# --- Standard Libraries ---
import math # Potentially needed by some modeling utils not shown, good practice
from typing import Optional, Tuple, Union, Callable, List # Added List

# --- PyTorch ---
import torch
from torch import nn
import torch.nn.functional as F # Used in eager_attention_forward
import torch.utils.checkpoint # Used in Qwen3Model forward

# --- Hugging Face Transformers ---
from transformers import PreTrainedModel, PretrainedConfig # Base classes
from transformers.activations import ACT2FN # For MLP activation
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast # Return types
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS # For attention backend selection
# GradientCheckpointingLayer is not directly inherited by Qwen3DecoderLayer in the target non-shared version
from transformers.generation import GenerationMixin # For Qwen3ForCausalLM
from transformers.integrations import use_kernel_forward_from_hub # Decorator for RMSNorm
from transformers.utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, can_return_tuple, replace_return_docstrings, is_torch_flex_attn_available # Utilities & Decorators
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache # KV Cache classes
from transformers.modeling_attn_mask_utils import AttentionMaskConverter # For mask handling
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation # RoPE utilities
from transformers.processing_utils import Unpack # For Kwargs type hint
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import LossKwargs # Assuming LossKwargs is here

# Conditional import for Flex Attention
if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask # Used in type hints or internal checks
    from transformers.integrations.flex_attention import make_flex_block_causal_mask # Used in _update_causal_mask

# --- Local Imports (Placeholders - Adjust Paths) ---
# Assuming DistilQwen3Config is defined in a local file
from configuration_distilqwen3 import DistilQwen3Config # Example import path

# The rest of the model classes (Qwen3RMSNorm, etc.) are defined within this file based on the provided code.

# --- Start of your Code ---
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "distil-qwen3-placeholder"
_CONFIG_FOR_DOC = "DistilQwen3Config"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module, # module is Qwen3Attention instance
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs, # Includes sliding_window potentially passed from Qwen3Attention.forward
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class DistilQwen3PreTrainedModel(PreTrainedModel):
    config_class = DistilQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # *** MODIFIED: Update the class name for the non-shared layer ***
    _no_split_modules = ["Qwen3DecoderLayer"] # Changed from Qwen3DecoderLayerShared
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLPWithBottleneck(nn.Module):
    """
    Qwen3 MLP module modified with an optional bottleneck dimension.
    This module remains unchanged as per the request to keep bottleneck learning.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.bottleneck_dim = getattr(config, "mlp_bottleneck_dim", None)
        if self.bottleneck_dim is None or self.bottleneck_dim >= self.intermediate_size:
            logger.info(
                f"No bottleneck dimension specified or bottleneck_dim ({self.bottleneck_dim}) >= intermediate_size ({self.intermediate_size}). "
                f"Using standard MLP structure with intermediate_size = {self.intermediate_size} as bottleneck_dim."
            )
            self.bottleneck_dim = self.intermediate_size
        else:
             logger.info(
                f"Using MLP bottleneck dimension: {self.bottleneck_dim}"
            )

        self.gate_proj = nn.Linear(self.hidden_size, self.bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.bottleneck_dim, bias=False)
        self.down_proj = nn.Linear(self.bottleneck_dim, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate_act = self.act_fn(gate_output) * up_output
        down_output = self.down_proj(intermediate_act)
        return down_output

class Qwen3Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    `layer_idx` is now passed during __init__ and used for sliding window logic.
    """
    def __init__(self, config: DistilQwen3Config, layer_idx: int): # layer_idx is now an init arg
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # Store layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Determine sliding_window based on self.layer_idx (stored from init)
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers # Use self.layer_idx
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        # current_layer_idx: int, # REMOVED: layer_idx is now part of self
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs], # Includes output_attentions
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Use self.layer_idx for cache update, as this is a distinct layer instance
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, # Pass module instance
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window, # Pass pre-determined sliding_window
            **kwargs, # Pass other kwargs like output_attentions
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # output_attentions is typically passed in kwargs
        if not kwargs.get("output_attentions", False):
             attn_weights = None

        # The past_key_value object is managed by the Cache class and updated in place.
        # It's returned at the model level, not from each attention layer.
        return attn_output, attn_weights


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DistilQwen3Config`]): # Changed to DistilQwen3Config
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3RotaryEmbedding(nn.Module): # Unchanged from original
    def __init__(self, config: DistilQwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

# Qwen3AttentionShared is removed as we are not using shared attention layers.

class Qwen3DecoderLayer(nn.Module): # Renamed from Qwen3DecoderLayerShared
    """
    Standard Qwen3DecoderLayer with its own parameters.
    Uses Qwen3Attention and Qwen3MLPWithBottleneck.
    `layer_idx` is passed to Qwen3Attention during its initialization.
    """
    def __init__(self, config: DistilQwen3Config, layer_idx: int): # Takes layer_idx for its components
        super().__init__()
        self.hidden_size = config.hidden_size
        # Pass layer_idx to Qwen3Attention constructor
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLPWithBottleneck(config) # Retains bottleneck MLP
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # layer_idx: int, # REMOVED: Not needed for forward pass of an independent layer
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False, # use_cache is relevant for past_key_value handling
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs], # Pass FlashAttentionKwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # `layer_idx` is implicitly handled by self.self_attn as it was set during __init__
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # position_ids=position_ids, # Qwen3Attention doesn't use it directly, but RoPE does via position_embeddings
            past_key_value=past_key_value, # Pass the cache for this layer
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            # Pass output_attentions flag down to Qwen3Attention
            output_attentions=output_attentions,
            **kwargs, # Pass other kwargs like those for flash attention
        )
        hidden_states = attn_outputs[0] # Attention output
        self_attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions and self_attn_weights is not None:
            outputs += (self_attn_weights,)
        # Cache is handled by the past_key_value object itself if use_cache is True.
        # The updated cache object is returned at the model level.
        return outputs


class Qwen3Model(DistilQwen3PreTrainedModel): # Renamed from Qwen3ModelSharedLayers
    """
    The Qwen3 Model transformer with independent layers.
    Each layer is an instance of Qwen3DecoderLayer.
    """
    # config_class, base_model_prefix, etc. are inherited from DistilQwen3PreTrainedModel

    def __init__(self, config: DistilQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # self.num_hidden_layers = config.num_hidden_layers # Already in config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # *** MODIFICATION: Create a ModuleList of independent Qwen3DecoderLayer instances ***
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # self.shared_layer = ... # REMOVED

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False # Default, can be enabled

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
             inputs_embeds = self.embed_tokens(input_ids)

        # KV Cache setup
        # past_key_values is a Cache instance that manages states for all layers.
        # Each Qwen3Attention layer will use its self.layer_idx to interact with the cache.
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache() # Or other cache type based on config
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length(layer_idx=0) # layer_idx=0 is fine here as it's for total seq len
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
        else:
             past_key_values = None

        if position_ids is None:
            if cache_position is None:
                 cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # next_decoder_cache is effectively the updated past_key_values object itself.

        # *** MODIFICATION: Iterate through self.layers (ModuleList) ***
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # The past_key_value for the current layer is handled internally by the Cache object
            # when past_key_value is passed to the layer.
            current_past_key_value_for_layer = past_key_values

            if self.gradient_checkpointing and self.training:
                # Define a closure for checkpointing, ensuring all args are passed
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # The module (decoder_layer) handles its own layer_idx internally for its components.
                        # We don't pass layer_idx as an explicit argument to decoder_layer.forward()
                        # inputs: hidden_states, attention_mask, position_ids, past_key_value,
                        #         output_attentions, use_cache, cache_position, position_embeddings
                        # And **flash_attn_kwargs needs to be included
                        passed_kwargs = {k: v for k, v in flash_attn_kwargs.items()}
                        return module(inputs[0], # hidden_states
                                      attention_mask=inputs[1],
                                      position_ids=inputs[2], # Kept for consistency, though RoPE uses position_embeddings
                                      past_key_value=inputs[3],
                                      output_attentions=inputs[4],
                                      use_cache=inputs[5],
                                      cache_position=inputs[6],
                                      position_embeddings=inputs[7],
                                      **passed_kwargs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_mask,
                    position_ids, # Pass position_ids
                    current_past_key_value_for_layer, # Pass the cache object
                    output_attentions,
                    use_cache, # Pass use_cache flag
                    cache_position,
                    position_embeddings,
                    use_reentrant=self.config.gradient_checkpointing_kwargs.get("use_reentrant", False)
                                if hasattr(self.config, "gradient_checkpointing_kwargs") and self.config.gradient_checkpointing_kwargs is not None
                                else False, # Default to False if not specified or config doesn't have the new dict
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    # layer_idx is not passed here, it's part of the layer's own state/submodules
                    attention_mask=causal_mask,
                    position_ids=position_ids, # Pass position_ids
                    past_key_value=current_past_key_value_for_layer, # Pass the cache object
                    output_attentions=output_attentions,
                    use_cache=use_cache, # Pass use_cache flag
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            # *** End of MODIFICATION in loop ***

            hidden_states = layer_outputs[0]

            # Cache is updated in-place within the decoder_layer's attention if use_cache is True.
            # We don't need to collect next_decoder_cache per layer here as past_key_values is modified directly.

            if output_attentions:
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None # The cache object itself is the 'next_cache'

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache, # Return the (potentially updated) cache object
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        # This method's logic largely remains the same, as it depends on config and cache state,
        # not directly on whether layers are shared or not.
        # (Original implementation from your provided code)
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask: # This means it's a custom mask
                return attention_mask
            # For FA2, if no custom mask, return None to use its internal causal masking
            return None # This was the missing part for FA2 to use its internal causal mask
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        past_seen_tokens = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0 # layer_idx=0 for overall length
        sequence_length = input_tensor.shape[1]
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions # SDPA with output_attentions falls back to eager
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor, # Changed from hidden_states to input_tensor
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window, # Pass model's sliding_window config
                is_training=self.training,
            ):
                return None # Return None for SDPA to use its internal causal mask

        # Determine target_length for the mask
        if past_key_values is not None:
            target_length = past_seen_tokens + sequence_length
        else:
            target_length = sequence_length


        # The _prepare_4d_causal_attention_mask_with_cache_position expects target_length
        # which is the kv_sequence_length (past + current).
        # The query_length is sequence_length.
        return self._prepare_4d_causal_attention_mask_with_cache_position(
             attention_mask,
             sequence_length, # query_length
             target_length,   # kv_sequence_length
             input_tensor.dtype,
             cache_position,
             input_tensor.shape[0], # batch_size
             self.config, # Pass the model config
             past_key_values # Pass the cache object
         )


    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        kv_length: int, # Renamed target_length to kv_length for clarity
        dtype: torch.dtype,
        cache_position: torch.LongTensor,
        batch_size: int,
        config: DistilQwen3Config, # Added config typing
        past_key_values: Optional[Cache] = None, # Added past_key_values
    ):
        # (Original implementation from your provided code, with minor adjustments for clarity/completeness)
        # B x H x Q x K
        # Note: sliding_window logic within this static method needs careful review
        # if it's different from the per-layer sliding_window in Qwen3Attention.
        # For non-shared layers, Qwen3Attention itself handles its sliding_window logic.
        # This function primarily constructs the base causal mask and applies padding.

        # If a 4D mask is already provided, assume it's correctly formatted.
        if attention_mask is not None and attention_mask.dim() == 4:
            # TODO: It is not generally safe to return attention_mask directly.
            # It might not have the correct shape if the user is passing only the bi-dimensional
            # attention_mask. This was a Llama mistake.
            # Here we accept it for backward compatibility.
            # Fixing it would require a minor BC break.
            # if query_length != kv_length and attention_mask.shape[-2] == attention_mask.shape[-1]:
            #     attention_mask = attention_mask[..., :query_length, :kv_length]
            return attention_mask

        # Standard causal mask
        # TODO: make it possible to pass `causal=True` to `sdpa` and remove this mask creation
        # NOTE: we cause errors by creating a causal mask that has the wrong shape for the sliding window
        # attn mask. It needs to be [bsz, num_heads, q_len, kv_len]
        # We will create the mask using the maximum sequence length to avoid recomputation
        # of the mask during generation.
        # The mask is then cut to the correct length using the cache_position.
        # Though it will be less than the max seq len, this is generally not an issue with Pytorch XLA.
        # For SWA, we have a different logic for constructing causal mask, so we return None here.
        if (
            isinstance(past_key_values, SlidingWindowCache)
            and getattr(config, "sliding_window", None) is not None # Check if sliding_window is configured
            and kv_length > 1 # SWA is not applicable for single token
        ):
            # With SWA, the mask generation is more complex and often handled differently
            # or might rely on specific mechanisms in the attention implementation.
            # For simplicity in this static method, if SWA is active,
            # one might return a simpler mask or expect FA/SDPA to handle it.
            # However, eager attention still needs a mask.
            # The Qwen3Attention module itself applies its own `sliding_window` parameter
            # to the attention function call. This static mask prep function should
            # provide the base causal + padding mask.
            pass # Continue to create base causal mask, SWA modification happens in attention_interface

        # Fallback to creating a 4D mask
        # [bsz, num_heads, query_length, kv_length]
        # `AttentionMaskConverter._make_causal_mask` can be used here
        # For now, using a simplified direct creation for illustration, assuming cache_position gives correct device
        # This creates a lower triangular mask of shape (batch_size, 1, query_length, kv_length)
        # Mask values are 0 where attention is allowed, and -inf where it's masked.
        mask = torch.full(
            (batch_size, 1, query_length, kv_length), torch.finfo(dtype).min, device=cache_position.device
        )

        # `cache_position` has shape (sequence_length)
        # `seq_ids` has shape (batch_size, sequence_length)
        # `expanded_mask` has shape (batch_size, 1, sequence_length, sequence_length)
        if query_length == kv_length:
            # Standard self-attention case (no KV cache or KV cache length matches query length)
            # Create a lower triangular mask
            # mask.tril_() would fill upper triangle with 0, we need lower triangle to be 0 (unmasked)
            # and upper triangle to be -inf (masked).
            # indices_q.unsqueeze(-1) <= indices_k.unsqueeze(-2)
            # Creates a boolean mask, then convert to float.
            indices_q = torch.arange(query_length, device=cache_position.device).view(1, query_length, 1)
            indices_k = torch.arange(kv_length, device=cache_position.device).view(1, 1, kv_length)
            causal = indices_q >= indices_k # For causal, query pos must be >= key pos for attention
            # This results in True where q >= k. For additive mask, we want 0 there.
            # We need q < k to be masked (-inf).
            # So, where q >= k, mask is 0. Where q < k, mask is -inf.
            # Correct: mask where query_pos < key_pos
            # causal_shape = (1, 1, query_length, kv_length)
            # temp_mask = torch.ones(causal_shape, device=cache_position.device, dtype=torch.bool).tril(diagonal=0)
            # mask = torch.where(temp_mask, torch.zeros((), device=mask.device, dtype=dtype), torch.finfo(dtype).min)
            # A simpler way:
            # mask = torch.triu(torch.full((query_length, kv_length), torch.finfo(dtype).min, device=cache_position.device), 1)
            # mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            # Corrected simple causal mask creation:
            if query_length > 0: # Avoid issues with empty sequences
                # mask tensor for "where to attend" (0.0) and "where not to attend" (-inf)
                # This mask is (q_len, k_len)
                causal_2d_mask = torch.tril(torch.ones((query_length, kv_length), dtype=dtype, device=cache_position.device))
                # Invert for additive mask: 0 for allowed, -inf for masked
                causal_2d_mask = torch.where(causal_2d_mask.bool(), torch.zeros((), dtype=dtype, device=mask.device), torch.finfo(dtype).min)
                mask = causal_2d_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, kv_length)

        else: # query_length != kv_length (generation with KV cache)
            # The mask should be for (query_length, kv_length)
            # Query tokens can attend to all past key/value tokens + current token's key/value
            # cache_position gives the absolute positions of the query tokens.
            # Example: query is [tok_N], kv_cache is [tok_0, ..., tok_N-1]. kv_length = N. query_length = 1.
            # The single query token (at absolute position `cache_position[-1]`)
            # can attend to all kv_length tokens.
            # `col_indices` are absolute positions of keys: 0 to kv_length - 1
            # `row_indices` are absolute positions of queries: cache_position
            # `abs_mask[b, q_idx, k_idx] = 1 if query_tokens[b, q_idx] can attend to key_tokens[b, k_idx]`
            # `mask_cond = cache_position.unsqueeze(-1) >= torch.arange(kv_length, device=cache_position.device).unsqueeze(0)`
            # `current_mask = torch.where(mask_cond, 0.0, torch.finfo(dtype).min)`
            # `current_mask = current_mask.unsqueeze(1).expand(batch_size, 1, query_length, kv_length)`
            # This is more complex due to cache_position indicating actual positions.
            # A simpler approach assuming contiguous cache:
            # q_indices are from 0 to query_length-1
            # k_indices are from 0 to kv_length-1
            # A query token q_idx (which is at absolute position `past_seen + q_idx`)
            # can attend to key token k_idx (which is at absolute position `k_idx`)
            # if `past_seen + q_idx >= k_idx`.
            # `past_seen = kv_length - query_length` (assuming full cache up to query start)
            # So, `kv_length - query_length + q_idx >= k_idx`
            q_indices = torch.arange(query_length, device=cache_position.device).view(1, query_length, 1)
            k_indices = torch.arange(kv_length, device=cache_position.device).view(1, 1, kv_length)
            past_seen = kv_length - query_length # Number of elements in cache before current query tokens
            condition = (past_seen + q_indices) >= k_indices
            generated_mask = torch.where(condition, torch.zeros((), dtype=dtype, device=mask.device), torch.finfo(dtype).min)
            mask = generated_mask.unsqueeze(0).expand(batch_size, 1, query_length, kv_length)


        # Apply padding mask if provided
        if attention_mask is not None and attention_mask.dim() == 2: # (batch_size, kv_length) or (batch_size, query_length)
            # Ensure attention_mask is expanded to kv_length for key masking
            # The original Llama/Mistral code expects attention_mask to be (bsz, seq_len) where seq_len can be query_length.
            # It's then expanded for the kv_length dimension.
            # If attention_mask is (bsz, query_length), it means the mask is for queries.
            # If attention_mask is (bsz, kv_length), it directly masks keys.
            # Standard HF practice: attention_mask has shape (batch_size, sequence_length)
            # where sequence_length is the original input sequence length.
            # For generation, this usually means it's (batch_size, kv_length) after padding.
            padding_mask_dim = attention_mask.shape[-1]
            expected_padding_mask_shape = (batch_size, kv_length)

            if padding_mask_dim == kv_length: # Mask is for keys
                 _attention_mask = attention_mask[:, None, None, :].to(dtype) # (bsz, 1, 1, kv_len)
            elif padding_mask_dim == query_length: # Mask is for queries, expand to affect all keys for that query
                 _attention_mask = attention_mask[:, None, :, None].to(dtype) # (bsz, 1, q_len, 1)
            else: # Fallback or error, assume kv_length if unsure, or raise error
                # This case should be handled carefully based on expected behavior.
                # Assuming it should be kv_length for safety.
                logger.warning_once(
                    f"Attention mask shape {attention_mask.shape} does not match query_length {query_length} or kv_length {kv_length}. "
                    f"Assuming it's for kv_length if it's for padding."
                ) # Forcing kv_length here if not q_length
                if attention_mask.shape[1] > kv_length : # if original seq_len > kv_len (e.g. with sliding window cache)
                    attention_mask = attention_mask[:, :kv_length]
                elif attention_mask.shape[1] < kv_length and query_length == 1 : # Pre-fill phase or generation with full cache context
                     # This happens when the input `attention_mask` is for the full sequence (e.g., from tokenizer)
                     # but `kv_length` is smaller due to, e.g., `max_position_embeddings`.
                     # Or, more commonly, kv_length is larger (includes cache).
                     # If kv_length > attention_mask.shape[1], pad the attention_mask.
                     # This is tricky. Let's assume the common case where attention_mask from input covers kv_length if it's for padding.
                     pass # Keep as is, rely on broadcasting or ensure it's kv_length before this function

                _attention_mask = attention_mask[:, None, None, :padding_mask_dim].expand(batch_size, 1, query_length, padding_mask_dim).to(dtype)
                # If padding_mask_dim is not kv_length, this expansion might be problematic.
                # It's safer to ensure attention_mask corresponds to kv_length for padding.
                # If it's a padding mask, it should affect the K dimension.

            # Combine with causal mask: if either causal OR padding mask says to mask, then mask.
            # Mask is additive: 0 means allow, -inf means disallow.
            # So we add them. If _attention_mask has 0 for unmasked and -inf for masked (like `mask`).
            # No, HF standard is 1 for NOT MASKED, 0 for MASKED.
            # So, where attention_mask == 0, we should put -inf.
            padding_additive_mask = (_attention_mask == 0) * torch.finfo(dtype).min
            mask = mask + padding_additive_mask # Broadcasting if shapes differ slightly (e.g. _attention_mask is (b,1,1,kv))

        # Ensure no NaNs or Infs creep in where not expected
        mask = torch.clamp(mask, min=torch.finfo(dtype).min)
        return mask


# --- Define the PreTrainedModel class if not imported ---
# class PreTrainedModel(nn.Module): ... # Placeholder
# class Qwen3Config: ... # Placeholder

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ... # Unchanged

class DistilQwen3ForCausalLM(DistilQwen3PreTrainedModel, GenerationMixin): # Unchanged structure, but uses new Qwen3Model
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        # *** Uses the refactored Qwen3Model with independent layers ***
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @can_return_tuple
    # @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0, # Keep this parameter
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs, # Pass through other kwargs (like flash_attn_kwargs)
        )

        hidden_states = outputs.last_hidden_state
        
        # Logic for logits_to_keep
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            slice_indices = slice(-logits_to_keep, None)
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None: # Adjust labels accordingly if they are provided
                 labels = labels[:, slice_indices]
        elif isinstance(logits_to_keep, torch.Tensor): # Tensor of indices
            slice_indices = logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                 labels = labels[:, slice_indices]
        else: # logits_to_keep is 0 or not a positive int/tensor, compute for all
            logits = self.lm_head(hidden_states)


        loss = None
        if labels is not None:
            # Ensure loss_function is defined or imported if it's a custom one.
            # Assuming a standard CrossEntropyLoss if not specified otherwise.
            # The loss function should handle vocab_size and potential ignore_index.
            loss_fct = nn.CrossEntropyLoss()
            # Shift logits and labels for next token prediction if that's the model's objective
            # For many Causal LM, this is handled by how labels are prepared.
            # If your model directly predicts next token given current, labels might not need shifting here
            # if they are already shifted during data prep (e.g., labels[i] = input_ids[i+1]).
            # Assuming logits are [batch_size, seq_len, vocab_size] and labels are [batch_size, seq_len]
            loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            # If your `loss_function` was a custom method (self.loss_function), ensure it's defined.
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs.get("loss_kwargs", {}))


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )