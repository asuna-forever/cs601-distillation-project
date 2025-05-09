# configuration_distilqwen3.py

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

class DistilQwen3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DistilQwen3Model`].
    It is used to instantiate a DistilQwen3 model according to the specified arguments,
    defining the model architecture. This configuration is designed for a distilled version
    of Qwen3, featuring **independent decoder layers** and an optional MLP bottleneck.

    Note: With independent layers, the total parameter count will be substantially higher
    than a model with shared layers if `num_hidden_layers` and other dimensional parameters
    are kept the same. Adjust default values if targeting a specific total parameter count.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the
    model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the DistilQwen3 model.
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations in each decoder layer.
        intermediate_size (`int`, *optional*, defaults to 8192):
            *Base* dimension of the MLP representations within each decoder layer if the
            bottleneck is not effectively used. Typically calculated as (8/3 * hidden_size).
        mlp_bottleneck_dim (`int`, *optional*, defaults to 1536):
            Dimension of the bottleneck layer within each MLP block. If set and smaller than
            `intermediate_size`, it reduces parameters in that MLP block. Defaults to `hidden_size / 2`.
            Set to `None` or a value `>= intermediate_size` to effectively disable the bottleneck.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of independent decoder layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads (Grouped Query Attention) for each attention layer.
            Must divide `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 96):
            The dimension of each attention head (`hidden_size // num_attention_heads`).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the MLP of each decoder layer.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (past_key_values)
            to speed up sequential decoding.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the Rotary Position Embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for RoPE embeddings.
            Example: `{"type": "linear", "factor": 2.0}`
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the query, key, value, and output projection layers
            of the attention mechanism.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention in the initial layers.
        sliding_window (`int`, *optional*, defaults to 4096):
            The size of the sliding window for attention. Only applicable if `use_sliding_window` is `True`.
        max_window_layers (`int`, *optional*, defaults to 20):
            Number of initial independent decoder layers that will use Sliding Window Attention (SWA).
            Subsequent layers will use full attention if SWA is enabled. Must be <= `num_hidden_layers`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from configuration_distilqwen3 import DistilQwen3Config
    >>> from transformers import AutoModelForCausalLM # Or your custom DistilQwen3ForCausalLM

    >>> # Initializing a DistilQwen3 configuration for a model with independent layers
    >>> configuration = DistilQwen3Config()

    >>> # Initializing a model (with random weights) from this configuration
    >>> # model = DistilQwen3ForCausalLM(configuration) # If using your custom class directly
    >>> model = AutoModelForCausalLM.from_config(configuration) # If registered with AutoModel

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # model_type can be kept, or changed if you want to strongly distinguish
    # this non-shared version (e.g., "distilqwen3_nonshared").
    # For AutoModel discovery, this string is key.
    model_type = "distilqwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=24,  # Number of independent layers
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None, # Will be calculated if None
        max_window_layers=20,
        mlp_bottleneck_dim=1536,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_bottleneck_dim = mlp_bottleneck_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            # Default to the same number of heads as attention heads (Multi-Head Attention)
            # or a common ratio for Grouped Query Attention.
            # Qwen models often use GQA, so providing a default if not specified.
            # Example: num_key_value_heads = num_attention_heads // 4 or just num_attention_heads
             num_key_value_heads = num_attention_heads # Default to MHA if not specified, or choose a GQA default
        self.num_key_value_heads = num_key_value_heads

        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim
        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads}) "
                f"and equal to `head_dim` ({self.head_dim}) * `num_attention_heads`."
            )

        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        if self.use_sliding_window:
            self.max_window_layers = min(max_window_layers, num_hidden_layers)
        else:
            # If not using SWA, max_window_layers isn't strictly necessary but set for consistency
            self.max_window_layers = num_hidden_layers


        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate RoPE config:
        # The `type` field in `rope_scaling` is deprecated just use `rope_type`
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.pop("type")
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

# To make it discoverable by `from . import DistilQwen3Config`
__all__ = ["DistilQwen3Config"]