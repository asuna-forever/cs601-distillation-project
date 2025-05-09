# scripts/distillation_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_config

class DistillationLoss(nn.Module):
    """
    Computes the distillation loss. Aligns with Hugging Face standard
    practice of using -100 as the ignore_index for CrossEntropyLoss.
    """
    def __init__(self, base_criterion, teacher_model, student_tokenizer, alpha=train_config.DISTILLATION_ALPHA, temperature=train_config.DISTILLATION_TEMP):
        """
        Args:
            base_criterion: Ignored. Will use standard CrossEntropyLoss with ignore_index=-100.
            teacher_model: The pre-loaded teacher model (set to eval mode).
            student_tokenizer: The student model's tokenizer (used to get vocab size).
            alpha (float): Weighting factor for the distillation loss component.
            temperature (float): Temperature scaling factor.
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        # *** Consistently use -100 as the ignore index for the loss function ***
        self.ignore_index = -100
        self.student_vocab_size = len(student_tokenizer) # Store vocab size for validation

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            print("Warning: No teacher model provided to DistillationLoss. Will only compute base loss.")
            self.alpha = 0.0

        # *** Initialize CrossEntropyLoss with ignore_index = -100 ***
        print(f"Initializing CrossEntropyLoss with ignore_index = {self.ignore_index}")
        self.base_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.kl_div_loss = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, student_outputs, labels, input_ids, attention_mask, **kwargs):
        """
        Calculates the combined distillation loss.
        Assumes labels are correctly shifted by the DataCollator and contain -100 for padding.
        """
        student_logits = student_outputs.logits
        # Use stored vocab size for validation consistency
        vocab_size = student_logits.size(-1) 

        # --- Validate Labels ---
        if self.alpha < 1.0: # Only check if CE loss is actually computed
            invalid_labels = []
            labels_flat = labels.view(-1)
            for i, label_val in enumerate(labels_flat):
                # Check labels that are NOT the ignore_index (-100)
                if label_val != self.ignore_index:
                    # Check if the valid label is within the expected range
                    if not (0 <= label_val < vocab_size):
                        invalid_labels.append((i, label_val.item()))

            if invalid_labels:
                print(f"错误：发现无效的 Label ID (不在 [0, {vocab_size-1}] 范围内且不是 ignore_index={self.ignore_index})!")
                for idx, val in invalid_labels[:10]:
                     print(f"  - Flat Index {idx}: Value {val}")
                # This should ideally not happen if input ID validation passed
                raise ValueError(f"Invalid label IDs detected after data collation. Vocab size is {vocab_size}. Check data processing pipeline.")
        # --- Validation End ---


        # 1. Standard Student Loss (Cross-Entropy)
        loss_ce = torch.tensor(0.0, device=student_logits.device)
        if self.alpha < 1.0:
            try:
                # Ensure logits vocab dimension matches expected vocab size
                # if student_logits.size(-1) != vocab_size:
                #    raise ValueError(f"Logits last dimension ({student_logits.size(-1)}) does not match expected vocab size ({vocab_size})")
                vocab_size = student_logits.size(-1)
                loss_ce = self.base_criterion(
                    student_logits.view(-1, vocab_size), # (batch * seq_len, vocab_size)
                    labels.view(-1)                   # (batch * seq_len)
                )
                if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                     print("Warn: Cross Entropy Loss is NaN or Inf!")
                     loss_ce = torch.tensor(0.0, device=student_logits.device)
            except RuntimeError as e:
                 print(f"Error when calculating Cross Entropy Loss: {e}")
                 print(f"  Logits shape: {student_logits.shape}")
                 print(f"  Labels shape: {labels.shape}")
                 print(f"  Min label value: {labels.min().item()}")
                 print(f"  Max label value: {labels.max().item()}")
                 print(f"  Vocab size used in check: {vocab_size}")
                 print(f"  Ignore index used: {self.ignore_index}")
                 raise e

        # 2. Distillation Loss (KL Divergence)
        loss_kl = torch.tensor(0.0, device=student_logits.device)
        if self.teacher_model is not None and self.alpha > 0.0:
            try:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits.to(student_logits.device)

                seq_len_student = student_logits.shape[1]
                seq_len_teacher = teacher_logits.shape[1]
                if seq_len_student != seq_len_teacher:
                    min_seq_len = min(seq_len_student, seq_len_teacher)
                    student_logits = student_logits[:, :min_seq_len, :]
                    teacher_logits = teacher_logits[:, :min_seq_len, :]
                    labels = labels[:, :min_seq_len] # Ensure labels length matches

                student_log_probs_t = F.log_softmax(student_logits / self.temperature, dim=-1)
                teacher_probs_t = F.softmax(teacher_logits / self.temperature, dim=-1)
                kl_loss_all = self.kl_div_loss(student_log_probs_t, teacher_probs_t).sum(dim=-1)

                # *** Use ignore_index = -100 for the mask ***
                pad_mask = (labels != self.ignore_index) # (batch_size, seq_len)

                masked_kl_loss = kl_loss_all * pad_mask
                num_active_elements = pad_mask.sum()

                if num_active_elements > 0:
                    loss_kl = masked_kl_loss.sum() / num_active_elements
                else:
                    loss_kl = torch.tensor(0.0, device=student_logits.device)

                loss_kl = (self.temperature ** 2) * loss_kl

                if torch.isnan(loss_kl) or torch.isinf(loss_kl):
                     print("Warn: KL Divergence Loss is NaN or Inf!")
                     loss_kl = torch.tensor(0.0, device=student_logits.device)

            except Exception as e:
                print(f"Error when calculating KL divergence: {e}")
                loss_kl = torch.tensor(0.0, device=student_logits.device)

        # 3. Combine Losses
        total_loss = self.alpha * loss_kl + (1.0 - self.alpha) * loss_ce

        loss_dict = {
            "total_loss": total_loss,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl
        }

        return total_loss, loss_dict

# --- Example Usage (保持不变) ---
if __name__ == "__main__":
    class DummyTokenizer:
        # Simulate tokenizer where pad token ID is different from -100
        pad_token_id = 0
        eos_token_id = 1
        vocab_size = 10
        def __len__(self): return self.vocab_size

    dummy_tokenizer = DummyTokenizer()

    batch_size, seq_len, vocab_size = 2, 5, dummy_tokenizer.vocab_size
    dummy_student_logits = torch.randn(batch_size, seq_len, vocab_size)
    dummy_teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Simulate labels from DataCollator: padding is -100
    dummy_labels = torch.randint(1, vocab_size, (batch_size, seq_len)) # Valid labels
    dummy_labels[0, -2:] = -100 # Use -100 for padding simulation

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    class DummyModel(nn.Module):
        def forward(self, **kwargs): return type('obj', (object,), {'logits': dummy_teacher_logits})()
    dummy_teacher = DummyModel()
    dummy_teacher.eval()

    # Instantiate the loss - pass the tokenizer
    distill_loss_fn = DistillationLoss(None, dummy_teacher, dummy_tokenizer, alpha=0.7, temperature=3.0)

    # Calculate loss
    try:
        total_loss, loss_components = distill_loss_fn(
            student_outputs=type('obj', (object,), {'logits': dummy_student_logits})(),
            labels=dummy_labels,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        )

        print(f"Total Loss: {total_loss.item()}")
        print(f"CE Loss: {loss_components['loss_ce'].item()}")
        print(f"KL Loss: {loss_components['loss_kl'].item()}")
        print(f"Using ignore_index: {distill_loss_fn.ignore_index}") # Should now print -100
    except ValueError as e:
         print(f"Caught error during testing: {e}") # Should not happen if labels are correct
