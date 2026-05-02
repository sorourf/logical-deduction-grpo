"""
Step 5b: Load Qwen2.5-3B-Instruct with LoRA adapters via PEFT.

Used by both train_grpo.py (training) and post-train eval.

Quick sanity check:
    python model_loader.py
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_formatter import MODEL_NAME


# ---- LoRA hyperparameters ---------------------------------------------------

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TARGET_MODULES   = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model_and_tokenizer():
    """Return (model_with_lora, tokenizer) ready for GRPO training."""
    device, dtype = pick_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = TARGET_MODULES,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    # gradient checkpointing trades compute for activation memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model, tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()
    model.print_trainable_parameters()
    print(f"device     : {next(model.parameters()).device}")
    print(f"dtype      : {next(model.parameters()).dtype}")


if __name__ == "__main__":
    main()
