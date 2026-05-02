"""
Step 6: Evaluate the GRPO-trained adapter on BIG-bench logical_deduction
validation, then compare to the baseline.

Run on the pod (after training):
    python eval_trained.py
"""

import json
import os
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_formatter import MODEL_NAME, format_example
from scorer import score_letter_answer

ADAPTER_DIR    = "checkpoints/final"
RESULTS_DIR    = "results"
OUTPUT_FILE    = os.path.join(RESULTS_DIR, "post_train_results.json")
MAX_NEW_TOKENS = 1024


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main():
    device, dtype = pick_device_and_dtype()
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"loading base model {MODEL_NAME} ...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

    print(f"loading LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
    model.eval()

    eval_set = load_dataset("tasksource/bigbench", "logical_deduction")["validation"]
    n = len(eval_set)
    print(f"evaluating on {n} examples")

    results = []
    correct = 0.0

    for i in range(n):
        row = eval_set[i]
        messages, gold = format_example(row)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        new_token_slice = out[:, prompt_len:]
        response = tokenizer.batch_decode(new_token_slice, skip_special_tokens=True)[0]

        score = score_letter_answer(response, gold)
        correct += score

        results.append({
            "idx":      int(row["idx"]),
            "gold":     gold,
            "score":    score,
            "response": response,
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            running = correct / (i + 1)
            print(f"  {i+1}/{n}   running_acc={running:.3f}")

    accuracy = correct / n
    summary = {
        "model":    MODEL_NAME,
        "adapter":  ADAPTER_DIR,
        "n":        n,
        "correct":  correct,
        "accuracy": accuracy,
        "results":  results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\npost-train accuracy = {accuracy:.4f}  ({int(correct)}/{n})")
    print(f"saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
