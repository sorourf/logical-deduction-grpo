"""
Step 4: Zero-shot baseline accuracy of Qwen2.5-3B-Instruct on
BIG-bench logical_deduction (full validation split).

Saves per-example results to baseline_results.json.

Run:  python baseline_eval.py
"""

import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_formatter import MODEL_NAME, format_example
from scorer import score_letter_answer

RESULTS_DIR    = "results"
OUTPUT_FILE    = os.path.join(RESULTS_DIR, "baseline_results.json")
MAX_NEW_TOKENS = 512


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
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = out[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

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
        "n":        n,
        "correct":  correct,
        "accuracy": accuracy,
        "results":  results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\naccuracy = {accuracy:.4f}  ({int(correct)}/{n})")
    print(f"saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
