"""
Step 4: Zero-shot baseline accuracy of Qwen2.5-3B-Instruct on
BIG-bench logical_deduction (validation split).

Run:  python baseline_eval.py

For a quick smoke test, leave N_EVAL small (10).
On the pod, set N_EVAL = None to run the full validation split.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_formatter import MODEL_NAME, format_example
from scorer import score_letter_answer


# ---- config -----------------------------------------------------------------

N_EVAL: int | None = 10        # set to None for the full validation set
MAX_NEW_TOKENS     = 512       # plenty for <think>...</think><answer>(B)</answer>
SHOW_FIRST_K       = 3         # print this many full responses for inspection


# ---- helpers ----------------------------------------------------------------

def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16   # bf16 support on MPS is patchy
    return "cpu", torch.float32


def main():
    device, dtype = pick_device_and_dtype()
    print(f"Device: {device}  | dtype: {dtype}")

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading dataset ...")
    ds = load_dataset("tasksource/bigbench", "logical_deduction")
    eval_set = ds["validation"]
    n = len(eval_set) if N_EVAL is None else min(N_EVAL, len(eval_set))
    print(f"Evaluating on {n} examples (validation has {len(eval_set)} total)\n")

    correct = 0.0
    for i in range(n):
        row = eval_set[i]
        messages, gold_letter = format_example(row)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                         # greedy baseline
                pad_token_id=tokenizer.pad_token_id,
            )

        # slice off the prompt tokens; only decode the model's continuation
        new_tokens = out[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        score = score_letter_answer(response, gold_letter)
        correct += score

        marker = "✓" if score == 1.0 else "✗"
        print(f"[{i+1:>3}/{n}] {marker} gold={gold_letter}  "
              f"score={score}  resp_tail={response.strip()[-80:]!r}")

        if i < SHOW_FIRST_K:
            print(f"        --- full response ---\n{response}\n        ---------------------")

    accuracy = correct / n
    print(f"\nBaseline accuracy: {accuracy:.2%}  ({int(correct)}/{n})")


if __name__ == "__main__":
    main()
