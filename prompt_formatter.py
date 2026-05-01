"""
Step 2: Format one BIG-bench logical_deduction row into a chat prompt + gold letter.

Run this file directly to sanity-check on the first dataset example:
    python prompt_formatter.py
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from rich import print

labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a careful logical reasoner. Solve the puzzle step by step.\n"
    "Show your reasoning inside <think>...</think> tags, then output ONLY the "
    "final answer letter (e.g. \"(B)\") inside <answer>...</answer> tags."
)


def format_example(row: dict) -> tuple[list[dict], str]:
    """Turn one dataset row into (chat messages, gold letter).

    Returns:
        messages:    list of {role, content} dicts ready for apply_chat_template
        gold_letter: a single uppercase letter, e.g. "B"
    """
    options: list[str] = row["multiple_choice_targets"]
    scores: list[int]  = row["multiple_choice_scores"]

    # TODO 1: build the lettered options block.
    # Goal: a string like "(A) ...\n(B) ...\n(C) ...\n"
    # Hint: use chr(ord("A") + i) to get letters; enumerate(options).
    lettered=""
    for label, option in zip(labels, options):
        lettered += f"{label} {option}\n"

    # TODO 2: find the gold letter.
    # Hint: scores is a list of 0/1; the index where score == 1 is the correct option.
    gold_index = scores.index(1)
    gold_letter = labels[gold_index]   # e.g. "(B)"

    # TODO 3: build the user message.
    # The puzzle text is in row["inputs"]. Append a blank line, then
    # "Which of the following is true?\n", then the lettered options block.
    user_content = f"{row['inputs']}\n\nWhich of the following is true?\n{lettered}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return messages, gold_letter


def main():
    ds = load_dataset("tasksource/bigbench", "logical_deduction")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    row = ds["train"][0]
    messages, gold_letter = format_example(row)

    # Render with the model's chat template, including the assistant turn opener.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("[bold]--- PROMPT ---[/bold]")
    print(prompt)
    print(f"\n[bold green]Gold letter: {gold_letter}[/bold green]")


if __name__ == "__main__":
    main()
