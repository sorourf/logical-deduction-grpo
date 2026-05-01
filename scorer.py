"""
Step 3: Answer extractor + multiple-choice correctness scorer for BIG-bench.

Run directly to test against a few hand-written cases:
    python scorer.py
"""

from __future__ import annotations  # so `str | None` works on Python 3.9

import re


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
PAREN_RE      = re.compile(r"\(\s*([A-Ga-g])\s*\)")             # "(B)" or "( b )"
WORD_RE       = re.compile(r"(?<![A-Za-z])([A-Ga-g])(?![A-Za-z])")  # standalone letter


def extract_answer_text(response: str) -> str:
    """Return the substring between <answer> and </answer>, or '' if absent."""
    m = ANSWER_TAG_RE.search(response)
    return m.group(1).strip() if m else ""


def normalize_letter(text: str) -> str | None:
    """
    Pull the *unique* answer letter out of a free-form string.

    Returns an uppercase letter ("A".."G") if exactly one candidate letter
    appears, else None. Examples:
        "(B)"                          -> "B"
        "B"                            -> "B"
        "b"                            -> "B"
        "(B) The owl is the rightmost" -> "B"
        "B or C"                       -> None   (ambiguous)
        ""                             -> None
    """
    # Tier 1: prefer a parenthesized letter like "(B)".
    matches = PAREN_RE.findall(text)
    if matches:
        distinct = {m.upper() for m in matches}
        return distinct.pop() if len(distinct) == 1 else None

    # Tier 2: fall back to a standalone letter (not embedded in a word).
    matches = WORD_RE.findall(text)
    distinct = {m.upper() for m in matches}
    return distinct.pop() if len(distinct) == 1 else None


def score_letter_answer(response: str, gold_letter: str) -> float:
    """
    Top-level scorer used by the reward function.

    Args:
        response:    full model output (the part *after* the prompt).
        gold_letter: from the prompt formatter, e.g. "(B)".

    Returns:
        1.0 if the predicted letter matches gold; 0.0 otherwise.
    """
    # TODO 3: pull text from <answer>...</answer>, then normalize to a single letter.
    inner = extract_answer_text(response)
    pred  = normalize_letter(inner)

    # TODO 4: normalize gold from "(B)" -> "B" and compare.
    gold = normalize_letter(gold_letter)
    return 1.0 if (pred is not None and pred == gold) else 0.0


# ---------- self-test ----------

TEST_CASES = [
    # (model_response, gold_letter, expected_score)
    ("<answer>(B)</answer>",                                "(B)", 1.0),
    ("<answer>B</answer>",                                  "(B)", 1.0),
    ("<answer>b</answer>",                                  "(B)", 1.0),
    ("<answer>(B) The owl is the rightmost.</answer>",      "(B)", 1.0),
    ("<answer>(C)</answer>",                                "(B)", 0.0),
    ("<answer>B or C</answer>",                             "(B)", 0.0),
    ("<answer></answer>",                                   "(B)", 0.0),
    ("no answer tags at all",                               "(B)", 0.0),
    ("<think>I think (B)</think><answer>(B)</answer>",      "(B)", 1.0),
]


def main():
    passed = 0
    for i, (resp, gold, expected) in enumerate(TEST_CASES):
        got = score_letter_answer(resp, gold)
        ok = "PASS" if got == expected else "FAIL"
        if ok == "PASS":
            passed += 1
        print(f"[{ok}] case {i}: gold={gold} expected={expected} got={got}  | resp={resp!r}")
    print(f"\n{passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
