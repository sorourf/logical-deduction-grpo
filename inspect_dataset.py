"""
Step 1: Load and inspect tasksource/bigbench logical_deduction.

Goal: confirm the column names and the format of the ground-truth answer
so we know what the reward function has to match against.

Run:  python inspect_dataset.py
"""

from datasets import load_dataset
from rich import print


def main():
    # TODO 1: load the dataset.
    # Hint: load_dataset("tasksource/bigbench", "logical_deduction")
    # The config has multiple subtasks (3, 5, 7 objects). Start with the default split.
    ds = load_dataset("tasksource/bigbench", "logical_deduction")


    # TODO 2: print the available splits and column names.
    # Hint: ds is a DatasetDict; iterate ds.keys() and look at ds[split].column_names
    for split in ds.keys():
        print(f"Split: {split}")
        print(f"Column names: {ds[split].column_names}")

    # TODO 3: print one full example from the train split (or whichever exists).
    # Look at: inputs, targets, multiple_choice_targets, multiple_choice_scores
    example = ds["train"][0]
    print(example)

    # TODO 4: print 3 more examples' targets only, to see the answer-shape pattern.
    # Are they letters like "(A)"? Full strings? Lists?
    for i in range(1, 4):
        print(ds["train"][i]["targets"])


if __name__ == "__main__":
    main()
