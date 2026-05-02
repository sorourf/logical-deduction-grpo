"""
Step 5c: GRPO training loop on BIG-bench logical_deduction.

Trains LoRA adapters on Qwen2.5-3B-Instruct using group-relative
advantages computed from format + correctness rewards.

Run:  python train_grpo.py
"""

import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset

from grpo_util import (
    calculate_grpo_loss,
    calculate_logits,
    calculate_rewards,
    generate_responses,
)
from model_loader import load_model_and_tokenizer
from prompt_formatter import format_example


# ---- hyperparameters --------------------------------------------------------

BATCH_PROMPTS    = 2          # B: number of distinct prompts per step
GROUP_SIZE       = 8          # G: rollouts sampled per prompt
MAX_NEW_TOKENS   = 1024        # cap on rollout length
TEMPERATURE      = 1.0
TOP_P            = 0.95

LEARNING_RATE    = 1e-5
CLIP_EPSILON     = 0.2
NUM_STEPS        = 400
LOG_EVERY        = 1
SAVE_EVERY       = 50
LOSS_VARIANT     = "grpo"     # "grpo" | "dr_grpo" | "bnpo"

OUTPUT_DIR       = "checkpoints"
LOG_FILE         = "results/training_log1.json"
SEED             = 0


# ---- helpers ----------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt_batch(rows, tokenizer):
    """Format a list of dataset rows -> (prompt strings, gold letters)."""
    prompts, golds = [], []
    for row in rows:
        messages, gold = format_example(row)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        golds.append(gold)
    return prompts, golds


def build_response_mask(full_ids, prompt_lens, pad_token_id):
    """
    Mask shape (B, L-1) with 1 where the *generated* tokens live, 0 for
    prompt and padding. Aligned to the shifted log-probs (length L-1).
    """
    B, L = full_ids.shape
    positions = torch.arange(L - 1, device=full_ids.device).unsqueeze(0)         # (1, L-1)
    targets = full_ids[:, 1:]                                                    # (B, L-1)
    prompt_lens_t = torch.tensor(prompt_lens, device=full_ids.device).unsqueeze(1)  # (B, 1)
    not_prompt = positions >= (prompt_lens_t - 1)                                # 1 once we're at a generated target
    not_pad    = targets != pad_token_id
    return (not_prompt & not_pad).float()


# ---- main -------------------------------------------------------------------

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print("Loading model + tokenizer ...")
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    print("Loading dataset ...")
    train_set = load_dataset("tasksource/bigbench", "logical_deduction")["train"]
    print(f"train size: {len(train_set)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    history = []
    step = 0
    train_indices = list(range(len(train_set)))

    while step < NUM_STEPS:
        # pick BATCH_PROMPTS random rows (sampling with replacement is fine here)
        rows = [train_set[i] for i in random.sample(train_indices, BATCH_PROMPTS)]
        prompts, golds = build_prompt_batch(rows, tokenizer)

        # --- 1. tokenize prompts (left-padded so generation is easy) ----------
        tokenizer.padding_side = "left"
        prompt_ids = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        prompt_len = prompt_ids.input_ids.shape[1]

        # --- 2. roll out G samples per prompt ---------------------------------
        model.eval()
        with torch.no_grad():
            generated = generate_responses(
                model,
                input_ids       = prompt_ids.input_ids,
                attention_mask  = prompt_ids.attention_mask,
                eos_token_id    = tokenizer.eos_token_id,
                n_rollouts      = GROUP_SIZE,
                max_new_tokens  = MAX_NEW_TOKENS,
                top_p           = TOP_P,
                temperature     = TEMPERATURE,
            )
        # generated.shape = (B*G, prompt_len + new_len)
        full_ids = generated
        full_attn = (full_ids != pad_id).long()
        # B*G prompt lengths are all the same since we left-padded to a common length
        prompt_lens = [prompt_len] * full_ids.shape[0]

        # decode rollouts (text after the prompt) for reward calc
        new_token_slice = full_ids[:, prompt_len:]
        responses = tokenizer.batch_decode(new_token_slice, skip_special_tokens=True)

        # repeat each gold letter G times so it lines up with B*G rollouts
        golds_repeated = [g for g in golds for _ in range(GROUP_SIZE)]

        # --- 3. rewards & group-relative advantages ---------------------------
        rewards = calculate_rewards(responses, golds_repeated)               # (B*G,)
        rewards_grouped = rewards.reshape(BATCH_PROMPTS, GROUP_SIZE)
        means = rewards_grouped.mean(axis=1, keepdims=True)
        stds  = rewards_grouped.std(axis=1, keepdims=True) + 1e-8
        advantages = ((rewards_grouped - means) / stds).reshape(-1)          # (B*G,)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # --- 4. log-probs (current policy and old policy) ---------------------
        model.train()
        log_probs = calculate_logits(model, full_ids, full_attn)             # (B*G, L-1)

        with torch.no_grad():
            with model.disable_adapter():
                old_log_probs = calculate_logits(model, full_ids, full_attn)

        # --- 5. response mask + broadcast advantages to per-token -------------
        resp_mask = build_response_mask(full_ids, prompt_lens, pad_id)       # (B*G, L-1)
        adv_per_token = advantages_t.unsqueeze(-1).expand_as(log_probs)      # (B*G, L-1)

        # --- 6. GRPO loss + step ----------------------------------------------
        loss = calculate_grpo_loss(
            log_probs        = log_probs,
            old_log_probs    = old_log_probs,
            response_mask    = resp_mask,
            advantages       = adv_per_token,
            clip_epsilon     = CLIP_EPSILON,
            loss_implementation = LOSS_VARIANT,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        # --- 7. logging --------------------------------------------------------
        record = {
            "step":          step,
            "loss":          float(loss.detach()),
            "reward_mean":   float(rewards.mean()),
            "reward_max":    float(rewards.max()),
            "reward_min":    float(rewards.min()),
            "correct_frac":  float((rewards > 0.5).mean()),  # heuristic
        }
        history.append(record)
        if step % LOG_EVERY == 0:
            print(f"step {step:>4}  loss={record['loss']:+.4f}  "
                  f"reward_mean={record['reward_mean']:+.3f}  "
                  f"reward_max={record['reward_max']:+.3f}  "
                  f"correct≈{record['correct_frac']:.2f}")

        if step > 0 and step % SAVE_EVERY == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{step}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with open(LOG_FILE, "w") as f:
                json.dump(history, f, indent=2)

        step += 1

    # final save
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\ndone. saved adapters -> {final_dir}")
    print(f"log -> {LOG_FILE}")


if __name__ == "__main__":
    main()
