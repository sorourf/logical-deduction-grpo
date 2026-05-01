# GRPO Replication — Notes & References

## The video
- **Title:** How I trained Small Language Models to reason with Reinforcement Learning! (also titled "How I finetuned a Small LM to THINK and solve puzzles on its own (GRPO & RL!)")
- **Author:** Avishek Biswas (`@neural_avb`, channel: *Neural Breakdown with AVB*)
- **URL:** https://www.youtube.com/watch?v=yGkJj_4bjpE
- **Topic:** Implementing GRPO (Group Relative Policy Optimization, the RL algorithm behind DeepSeek-R1) from scratch in PyTorch to fine-tune a small LM to reason.

No public companion repo was found for this specific video. Avishek's [finetuning_recipes](https://github.com/avbiswas/finetuning_recipes) repo covers SFT/CPT/instruction-tuning but does not (yet) include a GRPO directory.

## Open decisions before implementing
1. **Base model** — small enough to train on modest hardware. Default candidate: Qwen2.5-0.5B or Qwen3-0.6B.
2. **Hardware target** — Mac (MPS / MLX), NVIDIA GPU, or Colab? Changes the stack.
3. **Task / dataset** — **syllogism + propositional-logic puzzles** (confirmed). Likely synthesized programmatically so the verifier can check answers deterministically (a natural fit for GRPO's verifiable-reward setup). Candidate datasets if not synthesizing: FOLIO, ProofWriter, ProntoQA, LogicNLI.
4. **"From scratch" depth** — implement GRPO loss + rollout loop on top of HuggingFace `transformers` + PEFT/LoRA (typical interpretation), or go lower (no PEFT/TRL)?

## Reference implementations (community)
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) — DeepSeek R1's GRPO from scratch, minimal deps (tokenizers + pytorch)
- [simple_GRPO](https://github.com/lsdefine/simple_GRPO) — very small GRPO reproduction of r1-like LLM thinking
- [Hands-On LLM Alignment: Coding GRPO from Scratch (Medium, Baicen Xiao)](https://medium.com/@baicenxiao/hands-on-llm-alignment-coding-grpo-from-scratch-step-by-step-30c6aa4a2146)
- [Burkov's minimalist GRPO (X post)](https://x.com/burkov/status/1890566690058170708)
- [Unsloth GRPO notebook (Llama 3.1 8B, low-VRAM)](https://x.com/UnslothAI/status/1889726411478278183)

## Author links
- GitHub: https://github.com/avbiswas
- X / Twitter: https://x.com/neural_avb
