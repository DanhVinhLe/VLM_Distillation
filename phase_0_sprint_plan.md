# Phase 0 Sprint Plan — Unit-Aligned VLM Distillation (Short Paper)

**Submission target:** ARR May cycle, internal cutoff **2026-05-22** (3-day buffer before ARR's May 25 deadline).
**Goal:** one decisive table — 4 methods × 1 teacher/student pair × 4 benchmarks — supporting a short-paper claim that joint text-span + vision-token alignment beats either single side on a cross-architecture VLM pair.
**Architecture:** Existing `Distiller` + `DistillTrainer` reused. New `UnitAlignedDistillationCriterion` (single shared student/teacher forward), `CEOnlyCriterion` baseline. SRE upgraded with learnable student→teacher projector replacing `_crop_last_dim`. EM-KD α/β/γ/T now read from CLI args.
**Tech stack:** Qwen2.5-VL-7B teacher → Qwen2-VL-2B student, LoRA r=64, bf16, Cambrian-7M subset (or `llava_v1_5_mix665k` already on disk), VLMEvalKit pinned commit.

---

## Decision: what we are NOT doing this sprint

- **Only one teacher/student pair.** Strategy B (3 pairs incl. Qwen3-VL, LLaVA-OV) is the ARR June target if this short paper rolls forward.
- **No DSKD-CMA / GenRecal reimplementation.** We compare against published numbers in related-work prose, not in our compute-matched table.
- **No T1–T7 / V1–V6 SRE/EM-KD ablations.** One headline ablation table only: each method removes one component.
- **No frontier-scale stretch.** Only LoRA r=64; no full-finetune.
- **No multi-seed.** Single seed 1337. Acknowledge in limitations.

These constraints make the 7-day budget feasible. The full grid lives in `unit_aligned_distillation_audit_and_plan.md` and remains the post-May-22 backlog.

---

## Status snapshot (filled in as we go)

| Item | Status |
|---|---|
| Branch `phase-0-paper-sprint` | ✅ created |
| 4 glaring bug fixes | ✅ landed (this commit) |
| `UnitAlignedDistillationCriterion` | ✅ landed |
| `CEOnlyCriterion` | ✅ landed |
| UMI computation script | ⬜ Day 1 |
| Teacher logit cache | ⬜ Day 2 |
| SFT baseline run | ⬜ Day 2-3 |
| EM-KD-only run | ⬜ Day 3-4 |
| SRE-only run | ⬜ Day 4-5 |
| Unit-Aligned run | ⬜ Day 5-6 |
| VLMEvalKit on 4 benchmarks × 4 ckpts | ⬜ Day 6-7 |
| Paper draft (short, 4–5 pages) | ⬜ Day 6-7 |
| Submit | ⬜ May 22 |

---

## Day-by-day plan

### Day 1 (May 15 — today) — Code & UMI

- [x] **Task 1: Bug fix — `kd_loss_type` default.** `src/arguments.py:100`. Default flipped from broken `"contrastive_rkd"` to `"ce_only"` (safe-by-default; KD criteria must be explicitly selected).
- [x] **Task 2: Bug fix — EM-KD weights from args.** `src/criterions/em_kd.py:107-115`. α/β/γ/T now read via `getattr(args, "em_kd_*", default)`; previously hard-coded, silently shadowing CLI flags.
- [x] **Task 3: Bug fix — drop dead RKD args.** Removed `rkd_distance_weight` and `rkd_angle_weight` from `TrainingArguments` (`src/arguments.py`). Help text on `kd_loss_type` updated.
- [x] **Task 4: Bug fix — SRE projector path.** `src/criterions/sre.py`: new `_align_hidden_dims(student_h, teacher_h, projector)` replaces `_crop_last_dim` as the principled path. `SRECriterion._get_projector(distiller, idx)` looks up `distiller.projectors[idx]` (layer-mapping-aware, falls back to `[0]`). Gated by `--sre_use_projector` (default True). Crop is kept only as no-projector fallback.
- [x] **Task 5: Joint criterion + CE-only baseline.** New `src/criterions/ce_only.py` and `src/criterions/unit_aligned.py`. `UnitAlignedDistillationCriterion` runs **one** student/teacher forward pass and calls `SRECriterion._knowledge_distillation_loss` + `EMKDCriterion._response_logit_distillation` + `EMKDCriterion._vision_distillation` directly (no 2× compute). Registered in `src/criterions/__init__.py` under keys `joint`, `unit_aligned`, `unit_aligned_distillation`.
- [ ] **Task 6: UMI computation.** New file `scripts/compute_umi.py` (below). Outputs JSON `{pair_name, vocab_jaccard, vision_token_ratio_mean, umi}` for the Qwen2.5-VL-7B / Qwen2-VL-2B pair on a 5K-response × 1K-image probe.
- [ ] **Task 7: Training scripts.** Four scripts under `script_train/`:
  - `ce_only/train_qwen25_teacher_7b_qwen2_student_2b_ce_only.sh`
  - existing `em_kd/train_qwen25_teacher_7b_qwen2_student_2b_emkd.sh` — sanity-check args still parse
  - existing `train_qwen25_teacher_qwen2_student_lora_test.sh` — bumped to a real SRE run (rename to `sre/train_qwen25_teacher_7b_qwen2_student_2b_sre.sh`)
  - `unit_aligned/train_qwen25_teacher_7b_qwen2_student_2b_unit_aligned.sh` (new)
  All four point at the same `data_path`, `image_dir`, and use identical `lora_r=64`, `lora_alpha=64`, `per_device_train_batch_size=1`, `grad_accum=8`, `lr=1e-5`, `warmup_ratio=0.03`, `num_train_epochs=1`, `image_resolution=mid`. Only `--kd_loss_type` and method-specific weights differ.

**Day-1 gate:** `python3 -m py_compile src/...` clean ✅; UMI script returns a number for the pair; one training script smoke-runs `--percent_data 0.001` for 5 steps without crashing.

### Day 2 (May 16) — Teacher cache + SFT launch

- [ ] **Task 8: Cache teacher final-layer hidden states + vision-logit-projected outputs** for the 100K-sample subset we'll use. Saves ~70% teacher cost across the 3 KD methods. One-time ~12-18h job; can run overnight if it starts early.
- [ ] **Task 9: Launch CE-only baseline.** `bash script_train/ce_only/...sh` with `--percent_data 1.0`. Expected ~10-14h on Mac Studio M3 Ultra at LoRA r=64 with 100K samples.
- [ ] **Task 10: Write paper skeleton.** intro + related work + UMI definition + method paragraph + table stubs. Keep at 4 pages (short paper). Cite SRA (arXiv 2605.01205), EM-KD (arXiv 2511.21106), GenRecal (arXiv 2506.15681), LLaVA-KD (arXiv 2410.16236).

**Day-2 gate:** teacher cache running; CE-only training running; paper skeleton committed.

### Day 3 (May 17) — EM-KD-only

- [ ] **Task 11: CE-only checkpoint saved**, sanity-check it generates coherent text on 4 sample prompts.
- [ ] **Task 12: Launch EM-KD-only run.** Same config as CE-only + `--kd_loss_type emkd`. Uses cached teacher.
- [ ] **Task 13: VLMEvalKit setup.** Install, pin commit, smoke-test on the CE-only checkpoint with MMB-EN dev (≤500 samples for speed). Document judge model (Qwen2.5-VL-72B as open fallback).

**Day-3 gate:** CE-only on MMB-EN dev → number lands within ±2 pts of upstream Qwen2-VL-2B-Instruct SFT-on-this-data baseline (smoke for eval correctness).

### Day 4 (May 18) — SRE-only

- [ ] **Task 14: EM-KD-only checkpoint saved**, generation smoke test.
- [ ] **Task 15: Launch SRE-only run.** `--kd_loss_type sre --teacher_layer_mapping "[27]" --student_layer_mapping "[27]" --sre_use_projector true`. Single last-layer mapping ⇒ exactly 1 student→teacher projector wired up.
- [ ] **Task 16: Begin CE-only evals on MMStar, MMMU-val, MathVista-MINI.** Run sequentially; ~3-4h per benchmark.

### Day 5 (May 19) — Unit-Aligned

- [ ] **Task 17: SRE-only checkpoint saved.**
- [ ] **Task 18: Launch Unit-Aligned run.** `--kd_loss_type unit_aligned --teacher_layer_mapping "[27]" --student_layer_mapping "[27]" --joint_ce_weight 0.5 --joint_emkd_weight 1.0 --joint_sre_weight 1.0`.
- [ ] **Task 19: EM-KD-only evals begin.**

### Day 6 (May 20) — Eval + Results

- [ ] **Task 20: Unit-Aligned checkpoint saved.**
- [ ] **Task 21: SRE-only and Unit-Aligned evals.** All four checkpoints × four benchmarks = 16 cells in main table.
- [ ] **Task 22: Generate Table 1 (UMI per pair, 1 row), Table 2 (method × benchmark, main result).**
- [ ] **Task 23: Component-removal ablation (1 table, 4 rows).** From Unit-Aligned: drop SRE → "EM-KD only" (reuse Day-3 ckpt); drop EM-KD → "SRE only" (reuse Day-4 ckpt); drop projector → "Unit-Aligned no-proj" (extra small ablation run, 100 steps fine-tune from joint ckpt); full Unit-Aligned. No new training needed for first 3 rows since they're already in main table; only 1 extra ablation run for no-proj.

### Day 7 (May 21) — Write + buffer

- [ ] **Task 24: Write results + discussion sections.** Lead with the unit-mismatch framing (cite UMI from Day 1). Honest acknowledgement: same-family pair = low UMI; gain expectation modest; cross-architecture pairs are future work.
- [ ] **Task 25: Camera-ready polish.** Tighten to 4 pages content + refs. Make sure GenRecal is cited and differentiated in related work (avoid auto-reject).

### Day 8 (May 22) — Submit

- [ ] **Task 26: Final reproducibility checklist.** Pin VLMEvalKit commit in `phase_0_sprint_plan.md` (this file) and in the paper. Verify scripts in `script_train/` run end-to-end with documented args. Push branch to remote.
- [ ] **Task 27: Submit to ARR May cycle.**

---

## The single-pair training matrix

All four runs share these flags (only `--kd_loss_type` and method-specific weights differ):

```bash
STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.05
PER_DEVICE_BS=1
GRAD_ACCUM=8        # effective batch size 8 on 1 GPU
LR=1e-5
WARMUP=0.03
EPOCHS=1
MAX_LEN=2048
IMAGE_RESOLUTION=mid
PERCENT_DATA=0.15   # ~100K of llava_v1_5_mix665k
SEED=1337
```

| Method | `--kd_loss_type` | Extra args |
|---|---|---|
| CE-only baseline | `ce_only` | — |
| EM-KD-only | `emkd` | (uses CLI defaults for `em_kd_alpha/beta/gamma/temperature`) |
| SRE-only | `sre` | `--teacher_layer_mapping "[27]" --student_layer_mapping "[27]" --sre_use_projector true` |
| Unit-Aligned | `unit_aligned` | same SRE mapping + `--joint_ce_weight 0.5 --joint_emkd_weight 1.0 --joint_sre_weight 1.0` |

Layer index `27` is the last hidden layer for Qwen2-VL-2B (28 layers). Confirm against `model.config.num_hidden_layers` at run start.

---

## Headline metric & decision criterion

**Main claim attempted in the short paper:**

> On the Qwen2.5-VL-7B → Qwen2-VL-2B pair (same-family, low UMI), joint unit-aligned distillation matches or exceeds the better of EM-KD-only and SRE-only on ≥3 of 4 benchmarks, and ablations show the projector-based hidden alignment (vs `_crop_last_dim`) contributes ≥0.3 pts averaged.

**Submit if:** Unit-Aligned ≥ max(EM-KD-only, SRE-only) − 0.3 pts averaged, AND Unit-Aligned > CE-only averaged. (Even a small positive delta is publishable as a short paper when supported by clean ablations and the projector contribution.)

**Reframe before submitting if:** Unit-Aligned < CE-only. Rewrite to a robustness / negative-result framing — *"joint training is the only method that does not collapse on benchmark X"* — and ship to a workshop instead of ARR.

**Abandon and pivot if:** all KD methods underperform CE-only by ≥1 pt. Add a fourth, cross-architecture pair (Qwen3-VL-8B → Qwen2.5-VL-3B) and target ARR June.

---

## Bug-fix summary (this commit)

| File | Change |
|---|---|
| `src/arguments.py` | `kd_loss_type` default `"contrastive_rkd"` → `"ce_only"`; removed dead `rkd_distance_weight`, `rkd_angle_weight`; added `sre_use_projector`, `joint_ce_weight`, `joint_emkd_weight`, `joint_sre_weight`. |
| `src/criterions/em_kd.py` | `EMKDCriterion.__init__` now reads α/β/γ/T from args via `getattr`. Adds optional `em_kd_max_vision_tokens` / `em_kd_max_text_tokens` budget caps to keep Hungarian tractable on high-res inputs. Splits forward into `forward()` + `compute_losses()` so the joint criterion can reuse the forward pass. |
| `src/criterions/sre.py` | New `_align_hidden_dims(student_h, teacher_h, projector)` replaces inline `_crop_last_dim` in cosine + geometry losses. `SRECriterion._get_projector(distiller, idx)` resolves the right `distiller.projectors[idx]`. Span-loss and span-pooled-hidden-loss pass the per-layer projector. Forward split into `forward()` + `compute_losses()`. |
| `src/criterions/ce_only.py` | New — `CEOnlyCriterion` (baseline that keeps teacher in batch but discards KD signal). |
| `src/criterions/unit_aligned.py` | New — `UnitAlignedDistillationCriterion` (joint SRE + EM-KD, **shared single forward pass**, configurable CE/EMKD/SRE weights). |
| `src/criterions/__init__.py` | Registry updated. Keys: `ce_only`, `default`/`default_distillation`, `emkd`/`em_kd`, `sre`, `joint`/`unit_aligned`/`unit_aligned_distillation`. Empty-string fallback now resolves to `ce_only`. |

**Smoke test:** `python3 -m py_compile` clean on all 7 files.
