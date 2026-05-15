# Unit-Aligned Distillation for Cross-Architecture VLMs

## Publication-Readiness Audit + Phased Execution Plan (EMNLP 2026 / AAAI 2027)

---

## TL;DR

- **Your paper is publishable but only if reframed around a *measurable* cross-architecture regime and a joint-training non-additivity claim.** As-is ("SRE + EMKD = paper") it will be rejected as the union of SRA (arXiv 2605.01205), EM-KD (arXiv 2511.21106), and GenRecal (arXiv 2506.15681). The defensible novelty is (a) the *first* joint text-span + vision-Hungarian framework for VLM KD, (b) a quantitative **Unit-Mismatch Index** that predicts when joint training beats single-side methods, and (c) demonstration that text- and vision-side mismatches *interact non-additively*.
- **Three primary threats already exist on arXiv**: EM-KD owns Hungarian-matched vision-logit KD + VL affinity; SRA owns attention-weighted span CoM + geometric regularizer; GenRecal owns the cross-architecture VLM distillation framing itself via a learned Recalibrator. No paper combines text-span and vision-Hungarian alignment in one VLM framework — that empty cell is your wedge.
- **Decisive minimum experiment on a single Mac Studio M3 Ultra (512GB unified memory, 819 GB/s)**: three teacher/student pairs (Qwen2.5-VL-7B→Qwen2-VL-2B same-family, Qwen3-VL-8B→Qwen2.5-VL-3B cross-tokenizer-cross-vision, LLaVA-OneVision-7B→Qwen2.5-VL-3B cross-family), LoRA students, ~500K-sample Cambrian-7M subset, evaluated on MMB / MMStar / MMMU / MathVista / MM-Vet / AI2D / ChartQA / OCRBench / RealWorldQA / BLINK / POPE, with UMI reported per pair and compute-matched baselines (SFT, LLaVA-KD, EM-KD-only, SRE-only, GenRecal-style implicit-bridge). If joint dual-granularity beats the better of EM-KD-only / SRE-only by ≥1.5 pts averaged across these benchmarks on the two cross-architecture pairs (and matches on the same-family pair), submit.

---

## Key Findings

1. **GenRecal (arXiv 2506.15681, Lee, Hachiuma, Ro, Wang, Wu — NVIDIA / KAIST / National Taiwan University, June 2025) is the most dangerous prior art** and was likely not in your threat model. It already claims "general-purpose distillation for VLMs … across different types of VLMs … built on different LLMs and employ varying token types — differing in vocabulary size, token splits, and token index ordering." However, its mechanism is fundamentally different: a two-decoder-block "Recalibrator" between student question-features and teacher answer-features, supervision through the teacher's VLM-head producing "Recalib-based logits" with KL to teacher logits + a fragile regularization term (without which MM-Vet drops 73.2→63.5, MMMU 68.1→58.9 per Table 3). **No per-token text alignment, no spatial vision-token matching.** Authors explicitly admit: "the current GenRecal framework focuses on distilling final-layer features, potentially missing out on finer-grained knowledge captured in intermediate layers."

2. **EM-KD (arXiv 2511.21106, AAAI 2026) owns the vision-side Hungarian story.** Manhattan-distance cost on per-side LM-head-projected vision logits, Hungarian linear-sum assignment, reverse-KL on aligned vision logits (VSD), smooth-L1 on text↔vision affinity matrices (VLAD). Your EMKD module is functionally a re-implementation. You **cannot** claim Hungarian-matched vision-logit KD or VL affinity as new. EM-KD assumes *shared vocabulary*, which is your one defensible delta.

3. **SRA (arXiv 2605.01205, May 2026) owns the text-side span story.** Tokenizer-agnostic spans via character-boundary alignment, attention-weighted Center-of-Mass span representations, geometric regularizer, aligned span logit distillation, framed as a Multi-Particle Dynamical System. Your SRE module is functionally equivalent.

4. **No 2025–2026 paper combines text-span and vision-token matching for VLMs.** This is the empty cell. Related but non-overlapping: CTPD (arXiv 2601.11865), DWA-KD (arXiv 2602.21669), ALM (arXiv 2503.20083), Align-TI (arXiv 2602.09483), HKD4VLM (arXiv 2506.13038), Masters (arXiv 2512.22238).

5. **Qwen3-VL (arXiv 2511.21631, 26 November 2025)** is released in 2B/4B/8B/32B dense + 30B-A3B/235B-A22B MoE, Instruct & Thinking. Uses the **Qwen3 tokenizer** (not Qwen2.5's), **different vision-patch quantization** (multiples of 32 vs Qwen2.5-VL's 28), plus **DeepStack** multi-layer vision-token injection and 256K native context. Qwen3-VL ≠ Qwen2.5-VL on *both* tokenizer and vision-token layout — uncontested empirical territory that postdates GenRecal.

6. **Your stated weaknesses in code (`_crop_last_dim`, unread α/β/γ, "contrastive_rkd" default, dead RKD args) are reviewer-bait — fix three, drop two, before submission.**

---

## 1. Prior-Art Landscape (2024–2026)

**Cross-tokenizer LLM distillation:**
- **MiniLLM** (arXiv 2306.08543) — reverse-KL + on-policy. Same-tokenizer only. Foundational reference.
- **DistiLLM-2** (arXiv 2503.07067, ICML 2025 Oral) — skewed-KL with contrastive teacher/student data synergy. Cite as the reverse-KL baseline you build on.
- **ULD** (arXiv 2402.12030, Boizard et al.) — token-wise OT on padded logits.
- **MultiLevelOT** (arXiv 2412.14528) — Sinkhorn at token + sequence levels.
- **DSKD / DSKDv2** (arXiv 2406.17328 EMNLP'24 + arXiv 2504.11426 v2) — cross-model attention projector unifying output spaces. Strongest LLM-side cross-tokenizer SOTA pre-SRA.
- **MinED** — Levenshtein-minimum greedy vocabulary alignment + per-position KL.
- **ALM** (arXiv 2503.20083, ICLR 2026) — approximate likelihood matching over chunks.
- **DWA-KD** (arXiv 2602.21669) — Soft-DTW on embedding & final-hidden states.
- **CTPD** (arXiv 2601.11865, AAAI 2026) — aligned-span lattice for preference distillation.
- **SRA** (arXiv 2605.01205, May 2026) — direct overlap with your SRE.
- **Byte-Level Distillation** (arXiv 2604.07466, 2026) — baseline for tokenizer-free transfer.

**VLM / MLLM distillation:**
- **LLaVA-MoD** (arXiv 2408.15881, ICLR 2025) — MoE student + mimic-then-DPO.
- **LLaVA-KD** (arXiv 2410.16236, ICCV 2025) — MDist + RDist + 3-stage curriculum. Canonical same-tokenizer MLLM KD baseline.
- **EM-KD** (arXiv 2511.21106, AAAI 2026) — direct overlap with your EMKD.
- **GenRecal** (arXiv 2506.15681) — your most dangerous direct competitor.
- **Masters** (arXiv 2512.22238) — magnitude-mask + offline RL; same-family.
- **HKD4VLM** (arXiv 2506.13038) — hybrid progressive KD for hallucination/factuality.
- **Align-TI** (arXiv 2602.09483) — IVA + TPA; same-tokenizer.
- **FastVLM** (arXiv 2412.13303, Apple, CVPR 2025) — FastViT-HD encoder, 4× fewer tokens than FastViT, 16× fewer than ViT-L/14 at 336px.
- **LLaVA-Mini** (arXiv 2501.03895, ICLR 2025) — collapse to 1 vision token. Useful as "high-imbalance" student.

**Vision-token assignment / compression touching Hungarian or OT:** EM-KD is the only KD work using Hungarian on vision logits. Selective Sinkhorn Routing, LUVC, Fourier-VLM, VoCo-LLaMA/HTC-VLM, LLaVA-PruMerge, MQT-LLaVA, ATP-LLaVA, VScan — all are inference compression/pruning, not KD. Cite as motivation for vision-token imbalance.

---

## 2. Positioning Refinement

**Why "SRE + EMKD = paper" is dead on arrival.** SRA covers SRE, EM-KD covers EMKD, GenRecal covers cross-architecture VLM framing. A reviewer with one afternoon and Google Scholar will write a 2-paragraph rejection.

**Three defensible reframings (use all three together):**

(a) **Reframe as "Unit-Aligned Distillation for the Cross-Architecture VLM Regime."** Define a measurable **Unit-Mismatch Index (UMI)** per teacher/student pair:

```
UMI = 0.5 · (1 − shared-vocab Jaccard on tokens-actually-emitted in fixed instruction-tuning corpus)
    + 0.5 · (1 − min(n_v^s, n_v^t) / max(n_v^s, n_v^t) averaged over held-out image set)
```

The paper's main empirical claim becomes: *gains from dual-granularity alignment over the best single-side method scale monotonically with UMI, and exceed the additive sum of text-only and vision-only gains when both components of UMI are large.* Falsifiable, paper-defining, not made by SRA / EM-KD / GenRecal.

(b) **Reframe span alignment as "assistant-response unit alignment under ChatML gating with shared-vocabulary anchoring,"** emphasizing response-region-only operation (labels ≠ −100). Cite SRA as "concurrent and complementary" but distinguish: (i) you anchor span pooling to the *last-query-token's causal attention* over assistant tokens (asymmetric instruction-conditioned weighting SRA's symmetric CoM cannot express); (ii) shared-subvocabulary KL soft-target avoiding OT machinery.

(c) **Reframe the vision side as "vocabulary-anchored vision-token matching with non-shared-head support."** EM-KD assumes shared vocabulary. Your contribution: vocab-cropped projection via *each side's own LM head*, so the method works when student and teacher have **different** vocabularies. Narrow, defensible delta — exactly what GenRecal's setting demands.

**Conceptually new claim (centerpiece):** *Text-side and vision-side unit mismatches interact.* In the diagnostic ablation, when both shared-vocab coverage is low AND vision-token-count ratio is far from 1, single-side methods *under-perform their isolated regime*, but joint method recovers the gain. This non-additivity is what makes the paper more than the sum of SRA + EM-KD.

**Recommended title:** *"Unit-Aligned Distillation: Joint Text-Span and Vision-Token Correspondence for Cross-Architecture Vision-Language Model Distillation."* Subtitle in abstract: "We propose the first joint cross-tokenizer span alignment and Hungarian vision-token assignment for VLM distillation, and introduce a Unit-Mismatch Index that quantitatively defines the cross-architecture regime where joint training beats both single-side methods."

**Abstract structure (200 words):** (1) Two sentences: current VLM KD assumes shared vocabulary and matched vision-token counts; modern open-source pairs violate both. (2) One sentence introducing UMI. (3) Two sentences describing method — assistant-response span alignment with attention-weighted pooling and shared-vocab logit anchoring + Hungarian vision-token matching on vocab-cropped logits + VL affinity. (4) One sentence on joint training. (5) Two sentences on results — point estimates on 3 pairs, gain vs EM-KD-only and SRA/SRE-only, non-additivity finding correlated with UMI. (6) Code/checkpoints release.

---

## 3. Experiment Design Refinement

### Teacher / student pairs — concrete HF IDs

| Pair type | Teacher | Student | UMI character |
|---|---|---|---|
| Same-family-larger | `Qwen/Qwen2.5-VL-7B-Instruct` | `Qwen/Qwen2-VL-2B-Instruct` | low (different ViT: window-attn vs naive-dynamic; same Qwen2 BPE) |
| Same-family-newer | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen2.5-VL-3B-Instruct` | high (Qwen3 vs Qwen2.5 BPE; patch multiple 32 vs 28; DeepStack vs single-layer) |
| Cross-family LLaVA→Qwen | `lmms-lab/llava-onevision-qwen2-7b-ov` | `Qwen/Qwen2.5-VL-3B-Instruct` | medium-high (SigLIP+AnyRes 9 patches × 729=6,561 max; Qwen2.5 native-dynamic with window-attn) |
| FastVLM→Qwen | `apple/FastVLM-7B` (FastViT-HD + Qwen2-7B base) | `Qwen/Qwen2.5-VL-3B-Instruct` | medium (shared Qwen2 tokenizer family; ~256 vs >2,000 vision tokens) |
| Vision-token-imbalance | `lmms-lab/llava-onevision-qwen2-7b-ov` (up to ~7,290 tokens) | `Qwen/Qwen2-VL-2B-Instruct` (efficient packing) | high vision-only |
| Cross-architecture stress | `OpenGVLab/InternVL3-8B` or `OpenGVLab/InternVL3-5-8B` | `Qwen/Qwen2.5-VL-3B-Instruct` | high (InternViT vs Qwen-ViT; tokenizers differ on long-tail tokens) |

For each pair document: (i) tokenizer vocab size, (ii) shared-vocab Jaccard on response corpus, (iii) min/max/mean vision-token count over 1K-image probe.

### Vision-token-count cheat sheet
- **Qwen2-VL**: 2D-RoPE, 2×2 MLP-merge → variable, 448×448 image ≈ 256 tokens before merge, 64 after.
- **Qwen2.5-VL**: window-attention in 112×112 windows (8×8 patches), 2×2 merge.
- **Qwen3-VL**: SigLIP-2 encoder + 2×2 MLP merger + DeepStack (3 ViT depths injected to first 3 LLM layers); pixel rounding to multiples of 32; native 256K context.
- **LLaVA-OneVision**: SigLIP at 384 → 729 tokens per crop, AnyRes-max-9 → up to 10×729 = 7,290 tokens per image.
- **LLaVA-NeXT**: 2,880 tokens for 4-crop 672×672.
- **FastVLM**: hierarchical FastViT-HD, 4× fewer than FastViT, 16× fewer than ViT-L/14 at 336px.
- **InternVL2.5/3**: pixel-shuffle dynamic crops to 448 tile, 256–1,024 tokens.

### Control experiments

- **C1 — same-tokenizer-different-vision:** Qwen2-VL-7B → Qwen2-VL-2B at different image-resolution settings. Isolates vision-side mismatch.
- **C2 — same-vision-encoder-different-LLM:** LLaVA-OneVision-7B (Qwen2 LLM) → Vicuna-LLM reimplementation if available, or two LLaVA-NeXT variants with different LLM backbones but identical SigLIP+AnyRes. Isolates text-side mismatch.
- **C3 — shared-vocabulary degradation:** randomly mask k% of shared-vocab entries from SRE logit KL and plot gain curve as k → 0.
- **C4 — vision-token count degradation:** synthetically downsample teacher vision tokens to match student's count (no Hungarian needed). Measures gain attributable purely to *matching* (not supervision signal).
- **C5 — interaction grid:** factorial 2×2 on (text-mismatch high/low) × (vision-mismatch high/low) using the 6 pairs. Fit linear model `Δ = β₀ + β_t·UMI_text + β_v·UMI_vision + β_tv·UMI_text·UMI_vision`. Interaction term β_tv > 0 is your *new* finding.

### Training data
- **Stage A (alignment / DPT analog):** 500K-sample stratified subset of **Cambrian-7M** (`nyu-visionx/Cambrian-10M`, filtered).
- **Stage B (joint distillation / DFT analog):** 200K-sample mix of **LLaVA-OneVision data** (3.2M single-image SFT pool; sample 150K) + **Eagle-1.8M** OCR/chart subset (50K).
- Avoid LLaVA-Instruct-150K as primary corpus; report it as small-scale legacy comparison only.

### Benchmark sets
- **Minimal (must-have) — 8 benchmarks:** MMBench-EN-dev, MMStar, MMMU-val, MathVista-MINI, MM-Vet, AI2D, ChartQA, POPE.
- **Broad (camera-ready):** + MMMU-Pro, BLINK, MMVP, RealWorldQA, OCRBench, DocVQA-val, InfoVQA-val, SEED-Bench-2-Plus, HallusionBench, MMT-Bench.
- **Skip:** ScienceQA-Image (saturated), TextVQA (redundant with OCRBench), RefCOCO unless claiming grounding gains.
- **Run via VLMEvalKit** with documented judge model (GPT-4o-2024-08-06 ideally; Qwen2.5-VL-72B as open-judge fallback).

### Inference-efficiency table (2026 expectation)
- TTFT at 1024×1024 (ms, A100 80GB *and* Mac Studio M3 Ultra).
- Decoding tokens/sec at 256-token output.
- Peak GPU memory.
- Total prefill FLOPs (`torch.profiler` or `fvcore`).
- KV-cache footprint at 4K context.

### Compute-matched baselines (pre-empt "trained for longer")
1. Report wall-clock training time and total tokens-seen for every method.
2. SFT at same wall-clock as your method.
3. LLaVA-KD MDist+RDist at compute-matched.
4. EM-KD only (your EMKD without SRE).
5. SRE only (your SRE without EMKD).
6. **GenRecal-style implicit bridge** ablation: small MLP projector from student final-hidden to teacher's vocab dim + KL on response logits, compute-matched. Won't beat full GenRecal (256× A100, ZeRO-3, 9M samples), but will beat fair-compute approximation.

---

## 4. Ablation Study Design

### Text-side (which SRE component matters)
- **T1** LCS over offset_mapping vs character-level Smith-Waterman vs byte-level vs DTW vs no-alignment-uniform-chunking. *Does alignment quality matter or only span existence?*
- **T2** Pool by uniform mean / max / last-token-only / attention-weighted (yours) / cross-attention-from-query-to-span. *Does instruction-conditioned weighting beat content-only?*
- **T3** Loss components: span-cosine only / +geometry MSE / +shared-vocab logit KL / all three. Full 2³ factorial. *How much does each add?*
- **T4** Per-layer schedule (1,1,2,2,3,3,4,4,5,5,8,10) vs uniform=1 vs learned-scalars vs only-last vs only-middle-third. *Is the late-layer bias load-bearing?*
- **T5** Shared-vocab KL: full-vocab forward-KL / cropped-to-min / shared-subset-only (yours) / no-logit. *Is the shared-subset restriction stable?*
- **T6** Span granularity: char / word / subword / sentence. *Which granularity preserves teacher signal best?*
- **T7** Span-density gain curve: bucket samples by (#aligned spans / #response tokens), plot ΔAccuracy. *Does SRE help most where alignment is densest?*

### Vision-side (which EMKD component matters)
- **V1** Assignment: Hungarian / nearest-neighbor / Sinkhorn τ ∈ {0.1, 0.5, 1.0} / random / identity-when-equal.
- **V2** Cost: L1 / L2 / cosine / KL on softmaxed logits / Wasserstein.
- **V3** Projection: vocab-projected via own LM head (yours) / raw hidden / SigLIP-projected back to image space / shared learned projector.
- **V4** Affinity-matrix loss on/off; affinity to assistant-only / all-text / user+assistant.
- **V5** Vision-logit KL direction: forward / reverse (yours) / JS / skewed-KL.
- **V6** Matching density: full bipartite / top-k by margin / threshold on cost.

### Interaction / joint (paper-making ablations)
- **J1** Text-only / vision-only / joint. *Headline table.*
- **J2** Sequential (text → vision two-stage) vs joint. *Predicted: joint > sequential when UMI is high; equivalent when low.*
- **J3** Weight-balance sweep: λ_sre / λ_emkd ∈ {0.1, 0.3, 1, 3, 10} grid. *Is joint training brittle?*
- **J4** Non-additivity test: across pairs, Δ_joint − (Δ_text + Δ_vision). Plot vs UMI. *Main scientific claim.*

### Diagnostics
- **D1** Per-sample: cosine similarity between span-density rank and improvement rank (Spearman).
- **D2** Hungarian-cost-distribution histograms before vs after training: method *resolves* mismatch rather than working around it.
- **D3** Visualize matched vision-token pairs back to image patches for qualitative figure.

### Robustness
- **R1** Compute-matched runs.
- **R2** LoRA rank ∈ {16, 32, 64, 128} and full-finetune.
- **R3** Seed × 3 (1337, 42, 2026).
- **R4** Order-of-data ablation if using curriculum.
- **R5** Sensitivity to ChatML formatting edge cases.

---

## 5. Weakness Triage

| Issue | Verdict | Action |
|---|---|---|
| SRE `_crop_last_dim` for hidden-dim mismatch | **Fix-and-keep** | Replace with 1-layer linear projector per-side (student→shared d, teacher→shared d), trained jointly. Add ablation: projector vs crop, expect ≥0.5pt gain. |
| EMKD hard-coded α=0.5, β=0.25, γ=25.0, T=1.0 | **Fix-and-keep** | Read from args; report swept values; show chosen point is robust within ±50% sensitivity. |
| Default `kd_loss_type = "contrastive_rkd"` unregistered | **Fix-and-keep** | Change default to `"ce_only"`. If reviewers run your code and it crashes, you're done. |
| Dead RKD args (`rkd_distance_weight`, `rkd_angle_weight`) | **Drop-and-acknowledge** | Remove from argparse. Never mention in paper. |
| Hard-coded per-layer schedule (1,1,2,2,3,3,4,4,5,5,8,10) | **Fix-and-keep** | Parameterize with (start-weight, end-weight, schedule_type ∈ {linear, exponential, learned}). Add T4 ablation showing choice is non-critical. |

---

## 6. Expected Reviewer Attacks

1. **"This is just SRA + EM-KD with a project name change."** → UMI/non-additivity claim + Table comparing single-side vs joint with interaction coefficient β_tv reported.
2. **"GenRecal already does cross-architecture VLM distillation."** → Differentiate on (a) no need for teacher's VLM-head at inference, (b) explicit token correspondences vs implicit decoder bridge, (c) intermediate-layer alignment (GenRecal's explicit limitation), (d) demonstrated on Qwen3-VL/InternVL3 pairs that postdate GenRecal.
3. **"Why not Optimal Transport instead of Hungarian?"** → Sinkhorn ablation V1. Hungarian within noise of Sinkhorn at τ=0.5 but 5× faster on n=512 tokens.
4. **"Why not just use DSKD-CMA / ALM?"** → T5 shows shared-subvocab KL matches DSKD-CMA and beats ULD/ALM at compute-matched, *without* learnable cross-model attention parameters (parameter-free at inference).
5. **"Trained for longer than baselines."** → Compute-matched table. Wall-clock and tokens-seen.
6. **"LoRA limits the conclusion."** → R2 ablation with full-finetune on smallest pair; direction preserved.
7. **"Only Qwen-family results."** → LLaVA-OneVision → Qwen and InternVL3 → Qwen pairs in main table.
8. **"Reverse-KL on vision logits is a re-implementation of EM-KD."** → Honest acknowledgement + credit + differentiation: EM-KD requires shared vocab; you support different-vocab teachers. Row in Table 1: "EM-KD as published (requires shared vocab): N/A on Qwen3-VL→Qwen2.5-VL pair."
9. **"LCS over offset_mapping is brittle on non-Latin scripts."** → T1 ablation includes character-level alternatives; script-stratified breakdown on multilingual subset.
10. **"γ=25 is unprincipled."** → V4 sensitivity sweep + learnable-γ version.
11. **"Benchmark suite is cherry-picked."** → Full VLMEvalKit minimal+broad set, pre-registered.
12. **"Doesn't show this works on a frontier model."** → Optional 8B-student headline pair (Qwen3-VL-8B from Qwen3-VL-32B) even at one seed.

---

## 7. Recommended Minimal Decisive Experiment

Single-node Mac Studio M3 Ultra (March 2025) configured with up to 512GB unified memory at 819 GB/s + modest cloud spillover (1× H100 80GB for 2–3 short windows for teacher-forward batching).

**Models (3 pairs):**
1. Qwen2.5-VL-7B-Instruct → Qwen2-VL-2B-Instruct (low UMI control)
2. Qwen3-VL-8B-Instruct → Qwen2.5-VL-3B-Instruct (high UMI, novel territory)
3. llava-onevision-qwen2-7b-ov → Qwen2.5-VL-3B-Instruct (cross-family medium-high UMI)

**Training:** LoRA r=64 on student LLM + vision-language projector; 1 epoch on 300K Cambrian-7M-subset + 100K LLaVA-OV-subset; teacher inference offloaded in chunks to disk-cache logits + final-layer hidden states + vision-projected logits for 100K representative samples (saves ~70% teacher cost).

**Methods × pairs grid (5 × 3 = 15 main runs):**
- SFT baseline
- LLaVA-KD-style MDist+RDist (compute-matched re-implementation)
- EM-KD-only
- SRE-only
- Joint = SRE + EMKD (yours)

**Evaluation:** 8-benchmark minimal set via VLMEvalKit + RealWorldQA + BLINK (10 total).

**Diagnostic add-ons:**
- UMI per pair from 1K-image, 5K-response probe.
- Non-additivity table: Δ_joint vs (Δ_SRE-only + Δ_EM-only) per pair.
- Span-density gain curve (T7) on cross-family pair.
- Hungarian-cost-distribution before/after on high-UMI pair.

**Decision threshold:**
- Low-UMI pair: joint within ±0.5 pt of best single-side.
- Two high-UMI pairs: joint ≥ best-single-side + 1.5 pts averaged, positive sign on ≥7 of 10 benchmarks.
- β_tv interaction positive with bootstrap CI not crossing zero.

If all three threshold criteria clear, submit to CVPR/ICCV/NeurIPS/EMNLP/AAAI. If only one high-UMI pair clears, expand to fourth pair before submitting.

---

# Phased Execution Plan (EMNLP 2026 / AAAI 2027)

## Verified submission windows (as of May 14, 2026)

- **EMNLP 2026 via ARR May cycle:** Submission deadline **May 25, 2026** (11 days from now). Reviewer registration May 27. Author response July 7-13. EMNLP commitment ~early August. Conference Oct 24-29, Budapest.
- **EMNLP 2026 via ARR June cycle:** ~late June 2026 submission, ~late July reviews, EMNLP commitment ~Aug 1.
- **AAAI 2027:** Deadlines TBD but historically late July / early August 2026. Conference Feb 16-23, 2027, Montréal. 7-page content limit.

## Three submission strategies

| Strategy | Target | Required by submission | Risk |
|---|---|---|---|
| A — Aggressive | ARR May (May 25) | 1 pair + slim short paper | High (30-40% acceptance); "why only one pair?" |
| B — Pragmatic | ARR June (~June 25) → EMNLP commit | 3 pairs + core ablations + UMI validated | Moderate; standard publishable unit |
| C — AAAI 2027 | ~late July / early August | Full grid + ablations + frontier pair | Lowest risk; tighter writing window (7 pages) |

**Recommendation: Plan Strategy B as default, Strategy A as Phase 1 stretch goal, Strategy C as fallback if Phase 2 reveals method issues.**

## Phase 0 — Infrastructure (May 14-20, 7 days) — non-negotiable

| Work | Hours |
|---|---|
| Fix 4 reviewer-bait bugs: `kd_loss_type` default, hard-coded EMKD α/β/γ, dead RKD args, `_crop_last_dim` → paired linear projectors initialized from teacher word-embedding subspace | 8h |
| Implement UMI computation: shared-vocab Jaccard on 5K-response probe + vision-token count ratio on 1K-image probe | 4h |
| Compute UMI for 3 priority pairs (Q2.5-7B→Q2-2B, Q3-8B→Q2.5-3B, LLaVA-OV-7B→Q2.5-3B) | 2h |
| Stand up VLMEvalKit pinned commit; smoke-test on 1 student across 4 benchmarks (MMB, MMStar, MMMU, MathVista) | 6h |
| Build teacher-forward caching: pre-compute and disk-cache teacher final-layer hidden states + vision-projected logits for 100K sample subset (saves ~70% teacher cost across 5 method variants) | 8h |
| Pre-write paper skeleton: intro, related work (SRA, EM-KD, GenRecal, LLaVA-KD, DSKD), UMI definition, method section. Leave results blank. | 8h |

**Phase 0 deliverable:** Working pipeline + UMI numbers + paper skeleton committed.

**Gate 0 (May 20):** Can you run a full SFT training + eval cycle end-to-end on one pair without crashes? If no, Strategy A is dead — move to Strategy B with Phase 1 starting May 21.

## Phase 1 — Minimum-Viable Result (May 20-25, 5 days)

Single pair, single decisive table.

| Work | Notes |
|---|---|
| Train 5 methods on **Qwen2.5-VL-7B → Qwen2-VL-2B** (lowest UMI, fastest, most cached teacher data): SFT, LLaVA-KD-repro, EM-KD-only, SRE-only, Joint | LoRA r=64; 100K samples; ~12-18h each on Mac Studio M3 Ultra with teacher cached |
| Eval all 5 on 4 benchmarks: MMB-EN, MMStar, MMMU-val, MathVista-MINI | ~3-4h per checkpoint |
| Draft Tables 1, 2 (UMI numbers per pair + method comparison) + results paragraph | In parallel with training |

**Phase 1 deliverable:** Table 1 row 1 (low-UMI pair, 5 methods × 4 benchmarks). UMI for all 3 pairs reported.

**Gate 1 (May 25, ARR May deadline):**
- **Strategy A trigger:** If joint > best-single-side by ≥1.5pt averaged AND positive on ≥3/4 benchmarks → submit slim short paper to ARR May.
- **Default:** Submit to ARR June cycle. Continue to Phase 2.

> ⚠️ Single-pair + 4 benchmarks is short-paper quality. Strategy A is 30-40% acceptance probability. Strategy B with 3 pairs is the stronger first submission.

## Phase 2 — Core Empirical Story (May 26 - June 22, ~4 weeks)

Three pairs, joint vs single-side, first ablations.

| Work | Notes |
|---|---|
| Run 5 methods on **Qwen3-VL-8B → Qwen2.5-VL-3B** (high-UMI flagship) | Qwen3 tokenizer + DeepStack + patch-32 vs patch-28 — novel territory |
| Run 5 methods on **llava-onevision-qwen2-7b-ov → Qwen2.5-VL-3B** (cross-family, medium-high UMI) | Most compute-expensive due to 7,290-vision-token teacher forwards |
| Expand eval to 8-benchmark minimal set (add MM-Vet, AI2D, ChartQA, POPE) | VLMEvalKit batch eval |
| **J4 non-additivity test:** Compute Δ_joint vs (Δ_SRE-only + Δ_EM-only) across 3 pairs. Bootstrap β_tv interaction coefficient. | The paper-making analysis |
| **First ablation tier (5 ablations on high-UMI pair):** T3 (SRE component factorial), T5 (shared-vocab KL), V1 (Hungarian vs NN vs Sinkhorn), V4 (affinity on/off), J2 (sequential vs joint) | ~3-4 days compute |
| Write results, discussion, ablation sections | In parallel |

**Phase 2 deliverable:** 3-pair main table on 8 benchmarks + 5 ablations + non-additivity figure. **Minimum publishable unit.**

**Gate 2 (June 22, ARR June deadline ~June 25):**
- **Strategy B trigger:** If 3-pair table shows joint ≥ best-single-side on ≥2 pairs AND β_tv interaction is positive (bootstrap CI doesn't cross zero) → submit long paper to ARR June. EMNLP commitment by ~Aug 1.
- **Fallback:** If only high-UMI pair shows clear gain, reframe as "we identify cross-architecture as a distinct distillation regime." Still submittable.
- **Kill criterion:** If joint < best-single-side on ≥2 pairs, stop and reframe to workshop or robustness paper. Don't waste another month.

## Phase 3 — Full Story for AAAI 2027 (June 23 - July 25, ~5 weeks)

Execute if not submitting to ARR June, OR if ARR June got reviews and AAAI runs as parallel/backup.

| Work | Notes |
|---|---|
| Add 4th pair: **InternVL3-8B → Qwen2.5-VL-3B** OR **FastVLM-7B → Qwen2.5-VL-3B** | Pick based on Phase 2 — InternVL3 for 3-VLM-family claim, FastVLM for efficient-vision-encoder story |
| Expand benchmarks to broad set: MMMU-Pro, BLINK, MMVP, RealWorldQA, OCRBench, DocVQA, HallusionBench | Camera-ready table |
| Second ablation tier: T2 (pooling), T4 (per-layer schedule), T7 (span-density curve), V2 (cost function), V3 (projection), J3 (weight-balance sweep) | ~5 ablations |
| Control experiments: **C1** (same-tokenizer-different-vision), **C3** (shared-vocab degradation curve) | High reviewer-value mechanism isolations |
| Inference-efficiency table: TTFT, decode tok/s, KV-cache, FLOPs on A100 (rent ~$200) and Mac Studio | Required for VLM efficiency papers in 2026 |
| Robustness: 3 seeds on flagship pair, LoRA-rank sensitivity | ~3 days |
| Tighten paper to 7-page AAAI format, write appendix | 1 week |

**Phase 3 deliverable:** Camera-ready-quality paper with 4 pairs × ≥10 benchmarks + ~10 ablations + 2 controls + efficiency table + robustness.

**Gate 3 (July 25, ~AAAI deadline):**
- Submit to AAAI 2027.
- If ARR June got positive reviews, decide between EMNLP commitment (~Aug 1) and AAAI based on review temperature. Check AAAI "substantially different" policy if dual-tracking.

## Phase 4 — Stretch (Optional, August)

Frontier-scale pair (Qwen3-VL-32B → Qwen3-VL-8B) if compute allows. Reviewer-requested revisions.

## Compute Budget Reality Check

| Item | Estimate |
|---|---|
| Per-pair-method run (LoRA r=64, 100K samples, 1 epoch) | ~12-18h on Mac Studio with cached teacher |
| 5 methods × 3 pairs = 15 runs | ~10-12 days serial; ~6 days with smart scheduling |
| Eval (10 benchmarks × ~15 checkpoints via VLMEvalKit) | ~60-80h |
| Teacher caching (one-time) | ~24-36h |
| H100 spillover for 7B+ student finetune | ~$500-1000 over project |

Phase 2 fits comfortably in 4 weeks. Phase 1 in 5 days is tight but doable.

## Recommended execution order

1. **May 14-20:** Phase 0, hard deadline. Bug fixes, UMI, infra. No experiments yet.
2. **May 18-23:** Phase 1 pilot. Start 5 training runs Day 1, write paper skeleton Days 1-3, eval Days 4-5.
3. **May 24-25:** Gate 1 decision. Default = ARR June, not ARR May.
4. **May 26 onward:** Phase 2 in earnest. Start Qwen3-VL teacher caching immediately.
5. **Target ARR June (~June 25) for 3-pair long paper**, AAAI 2027 as parallel/backup.

The discipline that matters most: **don't skip Phase 0**. A bug discovered on May 23 kills EMNLP; the same bug found on May 18 doesn't.

---

## Recommendations (decision-ready, staged)

**Stage 0 — this week:**
1. Acknowledge GenRecal (arXiv 2506.15681) as central competitor in related-work and method comparison sections. Failing to do so is auto-reject.
2. Fix the four code bugs in §5 before any experiments — they contaminate every comparison.
3. Replace `_crop_last_dim` with paired linear projectors; add a 0.1-line projector-init from teacher's word-embedding subspace (DSKD trick) to stabilize training.
4. Rename the paper: *Unit-Aligned Distillation for Cross-Architecture Vision-Language Models.* Use "dual-granularity" as descriptive subtitle.

**Stage 1 — two weeks: Pre-registration and pilot.**
5. Write UMI definition section + non-additivity hypothesis explicitly in paper draft before running experiments. Pre-register the 15-run main grid.
6. Run a 24-hour pilot on Qwen2.5-VL-7B→Qwen2-VL-2B pair with 5K samples to validate plumbing for all 5 methods. If joint training is unstable, fix it now (most likely cause: γ=25 affinity loss dominates; rescale by 1/n_v).

**Stage 2 — four to six weeks: Decisive grid.**
7. Run the 15-run main grid + 8 ablations (T2, T3, T5, V1, V2, V4, J2, J4) at compute-matched.
8. Compute UMI for all pairs at start; report in Table 1.
9. Run non-additivity analysis and decide go/no-go.

**Stage 3 — final two weeks before deadline:**
10. Add 1–2 frontier-scale or stress-test pairs (Qwen3-VL-32B→Qwen3-VL-8B if compute permits; otherwise InternVL3-8B→Qwen2.5-VL-3B).
11. Generate efficiency table on both A100 and Mac Studio.
12. Release code + matched HF checkpoints; pin VLMEvalKit commit.

**Benchmarks that would change the recommendation:**
- If GenRecal-paper-v3 (arXiv 2506.15681 cs.CL, 2 March 2026) appears with explicit cross-tokenizer span alignment, downgrade Stage 0 priority and add head-to-head on shared pairs.
- If a 2026 arXiv paper appears that *exactly* combines text-span + vision-Hungarian for VLMs before your submission, pivot to "first measured Unit-Mismatch Index + first non-additivity demonstration" framing exclusively.
- If single-side methods already outperform joint on the high-UMI pair in pilot, abandon "additive gain" framing and reposition as a *robustness* paper: *"joint dual-granularity is the only method that does not collapse on at least one of the 3 pair types."*

---

## Caveats

- **GenRecal v3 (arXiv 2506.15681) cs.CL revision was posted 2 March 2026**; re-pull the latest before submitting — it may already address intermediate-layer alignment, which would erode differentiation.
- **EM-KD reports its Hungarian-on-vision-logits results on same-family efficient-vs-vanilla pairs**; they do *not* test cross-tokenizer setups. Your cross-tokenizer Hungarian results are genuinely novel territory, but the *method* is a small delta — be honest in framing.
- The **Unit-Mismatch Index** as defined here is a *proposed* construct. Validate predictive validity in the paper (J4 ablation). If β_tv is not significantly positive, positioning is at risk; have fallback narrative (robustness framing) ready.
- **Mac Studio M3 Ultra Metal/MLX-LoRA stacks for VLMs are immature** as of May 2026; 7B-student-LoRA path is fine but full-finetune of 7B will likely require cloud spillover. Budget ~$2K cloud as hedge.
- **Apple's FastVLM (arXiv 2412.13303, CVPR 2025) license/weights checks** — verify before including in main table.
- Treat all reported teacher scores as needing reproduction with your VLMEvalKit pin.
- "Interaction term β_tv > 0" prediction is *not* certain — single-modality KD theory does not strictly predict super-additive gain. The empirical demonstration is the contribution; do not over-claim it as theoretically necessary.
