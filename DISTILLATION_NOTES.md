# VLM Distillation — Approach Notes

Detailed walk-through of the distillation framework in this repo: how training is dispatched, how the data pipeline produces aligned student/teacher batches, what each of the three KD criteria does mathematically, and which design choices are load-bearing vs. vestigial.

All references are `path:line` into the working tree. The pipeline supports six VLM backbones (FastVLM, LLaVA-Next, LLaVA-OneVision, Qwen2-VL, Qwen2.5-VL, Qwen3-VL) and three KD recipes (`default`, `emkd`, `sre`) plus a vestigial `contrastive_rkd` default that does not work as-shipped (see §6).

---

## 1. Overall setup

The goal: train a small student VLM (e.g. `Qwen2-VL-2B-Instruct`) to imitate a larger teacher VLM (e.g. `Qwen2.5-VL-7B-Instruct`) on LLaVA-style instruction data, using LoRA on the student. The teacher stays frozen.

Two modes share the same `train.py` entry point, gated by whether `--teacher_model_name` is set (`train.py:44-45,86-100`):

- **SFT mode** — single `VLMModel`, plain next-token CE on assistant tokens.
- **Distillation mode** — a `Distiller` (`src/distiller.py:40`) owning:
  - `student` (`VLMModel.build`, trainable, optionally LoRA-wrapped)
  - `teacher` (`VLMModel.load`, params frozen, switched to inference mode at `src/distiller.py:76-78`)
  - `projectors` — optional `nn.Linear` modules bridging student → teacher hidden dims, used for per-layer KD
  - plus a separately-constructed **criterion** that owns the loss math.

`DistillTrainer.compute_loss` (`src/trainer.py:79`) detects mode via `isinstance(model, Distiller)` and either:
- calls the model directly and uses `outputs.loss` (SFT), or
- calls `model(criterion, batch)`, which forwards to `criterion(distiller, batch)` (distillation).

This separation means the criterion is the *only* place that knows about teacher forward passes, hidden-state alignment, etc. — the model itself is criterion-agnostic.

---

## 2. Data flow — why labels and offsets matter

Every loss starts with `student_outputs = distiller.student(**student_inputs)` and `teacher_outputs = distiller.teacher(**teacher_inputs)`. Two batch fields are load-bearing.

### Assistant-only labels (`src/data/dataset.py:463-522`)

`_make_labels_chatml` scans the tokenized sequence for `<|im_start|>assistant…<|im_end|>` token spans and masks **everything else with `-100`**. So `student_outputs.loss` is CE on assistant tokens only, and every KD loss can gate "response position" using `labels.ne(-100)`. This works because all six supported backbones use ChatML chat templates.

### Per-tokenizer offset mappings (`src/data/dataset.py:548, 568-569`)

When `kd_loss_type=sre` **and** a teacher is configured, the collator additionally pre-computes a token-span alignment between student and teacher tokenizations via LCS over `offset_mapping` (`_prepare_sre_pooler` at `src/data/dataset.py:103-140`). This produces `pooler_safe_idx` and `pooler_mask` written into both `student_inputs` and `teacher_inputs`. SRE uses these to pool hidden states into "spans of equivalent meaning" across heterogeneous tokenizers. Details in §5.

### Forced model config (`src/model/model.py:65-87`)

`_force_eager_attention` forces `output_hidden_states=True`, `output_attentions=True`, `_attn_implementation="eager"` on the main config and every sub-config (`text_config`, `vision_config`, …). Every KD criterion needs at least hidden states; SRE additionally needs attentions for causal-attention-weighted pooling. Do not disable this.

`padding_side` is forced to `"left"` on config and tokenizer.

---

## 3. The three KD criteria — at a glance

`src/criterions/__init__.py` maps `--kd_loss_type` strings to criterion classes:

| `kd_loss_type` | Class | Key idea |
|---|---|---|
| `default` / `default_distillation` | `DefaultDistillationCriterion` | MSE on projected mean-pooled hidden (with layer mapping) OR temperature-KL on response logits |
| `emkd` / `em_kd` | `EMKDCriterion` | Reverse-KL on response logits + Hungarian-matched vision-logit distillation + vision↔text affinity matrices |
| `sre` | `SRECriterion` | Weighted cosine span loss + geometry MSE on similarity matrices + shared-vocab soft-label KL |

All criteria receive `(distiller, batch)` and return `{"loss": ...}` (additional keys are detached metrics).

### 3a. `default` / `default_distillation`

`src/criterions/default_distillation.py`. Despite the `rkd_distance_weight` / `rkd_angle_weight` args in `arguments.py` (which the current code does not read — see §6), this criterion is **not** RKD. It branches:

- **If `student_layer_mapping` and `teacher_layer_mapping` are both non-empty** (line 64-78): for each mapped (student_layer, teacher_layer) pair, take the layer's hidden states, mean-pool with the attention mask, push the student vector through its corresponding projector, then take MSE against the teacher's mean-pooled vector:

$$
L_{\text{hidden}} = \frac{1}{N}\sum_{i=1}^{N} \big\|\, P_i(\bar h^{S}_i) - \bar h^{T}_i\, \big\|^2
$$

- **Else, if logits exist** (line 80-106): temperature-scaled forward-KL on response positions only:

$$
L_{\text{logit}} = \frac{\sum_t m_t \cdot \text{KL}(\sigma(z^S_t / T) \parallel \sigma(z^T_t / T))}{\sum_t m_t}, \quad m_t = \mathbb{1}[\text{label}_t \neq -100]
$$

- **Else** (vocab mismatch and no mapping): warn once and fall back to pure CE.

Total: `ce_loss + kd_weight * kd_loss`. If student and teacher have different vocab sizes and no projector is configured, this silently degenerates to SFT — watch for the one-shot warning at line 107-112.

### 3b. `emkd` — Efficient Multimodal KD

`src/criterions/em_kd.py`. Four-term loss combining response distillation with explicit vision-token distillation. Hard-coded weights at `EMKDCriterion.__init__` (lines 96-99) override the `em_kd_*` args.

With $\alpha=0.5$, $\beta=0.25$, $\gamma=25.0$, $T=1.0$:

1. **Supervised CE** (`student_outputs.loss`) — assistant-only CE.

2. **Response logit distillation** — reverse-KL on response-position logits, vocab-cropped to the smaller of the two:
$$
L_{\text{resp}} = T^2 \cdot \mathbb{E}_{t \in \text{response}}\Big[ \sum_v p^S_{t,v}(\log p^S_{t,v} - \log p^T_{t,v}) \Big]
$$
Reverse-KL is mode-seeking: student concentrates on teacher's high-probability modes rather than spreading mass.

3. **Matched vision-logit distillation** — see §4.

4. **Vision↔language affinity distillation** — cosine-similarity matrices between matched vision tokens and text tokens, smooth-L1 between student and teacher matrices. The $\gamma=25.0$ weight on this term says structural cross-modal alignment matters more than logit matching.

$$
L = \alpha \cdot L_{\text{CE}} + (1-\alpha) \cdot L_{\text{resp}} + \beta \cdot L_{\text{vision-logit}} + \gamma \cdot L_{\text{affinity}}
$$

### 3c. `sre` — span-based, structurally aware

`src/criterions/sre.py`. Three KD terms plus CE, read from `args` (`sre_alpha`, `sre_p`, `sre_span_loss_weight`, `sre_geom_loss_weight`, `sre_logit_loss_weight`, `sre_temperature`).

$$
L = \alpha \cdot L_{\text{CE}} + (1-\alpha) \cdot \big(w_{\text{span}} L_{\text{span}} + w_{\text{geom}} L_{\text{geom}} + w_{\text{logit}} L_{\text{logit}}\big)
$$

Two execution paths chosen by `_has_pooler` (line 358):

- **Pooler-based (path A)** — fires when the collator wrote `pooler_safe_idx`/`pooler_mask`. Uses LCS-aligned spans and causal-attention-weighted pooling. Details in §5.
- **Token-weight fallback (path B)** — no LCS spans; per-token hiddens with `labels.ne(-100)` as token mask raised to power `p`.

Multi-layer span loss applies `student_layer_mapping`/`teacher_layer_mapping` and combines per-layer losses with the hard-coded weight schedule `DEFAULT_HIDDEN_LOSS_WEIGHTS = (1,1,2,2,3,3,4,4,5,5,8,10)` normalized to sum to 1 — later layers get more weight.

The **shared-vocabulary soft-label loss** (`_aligned_soft_label_loss`, line 280-295) handles tokenizer mismatch: at first call, `_build_shared_vocab_cpu` builds a mapping of `(student_id, teacher_id)` pairs for every token string present in both vocabs. KL is computed only on this shared subset, indexed once per device:
$$
L_{\text{logit}} = \frac{T^2}{B} \sum_{t \in \text{valid}} \text{KL}\big(\sigma(z^S_t[\mathcal V_{\text{shared}}] / T) \parallel \sigma(z^T_t[\mathcal V_{\text{shared}}] / T)\big)
$$

---

## 4. Deep-dive A: SRE geometry loss — what relational similarity-matrix MSE preserves

The loss (`src/criterions/sre.py:177-184`):

```python
student_hidden = F.normalize(student_hidden.float(), dim=-1)
teacher_hidden = F.normalize(teacher_hidden.float(), dim=-1)
student_scores = student_hidden @ student_hidden.transpose(-1, -2)
teacher_scores = teacher_hidden @ teacher_hidden.transpose(-1, -2)
loss = (mse(student_scores, teacher_scores) * pair_weights).sum() / B
```

### What the math says

After L2-normalization each $\hat h_i$ lives on the unit sphere $S^{d-1}$. The "score" matrix is pairwise cosine similarity:
$$
S^S_{ij} = \langle \hat h^S_i, \hat h^S_j\rangle = \cos\theta_{ij}^S, \qquad S^T_{ij} = \cos\theta_{ij}^T
$$

The loss is the weighted Frobenius distance between the two angle matrices:
$$
L_{\text{geom}} = \frac{1}{B}\sum_{i \neq j} w_{ij}\,(S^S_{ij} - S^T_{ij})^2, \qquad w_{ij} = \frac{w_i w_j}{\sum_{k\neq l} w_k w_l}
$$
Diagonal is zeroed because $S_{ii} \equiv 1$ contributes nothing informative.

### What's preserved (and what isn't)

The thing being matched is the **angular geometry between token representations**, not the representations themselves.

**1. Invariance to any orthogonal change of basis in the student.** If $H^S = H^T O$ for any orthogonal $O$ (rotation or reflection), then
$\hat H^S \hat H^{S\top} = \hat H^T OO^\top \hat H^{T\top} = \hat H^T \hat H^{T\top}$,
so the geometry loss is **identically zero**. The student is free to land in any rotation of the teacher's representation space and still satisfy this term perfectly. Hidden-state MSE (what `default_distillation` does) requires identity, not orthogonal equivalence — a much stronger and noisier objective.

**2. Invariance to global scaling.** Normalization kills magnitudes, so the student doesn't need to match the teacher's norm distribution either.

**Limitation:** $S_{ij}$ is indexed by position, so token $i$ must occupy the same "role" in student and teacher sequences. This is why the SRE pipeline does LCS span alignment first — so the $i$-th student span and $i$-th teacher span correspond to the same chunk of input text. Without that alignment, the geometry loss is meaningless across heterogeneous tokenizers.

### Why this composes with the other SRE terms

- **Span loss** — weighted cosine on per-span vectors. Pulls student spans *toward* the teacher's absolute span vectors. Needs `_crop_last_dim` for different hidden dims (line 170), breaks under arbitrary rotations.
- **Geometry loss** — relational. Forgives rotations.
- **Logit loss** — shared-vocab KL. Operates in vocabulary space, agnostic to hidden geometry.

So SRE is asking the student: "land *somewhere* whose angular structure matches the teacher (geom), AND whose vectors point roughly at the teacher's directions in the cropped subspace (span), AND whose output distributions agree on shared tokens (logit)." If span loss were the only hidden-state objective, you'd over-constrain the student to mimic the teacher's specific basis, which is wasteful — the teacher's basis is mostly an arbitrary artifact of pre-training.

### A subtle code detail

The geometry loss calls `_crop_last_dim` (line 178) before normalizing. With student d=896, teacher d=1536, it keeps the first 896 dims of teacher. This isn't necessary for geometry in principle (you could normalize each side independently and the cosine matrices would still be defined), but the implementation chose to crop for code uniformity with the span loss. The practical effect: you're matching the student's angular geometry against the angular geometry of the **first 896 components of the teacher**. The teacher dims aren't ordered by importance, so this is closer to "match a random projection of the teacher's geometry." Probably fine in practice but not theoretically clean.

---

## 5. Deep-dive B: EM-KD Hungarian matching — what it buys you

The vision-distillation block (`src/criterions/em_kd.py:162-191`):

```python
vl_s, vl_t = _crop_last_dim(student_head(vhs_s), teacher_head(vhs_t))
with torch.no_grad():
    cost = torch.cdist(vl_s.float(), vl_t.float(), p=1).detach()
    idx_s, idx_t = _hungarian_match(cost)
vl_s_matched, vl_t_matched = vl_s[idx_s], vl_t[idx_t]
vision_logit_losses.append(_reverse_kl(vl_s_matched, vl_t_matched, T))
```

### The setup: why matching is needed

Student and teacher disagree on **how many vision tokens an image becomes**:
- Qwen2-VL uses dynamic patching tied to image resolution.
- Qwen2.5-VL uses window-attention with a different patch grid.
- Same image → student might produce 256 vision tokens, teacher 484.

Their `vision_feature_mask` outputs at line 153-154 give you the token positions, but $n_S \neq n_T$ and the $k$-th vision token in student does **not** correspond to the $k$-th in teacher (their spatial layouts differ).

Two naive options, both bad:
1. **Pool both to single vectors** — destroys spatial information.
2. **Nearest-neighbor matching** — each student token matched to its closest teacher token. Allows many-to-one collisions; multiple student tokens collapse onto the same teacher token, losing the distinction.

Hungarian gives you the third option: **a one-to-one bijection that minimizes total cost globally**.

### What the L1 cost matrix encodes

`vl_s` is "vision-as-vocab-distribution": student's vision token hidden state pushed through the **student's own LM head**. So $vl_s[i] \in \mathbb{R}^V$ asks: "if this vision token were a text token to predict, what vocabulary distribution would the model produce?"

The cost
$$
C_{ij} = \|vl_s[i] - vl_t[j]\|_1 = \sum_v \big|z^S_{i,v} - z^T_{j,v}\big|
$$
asks: "how different are the vocab profiles these two vision tokens project into?" Two vision tokens are deemed similar if they describe the same image content **in vocabulary terms** — even if they live in entirely different hidden spaces (student and teacher each project through their own head).

This is a semantically meaningful cost: it doesn't require the hidden spaces to align, only their projections into shared vocabulary space. Vocabulary is the one thing both models definitionally share (modulo tokenization, which here is handled by `_crop_last_dim` to the min vocab size).

L1 instead of L2: logits are unbounded and a few outlier dimensions dominate L2 distance. L1 is more robust and produces flatter, more uniform matchings.

### Why `cost.detach()` and `no_grad` around Hungarian

`linear_sum_assignment` is non-differentiable — it solves a combinatorial assignment problem and returns discrete indices. Detaching the cost matrix and wrapping in `torch.no_grad()` is correct and necessary.

But — and this matters — **gradients still flow** through `vl_s[idx_s]` in the subsequent `_reverse_kl` call. The matching is fixed (no gradient through `idx_s`), but the matched student logits still receive gradient from the KL term. So the loss says: "given this matching computed once per step, push the student's matched vision-logits toward the teacher's matched vision-logits."

This is an EM-style decomposition (hence the criterion's name): E-step computes the matching, M-step updates the student. Matching is recomputed every forward pass, so it adapts as the student improves.

### The reverse-KL choice

Reverse-KL ($p_s \| p_t$ with $p_s$ as the sampling distribution) instead of forward-KL:
$$
L_{\text{vision}} = T^2 \sum_v p^S_v \log\frac{p^S_v}{p^T_v}
$$

Reverse-KL is **mode-seeking**: student concentrates probability mass on teacher's high-probability modes, ignoring teacher's low-probability spread. Forward-KL is mass-covering — student would try to cover all of teacher's support, including improbable tokens. For vision tokens (which describe a small set of relevant concepts in vocabulary space), mode-seeking is desirable: you want the student to focus, not hedge.

### The affinity loss is what makes matching scalable

Just doing the Hungarian-matched reverse-KL would be a fragile signal — the matching might be noisy, and you'd be training on whatever assignment happened to minimize L1 cost. The affinity term (line 187-191) adds robustness:

```python
affinity_s = cosine_similarity(vhs_s_matched.unsqueeze(1), lhs_s.unsqueeze(0))   # (n_v, n_t)
affinity_t = cosine_similarity(vhs_t_matched.unsqueeze(1), lhs_t.unsqueeze(0))
losses.append(smooth_l1_loss(affinity_s[:, :min_text_len], affinity_t[:, :min_text_len]))
```

For each matched vision token $i$, you compute its cosine similarity to every text (assistant) hidden state $j$. So `affinity_s[i, j]` = "how strongly does matched student vision token $i$ relate to text token $j$, in student's hidden space." Match this matrix against the teacher's. **The matrix doesn't depend on the absolute hidden geometry — only on the cross-modal attention pattern.**

So the matching can be slightly wrong and the affinity loss still pulls the student toward "vision tokens that play the same role relative to text" as the teacher. The $\gamma = 25.0$ weight at line 98 reflects this — the affinity term carries the bulk of cross-modal alignment.

### Cost and limits

- Hungarian is $O(n^3)$ in the larger of $(n_S, n_T)$. For 256-vs-484 vision tokens that's ~10⁸ ops on CPU per sample per step — tolerable. For high-res inputs with thousands of vision tokens, this would dominate runtime.
- The L1 cost in cropped vocab space ignores the rest of the vocabulary. If cropping removes the dimensions that distinguish vision tokens, matching degrades to noise. In practice student vocab ⊆ teacher vocab here, so cropping to min works out.
- The fallback at line 168-170 silently skips a sample's vision KD if any of the four masks (student/teacher × vision/text) is empty — common when a sample has no image or no assistant turn. Watch the `matched_vision_samples` metric to see how often this fires.

---

## 6. Deep-dive C: SRE LCS span alignment — building cross-tokenizer spans

The most under-documented and most interesting piece. Lives in `src/data/dataset.py`, runs inside the collator.

### Why it exists

When student tokenizer ≠ teacher tokenizer (e.g. student=Qwen2-VL, teacher=Qwen2.5-VL or Qwen3-VL), the **same text becomes different token sequences with different lengths**:

```
text:    "The cat sat"
student: ["The", " cat", " sat"]              -> 3 tokens
teacher: ["Th",  "e c",  "at sat"]            -> 3 tokens, different splits
```

Hidden states $h^S_i$ and $h^T_i$ at position $i$ correspond to entirely different chunks of input text. You cannot do token-aligned distillation. You can only align on **character boundaries that both tokenizers agree on**.

The HF tokenizer's `return_offsets_mapping=True` gives, for each token, the `(start_char, end_char)` in the original text. The collator passes this flag through to `apply_chat_template` at `src/data/dataset.py:568-569`.

### `_longest_common_subsequence_offsets` — the alignment kernel

`src/data/dataset.py:26-62`. The name "LCS" is a slight misnomer — this is a two-pointer scan over **end offsets** of the two tokenization streams, finding **shared character-boundary positions**:

```python
while i < len(a_list) and j < len(b_list):
    if a_list[i][1] == 0:  i += 1; continue   # skip special tokens (end=0)
    if b_list[j][1] == 0:  j += 1; continue

    if a_list[i][1] == b_list[j][1]:           # both tokens end at same char
        result.append((i + 1, j + 1))           # boundary point
        i += 1; j += 1
    elif a_list[i][1] < b_list[j][1]:
        i += 1                                  # student behind, advance
    else:
        j += 1                                  # teacher behind, advance
```

The output `result` is a list of pairs $(i_k, j_k)$ such that "student tokens $[0..i_k)$ and teacher tokens $[0..j_k)$ cover the same prefix of the input text." Between two consecutive shared boundaries $(i_{k-1}, j_{k-1})$ and $(i_k, j_k)$, you have a **span** of student tokens $i_{k-1}..i_k-1$ that covers the same text as teacher tokens $j_{k-1}..j_k-1$. The number of tokens per side inside a span can differ.

Example walk-through for "The cat sat":
```
student end-offsets: [3,  7, 11]   for tokens ["The", " cat", " sat"]
teacher end-offsets: [2,  5, 11]   for tokens ["Th",  "e c", "at sat"]

i=0, j=0:   3  vs   2   → 3>2, j++            (teacher behind)
i=0, j=1:   3  vs   5   → 3<5, i++            (student behind)
i=1, j=1:   7  vs   5   → 7>5, j++
i=1, j=2:   7  vs  11   → 7<11, i++
i=2, j=2:  11  vs  11   → match → append (3, 3); i++; j++
```

Result: `[(3, 3)]`. With the seed pair $(s\_start, t\_start) = (0, 0)$ prepended, this gives one span: student tokens 0..2, teacher tokens 0..2. The text was too short for any interior boundaries. Longer text → more shared boundaries → more spans.

### The cap and the special-token rescue

**Cap at `SRE_MAX_SPANS = 1024`** (line 19, 59-61):
```python
if len(result) > SRE_MAX_SPANS:
    step = len(result) / SRE_MAX_SPANS
    return [result[int((idx + 1) * step) - 1] for idx in range(SRE_MAX_SPANS)]
```
Uniform subsampling. Prevents pathological cases (mostly-ASCII text where most boundaries align) from creating gigantic pooler tensors.

**Special-token rescue** (line 49-57): when one side hits a special token (end-offset 0 — e.g. `<|im_start|>`, padding), the scan would stall. The rescue block "fast-forwards both streams past any active alignment region until both are at a special-token boundary, then re-syncs and records the new pair." This handles boundaries between conversation turns and the system-prompt prefix, where both tokenizers re-align at structural markers like `<|im_start|>`.

### `_prepare_sre_pooler` — turning boundaries into gather indices

`src/data/dataset.py:103-140`. For each sample in the batch:

```python
student_start = _first_supervised_token(student_inputs["labels"], ...)   # first assistant position
teacher_start = _first_supervised_token(teacher_labels, ...)

common_offsets = [(student_start, teacher_start)] + _longest_common_subsequence_offsets(
    student_offset, teacher_offset, student_start, teacher_start)
```

The alignment is **anchored at the first assistant token**, not at position 0. Spans only exist over the response region — system prompt and user turns don't contribute to span loss, which matches the labels masking. Then for each consecutive pair of boundaries:

```python
student_seg_idx.append(arange(s_prev, s_cur))   # token indices for this student span
teacher_seg_idx.append(arange(t_prev, t_cur))   # token indices for this teacher span
```

`_get_pooler_tensor` (line 65-84) packs all this into batched tensors:
- `safe_idx[B, n_spans, max_span_width]` — token indices, padded with `-1` for shorter spans, then masked to `0` for safe gathering.
- `pooler_mask[B, n_spans, max_span_width]` — bool, true at valid positions.

Both go into `student_inputs` and `teacher_inputs`. The student has its own (`safe_idx`, `mask`) shape; the teacher has a different shape (different token counts per span) but the same number of spans.

### `_pool_hidden_states_with_weights` — using the spans

`src/criterions/sre.py:139-166`. Given hidden states `H[B, L, D]`, span indices `safe_idx[B, S, W]`, and mask:

```python
gathered = H[batch_idx, safe_idx]              # [B, S, W, D]
gathered = gathered * pooler_mask.unsqueeze(-1)
```

Now the **causal attention weight** trick. If the model returned `attentions`, `_causal_attention_weights` (line 127-136) extracts the attention paid by the **last query token** to every key token, summed over heads:

```python
weights = attention.sum(dim=1)[:, -1]          # [B, L], heads summed, last query position
```

This is "how much does the model's final generated token attend to each previous token." Tokens the model *uses* get higher weights. These weights then gate the per-token pooling within each span:

```python
span_token_weights = token_weights[batch_idx, safe_idx] * pooler_mask
pooled = (gathered * span_token_weights[..., None]).sum(dim=2) / span_token_weights.sum(dim=2)
```

Effect: within a span like teacher tokens `["e c"]` covering the same text as student `[" cat"]`, you don't average teacher tokens uniformly — you upweight the tokens the model actually attends to when producing its response. Common-but-uninformative tokens (e.g. punctuation, articles) get downweighted automatically.

The teacher's `span_weights` is then used to weight the per-span cosine losses at line 441 (each layer's loss is weighted by how much the teacher considers that span important). This is a self-supervised importance signal — you're not setting span weights manually, the teacher's attention pattern tells you which spans matter.

### Why this whole machine is worth it

Without LCS spans, you have three bad options:
1. Match by token position — wrong, because tokens at the same position mean different things across tokenizers.
2. Match by raw text content — requires re-decoding and re-tokenizing, lossy.
3. Skip cross-tokenizer pairs entirely — restricts you to same-tokenizer student/teacher pairs.

LCS spans let you distill across **architecturally different** teachers (different vocab, different model family) by aligning at character boundaries where the tokenizers happen to agree, then handling within-span tokenization differences via attention-weighted pooling.

The cost: collation time grows with sequence length (LCS scan is $O(n_S + n_T)$ per sample, plus offset-mapping computation). For mix665k-scale data this is a fraction of a percent of training time, easily absorbed.

The limitation: if shared boundaries are rare (e.g. one tokenizer is BPE, the other is character-level), you end up with very few, very long spans. The span loss degenerates toward "match the mean pooled hidden state of the entire response," which is much weaker. Healthy tokenizer pairs (Qwen2 ↔ Qwen2.5) produce hundreds of well-distributed boundaries per response.

---

## 7. Deep-dive D: The vestigial RKD args — intended vs actual behavior

### Args that aren't read

In `src/arguments.py`, the `TrainingArguments` declares:
```python
kd_weight: float = 0.01
rkd_distance_weight: float = 1.0
rkd_angle_weight: float = 2.0
kd_loss_type: str = "contrastive_rkd"
```

And `DefaultDistillationCriterion` only reads `kd_weight` (line 44). The other three are dangling.

Worse, the **default value of `kd_loss_type` is `"contrastive_rkd"`**, which is not registered in `criterion_list` — running with the defaults raises `ValueError: Unsupported kd_loss_type: contrastive_rkd` at `build_criterion`. Every training script overrides `--kd_loss_type` to one of `default`, `emkd`, or `sre`, so this never trips in practice. But the default-as-shipped is broken.

### What "contrastive RKD" was probably meant to be

Original RKD (Park, Kim, Lu, Cho 2019) defines two relational losses on example-level representations.

**Distance-wise:**
For all pairs $(i,j)$ in the batch, compute
$$
d^X_{ij} = \|h^X_i - h^X_j\|_2, \qquad \hat d^X_{ij} = d^X_{ij} \,/\, \mu_X
$$
where $\mu_X = \frac{1}{|P|}\sum_{(i,j)} d^X_{ij}$ normalizes by mean pair distance. Then
$$
L_d = \sum_{(i,j)} \ell_h(\hat d^S_{ij}, \hat d^T_{ij}),
$$
where $\ell_h$ is the Huber loss.

**Angle-wise:**
For triplets $(i, j, k)$, the unit vectors
$$
e^X_{ji} = \frac{h^X_i - h^X_j}{\|h^X_i - h^X_j\|}, \qquad e^X_{jk} = \frac{h^X_k - h^X_j}{\|h^X_k - h^X_j\|}
$$
and the angle at vertex $j$:
$$
a^X_{ijk} = \langle e^X_{ji}, e^X_{jk}\rangle = \cos\theta_{ijk}^X
$$
Then $L_a = \sum_{(i,j,k)} \ell_h(a^S_{ijk}, a^T_{ijk})$.

Total: $L_{RKD} = w_d L_d + w_a L_a$ with the original paper's recommendation $w_d=1, w_a=2$ — which exactly matches the defaults in `arguments.py`. So someone intended to follow the original RKD recipe.

The "contrastive" prefix probably referred to combining RKD with an InfoNCE-style loss using same-example as positive and other-batch examples as negative. This would interact with the `pooling`, `normalize`, `temperature`, `num_hardneg` args in `ModelArguments`, which also look orphaned (none are read by VLM distillation). These args all appear to come from a retrieval/embedding training codebase that was repurposed for VLM distillation but never fully ported.

### What the current code actually does

Two branches in `DefaultDistillationCriterion.forward` (§3a):

1. **Layer-mapped MSE** (when `student_layer_mapping` & `teacher_layer_mapping` are set):
   - Mean-pool each mapped student layer's hidden states across sequence
   - Project through `distiller.projectors[idx]` to teacher dim
   - MSE against teacher's mean-pooled hidden
   - $L = \frac{1}{N}\sum_i \|P_i(\bar h^S_i) - \bar h^T_i\|^2$
   
2. **Token-level forward-KL** (fallback when no mapping, logits compatible):
   - Standard distillation: $T^2 \cdot \text{KL}(\sigma(z^S/T) \parallel \sigma(z^T/T))$, gated to assistant tokens.

Neither uses RKD's distance or angle.

### Sketch of an actual RKD criterion

If you genuinely want RKD as a baseline, this would go in `src/criterions/rkd.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distance(h):
    """h: [B, D]; returns [B, B] of pairwise L2 distances."""
    sq = (h * h).sum(dim=-1, keepdim=True)
    dist_sq = sq + sq.T - 2 * h @ h.T
    return dist_sq.clamp_min(0.0).sqrt()


def _normalized_pairwise_distance(h):
    d = _pairwise_distance(h)
    mu = d[d > 0].mean()
    return d / mu.clamp_min(1e-6)


def _angle_tensor(h):
    """h: [B, D]; returns [B, B, B] of angles at each vertex."""
    diff = h.unsqueeze(0) - h.unsqueeze(1)          # [B, B, D]
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    e = diff / norm
    return torch.einsum("jid,jkd->jik", e, e)


def _attn_masked_mean(hidden, attention_mask):
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(dtype=hidden.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom


class RKDCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = float(getattr(args, "kd_weight", 1.0))
        self.w_d = float(getattr(args, "rkd_distance_weight", 1.0))
        self.w_a = float(getattr(args, "rkd_angle_weight", 2.0))

    def forward(self, distiller, batch):
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        ce_loss = student_outputs.loss

        student_pool = _attn_masked_mean(
            student_outputs.hidden_states[-1],
            student_inputs.get("attention_mask"),
        )
        teacher_pool = _attn_masked_mean(
            teacher_outputs.hidden_states[-1],
            teacher_inputs.get("attention_mask"),
        )
        if hasattr(distiller, "projectors") and len(distiller.projectors) > 0:
            student_pool = distiller.projectors[0](student_pool)

        student_pool = student_pool.float()
        teacher_pool = teacher_pool.to(device=student_pool.device).float()

        d_s = _normalized_pairwise_distance(student_pool)
        d_t = _normalized_pairwise_distance(teacher_pool)
        loss_d = F.smooth_l1_loss(d_s, d_t)

        a_s = _angle_tensor(student_pool)
        a_t = _angle_tensor(teacher_pool)
        loss_a = F.smooth_l1_loss(a_s, a_t)

        rkd_loss = self.w_d * loss_d + self.w_a * loss_a
        return {
            "loss": ce_loss + self.alpha * rkd_loss,
            "rkd_distance": loss_d.detach(),
            "rkd_angle": loss_a.detach(),
        }
```

Then register in `src/criterions/__init__.py`:
```python
criterion_list = {
    "default": DefaultDistillationCriterion,
    "default_distillation": DefaultDistillationCriterion,
    "emkd": EMKDCriterion,
    "em_kd": EMKDCriterion,
    "sre": SRECriterion,
    "rkd": RKDCriterion,
    "contrastive_rkd": RKDCriterion,   # alias for the (currently broken) default
}
```

**Computational notes:**
- Pairwise tensor is $O(B^2)$ memory, angle tensor is $O(B^3)$. For $B=32$ that's ~32K floats — trivial. For $B=128$ it's ~2M floats — still fine. Beyond $B=256$ the angle tensor starts to bite.
- Mean-pooling discards all sequence information. For VLMs you might want to pool only over assistant tokens (label != -100), or only over vision tokens, depending on what you want preserved.
- Without contrastive sampling, this is "pure" RKD. To make it contrastive, add an InfoNCE term with student/teacher pooled vectors as positives and other-batch pairs as negatives.

### Should you implement this?

Probably not as-is. The SRE geometry loss is essentially a token-level generalization of RKD's angle term — it's preserving the same structural information (pairwise angles) but at much higher resolution (every token, not just every example). EM-KD's affinity loss does something similar in the cross-modal vision-text setting. So RKD's distinctive value (relational preservation) is already captured by SRE at finer granularity.

The args were inherited from earlier work and not cleaned up. The pragmatic three-line fix is to **delete the dead args and fix the broken default**:

```python
# in arguments.py
kd_weight: float = field(default=0.01, ...)          # keep — used by default criterion
kd_loss_type: str = field(default="default", ...)    # was "contrastive_rkd"
# DELETE rkd_distance_weight, rkd_angle_weight
```

That's a small PR and fixes the broken default. If you want RKD as a baseline, add the criterion above as a separate file.

---

## 8. Summary recipe

| Criterion | CE | Logit KL | Hidden alignment | Vision-specific | Tokenizer mismatch handling |
|---|---|---|---|---|---|
| `default` | ✓ ($\alpha=1$ always) | forward-KL on response | MSE on projected mean-pooled hidden | — | crop to min vocab (forward-KL) |
| `emkd` | ✓ ($\alpha=0.5$) | reverse-KL on response | — | Hungarian-matched vision logits + affinity matrix | crop to min vocab |
| `sre` | ✓ ($\alpha=0.5$) | forward-KL via shared vocab | weighted cosine on LCS spans (per layer) + geometry MSE | — (treats vision tokens uniformly) | LCS span pooling + shared-vocab logit lookup |

**Things that bite if you forget them:**
- KD criteria need hidden states & attentions — don't disable `_force_eager_attention`.
- `data_args.kd_loss_type` is set in `train.py:141` (not on the dataclass) so the collator can branch. Move that line and SRE silently loses its pooler path.
- EM-KD's $\alpha, \beta, \gamma, T$ are **hard-coded in the class** (`em_kd.py:96-99`), not read from `em_kd_*` args. Edit the class to tune them.
- The `default` criterion silently degenerates to CE-only when teacher vocab ≠ student vocab and no mapping is set — watch for the one-shot warning at `default_distillation.py:108-112`.
- `kd_loss_type` default is broken (`"contrastive_rkd"` is not registered). Every script overrides it; never run without `--kd_loss_type`.

---

## 9. Paper-worth review: is this idea publishable?

Short answer: **not as-is, but there is a publishable direction if the paper is reframed around cross-architecture VLM alignment rather than "a VLM distillation framework."**

As currently written in the repo, the work is a strong engineering adaptation of existing KD ideas:

- `sre` brings span-level, attention-weighted, geometry-aware cross-tokenizer distillation into the VLM training pipeline.
- `emkd` brings vision-token matching, reverse-KL vision semantic distillation, and vision-language affinity distillation into the same training pipeline.
- `default` is a conventional CE + hidden-MSE or logit-KL baseline.

That is useful, but by itself reviewers will likely read it as integration, not a new method. The paper needs to name a failure mode that existing papers do not fully solve, then show that this repo's method addresses that failure mode better than clean baselines.

### Prior-art risk

Two recent papers create the main novelty pressure.

1. **SRA: Span Representation Alignment for Large Language Model Distillation** (`arXiv:2605.01205`, submitted May 2026): https://arxiv.org/abs/2605.01205
   - This is very close to the text-side SRE story. It shifts cross-tokenizer KD from token alignment to tokenizer-agnostic spans, uses attention-weighted span representations, adds a geometric regularizer, and performs aligned span logit distillation.
   - Therefore, the paper should **not** claim span representation alignment as the core novelty unless the claim is specifically about the VLM/multimodal setting and backed by multimodal evidence.

2. **EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens** (`arXiv:2511.21106`, accepted AAAI 2026): https://arxiv.org/abs/2511.21106
   - This is very close to the current `emkd` implementation. It uses Manhattan distance between teacher/student vision logits, Hungarian matching, vision-language affinity distillation, and reverse-KL vision semantic distillation.
   - Therefore, the paper should **not** present Hungarian-matched vision-logit KD or VL affinity distillation as new unless the implementation materially changes the objective or applies it to a distinct cross-architecture setting.

So the unsafe framing is:

> We propose span alignment and EM-style vision-token matching for VLM distillation.

That will look like SRA + EM-KD combined.

### Stronger paper hypothesis

A defensible hypothesis is:

> Cross-architecture VLM distillation fails because teacher and student disagree on both text-side token boundaries and vision-side visual units. Token-level KD, hidden-state MSE, and pooled representation KD all assume comparable units. A student learns better when teacher supervision is transferred through semantically comparable units: assistant-response text spans aligned by shared character boundaries, and image tokens matched in vocabulary-projected semantic space, with relational geometry preserving structure inside each aligned unit set.

This changes the paper from "we implemented KD losses" to:

> **Unit mismatch is the bottleneck in cross-architecture VLM distillation.**

The method then becomes a principled answer to that bottleneck:

- text units: LCS/shared-boundary response spans, not raw tokens;
- text importance: teacher attention-weighted span pooling, not uniform averaging;
- text structure: cosine-similarity geometry over aligned spans, not absolute hidden MSE only;
- vision units: Hungarian one-to-one matching over vocab-projected vision logits, not index matching or global pooling;
- cross-modal structure: vision-text affinity preservation, not vision-only logit imitation;
- output behavior: shared-vocabulary soft-label KL for tokenizer mismatch.

Potential title:

> **Dual-Granularity Alignment for Cross-Architecture Vision-Language Model Distillation**

Alternative titles:

- **Unit-Aligned Distillation for Heterogeneous Vision-Language Models**
- **Aligning What the Teacher and Student Actually Share: Span and Vision-Unit Distillation for VLMs**
- **From Tokens to Units: Cross-Architecture Distillation for Vision-Language Models**

### What is genuinely interesting in this repo

The strongest paper kernel is the **joint text-and-vision unit mismatch**:

- Text mismatch: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA, and FastVLM may tokenize the same assistant response differently. The SRE path solves this with response-only span pooling via `offset_mapping` in `src/data/dataset.py`.
- Vision mismatch: different VLM backbones produce different numbers and layouts of vision tokens for the same image. The EM-KD path solves this with vocabulary-space matching in `src/criterions/em_kd.py`.
- Cross-modal mismatch: VLM performance often depends less on matching a vision token's hidden vector and more on preserving which visual regions relate to which text tokens. The affinity loss is a better paper argument than plain vision-logit KL.

That combined failure mode is more VLM-specific than either SRA or EM-KD alone.

### What is weak today

The current codebase has several issues that would weaken a submission if left unresolved:

1. **The default KD setting is broken.** `TrainingArguments.kd_loss_type` defaults to `"contrastive_rkd"`, but `criterion_list` does not register it. This is an easy reviewer-reproducibility problem.

2. **Some declared arguments do not control the implementation.** `em_kd_alpha`, `em_kd_beta`, `em_kd_gamma`, and `em_kd_temperature` are declared in `src/arguments.py`, but `EMKDCriterion.__init__` hard-codes the values. A paper cannot present these as tunable experimental knobs until this is fixed.

3. **SRE geometry currently crops hidden dimensions.** `_geometry_loss` calls `_crop_last_dim`, so a 896-dim student is compared against the first 896 dims of a 1536-dim teacher. That is not a theoretically clean geometry objective. For a paper, use learned projectors or compute geometry independently on each side's native hidden dimension.

4. **There is no visible evaluation suite.** The repo has training and smoke-test scripts, but no benchmark runner or result table. Without strong results, this remains a method note.

5. **The top-level `sre/` folder is reference text-only code.** The actual VLM criterion is self-contained in `src/criterions/sre.py`. The paper should avoid implying that the standalone `sre/` implementation is the core VLM method.

### Minimal experiment matrix

A serious paper needs at least this matrix.

**Student/teacher pairs**

| Pair type | Teacher | Student | Why it matters |
|---|---|---|---|
| Same family, larger teacher | Qwen2.5-VL-7B | Qwen2-VL-2B | Basic size distillation; likely easier |
| Same family, newer teacher | Qwen3-VL-8B | Qwen2.5-VL-3B | Tests tokenizer/model-generation shift |
| Cross-family | LLaVA-OneVision or FastVLM teacher | Qwen student, or reverse | Tests whether unit alignment beats family-specific assumptions |
| Vision-token imbalance | high-res teacher/student mismatch | efficient student | Tests the EM-KD-like vision matching claim |

**Baselines**

- Student SFT only.
- Vanilla response-logit KD on cropped/shared vocabulary.
- Hidden-state MSE with projectors (`default` criterion).
- SRE-only text-span distillation.
- EM-KD-only vision distillation.
- Combined text-span + vision-unit method.
- Optional: same-tokenizer teacher/student control, where token mismatch is reduced.

**Ablations**

| Ablation | Question answered |
|---|---|
| no LCS span alignment | Does response-span alignment matter beyond ordinary token KD? |
| uniform span pooling | Does teacher-attention weighting matter? |
| no geometry loss | Does relational structure add anything beyond span cosine loss? |
| no shared-vocab logit KD | Does output-space supervision matter under tokenizer mismatch? |
| no Hungarian vision matching | Is one-to-one visual unit matching necessary? |
| nearest-neighbor instead of Hungarian | Is global assignment better than greedy local matching? |
| no vision-language affinity | Is cross-modal relational structure the real driver? |
| no vision-logit reverse-KL | Does semantic vision-token supervision add signal beyond affinity? |

**Benchmarks**

Use a mix of general VLM, OCR, reasoning, hallucination, and chart/table tasks:

- MMBench / MMStar for broad multimodal capability.
- MMMU for multimodal reasoning.
- ScienceQA or AI2D for diagram/question answering.
- TextVQA and OCRBench for text-in-image grounding.
- ChartQA for structured visual reasoning.
- POPE or MME hallucination subsets for object grounding.

**Diagnostics**

These will make the paper more reviewer-resistant than aggregate scores alone:

- span density: number of aligned spans per response and average span width;
- shared-vocabulary coverage between student and teacher;
- percentage of samples with valid vision/text masks for EM-KD;
- Hungarian matching cost distribution over training;
- relation between span density and improvement over SFT/KD;
- runtime and memory overhead of span pooling and Hungarian matching.

### What result would make this paper credible?

A credible result is not just "combined loss beats SFT." The paper needs to show:

1. **Unit-aligned KD beats vanilla KD and hidden-MSE KD** on cross-tokenizer/cross-backbone pairs.
2. **Text-span alignment helps text-heavy visual tasks**, especially OCR/TextVQA/ChartQA-style tasks where answer tokens depend on exact visual text or structured reasoning.
3. **Vision-unit matching helps vision-token-imbalanced settings**, especially high-resolution or efficient-student setups.
4. **The combined method is additive**: SRE-only improves some cases, EM-KD-only improves others, and combined wins when both text and vision unit mismatch are present.
5. **The method is not just extra compute.** Report overhead and compare against a stronger SFT or longer-training baseline with matched compute where possible.

If the gains are small and only appear against SFT, this is not enough. If the gains are consistent across cross-architecture pairs and the diagnostics show the improvements correlate with token/span/vision mismatch, the paper becomes much stronger.

### Recommended next steps

1. Fix reproducibility issues:
   - change `kd_loss_type` default to `"default"`;
   - either delete dead RKD args or implement/register an actual RKD baseline;
   - wire `em_kd_*` args into `EMKDCriterion`;
   - make SRE geometry dimension handling defensible.

2. Add a benchmark/eval runner before adding more method complexity.

3. Run the smallest decisive comparison first:
   - Qwen2.5-VL-7B teacher -> Qwen2-VL-2B student;
   - same data, same steps;
   - compare SFT, default KD, SRE, EM-KD, combined;
   - evaluate on one OCR-heavy task, one general VLM task, and one reasoning task.

4. If combined wins, expand to cross-family pairs and build the paper around **unit mismatch**.

5. If combined does not win, do not write this as a method paper. Instead, keep it as a strong open-source distillation toolkit or write a negative/diagnostic report about when cross-tokenizer and vision-token KD fail.

### Final verdict

Current paper score: **3/10**.

Reason: as-is, the method overlaps heavily with SRA on the text/span side and EM-KD on the vision side, with no visible benchmark evidence yet.

Potential score after refactor + experiments: **7/10**.

Reason: a clean paper about **dual text/vision unit alignment for cross-architecture VLM distillation** could be defensible if the experiments show additive gains over SRE-only, EM-KD-only, vanilla KD, and hidden-MSE KD. The key is to make the novelty the VLM-specific failure mode and the evidence, not the individual losses in isolation.
