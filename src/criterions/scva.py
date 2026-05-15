"""Semantic-Cluster Visual Attention (SCVA) distillation.

For each generation step t, instead of forcing the student to match the
teacher's per-patch visual attention a_T^(t) — which is overly strict because
both can attend to slightly different vision tokens while focusing on the same
semantic region — we cluster the teacher's vision tokens into M semantic groups
and align the *cluster-level* attention distributions r_T^(t), r_S^(t) in R^M.

Loss per draft.pdf §SCVA:
    L_SCVA = ( Σ_t u_t · KL(r_T^(t) || r_S^(t)) ) / Σ_t u_t
    u_t    = exp( -H(a_T^(t)) / log n )       (teacher-confidence weight)
    r_X^(t)(m) = Σ_{i ∈ C_m} a_X^(t)(i)        (cluster-level attention)

The single-method criterion below mixes L_SCVA with supervised CE following the
EM-KD / SRE convention:
    L = α · L_CE + (1 - α) · w · L_SCVA
This keeps the standalone-SCVA criterion compositionally similar to the others
in this repo. The joint draft method (SCVA + CGKD) uses the draft's additive
formula directly and calls ``SCVACriterion._scva_loss`` to access the pure KD
component.

Assumptions:
  - Teacher and student emit the same number of vision tokens per image (same
    vision encoder family at the same resolution). Samples with n_v_teacher !=
    n_v_student are skipped from SCVA loss. Cross-architecture pairs are out
    of scope for this implementation; a Hungarian or NN remapping would be
    needed to extend.
  - ``output.attentions`` is populated (forced by ``_force_eager_attention`` in
    src/model/model.py).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def _last_hidden(outputs) -> torch.Tensor:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError("SCVA requires model outputs with hidden_states.")
    return hs[-1]


def _resolve_layer(attentions, requested: int) -> int:
    """Resolve a (possibly negative) attention-layer index against the actual
    number of attention tensors returned by the model."""
    if attentions is None:
        return -1
    n = len(attentions)
    if n == 0:
        return -1
    idx = requested if requested >= 0 else n + requested
    return max(0, min(idx, n - 1))


def _kmeans(x: torch.Tensor, n_clusters: int, n_iter: int = 10) -> torch.Tensor:
    """Pure-torch Lloyd K-means on a (n, d) tensor. Returns int64 [n] assignments."""
    n, d = x.shape
    if n_clusters <= 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long, device=x.device)
    if n <= n_clusters:
        # Degenerate: each point is its own (truncated) cluster.
        return torch.arange(n, dtype=torch.long, device=x.device).clamp(max=n_clusters - 1)

    init_idx = torch.randperm(n, device=x.device)[:n_clusters]
    centroids = x[init_idx].clone()

    for _ in range(max(int(n_iter), 1)):
        dists = torch.cdist(x, centroids)            # [n, k]
        assignments = dists.argmin(dim=-1)           # [n]
        new_centroids = centroids.clone()
        for k in range(n_clusters):
            mask = assignments.eq(k)
            if mask.any():
                new_centroids[k] = x[mask].mean(dim=0)
        # Halt early if stable.
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    return torch.cdist(x, centroids).argmin(dim=-1)


class SCVACriterion(nn.Module):
    """Standalone SCVA-only criterion. See module docstring for the loss form."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = float(getattr(args, "scva_alpha", 0.5))
        self.weight = float(getattr(args, "scva_weight", 1.0))
        self.n_clusters = max(int(getattr(args, "scva_n_clusters", 16)), 1)
        self.kmeans_iters = max(int(getattr(args, "scva_kmeans_iters", 10)), 1)
        self.attention_layer = int(getattr(args, "scva_attention_layer", -1))
        self.min_vision_tokens = max(int(getattr(args, "scva_min_vision_tokens", 4)), 1)

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running SCVA.")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        return self.compute_losses(distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs)

    def compute_losses(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        ce = student_outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        scva_kd = self._scva_loss(student_outputs, teacher_outputs, student_inputs, teacher_inputs)
        total = self.alpha * ce + (1.0 - self.alpha) * self.weight * scva_kd
        return {
            "loss": total,
            "hard_loss": ce.detach(),
            "scva_loss": scva_kd.detach(),
        }

    def _scva_loss(
        self,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        teacher_hidden = _last_hidden(teacher_outputs)        # [B, L_t, D]
        teacher_vision_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        student_vision_mask = getattr(student_outputs, "vision_feature_mask", None)
        if teacher_vision_mask is None or student_vision_mask is None:
            return teacher_hidden.new_zeros(())

        t_attns = getattr(teacher_outputs, "attentions", None)
        s_attns = getattr(student_outputs, "attentions", None)
        if not t_attns or not s_attns:
            return teacher_hidden.new_zeros(())

        t_layer = _resolve_layer(t_attns, self.attention_layer)
        s_layer = _resolve_layer(s_attns, self.attention_layer)
        if t_layer < 0 or s_layer < 0:
            return teacher_hidden.new_zeros(())

        # Head-averaged attention. Use float for KL stability.
        t_attn_avg = t_attns[t_layer].float().mean(dim=1)     # [B, L_t, L_t]
        s_attn_avg = s_attns[s_layer].float().mean(dim=1)     # [B, L_s, L_s]

        t_labels = teacher_inputs.get("labels")
        s_labels = student_inputs.get("labels")

        sample_losses = []
        for b in range(teacher_hidden.shape[0]):
            t_v_mask = teacher_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            s_v_mask = student_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            if t_v_mask.sum().item() < self.min_vision_tokens or s_v_mask.sum().item() < self.min_vision_tokens:
                continue
            if t_v_mask.sum().item() != s_v_mask.sum().item():
                # Index-aligned SCVA requires same n_v on both sides. Skip
                # mismatched samples (cross-architecture pairs need a separate
                # remapping path, out of scope here).
                continue

            t_v_idx = t_v_mask.nonzero(as_tuple=True)[0]
            s_v_idx = s_v_mask.nonzero(as_tuple=True)[0]
            n_v = int(t_v_idx.numel())

            if t_labels is None or s_labels is None:
                continue
            t_r_idx = (t_labels[b].to(teacher_hidden.device) != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            s_r_idx = (s_labels[b].to(teacher_hidden.device) != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if t_r_idx.numel() == 0 or s_r_idx.numel() == 0:
                continue

            # K-means cluster teacher vision-token hidden states (no grad).
            with torch.no_grad():
                cluster_idx = _kmeans(
                    teacher_hidden[b, t_v_idx].float(),
                    self.n_clusters,
                    self.kmeans_iters,
                )                                              # [n_v]
            cluster_onehot = F.one_hot(cluster_idx, num_classes=self.n_clusters).float()  # [n_v, M]

            # Teacher response→vision attention rows. Renormalize so each row is a distribution over n_v.
            t_resp_to_vis = t_attn_avg[b][t_r_idx][:, t_v_idx]                  # [T_t, n_v]
            t_resp_to_vis = t_resp_to_vis / t_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            s_resp_to_vis = s_attn_avg[b][s_r_idx][:, s_v_idx]                  # [T_s, n_v]
            s_resp_to_vis = s_resp_to_vis / s_resp_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Align response length: assistant turns should match across student/teacher in a shared batch
            # but we crop defensively.
            T_aligned = min(t_resp_to_vis.shape[0], s_resp_to_vis.shape[0])
            t_resp_to_vis = t_resp_to_vis[:T_aligned]
            s_resp_to_vis = s_resp_to_vis[:T_aligned]

            # Cluster-level distributions (T_aligned × M).
            r_T = (t_resp_to_vis @ cluster_onehot).clamp_min(1e-12)
            r_S = (s_resp_to_vis @ cluster_onehot).clamp_min(1e-12)
            r_T = r_T / r_T.sum(dim=-1, keepdim=True)
            r_S = r_S / r_S.sum(dim=-1, keepdim=True)

            # Teacher-attention entropy → confidence weight u_t.
            h_t = -(t_resp_to_vis.clamp_min(1e-12).log() * t_resp_to_vis).sum(dim=-1)
            u = (-h_t / max(math.log(max(n_v, 2)), 1e-6)).exp()                # [T_aligned]

            kl = F.kl_div(r_S.log(), r_T, reduction="none").sum(dim=-1)        # [T_aligned]
            sample_losses.append((u * kl).sum() / u.sum().clamp_min(1e-6))

        if not sample_losses:
            return teacher_hidden.new_zeros(())
        return torch.stack(sample_losses).mean()
