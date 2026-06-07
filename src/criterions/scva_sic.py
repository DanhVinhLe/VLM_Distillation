"""SCVA + Semantic Intervention Consistency (SIC) distillation.

This criterion keeps the existing SCVA term intact and adds the Semantic
Intervention Consistency loss from the Overleaf draft.  SCVA clusters define the
semantic visual regions C_m.  For each cluster, SIC replays the language model
with that cluster's visual-token embeddings masked, measures the induced
log-probability shift, projects the shift through each model's token embedding
matrix, and aligns the resulting semantic directions with cosine distance.

Total loss:
    L = ce_weight * L_CE + lambda_v * L_SCVA + lambda_sic * L_SIC

The implementation intentionally reuses SCVA's clustering/mapping helpers so the
new criterion follows the same teacher-cluster/student-grid flow as the existing
SCVA code.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.criterions.scva import (
    IGNORE_INDEX,
    SCVACriterion,
    _cluster_onehot_from_labels,
    _cluster_vision_tokens_dbscan,
    _grid_from_inputs,
    _last_hidden,
    _map_teacher_clusters_to_student_onehot,
)


def _embedding_weight(model: Any) -> torch.Tensor:
    encoder = getattr(model, "encoder", model)
    if not hasattr(encoder, "get_input_embeddings"):
        raise RuntimeError("SIC requires models exposing get_input_embeddings().")
    embeddings = encoder.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise RuntimeError("SIC could not resolve an input embedding matrix.")
    return embeddings.weight


def _drop_visual_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove raw visual tensors so a forward with inputs_embeds is not overwritten."""
    dropped = dict(inputs)
    for key in ("pixel_values", "pixel_values_videos", "images", "videos"):
        dropped.pop(key, None)
    return dropped


def _slice_batch(inputs: Dict[str, Any], sample_idx: int, batch_size: int) -> Dict[str, Any]:
    """Best-effort single-sample slice for nested trainer input dictionaries."""
    sliced: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            sliced[key] = value[sample_idx : sample_idx + 1]
        elif isinstance(value, list) and len(value) == batch_size:
            sliced[key] = value[sample_idx : sample_idx + 1]
        else:
            sliced[key] = value
    return sliced


def _with_masked_cluster_embeds(
    inputs: Dict[str, Any],
    embedding_hidden: torch.Tensor,
    vision_indices: torch.Tensor,
    cluster_onehot: torch.Tensor,
    cluster_idx: int,
) -> Dict[str, Any]:
    masked_inputs = _drop_visual_inputs(inputs)
    inputs_embeds = embedding_hidden.clone()
    cluster_members = cluster_onehot[:, cluster_idx].to(device=vision_indices.device).bool()
    if cluster_members.any():
        positions = vision_indices[cluster_members]
        inputs_embeds[:, positions, :] = 0
    masked_inputs["inputs_embeds"] = inputs_embeds
    return masked_inputs


def _response_indices(inputs: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    labels = inputs.get("labels")
    if labels is None:
        return None
    if labels.ndim == 2:
        labels = labels[0]
    idx = labels.to(device=device).ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    return idx if idx.numel() > 0 else None


def _semantic_direction(delta_log_probs: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Project intervention-induced log-probability shifts into embedding space."""
    vocab = min(delta_log_probs.shape[-1], embedding_weight.shape[0])
    return delta_log_probs[..., :vocab].float() @ embedding_weight[:vocab].to(delta_log_probs.device).float()


def _align_dims(
    student_direction: torch.Tensor,
    teacher_direction: torch.Tensor,
    projector: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projector is not None:
        target_dtype = next(projector.parameters()).dtype
        student_direction = projector(student_direction.to(target_dtype)).float()
        teacher_direction = teacher_direction.to(device=student_direction.device, dtype=torch.float32)
        return student_direction, teacher_direction
    dim = min(student_direction.shape[-1], teacher_direction.shape[-1])
    return student_direction[..., :dim].float(), teacher_direction[..., :dim].float()


class SCVASICCriterion(nn.Module):
    """Joint SCVA + SIC criterion using one full forward plus cluster interventions."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scva = SCVACriterion(args)
        self.ce_weight = float(getattr(args, "scva_sic_ce_weight", 1.0))
        self.lambda_v = float(getattr(args, "scva_sic_lambda_v", 1.0))
        self.lambda_sic = float(getattr(args, "scva_sic_lambda_sic", 1.0))
        self.max_clusters = int(getattr(args, "sic_max_clusters", 0))
        self.use_projector = bool(getattr(args, "sic_use_projector", False))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running SCVA+SIC.")

        student_outputs = distiller.student(**student_inputs)
        with torch.no_grad():
            teacher_outputs = distiller.teacher(**teacher_inputs)

        ce = student_outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return CE loss; labels may be missing.")

        scva_kd = self.scva._scva_loss(student_outputs, teacher_outputs, student_inputs, teacher_inputs)
        sic_kd = self._sic_loss(distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs)

        total = self.ce_weight * ce + self.lambda_v * scva_kd + self.lambda_sic * sic_kd
        out: Dict[str, torch.Tensor] = {
            "loss": total,
            "hard_loss": ce.detach(),
            "scva_loss": scva_kd.detach(),
            "sic_loss": sic_kd.detach(),
        }
        out.update(self.scva._scalarize_counts(ce))
        out.update(self._scalarize_sic_counts(ce))
        return out

    def _scalarize_sic_counts(self, like: torch.Tensor) -> Dict[str, torch.Tensor]:
        counts = getattr(self, "_last_sic_counts", None) or {}
        return {f"sic_{key}": like.new_tensor(float(value)) for key, value in counts.items()}

    def _projector(self, distiller: Any) -> Optional[nn.Module]:
        if not self.use_projector:
            return None
        projectors = getattr(distiller, "projectors", None)
        if projectors is None:
            return None
        if isinstance(projectors, nn.ModuleList) and len(projectors) > 0:
            return projectors[0]
        if isinstance(projectors, nn.ModuleDict) and len(projectors) > 0:
            return next(iter(projectors.values()))
        return None

    def _sic_loss(
        self,
        distiller: Any,
        student_outputs,
        teacher_outputs,
        student_inputs: Dict[str, Any],
        teacher_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        counts: Dict[str, int] = {
            "valid": 0,
            "skipped_mask": 0,
            "skipped_no_clusters": 0,
            "skipped_empty_response": 0,
        }
        self._last_sic_counts = counts

        s_logits = getattr(student_outputs, "logits", None)
        t_logits = getattr(teacher_outputs, "logits", None)
        if s_logits is None or t_logits is None:
            raise RuntimeError("SIC requires both student and teacher logits.")

        student_hidden0 = student_outputs.hidden_states[0]
        teacher_hidden0 = teacher_outputs.hidden_states[0]
        teacher_hidden = _last_hidden(teacher_outputs)
        teacher_vision_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        student_vision_mask = getattr(student_outputs, "vision_feature_mask", None)
        if teacher_vision_mask is None or student_vision_mask is None:
            counts["skipped_mask"] = int(teacher_hidden.shape[0])
            return t_logits.new_zeros(())

        s_embed = _embedding_weight(distiller.student)
        t_embed = _embedding_weight(distiller.teacher)
        projector = self._projector(distiller)

        sample_losses = []
        batch_size = int(teacher_hidden.shape[0])
        for b in range(batch_size):
            t_v_mask = teacher_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            s_v_mask = student_vision_mask[b].to(device=teacher_hidden.device, dtype=torch.bool)
            if t_v_mask.sum().item() < self.scva.min_vision_tokens or s_v_mask.sum().item() < self.scva.min_vision_tokens:
                counts["skipped_mask"] += 1
                continue

            t_resp_idx = _response_indices(_slice_batch(teacher_inputs, b, batch_size), teacher_hidden.device)
            s_resp_idx = _response_indices(_slice_batch(student_inputs, b, batch_size), teacher_hidden.device)
            if t_resp_idx is None or s_resp_idx is None:
                counts["skipped_empty_response"] += 1
                continue

            t_v_idx = t_v_mask.nonzero(as_tuple=True)[0]
            s_v_idx = s_v_mask.nonzero(as_tuple=True)[0]
            n_t_v = int(t_v_idx.numel())
            n_s_v = int(s_v_idx.numel())

            with torch.no_grad():
                teacher_grid = _grid_from_inputs(teacher_inputs, b, n_t_v)
                student_grid = _grid_from_inputs(student_inputs, b, n_s_v)
                cluster_labels = _cluster_vision_tokens_dbscan(
                    teacher_hidden[b, t_v_idx].float(),
                    teacher_grid[0],
                    teacher_grid[1],
                    self.scva.spatial_weight,
                    self.scva.dbscan_min_samples,
                    self.scva.dbscan_eps_percentile,
                )
                teacher_cluster_onehot, remapped_labels = _cluster_onehot_from_labels(cluster_labels)
                student_cluster_onehot = _map_teacher_clusters_to_student_onehot(
                    remapped_labels,
                    teacher_grid,
                    student_grid,
                    n_s_v,
                    teacher_cluster_onehot.shape[-1],
                )

            n_clusters = int(teacher_cluster_onehot.shape[-1])
            if n_clusters == 0 or student_cluster_onehot.sum().item() == 0:
                counts["skipped_no_clusters"] += 1
                continue
            if self.max_clusters > 0:
                n_clusters = min(n_clusters, self.max_clusters)

            t_full_log_probs = F.log_softmax(t_logits[b : b + 1].float(), dim=-1)
            s_full_log_probs = F.log_softmax(s_logits[b : b + 1].float(), dim=-1)
            T_aligned = min(int(t_resp_idx.numel()), int(s_resp_idx.numel()))
            t_resp = t_resp_idx[:T_aligned]
            s_resp = s_resp_idx[:T_aligned]

            # u_t: confidence of the teacher's full-image next-token distribution.
            t_probs_resp = t_full_log_probs[:, t_resp].exp()
            entropy = -(t_probs_resp * t_full_log_probs[:, t_resp]).sum(dim=-1).squeeze(0)
            u = (-entropy / max(math.log(max(t_logits.shape[-1], 2)), 1e-6)).exp()

            t_one = _slice_batch(teacher_inputs, b, batch_size)
            s_one = _slice_batch(student_inputs, b, batch_size)
            cluster_losses = []
            for m in range(n_clusters):
                t_masked_inputs = _with_masked_cluster_embeds(
                    t_one,
                    teacher_hidden0[b : b + 1],
                    t_v_idx,
                    teacher_cluster_onehot,
                    m,
                )
                s_masked_inputs = _with_masked_cluster_embeds(
                    s_one,
                    student_hidden0[b : b + 1],
                    s_v_idx,
                    student_cluster_onehot,
                    m,
                )

                with torch.no_grad():
                    t_masked_outputs = distiller.teacher(**t_masked_inputs)
                s_masked_outputs = distiller.student(**s_masked_inputs)

                t_masked_log_probs = F.log_softmax(t_masked_outputs.logits.float(), dim=-1)
                s_masked_log_probs = F.log_softmax(s_masked_outputs.logits.float(), dim=-1)

                t_delta = t_full_log_probs[:, t_resp] - t_masked_log_probs[:, t_resp]
                s_delta = s_full_log_probs[:, s_resp] - s_masked_log_probs[:, s_resp]
                t_direction = _semantic_direction(t_delta.squeeze(0), t_embed)
                s_direction = _semantic_direction(s_delta.squeeze(0), s_embed)
                s_direction, t_direction = _align_dims(s_direction, t_direction, projector)

                cosine_loss = 1.0 - F.cosine_similarity(s_direction, t_direction, dim=-1, eps=1e-6)
                cluster_losses.append((u.to(cosine_loss.device) * cosine_loss).sum())

            if cluster_losses:
                sample_losses.append(torch.stack(cluster_losses).sum() / u.sum().clamp_min(1e-6))
                counts["valid"] += 1

        if not sample_losses:
            return t_logits.new_zeros(())
        return torch.stack(sample_losses).mean()
