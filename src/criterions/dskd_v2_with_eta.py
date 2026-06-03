from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.criterions.various_divergence import VariousDivergence


def get_hidden_states(outputs) -> Tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("DSKDv2 requires model outputs with hidden_states.")
    return tuple(hidden_states)


def get_output_head(model: Any):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "get_output_embeddings"):
        head = encoder.get_output_embeddings()
        if head is not None:
            return head
    if hasattr(encoder, "lm_head"):
        return encoder.lm_head
    raise RuntimeError("Could not find output embedding/lm_head for DSKDv2.")


def call_with_hidden_states(model: nn.Module, inputs: Dict[str, Any]):
    model_inputs = dict(inputs)
    model_inputs.setdefault("output_hidden_states", True)
    return model(**model_inputs)


def project(projector: nn.Module, value: torch.Tensor) -> torch.Tensor:
    target_dtype = next(projector.parameters()).dtype
    return projector(value.to(dtype=target_dtype)).to(dtype=torch.float32)


def require_projector(projectors: Any, name: str) -> nn.Module:
    if not isinstance(projectors, nn.ModuleDict) or name not in projectors:
        raise RuntimeError(
            f"DSKDv2 requires a named projector `{name}` in projector_config_path. "
            "Expected projectors: t2s, s2t."
        )
    return projectors[name]


def align_sequences(
    teacher_tokens: List[str],
    student_tokens: List[str],
    student_tokenizer,
    teacher_tokenizer,
) -> Tuple[List[int], List[int]]:
    i, j = 0, 0
    teacher_align, student_align = [], []
    teacher_history, student_history = "", ""

    teacher_tokens = [str(token).replace("▁", "").replace("Ġ", "") for token in teacher_tokens]
    student_tokens = [str(token).replace("▁", "").replace("Ġ", "") for token in student_tokens]
    teacher_eos = str(getattr(teacher_tokenizer, "eos_token", ""))
    student_eos = str(getattr(student_tokenizer, "eos_token", ""))

    while i < len(teacher_tokens) and j < len(student_tokens):
        same_token = teacher_tokens[i] == student_tokens[j]
        same_eos = teacher_tokens[i] == teacher_eos and student_tokens[j] == student_eos
        if teacher_history == student_history and (same_token or same_eos):
            teacher_history += teacher_tokens[i]
            student_history += student_tokens[j]
            teacher_align.append(i)
            student_align.append(j)
            i += 1
            j += 1
        elif len(teacher_history) > len(student_history):
            student_history += student_tokens[j]
            j += 1
        elif len(teacher_history) < len(student_history):
            teacher_history += teacher_tokens[i]
            i += 1
        else:
            teacher_history += teacher_tokens[i]
            student_history += student_tokens[j]
            i += 1
            j += 1

    return teacher_align, student_align


class DSKDv2WithETACriterion(VariousDivergence):
    """Dual-Space KD v2 with ETA, adapted to the VLM distillation API.

    The reference implementation is text-only. This port restricts every KD
    term to shifted assistant text positions using labels and text_feature_mask.
    """

    def __init__(self, args):
        super().__init__(args)
        self.only_stu_kd = bool(getattr(args, "only_stu_kd", False))
        self.only_tea_kd = bool(getattr(args, "only_tea_kd", False))
        self.init_s2t_projector = bool(getattr(args, "init_s2t_projector", False))
        self.t2s_agreement_threshold = float(getattr(args, "t2s_agreement", 1.0))

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_inputs = batch["student_inputs"]
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise RuntimeError("teacher_inputs are missing while running DSKDv2.")

        student_outputs = call_with_hidden_states(distiller.student, student_inputs)
        labels = student_inputs["labels"].to(device=student_outputs.logits.device)
        supervised_loss_sum = self.compute_cross_entropy_loss(student_outputs.logits, labels)
        _shifted_logits, shifted_labels = self.shift_logits_and_labels(student_outputs.logits, labels)
        supervised_loss = supervised_loss_sum / shifted_labels.ne(self.padding_id).sum().float().clamp_min(1.0)

        with torch.no_grad():
            teacher_outputs = call_with_hidden_states(distiller.teacher, teacher_inputs)

        teacher_labels, _teacher_mask = self.teacher_targets(teacher_inputs, student_outputs.logits.device)
        kd_loss, extra = self._dual_space_kd_loss_with_eta(
            distiller,
            student_inputs,
            teacher_inputs,
            student_outputs,
            teacher_outputs,
            labels,
            teacher_labels,
        )

        loss = (1.0 - self.kd_rate) * supervised_loss + self.kd_rate * kd_loss
        result = {
            "loss": loss,
            "supervised_loss": supervised_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "token_accuracy": self.compute_token_accuracy(student_outputs.logits, labels).detach(),
        }
        result.update({name: value.detach() for name, value in extra.items()})
        return result

    def _dual_space_kd_loss_with_eta(
        self,
        distiller: Any,
        student_inputs: Dict[str, torch.Tensor],
        teacher_inputs: Dict[str, torch.Tensor],
        student_outputs,
        teacher_outputs,
        labels: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_input, target, student_mask = self.shift_inputs_for_causal_targets(
            student_inputs["input_ids"].to(device=labels.device),
            labels,
        )
        teacher_input, teacher_target, teacher_mask = self.shift_inputs_for_causal_targets(
            teacher_inputs["input_ids"].to(device=labels.device),
            teacher_labels.to(device=labels.device),
        )
        del student_input, teacher_input

        student_mask = student_mask & self._shifted_text_mask(student_outputs, target.shape[1], labels.device)
        teacher_mask = teacher_mask & self._shifted_text_mask(teacher_outputs, teacher_target.shape[1], labels.device)
        target = target.masked_fill(~student_mask, self.padding_id)
        teacher_target = teacher_target.masked_fill(~teacher_mask, self.padding_id)

        student_hidden = get_hidden_states(student_outputs)[-1][:, : target.shape[1]]
        teacher_hidden = get_hidden_states(teacher_outputs)[-1][:, : teacher_target.shape[1]].to(device=student_hidden.device)
        student_logits = student_outputs.logits[:, : target.shape[1]]
        teacher_logits = teacher_outputs.logits[:, : teacher_target.shape[1]].to(device=student_hidden.device)

        student_value = self._student_to_teacher_value(distiller, student_hidden)
        teacher_value = project(require_projector(distiller.projectors, "t2s"), teacher_hidden)

        teacher_preds = teacher_logits.argmax(dim=-1)
        teacher_preds = torch.where(teacher_mask, teacher_preds, teacher_target)

        batch_size = target.shape[0]
        t2s_hiddens_align = torch.zeros_like(student_hidden, dtype=teacher_value.dtype)
        s2t_hiddens_align = torch.zeros_like(teacher_hidden, dtype=student_value.dtype)
        t_preds_as_label_rows = []
        align_ratios = []

        student_tokenizer = self._tokenizer(distiller, teacher=False)
        teacher_tokenizer = self._tokenizer(distiller, teacher=True)

        for batch_index in range(batch_size):
            teacher_positions = teacher_mask[batch_index].nonzero(as_tuple=False).flatten()
            student_positions = student_mask[batch_index].nonzero(as_tuple=False).flatten()
            if teacher_positions.numel() == 0 or student_positions.numel() == 0:
                t_preds_as_label_rows.append(torch.full_like(target[batch_index], self.padding_id))
                align_ratios.append(target.new_tensor(1.0, dtype=torch.float32))
                continue

            teacher_span = teacher_positions[0] + torch.arange(
                teacher_positions[-1] - teacher_positions[0] + 1,
                device=teacher_positions.device,
            )
            student_span = student_positions[0] + torch.arange(
                student_positions[-1] - student_positions[0] + 1,
                device=student_positions.device,
            )

            teacher_target_tokens = self._tokens_for_ids(
                teacher_target[batch_index].index_select(0, teacher_span),
                teacher_tokenizer,
            )
            student_target_tokens = self._tokens_for_ids(
                target[batch_index].index_select(0, student_span),
                student_tokenizer,
            )
            teacher_align, student_align = align_sequences(
                teacher_target_tokens,
                student_target_tokens,
                student_tokenizer,
                teacher_tokenizer,
            )
            if not teacher_align and not student_align:
                t_preds_as_label_rows.append(torch.full_like(target[batch_index], self.padding_id))
                align_ratios.append(target.new_tensor(1.0, dtype=torch.float32))
                continue

            align_ratios.append(
                target.new_tensor(float(len(student_align)) / max(float(student_span.numel()), 1.0), dtype=torch.float32)
            )

            cur_t_preds_as_label = torch.full_like(target[batch_index], self.padding_id)
            for teacher_offset, student_offset in zip(teacher_align, student_align):
                teacher_pos = int(teacher_span[teacher_offset].item())
                student_pos = int(student_span[student_offset].item())
                pred_id = int(teacher_preds[batch_index, teacher_pos].item())
                mapped_id = self._teacher_pred_to_student_id(pred_id, student_tokenizer, teacher_tokenizer)
                if mapped_id is None:
                    continue
                cur_t_preds_as_label[student_pos] = mapped_id
                s2t_hiddens_align[batch_index, teacher_pos] = student_value[batch_index, student_pos]
                t2s_hiddens_align[batch_index, student_pos] = teacher_value[batch_index, teacher_pos]
            t_preds_as_label_rows.append(cur_t_preds_as_label)

        align_ratio = torch.stack(align_ratios).mean() if align_ratios else target.new_zeros((), dtype=torch.float32)
        t_preds_as_label = torch.stack(t_preds_as_label_rows, dim=0) if t_preds_as_label_rows else torch.full_like(target, self.padding_id)

        student_head = get_output_head(distiller.student)
        teacher_head = get_output_head(distiller.teacher)
        student_head_weight = student_head.weight.detach().to(device=t2s_hiddens_align.device, dtype=t2s_hiddens_align.dtype)
        t2s_logits = t2s_hiddens_align.matmul(student_head_weight.transpose(-1, -2))
        s2t_logits = teacher_head(s2t_hiddens_align.to(dtype=teacher_hidden.dtype))

        label_mask = t_preds_as_label.ne(self.padding_id)
        stu_align_token_num = label_mask.sum().float().clamp_min(1e-3)
        t2s_agreement_mask = t2s_logits.argmax(dim=-1).eq(t_preds_as_label) & label_mask
        stu_agreement_num = t2s_agreement_mask.sum().float().clamp_min(1e-3)
        valid_student_tokens = student_mask.sum().float().clamp_min(1.0)

        t2s_agreement = t2s_agreement_mask.sum().float() / stu_align_token_num
        t2s_agreement_ratio = t2s_agreement_mask.sum().float() / valid_student_tokens
        t2s_acc_mask = t2s_logits.argmax(dim=-1).eq(target)
        t2s_acc = (t2s_acc_mask & student_mask).sum().float() / valid_student_tokens
        t2s_acc_ratio = t2s_acc_mask.sum().float() / valid_student_tokens
        t_preds_as_label_acc = (t_preds_as_label.eq(target) & label_mask).sum().float() / stu_align_token_num

        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, t_preds_as_label, shift=False) / stu_align_token_num
        t2s_kd_vec = self.dist_func(student_logits, t2s_logits.detach(), target, reduction="none")
        if t2s_agreement <= self.t2s_agreement_threshold:
            t2s_kd_loss = (t2s_kd_vec * t2s_agreement_mask.float()).sum() / stu_agreement_num
        else:
            t2s_kd_loss = (t2s_kd_vec * label_mask.float()).sum() / stu_align_token_num

        s2t_kd_vec = self.dist_func(
            s2t_logits,
            teacher_logits.to(device=s2t_logits.device),
            teacher_target,
            reduction="none",
        )
        s2t_mask = ~s2t_hiddens_align.eq(0).all(dim=-1)
        s2t_kd_loss = (s2t_kd_vec * s2t_mask.float()).sum() / s2t_mask.sum().float().clamp_min(1e-8)

        if self.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        return kd_loss, {
            "align_ratio": align_ratio,
            "t2s_agreement": t2s_agreement,
            "t2s_agreement_ratio": t2s_agreement_ratio,
            "t2s_acc": t2s_acc,
            "t2s_acc_ratio": t2s_acc_ratio,
            "t_preds_as_label_acc": t_preds_as_label_acc,
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "s2t_kd_loss": s2t_kd_loss,
        }

    def _student_to_teacher_value(self, distiller: Any, student_hidden: torch.Tensor) -> torch.Tensor:
        if self.init_s2t_projector and hasattr(distiller, "part_teacher_head_pinv"):
            student_head = get_output_head(distiller.student).weight.detach().transpose(0, 1)
            overlap_ids = getattr(distiller, "student_overlap_token_ids", None)
            if overlap_ids is not None:
                student_head = student_head[:, overlap_ids.to(device=student_head.device)]
            topk_vocab = int(getattr(self.args, "dskd_topk_vocab", getattr(self.args, "topk_vocab", -1)) or -1)
            if topk_vocab != -1:
                student_head = student_head[:, :topk_vocab]
            s2t_projector = student_head.float() @ distiller.part_teacher_head_pinv.float()
            return student_hidden.float() @ s2t_projector.to(device=student_hidden.device)
        return project(require_projector(distiller.projectors, "s2t"), student_hidden)

    def _teacher_pred_to_student_id(self, teacher_id: int, student_tokenizer, teacher_tokenizer) -> int | None:
        if teacher_id == getattr(teacher_tokenizer, "eos_token_id", None):
            return getattr(student_tokenizer, "eos_token_id", None)
        teacher_tokens = self._tokens_for_ids(torch.tensor([teacher_id]), teacher_tokenizer)
        try:
            converted = student_tokenizer.convert_tokens_to_ids(teacher_tokens)
        except Exception:
            return None
        if isinstance(converted, int):
            return converted
        if isinstance(converted, list) and len(converted) == 1 and converted[0] is not None:
            return int(converted[0])
        return None

    def _tokens_for_ids(self, token_ids: torch.Tensor, tokenizer) -> List[str]:
        ids = token_ids.detach().cpu().tolist()
        ids = [int(token_id) for token_id in ids if int(token_id) != self.padding_id]
        if tokenizer is None:
            return [str(token_id) for token_id in ids]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if tokens is None:
            return [str(token_id) for token_id in ids]
        if isinstance(tokens, str):
            tokens = [tokens]
        return [str(token) for token in tokens]

    def _tokenizer(self, distiller: Any, teacher: bool):
        processor_getter = distiller.get_teacher_processor if teacher else distiller.get_student_processor
        try:
            processor = processor_getter()
        except Exception:
            return None
        return getattr(processor, "tokenizer", processor)

    def _shifted_text_mask(self, outputs, target_len: int, device: torch.device) -> torch.Tensor:
        text_mask = getattr(outputs, "text_feature_mask", None)
        if text_mask is None:
            return torch.ones(
                get_hidden_states(outputs)[-1].shape[0],
                target_len,
                dtype=torch.bool,
                device=device,
            )
        text_mask = text_mask.to(device=device, dtype=torch.bool)
        shifted = text_mask[:, :target_len]
        if shifted.shape[1] < target_len:
            pad = torch.zeros(shifted.shape[0], target_len - shifted.shape[1], dtype=torch.bool, device=device)
            shifted = torch.cat([shifted, pad], dim=1)
        return shifted
