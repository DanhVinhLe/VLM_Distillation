from src.criterions.default_distillation import (
    DefaultDistillationCriterion,
    default_distillation_criterion,
)
from src.criterions.em_kd import EMKDCriterion
from src.criterions.sre import SRECriterion


criterion_list = {
    "default": DefaultDistillationCriterion,
    "default_distillation": DefaultDistillationCriterion,
    "emkd": EMKDCriterion,
    "em_kd": EMKDCriterion,
    "sre": SRECriterion,
}


def build_criterion(training_args):
    kd_loss_type = getattr(training_args, "kd_loss_type", None)
    if kd_loss_type in (None, ""):
        kd_loss_type = "default"

    if kd_loss_type in criterion_list:
        return criterion_list[kd_loss_type](training_args)

    raise ValueError(f"Unsupported kd_loss_type: {kd_loss_type}")


__all__ = [
    "DefaultDistillationCriterion",
    "EMKDCriterion",
    "SRECriterion",
    "build_criterion",
    "criterion_list",
    "default_distillation_criterion",
]
