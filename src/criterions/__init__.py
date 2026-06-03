from src.criterions.ce_only import CEOnlyCriterion
from src.criterions.cgkd import CGKDCriterion
from src.criterions.default_distillation import (
    DefaultDistillationCriterion,
    default_distillation_criterion,
)
from src.criterions.dwa_kd import DWAKDCriterion
from src.criterions.dskd_v2_with_eta import DSKDv2WithETACriterion
from src.criterions.mcw_kd import MCWKDCriterion
try:
    from src.criterions.em_kd import EMKDCriterion
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise

    class EMKDCriterion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("EM-KD requires scipy. Install requirements.txt before using kd_loss_type='em_kd'.")

try:
    from src.criterions.scva import SCVACriterion
    from src.criterions.scva_cgkd import SCVACGKDCriterion
except ModuleNotFoundError as exc:
    if exc.name != "hdbscan":
        raise

    class SCVACriterion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("SCVA requires hdbscan. Install requirements.txt before using kd_loss_type='scva'.")

    class SCVACGKDCriterion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "SCVA+CGKD requires hdbscan. Install requirements.txt before using kd_loss_type='scva_cgkd'."
            )
from src.criterions.sre import SRECriterion
try:
    from src.criterions.unit_aligned import UnitAlignedDistillationCriterion
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise

    class UnitAlignedDistillationCriterion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Unit-Aligned distillation requires scipy. Install requirements.txt before using kd_loss_type='unit_aligned'."
            )


criterion_list = {
    "ce_only": CEOnlyCriterion,
    "default": DefaultDistillationCriterion,
    "default_distillation": DefaultDistillationCriterion,
    "emkd": EMKDCriterion,
    "em_kd": EMKDCriterion,
    "sre": SRECriterion,
    "dwa_kd": DWAKDCriterion,
    "dwakd": DWAKDCriterion,
    "dskd_v2_with_eta": DSKDv2WithETACriterion,
    "dskdv2_with_eta": DSKDv2WithETACriterion,
    "dskd_v2": DSKDv2WithETACriterion,
    "dskdv2": DSKDv2WithETACriterion,
    "mcw_kd": MCWKDCriterion,
    "mcwkd": MCWKDCriterion,
    "joint": UnitAlignedDistillationCriterion,
    "unit_aligned": UnitAlignedDistillationCriterion,
    "unit_aligned_distillation": UnitAlignedDistillationCriterion,
    "scva": SCVACriterion,
    "cgkd": CGKDCriterion,
    "scva_cgkd": SCVACGKDCriterion,
    "draft": SCVACGKDCriterion,
}


def build_criterion(training_args):
    kd_loss_type = getattr(training_args, "kd_loss_type", None)
    if kd_loss_type in (None, ""):
        kd_loss_type = "ce_only"

    if kd_loss_type in criterion_list:
        return criterion_list[kd_loss_type](training_args)

    raise ValueError(f"Unsupported kd_loss_type: {kd_loss_type}")


__all__ = [
    "DefaultDistillationCriterion",
    "DWAKDCriterion",
    "DSKDv2WithETACriterion",
    "MCWKDCriterion",
    "CEOnlyCriterion",
    "EMKDCriterion",
    "SCVACriterion",
    "CGKDCriterion",
    "SCVACGKDCriterion",
    "SRECriterion",
    "UnitAlignedDistillationCriterion",
    "build_criterion",
    "criterion_list",
    "default_distillation_criterion",
]
