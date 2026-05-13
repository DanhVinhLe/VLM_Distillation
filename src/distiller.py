import json
import os

import torch
import torch.nn as nn

from src.arguments import ModelArguments, TrainingArguments
from src.model.model import VLMModel
from src.model.processor import load_processor
from src.utils import print_master, print_rank


def _create_teacher_model_args(model_args: ModelArguments) -> ModelArguments:
    return ModelArguments(
        model_name=model_args.teacher_model_name,
        model_backbone=model_args.teacher_backbone,
        lora=model_args.teacher_lora,
        lora_r=model_args.teacher_lora_r,
        lora_alpha=model_args.teacher_lora_alpha,
        lora_dropout=model_args.teacher_lora_dropout,
        lora_target_modules=model_args.teacher_lora_target_modules,
        pooling=model_args.teacher_pooling,
        normalize=model_args.teacher_normalize,
    )


def _init_semi_orthogonal(tensor: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor.shape
    if rows >= cols:
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q[:, :cols]
    else:
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q.T[:rows, :]
    return tensor


class Distiller(nn.Module):
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args

        self.student = self._load_student()
        self.teacher = self._load_teacher()

        self.student_hidden_dim = model_args.student_hidden_dim
        self.teacher_hidden_dim = model_args.teacher_hidden_dim

        self.set_projector()
        self.load_projectors_if_needed()
        print_master("Projectors set.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_student(self) -> VLMModel:
        print_master(
            f"Loading student: {self.model_args.model_name} "
            f"(lora={self.model_args.lora}, r={self.model_args.lora_r})"
        )
        student = VLMModel.build(self.model_args)
        print_master("Student model built.")
        return student

    def _load_teacher(self) -> VLMModel:
        teacher_model_args = _create_teacher_model_args(self.model_args)
        print_master(
            f"Loading teacher: {teacher_model_args.model_name} "
            f"(lora={teacher_model_args.lora}, r={teacher_model_args.lora_r})"
        )
        teacher = VLMModel.load(teacher_model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        print_master("Teacher model loaded.")
        return teacher

    # ------------------------------------------------------------------
    # Processor helpers
    # ------------------------------------------------------------------

    def get_student_processor(self):
        if hasattr(self, "_student_processor"):
            return self._student_processor
        processor = load_processor(self.model_args)
        self._student_processor = processor
        print_master("Student processor loaded.")
        return processor

    def get_teacher_processor(self):
        if hasattr(self, "_teacher_processor"):
            return self._teacher_processor
        teacher_model_args = _create_teacher_model_args(self.model_args)
        processor = load_processor(teacher_model_args)
        self._teacher_processor = processor
        print_master("Teacher processor loaded.")
        return processor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, criterion, batch):
        return criterion(self, batch)

    # ------------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------------

    def set_projector(self):
        if self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            with open(self.model_args.projector_config_path) as f:
                projector_config = json.load(f)

            dim_map = {"s": self.student_hidden_dim, "t": self.teacher_hidden_dim}

            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue

                seq = nn.Sequential()
                parts = cfg["structure"].split("-")
                parsed = []

                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    else:
                        suffix = p[-1]
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * dim_map[suffix])

                for i in range(len(parsed) - 1):
                    a, b = parsed[i], parsed[i + 1]
                    if isinstance(a, int) and isinstance(b, int):
                        layer = nn.Linear(a, b)
                        _init_semi_orthogonal(layer.weight)
                        seq.append(layer.to(dtype=torch.bfloat16))
                    elif b == "relu":
                        seq.append(nn.ReLU())
                    elif a == "relu" and isinstance(b, int):
                        prev = parsed[i - 1] if i > 0 and isinstance(parsed[i - 1], int) else None
                        if prev is not None:
                            layer = nn.Linear(prev, b)
                            _init_semi_orthogonal(layer.weight)
                            seq.append(layer.to(dtype=torch.bfloat16))

                self.projectors[name] = seq
        else:
            projector_list = nn.ModuleList()
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector_list.append(
                    nn.Linear(self.student_hidden_dim, self.teacher_hidden_dim, dtype=torch.bfloat16)
                )
            self.projectors = projector_list

        print_master(f"Created {len(self.projectors)} projector(s).")

    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors") and self.projectors is not None:
            lr = self.model_args.projector_lr or self.training_args.learning_rate
            optimizer.add_param_group({"params": self.projectors.parameters(), "lr": lr})
            print_master("Projector parameters added to optimizer.")
        return optimizer

    def save_projectors(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(self.projectors, nn.ModuleDict):
            for name, proj in self.projectors.items():
                path = os.path.join(save_dir, f"projector_{name}.pt")
                torch.save(proj.state_dict(), path)
                print_master(f"Projector '{name}' saved to {path}")
        else:
            for i, proj in enumerate(self.projectors):
                path = os.path.join(save_dir, f"projector_{i}.pt")
                torch.save(proj.state_dict(), path)
                print_master(f"Projector {i} saved to {path}")

    def load_projectors_if_needed(self):
        if not self.model_args.projector_path:
            return

        path = self.model_args.projector_path
        if isinstance(self.projectors, nn.ModuleDict):
            if os.path.isfile(path):
                state = torch.load(path, map_location="cpu")
                self.projectors.load_state_dict(state)
                print_master(f"Projectors loaded from {path}")
                return

            for name, proj in self.projectors.items():
                proj_path = os.path.join(path, f"projector_{name}.pt")
                if os.path.exists(proj_path):
                    proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
                    print_master(f"Projector '{name}' loaded from {proj_path}")
            return

        if os.path.isfile(path):
            state = torch.load(path, map_location="cpu")
            self.projectors.load_state_dict(state)
            print_master(f"Projectors loaded from {path}")
            return

        for i, proj in enumerate(self.projectors):
            proj_path = os.path.join(path, f"projector_{i}.pt")
            if os.path.exists(proj_path):
                proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
                print_master(f"Projector {i} loaded from {proj_path}")
