# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned stack: `transformers==5.2.0`, `torch==2.9.0`, `peft==0.19.1`, `accelerate==1.13.0` (see `requirements.txt`). Dataset download and the expected `train_data/` layout are documented in `SETUP.md`.

## Common commands

Training is launched via `torchrun` against `train.py`. Curated launch scripts live under `script_train/`:

```bash
# SFT (no teacher)
bash script_train/sft/train_qwen2_vl_2b_sft.sh

# Distillation (student + teacher, kd_loss_type selects the criterion)
bash script_train/em_kd/train_qwen25_teacher_7b_qwen2_student_2b_emkd.sh
bash script_train/train_qwen25_teacher_qwen2_student_lora_test.sh   # SRE smoke test
```

Every script hard-codes `PROJECT_DIR` near the top — **always update it for the target machine** before running. `NPROC_PER_NODE` and `MASTER_PORT` are env-overridable. `--percent_data 0.01` is set on most scripts as a smoke-test default; raise it for real runs.

Resume training with `--resume_from auto` (latest checkpoint in `output_dir`) or `--resume_from <step>`; `none` disables resume.

There is no formal test suite, lint config, or build step. Standalone smoke-test scripts:

```bash
python test_load_dataset.py --data_path ... --student_model ...  # dataset + collator
python test_loadmodel/test_qwen2_vl.py                            # per-backbone load
```

## Architecture

### Two training modes, one entry point

`train.py:main()` dispatches based on whether `--teacher_model_name` is set (`_has_teacher`):

- **SFT mode** → builds a single `VLMModel` (`src/model/model.py`). `DistillTrainer.compute_loss` forwards the batch directly and uses the LM's own CE loss against `labels`.
- **Distillation mode** → builds a `Distiller` (`src/distiller.py`) wrapping student + frozen teacher + optional projectors, plus a criterion from `build_criterion(training_args)`. `DistillTrainer.compute_loss` calls `model(criterion, batch)`, which delegates to `criterion(distiller, batch)`.

The trainer detects the mode at construction time via `isinstance(model, Distiller)` — both code paths share the same Trainer class.

### Backbone abstraction

`src/model/processor.py` registers six backbones: `fast_vlm`, `llava_next`, `llava_onevision`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`. Each has a vendored implementation under `src/model/vlm_backbone/<name>/` (forked HF model code, not the upstream package) and a backbone-specific processor loader. `VLMModel._load_base_model` resolves the backbone from the HF config and instantiates the right class via `backbone2model`.

Two non-obvious config mutations happen at load time:
- `_force_eager_attention` sets `_attn_implementation="eager"` and forces `output_hidden_states=True`, `output_attentions=True` on the main config and all sub-configs (`text_config`, `vision_config`, …). KD criteria depend on these being on.
- `padding_side` is forced to `"left"` on both config and tokenizer.

LoRA is wired through `peft`. `init_lora_model`/`load_pretrained_lora`/`checkpoint_path` interact in `_wrap_lora_if_needed`: a checkpoint path with `lora=true` loads an existing adapter; bare `lora=true` initializes a new DoRA adapter using `lora_target_modules` (or, for LLaVA-OV/Next, those modules scoped to `language_model`).

### Distiller, projectors, optimizer

`Distiller.__init__` loads the student (trainable) via `VLMModel.build` and the teacher via `VLMModel.load`, then freezes the teacher's parameters and switches it to inference mode. Teacher `ModelArguments` are derived by `_create_teacher_model_args` — only the `teacher_*` fields on `ModelArguments` are mirrored into a fresh dataclass.

`set_projector` builds projectors two ways:
1. If `projector_config_path` is set, parse a JSON of named `{enabled, structure}` entries (e.g. `"1s-relu-1t"` where `s`/`t` resolve to student/teacher hidden dims) into an `nn.ModuleDict`.
2. Otherwise, build one `Linear(student_hidden_dim → teacher_hidden_dim)` per entry in `teacher_layer_mapping` as an `nn.ModuleList`.

`DistillTrainer.create_optimizer` calls `super().create_optimizer()`, then if `projector_lr` differs from `learning_rate`, pulls projector params out of the existing groups and adds them as a new group at `projector_lr`. Checkpoints save the student via `VLMModel.save` and projectors to `<output_dir>/projectors/` separately.

### KD criteria registry

`src/criterions/__init__.py` maps `--kd_loss_type` strings to criterion classes:

| `kd_loss_type` | Class | Key idea |
|---|---|---|
| `ce_only` *(default)* | `CEOnlyCriterion` | Supervised CE only; teacher still in batch for fair compute-matched baseline. |
| `default` / `default_distillation` | `DefaultDistillationCriterion` | Hidden-state MSE on mean-pooled layers (per `*_layer_mapping`), else temperature-KL on response logits. |
| `emkd` / `em_kd` | `EMKDCriterion` | Reverse-KL on response logits + Hungarian-matched vision-logit distillation + vision↔text affinity. Reads `em_kd_alpha/beta/gamma/temperature` from CLI args. Optional `em_kd_max_vision_tokens` / `em_kd_max_text_tokens` budget caps. |
| `sre` | `SRECriterion` | Weighted cosine span loss + geometry MSE + shared-vocab soft-label KL. Hidden alignment via learnable student→teacher projector (`--sre_use_projector true`, from `distiller.projectors[idx]`). |
| `joint` / `unit_aligned` / `unit_aligned_distillation` | `UnitAlignedDistillationCriterion` | Joint SRE + EM-KD, single shared student/teacher forward pass. Weighted by `joint_ce_weight/emkd_weight/sre_weight`. |

All criteria receive `(distiller, batch)` and must return `{"loss": ...}` (additional keys are detached metrics). They depend on `output.hidden_states` and (for some) `output.attentions` being populated, which is why `_force_eager_attention` is non-negotiable.

`EMKDCriterion` and `SRECriterion` expose `compute_losses(distiller, student_outputs, teacher_outputs, student_inputs, teacher_inputs)` alongside `forward(distiller, batch)` so the joint criterion can run them without paying 2× forward cost.

Loss weights live on `TrainingArguments` (`src/arguments.py`): `kd_weight`; `em_kd_alpha/beta/gamma/temperature` (+ optional `em_kd_max_vision_tokens`, `em_kd_max_text_tokens`); `sre_alpha/p/span_loss_weight/geom_loss_weight/logit_loss_weight/temperature/use_projector`; `joint_ce_weight/emkd_weight/sre_weight`.

### Data pipeline

`src/data/dataset.py`:

- `LazyVlmDistillDataset` reads LLaVA-style JSON/JSONL (`{id, image, conversations: [{from, value}]}`). Image paths are resolved against `image_dir` (or the JSON's directory), and samples whose image file is **missing on disk are silently skipped** — count is tracked in `skipped_missing_images`. `_normalize_conversations` maps `human/gpt/user/assistant/system` roles into a uniform `{role, value}` list, stripping `<image>` tokens (the chat template re-adds them).
- `__getitem__` builds an OAI-style `messages` list with the image attached to the first user turn only.
- `VlmDistillDataCollator` applies `apply_chat_template` separately for student and teacher (when present), tokenizing into `student_inputs` / `teacher_inputs`. `_add_labels` calls `_make_labels_chatml`, which scans for `<|im_start|>assistant…<|im_end|>` token spans and masks **everything else with `-100`**. This works because every supported backbone uses a ChatML-style template.
- When `kd_loss_type=sre` **and** a teacher is configured, the collator additionally computes per-sample LCS span alignment (`_prepare_sre_pooler`) over the student/teacher offset mappings, writing `pooler_safe_idx`/`pooler_mask` into both input dicts. `SRECriterion._has_pooler` switches to a span-pooled hidden loss when these are present.

`DistillTrainer` disables `remove_unused_columns` because batches are nested dicts (`student_inputs`, `teacher_inputs`, plus metadata keys in `_BATCH_METADATA_KEYS` that get stripped in `_build_sft_model_inputs`).

### Folders that are NOT part of the main pipeline

- `sre/` (top-level, not `src/criterions/sre.py`) — standalone reference SRE implementation for text-only LLMs. The in-tree `SRECriterion` does **not** import from it ("self-contained adaptation" per the docstring). Treat as reference material only.
- `test_loadmodel/` — per-backbone smoke tests, useful for verifying a checkpoint loads but not run as part of training.

## Conventions worth knowing

- **Argument plumbing is dataclass-based.** Add new flags by extending `ModelArguments`, `DataArguments`, or `TrainingArguments` in `src/arguments.py`; HF's `HfArgumentParser` picks them up automatically. `TrainingArguments` subclasses HF's, so any HF flag works too.
- **`data_args.kd_loss_type` is set in `train.py`** (not declared on `DataArguments`) before building the data module so the collator can branch on it without seeing `TrainingArguments`. If you move this assignment, the SRE pooler path silently turns off.
- **`bf16` is the only tested precision.** `_load_base_model` hard-codes `torch_dtype=torch.bfloat16` and projectors are cast to bf16; scripts pass `--bf16 true`.
- **Single-GPU and multi-GPU share the same launch command.** `VLMModel._distributed_context` derives rank/world from `LOCAL_RANK`/`WORLD_SIZE` env vars set by `torchrun`. DeepSpeed/FSDP is gated by `ACCELERATE_USE_DEEPSPEED`/`ACCELERATE_USE_FSDP`; `--deepspeed_config` and `--ds_config` are aliases (`_maybe_set_deepspeed_alias`).
- **Logging is intentionally quiet.** `disable_verbose_logs()` silences `transformers`, `httpx`, `huggingface_hub`, etc. before parsing args. Use `print_master`/`print_rank` from `src/utils.py` for rank-aware output.
