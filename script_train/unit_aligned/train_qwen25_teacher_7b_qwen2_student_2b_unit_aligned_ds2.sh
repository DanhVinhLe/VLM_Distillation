#!/usr/bin/env bash
# Unit-Aligned with DeepSpeed ZeRO-2 — multi-GPU variant.
# Set NPROC_PER_NODE to your GPU count (e.g., NPROC_PER_NODE=4 bash this.sh).
# Switch to configs/ds_z2_offload.json if you OOM on optimizer state (A100 40GB).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/ComfyUI/models/instantid/VLM_Distill}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
DS_CONFIG="${DS_CONFIG:-${PROJECT_DIR}/configs/ds_z2.json}"

STUDENT_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json"
IMAGE_DIR="${PROJECT_DIR}/train_data"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen25_teacher_7b_qwen2_student_2b_unit_aligned_ds2"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

cd "${PROJECT_DIR}"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/script_train/_common.sh"

"${TORCHRUN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --data_path "${DATA_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --percent_data 0.15 \
  --lora true \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --logging_steps 50 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --report_to "${REPORT_TO}" \
  --seed 1337 \
  --kd_loss_type unit_aligned \
  --sre_use_projector true \
  --teacher_layer_mapping 27 \
  --student_layer_mapping 27 \
  --joint_ce_weight 0.5 \
  --joint_emkd_weight 1.0 \
  --joint_sre_weight 1.0 \
  --em_kd_alpha 0.5 \
  --em_kd_beta 0.25 \
  --em_kd_gamma 25.0 \
  --em_kd_temperature 1.0 \
  --sre_alpha 0.5 \
  --sre_p 1.0 \
  --sre_span_loss_weight 1.0 \
  --sre_geom_loss_weight 50 \
  --sre_logit_loss_weight 1.0 \
  --sre_temperature 2.0 \
  --projector_lr 1e-4 \
  --em_kd_max_vision_tokens 512 \
  --em_kd_max_text_tokens 1024 \
  --ds_config "${DS_CONFIG}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
