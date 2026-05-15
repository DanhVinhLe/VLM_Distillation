#!/usr/bin/env bash
# CE-only baseline (teacher in batch, KD discarded) — DeepSpeed ZeRO-2 multi-GPU.
# Launch:   NPROC_PER_NODE=4 bash script_train/ce_only/train_..._ce_only_ds2.sh
# Tight VRAM? Switch with: DS_CONFIG="${PROJECT_DIR}/configs/ds_z2_offload.json"
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/ComfyUI/models/instantid/VLM_Distill}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
DS_CONFIG="${DS_CONFIG:-${PROJECT_DIR}/configs/ds_z2.json}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen25_teacher_7b_qwen2_student_2b_ce_only_ds2}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"
PERCENT_DATA="${PERCENT_DATA:-0.10}"

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
  --percent_data "${PERCENT_DATA}" \
  --lora true \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 50 \
  --dataloader_num_workers 2 \
  --max_len 2048 \
  --image_resolution low \
  --resume_from none \
  --report_to "${REPORT_TO}" \
  --seed 1337 \
  --kd_loss_type ce_only \
  --ds_config "${DS_CONFIG}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
