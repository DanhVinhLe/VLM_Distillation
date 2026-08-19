#!/usr/bin/env bash
# SCVA + SIC joint — semantic intervention consistency, DeepSpeed ZeRO-2 multi-GPU.
# Launch:   NPROC_PER_NODE=2 bash script_train/scva_sic/train_..._scva_sic_ds2.sh
# Tight VRAM? Switch to configs/ds_z2_offload.json via DS_CONFIG.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gdi-user/enguyen/research_vllm/test/VLM_Distillation}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="/home/gdi-user/miniconda3/envs/distil/bin/torchrun"
DS_CONFIG="${DS_CONFIG:-${PROJECT_DIR}/configs/ds_z2.json}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen25_teacher_7b_qwen2_student_2b_scva_sic_ds2}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
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
  --kd_loss_type scva_sic \
  --scva_n_clusters 16 \
  --scva_kmeans_iters 10 \
  --scva_min_vision_tokens 4 \
  --scva_sic_ce_weight "${SCVA_SIC_CE_WEIGHT:-1.0}" \
  --scva_sic_lambda_v "${SCVA_SIC_LAMBDA_V:-0.3}" \
  --scva_sic_lambda_sic "${SCVA_SIC_LAMBDA_SIC:-0.3}" \
  --sic_num_regions "${SIC_NUM_REGIONS:-${SIC_MAX_CLUSTERS:-16}}" \
  --sic_max_clusters "${SIC_MAX_CLUSTERS:-0}" \
  --sic_use_projector "${SIC_USE_PROJECTOR:-true}" \
  --ds_config "${DS_CONFIG}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
