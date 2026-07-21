#!/usr/bin/env bash
# Run all baselines sequentially for both teacher/student pairs:
#   1) Qwen3-VL-4B teacher / FastVLM-0.5B student
#   2) Qwen3-VL-8B teacher / Qwen2.5-VL-3B student
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_DIR

# --- Pair 1: Qwen3-VL-4B teacher / FastVLM-0.5B student ---
PAIR1_DIR="${SCRIPT_DIR}/qwen3_teacher_4b_fastvlm_student_05b"
PAIR1_SCRIPTS=(
  "train_qwen3_teacher_4b_fastvlm_student_05b_ce_only.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_dskd_v2_with_eta.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_dwa_kd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_emkd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_mcw_kd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_scva_cgkd.sh"
  "train_qwen3_teacher_4b_fastvlm_student_05b_sre.sh"
)

# --- Pair 2: Qwen3-VL-8B teacher / Qwen2.5-VL-3B student ---
PAIR2_DIR="${SCRIPT_DIR}/qwen3_teacher_8b_qwen25_student_3b"
PAIR2_SCRIPTS=(
  "train_qwen3_teacher_8b_qwen25_student_3b_ce_only.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_dskd_v2_with_eta.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_dwa_kd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_emkd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_mcw_kd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_scva_cgkd.sh"
  "train_qwen3_teacher_8b_qwen25_student_3b_sre.sh"
)

run_pair() {
  local pair_name="$1"
  local pair_dir="$2"
  shift 2
  local scripts=("$@")

  printf '\n=== Starting pair: %s ===\n' "${pair_name}"

  for script_name in "${scripts[@]}"; do
    script_path="${pair_dir}/${script_name}"
    [[ -f "${script_path}" ]] || { echo "Missing script: ${script_path}" >&2; exit 1; }
    printf '\n[%s] Starting %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
    bash "${script_path}"
    printf '[%s] Finished %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${script_name}"
  done

  printf '\nAll baselines completed successfully for %s.\n' "${pair_name}"
}

run_pair "qwen3_teacher_4b_fastvlm_student_05b" "${PAIR1_DIR}" "${PAIR1_SCRIPTS[@]}"
run_pair "qwen3_teacher_8b_qwen25_student_3b" "${PAIR2_DIR}" "${PAIR2_SCRIPTS[@]}"

printf '\nAll baselines completed successfully for both pairs.\n'