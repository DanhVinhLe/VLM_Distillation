#!/usr/bin/env bash
# Shared training-script helpers. Source from each script after setting RUN_NAME.
#
# Provides:
#   - Loads ${PROJECT_DIR}/.env if present (WANDB_API_KEY, HF_TOKEN, etc.)
#   - REPORT_TO    : "wandb" if WANDB_API_KEY set, otherwise "none". Pass via --report_to.
#   - HUB_FLAGS[]  : --push_to_hub / --hub_model_id / --hub_token / --hub_strategy when
#                    HF_TOKEN and HF_HUB_REPO_PREFIX are both set. Expand with
#                    ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"} to be safe under `set -u`.
#
# Requires RUN_NAME to be set before sourcing (used for WANDB_NAME and HF repo id).

if [[ -z "${RUN_NAME:-}" ]]; then
  echo "_common.sh: RUN_NAME must be set before sourcing." >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# ----- WandB -----
REPORT_TO="${REPORT_TO:-none}"
if [[ -n "${WANDB_API_KEY:-}" && "${REPORT_TO}" == "none" ]]; then
  REPORT_TO="wandb"
fi
if [[ "${REPORT_TO}" == "wandb" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-vlm-distillation}"
  export WANDB_NAME="${WANDB_NAME_PREFIX:-}${RUN_NAME}"
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
  fi
fi

# ----- HF Hub -----
HUB_FLAGS=()
if [[ -n "${HF_TOKEN:-}" && -n "${HF_HUB_REPO_PREFIX:-}" ]]; then
  HUB_FLAGS=(
    --push_to_hub true
    --hub_model_id "${HF_HUB_REPO_PREFIX}${RUN_NAME}"
    --hub_token "${HF_TOKEN}"
    --hub_strategy "${HF_HUB_STRATEGY:-every_save}"
  )
  if [[ "${HF_HUB_PRIVATE:-true}" == "true" ]]; then
    HUB_FLAGS+=(--hub_private_repo true)
  fi
fi

export REPORT_TO
