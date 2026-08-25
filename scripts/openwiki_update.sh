#!/usr/bin/env bash
# Run OpenWiki against a rotating list of free OpenRouter models.
#
# scripts/openwiki_pick_model.py lists :free ids (prefer tool-calling), ranked
# by Artificial Analysis coding_index when present, else context_length, then
# created. This wrapper exports OPENWIKI_MODEL_ID to the first id, then retries
# the next on 429 / 402 / rate-limit. Do not pin a paid model such as
# z-ai/glm-5.2 (without :free).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PICKER="${SCRIPT_DIR}/openwiki_pick_model.py"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "::warning::OPENROUTER_API_KEY is not set. Add the GitHub Actions secret, then re-run this workflow (workflow_dispatch or next merge to main)."
  if [[ "${GITHUB_EVENT_NAME:-}" == "workflow_dispatch" ]]; then
    exit 1
  fi
  exit 0
fi

if ! command -v openwiki >/dev/null 2>&1; then
  echo "openwiki is not on PATH. Install with: npm install --global openwiki@latest" >&2
  exit 1
fi

pick_out="$(mktemp)"
if ! python3 "${PICKER}" >"${pick_out}"; then
  rm -f "${pick_out}"
  echo "Failed to pick free OpenRouter models." >&2
  exit 1
fi
mapfile -t MODELS < "${pick_out}"
rm -f "${pick_out}"
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No free OpenRouter models selected." >&2
  exit 1
fi

cd "${REPO_ROOT}"

is_rate_or_payment_error() {
  local log_file="$1"
  grep -Eiq '([^0-9]|^)(429|402)([^0-9]|$)|rate[-_ ]?limit|too many requests|payment required|insufficient.?credit|quota.?exceeded' "${log_file}"
}

last_status=1
for model_id in "${MODELS[@]}"; do
  export OPENWIKI_MODEL_ID="${model_id}"
  echo "Running OpenWiki with OPENWIKI_MODEL_ID=${OPENWIKI_MODEL_ID}"
  log_file="$(mktemp)"
  set +e
  openwiki code --update --print >"${log_file}" 2>&1
  last_status=$?
  set -e
  cat "${log_file}"
  if [[ "${last_status}" -eq 0 ]]; then
    rm -f "${log_file}"
    exit 0
  fi
  if is_rate_or_payment_error "${log_file}"; then
    echo "OpenWiki failed on ${model_id} with 429/402/rate-limit. Trying the next free model."
    rm -f "${log_file}"
    continue
  fi
  echo "OpenWiki failed on ${model_id} with a non-rate-limit error (exit ${last_status})." >&2
  rm -f "${log_file}"
  exit "${last_status}"
done

echo "All ${#MODELS[@]} free OpenRouter models failed with 429/402/rate-limit." >&2
exit "${last_status}"
