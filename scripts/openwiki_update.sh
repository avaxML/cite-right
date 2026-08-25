#!/usr/bin/env bash
# Run OpenWiki against a rotating list of free OpenRouter models.
#
# scripts/openwiki_pick_model.py lists :free ids (prefer tool-calling), ranked
# by Artificial Analysis coding_index when present, else context_length, then
# created. This wrapper exports OPENWIKI_MODEL_ID to the first id, then retries
# the next on 429/402/rate-limit, 403/404, agentic-harness blocks, and
# model-unavailable errors. On 429, sleep retry_after_seconds or until
# X-RateLimit-Reset (cap 90s) before the next model. Do not pin a paid model
# such as z-ai/glm-5.2 (without :free).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PICKER="${SCRIPT_DIR}/openwiki_pick_model.py"
MAX_RATE_LIMIT_SLEEP=90

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

# Provider errors that mean "this free model is unusable right now, try the next".
# Real OpenWiki crashes and other unexpected errors still fail the job.
is_retryable_provider_error() {
  local log_file="$1"
  grep -Eiq \
    'HTTP[/ ]*(429|402|403|404)([^0-9]|$)|Error:[[:space:]]*(429|402|403|404)([^0-9]|$)|status["'\''[:space:]=]+(429|402|403|404)([^0-9]|$)|(429|402|403|404)[[:space:]]+(Forbidden|Not Found|Too Many|Payment Required)|rate[-_ ]?limit|too many requests|free-models-per-min|payment required|insufficient.?credit|quota.?exceeded|only available on agentic harnesses|model not found|no endpoints|\bunavailable\b' \
    "${log_file}"
}

is_rate_limit_error() {
  local log_file="$1"
  grep -Eiq \
    'HTTP[/ ]*429([^0-9]|$)|Error:[[:space:]]*429([^0-9]|$)|(429)[[:space:]]+Too Many|rate[-_ ]?limit|too many requests|free-models-per-min' \
    "${log_file}"
}

rate_limit_sleep_seconds() {
  python3 - "${1}" "${MAX_RATE_LIMIT_SLEEP}" <<'PY'
import re
import sys
import time

path, cap_s = sys.argv[1], int(sys.argv[2])
text = open(path, encoding="utf-8", errors="replace").read()
wait = None

match = re.search(
    r"retry_after_seconds[\"'\s:=]+(\d+(?:\.\d+)?)", text, re.IGNORECASE
)
if match:
    wait = float(match.group(1))
else:
    match = re.search(
        r"x-ratelimit-reset[\"'\s:=]+(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if match:
        value = float(match.group(1))
        now = time.time()
        if value >= 10**12:
            wait = value / 1000.0 - now
        elif value >= 10**9:
            wait = value - now
        else:
            wait = value
            if wait > cap_s and wait <= 90_000:
                wait = wait / 1000.0

if wait is None or wait <= 0:
    raise SystemExit(0)
print(min(cap_s, int(wait + 0.999)))
PY
}

sleep_rate_limit_backoff() {
  local log_file="$1"
  local seconds
  seconds="$(rate_limit_sleep_seconds "${log_file}" || true)"
  if [[ -n "${seconds}" && "${seconds}" -gt 0 ]]; then
    echo "Rate limited. Sleeping ${seconds}s (cap ${MAX_RATE_LIMIT_SLEEP}s) before the next model."
    sleep "${seconds}"
  fi
}

last_status=1
model_index=0
for model_id in "${MODELS[@]}"; do
  model_index=$((model_index + 1))
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
  if is_retryable_provider_error "${log_file}"; then
    if is_rate_limit_error "${log_file}" && [[ "${model_index}" -lt "${#MODELS[@]}" ]]; then
      sleep_rate_limit_backoff "${log_file}"
    fi
    echo "OpenWiki failed on ${model_id} with a retryable provider error (429/402/403/404/unavailable). Trying the next free model."
    rm -f "${log_file}"
    continue
  fi
  echo "OpenWiki failed on ${model_id} with a non-retryable error (exit ${last_status})." >&2
  rm -f "${log_file}"
  exit "${last_status}"
done

echo "All ${#MODELS[@]} free OpenRouter models failed with retryable provider errors." >&2
exit "${last_status}"
