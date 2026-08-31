#!/usr/bin/env bash
# Tear down only the demo process this verification run started.
# Never kill by process name. Never delete VERIFY_EVIDENCE_DIR.
set -euo pipefail

PID_FILE="${CITE_RIGHT_VERIFY_PID_FILE:-}"
if [[ -z "${PID_FILE}" ]]; then
  if [[ -n "${VERIFY_RUN_ID:-}" ]]; then
    PID_FILE="/tmp/cite-right-verify-${VERIFY_RUN_ID}.uvicorn.pid"
  else
    echo "teardown: no CITE_RIGHT_VERIFY_PID_FILE or VERIFY_RUN_ID; nothing to stop"
    exit 0
  fi
fi

if [[ ! -f "${PID_FILE}" ]]; then
  echo "teardown: no pid file at ${PID_FILE}; nothing to stop"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if [[ -z "${pid}" ]]; then
  rm -f "${PID_FILE}"
  echo "teardown: empty pid file removed"
  exit 0
fi

if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}" || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" || true
  fi
  echo "teardown: stopped pid ${pid}"
else
  echo "teardown: pid ${pid} already gone"
fi
rm -f "${PID_FILE}"
