#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="./runs/ia2c_large"
PID=""
SEND_SIGNAL_DUMP=0

POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --signal-dump)
      SEND_SIGNAL_DUMP=1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  RUN_DIR="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  PID="${POSITIONAL[1]}"
fi

if [[ -z "${PID}" ]]; then
  PID="$(pgrep -f "python .*main.py --base-dir ${RUN_DIR} train" | head -n1 || true)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RUN_DIR}/triage"
OUT_FILE="${OUT_DIR}/freeze_snapshot_${TS}.txt"
mkdir -p "${OUT_DIR}"

{
  echo "timestamp=$(date -Is)"
  echo "run_dir=${RUN_DIR}"
  echo "pid=${PID:-none}"
  echo

  echo "==== disk ===="
  df -h .
  echo

  echo "==== memory ===="
  free -h || true
  echo

  echo "==== processes ===="
  pgrep -af "python .*main.py --base-dir .* train" || true
  if [[ -n "${PID}" ]]; then
    echo
    echo "-- ps tree for pid ${PID} --"
    pstree -ap "${PID}" || true
    echo
    echo "-- ps summary --"
    ps -o pid,ppid,stat,etime,%cpu,%mem,rss,vsz,cmd -p "${PID}" || true
    CHILD_SUMO="$(pgrep -P "${PID}" sumo | head -n1 || true)"
    if [[ -n "${CHILD_SUMO}" ]]; then
      ps -o pid,ppid,stat,etime,%cpu,%mem,rss,vsz,cmd -p "${CHILD_SUMO}" || true
    fi
    echo
    echo "-- top threads snapshot --"
    top -H -b -n 1 -p "${PID}" | sed -n '1,60p' || true
    echo
    echo "-- /proc status --"
    sed -n '1,80p' "/proc/${PID}/status" || true
    if [[ -n "${CHILD_SUMO}" ]]; then
      sed -n '1,80p' "/proc/${CHILD_SUMO}/status" || true
    fi
  fi
  echo

  echo "==== sockets ===="
  ss -tnp | rg "127.0.0.1:(8000|[0-9]+)" || true
  echo

  echo "==== latest log tail ===="
  LATEST_LOG="$(ls -1t "${RUN_DIR}/log/"*.log 2>/dev/null | head -n1 || true)"
  echo "latest_log=${LATEST_LOG}"
  if [[ -n "${LATEST_LOG}" ]]; then
    tail -n 200 "${LATEST_LOG}" || true
  fi
  echo

  echo "==== watchdog dump tail ===="
  if [[ -f "${RUN_DIR}/data/stall_watchdog_stacks.log" ]]; then
    tail -n 200 "${RUN_DIR}/data/stall_watchdog_stacks.log" || true
  else
    echo "missing ${RUN_DIR}/data/stall_watchdog_stacks.log"
  fi
  echo

  echo "==== manual signal dump ===="
  if [[ "${SEND_SIGNAL_DUMP}" -eq 1 && -n "${PID}" ]]; then
    echo "sending SIGUSR1 to ${PID}"
    kill -USR1 "${PID}" || true
    sleep 1
    if [[ -f "${RUN_DIR}/data/manual_signal_stacks.log" ]]; then
      tail -n 120 "${RUN_DIR}/data/manual_signal_stacks.log" || true
    else
      echo "missing ${RUN_DIR}/data/manual_signal_stacks.log"
    fi
  elif [[ "${SEND_SIGNAL_DUMP}" -eq 1 ]]; then
    echo "no pid detected, skipping SIGUSR1"
  else
    echo "disabled (pass --signal-dump to enable)"
  fi
} > "${OUT_FILE}" 2>&1

echo "Wrote freeze triage snapshot to ${OUT_FILE}"
