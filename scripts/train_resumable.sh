#!/usr/bin/env bash
export JEV_MAX_REQUESTS="${JEV_MAX_REQUESTS:-200000}"
source "$(dirname "$0")/env.sh"
export JEV_BUDGET_ID="${JEV_BUDGET_ID:-expanded-seed42}"
exec python -u -m src.launch_training "$@"
