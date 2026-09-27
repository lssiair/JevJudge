#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
export JEV_MAX_REQUESTS=200000
export JEV_BUDGET_ID=expanded-seed42
exec python -u -m src.train_expanded
