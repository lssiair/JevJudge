#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
exec python -m src.run_experiment "$@"
