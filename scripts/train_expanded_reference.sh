#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
exec python -u - <<'PY'
import os, socket, subprocess, sys
from pathlib import Path
from src.common import dump_json
from src.run_experiment import server, stop_server

target = Path("outputs/expanded-reference-seed42")
if target.exists():
    raise FileExistsError(f"Refusing to reuse {target}")
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise RuntimeError("Port 8000 occupied; refusing to use another server")
status_path = Path("outputs/expanded-reference-launch-status.json")
state = dict(status="starting_server", arm="reference", steps=1602, global_batch=112, num_generations=8)
dump_json(status_path, state)
process = log = None
try:
    process, log = server()
    state["status"] = "training"
    dump_json(status_path, state)
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=7",
                    "-m", "src.train_grpo", "--config", "configs/grpo_expanded_reference.yaml",
                    "--output", str(target)],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"), check=True)
    state["status"] = "completed"
except BaseException as exc:
    state.update(status="failed", error=str(exc))
    raise
finally:
    if process is not None:
        stop_server(process, log)
    dump_json(status_path, state)
PY
