#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
export JEV_MAX_REQUESTS=200000
export JEV_BUDGET_ID=expanded-seed42
exec python -u - <<'PY'
import os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
from src.common import dump_json
from src.run_experiment import stop_server

target = Path("outputs/expanded-jev-seed42-retry1")
if target.exists():
    raise FileExistsError(f"Refusing to reuse {target}")
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise RuntimeError("Port 8000 occupied")
root = Path("outputs/expanded-jev-retry1-launch")
root.mkdir(exist_ok=False)
state = dict(status="starting_server", host=socket.gethostname(), arm="jev",
             steps=1602, global_batch=112, num_generations=8, output=str(target))
dump_json(root / "status.json", state)
log = (root / "vllm.log").open("x")
process = None
try:
    process = subprocess.Popen(["bash", "scripts/start_vllm.sh"], stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError("vLLM exited; inspect vllm.log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("vLLM startup timed out")
    state["status"] = "training"
    dump_json(root / "status.json", state)
    with (root / "training.log").open("x") as training_log:
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=7", "-m", "src.train_grpo",
                        "--config", "configs/grpo_expanded_jev.yaml", "--output", str(target)],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"),
                       stdout=training_log, stderr=subprocess.STDOUT, check=True)
    state["status"] = "completed"
except BaseException as exc:
    state.update(status="failed", error=str(exc))
    raise
finally:
    if process is not None:
        stop_server(process, log)
    else:
        log.close()
    dump_json(root / "status.json", state)
PY
