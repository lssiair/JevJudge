"""Train both expanded arms sequentially, without evaluation or API audits."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from src.common import config, dump_json
from src.run_experiment import server, stop_server

def main():
    root = Path("outputs/expanded-seed42")
    root.mkdir(exist_ok=False)
    state = dict(status="starting", arms=["jev", "reference"],
                 global_batch=112, num_generations=8, steps_per_arm=1602,
                 rollouts_per_arm=179424, request_budget=200000)
    dump_json(root / "status.json", state)
    try:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("Port 8000 is occupied; refusing to attach to another server")
        validation = json.loads(Path("results/reward_validation.json").read_text())
        assert validation["status"] == "completed" and validation["scope"] == "full_validation"
        for arm in state["arms"]:
            target = Path(f"outputs/expanded-{arm}-seed42")
            if target.exists():
                raise FileExistsError(f"Refusing to reuse {target}")
            cfg = config(f"configs/grpo_expanded_{arm}.yaml")
            assert cfg["max_steps"] * 7 * cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] == 179424
            dump_json(root / f"{arm}-config.json", cfg)
        for arm in state["arms"]:
            state.update(status="starting_server", active_arm=arm)
            dump_json(root / "status.json", state)
            process, log = server()
            try:
                state.update(status="training", active_arm=arm)
                dump_json(root / "status.json", state)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc_per_node=7", "-m", "src.train_grpo",
                           "--config", f"configs/grpo_expanded_{arm}.yaml",
                           "--output", f"outputs/expanded-{arm}-seed42"]
                print("Starting", arm, flush=True)
                with (root / f"{arm}.log").open("x") as training_log:
                    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"),
                                   stdout=training_log, stderr=subprocess.STDOUT, check=True)
            finally:
                stop_server(process, log)
            state.setdefault("completed_arms", []).append(arm)
            dump_json(root / "status.json", state)
        state.update(status="completed", active_arm=None)
        dump_json(root / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        dump_json(root / "status.json", state)
        raise

if __name__ == "__main__":
    main()
