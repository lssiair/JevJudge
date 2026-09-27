"""Launch one training arm with an isolated vLLM log, optionally from a full checkpoint."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from src.common import dump_json, config
from src.run_experiment import stop_server

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()
    cfg = config(args.config)
    if Path(args.output).exists():
        raise FileExistsError("Choose a fresh --output, including when resuming")
    if args.resume:
        from src.checkpoints import validate_checkpoint
        validate_checkpoint(args.resume, 7)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Port 8000 occupied; will not attach to another server")
    root = Path(args.output + ".launch")
    root.mkdir(parents=True, exist_ok=False)
    state = dict(status="starting_server", host=socket.gethostname(), **vars(args))
    dump_json(root / "status.json", state)
    process = None
    log = (root / "vllm.log").open("x")
    try:
        process = subprocess.Popen(["bash", "scripts/start_vllm.sh"], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, MODEL_PATH=cfg["model_path"],
                                            VLLM_MAX_MODEL_LEN=str(cfg.get("vllm_max_model_len",2048))))
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError("vLLM exited; inspect vllm.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("vLLM startup timed out")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=7",
                   "-m", "src.train_grpo", "--config", args.config, "--output", args.output,
                   "--save-steps", str(args.save_steps)]
        if args.resume:
            command += ["--resume", args.resume]
        state["status"] = "training"
        dump_json(root / "status.json", state)
        with (root / "training.log").open("x") as training_log:
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"),
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

if __name__ == "__main__":
    main()
