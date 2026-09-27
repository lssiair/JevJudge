"""Validate full DDP checkpoints before resuming training."""
import json
from pathlib import Path


def validate_checkpoint(path, world_size=7):
    root = Path(path).resolve()
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    required += ([f"rng_state_{rank}.pth" for rank in range(world_size)]
                 if world_size > 1 else ["rng_state.pth"])
    missing = [name for name in required if not (root / name).is_file()]
    weights = list(root.glob("model*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not weights:
        missing.append("model weights")
    for index in root.glob("*.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map", {})
        missing.extend(name for name in set(mapping.values()) if not (root / name).is_file())
    if missing:
        raise ValueError(f"Incomplete checkpoint {root}: missing {sorted(set(missing))}")
    state = json.loads((root / "trainer_state.json").read_text())
    if not isinstance(state.get("global_step"), int) or state["global_step"] < 1:
        raise ValueError("Checkpoint has no positive global_step")
    return root
