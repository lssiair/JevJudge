import json
import pytest
from src.checkpoints import validate_checkpoint

def test_reject_weights_only(tmp_path):
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="optimizer.pt"):
        validate_checkpoint(tmp_path, 2)

def test_full_checkpoint_and_missing_rank(tmp_path):
    for name in ["model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state_0.pth", "rng_state_1.pth"]:
        (tmp_path / name).touch()
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 50}))
    assert validate_checkpoint(tmp_path, 2) == tmp_path
    (tmp_path / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="rng_state_1.pth"):
        validate_checkpoint(tmp_path, 2)

def test_missing_weight_shard(tmp_path):
    for name in ["model-00001.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        (tmp_path / name).touch()
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 50}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
    with pytest.raises(ValueError, match="model-00002"):
        validate_checkpoint(tmp_path, 1)
