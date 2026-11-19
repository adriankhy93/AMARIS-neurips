"""Memory-bounded access to local safetensors checkpoints."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


class SafeTensorCheckpoint:
    """Load one named tensor at a time without materializing a full checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self._weight_map = index.get("weight_map", {})
            if not self._weight_map:
                raise ValueError(f"No weight_map in {index_path}")
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(
                    f"Expected model.safetensors or model.safetensors.index.json in {self.model_dir}"
                )
            with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
                self._weight_map = {key: files[0].name for key in handle.keys()}

    @property
    def keys(self) -> set[str]:
        keys = set(self._weight_map)
        if "model.embed_tokens.weight" in keys:
            keys.add("lm_head.weight")
        return keys

    def get_tensor(self, name: str) -> torch.Tensor:
        if name == "lm_head.weight" and name not in self._weight_map and "model.embed_tokens.weight" in self._weight_map:
            name = "model.embed_tokens.weight"
        try:
            shard_name = self._weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Checkpoint {self.model_dir} has no tensor named {name}") from exc
        shard_path = self.model_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)
