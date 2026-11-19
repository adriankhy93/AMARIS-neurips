import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import argparse
from utils.layers import (
    layer_name_to_filename,
)
from tqdm import tqdm


def setup_amaris_covariances(
    cov_dir: str | Path,
) -> dict[str, torch.Tensor]:
    """Load amaris covariances

    Returns: dict mapping parameter name (e.g. 'model.layers.0.self_attn.q_proj.weight') to the corresponding input cov matrix

    """
    cov_dir = Path(cov_dir)

    # get linear layers by finding all files matching input_cov*.pt and extracting layer names, to ensure we only attempt to load covariances for layers that were actually computed
    linear_layers = [
        f.stem.replace("_input_cov", "") for f in cov_dir.glob("*input_cov.pt")
    ]
    # linear_layers = [filename_to_layer_name(fname) for fname in linear_layers]
    input_cov_dict: dict[str, torch.Tensor] = {}

    # Collect valid layers and their file paths before loading
    load_tasks = []
    for fname in linear_layers:
        input_cov_pt_path = cov_dir / f"{fname}_input_cov.pt"

        if input_cov_pt_path.exists():
            load_tasks.append((fname, input_cov_pt_path))
        else:
            raise ValueError(
                f"Expected covariance file {input_cov_pt_path} does not exist"
            )

    def _load_pair(
        task: tuple[str, Path],
    ) -> tuple[str, torch.Tensor]:
        fname, input_path = task
        # Leave covariances on CPU to minimise GPU memory usage
        input_cov = torch.load(input_path, map_location="cpu", weights_only=True)
        if not isinstance(input_cov, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor in {input_path}, got {type(input_cov).__name__}"
            )
        return fname, input_cov

    with ThreadPoolExecutor() as executor:
        for fname, input_cov in tqdm(
            executor.map(_load_pair, load_tasks),
            desc="Loading amaris covariances",
            total=len(load_tasks),
        ):
            input_cov_dict[fname] = input_cov

    return input_cov_dict


def main():
    parser = argparse.ArgumentParser(
        description="Combine amaris covariances by addition"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save covariance matrices",
    )

    parser.add_argument(
        "--tags",
        type=str,
        default="command-separated list of tags corresponding to covariance directories to combine (e.g. 'math,if')",
    )

    parser.add_argument(
        "model_num", type=int, help="Model number to use for combining covariances"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    tags = [tag.strip() for tag in args.tags.split(",")]

    for tag in tags:
        if not tag:
            raise ValueError(f"Invalid tag: '{tag}'")

    cov_dirs = [output_path / f"cov_{tag}" for tag in tags]
    # found
    print(f"Found covariance directories: {cov_dirs}")

    cov_dicts = [setup_amaris_covariances(cov_dir) for cov_dir in cov_dirs]

    combined_cov_dict: dict[str, torch.Tensor] = {}
    for lname in cov_dicts[0].keys():
        if all(lname in cov_dict for cov_dict in cov_dicts[1:]):
            # add all covariances (torch tensors) together for layers that are present in both dicts
            combined_cov = cov_dicts[0][lname].clone()
            for cov_dict in cov_dicts[1:]:
                combined_cov += cov_dict[lname]
            combined_cov_dict[lname] = combined_cov
        else:
            raise ValueError(f"Layer {lname} is not present in all covariance dicts")

    # save combined covariances
    save_dir = output_path / f"cov_combined_{args.model_num}/"
    save_dir.mkdir(parents=True, exist_ok=True)

    for lname, cov in combined_cov_dict.items():
        save_path = save_dir / f"{layer_name_to_filename(lname)}_input_cov.pt"
        torch.save(cov, save_path)
        print(f"Saved combined covariance for layer {lname} to {save_path}")


if __name__ == "__main__":
    main()
