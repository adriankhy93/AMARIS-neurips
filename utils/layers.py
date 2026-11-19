"""Shared utilities for enumerating and naming linear layers."""

import torch.nn as nn


def get_linear_layers(
    model: nn.Module, layer_filter: str = "all"
) -> dict[str, nn.Linear]:
    """Return a dict of name → nn.Linear for targeted linear layers.

    Args:
        model: The model to enumerate.
        layer_filter: One of "all", "attn", or "mlp".
    """
    attn_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_suffixes = ("up_proj", "down_proj", "gate_proj")
    lm_head_suffix = ("lm_head",)

    if layer_filter == "attn":
        target_suffixes = attn_suffixes
    elif layer_filter == "mlp":
        target_suffixes = mlp_suffixes
        # include lm_head
        target_suffixes += lm_head_suffix
    else:
        target_suffixes = attn_suffixes + mlp_suffixes + lm_head_suffix

    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(s) for s in target_suffixes
        ):
            layers[name] = module
    return layers


def layer_name_to_filename(name: str) -> str:
    """Convert 'model.layers.0.self_attn.q_proj' → 'layer_0_q_proj'."""
    parts = name.split(".")
    layer_idx = None
    proj_type = parts[-1]
    for p in parts:
        if p.isdigit():
            layer_idx = p
            break
    if layer_idx is None:
        return name.replace(".", "_")
    return f"layer_{layer_idx}_{proj_type}"


def filename_to_layer_name(filename: str) -> str:
    """
    Convert 'layer_0_q_proj' → 'model.layers.0.mlp.q_proj'.
    Convert 'layer_0_k_proj' → 'model.layers.0.self_attn.k_proj'.
    Convert 'lm_head' → 'lm_head'.
    """
    parts = filename.split("_")
    if len(parts) < 3 or parts[0] != "layer":
        raise ValueError(f"Invalid filename format: {filename}")
    layer_idx = parts[1]
    proj_type = "_".join(parts[2:])
    if proj_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"model.layers.{layer_idx}.self_attn.{proj_type}"
    elif proj_type in ("up_proj", "down_proj", "gate_proj"):
        return f"model.layers.{layer_idx}.mlp.{proj_type}"
    elif proj_type == "lm_head":
        return "lm_head"
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")
