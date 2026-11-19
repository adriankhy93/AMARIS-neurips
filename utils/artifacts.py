"""Output naming and atomic completion records for experiment stages."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def beta_output_dirs(betas, *, output_base_dir=None, output_dir=None):
    if not betas or any(not math.isfinite(beta) or not 0 <= beta <= 1 for beta in betas):
        raise ValueError("Betas must be a nonempty list of finite values in [0, 1]")
    if len(set(betas)) != len(betas):
        raise ValueError("Duplicate beta values are not allowed")
    if output_dir is not None:
        if len(betas) != 1:
            raise ValueError("A single output directory requires one beta")
        return {betas[0]: Path(output_dir)}
    outputs = {}
    for beta in betas:
        # Retain historical names for values representable with two decimals.
        label = f"{beta:.2f}" if float(f"{beta:.2f}") == beta else repr(beta)
        outputs[beta] = Path(f"{output_base_dir}-beta-{label}")
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Beta output paths collide")
    return outputs


def inventory(root, patterns):
    records = {}
    for pattern in patterns:
        files = sorted(path for path in root.glob(pattern) if path.is_file())
        if not files:
            raise ValueError(f"No artifacts match {pattern}")
        for path in files:
            stat = path.stat()
            records[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return records


def input_inventory(paths):
    records = {}
    for name in paths:
        path = Path(name).resolve()
        if not path.exists():
            raise ValueError(f"Missing stage input: {path}")
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        for file in files:
            stat = file.stat()
            records[str(file)] = [stat.st_size, stat.st_mtime_ns]
    return records


def stage_state(action, root, stage, config, patterns, inputs=(), parameters=()):
    """Only finish after the stage command succeeds; partial outputs never count."""
    manifest = root / '.stages' / f'{stage}.json'
    if action == 'begin':
        manifest.unlink(missing_ok=True)
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    path.unlink()
        return True
    try:
        record = {
            'schema_version': 1,
            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'inputs': input_inventory(inputs),
            'artifacts': inventory(root, patterns),
        }
        if parameters:
            record['parameters'] = sorted(parameters)
        if action == 'check':
            return json.loads(manifest.read_text()) == record
    except (OSError, ValueError):
        if action == 'check':
            return False
        raise
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(manifest)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'begin', 'finish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', action='append', required=True)
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--parameter', action='append', default=[])
    args = parser.parse_args()
    if not stage_state(args.action, args.root, args.stage, args.config, args.output,
                       args.input, args.parameter):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
