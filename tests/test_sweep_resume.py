"""Exercise real shell stage orchestration with lightweight stage producers."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKER = """
import json
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
module = args[2]
if module == 'utils.artifacts':
    raise SystemExit(subprocess.call([sys.executable, *args[1:]]))
def argument(name):
    return args[args.index(name) + 1]
root = Path(argument('--output_dir'))
with (root / 'calls.log').open('a') as log:
    log.write(module + '\\n')
def write(name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('complete')
if module == 'data.tokenize_dataset':
    for entry in argument('--datasets').split(','):
        write('tokenized/' + entry.split(':', 1)[0] + '.jsonl')
elif module == 'data.split':
    for path in (root / 'tokenized').glob('*.jsonl'):
        for split in ('train', 'val'):
            write(f'split_tokenized/{split}_{path.stem}.jsonl')
elif module == 'data.pack':
    for path in (root / 'split_tokenized').glob('*.jsonl'):
        write(f'packed/{path.stem}_packed.pt')
elif module == 'amaris_merge.compute_covariances':
    tag = argument('--tag')
    for layer in (0, 1):
        write(f'cov_{tag}/layer_{layer}_q_proj_input_cov.pt')
    write(f'metadata/{tag}_covariances_metadata.json')
elif module == 'amaris_merge.get_delta':
    for layer in (0, 1):
        write(f'delta/layer_{layer}_q_proj_delta.pt')
    write('metadata/delta_metadata.json')
elif module == 'amaris_merge.combine_covariances':
    for layer in (0, 1):
        write(f'cov_combined_{args[-1]}/layer_{layer}_q_proj_input_cov.pt')
else:
    raise AssertionError(module)
"""


class SweepResumeTests(unittest.TestCase):
    def test_completed_stages_resume_and_missing_layers_trigger_recovery(self):
        script = (ROOT / "merge.sh").read_text()
        names = (
            "trim_whitespace",
            "stage_state",
            "ensure_dir_link",
            "run_shared_preprocessing_once",
            "prepare_combined_covariances",
        )
        functions = "\n".join(
            re.search(r"^" + name + r"\(\) \{.*?^\}", script, re.M | re.S).group()
            for name in names
        )
        functions += "\n" + "\n".join(re.findall(r"^shared_\w+\(\).*$", script, re.M))
        harness = """set -euo pipefail
TEST_PYTHON="$1"
TEST_WORKER="$2"
SHARED_OUTPUT_DIR="$3/shared"
MODEL1_DIR="$3/base"
MODEL2_DIR="$3/specialist"
CONFIG_PATH="$3/config.yaml"
DATASET1="core:$3/data.jsonl,extra:$3/data.jsonl"
DATASET2="fin:$3/data.jsonl"
tags1=(core extra)
tags2=(fin)
all_tags=(core extra fin)
MAX_SEQ_LEN=4 MAX_NUM_TOKENS=100 TRAIN_PROPORTION=0.8
TOKENIZER_STREAMING=false TRUNCATE_PACKED_DOCUMENTS=false
NUM_CHUNKS=1 COV_DTYPE=float32 COV_NUM_SAMPLES=10 LAYERS_TO_IGNORE=""
run_cmd() { "$TEST_PYTHON" "$TEST_WORKER" "$@"; }
"""
        harness += functions + """
run_shared_preprocessing_once
mkdir -p "$SHARED_OUTPUT_DIR/cov_combined"
prepare_combined_covariances 1 tags1 core,extra
prepare_combined_covariances 2 tags2 fin
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("base", "specialist"):
                (root / name).mkdir()
                (root / name / "config.json").write_text("{}")
            (root / "config.yaml").write_text("max_k: 256")
            (root / "data.jsonl").write_text("{}")
            worker = root / "worker.py"
            worker.write_text(WORKER)
            shared = root / "shared"
            partial = shared / "cov_core/layer_0_q_proj_input_cov.pt"
            partial.parent.mkdir(parents=True)
            partial.write_text("interrupted")
            command = [
                "bash",
                "-c",
                harness,
                "sweep-test",
                sys.executable,
                str(worker),
                str(root),
            ]

            def run():
                result = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run()
            self.assertEqual(partial.read_text(), "complete")
            calls = (shared / "calls.log").read_text().splitlines()
            run()
            self.assertEqual((shared / "calls.log").read_text().splitlines(), calls)
            (shared / "delta/layer_1_q_proj_delta.pt").unlink()
            (shared / "cov_core/layer_1_q_proj_input_cov.pt").unlink()
            run()
            recovered = (shared / "calls.log").read_text().splitlines()[len(calls) :]
            self.assertEqual(recovered.count("amaris_merge.compute_covariances"), 2)
            self.assertEqual(recovered.count("amaris_merge.get_delta"), 1)
            self.assertEqual(recovered.count("amaris_merge.combine_covariances"), 1)
            self.assertTrue((shared / "delta/layer_1_q_proj_delta.pt").exists())
            self.assertTrue((shared / "cov_core/layer_1_q_proj_input_cov.pt").exists())


if __name__ == "__main__":
    unittest.main()
