"""Exercise the merge CLI and caching without loading real checkpoints."""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "merge.sh"

WORKER = """
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
module = args[2]
if module == 'utils.artifacts':
    raise SystemExit(subprocess.call([sys.executable, *args[1:]]))
with Path(os.environ['MERGE_TEST_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\\n')
def argument(name):
    return args[args.index(name) + 1]
if module == 'amaris_merge.merge_one_step':
    from utils.artifacts import beta_output_dirs
    beta = float(argument('--betas'))
    output = beta_output_dirs([beta], output_base_dir=argument('--output_base_dir'))[beta]
    (output / 'model').mkdir(parents=True)
    (output / 'model/config.json').write_text('{}')
    raise SystemExit(0)
root = Path(argument('--output_dir'))
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
    write(f'cov_{tag}/layer_0_q_proj_input_cov.pt')
    write(f'metadata/{tag}_covariances_metadata.json')
elif module == 'amaris_merge.get_delta':
    write('delta/layer_0_q_proj_delta.pt')
    write('metadata/delta_metadata.json')
elif module == 'amaris_merge.compute_basis_by_percentage':
    alpha = argument('--alpha')
    write(f'basis_alpha_{alpha}/layer_0_q_proj_input_basis.pt')
    write(f'basis_alpha_{alpha}/basis_metadata.json')
else:
    raise AssertionError(module)
"""


class MergeCliTests(unittest.TestCase):
    def config_text(self, root, alpha="0.1", beta="0.5"):
        return "\n".join(
            [
                f'model1_dir: {root / "base"}',
                f'model2_dir: {root / "specialist"}',
                f'datasets1: core:{root / "core.jsonl"}',
                f'dataset2: fin:{root / "finance.jsonl"}',
                "train_proportion: 0.8",
                "max_seq_len: 4",
                "max_num_tokens: 100",
                f"alpha: {alpha}",
                f"beta: {beta}",
                f'output_dir: {root / "output"}',
            ]
        )

    def test_invalid_values_and_arguments_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yaml"
            cases = [
                ("[0.1]", "0.5", [], "alpha"),
                ("0.1,0.2", "0.5", [], "alpha"),
                ("0.1", "[0.5, 1]", [], "beta"),
                ("[0.1]", "0.5", ["--alpha", "0.2"], "alpha"),
                ("0.1", "0.5", ["--alpha", "-0.1"], "alpha"),
                ("0.1", "0.5", ["--beta", "1.1"], "beta"),
                ("0.1", "0.5", ["--alpha", "nan"], "alpha"),
                ("0.1", "0.5", ["--beta", "inf"], "beta"),
                ("0.1", "0.5", ["--alpha", "[0.1,0.2]"], "alpha"),
                ("0.1", "0.5", ["--beta"], "Missing value"),
                ("0.1", "0.5", ["--alpha", "--beta", "0.5"], "Missing value"),
                ("0.1", "0.5", ["--beta", ""], "Missing value"),
                ("0.1", "0.5", ["--unknown", "0.5"], "Unknown argument"),
            ]
            for alpha, beta, flags, error in cases:
                with self.subTest(alpha=alpha, beta=beta, flags=flags):
                    config.write_text(self.config_text(root, alpha, beta))
                    result = subprocess.run(
                        [str(SCRIPT), str(config), *flags],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertFalse((root / "output-shared").exists())

    def test_overrides_produce_one_model_and_reuse_shared_and_basis_stages(self):
        with tempfile.TemporaryDirectory(prefix="merge cli ") as temp:
            root = Path(temp)
            for name in ("base", "specialist"):
                (root / name).mkdir()
                (root / name / "config.json").write_text("{}")
            for name in ("core.jsonl", "finance.jsonl"):
                (root / name).write_text("{}\n")
            config = root / "config.yaml"
            original = self.config_text(root)
            config.write_text(original)
            worker = root / "worker.py"
            worker.write_text(WORKER)
            # Replace expensive stage producers while preserving the real CLI,
            # shell orchestration, completion checks, and artifact inventories.
            runner = f"{shlex.quote(sys.executable)} {shlex.quote(str(worker))}"
            text = re.sub(
                r"^run_cmd\(\) \{.*?^\}",
                lambda _: "run_cmd() {\n  " + runner + ' "$@"\n}',
                SCRIPT.read_text(),
                flags=re.M | re.S,
            )
            script = root / "merge.sh"
            script.write_text(text)
            log = root / "calls.jsonl"
            env = dict(os.environ, PYTHONPATH=str(ROOT), MERGE_TEST_LOG=str(log))

            def run(*flags):
                previous = len(log.read_text().splitlines()) if log.exists() else 0
                result = subprocess.run(
                    ["bash", str(script), str(config), *flags],
                    cwd="/tmp",
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(config.read_text(), original)
                return [
                    json.loads(line) for line in log.read_text().splitlines()[previous:]
                ]

            calls = run()
            self.assertEqual(
                sum(call[2] == "amaris_merge.merge_one_step" for call in calls), 1
            )
            self.assertTrue(
                (root / "output-alpha-0.1-beta-0.50/model/config.json").exists()
            )

            calls = run("--beta", "0.504")
            self.assertEqual(
                [call[2] for call in calls], ["amaris_merge.merge_one_step"]
            )
            self.assertTrue(
                (root / "output-alpha-0.1-beta-0.504/model/config.json").exists()
            )

            calls = run("--alpha", "1e-3", "--beta", "1")
            self.assertEqual(
                [call[2] for call in calls],
                [
                    "amaris_merge.compute_basis_by_percentage",
                    "amaris_merge.merge_one_step",
                ],
            )
            self.assertTrue(
                (root / "output-alpha-0.001-beta-1.00/model/config.json").exists()
            )
            self.assertEqual(len(list(root.glob("output-alpha-*-beta-*/model"))), 3)
            basis_record = root / "output-alpha-0.001/.stages/basis.json"
            self.assertEqual(
                json.loads(basis_record.read_text())["parameters"], ["alpha=0.001"]
            )

            calls = run("--alpha", "0.001", "--beta", "0")
            self.assertEqual(
                [call[2] for call in calls], ["amaris_merge.merge_one_step"]
            )
            self.assertTrue(
                (root / "output-alpha-0.001-beta-0.00/model/config.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
