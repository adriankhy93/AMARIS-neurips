import subprocess
import unittest
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'merge.sh'
CONFIGS = sorted((ROOT / 'config').glob('*.yaml'))


class SweepConfigurationTests(unittest.TestCase):
    def test_sweep_script_has_valid_bash_syntax(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)

    def test_configs_use_active_alpha_beta_schema_only(self):
        self.assertGreater(len(CONFIGS), 0)
        for config_path in CONFIGS:
            with self.subTest(config=config_path.name):
                config = config_path.read_text()
                self.assertIn('alpha:', config)
                self.assertIn('output_dir:', config)
                self.assertNotIn('generate_completions:', config)
                self.assertNotIn('apply_assistant_mask:', config)
                if 'beta:' in config:
                    beta_value = next(line for line in config.splitlines() if line.startswith('beta:'))
                    self.assertNotEqual(beta_value.split(':', 1)[1].strip(), '[]')
                self.assertIn('enable_delta_variant:', config)
                self.assertIn('layers_to_ignore:', config)
                self.assertIn('basis_gpus: all', config)
                self.assertRegex(config, r'(?m)^num_chunks: \d+(?:\s+#.*)?$', msg=config_path.name)
                self.assertIn('cov_dtype: float32', config)
                self.assertRegex(config, r'(?m)^cov_num_samples: \d+(?:\s+#.*)?$', msg=config_path.name)
                for key in ('tokenizer_streaming', 'truncate_packed_documents'):
                    self.assertRegex(config, rf'(?m)^{key}: (true|false)(?:\s+#.*)?$')
                values = {line.split(':', 1)[0]: line.split(':', 1)[1].split('#', 1)[0].strip()
                          for line in config.splitlines() if ':' in line and not line.startswith('#')}
                for key in ('alpha', 'beta'):
                    self.assertGreaterEqual(float(values[key]), 0)
                    self.assertLessEqual(float(values[key]), 1)
                for key in ('max_k', 'lobpcg_max_iterations', 'lobpcg_max_retries',
                            'num_chunks', 'cov_num_samples'):
                    self.assertGreater(int(values[key]), 0, key)
                for key in ('lobpcg_tolerance', 'lobpcg_max_ridge_scale'):
                    self.assertGreater(float(values[key]), 0, key)
                self.assertIn(values['basis_solver'], ('lobpcg', 'dense'))

    def test_sweep_defaults_beta_and_names_artifacts_by_alpha_and_beta(self):
        script = SCRIPT.read_text()
        self.assertIn('get_key_or_default "beta" "1.0"', script)
        self.assertIn('ALPHA_OUTPUT_DIR="${BASE_OUTPUT_DIR}-alpha-${alpha}${VARIANT_SUFFIX}"', script)
        self.assertIn('--basis_dir "$ALPHA_OUTPUT_DIR/basis_alpha_${alpha}"', script)
        self.assertIn('--output_base_dir "$ALPHA_OUTPUT_DIR"', script)
        self.assertIn('--betas "$BETA"', script)
        self.assertIn('BASIS_GPUS="$(get_key_or_default "basis_gpus" "all")"', script)
        self.assertIn('get_key_or_default "enable_delta_variant" "false"', script)
        self.assertIn('VARIANT_SUFFIX="-delta-variant"', script)
        self.assertIn('basis_args+=(--enable_delta_variant)', script)
        self.assertIn('basis_metadata.json', script)
        self.assertIn('Remove only generated basis artifacts', script)
        self.assertIn('get_key_or_default "cov_dtype" "float32"', script)
        self.assertIn('get_key_or_default "cov_num_samples" "10000"', script)
        self.assertIn('get_key_or_default "tokenizer_streaming" "false"', script)
        self.assertIn('get_key_or_default "truncate_packed_documents" "false"', script)
        self.assertIn('pack_args+=(--truncate_documents)', script)
        self.assertIn('--cov_dtype "$COV_DTYPE"', script)
        self.assertIn('--num_samples "$COV_NUM_SAMPLES"', script)
        self.assertIn('tokenize_args+=(--streaming)', script)
        self.assertIn('get_key_or_default "basis_solver" "lobpcg"', script)
        self.assertIn('--basis_solver "$BASIS_SOLVER"', script)
        self.assertIn("printf '%.12g' \"$ALPHA\"", script)
        self.assertIn('--lobpcg_max_retries "$LOBPCG_MAX_RETRIES"', script)
        self.assertIn('--candidate_cache_dir "$SHARED_OUTPUT_DIR/generalized_eigenpairs${VARIANT_SUFFIX}"', script)
        self.assertGreaterEqual(script.count('--layers_to_ignore "$LAYERS_TO_IGNORE"'), 3)


if __name__ == '__main__':
    unittest.main()
