# **AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance**.

[NeurIPS 2026 poster page](https://neurips.cc/virtual/2026/loc/paris/poster/154547)

AMARIS transfers specialist updates along selected activation subspaces while controlling a regularized base-output perturbation budget. It uses calibration statistics without additional training and produces a single static Hugging Face checkpoint, with no inference-time adapters or routing. The [`merge.sh`](merge.sh) script runs the merge pipeline and saves one checkpoint per invocation.

The configurations and commands below are general usage examples. They are not complete recipes for reproducing the paper's results.

## Requirements

- Linux with Bash 4.3 or newer.
- NVIDIA GPU(s) with enough memory for the models and covariance/basis computation, plus sufficient RAM and disk space for intermediate artifacts and output checkpoints.
- A CUDA-enabled Python environment with PyTorch, Transformers, Accelerate, Datasets, Safetensors, tqdm, and FlashAttention 2. The pipeline loads models with `flash_attention_2`.
- Two compatible local Hugging Face model directories with matching architecture, parameter shapes, and tokenizer vocabulary. Supply Safetensors weights, either a single file or indexed shards, along with model configuration and tokenizer files.
- **Two calibration sets:** a generalist set covering tasks where the generalist model performs better than the specialist, and a specialist set covering tasks where the specialist performs better than the generalist. Each set can contain one or more datasets.

The [`dockerfile`](dockerfile) describes the GPU environment. From the repository root, build and enter it with Docker and NVIDIA Container Toolkit installed:

```bash
docker build -f dockerfile -t amaris .
docker run --gpus all --rm -it \
  -v "$PWD:/workspace/AMARIS" -w /workspace/AMARIS amaris bash
```

Mount model and dataset directories into the container as well if they are outside the repository, and use their container paths in your configuration.

**`merge.sh` invokes `/usr/bin/python3` explicitly.** Dependencies must be available to that interpreter; activating another virtual environment alone does not change it. The Dockerfile also creates `/opt/vllm-env` for optional response generation. That separate environment is not used by the merge launcher. The image uses an upstream `latest` tag rather than a pinned environment.

Check the merge interpreter before launching:

```bash
/usr/bin/python3 -c 'import torch, transformers, accelerate, datasets, safetensors, tqdm, flash_attn; print("CUDA available:", torch.cuda.is_available())'
```

## 1. Configure the merge

Copy an example configuration, such as [`config/qwen-finance.yaml`](config/qwen-finance.yaml):

```bash
cp config/qwen-finance.yaml config/my-merge.yaml
```

Edit the configuration before running. The supplied paths refer to the original development machine. Set these fields to your own locations:

```yaml
model1_dir: "/path/to/base-model"
model2_dir: "/path/to/specialist-model"
datasets1: "core:/path/to/core-calibration.jsonl"
dataset2: "fin:/path/to/finance-calibration.jsonl"
output_dir: "outputs/my-merge"
alpha: 0.001
beta: 0.5
```

`model1_dir` is the generalist (base) checkpoint; `model2_dir` is the specialist. **Both calibration sets are required:**

- **Generalist calibration set (`datasets1`):** tasks where the generalist performs better than the specialist, representing capabilities you want to preserve.
- **Specialist calibration set (`dataset2`):** tasks where the specialist performs better than the generalist, representing capabilities you want to transfer.

Each calibration set can combine two or more datasets, such as math and instruction-following datasets. Supply comma-separated `tag:path` entries for either set; you do not need to concatenate the files manually. For example:

```yaml
datasets1: "math:/path/to/math.jsonl,instruction:/path/to/instruction-following.jsonl"
dataset2: "finance_qa:/path/to/finance-qa.jsonl,finance_sentiment:/path/to/finance-sentiment.jsonl"
```

Choose which set a dataset belongs to based on the relative strengths of your two checkpoints. Both sets are tokenized with the generalist model's tokenizer; covariance collection uses the corresponding model for each set. Use distinct dataset tags across both sets.

A supported calibration format is JSONL with one conversation per line, including the assistant response:

```json
{"messages":[{"role":"user","content":"What is compound interest?"},{"role":"assistant","content":"Compound interest is interest earned on both principal and accumulated interest."}]}
```

### Prepare calibration data

Calibration datasets are not bundled with the example configurations. Prepare both sets using prompts that reflect the relative strengths described above, keeping evaluation examples separate. Export completed conversations in the JSONL format above, then list the files for the generalist set in `datasets1` and those for the specialist set in `dataset2`. Each set may use one file or multiple tagged files.

If you have prompts without responses, [`data/generate.py`](data/generate.py) can generate completions. For example, in the Docker environment, use the base model for core prompts:

```bash
/opt/vllm-env/bin/python -m data.generate \
  --model /path/to/base-model \
  --capability_datasets core:/path/to/core-prompts.jsonl \
  --num_samples 1000 \
  --max_seq_len 4096 --max_new_tokens 2048 \
  --output_dir calibration_data/base
```

Prompt files can contain rows such as `{"prompt":"Explain compound interest."}`. Repeat with the specialist model, specialist prompts, a distinct tag such as `fin`, and a separate output directory. Without `--thinking`, the example writes `calibration_data/base/core_no-think_text.jsonl`; use that completed-conversation file as the core calibration input. Generation settings and sample counts here are illustrative, not paper reproduction settings.

Prepare the datasets before launching; `merge.sh` does not generate model responses. Relative model, dataset, and output paths in the configuration resolve from the repository root. Keep configuration keys at the top level, as in the supplied files; the launcher uses a simple parser rather than a full YAML parser.

### Choose alpha and beta

**Start with a low alpha, such as `0.001` or `0.0001`.** Evaluate the merged model on both base and specialist tasks, then increase alpha gradually if more specialist transfer is needed. A smaller perturbation budget does not guarantee better downstream performance.

```yaml
alpha: 0.001
beta: 0.5
```

- **Alpha** controls the regularized base-output perturbation budget used to select the projection basis. Smaller values impose a stricter budget. It is not a direct weight-averaging coefficient.
- **Beta** mixes parameters without projected updates: `beta * base + (1 - beta) * specialist`. These include embeddings, normalization parameters, biases, and non-projected linear weights. `beta: 1` keeps those parameters at their base values; `beta: 0` takes them from the specialist. For a projected weight, AMARIS applies `W_base + (W_specialist - W_base) U Uᵀ`; beta does not scale this update or mix its complementary directions.

Each setting must be a single finite number between 0 and 1; lists are rejected, including lists containing one number. Alpha is required; omitted beta defaults to `1.0`. Command-line flags override these defaults without editing the configuration:

```bash
./merge.sh config/my-merge.yaml --alpha 0.001 --beta 0.5
```

```bash
CUDA_VISIBLE_DEVICES=3 ./merge.sh config/my-merge.yaml --alpha 0.001 --beta 0.5
```

You can override either setting independently. Run separate commands to compare different values; each command saves one checkpoint with alpha and beta in its output path.

Try **beta values from `0.0` to `1.0` in steps of `0.1`**, keeping alpha fixed while comparing them. For example:

```bash
for beta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  ./merge.sh config/my-merge.yaml --alpha 0.001 --beta "$beta" || break
done
```

This runs 11 merges and saves 11 checkpoints. Valid shared preprocessing and the basis for this alpha are reused. Compare the checkpoints on held-out generalist and specialist tasks to choose beta.

### Other configuration controls

| Key | Purpose |
| --- | --- |
| `train_proportion` | Fraction of tokenized examples assigned to the training split; the remainder is held out. |
| `max_seq_len` | Sequence length used during data preparation and packing. |
| `max_num_tokens` | Token budget applied when tokenizing each tagged dataset. |
| `cov_num_samples` | Maximum packed samples used for covariance collection. |
| `num_chunks` | Increase to reduce covariance GPU memory use at the cost of additional passes. |
| `cov_dtype` | Covariance precision: `float32` or `float64`. |
| `basis_gpus` | `all` uses all visible GPUs for basis construction. |
| `max_k` | Caps candidate/selected rank for the LOBPCG solver. Lower values can reduce compute and memory, while limiting the available subspace. |
| `basis_solver` | `lobpcg` for iterative solving or `dense` for full eigendecomposition. |
| `lobpcg_*` | Tolerance, iteration limit, numerical retry count, and maximum ridge scale. |
| `layers_to_ignore` | Excludes matching layers from covariance, delta, basis, and projection stages; their parameters use beta mixing. |
| `tokenizer_streaming` | Streams source datasets during tokenization when enabled. |
| `truncate_packed_documents` | Truncates individual documents beyond `max_seq_len` during packing when enabled. |
| `enable_delta_variant` | Enables experimental delta-aware basis ranking; leave `false` for the standard path. |

## 2. Run the merge

From the repository root:

```bash
./merge.sh config/my-merge.yaml
```

Select GPUs through the environment, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./merge.sh config/my-merge.yaml
```

The script tokenizes, splits, and packs the datasets; computes model-specific covariances and checkpoint deltas; combines covariances for each model; constructs a basis for the effective alpha; and saves one merged model using the effective beta. It stops if a stage fails.

To inspect usage or check shell syntax without starting a merge:

```bash
./merge.sh --help
bash -n merge.sh
```

## 3. Find the merged model

For `output_dir: "outputs/my-merge"`, `alpha: 0.001`, and `beta: 0.5`, the main outputs are:

```text
outputs/
├── my-merge-shared/                 # Prepared data, covariances, deltas, caches
├── my-merge-alpha-0.001/             # Basis and links to shared artifacts
└── my-merge-alpha-0.001-beta-0.50/
    ├── model/                      # Loadable model and tokenizer
    └── amaris_run.json              # Merge metadata and diagnostics
```

Use `outputs/my-merge-alpha-0.001-beta-0.50/model` as the model path for inference or evaluation. Running with different alpha/beta values creates separate checkpoint directories. With `enable_delta_variant: true`, alpha directory names include `-delta-variant`.

## Resume and evaluate

Rerun the same command to resume an interrupted run. Preprocessing and basis stages reuse verified completion records when the configuration, tracked inputs, and outputs match. Changing only CLI alpha/beta reuses valid shared preprocessing; a different alpha gets its own basis, and changing beta reuses the basis for the same alpha. Incomplete or invalidated stages are recomputed. Editing the config file conservatively invalidates stage records, so use CLI overrides when comparing alpha/beta values. The final merge/save stage runs again, so existing checkpoint outputs can be rewritten. Use a new `output_dir` to keep separate experiments.

The merge does not automatically choose the best checkpoint or run benchmark evaluation. Compare saved models on your base and specialist tasks. An optional validation-loss command, using the held-out packs created by the merge, is:

```bash
/usr/bin/python3 -m utils.validate_models \
  --model base /path/to/base-model \
  --model amaris-alpha-0.001-beta-0.50 outputs/my-merge-alpha-0.001-beta-0.50/model \
  --dataset core outputs/my-merge-shared/packed/val_core_packed.pt \
  --dataset fin outputs/my-merge-shared/packed/val_fin_packed.pt \
  --output outputs/my-merge-validation.json
```

Substitute your actual model paths and dataset tags. Surrogate-tightness analysis is an optional diagnostic and is not required to create merged checkpoints.