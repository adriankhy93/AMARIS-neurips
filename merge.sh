#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <config_path> [--alpha VALUE] [--beta VALUE]"
  echo "Run one merge. Flags override the scalar values in the configuration."
}

if [[ "$#" -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 1
fi

CONFIG_PATH="$1"
shift
ALPHA_OVERRIDE=""
BETA_OVERRIDE=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --alpha|--beta)
      option="$1"
      if [[ "$#" -lt 2 || "$2" == --* ]]; then
        echo "Missing value for $option" >&2
        exit 1
      fi
      if [[ "$option" == --alpha ]]; then
        ALPHA_OVERRIDE="$2"
      else
        BETA_OVERRIDE="$2"
      fi
      # Reject empty arguments instead of silently using the config default.
      if [[ -z "$2" ]]; then
        echo "Missing value for $option" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Resolve before changing to the repository root.
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PWD/$CONFIG_PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

yaml_get() {
  local key="$1"
  local file="$2"

  awk -v k="$key" '
    BEGIN { FS=":" }
    $0 ~ "^[[:space:]]*" k "[[:space:]]*:" {
      sub("^[[:space:]]*" k "[[:space:]]*:[[:space:]]*", "", $0)
      sub(/[[:space:]]+#.*/, "", $0)
      gsub(/^"|"$/, "", $0)
      gsub(/^\047|\047$/, "", $0)
      gsub(/[[:space:]]+$/, "", $0)
      print $0
      exit
    }
  ' "$file"
}

require_key() {
  local key="$1"
  local value
  value="$(yaml_get "$key" "$CONFIG_PATH")"
  if [[ -z "$value" ]]; then
    echo "Missing required key in $CONFIG_PATH: $key" >&2
    exit 1
  fi
  printf '%s' "$value"
}

require_any_key() {
  local keys_csv="$1"
  local key
  local value

  IFS=',' read -r -a _keys <<< "$keys_csv"
  for key in "${_keys[@]}"; do
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="$(yaml_get "$key" "$CONFIG_PATH")"
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return 0
    fi
  done

  echo "Missing required key in $CONFIG_PATH: one of [$keys_csv]" >&2
  exit 1
}

get_key_or_default() {
  local key="$1"
  local default_value="$2"
  local value

  value="$(yaml_get "$key" "$CONFIG_PATH")"
  if [[ -z "$value" ]]; then
    printf '%s' "$default_value"
  else
    printf '%s' "$value"
  fi
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

validate_parameter() {
  /usr/bin/python3 - "$1" "$2" <<'PY'
import math
import sys

name, raw = sys.argv[1:]
try:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError
except ValueError:
    sys.exit(f"{name} must be one finite scalar number in [0, 1], got {raw!r}; lists are not supported")
print(value)
PY
}

run_cmd() {
  local -a cmd=("$@")
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

has_any_file_matching() {
  local search_dir="$1"
  local pattern="$2"
  [[ -d "$search_dir" ]] && find "$search_dir" -name "$pattern" -print -quit | grep -q .
}

ensure_dir_link() {
  local link_path="$1"
  local target_path="$2"

  if [[ -L "$link_path" || -e "$link_path" ]]; then
    rm -rf "$link_path"
  fi
  ln -s "$target_path" "$link_path"
}

parse_tags() {
  local dataset_csv="$1"
  local -n tags_ref="$2"

  tags_ref=()
  IFS=',' read -r -a dataset_entries <<< "$dataset_csv"
  for entry in "${dataset_entries[@]}"; do
    entry="$(trim_whitespace "$entry")"
    local tag="${entry%%:*}"
    if [[ -z "$entry" || "$entry" == "$tag" || -z "$tag" ]]; then
      echo "Invalid dataset entry: '$entry' (expected format tag:path)" >&2
      exit 1
    fi
    tags_ref+=("$tag")
  done
}

build_all_tags() {
  local -n tags1_ref="$1"
  local -n tags2_ref="$2"
  local -n out_ref="$3"
  local tag
  declare -A seen=()

  out_ref=()
  for tag in "${tags1_ref[@]}" "${tags2_ref[@]}"; do
    if [[ -z "${seen[$tag]+x}" ]]; then
      out_ref+=("$tag")
      seen[$tag]=1
    fi
  done
}

tags_to_csv() {
  local -n tags_ref="$1"
  local csv=""
  local tag

  for tag in "${tags_ref[@]}"; do
    if [[ -z "$csv" ]]; then
      csv="$tag"
    else
      csv+=",$tag"
    fi
  done

  printf '%s' "$csv"
}

# A completion record is written only after every command in a stage succeeds.
stage_state() {
  local action="$1" stage="$2" tag entry
  local -a state_args=("$action" --root "$SHARED_OUTPUT_DIR" --stage "$stage" --config "$CONFIG_PATH")
  local -a stage_patterns=() upstream=()
  case "$stage" in
    tokenized)
      for tag in "${all_tags[@]}"; do stage_patterns+=("tokenized/${tag}.jsonl"); done ;;
    split)
      for tag in "${all_tags[@]}"; do stage_patterns+=("split_tokenized/train_${tag}.jsonl" "split_tokenized/val_${tag}.jsonl"); done
      upstream+=(tokenized) ;;
    packed)
      for tag in "${all_tags[@]}"; do stage_patterns+=("packed/train_${tag}_packed.pt" "packed/val_${tag}_packed.pt"); done
      upstream+=(split) ;;
    covariance1|covariance2)
      local tags_name=tags1
      [[ "$stage" == covariance2 ]] && tags_name=tags2
      local -n stage_tags="$tags_name"
      for tag in "${stage_tags[@]}"; do stage_patterns+=("cov_${tag}/*_input_cov.pt" "metadata/${tag}_covariances_metadata.json"); done
      upstream+=(packed) ;;
    delta)
      stage_patterns=("delta/*_delta.pt" "metadata/delta_metadata.json") ;;
    combined1|combined2)
      local number="${stage#combined}"
      stage_patterns+=("cov_combined_${number}/*_input_cov.pt")
      upstream+=("covariance${number}") ;;
  esac
  for entry in "${stage_patterns[@]}"; do state_args+=(--output "$entry"); done
  for entry in "${upstream[@]}"; do state_args+=(--input "$SHARED_OUTPUT_DIR/.stages/${entry}.json"); done
  state_args+=(--input "$MODEL1_DIR" --input "$MODEL2_DIR")
  local -a source_entries=()
  IFS=',' read -r -a source_entries <<< "$DATASET1,$DATASET2"
  for entry in "${source_entries[@]}"; do
    entry="$(trim_whitespace "$entry")"
    # Hub dataset identifiers are covered by the configuration fingerprint;
    # local sources additionally contribute their current file inventory.
    if [[ -e "${entry#*:}" ]]; then state_args+=(--input "${entry#*:}"); fi
  done
  run_cmd /usr/bin/python3 -m utils.artifacts "${state_args[@]}"
}

shared_tokenized_done() { stage_state check tokenized; }
shared_split_done() { stage_state check split; }
shared_packed_done() { stage_state check packed; }
shared_covariances_done_model1() { stage_state check covariance1; }
shared_covariances_done_model2() { stage_state check covariance2; }
shared_deltas_done() { stage_state check delta; }

prepare_combined_covariances() {
  local model_num="$1"
  local tags_name="$2"
  local tags_csv="$3"
  local -n tags_ref="$tags_name"
  local combined_dir="$SHARED_OUTPUT_DIR/cov_combined_${model_num}"

  if stage_state check "combined${model_num}"; then
    echo "Resume: verified combined covariances for model${model_num}."
  else
    # A single-tag combined path is a view of a completed covariance stage.
    if [[ ${#tags_ref[@]} -eq 1 ]]; then
      ensure_dir_link "$combined_dir" "$SHARED_OUTPUT_DIR/cov_${tags_ref[0]}"
    else
      if [[ -L "$combined_dir" ]]; then unlink "$combined_dir"; fi
      stage_state begin "combined${model_num}"
      run_cmd /usr/bin/python3 -m amaris_merge.combine_covariances \
        --output_dir "$SHARED_OUTPUT_DIR" \
        --tags "$tags_csv" \
        "$model_num"
    fi
    stage_state finish "combined${model_num}"
  fi

  ensure_dir_link "$SHARED_OUTPUT_DIR/cov_combined/cov_combined_${model_num}" "$combined_dir"
}
basis_percentage_done_for_alpha() {
  local alpha_output_dir="$1"
  local alpha="$2"
  # Keep small budgets (for example 0.001) distinct in artifact names.
  alpha="$(printf '%.12g' "$alpha")"
  basis_stage_state check "$alpha_output_dir" "$alpha"
}

basis_stage_state() {
  local action="$1" alpha_output_dir="$2" alpha="$3"
  run_cmd /usr/bin/python3 -m utils.artifacts "$action" \
    --root "$alpha_output_dir" --stage basis --config "$CONFIG_PATH" \
    --parameter "alpha=$alpha" \
    --output "basis_alpha_${alpha}/*" \
    --input "$SHARED_OUTPUT_DIR/.stages/combined1.json" \
    --input "$SHARED_OUTPUT_DIR/.stages/combined2.json" \
    --input "$SHARED_OUTPUT_DIR/.stages/delta.json"
}

run_shared_preprocessing_once() {
  echo "Running shared preprocessing once at: $SHARED_OUTPUT_DIR"
  mkdir -p "$SHARED_OUTPUT_DIR"

  if shared_tokenized_done; then
    echo "Resume: tokenized files already exist, skipping tokenize stage."
  else
    stage_state begin tokenized
    local -a tokenize_args=(--tokenizer "$MODEL1_DIR" --max_seq_len "$MAX_SEQ_LEN" --max_num_tokens "$MAX_NUM_TOKENS" --output_dir "$SHARED_OUTPUT_DIR")
    if [[ "$TOKENIZER_STREAMING" == true ]]; then
      tokenize_args+=(--streaming)
    fi
    run_cmd /usr/bin/python3 -m data.tokenize_dataset \
      --datasets "$DATASET1" \
      "${tokenize_args[@]}"

    run_cmd /usr/bin/python3 -m data.tokenize_dataset \
      --datasets "$DATASET2" \
      "${tokenize_args[@]}"
    stage_state finish tokenized
  fi

  if shared_split_done; then
    echo "Resume: split files already exist, skipping split stage."
  else
    stage_state begin split
    run_cmd /usr/bin/python3 -m data.split \
      --output_dir "$SHARED_OUTPUT_DIR" \
      --train_proportion "$TRAIN_PROPORTION"
    stage_state finish split
  fi

  if shared_packed_done; then
    echo "Resume: packed files already exist, skipping pack stage."
  else
    stage_state begin packed
    local -a pack_args=()
    if [[ "$TRUNCATE_PACKED_DOCUMENTS" == true ]]; then
      pack_args+=(--truncate_documents)
    fi
    run_cmd /usr/bin/python3 -m data.pack \
      --output_dir "$SHARED_OUTPUT_DIR" \
      --max_seq_length "$MAX_SEQ_LEN" \
      "${pack_args[@]}"
    stage_state finish packed
  fi

  if shared_covariances_done_model1; then
    echo "Resume: model1 covariance files already exist, skipping covariance stage."
  else
    local tag
    stage_state begin covariance1
    for tag in "${tags1[@]}"; do
      run_cmd /usr/bin/python3 -m amaris_merge.compute_covariances \
        --model "$MODEL1_DIR" \
        --output_dir "$SHARED_OUTPUT_DIR" \
        --num_chunks "$NUM_CHUNKS" \
        --cov_dtype "$COV_DTYPE" \
        --num_samples "$COV_NUM_SAMPLES" \
        --layers_to_ignore "$LAYERS_TO_IGNORE" \
        --tag "$tag"
    done
    stage_state finish covariance1
  fi

  if shared_covariances_done_model2; then
    echo "Resume: model2 covariance files already exist, skipping covariance stage."
  else
    local tag
    stage_state begin covariance2
    for tag in "${tags2[@]}"; do
      run_cmd /usr/bin/python3 -m amaris_merge.compute_covariances \
        --model "$MODEL2_DIR" \
        --output_dir "$SHARED_OUTPUT_DIR" \
        --num_chunks "$NUM_CHUNKS" \
        --cov_dtype "$COV_DTYPE" \
        --num_samples "$COV_NUM_SAMPLES" \
        --layers_to_ignore "$LAYERS_TO_IGNORE" \
        --tag "$tag"
    done
    stage_state finish covariance2
  fi

  if shared_deltas_done; then
    echo "Resume: delta files already exist, skipping delta stage."
  else
    stage_state begin delta
    run_cmd /usr/bin/python3 -m amaris_merge.get_delta \
      --model1_dir "$MODEL1_DIR" \
      --model2_dir "$MODEL2_DIR" \
      --output_dir "$SHARED_OUTPUT_DIR" \
      --layers_to_ignore "$LAYERS_TO_IGNORE"
    stage_state finish delta
  fi
}

MODEL1_DIR="$(require_key "model1_dir")"
MODEL2_DIR="$(require_key "model2_dir")"
DATASET1="$(require_any_key "datasets1,dataset1")"
DATASET2="$(require_any_key "dataset2,datasets2")"
TRAIN_PROPORTION="$(require_key "train_proportion")"
MAX_SEQ_LEN="$(require_key "max_seq_len")"
MAX_NUM_TOKENS="$(require_key "max_num_tokens")"
# Validate config scalars even when flags override them, so list configs fail early.
ALPHA="$(validate_parameter alpha "$(require_key "alpha")")"
BETA="$(validate_parameter beta "$(get_key_or_default "beta" "1.0")")"
if [[ -n "$ALPHA_OVERRIDE" ]]; then
  ALPHA="$(validate_parameter alpha "$ALPHA_OVERRIDE")"
fi
if [[ -n "$BETA_OVERRIDE" ]]; then
  BETA="$(validate_parameter beta "$BETA_OVERRIDE")"
fi
BASE_OUTPUT_DIR="$(require_key "output_dir")"
LAYERS_TO_IGNORE="$(get_key_or_default "layers_to_ignore" "")"
ENABLE_DELTA_VARIANT="$(get_key_or_default "enable_delta_variant" "false")"
case "$ENABLE_DELTA_VARIANT" in
  true|True|TRUE) ENABLE_DELTA_VARIANT=true ;;
  false|False|FALSE) ENABLE_DELTA_VARIANT=false ;;
  *)
    echo "enable_delta_variant must be true or false in $CONFIG_PATH" >&2
    exit 1
    ;;
esac
NUM_CHUNKS="$(get_key_or_default "num_chunks" "1")"
COV_DTYPE="$(get_key_or_default "cov_dtype" "float32")"
COV_NUM_SAMPLES="$(get_key_or_default "cov_num_samples" "10000")"
TOKENIZER_STREAMING="$(get_key_or_default "tokenizer_streaming" "false")"
TRUNCATE_PACKED_DOCUMENTS="$(get_key_or_default "truncate_packed_documents" "false")"
MAX_K="$(get_key_or_default "max_k" "256")"
BASIS_SOLVER="$(get_key_or_default "basis_solver" "lobpcg")"
LOBPCG_TOLERANCE="$(get_key_or_default "lobpcg_tolerance" "1e-3")"
LOBPCG_MAX_ITERATIONS="$(get_key_or_default "lobpcg_max_iterations" "50")"
LOBPCG_MAX_RETRIES="$(get_key_or_default "lobpcg_max_retries" "10")"
LOBPCG_MAX_RIDGE_SCALE="$(get_key_or_default "lobpcg_max_ridge_scale" "1e-2")"
BASIS_GPUS="$(get_key_or_default "basis_gpus" "all")"
case "$BASIS_SOLVER" in
  lobpcg|dense) ;;
  *)
    echo "basis_solver must be lobpcg or dense in $CONFIG_PATH" >&2
    exit 1
    ;;
esac
case "$TOKENIZER_STREAMING" in
  true|True|TRUE) TOKENIZER_STREAMING=true ;;
  false|False|FALSE) TOKENIZER_STREAMING=false ;;
  *)
    echo "tokenizer_streaming must be true or false in $CONFIG_PATH" >&2
    exit 1
    ;;
esac
case "$TRUNCATE_PACKED_DOCUMENTS" in
  true|True|TRUE) TRUNCATE_PACKED_DOCUMENTS=true ;;
  false|False|FALSE) TRUNCATE_PACKED_DOCUMENTS=false ;;
  *)
    echo "truncate_packed_documents must be true or false in $CONFIG_PATH" >&2
    exit 1
    ;;
esac
case "$COV_DTYPE" in
  float32|float64) ;;
  *)
    echo "cov_dtype must be float32 or float64 in $CONFIG_PATH" >&2
    exit 1
    ;;
esac

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR%/}"
if [[ "$BASE_OUTPUT_DIR" != /* ]]; then
  BASE_OUTPUT_DIR="$ROOT_DIR/$BASE_OUTPUT_DIR"
fi
SHARED_OUTPUT_DIR="${BASE_OUTPUT_DIR}-shared"

parse_tags "$DATASET1" tags1
parse_tags "$DATASET2" tags2
build_all_tags tags1 tags2 all_tags
# Covariances depend on the model, so cross-model tag reuse is unsupported.
for tag1 in "${tags1[@]}"; do
  for tag2 in "${tags2[@]}"; do
    if [[ "$tag1" == "$tag2" ]]; then
      echo "Dataset tags must be distinct between models: $tag1" >&2
      exit 1
    fi
  done
done

if [[ ${#tags1[@]} -lt 1 || ${#tags2[@]} -lt 1 ]]; then
  echo "Expected at least one dataset tag for datasets1/dataset1 and dataset2/datasets2." >&2
  exit 1
fi


echo "Starting merge using: $CONFIG_PATH"
echo "alpha: $ALPHA"
echo "beta: $BETA"
echo "shared_output_dir: $SHARED_OUTPUT_DIR"
echo "layers_to_ignore: ${LAYERS_TO_IGNORE:-<none>}"
echo "enable_delta_variant: $ENABLE_DELTA_VARIANT"
echo "basis_gpus: $BASIS_GPUS"
echo "num_chunks: $NUM_CHUNKS"
echo "cov_dtype: $COV_DTYPE"
echo "cov_num_samples: $COV_NUM_SAMPLES"
echo "tokenizer_streaming: $TOKENIZER_STREAMING"
echo "truncate_packed_documents: $TRUNCATE_PACKED_DOCUMENTS"
echo "basis_solver: $BASIS_SOLVER"
echo "max_k: $MAX_K"

run_shared_preprocessing_once


TAGS1_CSV="$(tags_to_csv tags1)"
TAGS2_CSV="$(tags_to_csv tags2)"

mkdir -p "$SHARED_OUTPUT_DIR/cov_combined"

prepare_combined_covariances 1 tags1 "$TAGS1_CSV"
prepare_combined_covariances 2 tags2 "$TAGS2_CSV"


alpha="$(printf '%.12g' "$ALPHA")"
VARIANT_SUFFIX=""
if [[ "$ENABLE_DELTA_VARIANT" == true ]]; then
  VARIANT_SUFFIX="-delta-variant"
fi
ALPHA_OUTPUT_DIR="${BASE_OUTPUT_DIR}-alpha-${alpha}${VARIANT_SUFFIX}"
echo "\n=== alpha=$alpha ==="
echo "basis_output_dir=$ALPHA_OUTPUT_DIR"

mkdir -p "$ALPHA_OUTPUT_DIR"
ensure_dir_link "$ALPHA_OUTPUT_DIR/packed" "$SHARED_OUTPUT_DIR/packed"
mkdir -p "$ALPHA_OUTPUT_DIR/cov_combined"
ensure_dir_link "$ALPHA_OUTPUT_DIR/cov_combined/cov_combined_1" "$SHARED_OUTPUT_DIR/cov_combined_1"
ensure_dir_link "$ALPHA_OUTPUT_DIR/cov_combined/cov_combined_2" "$SHARED_OUTPUT_DIR/cov_combined_2"
ensure_dir_link "$ALPHA_OUTPUT_DIR/delta" "$SHARED_OUTPUT_DIR/delta"

if basis_percentage_done_for_alpha "$ALPHA_OUTPUT_DIR" "$alpha"; then
  echo "Resume: basis already exists for alpha=$alpha, skipping basis stage."
else
  # A prior interrupted basis run may have written a prefix of basis files.
  # Remove only generated basis artifacts before recomputing this alpha so
  # they cannot be mistaken for a completed partial result.
  BASIS_DIR="$ALPHA_OUTPUT_DIR/basis_alpha_${alpha}"
  rm -f "$BASIS_DIR"/*_input_basis.pt "$BASIS_DIR/basis_metadata.json"
  basis_args=(
    /usr/bin/python3 -m amaris_merge.compute_basis_by_percentage
    --alpha "$alpha"
    --output_dir "$ALPHA_OUTPUT_DIR"
    --delta_dir "$ALPHA_OUTPUT_DIR/delta"
    --candidate_cache_dir "$SHARED_OUTPUT_DIR/generalized_eigenpairs${VARIANT_SUFFIX}"
    --max_k "$MAX_K"
    --basis_solver "$BASIS_SOLVER"
    --lobpcg_tolerance "$LOBPCG_TOLERANCE"
    --lobpcg_max_iterations "$LOBPCG_MAX_ITERATIONS"
    --lobpcg_max_retries "$LOBPCG_MAX_RETRIES"
    --lobpcg_max_ridge_scale "$LOBPCG_MAX_RIDGE_SCALE"
  )
  if [[ "$ENABLE_DELTA_VARIANT" == true ]]; then
    basis_args+=(--enable_delta_variant)
  fi
  if [[ "$BASIS_GPUS" != "all" ]]; then
    basis_args+=(--gpus "$BASIS_GPUS")
  fi
  run_cmd "${basis_args[@]}"
  basis_stage_state finish "$ALPHA_OUTPUT_DIR" "$alpha"
fi

merge_args=(
  /usr/bin/python3 -m amaris_merge.merge_one_step
  --model1_dir "$MODEL1_DIR"
  --model2_dir "$MODEL2_DIR"
  --output_base_dir "$ALPHA_OUTPUT_DIR"
  --basis_dir "$ALPHA_OUTPUT_DIR/basis_alpha_${alpha}"
  --betas "$BETA"
)
if [[ -n "$LAYERS_TO_IGNORE" ]]; then
  merge_args+=(--layers_to_ignore "$LAYERS_TO_IGNORE")
fi
run_cmd "${merge_args[@]}"
echo "Merge complete."
