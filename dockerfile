FROM huggingface/transformers-all-latest-gpu:flash-attn

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# install in system python
RUN uv pip install --system ipykernel accelerate

# Keep base image environment untouched
RUN uv venv /opt/vllm-env
ENV PATH="/opt/vllm-env/bin:${PATH}"

# Install directly into the vllm venv and pin compatible data stack versions.
RUN uv pip install --python /opt/vllm-env/bin/python vllm ipykernel "datasets>=2.14.4,<3" "pyarrow<21" "fsspec==2023.6.0"