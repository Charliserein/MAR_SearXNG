#!/usr/bin/env bash
# 在独立虚拟环境启动本地模型；不会停止或修改其他推理服务。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MAR_QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.8-27B-FP8}"
VLLM_BIN="$PROJECT_ROOT/.venv-vllm/bin/vllm"
if [[ ! -f "$MODEL_DIR/download-manifest.json" ]]; then
  echo "模型尚未完成校验，请先运行 python3 scripts/download_model.py" >&2
  exit 1
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "请先在 .venv-vllm 安装 deploy/qwen/requirements.txt" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${MAR_QWEN_GPU:-0}"
CUDA_PACKAGE_DIR="$("$PROJECT_ROOT/.venv-vllm/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
export CUDA_HOME="$CUDA_PACKAGE_DIR"
export PATH="$PROJECT_ROOT/.venv-vllm/bin:$CUDA_PACKAGE_DIR/bin:$PATH"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$PROJECT_ROOT/cache/vllm"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/runtime"
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "${MAR_QWEN_PORT:-19888}" \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${MAR_QWEN_GPU_MEMORY:-0.78}" \
  --max-model-len "${MAR_QWEN_CONTEXT:-16384}" \
  --max-num-seqs "${MAR_QWEN_MAX_SEQS:-8}" \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1003520}' \
  "$@"
