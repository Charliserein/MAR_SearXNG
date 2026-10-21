#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
command="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "${command}" in
  prepare) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/prepare_cached_evidence.py" "$@" ;;
  warm) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/warm_search_cache.py" "$@" ;;
  evaluate) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/evaluate_batch.py" "$@" ;;
  report) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/report_evaluation.py" "$@" ;;
  experiment) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/run_experiment.py" "$@" ;;
  search) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/search_cache.py" "$@" ;;
  status) exec "${project_dir}/.venv/bin/python" "${project_dir}/scripts/status.py" "$@" ;;
  searxng) exec bash "${project_dir}/scripts/searxng.sh" "$@" ;;
  model) exec python3 "${project_dir}/deploy/qwen/start.py" "$@" ;;
  *) printf '%s\n' '用法：./run.sh {prepare|warm|evaluate|report|experiment|search|status|searxng|model} [参数]' ;;
esac
