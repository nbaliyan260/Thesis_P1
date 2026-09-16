#!/bin/bash
#SBATCH --job-name=recut3-study
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/stage3-%A_%a.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/stage3-%A_%a.err
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
export HF_HOME=/l/users/nazish.baliyan/recut-20260916/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "${SLURM_ARRAY_TASK_ID}" in
  0) recut_model='HuggingFaceTB/SmolLM2-135M'; recut_slug='smol135m' ;;
  1) recut_model='Qwen/Qwen2.5-0.5B'; recut_slug='qwen05b' ;;
  2) recut_model='Qwen/Qwen2.5-7B'; recut_slug='qwen7b' ;;
  *) exit 2 ;;
esac
date -Is
hostname
nvidia-smi
.venv313/bin/python stage3/freeze_study.py verify
case "${RECUT_PHASE:-debug}" in
  debug)
    .venv313/bin/python -m pytest -q stage3/test_stage3_controller.py stage2/test_transactional.py experiments/test_recut_engine.py experiments/tests_independent.py
    recut_config='stage3/config_debug.json'; recut_output="runs/stage3-debug-v1/${recut_slug}"
    ;;
  main)
    recut_config='stage3/config_study.json'; recut_output="runs/stage3-main-v1/${recut_slug}"
    ;;
  *) exit 2 ;;
esac
.venv313/bin/python -u stage3/run_study.py --config "${recut_config}" --manifest stage3/model_manifest.json --model "${recut_model}" --output "${recut_output}"
.venv313/bin/python stage3/analyze_study.py "${recut_output}"
.venv313/bin/python stage3/audit_study.py "${recut_output}" --config "${recut_config}" --manifest stage3/model_manifest.json
.venv313/bin/python stage3/freeze_study.py verify
date -Is
