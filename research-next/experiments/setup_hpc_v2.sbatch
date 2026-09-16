#!/bin/bash
#SBATCH --job-name=recut-setup-v2
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/setup-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/setup-%j.err
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
export HF_HOME=/l/users/nazish.baliyan/recut-20260916/hf-cache
export PIP_CACHE_DIR=/l/users/nazish.baliyan/recut-20260916/pip-cache
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1
date -Is
hostname
nvidia-smi
/home/nazish.baliyan/miniconda3/bin/python -m venv .venv313
.venv313/bin/python -m pip install --upgrade pip
.venv313/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv313/bin/python -m pip install -r experiments/requirements.txt
.venv313/bin/python experiments/prepare_models.py
.venv313/bin/python -m pip freeze > experiments/environment_freeze.txt
date -Is
