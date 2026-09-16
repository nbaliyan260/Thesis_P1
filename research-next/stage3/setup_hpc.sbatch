#!/bin/bash
#SBATCH --job-name=recut3-fetch
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/l/users/nazish.baliyan/recut-20260916/logs/stage3-setup-%j.out
#SBATCH --error=/l/users/nazish.baliyan/recut-20260916/logs/stage3-setup-%j.err
set -euo pipefail
cd /l/users/nazish.baliyan/recut-20260916
export HF_HOME=/l/users/nazish.baliyan/recut-20260916/hf-cache
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
date -Is
hostname
.venv313/bin/python stage3/fetch_models.py --manifest stage3/model_manifest.json --output stage3/downloaded_manifest.json
date -Is
