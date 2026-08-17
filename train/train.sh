#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
OUTPUT_PATH="${OUTPUT_PATH:-./checkpoints/Eevee_v0}"

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer." >&2
  exit 1
fi
if (( NUM_PROCESSES > ${#VISIBLE_GPUS[@]} )); then
  echo "NUM_PROCESSES (${NUM_PROCESSES}) exceeds the number of visible GPUs (${#VISIBLE_GPUS[@]})." >&2
  exit 1
fi

accelerate launch --num_processes="${NUM_PROCESSES}" train/train.py \
  --dresses_dataset_base_path ./data/Eevee/dresses \
  --dresses_dataset_metadata_path ./data/Eevee/dresses_train.csv \
  --lower_dataset_base_path ./data/Eevee/lower_body \
  --lower_dataset_metadata_path ./data/Eevee/lower_train.csv \
  --upper_dataset_base_path ./data/Eevee/upper_body \
  --upper_dataset_metadata_path ./data/Eevee/upper_train.csv \
  --height 816 \
  --width 1088 \
  --num_frames 49 \
  --vae_model_path "./checkpoints/Wan2.1-VACE-14B/Wan2.1_VAE.pth" \
  --text_encoder_model_path "./checkpoints/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --dit_model_path \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors" \
    "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors" \
  --tokenizer_path "./checkpoints/Wan2.1-VACE-14B/google/umt5-xxl" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --output_path "${OUTPUT_PATH}" \
  --remove_prefix_in_ckpt "pipe.vace." \
  --learning_rate 1e-5 \
  --save_steps "${SAVE_STEPS}" \
  --num_epochs "${NUM_EPOCHS}" \
  "$@"
