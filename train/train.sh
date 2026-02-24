export PYTHONPATH="${PYTHONPATH}:./"


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup accelerate launch --num_processes=8 train/train.py \
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
  --tokenizer_path "./checkpoints/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --output_path "./checkpoints/Eevee_v0" \
  --learning_rate 1e-5 \
  --save_steps 1 \
  --num_epochs 100 > log.out 2>&1 & 