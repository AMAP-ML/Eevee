export PYTHONPATH="${PYTHONPATH}:/mnt/xmap_nas_ml/zengjianhao/Eevee"


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --num_processes=4 train/train.py \
  --dresses_dataset_base_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses \
  --dresses_dataset_metadata_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses_train.csv \
  --lower_dataset_base_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/lower_body \
  --lower_dataset_metadata_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/lower_train.csv \
  --upper_dataset_base_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/upper_body \
  --upper_dataset_metadata_path /mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/upper_train.csv \
  --height 816 \
  --width 1088 \
  --num_frames 49 \
  --vae_model_path "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --text_encoder_model_path "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --dit_model_path \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors" \
    "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors" \
  --tokenizer_path "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --output_path "./checkpoints/V1" \
  --learning_rate 1e-5 \
  --save_steps 1 \
  --num_epochs 100 > log.out 2>&1 & 




