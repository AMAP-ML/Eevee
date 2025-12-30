import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--dresses_dataset_base_path", type=str, default="", required=True, help="Base path of the dresses dataset.")
    parser.add_argument("--dresses_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dresses dataset.")
    parser.add_argument("--lower_dataset_base_path", type=str, default="", required=True, help="Base path of the lower body dataset.")
    parser.add_argument("--lower_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the lower body dataset.")
    parser.add_argument("--upper_dataset_base_path", type=str, default="", required=True, help="Base path of the upper body dataset.")
    parser.add_argument("--upper_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the upper body dataset.")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames per video. Frames are sampled from the video prefix.")

    parser.add_argument("--vae_model_path", type=str, default=None, help="Path of VAE model.")
    parser.add_argument("--text_encoder_model_path", type=str, default=None, help="Path of Text Encoder model.")
    parser.add_argument("--dit_model_path", type=str, nargs='+', default=None, help="Paths of DIT model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path of Tokenizer model.")
    
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")

    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")

    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")

    return parser