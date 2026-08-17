import os
from pathlib import Path

from utils.parser import parse_args
from utils.logger import ModelLogger
from utils.launch import launch_training_task
from dataset.eevee_dataset import EeveeDataset
from models.training import TrainingModule


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = parse_args()
args = parser.parse_args()
if args.remove_prefix_in_ckpt is None:
    args.remove_prefix_in_ckpt = f"pipe.{args.lora_base_model}."


def validate_training_inputs(args):
    if args.height is None or args.width is None:
        raise ValueError("--height and --width are required for Eevee training.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.num_frames <= 0 or args.num_frames % 4 != 1:
        raise ValueError("--num_frames must be positive and satisfy num_frames % 4 == 1.")
    if args.lora_rank <= 0:
        raise ValueError("--lora_rank must be positive.")
    if args.num_epochs <= 0:
        raise ValueError("--num_epochs must be positive.")
    if args.save_steps is not None and args.save_steps <= 0:
        raise ValueError("--save_steps must be positive when provided.")
    if args.dataset_num_workers < 0:
        raise ValueError("--dataset_num_workers cannot be negative.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if not 0 <= args.min_timestep_boundary < args.max_timestep_boundary <= 1:
        raise ValueError(
            "Timestep boundaries must satisfy "
            "0 <= min_timestep_boundary < max_timestep_boundary <= 1."
        )

    required_directories = [
        Path(args.dresses_dataset_base_path),
        Path(args.lower_dataset_base_path),
        Path(args.upper_dataset_base_path),
        Path(args.tokenizer_path),
    ]
    required_files = [
        Path(args.dresses_dataset_metadata_path),
        Path(args.lower_dataset_metadata_path),
        Path(args.upper_dataset_metadata_path),
        Path(args.vae_model_path),
        Path(args.text_encoder_model_path),
        *(Path(path) for path in args.dit_model_path),
        Path(args.tokenizer_path) / "tokenizer_config.json",
        Path(args.tokenizer_path) / "spiece.model",
    ]
    missing_paths = [
        path for path in required_directories if not path.is_dir()
    ] + [path for path in required_files if not path.is_file()]
    if missing_paths:
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Required training paths are missing:\n{missing_list}")


validate_training_inputs(args)


dataset = EeveeDataset(
    dresses_dataset_base_path = args.dresses_dataset_base_path,
    dresses_dataset_metadata_path = args.dresses_dataset_metadata_path,
    lower_dataset_base_path = args.lower_dataset_base_path,
    lower_dataset_metadata_path = args.lower_dataset_metadata_path,
    upper_dataset_base_path = args.upper_dataset_base_path,
    upper_dataset_metadata_path = args.upper_dataset_metadata_path,
    target_height = args.height,
    target_width = args.width,
    num_frames = args.num_frames
)
if len(dataset) == 0:
    raise ValueError("The training metadata files do not contain any samples.")

model = TrainingModule(
    vae_model_path = args.vae_model_path,                                               # 
    text_encoder_model_path = args.text_encoder_model_path,                             # 
    dit_model_path = args.dit_model_path,                                               # 
    tokenizer_path = args.tokenizer_path,                                               # 
    lora_base_model = args.lora_base_model,                                             # "vace"
    lora_target_modules = args.lora_target_modules,                                     # "q,k,v,o,ffn.0,ffn.2"
    lora_rank = args.lora_rank,                                                         # 32
    max_timestep_boundary = args.max_timestep_boundary,                                 # 1.0
    min_timestep_boundary = args.min_timestep_boundary,                                 # 0.0
)

model_logger = ModelLogger(
    args.output_path,
    remove_prefix_in_ckpt = args.remove_prefix_in_ckpt
)

launch_training_task(dataset, model, model_logger, args=args)
