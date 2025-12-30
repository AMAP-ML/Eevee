import os

from utils.parser import parse_args
from utils.logger import ModelLogger
from utils.launch import launch_training_task
from dataset.eevee_dataset import EeveeDataset
from models.training import TrainingModule


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = parse_args()
args = parser.parse_args()


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