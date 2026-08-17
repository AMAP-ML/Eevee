import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.pipeline import EeveePipeline


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Eevee inference on one dataset case.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints/Wan2.1-VACE-14B"),
        help="Directory containing the Wan2.1-VACE-14B checkpoint.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=Path("./checkpoints/Eevee/step-3000.safetensors"),
        help="Path to the Eevee LoRA checkpoint.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("./data/Eevee/dresses/00030"),
        help="Directory of one Eevee dataset case.",
    )
    parser.add_argument(
        "--video-id",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use the full-shot (0) or close-up (1) input video.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="Reference garment image. Defaults to <case-dir>/garment_detail.png.",
    )
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=816)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output video path. Defaults to outputs/eevee_<case>_video_<id>.mp4.",
    )
    return parser.parse_args()


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def load_video(video_path, num_frames, height, width):
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        available_frames = reader.count_frames()
        if available_frames < num_frames:
            raise ValueError(
                f"{video_path} contains {available_frames} frames, but {num_frames} are required."
            )
        for frame_id in range(num_frames):
            frame = Image.fromarray(np.asarray(reader.get_data(frame_id))).convert("RGB")
            frames.append(crop_and_resize(frame, height, width))
    finally:
        reader.close()
    return frames


def save_video(frames, save_path, fps, quality=5):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    try:
        for frame in tqdm(frames, desc="Saving video"):
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def require_files(paths):
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Required test files are missing:\n{missing_list}")


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.num_frames <= 0 or args.num_frames % 4 != 1:
        raise ValueError("--num-frames must be positive and satisfy num_frames % 4 == 1.")

    checkpoint_dir = args.checkpoint_dir
    case_dir = args.case_dir
    output_path = args.output_path or Path(
        f"./outputs/eevee_{case_dir.name}_video_{args.video_id}.mp4"
    )
    reference_image_path = args.reference_image or case_dir / "garment_detail.png"
    caption_path = case_dir / "garment_caption.txt"
    input_video_path = case_dir / f"video_{args.video_id}_agnostic.mp4"
    mask_video_path = case_dir / f"video_{args.video_id}_mask.mp4"
    dit_model_paths = [
        checkpoint_dir / f"diffusion_pytorch_model-{shard_id:05d}-of-00007.safetensors"
        for shard_id in range(1, 8)
    ]

    require_files(
        [
            checkpoint_dir / "Wan2.1_VAE.pth",
            checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            checkpoint_dir / "google/umt5-xxl/tokenizer_config.json",
            checkpoint_dir / "google/umt5-xxl/spiece.model",
            *dit_model_paths,
            args.lora_path,
            caption_path,
            reference_image_path,
            input_video_path,
            mask_video_path,
        ]
    )

    vace_video = load_video(
        input_video_path, args.num_frames, args.height, args.width
    )
    vace_video_mask = load_video(
        mask_video_path, args.num_frames, args.height, args.width
    )
    vace_reference_image = crop_and_resize(
        Image.open(reference_image_path).convert("RGB"), args.height, args.width
    )
    prompt = "Model is wearing " + caption_path.read_text(encoding="utf-8").strip()

    pipe = EeveePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        vae_model_path=str(checkpoint_dir / "Wan2.1_VAE.pth"),
        text_encoder_model_path=str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        dit_model_path=[str(path) for path in dit_model_paths],
        tokenizer_path=str(checkpoint_dir / "google/umt5-xxl"),
    )
    pipe.load_lora(pipe.vace, str(args.lora_path), alpha=1)

    output_frames = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        vace_video=vace_video,
        vace_reference_image=vace_reference_image,
        vace_video_mask=vace_video_mask,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )
    save_video(output_frames, output_path, fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
