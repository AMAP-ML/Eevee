import torch, imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.pipeline import EeveePipeline


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()

class VideoData:
    def __init__(self, video_file=None, image_folder=None, height=None, width=None, **kwargs):
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        pass

    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


pipe = EeveePipeline.from_pretrained(
    torch_dtype = torch.bfloat16,
    device = "cuda",
    vae_model_path = "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    text_encoder_model_path = "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    dit_model_path = [
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors",
        "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    tokenizer_path = "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
)
# pipe.load_lora(pipe.vace, "", alpha=1)


video = VideoData("/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses/00137/video_0_agnostic.mp4", height=1088, width=816)
video = [video[i] for i in range(49)]
video_mask = VideoData("/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses/00137/video_0_agnostic_mask.mp4", height=1088, width=816)
video_mask = [video_mask[i] for i in range(49)]

vace_reference_image=Image.open("/mnt/xmap_nas_ml/zengjianhao/Data/Eevee4/dresses/00137/in.png").resize((816, 1088))


video = pipe(
    prompt="Model is wearing A pale yellow sleeveless dress features a simple yet elegant design. A V-neckline gracefully frames the upper part, leading down to a fitted bodice that smoothly transitions into a gently flared skirt. A subtle seam runs horizontally across the waist, accentuating the silhouette. A small, rectangular tag is visible at the neckline, bearing text too faint to discern clearly. A smooth texture suggests a soft fabric, likely designed for comfortable wear. A clean and minimalist aesthetic defines the overall style, with no additional patterns or graphics adorning the surface.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video,
    vace_reference_image=vace_reference_image,
    vace_video_mask = video_mask,
    width=816,
    height=1088,
    num_frames=49,
    seed=1,
    tiled=True
)
save_video(video, "1.mp4", fps=25, quality=5)