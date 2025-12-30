import torch
import numpy as np
from einops import repeat,reduce, rearrange
from tqdm import tqdm
from PIL import Image

from models.scheduler import FlowMatchScheduler
from models.manager import ModelManager, load_state_dict
from models.prompter import WanPrompter
from models.dit import sinusoidal_embedding_1d


class EeveePipeline(torch.nn.Module):

    def __init__(
        self,
        device = "cuda",
        torch_dtype = torch.bfloat16,
        height_division_factor = 16,
        width_division_factor = 16,
        time_division_factor = 4,
        time_division_remainder = 1,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False


        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.in_iteration_models = ("dit", "vace")
        self.model_fn = model_fn_wan_video

    @staticmethod
    def from_pretrained(
        torch_dtype = torch.bfloat16,
        device = "cuda",
        vae_model_path = None,
        text_encoder_model_path = None,
        dit_model_path = None,
        tokenizer_path = None,
    ):
        
        # Initialize pipeline
        pipe = EeveePipeline(device=device, torch_dtype=torch_dtype)
        
        # Download and load models
        model_manager = ModelManager()
        model_manager.load_model(vae_model_path, "vae", device=device, torch_dtype=torch_dtype)
        model_manager.load_model(text_encoder_model_path, "text_encoder", device=device, torch_dtype=torch_dtype)
        model_manager.load_model(dit_model_path, "dit", device=device, torch_dtype=torch_dtype)

        # Load models
        pipe.text_encoder =  model_manager.fetch_model("video_text_encoder")
        pipe.vae = model_manager.fetch_model("video_vae")
        pipe.dit = model_manager.fetch_model("video_dit")
        pipe.vace = model_manager.fetch_model("video_vace")

        # Size division factor
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        pipe.prompter = WanPrompter()
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_path)

        return pipe

    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames
    
    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image
    
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss
    
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def load_lora(
        self,
        module,
        lora_config = None,
        alpha = 1
    ):
        lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video
    
    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image
    


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt,
        vace_video = None,
        vace_video_mask= None,
        vace_reference_image = None,
        vace_scale = 1.0,
        width = 816,
        height = 1088,
        num_frames = 49,
        seed = 1,
        # scheduler
        num_inference_steps = 50,
        denoising_strength = 1.0,
        sigma_shift = 5.0,
        # Classifier-free guidance
        cfg_scale = 5.0,
        cfg_merge = False,
        # VAE tiling
        tiled = True,
        tile_size = (30, 52),
        tile_stride = (15, 26),
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"prompt": negative_prompt}

        inputs = {
            "input_video": None,
            "vace_video": vace_video, 
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "height": height, 
            "width": width, 
            "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "rand_device": self.device,
            # "use_gradient_checkpointing": self.use_gradient_checkpointing,
            # "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }


        inputs = self.shape_checker(inputs)
        inputs = self.noise_initializer(inputs)
        inputs_posi = self.prompt_embedder(inputs_posi)
        inputs_nega = self.prompt_embedder(inputs_nega)
        inputs = self.input_video_embedder(inputs)
        inputs = self.unit_vace(inputs)
        
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs, **inputs_posi, timestep=timestep)
            noise_pred_nega = self.model_fn(**models, **inputs, **inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            inputs["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs["latents"])
        
        f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
        inputs["latents"] = inputs["latents"][:, :, f:]
        
        video = self.vae.decode(inputs["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)

        return video



    def shape_checker(self, inputs):
        height, width, num_frames = self.check_resize_height_width(inputs["height"], inputs["width"], inputs["num_frames"])
        inputs["height"] = height
        inputs["width"] = width
        inputs["num_frames"] = num_frames
        return inputs

    def noise_initializer(self, inputs):
        length = (inputs["num_frames"] - 1) // 4 + 1 + 1
        shape = (1, self.vae.model.z_dim, length, inputs["height"] // self.vae.upsampling_factor, inputs["width"] // self.vae.upsampling_factor)
        noise = self.generate_noise(shape, seed=None, rand_device=inputs["rand_device"])
        noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        inputs["noise"] = noise
        return inputs

    def prompt_embedder(self, inputs):
        self.load_models_to_device(("text_encoder",))
        prompt_emb = self.prompter.encode_prompt(inputs["prompt"], positive=None, device=self.device)
        inputs["context"] = prompt_emb
        return inputs

    def input_video_embedder(self, inputs):
        if inputs["input_video"] is None:
            inputs["latents"] = inputs["noise"]
            return inputs
        self.load_models_to_device(["vae"])
        input_video = self.preprocess_video(inputs["input_video"])
        input_latents = self.vae.encode(input_video, device=self.device, tiled=inputs["tiled"], tile_size=None, tile_stride=None).to(dtype=self.torch_dtype, device=self.pipe.device)
        vace_reference_image = [inputs["vace_reference_image"]]
        vace_reference_image = self.preprocess_video(vace_reference_image)
        vace_reference_latents = self.vae.encode(vace_reference_image, device=self.device).to(dtype=self.torch_dtype, device=self.device)
        input_latents = torch.concat([vace_reference_latents, input_latents], dim=2) # torch.Size([1, 16, 6, 60, 104])
        inputs["latents"] = inputs["noise"]
        inputs["input_latents"] = input_latents
        return inputs


    def unit_vace(self, inputs):
        self.load_models_to_device(["vae"])
        vace_video = self.preprocess_video(inputs["vace_video"])
        vace_video_mask = self.preprocess_video(inputs["vace_video_mask"], min_value=0, max_value=1)
        
        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
        inactive = self.vae.encode(inactive, device=self.device, tiled=inputs["tiled"], tile_size=inputs["tile_size"], tile_stride=inputs["tile_stride"]).to(dtype=self.torch_dtype, device=self.device)
        reactive = self.vae.encode(reactive, device=self.device, tiled=inputs["tiled"], tile_size=inputs["tile_size"], tile_stride=inputs["tile_stride"]).to(dtype=self.torch_dtype, device=self.device)
        vace_video_latents = torch.concat((inactive, reactive), dim=1)
        vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
        vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

        vace_reference_image = self.preprocess_video([inputs["vace_reference_image"]])
        bs, c, f, h, w = vace_reference_image.shape
        new_vace_ref_images = []
        for j in range(f):
            new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
        vace_reference_image = new_vace_ref_images
        vace_reference_latents = self.vae.encode(vace_reference_image, device=self.device, tiled=inputs["tiled"],  tile_size=inputs["tile_size"], tile_stride=inputs["tile_stride"]).to(dtype=self.torch_dtype, device=self.device)
        vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
        vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

        vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
        vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

        inputs["vace_context"] = vace_context
        return inputs


def model_fn_wan_video(
    dit,
    vace = None,
    latents = None,
    timestep = None,
    context= None,
    vace_context = None,
    vace_scale = 1.0,
    use_gradient_checkpointing = True,
    use_gradient_checkpointing_offload = True,
    **kwargs,
):

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents

    x, (f, h, w) = dit.patchify(x, None)
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
        
    vace_hints = vace(
        x,
        vace_context,
        context,
        t_mod,
        freqs,
        use_gradient_checkpointing = use_gradient_checkpointing,
        use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
    )
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    

    for block_id, block in enumerate(dit.blocks):
        with torch.autograd.graph.save_on_cpu():
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        if vace_context is not None and block_id in vace.vace_layers_mapping:
            current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
            x = x + current_vace_hint * vace_scale
            
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")




        



if __name__ == "__main__":
    pipe = EeveePipeline.from_pretrained(
        torch_dtype = torch.bfloat16,
        device = "cpu",
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
        tokenizer_path = "/mnt/xmap_nas_ml/zengjianhao/DiffSynth/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    )