import torch
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model

from models.pipeline import EeveePipeline


class TrainingModule(torch.nn.Module):
    def __init__(
        self,
        vae_model_path = None,                          #
        text_encoder_model_path = None,                 #
        dit_model_path = None,                          #
        tokenizer_path = None,                          #

        lora_base_model = None,                         # "vace"
        lora_target_modules = "q,k,v,o,ffn.0,ffn.2",    # "q,k,v,o,ffn.0,ffn.2"
        lora_rank = 32,                                 # 32

        use_gradient_checkpointing = True,              # True
        use_gradient_checkpointing_offload = True,      # True
        extra_inputs = None,                            # "vace_video,vace_reference_image"
        max_timestep_boundary = 1.0,                    # 1.0
        min_timestep_boundary = 0.0,                    # 0.0
    ):
        super().__init__()
        
        self.pipe = EeveePipeline.from_pretrained(
            torch_dtype = torch.bfloat16,
            device = "cpu",
            vae_model_path = vae_model_path,
            text_encoder_model_path = text_encoder_model_path,
            dit_model_path = dit_model_path,
            tokenizer_path = tokenizer_path
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            lora_base_model,
            lora_target_modules,
            lora_rank
        )
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    
    def forward(self, data):
        inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models} # todo
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


    def switch_pipe_to_training_mode(
        self,
        pipe,
        lora_base_model,        # "vace"
        lora_target_modules,    # "q,k,v,o,ffn.0,ffn.2"
        lora_rank,              # 32
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([])
        
        # Add LoRA to the base models
        model = self.add_lora_to_model(
            getattr(pipe, lora_base_model),
            target_modules = lora_target_modules.split(","),
            lora_rank = lora_rank,
            upcast_dtype = pipe.torch_dtype,
        )
        setattr(pipe, lora_base_model, model)


    def add_lora_to_model(
            self,
            model,
            target_modules,
            lora_rank,
            upcast_dtype = None
        ):
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(upcast_dtype)
        return model
    

    def forward_preprocess(self, data):
        inputs = {
            "prompt": data["prompt"],
            "vace_reference_image": data["vace_reference_image"][0],
            "vace_video": data["vace_video"],
            "vace_video_mask": data["vace_video_mask"],
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        inputs = self.shape_checker(inputs)
        inputs = self.noise_initializer(inputs)
        inputs = self.prompt_embedder(inputs)
        inputs = self.input_video_embedder(inputs)
        inputs = self.vace(inputs)

        return inputs


    def shape_checker(self, inputs):
        height, width, num_frames = self.pipe.check_resize_height_width(inputs["height"], inputs["width"], inputs["num_frames"])
        inputs["height"] = height
        inputs["width"] = width
        inputs["num_frames"] = num_frames
        return inputs


    def noise_initializer(self, inputs):
        length = (inputs["num_frames"] - 1) // 4 + 1 + 1
        shape = (1, self.pipe.vae.model.z_dim, length, inputs["height"] // self.pipe.vae.upsampling_factor, inputs["width"] // self.pipe.vae.upsampling_factor)
        noise = self.pipe.generate_noise(shape, seed=None, rand_device=inputs["rand_device"])
        noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        inputs["noise"] = noise
        return inputs

    
    def prompt_embedder(self, inputs):
        self.pipe.load_models_to_device(("text_encoder",))
        prompt_emb = self.pipe.prompter.encode_prompt(inputs["prompt"], positive=None, device=self.pipe.device)
        inputs["context"] = prompt_emb
        return inputs
    

    def input_video_embedder(self, inputs):
        self.pipe.load_models_to_device(["vae"])
        input_video = self.pipe.preprocess_video(inputs["input_video"])
        input_latents = self.pipe.vae.encode(input_video, device=self.pipe.device, tiled=inputs["tiled"], tile_size=None, tile_stride=None).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        vace_reference_image = [inputs["vace_reference_image"]]
        vace_reference_image = self.pipe.preprocess_video(vace_reference_image)
        vace_reference_latents = self.pipe.vae.encode(vace_reference_image, device=self.pipe.device).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        input_latents = torch.concat([vace_reference_latents, input_latents], dim=2) # torch.Size([1, 16, 6, 60, 104])
        inputs["latents"] = inputs["noise"]
        inputs["input_latents"] = input_latents
        return inputs
    

    def vace(self, inputs):
        self.pipe.load_models_to_device(["vae"])
        vace_video = self.pipe.preprocess_video(inputs["vace_video"])
        vace_video_mask = self.pipe.preprocess_video(inputs["vace_video_mask"], min_value=0, max_value=1)
        
        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
        inactive = self.pipe.vae.encode(inactive, device=self.pipe.device, tiled=inputs["tiled"], tile_size=None, tile_stride=None).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        reactive = self.pipe.vae.encode(reactive, device=self.pipe.device, tiled=inputs["tiled"], tile_size=None, tile_stride=None).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        vace_video_latents = torch.concat((inactive, reactive), dim=1)
        vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
        vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

        vace_reference_image = self.pipe.preprocess_video([inputs["vace_reference_image"]])
        bs, c, f, h, w = vace_reference_image.shape
        new_vace_ref_images = []
        for j in range(f):
            new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
        vace_reference_image = new_vace_ref_images
        vace_reference_latents = self.pipe.vae.encode(vace_reference_image, device=self.pipe.device, tiled=inputs["tiled"], tile_size=None, tile_stride=None).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
        vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

        vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
        vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

        inputs["vace_context"] = vace_context
        return inputs


    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    