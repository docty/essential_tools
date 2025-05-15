import sys
import os
import time
import compel
import diffusers
import torch

class DiffuserPipeline(object):
    
    def __init__(
        self,
        model_path,
        scheduler = "DPMSMS",
        scheduler_configs = None,
        lora_paths = [],
        lora_adapter_names = [],
        lora_scales = [],
        clip_skip = 2,
        textual_inversion_paths = [],
        textual_inversion_tokens = [],
        safety_checker = None,
        use_prompt_embeddings = True,
        use_compel = True,
        img2img = False,
        torch_dtype = torch.float16, 
        checkpoint = None
    ):
        
        
        self.torch_dtype = torch_dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        self.pipe = None
        self.img2img = img2img  
        self.checkpoint = checkpoint

        self.load_model_weights(model_path, torch_dtype)

        self.pipe.safety_checker = safety_checker
        
        self.load_lora_weights(lora_paths, lora_adapter_names, lora_scales)

        self.clip_skip(clip_skip)

        self.set_scheduler(scheduler, scheduler_configs)

        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_model_cpu_offload()

        #self.load_textual_inversion_weights(textual_inversion_paths, textual_inversion_tokens)

        self.prompt = None
        self.negative_prompt = None
        self.prompt_embeddings = None
        self.negative_prompt_embeddings = None
        self.use_compel = use_compel
        
        
        if self.use_compel is True:
            if len(textual_inversion_paths) > 0:
                textual_inversion_manager = compel.DiffusersTextualInversionManager(self.pipe)
            else:
                textual_inversion_manager = None
                
            self.compel = compel.Compel(tokenizer=self.pipe.tokenizer, 
                                        text_encoder=self.pipe.text_encoder,
                                        textual_inversion_manager=textual_inversion_manager, 
                                        truncate_long_prompts=False)
        else:
            self.compel = None
        
        

     
    def load_model_weights(self, model_path, torch_dtype):
        
        if os.path.splitext(model_path)[-1] == ".safetensors":
            if self.img2img is True:
                self.pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch_dtype)
            else:
                if self.checkpoint == None:
                    self.pipe = diffusers.StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype)
                elif self.checkpoint == 'XL':
                    self.pipe = diffusers.StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch_dtype)
        else:
            if self.img2img is True:
                self.pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
            else:
                self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
            
    
    def load_lora_weights(self, lora_paths = [], lora_adapter_names = [], lora_scales = [], fuse_scale = 1.0):
        
        if len(lora_adapter_names) < len(lora_paths):
            adapters = [os.path.splitext(os.path.basename(p))[0] for p in lora_paths[len(lora_adapter_names):]]
            adapters = [a.replace(".", "_") for a in adapters]
            lora_adapter_names += adapters

        if len(lora_scales) < len(lora_paths):
            lora_scales += [1.0] * (len(lora_paths) - len(lora_scales))

        
        for lp, la in zip(lora_paths, lora_adapter_names):
            self.pipe.load_lora_weights(lp, adapter_name = la)
            
        if len(lora_adapter_names) > 0:
            self.pipe.set_adapters(lora_adapter_names, adapter_weights = lora_scales)
            self.pipe.fuse_lora(adapter_names = lora_adapter_names, lora_scale = fuse_scale)

    
    def clip_skip(self, clip_skip = 0):
        if clip_skip > 0:
            clip_layers = self.pipe.text_encoder.text_model.encoder.layers
            self.pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    
    def load_textual_inversion_weights(self, textual_inversion_paths = [], textual_inversion_tokens = []):
        """Loads textual inversion pre-trained model weights into the StableDiffusionPipeline.
        Textual inversion is a technique for learning a specific concept from some images which you can use to 
        generate new images conditioned on that concept.
        https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading_adapters#textual-inversion
        https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts#textual-inversion
        
        Inputs:
            textual_inversion_paths: list
                List of paths to the pre-trained textual inversion models.
            textual_inversion_tokens:
                Optional list of corresponding textual inversion tokens. Specify tokens for more control.
        """
        if len(textual_inversion_paths) > len(textual_inversion_tokens):
            diff = [None] * (len(textual_inversion_paths) - len(textual_inversion_tokens))
            textual_inversion_tokens = textual_inversion_tokens + diff

        for tid, tis in zip(textual_inversion_paths, textual_inversion_tokens):
            if tis is None:
                # Positive textual inversion. No special token. Less control over the output.
                self.pipe.load_textual_inversion(tid)
            else:
                # Negative textual inversion. Requires a special token. More control over the output.
                self.pipe.load_textual_inversion(tid, token = tis)

    
    def set_scheduler(self, scheduler = None, scheduler_configs = None):
        
        if scheduler_configs is None or len(scheduler_configs) == 0:
            scheduler_configs = self.pipe.scheduler.config
        else:
            for k in self.pipe.scheduler.config.keys():
                scheduler_configs[k] = self.pipe.scheduler.config.get(k)
        
        if scheduler in ["EulerAncestralDiscreteScheduler", "EADS"]:
            self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(scheduler_configs)
        elif scheduler in ["EulerDiscreteScheduler", "EDS"]:
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_configs)
        elif scheduler in ["DPMSolverMultistepScheduler", "DPMSMS"]:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(scheduler_configs)

    
    def set_prompts(
        self,
        prompt = None,
        negative_prompt = None,
        use_prompt_embeddings = True,
        use_compel = False
    ):
        
        if prompt is not None:
            self.prompt = prompt
        if negative_prompt is not None:
            self.negative_prompt = negative_prompt

        # Currently making prompt embeddings require both the prompt and
        # the negative prompt to be set.
        if type(self.prompt) == str and type(self.negative_prompt) == str:
            if use_prompt_embeddings is True:
                if use_compel is True:
                    self.get_compel_prompt_embeddings()
                else:
                    self.get_prompt_embeddings()

    def get_compel_prompt_embeddings(self, return_embeddings = False):
        
        prompt_embeddings = self.compel([self.prompt])
        negative_prompt_embeddings = self.compel([self.negative_prompt])
        
        [self.prompt_embeddings, self.negative_prompt_embeddings] = self.compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeddings, negative_prompt_embeddings]
        )
        if return_embeddings is True:
            return self.prompt_embeddings, self.negative_prompt_embeddings
                    
    
    def get_prompt_embeddings(self, return_embeddings = False):
        
        max_length = self.pipe.tokenizer.model_max_length

        input_ids = self.pipe.tokenizer(
            self.prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(self.device)
        negative_ids = self.pipe.tokenizer(
            self.negative_prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(self.device)

        if input_ids.shape[-1] >= negative_ids.shape[-1]:
            shape_max_length = input_ids.shape[-1]
            negative_ids = self.pipe.tokenizer(
                self.negative_prompt, return_tensors = "pt", truncation = False, 
                padding = "max_length", max_length = shape_max_length
            ).input_ids.to(self.device)
        else:
            shape_max_length = negative_ids.shape[-1]
            input_ids = self.pipe.tokenizer(
                self.prompt, return_tensors = "pt", truncation = False, 
                padding = "max_length", max_length = shape_max_length
            ).input_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(self.pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        self.prompt_embeddings = torch.cat(concat_embeds, dim = 1)
        self.negative_prompt_embeddings = torch.cat(neg_embeds, dim = 1)

        if return_embeddings is True:
            return self.prompt_embeddings, self.negative_prompt_embeddings

    
    def run_pipe(
        self,
        prompt = None,
        negative_prompt= None,
        steps = 30,
        width = 512,
        height = 768,
        scale = 7.0,
        seed = None,
        image = None,
        strength = 0.8,
        num_images_per_prompt = 1,
        use_prompt_embeddings = True,
        verbose = False,
    ):

        self.set_prompts(prompt, negative_prompt, use_prompt_embeddings, self.use_compel)
        
         
        if self.prompt is None and self.prompt_embeddings is None:
            return

        if self.pipe is None:
            return

        start_time = time.time()

        if use_prompt_embeddings is True:
            prompt = None
            negative_prompt = None
            prompt_embeds = self.prompt_embeddings
            negative_prompt_embeds = self.negative_prompt_embeddings
        else:
            prompt = self.prompt
            negative_prompt = self.negative_prompt
            prompt_embeds = None
            negative_prompt_embeds = None

        imgs = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            image = image,
            strength = strength,
            width = width,
            height = height,
            guidance_scale = scale,
            num_inference_steps = steps,
            num_images_per_prompt = num_images_per_prompt,
            generator = None if seed == None else torch.manual_seed(seed),
            
        ).images

        end_time = time.time()
        time_elapsed = end_time - start_time
        if verbose is True:
            sys.stdout.write("{:.2f}s.\n".format(time_elapsed));
        return imgs
