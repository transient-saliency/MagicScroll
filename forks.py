# !pip install diffusers==0.20.0
# !pip install transformers
# !pip install accelerate
# !pip install compel


import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

import numpy as np
import pandas as pd
import cv2
import PIL

from PIL import Image
from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    PIL_INTERPOLATION,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_omegaconf_available,
    is_transformers_available,
    logging,
    load_image
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from transformers import CLIPTextModel, CLIPTokenizer, logging as transformers_logging

from huggingface_hub import hf_hub_download

from compel import Compel

# suppress partial model loading warning
transformers_logging.set_verbosity_error()


## 准备random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

## 准备滑动窗
# 把整张大图片分解成多张小图片的集合
def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

# 修改步长
def get_dilated_views(panorama_height, panorama_width, window_size=64, stride=8, dilation_rate=1):
    panorama_height /= 8
    panorama_width /= 8
    stride *= dilation_rate  # 修改步长以包括扩张率
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def get_multi_scale_views(panorama_height, panorama_width, window_sizes=[64], strides=[8]):
    multi_scale_views_1 = []
    multi_scale_views_2 = []
    panorama_height = int(panorama_height / 8)
    panorama_width = int(panorama_width / 8)
    
    # 第一组视图：从左上角到右下角
    for window_size, stride in zip(window_sizes, strides):
        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        for i in range(num_blocks_height * num_blocks_width):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            multi_scale_views_1.append((h_start, h_end, w_start, w_end))
            multi_scale_views_2.append((h_start, h_end, w_start, w_end))
    
    return multi_scale_views_1, multi_scale_views_2


## 输入图片的尺寸、值、通道调整
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class MagicScroll(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == 'xl':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"       
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            model_key = self.sd_version  # For custom models or fine-tunes, allow people to use arbitrary versions
            # raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        self.compel_proc = Compel(tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    ## 图片编解码
    # 图片编码
    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents
    
    # 图片解码
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    

    ## 输入图像预处理
    # 将numpy图像转换为pytorch张量
    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    # 将pil图像转换为numpy
    @staticmethod
    def pil_to_numpy(images) -> np.ndarray:
        """
        Convert a PIL image or a list of PIL images to NumPy arrays.
        """
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    # 图像值的归一化
    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    # 调整图像大小
    @staticmethod
    def resize(image, height, width):
        """
        Resize image.
        """
        image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
        return image

    # 调整高度和宽度为 8 的整数倍
    @staticmethod
    def get_default_height_width(image, height, width):
        """
        This function return the height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.
        """

        if height is None:
            height = image.height // 8
        if width is None:
            width = image.width // 8

        width, height = (
            x - x % 8 for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor
        print("image_shape1:", image)
        return height, width

    # 图像处理合集
    @torch.no_grad()
    def img_preprocess(self, image, height, width) -> torch.Tensor:
        """
        Preprocess the image input. Accepted formats are PIL images.
        """
        image = [image]
        height, width = height // 8, width // 8
        image = [self.resize(i, height, width) for i in image]
        image = self.pil_to_numpy(image)  # to np
        image = self.numpy_to_pt(image)  # to pt

        # expected range [0,1], normalize to [-1,1]
        do_normalize = True
        if image.min() < 0 and do_normalize:
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)
        return image

    ## 全局latent预处理
    # 根据图像大小初始化latents并添加噪声
    @torch.no_grad()
    def prepare_latents(self, batch_size, num_channels_latents, image, timestep, height, width, dtype, device,
                        generator=None, latents=None):
        image = preprocess(image)
        print("image_shape2:", image.shape)
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * 1

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i: i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(
                    image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        print("init_latents_shape1:", init_latents.shape)  # ---------------------------
        shape = init_latents.shape

        noise1 = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        print("noise1_shape:", noise1.shape)  # ---------------------------
        init_latents = self.scheduler.add_noise(init_latents, noise1, timestep)

        print("init_latents_shape_add_noise:", init_latents.shape)  # ---------------------------

        if latents is None:
            latents = init_latents
        else:
            latents = latents.to(device)
        print("init_latents_shape2:", latents.shape)  # -----------------------------------

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    ## sliding window预处理
    # 生成512*512的随机背景
    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])
    

    @torch.no_grad()
    def get_constant_background(self, n_samples, background_color=(0.5, 0.5, 0.5)):
        # Create a constant background with the specified RGB value
        background_tensor = torch.tensor(background_color, device=self.device).view(1, 3, 1, 1).repeat(n_samples, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in background_tensor])

    
    # 计算滑动窗的权重(latent smoothing)
    @torch.no_grad()
    def edge_weights(self, window_size, weight_type='gaussian', sigma=1):
        """
        生成用于平滑latent space的权重核

        Args:
            window_size: 权重核的大小
            weight_type: 权重类型
            sigma: 高斯分布的标准差（默认为 0.5）
        """
        if weight_type == 'linear':
            weights = np.linspace(0, 1, window_size)
            weights = np.outer(weights, weights)
        elif weight_type == 'gaussian':
            x, y = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
            d = np.sqrt(x * x + y * y)
            weights = np.exp(-((d ** 2) / (2.0 * sigma ** 2)))
            # weights = np.outer(weights, weights)
        elif weight_type == 'cosine':
            x = np.linspace(-1, 1, window_size)
            weights = np.abs(np.cos(x * np.pi / 2))
            weights = np.outer(weights, weights)
        elif weight_type == 'triangle':
            x = np.linspace(-1, 1, window_size)
            weights = np.maximum(0, 1 - np.abs(x))
            weights = np.outer(weights, weights)
        else:
            raise ValueError(f'Invalid weight type: {weight_type}')

        return torch.tensor(weights, dtype=torch.float32)
    

    ## 文本预处理和编码
    # 文本基础编码
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt', is_split_into_words=True)
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # compel编码
    @torch.no_grad()
    def get_text_embeds_compel(self, prompts, negative_prompt):
        prompt_embeds = torch.cat([self.compel_proc(negative_prompt), self.compel_proc(prompts)])
        print("prompt_embeds", prompt_embeds.shape)
        return prompt_embeds


    ## 去噪过程
    # 原始去噪
    @torch.no_grad()
    def denoise(self, i, t, views, masks, latent, prompts, background_prompts, bootstrapping_backgrounds, bootstrapping,
                width, noise, text_embeds, guidance_scale, value, count):
        for h_start, h_end, w_start, w_end in views:
            masks_view = masks[:, :, h_start:h_end, w_start:w_end]
            latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(len(prompts), 1, 1, 1)
            if i % 2 == 1:
                if background_prompts is None:
                    bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                else:
                    bg = []
                    block_width = width // len(background_prompts)
                    for j in range(len(prompts) - 1):
                        start_idx = (w_start // block_width) % len(background_prompts)
                        bg.append(bootstrapping_backgrounds[start_idx])
                    bg = torch.stack(bg)
                bg = self.scheduler.add_noise(bg, noise[:, :, h_start:h_end, w_start:w_end], t)
                latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (1 - masks_view[1:])

            latent_model_input = torch.cat([latent_view] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
            value[:, :, h_start:h_end, w_start:w_end] += (latents_view_denoised * masks_view).sum(dim=0, keepdims=True)
            count[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdims=True)

        return value, count

    # 带latent smoothing的去噪  
    @torch.no_grad()
    def denoise_interp(self, i, t, views, masks, latent, prompts, background_prompts, bootstrapping_backgrounds,
                       bootstrapping,
                       width, noise, text_embeds, guidance_scale, value, count):
        edge_weights_tensor = self.edge_weights(window_size=64).to(self.device)
        for h_start, h_end, w_start, w_end in views:
            masks_view = masks[:, :, h_start:h_end, w_start:w_end]
            latent_view = latent[:, :, h_start:h_end, w_start:w_end].repeat(len(prompts), 1, 1, 1)
            if i % 2 == 1:
                if background_prompts is None:
                    bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(prompts) - 1,))]
                else:
                    bg = []
                    block_width = width // len(background_prompts)
                    for j in range(len(prompts) - 1):
                        start_idx = (w_start // block_width) % len(background_prompts)
                        bg.append(bootstrapping_backgrounds[start_idx])
                    bg = torch.stack(bg)
                bg = self.scheduler.add_noise(bg, noise[:, :, h_start:h_end, w_start:w_end], t)
                latent_view[1:] = latent_view[1:] * masks_view[1:] + bg * (1 - masks_view[1:])
            latent_model_input = torch.cat([latent_view] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']

            # 使用权重进行加权平均
            weighted_latents_view_denoised = latents_view_denoised * masks_view * edge_weights_tensor
            value[:, :, h_start:h_end, w_start:w_end] += weighted_latents_view_denoised.sum(dim=0, keepdims=True)
            count[:, :, h_start:h_end, w_start:w_end] += (masks_view * edge_weights_tensor).sum(dim=0, keepdims=True)

        return value, count


    # 获取推理步骤数
    @torch.no_grad()
    def get_timesteps(self, num_inference_steps):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        return timesteps, num_inference_steps - t_start

    # 生成总指令
    @torch.no_grad()
    def generate(self, fg_masks_1, fg_prompts_1, mg_masks_1, mg_prompts_1, images_1, fg_masks_2, fg_prompts_2, mg_masks_2, mg_prompts_2, images_2, 
                 fg_negative_prompts_1='', mg_negative_prompts_1='', fg_negative_prompts_2='', mg_negative_prompts_2='',
                 height=512, width=2048, num_inference_steps=50,
                 guidance_scale=7.5, bootstrapping_1=10, bootstrapping_2=40, background_prompts=None, use_ref_img=True):

        bootstrapping_backgrounds_1 = self.get_random_background(bootstrapping_1)
        bootstrapping_backgrounds_2 = self.get_random_background(bootstrapping_2)

        # bootstrapping_backgrounds_1 = self.get_constant_background(bootstrapping_1)
        # bootstrapping_backgrounds_2 = self.get_constant_background(bootstrapping_2)

        # Prompts -> text embeds for fg and mg
        text_embeds_fg_1 = self.get_text_embeds_compel(fg_prompts_1, fg_negative_prompts_1)  # Text embeddings for fg
        print("text_embeds_fg_1",text_embeds_fg_1.shape)
        text_embeds_mg_1 = self.get_text_embeds_compel(mg_prompts_1, mg_negative_prompts_1)  # Text embeddings for mg
        print("text_embeds_mg_1", text_embeds_mg_1.shape)  # ------------------------------------------------------------------
        text_embeds_fg_2 = self.get_text_embeds_compel(fg_prompts_2, fg_negative_prompts_2)  # Text embeddings for fg
        print("text_embeds_fg_2",text_embeds_fg_2.shape)
        text_embeds_mg_2 = self.get_text_embeds_compel(mg_prompts_2, mg_negative_prompts_2)  # Text embeddings for mg
        print("text_embeds_mg_2", text_embeds_mg_2.shape)  # ------------------------------------------------------------------


        # Define panorama grid and get views
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        batch_size = 1
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps)
        latent_timestep = timesteps[:1].repeat(batch_size)

        if use_ref_img:
            latent_1 = self.prepare_latents(
                batch_size,
                self.unet.in_channels,
                images_1,
                latent_timestep,
                height,
                width,
                text_embeds_mg_1[0].dtype,
                device
            )
            latent_2 = self.prepare_latents(
                batch_size,
                self.unet.in_channels,
                images_2,
                latent_timestep,
                height,
                width,
                text_embeds_mg_2[0].dtype,
                device
            )
        else:
            latent_1 = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
            latent_2 = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)


        fg_noise_1 = latent_1.clone().repeat(len(fg_prompts_1) - 1, 1, 1, 1)
        mg_noise_1 = latent_1.clone().repeat(len(mg_prompts_1) - 1, 1, 1, 1)        
        fg_noise_2 = latent_2.clone().repeat(len(fg_prompts_2) - 1, 1, 1, 1)
        mg_noise_2 = latent_2.clone().repeat(len(mg_prompts_2) - 1, 1, 1, 1)
        # sliding window方式1：常规
        # views = get_views(height, width)

        # sliding window方式2：多种stride(dilation*8)
        # dilation_rates = [1, 2, 4]  # 例如，使用3个不同的扩张率

        # sliding window方式3：多种窗口大小和步长
        # views = []
        # for dilation_rate in dilation_rates:
        #     views.extend(get_dilated_views(height, width, dilation_rate=dilation_rate))
        window_sizes = [64]  # 例如，使用两个不同大小的滑动窗口
        strides = [8]  # 为每个滑动窗口大小指定步长
        views_1, views_2 = get_multi_scale_views(height, width, window_sizes, strides)

        print("fg_masks_1",fg_masks_1.shape)
        print("latent_1",latent_1.shape)
        print("bootstrapping_backgrounds_1",bootstrapping_backgrounds_1.shape)
        print("fg_masks_2",fg_masks_2.shape)
        print("latent_2",latent_2.shape)
        print("bootstrapping_backgrounds_2",bootstrapping_backgrounds_2.shape)

        count_1 = torch.zeros_like(latent_1)
        value_1 = torch.zeros_like(latent_1)
        count_2 = torch.zeros_like(latent_2)
        value_2 = torch.zeros_like(latent_2)
        # -----------------------------------------------------------------------------------------------------------

        with torch.autocast('cuda'):
            # 根据目前的去噪步数去判断是去噪前景或是中景
            for i, t in enumerate(self.scheduler.timesteps):
                count_1.zero_()
                value_1.zero_()
                count_2.zero_()
                value_2.zero_()

                # 在处理完所有视图后，将最右边的 latent 复制到最左边
                # window_width = 64  # 假设窗口宽度为64
                # latent[:, :, :, :window_width] = latent[:, :, :, -window_width:]

                # 在处理完所有视图后，将最右边的 latent 复制到最左边————这部分可以继续优化
                if i < num_inference_steps:
                    if i % 2 == 0:
                        latent_1[:, :, :, :width//16] = latent_2[:, :, :, :width//16]
                    else: 
                        latent_2[:, :, :, :width//16] = latent_1[:, :, :, :width//16]

                if i > bootstrapping_1:
                    # 前景去噪
                    value_1, count_1 = self.denoise_interp(i, t, views_1, fg_masks_1, latent_1, fg_prompts_1, background_prompts,
                                                    bootstrapping_backgrounds_2, bootstrapping_2, width, fg_noise_1,
                                                    text_embeds_fg_1, guidance_scale, value_1, count_1)
                    value_2, count_2 = self.denoise_interp(i, t, views_2, fg_masks_2, latent_2, fg_prompts_2, background_prompts,
                                                    bootstrapping_backgrounds_2, bootstrapping_2, width, fg_noise_2,
                                                    text_embeds_fg_2, guidance_scale, value_2, count_2)
                else:
                    value_1, count_1 = self.denoise_interp(i, t, views_1, mg_masks_1, latent_1, mg_prompts_1, background_prompts,
                                                    bootstrapping_backgrounds_1, bootstrapping_1, width, mg_noise_1,
                                                    text_embeds_mg_1, guidance_scale, value_1, count_1)
                    value_2, count_2 = self.denoise_interp(i, t, views_2, mg_masks_2, latent_2, mg_prompts_2, background_prompts,
                                                    bootstrapping_backgrounds_1, bootstrapping_1, width, mg_noise_2,
                                                    text_embeds_mg_2, guidance_scale, value_2, count_2)
                    
                # take the MultiDiffusion step
                latent_1 = torch.where(count_1 > 0, value_1 / count_1, value_1)
                latent_2 = torch.where(count_2 > 0, value_2 / count_2, value_2)
                # latent[:, :, :, :width//2] = latent[:, :, :, width:(width+width//2)]


        # Img latents -> imgs
        imgs_1 = self.decode_latents(latent_1)  # [1, 3, 512, 512]
        img_1 = T.ToPILImage()(imgs_1[0].cpu())
        imgs_2 = self.decode_latents(latent_2)  # [1, 3, 512, 512]
        img_2 = T.ToPILImage()(imgs_2[0].cpu())
        return img_1, img_2

    ## textual inversion相关代码
    def load_textual_inversion_state_dicts(self, pretrained_model_name_or_paths, **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch",
        }
        state_dicts = []
        for pretrained_model_name_or_path in pretrained_model_name_or_paths:
            if not isinstance(pretrained_model_name_or_path, (dict, torch.Tensor)):
                # 3.1. Load textual inversion file
                model_file = None

                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                        weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=weight_name or "learned_embeds.safetensors",
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                        )
                        state_dict = safetensors.torch.load_file(model_file, device="cpu")
                    except Exception as e:
                        if not allow_pickle:
                            raise e

                        model_file = None

                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or "learned_embeds.bin",
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                state_dict = pretrained_model_name_or_path

            state_dicts.append(state_dict)

        return state_dicts

    @staticmethod
    def _retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer):
        all_tokens = []
        all_embeddings = []
        for state_dict, token in zip(state_dicts, tokens):
            if isinstance(state_dict, torch.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                loaded_token = token
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]
            else:
                raise ValueError(
                    f"Loaded state dictonary is incorrect: {state_dict}. \n\n"
                    "Please verify that the loaded state dictionary of the textual embedding either only has a single key or includes the `string_to_param`"
                    " input key."
                )

            if token is not None and loaded_token != token:
                logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            if token in tokenizer.get_vocab():
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )

            all_tokens.append(token)
            all_embeddings.append(embedding)

        return all_tokens, all_embeddings

    @staticmethod
    def _extend_tokens_and_embeddings(tokens, embeddings, tokenizer):
        all_tokens = []
        all_embeddings = []

        for embedding, token in zip(embeddings, tokens):
            if f"{token}_1" in tokenizer.get_vocab():
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
            if is_multi_vector:
                all_tokens += [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                all_embeddings += [e for e in embedding]  # noqa: C416
            else:
                all_tokens += [token]
                all_embeddings += [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        return all_tokens, all_embeddings

    def _check_text_inv_inputs(self, tokenizer, text_encoder, pretrained_model_name_or_paths, tokens):
        if tokenizer is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` or passing a `tokenizer` of type `PreTrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if text_encoder is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` or passing a `text_encoder` of type `PreTrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)} "
                f"Make sure both lists have the same length."
            )

        valid_tokens = [t for t in tokens if t is not None]
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

    def load_textual_inversion(
            self,
            pretrained_model_name_or_path: str,  # 替换为正确的模型路径
            token_name: str,  # 替换为正确的标记名称
            token: Optional[Union[str, List[str]]] = None,
            weight_name: Optional[str] = None,  # 替换为正确的权重名称，如果不需要可以保留为None
            use_safetensors: Optional[bool] = False,  # 如果需要使用 SafeTensors，请将其设置为True，否则保留为False
            **kwargs,
    ):
        # 1. Set correct tokenizer and text encoder
        tokenizer = self.tokenizer
        text_encoder = self.text_encoder

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )
        tokens = len(pretrained_model_name_or_paths) * [token] if (isinstance(token, str) or token is None) else token

        # 3. Check inputs
        self._check_text_inv_inputs(tokenizer, text_encoder, pretrained_model_name_or_paths, tokens)

        # 4. Load state dicts of textual embeddings
        state_dicts = self.load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs)

        # 4. Retrieve tokens and embeddings
        tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer)

        # 5. Extend tokens and embeddings for multi vector
        tokens, embeddings = self._extend_tokens_and_embeddings(tokens, embeddings, tokenizer)

        # 6. Make sure all embeddings have the correct size
        expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
        if any(expected_emb_dim != emb.shape[-1] for emb in embeddings):
            raise ValueError(
                "Loaded embeddings are of incorrect shape. Expected each textual inversion embedding "
                "to be of shape {input_embeddings.shape[-1]}, but are {embeddings.shape[-1]} "
            )

        # 7.2 save expected device and dtype
        device = text_encoder.device
        dtype = text_encoder.dtype

        # 7.3 Increase token embedding matrix
        text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
        input_embeddings = text_encoder.get_input_embeddings().weight

        # 7.4 Load token and embedding
        for token, embedding in zip(tokens, embeddings):
            # add tokens and get ids
            tokenizer.add_tokens(token)
            token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings.data[token_id] = embedding
            print("Loaded textual inversion embedding for {token}.")

        input_embeddings.to(dtype=dtype, device=device)

        # # 7.5 Offload the model again
        # if is_model_cpu_offload:
        #     self.enable_model_cpu_offload()
        # elif is_sequential_cpu_offload:
        #     self.enable_sequential_cpu_offload()

## 准备mask
# 单层mask
def preprocess_mask(mask_path, h, w, device):
    """
    读取mask图片，将其转换为二值化的torch张量，并按照给定的高度和宽度进行下采样

    Args:
        mask_path: str，mask图片的路径
        h: 图片的高度
        w: 图片的宽度
        device: 运行设备
    """
    # 使用PIL库打开掩码图片，并将其转换为灰度模式，然后使用numpy库将其转换为一个二维数组
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h // 8, w // 8), mode='nearest')
    return mask

# 多层mask
def load_and_preprocess_masks(i, H, W, device):
    # Directories for the mask outputs
    mask_output_fg_dir = f'{base_dir}{i}/mask_output_fg{i}'
    mask_output_mg_dir = f'{base_dir}{i}/mask_output_mg{i}'

    # Load the foreground and middle ground masks
    fg_mask_paths = [os.path.join(mask_output_fg_dir, file) for file in os.listdir(mask_output_fg_dir) if file.endswith('.png')]
    mg_mask_paths = [os.path.join(mask_output_mg_dir, file) for file in os.listdir(mask_output_mg_dir) if file.endswith('.png')]

    # Preprocess the masks
    fg_masks = torch.cat([preprocess_mask(mask_path, H, W, device) for mask_path in fg_mask_paths])
    mg_masks = torch.cat([preprocess_mask(mask_path, H, W, device) for mask_path in mg_mask_paths])

    # Create the combined masks for foreground and middle ground
    fg_bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    fg_bg_mask[fg_bg_mask < 0] = 0
    fg_bg_masks = torch.cat([fg_bg_mask, fg_masks])

    mg_bg_mask = 1 - torch.sum(mg_masks, dim=0, keepdim=True)  # This line seems to be using fg_masks, adjust if necessary
    mg_bg_mask[mg_bg_mask < 0] = 0
    mg_bg_masks = torch.cat([mg_bg_mask, mg_masks])

    return fg_bg_masks, mg_bg_masks


## 准备prompt
# 前景prompt
def read_fg_prompts(index):
    # Construct the path to the object_color.txt file
    file_path = f'{base_dir}{index}/object_color.txt'
    fg_prompts = []
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the file and extract prompts
        with open(file_path, 'r') as file:
            for line in file:
                # Split each line at ':' and take the first part as the prompt
                prompt = line.split(':')[0]
                fg_prompts.append(prompt)
    return fg_prompts

# 中景prompt
def read_mg_prompts(row):
    # Assuming the text is in the 7th column (index 6) of the DataFrame row
    text = row.iloc[6]
    # Split the text by '.' and return the resulting list
    split_text = text.split('.')
    # Exclude the last element using slicing
    return split_text[:-1]

def load_images_from_folder(folder_path, width, height):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")

            # 获取当前图像的尺寸
            original_width, original_height = image.size

            # 计算缩放比例，确保宽度和高度都小于最大值
            ratio = min(width / original_width, height / original_height)

            # 计算新的尺寸
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            # 调整图像大小
            image = image.resize((new_width, new_height), Image.LANCZOS)
            images.append(image)
    
    return images


## 单张图执行命令
def process_data_point(index, H, W, ref_image_path_1, ref_image_path_2, device):
    use_ref_img = True
    seed = 128
    steps = 30
    bootstrapping_1 = 15
    bootstrapping_2 = 20
    
    image_path_1 = ref_image_path_1
    init_image_1 = Image.open(image_path_1).convert("RGB")
    init_image_1 = init_image_1.resize((W, H))
    image_path_2 = ref_image_path_2
    init_image_2 = Image.open(image_path_2).convert("RGB")
    init_image_2 = init_image_2.resize((W, H))

    style_prompt = "traditional Chinese painting of <dqc>"
    # style_prompt = "traditional Chinese painting"
    bg_prompt = [style_prompt]
    bg_negative = 'ugly, dirty, polluted'
    fg_negative = 'ugly, dirty, unhealthy, bland, unappetizing'
    mg_negative = 'ugly, dirty, unhealthy, bland, unappetizing'

    # Load and preprocess masks
    fg_bg_masks_1, mg_bg_masks_1 = load_and_preprocess_masks(index+1, H, W, device)
    fg_bg_masks_2, mg_bg_masks_2 = load_and_preprocess_masks(index+2, H, W, device)

    # Assuming the fg_prompts and mg_prompts are related to the masks
    # Get the list of all files in mask_output_fg_dir
    fg_file_paths_1 = [os.path.join(mask_output_fg_dir_1, file) for file in os.listdir(mask_output_fg_dir_1) if file.endswith('.png')]
    fg_file_paths_2 = [os.path.join(mask_output_fg_dir_2, file) for file in os.listdir(mask_output_fg_dir_2) if file.endswith('.png')]

    # Now create fg_prompts using the list of file_paths
    fg_prompts_1 = bg_prompt + [f'{os.path.splitext(os.path.basename(path))[0]}' for path in fg_file_paths_1]
    print(fg_prompts_1)
    fg_prompts_2 = bg_prompt + [f'{os.path.splitext(os.path.basename(path))[0]}' for path in fg_file_paths_2]
    print(fg_prompts_2)

    mg_file_paths_1 = [os.path.join(mask_output_mg_dir_1, file) for file in os.listdir(mask_output_mg_dir_1) if file.endswith('.png')]
    mg_file_paths_2 = [os.path.join(mask_output_mg_dir_2, file) for file in os.listdir(mask_output_mg_dir_2) if file.endswith('.png')]

    mg_prompts_1 = [f'{os.path.splitext(os.path.basename(path))[0]} in the style of {style_prompt}' for path in mg_file_paths_1]
    print(mg_prompts_1)
    mg_prompts_2 = [f'{os.path.splitext(os.path.basename(path))[0]} in the style of {style_prompt}' for path in mg_file_paths_2]
    print(mg_prompts_2)


    # Construct negative prompts similarly
    fg_neg_prompts_1 = [bg_negative] + [fg_negative] * (len(fg_bg_masks_1)-1 ) 
    fg_neg_prompts_2 = [bg_negative] + [fg_negative] * (len(fg_bg_masks_2)-1 ) 
    mg_neg_prompts_1 = [bg_negative] + [mg_negative] * (len(mg_bg_masks_1)-1 ) 
    mg_neg_prompts_2 = [bg_negative] + [mg_negative] * (len(mg_bg_masks_2)-1 ) 

    # Assuming fg_masks and fg_prompts are already defined
    boxes_1 = []
    boxes_2 = []

    for mask in mg_bg_masks_1[1:]:  # Excluding the background mask which is at fg_masks[0]
        # Convert the mask tensor to a numpy array and then to uint8
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Keep track of the number of boxes before processing this mask
        boxes_before = len(boxes_1)
        # Get the bounding box for each contour and add to boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Normalize the bounding box coordinates to be between 0 and 1
            normalized_box = [x / mask_np.shape[1], y / mask_np.shape[0], (x + w) / mask_np.shape[1], (y + h) / mask_np.shape[0]]
            boxes_1.append(normalized_box)
        # Check if any new boxes were added for this mask
        if len(boxes_1) == boxes_before:
            # No new boxes were added for this mask, so add a (0,0,0,0) box
            boxes_1.append([0, 0, 0, 0])

    for mask in mg_bg_masks_2[1:]:  # Excluding the background mask which is at fg_masks[0]
        # Convert the mask tensor to a numpy array and then to uint8
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Keep track of the number of boxes before processing this mask
        boxes_before = len(boxes_2)
        # Get the bounding box for each contour and add to boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Normalize the bounding box coordinates to be between 0 and 1
            normalized_box = [x / mask_np.shape[1], y / mask_np.shape[0], (x + w) / mask_np.shape[1], (y + h) / mask_np.shape[0]]
            boxes_2.append(normalized_box)
        # Check if any new boxes were added for this mask
        if len(boxes_2) == boxes_before:
            # No new boxes were added for this mask, so add a (0,0,0,0) box
            boxes_2.append([0, 0, 0, 0])
    
    model_output_1, model_output_2 = model.generate(
        fg_bg_masks_1, fg_prompts_1, mg_bg_masks_1[1:], mg_prompts_1, init_image_1, 
        fg_bg_masks_2, fg_prompts_2, mg_bg_masks_2[1:], mg_prompts_2, init_image_2, 
        fg_neg_prompts_1, mg_neg_prompts_1[1:], fg_neg_prompts_2, mg_neg_prompts_2[1:],
        H, W, steps, bootstrapping_1=bootstrapping_1, bootstrapping_2=bootstrapping_2
    )

    return {'model_output_1': model_output_1, 'model_output_2': model_output_2,}


if __name__ == '__main__':
    data = pd.read_excel('./shijing_mask/data_description.xlsx', header=None)
    device = torch.device('cuda')
    sd_version = '2.1'

    # Base directory for the mask outputs
    base_dir = './shijing_mask/mask_output_'
    model = MagicScroll(device, sd_version)
    repo_id_embeds = "sd-concepts-library/dongqichang2-1"
    token_name = "<dqc>"
    model.load_textual_inversion(repo_id_embeds,token_name)

    ref_image_path_1='./layout.png'
    ref_image_path_2='./layout.png'

    # Define a directory to save the images
    save_dir = "./test" # 之前的all是缺中景的
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it does not exist

    # Process all the data
    results = []
    # for i, row in data.iterrows():  # Assuming data is your DataFrame
    for i in range(0, 4):  # Starts the loop at i=3
    # for i in [1,3,4]:
        row_1 = data.iloc[i]
        row_2 = data.iloc[i+1]

        print(i)
        mask_output_fg_dir_1 = f'{base_dir}{i+1}/mask_output_fg{i+1}' 
        mask_output_mg_dir_1 = f'{base_dir}{i+1}/mask_output_mg{i+1}' 
        mask_output_fg_dir_2 = f'{base_dir}{i+2}/mask_output_fg{i+2}' 
        mask_output_mg_dir_2 = f'{base_dir}{i+2}/mask_output_mg{i+2}' 

        # Check if the directories are empty
        if not os.path.isdir (mask_output_fg_dir_2) or not os.path.isdir (mask_output_mg_dir_2) or not os.listdir(mask_output_fg_dir_2) or not os.listdir(mask_output_mg_dir_2):
            print(f'Skipping iteration {i} as one or both mask_output directories are empty.')
            continue

        result = process_data_point(i, 512, 2048, ref_image_path_1, ref_image_path_2, device) 
        results.append(result)

        model_output_1 = result['model_output_1']
        model_output_2 = result['model_output_2']

        model_file_path_1 = os.path.join(save_dir, f'test_{i}_3_1.png')
        model_file_path_2 = os.path.join(save_dir, f'test_{i}_3_2.png')

        model_output_1.save(model_file_path_1)
        model_output_2.save(model_file_path_2)