import os
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from typing import Any, Callable, Optional, List, Dict, Tuple

# === Keep only the required degraders and basic utilities (all CPU-safe for DataLoader workers) ===
import math
import yaml
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.data.transforms import augment
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

# —— Existing prompt/constant imports (SPACIAL_TOKEN/GENERATE_TOKEN kept for consistency;
#    not used inside the dataset itself) —— #
from fapeir.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN

def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())
    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))
    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

class RealESRGAN_degradation(object):
    def __init__(self, degradation_config=None, device='cpu'):
        if degradation_config is None:
            degradation_config = {
                'scale': 4,
                'color_jitter_prob': 0.0,
                'gray_prob': 0.0,
                'resize_prob': [0.2, 0.7, 0.1],
                'resize_range': [0.15, 1.5],
                'gaussian_noise_prob': 0.5,
                'noise_range': [1, 30],
                'poisson_scale_range': [0.05, 3.0],
                'gray_noise_prob': 0.4,
                'jpeg_range': [30, 95],
                'second_blur_prob': 0.8,
                'resize_prob2': [0.3, 0.4, 0.3],
                'resize_range2': [0.3, 1.2],
                'gaussian_noise_prob2': 0.5,
                'noise_range2': [1, 25],
                'poisson_scale_range2': [0.05, 2.5],
                'gray_noise_prob2': 0.4,
                'jpeg_range2': [30, 95],
                'kernel_info': {
                    'blur_kernel_size': 21,
                    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                    'sinc_prob': 0.1,
                    'blur_sigma': [0.2, 3],
                    'betag_range': [0.5, 4],
                    'betap_range': [1, 2],
                    'blur_kernel_size2': 21,
                    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                    'sinc_prob2': 0.1,
                    'blur_sigma2': [0.2, 1.5],
                    'betag_range2': [0.5, 4],
                    'betap_range2': [1, 2],
                    'final_sinc_prob': 0.8
                }
            }
        self.opt = degradation_config
        self.device = device
        optk = self.opt['kernel_info']
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        self.final_sinc_prob = optk['final_sinc_prob']
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)

    def random_augment(self, img_gt):
        if isinstance(img_gt, Image.Image):
            img_gt = np.array(img_gt)
        img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        return img_gt

    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        from torchvision.transforms.functional import (
            adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
        )
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)
            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)
            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)
            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_kernels(self):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, kernel_size,
                self.blur_sigma, self.blur_sigma, [-math.pi, math.pi],
                self.betag_range, self.betap_range, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size,
                self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2, self.betap_range2, noise_range=None)

        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.random.uniform() < self.opt['kernel_info']['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        return kernel, kernel2, sinc_kernel

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=True):
        img_gt = np.asarray(img_gt)/255.
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        ori_h, ori_w = img_gt.size()[2:4]
        scale_final = 4
        out = filter2D(img_gt, kernel1)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)
        if np.random.uniform() < 0.5:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
        if np.random.uniform() < self.opt['gray_prob']:
            out = torch.mean(out, dim=1, keepdim=True).repeat(1, 3, 1, 1)  # fake grayscale, keep 3 channels
        if np.random.uniform() < self.opt['color_jitter_prob']:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)
        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        return img_gt, img_lq

class Qwen2VLDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset: holds no LVLM/processor and performs no text generation
    or tokenization.
    Responsibilities:
      - Read data_txt / json manifests
      - Produce LR/GT pairs based on need_degradation (or use pre-paired input/GT)
      - Return PIL images or numpy tensors for unified processing in the main process
    """
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        transform: Callable,                # final tensor transform for GT → [-1, 1] (can also be applied in the main process)
        lvlm_model=None,                    # compatibility parameter (ignored)
        tokenizer=None,                     # compatibility parameter (ignored)
        prompter=None,                      # compatibility parameter (ignored)
        image_processor=None,               # compatibility parameter (ignored)
        processor=None,                     # compatibility parameter (ignored)
        min_pixels: int = 384*384,
        max_pixels: int = 384*384,
        image_token_length: int = 729,
        only_generated_task: bool = True,   # compatibility parameter (ignored)
        drop_prompt_rate: float = 0.0,      # compatibility parameter (ignored)
        anyres: bool = False,
        mask_weight_type: str = 'log',      # only passed through as need_weight
        siglip_processor=None,              # compatibility parameter (ignored)
        ocr_enhancer: bool = False,         # compatibility parameter (ignored)
        random_data: bool = False,
        maxnum_per_data: int = -1,
        notry: bool = False,
        enable_degradation: bool = True,
        degradation_config: dict = None,
        device: str = 'cpu',
        # All remaining parameters are accepted for compatibility and ignored
        **kwargs
    ):
        assert dataset_type in ('qwen2vl', 'qwen2p5vl', 'llava')
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self.transform = transform
        self.enable_degradation = enable_degradation
        self.notry = notry
        self.force_superres_label_field = kwargs.get("force_superres_infer", "force_superres_infer")
        self.respect_size_superres = kwargs.get("respect_size_superres", True)

        if self.enable_degradation:
            self.degradation_processor = RealESRGAN_degradation(degradation_config, device='cpu')  # CPU-safe inside DataLoader workers

        self._load_data(maxnum_per_data)

    def _load_data(self, maxnum_per_data=-1):
        for dataset in self.datasets:
            parts = dataset.split(",")
            if len(parts) == 3:
                image_root, json_file, need_weight = parts
                need_degradation = "true"
            elif len(parts) == 4:
                image_root, json_file, need_weight, need_degradation = parts
            else:
                raise ValueError(f"Invalid dataset format: {dataset}")
            with open(json_file, "r") as f:
                data = json.load(f)
            if maxnum_per_data > 0 and maxnum_per_data < len(data):
                data = random.sample(data, maxnum_per_data)
            for line in tqdm(data, desc=f"loading {os.path.basename(json_file)}"):
                if "image" not in line:
                    line["image"] = []
                if isinstance(line["image"], str):
                    line["image"] = [line["image"]]
                assert isinstance(line["image"], list)
                # Resolve to absolute paths
                line["image"] = [os.path.join(image_root, p) for p in line["image"]]
                line["need_weight"] = need_weight
                line["need_degradation"] = (str(need_degradation).lower() == "true")
                # Default the force-SR flag to False if absent
                if self.force_superres_label_field not in line:
                    line[self.force_superres_label_field] = False
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def _open_as_rgb(self, p) -> Image.Image:
        if isinstance(p, Image.Image):
            return p.convert('RGB')
        return Image.open(p).convert('RGB')

    def _random_crop_512(self, image_path):
        img = self._open_as_rgb(image_path)
        w, h = img.size
        if w < 512 or h < 512:
            scale = max(512 / w, 512 / h)
            img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
            w, h = img.size
        left = random.randint(0, w - 512)
        top  = random.randint(0, h - 512)
        return img.crop((left, top, left+512, top+512))

    def _resize_512(self, image_path):
        return self._open_as_rgb(image_path).resize((512, 512), Image.Resampling.BICUBIC)

    def __getitem__(self, idx):
        data = self.data[idx]
        # Return raw materials only; LVLM inference and tokenization are handled uniformly in the main process
        if len(data["image"]) == 0:
            raise ValueError("Sample has no images.")
        need_degradation = data.get("need_degradation", True)
        need_weight = data.get("need_weight", "true")

        # Reference images (optional) + LR + GT
        refs: List[Image.Image] = []
        lr_pil: Image.Image = None
        gt_pil: Image.Image = None

        if need_degradation:
            # Degradation path: randomly crop GT to 512 × 512, then synthesise LR via degradation
            target_path = data["image"][-1]
            gt_pil = self._random_crop_512(target_path)
            if self.enable_degradation:
                gt_tensor, lq_tensor = self.degradation_processor.degrade_process(gt_pil, resize_bak=False)
                lr_np = tensor2img(lq_tensor.squeeze(0).cpu(), rgb2bgr=False, out_type=np.uint8)
                lr_pil = Image.fromarray(lr_np)
                # Overwrite gt_pil with the tensor2img-synchronised version
                gt_np = tensor2img(gt_tensor.squeeze(0).cpu(), rgb2bgr=False, out_type=np.uint8)
                gt_pil = Image.fromarray(gt_np)
            else:
                lr_pil = gt_pil.copy()
            if len(data["image"]) > 1:
                for p in data["image"][:-1]:
                    refs.append(self._open_as_rgb(p))
        else:
            # Paired path: resize both input and target directly to 512 × 512
            if len(data["image"]) < 2:
                raise ValueError("Paired dataset needs at least 2 images.")
            input_path = data["image"][0]
            target_path = data["image"][-1]
            lr_pil = self._resize_512(input_path)
            gt_pil = self._resize_512(target_path)
            if len(data["image"]) > 2:
                for p in data["image"][1:-1]:
                    refs.append(self._resize_512(p))

        # Target tensor normalised to [-1, 1] (required during training)
        gt_tensor = torch.tensor(np.array(gt_pil)).float().div(255.0)
        gt_tensor = rearrange(gt_tensor, "h w c -> c h w")
        gt_tensor = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])(gt_tensor)

        return {
            "lr_pil": lr_pil,              # used for LVLM text generation, processor image path, and VAE LQ encoding
            "gt_pil": gt_pil,              # used for saving/visualisation only; training uses gt_tensor
            "gt_tensor": gt_tensor,        # [-1, 1] normalised GT tensor
            "refs": refs,                  # reference images (may be empty)
            "need_weight": need_weight,    # used for weight mask computation
            "meta": {
                "all_image_paths": data["image"],
                "need_degradation": need_degradation,
            }
        }