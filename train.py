import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

import math
import lpips
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from vision_aided_loss.cv_losses import multilevel_loss
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.serialization import safe_globals
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.utils.tensor_fragment import fragment_address

from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast, AutoImageProcessor, AutoTokenizer, AutoProcessor, PreTrainedTokenizer

from fapeir.models import MODEL_TYPE
from fapeir.dataset import DATASET_TYPE
from fapeir.utils.prompter import PROMPT_TYPE
from fapeir.utils.flux_pipeline import FluxPipeline
from fapeir.utils.anyres_util import dynamic_resize
from fapeir.dataset.data_collator import RawTrainCollator
from fapeir.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from fapeir.utils.denoiser_prompt_embedding_flux import encode_prompt
from fapeir.dataset.drealsr_test_dataset import create_multi_scale_datasets
from fapeir.dataset.realsr_test_dataset import create_multi_realsr_datasets
from fapeir.training.configuration_denoise import FAPEIRTrainingDenoiseConfig
from fapeir.dataset.weather_test_dataset import create_multi_weather_datasets
from fapeir.dataset.generic_test_dataset import create_multi_generic_datasets

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    with safe_globals([LossScaler, ZeroStageEnum, fragment_address]):
        return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

USE_GAN_DEFAULT = True
LAMBDA_L2_DEFAULT = 50.0
LAMBDA_LPIPS_DEFAULT = 5.0
LAMBDA_GAN_DEFAULT = 0.5
FIXED_TIMESTEP_DEFAULT = 300
LAMBDA_FREQ_DEFAULT = 1e-3

def _cfg_get(cfg, key, default=None):
    """Read config value compatible with both dict and HF Config objects."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def cfg_set(cfg, key, value):
    """Write config value compatible with both dict and HF Config objects."""
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
                raise NotImplementedError
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

EXPERT_USER_PROMPT = r"""You are an expert image restoration task classifier.

Available tasks (use EXACT lowercase tokens):
deraining | desnowing | dehazing | deblur | denoise | light_enhancement | super_resolution

Critical distinctions:

**RAIN (deraining)**
- Linear streaks: parallel/near-parallel elongated lines
- Overlay effect: streaks cross sharp edges WITHOUT blurring them
- Directional: consistent orientation (diagonal/vertical)
- Can coexist with low contrast, but streaks are visible as distinct overlays

**SNOW (desnowing)**
- Particles: round/irregular white blobs, bokeh discs
- Size variation: larger near, smaller far
- Random distribution, NOT linear/parallel

**HAZE (dehazing)**
- Depth-dependent: far objects more degraded than near
- Milky appearance with desaturation
- Distance gradient clearly visible

**BLUR (deblur)**
- Edge smearing: boundaries themselves are widened/soft
- Uniform softness or motion trails
- Object edges lose geometric precision

**NOISE (denoise)**
- Grain on crisp edges: edge structure intact but covered by speckles
- Random texture in flat areas
- High-ISO appearance

**UNDEREXPOSED (light_enhancement)**
- Globally dark, no depth gradient
- Histogram left-biased, shadows crushed
- Color cast possible

**LOW RESOLUTION (super_resolution)**
- Insufficient spatial sampling: small native H×W or strong aliasing
- Loss of fine textures; blockiness/jagged edges when upscaled
- Distinct from blur: edges can appear jagged/aliased rather than uniformly smeared

Frequency decision rules:

Choose **high** when degradation is dominated by fine-scale artifacts or missing detail that lives in high spatial frequencies:
- deblur (recover sharp edges and textures)
- denoise (suppress noisy high-frequency speckles while preserving true details)
- deraining / desnowing (remove thin streaks/particles and recover edge micro-structure)

Choose **low** when degradation is dominated by global/slow-varying components:
- dehazing (restore low-frequency contrast/airlight; depth-dependent veiling)
- light_enhancement (global exposure/illumination and tone mapping)

If signals suggest mixed conditions, choose the dominant component according to the most visible impairment.

Pipeline rules:
- DO NOT use any task names in the Pipeline.

Output format (EXACTLY 1 plain-text line):
Task: <task_token>, Focus: <high|low>, Rationale: <brief reason>, Pipeline: <step1 -> step2 -> ...>.
"""
NEUTRAL_PROMPT = "Analyze and return 1 line: Task, Focus, Rationale, Pipeline."

RAIN_LINE_THR = 0.16; RAIN_ANISO_THR = 0.40; RAIN_FREQ_THR = 1.05
SNOW_BLOB_MIN = 25; SNOW_ANISO_MAX = 0.42
NOISE_MAD_THR = 0.0050; NOISE_CHROMA_THR = 0.0095; NOISE_SCORE_THR = 0.45; VHF_NOISE_MIN = 0.30
BLUR_GRAD95_THR = 0.17; BLUR_LAP_THR = 0.27; BLUR_HF_THR = 0.052
HAZE_SCORE_THR = 0.50; HAZE_DEPTH_GRAD = 0.03
UNDEREXP_MEAN_THR = 0.32; UNDEREXP_P50_THR = 0.26

import re
_TASK_LINE_RE = re.compile(r"^\s*Task\s*:\s*([a-zA-Z_]+)")
ALL_TASKS_ORDERED = ["deraining", "desnowing", "dehazing", "deblur", "denoise", "light_enhancement", "super_resolution"]
ALLOWED_TASKS_PARSE = set(ALL_TASKS_ORDERED)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _mean3(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="reflect")
    s = (p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
         p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
         p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:])
    return s / 9.0

def prepare_image_for_infer_keep_detail(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side: return img
    scale = max_side / float(m)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    try: resample = Image.Resampling.LANCZOS
    except AttributeError: resample = Image.LANCZOS
    return img.resize(new_size, resample)

def compute_rain_features(img: Image.Image) -> Dict[str, Any]:
    gray = img.convert("L")
    g = np.asarray(gray, dtype=np.float32) / 255.0
    gy, gx = np.gradient(g); mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(ang, bins=18, range=(0.0, np.pi), weights=mag)
    hist_sum = float(hist.sum()) + 1e-6
    line_score = float(hist.max()) / hist_sum
    anisotropy = float((hist.max() - hist.mean()) / (hist.mean() + 1e-6))
    F = np.fft.fft2(g)
    ps = np.abs(F) ** 2; ps = np.fft.fftshift(ps)
    h, w = ps.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    rmax = rr.max() + 1e-6
    mid_mask = (rr > 0.15 * rmax) & (rr < 0.45 * rmax)
    low_mask = rr < 0.15 * rmax
    mid_energy = ps[mid_mask].mean() if np.any(mid_mask) else ps.mean()
    low_energy = ps[low_mask].mean() if np.any(low_mask) else ps.mean()
    freq_ratio = float(mid_energy / (low_energy + 1e-6))
    return {"line_score": line_score, "anisotropy": anisotropy, "freq_ratio": freq_ratio,
            "hint": f"Rain: line={line_score:.2f}, aniso={anisotropy:.2f}, freq={freq_ratio:.2f}"}

def compute_snow_features(img: Image.Image) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g); mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(ang, bins=18, range=(0.0, np.pi), weights=mag)
    anisotropy = float((hist.max() - hist.mean()) / (hist.mean() + 1e-6))
    # blob detection
    try:
        from scipy import ndimage
        bright = (g > 0.78).astype(np.uint8)
        labeled, num = ndimage.label(bright)
        sizes = ndimage.sum(bright, labeled, range(1, num+1))
        small_blobs = int(np.sum((sizes >= 3) & (sizes <= 200)))
    except Exception:
        # fallback simplified implementation
        bw = (g > 0.78).astype(np.uint8)
        H, W = bw.shape
        visited = np.zeros_like(bw, dtype=bool)
        small_blobs = 0
        for y in range(H):
            for x in range(W):
                if bw[y, x] and not visited[y, x]:
                    stack = [(y, x)]; visited[y, x] = True; size = 0
                    while stack:
                        cy, cx = stack.pop(); size += 1
                        for ny in range(max(0, cy-1), min(H, cy+2)):
                            for nx in range(max(0, cx-1), min(W, cx+2)):
                                if bw[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True; stack.append((ny, nx))
                    if 3 <= size <= 200: small_blobs += 1
    return {"small_blobs": small_blobs, "anisotropy": anisotropy, "hint": f"Snow: blobs={small_blobs}, aniso={anisotropy:.2f}"}

def compute_noise_features(img: Image.Image) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g); grad_mag = np.hypot(gx, gy)
    thr = float(np.quantile(grad_mag, 0.25)); flat = grad_mag < thr
    g_blur = _mean3(g); resid = g - g_blur; r_flat = resid[flat]
    med = np.median(r_flat); noise_mad = float(np.median(np.abs(r_flat - med)))
    ycbcr = np.asarray(img.convert("YCbCr"), dtype=np.float32) / 255.0
    cb, cr = ycbcr[...,1], ycbcr[...,2]; chroma_std = float(0.5*(cb[flat].std()+cr[flat].std()))
    s1 = _sigmoid(50.0 * (noise_mad - NOISE_MAD_THR))
    s2 = _sigmoid(50.0 * (chroma_std - NOISE_CHROMA_THR))
    noise_score = float(0.6*s1 + 0.4*s2)
    return {"noise_mad": noise_mad, "chroma_std": chroma_std, "noise_score": noise_score,
            "hint": f"Noise: mad={noise_mad:.4f}, chroma={chroma_std:.4f}, score={noise_score:.2f}"}

def compute_blur_features(img: Image.Image) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    p = np.pad(g, 1, mode="reflect")
    lap = (k[0,0]*p[:-2,:-2] + k[0,1]*p[:-2,1:-1] + k[0,2]*p[:-2,2:] +
           k[1,0]*p[1:-1,:-2] + k[1,1]*p[1:-1,1:-1] + k[1,2]*p[1:-1,2:] +
           k[2,0]*p[2:,:-2] + k[2,1]*p[2:,1:-1] + k[2,2]*p[2:,2:])
    lap_var = float(lap.var())
    F = np.fft.fft2(g); ps = np.abs(F)**2; ps = np.fft.fftshift(ps)
    h, w = ps.shape; yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy-h/2)**2 + (xx-w/2)**2); rmax = rr.max()+1e-6
    outer_mask = (rr > 0.35*rmax) & (rr <= 0.9*rmax)
    inner_mask = (rr <= 0.35*rmax)
    outer_vals = ps[outer_mask]; inner_vals = ps[inner_mask]
    hf_energy = float((outer_vals.mean() if outer_vals.size else ps.mean()) /
                      ((inner_vals.mean() if inner_vals.size else ps.mean()) + 1e-6))
    g_s = _mean3(_mean3(g)); gy, gx = np.gradient(g_s)
    mag = np.hypot(gx, gy); grad95 = float(np.quantile(mag, 0.95))
    return {"lap_var": lap_var, "hf_energy": hf_energy, "grad95": grad95,
            "hint": f"Blur: lapVar={lap_var:.3f}, hf={hf_energy:.3f}, grad95={grad95:.3f}"}

def compute_haze_features(img: Image.Image) -> Dict[str, Any]:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2: arr = np.stack([arr,arr,arr], axis=-1)
    R,G,B = arr[...,0], arr[...,1], arr[...,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    dark = np.minimum(np.minimum(R,G),B); dark_mean = float(dark.mean())
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    sat_mean = float(hsv[...,1].mean())
    h = Y.shape[0]; top_slice = slice(0, max(1,int(h*0.4))); bot_slice = slice(int(h*0.6), h)
    top_bright = float(Y[top_slice,:].mean()); bot_bright = float(Y[bot_slice,:].mean())
    depth_grad = top_bright - bot_bright
    s1 = _sigmoid(7.0 * (dark_mean - 0.33))
    s2 = _sigmoid(7.0 * (0.30 - sat_mean))
    s3 = _sigmoid(8.0 * (depth_grad - 0.03))
    haze_score = float(0.4*s1 + 0.3*s2 + 0.3*s3)
    return {"haze_score": haze_score, "depth_grad": depth_grad,
            "dark_mean": dark_mean, "sat_mean": sat_mean,
            "hint": f"Haze: score={haze_score:.2f}, depth_grad={depth_grad:.3f}"}

def compute_exposure_features(img: Image.Image) -> Dict[str, Any]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    Y = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    meanY = float(Y.mean()); p50 = float(np.quantile(Y, 0.50))
    underexp = (meanY < UNDEREXP_MEAN_THR) or (p50 < UNDEREXP_P50_THR)
    return {"meanY": meanY, "p50": p50, "underexposed": bool(underexp),
            "hint": f"Exposure: meanY={meanY:.3f}, p50={p50:.3f}"}

def compute_all_features_for_infer(img: Image.Image) -> Dict[str, Dict]:
    img_for_feat = prepare_image_for_infer_keep_detail(img, max_side=1024)
    return {
        "rain": compute_rain_features(img_for_feat),
        "snow": compute_snow_features(img_for_feat),
        "noise": compute_noise_features(img_for_feat),
        "blur": compute_blur_features(img_for_feat),
        "haze": compute_haze_features(img_for_feat),
        "expo": compute_exposure_features(img_for_feat),
    }

def generate_feature_hints_string(features: Dict[str, Dict]) -> str:
    return "\n".join([
        features["rain"]["hint"],
        features["snow"]["hint"],
        features["noise"]["hint"],
        features["blur"]["hint"],
        features["haze"]["hint"],
        features["expo"]["hint"],
    ])

def parse_predicted_task(generated_text: str) -> Optional[str]:
    if not generated_text: return None
    for raw_line in generated_text.splitlines():
        m = _TASK_LINE_RE.match(raw_line)
        if m:
            tok = m.group(1).strip().lower().replace("-", "_")
            return tok if tok in ALLOWED_TASKS_PARSE else None
    return None

def check_strong_conflict(pred_task: str, f: Dict[str, Dict]) -> Optional[List[str]]:
    rain, snow, noise, blur, haze, expo = f["rain"], f["snow"], f["noise"], f["blur"], f["haze"], f["expo"]
    if pred_task in ["deblur", "dehazing"]:
        rain_confident = ((rain["line_score"] > RAIN_LINE_THR and rain["anisotropy"] > RAIN_ANISO_THR) or
                          (rain["line_score"] > RAIN_LINE_THR*1.3))
        if rain_confident: return ["deraining", "desnowing", "deblur", "denoise"]
    if pred_task in ["deblur", "dehazing"]:
        noise_confident = (noise["noise_score"] > NOISE_SCORE_THR and blur["grad95"] > BLUR_GRAD95_THR)
        if noise_confident: return ["denoise", "light_enhancement"]
    if pred_task in ["deblur", "denoise"]:
        haze_confident = (haze["haze_score"] > HAZE_SCORE_THR and haze["depth_grad"] > HAZE_DEPTH_GRAD)
        if haze_confident: return ["dehazing", "light_enhancement"]
    if pred_task == "deraining":
        if snow["small_blobs"] > SNOW_BLOB_MIN and snow["anisotropy"] < SNOW_ANISO_MAX:
            return ["desnowing", "denoise"]
    return None

# ======== Generation / tokenization helpers ========
def _normalize_messages_for_processor(messages: List[dict]) -> List[dict]:
    norm = []
    for m in messages:
        m = dict(m)
        c = m.get("content", "")
        if isinstance(c, str):
            m["content"] = [{"type": "text", "text": c}]
        elif isinstance(c, list):
            new_list = []
            for item in c:
                if isinstance(item, dict) and "type" in item:
                    if item["type"] in ("text","image","video"):
                        new_list.append(item)
                    else:
                        new_list.append({"type":"text","text":str(item)})
                elif isinstance(item, str):
                    new_list.append({"type":"text","text":item})
                else:
                    try:
                        if isinstance(item, Image.Image):
                            new_list.append({"type":"image","image":item})
                        else:
                            new_list.append({"type":"text","text":str(item)})
                    except Exception:
                        new_list.append({"type":"text","text":str(item)})
            if not new_list: new_list=[{"type":"text","text":""}]
            m["content"]=new_list
        else:
            m["content"]=[{"type":"text","text":""}]
        norm.append(m)
    return norm

def _combine_user_text(use_expert_prompt: bool, prompt_used: str) -> str:
    return (EXPERT_USER_PROMPT if use_expert_prompt else "") + prompt_used

@torch.no_grad()
def _generate_text_once(lvlm_model, processor, tokenizer, prompt_used: str, first_img_pil: Image.Image,
                        generation_max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, gen_max_pixels=512*512,
                        use_expert_prompt=True):
    combined_text = _combine_user_text(use_expert_prompt, prompt_used)
    messages = [{"role":"user","content":[{"type":"image","image":first_img_pil}, {"type":"text","text":combined_text}]}]
    messages = _normalize_messages_for_processor(messages)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True, max_pixels=gen_max_pixels,
    )
    device = next(lvlm_model.parameters()).device
    model_dtype = next(lvlm_model.parameters()).dtype
    if hasattr(inputs, "to"): inputs = inputs.to(device)
    for k in ("pixel_values","image_grid_thw","pixel_values_videos","video_grid_thw","aspect_ratio_ids"):
        if k in inputs and inputs[k] is not None and isinstance(inputs[k], torch.Tensor):
            if inputs[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
                inputs[k] = inputs[k].to(device, dtype=model_dtype)
            else:
                inputs[k] = inputs[k].to(device)
    embed_device = getattr(lvlm_model, "model", lvlm_model).embed_tokens.weight.device if hasattr(lvlm_model, "model") else device
    inputs["input_ids"] = inputs["input_ids"].to(embed_device)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        inputs["attention_mask"] = inputs["attention_mask"].to(embed_device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    gen_cfg = dict(max_new_tokens=generation_max_new_tokens, temperature=temperature, top_p=top_p,
                   do_sample=do_sample, pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True)
    outputs = lvlm_model.generate(**inputs, generation_config=None, output_type="lvlm", return_dict_in_generate=True, **gen_cfg)
    seq = outputs.sequences
    input_len = inputs["input_ids"].shape[1]
    gen_ids = seq[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def infer_5lines_with_repicks(first_img_pil: Image.Image, lvlm_model, processor, tokenizer, prompt_text=NEUTRAL_PROMPT, append_global_feature_hints=True,
                              use_expert_prompt=True, respect_size_superres=True, disable_superres=True, repick_if_superres=True, selective_repick=True,
                              repick_temperature=0.2, repick_top_p=0.5, gen_max_pixels=512*512, generation_max_new_tokens=512, generation_temperature=0.7,
                              generation_top_p=0.9, generation_do_sample=True) -> Tuple[str, Optional[str], Dict]:
    features = compute_all_features_for_infer(first_img_pil)
    feature_hint = generate_feature_hints_string(features) if append_global_feature_hints else ""
    prompt_used = f"{prompt_text}\n\n{feature_hint}" if feature_hint else prompt_text
    w,h = first_img_pil.size
    if respect_size_superres and (w<512 and h<512):
        text_out = _generate_text_once(
            lvlm_model, processor, tokenizer, prompt_used, first_img_pil,
            generation_max_new_tokens, generation_temperature, generation_top_p,
            generation_do_sample, gen_max_pixels, use_expert_prompt
        )
        text_out = "Task: super_resolution, Pipeline: upscale > detail_refine"
        return text_out, "super_resolution", features
    text_out = _generate_text_once(
        lvlm_model, processor, tokenizer, prompt_used, first_img_pil,
        generation_max_new_tokens, generation_temperature, generation_top_p,
        generation_do_sample, gen_max_pixels, use_expert_prompt
    )
    pred_task = parse_predicted_task(text_out)
    if disable_superres and pred_task == "super_resolution" and repick_if_superres:
        forbid = "\n\nIMPORTANT: Do not choose super_resolution. Choose from the 6 main tasks."
        text2 = _generate_text_once(
            lvlm_model, processor, tokenizer, prompt_used + forbid, first_img_pil,
            generation_max_new_tokens, repick_temperature, repick_top_p,
            True, gen_max_pixels, use_expert_prompt
        )
        pred2 = parse_predicted_task(text2)
        if pred2 and pred2 != "super_resolution":
            text_out, pred_task = text2, pred2
    if selective_repick and pred_task:
        suggested = check_strong_conflict(pred_task, features)
        if suggested:
            allowed_str = " | ".join(suggested)
            forbid2 = f"\n\nIMPORTANT: Based on strong evidence, choose ONE from: {allowed_str}"
            text3 = _generate_text_once(
                lvlm_model, processor, tokenizer, prompt_used + forbid2, first_img_pil,
                generation_max_new_tokens, repick_temperature, repick_top_p,
                True, gen_max_pixels, use_expert_prompt
            )
            pred3 = parse_predicted_task(text3)
            if pred3 in suggested:
                text_out, pred_task = text3, pred3
    return text_out, pred_task, features

def preprocess_for_siglip(siglip_processor, images, input_size: int):
    """
    Preprocess images for SigLIP.
    images: a single PIL.Image or list[PIL.Image]
    Returns: pixel_values tensor of shape (N, 3, H, W)
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
    batch = siglip_processor(
        images=images,
        return_tensors="pt",
        do_resize=True,
        size={"height": input_size, "width": input_size},
        do_center_crop=True,
        crop_size={"height": input_size, "width": input_size},
        do_rescale=True,
        do_normalize=True,
    )
    return batch.pixel_values  # (N, 3, H, W)

def pad_2d_right(x: torch.Tensor, tgt_len: int, pad_value: int):
    L = x.shape[1]
    if L >= tgt_len: return x[:, :tgt_len]
    pad = torch.full((1, tgt_len - L), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)

def _visual_forward_siglip(model, image: torch.Tensor, return_feats: bool = False, return_pooled_feats: bool = False, num_spatial_levels: int = 3):
    assert image.dim() == 4 and image.size(1) == 3, "image must be BCHW with 3 channels."
    with torch.no_grad():
        outputs = model(pixel_values=image.to(model.dtype), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled = getattr(outputs, "pooler_output", None)
    B, _, H, W = image.shape
    img_size = getattr(model.config, "image_size", min(H, W))
    patch = getattr(model.config, "patch_size", 16)
    gh = img_size // patch
    gw = img_size // patch
    n_layers = model.config.num_hidden_layers
    candidate_ids = [max(1, int(n_layers * 0.25)), max(1, int(n_layers * 0.5)), n_layers]
    layer_ids = sorted(set(candidate_ids))
    if len(layer_ids) > num_spatial_levels:
        layer_ids = layer_ids[-num_spatial_levels:]
    elif len(layer_ids) < num_spatial_levels:
        layer_ids = layer_ids + [n_layers] * (num_spatial_levels - len(layer_ids))
    spatial_feats = []
    for lid in layer_ids:
        hs = hidden_states[lid]  # (B, seq, D)
        B_, S, D = hs.shape
        if S == gh * gw + 1:
            tokens = hs[:, 1:, :]
        elif S == gh * gw:
            tokens = hs
        else:
            g = int(math.sqrt(S))
            if g * g == S:
                gh = gw = g
                tokens = hs
            else:
                raise ValueError(f"Unexpected sequence length for SigLIP: {S}, cannot form {gh}x{gw} grid.")
        fmap = tokens.reshape(B_, gh, gw, D).permute(0, 3, 1, 2).contiguous()
        spatial_feats.append(fmap)
    if pooled is None:
        last_hs = hidden_states[-1]
        if last_hs.shape[1] == gh * gw + 1:
            pooled = last_hs[:, 1:, :].mean(dim=1)
        else:
            pooled = last_hs.mean(dim=1)
    if return_feats and not return_pooled_feats:
        return spatial_feats
    if return_pooled_feats:
        return spatial_feats + [pooled]
    return pooled

class ImageSigLIPBackbone(nn.Module):
    def __init__(self, siglip_model):
        super().__init__()
        self.model = siglip_model.eval().requires_grad_(False)

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward_siglip(
            self.model,
            image,
            return_feats=return_feats,
            return_pooled_feats=return_pooled_feats,
            num_spatial_levels=3,
        )

class ImageSigLIPDiscriminator(nn.Module):
    def __init__(self, siglip_model, siglip_processor=None, input_size: Optional[int] = None, precision: str = "fp32"):
        super().__init__()
        if siglip_model is None:
            raise ValueError("ImageSigLIPDiscriminator: `siglip_model` cannot be None. Please ensure SiglipVisionModel is loaded in main().")
        self.backbone = ImageSigLIPBackbone(siglip_model)
        self.backbone.eval().requires_grad_(False)
        self.input_size = int(input_size or getattr(siglip_model.config, "image_size", 224))
        if siglip_processor is not None and hasattr(siglip_processor, "image_mean") and hasattr(siglip_processor, "image_std"):
            mean = torch.tensor(siglip_processor.image_mean, dtype=torch.float32)
            std = torch.tensor(siglip_processor.image_std, dtype=torch.float32)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)
        self._detect_feature_dims()
        self.loss_fn = multilevel_loss(alpha=0.8)

    def _detect_feature_dims(self):
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=device, dtype=dtype)
        with torch.no_grad():
            feats = self.backbone.encode_image(dummy, return_pooled_feats=True)
        spatial_features = []
        pooled_features = []
        for i, f in enumerate(feats):
            if f.dim() == 4:
                spatial_features.append(f)
            else:
                pooled_features.append(f)
        if len(spatial_features) == 0:
            raise ValueError("No spatial features found from SigLIP.")
        spatial_channels = [f.shape[1] for f in spatial_features]
        pooled_dim = pooled_features[-1].shape[-1] if pooled_features else spatial_features[-1].shape[1]
        self.decoder = MultiLevelDConv(
            level=len(spatial_features) + 1,
            in_ch1=tuple(spatial_channels),
            in_ch2=pooled_dim,
            out_ch=512,
            down=2,
        )

    def train(self, mode=True):
        self.decoder.train(mode)
        return self

    def eval(self):
        self.train(False)
        return self

    def requires_grad_(self, requires_grad=True):
        self.decoder.requires_grad_(requires_grad)
        return self

    def forward(self, x, for_real=True, for_G=False, verbose=False, return_logits=False):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        x = x * 0.5 + 0.5
        x = (x - self.image_mean[None, :, None, None]) / self.image_std[None, :, None, None]
        features = self.backbone.encode_image(x, return_pooled_feats=True)
        dec_dtype = next(self.decoder.parameters()).dtype
        spatial_features, pooled_features = [], []
        for feat in features:
            if feat.dim() == 4:
                spatial_features.append(feat.to(dec_dtype))
            else:
                pooled_features.append(feat.to(dec_dtype))
        organized_features = spatial_features + [pooled_features[-1]]
        logits_per_level = self.decoder(organized_features)
        loss = self.loss_fn(logits_per_level, for_real=for_real, for_G=for_G)
        if not return_logits:
            return loss
        else:
            return loss, logits_per_level

class MultiLevelDConv(nn.Module):
    def __init__(self, level=4, in_ch1=(384, 768, 1536), in_ch2=1024, out_ch=512, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), down=2):
        super().__init__()
        self.decoder = nn.ModuleList()
        self.level = level
        if len(in_ch1) != level - 1:
            print(f"Warning: in_ch1 length ({len(in_ch1)}) != level-1 ({level-1})")
            if len(in_ch1) < level - 1:
                in_ch1 = list(in_ch1) + [in_ch1[-1]] * (level - 1 - len(in_ch1))
            else:
                in_ch1 = in_ch1[:level-1]
        for i in range(level - 1):
            in_channels = in_ch1[i]
            self.decoder.append(
                nn.Sequential(
                    BlurPool(in_channels, pad_type="zero", stride=1, pad_off=1) if down > 1 else nn.Identity(),
                    spectral_norm(nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                    activation,
                    BlurPool(out_ch, pad_type="zero", stride=1),
                    spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)),
                )
            )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c=None):
        """x: [fmap1 (B,C1,H,W), fmap2, fmap3, ..., pooled (B,in_ch2)]"""
        final_pred = []
        dec_dtype = next(self.decoder[0].parameters()).dtype
        for i in range(self.level - 1):
            if i < len(x) and x[i].dim() == 4:
                xi = x[i]
                if xi.dtype != dec_dtype:
                    xi = xi.to(dec_dtype)
                final_pred.append(self.decoder[i](xi).squeeze(1))
            else:
                print(f"Warning: Missing or invalid spatial feature at level {i}")
                B = x[0].shape[0] if len(x) > 0 and torch.is_tensor(x[0]) else 1
                dummy_out = torch.zeros(B, 1, device=self.out.weight.device, dtype=dec_dtype)
                final_pred.append(dummy_out)
        if len(x) > 0:
            pooled_feat = x[-1]
            if pooled_feat.dim() > 2:
                pooled_feat = pooled_feat.view(pooled_feat.shape[0], -1)
            if pooled_feat.dtype != dec_dtype:
                pooled_feat = pooled_feat.to(dec_dtype)
            h = self.decoder[-1](pooled_feat)
            out = self.out(h)
            if self.embed is not None and c is not None:
                out = out + torch.sum(self.embed(c).to(dec_dtype) * h, 1, keepdim=True)
            final_pred.append(out)
        return final_pred


# ===== Image preprocessing aligned with the legacy pipeline =====
def _process_vision_info_old(vision_infos: list, factor: int = 1):
    """
    Equivalent to the legacy process_vision_info used by _load_image:
    - Supports str/PIL inputs
    - First adjusts image area according to min/max pixels, then aligns H/W to multiples of factor
    - Returns image_inputs (list[PIL]) and video_inputs (None)
    """
    def _to_pil(x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.open(x).convert("RGB")

    image_inputs = []
    for info in vision_infos:
        img = _to_pil(info["image"])
        min_pixels = int(info.get("min_pixels", img.width * img.height))
        max_pixels = int(info.get("max_pixels", img.width * img.height))

        w, h = img.size
        area = w * h
        # Scale according to area constraints
        if area > max_pixels:
            scale = (max_pixels / float(area)) ** 0.5
        elif area < min_pixels:
            scale = (min_pixels / float(area)) ** 0.5
        else:
            scale = 1.0
        if abs(scale - 1.0) > 1e-6:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img = img.resize((nw, nh), Image.Resampling.BICUBIC)

        # Align side lengths to multiples of factor
        if factor and factor > 1:
            w, h = img.size
            nw = (w // factor) * factor
            nh = (h // factor) * factor
            nw = max(factor, nw)
            nh = max(factor, nh)
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), Image.Resampling.BICUBIC)

        image_inputs.append(img)

    return image_inputs, None  # video_inputs = None


def _load_image_old(
    image_slice,                 # list[str|PIL], refs first, last entry is the input (degraded) image
    max_pixels: int,
    min_pixels: int,
    processor,
    image_token: str,
    factor: int,
    drop_prompt: bool,
    prompt: str,
    mask_weight_type: str,
    siglip_processor,
    need_weight: str,
):
    """
    Equivalent implementation of the legacy _load_image:
    - When drop_prompt=True, pixel_values / image_grid / image_token_lengths are NOT injected (consistent with legacy logic)
    - Still collects pil_pixel_values (for weight mask) and siglip_pixel_values (last image only)
    """
    pixel_values_list = []
    image_grid_thw_list = []
    image_token_lengths = []
    pil_pixel_values = []
    siglip_pixel_values = []

    for image_path in image_slice:
        vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
        image_inputs, video_inputs = _process_vision_info_old([vision_infos], factor=factor)
        _inputs = processor(
            text=[f"dummy {image_token}"],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Consistent with legacy: when drop_prompt=True, do not populate pixel_values / grid / token_lengths
        if not drop_prompt:
            pixel_values_list.append(_inputs.pixel_values)   # (T, C, H, W) where T=1
            image_grid_thw_list.append(_inputs.image_grid_thw)
            tok_id = processor.tokenizer.convert_tokens_to_ids(image_token)
            img_tok_len = int((_inputs.input_ids[0] == tok_id).sum())
            image_token_lengths.append(img_tok_len)

        pil_pixel_values.append(image_inputs[0])

        # siglip_pixel_values retains only the last assigned value;
        # the legacy code also overwrites each iteration, equivalent to keeping only the last (input) image
        if siglip_processor is not None:
            img_for_siglip = image_path if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
            siglip_pixel_values = siglip_processor(images=img_for_siglip, return_tensors="pt").pixel_values

    if len(pixel_values_list) > 0:
        pixel_values = torch.concat(pixel_values_list, dim=0)
        image_grid_thw = torch.concat(image_grid_thw_list, dim=0)
    else:
        pixel_values = None
        image_grid_thw = None

    # Weight mask (consistent with legacy)
    if mask_weight_type is not None:
        from fapeir.utils.get_mask import get_weight_mask
        _, weights = get_weight_mask(pil_pixel_values, prompt, mask_weight_type, need_weight)
        if str(need_weight).lower() == 'false':
            # Legacy code asserts all ones when need_weight == 'false'
            assert torch.all(weights == 1)
    else:
        weights = torch.tensor([])

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "image_token_lengths": image_token_lengths,
        "pil_pixel_values": pil_pixel_values,
        "siglip_pixel_values": siglip_pixel_values,
        "weights": weights,
    }


def build_training_batch_from_raw_samples(
    raw_batch: Dict[str, Any],
    lvlm_model, tokenizer: PreTrainedTokenizer, processor,
    prompter, image_token: str, image_begin_token: str, image_end_token: str,
    gen_cfg: Dict[str, Any],
    repick_cfg: Dict[str, Any],              # kept in signature but no longer used
    append_global_feature_hints=True,        # kept in signature but no longer used
    use_expert_prompt=True,                  # kept in signature but no longer used
    mask_weight_type: Optional[str] = None,
    factor: int = 2,
    max_pixels: int = 384 * 384,
    min_pixels: int = 384 * 384,
    siglip_processor=None,
    siglip_input_size: Optional[int] = None, # kept in signature but no longer used
    # Legacy pipeline control flags
    only_generated_task: bool = True,
    drop_prompt_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Fully replicates the legacy getitem pipeline (no "5-line inference & repick"):
    - User segment contains <image>; assistant segment contains generated placeholder
    - Assistant segment is replaced only with begin token, end token is NOT appended (strictly matches legacy behavior)
    - prompt_for_t5 = prompt.replace('<image>', '').replace('\\n', '')
    - When drop_prompt is active: pixel_values / image_token_lengths are NOT injected
    """
    B = len(raw_batch["lr_pil_list"])
    vocab = tokenizer.get_vocab()
    assert image_token in vocab, "image_token is not in the vocabulary"
    image_token_id       = tokenizer.convert_tokens_to_ids(image_token)
    image_begin_token_id = tokenizer.convert_tokens_to_ids(image_begin_token)
    image_end_token_id   = tokenizer.convert_tokens_to_ids(image_end_token)
    assert isinstance(image_begin_token_id, int), f"tokenizer missing image begin token `{image_begin_token}`"
    assert isinstance(image_end_token_id, int),   f"tokenizer missing image end token `{image_end_token}`"

    input_ids_list, labels_list = [], []
    prompts_list, image_position_list = [], []
    all_pixel_values, all_image_grid_thw, all_pil_pixel_values = [], [], []
    all_siglip_pixel_values = []
    vae_pixels_list, weights_list = [], []
    t5_texts: List[str] = []   # Key output for legacy pipeline: text used by T5

    GENERATED = GENERATE_TOKEN
    EOS = prompter.eos_token if hasattr(prompter, "eos_token") else ""

    for i in range(B):
        lr_pil: Image.Image   = raw_batch["lr_pil_list"][i]
        gt_pil: Image.Image   = raw_batch["gt_pil_list"][i]   # used for PIL saving or alignment only
        refs: List[Image.Image] = raw_batch["refs_list"][i]
        need_weight = raw_batch["need_weight_list"][i]

        gen_text, pred_task, features = infer_5lines_with_repicks(
            lr_pil, lvlm_model, processor, tokenizer,
            prompt_text=NEUTRAL_PROMPT,
            append_global_feature_hints=append_global_feature_hints,
            use_expert_prompt=use_expert_prompt,
            respect_size_superres=True,
            disable_superres=repick_cfg["disable_superres"],
            repick_if_superres=repick_cfg["repick_if_superres"],
            selective_repick=repick_cfg["selective_repick"],
            repick_temperature=repick_cfg["repick_temperature"],
            repick_top_p=repick_cfg["repick_top_p"],
            gen_max_pixels=gen_cfg["gen_max_pixels"],
            generation_max_new_tokens=gen_cfg["generation_max_new_tokens"],
            generation_temperature=gen_cfg["generation_temperature"],
            generation_top_p=gen_cfg["generation_top_p"],
            generation_do_sample=gen_cfg["generation_do_sample"],
        )

        # Assemble legacy-style conversation:
        #   human turn: <image> + natural language instruction
        #   assistant turn: GENERATED placeholder
        # Note: temp_text is used as the "last-turn instruction" since the simplified
        # dataset does not have data["conversations"]. It will be stripped to form the T5 prompt.
        temp_text = (
            "Please restore this real-world degraded image by removing any possible degradations according to {0}. Produce a clear and natural result "
            "that preserves fine details, textures, and the original structure of the scene.".format(gen_text)
        )
        conversations_raw = [
            {"from": "human", "value": f"<image>\n{temp_text}"},
            {"from": "gpt",   "value": GENERATED},
        ]

        # Legacy pipeline: T5 / mask use the raw text of the last human turn
        prompt_text_for_mask = ""
        for msg in reversed(conversations_raw):
            if msg["from"] == "human":
                prompt_text_for_mask = msg["value"]
                break
        assert prompt_text_for_mask != "", "last human turn not found"

        # Version passed to prompter using its own role name definitions
        conversations_for_prompter = [
            {"from": prompter.user_role,      "value": f"<image>\n{temp_text}"},
            {"from": prompter.assistant_role, "value": GENERATED},
        ]

        # drop_prompt logic (consistent with legacy)
        drop_prompt = False
        if only_generated_task:
            if random.random() >= drop_prompt_rate:
                prompt_list = prompter.get_train_prompt(conversations_for_prompter)
            else:
                drop_prompt = True
                prompt_list = [
                    {"from": prompter.system_role,   "value": "You are a helpful assistant."},
                    {"from": prompter.user_role,     "value": "Generate an image."},
                    {"from": prompter.assistant_role,"value": GENERATED},
                ]
                prompt_list = prompter.get_train_prompt(prompt_list)
        else:
            prompt_list = prompter.get_train_prompt(conversations_for_prompter)


        # Text replacement stage (strictly follows legacy behavior:
        # assistant segment is replaced with begin token only; end token is NOT appended)
        has_generated_image = False
        prompt_text_for_mask = ""  # text used by get_mask in legacy code
        for it in prompt_list:
            # Replace <image> with image_token placeholder (user segment only)
            it["prompt"] = it["prompt"].replace("<image>", image_token)

            # Assistant segment: replace "GENERATED + <|im_end|>" with "image_begin_token" (no end appended)
            if GENERATED in it["prompt"]:
                assert it["from"] == prompter.assistant_role, "generation placeholder must be in assistant segment"
                if f"{GENERATED}{EOS}" in it["prompt"]:
                    it["prompt"] = it["prompt"].replace(f"{GENERATED}{EOS}", image_begin_token)
                    if EOS and not it["prompt"].endswith(EOS):
                        it["prompt"] = it["prompt"] + EOS  # extreme fallback (usually already has eos)
                else:
                    it["prompt"] = it["prompt"].replace(GENERATED, image_begin_token)
                    if EOS and not it["prompt"].endswith(EOS):
                        it["prompt"] = it["prompt"] + EOS
                has_generated_image = True

        if only_generated_task and (not has_generated_image):
            raise ValueError("Only generated task is not supported. But this prompt does not contain generated image token.")

        # Tokenize segment by segment (consistent with legacy)
        ids_list, lab_list = [], []
        for it in prompt_list:
            toks = tokenizer(it["prompt"], return_tensors="pt", truncation=True, max_length=1024)
            if it.get("is_labels", False):
                lab_list.append(toks.input_ids)
            else:
                lab_list.append(torch.full_like(toks.input_ids, -100))
            ids_list.append(toks.input_ids)

        input_ids = torch.cat(ids_list, dim=1)
        labels    = torch.cat(lab_list, dim=1)

        # Visual side: refs + input image (lr), consistent with legacy getitem's image_slice
        image_slice = list(refs) + [lr_pil]

        # Legacy _load_image equivalent
        # Note: when drop_prompt=True, pixel_values are NOT injected
        image_dict = _load_image_old(
            image_slice=image_slice,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            processor=processor,
            image_token=image_token,
            factor=factor,
            drop_prompt=drop_prompt,
            prompt=prompt_text_for_mask,
            mask_weight_type=mask_weight_type,
            siglip_processor=siglip_processor,
            need_weight=need_weight,
        )
        image_token_lengths = image_dict["image_token_lengths"]
        pixel_values        = image_dict["pixel_values"]
        image_grid_thw      = image_dict["image_grid_thw"]
        pil_pixel_values    = image_dict["pil_pixel_values"]
        siglip_pix          = image_dict["siglip_pixel_values"]
        weights             = image_dict["weights"]

        input_ids, labels, image_position = _process_image_token(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_begin_token_id=image_begin_token_id,
            image_end_token_id=image_end_token_id,
            image_token_lengths=image_token_lengths,
            labels=labels,
        )

        # VAE input: use the last image in image_slice (i.e., the LQ/degraded input image)
        inp_np = np.array(image_slice[-1])
        inp_tensor = torch.from_numpy(inp_np).float().div(255.0).permute(2, 0, 1)
        vae_pixels_list.append(transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])(inp_tensor))

        # Collect batch lists
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        image_position_list.append(image_position)
        prompts_list.append(prompt_text_for_mask)
        all_pixel_values.append(pixel_values)
        all_image_grid_thw.append(image_grid_thw)
        all_pil_pixel_values.append(pil_pixel_values)
        all_siglip_pixel_values.append(siglip_pix if isinstance(siglip_pix, torch.Tensor) else torch.empty(0))
        weights_list.append(weights)

        # Key: T5 prompt text (legacy pipeline)
        # Use the last human-turn text, strip <image> and newlines
        t5_text = prompt_text_for_mask.replace("<image>", "").replace("\n", "")
        t5_texts.append(t5_text if not drop_prompt else "")  # empty string when drop_prompt=True

    # Pad text sequences to the maximum length in the batch (consistent with legacy)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    max_len = max(x.shape[1] for x in input_ids_list)
    input_ids = torch.cat([_pad_2d_right(x, max_len, pad_id) for x in input_ids_list], dim=0)
    labels    = torch.cat([_pad_2d_right(x, max_len, -100)   for x in labels_list], dim=0)
    attention_mask = (input_ids != pad_id)

    # Concatenate pixel batches
    if any(x is None for x in all_pixel_values):
        pixel_values   = None
        image_grid_thw = None
    else:
        pixel_values   = torch.cat(all_pixel_values, dim=0)
        image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

    # SigLIP batch (legacy logic uses only the "last image" per sample;
    # unified here as BxCxHxW, or None if unavailable)
    if len(all_siglip_pixel_values) > 0 and isinstance(all_siglip_pixel_values[0], torch.Tensor) and all_siglip_pixel_values[0].numel() > 0:
        siglip_pixel_values = torch.cat(all_siglip_pixel_values, dim=0)  # (B,3,H,W)
    else:
        siglip_pixel_values = None

    # VAE batch
    vae_pixel_values = torch.stack(vae_pixels_list, dim=0) if len(vae_pixels_list) > 0 else None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "image_position": image_position_list,
        "vae_pixel_values": vae_pixel_values,
        "pil_pixel_values": all_pil_pixel_values,
        "weights": weights_list,
        "prompts": prompts_list,
        "siglip_pixel_values": siglip_pixel_values,
        # New addition: text for T5 (strict legacy processing)
        "t5_texts": t5_texts,
    }


# Legacy utility: equivalent to the original, renamed as private to avoid conflicts
def _pad_2d_right(x: torch.Tensor, tgt_len: int, pad_value: int):
    L = x.shape[1]
    if L >= tgt_len: return x[:, :tgt_len]
    pad = torch.full((x.shape[0], tgt_len - L), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


@torch.no_grad()
def _process_image_token(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_begin_token_id: int,
    image_end_token_id: int,
    image_token_lengths: List[int],
    labels: Optional[torch.Tensor] = None,
):
    # Kept fully consistent with the legacy version
    image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
    image_position = []
    offset = 0
    cur_i = 0
    if isinstance(image_token_lengths, int):
        image_token_lengths = [image_token_lengths] * len(image_token_indices[1])
    for idx in image_token_indices[1]:
        image_token_length = image_token_lengths[cur_i]
        adjusted_idx = idx + offset
        assert input_ids[0, adjusted_idx] == image_token_id, "assert input_ids[0, adjusted_idx] == image_token_id"

        # Insert begin + image_token * len + end
        input_ids = torch.cat(
            [
                input_ids[:, :adjusted_idx],
                input_ids.new_full((input_ids.shape[0], 1), image_begin_token_id),
                input_ids.new_full((input_ids.shape[0], image_token_length), image_token_id),
                input_ids.new_full((input_ids.shape[0], 1), image_end_token_id),
                input_ids[:, adjusted_idx + 1 :],
            ],
            dim=1,
        )
        if labels is not None:
            labels = torch.cat(
                [
                    labels[:, :adjusted_idx],
                    labels.new_full((labels.shape[0], 1), image_begin_token_id),
                    labels.new_full((labels.shape[0], image_token_length), -100),
                    labels.new_full((labels.shape[0], 1), -100),
                    labels[:, adjusted_idx + 1 :],
                ],
                dim=1,
            )

        adjusted_idx += 1  # skip begin token
        image_position.append(adjusted_idx.item())
        offset += image_token_length - 1
        offset += 2  # begin/end
        cur_i += 1
    return input_ids, labels, image_position

from validation import run_validation

# ==================== Entry point: main ====================
def main(args: FAPEIRTrainingDenoiseConfig, attn_implementation='sdpa'):
    use_gan = _cfg_get(args.training_config, "use_gan", USE_GAN_DEFAULT)
    lambda_l2 = float(_cfg_get(args.training_config, "lambda_l2", LAMBDA_L2_DEFAULT))
    lambda_lpips = float(_cfg_get(args.training_config, "lambda_lpips", LAMBDA_LPIPS_DEFAULT))
    lambda_gan = float(_cfg_get(args.training_config, "lambda_gan", LAMBDA_GAN_DEFAULT))
    fixed_timestep = int(_cfg_get(args.training_config, "fixed_timestep", FIXED_TIMESTEP_DEFAULT))
    lambda_freq = float(_cfg_get(args.training_config, "lambda_freq", LAMBDA_FREQ_DEFAULT))
    # Test dataset path configuration
    test_data_root = _cfg_get(args.dataset_config, "test_data_root", 'datasets/sr_testing_data/DrealSR')
    test_scales = _cfg_get(args.dataset_config, "test_scales", [2, 4])
    weather_test_data_root = _cfg_get(args.dataset_config, "weather_test_data_root", 'datasets/sr_testing_data')
    weather_test_types = _cfg_get(args.dataset_config, "weather_test_types", ["weather1", "weather2", "Snow100K-L", "Snow100K-S"])
    realsr_test_data_root = _cfg_get(args.dataset_config, "realsr_test_data_root", 'datasets/sr_testing_data/RealSR')
    realsr_camera_types = _cfg_get(args.dataset_config, "realsr_camera_types", ["Canon", "Nikon"])
    realsr_scale_factors = _cfg_get(args.dataset_config, "realsr_scale_factors", [2, 4])
    generic_test_data_root = _cfg_get(args.dataset_config, "generic_test_data_root", 'datasets/sr_testing_data')
    generic_dataset_types = _cfg_get(args.dataset_config, "generic_dataset_types", ["dehazing_test", "LOL2", "RealBlur_J", "RealBlur_R",
                                                                                     "GoPro", "Urban100_15", "Urban100_25", "Urban100_50"])
    num_validation_samples_per_scale = _cfg_get(args.training_config, "num_validation_samples_per_scale", 100000)
    logging_dir = Path(args.training_config.output_dir, args.training_config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.training_config.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        if args.training_config.output_dir is not None:
            os.makedirs(args.training_config.output_dir, exist_ok=True)
    set_seed(args.training_config.seed, device_specific=True)
    
    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize image quality metrics
    quality_metrics = None
    resume_checkpoint_path = None
    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint != "latest":
            path = os.path.basename(args.training_config.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.training_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.training_config.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.training_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            resume_checkpoint_path = os.path.join(args.training_config.output_dir, path)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
    else:
        initial_global_step = 0
    dataset_type = args.dataset_config.dataset_type
    model_class = MODEL_TYPE[dataset_type]
    dataset_class = DATASET_TYPE[dataset_type]

    def set_grad_for_params(model, name_patterns, requires: bool):
        for name, param in model.named_parameters():
            if any(pat in name for pat in name_patterns):
                param.requires_grad_(requires)
               
    # Setup model saving hook — triggered before accelerator.save_state(save_path)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), model_class):
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(output_dir, "fapeir"))
                    if processor is not None:
                        processor.save_pretrained(os.path.join(output_dir, "fapeir"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                if weights:
                    weights.pop()
            if use_gan and (D is not None):
                discriminator_save_path = os.path.join(output_dir, "discriminator")
                os.makedirs(discriminator_save_path, exist_ok=True)
                accelerator.print(f"[Checkpoint] Saving discriminator to {discriminator_save_path}")
                try:
                    D_state = accelerator.get_state_dict(D)
                except Exception:
                    D_state = accelerator.unwrap_model(D).state_dict()
                torch.save(D_state, os.path.join(discriminator_save_path, "D_state_dict.pth"))
                if 'opt_D' in locals() or 'opt_D' in globals():
                    if opt_D is not None:
                        try:
                            torch.save(opt_D.state_dict(), os.path.join(discriminator_save_path, "opt_D_state_dict.pth"))
                        except Exception as e:
                            accelerator.print(f"[Checkpoint] Warning: failed to save opt_D.state_dict(): {e}")
    accelerator.register_save_state_pre_hook(save_model_hook)
     
    # Load models
    lvlm_model = model_class.from_pretrained(args.model_config.pretrained_lvlm_name_or_path, attn_implementation=attn_implementation, torch_dtype=weight_dtype)
    accelerator.wait_for_everyone()
    free_memory()
    lvlm_tokenizer, image_processor, processor = None, None, None
    if dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl':
        processor = AutoProcessor.from_pretrained(args.model_config.pretrained_lvlm_name_or_path)
        lvlm_tokenizer = processor.tokenizer
        lvlm_tokenizer.padding_side = 'left'
        image_processor = processor.image_processor
    elif dataset_type == 'llava':
        lvlm_tokenizer = AutoTokenizer.from_pretrained(args.model_config.pretrained_lvlm_name_or_path)
        image_processor = AutoImageProcessor.from_pretrained(args.model_config.pretrained_lvlm_name_or_path)
    else:
         raise NotImplementedError(f"Only support dataset_type in ['qwen2vl', 'qwen2p5vl', 'llava'], but found {dataset_type}")
    siglip_processor, siglip_model = None, None
    if args.model_config.pretrained_siglip_name_or_path is not None:
        from transformers import SiglipImageProcessor, SiglipVisionModel
        siglip_processor = SiglipImageProcessor.from_pretrained(args.model_config.pretrained_siglip_name_or_path)
        siglip_model = SiglipVisionModel.from_pretrained(args.model_config.pretrained_siglip_name_or_path, attn_implementation=attn_implementation)
    tokenizer_one = CLIPTokenizer.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="tokenizer_2")
    text_encoder_cls_one = CLIPTextModel.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="text_encoder")
    text_encoder_cls_two = T5EncoderModel.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="text_encoder_2")
    text_encoder_cls_one.eval()
    text_encoder_cls_two.eval()
    vae = AutoencoderKL.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="vae")
    vae_dtype = torch.float32 if args.model_config.vae_fp32 else weight_dtype
    if siglip_model is not None:
        siglip_model.to(accelerator.device, dtype=weight_dtype)
    for mod in (vae, text_encoder_cls_one, text_encoder_cls_two, lvlm_model, siglip_model):
        if mod is not None:
            mod.requires_grad_(False)
    lvlm_model.denoise_tower.requires_grad_(False)
    if _cfg_get(lvlm_model.config, "shortcut_image_embeds", False) and hasattr(lvlm_model.vision_tower, "shortcut_projector"):
        accelerator.print("Training shortcut projector")
        lvlm_model.vision_tower.shortcut_projector.requires_grad_(True)
    if args.training_config.gradient_checkpointing:
        lvlm_model.denoise_tower.denoiser.enable_gradient_checkpointing()
    def set_grad_for_params_(requires, *name_patterns):
        set_grad_for_params(lvlm_model, list(name_patterns), requires)
    set_grad_for_params_(False, 'denoise_tower.denoise_projector', 'denoise_tower.vae_projector', 'denoise_tower.siglip_projector')
    if args.model_config.only_tune_mlp2:
        set_grad_for_params_(True, 'denoise_tower.denoise_projector')
    elif args.model_config.only_tune_mlp3:
        set_grad_for_params_(True, 'denoise_tower.vae_projector')
    elif args.model_config.only_tune_siglip_mlp:
        set_grad_for_params_(True, 'denoise_tower.siglip_projector')
    if args.model_config.with_tune_mlp2:
        set_grad_for_params_(True, 'denoise_tower.denoise_projector')
    if args.model_config.with_tune_mlp3:
        set_grad_for_params_(True, 'denoise_tower.vae_projector')
    if args.model_config.with_tune_siglip_mlp:
        set_grad_for_params_(True, 'denoise_tower.siglip_projector')
    if hasattr(lvlm_model.denoise_tower, "enable_lora_moe"):
        lvlm_model.denoise_tower.enable_lora_moe()
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_cls_one, text_encoder_cls_two]
    
    # Load scheduler (primarily used for single-step decoding)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_config.pretrained_denoiser_name_or_path, subfolder="scheduler")
    lpips_ckpt = getattr(args.training_config, "lpips_weights_path", 'checkpoints/vgg.pth')
    if lpips_ckpt is not None and os.path.isfile(lpips_ckpt):
        net_lpips = lpips.LPIPS(net="vgg", pretrained=False, verbose=False).to(accelerator.device)
        sd = torch.load(lpips_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = net_lpips.load_state_dict(sd, strict=False)
        if accelerator.is_main_process:
            print(f"[LPIPS] loaded from local: {lpips_ckpt}")
            if missing:   print(f"[LPIPS] missing keys: {missing}")
            if unexpected:print(f"[LPIPS] unexpected keys: {unexpected}")
    else:
        net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device)
    net_lpips.eval().requires_grad_(False)

    # Initialize D and opt_D to None
    D, opt_D = None, None

    if use_gan:
        if siglip_model is None:
            accelerator.print("[GAN] use_gan=True but SigLIP was not loaded. GAN is automatically disabled. Please set model_config.pretrained_siglip_name_or_path.")
            use_gan = False
        else:
            disc_precision = "fp32"
            D = ImageSigLIPDiscriminator(
                siglip_model=siglip_model,
                siglip_processor=siglip_processor,
                input_size=getattr(siglip_model.config, "image_size", 224),
                precision=disc_precision,
            )
            opt_D = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, D.parameters()),
                lr=args.training_config.learning_rate,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                eps=args.training_config.adam_epsilon,
                weight_decay=args.training_config.adam_weight_decay
            )

    # Generator optimizer parameters (G only; D is excluded)
    use_deepspeed_optimizer = (accelerator.state.deepspeed_plugin is not None and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config)
    use_deepspeed_scheduler = (accelerator.state.deepspeed_plugin is not None and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config)

    gen_trainable_params = []
    gen_trainable_names = []
    for name, param in lvlm_model.named_parameters():
        if param.requires_grad:
            gen_trainable_params.append(param)
            gen_trainable_names.append(name)
    if accelerator.is_main_process:
        with open("trainable_params_G.txt", "w") as f:
            for name in gen_trainable_names:
                f.write(f"{name}\n")
        accelerator.print(f"Generator trainable params: {len(gen_trainable_names)} (written to trainable_params_G.txt)")
        if use_gan and D is not None:
            disc_trainable_names = [name for name, p in D.named_parameters() if p.requires_grad]
            with open("trainable_params_D.txt", "w") as f:
                for name in disc_trainable_names:
                    f.write(f"{name}\n")
            accelerator.print(f"Discriminator trainable params: {len(disc_trainable_names)} (written to trainable_params_D.txt)")

    # Preserve the mixed optimizer approach as originally intended
    if accelerator.state.deepspeed_plugin is not None:
        import prodigyopt
        optimizer = prodigyopt.Prodigy(
            gen_trainable_params,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            beta3=args.training_config.prodigy_beta3,
            weight_decay=args.training_config.adam_weight_decay,
            eps=args.training_config.adam_epsilon,
            decouple=args.training_config.prodigy_decouple,
            use_bias_correction=args.training_config.prodigy_use_bias_correction,
            safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
            d_coef=args.training_config.prodigy_d_coef,
        )
    else:
        if args.training_config.optimizer.lower() == "prodigy":
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                gen_trainable_params,
                betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
                beta3=args.training_config.prodigy_beta3,
                weight_decay=args.training_config.adam_weight_decay,
                eps=args.training_config.adam_epsilon,
                decouple=args.training_config.prodigy_decouple,
                use_bias_correction=args.training_config.prodigy_use_bias_correction,
                safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
                d_coef=args.training_config.prodigy_d_coef,
            )
        else:
            raise ValueError("In non-DeepSpeed mode, use a different optimizer or implement your own.")

    prompter = PROMPT_TYPE[dataset_type]()
    anchor_pixels = args.dataset_config.height * args.dataset_config.width
    resize_lambda = transforms.Lambda(
        lambda img: transforms.Resize(
            dynamic_resize(img.shape[1], img.shape[2], args.dataset_config.anyres, anchor_pixels),
            interpolation=transforms.InterpolationMode.BICUBIC
        )(img)
    )
    transform = transforms.Compose([
        resize_lambda, transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    dataset = dataset_class(
        lvlm_model=None, tokenizer=None, prompter=None, image_processor=None, processor=None,
        dataset_type=dataset_type,
        data_txt=args.dataset_config.data_txt,
        transform=transform,
        min_pixels=args.dataset_config.min_pixels,
        max_pixels=args.dataset_config.max_pixels,
        anyres=args.dataset_config.anyres,
        mask_weight_type=args.training_config.mask_weight_type,
        ocr_enhancer=args.dataset_config.ocr_enhancer,
        random_data=args.dataset_config.random_data,
    )
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.dataset_config.batch_size,
        shuffle=True,
        pin_memory=args.dataset_config.pin_memory,
        num_workers=args.dataset_config.num_workers,
        collate_fn=RawTrainCollator(),
        prefetch_factor=None if args.dataset_config.num_workers == 0 else 4, 
    )

    # Load test datasets
    test_datasets_dict = {}
    if test_data_root is not None and os.path.exists(test_data_root):
        drealsr_datasets = create_multi_scale_datasets(
            lvlm_model=lvlm_model,
            tokenizer=lvlm_tokenizer,
            data_root=test_data_root,
            scales=test_scales,
            output_dir=None,
            transform=transform,
            prompter=prompter,
            min_pixels=args.dataset_config.min_pixels,
            max_pixels=args.dataset_config.max_pixels,
            mask_weight_type=args.training_config.mask_weight_type,
            siglip_processor=siglip_processor,
            seed=42,
        )
        test_datasets_dict["DrealSR"] = drealsr_datasets
        accelerator.print(f"✅ DrealSR datasets loaded successfully")
    else:
        accelerator.print(f"⚠️  DrealSR data root not found: {test_data_root}")

    if weather_test_data_root is not None and os.path.exists(weather_test_data_root):
        weather_datasets = create_multi_weather_datasets(
            lvlm_model=lvlm_model,
            tokenizer=lvlm_tokenizer,
            data_root=weather_test_data_root,
            weather_types=weather_test_types,
            output_dir=None,
            transform=transform,
            prompter=prompter,
            dataset_type=dataset_type,
            min_pixels=args.dataset_config.min_pixels,
            max_pixels=args.dataset_config.max_pixels,
            mask_weight_type=args.training_config.mask_weight_type,
            siglip_processor=siglip_processor,
            seed=42,
            resize_mode="none",
            target_size=512,
            naming_patterns={
                "weather1": {"input_suffix": "_rain","gt_suffix": "_clean"},
                "weather2": {"input_suffix": "_input","gt_suffix": "_gt"},
                "Snow100K-L": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-S": {"input_suffix": "","gt_suffix": ""},
                "Rain100L": {"input_suffix": "","gt_suffix": ""},
                "RainTrainL": {"input_suffix": "","gt_suffix": ""}
            }
        )
        test_datasets_dict["Weather"] = weather_datasets
        accelerator.print(f"✅ Enhanced Weather datasets (including Snow100K) loaded successfully")
    else:
        accelerator.print(f"⚠️  Weather data root not found: {weather_test_data_root}")

    if realsr_test_data_root is not None and os.path.exists(realsr_test_data_root):
        realsr_datasets = create_multi_realsr_datasets(
            lvlm_model=lvlm_model,
            tokenizer=lvlm_tokenizer,
            data_root=realsr_test_data_root,
            camera_types=realsr_camera_types,
            scale_factors=realsr_scale_factors,
            output_dir=None,
            transform=transform,
            prompter=prompter,
            dataset_type=dataset_type,
            min_pixels=args.dataset_config.min_pixels,
            max_pixels=args.dataset_config.max_pixels,
            mask_weight_type=args.training_config.mask_weight_type,
            siglip_processor=siglip_processor,
            seed=42,
            resize_mode="none",
            target_size=512,
        )
        test_datasets_dict["RealSR"] = realsr_datasets
        accelerator.print(f"✅ RealSR datasets loaded successfully")
    else:
        accelerator.print(f"⚠️  RealSR data root not found: {realsr_test_data_root}")

    if generic_test_data_root is not None and os.path.exists(generic_test_data_root):
        accelerator.print(f"Loading Generic test datasets from: {generic_test_data_root}")
        generic_datasets = create_multi_generic_datasets(
            lvlm_model=lvlm_model,
            tokenizer=lvlm_tokenizer,
            data_root=generic_test_data_root,
            dataset_types=generic_dataset_types,
            output_dir=None,
            target_size=512,
            transform=transform,
            prompter=prompter,
            dataset_type=dataset_type,
            min_pixels=args.dataset_config.min_pixels,
            max_pixels=args.dataset_config.max_pixels,
            mask_weight_type=args.training_config.mask_weight_type,
            siglip_processor=siglip_processor,
            seed=42,
            resize_mode="none",
        )
        test_datasets_dict["Generic"] = generic_datasets
        accelerator.print(f"✅ Generic datasets loaded successfully")
    else:
        accelerator.print(f"⚠️  Generic data root not found: {generic_test_data_root}")

    if test_datasets_dict:
        accelerator.print(f"\n🎯 Test datasets summary:")
        for dataset_type_name, datasets_obj in test_datasets_dict.items():
            if hasattr(datasets_obj, 'print_summary'):
                datasets_obj.print_summary()
            elif hasattr(datasets_obj, 'get_dataset_names'):
                dataset_names = datasets_obj.get_dataset_names() if hasattr(datasets_obj, 'get_dataset_names') else list(datasets_obj.get_all_datasets().keys())
                accelerator.print(f"  {dataset_type_name}: {dataset_names}")
    else:
        accelerator.print("⚠️  No test datasets loaded. Validation will be skipped during training.")
    
    image_token = SPACIAL_TOKEN[dataset_type]['image_token']
    image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
    image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']

    cfg_set(lvlm_model.config, 'image_token_id', lvlm_tokenizer.convert_tokens_to_ids(image_token))
    cfg_set(lvlm_model.config, 'image_begin_token_id', lvlm_tokenizer.convert_tokens_to_ids(image_begin_token))
    cfg_set(lvlm_model.config, 'image_end_token_id', lvlm_tokenizer.convert_tokens_to_ids(image_end_token))


    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (args.training_config.num_train_epochs * num_update_steps_per_epoch)
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.training_config.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.training_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
            num_cycles=args.training_config.lr_num_cycles,
            power=args.training_config.lr_power,
        )

    lvlm_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(lvlm_model, optimizer, lr_scheduler, train_dataloader)
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder_cls_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_cls_two.to(accelerator.device, dtype=weight_dtype)

    empty_t5_prompt_embeds, empty_pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt="",
        max_sequence_length=256,
        device=accelerator.device,
        num_images_per_prompt=1,
    )
    empty_pooled_prompt_embeds = empty_pooled_prompt_embeds.repeat(args.dataset_config.batch_size, 1)
    if args.training_config.drop_t5_rate == 1.0:
        del text_encoders, text_encoder_cls_one, text_encoder_cls_two
        text_encoders = None
        free_memory()
    free_memory()
    if use_gan and D is not None:
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            D.to(accelerator.device)
        else:
            D, opt_D = accelerator.prepare(D, opt_D)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training_config.max_train_steps = (args.training_config.num_train_steps if hasattr(args.training_config, "num_train_steps") and args.training_config.num_train_steps is not None
                                                else args.training_config.num_train_epochs * num_update_steps_per_epoch)
    args.training_config.num_train_epochs = math.ceil(args.training_config.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        args.dataset_config.batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps
    )
    accelerator.print("***** Running Enhanced HYPIR-style training with Quality Metrics *****")
    accelerator.print(f"  Num examples = {len(train_dataloader)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.training_config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.dataset_config.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.training_config.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    accelerator.print(f"  Fixed timestep = {fixed_timestep}")
    if test_datasets_dict:
        accelerator.print(f"  Test dataset types loaded: {list(test_datasets_dict.keys())}")
        accelerator.print(f"  Validation samples per scale/type: {num_validation_samples_per_scale}")
    if quality_metrics is not None:
        accelerator.print(f"  Quality metrics enabled")

    # Begin training
    global_step = initial_global_step
    first_epoch = global_step // num_update_steps_per_epoch
    if resume_checkpoint_path is not None:
        accelerator.load_state(resume_checkpoint_path)
        if use_gan and D is not None:
            disc_dir = os.path.join(resume_checkpoint_path, "discriminator")
            D_path = os.path.join(disc_dir, "D_state_dict.pth")
            optD_path = os.path.join(disc_dir, "opt_D_state_dict.pth")
            if os.path.exists(D_path):
                try:
                    state = torch.load(D_path, map_location="cpu")
                    accelerator.unwrap_model(D).load_state_dict(state, strict=False)
                    accelerator.print(f"[Resume] Loaded discriminator state from {D_path}")
                except Exception as e:
                    accelerator.print(f"[Resume] Warning: failed to load D_state_dict.pth: {e}")
            if os.path.exists(optD_path) and (opt_D is not None):
                try:
                    opt_D.load_state_dict(torch.load(optD_path, map_location="cpu"))
                    accelerator.print(f"[Resume] Loaded discriminator optimizer state from {optD_path}")
                except Exception as e:
                    accelerator.print(f"[Resume] Warning: failed to load opt_D_state_dict.pth: {e}")
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(0, args.training_config.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    def vae_encode(x):
        with torch.no_grad():
            model_input = vae.encode(x).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        return model_input
    
    def encode_vae_and_pack_fn(x_input):
        x_input = x_input.to(dtype=torch.float32)
        x_input = x_input.to(device=vae.device, dtype=vae.dtype, non_blocking=True)
        x = vae.encode(x_input).latent_dist.sample()
        x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        x = x.to(dtype=weight_dtype)
        x_pack = FluxPipeline._pack_latents(
            x,
            batch_size=x.shape[0],
            num_channels_latents=x.shape[1],
            height=x.shape[2],
            width=x.shape[3],
        )
        return x, x_pack

    latent_image_ids_dict = {}

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        lvlm_model.train()
        if use_gan and D is not None:
            D.train().requires_grad_(True)

        for step, raw_batch in enumerate(train_dataloader):

            # Build batch on all ranks
            built = build_training_batch_from_raw_samples(
                raw_batch=raw_batch,
                lvlm_model=lvlm_model,
                tokenizer=lvlm_tokenizer,
                processor=processor,
                prompter=prompter,
                image_token=image_token,
                image_begin_token=SPACIAL_TOKEN[dataset_type]['image_begin_token'],
                image_end_token=SPACIAL_TOKEN[dataset_type]['image_end_token'],
                gen_cfg=dict(
                    gen_max_pixels=args.model_config.get("gen_max_pixels", 512*512),
                    generation_max_new_tokens=args.model_config.get("generation_max_new_tokens", 512),
                    generation_temperature=args.model_config.get("generation_temperature", 0.7),
                    generation_top_p=args.model_config.get("generation_top_p", 0.9),
                    generation_do_sample=args.model_config.get("generation_do_sample", True),
                ),
                repick_cfg=dict(
                    disable_superres=True,
                    repick_if_superres=True,
                    selective_repick=True,
                    repick_temperature=0.2,
                    repick_top_p=0.5,
                ),
                append_global_feature_hints=True,
                use_expert_prompt=True,
                mask_weight_type=args.training_config.mask_weight_type,
                max_pixels=args.dataset_config.max_pixels,
                min_pixels=args.dataset_config.min_pixels,
                siglip_processor=siglip_processor,
                siglip_input_size=getattr(siglip_model.config, "image_size", None) if siglip_model is not None else None,
            )

            # Compute a globally consistent skip flag:
            # if any rank is missing LQ data, all ranks skip the optimization step
            skip_local = built["vae_pixel_values"] is None
            skip_any = accelerator.reduce(torch.tensor(1 if skip_local else 0, device=accelerator.device), reduction="max").item() > 0
            with accelerator.accumulate(lvlm_model):
                # Initialize defaults so skipped steps still have loggable scalars
                loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_l2 = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_lpips = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_adv = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_D_real = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_D_fake = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                do_train_this_step = not skip_any

                if do_train_this_step:
                    # Assemble tensors
                    gt_tensor_list: List[torch.Tensor] = raw_batch["gt_tensor_list"]
                    generated_image = torch.stack(gt_tensor_list, dim=0).to(accelerator.device, dtype=vae.dtype, non_blocking=True)  # BCHW

                    vae_pixel_values = built["vae_pixel_values"].to(accelerator.device, dtype=vae.dtype, non_blocking=True)
                    target_hw = (512, 512)
                    if generated_image.shape[-2:] != target_hw:
                        generated_image = F.interpolate(generated_image, size=target_hw, mode="bicubic", align_corners=False)
                    if built["vae_pixel_values"] is not None and built["vae_pixel_values"].shape[-2:] != target_hw:
                        vae_pixel_values = F.interpolate(vae_pixel_values, size=target_hw, mode="bicubic", align_corners=False)

                    input_ids = built["input_ids"].to(accelerator.device, non_blocking=True)
                    attention_mask = built["attention_mask"].to(accelerator.device, non_blocking=True)
                    pixel_values = built["pixel_values"].to(accelerator.device, dtype=weight_dtype, non_blocking=True) if built["pixel_values"] is not None else None
                    image_position = built["image_position"]
                    image_grid_thw = built["image_grid_thw"].to(accelerator.device, non_blocking=True) if built["image_grid_thw"] is not None else None
                    prompts = built["prompts"]

                    # VAE hidden states
                    vae_hidden_states = None
                    if vae is not None and len(vae_pixel_values) > 0:
                        with torch.no_grad():
                            _, vae_hidden_states = encode_vae_and_pack_fn(vae_pixel_values)

                    # SigLIP hidden states
                    siglip_hidden_states = None
                    if (siglip_model is not None) and (built.get("siglip_pixel_values") is not None):
                        siglip_pixels = built["siglip_pixel_values"].to(device=accelerator.device, dtype=next(siglip_model.parameters()).dtype, non_blocking=True)
                        with torch.no_grad():
                            siglip_hidden_states = siglip_model(siglip_pixels).last_hidden_state

                    # Encode LQ/GT latents
                    lq_latents = vae_encode(vae_pixel_values)
                    gt_latents = vae_encode(generated_image)
                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    bsz = lq_latents.shape[0]

                    # Latent image IDs (cached by shape)
                    if lq_latents.shape in latent_image_ids_dict:
                        latent_image_ids = latent_image_ids_dict[lq_latents.shape]
                    else:
                        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                            lq_latents.shape[0], lq_latents.shape[2] // 2, lq_latents.shape[3] // 2, accelerator.device, weight_dtype
                        )
                        latent_image_ids_dict[lq_latents.shape] = latent_image_ids

                    # Fixed timestep
                    timesteps = torch.full((bsz,), fixed_timestep, dtype=torch.long, device=accelerator.device)
                    timestep_normalized = (timesteps.float() / 1000.0).to(weight_dtype)

                    # Pack LQ latents
                    packed_lq_latents = FluxPipeline._pack_latents(
                        lq_latents, batch_size=lq_latents.shape[0], num_channels_latents=lq_latents.shape[1],
                        height=lq_latents.shape[2], width=lq_latents.shape[3],
                    )
                    guidance = torch.full((bsz,), fill_value=args.model_config.guidance_scale, device=accelerator.device, dtype=weight_dtype)

                    # Encode T5 prompt using t5_texts from the legacy pipeline
                    # (falls back to empty embeddings if text_encoders is None or drop_t5_rate==1.0)
                    if (text_encoders is not None) and (built.get("t5_texts") is not None):
                        t5_prompt_embeds, pooled_proj = encode_prompt(
                            text_encoders,
                            tokenizers,
                            prompt=built["t5_texts"],       # list[str], aligned with batch
                            max_sequence_length=256,
                            device=accelerator.device,
                            num_images_per_prompt=1,
                        )
                    else:
                        t5_prompt_embeds = None
                        pooled_proj = empty_pooled_prompt_embeds[:bsz].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    model_pred = lvlm_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_position=image_position,
                        image_grid_thw=image_grid_thw,
                        output_type="denoise_model_pred",
                        only_use_t5=args.model_config.only_use_t5,
                        ref_features_for_vlm=None,
                        vlm_residual_image_factor=args.model_config.vlm_residual_image_factor,
                        siglip_hidden_states=siglip_hidden_states,
                        vae_hidden_states=vae_hidden_states,
                        denoiser_kwargs={
                            "prefix_prompt_embeds": t5_prompt_embeds,
                            "hidden_states": packed_lq_latents.to(weight_dtype),
                            "timestep": timestep_normalized,
                            "guidance": guidance,
                            "pooled_projections": pooled_proj,
                            "img_ids": latent_image_ids,
                            "joint_attention_kwargs": {},
                        },
                    )
                    model_pred = FluxPipeline._unpack_latents(model_pred, height=lq_latents.shape[2]*vae_scale_factor, width=lq_latents.shape[3]*vae_scale_factor, vae_scale_factor=vae_scale_factor)
                    final_latents = (model_pred / vae.config.scaling_factor) + vae.config.shift_factor
                    x_hat = vae.decode(final_latents.to(vae.dtype)).sample.float()  # [-1,1]

                    # Compute losses
                    loss_l2 = F.mse_loss(x_hat, generated_image) * lambda_l2
                    loss_lpips = net_lpips(x_hat, generated_image).mean() * lambda_lpips
                    loss_adv = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                    if use_gan and D is not None:
                        D.eval().requires_grad_(False)
                        loss_adv = D(x_hat.float(), for_G=True).mean() * lambda_gan
                    
                    # Collect and add frequency-domain loss
                    L_freq_total = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                    # L_freq values are appended per layer inside FAPEIRDenoiseTower.forward -> tower._freq_losses
                    tower = getattr(lvlm_model, "denoise_tower", None)
                    if tower is not None and hasattr(tower, "_freq_losses") and len(tower._freq_losses) > 0:
                        # Sum all sub-terms (already computed with correct dtype/device in their respective layers)
                        L_freq_total = torch.stack([lf.to(accelerator.device, dtype=weight_dtype) for lf in tower._freq_losses]).sum()
                        # Clear cache to prevent cross-step accumulation
                        tower._freq_losses.clear()

                    loss = loss_l2 + loss_lpips + loss_adv + lambda_freq * L_freq_total

                    # Backpropagation and optimizer step
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lvlm_model.parameters(), args.training_config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Gather metrics (always done even on skipped steps to keep collective counts consistent)
                avg_loss_list = accelerator.gather_for_metrics(loss.detach())

                # Discriminator D: train only on sync steps (sync_gradients=True) when the step was not skipped
                if use_gan and D is not None:
                    if accelerator.sync_gradients:   # train D only on aligned steps
                        with torch.no_grad():
                            x_hat_detached = x_hat.detach()
                        D.train().requires_grad_(True)
                        loss_D_real, _ = D(generated_image.float(), for_real=True, return_logits=True)
                        loss_D_fake, _ = D(x_hat_detached.float(),  for_real=False, return_logits=True)
                        loss_D = loss_D_real.mean() + loss_D_fake.mean()
                        accelerator.backward(loss_D)
                        accelerator.clip_grad_norm_(D.parameters(), args.training_config.max_grad_norm)
                        opt_D.step()
                        opt_D.zero_grad()
                        D.eval().requires_grad_(False)

                # Checkpointing / validation / logging
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED) and global_step % args.training_config.checkpointing_steps == 0:
                        if args.training_config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.training_config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if (len(checkpoints) >= args.training_config.checkpoints_total_limit):
                                num_to_remove = (len(checkpoints) - args.training_config.checkpoints_total_limit + 1)
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                accelerator.print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.training_config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.training_config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if args.model_config.only_tune_mlp2 or args.model_config.with_tune_mlp2:
                            keys_to_match = ['denoise_tower.denoise_projector']
                            weight_mlp2 = get_mm_adapter_state_maybe_zero_3(lvlm_model.named_parameters(), keys_to_match)
                            weight_mlp2 = {k.replace('module.', ''): v for k, v in weight_mlp2.items()}
                            torch.save(weight_mlp2, os.path.join(save_path, 'denoise_projector.bin'))
                        if args.model_config.only_tune_mlp3 or args.model_config.with_tune_mlp3:
                            keys_to_match = ['denoise_tower.vae_projector']
                            weight_mlp3 = get_mm_adapter_state_maybe_zero_3(lvlm_model.named_parameters(), keys_to_match)
                            weight_mlp3 = {k.replace('module.', ''): v for k, v in weight_mlp3.items()}
                            torch.save(weight_mlp3, os.path.join(save_path, 'vae_projector.bin'))
                        if args.model_config.only_tune_siglip_mlp or args.model_config.with_tune_siglip_mlp:
                            keys_to_match = ['denoise_tower.siglip_projector']
                            weight_siglip_mlp = get_mm_adapter_state_maybe_zero_3(lvlm_model.named_parameters(), keys_to_match)
                            weight_siglip_mlp = {k.replace('module.', ''): v for k, v in weight_siglip_mlp.items()}
                            torch.save(weight_siglip_mlp, os.path.join(save_path, 'siglip_projector.bin'))
                        accelerator.print(f"Saved state to {save_path}")

                    if hasattr(args.training_config, 'validation_steps') and args.training_config.validation_steps is not None:
                        if (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED) and global_step % args.training_config.validation_steps == 0:
                            if test_datasets_dict:
                                try:
                                    avg_psnr, avg_ssim = run_validation(
                                        args=args,
                                        lvlm_model=lvlm_model,
                                        test_datasets_dict=test_datasets_dict,
                                        vae=vae,
                                        tokenizers=[tokenizer_one, tokenizer_two],
                                        text_encoders=text_encoders if text_encoders is not None else None,
                                        empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                                        noise_scheduler_copy=noise_scheduler,
                                        accelerator=accelerator,
                                        weight_dtype=weight_dtype,
                                        global_step=global_step,
                                        output_dir=args.training_config.output_dir,
                                        flux_pipeline=None,
                                        siglip_model=siglip_model,
                                        num_validation_samples=num_validation_samples_per_scale,
                                        collator_tokenizer=lvlm_tokenizer,
                                        quality_metrics=quality_metrics
                                    )
                                    accelerator.print(f"Enhanced validation with quality metrics at step {global_step}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
                                except Exception as e:
                                    accelerator.print(f"Validation failed at step {global_step}: {str(e)}")
                                    import traceback
                                    accelerator.print(f"Traceback: {traceback.format_exc()}")
                            else:
                                accelerator.print(f"Skipping validation at step {global_step}: No test datasets available")

                # Logging
                log_interval = 1
                if global_step % log_interval == 0:
                    logs = {
                        "loss": float(avg_loss_list.mean().detach().item()) if avg_loss_list is not None else float(loss.detach().item()),
                        "lr": (lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]),
                        "loss_l2": float(loss_l2.detach().item()),
                        "loss_lpips": float(loss_lpips.detach().item()),
                        "fixed_timestep": fixed_timestep,
                        "skipped": int(skip_any),
                        # Frequency-domain loss
                        "loss_freq": float(L_freq_total.detach().item()),
                        "lambda_freq": lambda_freq,
                    }
                    if use_gan:
                        # May be 0 if step was skipped or D was not trained this step
                        logs["loss_g_adv"] = float(loss_adv.detach().item())
                        if 'loss_D' in locals():
                            logs["loss_D_real"] = float(loss_D_real.detach().item())
                            logs["loss_D_fake"] = float(loss_D_fake.detach().item())
                            logs["loss_D"] = float(loss_D.detach().item()) if 'loss_D' in locals() else 0.0
                        else:
                            logs["loss_D_real"] = float(loss_D_real.detach().item())
                            logs["loss_D_fake"] = float(loss_D_fake.detach().item())
                            logs["loss_D"] = 0.0
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if global_step >= args.training_config.max_train_steps:
                    break

    accelerator.end_training()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(FAPEIRTrainingDenoiseConfig)
    conf = OmegaConf.merge(schema, config)
    main(conf, attn_implementation='flash_attention_2')