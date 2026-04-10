import os
import time
import json
import pyiqa
import numpy as np
from PIL import Image
from datetime import datetime
from collections.abc import Sequence
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from fapeir.utils.flux_pipeline import FluxPipeline
from fapeir.dataset.test_data_collator import ValidationCollator
from fapeir.utils.denoiser_prompt_embedding_flux import encode_prompt

def slice_sample(x, i: int, B: int):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.ndim >= 1 and x.size(0) == B:
            return x[i:i+1]
        return x
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        if len(x) == B:
            xi = x[i]
            if isinstance(xi, torch.Tensor) and (xi.ndim in (1, 2, 3)):
                return xi.unsqueeze(0)
            return xi
        return x
    if isinstance(x, dict):
        return {k: slice_sample(v, i, B) for k, v in x.items()}
    return x


# ========= Configurable: whether to rebuild LVLM input from the 5-line prompt (full training alignment) =========
USE_PROMPT_TO_REBUILD_IDS = True  # Default False; set True for strict alignment with training

# ========= Required for 5-line task inference: expert prompt & feature thresholds =========
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
ALLOWED_TASKS_PARSE = set(["deraining", "desnowing", "dehazing", "deblur", "denoise", "light_enhancement", "super_resolution"])

# ========= Lightweight image statistics (consistent with training) =========

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _mean3(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="reflect")
    s = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    )
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
    F = np.fft.fft2(g); ps = np.abs(F) ** 2; ps = np.fft.fftshift(ps)
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
    # Blob counting (downscale to <= 256 on the longest side)
    w, h = img.size
    if max(w, h) > 256:
        scale = 256 / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img2 = img.resize(new_size, Image.LANCZOS)
    else:
        img2 = img
    g2 = np.asarray(img2.convert("L"), dtype=np.float32) / 255.0
    from scipy import ndimage
    bright = (g2 > 0.78).astype(np.uint8)
    labeled, num = ndimage.label(bright)
    sizes = ndimage.sum(bright, labeled, range(1, num + 1))
    small_blobs = int(np.sum((sizes >= 3) & (sizes <= 200)))
    return {"small_blobs": small_blobs, "anisotropy": anisotropy,
            "hint": f"Snow: blobs={small_blobs}, aniso={anisotropy:.2f}"}

def compute_noise_features(img: Image.Image) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g); grad_mag = np.hypot(gx, gy)
    thr = float(np.quantile(grad_mag, 0.25)); flat = grad_mag < thr
    g_blur = _mean3(g); resid = g - g_blur; r_flat = resid[flat]
    med = np.median(r_flat); noise_mad = float(np.median(np.abs(r_flat - med)))
    ycbcr = np.asarray(img.convert("YCbCr"), dtype=np.float32) / 255.0
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]; chroma_std = float(0.5 * (cb[flat].std() + cr[flat].std()))
    s1 = _sigmoid(50.0 * (noise_mad - NOISE_MAD_THR))
    s2 = _sigmoid(50.0 * (chroma_std - NOISE_CHROMA_THR))
    noise_score = float(0.6 * s1 + 0.4 * s2)
    return {"noise_mad": noise_mad, "chroma_std": chroma_std, "noise_score": noise_score,
            "hint": f"Noise: mad={noise_mad:.4f}, chroma={chroma_std:.4f}, score={noise_score:.2f}"}

def compute_blur_features(img: Image.Image) -> Dict[str, Any]:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    p = np.pad(g, 1, mode="reflect")
    lap = (k[0, 0]*p[:-2, :-2] + k[0, 1]*p[:-2, 1:-1] + k[0, 2]*p[:-2, 2:] +
           k[1, 0]*p[1:-1, :-2] + k[1, 1]*p[1:-1, 1:-1] + k[1, 2]*p[1:-1, 2:] +
           k[2, 0]*p[2:, :-2] + k[2, 1]*p[2:, 1:-1] + k[2, 2]*p[2:, 2:])
    lap_var = float(lap.var())
    F = np.fft.fft2(g); ps = np.abs(F) ** 2; ps = np.fft.fftshift(ps)
    h, w = ps.shape; yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2); rmax = rr.max() + 1e-6
    outer_mask = (rr > 0.35 * rmax) & (rr <= 0.9 * rmax)
    inner_mask = (rr <= 0.35 * rmax)
    outer_vals = ps[outer_mask]; inner_vals = ps[inner_mask]
    hf_energy = float((outer_vals.mean() if outer_vals.size else ps.mean()) /
                      ((inner_vals.mean() if inner_vals.size else ps.mean()) + 1e-6))
    g_s = _mean3(_mean3(g)); gy, gx = np.gradient(g_s)
    mag = np.hypot(gx, gy); grad95 = float(np.quantile(mag, 0.95))
    return {"lap_var": lap_var, "hf_energy": hf_energy, "grad95": grad95,
            "hint": f"Blur: lapVar={lap_var:.3f}, hf={hf_energy:.3f}, grad95={grad95:.3f}"}

def compute_haze_features(img: Image.Image) -> Dict[str, Any]:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=-1)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    dark = np.minimum(np.minimum(R, G), B); dark_mean = float(dark.mean())
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    sat_mean = float(hsv[..., 1].mean())
    h = Y.shape[0]; top_slice = slice(0, max(1, int(h * 0.4))); bot_slice = slice(int(h * 0.6), h)
    top_bright = float(Y[top_slice, :].mean()); bot_bright = float(Y[bot_slice, :].mean())
    depth_grad = top_bright - bot_bright
    s1 = _sigmoid(7.0 * (dark_mean - 0.33))
    s2 = _sigmoid(7.0 * (0.30 - sat_mean))
    s3 = _sigmoid(8.0 * (depth_grad - 0.03))
    haze_score = float(0.4 * s1 + 0.3 * s2 + 0.3 * s3)
    return {"haze_score": haze_score, "depth_grad": depth_grad,
            "dark_mean": dark_mean, "sat_mean": sat_mean,
            "hint": f"Haze: score={haze_score:.2f}, depth_grad={depth_grad:.3f}"}

def compute_exposure_features(img: Image.Image) -> Dict[str, Any]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
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
                          (rain["line_score"] > RAIN_LINE_THR * 1.3))
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

def _normalize_messages_for_processor(messages: List[dict]) -> List[dict]:
    norm_msgs = []
    for m in messages:
        m = dict(m)
        c = m.get("content", "")
        if isinstance(c, str):
            m["content"] = [{"type": "text", "text": c}]
        elif isinstance(c, list):
            new_list = []
            for item in c:
                if isinstance(item, dict) and "type" in item:
                    if item["type"] == "text":
                        new_list.append({"type": "text", "text": str(item.get("text", ""))})
                    elif item["type"] in ("image", "video"):
                        new_list.append(item)
                    else:
                        new_list.append({"type": "text", "text": str(item)})
                elif isinstance(item, str):
                    new_list.append({"type": "text", "text": item})
                else:
                    try:
                        if isinstance(item, Image.Image):
                            new_list.append({"type": "image", "image": item})
                        else:
                            new_list.append({"type": "text", "text": str(item)})
                    except Exception:
                        new_list.append({"type": "text", "text": str(item)})
            if not new_list: new_list = [{"type": "text", "text": ""}]
            m["content"] = new_list
        else:
            m["content"] = [{"type": "text", "text": ""}]
        norm_msgs.append(m)
    return norm_msgs

def _combine_user_text(prompt_used: str, use_expert_prompt: bool = True) -> str:
    prefix = EXPERT_USER_PROMPT if use_expert_prompt else ""
    return f"{prefix}{prompt_used}"

@torch.no_grad()
def _generate_text_once_for_validation(
    lvlm_model, processor, tokenizer, prompt_used: str, first_img_pil: Image.Image,
    generation_max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True,
    gen_max_pixels=512 * 512, use_expert_prompt=True,
):
    combined_text = _combine_user_text(prompt_used, use_expert_prompt)
    messages = [{"role": "user", "content": [{"type": "image", "image": first_img_pil},
                                              {"type": "text", "text": combined_text}]}]
    messages = _normalize_messages_for_processor(messages)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, max_pixels=gen_max_pixels,
    )
    device = next(lvlm_model.parameters()).device
    model_dtype = next(lvlm_model.parameters()).dtype
    if hasattr(inputs, "to"): inputs = inputs.to(device)
    keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "aspect_ratio_ids")
    for k in keys:
        if k in inputs and inputs[k] is not None and isinstance(inputs[k], torch.Tensor):
            if inputs[k].dtype in (torch.float16, torch.float32, torch.bfloat16):
                inputs[k] = inputs[k].to(device, dtype=model_dtype)
            else:
                inputs[k] = inputs[k].to(device)
    embed_device = getattr(lvlm_model.model.embed_tokens.weight, "device", device)
    inputs["input_ids"] = inputs["input_ids"].to(embed_device)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        inputs["attention_mask"] = inputs["attention_mask"].to(embed_device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    gen_cfg = GenerationConfig(
        max_new_tokens=generation_max_new_tokens, temperature=temperature, top_p=top_p,
        do_sample=do_sample, pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True
    )
    outputs = lvlm_model.generate(**inputs, generation_config=gen_cfg,
                                  output_type="lvlm", return_dict_in_generate=True)
    seq = outputs.sequences
    input_len = inputs["input_ids"].shape[1]
    gen_ids = seq[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, inputs  # also return inputs for optional LVLM input reconstruction

def infer_5lines_with_repicks_for_validation(
    first_img_pil: Image.Image, lvlm_model, processor, tokenizer,
    prompt_text=NEUTRAL_PROMPT, append_global_feature_hints=True, use_expert_prompt=True,
    respect_size_superres=True, disable_superres=True, repick_if_superres=True, selective_repick=True,
    repick_temperature=0.2, repick_top_p=0.5, gen_max_pixels=512 * 512,
    generation_max_new_tokens=512, generation_temperature=0.7,
    generation_top_p=0.9, generation_do_sample=True
):
    features = compute_all_features_for_infer(first_img_pil)
    feature_hint = generate_feature_hints_string(features) if append_global_feature_hints else ""
    prompt_used = f"{prompt_text}\n\n{feature_hint}" if feature_hint else prompt_text
    w, h = first_img_pil.size
    if respect_size_superres and (w < 512 and h < 512):
        text_out, rebuilt_inputs = _generate_text_once_for_validation(
            lvlm_model, processor, tokenizer, prompt_used, first_img_pil,
            generation_max_new_tokens, generation_temperature, generation_top_p,
            generation_do_sample, gen_max_pixels, use_expert_prompt
        )
        return text_out, "super_resolution", features, rebuilt_inputs
    text_out, rebuilt_inputs = _generate_text_once_for_validation(
        lvlm_model, processor, tokenizer, prompt_used, first_img_pil,
        generation_max_new_tokens, generation_temperature, generation_top_p,
        generation_do_sample, gen_max_pixels, use_expert_prompt
    )
    pred_task = parse_predicted_task(text_out)
    if disable_superres and pred_task == "super_resolution" and repick_if_superres:
        forbid = "\n\nIMPORTANT: Do not choose super_resolution. Choose from the 6 main tasks."
        text2, rebuilt2 = _generate_text_once_for_validation(
            lvlm_model, processor, tokenizer, prompt_used + forbid, first_img_pil,
            generation_max_new_tokens, repick_temperature, repick_top_p,
            True, gen_max_pixels, use_expert_prompt
        )
        pred2 = parse_predicted_task(text2)
        if pred2 and pred2 != "super_resolution":
            text_out, pred_task, rebuilt_inputs = text2, pred2, rebuilt2
    if selective_repick and pred_task:
        suggested = check_strong_conflict(pred_task, features)
        if suggested:
            allowed_str = " | ".join(suggested)
            forbid2 = f"\n\nIMPORTANT: Based on strong evidence, choose ONE from: {allowed_str}"
            text3, rebuilt3 = _generate_text_once_for_validation(
                lvlm_model, processor, tokenizer, prompt_used + forbid2, first_img_pil,
                generation_max_new_tokens, repick_temperature, repick_top_p,
                True, gen_max_pixels, use_expert_prompt
            )
            pred3 = parse_predicted_task(text3)
            if pred3 in suggested:
                text_out, pred_task, rebuilt_inputs = text3, pred3, rebuilt3
    return text_out, pred_task, features, rebuilt_inputs


def _init_iqa_metrics(device: torch.device) -> Dict[str, Any]:
    """
    Initialize all pyiqa metrics once and return as a dict.
    Full-reference metrics: PSNR, SSIM, LPIPS, DISTS
    No-reference metrics:   CLIPIQA, NIQE, MUSIQ, MANIQA, TOPIQA
    """
    return {
        'PSNR':    pyiqa.create_metric('psnr',         test_y_channel=True, color_space='ycbcr').to(device),
        'SSIM':    pyiqa.create_metric('ssim',         test_y_channel=True, color_space='ycbcr').to(device),
        'LPIPS':   pyiqa.create_metric('lpips',        device=device),
        'DISTS':   pyiqa.create_metric('dists',        device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa',      device=device),
        'NIQE':    pyiqa.create_metric('niqe',         device=device),
        'MUSIQ':   pyiqa.create_metric('musiq',        device=device),
        'MANIQA':  pyiqa.create_metric('maniqa-pipal', device=device),
        'TOPIQA':  pyiqa.create_metric('topiq_nr',     device=device),
    }


# No-reference metrics that receive only the SR tensor (no GT needed)
_NR_METRICS = {'CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'TOPIQA'}


def _compute_iqa_metrics(
    iqa_metrics: Dict[str, Any],
    sr_tensor: torch.Tensor,   # (1, 3, H, W), float32, [0, 1]
    gt_tensor: torch.Tensor,   # (1, 3, H, W), float32, [0, 1]
) -> Dict[str, float]:
    """
    Compute all IQA metrics for a single image pair.
    Full-reference metrics use both sr_tensor and gt_tensor.
    No-reference metrics use only sr_tensor.
    """
    results = {}
    with torch.no_grad():
        for name, metric in iqa_metrics.items():
            try:
                if name in _NR_METRICS:
                    results[name] = float(metric(sr_tensor).item())
                else:
                    results[name] = float(metric(sr_tensor, gt_tensor).item())
            except Exception as e:
                results[name] = float('nan')
    return results


@torch.no_grad()
def run_validation(
    args, lvlm_model, test_datasets_dict, vae, tokenizers, text_encoders,
    empty_pooled_prompt_embeds, noise_scheduler_copy, accelerator, weight_dtype,
    global_step, output_dir, flux_pipeline=None, siglip_model=None,
    num_validation_samples=100000, collator_tokenizer=None, quality_metrics=None
):
    if not accelerator.is_main_process:
        quality_metrics = None
    if collator_tokenizer.pad_token_id is None:
        if collator_tokenizer.eos_token is not None:
            collator_tokenizer.pad_token = collator_tokenizer.eos_token
        else:
            collator_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    accelerator.print(f"Running validation with pyiqa metrics on test datasets at step {global_step}")
    lvlm_model.eval()

    validation_dir = os.path.join(output_dir, f"validation_step_{global_step}")
    os.makedirs(validation_dir, exist_ok=True)

    # Initialize pyiqa metrics once for the entire validation run
    iqa_metrics = _init_iqa_metrics(accelerator.device)
    accelerator.print("pyiqa metrics initialized: " + ", ".join(iqa_metrics.keys()))

    total_results = {}
    all_results = []
    all_sub_results = {}
    global_timing_stats = []
    global_quality_stats = {}

    def _encode_vae_and_pack(x_input: torch.Tensor):
        x_input = x_input.to(dtype=torch.float32, device=vae.device)
        z = vae.encode(x_input).latent_dist.sample()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
        z = z.to(dtype=weight_dtype)
        z_pack = FluxPipeline._pack_latents(
            z, batch_size=z.shape[0], num_channels_latents=z.shape[1],
            height=z.shape[2], width=z.shape[3]
        )
        return z, z_pack

    latent_image_ids_cache = {}

    for dataset_type, datasets_obj in test_datasets_dict.items():
        accelerator.print(f"\n🔥 Validating on {dataset_type} datasets")
        dataset_type_dir = os.path.join(validation_dir, dataset_type)
        os.makedirs(dataset_type_dir, exist_ok=True)
        dataset_type_results = {}
        dataset_type_samples = 0
        dataset_type_psnr = 0.0
        dataset_type_ssim = 0.0
        dataset_quality_stats = {}

        if dataset_type == "DrealSR":
            for scale_name, dataset in datasets_obj:
                accelerator.print(f"  Validating on {scale_name} dataset ({len(dataset)} samples)")
                scale_dir = os.path.join(dataset_type_dir, scale_name)
                gen_dir  = os.path.join(scale_dir, "generated_images")
                tgt_dir  = os.path.join(scale_dir, "target_images")
                lr_dir   = os.path.join(scale_dir, "lr_images")
                cmp_dir  = os.path.join(scale_dir, "comparisons")
                for d in (gen_dir, tgt_dir, lr_dir, cmp_dir):
                    os.makedirs(d, exist_ok=True)
                max_samples = min(num_validation_samples, len(dataset))
                scale_psnr, scale_ssim, scale_processed, scale_results, scale_quality_stats = validate_single_dataset(
                    dataset=dataset,
                    dataset_name=f"{dataset_type}_{scale_name}",
                    args=args,
                    lvlm_model=lvlm_model,
                    vae=vae,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                    siglip_model=siglip_model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    collator_tokenizer=collator_tokenizer,
                    latent_image_ids_cache=latent_image_ids_cache,
                    _encode_vae_and_pack=_encode_vae_and_pack,
                    max_samples=max_samples,
                    output_dirs={'gen_dir': gen_dir, 'tgt_dir': tgt_dir,
                                 'lr_dir': lr_dir, 'cmp_dir': cmp_dir},
                    iqa_metrics=iqa_metrics,
                    quality_metrics=quality_metrics,
                )
                dataset_type_results[scale_name] = {
                    "num_samples": scale_processed,
                    "avg_psnr": float(scale_psnr / scale_processed) if scale_processed > 0 else 0.0,
                    "avg_ssim": float(scale_ssim / scale_processed) if scale_processed > 0 else 0.0,
                    "detailed_results": scale_results,
                    "quality_metrics": scale_quality_stats,
                }
                all_sub_results[scale_name] = dataset_type_results[scale_name]
                dataset_type_samples += scale_processed
                dataset_type_psnr += scale_psnr
                dataset_type_ssim += scale_ssim
                all_results.extend(scale_results)
                for metric_name, values in scale_quality_stats.items():
                    dataset_quality_stats.setdefault(metric_name, []).extend(values)
                quality_summary = _format_quality_summary(scale_quality_stats)
                accelerator.print(
                    f"    ✅ {scale_name}: PSNR={scale_psnr/scale_processed:.4f}, "
                    f"SSIM={scale_ssim/scale_processed:.4f}{quality_summary} ({scale_processed} samples)"
                )

        elif dataset_type == "Weather":
            for weather_name, dataset in datasets_obj:
                accelerator.print(f"  Validating on {weather_name} dataset ({len(dataset)} samples)")
                accelerator.print(f"    Weather dataset target_size: {dataset.target_size}")
                accelerator.print(f"    Weather dataset resize_mode: {dataset.resize_mode}")
                weather_dir = os.path.join(dataset_type_dir, weather_name)
                gen_dir   = os.path.join(weather_dir, "generated_images")
                tgt_dir   = os.path.join(weather_dir, "target_images")
                input_dir = os.path.join(weather_dir, "input_images")
                cmp_dir   = os.path.join(weather_dir, "comparisons")
                for d in (gen_dir, tgt_dir, input_dir, cmp_dir):
                    os.makedirs(d, exist_ok=True)
                max_samples = min(num_validation_samples, len(dataset))
                weather_psnr, weather_ssim, weather_processed, weather_results, weather_quality_stats = validate_single_dataset(
                    dataset=dataset,
                    dataset_name=f"{dataset_type}_{weather_name}",
                    args=args,
                    lvlm_model=lvlm_model,
                    vae=vae,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                    siglip_model=siglip_model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    collator_tokenizer=collator_tokenizer,
                    latent_image_ids_cache=latent_image_ids_cache,
                    _encode_vae_and_pack=_encode_vae_and_pack,
                    max_samples=max_samples,
                    output_dirs={'gen_dir': gen_dir, 'tgt_dir': tgt_dir,
                                 'lr_dir': input_dir, 'cmp_dir': cmp_dir},
                    is_weather_dataset=True,
                    iqa_metrics=iqa_metrics,
                    quality_metrics=quality_metrics,
                )
                dataset_type_results[weather_name] = {
                    "num_samples": weather_processed,
                    "avg_psnr": float(weather_psnr / weather_processed) if weather_processed > 0 else 0.0,
                    "avg_ssim": float(weather_ssim / weather_processed) if weather_processed > 0 else 0.0,
                    "detailed_results": weather_results,
                    "quality_metrics": weather_quality_stats,
                }
                all_sub_results[weather_name] = dataset_type_results[weather_name]
                dataset_type_samples += weather_processed
                dataset_type_psnr += weather_psnr
                dataset_type_ssim += weather_ssim
                all_results.extend(weather_results)
                for metric_name, values in weather_quality_stats.items():
                    dataset_quality_stats.setdefault(metric_name, []).extend(values)
                quality_summary = _format_quality_summary(weather_quality_stats)
                accelerator.print(
                    f"    ✅ {weather_name}: PSNR={weather_psnr/weather_processed:.4f}, "
                    f"SSIM={weather_ssim/weather_processed:.4f}{quality_summary} ({weather_processed} samples)"
                )

        elif dataset_type == "RealSR":
            for realsr_name, dataset in datasets_obj:
                accelerator.print(f"  Validating on {realsr_name} dataset ({len(dataset)} samples)")
                accelerator.print(f"    RealSR dataset target_size: {dataset.target_size}")
                accelerator.print(f"    RealSR dataset resize_mode: {dataset.resize_mode}")
                realsr_dir = os.path.join(dataset_type_dir, realsr_name)
                gen_dir = os.path.join(realsr_dir, "generated_images")
                tgt_dir = os.path.join(realsr_dir, "target_images")
                lr_dir  = os.path.join(realsr_dir, "lr_images")
                cmp_dir = os.path.join(realsr_dir, "comparisons")
                for d in (gen_dir, tgt_dir, lr_dir, cmp_dir):
                    os.makedirs(d, exist_ok=True)
                max_samples = min(num_validation_samples, len(dataset))
                realsr_psnr, realsr_ssim, realsr_processed, realsr_results, realsr_quality_stats = validate_single_dataset(
                    dataset=dataset,
                    dataset_name=f"{dataset_type}_{realsr_name}",
                    args=args,
                    lvlm_model=lvlm_model,
                    vae=vae,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                    siglip_model=siglip_model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    collator_tokenizer=collator_tokenizer,
                    latent_image_ids_cache=latent_image_ids_cache,
                    _encode_vae_and_pack=_encode_vae_and_pack,
                    max_samples=max_samples,
                    output_dirs={'gen_dir': gen_dir, 'tgt_dir': tgt_dir,
                                 'lr_dir': lr_dir, 'cmp_dir': cmp_dir},
                    is_realsr_dataset=True,
                    iqa_metrics=iqa_metrics,
                    quality_metrics=quality_metrics,
                )
                dataset_type_results[realsr_name] = {
                    "num_samples": realsr_processed,
                    "avg_psnr": float(realsr_psnr / realsr_processed) if realsr_processed > 0 else 0.0,
                    "avg_ssim": float(realsr_ssim / realsr_processed) if realsr_processed > 0 else 0.0,
                    "detailed_results": realsr_results,
                    "quality_metrics": realsr_quality_stats,
                }
                all_sub_results[realsr_name] = dataset_type_results[realsr_name]
                dataset_type_samples += realsr_processed
                dataset_type_psnr += realsr_psnr
                dataset_type_ssim += realsr_ssim
                all_results.extend(realsr_results)
                for metric_name, values in realsr_quality_stats.items():
                    dataset_quality_stats.setdefault(metric_name, []).extend(values)
                quality_summary = _format_quality_summary(realsr_quality_stats)
                accelerator.print(
                    f"    ✅ {realsr_name}: PSNR={realsr_psnr/realsr_processed:.4f}, "
                    f"SSIM={realsr_ssim/realsr_processed:.4f}{quality_summary} ({realsr_processed} samples)"
                )

        elif dataset_type == "Generic":
            for generic_name, dataset in datasets_obj:
                accelerator.print(f"  Validating on {generic_name} dataset ({len(dataset)} samples)")
                accelerator.print(f"    Generic dataset target_size: {dataset.target_size}")
                accelerator.print(f"    Generic dataset resize_mode: {dataset.resize_mode}")
                generic_dir = os.path.join(dataset_type_dir, generic_name)
                gen_dir   = os.path.join(generic_dir, "generated_images")
                tgt_dir   = os.path.join(generic_dir, "target_images")
                input_dir = os.path.join(generic_dir, "input_images")
                cmp_dir   = os.path.join(generic_dir, "comparisons")
                for d in (gen_dir, tgt_dir, input_dir, cmp_dir):
                    os.makedirs(d, exist_ok=True)
                max_samples = min(num_validation_samples, len(dataset))
                generic_psnr, generic_ssim, generic_processed, generic_results, generic_quality_stats = validate_single_dataset(
                    dataset=dataset,
                    dataset_name=f"{dataset_type}_{generic_name}",
                    args=args,
                    lvlm_model=lvlm_model,
                    vae=vae,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    empty_pooled_prompt_embeds=empty_pooled_prompt_embeds,
                    siglip_model=siglip_model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    collator_tokenizer=collator_tokenizer,
                    latent_image_ids_cache=latent_image_ids_cache,
                    _encode_vae_and_pack=_encode_vae_and_pack,
                    max_samples=max_samples,
                    output_dirs={'gen_dir': gen_dir, 'tgt_dir': tgt_dir,
                                 'lr_dir': input_dir, 'cmp_dir': cmp_dir},
                    is_generic_dataset=True,
                    iqa_metrics=iqa_metrics,
                    quality_metrics=quality_metrics,
                )
                dataset_type_results[generic_name] = {
                    "num_samples": generic_processed,
                    "avg_psnr": float(generic_psnr / generic_processed) if generic_processed > 0 else 0.0,
                    "avg_ssim": float(generic_ssim / generic_processed) if generic_processed > 0 else 0.0,
                    "detailed_results": generic_results,
                    "quality_metrics": generic_quality_stats,
                }
                all_sub_results[generic_name] = dataset_type_results[generic_name]
                dataset_type_samples += generic_processed
                dataset_type_psnr += generic_psnr
                dataset_type_ssim += generic_ssim
                all_results.extend(generic_results)
                for metric_name, values in generic_quality_stats.items():
                    dataset_quality_stats.setdefault(metric_name, []).extend(values)
                quality_summary = _format_quality_summary(generic_quality_stats)
                accelerator.print(
                    f"    ✅ {generic_name}: PSNR={generic_psnr/generic_processed:.4f}, "
                    f"SSIM={generic_ssim/generic_processed:.4f}{quality_summary} ({generic_processed} samples)"
                )

        avg_dataset_type_psnr = dataset_type_psnr / dataset_type_samples if dataset_type_samples > 0 else 0.0
        avg_dataset_type_ssim = dataset_type_ssim / dataset_type_samples if dataset_type_samples > 0 else 0.0
        avg_dataset_quality = {
            k: sum(v) / len(v) for k, v in dataset_quality_stats.items() if v
        }
        total_results[dataset_type] = {
            "num_samples": dataset_type_samples,
            "avg_psnr": float(avg_dataset_type_psnr),
            "avg_ssim": float(avg_dataset_type_ssim),
            "detailed_results": dataset_type_results,
            "avg_quality_metrics": avg_dataset_quality,
        }
        for metric_name, values in dataset_quality_stats.items():
            global_quality_stats.setdefault(metric_name, []).extend(values)

        dataset_type_summary = {
            "dataset_type": dataset_type,
            "step": global_step,
            "timestamp": datetime.now().isoformat(),
            "num_samples": dataset_type_samples,
            "avg_psnr": float(avg_dataset_type_psnr),
            "avg_ssim": float(avg_dataset_type_ssim),
            "method": "hypir_style_restoration",
            "detailed_results": dataset_type_results,
            "avg_quality_metrics": avg_dataset_quality,
        }
        with open(os.path.join(dataset_type_dir, f"{dataset_type}_results.json"), "w") as f:
            json.dump(dataset_type_summary, f, indent=2)

        quality_summary = _format_quality_summary_from_avg(avg_dataset_quality)
        accelerator.print(
            f"  🎯 {dataset_type} overall: PSNR={avg_dataset_type_psnr:.4f}, "
            f"SSIM={avg_dataset_type_ssim:.4f}{quality_summary} ({dataset_type_samples} samples)"
        )

    # Collect timing records from detailed results
    for dataset_type, dataset_result in total_results.items():
        if "detailed_results" in dataset_result:
            for scale_name, scale_results in dataset_result["detailed_results"].items():
                if "detailed_results" in scale_results:
                    for result in scale_results["detailed_results"]:
                        if "timing" in result:
                            timing_info = result["timing"].copy()
                            timing_info["dataset_type"] = dataset_type
                            timing_info["scale_name"] = scale_name
                            global_timing_stats.append(timing_info)

    total_samples = sum(result["num_samples"] for result in total_results.values())
    if total_samples > 0:
        weighted_psnr = sum(
            result["avg_psnr"] * result["num_samples"] for result in total_results.values()
        ) / total_samples
        weighted_ssim = sum(
            result["avg_ssim"] * result["num_samples"] for result in total_results.values()
        ) / total_samples
    else:
        weighted_psnr = weighted_ssim = 0.0

    global_avg_quality = {k: sum(v) / len(v) for k, v in global_quality_stats.items() if v}

    overall_summary = {
        "step": global_step,
        "timestamp": datetime.now().isoformat(),
        "method": "hypir_style_restoration",
        "total_samples": total_samples,
        "weighted_avg_psnr": float(weighted_psnr),
        "weighted_avg_ssim": float(weighted_ssim),
        "dataset_results": total_results,
        "sub_dataset_results": all_sub_results,
        "all_detailed_results": all_results,
        "global_avg_quality_metrics": global_avg_quality,
    }

    if global_timing_stats:
        overall_avg_total_time       = sum(r["total_time"]            for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_inference_time   = sum(r["model_inference_time"]  for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_vae_decode_time  = sum(r["vae_decode_time"]       for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_preprocess_time  = sum(r["preprocess_time"]       for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_postprocess_time = sum(r["postprocess_time"]      for r in global_timing_stats) / len(global_timing_stats)
        overall_summary["timing_statistics"] = {
            "total_images_processed":          len(global_timing_stats),
            "average_total_time_per_image":    float(overall_avg_total_time),
            "average_inference_time_per_image": float(overall_avg_inference_time),
            "average_vae_decode_time_per_image": float(overall_avg_vae_decode_time),
            "average_preprocess_time_per_image": float(overall_avg_preprocess_time),
            "average_postprocess_time_per_image": float(overall_avg_postprocess_time),
            "images_per_second":               float(1.0 / overall_avg_total_time),
            "detailed_timing_records":         global_timing_stats,
        }

    with open(os.path.join(validation_dir, "overall_validation_results.json"), "w") as f:
        json.dump(overall_summary, f, indent=2)

    # Build text report
    report_txt = (
        f"Enhanced Multi-Dataset Validation Report - Step {global_step} (HYPIR Style with pyiqa Metrics)\n"
        f"==========================================================================================\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Method: HYPIR-style Image Restoration\n"
        f"Total samples: {total_samples}\n"
        f"Weighted Average PSNR: {weighted_psnr:.4f} dB\n"
        f"Weighted Average SSIM: {weighted_ssim:.4f}\n\n"
    )
    if global_avg_quality:
        report_txt += "Global Average Quality Metrics:\n"
        report_txt += "-" * 50 + "\n"
        for metric_name, avg_val in global_avg_quality.items():
            direction = "↑" if any(x in metric_name.lower() for x in ["musiq", "maniqa", "clipiqa", "topiq"]) else "↓"
            report_txt += f"{metric_name:>15}: {avg_val:>8.4f} {direction}\n"
        report_txt += "\n"

    report_txt += "Results by Scale/Type:\n"
    report_txt += "-" * 50 + "\n"
    for sub_name, sub_result in all_sub_results.items():
        display_name = f"{sub_name[:-1]}x Scale" if sub_name.endswith('x') else f"{sub_name} Type"
        report_txt += (
            f"{display_name:>12}: {sub_result['num_samples']:>4} samples | "
            f"PSNR: {sub_result['avg_psnr']:>6.2f} dB | "
            f"SSIM: {sub_result['avg_ssim']:>6.4f}\n"
        )

    report_txt += "\n" + "Dataset Type Summary:\n"
    report_txt += "-" * 50 + "\n"
    for dataset_type, result in total_results.items():
        report_txt += (
            f"{dataset_type:>8} Dataset: {result['num_samples']:>4} samples | "
            f"PSNR: {result['avg_psnr']:>6.2f} dB | "
            f"SSIM: {result['avg_ssim']:>6.4f}\n"
        )

    if global_timing_stats:
        overall_avg_total_time      = sum(r["total_time"]           for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_inference_time  = sum(r["model_inference_time"] for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_vae_decode_time = sum(r["vae_decode_time"]      for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_preprocess_time = sum(r["preprocess_time"]      for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_postprocess_time= sum(r["postprocess_time"]     for r in global_timing_stats) / len(global_timing_stats)
        report_txt += f"\nTiming Statistics:\n"
        report_txt += "-" * 50 + "\n"
        report_txt += f"Total images processed: {len(global_timing_stats)}\n"
        report_txt += f"Average processing time per image: {overall_avg_total_time:.3f}s\n"
        report_txt += f"Average model inference time: {overall_avg_inference_time:.3f}s ({overall_avg_inference_time/overall_avg_total_time*100:.1f}%)\n"
        report_txt += f"Average VAE decode time: {overall_avg_vae_decode_time:.3f}s ({overall_avg_vae_decode_time/overall_avg_total_time*100:.1f}%)\n"
        report_txt += f"Average preprocess time: {overall_avg_preprocess_time:.3f}s ({overall_avg_preprocess_time/overall_avg_total_time*100:.1f}%)\n"
        report_txt += f"Average postprocess time: {overall_avg_postprocess_time:.3f}s ({overall_avg_postprocess_time/overall_avg_total_time*100:.1f}%)\n"
        report_txt += f"Processing throughput: {1.0/overall_avg_total_time:.2f} images/second\n"

    report_txt += (
        f"\nFiles saved in: {validation_dir}\n"
        f"Each scale/type has its own subdirectory with:\n"
        f"- Generated images\n"
        f"- Target images\n"
        f"- Input/LR images\n"
        f"- Comparison images (layout: [LR | Target | Generated])\n"
    )
    with open(os.path.join(validation_dir, "validation_report.txt"), "w") as f:
        f.write(report_txt)

    # Build accelerator log dict
    log_dict = {
        "val_weighted_psnr":    weighted_psnr,
        "val_weighted_ssim":    weighted_ssim,
        "val_total_samples":    total_samples,
    }
    for metric_name, avg_val in global_avg_quality.items():
        log_dict[f"val_global_{metric_name}"] = avg_val
    for sub_name, sub_result in all_sub_results.items():
        log_dict[f"val_{sub_name}_psnr"]    = sub_result["avg_psnr"]
        log_dict[f"val_{sub_name}_ssim"]    = sub_result["avg_ssim"]
        log_dict[f"val_{sub_name}_samples"] = sub_result["num_samples"]
    for dataset_type, result in total_results.items():
        log_dict[f"val_{dataset_type}_overall_psnr"]    = result["avg_psnr"]
        log_dict[f"val_{dataset_type}_overall_ssim"]    = result["avg_ssim"]
        log_dict[f"val_{dataset_type}_overall_samples"] = result["num_samples"]
    if global_timing_stats:
        overall_avg_total_time      = sum(r["total_time"]           for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_inference_time  = sum(r["model_inference_time"] for r in global_timing_stats) / len(global_timing_stats)
        overall_avg_vae_decode_time = sum(r["vae_decode_time"]      for r in global_timing_stats) / len(global_timing_stats)
        log_dict.update({
            "val_avg_total_time":      overall_avg_total_time,
            "val_avg_inference_time":  overall_avg_inference_time,
            "val_avg_vae_decode_time": overall_avg_vae_decode_time,
            "val_images_per_second":   1.0 / overall_avg_total_time,
        })
        accelerator.print(f"\n⏱️  Global Timing Summary:")
        accelerator.print(f"  Total images processed: {len(global_timing_stats)}")
        accelerator.print(f"  Overall average time per image: {overall_avg_total_time:.3f}s")
        accelerator.print(f"  Overall average inference time: {overall_avg_inference_time:.3f}s ({overall_avg_inference_time/overall_avg_total_time*100:.1f}%)")
        accelerator.print(f"  Overall average VAE decode time: {overall_avg_vae_decode_time:.3f}s ({overall_avg_vae_decode_time/overall_avg_total_time*100:.1f}%)")
        accelerator.print(f"  Overall images per second: {1.0/overall_avg_total_time:.2f}")

    if global_avg_quality:
        accelerator.print(f"\n📊 Global Quality Metrics Summary:")
        for metric_name, avg_val in global_avg_quality.items():
            direction = "↑" if any(x in metric_name.lower() for x in ["musiq", "maniqa", "clipiqa", "topiq"]) else "↓"
            accelerator.print(f"  {metric_name:>15}: {avg_val:>8.4f} {direction}")

    accelerator.log(log_dict, step=global_step)
    accelerator.print(f"[VAL DONE] Overall: PSNR={weighted_psnr:.4f}, SSIM={weighted_ssim:.4f}, saved to {validation_dir}")
    accelerator.print("\nResults by Scale/Type:")
    for sub_name, sub_result in all_sub_results.items():
        display_name = f"{sub_name[:-1]}x Scale" if sub_name.endswith('x') else f"{sub_name} Type"
        accelerator.print(f"  {display_name:>12}: PSNR={sub_result['avg_psnr']:.4f}, SSIM={sub_result['avg_ssim']:.4f}")
    accelerator.print("\nDataset-wise summary:")
    for dataset_type, result in total_results.items():
        accelerator.print(f"  {dataset_type}: PSNR={result['avg_psnr']:.4f}, SSIM={result['avg_ssim']:.4f}")

    lvlm_model.train()
    return weighted_psnr, weighted_ssim


# ──────────────────────────────────────────────────────────────────────────────
# Helper: format quality summary strings for logging
# ──────────────────────────────────────────────────────────────────────────────

def _format_quality_summary(quality_stats: Dict[str, List]) -> str:
    """Return a formatted summary string from a dict of metric_name -> list_of_values."""
    if not quality_stats:
        return ""
    items = []
    for metric_name, values in quality_stats.items():
        if values:
            avg_val = sum(values) / len(values)
            items.append(f"{metric_name}={avg_val:.4f}")
    return (", " + ", ".join(items)) if items else ""


def _format_quality_summary_from_avg(avg_quality: Dict[str, float]) -> str:
    """Return a formatted summary string from a dict of metric_name -> avg_value."""
    if not avg_quality:
        return ""
    items = [f"{k}={v:.4f}" for k, v in avg_quality.items()]
    return (", " + ", ".join(items)) if items else ""


# ──────────────────────────────────────────────────────────────────────────────
# validate_single_dataset
# ──────────────────────────────────────────────────────────────────────────────

def validate_single_dataset(
    dataset, dataset_name, args, lvlm_model, vae, text_encoders, tokenizers,
    empty_pooled_prompt_embeds, siglip_model, accelerator, weight_dtype,
    collator_tokenizer, latent_image_ids_cache, _encode_vae_and_pack,
    max_samples, output_dirs,
    is_weather_dataset=False, is_realsr_dataset=False, is_generic_dataset=False,
    iqa_metrics: Optional[Dict[str, Any]] = None,
    quality_metrics=None,
):
    """
    Run inference and compute IQA metrics for a single dataset.

    IQA metrics are computed using pyiqa (passed in as `iqa_metrics`):
      - Full-reference: PSNR, SSIM, LPIPS, DISTS  (require both SR and GT tensors)
      - No-reference:   CLIPIQA, NIQE, MUSIQ, MANIQA, TOPIQA  (SR tensor only)

    `total_psnr` and `total_ssim` are accumulated from the pyiqa PSNR/SSIM values
    for backward-compatible return values used by the caller.
    """
    data_collator = ValidationCollator(tokenizer=collator_tokenizer, padding_side='left')
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.dataset_config.batch_size,
        pin_memory=args.dataset_config.pin_memory,
        num_workers=0,
        collate_fn=data_collator,
        prefetch_factor=None,
    )

    results = []
    total_psnr, total_ssim, processed = 0.0, 0.0, 0
    timing_records = []
    quality_stats: Dict[str, List] = {}
    fid_accumulator = None

    processor      = getattr(lvlm_model, "processor", None)
    tokenizer_vlm  = getattr(processor, "tokenizer", None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= max_samples:
                break

            generated_image    = batch["generated_image"].to(accelerator.device)
            input_ids          = batch["input_ids"].to(accelerator.device)
            attention_mask     = batch["attention_mask"].to(accelerator.device)
            pixel_values       = (None if batch["pixel_values"] is None
                                  else batch["pixel_values"].to(accelerator.device, dtype=weight_dtype))
            image_position     = batch["image_position"]
            image_grid_thw     = (None if batch["image_grid_thw"] is None
                                  else batch["image_grid_thw"].to(accelerator.device))
            prompts            = batch.get("prompts", [""] * generated_image.size(0))
            pil_pixel_values   = batch.get("pil_pixel_values", None)  # [[PIL(LR)], ...]
            siglip_pixel_values = batch["siglip_pixel_values"]
            vae_pixel_values   = batch["vae_pixel_values"]

            B, _, H, W = generated_image.shape
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            target_h = max((H // vae_scale_factor) * vae_scale_factor, vae_scale_factor)
            target_w = max((W // vae_scale_factor) * vae_scale_factor, vae_scale_factor)

            for i in range(B):
                single_sample_start = time.time()

                # ── 1) Obtain the LR PIL for this sample (used for 5-line generation) ──
                lr_pil = None
                if isinstance(pil_pixel_values, list) and i < len(pil_pixel_values):
                    cur = pil_pixel_values[i]
                    if isinstance(cur, (list, tuple)) and len(cur) > 0 and isinstance(cur[0], Image.Image):
                        lr_pil = cur[0]
                if lr_pil is None:
                    # Fallback: reconstruct PIL from VAE tensor (text generation only; not training input)
                    x = vae_pixel_values[i:i+1].to(dtype=torch.float32, device=vae.device)
                    x = (x * 0.5 + 0.5).clamp(0, 1)
                    lr_pil = transforms.ToPILImage()(x[0].cpu())

                # ── 2) Generate 5-line text (consistent with training: expert prompt + features + repick) ──
                gen_text, pred_task, _features, rebuilt_inputs = infer_5lines_with_repicks_for_validation(
                    first_img_pil=lr_pil, lvlm_model=lvlm_model,
                    processor=processor, tokenizer=tokenizer_vlm,
                    prompt_text=NEUTRAL_PROMPT, append_global_feature_hints=True,
                    use_expert_prompt=True, respect_size_superres=True,
                    disable_superres=True, repick_if_superres=True, selective_repick=True,
                    repick_temperature=0.2, repick_top_p=0.5,
                    gen_max_pixels=getattr(args.model_config, "gen_max_pixels", 512 * 512),
                    generation_max_new_tokens=getattr(args.model_config, "generation_max_new_tokens", 512),
                    generation_temperature=getattr(args.model_config, "generation_temperature", 0.7),
                    generation_top_p=getattr(args.model_config, "generation_top_p", 0.9),
                    generation_do_sample=getattr(args.model_config, "generation_do_sample", True),
                )

                # ── 3A) Build T5 prompt from the 5-line output ──
                single_prompt = (
                    "Please restore this real-world degraded image by removing any possible degradations "
                    "according to {0}. Produce a clear and natural result that preserves fine details, "
                    "textures, and the original structure of the scene.".format(gen_text)
                )

                # ── 3B) Optionally rebuild LVLM input IDs from the 5-line text (strict training alignment) ──
                if USE_PROMPT_TO_REBUILD_IDS:
                    conversations = [
                        {"role": "user", "content": [{"type": "image", "image": lr_pil},
                                                     {"type": "text", "text": gen_text}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "<|image_gen|>"}]},
                    ]
                    msgs = _normalize_messages_for_processor(conversations)
                    rebuilt = processor.apply_chat_template(
                        msgs, add_generation_prompt=False, tokenize=True,
                        return_tensors="pt", return_dict=True,
                        max_pixels=getattr(args.model_config, "gen_max_pixels", 512 * 512),
                    )
                    ids  = rebuilt["input_ids"].to(accelerator.device)
                    attn = rebuilt.get("attention_mask", None)
                    if attn is not None:  attn = attn.to(accelerator.device)
                    px   = rebuilt.get("pixel_values", None)
                    if px is not None:   px = px.to(accelerator.device, dtype=weight_dtype)
                    grid = rebuilt.get("image_grid_thw", None)
                    if grid is not None: grid = grid.to(accelerator.device)
                else:
                    # Slice the i-th sample from the DataLoader batch
                    ids  = slice_sample(input_ids,      i, B)
                    attn = slice_sample(attention_mask,  i, B)
                    px   = slice_sample(pixel_values,    i, B)
                    grid = slice_sample(image_grid_thw,  i, B)
                img_pos = slice_sample(image_position, i, B)

                # ── 4) T5 prompt embeddings (if text encoders are available) ──
                t5_prompt_embeds = None
                if (text_encoders is not None and len(text_encoders) > 1
                        and text_encoders[1] is not None
                        and args.training_config.drop_t5_rate < 1.0):
                    t5_prompt_embeds, _ = encode_prompt(
                        [None, text_encoders[1]],
                        tokenizers,
                        prompt=[single_prompt],
                        max_sequence_length=256,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                    )

                # ── 5) SigLIP / VAE hidden states ──
                siglip_hidden = None
                if (siglip_model is not None
                        and isinstance(siglip_pixel_values, torch.Tensor)
                        and siglip_pixel_values.size(0) == B):
                    siglip_px = siglip_pixel_values[i:i+1].to(accelerator.device, dtype=siglip_model.dtype)
                    siglip_hidden = siglip_model(siglip_px).last_hidden_state

                vae_hidden = None
                if isinstance(vae_pixel_values, torch.Tensor) and vae_pixel_values.size(0) == B:
                    vae_px = vae_pixel_values[i:i+1].to(accelerator.device, dtype=vae.dtype)
                    _, vae_hidden = _encode_vae_and_pack(vae_px)
                    vae_hidden = vae_hidden.to(weight_dtype)

                preprocess_time = time.time() - single_sample_start

                # ── 6) Spatial shape alignment ──
                gt_img = slice_sample(generated_image, i, B)
                model_C = getattr(vae.config, "latent_channels", 16)
                model_H = target_h // (2 ** (len(vae.config.block_out_channels) - 1))
                model_W = target_w // (2 ** (len(vae.config.block_out_channels) - 1))

                input_img = vae_pixel_values[i:i+1].to(device=accelerator.device, dtype=vae.dtype)
                input_h, input_w = input_img.shape[2], input_img.shape[3]
                input_img_resized = (
                    F.interpolate(input_img, size=(target_h, target_w), mode="bicubic", align_corners=False)
                    if (input_h != target_h or input_w != target_w) else input_img
                )
                gt_h, gt_w = gt_img.shape[2], gt_img.shape[3]
                gt_img_resized = (
                    F.interpolate(gt_img, size=(target_h, target_w), mode="bicubic", align_corners=False)
                    if (gt_h != target_h or gt_w != target_w) else gt_img
                )

                # ── 7) VAE encode + prepare latent IDs ──
                lq_latents = vae.encode(input_img_resized).latent_dist.sample()
                lq_latents = (lq_latents - vae.config.shift_factor) * vae.config.scaling_factor
                lq_latents = lq_latents.to(dtype=weight_dtype)

                model_shape = (1, model_C, model_H, model_W)
                if model_shape in latent_image_ids_cache:
                    latent_image_ids = latent_image_ids_cache[model_shape]
                else:
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        1, model_H // 2, model_W // 2, accelerator.device, weight_dtype
                    )
                    latent_image_ids_cache[model_shape] = latent_image_ids

                fixed_timestep      = torch.full((1,), 300, dtype=torch.long, device=accelerator.device)
                timestep_normalized = (fixed_timestep / 1000.0).to(weight_dtype)
                packed_lq_latents = FluxPipeline._pack_latents(
                    lq_latents, batch_size=1, num_channels_latents=lq_latents.shape[1],
                    height=lq_latents.shape[2], width=lq_latents.shape[3],
                )
                guidance = torch.full(
                    (1,), args.model_config.guidance_scale, device=accelerator.device, dtype=weight_dtype
                )

                # ── 8) Model forward pass ──
                model_inference_start = time.time()
                pred = lvlm_model(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=px,
                    image_position=img_pos,
                    image_grid_thw=grid,
                    output_type="denoise_model_pred",
                    only_use_t5=args.model_config.only_use_t5,
                    ref_features_for_vlm=None,
                    vlm_residual_image_factor=args.model_config.vlm_residual_image_factor,
                    siglip_hidden_states=siglip_hidden,
                    vae_hidden_states=vae_hidden,
                    denoiser_kwargs={
                        "prefix_prompt_embeds": t5_prompt_embeds,
                        "hidden_states":        packed_lq_latents.to(weight_dtype),
                        "timestep":             timestep_normalized,
                        "guidance":             guidance,
                        "pooled_projections":   empty_pooled_prompt_embeds[:1],
                        "img_ids":              latent_image_ids,
                        "joint_attention_kwargs": {},
                    },
                )
                model_inference_time = time.time() - model_inference_start

                # ── 9) VAE decode ──
                vae_decode_start = time.time()
                pred_latents = FluxPipeline._unpack_latents(
                    pred, height=target_h, width=target_w,
                    vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1)
                )
                pred_latents = pred_latents / vae.config.scaling_factor + vae.config.shift_factor
                restored_img = vae.decode(pred_latents.to(dtype=vae.dtype)).sample
                vae_decode_time = time.time() - vae_decode_start

                # ── 10) Post-processing ──
                postprocess_start = time.time()
                restored_img      = torch.clamp(restored_img,      -1.0, 1.0)
                gt_img_resized    = torch.clamp(gt_img_resized,    -1.0, 1.0)
                input_img_resized = torch.clamp(input_img_resized, -1.0, 1.0)

                # ── 11) Compute IQA metrics via pyiqa ──
                # Convert from [-1, 1] to [0, 1] for pyiqa; use float32
                sr_01 = (restored_img.float() * 0.5 + 0.5).clamp(0.0, 1.0)
                gt_01 = (gt_img_resized.float() * 0.5 + 0.5).clamp(0.0, 1.0)

                iqa_results: Dict[str, float] = {}
                if iqa_metrics is not None:
                    iqa_results = _compute_iqa_metrics(iqa_metrics, sr_01, gt_01)

                # Accumulate all metric values into quality_stats
                for metric_name, metric_value in iqa_results.items():
                    if not np.isnan(metric_value):
                        quality_stats.setdefault(metric_name, []).append(metric_value)

                # Extract PSNR / SSIM for the caller's running totals
                psnr = iqa_results.get('PSNR', 0.0)
                ssim = iqa_results.get('SSIM', 0.0)
                if np.isnan(psnr): psnr = 0.0
                if np.isnan(ssim): ssim = 0.0

                # Also run any additional quality_metrics object if provided
                extra_quality_results = {}
                if quality_metrics is not None:
                    try:
                        extra_quality_results = quality_metrics.calculate_all_metrics(restored_img, gt_img_resized)
                        for metric_name, metric_value in extra_quality_results.items():
                            quality_stats.setdefault(metric_name, []).append(metric_value)
                    except Exception as e:
                        accelerator.print(f"Warning: additional quality_metrics failed for sample {i}: {e}")

                # ── 12) Save images ──
                restored_pil = transforms.ToPILImage()(sr_01[0].cpu())
                gt_pil       = transforms.ToPILImage()(gt_01[0].cpu())
                input_pil    = transforms.ToPILImage()((input_img_resized.float() * 0.5 + 0.5).clamp(0, 1)[0].cpu())
                sample_id    = f"{dataset_name}_{batch_idx:04d}_{i:02d}"

                for d in output_dirs.values():
                    os.makedirs(d, exist_ok=True)
                restored_pil.save(os.path.join(output_dirs['gen_dir'], f"{sample_id}_generated.png"))
                gt_pil.save(os.path.join(output_dirs['tgt_dir'],       f"{sample_id}_target.png"))
                input_pil.save(os.path.join(output_dirs['lr_dir'],     f"{sample_id}_input.png"))

                # Side-by-side comparison: [Input | GT | Restored]
                comparison_img = Image.new('RGB', (restored_pil.width * 3, restored_pil.height))
                comparison_img.paste(input_pil,    (0, 0))
                comparison_img.paste(gt_pil,       (restored_pil.width, 0))
                comparison_img.paste(restored_pil, (restored_pil.width * 2, 0))
                comparison_img.save(os.path.join(output_dirs['cmp_dir'], f"{sample_id}_comparison.png"))

                postprocess_time         = time.time() - postprocess_start
                single_sample_total_time = time.time() - single_sample_start

                timing_record = {
                    "sample_id":           sample_id,
                    "total_time":          single_sample_total_time,
                    "preprocess_time":     preprocess_time,
                    "model_inference_time": model_inference_time,
                    "vae_decode_time":     vae_decode_time,
                    "postprocess_time":    postprocess_time,
                    "inference_time":      model_inference_time,
                }
                timing_records.append(timing_record)

                total_psnr += psnr
                total_ssim += ssim
                processed  += 1

                # Merge all metric results for per-sample record
                all_metric_results = {**iqa_results, **extra_quality_results}

                sample_result = {
                    "sample_id":     sample_id,
                    "psnr":          float(psnr),
                    "ssim":          float(ssim),
                    "prompt":        single_prompt,
                    "timing":        timing_record,
                    "iqa_metrics":   {k: (float(v) if not np.isnan(v) else None)
                                      for k, v in iqa_results.items()},
                    "quality_metrics": extra_quality_results,
                }

                # Attach dataset-specific metadata
                if is_realsr_dataset:
                    sample_result["camera_type"]  = batch.get("camera_type",  ["unknown"])[i] if hasattr(batch, "camera_type")  else "unknown"
                    sample_result["scale_factor"] = batch.get("scale_factor", [1])[i]          if hasattr(batch, "scale_factor") else 1
                    sample_result["lr_path"]      = batch.get("lr_path",      ["unknown"])[i]  if hasattr(batch, "lr_path")      else "unknown"
                    sample_result["hr_path"]      = batch.get("hr_path",      ["unknown"])[i]  if hasattr(batch, "hr_path")      else "unknown"
                elif is_weather_dataset:
                    sample_result["weather_type"] = batch.get("weather_type", ["unknown"])[i]  if hasattr(batch, "weather_type") else "unknown"
                    sample_result["input_path"]   = batch.get("input_path",   ["unknown"])[i]  if hasattr(batch, "input_path")   else "unknown"
                    sample_result["gt_path"]      = batch.get("gt_path",      ["unknown"])[i]  if hasattr(batch, "gt_path")      else "unknown"
                elif is_generic_dataset:
                    sample_result["dataset_name"] = batch.get("dataset_name", ["unknown"])[i]  if hasattr(batch, "dataset_name") else "unknown"
                    sample_result["input_path"]   = batch.get("input_path",   ["unknown"])[i]  if hasattr(batch, "input_path")   else "unknown"
                    sample_result["gt_path"]      = batch.get("gt_path",      ["unknown"])[i]  if hasattr(batch, "gt_path")      else "unknown"

                results.append(sample_result)

                # Periodic progress log
                if processed % 10 == 0:
                    avg_psnr = total_psnr / processed
                    avg_ssim = total_ssim / processed
                    last_n   = timing_records[-min(10, len(timing_records)):]
                    avg_inference_time = sum(r["model_inference_time"] for r in last_n) / len(last_n) if last_n else 0.0
                    avg_total_time     = sum(r["total_time"]           for r in last_n) / len(last_n) if last_n else 0.0

                    # Summarize recent IQA metrics
                    quality_summary_parts = []
                    for metric_name, values in quality_stats.items():
                        if values:
                            recent = values[-min(10, len(values)):]
                            quality_summary_parts.append(f"{metric_name}={sum(recent)/len(recent):.4f}")
                    quality_summary = (", " + ", ".join(quality_summary_parts)) if quality_summary_parts else ""

                    accelerator.print(
                        f" Processed {processed} samples: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, "
                        f"Avg Inference Time={avg_inference_time:.3f}s, "
                        f"Avg Total Time={avg_total_time:.3f}s{quality_summary}"
                    )

    # ── Final timing summary ──
    if timing_records:
        avg_total_time       = sum(r["total_time"]            for r in timing_records) / len(timing_records)
        avg_inference_time   = sum(r["model_inference_time"]  for r in timing_records) / len(timing_records)
        avg_vae_decode_time  = sum(r["vae_decode_time"]       for r in timing_records) / len(timing_records)
        avg_preprocess_time  = sum(r["preprocess_time"]       for r in timing_records) / len(timing_records)
        avg_postprocess_time = sum(r["postprocess_time"]      for r in timing_records) / len(timing_records)
        accelerator.print(f"\n⏱️  Dataset {dataset_name} Timing Summary:")
        accelerator.print(f"  Total samples: {len(timing_records)}")
        accelerator.print(f"  Average total time per image: {avg_total_time:.3f}s")
        accelerator.print(f"  Average model inference time: {avg_inference_time:.3f}s ({avg_inference_time/avg_total_time*100:.1f}%)")
        accelerator.print(f"  Average VAE decode time: {avg_vae_decode_time:.3f}s ({avg_vae_decode_time/avg_total_time*100:.1f}%)")
        accelerator.print(f"  Average preprocess time: {avg_preprocess_time:.3f}s ({avg_preprocess_time/avg_total_time*100:.1f}%)")
        accelerator.print(f"  Average postprocess time: {avg_postprocess_time:.3f}s ({avg_postprocess_time/avg_total_time*100:.1f}%)")
        accelerator.print(f"  Images per second: {1.0/avg_total_time:.2f}")

    # ── Final IQA metrics summary ──
    if quality_stats:
        accelerator.print(f"\n📊 Dataset {dataset_name} Quality Metrics Summary:")
        for metric_name, values in quality_stats.items():
            if values:
                avg_val   = sum(values) / len(values)
                direction = "↑" if any(x in metric_name.lower() for x in ["musiq", "maniqa", "clipiqa", "topiq"]) else "↓"
                accelerator.print(f"  {metric_name:>15}: {avg_val:>8.4f} {direction} (n={len(values)})")

    if fid_accumulator is not None:
        fid_score = fid_accumulator.compute_fid()
        if fid_score is not None:
            quality_stats['fid'] = [fid_score]
            accelerator.print(f"FID Score for {dataset_name}: {fid_score:.4f}")

    return total_psnr, total_ssim, processed, results, quality_stats