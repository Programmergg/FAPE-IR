import os
import re
import torch
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from einops import rearrange
from typing import Callable, List, Optional, Tuple, Dict, Any

from torchvision import transforms
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from fapeir.utils.prompter import Prompter, PROMPT_TYPE
from fapeir.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from qwen_vl_utils.vision_process import to_rgb, smart_resize, fetch_video
from fapeir.models.qwen2p5vl.modeling_fapeir_qwen2p5vl_moe import FAPEIRQwen2p5VLForConditionalGeneration


# ---------------------------
# Image/Video preprocess
# ---------------------------

def fetch_image(ele: dict, size_factor: int = 28, resize_mode: str = "smart") -> Image.Image:
    """
    Load a single image; supports PIL / local path / http(s) / data:image;base64 / file://.
    Applies smart_resize based on resize_mode.
    """
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, stream=True)
            image_obj = Image.open(BytesIO(response.content))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )

    image = to_rgb(image_obj)
    if resize_mode == "none":
        return image

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(
            height, width, factor=size_factor, min_pixels=min_pixels, max_pixels=max_pixels,
        )
    return image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)


def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1,
    resize_mode: str = "smart",
) -> tuple:
    image_inputs, video_inputs, video_sample_fps_list = [], [], []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28 * factor, resize_mode=resize_mode))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {"fps": video_sample_fps_list}
    return image_inputs, video_inputs


# =========================================================
#         WeatherTestDataset (no 5-line generation)
# =========================================================

class WeatherTestDataset(Dataset):
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        weather_type: str,
        transform: Callable,
        prompter: Prompter,
        output_dir: Optional[str] = None,
        dataset_type: str = 'qwen2vl',
        min_pixels: int = 384*384,
        max_pixels: int = 384*384,
        image_token_length: int = 729,
        mask_weight_type: str = 'log',
        siglip_processor: Callable = None,
        seed: int = 42,
        resize_mode: str = "none",
        target_size: int = 512,
        naming_patterns: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        assert dataset_type in ('qwen2vl', 'qwen2p5vl')
        self.data_root = data_root
        self.weather_type = weather_type
        self.transform = transform
        self.resize_mode = resize_mode
        self.target_size = target_size
        self.vae_transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # Model / tokenizer / processor
        self.lvlm_model = lvlm_model
        self.lvlm_model.eval()
        self.device = next(self.lvlm_model.parameters()).device
        self.processor = getattr(self.lvlm_model, "processor", None)
        assert self.processor is not None, "lvlm_model must carry a `processor`"
        self.tokenizer = tokenizer or getattr(self.processor, "tokenizer", None)
        assert self.tokenizer is not None, "tokenizer not found; pass one or ensure processor has tokenizer"

        self.prompter = prompter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token = SPACIAL_TOKEN[dataset_type]['image_token']
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = self.processor.image_processor
        self.factor = 2
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.seed = seed
        self.output_dir = output_dir

        # Naming patterns
        default_naming_patterns = {
                "weather1": {"input_suffix": "_rain","gt_suffix": "_clean"},
                "weather2": {"input_suffix": "_input","gt_suffix": "_gt"},
                "Snow100K-L": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-S": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-L-ori": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-S-ori": {"input_suffix": "","gt_suffix": ""},
                "Rain100L": {"input_suffix": "","gt_suffix": ""},
                "RainTrainL": {"input_suffix": "","gt_suffix": ""},
                "Rain100_H_test": {"input_suffix": "","gt_suffix": ""},
                "Rain100_L_test": {"input_suffix": "","gt_suffix": ""},
                "multi_haze_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_haze_snow": {"input_suffix": "","gt_suffix": ""},
                "multi_low_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_low_snow": {"input_suffix": "","gt_suffix": ""},
                "multi_low_haze_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_low_haze_snow": {"input_suffix": "","gt_suffix": ""}
            }
        self.naming_patterns = naming_patterns or default_naming_patterns

        # Directories
        self.weather_folder = os.path.join(data_root, weather_type)
        self.input_folder = os.path.join(self.weather_folder, 'input')
        self.gt_folder = os.path.join(self.weather_folder, 'gt')
        if self.output_dir:
            self.output_input_dir = os.path.join(output_dir, f'resized_{weather_type}', 'input')
            self.output_gt_dir = os.path.join(output_dir, f'resized_{weather_type}', 'gt')
            os.makedirs(self.output_input_dir, exist_ok=True)
            os.makedirs(self.output_gt_dir, exist_ok=True)
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder not found: {self.input_folder}")
        if not os.path.exists(self.gt_folder):
            raise ValueError(f"GT folder not found: {self.gt_folder}")

        # Special tokens
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(self.image_begin_token)
        assert isinstance(self.image_begin_token_id, int), (f"tokenizer miss image begin token `{self.image_begin_token}`")
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        assert isinstance(self.image_end_token_id, int), (f"tokenizer miss image end token `{self.image_end_token}`")

        # Load image pairs
        self.image_pairs = self._load_image_pairs()
        print(f"🎯 WeatherTestDataset initialized: {self.weather_type} | total pairs={len(self.image_pairs)} | target={self.target_size} | resize_mode={self.resize_mode}")

    # ---------- I/O ----------
    def _extract_base_name(self, filename: str, is_input: bool = True) -> str:
        name_without_ext = os.path.splitext(filename)[0]
        if self.weather_type in self.naming_patterns:
            pattern = self.naming_patterns[self.weather_type]
            suffix = pattern.get("input_suffix" if is_input else "gt_suffix", "")
            if suffix and name_without_ext.endswith(suffix):
                return name_without_ext[:-len(suffix)]
        if self.weather_type.startswith("Snow100K"):
            return name_without_ext
        m = re.match(r'^(\d+)', name_without_ext)
        return m.group(1) if m else name_without_ext

    def _find_matching_gt(self, input_filename: str, gt_files: Dict[str, str]) -> Optional[str]:
        input_base = self._extract_base_name(input_filename, is_input=True)
        # Direct name match
        if input_base in gt_files:
            return gt_files[input_base]
        # Suffix substitution
        if self.weather_type in self.naming_patterns:
            pat = self.naming_patterns[self.weather_type]
            insuf = pat.get("input_suffix", ""); gtsuf = pat.get("gt_suffix", "")
            if insuf and gtsuf:
                nw = os.path.splitext(input_filename)[0]
                if nw.endswith(insuf):
                    expect = nw[:-len(insuf)] + gtsuf
                    for bn, p in gt_files.items():
                        if os.path.splitext(os.path.basename(p))[0] == expect:
                            return p
        # Fuzzy numeric match
        input_numbers = re.findall(r'\d+', input_filename)
        if input_numbers:
            input_num = input_numbers[0]
            for p in gt_files.values():
                gt_numbers = re.findall(r'\d+', os.path.basename(p))
                if gt_numbers and gt_numbers[0] == input_num:
                    return p
        return None

    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        input_files = []
        for f in sorted(os.listdir(self.input_folder)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_files.append((f, os.path.join(self.input_folder, f)))
        gt_files: Dict[str, str] = {}
        for f in sorted(os.listdir(self.gt_folder)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                base = self._extract_base_name(f, is_input=False)
                gt_files[base] = os.path.join(self.gt_folder, f)
        image_pairs = []
        matched = 0
        for in_name, in_path in input_files:
            gt_path = self._find_matching_gt(in_name, gt_files)
            if gt_path:
                image_pairs.append((in_path, gt_path))
                matched += 1
        if matched == 0:
            raise ValueError(f"No matching pairs in {self.weather_folder}")
        return image_pairs

    def _resize_pair(self, input_path: str, gt_path: str, index: int) -> Tuple[Image.Image, Image.Image]:
        inp = Image.open(input_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')
        target = (self.target_size, self.target_size)
        inp_r = inp.resize(target, Image.Resampling.BICUBIC)
        gt_r  = gt.resize(target, Image.Resampling.BICUBIC)
        if self.output_dir:
            base = os.path.splitext(os.path.basename(input_path))[0]
            suf = f"_resized_{index:04d}"
            ip = os.path.join(self.output_input_dir, f"{base}{suf}.png")
            gp = os.path.join(self.output_gt_dir, f"{base}{suf}.png")
            os.makedirs(self.output_input_dir, exist_ok=True)
            os.makedirs(self.output_gt_dir, exist_ok=True)
            inp_r.save(ip); gt_r.save(gp)
        return inp_r, gt_r

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, gt_path = self.image_pairs[idx]
        input_image, gt_image = self._resize_pair(input_path, gt_path, idx)

        # ---- Minimal conversation (no 5-line generation) ----
        raw_conversations = [
            {"from": "human", "value": "<image>"},
            {"from": "gpt",   "value": GENERATE_TOKEN},
        ]
        conversations = [
            {
                "from": (self.prompter.user_role if item["from"] == "human" else self.prompter.assistant_role),
                "value": item["value"],
            }
            for item in raw_conversations
        ]
        prompt_list = self.prompter.get_train_prompt(conversations)

        input_ids, labels, has_generated_image = [], [], False
        for it in prompt_list:
            it["prompt"] = it["prompt"].replace('<image>', self.image_token)
            if GENERATE_TOKEN in it["prompt"]:
                assert it["from"] == self.prompter.assistant_role
                assert (f"{GENERATE_TOKEN}{self.prompter.eos_token}" in it["prompt"])
                it["prompt"] = it["prompt"].replace(
                    f"{GENERATE_TOKEN}{self.prompter.eos_token}",
                    self.image_begin_token,
                )
                has_generated_image = True
            tok = self.tokenizer(it["prompt"], return_tensors="pt", truncation=True, max_length=1024)
            labels.append(tok.input_ids if it.get("is_labels", False) else torch.full_like(tok.input_ids, -100))
            input_ids.append(tok.input_ids)

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # Visual encoding (input image)
        image_slice = [input_image]
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels,
            processor=self.processor, image_token=self.image_token,
            factor=self.factor, last_image=gt_image,
            vae_image_transform=self.vae_transform, drop_prompt=False, prompt="",
            mask_weight_type=self.mask_weight_type, siglip_processor=self.siglip_processor,
            need_weight='false', resize_mode=self.resize_mode,
        )
        image_token_lengths  = image_dict['image_token_lengths']
        pixel_values         = image_dict['pixel_values']
        image_grid_thw       = image_dict['image_grid_thw']
        pil_pixel_values     = image_dict['pil_pixel_values']     # input PIL image for external 5-line generation or visualisation
        siglip_pixel_values  = image_dict['siglip_pixel_values']
        weights              = image_dict['weights']

        input_ids, labels, image_position = self._process_image_token(
            input_ids,
            labels=labels,
            image_token_id=self.image_token_id,
            image_begin_token_id=self.image_begin_token_id,
            image_end_token_id=self.image_end_token_id,
            image_token_lengths=image_token_lengths,
        )

        # VAE input
        img_np = np.array(input_image)
        img_tensor = torch.from_numpy(img_np).float().div(255.0)
        img_tensor = rearrange(img_tensor, "h w c -> c h w")
        vae_pixel_values = self.vae_transform(img_tensor).unsqueeze(0).to(torch.bfloat16)

        ret = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "prompt": "",                          # 5-line prompt is not generated inside the Dataset
            "pil_pixel_values": pil_pixel_values,  # [PIL(input)]
            "siglip_pixel_values": siglip_pixel_values,
            "vae_pixel_values": vae_pixel_values,
            "weights": weights,
            "generated_image": [],
            "input_path": input_path,
            "gt_path": gt_path,
            "weather_type": self.weather_type,
            "pred_task": None,                     # can optionally be generated and recorded externally
        }

        if has_generated_image:
            g = torch.tensor(np.array(gt_image)) / 255.0
            g = rearrange(g, "h w c -> c h w")
            ret["generated_image"] = self.transform(g)
        return ret

    # ---- Static helpers (forward resize_mode through) ----
    @staticmethod
    def _load_image(
        image_slice: List,
        max_pixels: int = 512*512,
        min_pixels: int = 512*512,
        processor: Callable = None,
        image_processor: Callable = None,
        image_token_lengths: int = 729,
        image_token: str = '<|image_pad|>',
        factor: int = 1,
        last_image: Optional = None,
        vae_image_transform: Callable = None,
        drop_prompt: bool = False,
        prompt: str = '',
        mask_weight_type: str = None,
        siglip_processor: Callable = None,
        need_weight: str = 'false',
        resize_mode: str = "smart",
    ):
        image_token_lengths = []
        pixel_values = []
        image_grid_thw = []
        pil_pixel_values = []
        siglip_pixel_values = []
        for image_path in image_slice:
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            image_inputs, video_inputs = process_vision_info([vision_infos], resize_mode=resize_mode, factor=factor)
            inputs = processor(text=[f'dummy {image_token}'], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)).sum()
                image_token_lengths.append(image_token_length)
            pil_pixel_values.append(image_inputs[0])
            if siglip_processor is not None:
                if isinstance(image_path, str):
                    siglip_input = Image.open(image_path).convert('RGB')
                else:
                    siglip_input = image_path
                siglip_pixel_values = siglip_processor(images=siglip_input, return_tensors="pt").pixel_values

        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)
        weights = []  # weights are not needed during testing
        return {
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'image_token_lengths': image_token_lengths,
            'pil_pixel_values': pil_pixel_values,
            'siglip_pixel_values': siglip_pixel_values,
            'weights': weights,
        }

    @staticmethod
    def _process_image_token(
        input_ids: torch.Tensor,
        image_token_id: int,
        image_begin_token_id: int,
        image_end_token_id: int,
        image_token_lengths: List[int],
        labels: Optional[torch.Tensor] = None,
    ):
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        image_position = []
        offset = 0
        cur_i = 0
        if isinstance(image_token_lengths, int):
            image_token_lengths = [image_token_lengths] * len(image_token_indices[1])
        for idx in image_token_indices[1]:
            image_token_length = image_token_lengths[cur_i]
            adjusted_idx = idx + offset
            assert input_ids[0, adjusted_idx] == image_token_id

            input_ids = torch.cat(
                [
                    input_ids[:, :adjusted_idx],
                    input_ids.new_full((1, 1), image_begin_token_id),
                    input_ids.new_full((1, image_token_length), image_token_id),
                    input_ids.new_full((1, 1), image_end_token_id),
                    input_ids[:, adjusted_idx + 1 :],
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [
                        labels[:, :adjusted_idx],
                        labels.new_full((1, 1), image_begin_token_id),
                        labels.new_full((1, image_token_length), -100),
                        labels.new_full((1, 1), -100),
                        labels[:, adjusted_idx + 1 :],
                    ],
                    dim=1,
                )
            adjusted_idx += 1
            image_position.append(adjusted_idx.item())
            offset += image_token_length - 1
            offset += 2
            cur_i += 1
        return input_ids, labels, image_position


# =========================================================
#      MultiWeatherTestDatasets (no 5-line generation)
# =========================================================

class MultiWeatherTestDatasets:
    """
    Manages multiple weather test datasets (including Snow100K); applies direct
    resize without cropping; does not generate 5-line prompts inside the datasets.
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        data_root: str,
        weather_types: List[str] = ["multi_haze_snow", "multi_haze_rain", "multi_low_rain", "multi_low_snow", "multi_low_haze_rain", "multi_low_haze_snow",
                                    "weather1", "weather2", "Snow100K-L", "Snow100K-S", "Snow100K-L-ori", "Snow100K-S-ori", "Rain100L", "RainTrainL", "Rain100_H_test", "Rain100_L_test"],
        output_dir: Optional[str] = None,
        naming_patterns: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs
    ):
        self.data_root = data_root
        self.weather_types = weather_types
        self.output_dir = output_dir
        self.naming_patterns = naming_patterns
        self.kwargs = kwargs

        default_naming_patterns = {
                "weather1": {"input_suffix": "_rain","gt_suffix": "_clean"},
                "weather2": {"input_suffix": "_input","gt_suffix": "_gt"},
                "Snow100K-L": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-S": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-L-ori": {"input_suffix": "","gt_suffix": ""},
                "Snow100K-S-ori": {"input_suffix": "","gt_suffix": ""},
                "Rain100L": {"input_suffix": "","gt_suffix": ""},
                "RainTrainL": {"input_suffix": "","gt_suffix": ""},
                "Rain100_H_test": {"input_suffix": "","gt_suffix": ""},
                "Rain100_L_test": {"input_suffix": "","gt_suffix": ""},
                "multi_haze_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_haze_snow": {"input_suffix": "","gt_suffix": ""},
                "multi_low_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_low_snow": {"input_suffix": "","gt_suffix": ""},
                "multi_low_haze_rain": {"input_suffix": "","gt_suffix": ""},
                "multi_low_haze_snow": {"input_suffix": "","gt_suffix": ""}
            }
        self.naming_patterns = naming_patterns or default_naming_patterns

        self.datasets: Dict[str, WeatherTestDataset] = {}
        self.dataset_info: Dict[str, dict] = {}

        for wt in weather_types:
            folder = os.path.join(data_root, wt)
            if not os.path.exists(folder):
                print(f"⚠️  {wt} folder not found: {folder}")
                continue
            try:
                dataset = WeatherTestDataset(
                    lvlm_model=kwargs.get("lvlm_model", lvlm_model),
                    tokenizer=kwargs.get("tokenizer", tokenizer),
                    data_root=data_root,
                    weather_type=wt,
                    output_dir=output_dir,
                    naming_patterns=self.naming_patterns,
                    transform=kwargs["transform"],
                    prompter=kwargs.get("prompter") or PROMPT_TYPE['qwen2p5vl'](),
                    **{k: v for k, v in kwargs.items() if k not in ("transform","prompter","lvlm_model","tokenizer")}
                )
                self.datasets[wt] = dataset
                self.dataset_info[wt] = {
                    'weather_type': wt,
                    'num_samples': len(dataset),
                    'input_folder': dataset.input_folder,
                    'gt_folder': dataset.gt_folder,
                    'naming_pattern': self.naming_patterns.get(wt, {}),
                    'target_size': dataset.target_size,
                }
                print(f"✅ Successfully loaded {wt}: {len(dataset)} samples")
            except Exception as e:
                print(f"❌ Failed to load {wt}: {e}")

    def get_dataset(self, weather_type: str) -> Optional[WeatherTestDataset]:
        return self.datasets.get(weather_type)

    def get_all_datasets(self) -> Dict[str, WeatherTestDataset]:
        return self.datasets

    def get_dataset_names(self) -> List[str]:
        return list(self.datasets.keys())

    def get_dataset_info(self) -> Dict[str, dict]:
        return self.dataset_info

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, key: str):
        return self.get_dataset(key)

    def __iter__(self):
        return iter(self.datasets.items())

    def print_summary(self):
        print("\n" + "="*80)
        print("🌦️❄️  MULTI-WEATHER TEST DATASETS SUMMARY (NO IN-DATASET PROMPT)")
        print("="*80)
        total = 0
        for wt, info in self.dataset_info.items():
            naming = info.get('naming_pattern', {})
            ins, gts = naming.get('input_suffix','N/A'), naming.get('gt_suffix','N/A')
            target = info.get('target_size', 512)
            icon = "❄️ " if wt.startswith("Snow100K") else ("🌧️ " if wt=="weather1" else "🌦️ ")
            print(f"{icon}{wt:>12} Dataset: {info['num_samples']:>4} samples")
            if ins == "" and gts == "":
                print(f"     🏷️  Naming: Same filename in input/gt")
            else:
                print(f"     🏷️  Naming: *{ins}.ext -> *{gts}.ext")
            print(f"     📐  Target Size: {target}x{target} (resize)")
            total += info['num_samples']
        print("-"*80)
        print(f"📈 Total Datasets: {len(self.datasets)} | Total Samples: {total}")
        print(f"📁 Data Root: {self.data_root}")
        if self.output_dir: print(f"💾 Output Dir: {self.output_dir}")
        print(f"🔧 Processing Mode: Direct Resize (No Cropping, No In-Dataset Prompt)")
        print("="*80 + "\n")


# ---------------------------
# Factory
# ---------------------------

def create_multi_weather_datasets(
    lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    data_root: str,
    weather_types: List[str] = ["multi_haze_snow", "multi_haze_rain", "multi_low_rain", "multi_low_snow", "multi_low_haze_rain", "multi_low_haze_snow",
                                "weather1", "weather2", "Snow100K-L", "Snow100K-S", "Snow100K-L-ori", "Snow100K-S-ori", "Rain100L", "RainTrainL", "Rain100_H_test", "Rain100_L_test"],
    output_dir: Optional[str] = None,
    naming_patterns: Optional[Dict[str, Dict[str, str]]] = None,
    **kwargs
) -> MultiWeatherTestDatasets:
    """
    Instantiate multiple weather test datasets
    (5-line expert prompts are not built inside the datasets).
    """
    return MultiWeatherTestDatasets(
        lvlm_model=lvlm_model,
        tokenizer=tokenizer,
        data_root=data_root,
        weather_types=weather_types,
        output_dir=output_dir,
        naming_patterns=naming_patterns,
        **kwargs
    )