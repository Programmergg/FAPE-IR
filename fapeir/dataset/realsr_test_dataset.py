import os
import torch
import random
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from einops import rearrange
from typing import Callable, List, Optional, Tuple, Dict, Any, Union

from torchvision import transforms
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# ============== Project-level dependencies ==============
from fapeir.utils.anyres_util import dynamic_resize
from fapeir.utils.prompter import Prompter, PROMPT_TYPE
from fapeir.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from qwen_vl_utils.vision_process import to_rgb, smart_resize, fetch_video
from fapeir.models.qwen2p5vl.modeling_fapeir_qwen2p5vl_moe import FAPEIRQwen2p5VLForConditionalGeneration


# ============== Image loading (with resize_mode switch) ==============
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

    # smart_resize
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
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
    return image


def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1,
    resize_mode: str = "smart",
) -> tuple:
    """
    Convert vision_infos into images/videos accepted by the processor.
    """
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
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


# =============================================================================
#                        Dataset (generic multi-scale SR)
# =============================================================================
class Qwen2VLTestDataset(Dataset):
    """
    Generic multi-scale SR validation dataset
    (directory layout: data_root/Test_x{scale}/test_LR, test_HR).
    - Does not build 5-line prompts inside the Dataset;
    - Only constructs a minimal conversation: <image> -> <gen_image>;
    - Retains the LR PIL image (for 5-line text generation at validation time);
    - Packages tensors required by VAE / SigLIP.
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        scale_factor: int,
        transform: Callable,
        prompter: Prompter,
        output_dir: Optional[str] = None,
        dataset_type: str = "qwen2vl",
        min_pixels: int = 384 * 384,
        max_pixels: int = 384 * 384,
        image_token_length: int = 729,
        mask_weight_type: str = "log",
        siglip_processor: Callable = None,
        seed: int = 42,
        resize_mode: str = "none",
    ):
        assert dataset_type in ("qwen2vl", "qwen2p5vl")
        assert scale_factor in [2, 3, 4], "scale_factor must be 2, 3, or 4"

        self.data_root = data_root
        self.scale_factor = scale_factor
        self.transform = transform
        self.resize_mode = resize_mode
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
        self.image_token = SPACIAL_TOKEN[dataset_type]["image_token"]
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]["image_begin_token"]
        self.image_end_token = SPACIAL_TOKEN[dataset_type]["image_end_token"]
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = self.processor.image_processor
        self.factor = 2
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.seed = seed
        self.output_dir = output_dir

        self.test_folder = os.path.join(data_root, f"Test_x{scale_factor}")
        self.lr_folder = os.path.join(self.test_folder, "test_LR")
        self.hr_folder = os.path.join(self.test_folder, "test_HR")

        if self.output_dir:
            self.output_lr_dir = os.path.join(output_dir, f"cropped_x{scale_factor}", "LR")
            self.output_hr_dir = os.path.join(output_dir, f"cropped_x{scale_factor}", "HR")
            os.makedirs(self.output_lr_dir, exist_ok=True)
            os.makedirs(self.output_hr_dir, exist_ok=True)

        if not os.path.exists(self.lr_folder):
            raise ValueError(f"LR folder not found: {self.lr_folder}")
        if not os.path.exists(self.hr_folder):
            raise ValueError(f"HR folder not found: {self.hr_folder}")

        # Fixed crop settings: HR is always 512, LR is scaled accordingly
        self.crop_sizes = {
            2: {"lr": 256, "hr": 512},
            3: {"lr": 168, "hr": 512},
            4: {"lr": 128, "hr": 512},
        }

        # Special tokens
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(self.image_begin_token)
        assert isinstance(self.image_begin_token_id, int)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        assert isinstance(self.image_end_token_id, int)

        # Load paired samples
        self.image_pairs = self._load_image_pairs()

    # ---------- I/O ----------
    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        lr_files = {}
        for file in sorted(os.listdir(self.lr_folder)):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                name_without_ext = os.path.splitext(file)[0]
                if name_without_ext.endswith("_x1"):
                    base_name = name_without_ext[:-3]
                else:
                    base_name = name_without_ext
                lr_files[base_name] = os.path.join(self.lr_folder, file)

        hr_files = {}
        expected_suffix = f"_x{self.scale_factor}"
        for file in sorted(os.listdir(self.hr_folder)):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                name_without_ext = os.path.splitext(file)[0]
                if name_without_ext.endswith(expected_suffix):
                    base_name = name_without_ext[: -len(expected_suffix)]
                else:
                    base_name = name_without_ext
                hr_files[base_name] = os.path.join(self.hr_folder, file)

        image_pairs = []
        matched_count = 0
        for base_name in sorted(lr_files.keys()):
            if base_name in hr_files:
                image_pairs.append((lr_files[base_name], hr_files[base_name]))
                matched_count += 1

        print(f"Found {matched_count} matching image pairs for {self.scale_factor}x restoration in {self.test_folder}")
        if matched_count == 0:
            print("No matches found with naming convention, trying direct filename matching...")
            lr_names = {
                os.path.splitext(f)[0]
                for f in os.listdir(self.lr_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            }
            hr_names = {
                os.path.splitext(f)[0]
                for f in os.listdir(self.hr_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            }
            common_names = lr_names.intersection(hr_names)
            for name in common_names:
                lr_file = os.path.join(self.lr_folder, name + ".png")
                if not os.path.exists(lr_file):
                    for lf in os.listdir(self.lr_folder):
                        if os.path.splitext(lf)[0] == name:
                            lr_file = os.path.join(self.lr_folder, lf)
                            break
                hr_file = os.path.join(self.hr_folder, name + ".png")
                if not os.path.exists(hr_file):
                    for hf in os.listdir(self.hr_folder):
                        if os.path.splitext(hf)[0] == name:
                            hr_file = os.path.join(self.hr_folder, hf)
                            break
                if os.path.exists(lr_file) and os.path.exists(hr_file):
                    image_pairs.append((lr_file, hr_file))
            print(f"Fallback matching found {len(image_pairs)} pairs")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    # ---------- Cropping ----------
    def _get_consistent_crop_coords(
        self, lr_image: Image.Image, hr_image: Image.Image, index: int
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Map a random LR crop to a corresponding HR crop via relative position,
        so that both crops align at their respective scales.
        """
        lr_crop_size = self.crop_sizes[self.scale_factor]["lr"]
        hr_crop_size = self.crop_sizes[self.scale_factor]["hr"]
        lr_width, lr_height = lr_image.size
        hr_width, hr_height = hr_image.size
        rng = random.Random(self.seed + index)

        # Upscale LR if it is smaller than the required crop size
        if lr_width < lr_crop_size or lr_height < lr_crop_size:
            scale = max(lr_crop_size / lr_width, lr_crop_size / lr_height)
            new_lr_width = int(lr_width * scale)
            new_lr_height = int(lr_height * scale)
            lr_image = lr_image.resize((new_lr_width, new_lr_height), Image.Resampling.BICUBIC)
            lr_width, lr_height = lr_image.size

        max_lr_left = max(0, lr_width - lr_crop_size)
        max_lr_top = max(0, lr_height - lr_crop_size)
        lr_left = rng.randint(0, max_lr_left) if max_lr_left > 0 else 0
        lr_top = rng.randint(0, max_lr_top) if max_lr_top > 0 else 0

        lr_rel_x = lr_left / max(lr_width - lr_crop_size, 1)
        lr_rel_y = lr_top / max(lr_height - lr_crop_size, 1)

        max_hr_left = max(0, hr_width - hr_crop_size)
        max_hr_top = max(0, hr_height - hr_crop_size)
        hr_left = int(lr_rel_x * max_hr_left)
        hr_top = int(lr_rel_y * max_hr_top)

        return (lr_left, lr_top, lr_left + lr_crop_size, lr_top + lr_crop_size), \
               (hr_left, hr_top, hr_left + hr_crop_size, hr_top + hr_crop_size)

    def _crop_image_pair(self, lr_path: str, hr_path: str, index: int) -> Tuple[Image.Image, Image.Image]:
        """
        Consistently crop the LR/HR pair; optionally save cropped results to output_dir.
        """
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")
        lr_coords, hr_coords = self._get_consistent_crop_coords(lr_image, hr_image, index)

        lr_cropped = lr_image.crop(lr_coords)
        hr_cropped = hr_image.crop(hr_coords)

        lr_crop_size = self.crop_sizes[self.scale_factor]["lr"]
        hr_crop_size = self.crop_sizes[self.scale_factor]["hr"]
        if lr_cropped.size != (lr_crop_size, lr_crop_size):
            lr_cropped = lr_cropped.resize((lr_crop_size, lr_crop_size), Image.Resampling.BICUBIC)
        if hr_cropped.size != (hr_crop_size, hr_crop_size):
            hr_cropped = hr_cropped.resize((hr_crop_size, hr_crop_size), Image.Resampling.BICUBIC)

        if self.output_dir:
            base_name = os.path.splitext(os.path.basename(lr_path))[0]
            crop_suffix = f"_crop_{index:04d}"
            lr_save_path = os.path.join(self.output_lr_dir, f"{base_name}{crop_suffix}.png")
            hr_save_path = os.path.join(self.output_hr_dir, f"{base_name}{crop_suffix}.png")
            lr_cropped.save(lr_save_path)
            hr_cropped.save(hr_save_path)

        return lr_cropped, hr_cropped

    # ---------- Visual preprocessing wrapper ----------
    @staticmethod
    def _load_image(
        image_slice: List,
        max_pixels: int = 512 * 512,
        min_pixels: int = 512 * 512,
        processor: Callable = None,
        image_processor: Callable = None,
        image_token_lengths: int = 729,
        image_token: str = "<|image_pad|>",
        factor: int = 1,
        last_image: Optional = None,
        vae_image_transform: Callable = None,
        drop_prompt: bool = False,
        prompt: str = "",
        mask_weight_type: str = None,
        siglip_processor: Callable = None,
        need_weight: str = "false",
        resize_mode: str = "smart",
    ):
        """
        Visual preprocessing wrapper consistent with training;
        returns pixel_values / image_grid_thw / PIL images, etc.
        """
        pixel_values = []
        image_grid_thw = []
        image_token_lengths = []
        pil_pixel_values = []
        siglip_pixel_values = []

        for image_path in image_slice:
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            image_inputs, video_inputs = process_vision_info(
                [vision_infos], resize_mode=resize_mode, factor=factor
            )
            inputs = processor(
                text=[f"dummy {image_token}"],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (
                    inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)
                ).sum()
                image_token_lengths.append(image_token_length)
            pil_pixel_values.append(image_inputs[0])
            if siglip_processor is not None:
                siglip_input = image_path if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
                siglip_pixel_values = siglip_processor(images=siglip_input, return_tensors="pt").pixel_values

        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)
        weights = []
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_token_lengths": image_token_lengths,
            "pil_pixel_values": pil_pixel_values,
            "siglip_pixel_values": siglip_pixel_values,
            "weights": weights,
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
        """
        Replace each <image_token> placeholder with
        <image_begin_token> + N × <image_token> + <image_end_token>
        and record the resulting position indices (image_position).
        """
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

    # ---------- Core: minimal conversation + visual tensors ----------
    def __getitem__(self, idx):
        lr_path, hr_path = self.image_pairs[idx]
        lr_image, hr_image = self._crop_image_pair(lr_path, hr_path, idx)

        # Minimal conversation only: <image> -> <gen_image>; no 5-line prompt
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
        for item in prompt_list:
            item["prompt"] = item["prompt"].replace("<image>", self.image_token)
            if GENERATE_TOKEN in item["prompt"]:
                # Replace <gen_image><eos> with <image_begin_token>
                assert item["from"] == self.prompter.assistant_role
                assert f"{GENERATE_TOKEN}{self.prompter.eos_token}" in item["prompt"]
                item["prompt"] = item["prompt"].replace(
                    f"{GENERATE_TOKEN}{self.prompter.eos_token}", self.image_begin_token
                )
                has_generated_image = True
            tokenized_item = self.tokenizer(
                item["prompt"], return_tensors="pt", truncation=True, max_length=1024,
            )
            if item.get("is_labels", False):
                labels.append(tokenized_item.input_ids)
            else:
                labels.append(torch.full_like(tokenized_item.input_ids, -100))
            input_ids.append(tokenized_item.input_ids)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # Visual processing (LR image as input)
        image_slice = [lr_image]
        image_dict = self._load_image(
            image_slice,
            self.max_pixels,
            self.min_pixels,
            processor=self.processor,
            image_token=self.image_token,
            factor=self.factor,
            last_image=hr_image,
            vae_image_transform=self.vae_transform,
            drop_prompt=False,
            prompt="",
            mask_weight_type=self.mask_weight_type,
            siglip_processor=self.siglip_processor,
            need_weight="false",
            resize_mode=self.resize_mode,
        )
        image_token_lengths = image_dict["image_token_lengths"]
        pixel_values = image_dict["pixel_values"]
        image_grid_thw = image_dict["image_grid_thw"]
        pil_pixel_values = image_dict["pil_pixel_values"]     # LR PIL image for 5-line generation at validation time
        siglip_pixel_values = image_dict["siglip_pixel_values"]
        weights = image_dict["weights"]

        input_ids, labels, image_position = self._process_image_token(
            input_ids,
            labels=labels,
            image_token_id=self.image_token_id,
            image_begin_token_id=self.image_begin_token_id,
            image_end_token_id=self.image_end_token_id,
            image_token_lengths=image_token_lengths,
        )

        # VAE: use LR image as input
        img_np = np.array(lr_image)
        img_tensor = torch.from_numpy(img_np).float().div(255.0)
        img_tensor = rearrange(img_tensor, "h w c -> c h w")
        vae_pixel_values = self.vae_transform(img_tensor).unsqueeze(0).to(torch.bfloat16)

        # Assemble return dict
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "prompt": "",                           # empty string; 5-line prompt is generated externally at validation time
            "pil_pixel_values": pil_pixel_values,   # [PIL(LR)]
            "siglip_pixel_values": siglip_pixel_values,
            "vae_pixel_values": vae_pixel_values,
            "weights": weights,
        }

        # GT
        if has_generated_image:
            image_tensor = torch.tensor(np.array(hr_image)) / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []

        # Metadata
        return_data["lr_path"] = lr_path
        return_data["hr_path"] = hr_path
        return_data["scale_factor"] = self.scale_factor
        return_data["pred_task"] = None  # generated and recorded at validation time
        return return_data


class MultiScaleTestDatasets:
    """
    Wrapper managing multi-scale (2×/3×/4×) generic SR test datasets
    (5-line prompts are not generated inside the datasets).
    Directory layout: data_root/Test_x{scale}/test_LR, test_HR
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        scales: List[int] = [2, 3, 4],
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        self.data_root = data_root
        self.scales = scales
        self.output_dir = output_dir
        self.kwargs = kwargs

        self.datasets: Dict[str, Qwen2VLTestDataset] = {}
        self.dataset_info: Dict[str, dict] = {}

        for scale in scales:
            test_folder = os.path.join(data_root, f"Test_x{scale}")
            if os.path.exists(test_folder):
                try:
                    prompter = kwargs.get("prompter") or PROMPT_TYPE["qwen2p5vl"]()
                    transform = kwargs.get("transform")
                    assert transform is not None, "Please pass `transform` in kwargs."

                    dataset = Qwen2VLTestDataset(
                        lvlm_model=lvlm_model,
                        tokenizer=tokenizer,
                        data_root=data_root,
                        scale_factor=scale,
                        output_dir=output_dir,
                        transform=transform,
                        prompter=prompter,
                        **{k: v for k, v in kwargs.items() if k not in ("prompter", "transform")}
                    )
                    self.datasets[f"{scale}x"] = dataset
                    self.dataset_info[f"{scale}x"] = {
                        "scale_factor": scale,
                        "num_samples": len(dataset),
                        "lr_folder": dataset.lr_folder,
                        "hr_folder": dataset.hr_folder,
                    }
                    print(f"✅ Successfully loaded {scale}x dataset: {len(dataset)} samples")
                except Exception as e:
                    print(f"❌ Failed to load {scale}x dataset: {e}")
            else:
                print(f"⚠️  {scale}x test folder not found: {test_folder}")

    def get_dataset(self, scale: str) -> Optional[Qwen2VLTestDataset]:
        return self.datasets.get(scale)

    def get_all_datasets(self) -> Dict[str, Qwen2VLTestDataset]:
        return self.datasets

    def get_dataset_names(self) -> List[str]:
        return list(self.datasets.keys())

    def get_dataset_info(self) -> Dict[str, dict]:
        return self.dataset_info

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, scale: str):
        return self.get_dataset(scale)

    def __iter__(self):
        return iter(self.datasets.items())

    def print_summary(self):
        print("\n" + "=" * 60)
        print("🗂️  MULTI-SCALE TEST DATASETS SUMMARY")
        print("=" * 60)
        total_samples = 0
        for scale_name, info in self.dataset_info.items():
            print(
                f"📊 {scale_name:>4} Dataset: {info['num_samples']:>4} samples | "
                f"Scale Factor: {info['scale_factor']}x"
            )
            total_samples += info["num_samples"]
        print("-" * 60)
        print(f"📈 Total Datasets: {len(self.datasets)} | Total Samples: {total_samples}")
        print(f"📁 Data Root: {self.data_root}")
        if self.output_dir:
            print(f"💾 Output Dir: {self.output_dir}")
        print("=" * 60 + "\n")


def create_multi_scale_datasets(
    lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
    tokenizer: Optional[PreTrainedTokenizer],
    data_root: str,
    scales: List[int] = [2, 3, 4],
    output_dir: Optional[str] = None,
    **kwargs,
) -> MultiScaleTestDatasets:
    return MultiScaleTestDatasets(
        lvlm_model=lvlm_model,
        tokenizer=tokenizer,
        data_root=data_root,
        scales=scales,
        output_dir=output_dir,
        **kwargs,
    )


# =============================================================================
#                        Dataset (RealSR camera / scale)
# =============================================================================
class RealSRTestDataset(Dataset):
    """
    RealSR test dataset.
    Directory layout: data_root/<Camera>/<Scale>/
    File naming convention: *_HR.png / *_LR{scale}.png
    - Does not build 5-line prompts inside the Dataset;
    - Only constructs a minimal conversation: <image> -> <gen_image>;
    - Centre-crops LR/HR (HR = target_size, LR = target_size // scale);
    - Retains the LR PIL image (for 5-line text generation at validation time).
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        transform: Callable,
        prompter: Prompter,
        camera_types: List[str] = ("Canon", "Nikon"),
        scale_factors: List[int] = (2, 4),
        output_dir: Optional[str] = None,
        dataset_type: str = "qwen2vl",
        min_pixels: int = 384 * 384,
        max_pixels: int = 384 * 384,
        image_token_length: int = 729,
        mask_weight_type: str = "log",
        siglip_processor: Callable = None,
        seed: int = 42,
        resize_mode: str = "none",
        target_size: int = 512,
    ):
        assert dataset_type in ("qwen2vl", "qwen2p5vl")

        self.data_root = data_root
        self.camera_types = list(camera_types)
        self.scale_factors = list(scale_factors)
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
        self.image_token = SPACIAL_TOKEN[dataset_type]["image_token"]
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]["image_begin_token"]
        self.image_end_token = SPACIAL_TOKEN[dataset_type]["image_end_token"]
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = self.processor.image_processor
        self.factor = 2
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.seed = seed
        self.output_dir = output_dir

        # Pre-compute crop sizes per scale (HR = target_size, LR = target_size // scale)
        self.crop_sizes = {s: {"hr": target_size, "lr": target_size // s} for s in self.scale_factors}

        # Special tokens
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(self.image_begin_token)
        assert isinstance(self.image_begin_token_id, int)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        assert isinstance(self.image_end_token_id, int)

        # Load paired samples
        self.image_pairs = self._load_image_pairs()

    # ---------- RealSR I/O ----------
    def _load_image_pairs(self) -> List[Tuple[str, str, str, int]]:
        image_pairs = []
        for camera in self.camera_types:
            camera_dir = os.path.join(self.data_root, camera)
            if not os.path.exists(camera_dir):
                print(f"Warning: Camera directory not found: {camera_dir}")
                continue
            for scale in self.scale_factors:
                scale_dir = os.path.join(camera_dir, str(scale))
                if not os.path.exists(scale_dir):
                    print(f"Warning: Scale directory not found: {scale_dir}")
                    continue
                files = os.listdir(scale_dir)
                hr_files = [f for f in files if f.endswith("_HR.png")]
                lr_files = [f for f in files if f.endswith(f"_LR{scale}.png")]

                hr_dict = {f.replace("_HR.png", ""): os.path.join(scale_dir, f) for f in hr_files}
                lr_dict = {f.replace(f"_LR{scale}.png", ""): os.path.join(scale_dir, f) for f in lr_files}

                matched_count = 0
                for base_name in sorted(hr_dict.keys()):
                    if base_name in lr_dict:
                        image_pairs.append((lr_dict[base_name], hr_dict[base_name], camera, scale))
                        matched_count += 1
                print(f"Found {matched_count} matching pairs for {camera} {scale}x in {scale_dir}")
        print(f"Total RealSR image pairs loaded: {len(image_pairs)}")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    # ---------- Centre crop ----------
    def _get_center_crop_coords(self, image: Image.Image, crop_size: int) -> Tuple[int, int, int, int]:
        width, height = image.size
        left = max(0, (width - crop_size) // 2)
        top = max(0, (height - crop_size) // 2)
        right = min(width, left + crop_size)
        bottom = min(height, top + crop_size)
        return (left, top, right, bottom)

    def _crop_image_pair(self, lr_path: str, hr_path: str, scale_factor: int, index: int) -> Tuple[Image.Image, Image.Image]:
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")
        hr_crop_size = self.crop_sizes[scale_factor]["hr"]
        lr_crop_size = self.crop_sizes[scale_factor]["lr"]

        # Centre-crop the HR image
        hr_coords = self._get_center_crop_coords(hr_image, hr_crop_size)
        hr_cropped = hr_image.crop(hr_coords)

        # Map the HR crop region to LR scale
        hr_left, hr_top, hr_right, hr_bottom = hr_coords
        hr_width, hr_height = hr_image.size
        lr_width, lr_height = lr_image.size
        width_ratio = lr_width / hr_width
        height_ratio = lr_height / hr_height

        lr_left = int(hr_left * width_ratio)
        lr_top = int(hr_top * height_ratio)
        lr_right = int(hr_right * width_ratio)
        lr_bottom = int(hr_bottom * height_ratio)

        # Adjust to the exact lr_crop_size
        if (lr_right - lr_left) != lr_crop_size:
            cx = (lr_left + lr_right) // 2
            lr_left = cx - lr_crop_size // 2
            lr_right = lr_left + lr_crop_size
        if (lr_bottom - lr_top) != lr_crop_size:
            cy = (lr_top + lr_bottom) // 2
            lr_top = cy - lr_crop_size // 2
            lr_bottom = lr_top + lr_crop_size

        lr_left = max(0, min(lr_left, lr_width - lr_crop_size))
        lr_top = max(0, min(lr_top, lr_height - lr_crop_size))
        lr_right = lr_left + lr_crop_size
        lr_bottom = lr_top + lr_crop_size
        lr_coords = (lr_left, lr_top, lr_right, lr_bottom)
        lr_cropped = lr_image.crop(lr_coords)

        if hr_cropped.size != (hr_crop_size, hr_crop_size):
            hr_cropped = hr_cropped.resize((hr_crop_size, hr_crop_size), Image.Resampling.BICUBIC)
        if lr_cropped.size != (lr_crop_size, lr_crop_size):
            lr_cropped = lr_cropped.resize((lr_crop_size, lr_crop_size), Image.Resampling.BICUBIC)

        if self.output_dir:
            lr_path_obj = os.path.basename(lr_path)
            base_name = lr_path_obj.replace(f"_LR{scale_factor}.png", "")
            camera_type = self.image_pairs[index][2]
            lr_save_dir = os.path.join(self.output_dir, f"RealSR_cropped_{camera_type}_x{scale_factor}", "LR")
            hr_save_dir = os.path.join(self.output_dir, f"RealSR_cropped_{camera_type}_x{scale_factor}", "HR")
            os.makedirs(lr_save_dir, exist_ok=True)
            os.makedirs(hr_save_dir, exist_ok=True)
            crop_suffix = f"_crop_{index:04d}"
            lr_cropped.save(os.path.join(lr_save_dir, f"{base_name}{crop_suffix}.png"))
            hr_cropped.save(os.path.join(hr_save_dir, f"{base_name}{crop_suffix}.png"))

        return lr_cropped, hr_cropped

    # ---------- Visual preprocessing / packaging ----------
    @staticmethod
    def _load_image(
        image_slice: List,
        max_pixels: int = 512 * 512,
        min_pixels: int = 512 * 512,
        processor: Callable = None,
        image_processor: Callable = None,
        image_token_lengths: int = 729,
        image_token: str = "<|image_pad|>",
        factor: int = 1,
        last_image: Optional = None,
        vae_image_transform: Callable = None,
        drop_prompt: bool = False,
        prompt: str = "",
        mask_weight_type: str = None,
        siglip_processor: Callable = None,
        need_weight: str = "false",
        resize_mode: str = "smart",
    ):
        pixel_values = []
        image_grid_thw = []
        image_token_lengths = []
        pil_pixel_values = []
        siglip_pixel_values = []

        for image_path in image_slice:
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            image_inputs, video_inputs = process_vision_info([vision_infos], resize_mode=resize_mode, factor=factor)
            inputs = processor(
                text=[f"dummy {image_token}"],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (
                    inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)
                ).sum()
                image_token_lengths.append(image_token_length)
            pil_pixel_values.append(image_inputs[0])
            if siglip_processor is not None:
                siglip_input = image_path if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
                siglip_pixel_values = siglip_processor(images=siglip_input, return_tensors="pt").pixel_values

        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)
        weights = []
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_token_lengths": image_token_lengths,
            "pil_pixel_values": pil_pixel_values,
            "siglip_pixel_values": siglip_pixel_values,
            "weights": weights,
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

    def __getitem__(self, idx):
        lr_path, hr_path, camera_type, scale_factor = self.image_pairs[idx]
        lr_image, hr_image = self._crop_image_pair(lr_path, hr_path, scale_factor, idx)

        # Minimal conversation only: <image> -> <gen_image>; no 5-line prompt
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
        for item in prompt_list:
            item["prompt"] = item["prompt"].replace("<image>", self.image_token)
            if GENERATE_TOKEN in item["prompt"]:
                assert item["from"] == self.prompter.assistant_role
                assert f"{GENERATE_TOKEN}{self.prompter.eos_token}" in item["prompt"]
                item["prompt"] = item["prompt"].replace(
                    f"{GENERATE_TOKEN}{self.prompter.eos_token}", self.image_begin_token
                )
                has_generated_image = True
            tokenized_item = self.tokenizer(
                item["prompt"], return_tensors="pt", truncation=True, max_length=1024,
            )
            if item.get("is_labels", False):
                labels.append(tokenized_item.input_ids)
            else:
                labels.append(torch.full_like(tokenized_item.input_ids, -100))
            input_ids.append(tokenized_item.input_ids)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # Visual processing: LR image as input
        image_slice = [lr_image]
        image_dict = self._load_image(
            image_slice,
            self.max_pixels,
            self.min_pixels,
            processor=self.processor,
            image_token=self.image_token,
            factor=self.factor,
            last_image=hr_image,
            vae_image_transform=self.vae_transform,
            drop_prompt=False,
            prompt="",
            mask_weight_type=self.mask_weight_type,
            siglip_processor=self.siglip_processor,
            need_weight="false",
            resize_mode=self.resize_mode,
        )
        image_token_lengths = image_dict["image_token_lengths"]
        pixel_values = image_dict["pixel_values"]
        image_grid_thw = image_dict["image_grid_thw"]
        pil_pixel_values = image_dict["pil_pixel_values"]     # LR PIL image
        siglip_pixel_values = image_dict["siglip_pixel_values"]
        weights = image_dict["weights"]

        input_ids, labels, image_position = self._process_image_token(
            input_ids,
            labels=labels,
            image_token_id=self.image_token_id,
            image_begin_token_id=self.image_begin_token_id,
            image_end_token_id=self.image_end_token_id,
            image_token_lengths=image_token_lengths,
        )

        # VAE: use LR image as input
        img_np = np.array(lr_image)
        img_tensor = torch.from_numpy(img_np).float().div(255.0)
        img_tensor = rearrange(img_tensor, "h w c -> c h w")
        vae_pixel_values = self.vae_transform(img_tensor).unsqueeze(0).to(torch.bfloat16)

        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "prompt": "",                           # empty string; 5-line prompt is generated at validation time
            "pil_pixel_values": pil_pixel_values,   # [PIL(LR)]
            "siglip_pixel_values": siglip_pixel_values,
            "vae_pixel_values": vae_pixel_values,
            "weights": weights,
        }

        # GT
        if has_generated_image:
            image_tensor = torch.tensor(np.array(hr_image)) / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []

        # Metadata (for evaluation)
        return_data["lr_path"] = lr_path
        return_data["hr_path"] = hr_path
        return_data["camera_type"] = camera_type
        return_data["scale_factor"] = scale_factor
        return_data["pred_task"] = None
        return return_data


class MultiRealSRTestDatasets:
    """
    Wrapper managing multiple RealSR camera-type / scale test datasets
    (5-line prompts are not generated inside the datasets).
    Directory layout: data_root/<Camera>/<Scale>/
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        camera_types: List[str] = ("Canon", "Nikon"),
        scale_factors: List[int] = (2, 4),
        output_dir: Optional[str] = None,
        target_size: int = 512,
        **kwargs,
    ):
        self.data_root = data_root
        self.camera_types = list(camera_types)
        self.scale_factors = list(scale_factors)
        self.output_dir = output_dir
        self.target_size = target_size
        self.kwargs = kwargs

        self.datasets: Dict[str, RealSRTestDataset] = {}
        self.dataset_info: Dict[str, dict] = {}

        for camera in self.camera_types:
            for scale in self.scale_factors:
                dataset_key = f"{camera}_{scale}x"
                camera_dir = os.path.join(data_root, camera, str(scale))
                if os.path.exists(camera_dir):
                    try:
                        prompter = kwargs.get("prompter") or PROMPT_TYPE["qwen2p5vl"]()
                        transform = kwargs.get("transform")
                        assert transform is not None, "Please pass `transform` in kwargs."

                        dataset = RealSRTestDataset(
                            lvlm_model=lvlm_model,
                            tokenizer=tokenizer,
                            data_root=data_root,
                            transform=transform,
                            prompter=prompter,
                            camera_types=[camera],
                            scale_factors=[scale],
                            output_dir=output_dir,
                            target_size=target_size,
                            **{k: v for k, v in kwargs.items() if k not in ("prompter", "transform")}
                        )
                        if len(dataset) > 0:
                            self.datasets[dataset_key] = dataset
                            self.dataset_info[dataset_key] = {
                                "camera_type": camera,
                                "scale_factor": scale,
                                "num_samples": len(dataset),
                                "camera_dir": camera_dir,
                            }
                            print(f"✅ Successfully loaded {dataset_key} dataset: {len(dataset)} samples")
                        else:
                            print(f"⚠️  {dataset_key} dataset is empty")
                    except Exception as e:
                        print(f"❌ Failed to load {dataset_key} dataset: {e}")
                else:
                    print(f"⚠️  {dataset_key} camera directory not found: {camera_dir}")

    def get_dataset(self, key: str) -> Optional[RealSRTestDataset]:
        return self.datasets.get(key)

    def get_all_datasets(self) -> Dict[str, RealSRTestDataset]:
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
        print("\n" + "=" * 60)
        print("🗂️  MULTI-REALSR TEST DATASETS SUMMARY")
        print("=" * 60)
        total_samples = 0
        for key, info in self.dataset_info.items():
            print(
                f"📊 {key:>12} Dataset: {info['num_samples']:>4} samples | "
                f"Camera: {info['camera_type']} | Scale: {info['scale_factor']}x"
            )
            total_samples += info["num_samples"]
        print("-" * 60)
        print(f"📈 Total Datasets: {len(self.datasets)} | Total Samples: {total_samples}")
        print(f"📁 Data Root: {self.data_root}")
        print(f"🎯 Target HR Size: {self.target_size}x{self.target_size}")
        if self.output_dir:
            print(f"💾 Output Dir: {self.output_dir}")
        print("=" * 60 + "\n")


def create_multi_realsr_datasets(
    lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
    tokenizer: Optional[PreTrainedTokenizer],
    data_root: str,
    camera_types: List[str] = ("Canon", "Nikon"),
    scale_factors: List[int] = (2, 4),
    output_dir: Optional[str] = None,
    target_size: int = 512,
    **kwargs,
) -> MultiRealSRTestDatasets:
    """
    Instantiate multiple RealSR test datasets
    (5-line expert prompts are not built inside the datasets).
    """
    return MultiRealSRTestDatasets(
        lvlm_model=lvlm_model,
        tokenizer=tokenizer,
        data_root=data_root,
        camera_types=camera_types,
        scale_factors=scale_factors,
        output_dir=output_dir,
        target_size=target_size,
        **kwargs,
    )