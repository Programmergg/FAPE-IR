import os
import torch
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Callable, List, Optional, Tuple, Dict, Any

from einops import rearrange
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
    Load image / image_url and optionally apply smart_resize based on resize_mode.
    - resize_mode == "none": return the original image without resizing
    - otherwise: resample using qwen_vl_utils.smart_resize with pixel constraints
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
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    image = to_rgb(image_obj)

    if resize_mode == "none":
        return image

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
    return image


def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1,
    resize_mode: str = "smart",
) -> tuple:
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28*factor, resize_mode=resize_mode))
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
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs


# =========================================================
#          Generic test dataset (no 5-line generation inside Dataset)
# =========================================================

class GenericTestDataset(Dataset):
    """
    Generic base class for test datasets with 1:1 input/GT pairing.
    Compatible with the validation pipeline: 5-line generation is deferred
    to the inference/validation stage rather than being done inside the Dataset.

    Each returned sample aligns with ValidationCollator / validate_single_dataset:
      - input_ids, labels       (labels are placeholders for training compatibility;
                                 cross-entropy is not used during validation)
      - pixel_values, image_position, image_grid_thw  (VLM visual inputs)
      - vae_pixel_values        (image tensor normalised to [-1, 1] for VAE encoding)
      - siglip_pixel_values     (provided when siglip_processor is supplied)
      - generated_image         (GT image in [0, 1], shape C×H×W)
      - prompt                  (empty string; 5-line prompt is generated at validation time)
      - pil_pixel_values        ([PIL(input_image)], consumed by the validation stage)
      - input_path / gt_path / dataset_name / pred_task (None)
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: Optional[PreTrainedTokenizer],
        data_root: str,
        dataset_name: str,
        transform: Callable,
        prompter: Prompter,
        output_dir: Optional[str] = None,
        dataset_type: str = 'qwen2p5vl',
        min_pixels: int = 384*384,
        max_pixels: int = 384*384,
        image_token_length: int = 729,  # not used directly; kept for API compatibility
        mask_weight_type: str = 'log',
        siglip_processor: Callable = None,
        seed: int = 42,
        resize_mode: str = "none",
        target_size: int = 512,
        input_suffix: str = "",
        gt_suffix: str = "",
        file_extensions: List[str] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'),
    ):
        assert dataset_type in ('qwen2vl', 'qwen2p5vl')

        self.data_root = data_root
        self.dataset_name = dataset_name
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
        self.input_suffix = input_suffix
        self.gt_suffix = gt_suffix
        self.file_extensions = list(file_extensions)

        # Output directories for saving intermediate/cropped images
        if self.output_dir:
            self.output_input_dir = os.path.join(output_dir, f'{dataset_name}', 'input')
            self.output_gt_dir = os.path.join(output_dir, f'{dataset_name}', 'gt')
            os.makedirs(self.output_input_dir, exist_ok=True)
            os.makedirs(self.output_gt_dir, exist_ok=True)

        # Special tokens
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(self.image_begin_token)
        assert isinstance(self.image_begin_token_id, int), f"tokenizer miss `{self.image_begin_token}`"
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        assert isinstance(self.image_end_token_id, int), f"tokenizer miss `{self.image_end_token}`"

        # Load paired samples
        self.image_pairs = self._load_image_pairs()

    # ---------- I/O ----------
    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        """Load input/GT image pairs with 1:1 filename correspondence."""
        input_dir = os.path.join(self.data_root, "input")
        gt_dir = os.path.join(self.data_root, "gt")
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        if not os.path.exists(gt_dir):
            raise ValueError(f"GT directory not found: {gt_dir}")

        def list_ext(root: str) -> List[str]:
            files = []
            for ext in self.file_extensions:
                files.extend([f for f in os.listdir(root) if f.lower().endswith(ext.lower())])
            return files

        input_files = list_ext(input_dir)
        gt_files = list_ext(gt_dir)

        image_pairs = []
        input_dict, gt_dict = {}, {}

        for input_file in input_files:
            base_name = self._extract_base_name(input_file, is_input=True)
            input_dict[base_name] = os.path.join(input_dir, input_file)
        for gt_file in gt_files:
            base_name = self._extract_base_name(gt_file, is_input=False)
            gt_dict[base_name] = os.path.join(gt_dir, gt_file)

        matched_count = 0
        for base_name in sorted(input_dict.keys()):
            if base_name in gt_dict:
                image_pairs.append((input_dict[base_name], gt_dict[base_name]))
                matched_count += 1
            else:
                print(f"Warning: No GT found for input {base_name}")

        print(f"Found {matched_count} matching pairs for {self.dataset_name} in {self.data_root}")
        return image_pairs

    def _extract_base_name(self, filename: str, is_input: bool) -> str:
        base_name = os.path.splitext(filename)[0]
        if is_input and self.input_suffix and base_name.endswith(self.input_suffix):
            base_name = base_name[:-len(self.input_suffix)]
        elif (not is_input) and self.gt_suffix and base_name.endswith(self.gt_suffix):
            base_name = base_name[:-len(self.gt_suffix)]
        return base_name

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """When resize_mode == 'none', resize to target_size via simple stretch
        (preserves the same behaviour as the original code)."""
        if self.resize_mode == "none":
            return image.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
        else:
            return image

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, gt_path = self.image_pairs[idx]

        # Load images
        input_image = Image.open(input_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        # Resize (consistent with previous behaviour)
        input_image = self._resize_image(input_image)
        gt_image = self._resize_image(gt_image)

        # ====== Build an empty-text conversation (5-line prompt is generated at validation time) ======
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

        # Assemble token ids: replace <image> with the spatial token and
        # replace <|image_gen|> with image_begin_token
        input_ids = []
        labels = []
        has_generated_image = False
        for item in prompt_list:
            item_prompt = item["prompt"].replace('<image>', self.image_token)
            if self.generated_image_token in item_prompt:
                assert item["from"] == self.prompter.assistant_role
                assert (f"{self.generated_image_token}{self.prompter.eos_token}" in item_prompt)
                item_prompt = item_prompt.replace(
                    f"{self.generated_image_token}{self.prompter.eos_token}",
                    self.image_begin_token,
                )
                has_generated_image = True
            tokenized_item = self.tokenizer(
                item_prompt, return_tensors="pt", truncation=True, max_length=1024,
            )
            if item.get("is_labels", False):
                labels.append(tokenized_item.input_ids)
            else:
                labels.append(torch.full_like(tokenized_item.input_ids, -100))
            input_ids.append(tokenized_item.input_ids)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)

        # ====== Visual encoding (input image as VLM pixel_values) ======
        image_slice = [input_image]
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels,
            processor=self.processor, image_token=self.image_token,
            factor=self.factor, last_image=gt_image,
            vae_image_transform=self.vae_transform,
            drop_prompt=False, prompt="",
            mask_weight_type=self.mask_weight_type,
            siglip_processor=self.siglip_processor,
            need_weight='false',
            resize_mode=self.resize_mode,
        )
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        pil_pixel_values = image_dict['pil_pixel_values']       # input PIL images for 5-line generation at validation time
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        # ====== Insert image_begin / image_end tokens and image token placeholders into input_ids ======
        input_ids, labels, image_position = self._process_image_token(
            input_ids,
            labels=labels,
            image_token_id=self.image_token_id,
            image_begin_token_id=self.image_begin_token_id,
            image_end_token_id=self.image_end_token_id,
            image_token_lengths=image_token_lengths,
        )

        # ====== VAE input: normalise to [-1, 1] (using the input image) ======
        img_np = np.array(input_image)
        img_tensor = torch.from_numpy(img_np).float().div(255.0)
        img_tensor = rearrange(img_tensor, "h w c -> c h w")
        vae_pixel_values = self.vae_transform(img_tensor).unsqueeze(0).to(torch.bfloat16)

        # ====== Assemble return dict (aligned with the validation script) ======
        return_data = {
            "input_ids": input_ids,                      # DataCollator will produce attention_mask
            "labels": labels,                            # not used for cross-entropy in validation; kept for compatibility
            "pixel_values": pixel_values,                # VLM visual features
            "image_position": image_position,            # VLM visual token positions
            "image_grid_thw": image_grid_thw,
            "prompt": "",                                # empty string; 5-line prompt is generated at validation time from the LR PIL
            "pil_pixel_values": pil_pixel_values,        # [PIL(input_image)], read at validation time for 5-line generation
            "siglip_pixel_values": siglip_pixel_values,  # SigLIP features extracted at validation time if available
            "vae_pixel_values": vae_pixel_values,        # [-1, 1] tensor for VAE encoding
            "weights": weights,
        }

        # GT image → generated_image (range [0, 1], shape C×H×W), consistent with the training pipeline
        if has_generated_image:
            image_tensor = torch.tensor(np.array(gt_image)).float() / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []

        # Metadata
        return_data["input_path"] = input_path
        return_data["gt_path"] = gt_path
        return_data["dataset_name"] = self.dataset_name
        return_data["pred_task"] = None
        return return_data

    # ---- Static helpers: forward resize_mode through ----
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
            inputs = processor(
                text=[f'dummy {image_token}'],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
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
        weights = []  # weights are not needed during validation
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


# ---------------------------
# Specific datasets (1:1)
# ---------------------------

class DehazingTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="dehazing_test", **kwargs)

class LOL1TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="LOL1", **kwargs)

class LOL2TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="LOL2", **kwargs)

class RealBlurJTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="RealBlur_J", **kwargs)

class RealBlurRTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="RealBlur_R", **kwargs)

class SIDDTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="SIDD", **kwargs)

class BSD68_15TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="BSD68_15", **kwargs)

class BSD68_25TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="BSD68_25", **kwargs)

class BSD68_50TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="BSD68_50", **kwargs)

class GoProTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="GoPro", **kwargs)

class GoProGammaTestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="GoPro_gamma", **kwargs)

class Urban100_15TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="Urban100_15", **kwargs)

class Urban100_25TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="Urban100_25", **kwargs)

class Urban100_50TestDataset(GenericTestDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="Urban100_50", **kwargs)


# ---------------------------
# ITS dataset (special structure: many inputs per GT)
# ---------------------------

class ITSTestDataset(GenericTestDataset):
    """
    ITS test set: many-to-one input/GT mapping.
    GT is matched by the prefix before the first underscore in the input filename.
    """
    def __init__(self, **kwargs):
        super().__init__(dataset_name="ITS", **kwargs)

    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        input_dir = os.path.join(self.data_root, "input")
        gt_dir = os.path.join(self.data_root, "gt")
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        if not os.path.exists(gt_dir):
            raise ValueError(f"GT directory not found: {gt_dir}")

        def list_with_ext(root: str) -> List[str]:
            files = []
            for ext in self.file_extensions:
                files.extend([f for f in os.listdir(root) if f.lower().endswith(ext.lower())])
            return files

        input_files = list_with_ext(input_dir)
        gt_files = list_with_ext(gt_dir)

        # Build GT mapping: '10001' -> '/.../gt/10001.png'
        gt_dict = {}
        for gt_file in gt_files:
            key = os.path.splitext(gt_file)[0]
            gt_dict[key] = os.path.join(gt_dir, gt_file)

        # Iterate inputs: use the prefix before the first underscore as the lookup key
        image_pairs = []
        matched = 0
        for in_file in sorted(input_files):
            base = os.path.splitext(in_file)[0]
            key = base.split('_')[0]
            if key in gt_dict:
                image_pairs.append((os.path.join(input_dir, in_file), gt_dict[key]))
                matched += 1
            else:
                print(f"Warning: No GT found for ITS input {in_file} (key={key})")

        print(f"Found {matched} matching pairs for ITS in {self.data_root}")
        return image_pairs


# ---------------------------
# Multi Datasets Wrapper
# ---------------------------

class MultiGenericTestDatasets:
    """
    Wrapper that manages multiple generic test datasets
    (5-line generation is not performed inside the datasets).
    """
    def __init__(
        self,
        lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        data_root: str,
        dataset_types: List[str] = (
            "dehazing_test", "LOL1", "LOL2", "RealBlur_J", "RealBlur_R", "BSD68_15", "BSD68_25", "BSD68_50",
            "GoPro", "GoPro_gamma", "Urban100_15", "Urban100_25", "Urban100_50", "ITS"
        ),
        output_dir: Optional[str] = None,
        target_size: int = 512,
        **kwargs
    ):
        self.data_root = data_root
        self.dataset_types = list(dataset_types)
        self.output_dir = output_dir
        self.target_size = target_size
        self.kwargs = kwargs

        dataset_classes = {
            "dehazing_test": DehazingTestDataset,
            "LOL1": LOL1TestDataset,
            "LOL2": LOL2TestDataset,
            "RealBlur_J": RealBlurJTestDataset,
            "RealBlur_R": RealBlurRTestDataset,
            "SIDD": SIDDTestDataset,
            "BSD68_15": BSD68_15TestDataset,
            "BSD68_25": BSD68_25TestDataset,
            "BSD68_50": BSD68_50TestDataset,
            "GoPro": GoProTestDataset,
            "GoPro_gamma": GoProGammaTestDataset,
            "Urban100_15": Urban100_15TestDataset,
            "Urban100_25": Urban100_25TestDataset,
            "Urban100_50": Urban100_50TestDataset,
            "ITS": ITSTestDataset,
        }

        self.datasets: Dict[str, GenericTestDataset] = {}
        self.dataset_info: Dict[str, dict] = {}

        for dataset_type in self.dataset_types:
            dataset_dir = os.path.join(data_root, dataset_type)
            if os.path.exists(dataset_dir):
                try:
                    dataset_class = dataset_classes.get(dataset_type, GenericTestDataset)
                    dataset = dataset_class(
                        lvlm_model=kwargs.get("lvlm_model", lvlm_model),
                        tokenizer=kwargs.get("tokenizer", tokenizer),
                        data_root=dataset_dir,
                        output_dir=output_dir,
                        target_size=target_size,
                        transform=kwargs["transform"],
                        prompter=kwargs.get("prompter") or PROMPT_TYPE['qwen2p5vl'](),
                        **{k: v for k, v in kwargs.items() if k not in ("transform", "prompter", "lvlm_model", "tokenizer")}
                    )
                    if len(dataset) > 0:
                        self.datasets[dataset_type] = dataset
                        self.dataset_info[dataset_type] = {
                            'dataset_type': dataset_type,
                            'num_samples': len(dataset),
                            'dataset_dir': dataset_dir,
                            'target_size': target_size,
                        }
                        print(f"✅ Successfully loaded {dataset_type} dataset: {len(dataset)} samples")
                    else:
                        print(f"⚠️  {dataset_type} dataset is empty")
                except Exception as e:
                    print(f"❌ Failed to load {dataset_type} dataset: {e}")
            else:
                print(f"⚠️  {dataset_type} dataset directory not found: {dataset_dir}")

    def get_dataset(self, key: str) -> Optional[GenericTestDataset]:
        return self.datasets.get(key)

    def get_all_datasets(self) -> Dict[str, GenericTestDataset]:
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
        print("\n" + "="*60)
        print("🗂️  MULTI-GENERIC TEST DATASETS SUMMARY")
        print("="*60)
        total_samples = 0
        for key, info in self.dataset_info.items():
            print(f"📊 {key:>15} Dataset: {info['num_samples']:>4} samples | Size: {info['target_size']}x{info['target_size']}")
            total_samples += info['num_samples']
        print("-"*60)
        print(f"📈 Total Datasets: {len(self.datasets)} | Total Samples: {total_samples}")
        print(f"📁 Data Root: {self.data_root}")
        print(f"🎯 Target Size: {self.target_size}x{self.target_size}")
        if self.output_dir:
            print(f"💾 Output Dir: {self.output_dir}")
        print("="*60 + "\n")


# ---------------------------
# Factory
# ---------------------------

def create_multi_generic_datasets(
    lvlm_model: FAPEIRQwen2p5VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    data_root: str,
    dataset_types: List[str] = (
        "dehazing_test", "LOL1", "LOL2", "RealBlur_J", "RealBlur_R", "BSD68_15", "BSD68_25", "BSD68_50",
        "GoPro", "GoPro_gamma", "Urban100_15", "Urban100_25", "Urban100_50", "ITS"
    ),
    output_dir: Optional[str] = None,
    target_size: int = 512,
    **kwargs
) -> MultiGenericTestDatasets:
    """
    Instantiate multiple generic test datasets
    (5-line generation is not performed inside the Dataset).
    """
    return MultiGenericTestDatasets(
        lvlm_model=lvlm_model,
        tokenizer=tokenizer,
        data_root=data_root,
        dataset_types=list(dataset_types),
        output_dir=output_dir,
        target_size=target_size,
        **kwargs
    )