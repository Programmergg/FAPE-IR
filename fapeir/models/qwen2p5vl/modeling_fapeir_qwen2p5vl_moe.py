import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Optional, List, Tuple, Union, Literal, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers import GenerationMixin
from transformers.modeling_utils import restore_default_torch_dtype
from fapeir.models.modeling_fapeir_denoise_tower_moe import FAPEIRDenoiseTower
from fapeir.models.qwen2p5vl.configuration_fapeir_qwen2p5vl import FAPEIRQwen2p5VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers import AutoProcessor, AutoTokenizer

class FAPEIRQwen2p5VLModel(Qwen2_5_VLModel):
    def __init__(self, config: FAPEIRQwen2p5VLConfig):
        super().__init__(config)
        self.config = config


class FAPEIRQwen2p5VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = FAPEIRQwen2p5VLConfig

    def __init__(self, config: FAPEIRQwen2p5VLConfig):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = FAPEIRQwen2p5VLModel(config)
        self.denoise_tower = FAPEIRDenoiseTower(config.denoise_tower)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.forward_denoiser = False
        self.post_init()
        self.processor = AutoProcessor.from_pretrained('checkpoints/uniworld')
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    @classmethod
    @restore_default_torch_dtype
    def from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", config.torch_dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        config = copy.deepcopy(config)
        if config._attn_implementation_internal is not None:
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None
        config._attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                check_device_map=False,
                torch_dtype=torch_dtype,
            )
        model = cls(config, **kwargs)

        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)
        return model

    def get_denoise_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[List[torch.FloatTensor]] = None,
        image_position: Optional[torch.LongTensor] = None,
    ):
        input_embeds = self(input_ids, images, image_position)[0]
        input_embeds = self.denoise_tower(input_embeds)
        return input_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @torch._dynamo.disable
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                # skip last boi, because last boi do NOT have true image_token
                vision_start_indices = vision_start_indices[vision_start_indices + 1 < len(input_ids)]
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    # @torch.compile
    def forward_visual(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw=grid_thw)

    # @torch.compile
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        output_type: Literal["lvlm", "denoise_model_pred", "denoise_embeds"] = "lvlm",
        denoiser_kwargs: Optional[Dict] = {},
        only_use_t5: bool = False,
        vlm_residual_image_factor: float = 0.0,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        # text_mask: 与插入后的新序列布局对齐，仅文本位置为 True，形状 [B, L_new]
        text_mask = None

        if not only_use_t5:
            if (self.forward_denoiser):  # Force forward denoiser, which is used in FSDP training
                return self.denoise_tower.denoiser(**kwargs)
            if "hidden_states" in kwargs:
                print("You are using this model as a denoiser, please use the forward_denoiser_context to forward the model.")
                print("For example:")
                print("with self.forward_denoiser_context():")
                print("    ... # Your code ...")
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.dtype)
                    image_embeds = self.forward_visual(pixel_values, grid_thw=image_grid_thw)
                    if self.config.shortcut_projector_type is not None:
                        shortcut_image_embeds_batch = image_embeds
                    else:
                        shortcut_image_embeds_batch = None
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
                    mask = input_ids == self.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_embeds = self.forward_visual(pixel_values_videos, grid_thw=video_grid_thw)
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}")
                    mask = input_ids == self.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)
                shortcut_image_embeds = []
                if pixel_values is not None and shortcut_image_embeds_batch is not None:
                    cum_image_len = 0
                    for batch_idx in range(input_ids.shape[0]):
                        cur_input_ids = input_ids[batch_idx]
                        num_blocks, start_end_index, lengths = self.find_true_blocks((cur_input_ids == self.config.image_token_id))
                        for i in range(num_blocks):
                            shortcut_image_embeds.append(
                                (
                                    batch_idx,
                                    start_end_index[i],
                                    lengths[i],
                                    shortcut_image_embeds_batch[cum_image_len: cum_image_len + lengths[i]],
                                )
                            )
                            cum_image_len = cum_image_len + lengths[i]
            if output_type == "denoise_model_pred":
                assert len(denoiser_kwargs) > 0, ("denoiser_kwargs should not be empty when output_type is denoise_model_pred")
                return_dict = False

            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            if output_type.startswith("denoise"):
                outputs = outputs[0]
        else:
            outputs = None
        # print(outputs.shape) # torch.Size([1, 354, 3584])==>torch.Size([1, 839, 3584])
        if output_type.startswith("denoise"):
            if outputs is not None and vlm_residual_image_factor > 0.0 and pixel_values is not None:
                old = outputs[image_mask[:, :, 0]]  # shape [N, D]
                blended = old * (1 - vlm_residual_image_factor) + image_embeds * vlm_residual_image_factor  # shape [N, D]
                outputs = outputs.masked_scatter(image_mask, blended)
            if outputs is not None and shortcut_image_embeds is not None and self.config.shortcut_image_embeds:
                for (batch_idx, pos, image_seq_length, image_embeds_item) in shortcut_image_embeds:
                    outputs[batch_idx, pos: pos + image_seq_length, :] = (
                        self.config.shortcut_image_embeds_scale * image_embeds_item
                        + (1 - self.config.shortcut_image_embeds_scale)
                        * outputs[batch_idx, pos: pos + image_seq_length, :]
                    )
            ref_features_for_vlm = kwargs.pop('ref_features_for_vlm', None)
            siglip_hidden_states = kwargs.pop('siglip_hidden_states', None)
            vae_hidden_states = kwargs.pop('vae_hidden_states', None)
            if outputs is not None:
                outputs = self.denoise_tower.denoise_projector(outputs)
                if ref_features_for_vlm is not None:
                    outputs_ref_features = self.denoise_tower.vae_projector(ref_features_for_vlm)
                    outputs = torch.cat([outputs, outputs_ref_features], dim=1)

                # 找到插入点（通常为 image_end_token 的位置）
                indices = self.find_all_token_positions(input_ids, self.config.image_end_token_id)

                # === 构造原序列“仅文本”的掩码（排除视觉相关 token） ===
                tokens_to_exclude = [
                    getattr(self.config, "image_token_id", None),
                    getattr(self.config, "video_token_id", None),
                    getattr(self.config, "vision_start_token_id", None),
                    getattr(self.config, "image_end_token_id", None),
                    getattr(self.config, "video_end_token_id", None),
                ]
                orig_is_text = torch.ones_like(input_ids, dtype=torch.bool)
                for tok in tokens_to_exclude:
                    if tok is not None:
                        orig_is_text &= (input_ids != tok)

                only_siglip = siglip_hidden_states is not None and vae_hidden_states is None
                only_vae = vae_hidden_states is not None and siglip_hidden_states is None
                both_exist = siglip_hidden_states is not None and vae_hidden_states is not None
                if only_siglip:
                    s = self.denoise_tower.siglip_projector(siglip_hidden_states)  # [B, Ls, D]
                    outputs, attention_mask, text_mask = self._insert_img_to_vlm(outputs, attention_mask, s, indices, orig_is_text)
                elif only_vae:
                    v = self.denoise_tower.vae_projector(vae_hidden_states)        # [B, Lv, D]
                    outputs, attention_mask, text_mask = self._insert_img_to_vlm(outputs, attention_mask, v, indices, orig_is_text)
                elif both_exist:
                    s = self.denoise_tower.siglip_projector(siglip_hidden_states)  # [B, L1, D]
                    v = self.denoise_tower.vae_projector(vae_hidden_states)        # [B, L2, D]
                    combined = torch.cat([s, v], dim=1)  # [B, L1 + L2, D]
                    outputs, attention_mask, text_mask = self._insert_img_to_vlm(outputs, attention_mask, combined, indices, orig_is_text)

            if output_type == "denoise_embeds":
                # LVLM outputs -> MLP2 -> prompt_embeds
                return outputs
            elif output_type == "denoise_model_pred":
                # LM outputs -> MLP2 -> Denoiser -> model_pred
                denoiser_kwargs['enc_attention_mask'] = attention_mask
                # 传入仅文本位置的掩码（对齐新布局）
                denoiser_kwargs['text_mask'] = text_mask
                return self.forward_denoise_tower(outputs, **denoiser_kwargs)
            else:
                raise ValueError(f"Unknown output_type: {output_type}.")
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def forward_denoise_tower(self, outputs, **denoiser_kwargs):
        # 确保 text_mask 被传递到 denoise_tower
        return self.denoise_tower(encoder_hidden_states=outputs, **denoiser_kwargs)

    # @torch._dynamo.disable
    def find_all_token_positions(self, input_ids, token_id):
        """
        返回一个列表，列表中每个元素是该 batch 中对应样本中 token_id 出现的位置索引（1D Tensor）
        """
        match = (input_ids == token_id)
        batch_indices, seq_indices = torch.where(match)
        batch_size = input_ids.size(0)
        result = [[] for _ in range(batch_size)]
        for b, s in zip(batch_indices.tolist(), seq_indices.tolist()):
            result[b].append(s)
        return result

    def find_true_blocks(self, tensor):
        """
        给定一个 1D bool 张量，返回连续 True 段的数量、每段的 [start, end]、以及各段长度。
        """
        tensor = tensor.bool()
        padded = torch.nn.functional.pad(tensor[None].float(), (1, 1))  # shape: (1, L+2)
        diff = padded[:, 1:] - padded[:, :-1]  # shape: (1, L+1)
        starts = (diff == 1).nonzero(as_tuple=True)[1]
        ends = (diff == -1).nonzero(as_tuple=True)[1] - 1
        lengths = ends - starts + 1
        num_blocks = starts.numel()
        return num_blocks, list(zip(starts, ends)), lengths

    def forward_denoiser_context(self):
        class ForwardDenoiserContext:
            def __init__(self, model):
                self.model = model
                self.backup_config = None

            def __enter__(self):
                self.backup_config = self.model.config
                self.model.config = self.model.denoise_tower.denoiser.config
                self.model.forward_denoiser = True
                return self.model

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model.forward_denoiser = False
                self.model.config = self.backup_config
                return False
        return ForwardDenoiserContext(self)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepared with rope_deltas in forward
        model_inputs["position_ids"] = None
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    @staticmethod
    def _insert_img_to_vlm(
        vlm_feature: torch.Tensor,                  # [B, L, D]
        attention_mask: Optional[torch.Tensor],     # [B, L] or None
        img_feature: torch.Tensor,                  # [B, Limg_total, D] (每个样本所有插入片段拼接)
        indices_list: List[List[int]],              # 每个样本中插入的起点索引（通常为 image_end_token 位置）
        orig_is_text_mask: torch.Tensor,            # [B, L] 原序列“仅文本”的布尔掩码
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        将 img_feature 插入到 vlm_feature 中指定位置；同步重排 attention_mask；返回与新布局对齐的仅文本掩码 new_text_mask。
        约定：每个样本插入次数 k_i = len(indices_list[i])；每次插入片段长度相同为 img_L。
        则对样本 i：其 img_feature 的总长度应为 k_i * img_L；所有样本拼接成 [B, sum(k_i*img_L), D] 后按样本独立切分使用。
        """
        B, L, D = vlm_feature.shape
        device = vlm_feature.device

        # 计算每个样本需要插入的次数
        insert_counts = [len(inds) for inds in indices_list]
        max_inserts = max(insert_counts) if len(insert_counts) > 0 else 0

        if max_inserts == 0:
            # 无需插入，直接返回（保持 attention_mask 与文本掩码与原来一致）
            new_len = L
            new_vlm_feature = vlm_feature
            new_attention = attention_mask
            new_text_mask = orig_is_text_mask
            return new_vlm_feature, new_attention, new_text_mask

        # 计算每次插入的片段长度：假设各样本一致
        # 通过总 Limg_total / max_inserts 得到单次插入长度
        total_Limg = img_feature.shape[1]
        if total_Limg % max_inserts != 0:
            # 在 batch 内插入次数不一致时，总长度仍能被 max_inserts 整除；如果不能整除，抛错更安全
            raise ValueError(f"img_feature total length ({total_Limg}) is not divisible by max inserts per sample ({max_inserts}).")
        img_L = total_Limg // max_inserts if max_inserts > 0 else 0

        # 将 img_feature 按样本拆分（每个样本长度为 k_i * img_L）
        per_sample_img_feats: List[torch.Tensor] = []
        cursor = 0
        for k in insert_counts:
            seg_len = k * img_L
            if seg_len == 0:
                per_sample_img_feats.append(vlm_feature.new_zeros((0, D)))
            else:
                per_sample_img_feats.append(img_feature[:, cursor: cursor + seg_len, :].squeeze(0))
            cursor += seg_len

        # 计算新长度与对齐长度
        new_lens = [L + k * img_L for k in insert_counts]
        max_new_len = max(new_lens)

        # 2D masks（不带通道维）
        img_mask_2d = torch.zeros((B, max_new_len), dtype=torch.bool, device=device)
        vlm_mask_2d = torch.zeros((B, max_new_len), dtype=torch.bool, device=device)

        # 为每个样本确定左侧 padding，并设置插入与原序列的落位位置
        for i, inds in enumerate(indices_list):
            k = insert_counts[i]
            pad = max_new_len - (L + k * img_L)
            # 插入区
            for j, pos in enumerate(inds):
                start = pad + j * img_L + pos
                end = start + img_L
                img_mask_2d[i, start:end] = True
            # 原序列区：除去左侧 pad 的剩余位置
            vlm_mask_2d[i, pad:] = ~img_mask_2d[i, pad:]

        # 构造新特征并填充
        new_vlm_feature = vlm_feature.new_zeros((B, max_new_len, D))
        # 先 scatter 插入片段
        if img_L > 0 and max_inserts > 0:
            # 把每个样本的插入片段（顺序与 inds 一致）直接平铺写入
            flat_img = torch.cat(
                [
                    per_sample_img_feats[i] if per_sample_img_feats[i].numel() > 0 else vlm_feature.new_zeros((0, D))
                    for i in range(B)
                ],
                dim=0
            )
            new_vlm_feature[img_mask_2d] = flat_img.reshape(-1, D)

        # 再 scatter 原序列片段
        new_vlm_feature[vlm_mask_2d] = vlm_feature.reshape(-1, D)

        # 重建 attention_mask（对齐新布局）
        new_attention = None
        if attention_mask is not None:
            new_attention = torch.zeros((B, max_new_len), dtype=attention_mask.dtype, device=attention_mask.device)
            # 原序列注意力落到 vlm_mask_2d 的 True 位置
            new_attention[vlm_mask_2d] = attention_mask.reshape(-1)
            # 插入的图像片段是否参与注意力：默认置 1（如需屏蔽改成 0）
            new_attention[img_mask_2d] = 1

        # 构造“仅文本”的新掩码：把原序列中标记为文本的位置，按照 vlm_mask_2d 的落位规则 scatter 到新布局
        new_text_mask = torch.zeros((B, max_new_len), dtype=torch.bool, device=device)
        new_text_mask[vlm_mask_2d] = orig_is_text_mask.reshape(-1)
        # 插入区域与左 pad 区自然为 False

        return new_vlm_feature, new_attention, new_text_mask

    def _get_image_nums_and_video_nums(self, input_ids: Optional[torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size,)`)
            video_nums (`torch.LongTensor` of shape `(batch_size,)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)
        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        if expand_size == 1:
            return input_ids, model_kwargs
        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, image_nums.tolist())
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "image_grid_thw":
                    lengths = image_nums.tolist()
                    dict_to_expand[key] = _repeat_interleave_samples(dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, video_nums.tolist())
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "video_grid_thw":
                    lengths = video_nums.tolist()
                    dict_to_expand[key] = _repeat_interleave_samples(dict_to_expand[key], lengths=lengths, repeat_times=expand_size)
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead.")
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = video_nums.tolist()
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        model_kwargs = _expand_dict_for_generation(model_kwargs)
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
        return input_ids, model_kwargs