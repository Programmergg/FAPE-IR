import torch
from typing import List, Dict, Any, Optional

class ValidationCollator:
    """
    A robust collator that:
      - Concatenates 1×... tensors on dim=0; stacks C×H×W tensors on a new batch dim
      - Builds `attention_mask` from `input_ids` (list or tensor)
      - Optionally pads `labels` lists to the same length with -100
      - Respects tokenizer.padding_side ("left" or "right")
    """
    def __init__(self, tokenizer, padding_side: str = "left"):
        self.tok = tokenizer
        if self.tok is not None:
            self.tok.padding_side = padding_side

    def _get_pad_id(self) -> int:
        if self.tok is not None:
            if getattr(self.tok, "pad_token_id", None) is not None:
                return int(self.tok.pad_token_id)
            if getattr(self.tok, "eos_token_id", None) is not None:
                return int(self.tok.eos_token_id)
        return 0

    def _pad_1d(
        self,
        seq_list: List[torch.Tensor],
        pad_value: int,
        make_mask: bool = True
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        assert len(seq_list) > 0, "seq_list is empty"
        max_len = max(x.shape[1] for x in seq_list)
        left = (self.tok is None) or (getattr(self.tok, "padding_side", "left") == "left")

        outs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for x in seq_list:
            assert x.ndim == 2 and x.shape[0] == 1, f"Expected [1, L], got {tuple(x.shape)}"
            L = x.shape[1]

            if L < max_len:
                pad = x.new_full((1, max_len - L), pad_value)
                x_pad = torch.cat([pad, x], dim=1) if left else torch.cat([x, pad], dim=1)
            else:
                x_pad = x
            outs.append(x_pad)

            if make_mask:
                if left:
                    m = torch.cat(
                        [
                            torch.zeros((1, max_len - L), dtype=torch.long, device=x.device),
                            torch.ones((1, L), dtype=torch.long, device=x.device),
                        ],
                        dim=1,
                    )
                else:
                    m = torch.cat(
                        [
                            torch.ones((1, L), dtype=torch.long, device=x.device),
                            torch.zeros((1, max_len - L), dtype=torch.long, device=x.device),
                        ],
                        dim=1,
                    )
                masks.append(m)

        out = torch.cat(outs, dim=0)  # [B, L_max]
        mask = torch.cat(masks, dim=0) if make_mask else None
        return out, mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Empty batch passed to ValidationCollator"

        keys = batch[0].keys()
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [b[k] for b in batch]
            v0 = vals[0]

            if isinstance(v0, torch.Tensor):
                # 1) 如果是 [1, ...]，沿 dim=0 cat（最常见）
                if v0.ndim >= 1 and v0.shape[0] == 1:
                    try:
                        out[k] = torch.cat(vals, dim=0)
                    except Exception:
                        out[k] = vals
                # 2) 如果是 [C, H, W]（没有 batch 维），用 stack 新建 batch 维
                elif v0.ndim == 3:
                    try:
                        out[k] = torch.stack(vals, dim=0)  # [B, C, H, W]
                    except Exception:
                        out[k] = vals
                # 3) 其他形状，尝试 cat，否则保留 list
                else:
                    try:
                        out[k] = torch.cat(vals, dim=0)
                    except Exception:
                        out[k] = vals
            else:
                out[k] = vals

        # attention_mask
        pad_id = self._get_pad_id()
        if "input_ids" in out and isinstance(out["input_ids"], list) and len(out["input_ids"]) > 0:
            id_list = [x for x in out["input_ids"] if isinstance(x, torch.Tensor)]
            if len(id_list) != len(out["input_ids"]):
                raise TypeError("All elements of input_ids list must be torch.Tensors of shape [1, L_i].")
            input_ids_padded, attn_mask = self._pad_1d(id_list, pad_value=pad_id, make_mask=True)
            out["input_ids"] = input_ids_padded
            out["attention_mask"] = attn_mask
        elif "input_ids" in out and isinstance(out["input_ids"], torch.Tensor):
            if ("attention_mask" not in out) or (out["attention_mask"] is None):
                out["attention_mask"] = (out["input_ids"] != pad_id).long()

        # labels（如为 list[1×L]，对齐到 [B, L]）
        if "labels" in out and isinstance(out["labels"], list) and len(out["labels"]) > 0:
            lab_list = [x for x in out["labels"] if isinstance(x, torch.Tensor)]
            if len(lab_list) == len(out["labels"]):
                labels_padded, _ = self._pad_1d(lab_list, pad_value=-100, make_mask=False)
                out["labels"] = labels_padded

        return out