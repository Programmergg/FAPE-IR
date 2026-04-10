# train_collator.py
# Lightweight collator for training: performs no tokenization and does not
# touch the LVLM; simply packs samples into lists for unified processing
# in the main process.
from typing import List, Dict

class RawTrainCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        # Return raw lists directly; the main process handles everything else
        return {
            "lr_pil_list":       [b["lr_pil"]        for b in batch],
            "gt_pil_list":       [b["gt_pil"]         for b in batch],
            "gt_tensor_list":    [b["gt_tensor"]      for b in batch],
            "refs_list":         [b["refs"]            for b in batch],
            "need_weight_list":  [b["need_weight"]    for b in batch],
            "meta_list":         [b["meta"]            for b in batch],
        }