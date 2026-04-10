import torch
from transformers.models.t5 import modeling_t5

class CompiledT5Block(modeling_t5.T5Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
modeling_t5.T5Block = CompiledT5Block