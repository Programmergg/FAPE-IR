from .modeling_fapeir import FAPEIRQwen2ForCausalLM
from .qwen2vl.modeling_fapeir_qwen2vl import FAPEIRQwen2VLForConditionalGeneration
from .qwen2p5vl.modeling_fapeir_qwen2p5vl_moe import FAPEIRQwen2p5VLForConditionalGeneration

MODEL_TYPE = {
    'llava': FAPEIRQwen2ForCausalLM, 
    'qwen2vl': FAPEIRQwen2VLForConditionalGeneration, 
    'qwen2p5vl': FAPEIRQwen2p5VLForConditionalGeneration
}