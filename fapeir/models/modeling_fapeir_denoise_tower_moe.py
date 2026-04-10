import torch
import numpy as np
import math
import weakref
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, Optional, Tuple, Union, List
from transformers.modeling_utils import PreTrainedModel
from fapeir.models.configuration_fapeir_denoise_tower import FAPEIRDenoiseTowerConfig

from diffusers.utils import is_torch_version
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from contextlib import nullcontext
import torch.nn.functional as F


def add_lora_moe_to_linear(linear_layer, num_experts=4, ranks=[8, 8, 8, 8], alpha=1.0, is_first_layer=False, model_ref=None):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    
    # Ensure ranks list matches num_experts
    if len(ranks) != num_experts:
        raise ValueError(f"Length of ranks list ({len(ranks)}) must match num_experts ({num_experts})")
    
    # Add LoRA MoE parameters directly to the linear layer
    linear_layer.num_experts = num_experts
    linear_layer.lora_ranks = ranks
    linear_layer.lora_alpha = alpha
    linear_layer.is_first_layer = is_first_layer
    # Use weak reference to avoid circular reference
    linear_layer.model_ref = weakref.ref(model_ref) if model_ref is not None else None
    
    # Calculate scale for each expert based on their rank
    linear_layer.lora_scales = [alpha / rank if rank > 0 else 1.0 for rank in ranks]
    
    # Determine the input dimension for LoRA layers
    if is_first_layer and model_ref is not None:
        # For first layer, use text_hidden_states dimension
        text_hidden_dim = 4096  # default
        if hasattr(model_ref, 'config'):
            text_hidden_dim = getattr(model_ref.config, 'text_hidden_dim', text_hidden_dim)
            text_hidden_dim = getattr(model_ref.config, 'output_hidden_size', text_hidden_dim)  # fallback
        lora_input_dim = text_hidden_dim
    else:
        # For other layers, use original input dimension
        lora_input_dim = in_features
    
    # Determine the input dimension for LoRA layers
    if is_first_layer and model_ref is not None:
        # For first layer, use text_hidden_states dimension
        text_hidden_dim = 4096  # default
        if hasattr(model_ref, 'config'):
            text_hidden_dim = getattr(model_ref.config, 'text_hidden_dim', text_hidden_dim)
            text_hidden_dim = getattr(model_ref.config, 'output_hidden_size', text_hidden_dim)  # fallback
        lora_input_dim = text_hidden_dim
        gate_input_dim = text_hidden_dim
    else:
        # For other layers, use original input dimension
        lora_input_dim = in_features
        gate_input_dim = in_features
    
    # Create multiple LoRA experts with different ranks
    linear_layer.lora_downs = nn.ModuleList([
        nn.Linear(lora_input_dim, rank, bias=False) for rank in ranks
    ])
    linear_layer.lora_ups = nn.ModuleList([
        nn.Linear(rank, out_features, bias=False) for rank in ranks
    ])
    
    # Gating network
    linear_layer.gate = nn.Linear(gate_input_dim, num_experts, bias=False)
    
    # ========== Text prior head (used in first layer) ==========
    text_dim = lora_input_dim if is_first_layer else in_features
    linear_layer.text_gate = nn.Linear(text_dim, num_experts, bias=False)  # [*, Dtxt] -> [*, E]
    
    # ========== Frequency-domain parameters, fusion parameters, constants (FIR low-pass + residual high-pass) ==========
    # Use depthwise separable 1D convolution as low-pass filter to avoid memory and dtype issues
    # introduced by FFT's complex numbers / FP32 promotion
    ks = getattr(model_ref.config, "freq_fir_ksize", 9) if model_ref is not None else 9
    ks = int(ks) if ks % 2 == 1 else int(ks + 1)  # ensure odd kernel size
    pad = ks // 2
    linear_layer.fir_ks = ks
    linear_layer.fir_pad = pad
    # Fixed Gaussian-approximated low-pass kernel, depthwise over D (= lora_input_dim)
    with torch.no_grad():
        grid = torch.arange(ks) - (ks // 2)
        sigma = max(2.0, ks / 6.0)
        g = torch.exp(-(grid**2) / (2 * sigma * sigma))
        g = (g / g.sum()).to(torch.float32)  # store in float32 for numerical stability
        w = g.view(1, 1, ks).repeat(lora_input_dim, 1, 1)  # [D,1,ks]
    # Register as buffer (not trainable); cast to x.dtype at forward time
    linear_layer.register_buffer("fir_weight", w, persistent=True)
    # Energy statistics pooling window (reduces variance and memory usage)
    linear_layer.energy_pool = getattr(model_ref.config, "freq_energy_pool", 8) if model_ref is not None else 8

    # Fusion weights and constants
    linear_layer.spectrum_blend = nn.Parameter(torch.tensor(0.5))  # spectral prior weight in [0, 1]
    linear_layer.text_blend     = nn.Parameter(torch.tensor(0.5))  # text prior weight in [0, 1]
    linear_layer._eps = 1e-8
    linear_layer.enable_freq_regularizer = True  # whether to enable frequency-domain loss

    # Initialize weights
    for i in range(num_experts):
        nn.init.kaiming_uniform_(linear_layer.lora_downs[i].weight, a=math.sqrt(5))
        nn.init.zeros_(linear_layer.lora_ups[i].weight)
    
    # Initialize gate & text_gate
    nn.init.normal_(linear_layer.gate.weight, std=0.01)
    nn.init.normal_(linear_layer.text_gate.weight, std=0.01)
    
    # Freeze original parameters
    linear_layer.weight.requires_grad = False
    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = False

    # ========== Frequency split: FIR low-pass + residual high-pass (real-valued, compatible with BF16/FP16) ==========
    def _spectral_split(self, x):
        """
        x: [B, L, D]
        Returns:
          x_low, x_high: [B, L, D]
          E_low, E_high: [B]  (float32 sample-level energy)
          cache: None  (FIR mode requires no spectral cache)
        """
        B, L, D = x.shape
        # Conv1d expects [B, C, L]; treat D as C
        x_t = x.transpose(1, 2)  # [B, D, L]
        # Cast buffer weight to the same dtype as x to avoid dtype promotion to FP32
        w = self.fir_weight.to(dtype=x.dtype, device=x.device)  # [D,1,ks]
        low_t = F.conv1d(x_t, w, bias=None, stride=1, padding=self.fir_pad, groups=D)  # [B,D,L]
        x_low = low_t.transpose(1, 2)   # [B,L,D]
        x_high = x - x_low              # residual serves as high-pass

        # Energy statistics (float32 for stability)
        e_low_tok  = (x_low.to(torch.float32)**2).mean(dim=-1)   # [B,L]
        e_high_tok = (x_high.to(torch.float32)**2).mean(dim=-1)  # [B,L]
        if self.energy_pool > 1 and L >= self.energy_pool:
            pool = int(self.energy_pool)
            e_low_pool  = F.avg_pool1d(e_low_tok.unsqueeze(1), pool, stride=pool, ceil_mode=True).squeeze(1)   # [B, ceil(L/p)]
            e_high_pool = F.avg_pool1d(e_high_tok.unsqueeze(1), pool, stride=pool, ceil_mode=True).squeeze(1)  # [B, ceil(L/p)]
            E_low  = e_low_pool.mean(dim=-1)   # [B]
            E_high = e_high_pool.mean(dim=-1)  # [B]
        else:
            E_low  = e_low_tok.mean(dim=-1)    # [B]
            E_high = e_high_tok.mean(dim=-1)   # [B]

        return x_low, x_high, E_low, E_high, None

    linear_layer._spectral_split = _spectral_split.__get__(linear_layer, linear_layer.__class__)

    # ========== forward (with text prior + spectral prior + frequency regularization; streaming accumulation to avoid stack) ==========
    def forward_with_lora_moe(self, x):
        # Original Linear pass
        original_output = F.linear(x, self.weight, self.bias)
        B, L, _ = x.shape

        # Gate fusion
        def _fuse_gates(learned_logits, stat_prior, text_prior, TEMP, topk_k, epsilon):
            learned_prob = torch.softmax(learned_logits / TEMP, dim=-1)  # [B,L,E]
            mix = learned_prob
            if stat_prior is not None:
                lam_s = torch.sigmoid(self.spectrum_blend)
                mix = (1 - lam_s) * mix + lam_s * stat_prior
            if text_prior is not None:
                lam_t = torch.sigmoid(self.text_blend)
                mix = (1 - lam_t) * mix + lam_t * text_prior
            mix = mix / (mix.sum(dim=-1, keepdim=True) + self._eps)
            if topk_k is not None and topk_k > 0 and topk_k < self.num_experts:
                topk_vals, topk_idx = torch.topk(mix, topk_k, dim=-1)
                mask = torch.zeros_like(mix)
                mask.scatter_(-1, topk_idx, 1.0)
                mix = mix * mask + epsilon * (1 - mask)
                mix = mix / (mix.sum(dim=-1, keepdim=True) + self._eps)
            return mix.unsqueeze(-2)  # [B,L,1,E]

        # Select input for this layer (first layer prefers Htxt)
        lora_input = x
        tower = self.model_ref() if self.model_ref is not None else None
        has_text = False
        if self.is_first_layer and tower is not None and hasattr(tower, 'text_hidden_states') and tower.text_hidden_states is not None:
            has_text = True
            Htxt = tower.text_hidden_states  # [B, K, Dtxt]
            if Htxt.shape[1] != L:
                if Htxt.shape[1] < L:
                    pad_length = L - Htxt.shape[1]
                    padding = torch.zeros(Htxt.shape[0], pad_length, Htxt.shape[2], 
                                          device=Htxt.device, dtype=Htxt.dtype)
                    Htxt = torch.cat([Htxt, padding], dim=1)
                else:
                    Htxt = Htxt[:, :L, :]
            lora_input = Htxt

        # Frequency split (bind expert inputs) + statistical prior (real-valued)
        x_low, x_high, E_low, E_high, _ = self._spectral_split(lora_input)
        p_high = (E_high / (E_low + E_high + self._eps)).clamp(0., 1.).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        p_low  = 1.0 - p_high
        stat_prior = torch.cat([p_low.expand(B, L, 1), p_high.expand(B, L, 1)], dim=-1)  # [B,L,2]

        # Text prior (first layer only)
        text_prior = None
        if has_text:
            text_logits = self.text_gate(lora_input)           # [B,L,2]
            text_prior  = torch.softmax(text_logits, dim=-1)   # [B,L,2]

        # Learned gate + prior fusion
        if self.is_first_layer and tower is not None and has_text:
            TEMP = tower.config.get("moe_gate_temp", 5.0) if isinstance(tower.config, dict) else getattr(tower.config, "moe_gate_temp", 5.0)
            k = min(self.num_experts, getattr(tower.config, "moe_topk", 1))  # recommend topk=1 later to save memory
            epsilon = getattr(tower.config, "moe_epsilon", 1e-2)
            learned_logits = self.gate(lora_input)  # [B,L,E]
            gate_weights = _fuse_gates(learned_logits, stat_prior, text_prior, TEMP, k, epsilon)  # [B,L,1,E]
            # Record gate weights (first layer)
            if not hasattr(tower, "_all_gate_weights"):
                tower._all_gate_weights = []
            tower._all_gate_weights.append(gate_weights.squeeze(-2))  # [B,L,E]
        elif self.is_first_layer and (tower is None or not has_text):
            TEMP = 5.0
            k = min(self.num_experts, 2)
            epsilon = 1e-2
            learned_logits = self.gate(lora_input)
            gate_weights = _fuse_gates(learned_logits, stat_prior, None, TEMP, k, epsilon)
        else:
            TEMP = 5.0
            learned_logits = self.gate(x)
            gate_weights = _fuse_gates(learned_logits, stat_prior, None, TEMP, None, 0.0)

        # Expert inputs bound to frequency bands: 0 = low-freq, 1 = high-freq
        expert_inputs = [x_low, x_high]

        # === Streaming accumulation (avoids the large [B,L,Dout,E] intermediate tensor from stack) ===
        out_dim = self.lora_ups[0].out_features
        moe_output = torch.zeros(B, L, out_dim, device=x.device, dtype=x.dtype)
        lora_outputs_cache = []  # used for frequency regularization only; can skip caching if memory is tight

        for i in range(self.num_experts):
            ei = expert_inputs[i]
            lora_out = self.lora_ups[i](self.lora_downs[i](ei)) * self.lora_scales[i]  # [B,L,Dout] (kept in x.dtype)
            w = gate_weights[..., i]  # [B,L,1]
            moe_output = moe_output + lora_out * w  # broadcast to [B,L,Dout]
            lora_outputs_cache.append(lora_out)

        # ====== Frequency regularization (FIR approximation):
        # out-of-band energy of low-freq expert & low-freq leakage of high-freq expert ======
        if self.enable_freq_regularizer and tower is not None and len(lora_outputs_cache) >= 2:
            # Low-freq expert output
            y0 = lora_outputs_cache[0]  # [B,L,Dout]
            # High-freq expert output
            y1 = lora_outputs_cache[1]  # [B,L,Dout]

            # Prepare FIR weight matching the output channel count and dtype of y*_t
            # Also handle cases where output channels exceed the FIR buffer size
            def _prepare_weight_for_out(out_ch: int, ref_tensor: torch.Tensor):
                buf = self.fir_weight  # [Din,1,ks], Din = lora_input_dim
                din = buf.shape[0]
                if out_ch <= din:
                    w_use = buf[:out_ch, :, :]
                else:
                    # Extend: repeat the last kernel row until out_ch is satisfied
                    pad = buf[-1:, :, :].repeat(out_ch - din, 1, 1)
                    w_use = torch.cat([buf, pad], dim=0)  # [out_ch,1,ks]
                return w_use.to(dtype=ref_tensor.dtype, device=ref_tensor.device)

            # High-pass residual of y0
            y0_t = y0.transpose(1, 2)  # [B,Dout,L]
            w_out = _prepare_weight_for_out(out_dim, y0_t)  # dtype matches y0_t (BF16/FP16 safe)
            y0_low_t = F.conv1d(y0_t, w_out, bias=None, stride=1, padding=self.fir_pad, groups=out_dim)
            y0_low = y0_low_t.transpose(1, 2)  # [B,L,Dout]
            y0_high = y0 - y0_low

            # Low-pass component of y1
            y1_t = y1.transpose(1, 2)
            # Reuse the same weight (same out_dim and dtype)
            y1_low_t = F.conv1d(y1_t, w_out, bias=None, stride=1, padding=self.fir_pad, groups=out_dim)
            y1_low = y1_low_t.transpose(1, 2)

            # Out-of-band energy (float32 statistics for stability)
            L_freq = (y0_high.to(torch.float32).pow(2).mean() + y1_low.to(torch.float32).pow(2).mean())

            if not hasattr(tower, "_freq_losses"):
                tower._freq_losses = []
            tower._freq_losses.append(L_freq)

        out = original_output + moe_output
        return out.to(x.dtype)

    linear_layer.forward = forward_with_lora_moe.__get__(linear_layer, linear_layer.__class__)
    return linear_layer


class FAPEIRDenoiseTower(PreTrainedModel):
    config_class = FAPEIRDenoiseTowerConfig
    base_model_prefix = "model"

    def __init__(self, config: FAPEIRDenoiseTowerConfig):
        super().__init__(config)
        self.config = config
        if config.denoiser_type == "flux":
            self.denoiser = FluxTransformer2DModel.from_config(config.denoiser_config)
        elif config.denoiser_type == "sd3":
            self.denoiser = SD3Transformer2DModel.from_config(config.denoiser_config)
        else:
            raise ValueError(f"Unknown denoiser type: {config.denoiser_type}")
        self._init_denoise_projector()
        self._init_vae_projector()
        self._init_siglip_projector()
        
        # Initialize text_hidden_states
        self.text_hidden_states = None
        
        # Apply LoRA MoE to specified layers with different ranks
        self._apply_lora_moe(num_experts=2, ranks=[8, 8])
        
    def _apply_lora_moe(self, num_experts: int = 4, ranks: List[int] = [4, 8, 16, 32], alpha: float = 1.0):
        """Apply LoRA MoE to specified attention layers with different ranks"""
        if self.config.denoiser_type == "flux":
            first_layer_applied = False
            
            # Apply LoRA MoE to transformer_blocks
            for i, block in enumerate(self.denoiser.transformer_blocks):
                # Determine if this is the first layer
                is_first = not first_layer_applied
                
                # Add LoRA MoE to attention layers
                add_lora_moe_to_linear(block.attn.to_q, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=is_first, model_ref=self)
                add_lora_moe_to_linear(block.attn.to_k, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                add_lora_moe_to_linear(block.attn.to_v, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                add_lora_moe_to_linear(block.attn.to_out[0], num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                
                if is_first:
                    first_layer_applied = True
                    print(f"Applied LoRA MoE to transformer_block {i} to_q (FIRST LAYER with text_hidden_states) with ranks {ranks}")
                    print(f"Applied LoRA MoE to transformer_block {i} to_k, to_v, to_out with ranks {ranks}")
                else:
                    print(f"Applied LoRA MoE to transformer_block {i} with ranks {ranks}")
            
            # Apply LoRA MoE to single_transformer_blocks
            for i, block in enumerate(self.denoiser.single_transformer_blocks):
                # Add LoRA MoE to attention layers (not first layer)
                add_lora_moe_to_linear(block.attn.to_q, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                add_lora_moe_to_linear(block.attn.to_k, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                add_lora_moe_to_linear(block.attn.to_v, num_experts=num_experts, ranks=ranks, 
                                     alpha=alpha, is_first_layer=False, model_ref=self)
                
                print(f"Applied LoRA MoE to single_transformer_block {i} with ranks {ranks}")
        
        elif self.config.denoiser_type == "sd3":
            # Similar implementation for SD3 if needed
            print("LoRA MoE for SD3 not implemented yet")

    def get_lora_moe_parameters(self):
        """Get all LoRA MoE parameters for optimization"""
        lora_moe_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                # LoRA parameters
                for lora_down, lora_up in zip(module.lora_downs, module.lora_ups):
                    lora_moe_params.extend(list(lora_down.parameters()))
                    lora_moe_params.extend(list(lora_up.parameters()))
                # Gate parameters
                lora_moe_params.extend(list(module.gate.parameters()))
                # Text and spectral prior related parameters
                if hasattr(module, 'text_gate'):
                    lora_moe_params.extend(list(module.text_gate.parameters()))
                if hasattr(module, 'spectrum_blend'):
                    lora_moe_params.append(module.spectrum_blend)
                if hasattr(module, 'text_blend'):
                    lora_moe_params.append(module.text_blend)
        return lora_moe_params

    def get_lora_moe_state_dict(self):
        """Get LoRA MoE state dict for saving"""
        lora_moe_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                # Save LoRA weights
                for i, (lora_down, lora_up) in enumerate(zip(module.lora_downs, module.lora_ups)):
                    lora_moe_state_dict[f"{name}.lora_downs.{i}.weight"] = lora_down.weight
                    lora_moe_state_dict[f"{name}.lora_ups.{i}.weight"] = lora_up.weight
                # Save gate weights
                lora_moe_state_dict[f"{name}.gate.weight"] = module.gate.weight
                # Save text gate / blend
                if hasattr(module, 'text_gate'):
                    lora_moe_state_dict[f"{name}.text_gate.weight"] = module.text_gate.weight
                if hasattr(module, 'spectrum_blend'):
                    lora_moe_state_dict[f"{name}.spectrum_blend"] = module.spectrum_blend.detach()
                if hasattr(module, 'text_blend'):
                    lora_moe_state_dict[f"{name}.text_blend"] = module.text_blend.detach()
                # Save configuration information
                lora_moe_state_dict[f"{name}.lora_ranks"] = module.lora_ranks
                lora_moe_state_dict[f"{name}.lora_scales"] = module.lora_scales
                lora_moe_state_dict[f"{name}.is_first_layer"] = module.is_first_layer
                # FIR weights are buffers; typically no need to save, but can be added manually if required
        return lora_moe_state_dict

    def save_lora_moe_weights(self, path: str):
        """Save only LoRA MoE weights"""
        lora_moe_state_dict = self.get_lora_moe_state_dict()
        torch.save(lora_moe_state_dict, path)
        print(f"LoRA MoE weights saved to {path}")

    def load_lora_moe_weights(self, path: str):
        """Load LoRA MoE weights"""
        lora_moe_state_dict = torch.load(path, map_location='cpu')
        
        # Load weights into LoRA MoE modules
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                # Load LoRA weights
                for i in range(len(module.lora_downs)):
                    down_key = f"{name}.lora_downs.{i}.weight"
                    up_key = f"{name}.lora_ups.{i}.weight"
                    
                    if down_key in lora_moe_state_dict:
                        module.lora_downs[i].weight.data = lora_moe_state_dict[down_key]
                    if up_key in lora_moe_state_dict:
                        module.lora_ups[i].weight.data = lora_moe_state_dict[up_key]
                
                # Load gate weights
                gate_key = f"{name}.gate.weight"
                if gate_key in lora_moe_state_dict:
                    module.gate.weight.data = lora_moe_state_dict[gate_key]
                
                # Load text gate / blend
                text_gate_key = f"{name}.text_gate.weight"
                if text_gate_key in lora_moe_state_dict and hasattr(module, 'text_gate'):
                    module.text_gate.weight.data = lora_moe_state_dict[text_gate_key]
                sp_blend_key = f"{name}.spectrum_blend"
                if sp_blend_key in lora_moe_state_dict and hasattr(module, 'spectrum_blend'):
                    module.spectrum_blend.data = lora_moe_state_dict[sp_blend_key]
                txt_blend_key = f"{name}.text_blend"
                if txt_blend_key in lora_moe_state_dict and hasattr(module, 'text_blend'):
                    module.text_blend.data = lora_moe_state_dict[txt_blend_key]
                
                # Load configuration information
                ranks_key = f"{name}.lora_ranks"
                scales_key = f"{name}.lora_scales"
                first_layer_key = f"{name}.is_first_layer"
                
                if ranks_key in lora_moe_state_dict:
                    module.lora_ranks = lora_moe_state_dict[ranks_key]
                if scales_key in lora_moe_state_dict:
                    module.lora_scales = lora_moe_state_dict[scales_key]
                if first_layer_key in lora_moe_state_dict:
                    module.is_first_layer = lora_moe_state_dict[first_layer_key]
        
        print(f"LoRA MoE weights loaded from {path}")

    def enable_lora_moe(self):
        """Enable LoRA MoE training"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                # Enable LoRA parameters
                for lora_down, lora_up in zip(module.lora_downs, module.lora_ups):
                    for param in lora_down.parameters():
                        param.requires_grad = True
                    for param in lora_up.parameters():
                        param.requires_grad = True
                # Enable gate parameters
                for param in module.gate.parameters():
                    param.requires_grad = True
                # Enable text / frequency parameters
                if hasattr(module, 'text_gate'):
                    for p in module.text_gate.parameters():
                        p.requires_grad = True
                if hasattr(module, 'spectrum_blend'):
                    module.spectrum_blend.requires_grad = True
                if hasattr(module, 'text_blend'):
                    module.text_blend.requires_grad = True

    def disable_lora_moe(self):
        """Disable LoRA MoE training"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                # Disable LoRA parameters
                for lora_down, lora_up in zip(module.lora_downs, module.lora_ups):
                    for param in lora_down.parameters():
                        param.requires_grad = False
                    for param in lora_up.parameters():
                        param.requires_grad = False
                # Disable gate parameters
                for param in module.gate.parameters():
                    param.requires_grad = False
                # Disable text / frequency parameters
                if hasattr(module, 'text_gate'):
                    for p in module.text_gate.parameters():
                        p.requires_grad = False
                if hasattr(module, 'spectrum_blend'):
                    module.spectrum_blend.requires_grad = False
                if hasattr(module, 'text_blend'):
                    module.text_blend.requires_grad = False

    def get_lora_moe_info(self):
        """Get information about LoRA MoE configuration"""
        info = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'lora_downs'):
                info[name] = {
                    'num_experts': module.num_experts,
                    'ranks': module.lora_ranks,
                    'scales': module.lora_scales,
                    'alpha': module.lora_alpha,
                    'is_first_layer': getattr(module, 'is_first_layer', False),
                    'has_text_gate': hasattr(module, 'text_gate'),
                }
        return info

    def _init_denoise_projector(self):
        """Initialize the denoise_projector for multi model input."""
        if self.config.denoise_projector_type == "mlp2x_gelu":
            self.denoise_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_hidden_size,
                    self.config.output_hidden_size * 3,
                ),
                nn.SiLU(),
                nn.Linear(self.config.output_hidden_size * 3, self.config.output_hidden_size),
            )
        else:
            raise ValueError(f"Unknown denoise_projector_type: {self.config.denoise_projector_type}")

    def _init_vae_projector(self):
        """Initialize the vae_projector for multi model input."""
        if self.config.vae_projector_type == "mlp2x_gelu":
            self.vae_projector = nn.Sequential(
                nn.Linear(
                    self.config.vae_input_hidden_size,
                    3072,  # HARDCODE, x_embedder from flux
                ),
                nn.SiLU(),
                nn.Linear(
                    3072,  # HARDCODE, x_embedder from flux
                    self.config.output_hidden_size
                ),
            ).to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown vae_projector_type: {self.config.vae_projector_type}")

    def _init_siglip_projector(self):
        """Initialize the siglip_projector for multi model input."""
        self.siglip_projector = nn.Sequential(
            nn.Linear(
                1152,  # HARDCODE, out from siglip
                4096 * 3,  # HARDCODE
            ),
            nn.SiLU(),
            nn.Linear(
                4096 * 3,  # HARDCODE
                4096,  # HARDCODE, context_embedder from flux
            ),
        )

    def _init_convnext_projector(self):
        """Initialize the convnext_projector for multi model input."""
        self.convnext_projector = nn.Sequential(
            nn.Linear(
                1152,  # HARDCODE, out from convnext
                4096 * 3,  # HARDCODE
            ),
            nn.SiLU(),
            nn.Linear(
                4096 * 3,  # HARDCODE
                4096,  # HARDCODE, context_embedder from flux
            ),
        )
        
    @staticmethod     
    def _insert_image_feats(encoder_h, img_feats, img_pos, output_hidden_size, vae_projector):
        """
        encoder_h: Tensor[B, L, D]
        img_feats: list of B lists: The i-th element is a list with length = len(img_pos[i]),
                where the k-th item is a Tensor[Nik, D]
        img_pos:   list of B lists: The i-th element is a position list [p_i0, p_i1, ...]
                len(img_pos[i]) == len(img_feats[i])
        returns:   Tensor[B, L + Nmax, D], after inserting at respective positions,
                pad the right side according to the maximum insertion count
        """
        B, L, D = encoder_h.shape
        device = encoder_h.device
        
        # -- 1. For each sample, concatenate multiple groups of feats into an "insertion stream"
        #       and expand the corresponding positions
        flat_feats = []
        flat_pos   = []
        for feats_list, pos_list in zip(img_feats, img_pos):
            assert len(feats_list) == len(pos_list)
            # feats_list = [Tensor[N0,D], Tensor[N1,D], ...]
            # pos_list   = [p0,      p1,       ...]
            # concatenate all tokens to be inserted
            if len(feats_list) == 0:
                concat_f = torch.empty(0, output_hidden_size, device=device)
                pos_expanded = torch.empty(0, dtype=torch.long, device=device)
            else:
                concat_f = torch.cat(feats_list, dim=0)    # [Ni_total, D]
                concat_f = vae_projector(concat_f)
                # expand corresponding positions to the same length
                # e.g. feats_list[0].shape[0] copies of p0, feats_list[1].shape[0] copies of p1, ...
                # NOTE: p-1
                pos_expanded = torch.cat([
                    torch.full((f.shape[0],), p-1, dtype=torch.long, device=device)
                    for f, p in zip(feats_list, pos_list)
                ], dim=0)                                   # [Ni_total]
            flat_feats.append(concat_f)
            flat_pos.append(pos_expanded)
        
        # -- 2. Pad to the same length Nmax
        padded_feats = pad_sequence(flat_feats, batch_first=True)    # [B, Nmax, D]
        pos_pad = pad_sequence(flat_pos, batch_first=True, padding_value=L)
        
        # -- 3. Prepare sort keys for all tokens
        # original token j's key = 2*j
        orig_key = (torch.arange(L, device=device) * 2).unsqueeze(0).expand(B, -1)       # [B, L]
        # inserted token's key = 2*pos + 1
        ins_key = pos_pad * 2 + 1                                                      # [B, Nmax]
        
        # -- 4. Concatenate, sort once + gather
        all_keys = torch.cat([orig_key, ins_key], dim=1)                        # [B, L+Nmax]
        all_feats = torch.cat([encoder_h, padded_feats], dim=1)                        # [B, L+Nmax, D]
        sort_idx = all_keys.argsort(dim=1)                                            # [B, L+Nmax]
        new_seq = all_feats.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, D))      # [B, L+Nmax, D]
        return new_seq

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self._all_gate_weights = []
        self._freq_losses = []           # frequency-domain regularization (collected per layer as scalars; weighted sum done on the training side)
        self.text_mask = kwargs.pop('text_mask')
        
        # Set text_hidden_states for the first layer to use
        self.text_hidden_states = encoder_hidden_states[self.text_mask.squeeze(-1)].unsqueeze(0) 
        
        if self.config.denoiser_type == "flux":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            if encoder_hidden_states is not None:
                if prefix_prompt_embeds is not None:
                    encoder_hidden_states = torch.concat([encoder_hidden_states, prefix_prompt_embeds], dim=1)
            else:
                assert prefix_prompt_embeds is not None
                encoder_hidden_states = prefix_prompt_embeds
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(hidden_states.device, dtype=hidden_states.dtype)
            joint_attention_kwargs = kwargs.pop('joint_attention_kwargs', None)
            enc_attention_mask = kwargs.pop('enc_attention_mask', None)
            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep,  # Note: timestep is in [0, 1]. It has been scaled by 1000 in the training script.
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                **kwargs,
            )[0]
        elif self.config.denoiser_type == "sd3":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            if prefix_prompt_embeds is not None:
                encoder_hidden_states = torch.concat([prefix_prompt_embeds, encoder_hidden_states], dim=1)
            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                **kwargs,
            )[0]