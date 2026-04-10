from typing import Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seed: int = 42
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 1e-2
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    resume_from_checkpoint: Optional[str] = None
    weighting_scheme: Optional[str] = ("logit_normal")  # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "null"]
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    max_grad_norm: float = 1.0
    checkpointing_steps: int = 100
    checkpoints_total_limit: Optional[int] = 500
    drop_condition_rate: float = 0.0
    drop_t5_rate: float = 1.0
    validation_steps: int = 100
    num_validation_images: int = 1
    mask_weight_type: Optional[str] = None     # ['log', 'exp']
    sigmas_as_weight: bool = False  # Used in Flux
    discrete_timestep: bool = True  # Used in Flux
    optimizer: str = 'adamw'  # ['adamw', 'prodigy']
    prodigy_use_bias_correction: bool = True
    prodigy_safeguard_warmup: bool = True
    prodigy_decouple: bool = True
    prodigy_beta3: Optional[float] = None 
    prodigy_d_coef: float = 1.0
    profile_out_dir: Optional[str] = None

@dataclass
class DatasetConfig:
    dataset_type: str
    data_txt: str
    batch_size: int = 16
    num_workers: int = 4
    height: int = 512
    width: int = 512
    min_pixels: int = 448*448
    max_pixels: int = 448*448
    anyres: str = 'any_1ratio'
    ocr_enhancer: bool = False
    random_data: bool = False
    padding_side: str = 'left'
    pin_memory: bool = True

@dataclass
class ModelConfig:
    pretrained_lvlm_name_or_path: str
    pretrained_denoiser_name_or_path: str
    pretrained_siglip_name_or_path: Optional[str] = None
    train_vision_tower_mm_projector: bool = False
    guidance_scale: float = 1.0  # Used in Flux
    tune_mlp1_only: bool = False
    pretrained_mlp1_path: Optional[str] = None
    with_tune_mlp2: bool = False
    only_tune_mlp2: bool = False
    pretrained_mlp2_path: Optional[str] = None
    only_tune_image_branch: bool = True  # Used in SD3
    with_tune_mlp3: bool = False
    only_tune_mlp3: bool = False
    pretrained_mlp3_path: Optional[str] = None
    with_tune_siglip_mlp: bool = False
    only_tune_siglip_mlp: bool = False
    pretrained_siglip_mlp_path: Optional[str] = None
    only_use_t5: bool = False
    vlm_residual_image_factor: float = 0.0
    vae_fp32: bool = True
    
@dataclass
class FAPEIRTrainingDenoiseConfig:
    training_config: TrainingConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig