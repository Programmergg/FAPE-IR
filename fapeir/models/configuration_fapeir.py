from typing import Optional
from transformers import Qwen2Config
from fapeir.models.configuration_fapeir_vision_tower import FAPEIRVisionTowerConfig
from fapeir.models.configuration_fapeir_denoise_tower import FAPEIRDenoiseTowerConfig

class FAPEIRConfig(Qwen2Config):
    model_type = "fapeir"
    sub_configs = {
        "vision_tower": FAPEIRVisionTowerConfig,
        "denoise_tower": FAPEIRDenoiseTowerConfig,
    }

    def __init__(
        self,
        vision_tower: FAPEIRVisionTowerConfig = None,
        denoise_tower: FAPEIRDenoiseTowerConfig = None,
        image_token_length: Optional[int] = None,
        shortcut_image_embeds: bool = False,
        shortcut_image_embeds_scale: float = 0.5,
        shortcut_projector_type: Optional[str] = "mlp2x_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_length = image_token_length
        self.shortcut_image_embeds = shortcut_image_embeds
        self.shortcut_image_embeds_scale = shortcut_image_embeds_scale

        if not shortcut_image_embeds:
            shortcut_projector_type = None

        if isinstance(vision_tower, dict):
            vision_tower["shortcut_projector_type"] = shortcut_projector_type
            self.vision_tower = FAPEIRVisionTowerConfig(**vision_tower)
        elif vision_tower is None:
            self.vision_tower = FAPEIRVisionTowerConfig(
                shortcut_projector_type=shortcut_projector_type
            )
        else:
            self.vision_tower = vision_tower
        if isinstance(denoise_tower, dict):
            denoise_tower["input_hidden_size"] = self.hidden_size
            self.denoise_tower = FAPEIRDenoiseTowerConfig(**denoise_tower)
        elif denoise_tower is None:
            self.denoise_tower = FAPEIRDenoiseTowerConfig(
                input_hidden_size=self.hidden_size
            )
        else:
            self.denoise_tower = denoise_tower