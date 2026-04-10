from dataclasses import dataclass
from transformers import TrainingArguments

@dataclass
class TrainingConfig(TrainingArguments): ...

@dataclass
class DatasetConfig:
    data_txt: str

@dataclass
class ModelConfig:
    pretrained_model_path_or_name: str
    image_processor_path: str
    train_vision_tower: bool = False
    train_mm_projector: bool = True
    train_llm: bool = True
    train_lm_head: bool = True

@dataclass
class FAPEIRTrainingConfig:
    training_config: TrainingConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig

    @classmethod
    def from_dict(cls, training_config: dict, dataset_config: dict, model_config: dict):
        return cls(
            training_config=TrainingConfig(**training_config),
            dataset_config=DatasetConfig(**dataset_config),
            model_config=ModelConfig(**model_config),
        )