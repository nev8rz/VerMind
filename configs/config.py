import os, sys, yaml
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import OmegaConf, MISSING

# ==================== 子配置定义 ====================

@dataclass
class ModelConfig:
    model_name: str = "VerMind-100M"
    vocab_size: int = 6400
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1


@dataclass
class DatasetConfig:
    train_path: str = MISSING
    valid_path: Optional[str] = None
    max_length: int = 2048
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-4
    gradient_accumulation: int = 1
    fp16: bool = True
    output_dir: str = "outputs/"
    save_steps: int = 500


# ==================== 主配置类 ====================

@dataclass
class ExperimentConfig:
    seed: int = 42
    stage: str = "sft"

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(train_path=MISSING))
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load_from_argv(cls) -> Tuple[OmegaConf, OmegaConf]:
        """
        示例:
            python train.py --config ../configs/sft_512.yaml lr=1e-4 batch_size=32
        返回: (结构化cfg_struct, 平铺cfg_flat)
        """
        if "--config" not in sys.argv:
            raise ValueError(" 必须指定 --config，例如: python train.py --config ../configs/sft_512.yaml lr=1e-4")

        idx = sys.argv.index("--config")
        if len(sys.argv) <= idx + 1:
            raise ValueError("请在 --config 后指定配置路径")

        cfg_path = sys.argv[idx + 1]
        overrides = [arg for arg in sys.argv[idx + 2:] if "=" in arg]
        return cls.load(cfg_path, overrides)

    @classmethod
    def load(cls, path: str, overrides: Optional[List[str]] = None) -> Tuple[OmegaConf, OmegaConf]:
        """
        从 YAML 文件加载配置，支持:
            - `_base_` 继承
            - dataclass 类型安全
            - 命令行覆盖
        返回: (结构化cfg_struct, 平铺cfg_flat)
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f" 未找到配置文件: {path}")

        # 加载base.yaml + 输入config
        def load_yaml_recursive(p: str):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "_base_" in data:
                base_path = os.path.join(os.path.dirname(p), data["_base_"])
                base_data = load_yaml_recursive(base_path) # 递归加载
                del data["_base_"]
                base_data.update(data) # 覆盖
                return base_data
            return data

        cfg_dict = load_yaml_recursive(path) # no flatten 

        cfg_struct = OmegaConf.structured(cls) # 记录st
        cfg_full = OmegaConf.merge(cfg_struct, cfg_dict)

        # flatten 便于训练时候使用
        plain = OmegaConf.to_container(cfg_full, resolve=True)
        flat = {**plain, **plain.get("model", {}), **plain.get("dataset", {}), **plain.get("training", {})}
        for k in ("model", "dataset", "training"):
            flat.pop(k, None)
        flat = OmegaConf.create(flat)  

        # ===  应用命令行覆盖 ===
        if overrides:
            cli_cfg = OmegaConf.from_dotlist(overrides)
            flat = OmegaConf.merge(flat, cli_cfg)
            cfg_full = OmegaConf.merge(cfg_full, cli_cfg)  # 同步更新结构化版本

        OmegaConf.resolve(flat)
        OmegaConf.resolve(cfg_full)
        return cfg_full, flat

    @staticmethod
    def save(cfg_struct: OmegaConf, save_path: str):
        """保存结构化配置（覆盖写入）"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        OmegaConf.save(cfg_struct, save_path)
