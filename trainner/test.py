
import argparse, sys, os
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omegaconf import OmegaConf
from configs.config import ExperimentConfig

def main():
    cfg_struct, cfg = ExperimentConfig.load_from_argv()

    print("✅ 平铺配置（训练时使用）:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # 保存结构化配置（美观、自动覆盖）
    cfg_name = os.path.splitext(os.path.basename(cfg.train_path))[0]
    save_path = os.path.join(cfg.output_dir, f"{cfg_name}_final.yaml")
    ExperimentConfig.save(cfg_struct, save_path)
    print(f"💾 已保存结构化配置到: {save_path}")

    # 示例：访问字段
    print(f"\n🚀 阶段: {cfg.stage}")
    print(f"🧠 模型: {cfg.model_name}")
    print(f"📁 数据集: {cfg.train_path}")
    print(f"🎯 学习率: {cfg.lr}")
    print(f"🕹️ 批大小: {cfg.batch_size}")

if __name__ == "__main__":
    main()