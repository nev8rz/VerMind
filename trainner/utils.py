'''
Author : yj z
Created: 2025-11-06 20:53
File   : utils.py
Description: Trainer utils
Version: 1.0.0
'''

import os
import random
import math
import torch
import numpy as np
import torch.distributed as dist
from model.VerMind import VerMindModel, VerMindConfig,VerMindForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import Sampler

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

import math

def get_lr(current_step: int,
           total_steps: int,
           base_lr: float,
           warmup_ratio: float = 0.1,
           stable_ratio: float = 0.4,
           decay_type: str = "cosine"):
    """
    通用WSD学习率调度 (Warmup → Stable → Decay)
    Args:
        current_step: 当前步数
        total_steps: 总步数
        base_lr: 基础学习率（峰值）
        warmup_ratio: warmup占比，例如0.1表示10%
        stable_ratio: stable阶段占比，例如0.4表示40%
        decay_type: 'cosine' 或 'linear'
    """
    # 各阶段步数
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_start = warmup_steps + stable_steps

    if current_step < warmup_steps:
        lr = base_lr * (current_step / warmup_steps)
    elif current_step < decay_start:
        lr = base_lr
    else:
        decay_progress = (current_step - decay_start) / (total_steps - decay_start)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        
        if decay_type == "cosine":
            lr = 0.5 * base_lr * (1 + math.cos(math.pi * decay_progress))
        else:  # 线性衰减
            lr = base_lr * (1 - decay_progress)
    
    return lr

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    



def save_checkpoint(model,optimizer,epoch,step,swanlab,save_dir,**kwargs):
    from torch.nn.parallel import DistributedDataParallel as DDP
    swanlab_id = None
    if swanlab:
        if hasattr(swanlab, 'get_run'):
            run = swanlab.get_run()
            swanlab_id = getattr(run, 'id', None) if run else None
        else:
            swanlab_id = getattr(swanlab, 'id', None)
        
    if is_main_process():
        save_path = os.path.join(save_dir, f"checkpoint_{step}.pth")
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'word_size': dist.get_world_size() if dist.is_initialized() else 1,
            'swanlab_id': swanlab_id
        }
        for key, value in checkpoint_data.items():
            if value is None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DDP):
                        checkpoint_data[key] = value.module.state_dict()
                    else:
                        checkpoint_data[key] = value.state_dict()
                else:
                    checkpoint_data[key] = value
        torch.save(checkpoint_data, save_path)
        
def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        print(f"加载检查点 {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        saved_ws = checkpoint.get("world_size", 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        checkpoint["step"] = checkpoint_path.split('step_')[-1].split('.pth')[0]
        if saved_ws != current_ws:
            Logger(f"GPU数量变化，从 {saved_ws} 变为 {current_ws}，调整训练步数...")
            checkpoint['step'] = checkpoint['step'] * saved_ws // current_ws
        
        return checkpoint
    else:
        print(f"检查点 {checkpoint_path} 不存在")
        return None


def load_model_and_tokenizer(config, device = 'cuda', model_path: str = None, tokenizer_path: str = "../VerMind"):
    model = VerMindForCausalLM(config)
    
    if model_path and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("No pre-trained model found, training from scratch.")
    
    print(f"模型可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
