'''
Author : yj z
Created: 2025-11-07 16:01
File   : pretrain.py
Description: 
Version: 1.0.0
'''


import argparse, sys, os,time
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omegaconf import OmegaConf
from configs.config import ExperimentConfig
from utils import *
from model.VerMind import VerMindConfig
from contextlib import nullcontext
from dataset_llm.pretrainDataset import PretrainDataset
from torch.utils.data import DataLoader, DistributedSampler

def train_epoch(epoch,dataloader,total_steps,start_step,cfg):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step,(X,Y,loss_mask) in enumerate(dataloader,start = start_step + 1):
        X,Y,loss_mask = X.to(cfg.device),Y.to(cfg.device),loss_mask.to(cfg.device)
        lr = get_lr(
            current_step = (epoch * total_steps) + step,
            total_steps = cfg.epochs * total_steps,
            base_lr = cfg.lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with autocast_ctx:
            results = model(input_ids=X)
            loss = loss_fct(
                results.logits.view(-1, results.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        
        if step % cfg.logging_steps == 0 or step == total_steps - 1:
            elapsed = time.time() - start_time
            curr_loss = loss.item() * cfg.gradient_accumulation_steps
            curr_lr = optimizer.param_groups[-1]['lr']
            eta_time = elapsed / (step + 1) * total_steps // 60 - elapsed // 60
            Logger(f"Epoch [{epoch}/{cfg.epochs}] Step [{step}/{total_steps}] "
                   f"Loss: {curr_loss:.4f} LR: {curr_lr:.6f} "
                     f"Elapsed: {elapsed/60:.2f} min ETA: {eta_time:.2f} min")
            if cfg.use_swanlab and swanlab is not None:
                swanlab.log({
                    "loss": curr_loss,
                    "lr": curr_lr,
                    "epoch": epoch,
                    "step": step + epoch * total_steps,
                    "eta_time_epoch": eta_time
                })
            
        if is_main_process() and (step % cfg.save_steps == 0 or step == total_steps - 1):
            model.eval()
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step + epoch * total_steps,
                swanlab=swanlab,
                save_dir=cfg.save_dir,
            )
            model.train()

            
if __name__ == "__main__":
    cfg_struct, cfg = ExperimentConfig.load_from_argv()
    OmegaConf.to_yaml(cfg,resolve=True)
    # print(cfg,cfg_struct)
    os.makedirs(cfg.save_dir, exist_ok=True)
    OmegaConf.save(cfg_struct, cfg.save_dir + "/train_args.yaml") 
    # 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): cfg.device = f'cuda:{local_rank}'
    setup_seed(cfg.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # 模型配置
    os.makedirs(cfg.save_dir, exist_ok=True)
    lm_config = VerMindConfig(
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        num_hidden_layers=cfg.num_hidden_layers,
    )
    cpk_data = load_checkpoint(cfg.resume_path) if cfg.resume != -1 else None
    
    # 混合精度
    device = cfg.device
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16 if cfg.dtype == "float16" else torch.float32
    autocast_ctx = nullcontext()  if device == "cpu" else torch.amp.autocast(device_type="cuda",dtype=dtype)
    
    
    # swanlab集成
    swanlab = None
    if cfg.use_swanlab and is_main_process():
        import swanlab
        swanlab_id = cpk_data.get('swanlab_id', None) if cpk_data is not None else None
        swanlab_run_name = f"VerMind_{cfg.stage}_Ep{cfg.epochs}_Bs{cfg.batch_size}_Lr{cfg.lr}"
        swanlab = swanlab.init(project="VerMind", name=swanlab_run_name, id=swanlab_id,config=cfg)
    # 加载模型 and tokenizer,数据，优化器等
    model,tokenizer = load_model_and_tokenizer(
        lm_config, device, model_path=cfg_struct.model.model_path,tokenizer_path="../VerMind"
    )
    
    train_dataset = PretrainDataset(cfg.train_path,tokenizer,max_length=cfg.max_length)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    
    scaler = torch.amp.GradScaler(device=device,enabled=(cfg.dtype == "float16")) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # cpk加载
    start_epoch, start_step = 0, 0
    if cpk_data is not None:
        model.load_state_dict(cpk_data['model'])
        optimizer.load_state_dict(cpk_data['optimizer'])
        scaler.load_state_dict(cpk_data['scaler'])
        start_epoch = cpk_data['epoch'] 
        start_step = cpk_data['step']
        print(f"Resumed from checkpoint: epoch {start_epoch}, step {start_step}")

    if dist.is_initialized():
        model._ddp_pramas_and_buffers_to_ignore = ['freqs_cos', 'freqs_sin']
        model = torch.nn.parallel.DistributedDataParallel(
            model,device_ids = [local_rank])
    
    for epoch in range(start_epoch, cfg.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_dataset)), cfg.batch_size, start_step)
            dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=cfg.num_workers,pin_memory=True)
            Logger(f"Epoch {epoch}: Resuming from step {start_step}")
            train_epoch(epoch, dataloader, len(dataloader) + start_step + 1, start_step, cfg)
        else:
            dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, shuffle=(train_sampler is None),pin_memory=True)
            train_epoch(epoch, dataloader, len(dataloader), 0, cfg)
    
    