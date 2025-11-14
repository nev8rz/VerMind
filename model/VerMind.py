'''
Author : yj z
Created: 2025-11-02 23:38
File   : VerMind.py
Description: VerMind Model
Version: 1.0.0
'''


import torch,math
from torch import (
    nn,
    Tensor,
)
from torch.nn import init 
from torch.nn import functional as F    
from typing import Optional,Tuple,List,Union
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class VerMindConfig(PretrainedConfig):
    model_type = "vermind"
    
    def __init__(
        self,
        dropout: float = 0.0,
        hidden_size: int = 768, 
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu", 
        intermediate_size: int = None, 
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps:float = 1e-05,
        rope_theta: int = 1e6,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        

class RMSNorm(nn.Module):
    
    def __init__(self,dim:int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self,x:Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim = -1,keepdim=True) + self.eps)
    def forward(self,x:Tensor) -> Tensor:
        return (self.weight * self._norm(x.float())).type_as(x)



def precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    half_dim = dim // 2
    freqs = 1.0 / (rope_base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        if end > orig_max:
            # 第一个波长大于 > original_max_position_embeddings的点
            corr_dim = next((i for i, f in enumerate(freqs) if 2 * math.pi / f > orig_max), half_dim)
            power = torch.linspace(0, 1, half_dim, device=freqs.device)
            beta = beta_slow + (beta_fast - beta_slow) * power
            scale = torch.where(
                torch.arange(half_dim, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
            freqs *= scale

    pos = torch.arange(end, device=freqs.device)
    freqs = torch.outer(pos, freqs)
    cos, sin = freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rot(x): return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def repeat_kv(x: Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # → shape: [bs, slen, num_kv_heads, 1, head_dim]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        # ↑ 复制 n_rep 次
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        # → 合并到 head 维度
    )


class GQA(nn.Module):
    
    def __init__(self,args:VerMindConfig):
        
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size,args.num_attention_heads * self.head_dim,bias = False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        
        
    def forward(self,
                x:Tensor,
                position_embeddings:Tuple[Tensor,Tensor],
                past_key_value:Optional[Tuple[Tensor,Tensor]] = None,
                use_cache = False,
                attention_mask: Optional[Tensor] = None,
                ):
        bsz,seq_len,_ = x.shape
        xq,xk,xv = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk = xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xv = xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        
        cos,sin = position_embeddings
        xq,xk = apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
        
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim = 1)
            xv = torch.cat([past_key_value[1],xv],dim = 1)
        past_kv = (xk,xv) if use_cache else None
        
        xq,xk,xv = (
            xq.transpose(1,2),
            repeat_kv(xk,self.n_rep).transpose(1,2),
             repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask[:, None, None, :]) * -1e9

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class SwiGLU(nn.Module):
    
    def __init__(self,config:VerMindConfig):
        super().__init__()
        
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(config.hidden_size,config.intermediate_size,bias = False)
        self.down_proj = nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
        self.up_proj = nn.Linear(config.hidden_size,config.intermediate_size,bias = False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]
        

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
class VerMindBlock(nn.Module):
    
    def __init__(self,layer_id:int,config:VerMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = GQA(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config) 
    
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class VerMindModel(nn.Module):  
    
    def __init__(self,config:VerMindConfig):
        super().__init__()
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([VerMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        
        bsz,seq_len  = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents
    
class VerMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VerMindConfig

    def __init__(self, config: VerMindConfig = None):
        self.config = config or VerMindConfig()
        super().__init__(self.config)
        self.model = VerMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT