import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset


class RMSNorm(nn.Module):
    """
    RMSNorm归一化层：相比LayerNorm计算更高效
    用于对隐藏状态进行归一化，保持深层网络的稳定性
    """
    def __init__(self, hidden_size, eps=1e-6):
        
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习的缩放参数
        self.variance_epsilon = eps  # 防止除零的小常数

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        # 计算均方根值
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 通过均方根归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()
    
def rotate_half(x):
    """
    将输入张量在最后一个维度上一分为二，并进行交叉旋转
    用于旋转位置编码的实现
    """
    x1, x2 = x.chunk(2, dim=-1)  # 沿最后一个维度将张量分成两半
    return torch.cat((-x2, x1), dim=-1)  # 第二部分取负并放在前面，第一部分放在后面

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """
    应用旋转位置编码到查询和键向量
    这是RoPE(Rotary Position Embedding)的核心实现
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin)  # 对查询向量应用旋转
    k_embed = (k*cos) + (rotate_half(k)*sin)  # 对键向量应用旋转
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    通过三角函数生成位置相关的编码，帮助模型理解序列中的位置信息
    """
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # 计算旋转角频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        # 预计算并缓存旋转角的三角函数值
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        # 获取对应序列长度的三角函数值并应用旋转
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)
    
def repeat_kv(hidden_states, n_rep):
    """
    重复键值向量以匹配多头注意力机制中头的数量
    用于实现多查询注意力(MQA)和分组查询注意力(GQA)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 扩展并重塑以复制键值头
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    """
    多头注意力机制模块
    支持GQA(分组查询注意力)和KV缓存，以提高推理效率
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads  # GQA参数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 每个KV头对应的查询头数量
        self.k_cache, self.v_cache = None, None  # KV缓存，用于推理加速
        self.is_causal = True  # 因果注意力掩码，确保只关注过去位置
        self.flash_attn = self.config.flash_attn  # 是否使用FlashAttention算法加速

        # 投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)  # 旋转位置编码
        
    def forward(self, hidden_states, use_kv_cache=False):
        b, s = hidden_states.shape[:2]  # 批次大小和序列长度
        
        # KV缓存逻辑：在推理时重用之前计算的键值，提高效率
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                # 首次推理或缓存不匹配时，计算所有q,k,v
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                # 增量推理，仅计算新token的q,k,v并与缓存合并
                token = hidden_states[:, -1:, :]
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1)
            self.k_cache, self.v_cache = k, v  # 更新缓存
            
        else:
            # 训练模式，无需缓存
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            
        # 重塑投影结果为多头格式
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)
        
        # 应用旋转位置编码
        q, k = self.rotary_emb(q, k)
        
        # 重复键值头以匹配查询头的数量（GQA实现）
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        
        # 调整维度顺序以便进行注意力计算
        q = q.transpose(1, 2)  # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2)  # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2)  # b, self.num_heads, s, self.head_dim
        
        if self.flash_attn:
            # 使用PyTorch提供的优化注意力计算函数
            # q*k转置，（b, self.num_heads, s, self.head_dim）* (b, self.num_heads, self.head_dim，s) = （b, self.num_heads, s, s）
            # q*k/sqrt(self.head_dim)*v  （b, self.num_heads, s, s）* (b, self.num_heads, s, self.head_dim) = b, self.num_heads, s, self.head_dim
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout if self.training else 0.0, 
                                                    is_causal=self.is_causal) 
        else:
            # 手动实现注意力计算
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + self.mask[:, :, :s, :s]  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出
        
        # 恢复原始形状并投影回原始维度
        output = output.transpose(1, 2).contiguous().view(b, s, -1)  # b, s, self.hidden_size
        
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output
    
    
class MLP(nn.Module):
    """
    多层感知机模块
    使用SwiGLU激活函数的前馈网络，用于特征变换
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # SwiGLU架构需要的三个投影层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x):
        # SwiGLU激活: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def load_balancing_loss_func(
    gate_logits,
    num_experts,
    top_k):
    """
    计算专家负载均衡损失
    确保各个专家被均匀使用，防止某些专家过载或闲置
    """
    # 合并所有层的门控逻辑
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)  # [layers X batch_size X sequence_length, num_experts]
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)  # 计算路由权重
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # 选择top-k专家
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)  # 创建专家掩码
    
    # 计算每个专家处理的平均token数量
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # 计算路由器分配给每个专家的平均概率
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    
    # 计算总体损失 - 鼓励均匀分配
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
    
class Gating(nn.Module):
    """
    专家门控机制
    决定每个输入应该由哪些专家处理
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk  # 每个token要选择的专家数量
        self.expert_num = config.expert_num  # 专家总数
        self.gate = nn.Linear(self.hidden_size, self.expert_num)  # 门控网络
        
    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # 门控: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1)  # 选择概率最大的k个专家
        
        # 创建稀疏门控矩阵
        zeros = torch.full_like(logits, float("-inf"))  # 创建负无穷矩阵用于屏蔽未选中的专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 仅保留topk专家的分数
        sparse_logits = F.softmax(sparse_logits, dim=-1)  # 对选中的专家概率重新归一化
        
        # 保存原始门控逻辑用于计算负载均衡损失
        gate_logit = logits.view(-1, self.expert_num)
        
        return sparse_logits, indices, gate_logit
    
class Expert(nn.Module):
    """
    单个专家模块
    实现与MLP相同的SwiGLU架构
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 与MLP相同的SwiGLU架构
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias) 
        
    def forward(self, x):
        # SwiGLU激活
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoE(nn.Module):
    """
    专家混合模块(Mixture of Experts)
    包含多个专家网络和一个门控机制，实现条件计算
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建专家池
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)  # 门控决策网络
        
    def forward(self, x):
        # 获取门控决策
        sparse_logits, indices, gate_logit = self.gating(x)
        
        final_outputs = torch.zeros_like(x)  # 初始化输出
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num)
        
        # 逐个专家处理相关输入
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # 找出被路由到当前专家的token (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1)  # 展平掩码 (batch_size * seq_len)
            
            if expert_mask_flat.any():  # 如果有任何token被路由到此专家
                expert_input = x_flat[expert_mask_flat]  # 提取相关输入 (seq_true, dim)
                export_output = expert(expert_input)  # 专家处理 (seq_true, dim)
                
                # 获取对应的门控分数
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true, 1)
                
                # 加权输出
                weighted_output = export_output * gate_scores  # (seq_true, dim)
                
                # 累加到最终输出
                final_outputs[expert_mask] += weighted_output
                
        return final_outputs, gate_logit
        
        

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含自注意力、MLP或MoE，以及归一化层
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)  # 自注意力层
        self.moe = MoE(config)  # 专家混合层
        self.mlp = MLP(config)  # 标准MLP层
        # 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx  # 层索引，用于决定使用MLP还是MoE
        
    def forward(
        self,
        hidden_states,
        use_kv_cache
    ):
        # 残差连接 + 层归一化 + 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            use_kv_cache=use_kv_cache
        )
        hidden_states = residual + hidden_states
        
        # 残差连接 + 层归一化 + 前馈网络(MLP或MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 偶数层使用MLP，奇数层使用MoE
        if self.layer_idx % 2 == 0:
            hidden_states = self.mlp(hidden_states)
            gate_logit = None
        else:
            hidden_states, gate_logit = self.moe(hidden_states)
            
        outputs = residual + hidden_states
        return outputs, gate_logit
   
   
class Config(PretrainedConfig):
    """
    模型配置类
    定义模型的超参数和结构选项
    
    编写自定义配置时需要记住的三个重要事项如下：
    1、必须继承自 PretrainedConfig
    2、PretrainedConfig 的 __init__ 方法必须接受任何 kwargs
    3、这些 kwargs 需要传递给超类的 __init__ 方法。
    """
    model_type = "moe_model"
    
    def __init__(self,
                hidden_size = 512,           # 隐藏层维度
                num_attention_heads = 16,    # 注意力头数量
                num_key_value_heads = 8,     # 键值头数量(GQA参数)
                flash_attn = True,           # 是否使用FlashAttention
                attention_bias = False,      # 注意力投影是否使用偏置
                max_seq_len = 512,           # 最大序列长度
                intermediate_size = 2048,    # 中间层维度
                mlp_bias = False,            # MLP是否使用偏置
                vocab_size = 6400,           # 词表大小
                n_layers = 8,                # 层数
                dropout = 0.0,               # Dropout比率
                expert_num = 4,              # 专家数量
                topk = 2,                    # 每个token选择的专家数
                output_router_logits = True, # 是否输出路由逻辑
                aux_loss_coef = 0.01,        # 辅助损失系数
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)
         

class LLM(PreTrainedModel):
    """
    完整的语言模型
    集成多个解码器层，实现文本生成功能
    """
    config_class = Config
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        
        # 词嵌入层
        self.tokon_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout) 
        
        # 创建解码器层堆栈
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx)) 
            
        # 最终层归一化和输出投影
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        
        # 权重绑定(共享embedding和输出权重，减少参数量)
        self.tokon_embeddings.weight = self.output.weight
        
        # 权重初始化
        self.apply(self._init_weights) 
        self.loss = None 
        self.aux_loss = None
        
        # 特殊初始化某些层，以提高深层网络稳定性
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            
        
    def forward(self, input_ids, labels, use_kv_cache=False):
        """
        前向传播
        处理输入序列，计算损失，并返回结果
        """
        all_router_logits = () if self.config.output_router_logits else None
       
        # 词嵌入和Dropout
        hidden_states = self.tokon_embeddings(input_ids) 
        hidden_states = self.dropout(hidden_states)  
        
        # 通过所有解码器层
        for idx, layer in enumerate(self.layers):
            hidden_states, gate_logit = layer(hidden_states, use_kv_cache=use_kv_cache)
            # 收集门控逻辑用于计算负载均衡损失
            if gate_logit is not None:
                all_router_logits += (gate_logit, )  

        # 最终层归一化
        hidden_states = self.norm(hidden_states) 
        
        # 损失计算
        if labels is not None:
            # 训练模式：计算整个序列的logits和交叉熵损失
            logits = self.output(hidden_states)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0) 
        else:
            # 推理模式：只计算序列最后一个位置的logits
            logits = self.output(hidden_states[:, [-1], :])  
            self.loss = None  
        
        # 计算辅助损失(负载均衡损失)
        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)
            
            if labels is not None:
                # 将辅助损失添加到主损失中
                self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        """
        文本生成方法
        支持温度采样、top-k采样和流式输出
        """
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        
        # 逐个生成新token
        while input_ids.shape[1] < max_new_tokens - 1:  
            # 计算下一个token的概率分布
            inference_res = self(input_ids, labels, use_kv_cache=use_kv_cache)  
            logits = inference_res.logits 
            logits = logits[:, -1, :]  # 只关注最后一个位置
            
            # 应用重复惩罚：降低已生成token的概率
            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty
            
            # 采样下一个token
            if temperature == 0.0:  # 贪婪解码
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:  # 温度采样
                logits = logits / temperature  
                if top_k is not None:  # top-k采样
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')  # 屏蔽top-k之外的token

                probs = F.softmax(logits, dim=-1)  # 计算概率分布
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  # 随机采样
            
            # 检查是否生成了结束符
            if idx_next == eos:  
                break
            
            # 将新token添加到输入序列
            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            
            # 流式输出
            if stream:  
                yield input_ids[:, s:]  
        
        # 非流式模式下，返回完整生成结果
        if not stream:  
            yield input_ids[:, s:]  
               
if __name__ == '__main__':   
    """
    主程序：创建模型并开始训练
    """
    # 创建模型配置和模型实例
    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 准备数据和训练器
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    
    # 配置训练参数
    args = TrainingArguments(output_dir='./moe',         # 输出目录
                            num_train_epochs=10,         # 训练轮数
                            do_train=True,               # 是否进行训练
                            per_device_train_batch_size=2, # 每个设备的批次大小
                            gradient_accumulation_steps=1, # 梯度累积步数
                            # max_steps=15000,           # 最大训练步数
                            logging_steps=1,             # 日志记录间隔
                            report_to='tensorboard',     # 报告工具
                            save_total_limit=5,          # 保存的检查点数量
                            bf16=True,                   # 是否使用BF16精度
                            learning_rate=2e-4,          # 学习率
                            lr_scheduler_type='cosine',  # 学习率调度器类型
                            dataloader_num_workers=8,    # 数据加载线程数
                            dataloader_pin_memory=True,  # 是否使用锁页内存
                            save_safetensors=False)      # 是否保存为safetensors格式
                            
    # 创建数据集
    dataset = LLMDataset('./train.jsonl', tokenizer=tokenizer, max_seq_len=512)
    
    # 创建训练器并开始训练
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型和训练状态
    trainer.save_model('./saves/moe')
    trainer.save_state()