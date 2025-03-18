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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
# 导入之前定义的MoE模型和配置类
from moe_train import LLM, Config

if __name__ == '__main__':
    # 向Hugging Face的自动配置注册我们的自定义模型类型
    # 这样可以通过from_pretrained方法加载我们的模型
    AutoConfig.register("moe_model", Config)
    # 注册自定义模型，使其与AutoModelForCausalLM兼容
    AutoModelForCausalLM.register(Config, LLM)
    
    # 从之前预训练的路径加载MoE模型
    model = AutoModelForCausalLM.from_pretrained('./saves/moe')
    # 打印模型参数量，帮助了解模型规模
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 准备数据处理器
    data_collator = DefaultDataCollator()
    # 加载分词器，用于将文本转换为模型可以处理的形式
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    
    # 设置SFT微调的训练参数
    args = TrainingArguments(
        output_dir='./sft',                    # 输出目录，用于保存模型检查点和日志
        num_train_epochs=5,                    # 训练轮数，完整数据集遍历5次
        do_train=True,                         # 执行训练
        per_device_train_batch_size=2,         # 每个GPU设备的批次大小
        gradient_accumulation_steps=1,         # 梯度累积步数，用于模拟更大批次
        # max_steps=15000,                     # 最大训练步数(这里被注释掉，使用epochs代替)
        logging_steps=1,                       # 每隔多少步记录一次日志
        report_to='tensorboard',               # 使用TensorBoard记录训练过程
        save_total_limit=5,                    # 最多保存的检查点数量
        bf16=True,                             # 使用BF16混合精度训练，提高训练速度
        learning_rate=2e-4,                    # 学习率
        lr_scheduler_type='cosine',            # 余弦学习率调度器，逐渐降低学习率
        dataloader_num_workers=1,              # 数据加载的工作线程数
        dataloader_pin_memory=True,            # 使用内存锁页，加速数据传输到GPU
        save_safetensors=False)                # 不使用safetensors格式保存模型
    
    # 创建SFT数据集，加载指令微调数据
    # SFTDataset通常包含问题-答案对，用于监督式微调
    dataset = SFTDataset('./sft.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    
    # 创建训练器并配置
    trainer = Trainer(
        model=model,                           # 要微调的模型
        args=args,                             # 训练参数
        train_dataset=dataset,                 # 训练数据集
        tokenizer=tokenizer,                   # 分词器
        data_collator=data_collator)           # 数据处理器
    
    # 开始训练
    # resume_from_checkpoint=False表示从头开始训练，不从检查点恢复
    # 如果要从中断的检查点继续训练，设为True
    trainer.train(resume_from_checkpoint=False)
    
    # 保存最终微调后的模型
    trainer.save_model('./saves/sft')
    # 保存训练状态，便于后续可能的继续训练
    trainer.save_state()