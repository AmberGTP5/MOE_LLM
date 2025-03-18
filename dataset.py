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


class LLMDataset(Dataset):
    """
    用于语言模型预训练的数据集类
    处理普通文本数据，以自回归方式准备输入和标签
    """
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        初始化数据集
        参数:
            data_path: 数据文件路径，通常是jsonl格式
            tokenizer: 分词器，用于将文本转换为token ID
            max_seq_len: 最大序列长度，超出部分会被截断
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # 读取数据文件的所有行
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        """返回数据集的大小（样本数量）"""
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        获取指定索引的数据样本
        参数:
            index: 样本索引
        返回:
            包含input_ids和labels的字典，用于自回归训练
        """
        # 读取并解析JSON行
        line = self.data[index]
        line = json.loads(line)
        # 添加开始和结束标记
        text = '<s>' + line['text'] + '</s>'
        # 将文本转换为token ID
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        
        # 处理序列长度：截断或填充
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]  # 截断过长序列
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)  # 填充到最大长度
            
        input_ids = np.array(input_ids)
        # 自回归处理：输入为去掉最后一个token的序列，输出为去掉第一个token的序列
        X = np.array(input_ids[:-1]).astype(np.int64)  # 输入：除最后一个token外的所有token
        Y = np.array(input_ids[1:]).astype(np.int64)   # 标签：除第一个token外的所有token
        
        # 返回torch张量
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }
        
class SFTDataset(Dataset):
    """
    监督微调(SFT)数据集类
    处理指令-输入-输出格式的数据，用于指令微调
    """
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        初始化SFT数据集
        参数:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 读取数据行
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
            
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)    
    
    def __getitem__(self, index):
        """
        获取特定索引的数据样本
        参数:
            index: 样本索引
        返回:
            处理后的输入ID和标签
        """
        # 读取并解析数据
        line = self.data[index]
        line = json.loads(line)
        
        # 获取指令、输入和输出文本
        instruction_text = line['instruction']  # 指令文本
        input_text = line['input']             # 用户输入
        output_text = line['output']           # 期望输出
        history = line['history']              # 对话历史
        
        # 合并指令和输入作为查询
        query = instruction_text + input_text
        # 将输出添加结束标记
        answer = output_text + self.tokenizer.eos_token
        
        # 构建对话消息列表
        messages = []
        if history:
            # 将历史对话添加到消息列表
            for i in history:
                messages.append({'role': 'user', 'content': i[0]})
                messages.append({'role': 'assistant', 'content': i[1]})
        
        # 添加当前查询
        messages.append({'role': 'user', 'content': query})
        
        # 应用聊天模板构建提示
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False) 
        
        # 编码提示和回答
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        
        # 合并提示和回答的token ID
        input_ids = prompt_input_ids + answer_input_ids
        
        # 创建标签：提示部分用0标记（不计算损失），回答部分使用正常标签
        labels = [0] * len(prompt_input_ids) + answer_input_ids
        
        # 处理序列长度：截断或填充
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)
        
        # 自回归处理：输入去掉最后一个token，标签去掉第一个token
        input_ids = input_ids[:-1]
        labels = labels[1:]
        
        return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}
    
    
# 内存不够，可使用如下方法加载数据
# class LLMDataset(IterableDataset):
#     def __init__(self, data_path, tokenizer, max_seq_len):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
    
#     def __iter__(self):
#         return self.data_process()
    
#     def data_process(self):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = json.loads(line)
#                 text = '<s>' + line['text'] + '</s>'
#                 input_ids = self.tokenizer.encode(text)
#                 text_len = len(input_ids)
#                 if text_len > self.max_seq_len:
#                     input_ids = input_ids[:self.max_seq_len]
#                 else:
#                     input_ids = input_ids + [0] * (self.max_seq_len - text_len)
#                 input_ids = np.array(input_ids)
#                 X = np.array(input_ids[:-1]).astype(np.int64)
#                 Y = np.array(input_ids[1:]).astype(np.int64)
#                 yield {
#                     'input_ids': torch.from_numpy(X),
#                     'labels': torch.from_numpy(Y),
#                 }

class DPODataset(Dataset):
    """
    直接偏好优化(DPO)数据集类
    处理包含偏好信息的数据，用于偏好优化训练
    """
    def __init__(self, data_path, tokenizer):
        """
        初始化DPO数据集
        参数:
            data_path: 数据文件路径
            tokenizer: 分词器
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        # 读取JSON格式的偏好数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
        
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        参数:
            index: 样本索引
        返回:
            包含提示、首选回答和拒绝回答的token ID列表
        """
        # 获取样本数据
        sample = self.datas[index]
        prompt = sample['prompt']          # 用户提示/问题
        chosen = sample['chosen']          # 首选回答
        rejected = sample['rejected']      # 拒绝的回答
        
        # 创建对话消息
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板格式化提示
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True     # 添加生成提示标记
        )
        
        # 编码各部分文本
        prompt_inputs = self.tokenizer(text=text)['input_ids']              # 编码提示
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]  # 拒绝回答
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]      # 首选回答
        
        # 返回提示、首选回答和拒绝回答的token ID
        return [prompt_inputs, chosen_inputs, rejected_inputs]
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.datas)
    
    
class DPODataCollator:
    """
    DPO训练的数据收集器
    将多个样本组合成批次，处理填充和标签创建
    """
    def __init__(self, tokenizer, max_seq_len):
        """
        初始化数据收集器
        参数:
            tokenizer: 分词器
            max_seq_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __call__(self, features):
        """
        处理批次数据
        参数:
            features: 样本列表，每个样本包含提示、首选回答和拒绝回答
        返回:
            批处理后的输入ID和标签
        """
        inputs_ids = []
        labels = []
        
        # 处理每个样本：首先处理首选回答
        for feature in features:
            # 合并提示和首选回答
            inputs_ids.append(feature[0] + feature[1])
            # 标签：提示部分为0，回答部分为正常token ID
            labels.append([0]*len(feature[0]) + feature[1])
            
        # 处理每个样本：然后处理拒绝回答
        for feature in features:
            # 合并提示和拒绝回答
            inputs_ids.append(feature[0] + feature[2])
            # 标签：提示部分为0，回答部分为正常token ID
            labels.append([0]*len(feature[0]) + feature[2])
            
        def process(inputs_ids, labels):
            """
            处理输入和标签：截断、填充和对齐
            """
            # 截断超长序列
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            
            # 找出批次中最长序列的长度
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            # 填充所有序列至相同长度
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    # 填充序列至相同长度
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    # 自回归处理：输入去掉最后一个token，标签去掉第一个token
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        # 处理所有输入和标签
        inputs_ids, labels = process(inputs_ids, labels)
        
        # 返回处理后的张量
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }