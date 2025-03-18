# 导入必要的库和模块
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # 导入Hugging Face的自动类
import torch  # PyTorch深度学习库
from moe_train import LLM, Config  # 导入我们自定义的MoE模型和配置类

# 加载分词器，用于将文本转换为模型可以理解的token序列
# 从本地保存的tokenizer目录加载
t = AutoTokenizer.from_pretrained('./saves/moe')

# 向Hugging Face的AutoConfig注册我们的自定义模型配置
# 这样系统就能识别"moe_model"这个模型类型
AutoConfig.register("moe_model", Config)

# 向AutoModelForCausalLM注册我们的LLM模型实现
# 这样就可以使用from_pretrained函数加载我们的模型
AutoModelForCausalLM.register(Config, LLM)

# 加载我们之前训练好并保存的MoE模型
model = AutoModelForCausalLM.from_pretrained('./saves/moe')

# 准备输入数据：
# 1. 添加开始标记(bos_token_id)
# 2. 对输入文本"1+1等于"进行编码，转换为token ID序列
input_data = [t.bos_token_id] + t.encode('汉高祖是谁？')
print(input_data)  # 打印编码后的token ID，方便调试

# 使用模型生成文本：
# 1. 将输入数据包装成字典并转换为torch tensor，增加批次维度
# 2. 设置生成的文本最大长度为20个token
# 3. stream=False表示等所有token生成完毕后一次性返回结果，而不是流式返回
# 4. temperature=0.0表示使用贪婪解码(始终选择概率最高的token)
# 5. top_k=1表示只考虑概率最高的1个候选token
for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0), "labels":None}, 
                           t.eos_token_id,  # 指定结束标记，生成到此标记时停止
                           20,              # 最大生成token数
                           stream=False,    # 非流式生成
                           temperature=0.0, # 温度参数，0表示贪婪解码
                           top_k=1):        # 只考虑概率最高的token
    # 解码生成的token序列并打印出来
    # token[0]取第一个(也是唯一一个)样本的结果
    print(t.decode(token[0]))