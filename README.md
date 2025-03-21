# MoE-LLM: Large-Scale Language Model Based on Mixture of Experts
This project implements a large-scale language model based on the Mixture of Experts (MoE) architecture, significantly improving computational efficiency and model performance through conditional computation. The model includes two phases: pre-training and supervised fine-tuning (SFT), enabling efficient handling of natural language understanding and generation tasks.

## Project Overview
The Mixture of Experts model utilizes sparse activation by distributing computations across multiple "expert" networks, reducing computational load while maintaining model performance. Unlike traditional Transformer architectures, the MoE model dynamically selects the most suitable expert sub-network for different inputs through a gating mechanism, ensuring efficient use of computational resources.

## Technical Features

- **MoE Architecture**: Implements conditional computation to enhance model capacity and inference efficiency.
- **Rotary Position Embedding (RoPE)**: Efficiently processes positional information and supports extrapolation to longer sequences.
- **RMSNorm**: A more computationally efficient normalization technique compared to LayerNorm.
- **SwiGLU Activation**: A gated linear unit variant that enhances model expressiveness.
- **Load Balancing Loss**: Ensures fair distribution of expert resources and prevents "expert collapse."
- **BF16 Mixed-Precision Training**: Accelerates training while reducing memory usage.

## Project Structure

```
├── moe/                  # Pre-trained model output directory
├── saves/                # Saved model checkpoints
│   ├── moe/             # Pre-trained model
│   └── sft/             # Fine-tuned model
├── tokenizer/            # Tokenizer files
├── moe_train.py          # Main code for pre-training phase
├── moe_sft_train.py      # Main code for SFT fine-tuning phase
├── dataset.py            # Dataset processing related code
├── train.jsonl           # Pre-training dataset
├── sft.jsonl             # Supervised fine-tuning dataset
└── README.md             # Project documentation
```

## Model Architecture Details

### Core Components

1. **RMSNorm Normalization Layer**
   - A more efficient normalization method compared to LayerNorm.
   - Uses root mean square (RMS) normalization to maintain deep network stability.

2. **Rotary Position Embedding (RoPE)**
   - Integrates positional information directly into attention computations.
   - Supports better sequence length extrapolation.

3. **Gating Mechanism**
   - Dynamically selects the most suitable expert for each input.
   - Enables sparse activation to enhance computational efficiency.

4. **Mixture of Experts (MoE) Layer**
   - Multiple parallel expert networks.
   - Gating mechanism routes inputs to different experts.
   - Weighted aggregation of expert outputs.

5. **Load Balancing Mechanism**
   - Ensures fair utilization of all experts.
   - Prevents overloading or underutilization of experts to optimize system efficiency.

### Decoder Architecture
The model adopts a Transformer decoder architecture, consisting of multiple DecoderLayers. Each layer includes:

1. Self-Attention Mechanism
2. MoE Feed-Forward Network (replacing traditional FFN)
3. Residual connections and normalization operations

## Training Process

The pre-training phase uses `moe_train.py` to learn language modeling capabilities from raw text corpora:

1. Configure model parameters (hidden dimensions, layers, number of experts, etc.).
2. Initialize the model and tokenizer.
3. Prepare the pre-training dataset.
4. Set training parameters (learning rate, batch size, etc.).
5. Execute training and save the model.

### SFT Fine-Tuning Phase

The SFT phase uses `moe_sft_train.py` to further enhance the model’s capabilities through high-quality instruction-response pairs:

1. Load the pre-trained model.
2. Prepare the SFT dataset.
3. Configure fine-tuning parameters.
4. Execute fine-tuning and save the results.

## Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- CUDA environment (recommended)

## Usage

### Install Dependencies

```bash
pip install torch transformers numpy pandas tensorboard
```

### Data Preparation

1. Prepare the pre-training dataset: JSONL format with training text in each line.
2. Prepare the SFT dataset: JSONL format with instruction-response pairs.

### Train the Model

1. Pre-training phase:

```bash
python moe_train.py
```

2. SFT fine-tuning phase:

```bash
python moe_sft_train.py
```

## Performance Optimization

- **Gradient Accumulation**: Set `gradient_accumulation_steps` to simulate larger batch sizes.
- **Mixed-Precision Training**: Use `bf16=True` to enable BrainFloat16 format for faster training.
- **Parameter Sharing**: Share embedding weights with the output layer to reduce parameter count.
- **Special Layer Initialization**: Apply specialized initialization strategies for deep network layers to improve stability.

## Future Work

- Extend support for multiple languages.
- Implement model parallelism for larger-scale parameters.
- Explore improvements in expert routing strategies.
- Add support for additional modalities (e.g., images, audio).

## References

- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
- (https://github.com/wyf3/llm_related/tree/main)
- [Mixture of Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## Download Data

- https://github.com/jingyaogong/minimind
- https://huggingface.co/datasets/lation/MoE_LLM_Dataset/tree/main

## Download Model

- https://huggingface.co/lation/MoE-LLM/blob/main/moe.rar
- https://huggingface.co/lation/MoE-LLM/blob/main/sft.rar

## Acknowledgments 
- Thanks to the [llm_related](https://github.com/wyf3/llm_related/tree/main) project for providing guidance and inspiration. 
