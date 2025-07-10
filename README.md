# English-Hungarian Neural Machine Translation with Knowledge Distillation

A complete implementation of Transformer-based neural machine translation from English to Hungarian, enhanced with an improved knowledge distillation system for model compression and acceleration.

## 🎯 Project Overview

This project implements a state-of-the-art neural machine translation system that:
- Translates English text to Hungarian using Transformer architecture
- Uses an improved knowledge distillation system to create efficient compressed models
- Provides multiple training configurations for different needs
- Includes complete training, testing, and evaluation pipeline

## 📁 Project Structure

```
├── data/
│   ├── cmn_hun.py                      # Data processing for English-Hungarian translation
│   └── hun-eng/                        # Hungarian-English parallel corpus
├── model/
│   └── transformer.py                  # Transformer model implementation
├── train_cmn_hun_transformer.py        # Teacher model training script
├── improved_distillation_trainer.py    # Improved knowledge distillation training
├── test_trained_student.py             # Student model testing and evaluation
├── inference_cmn_hun_transformer.py    # Translation inference script
└── train_process/                      # Training outputs and checkpoints
    ├── transformer-cmn-hun/            # Teacher model checkpoints
    └── distillation-[config]/          # Student model checkpoints by configuration
```

## 🚀 Quick Start

### 1. Train Teacher Model
```bash
python train_cmn_hun_transformer.py
```

### 2. Knowledge Distillation Training
```bash
# Run the improved distillation trainer
python improved_distillation_trainer.py

# Choose from 5 configurations:
# 1. Quick Test (15 minutes) - for verification
# 2. Conservative (2 hours) - stable training
# 3. Balanced (3 hours) - recommended
# 4. Aggressive (5 hours) - high compression
# 5. High Quality (8 hours) - best results
```

### 3. Test Student Model
```bash
python test_trained_student.py --model_path ./train_process/distillation-[config]/checkpoints/student_best.pt
```

### 4. Translation Inference
```bash
python inference_cmn_hun_transformer.py
```

## 📊 Training Configurations

### Available Configurations
| Configuration | Training Time | Compression Ratio | Use Case |
|---------------|---------------|-------------------|----------|
| **Quick Test** | 15 minutes | 1.8x | Code verification |
| **Conservative** | 2 hours | 1.7x | Stable training |
| **Balanced** | 3 hours | 2.0x | Recommended |
| **Aggressive** | 5 hours | 3.0x | High compression |
| **High Quality** | 8 hours | 1.8x | Best results |

### Expected Performance
- **Model Compression**: 1.7x - 3.0x depending on configuration
- **Inference Speedup**: 1.5x - 3.0x faster than teacher model
- **Training Time**: 15 minutes - 8 hours based on quality requirements

## 🛠️ Technical Features

### Improved Knowledge Distillation
- **Multiple Configurations**: 5 different training modes
- **Advanced Loss Function**: Label smoothing + improved masking
- **Optimized Training**: Cosine annealing + gradient accumulation
- **Comprehensive Monitoring**: TensorBoard integration + detailed logging

### Model Architecture
- **Transformer Encoder-Decoder**: Standard architecture with optimizations
- **Dynamic Compression**: Configurable compression ratios (1.7x - 3.0x)
- **Flexible Sequence Length**: Adaptive to different configurations
- **Vocabulary**: English and Hungarian tokenizers with caching

### Training Optimizations
- **AdamW Optimizer**: With weight decay and advanced scheduling
- **Gradient Management**: Clipping + accumulation for stability
- **Error Handling**: Comprehensive exception handling and recovery
- **Automatic Saving**: Best model selection and checkpoint management

## 🎯 Key Files Description

### Core Training Files
- **`train_cmn_hun_transformer.py`**: Train the teacher model from scratch
- **`improved_distillation_trainer.py`**: Advanced knowledge distillation with 5 configurations
- **`test_trained_student.py`**: Comprehensive testing and evaluation of student models

### Model & Data Files
- **`model/transformer.py`**: Transformer model implementation with optimizations
- **`data/cmn_hun.py`**: English-Hungarian data processing and loading
- **`inference_cmn_hun_transformer.py`**: Translation interface for both models

## 📈 Usage Examples

### Training Workflow
```bash
# 1. Train teacher model (if not already trained)
python train_cmn_hun_transformer.py

# 2. Run knowledge distillation
python improved_distillation_trainer.py
# Choose configuration: 3 (Balanced - recommended)

# 3. Test the trained student model
python test_trained_student.py --model_path ./train_process/distillation-balanced/checkpoints/student_best.pt

# 4. Use for translation
python inference_cmn_hun_transformer.py
```

### Interactive Translation
```bash
# Start interactive translator
python inference_cmn_hun_transformer.py

# Input examples:
en:Hello world
hun:Jó reggelt!
```

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.12+
- torchtext
- tqdm
- tensorboard
- numpy

## 🎉 Project Highlights

This project successfully demonstrates:
- **Advanced Knowledge Distillation**: Multiple training configurations for different needs
- **Flexible Compression**: 1.7x - 3.0x model compression options
- **Improved Training**: Stable and reliable distillation process
- **Complete Pipeline**: From teacher training to student evaluation
- **Production Ready**: Professional-grade training and testing tools

## 📖 Documentation

- `TRAINING_CONFIGURATIONS_GUIDE.md`: Detailed configuration guide
- `IMPROVED_DISTILLATION_SUMMARY.md`: Complete system overview

## 🚀 Future Improvements

- BLEU score evaluation integration
- Beam search decoding options
- Multi-GPU training support
- Web API interface development
- Real-time translation service

## 📄 License

This project is for educational and research purposes.

---

**English-Hungarian Neural Machine Translation with Improved Knowledge Distillation** - Professional, Flexible, and Reliable! 🎯









# 英匈神经机器翻译与改进知识蒸馏

一个完整的基于 Transformer 的英语到匈牙利语神经机器翻译实现，结合改进的知识蒸馏系统，实现模型压缩与加速。

## 🎯 项目概述

本项目实现了一个**先进的神经机器翻译系统**，具备以下特点：

* 使用 Transformer 架构将**英语翻译为匈牙利语**
* 使用改进的知识蒸馏系统，创建高效压缩模型
* 提供多种训练配置以满足不同需求
* 包含完整的训练、测试和评估流程

## 📁 项目结构

```
├── data/
│   ├── cmn_hun.py                      # 英匈翻译数据预处理
│   └── hun-eng/                        # 匈牙利语-英语平行语料
├── model/
│   └── transformer.py                  # Transformer 模型实现
├── train_cmn_hun_transformer.py        # 教师模型训练脚本
├── improved_distillation_trainer.py    # 改进的知识蒸馏训练脚本
├── test_trained_student.py             # 学生模型测试与评估
├── inference_cmn_hun_transformer.py    # 翻译推理脚本
└── train_process/                      # 训练输出及检查点
    ├── transformer-cmn-hun/            # 教师模型检查点
    └── distillation-[config]/          # 不同配置下学生模型检查点
```

## 🚀 快速开始

### 1️⃣ 训练教师模型

```bash
python train_cmn_hun_transformer.py
```

### 2️⃣ 进行知识蒸馏训练

```bash
# 启动改进的蒸馏训练
python improved_distillation_trainer.py

# 可选择以下 5 种配置：
# 1. 快速测试（15 分钟）- 验证用
# 2. 稳健模式（2 小时）- 稳定训练
# 3. 平衡模式（3 小时）- 推荐
# 4. 激进模式（5 小时）- 高压缩率
# 5. 高质量模式（8 小时）- 最佳效果
```

### 3️⃣ 测试学生模型

```bash
python test_trained_student.py --model_path ./train_process/distillation-[config]/checkpoints/student_best.pt
```

### 4️⃣ 翻译推理

```bash
python inference_cmn_hun_transformer.py
```

## 📊 训练配置

### 可用配置

| 配置        | 训练时长  | 压缩比  | 适用场景 |
| --------- | ----- | ---- | ---- |
| **快速测试**  | 15 分钟 | 1.8x | 验证代码 |
| **稳健模式**  | 2 小时  | 1.7x | 稳定训练 |
| **平衡模式**  | 3 小时  | 2.0x | 推荐   |
| **激进模式**  | 5 小时  | 3.0x | 高压缩  |
| **高质量模式** | 8 小时  | 1.8x | 最佳效果 |

### 预期性能

* **模型压缩**：1.7x - 3.0x（视配置而定）
* **推理加速**：比教师模型快 1.5x - 3.0x
* **训练时长**：15 分钟 - 8 小时（根据质量要求选择）

## 🛠️ 技术特性

### 改进的知识蒸馏

* **多配置支持**：5 种不同训练模式
* **高级损失函数**：标签平滑 + 改进的掩码处理
* **优化训练**：余弦退火调度 + 梯度累积
* **完整监控**：集成 TensorBoard + 详细日志记录

### 模型架构

* **Transformer 编码器-解码器结构**：标准架构并进行优化
* **动态压缩**：支持 1.7x - 3.0x 灵活压缩
* **灵活序列长度**：可根据不同配置自适应调整
* **词汇表**：英语与匈牙利语分词器并带缓存

### 训练优化

* **AdamW 优化器**：支持权重衰减与高级调度
* **梯度管理**：剪裁 + 累积确保稳定
* **错误处理**：完善的异常捕捉与恢复机制
* **自动保存**：最佳模型自动选择与检查点管理

## 🎯 核心文件说明

### 核心训练文件

* **`train_cmn_hun_transformer.py`**：训练教师模型
* **`improved_distillation_trainer.py`**：进行改进的知识蒸馏
* **`test_trained_student.py`**：测试和评估学生模型

### 模型与数据文件

* **`model/transformer.py`**：优化的 Transformer 模型实现
* **`data/cmn_hun.py`**：英语-匈牙利语数据处理
* **`inference_cmn_hun_transformer.py`**：翻译推理接口

## 📈 使用示例

### 完整训练流程

```bash
# 1. 训练教师模型（如尚未训练）
python train_cmn_hun_transformer.py

# 2. 运行知识蒸馏
python improved_distillation_trainer.py
# 建议选择配置：3（平衡模式）

# 3. 测试学生模型
python test_trained_student.py --model_path ./train_process/distillation-balanced/checkpoints/student_best.pt

# 4. 进行翻译推理
python inference_cmn_hun_transformer.py
```

### 交互式翻译

```bash
# 启动交互式翻译器
python inference_cmn_hun_transformer.py

# 输入示例：
en:Hello world
hun:Jó reggelt!
```

## 🔧 依赖环境

* Python 3.8+
* PyTorch 1.12+
* torchtext
* tqdm
* tensorboard
* numpy

## 🎉 项目亮点

本项目成功展示：

* **先进的知识蒸馏方法**：支持多种训练配置以满足不同需求
* **灵活压缩能力**：支持 1.7x - 3.0x 模型压缩
* **改进训练流程**：稳定、可靠的蒸馏过程
* **完整管线**：涵盖教师训练到学生评估全流程
* **生产可用**：专业级训练与测试工具

## 📖 文档

* `TRAINING_CONFIGURATIONS_GUIDE.md`：详细配置使用指南
* `IMPROVED_DISTILLATION_SUMMARY.md`：完整系统综述

## 🚀 后续改进计划

* 集成 BLEU 分数自动评估
* 增加 Beam Search 解码选项
* 支持多 GPU 训练
* 开发 Web API 接口
* 部署实时翻译服务

## 📄 许可证

本项目仅供教学与研究用途。

---

**英匈神经机器翻译 + 改进知识蒸馏**
专业、灵活、可靠，助你快速搭建高质量翻译系统！ 🎯

---

如果需要，我可以继续帮你生成该项目的完整可执行**训练、推理与可视化脚本**以便直接跑起来练习深度学习完整流程，告诉我即可。

