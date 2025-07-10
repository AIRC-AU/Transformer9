#!/usr/bin/env python3
"""
Improved Knowledge Distillation Trainer for Transformer Models
重新设计的知识蒸馏训练系统 - 多方案可选
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import math

from data.cmn_hun import TranslationDataset
from model.transformer import TranslationModel


class ImprovedDistillationLoss(nn.Module):
    """改进的蒸馏损失函数"""
    def __init__(self, temperature=3.0, alpha=0.7, beta=0.3, label_smoothing=0.1):
        super(ImprovedDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=label_smoothing)
        
    def forward(self, student_logits, teacher_logits, hard_targets):
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.reshape(-1, vocab_size)
        teacher_logits_flat = teacher_logits.reshape(-1, vocab_size)
        hard_targets_flat = hard_targets.reshape(-1)
        
        # Create mask for valid positions (not padding)
        mask = (hard_targets_flat != 2)
        
        if mask.sum() == 0:  # All padding
            return torch.tensor(0.0, device=student_logits.device)
        
        # Apply mask
        student_masked = student_logits_flat[mask]
        teacher_masked = teacher_logits_flat[mask]
        targets_masked = hard_targets_flat[mask]
        
        # Soft targets from teacher (with temperature)
        student_soft = F.log_softmax(student_masked / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_masked / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard target loss (cross entropy with label smoothing)
        hard_loss = self.ce_loss(student_masked, targets_masked)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * hard_loss
        
        return total_loss, distillation_loss, hard_loss


class TrainingConfig:
    """训练配置基类"""
    def __init__(self):
        self.name = "base"
        self.epochs = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        self.max_seq_length = 24
        self.student_reduction_factor = 0.5
        self.temperature = 3.0
        self.alpha = 0.7
        self.beta = 0.3
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        self.gradient_clip = 1.0
        self.accumulation_steps = 1
        self.save_every = 5
        self.eval_every = 2
        self.use_scheduler = True
        self.label_smoothing = 0.1
        self.data_subset_ratio = 1.0  # Use full dataset
        
    def get_description(self):
        return f"{self.name} - {self.epochs}轮, 批次{self.batch_size}, 学习率{self.learning_rate}"


class QuickTestConfig(TrainingConfig):
    """快速测试配置 - 验证代码正确性"""
    def __init__(self):
        super().__init__()
        self.name = "quick_test"
        self.epochs = 3
        self.batch_size = 64
        self.learning_rate = 0.002
        self.max_seq_length = 16
        self.student_reduction_factor = 0.6
        self.data_subset_ratio = 0.1  # Only 10% of data
        self.save_every = 1
        self.eval_every = 1
        
    def get_description(self):
        return "快速测试 - 3轮, 10%数据, 约15分钟"


class ConservativeConfig(TrainingConfig):
    """保守配置 - 稳定训练"""
    def __init__(self):
        super().__init__()
        self.name = "conservative"
        self.epochs = 30
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.max_seq_length = 28
        self.student_reduction_factor = 0.6
        self.temperature = 4.0
        self.alpha = 0.8
        self.beta = 0.2
        self.warmup_steps = 2000
        
    def get_description(self):
        return "保守训练 - 30轮, 60%压缩, 约2小时"


class BalancedConfig(TrainingConfig):
    """平衡配置 - 性能与效率平衡"""
    def __init__(self):
        super().__init__()
        self.name = "balanced"
        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.max_seq_length = 24
        self.student_reduction_factor = 0.5
        self.temperature = 3.5
        self.alpha = 0.75
        self.beta = 0.25
        self.accumulation_steps = 2
        
    def get_description(self):
        return "平衡训练 - 50轮, 50%压缩, 约3小时"


class AggressiveConfig(TrainingConfig):
    """激进配置 - 高压缩比"""
    def __init__(self):
        super().__init__()
        self.name = "aggressive"
        self.epochs = 80
        self.batch_size = 24
        self.learning_rate = 0.0008
        self.max_seq_length = 24
        self.student_reduction_factor = 0.35
        self.temperature = 5.0
        self.alpha = 0.85
        self.beta = 0.15
        self.warmup_steps = 3000
        self.accumulation_steps = 3
        
    def get_description(self):
        return "激进训练 - 80轮, 35%压缩, 约5小时"


class HighQualityConfig(TrainingConfig):
    """高质量配置 - 最佳效果"""
    def __init__(self):
        super().__init__()
        self.name = "high_quality"
        self.epochs = 100
        self.batch_size = 16
        self.learning_rate = 0.0003
        self.max_seq_length = 32
        self.student_reduction_factor = 0.55
        self.temperature = 4.5
        self.alpha = 0.8
        self.beta = 0.2
        self.warmup_steps = 4000
        self.accumulation_steps = 4
        self.weight_decay = 0.02
        
    def get_description(self):
        return "高质量训练 - 100轮, 55%压缩, 约8小时"


def get_config(config_name):
    """获取配置"""
    configs = {
        'quick': QuickTestConfig(),
        'conservative': ConservativeConfig(),
        'balanced': BalancedConfig(),
        'aggressive': AggressiveConfig(),
        'quality': HighQualityConfig()
    }
    return configs.get(config_name, BalancedConfig())


def create_student_model(teacher_model, config, dataset):
    """创建学生模型"""
    teacher_d_model = teacher_model.src_embedding.embedding_dim
    student_d_model = int(teacher_d_model * config.student_reduction_factor)
    
    # Ensure d_model is divisible by num_heads (8)
    student_d_model = max(8, (student_d_model // 8) * 8)
    
    print(f"Teacher d_model: {teacher_d_model}")
    print(f"Student d_model: {student_d_model} ({config.student_reduction_factor:.1%} of teacher)")
    
    student_model = TranslationModel(
        d_model=student_d_model,
        src_vocab=dataset.zh_vocab,
        tgt_vocab=dataset.hun_vocab,
        max_seq_length=config.max_seq_length,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    
    return student_model


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """余弦退火学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(batch, max_seq_length, device):
    """数据整理函数"""
    bs_id, eos_id, pad_id = 0, 1, 2
    src_list, tgt_list = [], []

    for _src, _tgt in batch:
        # Limit sequence length
        src_tensor = _src[:max_seq_length - 2].clone().detach()
        tgt_tensor = _tgt[:max_seq_length - 2].clone().detach()

        # Add special tokens
        processed_src = torch.cat([
            torch.tensor([bs_id], dtype=torch.int64),
            src_tensor,
            torch.tensor([eos_id], dtype=torch.int64)
        ])

        processed_tgt = torch.cat([
            torch.tensor([bs_id], dtype=torch.int64),
            tgt_tensor,
            torch.tensor([eos_id], dtype=torch.int64)
        ])

        # Pad to max length
        src_list.append(pad(
            processed_src,
            (0, max_seq_length - len(processed_src)),
            value=pad_id
        ))

        tgt_list.append(pad(
            processed_tgt,
            (0, max_seq_length - len(processed_tgt)),
            value=pad_id
        ))

    src = torch.stack(src_list).to(device)
    tgt = torch.stack(tgt_list).to(device)
    tgt_y = tgt[:, 1:]  # Target output (shifted)
    tgt = tgt[:, :-1]   # Target input

    return src, tgt, tgt_y


def evaluate_model(model, data_loader, criterion, device, max_batches=50):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, (src, tgt, tgt_y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                output = model(src, tgt)
                logits = model.predictor(output)
                
                # Calculate loss (only hard target loss for evaluation)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = tgt_y.reshape(-1)
                mask = (targets_flat != 2)
                
                if mask.sum() > 0:
                    loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
                    total_loss += loss.item()
                    total_batches += 1
                    
            except Exception as e:
                print(f"Evaluation error at batch {batch_idx}: {e}")
                continue
    
    model.train()
    return total_loss / max(total_batches, 1)


def train_student_model(config):
    """训练学生模型"""
    print(f"\n{'='*60}")
    print(f"开始知识蒸馏训练: {config.get_description()}")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # Paths
    teacher_model_path = "./train_process/transformer-cmn-hun/transformer_checkpoints/best.pt"
    output_dir = Path(f"./train_process/distillation-{config.name}")
    model_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("加载数据集...")
    dataset = TranslationDataset("data/hun-eng/hun.txt")
    
    # Create data subset if needed
    if config.data_subset_ratio < 1.0:
        subset_size = int(len(dataset) * config.data_subset_ratio)
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
        print(f"使用数据子集: {subset_size} 样本")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config.max_seq_length, device),
        num_workers=0,
        pin_memory=False  # Disable pin_memory to avoid CUDA tensor pinning issues
    )
    
    print(f"数据加载完成: {len(dataset)} 样本, {len(train_loader)} 批次")
    
    # Load teacher model
    print("加载教师模型...")
    if not Path(teacher_model_path).exists():
        raise FileNotFoundError(f"教师模型不存在: {teacher_model_path}")
    
    teacher_model = torch.load(teacher_model_path, map_location=device)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Create student model
    print("创建学生模型...")
    student_model = create_student_model(teacher_model, config, dataset.dataset if hasattr(dataset, 'dataset') else dataset)
    student_model.to(device)
    
    # Model info
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params
    
    print(f"\n模型信息:")
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    print(f"压缩比: {compression_ratio:.2f}x")
    print(f"参数减少: {(1 - student_params/teacher_params)*100:.1f}%")
    
    return student_model, teacher_model, train_loader, device, model_dir, log_dir


def run_training_loop(student_model, teacher_model, train_loader, config, device, model_dir, log_dir):
    """完整的训练循环"""

    # Setup training components
    criterion = ImprovedDistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta,
        label_smoothing=config.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config.epochs // config.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        config.warmup_steps,
        total_steps
    ) if config.use_scheduler else None

    # TensorBoard logging
    writer = SummaryWriter(log_dir)

    # Training state
    best_loss = float('inf')
    global_step = 0
    start_time = time.time()

    print(f"\n🚀 开始训练...")
    print(f"总步数: {total_steps}, 预计时间: {config.get_description().split('约')[-1] if '约' in config.get_description() else '未知'}")

    for epoch in range(config.epochs):
        student_model.train()
        epoch_loss = 0
        epoch_distill_loss = 0
        epoch_hard_loss = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, (src, tgt, tgt_y) in enumerate(pbar):
            try:
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_output = teacher_model(src, tgt)
                    teacher_logits = teacher_model.predictor(teacher_output)

                # Student forward pass
                student_output = student_model(src, tgt)
                student_logits = student_model.predictor(student_output)

                # Calculate loss
                total_loss, distill_loss, hard_loss = criterion(
                    student_logits, teacher_logits, tgt_y
                )

                # Scale loss for gradient accumulation
                total_loss = total_loss / config.accumulation_steps

                # Backward pass
                total_loss.backward()

                # Update weights
                if (batch_idx + 1) % config.accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.gradient_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler:
                        scheduler.step()

                    global_step += 1

                # Accumulate losses
                epoch_loss += total_loss.item() * config.accumulation_steps
                epoch_distill_loss += distill_loss.item()
                epoch_hard_loss += hard_loss.item()

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
                pbar.set_postfix({
                    'Loss': f'{total_loss.item() * config.accumulation_steps:.4f}',
                    'LR': f'{current_lr:.6f}'
                })

                # Log to TensorBoard
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/Total', total_loss.item() * config.accumulation_steps, global_step)
                    writer.add_scalar('Loss/Distillation', distill_loss.item(), global_step)
                    writer.add_scalar('Loss/Hard', hard_loss.item(), global_step)
                    writer.add_scalar('Learning_Rate', current_lr, global_step)

            except Exception as e:
                print(f"\n❌ 训练错误 (Epoch {epoch+1}, Batch {batch_idx}): {e}")
                continue

        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        avg_distill = epoch_distill_loss / len(train_loader)
        avg_hard = epoch_hard_loss / len(train_loader)

        elapsed_time = time.time() - start_time
        eta = elapsed_time / (epoch + 1) * (config.epochs - epoch - 1)

        print(f"\nEpoch {epoch+1}/{config.epochs} 完成:")
        print(f"  总损失: {avg_loss:.4f}")
        print(f"  蒸馏损失: {avg_distill:.4f}")
        print(f"  硬目标损失: {avg_hard:.4f}")
        print(f"  已用时间: {elapsed_time/60:.1f}分钟")
        print(f"  预计剩余: {eta/60:.1f}分钟")

        # Save model
        if (epoch + 1) % config.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(student_model, model_dir / 'student_best.pt')
                print(f"  ✅ 保存最佳模型 (损失: {avg_loss:.4f})")

            torch.save(student_model, model_dir / f'student_epoch_{epoch+1}.pt')
            print(f"  💾 保存检查点: epoch_{epoch+1}")

        # Evaluation
        if (epoch + 1) % config.eval_every == 0:
            print("  🔍 评估中...")
            eval_loss = evaluate_model(student_model, train_loader, criterion, device)
            print(f"  评估损失: {eval_loss:.4f}")
            writer.add_scalar('Eval/Loss', eval_loss, epoch)

    # Training completed
    total_time = time.time() - start_time
    print(f"\n🎉 训练完成!")
    print(f"总用时: {total_time/60:.1f}分钟")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {model_dir}")

    writer.close()
    return student_model


if __name__ == "__main__":
    # Show available configurations
    print("🎯 改进的知识蒸馏训练系统")
    print("=" * 60)
    print("可用的训练配置:")
    print("-" * 60)

    configs = ['quick', 'conservative', 'balanced', 'aggressive', 'quality']

    for i, config_name in enumerate(configs, 1):
        config = get_config(config_name)
        print(f"{i}. {config_name:12} - {config.get_description()}")

    print("-" * 60)
    print("请选择训练配置 (输入数字 1-5):")

    try:
        choice = int(input().strip())
        if 1 <= choice <= 5:
            config_name = configs[choice - 1]
            config = get_config(config_name)
            print(f"\n✅ 已选择: {config.get_description()}")

            # Confirm training
            print(f"\n⚠️  即将开始训练，预计用时: {config.get_description().split('约')[-1] if '约' in config.get_description() else '未知'}")
            confirm = input("确认开始训练? (y/N): ").strip().lower()

            if confirm in ['y', 'yes']:
                # Initialize and run training
                student_model, teacher_model, train_loader, device, model_dir, log_dir = train_student_model(config)

                # Run complete training
                trained_model = run_training_loop(
                    student_model, teacher_model, train_loader,
                    config, device, model_dir, log_dir
                )

                print(f"\n🎯 训练完成! 可以使用以下命令测试模型:")
                print(f"python test_trained_student.py --model_path {model_dir}/student_best.pt")

            else:
                print("❌ 训练已取消")
        else:
            print("❌ 无效选择，请输入 1-5 之间的数字")

    except ValueError:
        print("❌ 请输入有效的数字")
    except KeyboardInterrupt:
        print("\n❌ 用户取消操作")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
