#!/usr/bin/env python3
"""
测试训练好的学生模型
Test the trained student model
"""

import torch
import argparse
import time
from pathlib import Path
from data.cmn_hun import TranslationDataset


def load_model(model_path, device):
    """加载模型"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def safe_translate(model, src_text, src_vocab, tgt_vocab, tokenizer, device, is_english=True, max_len=None):
    """安全的翻译函数"""
    if max_len is None:
        max_len = model.positional_encoding.pe.size(1) - 2
    
    # Tokenize input
    if is_english:
        src_tokens = src_vocab(tokenizer(src_text))
    else:
        src_tokens = src_vocab(tokenizer(src_text))
    
    # Limit input length
    if len(src_tokens) > max_len - 2:
        src_tokens = src_tokens[:max_len - 2]
    
    # Create input tensor
    src_tensor = torch.tensor([0] + src_tokens + [1]).unsqueeze(0).to(device)
    
    # Start generation
    tgt = torch.tensor([[0]]).to(device)
    
    # Generate translation
    for _ in range(max_len):
        if tgt.size(1) >= max_len:
            break
        
        try:
            with torch.no_grad():
                output = model(src_tensor, tgt)
                logits = model.predictor(output[:, -1])
                next_token = torch.argmax(logits, dim=1)
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == 1:  # <eos>
                    break
        except Exception as e:
            print(f"生成错误: {e}")
            break
    
    # Convert to text
    token_list = tgt.squeeze().tolist()
    filtered_tokens = [token for token in token_list if token < len(tgt_vocab) and token not in [0, 1, 2]]
    
    if filtered_tokens:
        result = " ".join(tgt_vocab.lookup_tokens(filtered_tokens))
    else:
        result = "[翻译失败]"
    
    return result


def test_model_quality(model, dataset, device, num_tests=10):
    """测试模型质量"""
    print(f"\n🔍 测试模型质量 (随机 {num_tests} 个样本)...")
    
    test_cases = [
        # English to Hungarian
        ("hi", True),
        ("good morning", True),
        ("thank you", True),
        ("I love you", True),
        ("How are you?", True),
    ]
    
    success_count = 0
    
    for i, (text, is_english) in enumerate(test_cases[:num_tests], 1):
        print(f"\n--- 测试 {i}/{num_tests} ---")
        print(f"输入: {text} ({'英语' if is_english else '匈牙利语'})")
        
        start_time = time.time()
        
        if is_english:
            result = safe_translate(
                model, text, dataset.zh_vocab, dataset.hun_vocab, 
                dataset.zh_tokenizer, device, True
            )
            print(f"匈牙利语: {result}")
        else:
            result = safe_translate(
                model, text, dataset.hun_vocab, dataset.zh_vocab,
                dataset.hun_tokenizer, device, False
            )
            print(f"英语: {result}")
        
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time:.3f}秒")
        
        if result != "[翻译失败]":
            success_count += 1
            print("✅ 翻译成功")
        else:
            print("❌ 翻译失败")
    
    success_rate = success_count / num_tests * 100
    print(f"\n📊 测试结果:")
    print(f"成功率: {success_rate:.1f}% ({success_count}/{num_tests})")
    
    return success_rate

import time

def interactive_test(model, dataset, device):
    """简化版：仅支持英文 -> 匈牙利语翻译"""
    print(f"\n💬 交互式翻译测试")
    print("直接输入英文句子，输入 'quit' 退出")

    while True:
        try:
            user_input = input("\n请输入英文: ").strip()

            if user_input.lower() == 'quit':
                break

            if user_input:
                start_time = time.time()
                result = safe_translate(
                    model, user_input, 
                    dataset.zh_vocab, dataset.hun_vocab,
                    dataset.zh_tokenizer, device, True
                )
                inference_time = time.time() - start_time
                print(f"英文: {user_input}")
                print(f"匈牙利语: {result}")
                print(f"用时: {inference_time:.3f}秒")
            else:
                print("请输入有效的英文句子")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")



def compare_with_teacher(student_model, teacher_model_path, dataset, device):
    """与教师模型对比"""
    print(f"\n⚖️  与教师模型对比...")
    
    if not Path(teacher_model_path).exists():
        print(f"教师模型不存在: {teacher_model_path}")
        return
    
    teacher_model = load_model(teacher_model_path, device)
    
    # Model statistics
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    print(f"\n📊 模型对比:")
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    print(f"压缩比: {teacher_params/student_params:.2f}x")
    print(f"参数减少: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Translation comparison
    test_sentences = ["hi", "good morning", "thank you"]
    
    print(f"\n🔄 翻译质量对比:")
    print(f"{'输入':<15} {'教师模型':<25} {'学生模型':<25}")
    print("-" * 65)
    
    for sentence in test_sentences:
        teacher_result = safe_translate(
            teacher_model, sentence, dataset.zh_vocab, dataset.hun_vocab,
            dataset.zh_tokenizer, device, True
        )
        
        student_result = safe_translate(
            student_model, sentence, dataset.zh_vocab, dataset.hun_vocab,
            dataset.zh_tokenizer, device, True
        )
        
        print(f"{sentence:<15} {teacher_result:<25} {student_result:<25}")
    
    # Speed comparison
    print(f"\n⚡ 推理速度对比:")
    test_text = "hello world"
    
    # Teacher speed
    start_time = time.time()
    for _ in range(10):
        safe_translate(teacher_model, test_text, dataset.zh_vocab, dataset.hun_vocab,
                      dataset.zh_tokenizer, device, True)
    teacher_time = (time.time() - start_time) / 10
    
    # Student speed
    start_time = time.time()
    for _ in range(10):
        safe_translate(student_model, test_text, dataset.zh_vocab, dataset.hun_vocab,
                      dataset.zh_tokenizer, device, True)
    student_time = (time.time() - start_time) / 10
    
    speedup = teacher_time / student_time if student_time > 0 else 0
    
    print(f"教师模型平均时间: {teacher_time:.4f}秒")
    print(f"学生模型平均时间: {student_time:.4f}秒")
    print(f"速度提升: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="测试训练好的学生模型")
    parser.add_argument("--model_path", type=str, required=True, help="学生模型路径")
    parser.add_argument("--teacher_path", type=str, 
                       default="./train_process/transformer-cmn-hun/transformer_checkpoints/best.pt",
                       help="教师模型路径")
    parser.add_argument("--mode", type=str, choices=['test', 'interactive', 'compare', 'all'],
                       default='all', help="测试模式")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # Load dataset
    print("加载数据集...")
    dataset = TranslationDataset("data/hun-eng/hun.txt")
    
    # Load student model
    print(f"加载学生模型: {args.model_path}")
    student_model = load_model(args.model_path, device)
    
    print(f"\n🎯 学生模型测试开始")
    print("=" * 50)
    
    if args.mode in ['test', 'all']:
        success_rate = test_model_quality(student_model, dataset, device)
    
    if args.mode in ['compare', 'all']:
        compare_with_teacher(student_model, args.teacher_path, dataset, device)
    
    
    if args.mode in ['interactive', 'all']:
        interactive_test(student_model, dataset, device)
    
    print(f"\n✅ 测试完成!")


if __name__ == "__main__":
    main()
