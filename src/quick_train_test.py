#!/usr/bin/env python3
"""
快速训练测试脚本
使用最小配置测试训练流程是否正常
"""

import subprocess
import sys
import os

def run_quick_test():
    """运行快速训练测试"""
    
    # 构建训练命令，使用最小配置
    cmd = [
        "python", "src/train_embedding.py",
        "--data_path", "/data/AnySat/TreeSat/",
        "--output_dir", "./quick_test_outputs",
        "--embed_dim", "64",  # 非常小的embedding维度
        "--backbone", "resnet50",  # 使用ResNet50，比ViT更稳定
        "--batch_size", "2",  # 非常小的批次大小
        "--max_epochs", "1",  # 只训练1个epoch
        "--warmup_epochs", "0",  # 不使用warmup
        "--data_partition", "0.01",  # 只使用1%的数据
        "--num_workers", "0",  # 不使用多进程
        "--early_stopping_patience", "0",  # 不使用早停
        "--logger", "none",  # 不使用日志记录器
        "--precision", "32"  # 使用32位精度，更稳定
    ]
    
    print("=" * 60)
    print("快速训练测试")
    print("=" * 60)
    print("配置:")
    print("  - 数据: 1%的TreeSAT训练数据")
    print("  - 模型: ResNet50 + 64维embedding")
    print("  - 批次大小: 2")
    print("  - 训练轮数: 1")
    print("  - 精度: 32位")
    print("=" * 60)
    
    print("运行命令:")
    print(" ".join(cmd))
    print()
    
    # 运行训练
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 快速训练测试成功完成！")
            print("现在可以使用完整配置进行正式训练。")
            return True
        else:
            print(f"\n❌ 快速训练测试失败，返回代码: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 运行测试时出错: {e}")
        return False

def run_data_test():
    """先运行数据测试"""
    print("=" * 60)
    print("数据加载测试")
    print("=" * 60)
    
    try:
        cmd = ["python", "src/test_dataloader.py"]
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"数据测试失败: {e}")
        return False

def main():
    """主函数"""
    print("TreeSAT训练问题诊断和测试工具")
    print("=" * 60)
    
    # 检查数据路径
    data_path = "/data/AnySat/TreeSat/"
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("请修改脚本中的data_path为正确路径")
        return
    
    # 步骤1: 数据加载测试
    print("步骤1: 数据加载测试")
    data_ok = run_data_test()
    
    if not data_ok:
        print("❌ 数据加载测试失败，请先解决数据问题")
        return
    
    print("✅ 数据加载测试通过")
    
    # 步骤2: 快速训练测试
    print("\n步骤2: 快速训练测试")
    train_ok = run_quick_test()
    
    if train_ok:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("现在可以运行完整训练:")
        print("python src/train_embedding.py \\")
        print("    --data_path /data/AnySat/TreeSat/ \\")
        print("    --backbone vit_b_16 \\")
        print("    --embed_dim 512 \\")
        print("    --batch_size 32 \\")
        print("    --max_epochs 100 \\")
        print("    --logger wandb")
    else:
        print("\n❌ 训练测试失败，请检查错误信息")

if __name__ == '__main__':
    main() 