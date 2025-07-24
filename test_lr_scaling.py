#!/usr/bin/env python3
"""
测试不同batch size下的学习率缩放策略
"""

from src.models.embedding import MultiModalEmbeddingModel
import torch

def test_lr_scaling():
    """测试学习率缩放功能"""
    
    print("📊 批次大小与学习率缩放测试")
    print("=" * 60)
    
    # 基础配置
    base_config = {
        'embed_dim': 512,
        'temperature': 0.07,
        'learning_rate': 1e-4,  # 基础学习率
        'backbone': 'vit_b_16',
        'log_wandb': False
    }
    
    # 学习率缩放配置
    lr_scaling_config = {
        'enabled': True,
        'rule': 'linear',
        'base_batch_size': 32
    }
    
    # 测试不同的batch size
    batch_sizes = [16, 32, 64, 128, 320, 640, 1024]
    
    print("\n1. 线性缩放策略 (Linear Scaling)")
    print("-" * 40)
    for batch_size in batch_sizes:
        model = MultiModalEmbeddingModel(
            **base_config,
            lr_scaling=lr_scaling_config,
            batch_size=batch_size
        )
        scale_factor = batch_size / 32
        print(f"  Batch Size: {batch_size:4d} | LR: {model.learning_rate:.2e} | Scale: {scale_factor:.2f}x")
    
    print("\n2. 平方根缩放策略 (Square Root Scaling)")
    print("-" * 40)
    sqrt_config = lr_scaling_config.copy()
    sqrt_config['rule'] = 'sqrt'
    
    for batch_size in batch_sizes:
        model = MultiModalEmbeddingModel(
            **base_config,
            lr_scaling=sqrt_config,
            batch_size=batch_size
        )
        scale_factor = (batch_size / 32) ** 0.5
        print(f"  Batch Size: {batch_size:4d} | LR: {model.learning_rate:.2e} | Scale: {scale_factor:.2f}x")
    
    print("\n3. 不缩放策略 (No Scaling)")
    print("-" * 40)
    no_scaling_config = lr_scaling_config.copy()
    no_scaling_config['rule'] = 'none'
    
    for batch_size in batch_sizes[:3]:  # 只测试几个
        model = MultiModalEmbeddingModel(
            **base_config,
            lr_scaling=no_scaling_config,
            batch_size=batch_size
        )
        print(f"  Batch Size: {batch_size:4d} | LR: {model.learning_rate:.2e} | Scale: 1.00x")

def test_temperature_adjustment():
    """测试温度参数调整"""
    
    print("\n🌡️ 温度参数调整测试")
    print("=" * 60)
    
    base_config = {
        'embed_dim': 512,
        'temperature': 0.07,
        'learning_rate': 1e-4,
        'backbone': 'vit_b_16',
        'log_wandb': False
    }
    
    batch_sizes = [16, 32, 64, 128, 320, 640, 1024]
    
    for batch_size in batch_sizes:
        model = MultiModalEmbeddingModel(**base_config, batch_size=batch_size)
        adjusted_temp = model._adjust_temperature_for_batch_size(0.07, batch_size)
        ratio = adjusted_temp / 0.07
        print(f"  Batch Size: {batch_size:4d} | Temperature: {adjusted_temp:.4f} | Ratio: {ratio:.3f}x")

def demonstrate_contrastive_scaling():
    """演示对比学习中batch size的影响"""
    
    print("\n🎯 对比学习batch size影响演示")
    print("=" * 60)
    
    # 创建模型
    model = MultiModalEmbeddingModel(
        embed_dim=256, 
        temperature=0.07, 
        log_wandb=False
    )
    model.eval()
    
    batch_sizes = [8, 32, 128]
    
    for batch_size in batch_sizes:
        # 创建随机embedding
        emb1 = torch.randn(batch_size, 256)
        emb2 = torch.randn(batch_size, 256)
        
        # 归一化
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        
        # 计算对比损失
        loss = model.contrastive_loss(emb1, emb2)
        
        # 计算负样本数量
        negative_samples = batch_size - 1
        
        print(f"  Batch Size: {batch_size:3d} | 负样本数: {negative_samples:3d} | Loss: {loss.item():.4f}")

def recommendations():
    """输出建议"""
    
    print("\n💡 实用建议")
    print("=" * 60)
    
    recommendations = [
        "1. 线性缩放 (Linear Scaling):",
        "   - 适用于大多数情况",
        "   - batch_size增大k倍 → lr增大k倍",
        "   - 建议: batch_size=320时, lr=1e-3",
        "",
        "2. 平方根缩放 (Square Root Scaling):",
        "   - 更保守的缩放策略",
        "   - 适用于对学习率敏感的模型",
        "   - batch_size增大k倍 → lr增大√k倍",
        "",
        "3. Warmup策略:",
        "   - 大batch size训练建议使用warmup",
        "   - 建议warmup_epochs = 10-20",
        "   - 从0或小学习率逐渐增加到目标学习率",
        "",
        "4. 温度参数调整:",
        "   - 大batch size可能需要稍微增加温度",
        "   - 帮助平衡更多负样本带来的学习难度",
        "",
        "5. 对比学习特殊考虑:",
        "   - 更大batch size = 更多负样本 = 更难的对比任务",
        "   - 可能需要调整损失权重或温度参数",
        "   - 建议从小batch size开始调试"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")

if __name__ == "__main__":
    test_lr_scaling()
    test_temperature_adjustment()
    demonstrate_contrastive_scaling()
    recommendations()
    
    print(f"\n✅ 学习率缩放测试完成！") 