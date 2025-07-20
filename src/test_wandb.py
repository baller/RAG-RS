#!/usr/bin/env python3
"""
测试wandb集成的简单脚本
"""
import os
import torch
import wandb
from models.embedding import MultiModalEmbeddingModel

def test_wandb_integration():
    """测试wandb集成是否正常工作"""
    print("测试wandb集成...")
    
    try:
        # 设置wandb为离线模式进行测试
        os.environ['WANDB_MODE'] = 'offline'
        
        # 初始化wandb
        wandb.init(
            project="treesat-embedding-test",
            name="test-run",
            mode="offline"
        )
        
        # 创建模型
        model = MultiModalEmbeddingModel(
            embed_dim=128,
            backbone='resnet50',
            temperature=0.07,
            learning_rate=1e-4
        )
        
        # 设置为评估模式
        model.eval()
        
        # 创建假数据进行测试
        batch_size = 2
        fake_batch = {
            'aerial': torch.randn(batch_size, 4, 224, 224),
            's1': torch.randn(batch_size, 10, 2, 224, 224),  # 时序数据
            's2': torch.randn(batch_size, 15, 10, 224, 224),  # 时序数据
        }
        
        # 测试前向传播
        with torch.no_grad():
            embeddings = model(fake_batch)
            print(f"✅ 成功获得embeddings:")
            for modality, emb in embeddings.items():
                print(f"  {modality}: {emb.shape}")
        
        # 测试wandb日志记录
        model._log_wandb_metrics(embeddings, {}, 'test')
        
        print("✅ wandb集成测试成功")
        
        # 完成wandb运行
        wandb.finish()
        
        return True
        
    except Exception as e:
        print(f"❌ wandb集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            wandb.finish()
        except:
            pass
        
        return False

if __name__ == "__main__":
    success = test_wandb_integration()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败，请检查相关配置") 