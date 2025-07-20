#!/usr/bin/env python3
"""
简化的训练测试脚本，验证所有修复是否有效
"""
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量避免分布式训练问题
os.environ['WANDB_MODE'] = 'offline'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 强制使用单GPU

def test_training():
    """测试训练流程"""
    print("🧪 TreeSAT多模态嵌入训练测试")
    print("=" * 50)
    
    try:
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            print(f"✅ 当前GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  未检测到CUDA，将使用CPU")
        
        # 添加src目录到路径
        sys.path.append('src')
        
        from data.utils import get_treesat_classes
        from data.transforms.transform import TransformMAE
        from data.TreeSAT import TreeSAT
        from models.embedding import MultiModalEmbeddingModel
        import lightning as L
        
        # 数据路径
        data_path = "/data/zhangguiwei/KAN4RSImg/TreeSatAI/TreeSatAI_v1_0_processed"
        
        print("\n📊 测试数据加载...")
        classes = get_treesat_classes(data_path, verbose=False)
        print(f"✅ 加载类别数: {len(classes)}")
        
        # 创建数据集（使用最小数据量测试）
        transform = TransformMAE(p=0.0, size=224)
        dataset = TreeSAT(
            path=data_path,
            modalities=['aerial', 's1', 's2'],
            transform=transform,
            split='train',
            classes=classes,
            partition=0.01  # 使用1%数据进行快速测试
        )
        
        print(f"✅ 数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=dataset.collate_fn
        )
        
        print("✅ 数据加载器创建成功")
        
        print("\n🤖 测试模型创建...")
        # 创建模型
        model = MultiModalEmbeddingModel(
            embed_dim=128,
            backbone='resnet50',
            temperature=0.07,
            learning_rate=1e-4,
            log_wandb=False  # 禁用wandb避免冲突
        )
        
        print("✅ 模型创建成功")
        
        print("\n⚡ 测试简单前向传播...")
        model.eval()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            embeddings = model(batch)
            print(f"✅ 前向传播成功，embedding形状:")
            for modality, emb in embeddings.items():
                print(f"   {modality}: {emb.shape}")
        
        print("\n🏃 测试训练步骤...")
        model.train()
        
        # 测试training_step
        try:
            # 模拟trainer
            model.trainer = type('MockTrainer', (), {
                'global_step': 0,
                'current_epoch': 0,
                'optimizers': [torch.optim.Adam(model.parameters())]
            })()
            
            loss = model.training_step(batch, 0)
            print(f"✅ 训练步骤成功，损失: {loss.item():.4f}")
            
        except Exception as e:
            print(f"❌ 训练步骤失败: {e}")
            return False
        
        print("\n🎯 测试Lightning训练器...")
        
        # 创建Lightning训练器（单GPU，短时间）
        trainer = L.Trainer(
            max_epochs=1,
            devices=1 if torch.cuda.is_available() else 'cpu',
            precision='32',  # 使用FP32避免精度问题
            enable_checkpointing=False,
            logger=False,  # 禁用日志记录器
            enable_model_summary=False,
            num_sanity_val_steps=0,  # 跳过验证sanity check
            limit_train_batches=2,  # 只训练2个批次
            limit_val_batches=0,   # 跳过验证
            enable_progress_bar=True,
            accelerator='auto',
            strategy='auto',  # 自动选择策略
            deterministic=False  # 禁用确定性以避免性能警告
        )
        
        print("✅ Lightning训练器创建成功")
        
        # 重新创建模型（重置状态）
        model = MultiModalEmbeddingModel(
            embed_dim=128,
            backbone='resnet50',
            temperature=0.07,
            learning_rate=1e-4,
            log_wandb=False
        )
        
        print("🚀 开始训练测试...")
        trainer.fit(model, dataloader)
        
        print("🎉 训练测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ 所有测试通过！训练流程正常工作。")
        print("\n🎯 可以开始正式训练：")
        print("   python src/train_embedding.py --num_devices 1 --max_epochs 50")
    else:
        print("\n❌ 测试失败，请检查错误信息并修复问题。") 