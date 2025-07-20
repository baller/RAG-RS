#!/usr/bin/env python3
"""
测试TreeSAT数据加载功能
"""

import os
import sys
import torch

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.TreeSAT import TreeSAT
from data.transforms.transform import TransformMAE
from data.utils import get_treesat_classes

def test_single_sample(data_path):
    """测试单个样本加载"""
    print("=" * 50)
    print("测试单个样本加载")
    print("=" * 50)
    
    try:
        # 获取类别
        classes = get_treesat_classes(data_path, verbose=False)
        
        # 创建数据变换
        transform = TransformMAE(p=0.0, size=224)
        
        # 创建数据集（只使用很小的分区）
        dataset = TreeSAT(
            path=data_path,
            modalities=['aerial', 's1', 's2'],
            transform=transform,
            split='train',
            classes=classes,
            partition=0.01  # 只使用1%的数据
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 测试获取第一个样本
        print("尝试获取第一个样本...")
        sample = dataset[0]
        
        print("✅ 成功获取样本")
        print(f"样本包含的键: {list(sample.keys())}")
        
        # 分析每个模态的数据
        for key, value in sample.items():
            if key in ['label', 'name']:
                if key == 'label':
                    print(f"{key}: 形状={value.shape}, 类型={type(value)}")
                else:
                    print(f"{key}: {value}")
            else:
                if torch.is_tensor(value):
                    print(f"{key}: 形状={value.shape}, 数据类型={value.dtype}, 最小值={value.min():.3f}, 最大值={value.max():.3f}")
                else:
                    print(f"{key}: 类型={type(value)}, 值={value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader(data_path, batch_size=2):
    """测试数据加载器"""
    print("\n" + "=" * 50)
    print("测试数据加载器")
    print("=" * 50)
    
    try:
        # 获取类别
        classes = get_treesat_classes(data_path, verbose=False)
        
        # 创建数据变换
        transform = TransformMAE(p=0.0, size=224)
        
        # 创建数据集
        dataset = TreeSAT(
            path=data_path,
            modalities=['aerial', 's1', 's2'],
            transform=transform,
            split='train',
            classes=classes,
            partition=0.01  # 只使用1%的数据
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 使用0个worker避免多进程问题
            collate_fn=dataset.collate_fn
        )
        
        print(f"数据加载器创建成功，批次大小: {batch_size}")
        
        # 尝试获取第一个批次
        print("尝试获取第一个批次...")
        for i, batch in enumerate(dataloader):
            print(f"✅ 成功获取批次 {i+1}")
            print(f"批次包含的键: {list(batch.keys())}")
            
            # 分析批次中每个模态的数据
            for key, value in batch.items():
                if key in ['label', 'name']:
                    if key == 'label':
                        print(f"{key}: 形状={value.shape}, 类型={type(value)}")
                    else:
                        print(f"{key}: 长度={len(value)}")
                else:
                    if torch.is_tensor(value):
                        print(f"{key}: 形状={value.shape}, 数据类型={value.dtype}")
                    else:
                        print(f"{key}: 类型={type(value)}")
            
            # 只测试第一个批次
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward(data_path):
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("测试模型前向传播")
    print("=" * 50)
    
    try:
        # 创建一个简单的embedding模型进行测试
        from models.embedding import ModalityEncoder
        
        # 创建编码器
        aerial_encoder = ModalityEncoder(input_channels=4, embed_dim=128, backbone='resnet50')
        s1_encoder = ModalityEncoder(input_channels=2, embed_dim=128, backbone='resnet50')
        s2_encoder = ModalityEncoder(input_channels=10, embed_dim=128, backbone='resnet50')
        
        # 设置为评估模式，避免BatchNorm问题
        aerial_encoder.eval()
        s1_encoder.eval()
        s2_encoder.eval()
        
        print("✅ 模型创建成功")
        
        # 获取数据
        classes = get_treesat_classes(data_path, verbose=False)
        transform = TransformMAE(p=0.0, size=224)
        dataset = TreeSAT(
            path=data_path,
            modalities=['aerial', 's1', 's2'],
            transform=transform,
            split='train',
            classes=classes,
            partition=0.01
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,  # 使用批次大小2，避免BatchNorm问题
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        
        # 获取一个批次进行测试
        batch = next(iter(dataloader))
        
        # 测试每个编码器
        with torch.no_grad():  # 禁用梯度计算，节省内存
            if 'aerial' in batch:
                print("测试aerial编码器...")
                aerial_emb = aerial_encoder(batch['aerial'])
                print(f"✅ aerial embedding形状: {aerial_emb.shape}")
            
            if 's1' in batch:
                print("测试s1编码器...")
                s1_emb = s1_encoder(batch['s1'])
                print(f"✅ s1 embedding形状: {s1_emb.shape}")
            
            if 's2' in batch:
                print("测试s2编码器...")
                s2_emb = s2_encoder(batch['s2'])
                print(f"✅ s2 embedding形状: {s2_emb.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    data_path = "/data/AnySat/TreeSat/"
    
    # 检查数据路径
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    # 测试单个样本加载
    success1 = test_single_sample(data_path)
    
    if not success1:
        print("单个样本加载失败，停止后续测试")
        return
    
    # 测试数据加载器
    success2 = test_dataloader(data_path)
    
    if not success2:
        print("数据加载器测试失败，停止后续测试")
        return
    
    # 测试模型前向传播
    success3 = test_model_forward(data_path)
    
    if success1 and success2 and success3:
        print("\n🎉 所有测试通过！数据加载和模型前向传播正常。")
    else:
        print("\n❌ 部分测试失败，请检查相关组件。")

if __name__ == '__main__':
    print("TreeSAT数据加载测试工具")
    print("=" * 50)
    main() 