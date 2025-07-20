#!/usr/bin/env python3
"""
测试TreeSAT类别读取功能
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.utils import get_treesat_classes

def test_class_loading(data_path):
    """测试类别加载功能"""
    print("=" * 50)
    print("测试TreeSAT类别读取功能")
    print("=" * 50)
    
    try:
        # 测试类别读取
        classes = get_treesat_classes(data_path, verbose=True)
        
        print(f"\n成功读取 {len(classes)} 个类别:")
        print("类别列表:")
        for i, cls in enumerate(classes):
            print(f"  {i+1:2d}. {cls}")
        
        # 检查是否包含Pseudotsuga
        if 'Pseudotsuga' in classes:
            print(f"\n✅ 成功：Pseudotsuga 在类别列表中 (位置: {classes.index('Pseudotsuga') + 1})")
        else:
            print(f"\n❌ 错误：Pseudotsuga 不在类别列表中")
        
        return classes
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def test_dataset_creation(data_path):
    """测试数据集创建"""
    print("\n" + "=" * 50)
    print("测试数据集创建功能")
    print("=" * 50)
    
    try:
        from data.TreeSAT import TreeSAT
        from data.transforms.transform import TransformMAE
        
        # 获取类别
        classes = get_treesat_classes(data_path, verbose=False)
        
        # 创建数据变换
        transform = TransformMAE(p=0.0, size=224)
        
        # 尝试创建数据集（只测试很小的分区避免内存问题）
        print("尝试创建TreeSAT数据集...")
        dataset = TreeSAT(
            path=data_path,
            modalities=['aerial', 's1', 's2'],
            transform=transform,
            split='train',
            classes=classes,
            partition=0.01  # 只使用1%的数据进行测试
        )
        
        print(f"✅ 成功创建数据集，包含 {len(dataset)} 个样本")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ 成功获取样本，包含模态: {list(sample.keys())}")
            print(f"   标签形状: {sample['label'].shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    # 默认数据路径
    data_path = "/data/AnySat/TreeSat/"
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("请修改data_path为正确的TreeSAT数据集路径")
        return
    
    # 测试类别读取
    classes = test_class_loading(data_path)
    
    if classes is None:
        print("类别读取失败，停止测试")
        return
    
    # 测试数据集创建
    success = test_dataset_creation(data_path)
    
    if success:
        print("\n🎉 所有测试通过！现在可以正常训练模型了。")
    else:
        print("\n❌ 数据集创建测试失败，请检查数据格式")

if __name__ == '__main__':
    main() 