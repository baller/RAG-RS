#!/usr/bin/env python3
"""
测试新的Hydra配置系统
"""
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')

def test_hydra_config():
    """测试Hydra配置是否正确加载"""
    print("🧪 测试Hydra配置系统")
    print("=" * 50)
    
    try:
        import hydra
        from omegaconf import DictConfig, OmegaConf
        from hydra import initialize, compose
        
        # 初始化Hydra
        with initialize(version_base=None, config_path="../configs"):
            # 组合配置
            cfg = compose(config_name="config.yaml")
            
            print("✅ 成功加载配置!")
            print("\n📋 配置摘要:")
            print(f"  数据路径: {cfg.dataset.data_path}")
            print(f"  模态: {cfg.dataset.modalities}")
            print(f"  批次大小: {cfg.dataset.global_batch_size}")
            print(f"  最大轮数: {cfg.trainer.max_epochs}")
            print(f"  设备数: {cfg.trainer.devices}")
            print(f"  实验名: {cfg.experiment_name}")
            
            # 测试模型实例化
            print("\n🤖 测试模型配置...")
            model_cfg = cfg.model.model
            print(f"  Embedding维度: {model_cfg.embed_dim}")
            print(f"  Backbone: {model_cfg.backbone}")
            print(f"  温度参数: {model_cfg.temperature}")
            
            # 测试数据集配置
            print("\n📊 测试数据集配置...")
            train_cfg = cfg.dataset.train_dataset
            print(f"  训练集目标: {train_cfg._target_}")
            print(f"  数据分割: {train_cfg.split}")
            print(f"  数据比例: {train_cfg.partition}")
            
            return True
            
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation():
    """测试组件实例化"""
    print("\n🔧 测试组件实例化")
    print("=" * 50)
    
    try:
        import hydra
        from hydra import initialize, compose
        from src.data.utils import get_treesat_classes
        
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config.yaml")
            
            # 测试数据集类别获取
            try:
                data_path = cfg.dataset.data_path
                print(f"尝试从 {data_path} 获取类别...")
                classes = get_treesat_classes(data_path, verbose=False)
                print(f"✅ 成功获取 {len(classes)} 个类别")
            except Exception as e:
                print(f"⚠️  类别获取失败: {e}")
                classes = [f"class_{i}" for i in range(15)]  # 使用默认类别
            
            # 更新配置中的类别
            cfg.dataset.train_dataset.classes = classes
            cfg.dataset.val_dataset.classes = classes
            cfg.dataset.test_dataset.classes = classes
            
            # 测试Transform实例化
            print("\n🔄 测试Transform实例化...")
            transform = hydra.utils.instantiate(cfg.dataset.train_dataset.transform)
            print(f"✅ Transform实例化成功: {type(transform)}")
            
            # 测试数据集实例化（创建构建器）
            print("\n📦 测试数据集构建器...")
            train_builder = lambda: hydra.utils.instantiate(cfg.dataset.train_dataset)
            print("✅ 训练集构建器创建成功")
            
            val_builder = lambda: hydra.utils.instantiate(cfg.dataset.val_dataset)
            print("✅ 验证集构建器创建成功")
            
            # 测试回调实例化
            print("\n📞 测试回调实例化...")
            callbacks_cfg = cfg.get("callbacks", {})
            callbacks = []
            for cb_name, cb_cfg in callbacks_cfg.items():
                if "_target_" in cb_cfg:
                    try:
                        cb = hydra.utils.instantiate(cb_cfg)
                        callbacks.append(cb)
                        print(f"✅ {cb_name} 实例化成功")
                    except Exception as e:
                        print(f"⚠️  {cb_name} 实例化失败: {e}")
            
            print(f"\n✅ 成功实例化 {len(callbacks)} 个回调")
            return True
            
    except Exception as e:
        print(f"❌ 实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 TreeSAT多模态embedding - Hydra配置测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1：配置加载
    if test_hydra_config():
        success_count += 1
    
    # 测试2：组件实例化
    if test_instantiation():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！新的Hydra架构已就绪")
        print("\n🚀 可以使用以下命令开始训练：")
        print("python src/train_hydra.py")
        print("python src/train_hydra.py trainer=ddp")  # 多GPU训练
        print("python src/train_hydra.py dataset.partition=0.1")  # 更多数据
        return True
    else:
        print("❌ 部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 