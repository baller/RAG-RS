#!/usr/bin/env python3
"""
测试基于Hydra的训练系统
"""

import os
import sys

# 设置环境变量避免分布式训练问题
os.environ['WANDB_MODE'] = 'offline'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 强制使用单GPU

# 添加src目录到路径
sys.path.append('src')

def test_hydra_config():
    """测试Hydra配置系统是否正常工作"""
    print("🔧 测试Hydra配置系统...")
    
    try:
        import hydra
        from omegaconf import DictConfig
        
        @hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
        def test_config(cfg: DictConfig):
            print("✅ Hydra配置加载成功")
            print(f"  实验名称: {cfg.experiment_name}")
            print(f"  数据路径: {cfg.data_dir}")
            print(f"  模型: {cfg.model.name}")
            print(f"  设备数量: {cfg.trainer.devices}")
            print(f"  批次大小: {cfg.dataset.global_batch_size}")
            print(f"  最大轮数: {cfg.max_epochs}")
            return True
            
        return test_config()
        
    except Exception as e:
        print(f"❌ Hydra配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_hydra_train():
    """测试简单的Hydra训练"""
    print("\n🚀 测试Hydra训练系统...")
    
    # 使用简单配置进行测试
    cmd = """python src/train_hydra.py \
    trainer=gpu \
    trainer.devices=1 \
    max_epochs=1 \
    dataset.global_batch_size=4 \
    partition=0.001 \
    train=true \
    test=false \
    offline=true"""
    
    print("执行命令：")
    print(cmd)
    print("\n开始训练...")
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("✅ Hydra训练测试成功！")
        return True
    else:
        print("❌ Hydra训练测试失败")
        return False

def main():
    """主测试函数"""
    print("🧪 TreeSAT Hydra系统测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # 测试1：配置系统
    if test_hydra_config():
        success_count += 1
    
    # 测试2：训练系统
    if test_simple_hydra_train():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！")
        print("\n🎯 现在可以使用Hydra配置系统训练：")
        print("# 基础训练")
        print("python src/train_hydra.py")
        print("\n# 单GPU训练")
        print("python src/train_hydra.py trainer=gpu")
        print("\n# 分布式训练")  
        print("python src/train_hydra.py trainer=ddp")
        print("\n# 自定义参数")
        print("python src/train_hydra.py max_epochs=50 dataset.global_batch_size=64")
        print("\n# 不同backbone")
        print("python src/train_hydra.py model.backbone=vit_b_16")
        
        print("\n✅ Hydra系统优势：")
        print("   1. ✅ 模块化配置管理")
        print("   2. ✅ 优雅的分布式训练")
        print("   3. ✅ 任务包装器和错误处理")
        print("   4. ✅ 改进的日志记录系统")
        print("   5. ✅ 超参数搜索支持")
        print("   6. ✅ 实验管理和复现")
        
    else:
        print("❌ 部分测试失败，请检查错误信息")
        
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 