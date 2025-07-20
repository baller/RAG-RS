#!/usr/bin/env python3
"""
Weights & Biases (wandb) 使用示例

这个脚本演示了如何：
1. 配置wandb进行实验跟踪
2. 使用不同的backbone进行实验对比
3. 进行超参数搜索
4. 分析实验结果
"""

import os
import wandb
import subprocess
import argparse


def run_single_experiment(config, project_name="multimodal-rs-embedding"):
    """运行单个实验"""
    
    # 构建训练命令
    cmd = [
        "python", "src/train_embedding.py",
        "--data_path", config["data_path"],
        "--output_dir", f"./outputs/{config['exp_name']}",
        "--embed_dim", str(config["embed_dim"]),
        "--backbone", config["backbone"],
        "--temperature", str(config["temperature"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--max_epochs", str(config["max_epochs"]),
        "--logger", "wandb",
        "--experiment_name", config["exp_name"]
    ]
    
    # 添加模态权重
    if "modality_weights" in config:
        cmd.extend(["--modality_weights"] + [str(w) for w in config["modality_weights"]])
    
    print(f"运行实验: {config['exp_name']}")
    print(f"命令: {' '.join(cmd)}")
    
    # 运行训练
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"实验 {config['exp_name']} 完成成功")
    else:
        print(f"实验 {config['exp_name']} 失败:")
        print(result.stderr)
    
    return result.returncode == 0


def backbone_comparison_experiments(data_path):
    """backbone比较实验"""
    
    print("=" * 50)
    print("Backbone比较实验")
    print("=" * 50)
    
    base_config = {
        "data_path": data_path,
        "embed_dim": 512,
        "temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "max_epochs": 50,  # 较短的训练轮数用于比较
    }
    
    backbones = ["resnet50", "vit_b_16"]
    
    for backbone in backbones:
        config = base_config.copy()
        config["backbone"] = backbone
        config["exp_name"] = f"backbone_comparison_{backbone}"
        
        # ViT需要更小的batch size（通常显存占用更大）
        if backbone == "vit_b_16":
            config["batch_size"] = 32
        
        success = run_single_experiment(config)
        if not success:
            print(f"Backbone {backbone} 实验失败，跳过")


def hyperparameter_sweep(data_path, sweep_config=None):
    """超参数搜索实验"""
    
    print("=" * 50)
    print("超参数搜索实验")
    print("=" * 50)
    
    if sweep_config is None:
        sweep_config = {
            'method': 'bayes',  # 贝叶斯优化
            'metric': {
                'name': 'val/total_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'embed_dim': {
                    'values': [256, 512, 768]
                },
                'temperature': {
                    'min': 0.05,
                    'max': 0.15
                },
                'learning_rate': {
                    'values': [5e-5, 1e-4, 2e-4, 5e-4]
                },
                'batch_size': {
                    'values': [32, 64, 128]
                },
                'backbone': {
                    'values': ['resnet50', 'vit_b_16']
                }
            }
        }
    
    # 初始化sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="multimodal-rs-embedding-sweep"
    )
    
    def train_function():
        """训练函数，由wandb agent调用"""
        # 初始化wandb run
        run = wandb.init()
        config = wandb.config
        
        # 构建实验配置
        exp_config = {
            "data_path": data_path,
            "exp_name": f"sweep_{run.id}",
            "embed_dim": config.embed_dim,
            "backbone": config.backbone,
            "temperature": config.temperature,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": 30,  # 更短的训练用于快速搜索
        }
        
        # 运行实验
        success = run_single_experiment(exp_config)
        
        if not success:
            # 如果训练失败，记录失败状态
            wandb.log({"training_status": "failed"})
        
        # 完成run
        wandb.finish()
    
    print(f"启动wandb sweep: {sweep_id}")
    print("运行以下命令开始搜索:")
    print(f"wandb agent {sweep_id}")
    
    return sweep_id


def embedding_dimension_study(data_path):
    """embedding维度研究"""
    
    print("=" * 50)
    print("Embedding维度影响研究")
    print("=" * 50)
    
    base_config = {
        "data_path": data_path,
        "backbone": "resnet50",
        "temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "max_epochs": 30,
    }
    
    embed_dims = [128, 256, 512, 768, 1024]
    
    for embed_dim in embed_dims:
        config = base_config.copy()
        config["embed_dim"] = embed_dim
        config["exp_name"] = f"embed_dim_study_{embed_dim}"
        
        success = run_single_experiment(config)
        if not success:
            print(f"Embedding维度 {embed_dim} 实验失败，跳过")


def temperature_ablation(data_path):
    """温度参数消融实验"""
    
    print("=" * 50)
    print("温度参数消融实验")
    print("=" * 50)
    
    base_config = {
        "data_path": data_path,
        "backbone": "resnet50",
        "embed_dim": 512,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "max_epochs": 30,
    }
    
    temperatures = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    
    for temp in temperatures:
        config = base_config.copy()
        config["temperature"] = temp
        config["exp_name"] = f"temperature_ablation_{temp}"
        
        success = run_single_experiment(config)
        if not success:
            print(f"温度参数 {temp} 实验失败，跳过")


def analyze_experiments():
    """分析wandb实验结果"""
    
    print("=" * 50)
    print("实验结果分析")
    print("=" * 50)
    
    try:
        api = wandb.Api()
        
        # 获取项目中的所有runs
        runs = api.runs("multimodal-rs-embedding")
        
        print(f"找到 {len(runs)} 个实验")
        
        # 分析结果
        results = []
        for run in runs:
            if run.state == "finished":
                results.append({
                    "name": run.name,
                    "backbone": run.config.get("backbone", "unknown"),
                    "embed_dim": run.config.get("embed_dim", "unknown"),
                    "temperature": run.config.get("temperature", "unknown"),
                    "best_val_loss": run.summary.get("best_val_loss", float('inf')),
                    "total_epochs": run.summary.get("total_epochs", 0)
                })
        
        # 排序并显示最佳结果
        results.sort(key=lambda x: x["best_val_loss"])
        
        print("\n最佳实验结果 (按验证损失排序):")
        print("-" * 80)
        for i, result in enumerate(results[:10]):  # 显示前10个
            print(f"{i+1:2d}. {result['name'][:20]:20s} | "
                  f"Loss: {result['best_val_loss']:.4f} | "
                  f"Backbone: {result['backbone']:10s} | "
                  f"Dim: {result['embed_dim']:4s} | "
                  f"Temp: {result['temperature']}")
        
    except Exception as e:
        print(f"分析实验结果时出错: {e}")
        print("请确保已登录wandb且有访问权限")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='wandb实验管理')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='TreeSAT数据集路径')
    parser.add_argument('--experiment', type=str, 
                       choices=['backbone', 'sweep', 'embed_dim', 'temperature', 'analyze'],
                       default='backbone',
                       help='实验类型')
    parser.add_argument('--wandb_key', type=str, default=None,
                       help='wandb API key (可选)')
    
    args = parser.parse_args()
    
    # 设置wandb API key
    if args.wandb_key:
        os.environ['WANDB_API_KEY'] = args.wandb_key
    
    # 检查wandb登录状态
    try:
        wandb.login()
        print("wandb登录成功")
    except Exception as e:
        print(f"wandb登录失败: {e}")
        print("请运行 'wandb login' 或设置WANDB_API_KEY环境变量")
        return
    
    # 运行指定的实验
    if args.experiment == 'backbone':
        backbone_comparison_experiments(args.data_path)
    elif args.experiment == 'sweep':
        sweep_id = hyperparameter_sweep(args.data_path)
        print(f"\n超参数搜索已启动，sweep ID: {sweep_id}")
    elif args.experiment == 'embed_dim':
        embedding_dimension_study(args.data_path)
    elif args.experiment == 'temperature':
        temperature_ablation(args.data_path)
    elif args.experiment == 'analyze':
        analyze_experiments()
    
    print("\n实验管理完成!")


if __name__ == '__main__':
    print("Weights & Biases 实验管理工具")
    print("=" * 40)
    
    print("\n使用示例:")
    print("1. Backbone比较实验:")
    print("   python src/wandb_example.py --data_path /path/to/TreeSAT --experiment backbone")
    print("\n2. 超参数搜索:")
    print("   python src/wandb_example.py --data_path /path/to/TreeSAT --experiment sweep")
    print("\n3. 分析实验结果:")
    print("   python src/wandb_example.py --experiment analyze")
    
    print("\n开始运行...")
    main() 