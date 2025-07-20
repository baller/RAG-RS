import os
import sys
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import warnings
import wandb
warnings.filterwarnings("ignore")

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.embedding import MultiModalEmbeddingModel
from data.TreeSAT import TreeSAT
from data.datamodule import DataModule
from data.transforms.transform import TransformMAE


def setup_data(args):
    """设置数据模块"""
    # 添加导入路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data.utils import get_treesat_classes
    
    # 动态从TreeSAT标签文件中读取所有类别
    classes = get_treesat_classes(args.data_path, verbose=True)
    
    # 数据变换
    transform = TransformMAE(p=args.augmentation_prob, size=args.image_size)
    
    # 模态设置
    modalities = ['aerial', 's1', 's2']
    
    # 数据集构建器
    def train_dataset_builder():
        return TreeSAT(
            path=args.data_path,
            modalities=modalities,
            transform=transform,
            split='train',
            classes=classes,
            partition=args.data_partition
        )
    
    def val_dataset_builder():
        return TreeSAT(
            path=args.data_path,
            modalities=modalities,
            transform=transform,
            split='val',
            classes=classes,
            partition=1.0  # 验证集使用全部数据
        )
    
    def test_dataset_builder():
        return TreeSAT(
            path=args.data_path,
            modalities=modalities,
            transform=transform,
            split='test',
            classes=classes,
            partition=1.0  # 测试集使用全部数据
        )
    
    # 创建数据模块
    datamodule = DataModule(
        train_dataset=train_dataset_builder,
        val_dataset=val_dataset_builder,
        test_dataset=test_dataset_builder,
        global_batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_nodes=args.num_nodes,
        num_devices=args.num_devices
    )
    
    return datamodule


def setup_model(args):
    """设置模型"""
    model = MultiModalEmbeddingModel(
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        modality_weights=args.modality_weights,
        backbone=args.backbone,
        log_wandb=(args.logger == 'wandb')
    )
    
    return model


def setup_callbacks(args):
    """设置回调函数"""
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='embedding-{epoch:02d}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor='val/total_loss',
            mode='min',
            patience=args.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(args):
    """设置日志记录器"""
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name='embedding_logs',
            version=args.experiment_name
        )
    elif args.logger == 'wandb':
        # 初始化wandb配置
        wandb_config = {
            'embed_dim': args.embed_dim,
            'temperature': args.temperature,
            'backbone': args.backbone,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'warmup_epochs': args.warmup_epochs,
            'modality_weights': args.modality_weights,
            'data_partition': args.data_partition,
            'image_size': args.image_size,
            'augmentation_prob': args.augmentation_prob,
        }
        
        logger = WandbLogger(
            name=args.experiment_name,
            project='multimodal-rs-embedding',
            save_dir=args.output_dir,
            config=wandb_config,
            tags=[f'backbone_{args.backbone}', f'embed_dim_{args.embed_dim}'],
            notes=f'Multi-modal RS embedding training with {args.backbone} backbone'
        )
    else:
        logger = None
    
    return logger


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='多模态遥感Embedding训练')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='/data/AnySat/TreeSat/',
                       help='TreeSAT数据集路径')
    parser.add_argument('--data_partition', type=float, default=1.0,
                       help='使用的数据比例 (0.0-1.0)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='输入图像尺寸')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='数据增强概率')
    
    # 模型相关参数
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding维度')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='对比学习温度参数')
    parser.add_argument('--backbone', type=str, default='vit_b_16',
                       choices=['resnet50', 'vit_b_16'],
                       help='Backbone网络类型')
    parser.add_argument('--modality_weights', type=float, nargs=3, 
                       default=[1.0, 1.0, 1.0],
                       help='三种模态对的权重 [aerial-s1, aerial-s2, s1-s2]')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='学习率预热轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='早停耐心值，0表示不使用早停')
    
    # 计算资源参数
    parser.add_argument('--num_devices', type=int, default=2,
                       help='GPU数量')
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='节点数量')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['16-mixed', '32', 'bf16-mixed'],
                       help='训练精度')
    
    # 输出和日志参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='multimodal_embedding',
                       help='实验名称')
    parser.add_argument('--logger', type=str, default='wandb',
                       choices=['tensorboard', 'wandb', 'none'],
                       help='日志记录器类型')
    
    # 其他参数
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--compile_model', action='store_true',
                       help='是否编译模型以加速训练 (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    L.seed_everything(args.seed)
    
    # 优化RTX 4090等GPU的Tensor Cores性能
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')  # 或 'high'
        print("✅ 已启用Tensor Cores优化")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化wandb（如果使用）
    if args.logger == 'wandb':
        # 设置wandb API key（可选）
        # if 'WANDB_API_KEY' in os.environ:
        #     wandb.login(key=os.environ['WANDB_API_KEY'])
        
        # 初始化wandb run
        wandb.init(
            project='multimodal-rs-embedding',
            name=args.experiment_name,
            config=vars(args),
            tags=[f'backbone_{args.backbone}', f'embed_dim_{args.embed_dim}'],
            notes=f'Multi-modal RS embedding training with {args.backbone} backbone',
            dir=args.output_dir
        )
    
    # 打印配置信息
    print("=" * 50)
    print("训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # 设置数据
    print("设置数据模块...")
    datamodule = setup_data(args)
    
    # 设置模型
    print("设置模型...")
    model = setup_model(args)
    
    # 模型编译 (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        print("编译模型...")
        model = torch.compile(model)
    
    # 设置回调函数
    print("设置回调函数...")
    callbacks = setup_callbacks(args)
    
    # 设置日志记录器
    print("设置日志记录器...")
    logger = setup_logger(args)
    
    # 设置训练器
    print("设置训练器...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.num_devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_model_summary=True,
        gradient_clip_val=1.0,  # 梯度裁剪
        deterministic=True,
        strategy='ddp' if args.num_devices > 1 else 'auto'
    )
    
    # 开始训练
    print("开始训练...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # 测试最佳模型
    print("测试最佳模型...")
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path='best'
    )
    
    # 保存最终的embedding提取器
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"最佳模型保存在: {best_model_path}")
    
    # 保存训练完成的标志
    with open(os.path.join(args.output_dir, 'training_completed.txt'), 'w') as f:
        f.write(f"Training completed successfully!\n")
        f.write(f"Best model: {best_model_path}\n")
        f.write(f"Best validation loss: {trainer.checkpoint_callback.best_model_score}\n")
    
    # 如果使用wandb，记录最佳模型和完成训练
    if args.logger == 'wandb' and wandb.run is not None:
        # 记录最佳模型路径
        wandb.run.summary["best_model_path"] = best_model_path
        wandb.run.summary["best_val_loss"] = float(trainer.checkpoint_callback.best_model_score)
        wandb.run.summary["total_epochs"] = trainer.current_epoch
        wandb.run.summary["training_status"] = "completed"
        
        # 如果最佳模型文件存在，保存为wandb artifact
        if os.path.exists(best_model_path):
            artifact = wandb.Artifact(
                name=f"multimodal_embedding_model_{args.experiment_name}",
                type="model",
                description=f"Best multimodal embedding model with {args.backbone} backbone",
                metadata={
                    "backbone": args.backbone,
                    "embed_dim": args.embed_dim,
                    "best_val_loss": float(trainer.checkpoint_callback.best_model_score),
                    "epochs": trainer.current_epoch
                }
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        
        # 完成wandb run
        wandb.finish()
    
    print("训练完成!")


if __name__ == '__main__':
    main()
