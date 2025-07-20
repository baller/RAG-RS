"""
适用于Hydra的TreeSAT数据模块
"""

import lightning as L
import torch
from torch.utils.data import DataLoader
from typing import Optional

class HydraDataModule(L.LightningDataModule):
    """适用于Hydra的TreeSAT数据模块"""
    
    def __init__(
        self,
        train_dataset_config,
        val_dataset_config,
        test_dataset_config,
        global_batch_size: int = 32,
        num_workers: int = 4,
        num_nodes: int = 1,
        num_devices: int = 1,
        data_path: str = "/data/AnySat/TreeSat/",
        train_partition: float = 1.0,
    ):
        super().__init__()
        # 保存配置，但不立即实例化数据集
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.test_dataset_config = test_dataset_config
        
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        self.data_path = data_path
        self.train_partition = train_partition
        
        print(f"每个GPU将接收 {self.batch_size} 张图像")
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes = None
        
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        import hydra
        from data.utils import get_treesat_classes
        
        # 只获取一次类别信息
        if self.classes is None:
            self.classes = get_treesat_classes(self.data_path, verbose=True)
        
        if stage == "fit" or stage is None:
            # 创建训练数据集 - 使用配置文件中的值
            from src.data.TreeSAT import TreeSAT
            from src.data.transforms.transform import TransformMAE
            
            train_transform = TransformMAE(p=0.5, size=224)
            self.train_dataset = TreeSAT(
                path=self.data_path,
                modalities=["aerial", "s1", "s2"],
                transform=train_transform,
                split="train",
                classes=self.classes,
                partition=self.train_partition
            )
            
            # 创建验证数据集
            val_transform = TransformMAE(p=0.0, size=224)
            self.val_dataset = TreeSAT(
                path=self.data_path,
                modalities=["aerial", "s1", "s2"],
                transform=val_transform,
                split="val",
                classes=self.classes,
                partition=0.1
            )
            
            print(f"训练数据集大小: {len(self.train_dataset)}")
            print(f"验证数据集大小: {len(self.val_dataset)}")
            
        if stage == "test":
            # 创建测试数据集
            from src.data.TreeSAT import TreeSAT
            from src.data.transforms.transform import TransformMAE
            
            test_transform = TransformMAE(p=0.0, size=224)
            self.test_dataset = TreeSAT(
                path=self.data_path,
                modalities=["aerial", "s1", "s2"],
                transform=test_transform,
                split="test",
                classes=self.classes,
                partition=0.1
            )
            
            print(f"测试数据集大小: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True
        ) 