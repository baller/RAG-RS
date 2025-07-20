from lightning import LightningModule
import torch
import torch.nn.functional as F
from typing import Dict, Any
import hydra
from omegaconf import DictConfig

from utils import pylogger

log = pylogger.get_pylogger(__name__)


class MultiModalEmbeddingModule(LightningModule):
    """多模态embedding训练模块，基于OmniSat架构设计"""
    
    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig = None,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # 实例化模型
        self.model = hydra.utils.instantiate(model)
        
        # 存储配置
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        
        # 编译模型（可选）
        if compile_model:
            log.info("Compiling model for faster training!")
            self.model = torch.compile(self.model)

    def forward(self, batch):
        """前向传播"""
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 获取embeddings
        embeddings = self.model(batch)
        
        # 计算损失
        losses = self._compute_contrastive_losses(embeddings)
        
        # 记录所有损失
        for loss_name, loss_value in losses.items():
            self.log(
                f"train/{loss_name}",
                loss_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True if "total" in loss_name else False
            )
        
        # 记录embedding质量指标
        self._log_embedding_metrics(embeddings, "train")
        
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        with torch.no_grad():
            # 获取embeddings
            embeddings = self.model(batch)
            
            # 计算损失
            losses = self._compute_contrastive_losses(embeddings)
            
            # 记录损失
            for loss_name, loss_value in losses.items():
                self.log(
                    f"val/{loss_name}",
                    loss_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True if "total" in loss_name else False
                )
            
            # 记录embedding质量指标
            self._log_embedding_metrics(embeddings, "val")
        
        return losses["total_loss"]

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        with torch.no_grad():
            embeddings = self.model(batch)
            losses = self._compute_contrastive_losses(embeddings)
            
            for loss_name, loss_value in losses.items():
                self.log(
                    f"test/{loss_name}",
                    loss_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True
                )

    def _compute_contrastive_losses(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算对比学习损失"""
        losses = {}
        total_loss = 0.0
        count = 0
        
        # 所有模态对的组合
        modalities = list(embeddings.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                
                # 计算对比损失
                loss = self._contrastive_loss(embeddings[mod1], embeddings[mod2])
                loss_name = f"{mod1}_{mod2}_loss"
                losses[loss_name] = loss
                total_loss += loss
                count += 1
        
        if count > 0:
            losses["total_loss"] = total_loss / count
        else:
            losses["total_loss"] = torch.tensor(0.0, device=self.device)
        
        return losses

    def _contrastive_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """计算InfoNCE对比损失"""
        # 归一化embeddings
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(emb1, emb2.T) / temperature
        
        # 创建标签（对角线为正样本）
        batch_size = emb1.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算双向损失
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2

    def _log_embedding_metrics(self, embeddings: Dict[str, torch.Tensor], stage: str):
        """记录embedding质量指标"""
        for modality, emb in embeddings.items():
            # embedding标准差（衡量特征分布）
            std = emb.std().item()
            self.log(f"{stage}/{modality}_embedding_std", std, on_epoch=True, sync_dist=True)
            
            # embedding范数
            norm = emb.norm(dim=1).mean().item()
            self.log(f"{stage}/{modality}_embedding_norm", norm, on_epoch=True, sync_dist=True)
        
        # 模态间相似度
        if len(embeddings) >= 2:
            modalities = list(embeddings.keys())
            for i in range(len(modalities)):
                for j in range(i+1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    # 计算批次内平均相似度
                    similarity = F.cosine_similarity(
                        embeddings[mod1].mean(0), embeddings[mod2].mean(0), dim=0
                    ).item()
                    self.log(f"{stage}/{mod1}_{mod2}_similarity", similarity, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 实例化优化器
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        
        if self.scheduler_cfg is None:
            return {"optimizer": optimizer}
        
        # 实例化调度器
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        } 