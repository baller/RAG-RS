import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math
import wandb


class AerialEncoder(nn.Module):
    """Aerial编码器，专门处理高分辨率图像数据，支持ViT和ResNet backbone"""
    def __init__(self, input_channels=4, embed_dim=512, backbone='vit_b_16'):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.backbone_type = backbone
        
        if backbone == 'resnet50':
            # 使用预训练的ResNet50作为backbone
            backbone_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # 修改第一层卷积以适应不同的输入通道数
            if input_channels != 3:
                backbone_model.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # 移除最后的分类层
            backbone_model.fc = nn.Identity()
            self.backbone = backbone_model
            backbone_output_dim = 2048
            
        elif backbone == 'vit_b_16':
            # 使用预训练的Vision Transformer
            backbone_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            
            # 修改patch embedding以适应不同的输入通道数
            if input_channels != 3:
                original_conv = backbone_model.conv_proj
                backbone_model.conv_proj = nn.Conv2d(
                    input_channels, 
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None
                )
                
                # 如果输入通道数不是3，需要调整权重
                if input_channels != 3:
                    with torch.no_grad():
                        if input_channels < 3:
                            # 如果输入通道少于3，取前几个通道的权重
                            backbone_model.conv_proj.weight.data = original_conv.weight.data[:, :input_channels, :, :]
                        else:
                            # 如果输入通道多于3，重复RGB权重
                            new_weight = original_conv.weight.data.repeat(1, input_channels // 3 + 1, 1, 1)
                            backbone_model.conv_proj.weight.data = new_weight[:, :input_channels, :, :]
            
            # 移除分类头
            backbone_model.heads = nn.Identity()
            self.backbone = backbone_model
            backbone_output_dim = 768  # ViT-B/16的输出维度
            
        else:
            raise ValueError(f"AerialEncoder不支持的backbone类型: {backbone}")
        
        # 投影头：将backbone输出映射到embedding空间
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_output_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
    def forward(self, x):
        # 通过backbone提取特征
        features = self.backbone(x)  # (B, backbone_output_dim)
        
        # 通过投影头得到embedding
        embeddings = self.projection_head(features)  # (B, embed_dim)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class MLPEncoder(nn.Module):
    """MLP编码器，专门处理小尺寸时序数据（S1, S2），使用mean时序聚合"""
    def __init__(self, input_channels, embed_dim=512, spatial_size=6, dropout=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        
        # 计算展平后的特征维度
        # 输入: (B, T, C, H, W) -> mean聚合后: (B, C, H, W) -> 展平: (B, C*H*W)
        input_dim = input_channels * spatial_size * spatial_size
        
        # MLP投影网络
        self.projection = nn.Sequential(
            # 第一层：输入维度到中间维度
            nn.Linear(input_dim, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二层：中间维度到中间维度  
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第三层：中间维度到目标维度
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, x):
        # 处理时序数据
        if x.dim() == 5:  # (B, T, C, H, W) - 批次时序数据
            # 使用mean进行时序聚合
            x = torch.mean(x, dim=1)  # (B, C, H, W)
        elif x.dim() == 4 and x.shape[0] > 1:  # (T, C, H, W) - 单样本时序数据
            # 如果是单样本的时序数据，也进行mean聚合
            x = torch.mean(x, dim=0, keepdim=True)  # (1, C, H, W)
        
        # 展平空间维度
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (B, C*H*W)
        
        # 通过MLP得到embedding
        embeddings = self.projection(x)  # (B, embed_dim)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class MultiModalEmbeddingModel(L.LightningModule):
    """多模态embedding训练模型，使用异构编码器架构"""
    
    def __init__(
        self,
        embed_dim=1024,
        temperature=0.07,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=20,
        max_epochs=200,
        modality_weights=None,
        backbone='vit_b_16',
        log_wandb=True,
        lr_scaling=None,
        batch_size=None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 异构编码器架构
        # Aerial: 高分辨率图像 → AerialEncoder(ViT/ResNet)
        self.aerial_encoder = AerialEncoder(
            input_channels=4, 
            embed_dim=embed_dim, 
            backbone=backbone
        )
        
        # S1: 2通道×6×6时序 → MLPEncoder(mean聚合)
        self.s1_encoder = MLPEncoder(
            input_channels=2,
            embed_dim=embed_dim,
            spatial_size=6
        )
        
        # S2: 10通道×6×6时序 → MLPEncoder(mean聚合)
        self.s2_encoder = MLPEncoder(
            input_channels=10,
            embed_dim=embed_dim,
            spatial_size=6
        )
        
        # 超参数
        self.temperature = temperature
        self.base_learning_rate = learning_rate  # 保存原始学习率
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.backbone = backbone
        self.log_wandb = log_wandb
        
        # 学习率缩放逻辑
        self.learning_rate = self._calculate_scaled_lr(learning_rate, lr_scaling, batch_size)
        
        # 模态权重 (aerial-s1, aerial-s2, s1-s2)
        if modality_weights is None:
            self.modality_weights = [1.0, 1.0, 1.0]
        else:
            self.modality_weights = modality_weights
            
        # 用于监控训练过程
        self.automatic_optimization = True
    
    def _calculate_scaled_lr(self, base_lr, lr_scaling, batch_size):
        """根据batch size缩放学习率"""
        if lr_scaling is None or not lr_scaling.get('enabled', False) or batch_size is None:
            return base_lr
        
        rule = lr_scaling.get('rule', 'linear')
        base_batch_size = lr_scaling.get('base_batch_size', 32)
        
        scale_factor = batch_size / base_batch_size
        
        if rule == 'linear':
            scaled_lr = base_lr * scale_factor
        elif rule == 'sqrt':
            scaled_lr = base_lr * math.sqrt(scale_factor)
        elif rule == 'none':
            scaled_lr = base_lr
        else:
            raise ValueError(f"不支持的学习率缩放规则: {rule}")
        
        print(f"🔧 学习率缩放: {base_lr:.2e} → {scaled_lr:.2e} (batch_size: {batch_size}, rule: {rule})")
        return scaled_lr
    
    def _adjust_temperature_for_batch_size(self, base_temp, batch_size):
        """根据batch size调整温度参数
        
        对比学习中，更大的batch size意味着更多负样本，
        可能需要稍微调整温度参数来平衡学习难度
        """
        if batch_size is None:
            return base_temp
            
        # 经验公式：温度随batch size对数增长而轻微增加
        # 这有助于在大batch size时保持合适的学习难度
        temp_adjustment = 1.0 + 0.01 * math.log(batch_size / 32.0) if batch_size > 32 else 1.0
        adjusted_temp = base_temp * temp_adjustment
        
        if abs(adjusted_temp - base_temp) > 0.001:
            print(f"🌡️ 温度参数调整: {base_temp:.3f} → {adjusted_temp:.3f} (batch_size: {batch_size})")
        
        return adjusted_temp
        
    def contrastive_loss(self, embeddings1, embeddings2):
        """计算两个模态之间的对比学习损失"""
        batch_size = embeddings1.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(embeddings1, embeddings2.t()) / self.temperature
        
        # 标签：对角线上的元素为正样本
        labels = torch.arange(batch_size).to(self.device)
        
        # 计算交叉熵损失
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.t(), labels)
        
        return (loss_12 + loss_21) / 2
    
    def triple_contrastive_loss(self, aerial_emb, s1_emb, s2_emb):
        """计算三模态对比学习损失"""
        # 计算三种两两对比损失
        loss_aerial_s1 = self.contrastive_loss(aerial_emb, s1_emb)
        loss_aerial_s2 = self.contrastive_loss(aerial_emb, s2_emb)
        loss_s1_s2 = self.contrastive_loss(s1_emb, s2_emb)
        
        # 加权组合
        total_loss = (
            self.modality_weights[0] * loss_aerial_s1 +
            self.modality_weights[1] * loss_aerial_s2 +
            self.modality_weights[2] * loss_s1_s2
        ) / sum(self.modality_weights)
        
        # 记录各个损失分量
        loss_dict = {
            'aerial_s1_loss': loss_aerial_s1,
            'aerial_s2_loss': loss_aerial_s2,
            's1_s2_loss': loss_s1_s2,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def forward(self, batch):
        """前向传播"""
        embeddings = {}
        
        # 提取各模态的embedding
        if 'aerial' in batch:
            embeddings['aerial'] = self.aerial_encoder(batch['aerial'])
        
        if 's1' in batch:
            embeddings['s1'] = self.s1_encoder(batch['s1'])
        
        if 's2' in batch:
            embeddings['s2'] = self.s2_encoder(batch['s2'])
            
        return embeddings
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 获取embeddings
        embeddings = self(batch)
        
        # 检查是否有足够的模态
        available_modalities = list(embeddings.keys())
        if len(available_modalities) < 2:
            return None
            
        # 计算损失
        if len(available_modalities) == 3:
            # 如果三个模态都存在
            total_loss, loss_dict = self.triple_contrastive_loss(
                embeddings['aerial'], embeddings['s1'], embeddings['s2']
            )
        else:
            # 如果只有两个模态，计算对应的对比损失
            modality_pairs = {
                ('aerial', 's1'): 0,
                ('aerial', 's2'): 1,
                ('s1', 's2'): 2
            }
            
            total_loss = 0
            loss_dict = {}
            count = 0
            
            for (mod1, mod2), weight_idx in modality_pairs.items():
                if mod1 in available_modalities and mod2 in available_modalities:
                    loss = self.contrastive_loss(embeddings[mod1], embeddings[mod2])
                    total_loss += self.modality_weights[weight_idx] * loss
                    loss_dict[f'{mod1}_{mod2}_loss'] = loss
                    count += 1
            
            if count > 0:
                total_loss /= count
                loss_dict['total_loss'] = total_loss
        
        # 记录损失 - 学习OmniSat的日志记录方式
        for key, value in loss_dict.items():
            self.log(
                f'train/{key}',
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )
        
        # 计算embedding质量指标
        self._log_embedding_metrics(embeddings, 'train')
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        embeddings = self(batch)
        
        available_modalities = list(embeddings.keys())
        if len(available_modalities) < 2:
            return None
            
        # 计算验证损失
        if len(available_modalities) == 3:
            total_loss, loss_dict = self.triple_contrastive_loss(
                embeddings['aerial'], embeddings['s1'], embeddings['s2']
            )
        else:
            # 处理两个模态的情况
            modality_pairs = [('aerial', 's1'), ('aerial', 's2'), ('s1', 's2')]
            total_loss = 0
            loss_dict = {}
            count = 0
            
            for mod1, mod2 in modality_pairs:
                if mod1 in available_modalities and mod2 in available_modalities:
                    loss = self.contrastive_loss(embeddings[mod1], embeddings[mod2])
                    total_loss += loss
                    loss_dict[f'{mod1}_{mod2}_loss'] = loss
                    count += 1
            
            if count > 0:
                total_loss /= count
                loss_dict['total_loss'] = total_loss
        
        # 记录验证损失 - 学习OmniSat的日志记录方式
        for key, value in loss_dict.items():
            self.log(
                f'val/{key}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )
        
        # 计算embedding质量指标
        self._log_embedding_metrics(embeddings, 'val')
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        embeddings = self(batch)
        
        available_modalities = list(embeddings.keys())
        if len(available_modalities) < 2:
            return None
            
        # 计算测试损失
        if len(available_modalities) == 3:
            total_loss, loss_dict = self.triple_contrastive_loss(
                embeddings['aerial'], embeddings['s1'], embeddings['s2']
            )
        else:
            # 处理两个模态的情况
            modality_pairs = [('aerial', 's1'), ('aerial', 's2'), ('s1', 's2')]
            total_loss = 0
            loss_dict = {}
            count = 0
            
            for mod1, mod2 in modality_pairs:
                if mod1 in available_modalities and mod2 in available_modalities:
                    loss = self.contrastive_loss(embeddings[mod1], embeddings[mod2])
                    total_loss += loss
                    loss_dict[f'{mod1}_{mod2}_loss'] = loss
                    count += 1
            
            if count > 0:
                total_loss /= count
                loss_dict['total_loss'] = total_loss
        
        # 记录测试损失
        for key, value in loss_dict.items():
            self.log(
                f'test/{key}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )
        
        # 计算embedding质量指标
        self._log_embedding_metrics(embeddings, 'test')
        
        return total_loss
    
    def _log_embedding_metrics(self, embeddings, stage):
        """记录embedding质量指标"""
        for modality, emb in embeddings.items():
            # embedding标准差（衡量特征分布）
            std = emb.std().item()
            self.log(
                f'{stage}/{modality}_embedding_std',
                std,
                on_epoch=True,
                sync_dist=True
            )
            
            # embedding范数
            norm = emb.norm(dim=1).mean().item()
            self.log(
                f'{stage}/{modality}_embedding_norm',
                norm,
                on_epoch=True,
                sync_dist=True
            )
        
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
                    self.log(
                        f'{stage}/{mod1}_{mod2}_similarity',
                        similarity,
                        on_epoch=True,
                        sync_dist=True
                    )
    
    def _log_wandb_metrics(self, embeddings, loss_dict, stage):
        """记录详细的wandb指标"""
        if not hasattr(wandb, 'log') or wandb.run is None:
            return
            
        # 当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        # 基础指标
        wandb_log = {
            f'{stage}/learning_rate': current_lr,
            f'{stage}/epoch': self.current_epoch,
            f'{stage}/global_step': self.trainer.global_step,
        }
        
        # 损失函数指标
        for key, value in loss_dict.items():
            wandb_log[f'{stage}/{key}'] = value.item() if torch.is_tensor(value) else value
        
        # Embedding统计信息
        for modality, emb in embeddings.items():
            if emb.numel() > 0:
                # 基础统计
                wandb_log.update({
                    f'{stage}/{modality}_emb_mean': emb.mean().item(),
                    f'{stage}/{modality}_emb_std': emb.std().item(),
                    f'{stage}/{modality}_emb_min': emb.min().item(),
                    f'{stage}/{modality}_emb_max': emb.max().item(),
                    f'{stage}/{modality}_emb_norm_mean': emb.norm(dim=1).mean().item(),
                })
                
                # 梯度统计（仅训练时）
                if stage == 'train' and hasattr(self, f'{modality}_encoder'):
                    encoder = getattr(self, f'{modality}_encoder')
                    total_norm = 0
                    param_count = 0
                    for param in encoder.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        total_norm = total_norm ** (1. / 2)
                        wandb_log[f'{stage}/{modality}_grad_norm'] = total_norm
        
        # 模态间相似度矩阵
        if len(embeddings) >= 2:
            modalities = list(embeddings.keys())
            similarity_matrix = {}
            for i, mod1 in enumerate(modalities):
                for j, mod2 in enumerate(modalities):
                    if i != j and embeddings[mod1].numel() > 0 and embeddings[mod2].numel() > 0:
                        # 计算批次内的相似度
                        sim = F.cosine_similarity(
                            embeddings[mod1].mean(0, keepdim=True), 
                            embeddings[mod2].mean(0, keepdim=True), 
                            dim=1
                        ).item()
                        similarity_matrix[f'{mod1}_{mod2}'] = sim
            
            # 记录相似度矩阵
            if similarity_matrix:
                wandb_log[f'{stage}/similarity_matrix'] = similarity_matrix
        
        # 温度参数
        wandb_log[f'{stage}/temperature'] = self.temperature
        
        # 模型架构信息（仅第一次记录）
        if self.trainer.global_step == 0 and stage == 'train':
            wandb_log.update({
                'model/backbone': self.backbone,
                'model/embed_dim': self.hparams.embed_dim,
                'model/total_params': sum(p.numel() for p in self.parameters()),
                'model/trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            })
        
        # 记录到wandb (让Lightning自动管理步数)
        wandb.log(wandb_log)
        
        # 可视化embedding分布（每100步记录一次）
        if stage == 'val' and self.trainer.global_step % 100 == 0:
            self._log_embedding_distributions(embeddings)
    
    def _log_embedding_distributions(self, embeddings):
        """记录embedding分布的直方图"""
        if not hasattr(wandb, 'log') or wandb.run is None:
            return
            
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(embeddings), figsize=(5*len(embeddings), 4))
            if len(embeddings) == 1:
                axes = [axes]
            
            for idx, (modality, emb) in enumerate(embeddings.items()):
                if emb.numel() > 0:
                    # 绘制embedding分布直方图
                    emb_flat = emb.detach().cpu().numpy().flatten()
                    axes[idx].hist(emb_flat, bins=50, alpha=0.7, density=True)
                    axes[idx].set_title(f'{modality.upper()} Embedding Distribution')
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Density')
                    axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 记录到wandb
            wandb.log({
                'embeddings/distribution': wandb.Image(fig)
            })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"绘制embedding分布时出错: {e}")
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器：warmup + cosine annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / 
                                          (self.max_epochs - self.warmup_epochs)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class EmbeddingExtractor:
    """用于提取embedding的工具类"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = MultiModalEmbeddingModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(device)
    
    def extract_embeddings(self, dataloader, modalities=['aerial', 's1', 's2']):
        """从数据加载器中提取embeddings"""
        all_embeddings = {mod: [] for mod in modalities}
        all_labels = []
        all_names = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # 提取embeddings
                embeddings = self.model(batch)
                
                # 收集结果
                for mod in modalities:
                    if mod in embeddings:
                        all_embeddings[mod].append(embeddings[mod].cpu())
                
                all_labels.append(batch['label'].cpu())
                all_names.extend(batch['name'])
        
        # 拼接所有批次的结果
        for mod in modalities:
            if all_embeddings[mod]:
                all_embeddings[mod] = torch.cat(all_embeddings[mod], dim=0)
        
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_embeddings, all_labels, all_names
