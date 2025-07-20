#!/usr/bin/env python3
"""
基于Hydra的TreeSAT多模态embedding训练脚本
借鉴OmniSat项目的优秀实践
"""

from typing import List, Optional, Tuple
import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# 添加项目根目录到Python路径
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加src目录到Python路径  
src_dir = os.path.join(project_root, '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.utils import utils, pylogger, logging_utils

log = pylogger.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """训练模型。可以在训练期间额外评估测试集，使用训练期间获得的最佳权重。

    此方法包装在可选的@task_wrapper装饰器中，该装饰器控制失败期间的行为。
    对于多次运行、保存崩溃信息等很有用。

    Args:
        cfg (DictConfig): Hydra组合的配置。

    Returns:
        Tuple[dict, dict]: 包含指标的字典和包含所有实例化对象的字典。
    """
    
    # 设置矩阵乘法精度以优化性能
    torch.set_float32_matmul_precision('medium')

    # 为pytorch、numpy和python.random中的随机数生成器设置种子
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"实例化数据模块 <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"实例化模型 <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("实例化回调...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("实例化日志记录器...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    
    log.info(f"实例化训练器 <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("记录超参数!")
        logging_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("编译模型!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("开始训练!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("开始测试!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("找不到最佳ckpt! 使用当前权重进行测试...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"最佳ckpt路径: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # 合并训练和测试指标
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """主函数"""
    # 应用额外的实用程序
    # (例如，如果cfg中没有提供标签则询问标签，打印cfg树等)
    utils.extras(cfg)

    # 训练模型
    metric_dict, _ = train(cfg)

    # 安全地检索用于基于hydra的超参数优化的指标值
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # 返回优化的指标
    return metric_value


if __name__ == "__main__":
    main() 