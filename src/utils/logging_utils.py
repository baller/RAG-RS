from lightning.pytorch.utilities import rank_zero_only
from utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg.get("model", {})

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg.get("datamodule", {})
    hparams["trainer"] = cfg.get("trainer", {})
    
    hparams["callbacks"] = cfg.get("callbacks", {})
    hparams["logger"] = cfg.get("logger", {})
    
    hparams["task_name"] = cfg.get("task_name", "multimodal_embedding")
    hparams["tags"] = cfg.get("tags", ["train"])
    hparams["ckpt_path"] = cfg.get("ckpt_path", None)
    hparams["seed"] = cfg.get("seed", 42)

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams) 