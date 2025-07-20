from typing import Sequence

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.tree import Tree
from lightning.pytorch.utilities import rank_zero_only

from utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to save config to file.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(branch_content)

    # print config tree
    console = Console()
    console.print(tree)

    # save config tree to file
    if save_to_file:
        with open("config_tree.log", "w") as file:
            file_console = Console(file=file, width=120)
            file_console.print(tree)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""

    if not cfg.get("tags"):
        if "id" in cfg.get("hydra", {}).get("job", {}):
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = input(">>> Enter a list of comma separated tags (use 'dev' for development): ")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open("tags.log", "w") as file:
            file.write(str(tags))

        log.info(f"Tags: {tags}")

    else:
        log.info(f"Tags provided in config: {cfg.tags}")

    if save_to_file:
        with open("tags.log", "w") as file:
            file.write(str(cfg.get("tags", []))) 