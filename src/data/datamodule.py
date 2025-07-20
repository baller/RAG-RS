import lightning as L
from torch.utils.data import DataLoader
import time
import hydra
from omegaconf import DictConfig

from data.utils import get_treesat_classes

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        
        # 动态获取类别信息
        if callable(self._builders["train"]):
            # 如果是callable，先实例化一个临时数据集来获取类别信息
            temp_train = self._builders["train"]()
            classes = temp_train.classes if hasattr(temp_train, 'classes') else None
            data_path = temp_train.path if hasattr(temp_train, 'path') else None
            
            # 如果没有类别信息，动态获取
            if classes is None and data_path is not None:
                try:
                    classes = get_treesat_classes(data_path, verbose=True)
                    print(f"动态获取到 {len(classes)} 个类别")
                except Exception as e:
                    print(f"获取类别失败: {e}")
                    classes = None
            
            # 更新所有数据集构建器的类别信息
            for split in ["train", "val", "test"]:
                if split in self._builders and callable(self._builders[split]):
                    builder = self._builders[split]
                    if hasattr(builder, 'keywords') and builder.keywords is not None:
                        builder.keywords['classes'] = classes
                    elif hasattr(builder, 'func'):
                        # 对于functools.partial对象
                        if hasattr(builder.func, '__defaults__'):
                            builder.keywords = builder.keywords or {}
                            builder.keywords['classes'] = classes
        
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            print(f"Test dataset size: {len(self.test_dataset)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
        )