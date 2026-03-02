
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning.core.datamodule import LightningDataModule

log = logging.getLogger(__name__)


class SimpleData(LightningDataModule):
    def __init__(self, batch_size:int=32, num_workers:int=1):
        super().__init__()
        self.save_hyperparameters()
        log.debug("SimpleData 初始化: batch_size=%s", batch_size)

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        log.info("SimpleData setup stage=%s", stage)
        if stage == "fit" or stage is None:
            self.train_data = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
            log.info("train_data 样本数: 1000")
        if stage == "test" or stage is None:
            self.test_data = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))
            log.info("test_data 样本数: 200")
        if stage == "predict" or stage is None:
            self.predict_data = TensorDataset(torch.randn(50, 10))
            log.info("predict_data 样本数: 50")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
