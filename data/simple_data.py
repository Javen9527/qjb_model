

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning.core.datamodule import LightningDataModule

class SimpleData(LightningDataModule):
    def __init__(self, batch_size:int=32, num_workers:int=1):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage:str=None):
        if stage == 'fit' or stage is None:
            self.train_data = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
        if stage == 'test' or stage is None:
            self.test_data = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))
        if stage == 'predict' or stage is None:
            self.predict_data = TensorDataset(torch.randn(50, 10))  # 50条无标签数据

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
