"""
基于lightning框架的模型训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI

class MyModel(LightningModule):
    def __init__(self, lr:float=0.001, hidden_dim=128):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.MSELoss()(pred, y)
        self.log("train loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
class MyData(LightningDataModule):
    def __init__(self, batch_size:int=32):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage:str=None):
        if stage == 'fit' or stage is None:
            self.train_data = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
        if stage == 'test' or stage is None:
            self.test_data = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size)
    
if __name__ == '__main__':
    trainer_config = {
        "max_epochs": 50,
        "accelerator": "cpu",
        "log_every_n_steps": 10,
        "enable_checkpointing": True,
        "default_root_dir": "./logs"
    }
    cli = LightningCLI(MyModel, MyData, trainer_defaults=trainer_config)
    
