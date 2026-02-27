"""
基于lightning框架的模型训练
"""
import os
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
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        pred = self(x)

        print(f"\nPredict batch {batch_idx+1}")
        print(f"batch shape: {pred.shape}")
        print(f"batch predict[:5]: {pred[:5].numpy().flatten()}")
        return pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    # 可选：训练结束后手动保存最优模型权重
    def on_train_end(self):
        # 从 Trainer 中获取自动保存的最优 checkpoint 路径
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            save_dir = os.path.join(self.trainer.default_root_dir, 'final_model')
            os.makedirs(save_dir, exist_ok=True)
            # 加载最优模型并保存纯权重
            best_model = MyModel.load_from_checkpoint(best_ckpt_path)
            torch.save(best_model.state_dict(), f"{save_dir}/best_model_weights.pth")
            print(f"\nsave best model weights: {save_dir}/best_model_weights.pth")

    def on_predict_end(self):
        pass

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
        if stage == 'predict' or stage is None:
            self.predict_data = TensorDataset(torch.randn(50, 10))  # 50条无标签数据

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.hparams.batch_size)

## NOTE trainer块直接挪到yaml中进行配置

if __name__ == '__main__':
    cli = LightningCLI(MyModel, MyData, save_config_kwargs={"overwrite": True})