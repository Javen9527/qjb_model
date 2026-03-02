import logging
import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer

log = logging.getLogger(__name__)


class SimpleModel(LightningModule):
    def __init__(self, lr: float = 0.001, hidden_dim=128):
        super().__init__()
        self.save_hyperparameters()
        log.info("SimpleModel 初始化: lr=%s, hidden_dim=%s", lr, hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.MSELoss()(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        test_loss = nn.MSELoss()(pred, y)

        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        return {"test_loss": test_loss, "preds": pred, "labels": y}
    
    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        pred = self(x)

        log.info("predict batch_idx=%d shape=%s pred[:5]=%s", batch_idx + 1, pred.shape, pred[:5].detach().cpu().numpy().flatten())
        return pred
    
    def on_train_end(self):
        # 从 Trainer 中获取自动保存的最优 checkpoint 路径
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            save_dir = os.path.join(self.trainer.default_root_dir, 'final_model')
            os.makedirs(save_dir, exist_ok=True)
            # 加载最优模型并保存纯权重
            best_model = SimpleModel.load_from_checkpoint(best_ckpt_path)
            torch.save(best_model.state_dict(), f"{save_dir}/best_model_weights.pth")
            log.info("已保存最优模型权重: %s/best_model_weights.pth", save_dir)

    def on_test_end(self):
        pass

    def on_predict_end(self):
        pass
