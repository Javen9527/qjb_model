"""
纯手写模型训练流程, 仅用作对比学习本代码仓中的LightningCLI框架
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

class MyModel(nn.Module):
    def __init__(self, lr: float = 0.001, hidden_dim: int = 128):
        super().__init__()
        self.lr = lr
        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class MyData:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_data = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
        if stage == 'test' or stage is None:
            self.test_data = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

def train():
    lr = 0.001
    hidden_dim = 128
    batch_size = 32
    max_epochs = 50
    log_every_n_steps = 10  # 每10步打印一次日志（对应 Lightning 的 log_every_n_steps）
    device = torch.device("cpu")  # 对应 Lightning 的 accelerator='cpu'

    # 数据
    data_module = MyData(batch_size=batch_size)
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    # 模型
    model = MyModel(lr=lr, hidden_dim=hidden_dim).to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 损失函数
    criterion = nn.MSELoss()

    model.train()  # 切换训练模式
    start_time = time.time()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            pred = model(x)
            loss = criterion(pred, y)
            # 反向传播
            loss.backward()
            # 优化器更新
            optimizer.step()
            # 6. 累计损失
            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % log_every_n_steps == 0:
                print(f"Epoch [{epoch+1}/{max_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch+1}/{max_epochs}] Finished, Average Train Loss: {avg_epoch_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining finished! Total time: {total_time:.2f}s")
    print(f"Final model hyperparameters: lr={lr}, hidden_dim={hidden_dim}, batch_size={batch_size}")

if __name__ == '__main__':
    train()