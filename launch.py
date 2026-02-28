"""
基于lightning框架的模型训练
"""
import torch
from pytorch_lightning.cli import LightningCLI

torch.set_num_threads(1)  # 模型计算的CPU线程数
torch.set_num_interop_threads(1)  # 跨操作线程数

## NOTE trainer是CLI框架定义好的类，用户无需定义，直接到yaml中进行配置

if __name__ == '__main__':
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
