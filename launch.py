"""
基于 Lightning 框架的模型训练/测试/推理入口。
"""
import logging
import torch
from pytorch_lightning.cli import LightningCLI

from config.logging_config import setup_logging

log = logging.getLogger(__name__)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    setup_logging()
    log.info("启动 LightningCLI")
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    log.info("CLI 执行完成")
