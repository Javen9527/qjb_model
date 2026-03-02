"""
Transformer 模型数据模块。
提供序列对 (src, tgt) 用于 seq2seq 训练。
"""

import logging
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning.core.datamodule import LightningDataModule

log = logging.getLogger(__name__)


class SyntheticSeq2SeqDataset(Dataset):
    """合成序列对数据集。src 与 tgt 为随机 token 序列，用于演示。"""

    def __init__(
        self,
        num_samples: int,
        src_len: int = 20,
        tgt_len: int = 20,
        vocab_size: int = 1000,
        padding_idx: int = 0,
        seed: int | None = None,
    ):
        self.num_samples = num_samples
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
        src = torch.randint(1, self.vocab_size, (self.src_len,))
        tgt = torch.randint(1, self.vocab_size, (self.tgt_len,))
        return src, tgt


def _load_seq2seq_from_path(path: str):
    """从 .pt 加载序列对。期望 dict(src=..., tgt=...) 或 list of (src, tgt)。"""
    path = pathlib.Path(path)
    if not path.exists():
        log.error("数据路径不存在: %s", path)
        raise FileNotFoundError(f"数据路径不存在: {path}")
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError("仅支持 .pt / .pth 格式")

    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        src = data.get("src", data.get("source"))
        tgt = data.get("tgt", data.get("target"))
        if src is None or tgt is None:
            raise ValueError("需包含 'src'/'source' 与 'tgt'/'target'")
    elif isinstance(data, (list, tuple)) and len(data) >= 2:
        src, tgt = data[0], data[1]
    else:
        raise ValueError("数据格式需为 dict(src=..., tgt=...) 或 [src, tgt]")

    if isinstance(src, np.ndarray):
        src = torch.from_numpy(src).long()
    if isinstance(tgt, np.ndarray):
        tgt = torch.from_numpy(tgt).long()
    return src, tgt


class TransformerData(LightningDataModule):
    """Transformer 数据模块。支持合成数据或从 .pt 文件加载。"""

    _STAGE_TO_PATH = {"fit": "train_data_path", "test": "test_data_path", "predict": "predict_data_path"}

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        src_len: int = 20,
        tgt_len: int = 20,
        vocab_size: int = 1000,
        padding_idx: int = 0,
        train_samples: int = 10000,
        test_samples: int = 1000,
        predict_samples: int = 100,
        seed: int | None = 42,
        train_data_path: str | None = None,
        test_data_path: str | None = None,
        predict_data_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        log.debug("TransformerData 初始化: batch_size=%s, src_len=%s, tgt_len=%s", batch_size, src_len, tgt_len)

    def prepare_data(self):
        pass

    _STAGE_TO_PATH = {"fit": "train_data_path", "test": "test_data_path", "predict": "predict_data_path"}

    def _make_dataset(self, stage: str) -> Dataset:
        hp = self.hparams
        path_attr = self._STAGE_TO_PATH.get(stage, f"{stage}_data_path")
        path = getattr(hp, path_attr, None)
        fallback = hp.train_samples if stage == "fit" else (hp.test_samples if stage == "test" else hp.predict_samples)

        if path:
            log.info("stage=%s 从路径加载: %s", stage, path)
            src, tgt = _load_seq2seq_from_path(path)
            return TensorDataset(src, tgt)

        log.debug("stage=%s 使用合成数据, samples=%s", stage, fallback)
        return SyntheticSeq2SeqDataset(
            num_samples=fallback,
            src_len=hp.src_len,
            tgt_len=hp.tgt_len,
            vocab_size=hp.vocab_size,
            padding_idx=hp.padding_idx,
            seed=(hp.seed + {"fit": 0, "test": 1000, "predict": 2000}[stage]) if hp.seed is not None else None,
        )

    def setup(self, stage: str | None = None):
        log.info("TransformerData setup stage=%s", stage)
        if stage == "fit" or stage is None:
            self.train_data = self._make_dataset("fit")
            log.info("train_data 样本数: %d", len(self.train_data))
        if stage == "test" or stage is None:
            self.test_data = self._make_dataset("test")
            log.info("test_data 样本数: %d", len(self.test_data))
        if stage == "predict" or stage is None:
            self.predict_data = self._make_dataset("predict")
            log.info("predict_data 样本数: %d", len(self.predict_data))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
