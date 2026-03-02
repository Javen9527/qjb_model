"""
无约束扩散模型数据模块。
支持指定数据路径：文件夹（PNG/JPG 等图片）或 .pt / .npy 文件，或使用合成数据。
"""

import logging
import pathlib
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning.core.datamodule import LightningDataModule

log = logging.getLogger(__name__)

# 支持的图片后缀
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def _is_image_path(p: pathlib.Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def _load_image(path: pathlib.Path, img_size: int | None, to_grayscale: bool) -> torch.Tensor:
    """加载单张图片，转为 (C, H, W)，数值 [-1, 1]。"""
    img = Image.open(path).convert("L" if to_grayscale else "RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    x = torch.from_numpy(arr)
    if img_size and (x.shape[1] != img_size or x.shape[2] != img_size):
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    x = 2 * x - 1  # [0,1] -> [-1,1]
    return x


class ImageFolderDataset(Dataset):
    """从文件夹加载图片。支持 PNG、JPG 等格式。"""

    def __init__(
        self,
        root: str,
        img_size: int | None = 32,
        grayscale: bool = True,
        extensions: set | None = None,
    ):
        self.root = pathlib.Path(root)
        self.img_size = img_size
        self.grayscale = grayscale
        self.extensions = extensions or IMAGE_EXTENSIONS
        self.files = sorted([
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.extensions
        ])
        if not self.files:
            log.error("在 %s 下未找到图片（支持: %s）", root, self.extensions)
            raise FileNotFoundError(f"在 {root} 下未找到图片文件（支持: {self.extensions}）")
        log.debug("ImageFolderDataset: root=%s, 共 %d 张图片", root, len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return _load_image(self.files[idx], self.img_size, self.grayscale)


def _load_from_path(path: str, img_size: int | None = None, grayscale: bool = True):
    """从路径加载数据。支持文件夹（图片）或 .pt / .pth / .npy 文件。"""
    path = pathlib.Path(path)
    if not path.exists():
        log.error("数据路径不存在: %s", path)
        raise FileNotFoundError(f"数据路径不存在: {path}")

    if path.is_dir():
        log.info("从文件夹加载数据: %s", path)
        return ImageFolderDataset(str(path), img_size=img_size, grayscale=grayscale)

    log.info("从文件加载数据: %s", path)
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "x" in data:
            data = data["x"]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return TensorDataset(data)
    if suffix == ".npy":
        arr = np.load(path)
        data = torch.from_numpy(arr).float()
        return TensorDataset(data)
    raise ValueError(f"不支持的文件格式: {suffix}，请使用文件夹或 .pt / .pth / .npy")


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


class SyntheticImageDataset(Dataset):
    """合成 2D 图像。"""

    def __init__(self, num_samples: int, img_size: int = 32, channels: int = 1, seed: int | None = None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.channels = channels
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
        x = torch.randn(self.channels, self.img_size, self.img_size) * 0.3
        grid = torch.linspace(-1, 1, self.img_size)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        pattern = torch.exp(-(gx**2 + gy**2) / 2)
        x = x + pattern.unsqueeze(0).expand(self.channels, -1, -1)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = 2 * x - 1
        return x


class UnconstrainedDiffusionData(LightningDataModule):
    """无约束扩散模型数据。train_data_path 等可为文件夹（含 PNG/JPG 等）或 .pt/.npy 文件。"""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        img_size: int = 32,
        channels: int = 1,
        train_samples: int = 5000,
        test_samples: int = 500,
        predict_samples: int = 16,
        seed: int | None = 42,
        train_data_path: str | None = None,
        test_data_path: str | None = None,
        predict_data_path: str | None = None,
        grayscale: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        log.debug("UnconstrainedDiffusionData 初始化: batch_size=%s, img_size=%s", batch_size, img_size)

    _STAGE_TO_PATH = {"fit": "train_data_path", "test": "test_data_path", "predict": "predict_data_path"}

    def _make_dataset(self, stage: str) -> Dataset:
        hp = self.hparams
        path_attr = self._STAGE_TO_PATH.get(stage, f"{stage}_data_path")
        path = getattr(hp, path_attr, None)
        fallback = hp.train_samples if stage == "fit" else (hp.test_samples if stage == "test" else hp.predict_samples)

        if path:
            log.info("stage=%s 从路径加载: %s", stage, path)
            ds = _load_from_path(path, img_size=hp.img_size, grayscale=hp.grayscale)
            if isinstance(ds, TensorDataset):
                data = ds.tensors[0]
                data = _to_nchw(data)
                return TensorDataset(data)
            return ds
        log.debug("stage=%s 使用合成数据, samples=%s", stage, fallback)
        return SyntheticImageDataset(
            num_samples=fallback,
            img_size=hp.img_size,
            channels=hp.channels,
            seed=(hp.seed + {"fit": 0, "test": 1000, "predict": 2000}[stage]) if hp.seed is not None else None,
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        log.info("UnconstrainedDiffusionData setup stage=%s", stage)
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
