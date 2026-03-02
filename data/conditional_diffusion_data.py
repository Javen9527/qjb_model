"""
条件扩散模型数据模块。
支持指定数据路径：文件夹（子目录名=类别，内含 PNG/JPG 等）或 .pt/.npy 文件，或使用合成数据。
"""

import pathlib
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning.core.datamodule import LightningDataModule

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def _load_image(path: pathlib.Path, img_size: int | None, to_grayscale: bool) -> torch.Tensor:
    img = Image.open(path).convert("L" if to_grayscale else "RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)
    if img_size and (x.shape[1] != img_size or x.shape[2] != img_size):
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    x = 2 * x - 1
    return x


class ImageFolderConditionalDataset(Dataset):
    """从文件夹加载图片，子目录名作为类别标签。
    结构示例: root/class_0/img1.png, root/class_1/img2.jpg
    """

    def __init__(
        self,
        root: str,
        img_size: int | None = 32,
        grayscale: bool = True,
        class_to_idx: dict | None = None,
    ):
        self.root = pathlib.Path(root)
        self.img_size = img_size
        self.grayscale = grayscale
        self.samples: list[tuple[pathlib.Path, int]] = []
        self.classes: list[str] = []

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys())
        else:
            subdirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
            self.classes = [d.name for d in subdirs]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name, idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, idx))

        if not self.samples:
            raise FileNotFoundError(f"在 {root} 下未找到图片（需子目录结构，支持: {IMAGE_EXTENSIONS}）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = _load_image(path, self.img_size, self.grayscale)
        return x, torch.tensor(label, dtype=torch.long)


def _load_conditional_from_path(
    path: str,
    img_size: int | None = None,
    grayscale: bool = True,
):
    """从路径加载条件数据。支持文件夹（子目录=类别）或 .pt/.npy 文件。"""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据路径不存在: {path}")

    if path.is_dir():
        return ImageFolderConditionalDataset(str(path), img_size=img_size, grayscale=grayscale)

    suffix = path.suffix.lower()
    if suffix not in (".pt", ".pth", ".npy"):
        raise ValueError(f"不支持的文件格式: {suffix}，请使用文件夹或 .pt / .pth / .npy")

    if suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            x = data.get("x", data.get("images"))
            y = data.get("y", data.get("labels"))
            if x is None or y is None:
                raise ValueError("条件数据需包含 'x' 与 'y' (或 'images' 与 'labels')")
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            x, y = data[0], data[1]
        else:
            raise ValueError("条件数据应为 dict(x=..., y=...) 或 tuple(x, y)")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        return x, y

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 4:
            x = torch.from_numpy(arr).float()
            y = torch.zeros(x.size(0), dtype=torch.long)
        else:
            raise ValueError(".npy 条件数据需为 4D 数组 (N,C,H,W)")
        return x, y

    raise ValueError(f"不支持的文件格式: {suffix}")


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


class SyntheticConditionalDataset(Dataset):
    """合成带标签的图像。"""

    def __init__(
        self,
        num_samples: int,
        num_classes: int = 10,
        img_size: int = 32,
        channels: int = 1,
        seed: int | None = None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
        label = torch.randint(0, self.num_classes, (1,)).item()
        x = torch.randn(self.channels, self.img_size, self.img_size) * 0.3
        grid = torch.linspace(-1, 1, self.img_size)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        offset = (label / self.num_classes - 0.5) * 2
        pattern = torch.exp(-((gx - offset) ** 2 + gy**2) / 2)
        x = x + pattern.unsqueeze(0).expand(self.channels, -1, -1)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = 2 * x - 1
        return x, torch.tensor(label, dtype=torch.long)


class ConditionalDiffusionData(LightningDataModule):
    """条件扩散模型数据。路径可为文件夹（子目录=类别，内含 PNG/JPG 等）或 .pt/.npy 文件。"""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        img_size: int = 32,
        channels: int = 1,
        num_classes: int = 10,
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

    _STAGE_TO_PATH = {"fit": "train_data_path", "test": "test_data_path", "predict": "predict_data_path"}

    def _make_dataset(self, stage: str) -> Dataset:
        hp = self.hparams
        path_attr = self._STAGE_TO_PATH.get(stage, f"{stage}_data_path")
        path = getattr(hp, path_attr, None)
        fallback = hp.train_samples if stage == "fit" else (hp.test_samples if stage == "test" else hp.predict_samples)

        if path:
            result = _load_conditional_from_path(
                path,
                img_size=hp.img_size,
                grayscale=hp.grayscale,
            )
            if isinstance(result, Dataset):
                return result
            x, y = result
            x = _to_nchw(x)
            if y.dim() == 0:
                y = y.unsqueeze(0).expand(x.size(0))
            elif y.size(0) != x.size(0):
                raise ValueError(f"x 与 y 样本数不一致: {x.size(0)} vs {y.size(0)}")
            return TensorDataset(x, y)

        return SyntheticConditionalDataset(
            num_samples=fallback,
            num_classes=hp.num_classes,
            img_size=hp.img_size,
            channels=hp.channels,
            seed=(hp.seed + {"fit": 0, "test": 1000, "predict": 2000}[stage]) if hp.seed is not None else None,
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_data = self._make_dataset("fit")
        if stage == "test" or stage is None:
            self.test_data = self._make_dataset("test")
        if stage == "predict" or stage is None:
            self.predict_data = self._make_dataset("predict")

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
