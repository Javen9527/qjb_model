"""
RL Planner 数据模块。
支持合成数据和 .pt 文件加载，输出格式与 RLPlannerModel 对齐。
"""

import logging
import pathlib
from typing import Any, Dict

import torch
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class SyntheticRLPlannerDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1024,
        num_agents: int = 10,
        num_laneline_pts: int = 50,
        ego_dim: int = 5,
        agent_dim: int = 6,
        seed: int | None = 42,
    ):
        self.num_samples = num_samples
        self.num_agents = num_agents
        self.num_laneline_pts = num_laneline_pts
        self.ego_dim = ego_dim
        self.agent_dim = agent_dim
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.seed is not None:
            torch.manual_seed(self.seed + idx)
        ego = torch.randn(self.ego_dim)
        agent = torch.randn(self.num_agents, self.agent_dim)
        lane = torch.randn(self.num_laneline_pts, 2)
        ts = torch.tensor(0.0, dtype=torch.float32)
        return {
            "model_input": {
                "ego_curr_status": ego,
                "agent_status": agent,
                "laneline_pts": lane,
                "timestamp": ts,
            }
        }


class PTDictDataset(Dataset):
    """
    从 .pt 文件加载 list[dict] 或 dict[str, tensor]。
    返回统一格式: {"model_input": {...}}
    """

    def __init__(self, data: Any):
        self.data = data
        if isinstance(data, list):
            self.length = len(data)
        elif isinstance(data, dict):
            self.length = data["ego_curr_status"].shape[0]
        else:
            raise ValueError("仅支持 list[dict] 或 dict[str, tensor] 格式")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(self.data, list):
            item = self.data[idx]
            if "model_input" in item:
                return item
            return {"model_input": item}
        return {
            "model_input": {
                "ego_curr_status": self.data["ego_curr_status"][idx],
                "agent_status": self.data.get("agent_status", None)[idx]
                if self.data.get("agent_status", None) is not None
                else None,
                "laneline_pts": self.data.get("laneline_pts", None)[idx]
                if self.data.get("laneline_pts", None) is not None
                else None,
                "timestamp": self.data.get("timestamp", torch.zeros(self.length))[idx],
            }
        }


class RLPlannerData(LightningDataModule):
    _STAGE_TO_PATH = {"fit": "train_data_path", "test": "test_data_path", "predict": "predict_data_path"}

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        train_samples: int = 1024,
        test_samples: int = 128,
        predict_samples: int = 64,
        num_agents: int = 10,
        num_laneline_pts: int = 50,
        ego_dim: int = 5,
        agent_dim: int = 6,
        seed: int | None = 42,
        train_data_path: str | None = None,
        test_data_path: str | None = None,
        predict_data_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        log.debug("RLPlannerData 初始化: batch_size=%s", batch_size)

    def prepare_data(self):
        pass

    def _load_from_pt(self, path: str) -> Dataset:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"数据路径不存在: {path}")
        if p.suffix.lower() not in (".pt", ".pth"):
            raise ValueError("RLPlannerData 目前仅支持 .pt/.pth 文件输入")
        data = torch.load(p, map_location="cpu", weights_only=False)
        return PTDictDataset(data)

    def _make_dataset(self, stage: str) -> Dataset:
        hp = self.hparams
        path_attr = self._STAGE_TO_PATH.get(stage, f"{stage}_data_path")
        path = getattr(hp, path_attr, None)
        fallback = hp.train_samples if stage == "fit" else (hp.test_samples if stage == "test" else hp.predict_samples)
        if path:
            log.info("stage=%s 从文件加载 RL 数据: %s", stage, path)
            return self._load_from_pt(path)
        log.info("stage=%s 使用合成 RL 数据: samples=%s", stage, fallback)
        return SyntheticRLPlannerDataset(
            num_samples=fallback,
            num_agents=hp.num_agents,
            num_laneline_pts=hp.num_laneline_pts,
            ego_dim=hp.ego_dim,
            agent_dim=hp.agent_dim,
            seed=(hp.seed + {"fit": 0, "test": 1000, "predict": 2000}[stage]) if hp.seed is not None else None,
        )

    @staticmethod
    def _collate_fn(batch: list[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        model_inputs = [b["model_input"] for b in batch]
        keys = set().union(*[x.keys() for x in model_inputs])
        out_model_input = {}
        for k in keys:
            vals = [x.get(k) for x in model_inputs]
            valid = [v for v in vals if torch.is_tensor(v)]
            if valid:
                out_model_input[k] = torch.stack(valid, dim=0)
            else:
                out_model_input[k] = vals[0] if vals else None
        return {"model_input": out_model_input}

    def setup(self, stage: str | None = None):
        log.info("RLPlannerData setup stage=%s", stage)
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
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
        )
