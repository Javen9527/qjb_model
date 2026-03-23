"""
RL Planner 模型（Lightning 版本）。
将独立演示版逻辑拆分到仓库标准结构（model/data/config）。
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

log = logging.getLogger(__name__)


class PlannerCore(nn.Module):
    """简化的轨迹采样 + 奖励计算核心。"""

    def __init__(self, sample_number: int = 4):
        super().__init__()
        self.sample_number = sample_number
        self.trajectory_encoder = nn.Linear(32, 128)
        self.trajectory_decoder = nn.Linear(128, 50 * 5)

    def forward_sample_reward(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        ego_status = model_input.get("ego_curr_status")
        if ego_status is None:
            reward = torch.zeros(1, self.sample_number, 50)
            return {"batch_reward": {"total_reward_traj": reward}}

        bsz = ego_status.shape[0]
        sample_n = self.sample_number
        horizon = 50
        trajectories = []
        for m in range(sample_n):
            base = ego_status.clone()
            seq = []
            for t in range(horizon):
                dt = 0.2
                x_offset = base[:, 2] * dt * (t + 1)
                y_offset = base[:, 3] * dt * (t + 1)
                traj_t = torch.stack(
                    [
                        base[:, 0] + x_offset,
                        base[:, 1] + y_offset,
                        base[:, 2] + 0.1 * (m - sample_n // 2),
                        base[:, 3],
                        base[:, 4] + 0.01 * (m - sample_n // 2) * t,
                    ],
                    dim=1,
                )
                seq.append(traj_t)
            trajectories.append(torch.stack(seq, dim=1))

        ego_future_status = torch.cat(trajectories, dim=0)  # [B*M, T, 5]
        smoothness_reward = self._compute_smoothness_reward(ego_future_status)
        progress_reward = self._compute_progress_reward(ego_future_status)
        collision_reward = self._compute_collision_reward(ego_future_status)
        total_reward = (smoothness_reward + progress_reward + collision_reward).reshape(
            bsz, sample_n, horizon
        )
        return {
            "ego_future_status": ego_future_status,
            "batch_reward": {
                "total_reward_traj": total_reward,
                "smoothness_reward": smoothness_reward.reshape(bsz, sample_n, horizon),
                "progress_reward": progress_reward.reshape(bsz, sample_n, horizon),
                "collision_reward": collision_reward.reshape(bsz, sample_n, horizon),
            },
        }

    @staticmethod
    def _compute_smoothness_reward(traj: Tensor) -> Tensor:
        accel = traj[:, 2:, 2:4] - 2 * traj[:, 1:-1, 2:4] + traj[:, :-2, 2:4]
        accel_norm = torch.norm(accel, dim=-1)
        reward = -0.1 * accel_norm
        reward = torch.cat([reward[:, :1], reward, reward[:, -1:]], dim=1)
        return reward

    @staticmethod
    def _compute_progress_reward(traj: Tensor) -> Tensor:
        velocity = torch.norm(traj[:, :, 2:4], dim=-1)
        return 0.5 * torch.clamp(velocity, max=5.0) - 0.5

    @staticmethod
    def _compute_collision_reward(traj: Tensor) -> Tensor:
        x_pos = traj[:, :, 0]
        y_pos = traj[:, :, 1]
        out_of_bounds = (torch.abs(x_pos) > 10.0) | (torch.abs(y_pos) > 10.0)
        return torch.where(out_of_bounds, torch.full_like(x_pos, -1.0), torch.zeros_like(x_pos))


class SimpleDataManager:
    def __init__(self):
        self.ego_curr_status = None
        self.agent_status = None
        self.laneline_pts = None
        self.timestamp = None

    def load_from_model_input(self, model_input: Dict[str, Any]):
        self.ego_curr_status = model_input.get("ego_curr_status", torch.zeros(1, 5))
        self.agent_status = model_input.get("agent_status", None)
        self.laneline_pts = model_input.get("laneline_pts", None)
        self.timestamp = model_input.get("timestamp", 0.0)

    def build_model_input(self) -> Dict[str, Any]:
        return {
            "ego_curr_status": self.ego_curr_status,
            "agent_status": self.agent_status,
            "laneline_pts": self.laneline_pts,
            "timestamp": self.timestamp,
        }

    def update_ego_curr_from_future(self, next_state: Tensor):
        self.ego_curr_status = next_state.detach().clone()


class RLPlannerModel(LightningModule):
    """按仓库风格实现的 RL Planner 模型。"""

    def __init__(
        self,
        lr: float = 1e-3,
        warmup_steps: int = 100,
        sample_number: int = 4,
        closed_loop: bool = True,
        base_dt: float = 0.2,
        expected_interval_time: float = 0.6,
        sim_window_seconds: float = 5.0,
        closed_time_horizon: int = 10,
        debug: bool = False,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PlannerCore(sample_number=sample_number)
        self.step_count = 0
        log.info("RLPlannerModel 初始化: sample_number=%s, closed_loop=%s", sample_number, closed_loop)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def _detach_copy_model_input(data: Dict[str, Any]) -> Dict[str, Any]:
        def _detach_val(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                return v.detach().clone()
            if isinstance(v, dict):
                return {k: _detach_val(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_detach_val(x) for x in v)
            return copy.deepcopy(v)

        return {k: _detach_val(v) for k, v in data.items()}

    @staticmethod
    def _concat_model_inputs(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(dict_list) == 1:
            return dict_list[0]
        keys = set()
        for d in dict_list:
            keys.update(d.keys())
        result = {}
        for key in keys:
            values = [d.get(key) for d in dict_list]
            values_non_none = [v for v in values if v is not None]
            if not values_non_none:
                result[key] = None
                continue
            first_val = values_non_none[0]
            if torch.is_tensor(first_val):
                result[key] = torch.cat([v for v in values if torch.is_tensor(v)], dim=0)
            elif isinstance(first_val, (list, tuple)):
                merged = []
                for v in values:
                    if isinstance(v, (list, tuple)):
                        merged.extend(v)
                result[key] = merged
            else:
                result[key] = first_val
        return result

    def _run_closed_loop_rollout(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        base_dt = self.hparams.base_dt
        expected_interval_time = self.hparams.expected_interval_time
        closed_time_horizon = self.hparams.closed_time_horizon
        interval = max(1, int(round(expected_interval_time / base_dt)))
        total_steps = int(closed_time_horizon / expected_interval_time)
        sample_n = self.hparams.sample_number

        data_manager = SimpleDataManager()
        data_manager.load_from_model_input(model_input)
        data_managers = [data_manager]
        closed_loop_trajs = {0: []}

        for step_idx in range(total_steps):
            model_input_sim_list = [self._detach_copy_model_input(dm.build_model_input()) for dm in data_managers]
            model_input_batch = self._concat_model_inputs(model_input_sim_list)

            with torch.no_grad():
                model_output = self.model.forward_sample_reward(model_input_batch)

            ego_future_status = model_output["ego_future_status"]  # [B*M, T, 5]
            total_reward_traj = model_output["batch_reward"]["total_reward_traj"]  # [B, M, T]
            bsz = model_input["ego_curr_status"].shape[0]
            traj = ego_future_status.reshape(bsz, sample_n, ego_future_status.shape[1], 5)
            score = total_reward_traj.mean(dim=-1)
            best_idx = score.argmax(dim=1)
            best_traj = traj[torch.arange(bsz, device=traj.device), best_idx]
            lqr_step_idx = min(interval, best_traj.shape[1])
            next_state = best_traj[:, lqr_step_idx - 1, :].detach()
            for k in range(lqr_step_idx):
                closed_loop_trajs[0].append(best_traj[:, k, :].detach().clone())
            for dm in data_managers:
                dm.update_ego_curr_from_future(next_state)
                dm.timestamp = float(step_idx + 1) * expected_interval_time

        trajs = torch.stack(closed_loop_trajs[0])  # [T, B, 5]
        return {
            "ego_future_status": trajs,
            "batch_reward": {"total_reward_traj": trajs[:, :, 0]},
            "closed_loop_trajs": closed_loop_trajs,
        }

    def _inference_step(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        if self.hparams.closed_loop:
            return self._run_closed_loop_rollout(model_input)
        output = self.model.forward_sample_reward(model_input)
        output["closed_loop_trajs"] = None
        return output

    def _extract_model_input(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict) and "model_input" in batch:
            return batch["model_input"]
        if isinstance(batch, dict):
            return batch
        raise ValueError("RLPlannerModel 期望 batch 为 dict 或包含 model_input 的 dict")

    def training_step(self, batch, batch_idx):
        self.step_count += 1
        model_input = self._extract_model_input(batch)
        if "ego_curr_status" not in model_input:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]
        closed_loop_trajs = output.get("closed_loop_trajs")
        smoothness_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if closed_loop_trajs is not None and trajs.shape[0] > 2:
            accel = trajs[2:, :, 2:4] - 2 * trajs[1:-1, :, 2:4] + trajs[:-2, :, 2:4]
            smoothness_loss = torch.norm(accel, dim=-1).mean()
        l2_reg = sum(self.hparams.weight_decay * torch.sum(p**2) for p in self.parameters())
        loss = smoothness_loss + l2_reg
        self.log("train_loss", loss, prog_bar=True)
        self.log("smoothness_loss", smoothness_loss, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        model_input = self._extract_model_input(batch)
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]
        if trajs.dim() >= 3 and trajs.shape[0] > 2:
            accel = trajs[2:, :, 2:4] - 2 * trajs[1:-1, :, 2:4] + trajs[:-2, :, 2:4]
            loss = torch.norm(accel, dim=-1).mean()
        else:
            loss = torch.tensor(0.0, device=self.device)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        model_input = self._extract_model_input(batch)
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]
        log.info("RLPlanner predict batch_idx=%d traj_shape=%s", batch_idx, tuple(trajs.shape))
        return output

    def forward(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        return self._inference_step(model_input)
