"""
RL Planner 模型（Lightning 版本）。
将独立演示版逻辑拆分到仓库标准结构（model/data/config）。
核心功能：基于强化学习的智能体轨迹规划，支持开环/闭环两种模式，
          通过奖励机制筛选最优轨迹，通过平滑损失优化轨迹质量。
"""

# 基础库导入
import copy
import logging
from typing import Any, Dict, List, Optional

# 深度学习库导入
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule  # Lightning模型封装类
from torch import Tensor  # PyTorch张量类型注解

# 初始化日志器，用于输出模型运行信息
log = logging.getLogger(__name__)


class PlannerCore(nn.Module):
    """
    轨迹规划核心模块（简化版）
    核心职责：
    1. 基于智能体当前状态生成多条候选轨迹
    2. 计算每条轨迹的奖励（平滑性/进度/碰撞惩罚）
    输入：智能体当前状态字典
    输出：候选轨迹 + 各维度奖励
    """

    def __init__(self, sample_number: int = 4):
        """
        初始化轨迹生成核心
        :param sample_number: 每次生成的候选轨迹数量，默认4条
        """
        super().__init__()
        # 候选轨迹数量（超参数）
        self.sample_number = sample_number
        # 轨迹编码器（示例层，当前版本未实际使用，预留神经网络扩展）
        self.trajectory_encoder = nn.Linear(32, 128)
        # 轨迹解码器（示例层，当前版本未实际使用，预留神经网络扩展）
        self.trajectory_decoder = nn.Linear(128, 50 * 5)

    def forward_sample_reward(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心前向函数：生成候选轨迹 + 计算轨迹奖励
        :param model_input: 模型输入字典，必须包含"ego_curr_status"（智能体当前状态）
                            ego_curr_status维度：[batch_size, 5]
                            5维含义：x坐标, y坐标, x方向速度, y方向速度, 航向角θ
        :return: 包含候选轨迹和各维度奖励的字典
        """
        # 从输入中获取智能体当前状态，None则返回空奖励
        ego_status = model_input.get("ego_curr_status")
        if ego_status is None:
            # 无输入状态时返回全0奖励（避免程序崩溃）
            reward = torch.zeros(1, self.sample_number, 50)
            return {"batch_reward": {"total_reward_traj": reward}}

        # 提取关键参数
        bsz = ego_status.shape[0]  # batch size（批次大小）
        sample_n = self.sample_number  # 候选轨迹数量
        horizon = 50  # 轨迹预测时长（时间步数量）

        # 存储所有候选轨迹的列表
        trajectories = []
        # 循环生成sample_n条候选轨迹（每条轨迹添加不同扰动，模拟不同行驶策略）
        for m in range(sample_n):
            # 复制当前状态作为轨迹生成基准（避免修改原数据）
            base = ego_status.clone()
            # 存储单条轨迹的所有时间步
            seq = []
            # 生成单条轨迹的50个时间步
            for t in range(horizon):
                dt = 0.2  # 每个时间步的时间间隔（秒）
                # 计算t时刻的x方向位移（基于当前x速度）
                x_offset = base[:, 2] * dt * (t + 1)
                # 计算t时刻的y方向位移（基于当前y速度）
                y_offset = base[:, 3] * dt * (t + 1)
                # 构造t时刻的轨迹点（5维）
                traj_t = torch.stack(
                    [
                        base[:, 0] + x_offset,  # x坐标（基础+位移）
                        base[:, 1] + y_offset,  # y坐标（基础+位移）
                        # x方向速度（基础+扰动，不同轨迹编号扰动不同）
                        base[:, 2] + 0.1 * (m - sample_n // 2),
                        base[:, 3],  # y方向速度（保持基础值）
                        # 航向角θ（基础+随时间/轨迹编号变化的扰动）
                        base[:, 4] + 0.01 * (m - sample_n // 2) * t,
                    ],
                    dim=1,  # 在维度1拼接，保证shape=[bsz,5]
                )
                seq.append(traj_t)  # 将当前时间步轨迹点加入列表
            # 将单条轨迹的所有时间步拼接为tensor：[bsz, horizon, 5]
            trajectories.append(torch.stack(seq, dim=1))

        # 拼接所有候选轨迹：[bsz*sample_n, horizon, 5]
        ego_future_status = torch.cat(trajectories, dim=0)
        # 计算轨迹奖励（三个维度）
        smoothness_reward = self._compute_smoothness_reward(ego_future_status)  # 平滑性奖励
        progress_reward = self._compute_progress_reward(ego_future_status)      # 进度（速度）奖励
        collision_reward = self._compute_collision_reward(ego_future_status)    # 碰撞/越界惩罚

        # 计算总奖励并重塑形状：[bsz, sample_n, horizon]
        total_reward = (smoothness_reward + progress_reward + collision_reward).reshape(
            bsz, sample_n, horizon
        )

        # 返回轨迹和奖励结果
        return {
            "ego_future_status": ego_future_status,  # 所有候选轨迹
            "batch_reward": {
                "total_reward_traj": total_reward,          # 总奖励
                "smoothness_reward": smoothness_reward.reshape(bsz, sample_n, horizon),  # 平滑性奖励
                "progress_reward": progress_reward.reshape(bsz, sample_n, horizon),      # 进度奖励
                "collision_reward": collision_reward.reshape(bsz, sample_n, horizon),    # 碰撞惩罚
            },
        }

    @staticmethod
    def _compute_smoothness_reward(traj: Tensor) -> Tensor:
        """
        计算平滑性奖励（惩罚剧烈加减速）
        核心逻辑：通过二阶差分计算加速度，加速度越大，奖励越低（负向惩罚）
        :param traj: 轨迹张量，shape=[bsz*sample_n, horizon, 5]
        :return: 平滑性奖励，shape=[bsz*sample_n, horizon]
        """
        # 二阶差分计算加速度（离散形式）：a(t) = x(t+1) - 2x(t) + x(t-1)
        # 取速度维度（2:4）计算，结果shape=[bsz*sample_n, horizon-2, 2]
        accel = traj[:, 2:, 2:4] - 2 * traj[:, 1:-1, 2:4] + traj[:, :-2, 2:4]
        # 计算加速度的L2范数（标量），shape=[bsz*sample_n, horizon-2]
        accel_norm = torch.norm(accel, dim=-1)
        # 平滑性奖励：加速度越大，奖励越低（-0.1为惩罚系数）
        reward = -0.1 * accel_norm
        # 补全前后缺失的时间步（保持与原轨迹长度一致）
        reward = torch.cat([reward[:, :1], reward, reward[:, -1:]], dim=1)
        return reward

    @staticmethod
    def _compute_progress_reward(traj: Tensor) -> Tensor:
        """
        计算进度奖励（鼓励合理速度，避免静止/超速）
        :param traj: 轨迹张量，shape=[bsz*sample_n, horizon, 5]
        :return: 进度奖励，shape=[bsz*sample_n, horizon]
        """
        # 计算合速度（x/y速度的L2范数），shape=[bsz*sample_n, horizon]
        velocity = torch.norm(traj[:, :, 2:4], dim=-1)
        # 进度奖励：速度≤5时奖励为正，超过5则截断（避免超速），-0.5为基线偏移
        return 0.5 * torch.clamp(velocity, max=5.0) - 0.5

    @staticmethod
    def _compute_collision_reward(traj: Tensor) -> Tensor:
        """
        计算碰撞/越界惩罚（越界则扣分）
        :param traj: 轨迹张量，shape=[bsz*sample_n, horizon, 5]
        :return: 碰撞惩罚，shape=[bsz*sample_n, horizon]
        """
        # 提取x/y坐标
        x_pos = traj[:, :, 0]
        y_pos = traj[:, :, 1]
        # 判断是否越界（x/y超出±10范围）
        out_of_bounds = (torch.abs(x_pos) > 10.0) | (torch.abs(y_pos) > 10.0)
        # 越界则奖励为-1（惩罚），否则为0
        return torch.where(out_of_bounds, torch.full_like(x_pos, -1.0), torch.zeros_like(x_pos))


class SimpleDataManager:
    """
    数据管理工具类
    核心职责：
    1. 加载/存储模型输入数据（智能体状态、车道线、时间戳等）
    2. 构造模型输入字典
    3. 更新智能体当前状态（闭环模拟核心）
    """

    def __init__(self):
        """初始化数据管理器，所有字段默认None"""
        self.ego_curr_status = None  # 智能体当前状态 [bsz,5]
        self.agent_status = None     # 其他智能体状态（预留）
        self.laneline_pts = None     # 车道线坐标（预留）
        self.timestamp = None        # 时间戳（秒）

    def load_from_model_input(self, model_input: Dict[str, Any]):
        """
        从模型输入字典加载数据
        :param model_input: 模型输入字典
        """
        # 智能体当前状态，无数据则默认[1,5]全0张量
        self.ego_curr_status = model_input.get("ego_curr_status", torch.zeros(1, 5))
        self.agent_status = model_input.get("agent_status", None)  # 其他智能体状态
        self.laneline_pts = model_input.get("laneline_pts", None)  # 车道线坐标
        self.timestamp = model_input.get("timestamp", 0.0)         # 时间戳，默认0.0

    def build_model_input(self) -> Dict[str, Any]:
        """
        构造模型输入字典
        :return: 包含所有状态的模型输入字典
        """
        return {
            "ego_curr_status": self.ego_curr_status,
            "agent_status": self.agent_status,
            "laneline_pts": self.laneline_pts,
            "timestamp": self.timestamp,
        }

    def update_ego_curr_from_future(self, next_state: Tensor):
        """
        更新智能体当前状态（闭环模拟核心）
        :param next_state: 下一个时间步的状态张量 [bsz,5]
        """
        # detach()：分离梯度，避免反向传播影响；clone()：深拷贝，避免修改原数据
        self.ego_curr_status = next_state.detach().clone()


class RLPlannerModel(LightningModule):
    """
    RL Planner 顶层模型（Lightning封装版）
    核心职责：
    1. 整合轨迹生成核心、数据管理、闭环模拟逻辑
    2. 实现训练/测试/预测流程
    3. 配置优化器和学习率调度器
    """

    def __init__(
        self,
        lr: float = 1e-3,               # 学习率
        warmup_steps: int = 100,        # 学习率预热步数
        sample_number: int = 4,         # 候选轨迹数量
        closed_loop: bool = True,       # 是否启用闭环模式
        base_dt: float = 0.2,           # 基础时间步（秒）
        expected_interval_time: float = 0.6,  # 闭环重规划间隔（秒）
        sim_window_seconds: float = 5.0,     # 模拟窗口时长（预留）
        closed_time_horizon: int = 10,       # 闭环总时长（秒）
        debug: bool = False,                 # 调试模式（预留）
        weight_decay: float = 1e-4,          # L2正则系数
    ):
        """初始化顶层模型"""
        super().__init__()
        # 保存所有超参数（Lightning特性，支持后续调参/复现）
        self.save_hyperparameters()
        # 初始化轨迹生成核心
        self.model = PlannerCore(sample_number=sample_number)
        # 训练步数计数器
        self.step_count = 0
        # 输出初始化日志
        log.info("RLPlannerModel 初始化: sample_number=%s, closed_loop=%s", sample_number, closed_loop)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器（Lightning必需方法）
        :return: 优化器+调度器字典
        """
        # 使用AdamW优化器（带权重衰减的Adam）
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        # 学习率预热步数
        warmup_steps = self.hparams.warmup_steps

        # 定义学习率lambda函数：前warmup_steps步线性提升，之后保持1.0
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))  # 线性预热
            return 1.0  # 预热结束后保持学习率

        # 学习率调度器：基于lambda函数调整
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,          # 优化器
            "lr_scheduler": {                # 调度器配置
                "scheduler": scheduler,      # 调度器实例
                "interval": "step",          # 按步数更新学习率
            },
        }

    @staticmethod
    def _detach_copy_model_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        深拷贝模型输入并分离梯度（闭环模拟核心工具函数）
        作用：避免不同步的轨迹生成相互干扰梯度计算
        :param data: 模型输入字典
        :return: 深拷贝+梯度分离后的输入字典
        """
        # 递归处理每个值
        def _detach_val(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                return v.detach().clone()  # 张量：分离梯度+深拷贝
            if isinstance(v, dict):
                return {k: _detach_val(val) for k, val in v.items()}  # 字典：递归处理
            if isinstance(v, (list, tuple)):
                return type(v)(_detach_val(x) for x in v)  # 列表/元组：递归处理
            return copy.deepcopy(v)  # 其他类型：深拷贝

        return {k: _detach_val(v) for k, v in data.items()}

    @staticmethod
    def _concat_model_inputs(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        拼接多个模型输入字典（支持批量处理）
        :param dict_list: 输入字典列表
        :return: 拼接后的字典
        """
        if len(dict_list) == 1:
            return dict_list[0]  # 单字典直接返回
        # 收集所有字典的key
        keys = set()
        for d in dict_list:
            keys.update(d.keys())
        result = {}
        # 按key拼接值
        for key in keys:
            values = [d.get(key) for d in dict_list]
            values_non_none = [v for v in values if v is not None]
            if not values_non_none:
                result[key] = None
                continue
            first_val = values_non_none[0]
            # 张量：在维度0拼接（batch维度）
            if torch.is_tensor(first_val):
                result[key] = torch.cat([v for v in values if torch.is_tensor(v)], dim=0)
            # 列表/元组：合并
            elif isinstance(first_val, (list, tuple)):
                merged = []
                for v in values:
                    if isinstance(v, (list, tuple)):
                        merged.extend(v)
                result[key] = merged
            # 其他类型：取第一个非空值
            else:
                result[key] = first_val
        return result

    def _run_closed_loop_rollout(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行闭环轨迹模拟（核心方法）
        核心逻辑：走一小段→重规划→走一小段→重规划，循环迭代
        :param model_input: 初始模型输入字典
        :return: 闭环轨迹+奖励字典
        """
        # 提取闭环配置参数
        base_dt = self.hparams.base_dt                    # 基础时间步（0.2秒）
        expected_interval_time = self.hparams.expected_interval_time  # 重规划间隔（0.6秒）
        closed_time_horizon = self.hparams.closed_time_horizon        # 闭环总时长（10秒）
        # 每次重规划执行的步数（0.6/0.2=3步）
        interval = max(1, int(round(expected_interval_time / base_dt)))
        # 总重规划次数（10/0.6≈16次）
        total_steps = int(closed_time_horizon / expected_interval_time)
        sample_n = self.hparams.sample_number  # 候选轨迹数量

        # 初始化数据管理器，加载初始状态
        data_manager = SimpleDataManager()
        data_manager.load_from_model_input(model_input)
        data_managers = [data_manager]
        # 存储闭环轨迹（key=0表示主智能体）
        closed_loop_trajs = {0: []}

        # 闭环迭代：每次重规划+执行一小段
        for step_idx in range(total_steps):
            # 1. 构造当前步模型输入（深拷贝+梯度分离）
            model_input_sim_list = [self._detach_copy_model_input(dm.build_model_input()) for dm in data_managers]
            model_input_batch = self._concat_model_inputs(model_input_sim_list)

            # 2. 生成候选轨迹+计算奖励（无梯度计算，节省资源）
            with torch.no_grad():
                model_output = self.model.forward_sample_reward(model_input_batch)

            # 3. 选择最优轨迹（奖励最高）
            ego_future_status = model_output["ego_future_status"]  # 所有候选轨迹 [B*M, T, 5]
            total_reward_traj = model_output["batch_reward"]["total_reward_traj"]  # 总奖励 [B, M, T]
            bsz = model_input["ego_curr_status"].shape[0]  # batch size
            # 重塑轨迹形状：[B, M, T, 5]
            traj = ego_future_status.reshape(bsz, sample_n, ego_future_status.shape[1], 5)
            # 计算每条轨迹的平均奖励（时间步维度求均值）
            score = total_reward_traj.mean(dim=-1)
            # 选择奖励最高的轨迹索引 [B]
            best_idx = score.argmax(dim=1)
            # 提取最优轨迹 [B, T, 5]
            best_traj = traj[torch.arange(bsz, device=traj.device), best_idx]
            # 确定本次执行的步数（不超过轨迹总长度）
            lqr_step_idx = min(interval, best_traj.shape[1])
            # 取执行步数的最后一步作为新的当前状态
            next_state = best_traj[:, lqr_step_idx - 1, :].detach()

            # 4. 记录本次执行的轨迹段
            for k in range(lqr_step_idx):
                closed_loop_trajs[0].append(best_traj[:, k, :].detach().clone())

            # 5. 更新智能体状态（闭环核心：用执行后的状态作为新起点）
            for dm in data_managers:
                dm.update_ego_curr_from_future(next_state)  # 更新当前状态
                dm.timestamp = float(step_idx + 1) * expected_interval_time  # 更新时间戳

        # 拼接所有执行的轨迹段：[总时间步, B, 5]
        trajs = torch.stack(closed_loop_trajs[0])
        return {
            "ego_future_status": trajs,          # 闭环轨迹
            "batch_reward": {"total_reward_traj": trajs[:, :, 0]},  # 奖励（简化版）
            "closed_loop_trajs": closed_loop_trajs,  # 原始轨迹段
        }

    def _inference_step(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理步骤：根据配置选择开环/闭环模式
        :param model_input: 模型输入字典
        :return: 轨迹+奖励字典
        """
        if self.hparams.closed_loop:
            # 闭环模式：运行闭环模拟
            return self._run_closed_loop_rollout(model_input)
        # 开环模式：直接生成轨迹
        output = self.model.forward_sample_reward(model_input)
        output["closed_loop_trajs"] = None  # 开环无轨迹段
        return output

    def _extract_model_input(self, batch: Any) -> Dict[str, Any]:
        """
        从批次数据中提取模型输入字典
        :param batch: 批次数据（支持dict或包含model_input的dict）
        :return: 模型输入字典
        :raise ValueError: 输入格式错误时抛出异常
        """
        if isinstance(batch, dict) and "model_input" in batch:
            return batch["model_input"]
        if isinstance(batch, dict):
            return batch
        raise ValueError("RLPlannerModel 期望 batch 为 dict 或包含 model_input 的 dict")

    def training_step(self, batch, batch_idx):
        """
        训练步骤（Lightning必需方法）
        核心逻辑：生成轨迹→计算平滑损失→反向传播优化
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 损失值
        """
        self.step_count += 1  # 训练步数+1
        # 提取模型输入
        model_input = self._extract_model_input(batch)
        # 无智能体状态时返回0损失（避免崩溃）
        if "ego_curr_status" not in model_input:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_loss", loss, prog_bar=True)  # 记录训练损失
            return loss

        # 生成轨迹（开环/闭环）
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]  # 轨迹张量
        closed_loop_trajs = output.get("closed_loop_trajs")  # 闭环轨迹段

        # 计算平滑性损失（核心优化目标）
        smoothness_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if closed_loop_trajs is not None and trajs.shape[0] > 2:
            # 二阶差分计算加速度（平滑性损失核心）
            accel = trajs[2:, :, 2:4] - 2 * trajs[1:-1, :, 2:4] + trajs[:-2, :, 2:4]
            smoothness_loss = torch.norm(accel, dim=-1).mean()  # 加速度均值作为损失

        # 计算L2正则（防止过拟合）
        l2_reg = sum(self.hparams.weight_decay * torch.sum(p**2) for p in self.parameters())
        # 总损失 = 平滑损失 + L2正则
        loss = smoothness_loss + l2_reg

        # 记录损失日志（prog_bar=True表示在进度条显示）
        self.log("train_loss", loss, prog_bar=True)
        self.log("smoothness_loss", smoothness_loss, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        """
        测试步骤（Lightning可选方法）
        核心逻辑：生成轨迹→计算平滑损失→记录测试指标
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 测试损失字典
        """
        # 提取模型输入
        model_input = self._extract_model_input(batch)
        # 生成轨迹
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]

        # 计算平滑性损失
        if trajs.dim() >= 3 and trajs.shape[0] > 2:
            accel = trajs[2:, :, 2:4] - 2 * trajs[1:-1, :, 2:4] + trajs[:-2, :, 2:4]
            loss = torch.norm(accel, dim=-1).mean()
        else:
            loss = torch.tensor(0.0, device=self.device)

        # 记录测试损失（sync_dist=True支持多卡同步）
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        """
        预测步骤（Lightning可选方法）
        核心逻辑：生成轨迹→返回结果（无损失计算）
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 轨迹+奖励字典
        """
        # 提取模型输入
        model_input = self._extract_model_input(batch)
        # 生成轨迹
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]
        # 输出预测日志
        log.info("RLPlanner predict batch_idx=%d traj_shape=%s", batch_idx, tuple(trajs.shape))
        return output

    def forward(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        模型前向传播（Lightning必需方法）
        直接调用推理步骤，统一接口
        :param model_input: 模型输入字典
        :return: 轨迹+奖励字典
        """
        return self._inference_step(model_input)