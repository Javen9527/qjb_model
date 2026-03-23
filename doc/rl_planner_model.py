"""
完全独立可运行的 RL 规划模型演示
无需外部工程依赖，包含所有实现细节

使用方法：
    python rl_planner_model_runnable.py
"""

import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from torch import Tensor
import math


# =============================================
# 第一部分：模拟模型 (替代 RLPlannerModel)
# =============================================

class MockRLPlannerModel(nn.Module):
    """
    模拟的 RL 规划模型
    简化版本，仅用于演示
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.sample_number = config.get("sample_number", 4)
        
        # 简单的线性层用于轨迹生成
        self.trajectory_encoder = nn.Linear(32, 128)  # 输入特征维度 32
        self.trajectory_decoder = nn.Linear(128, 50 * 5)  # 输出 50步，每步5维
        
    def forward_sample_reward(
        self,
        model_input: Dict[str, Any],
        x_aug: Optional[Dict[str, Any]] = None,
        train_mode: bool = False,
        scenes: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成轨迹并计算奖励
        
        Args:
            model_input: 模型输入 (包含 ego_curr_status 等)
            x_aug: 增强后的输入 (可选)
            train_mode: 训练模式标志
            scenes: 场景列表
            
        Returns:
            包含轨迹和奖励的字典
        """
        
        # 提取输入
        ego_status = model_input.get("ego_curr_status")  # [B, 5]
        if ego_status is None:
            return {"batch_reward": {"total_reward_traj": torch.zeros(1, self.sample_number, 50)}}
        
        B = ego_status.shape[0]
        M = self.sample_number
        T = 50  # 时间步
        
        # 简单的轨迹生成：生成 M 个轨迹候选
        # 实际应该用扩散模型，这里用随机扰动模拟
        ego_status_expanded = ego_status.unsqueeze(1).expand(B, M, -1)  # [B, M, 5]
        
        # 为每个样本生成轨迹 [B*M, 5, T]
        trajectories = []
        for m in range(M):
            # 使用 ego_status 作为初始位置，生成简单的轨迹
            base_traj = ego_status.clone()  # [B, 5] -> [B, 1, 5]
            
            # 生成时间序列轨迹 [B, T, 5]
            traj_sequence = []
            for t in range(T):
                # 简单的物理模型：速度恒定移动
                dt = 0.2
                x_offset = base_traj[:, 2] * dt * (t + 1)  # 使用速度生成位移
                y_offset = base_traj[:, 3] * dt * (t + 1)  # 使用加速度
                
                traj_t = torch.stack([
                    base_traj[:, 0] + x_offset,  # x位置
                    base_traj[:, 1] + y_offset,  # y位置
                    base_traj[:, 2] + 0.1 * (m - M//2),  # 速度（不同样本不同）
                    base_traj[:, 3],  # 加速度
                    base_traj[:, 4] + 0.01 * (m - M//2) * t,  # 朝向
                ], dim=1)
                
                traj_sequence.append(traj_t)
            
            traj = torch.stack(traj_sequence, dim=1)  # [B, T, 5]
            trajectories.append(traj)
        
        # 拼接所有轨迹 [B*M, T, 5]
        ego_future_status = torch.cat(trajectories, dim=0)  # [B*M, T, 5]
        
        # 计算奖励：简单的奖励函数
        # 奖励高度模拟：平滑轨迹 + 避免碰撞
        smoothness_reward = self._compute_smoothness_reward(ego_future_status)  # [B*M, T]
        progress_reward = self._compute_progress_reward(ego_future_status)      # [B*M, T]
        collision_reward = self._compute_collision_reward(ego_future_status)    # [B*M, T]
        
        total_reward = smoothness_reward + progress_reward + collision_reward  # [B*M, T]
        total_reward = total_reward.reshape(B, M, T)  # [B, M, T]
        
        return {
            "ego_future_status": ego_future_status,
            "batch_reward": {
                "total_reward_traj": total_reward,
                "smoothness_reward": smoothness_reward.reshape(B, M, T),
                "progress_reward": progress_reward.reshape(B, M, T),
                "collision_reward": collision_reward.reshape(B, M, T),
            }
        }
    
    def _compute_smoothness_reward(self, traj: Tensor) -> Tensor:
        """计算平滑度奖励：低加速度为正奖励"""
        # 计算加速度 [B*M, T-2]
        accel = (traj[:, 2:, 2:4] - 2 * traj[:, 1:-1, 2:4] + traj[:, :-2, 2:4])
        accel_norm = torch.norm(accel, dim=-1)  # [B*M, T-2]
        
        # 奖励：加速度越小越好
        reward = -0.1 * accel_norm  # [B*M, T-2]
        
        # 补齐到 T 长度
        reward = torch.cat([
            reward[:, :1].expand(-1, 1),
            reward,
            reward[:, -1:].expand(-1, 1),
        ], dim=1)  # [B*M, T]
        
        return reward
    
    def _compute_progress_reward(self, traj: Tensor) -> Tensor:
        """计算进度奖励：向前移动为正奖励"""
        # 计算速度 [B*M, T]
        velocity = torch.norm(traj[:, :, 2:4], dim=-1)  # [B*M, T]
        
        # 奖励：速度越大越好（但要适度）
        reward = 0.5 * torch.clamp(velocity, max=5.0) - 0.5  # [B*M, T]
        
        return reward
    
    def _compute_collision_reward(self, traj: Tensor) -> Tensor:
        """计算碰撞奖励：避免碰撞为正奖励"""
        # 简单的碰撞检测：保持在范围内
        x_pos = traj[:, :, 0]  # [B*M, T]
        y_pos = traj[:, :, 1]  # [B*M, T]
        
        # 如果位置超出范围 [-10, 10]，给予惩罚
        out_of_bounds = (torch.abs(x_pos) > 10.0) | (torch.abs(y_pos) > 10.0)
        reward = torch.where(out_of_bounds, torch.full_like(x_pos, -1.0), torch.zeros_like(x_pos))
        
        return reward


# =============================================
# 第二部分：数据管理器 (替代 DataManager)
# =============================================

class SimpleDataManager:
    """
    简化的数据管理器
    管理一条完整轨迹数据
    """
    
    def __init__(self):
        self.ego_curr_status = None  # [B, 5]
        self.ego_history = []  # 历史状态
        self.agent_status = None  # agent 信息
        self.laneline_pts = None  # 车道线
        self.timestamp = None
        
    def load_from_model_input(self, model_input: Dict[str, Any]):
        """从 model_input 加载数据"""
        self.ego_curr_status = model_input.get("ego_curr_status", torch.zeros(1, 5))
        self.agent_status = model_input.get("agent_status", None)
        self.laneline_pts = model_input.get("laneline_pts", None)
        self.timestamp = model_input.get("timestamp", 0.0)
        
    def build_model_input(self) -> Dict[str, Any]:
        """构造当前时刻的 model_input"""
        return {
            "ego_curr_status": self.ego_curr_status,
            "agent_status": self.agent_status,
            "laneline_pts": self.laneline_pts,
            "timestamp": self.timestamp,
        }
    
    def update_ego_curr_from_future(self, next_state: Tensor):
        """
        更新 ego 当前状态为下一状态
        
        Args:
            next_state: [B, 5] - 下一时刻的状态
        """
        self.ego_curr_status = next_state.detach().clone()
    
    def update_ego_history(self, prev_pose: Tensor, curr_pose: Tensor):
        """更新 ego 历史状态"""
        if len(self.ego_history) < 5:  # 保留最近 5 帧
            self.ego_history.append(prev_pose)
        else:
            self.ego_history.pop(0)
            self.ego_history.append(prev_pose)


# =============================================
# 第三部分：简化的 Lightning 模型
# =============================================

class RunnableRLPlannerModel(torch.nn.Module):
    """
    完全独立可运行的 RL 规划模型
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        lr: float = 0.001,
        warmup_steps: int = 100,
        debug: bool = False,
    ):
        """
        初始化模型
        
        Args:
            config: 配置字典
            lr: 学习率
            warmup_steps: 预热步数
            debug: 调试模式
        """
        super().__init__()
        
        # 默认配置
        if config is None:
            config = {
                "sample_number": 4,
                "closed_loop": {
                    "is_closed_loop": True,
                    "base_dt": 0.2,
                    "expected_interval_time": 0.6,
                    "sim_window_seconds": 5.0,
                    "closed_time_horizon": 10,
                },
            }
        
        self._config = config
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.debug = debug
        self.step_count = 0
        
        # 核心模型
        self.model = MockRLPlannerModel(config)
        
        # 优化器（稍后初始化）
        self.optimizer = None
        self.scheduler = None
        
        # 训练配置
        self.sample_number = config.get("sample_number", 4)
        self.sample_number_config = self.sample_number
        
        # 指标跟踪
        self.total_loss = 0.0
        self.num_steps = 0
        
    def configure_optimizers(self, num_training_steps: int):
        """
        配置优化器和学习率调度
        
        Args:
            num_training_steps: 总训练步数
        """
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
        
        # 学习率调度：先预热后衰减
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - self.warmup_steps)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        return self.optimizer, self.scheduler
    
    def _detach_copy_model_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝数据并detach所有tensor"""
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
    
    def _concat_model_inputs(self, dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """沿 batch 维度拼接多个 model_input 字典"""
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
                result[key] = torch.cat(
                    [v for v in values if torch.is_tensor(v)],
                    dim=0
                )
            elif isinstance(first_val, (list, tuple)):
                merged = []
                for v in values:
                    if v is not None and isinstance(v, (list, tuple)):
                        merged.extend(v)
                result[key] = merged
            else:
                result[key] = first_val
        
        return result
    
    def _run_closed_loop_rollout(
        self,
        model_input: Dict[str, Any],
        sample_number: Optional[int] = None,
    ) -> tuple:
        """
        执行闭环仿真
        
        返回值：
            (closed_loop_trajs, sample_number_final)
        """
        
        # ===== 初始化参数 =====
        
        closed_loop_cfg = self._config.get("closed_loop", {})
        base_dt = closed_loop_cfg.get("base_dt", 0.2)
        expected_interval_time = closed_loop_cfg.get("expected_interval_time", 0.6)
        sim_window_seconds = closed_loop_cfg.get("sim_window_seconds", 5.0)
        closed_time_horizon = closed_loop_cfg.get("closed_time_horizon", 10)
        
        # 计算步长
        interval = max(1, int(round(expected_interval_time / base_dt)))
        sim_window_steps = int(round(sim_window_seconds / base_dt))
        total_steps = int(closed_time_horizon / expected_interval_time)
        
        if sample_number is None:
            sample_number = self.sample_number_config
        self.sample_number = sample_number
        self.model.sample_number = sample_number  # 重置模型的样本数
        
        device = model_input.get("ego_curr_status", torch.zeros(1, 5)).device
        batch_size = model_input.get("ego_curr_status", torch.zeros(1, 5)).shape[0]
        
        # ===== 初始化仿真 =====
        
        data_manager = SimpleDataManager()
        data_manager.load_from_model_input(model_input)
        
        data_managers = [data_manager]
        closed_loop_trajs = {0: []}
        
        if self.debug:
            print(f"[Closed-Loop] 启动仿真")
            print(f"  总步数: {total_steps}")
            print(f"  样本数: {sample_number}")
            print(f"  Batch大小: {batch_size}")
        
        # ===== 主循环 =====
        
        for step_idx in range(total_steps):
            if self.debug:
                print(f"  [Step {step_idx}/{total_steps}]")
            
            # 步骤1：为每个分支构造 model_input
            model_input_sim_list = []
            for dm in data_managers:
                model_input_sim = self._detach_copy_model_input(dm.build_model_input())
                model_input_sim_list.append(model_input_sim)
            
            # 步骤2：拼接成大 batch
            model_input_batch = self._concat_model_inputs(model_input_sim_list)
            
            # 步骤3：调用模型生成轨迹并计算奖励
            with torch.no_grad():
                model_output = self.model.forward_sample_reward(
                    model_input_batch,
                    train_mode=False,
                )
            
            # 步骤4：选择最优轨迹
            ego_future_status = model_output["ego_future_status"]  # [B*M, T, 5]
            batch_reward = model_output["batch_reward"]
            total_reward_traj = batch_reward["total_reward_traj"]  # [B, M, T]
            
            B = batch_size
            M = self.sample_number
            T = ego_future_status.shape[1]
            
            # Reshape 轨迹为 [B, M, T, 5]
            traj = ego_future_status.reshape(B, M, T, 5)
            
            # 计算奖励分数
            score = total_reward_traj.mean(dim=-1)  # [B, M]
            best_idx = score.argmax(dim=1)  # [B]
            best_traj = traj[torch.arange(B), best_idx]  # [B, T, 5]
            
            # 步骤5：提取下一状态
            lqr_step_idx = min(interval, T)
            next_state = best_traj[:, lqr_step_idx - 1, :].detach()  # [B, 5]
            
            # 保存轨迹
            for k in range(lqr_step_idx):
                closed_loop_trajs[0].append(best_traj[:, k, :].detach().clone())
            
            # 步骤6：更新状态
            for dm in data_managers:
                dm.update_ego_curr_from_future(next_state)
                dm.timestamp = float(step_idx + 1) * expected_interval_time
            
            # 步骤7：衰减样本数
            self.sample_number = max(int(self.sample_number / (4 ** (step_idx + 1))), 1)
            self.model.sample_number = self.sample_number  # 同步模型的样本数
            
            if self.debug:
                print(f"    奖励: {score.mean().item():.4f}, 样本数: {self.sample_number}")
        
        if self.debug:
            print(f"[Closed-Loop] 完成，轨迹长度: {len(closed_loop_trajs[0])}")
        
        return closed_loop_trajs, self.sample_number
    
    def _inference_step(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        内部推理方法：执行闭环或开环推理
        
        返回值：
            完整的模型输出字典
        """
        is_closed_loop = self._config.get("closed_loop", {}).get("is_closed_loop", False)
        
        if is_closed_loop:
            # 闭环推理
            closed_loop_trajs, final_sample = self._run_closed_loop_rollout(
                model_input,
                sample_number=self.sample_number_config,
            )
            
            if 0 in closed_loop_trajs and len(closed_loop_trajs[0]) > 0:
                trajs = torch.stack(closed_loop_trajs[0])  # [T, B, 5]
                return {
                    "ego_future_status": trajs,
                    "batch_reward": {"total_reward_traj": trajs[:, :, 0]},  # 占位符
                    "is_closed_loop": True,
                    "closed_loop_trajs": closed_loop_trajs,
                }
        
        # 开环推理：快速生成单次轨迹
        output = self.model.forward_sample_reward(model_input)
        output["is_closed_loop"] = False
        output["closed_loop_trajs"] = None
        return output
    
    def training_step(self, batch: Dict[str, Any]) -> Tensor:
        """
        训练一步
        
        Args:
            batch: 批数据
            
        Returns:
            损失值
        """
        
        self.step_count += 1
        
        if self.debug:
            print(f"\n[Training Step {self.step_count}]")
        
        # 提取数据
        model_input = batch.get("model_input", {})
        
        if not model_input or "ego_curr_status" not in model_input:
            return torch.tensor(0.0)
        
        # 执行推理
        output = self._inference_step(model_input)
        trajs = output["ego_future_status"]
        closed_loop_trajs = output.get("closed_loop_trajs")
        
        # 计算损失
        smoothness_loss = torch.tensor(0.0, requires_grad=True)
        
        if closed_loop_trajs is not None:
            # 闭环模式：基于完整轨迹计算损失
            if 0 in closed_loop_trajs and len(closed_loop_trajs[0]) > 1:
                # trajs shape [T, B, 5]
                if trajs.shape[0] > 2:
                    accel = trajs[2:, :, 2:4] - 2 * trajs[1:-1, :, 2:4] + trajs[:-2, :, 2:4]
                    smoothness_loss = torch.norm(accel, dim=-1).mean()
        else:
            # 开环模式：简单损失
            smoothness_loss = torch.tensor(0.0, requires_grad=True)
        
        # 计算总损失
        # 添加 L2 正则化确保梯度流经模型参数
        l2_reg = sum(0.0001 * torch.sum(p ** 2) for p in self.parameters())
        loss = smoothness_loss + l2_reg
        
        # 记录
        self.total_loss += loss.item() if isinstance(loss, torch.Tensor) else 0.0
        self.num_steps += 1
        
        if self.debug:
            print(f"  损失: {loss.item():.6f}")
        
        return loss
    
    def forward(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        前向传播
        
        根据 is_closed_loop 决定是否执行闭环仿真：
        - True: 执行完整闭环仿真（推理模式下也会执行）
        - False: 仅生成单次轨迹（快速推理）
        """
        output = self._inference_step(model_input)
        
        # 移除内部标记，只返回推理结果
        output.pop("is_closed_loop", None)
        output.pop("closed_loop_trajs", None)
        
        return output


# =============================================
# 第四部分：完整的训练演示
# =============================================

def create_dummy_batch(batch_size: int = 2) -> Dict[str, Any]:
    """
    创建虚拟batch数据
    
    Args:
        batch_size: batch大小
        
    Returns:
        包含 model_input 的batch字典
    """
    return {
        "model_input": {
            "ego_curr_status": torch.randn(batch_size, 5),  # [x, y, vx, vy, yaw]
            "agent_status": torch.randn(batch_size, 10, 6),  # 10个agents，每个6维
            "laneline_pts": torch.randn(batch_size, 50, 2),  # 50个车道线点
            "timestamp": 0.0,
        }
    }


def main():
    """完整的演示流程"""
    
    print("=" * 70)
    print("RL 规划模型独立可运行演示")
    print("=" * 70)
    
    # ===== 第1阶段：初始化 =====
    
    print("\n[1] 创建配置和模型")
    config = {
        "sample_number": 4,
        "closed_loop": {
            "is_closed_loop": True,
            "base_dt": 0.2,
            "expected_interval_time": 0.6,
            "sim_window_seconds": 5.0,
            "closed_time_horizon": 10,
        },
    }
    
    model = RunnableRLPlannerModel(
        config=config,
        lr=0.001,
        warmup_steps=100,
        debug=True,
    )
    
    print(f"✓ 模型创建成功")
    print(f"  - 样本数: {model.sample_number_config}")
    print(f"  - 学习率: {model.lr}")
    
    # ===== 第2阶段：配置优化器 =====
    
    print("\n[2] 配置优化器和学习率调度")
    num_training_steps = 1000
    optimizer, scheduler = model.configure_optimizers(num_training_steps)
    
    print(f"✓ 优化器和调度器配置完成")
    print(f"  - 优化器: AdamW")
    print(f"  - 总训练步数: {num_training_steps}")
    print(f"  - 预热步数: {model.warmup_steps}")
    
    # ===== 第3阶段：训练循环 =====
    
    print("\n[3] 执行训练循环")
    num_epochs = 2
    batch_size = 2
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        for batch_idx in range(3):  # 每个epoch 3个batch
            print(f"\nBatch {batch_idx + 1}/3:")
            
            # 创建虚拟batch
            batch = create_dummy_batch(batch_size=batch_size)
            
            # 前向传播和损失计算
            loss = model.training_step(batch)
            
            # 反向传播
            if model.optimizer is not None:
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                scheduler.step()
                
                # 打印当前学习率
                current_lr = model.optimizer.param_groups[0]['lr']
                print(f"  ✓ 参数更新完成，当前学习率: {current_lr:.6f}")
    
    # ===== 第4阶段：推理演示 =====
    
    print("\n[4] 推理演示（不更新参数）")
    model.eval()
    with torch.no_grad():
        test_batch = create_dummy_batch(batch_size=1)
        
        print("\n执行推理...")
        output = model.forward(test_batch["model_input"])
        
        print(f"✓ 推理完成")
        print(f"  - 输出键: {output.keys()}")
        print(f"  - 轨迹形状: {output['ego_future_status'].shape}")
        print(f"  - 奖励奖励形状: {output['batch_reward']['total_reward_traj'].shape}")
    
    # ===== 第5阶段：统计 =====
    
    print("\n[5] 训练统计")
    print(f"✓ 训练完成")
    print(f"  - 总训练步数: {model.step_count}")
    print(f"  - 平均损失: {model.total_loss / model.num_steps:.6f}")
    print(f"  - 最终样本数: {model.sample_number}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    main()
