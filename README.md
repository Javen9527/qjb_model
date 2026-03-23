# qjb-model

基于 PyTorch Lightning 的模型训练框架，支持多种模型与数据配置。

## 项目结构

```
qjb_model/
├── model/                 # 模型
│   ├── simple_model.py           # 简单 MLP 回归
│   ├── transformer_model.py      # Transformer (Attention is All You Need)
│   ├── unconstrained_diffusion_model.py   # 无约束扩散模型 (DDPM)
│   ├── conditional_diffusion_model.py     # 条件扩散模型
│   └── rl_planner_model.py                # RL 规划模型
├── data/                  # 数据模块
│   ├── simple_data.py
│   ├── transformer_data.py
│   ├── unconstrained_diffusion_data.py
│   ├── conditional_diffusion_data.py
│   └── rl_planner_data.py
├── config/                # 配置文件
│   ├── simple.yaml
│   ├── transformer.yaml
│   ├── unconstrained_diffusion.yaml
│   ├── conditional_diffusion.yaml
│   └── rl_planner.yaml
├── launch.py              # 入口
└── config/logging_config.py
```

## 环境

- Python >= 3.14
- PyTorch Lightning
- 使用 `uv` 管理依赖：`uv sync`

## 模型说明

| 模型 | 说明 | 配置 |
|------|------|------|
| **SimpleModel** | 简单 MLP 回归 | `config/simple.yaml` |
| **TransformerModel** | Transformer 编码器-解码器，序列到序列 | `config/transformer.yaml` |
| **UnconstrainedDiffusionModel** | 无约束扩散 (DDPM)，图像生成 | `config/unconstrained_diffusion.yaml` |
| **ConditionalDiffusionModel** | 条件扩散，以类别为条件生成 | `config/conditional_diffusion.yaml` |
| **RLPlannerModel** | RL 规划闭环 rollout 模型 | `config/rl_planner.yaml` |

## 使用方法

### 训练

```bash
# 简单模型
uv run python launch.py fit --config config/simple.yaml

# Transformer
uv run python launch.py fit --config config/transformer.yaml

# 无约束扩散（可指定图片文件夹）
uv run python launch.py fit --config config/unconstrained_diffusion.yaml

# 条件扩散（数据需子目录结构：root/class_0/img.png）
uv run python launch.py fit --config config/conditional_diffusion.yaml

# RL 规划
uv run python launch.py fit --config config/rl_planner.yaml
```

### 测试

```bash
uv run python launch.py test --config config/<config>.yaml --ckpt_path <path_to_checkpoint>
```

### 推理

```bash
uv run python launch.py predict --config config/<config>.yaml --ckpt_path <path_to_checkpoint>
```

扩散模型推理结果会保存到 `default_root_dir/predictions`，可通过 `predict_save_dir` 自定义。

## 数据配置

### Transformer

- 默认使用合成序列数据
- 可从 `.pt` 加载：`dict(src=..., tgt=...)`，或 `[src_tensor, tgt_tensor]`
- 配置：`train_data_path`, `test_data_path`, `predict_data_path`

### 扩散模型

- **无约束**：`train_data_path` 等可为图片文件夹（PNG/JPG 等）或 `.pt`/`.npy`
- **条件**：文件夹需子目录结构（子目录名 = 类别），或 `.pt` 含 `x` 与 `y`

### RL Planner

- 默认使用合成数据（`ego_curr_status/agent_status/laneline_pts`）
- 支持 `.pt/.pth` 文件输入
- 文件格式支持：
  - `list[{"model_input": {...}}]`
  - `dict(ego_curr_status=..., agent_status=..., laneline_pts=..., timestamp=...)`

## 日志

- 控制台：INFO
- 文件：`logs/app.log`（DEBUG）

## 参考

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
