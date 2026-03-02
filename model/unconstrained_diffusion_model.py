"""
无约束扩散模型 (Unconstrained Diffusion Model) / DDPM.
Denoising Diffusion Probabilistic Model，无条件生成。
"""

import logging
import math
import pathlib
import time
import torch
import torch.nn as nn
from PIL import Image
from pytorch_lightning import LightningModule

log = logging.getLogger(__name__)


def _save_tensor_as_images(tensor: torch.Tensor, save_dir: str, batch_idx: int, scale: int = 1):
    """将 (B,C,H,W) 张量保存为 PNG 图片。"""
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().cpu().float()
    x = (x + 1) / 2
    x = (x.clamp(0, 1) * 255).byte().numpy()
    for i in range(x.shape[0]):
        img = x[i]
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)
        Image.fromarray(img).save(save_path / f"batch{batch_idx:04d}_idx{i:04d}.png")
    log.debug("已保存 %d 张图片到 %s", x.shape[0], save_path)


def timestep_embedding(timesteps, dim):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = nn.SiLU()(h)
        h = self.conv1(h)
        h = h + self.mlp(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    """Simple U-Net for noise prediction."""

    def __init__(self, in_ch=1, out_ch=1, base_ch=64, ch_mult=(1, 2, 4), time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        chs = [base_ch * m for m in ch_mult]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        prev_ch = in_ch
        for ch in chs:
            self.down.append(ResBlock(prev_ch, ch, time_emb_dim))
            prev_ch = ch
        self.mid = ResBlock(prev_ch, prev_ch, time_emb_dim)
        for ch in reversed(chs[:-1]):
            self.up.append(nn.Conv2d(prev_ch * 2, ch, 3, padding=1))
            self.up.append(ResBlock(ch, ch, time_emb_dim))
            prev_ch = ch
        self.final = nn.Conv2d(base_ch * 2, out_ch, 1)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        hs = []
        h = x
        for block in self.down:
            h = block(h, t_emb)
            hs.append(h)
            h = nn.functional.avg_pool2d(h, 2)

        h = self.mid(h, t_emb)

        for i in range(0, len(self.up), 2):
            h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
            h = torch.cat([h, hs.pop()], dim=1)
            h = self.up[i + 1](self.up[i](h), t_emb)

        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, hs[0]], dim=1)
        return self.final(h)


class UnconstrainedDiffusionModel(LightningModule):
    """无约束扩散模型 (DDPM)。"""

    def __init__(
        self,
        lr: float = 1e-4,
        img_size: int = 32,
        in_channels: int = 1,
        base_ch: int = 64,
        ch_mult: tuple = (1, 2, 4),
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        save_predictions: bool = True,
        predict_save_dir: str | None = None,
        inference_timesteps: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        log.info("UnconstrainedDiffusionModel 初始化: lr=%s, img_size=%s, timesteps=%s", lr, img_size, timesteps)

        self.unet = UNet(
            in_ch=in_channels,
            out_ch=in_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
        )

        self.timesteps = timesteps
        if beta_schedule == "linear":
            self.register_buffer(
                "betas",
                torch.linspace(beta_start, beta_end, timesteps),
            )
        else:
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.register_buffer("betas", betas)

        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def _get_loss(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
        noise = torch.randn_like(x, device=x.device)
        x_noisy = self.q_sample(x, t, noise)
        pred_noise = self.unet(x_noisy, t)
        return nn.functional.mse_loss(pred_noise, noise)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def forward(self, x, t):
        return self.unet(x, t)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0 and self.global_step == 0:
            log.debug("training_step 首 batch: loss=%.6f", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return {"test_loss": loss}

    @torch.no_grad()
    def sample(self, shape, device=None, timesteps: int | None = None):
        if device is None:
            device = next(self.parameters()).device
        steps = timesteps if timesteps is not None else getattr(self.hparams, "inference_timesteps", None) or self.timesteps
        if steps != self.timesteps:
            step_indices = torch.linspace(self.timesteps - 1, 0, steps).long().tolist()
        else:
            step_indices = list(reversed(range(self.timesteps)))
        log.info("sample 开始: shape=%s, steps=%d (orig=%d)", shape, len(step_indices), self.timesteps)
        t0 = time.perf_counter()
        x = torch.randn(shape, device=device)
        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch)
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.ones_like(alpha)
            beta = self.betas[t]
            coef = (1 - alpha_prev) / (1 - alpha)
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / self.sqrt_one_minus_alphas_cumprod[t]) * pred_noise
            )
            if t > 0:
                sigma = torch.sqrt(beta * coef)
                x = x + sigma * torch.randn_like(x, device=device)
            if (i + 1) % 100 == 0 or i == len(step_indices) - 1:
                elapsed = time.perf_counter() - t0
                log.info("sample 进度: %d/%d steps, 已用 %.1fs", i + 1, len(step_indices), elapsed)
        log.info("sample 完成: 总耗时 %.1fs", time.perf_counter() - t0)
        return x

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        shape = x.shape
        log.info("predict_step batch_idx=%d shape=%s", batch_idx, shape)
        samples = self.sample(shape)
        if self.hparams.save_predictions:
            save_dir = self.hparams.predict_save_dir or str(pathlib.Path(self.trainer.default_root_dir) / "predictions")
            log.info("保存推理结果到 %s", save_dir)
            _save_tensor_as_images(
                samples,
                save_dir,
                batch_idx,
            )
        return samples
