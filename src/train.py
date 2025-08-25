import os
import math
import time
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler

# -------------------------------------------------
# Constants and dirs
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGES_DIR = os.path.join(PROJECT_ROOT, '.research', 'iteration1', 'images')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# R2-Diffusion (lightweight, dependency-minimal scaffold)
# -------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.LongTensor):
        if t.dtype in (torch.int32, torch.int64):
            t = t.float()
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def norm_act(ch: int) -> nn.Module:
    return nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU())


class ConvLoRA(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, padding: int = 0, stride: int = 1, bias: bool = True, r: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, stride=stride, bias=bias)
        self.r = r
        if r > 0:
            self.A = nn.Conv2d(in_ch, r, 1, bias=False)
            self.B = nn.Conv2d(r, out_ch, 1, bias=False)
        else:
            self.A = None
            self.B = None

    def forward(self, x, scale: Optional[torch.Tensor] = None):
        base = self.conv(x)
        if self.r and self.A is not None and self.B is not None:
            lora = self.B(self.A(x))
            if scale is not None:
                lora = lora * scale.view(scale.shape[0], 1, 1, 1)
            return base + lora
        return base


class AdditiveCouplingRevBlock(nn.Module):
    def __init__(self, ch: int, lora_rank: int = 4):
        super().__init__()
        assert ch % 2 == 0, "Channels must be even for coupling."
        c = ch // 2
        self.f = nn.Sequential(
            norm_act(c), ConvLoRA(c, c, 3, padding=1, r=lora_rank),
            norm_act(c), ConvLoRA(c, c, 3, padding=1, r=lora_rank),
        )
        self.g = nn.Sequential(
            norm_act(c), ConvLoRA(c, c, 3, padding=1, r=lora_rank),
            norm_act(c), ConvLoRA(c, c, 3, padding=1, r=lora_rank),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=1)


class R2Cell(nn.Module):
    def __init__(self, ch: int, lora_rank: int = 4, use_gru_gate: bool = True):
        super().__init__()
        self.rev = AdditiveCouplingRevBlock(ch, lora_rank=lora_rank)
        self.t_pos = SinusoidalEmbedding(128)
        self.t_mlp = nn.Sequential(nn.Linear(128, ch), nn.SiLU(), nn.Linear(ch, ch * 2))
        self.use_gru_gate = use_gru_gate
        if self.use_gru_gate:
            self.gru = nn.GRUCell(ch, ch)

    def forward(self, x: torch.Tensor, t: torch.LongTensor, state: Optional[torch.Tensor] = None):
        b, c, h, w = x.shape
        temb = self.t_pos(t)
        ts = self.t_mlp(temb).view(b, 2, c, 1, 1)
        scale, shift = ts[:, 0], ts[:, 1]
        x_mod = x * (1 + scale) + shift
        y = self.rev(x_mod)
        if self.use_gru_gate:
            pooled = y.mean(dim=(2, 3))
            if state is None:
                state = torch.zeros_like(pooled)
            h_new = self.gru(pooled, state)
            gate = torch.sigmoid(h_new).view(b, c, 1, 1)
            y = y * gate + x * (1 - gate)
            return y, h_new
        return y, None


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class R2DiffusionModel(nn.Module):
    def __init__(self, image_size: int = 32, in_channels: int = 3, base_channels: int = 128,
                 channel_mults: Tuple[int, ...] = (1, 2, 2), reversible: bool = True, recurrent: bool = True,
                 lora_rank: int = 4, sparse_attention: bool = False, K: int = 3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.recurrent = recurrent
        self.K = K

        self.downs = nn.ModuleList()
        self.cells = nn.ModuleList()
        self.ups = nn.ModuleList()

        cur_ch = base_channels
        for i, cm in enumerate(channel_mults):
            ch = base_channels * cm
            if i > 0:
                self.downs.append(Downsample(cur_ch))
                cur_ch = ch
            else:
                self.downs.append(nn.Identity())
            self.cells.append(R2Cell(cur_ch, lora_rank=lora_rank, use_gru_gate=True))

        for i, cm in reversed(list(enumerate(channel_mults))):
            self.ups.append(R2Cell(cur_ch, lora_rank=lora_rank, use_gru_gate=True))
            if i > 0:
                self.ups.append(Upsample(cur_ch))

        self.sample = None

    def forward(self, x: torch.Tensor, t: torch.LongTensor):
        h = self.in_conv(x)
        states = []
        for down, cell in zip(self.downs, self.cells):
            h = down(h)
            state = None
            if self.recurrent:
                for _ in range(self.K):
                    h, state = cell(h, t, state)
            else:
                h, state = cell(h, t, None)
            states.append(state)

        for mod in self.ups:
            if isinstance(mod, R2Cell):
                state = states.pop() if len(states) > 0 else None
                if self.recurrent:
                    for _ in range(self.K):
                        h, state = mod(h, t, state)
                else:
                    h, state = mod(h, t, None)
            else:
                h = mod(h)

        out = self.out_conv(h)
        class O: pass
        o = O(); o.sample = out
        return o


# -------------------------------------------------
# Configs and utilities
# -------------------------------------------------

@dataclass
class ModelConfig:
    model_type: str  # 'unet' or 'r2'
    in_channels: int = 3
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 2)
    image_size: int = 32
    lora_rank: int = 4
    recurrent: bool = True
    reversible: bool = True
    K: int = 3


@dataclass
class TrainConfig:
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 64
    grad_accum: int = 1
    ema_decay: float = 0.9999
    mixed_precision: str = 'fp16'  # 'no', 'fp16', 'bf16'
    num_workers: int = 4
    log_every: int = 100


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if hasattr(v, 'dtype') and v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and hasattr(v, 'dtype') and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)


class DummyGradQuantWrapper(nn.Module):
    def __init__(self, module: nn.Module, enabled: bool):
        super().__init__()
        self.module = module
        self.enabled = enabled

    def _q(self, g: torch.Tensor) -> torch.Tensor:
        if (g is None) or (not self.enabled) or (not torch.is_floating_point(g)):
            return g
        with torch.no_grad():
            s = g.abs().max().clamp(min=1e-12)
            scaled = (g / s * 127.0).round().clamp(-128, 127)
            q = scaled.to(torch.int8)
            deq = q.to(torch.float32) * (s / 127.0)
        return deq.to(g.dtype)

    def forward(self, *args, **kwargs):
        def register(t):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(self._q)
            return t
        args = tuple(register(a) for a in args)
        out = self.module(*args, **kwargs)
        if hasattr(out, 'sample') and isinstance(out.sample, torch.Tensor):
            register(out.sample)
        elif isinstance(out, torch.Tensor):
            register(out)
        return out


def make_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model_type == 'unet':
        model = UNet2DModel(
            sample_size=cfg.image_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,
            layers_per_block=2,
            block_out_channels=tuple(cfg.base_channels * m for m in cfg.channel_mults),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        try:
            model.enable_gradient_checkpointing()
        except Exception:
            pass
        return model
    elif cfg.model_type == 'r2':
        return R2DiffusionModel(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_mults=cfg.channel_mults,
            reversible=cfg.reversible,
            recurrent=cfg.recurrent,
            lora_rank=cfg.lora_rank,
            K=cfg.K,
        )
    else:
        raise ValueError(f"Unknown model_type {cfg.model_type}")


# -------------------------------------------------
# Training loop
# -------------------------------------------------

def train_one_epoch(accel: Accelerator, model, scheduler, opt, ema: Optional[EMA], dl, grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()
    t0 = time.time()
    total_loss = 0.0
    total_images = 0
    for step, (x, _) in enumerate(dl):
        with accel.accumulate(model):
            x = x.to(accel.device)
            b = x.size(0)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (b,), device=x.device, dtype=torch.long)
            noisy = scheduler.add_noise(x, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred, noise)
            accel.backward(loss)
            if grad_clip is not None:
                accel.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            total_loss += loss.detach().item() * b
            total_images += b
    t1 = time.time()
    return {"loss": total_loss / max(total_images, 1), "ips": total_images / (t1 - t0 + 1e-9), "time_sec": (t1 - t0)}


def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def train_pipeline(dataset, model_cfg: ModelConfig, train_cfg: TrainConfig, out_prefix: str,
                   use_ema: bool = True, grad_quant: bool = False) -> Dict[str, Any]:
    dl = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, drop_last=True)
    model = make_model(model_cfg)
    if grad_quant and model_cfg.model_type == 'r2':
        model = DummyGradQuantWrapper(model, enabled=True)
    accelerator = Accelerator(mixed_precision=train_cfg.mixed_precision)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=train_cfg.betas, weight_decay=train_cfg.weight_decay)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    ema = EMA(model, train_cfg.ema_decay) if use_ema else None

    # Reset CUDA peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    epoch_stats = []
    for ep in range(train_cfg.epochs):
        stats = train_one_epoch(accelerator, model, scheduler, optimizer, ema, dl)
        epoch_stats.append(stats)
        print(f"[train] epoch={ep+1}/{train_cfg.epochs} loss={stats['loss']:.4f} ips={stats['ips']:.1f}")

    peak_gb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Save checkpoint
    ckpt_name = f"{out_prefix}_last.pth"
    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    if use_ema and ema is not None:
        ema.copy_to(model)
    save_model(model, ckpt_path)

    report = {
        'epoch_stats': epoch_stats,
        'train_peak_gb': peak_gb,
        'ckpt_path': ckpt_path,
    }
    with open(os.path.join(MODELS_DIR, f"{out_prefix}_train_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    return {'model': model, 'accelerator': accelerator, 'scheduler': scheduler, 'report': report}
