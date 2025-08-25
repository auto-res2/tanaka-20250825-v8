import os
import time
import json
from typing import Optional, Dict, Any

import torch
import torchvision as tv
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import DDPMScheduler
from torch_fidelity import calculate_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGES_DIR = os.path.join(PROJECT_ROOT, '.research', 'iteration1', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


@torch.no_grad()
def sample_images(model, scheduler: DDPMScheduler, n: int, image_size: int, device: torch.device, batch: int = 64, steps: Optional[int] = None) -> torch.Tensor:
    model.eval()
    samples = []
    for i in range(0, n, batch):
        cur = min(batch, n - i)
        x = torch.randn(cur, 3, image_size, image_size, device=device)
        T = scheduler.num_train_timesteps if steps is None else steps
        scheduler.set_timesteps(T)
        for t in scheduler.timesteps:
            t_batch = torch.full((cur,), int(t), device=device, dtype=torch.long)
            noise_pred = model(x, t_batch).sample
            x = scheduler.step(noise_pred, int(t), x).prev_sample
        samples.append(x.clamp(-1, 1).cpu())
    return torch.cat(samples, dim=0)


def save_sample_grid(samples: torch.Tensor, filename_pdf: str, nrow: int = 8):
    os.makedirs(os.path.dirname(filename_pdf), exist_ok=True)
    grid = vutils.make_grid(samples[: nrow * nrow] * 0.5 + 0.5, nrow=nrow)
    # Save as high-quality PDF by plotting grid
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.savefig(filename_pdf, bbox_inches='tight')
    plt.close()


def compute_fidelity_metrics_from_samples(samples_dir: str, ref_dataset_name: str, ref_root: Optional[str]) -> Dict[str, float]:
    if ref_dataset_name.lower() == 'cifar10':
        ref = 'cifar10-train'
    elif ref_dataset_name.lower() == 'synthetic':
        return {"frechet_inception_distance": float('nan'), "inception_score_mean": float('nan'), "precision": float('nan'), "recall": float('nan')}
    else:
        assert ref_root is not None and os.path.exists(ref_root), 'Provide ref_root for this dataset.'
        ref = ref_root
    m = calculate_metrics(input1=samples_dir, input2=ref, fid=True, inception_score=True, precision=True, recall=True, verbose=False, cuda=torch.cuda.is_available())
    return {k: float(v) for k, v in m.items()}


def evaluate_and_save(model, scheduler, device, run_tag: str, dataset_name: str, ref_root: Optional[str],
                      image_size: int, num_samples: int = 512, batch: int = 64, steps: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.time()
    samples = sample_images(model, scheduler, n=num_samples, image_size=image_size, device=device, batch=batch, steps=steps)
    t1 = time.time()

    # Save samples for metrics (PNG) under images dir
    samples_dir = os.path.join(IMAGES_DIR, f"samples_{run_tag}")
    os.makedirs(samples_dir, exist_ok=True)
    for i, img in enumerate(samples):
        tv.utils.save_image(img * 0.5 + 0.5, os.path.join(samples_dir, f"{i:06d}.png"))

    # Also save a compact PDF grid for the paper
    grid_pdf = os.path.join(IMAGES_DIR, f"grid_{run_tag}.pdf")
    save_sample_grid(samples, grid_pdf, nrow=8)

    metrics = compute_fidelity_metrics_from_samples(samples_dir, dataset_name, ref_root)

    # Inference memory (50-step tail for speed)
    scheduler.set_timesteps(scheduler.num_train_timesteps)
    def infer_tail():
        b = min(batch, 8)
        x = torch.randn(b, 3, image_size, image_size, device=device)
        for t in scheduler.timesteps[-50:]:
            tb = torch.full((b,), int(t), device=device, dtype=torch.long)
            pred = model(x, tb).sample
            x = scheduler.step(pred, int(t), x).prev_sample
        return x

    reset_peak_memory()
    infer_tail()
    infer_peak = peak_memory_gb()

    report = {
        'samples_dir': samples_dir,
        'grid_pdf': grid_pdf,
        'sample_time_sec': (t1 - t0),
        'infer_peak_gb': infer_peak,
        'metrics': metrics
    }
    with open(os.path.join(IMAGES_DIR, f"eval_{run_tag}.json"), 'w') as f:
        json.dump(report, f, indent=2)
    return report
