import os
import sys
import json
import argparse
import random
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure relative imports from src when running as a script
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from preprocess import make_dataset
from train import train_pipeline, ModelConfig, TrainConfig, IMAGES_DIR
from evaluate import evaluate_and_save

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_plot(fig, filename: str):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_head2head(df: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='model', y='train_peak_gb', hue='model', ax=ax)
    ax.set_title('Peak training VRAM (GB)')
    ax.set_ylabel('GB'); ax.set_xlabel('Model')
    save_plot(fig, os.path.join(outdir, 'peak_memory_head2head.pdf'))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='model', y='ips', hue='model', ax=ax)
    ax.set_title('Throughput (images/sec)')
    ax.set_ylabel('images/sec'); ax.set_xlabel('Model')
    save_plot(fig, os.path.join(outdir, 'throughput_head2head.pdf'))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='model', y='loss', hue='model', ax=ax)
    ax.set_title('Training loss (per-image MSE)')
    ax.set_ylabel('loss'); ax.set_xlabel('Model')
    save_plot(fig, os.path.join(outdir, 'training_loss_head2head.pdf'))

    if 'fid' in df.columns and df['fid'].notnull().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x='model', y='fid', hue='model', ax=ax)
        ax.set_title('FID (lower is better)')
        ax.set_ylabel('FID'); ax.set_xlabel('Model')
        save_plot(fig, os.path.join(outdir, 'fid_head2head.pdf'))


def plot_ablation(df: pd.DataFrame, outdir: str):
    if 'lora_rank' in df.columns and df['lora_rank'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=df, x='lora_rank', y='fid', hue='reversible', style='recurrent', markers=True, ax=ax)
        ax.set_title('FID across LoRA rank and toggles')
        ax.set_xlabel('LoRA rank'); ax.set_ylabel('FID')
        save_plot(fig, os.path.join(outdir, 'fid_ablation.pdf'))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x='grad_quant', y='train_peak_gb', hue='reversible', ax=ax)
    ax.set_title('Peak VRAM (GB) across toggles')
    ax.set_xlabel('Grad quant'); ax.set_ylabel('GB')
    save_plot(fig, os.path.join(outdir, 'peak_memory_ablation.pdf'))


def run_quick_test(args):
    print('[QuickTest] Starting synthetic quick test')
    outdir = IMAGES_DIR
    rows = []
    for model_type in ['unet', 'r2']:
        set_seed(0)
        ds = make_dataset('synthetic', None, 32)
        mcfg = ModelConfig(model_type=model_type, image_size=32, base_channels=32, lora_rank=2, K=2)
        tcfg = TrainConfig(lr=1e-3, epochs=1, batch_size=16, num_workers=0, mixed_precision='no')
        result = train_pipeline(ds, mcfg, tcfg, out_prefix=f'quicktest_{model_type}', use_ema=True, grad_quant=False)
        model = result['model']
        accelerator = result['accelerator']
        scheduler = result['scheduler']
        ep0 = result['report']['epoch_stats'][0]
        peak_gb = result['report']['train_peak_gb']
        # Evaluate few samples (20 steps for speed)
        evalr = evaluate_and_save(model, scheduler, accelerator.device, run_tag=f'quicktest_{model_type}',
                                  dataset_name='synthetic', ref_root=None, image_size=32, num_samples=128, batch=8, steps=20)
        rows.append({'model': model_type, 'loss': ep0['loss'], 'ips': ep0['ips'], 'train_peak_gb': peak_gb,
                     'infer_peak_gb': evalr['infer_peak_gb']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'quick_test.csv'), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='model', y='loss', ax=ax)
    ax.set_title('Quick test training loss (synthetic)')
    ax.set_ylabel('loss')
    save_plot(fig, os.path.join(outdir, 'training_loss_quicktest.pdf'))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='model', y='train_peak_gb', ax=ax)
    ax.set_title('Quick test peak VRAM')
    ax.set_ylabel('GB')
    save_plot(fig, os.path.join(outdir, 'peak_memory_quicktest.pdf'))

    print('[QuickTest] Done. CSV and PDFs saved to', outdir)


def run_head2head(args):
    outdir = IMAGES_DIR
    all_rows: List[dict] = []
    ds = make_dataset(args.dataset, args.data_root, args.image_size)
    for model_type in ['unet', 'r2']:
        for seed in args.seeds:
            set_seed(seed)
            print(f"[Head2Head] model={model_type} seed={seed}")
            mcfg = ModelConfig(model_type=model_type, image_size=args.image_size, base_channels=args.base_channels, lora_rank=args.lora_rank, K=args.K)
            tcfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, mixed_precision=args.mixed_precision)
            result = train_pipeline(ds, mcfg, tcfg, out_prefix=f'head2head_{model_type}_s{seed}', use_ema=True, grad_quant=args.grad_quant and model_type=='r2')
            model = result['model']
            accelerator = result['accelerator']
            scheduler = result['scheduler']
            ep = result['report']['epoch_stats'][-1]
            peak_gb = result['report']['train_peak_gb']
            evalr = evaluate_and_save(model, scheduler, accelerator.device, run_tag=f'head2head_{model_type}_s{seed}',
                                      dataset_name=args.dataset, ref_root=args.data_root, image_size=args.image_size,
                                      num_samples=args.num_samples, batch=args.eval_batch, steps=None)
            row = {'model': model_type, 'seed': seed, 'loss': ep['loss'], 'ips': ep['ips'], 'train_peak_gb': peak_gb,
                   'infer_peak_gb': evalr['infer_peak_gb'], 'fid': evalr['metrics'].get('frechet_inception_distance', float('nan'))}
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(os.path.join(outdir, 'head2head.csv'), index=False)
    df = pd.DataFrame(all_rows)
    plot_head2head(df, outdir)
    print('[Head2Head] Done. Outputs in', outdir)


def run_ablation(args):
    outdir = IMAGES_DIR
    all_rows: List[dict] = []
    ds = make_dataset(args.dataset, args.data_root, args.image_size)
    for seed in args.seeds:
        set_seed(seed)
        for reversible in [True, False]:
            for recurrent in [True, False]:
                for lora_rank in [0, 2, 4, 8]:
                    for grad_quant in [False, True]:
                        tag = f"abl_s{seed}_rev{int(reversible)}_rec{int(recurrent)}_r{lora_rank}_q{int(grad_quant)}"
                        print('[Ablation]', tag)
                        mcfg = ModelConfig(model_type='r2', image_size=args.image_size, base_channels=args.base_channels,
                                           lora_rank=lora_rank, reversible=reversible, recurrent=recurrent, K=args.K)
                        tcfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, mixed_precision=args.mixed_precision)
                        result = train_pipeline(ds, mcfg, tcfg, out_prefix=tag, use_ema=True, grad_quant=grad_quant)
                        model = result['model']
                        accelerator = result['accelerator']
                        scheduler = result['scheduler']
                        ep = result['report']['epoch_stats'][-1]
                        peak_gb = result['report']['train_peak_gb']
                        evalr = evaluate_and_save(model, scheduler, accelerator.device, run_tag=tag,
                                                  dataset_name=args.dataset, ref_root=args.data_root, image_size=args.image_size,
                                                  num_samples=min(args.num_samples, 10000), batch=args.eval_batch, steps=200)
                        row = {'seed': seed, 'reversible': reversible, 'recurrent': recurrent, 'lora_rank': lora_rank, 'grad_quant': grad_quant,
                               'loss': ep['loss'], 'ips': ep['ips'], 'train_peak_gb': peak_gb,
                               'fid': evalr['metrics'].get('frechet_inception_distance', float('nan'))}
                        all_rows.append(row)
                        pd.DataFrame(all_rows).to_csv(os.path.join(outdir, 'ablation.csv'), index=False)
    df = pd.DataFrame(all_rows)
    plot_ablation(df, outdir)
    print('[Ablation] Done. Outputs in', outdir)


def run_hires(args):
    outdir = IMAGES_DIR
    rows = []
    ds = make_dataset(args.dataset, args.data_root, args.image_size)
    for model_type in ['unet', 'r2']:
        for seed in args.seeds:
            set_seed(seed)
            tag = f"hires_{model_type}_s{seed}"
            print('[HiRes] Training', tag)
            mcfg = ModelConfig(model_type=model_type, image_size=args.image_size, base_channels=args.base_channels, lora_rank=args.lora_rank, K=args.K)
            tcfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, mixed_precision=args.mixed_precision)
            result = train_pipeline(ds, mcfg, tcfg, out_prefix=tag, use_ema=False, grad_quant=False)
            model = result['model']
            accelerator = result['accelerator']
            scheduler = result['scheduler']
            for res in [256, 512] + ([1024] if args.test_1024 else []):
                print('[HiRes] Inference tail @', res)
                scheduler.set_timesteps(1000)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                # 50-step tail inference to gauge memory
                b = min(args.eval_batch, 2)
                x = torch.randn(b, 3, res, res, device=accelerator.device)
                for t in scheduler.timesteps[-50:]:
                    tb = torch.full((b,), int(t), device=accelerator.device, dtype=torch.long)
                    pred = model(x, tb).sample
                    x = scheduler.step(pred, int(t), x).prev_sample
                infer_peak = 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    infer_peak = torch.cuda.max_memory_allocated() / (1024**3)
                rows.append({'model': model_type, 'seed': seed, 'resolution': res, 'train_peak_gb': result['report']['train_peak_gb'], 'infer_peak_gb': infer_peak})
                pd.DataFrame(rows).to_csv(os.path.join(outdir, 'hires.csv'), index=False)
    df = pd.DataFrame(rows)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df, x='resolution', y='infer_peak_gb', hue='model', marker='o', ax=ax)
        ax.set_title('Inference peak VRAM vs resolution')
        ax.set_xlabel('Resolution'); ax.set_ylabel('Peak GB')
        save_plot(fig, os.path.join(outdir, 'inference_memory_hires.pdf'))
    print('[HiRes] Done. Outputs in', outdir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=str, required=True, choices=['quick_test', 'head2head', 'ablation', 'hires'])
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_batch', type=int, default=64)
    p.add_argument('--num_samples', type=int, default=5000)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--base_channels', type=int, default=128)
    p.add_argument('--lora_rank', type=int, default=4)
    p.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--grad_quant', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--test_1024', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    import torch
    args = parse_args()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(os.path.join(IMAGES_DIR, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.experiment == 'quick_test':
        run_quick_test(args)
    elif args.experiment == 'head2head':
        run_head2head(args)
    elif args.experiment == 'ablation':
        run_ablation(args)
    elif args.experiment == 'hires':
        run_hires(args)
