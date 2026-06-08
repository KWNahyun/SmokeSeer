"""
LightGaussian-based pruning for SmokeSeer checkpoints.

Loads a trained SmokeSeer Stage 2 checkpoint, computes importance scores
for surface and smoke Gaussians across all training views, prunes the
least important ones, and saves a new pruned checkpoint.

Usage:
  python prune_smokeseer.py -m ./output/red_sub/baseline --prune_percent 0.6
  python prune_smokeseer.py -m ./output/red_sub/baseline --prune_percent 0.6 --finetune_iters 5000
import sys
sys.path.insert(0, os.path.expanduser("~/LightGaussian/submodules/compress-diff-gaussian-rasterization"))
"""

import os
import gc
import sys
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, DeformModel, GaussianSmokeThermalModel, GaussianSurfaceThermalModel
from gaussian_renderer.gsplat_render import count_render
from arguments import PipelineParams


# ---------------------------------------------------------------------------
# Importance score computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_importance(gaussians, cameras, pipe):
    """Accumulate importance scores across all training cameras."""
    bg = torch.zeros(3, device="cuda")
    imp_score = None

    for cam in tqdm(cameras, desc="  Computing importance"):
        pkg = count_render(cam, gaussians, pipe)
        score = pkg["important_score"].detach()
        if imp_score is None:
            imp_score = score
        else:
            imp_score += score
        gc.collect()

    return imp_score  # (N,)


def prune_gaussians(gaussians, imp_score, prune_percent):
    """Prune the least important Gaussians."""
    n_total = gaussians.get_xyz.shape[0]
    n_prune = int(n_total * prune_percent)
    n_keep  = n_total - n_prune

    _, sorted_idx = torch.sort(imp_score, descending=True)
    keep_mask = torch.zeros(n_total, dtype=torch.bool, device="cuda")
    keep_mask[sorted_idx[:n_keep]] = True
    prune_mask = ~keep_mask

    gaussians.prune_points(prune_mask)
    print(f"  Pruned: {n_total:,} → {gaussians.get_xyz.shape[0]:,} "
          f"({prune_percent*100:.0f}% removed)")
    return gaussians


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="LightGaussian pruning for SmokeSeer")
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--prune_percent", type=float, default=0.6,
                        help="Fraction of Gaussians to prune (default: 0.6)")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Checkpoint iteration to load (default: 30000)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load config
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        print(f"cfg_args not found at {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        dataset = eval(f.read())

    # Pipeline params
    pp = PipelineParams(ArgumentParser())
    pipe = pp.extract(dataset)

    # Load checkpoints
    ckpt_surface = model_path / f"ft_chkpnt_surface_thermal{args.iteration}.pth"
    ckpt_smoke   = model_path / f"ft_chkpnt_smoke_thermal{args.iteration}.pth"
    if not ckpt_surface.exists() or not ckpt_smoke.exists():
        print(f"Checkpoints not found for iteration {args.iteration}")
        sys.exit(1)

    print(f"\nLoading checkpoints from {model_path}")
    gaussians_surface = GaussianSurfaceThermalModel(dataset.sh_degree_stage1)
    gaussians_smoke   = GaussianSmokeThermalModel(dataset.sh_degree_stage2)
    scene = Scene(dataset, gaussians_surface, shuffle=False)

    (params_s, _) = torch.load(ckpt_surface, map_location="cuda")
    gaussians_surface.restore(params_s, dataset)
    (params_k, _) = torch.load(ckpt_smoke, map_location="cuda")
    gaussians_smoke.restore(params_k, dataset)

    cameras = scene.getTrainCamerasUnshuffled()
    print(f"  Training cameras: {len(cameras)}")
    print(f"  Surface Gaussians before pruning: {gaussians_surface.get_xyz.shape[0]:,}")
    print(f"  Smoke   Gaussians before pruning: {gaussians_smoke.get_xyz.shape[0]:,}")

    # --- Prune surface ---
    print("\n[Surface] Computing importance scores...")
    imp_surface = compute_importance(gaussians_surface, cameras, pipe)
    prune_gaussians(gaussians_surface, imp_surface, args.prune_percent)

    # --- Prune smoke ---
    print("\n[Smoke] Computing importance scores...")
    imp_smoke = compute_importance(gaussians_smoke, cameras, pipe)
    prune_gaussians(gaussians_smoke, imp_smoke, args.prune_percent)

    # --- Save pruned checkpoints ---
    out_iter = f"{args.iteration}_pruned{int(args.prune_percent*100)}"
    out_surface = model_path / f"ft_chkpnt_surface_thermal{out_iter}.pth"
    out_smoke   = model_path / f"ft_chkpnt_smoke_thermal{out_iter}.pth"

    # Need optimizer for capture() — set up minimal one
    gaussians_surface.training_setup(dataset)
    gaussians_smoke.training_setup(dataset)

    torch.save((gaussians_surface.capture(), args.iteration), out_surface)
    torch.save((gaussians_smoke.capture(), args.iteration), out_smoke)

    print(f"\nSaved pruned checkpoints:")
    print(f"  {out_surface}")
    print(f"  {out_smoke}")
    print(f"\nTo finetune, run:")
    print(f"  python -W ignore train_finetune_thermal.py --eval "
          f"-s {dataset.source_path} -m {model_path} "
          f"--use_thermal --depths {dataset.depths} "
          f"--start_checkpoint {out_surface}")


if __name__ == "__main__":
    main()