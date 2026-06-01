"""
SmokeSeer Metrics Evaluation
Computes PSNR, SSIM, LPIPS between desmoked renders and GT desmoked images,
and measures rendering speed (ms/frame, FPS).

Image index convention in test_results_individual/:
  _00000 : gt_desmoke  (ground truth smoke-free)
  _00001 : gt          (ground truth with smoke)
  _00002 : thermal_gt
  _00003 : rendering   (surface + smoke combined)
  _00004 : render_surface  ← desmoked prediction
  _00005 : render_smoke

Usage:
  python metrics.py -m ./output/red_sub/baseline
  python metrics.py -m ./output/red_sub/baseline ./output/red_sub/exp1_grad0.0003
  python metrics.py -m ./output/red_sub/baseline --skip_quality   # speed only
  python metrics.py -m ./output/red_sub/baseline --n_warmup 10 --n_measure 100
"""

import os
import sys
import json
import time
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


IDX_GT_DESMOKE     = "00000"  # ground truth smoke-free
IDX_GT_SMOKE       = "00001"  # ground truth with smoke (input)
IDX_RENDER_FULL    = "00003"  # surface + smoke render
IDX_RENDER_SURFACE = "00004"  # desmoked render  ← main metric
IDX_RENDER_SMOKE   = "00005"  # smoke-only render


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def load_image(path: Path) -> torch.Tensor:
    """Load PNG as (1, 3, H, W) float32 CUDA tensor."""
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3].cuda()


def collect_pairs(individual_dir: Path, pred_idx: str, gt_idx: str):
    """Return sorted list of (pred_path, gt_path, frame_id) tuples."""
    pairs = []
    for fname in sorted(os.listdir(individual_dir)):
        if not fname.endswith(f"_{pred_idx}.png"):
            continue
        frame_id = fname.replace(f"_{pred_idx}.png", "")
        gt_fname = f"{frame_id}_{gt_idx}.png"
        gt_path  = individual_dir / gt_fname
        if gt_path.exists():
            pairs.append((individual_dir / fname, gt_path, frame_id))
    return pairs


def compute_metrics(pairs):
    """Compute per-frame and average PSNR/SSIM/LPIPS."""
    per_frame = {}
    psnrs, ssims, lpipss = [], [], []

    for pred_path, gt_path, frame_id in tqdm(pairs, desc="  Computing metrics"):
        pred = load_image(pred_path)
        gt   = load_image(gt_path)

        p = psnr(pred, gt).item()
        s = ssim(pred, gt).item()
        l = lpips(pred, gt, net_type='vgg').item()

        per_frame[frame_id] = {"PSNR": p, "SSIM": s, "LPIPS": l}
        psnrs.append(p);  ssims.append(s);  lpipss.append(l)

    avg = {
        "PSNR":  sum(psnrs)  / len(psnrs),
        "SSIM":  sum(ssims)  / len(ssims),
        "LPIPS": sum(lpipss) / len(lpipss),
        "n_frames": len(pairs),
    }
    return avg, per_frame


# ---------------------------------------------------------------------------
# Rendering speed
# ---------------------------------------------------------------------------

def measure_render_speed(model_path: Path, n_warmup: int = 10, n_measure: int = 100):
    """
    Load checkpoint and measure rendering speed (surface-only desmoked render).
    Returns dict with ms_per_frame, fps, n_surface_gaussians, n_smoke_gaussians.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from scene import Scene, DeformModel, GaussianSmokeThermalModel, GaussianSurfaceThermalModel
    from gaussian_renderer.gsplat_render import render_surface_smoke
    from arguments import PipelineParams

    print("\n[3] Rendering speed measurement")

    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        print("  cfg_args not found, skipping.")
        return None

    with open(cfg_path) as f:
        dataset = eval(f.read())

    ckpt_surface = model_path / "ft_chkpnt_surface_thermal30000.pth"
    ckpt_smoke   = model_path / "ft_chkpnt_smoke_thermal30000.pth"
    if not ckpt_surface.exists() or not ckpt_smoke.exists():
        print("  Checkpoints not found, skipping.")
        return None

    gaussians_surface = GaussianSurfaceThermalModel(dataset.sh_degree_stage1)
    gaussians_smoke   = GaussianSmokeThermalModel(dataset.sh_degree_stage2)
    scene = Scene(dataset, gaussians_surface, shuffle=False)

    (params_s, _) = torch.load(ckpt_surface, map_location="cuda")
    gaussians_surface.restore(params_s, dataset)
    (params_k, _) = torch.load(ckpt_smoke, map_location="cuda")
    gaussians_smoke.restore(params_k, dataset)

    deform = DeformModel(True, False)
    ft_path = str(model_path.parent / (model_path.name + "_ft_thermal"))
    if not os.path.exists(os.path.join(ft_path, "deform")):
        ft_path = str(model_path / "_ft_thermal")
    deform.load_weights(ft_path)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    parser_  = ArgumentParser()
    pp       = PipelineParams(parser_)
    pipe     = pp.extract(dataset)

    cameras  = scene.getTrainCamerasUnshuffled()
    n_cams   = len(cameras)
    if n_cams == 0:
        print("  No cameras found.")
        return None

    n_surface = gaussians_surface.get_xyz.shape[0]
    n_smoke   = gaussians_smoke.get_xyz.shape[0]
    print(f"  Surface Gaussians : {n_surface:,}")
    print(f"  Smoke   Gaussians : {n_smoke:,}")
    print(f"  Warmup: {n_warmup} frames  |  Measure: {n_measure} frames")

    def _render_one(cam):
        N = gaussians_smoke.get_xyz.shape[0]
        time_input = cam.fid.cuda().float().unsqueeze(0).expand(N, -1)
        d_xyz, d_rot, d_scl, d_op, d_col = deform.step(
            gaussians_smoke.get_xyz.detach(), time_input)
        deform_params = [d_xyz, d_rot, d_scl, d_op, d_col]
        with torch.no_grad():
            render_surface_smoke(cam, gaussians_surface, gaussians_smoke,
                                 pipe, bg_color, deform_parameters=deform_params)

    # Warmup
    torch.cuda.synchronize()
    for i in range(n_warmup):
        _render_one(cameras[i % n_cams])
    torch.cuda.synchronize()

    # Measure
    times = []
    for i in tqdm(range(n_measure), desc="  Measuring"):
        cam = cameras[i % n_cams]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _render_one(cam)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    ms_mean = sum(times) / len(times)
    ms_min  = min(times)
    ms_max  = max(times)
    fps     = 1000.0 / ms_mean

    print(f"  Avg : {ms_mean:.2f} ms/frame  ({fps:.2f} FPS)")
    print(f"  Min : {ms_min:.2f} ms  |  Max : {ms_max:.2f} ms")

    del gaussians_surface, gaussians_smoke, scene, deform
    torch.cuda.empty_cache()

    return {
        "ms_per_frame_avg": ms_mean,
        "ms_per_frame_min": ms_min,
        "ms_per_frame_max": ms_max,
        "fps": fps,
        "n_surface_gaussians": n_surface,
        "n_smoke_gaussians": n_smoke,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(model_paths, skip_quality=False, skip_speed=False,
             n_warmup=10, n_measure=100):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    all_results = {}

    for model_path in model_paths:
        model_path = Path(model_path)
        individual_dir = model_path / "test_results_individual"

        print(f"\n{'='*60}")
        print(f"Scene: {model_path}")
        print(f"{'='*60}")

        # Load existing results to avoid recomputing quality if already done
        out_path = model_path / "metrics.json"
        if out_path.exists():
            with open(out_path) as f:
                results = json.load(f)
        else:
            results = {}

        # --- Quality metrics ---
        if not skip_quality:
            if not individual_dir.exists():
                print(f"[SKIP quality] test_results_individual not found.")
            else:
                print("\n[1] Desmoked render  vs  GT desmoked")
                pairs = collect_pairs(individual_dir, IDX_RENDER_SURFACE, IDX_GT_DESMOKE)
                if pairs:
                    avg, per_frame = compute_metrics(pairs)
                    print(f"  PSNR  : {avg['PSNR']:.4f}")
                    print(f"  SSIM  : {avg['SSIM']:.4f}")
                    print(f"  LPIPS : {avg['LPIPS']:.4f}")
                    print(f"  Frames: {avg['n_frames']}")
                    results["desmoked_vs_gt_desmoked"] = {"avg": avg, "per_frame": per_frame}

                print("\n[2] Full render  vs  GT with smoke")
                pairs_full = collect_pairs(individual_dir, IDX_RENDER_FULL, IDX_GT_SMOKE)
                if pairs_full:
                    avg_full, per_frame_full = compute_metrics(pairs_full)
                    print(f"  PSNR  : {avg_full['PSNR']:.4f}")
                    print(f"  SSIM  : {avg_full['SSIM']:.4f}")
                    print(f"  LPIPS : {avg_full['LPIPS']:.4f}")
                    results["full_render_vs_gt_smoke"] = {"avg": avg_full, "per_frame": per_frame_full}

        # --- Rendering speed ---
        if not skip_speed:
            speed = measure_render_speed(model_path, n_warmup=n_warmup, n_measure=n_measure)
            if speed:
                results["render_speed"] = speed

        # --- Save ---
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

        all_results[str(model_path)] = results

    return all_results


if __name__ == "__main__":
    parser = ArgumentParser(description="SmokeSeer metrics evaluation")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--skip_quality", action="store_true",
                        help="Skip PSNR/SSIM/LPIPS (speed only)")
    parser.add_argument("--skip_speed", action="store_true",
                        help="Skip rendering speed measurement")
    parser.add_argument("--n_warmup",  type=int, default=10,
                        help="Warmup frames before timing (default: 10)")
    parser.add_argument("--n_measure", type=int, default=100,
                        help="Frames to measure for speed (default: 100)")
    args = parser.parse_args()
    evaluate(args.model_paths,
             skip_quality=args.skip_quality,
             skip_speed=args.skip_speed,
             n_warmup=args.n_warmup,
             n_measure=args.n_measure)