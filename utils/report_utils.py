import torch
import random
from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim

# ---------------------------------------------------------------------------
# wandb is optional — only imported when use_wandb=True
# ---------------------------------------------------------------------------
_wandb = None

def _get_wandb():
    global _wandb
    if _wandb is None:
        import wandb
        _wandb = wandb
    return _wandb


def _wlog(data, step, use_wandb):
    if use_wandb:
        _get_wandb().log(data, step=step)


def _wimage(tensor, use_wandb):
    if use_wandb:
        return _get_wandb().Image(tensor)
    return None


def normalize_depth(inv_depth):
    percentile_5  = torch.quantile(inv_depth, 0.05)
    percentile_95 = torch.quantile(inv_depth, 0.95)
    inv_depth = torch.clamp(inv_depth, percentile_5, percentile_95)
    return (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min())


# ---------------------------------------------------------------------------
# Stage 1 report
# ---------------------------------------------------------------------------

@torch.no_grad()
def training_report_stage1_thermal(
    losses_log, iteration, Ll1, loss, l1_loss, elapsed,
    testing_iterations, scene, gaussians_surface,
    renderFunc1, renderArgs,
    deform=None, use_thermal=False, use_wandb=False,
):
    _wlog(losses_log, iteration, use_wandb)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    print("Number of gaussians in surface: {}".format(gaussians_surface.get_xyz.shape[0]))

    configs = [
        {'name': 'test',  'cameras': scene.getTestCamerasThermal()},
        {'name': 'train', 'cameras': scene.getTrainCamerasThermal()},
    ]

    for config in configs:
        if not config['cameras']:
            continue

        l1_test   = 0.0
        psnr_test = 0.0

        for idx, viewpoint in enumerate(config['cameras']):
            pkg = renderFunc1(viewpoint, gaussians_surface, *renderArgs, is_thermal=True)
            thermal_image = pkg["render"]
            if viewpoint.alpha_mask is not None:
                thermal_image *= viewpoint.alpha_mask.cuda()
            thermal_inv_depth = pkg["inv_depth"]
            gt_thermal_image  = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if use_wandb and idx < 5:
                w = _get_wandb()
                _wlog({config['name'] + "_view_{}/render_thermal".format(viewpoint.image_name):
                       [w.Image(thermal_image)]}, iteration, use_wandb)
                _wlog({config['name'] + "_view_{}/ground_truth_thermal".format(viewpoint.image_name):
                       [w.Image(gt_thermal_image)]}, iteration, use_wandb)
                _wlog({config['name'] + "_view_{}/inv_depth_thermal".format(viewpoint.image_name):
                       [w.Image(thermal_inv_depth.unsqueeze(0))]}, iteration, use_wandb)

            l1_test   += l1_loss(thermal_image, gt_thermal_image).mean().double()
            psnr_test += psnr(thermal_image, gt_thermal_image).mean().double()

        n = len(config['cameras'])
        psnr_test /= n
        l1_test   /= n
        print("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f}".format(
            iteration, config['name'], l1_test, psnr_test))

        _wlog({
            config['name'] + '/loss_viewpoint - l1_loss_thermal': l1_test,
            config['name'] + '/loss_viewpoint - psnr_thermal':    psnr_test,
        }, iteration, use_wandb)

    if use_wandb:
        w = _get_wandb()
        try:
            _wlog({"scene/opacity_histogram_surface":
                   w.Histogram(gaussians_surface.get_opacity.cpu())}, iteration, use_wandb)
        except Exception:
            print(gaussians_surface.get_opacity.min(), gaussians_surface.get_opacity.max())
    _wlog({'total_points_surface': gaussians_surface.get_xyz.shape[0]}, iteration, use_wandb)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2 report
# ---------------------------------------------------------------------------

@torch.no_grad()
def training_report_s2(
    losses_log, iteration, Ll1, loss, l1_loss, elapsed,
    testing_iterations, scene, gaussians_smoke, gaussians_surface,
    renderFunc1, renderFunc2, renderArgs,
    deform=None, use_thermal=False, use_wandb=False,
):
    if iteration % 5 == 0:
        _wlog(losses_log, iteration, use_wandb)

    if iteration not in testing_iterations:
        return

    lpips_module = LPIPS(net_type='vgg').eval().to("cuda")

    print("Number of gaussians in surface: {}".format(gaussians_surface.get_xyz.shape[0]))
    print("Number of gaussians in smoke:   {}".format(gaussians_smoke.get_xyz.shape[0]))

    configs = [
        {'name': 'test',  'cameras': scene.getTestCamerasUnshuffled()},
        {'name': 'train', 'cameras': scene.getTrainCamerasUnshuffled()},
    ]
    thermal_configs = None
    if use_thermal:
        thermal_configs = [
            {'name': 'test',  'cameras': scene.getTestCamerasThermalUnshuffled()},
            {'name': 'train', 'cameras': scene.getTrainCamerasThermalUnshuffled()},
        ]

    random.seed(40)

    for idx_config, config in enumerate(configs):
        if not config['cameras']:
            continue

        l1_test        = 0.0
        psnr_test      = 0.0
        psnr_test_gt   = 0.0
        lpips_test_gt  = 0.0
        ssim_test_gt   = 0.0
        psnr_test_thermal = 0.0

        thermal_cameras  = thermal_configs[idx_config]['cameras'] if use_thermal else []
        random_indices   = set(random.sample(range(len(config['cameras'])),
                                             min(40, len(config['cameras']))))

        for idx, viewpoint in enumerate(config['cameras']):
            try:
                if iteration < 3000:
                    d_xyz = d_rotation = d_scaling = d_opacity = d_color = 0.0
                else:
                    N = gaussians_smoke.get_xyz.shape[0]
                    time_input = viewpoint.fid.unsqueeze(0).expand(N, -1)
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = \
                        deform.step(gaussians_smoke.get_xyz.detach(), time_input)

                deform_params = [d_xyz, d_rotation, d_scaling, d_opacity, d_color]

                # --- render ---
                pkg          = renderFunc2(viewpoint, gaussians_surface, gaussians_smoke,
                                           deform_parameters=deform_params, *renderArgs)
                image        = pkg["render"]
                depth_all    = normalize_depth(pkg["inv_depth"])

                pkg_surface  = renderFunc1(viewpoint, gaussians_surface, *renderArgs)
                image_desmoked   = pkg_surface["render"]
                inv_depth_surface = pkg_surface["inv_depth"]

                pkg_smoke    = renderFunc1(viewpoint, gaussians_smoke, *renderArgs,
                                           deform_parameters=deform_params)
                image_smoke      = pkg_smoke["render"]
                inv_depth_smoke  = pkg_smoke["inv_depth"]

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                gt_image_desmoked = torch.clamp(
                    (viewpoint.original_image_desmoked
                     if viewpoint.original_image_desmoked is not None
                     else viewpoint.original_image).to("cuda"), 0.0, 1.0
                ).float()

                # --- wandb image logging (sampled views only) ---
                if use_wandb and idx in random_indices:
                    w = _get_wandb()
                    _wlog({
                        config['name'] + "_view_{}/render".format(viewpoint.image_name):         [w.Image(image)],
                        config['name'] + "_view_{}/render_desmoked".format(viewpoint.image_name):[w.Image(image_desmoked)],
                        config['name'] + "_view_{}/render_smoke".format(viewpoint.image_name):   [w.Image(image_smoke)],
                    }, iteration, use_wandb)
                    _wlog({
                        config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name):          [w.Image(gt_image)],
                        config['name'] + "_view_{}/ground_truth_desmoked".format(viewpoint.image_name): [w.Image(gt_image_desmoked)],
                    }, iteration, use_wandb)
                    _wlog({config['name'] + "_view_{}/inv_depth_all".format(viewpoint.image_name):     [w.Image(depth_all)]},             iteration, use_wandb)
                    _wlog({config['name'] + "_view_{}/inv_depth_surface".format(viewpoint.image_name):[w.Image(inv_depth_surface)]},      iteration, use_wandb)
                    _wlog({config['name'] + "_view_{}/inv_depth_smoke".format(viewpoint.image_name):  [w.Image(inv_depth_smoke)]},        iteration, use_wandb)

                    if viewpoint.smoke_mask is not None:
                        _wlog({config['name'] + "_view_{}/gt_with_smoke_mask".format(viewpoint.image_name):
                               [w.Image(gt_image * viewpoint.smoke_mask.cuda())]}, iteration, use_wandb)

                    if use_thermal and idx < len(thermal_cameras):
                        tv = thermal_cameras[idx]
                        t_both    = renderFunc2(tv, gaussians_surface, gaussians_smoke,
                                                deform_parameters=deform_params, *renderArgs, is_thermal=True)["render"].clamp(0, 1)
                        t_surface = renderFunc1(tv, gaussians_surface, *renderArgs, is_thermal=True)["render"].clamp(0, 1)
                        t_smoke   = renderFunc1(tv, gaussians_smoke, *renderArgs,
                                                deform_parameters=deform_params, is_thermal=True)["render"].clamp(0, 1)
                        gt_t      = tv.original_image.to("cuda").clamp(0, 1)
                        _wlog({
                            config['name'] + "_view_{}/render_thermal_both".format(viewpoint.image_name):    [w.Image(t_both)],
                            config['name'] + "_view_{}/render_thermal_surface".format(viewpoint.image_name): [w.Image(t_surface)],
                            config['name'] + "_view_{}/render_thermal_smoke".format(viewpoint.image_name):   [w.Image(t_smoke)],
                            config['name'] + "_view_{}/ground_truth_thermal".format(viewpoint.image_name):   [w.Image(gt_t)],
                        }, iteration, use_wandb)
                        psnr_test_thermal += psnr(t_surface, gt_t).mean().double()

            except Exception as e:
                print(e)
                continue

            l1_test       += l1_loss(image, gt_image).mean().double()
            psnr_test     += psnr(image, gt_image).mean().double()
            psnr_test_gt  += psnr(image_desmoked, gt_image_desmoked).mean().double()
            lpips_test_gt += lpips_module(image_desmoked, gt_image_desmoked).mean().double()
            ssim_test_gt  += ssim(image_desmoked, gt_image_desmoked).mean().double()

        n = len(config['cameras'])
        psnr_test     /= n
        psnr_test_gt  /= n
        l1_test       /= n
        lpips_test_gt /= n
        ssim_test_gt  /= n

        print("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} PSNR_GT {:.4f} LPIPS {:.4f} SSIM {:.4f}".format(
            iteration, config['name'], l1_test, psnr_test, psnr_test_gt, lpips_test_gt, ssim_test_gt))

        _wlog({
            config['name'] + '/loss_viewpoint - l1_loss':   l1_test,
            config['name'] + '/loss_viewpoint - psnr':      psnr_test,
            config['name'] + '/loss_viewpoint - psnr_gt':   psnr_test_gt,
            config['name'] + '/loss_viewpoint - lpips_gt':  lpips_test_gt,
            config['name'] + '/loss_viewpoint - ssim_gt':   ssim_test_gt,
        }, iteration, use_wandb)

        if use_thermal:
            psnr_test_thermal /= n
            _wlog({config['name'] + '/loss_viewpoint - psnr_thermal': psnr_test_thermal},
                  iteration, use_wandb)

    if use_wandb:
        w = _get_wandb()
        try:
            _wlog({"scene/opacity_histogram_surface": w.Histogram(gaussians_surface.get_opacity.cpu()),
                   "scene/opacity_histogram_smoke":   w.Histogram(gaussians_smoke.get_opacity.cpu())},
                  iteration, use_wandb)
        except Exception:
            print(gaussians_smoke._opacity.min(),   gaussians_smoke._opacity.max())
            print(gaussians_surface.get_opacity.min(), gaussians_surface.get_opacity.max())

    _wlog({
        'total_points_smoke':   gaussians_smoke.get_xyz.shape[0],
        'total_points_surface': gaussians_surface.get_xyz.shape[0],
    }, iteration, use_wandb)
    torch.cuda.empty_cache()