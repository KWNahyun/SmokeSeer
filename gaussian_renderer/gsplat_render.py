# Refactored SmokeSeer renderer using gsplat backend
# Replaces both gaussian_renderer/__init__.py (diff-gaussian-rasterization) 
# and the original gsplat_render.py with a clean, unified implementation.

import math
import torch
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_thermal import GaussianSurfaceThermalModel, GaussianSmokeThermalModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_K(viewpoint_camera) -> torch.Tensor:
    """Build intrinsic matrix K from camera FoV."""
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = viewpoint_camera.image_width / (2 * tanfovx)
    fy = viewpoint_camera.image_height / (2 * tanfovy)
    cx = viewpoint_camera.image_width / 2.0
    cy = viewpoint_camera.image_height / 2.0
    return torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        device="cuda",
    )


def _get_fid_cuda(viewpoint_camera) -> torch.Tensor:
    """Return fid as a CUDA scalar tensor."""
    fid = viewpoint_camera.fid
    if isinstance(fid, torch.Tensor):
        return fid.cuda()
    return torch.tensor(fid, device="cuda")


def _rasterize(
    means3D, rotations, scales, opacity, colors,
    viewpoint_camera, bg_color, sh_degree,
    render_mode="RGB+ED",
):
    """Low-level gsplat rasterization call. Returns (image_depth, radii, info)."""
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    K = _build_K(viewpoint_camera)
    render_colors, render_alphas, info = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    radii = info["radii"].squeeze(0).max(dim=-1).values  # (N,)
    try:
        info["means2d"].retain_grad()
    except Exception:
        pass
    render_alphas = render_alphas.squeeze(0).permute(2, 0, 1)
    return render_colors[0], radii, render_alphas, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color=None,
    deform_parameters=None,
    is_thermal: bool = False,
):
    """Render a single Gaussian model (surface OR smoke).

    Returns dict with keys:
        render, viewspace_points, visibility_filter, radii, inv_depth
    """
    if deform_parameters is not None:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D   = pc.get_xyz + d_xyz
        scales    = pc.scaling_activation(pc._scaling + d_scaling) * scaling_modifier
        rotations = pc.rotation_activation(pc._rotation + d_rotation)
        opacity   = pc.get_opacity_thermal if is_thermal else pc.get_opacity_at_t(_get_fid_cuda(viewpoint_camera))
        colors    = pc.get_features_thermal if is_thermal else pc.get_features
    else:
        means3D   = pc.get_xyz
        scales    = pc.get_scaling * scaling_modifier
        rotations = pc.get_rotation
        opacity   = pc.get_opacity_thermal if is_thermal else pc.get_opacity
        colors    = pc.get_features_thermal if is_thermal else pc.get_features

    if override_color is not None:
        colors    = override_color
        sh_degree = None
    else:
        sh_degree = pc.active_sh_degree

    image_depth, radii, render_alphas, info = _rasterize(
        means3D, rotations, scales, opacity, colors,
        viewpoint_camera, bg_color, sh_degree, render_mode="RGB+ED",
    )

    rendered_image = image_depth[:, :, :3].permute(2, 0, 1)
    depth          = image_depth[:, :, 3]
    inv_depth      = 1.0 / (depth + 1e-6)

    return {
        "render":             rendered_image.clamp(0, 1),
        "viewspace_points":   info["means2d"],
        "visibility_filter":  radii > 0,
        "radii":              radii,
        "inv_depth":          inv_depth,
        "render_alphas":      render_alphas,
    }


def render_surface_only(
    viewpoint_camera,
    pc1: GaussianSurfaceThermalModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    is_thermal: bool = False,
):
    """Render surface Gaussians only (smoke removed).

    Useful for evaluating desmoked quality independently.
    Returns dict with keys: render, visibility_filter, radii, inv_depth
    """
    means3D   = pc1.get_xyz
    scales    = pc1.get_scaling * scaling_modifier
    rotations = pc1.get_rotation
    opacity   = pc1.get_opacity_thermal if is_thermal else pc1.get_opacity
    colors    = pc1.get_features_thermal if is_thermal else pc1.get_features

    image_depth, radii, render_alphas, info = _rasterize(
        means3D, rotations, scales, opacity, colors,
        viewpoint_camera, bg_color, pc1.active_sh_degree, render_mode="RGB+ED",
    )

    rendered_image = image_depth[:, :, :3].permute(2, 0, 1)
    inv_depth      = 1.0 / (image_depth[:, :, 3] + 1e-6)

    return {
        "render":            rendered_image.clamp(0, 1),
        "visibility_filter": radii > 0,
        "radii":             radii,
        "inv_depth":         inv_depth,
    }


def render_surface_smoke(
    viewpoint_camera,
    pc1: GaussianSurfaceThermalModel,
    pc2: GaussianSmokeThermalModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color=None,
    deform_parameters=None,
    scale_divide: float = 1.0,
    is_thermal: bool = False,
):
    """Render surface + smoke Gaussians together.

    Returns dict with keys:
        render, visibility_filter_surface, radii_surface,
        visibility_filter_smoke, radii_smoke,
        viewspace_points, inv_depth
    """
    fid = _get_fid_cuda(viewpoint_camera)

    if deform_parameters is not None:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D   = torch.cat([pc1.get_xyz, pc2.get_xyz + d_xyz], dim=0)
        scales    = torch.cat([
            pc1.get_scaling,
            pc2.scaling_activation(pc2._scaling + d_scaling),
        ], dim=0) * scaling_modifier
        rotations = torch.cat([
            pc1.get_rotation,
            pc2.rotation_activation(pc2._rotation + d_rotation),
        ], dim=0)
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal, pc2.get_opacity_thermal], dim=0)
            colors  = torch.cat([pc1.get_features_thermal, pc2.get_features_thermal], dim=0)
        else:
            opacity = torch.cat([pc1.get_opacity, pc2.get_opacity_at_t(fid)], dim=0)
            colors  = torch.cat([pc1.get_features, pc2.get_features], dim=0)
    else:
        means3D   = torch.cat([pc1.get_xyz, pc2.get_xyz], dim=0)
        scales    = torch.cat([pc1.get_scaling, pc2.get_scaling], dim=0) * scaling_modifier
        rotations = torch.cat([pc1.get_rotation, pc2.get_rotation], dim=0)
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal, pc2.get_opacity_thermal], dim=0)
            colors  = torch.cat([pc1.get_features_thermal, pc2.get_features_thermal], dim=0)
        else:
            opacity = torch.cat([pc1.get_opacity, pc2.get_opacity], dim=0)
            colors  = torch.cat([pc1.get_features, pc2.get_features], dim=0)

    image_depth, radii, render_alphas, info = _rasterize(
        means3D, rotations, scales, opacity, colors,
        viewpoint_camera, bg_color, pc1.active_sh_degree, render_mode="RGB+D",
    )

    n_surface      = pc1.get_xyz.shape[0]
    rendered_image = image_depth[:, :, :3].permute(2, 0, 1)
    inv_depth      = 1.0 / (image_depth[:, :, 3] + 1e-6)

    return {
        "render":                   rendered_image.clamp(0, 1),
        "visibility_filter_surface": radii[:n_surface] > 0,
        "radii_surface":             radii[:n_surface],
        "visibility_filter_smoke":   radii[n_surface:] > 0,
        "radii_smoke":               radii[n_surface:],
        "viewspace_points":          info["means2d"],
        "inv_depth":                 inv_depth,
        "render_alphas":             render_alphas,
    }


# ---------------------------------------------------------------------------
# Mini-Splatting: importance score & depth rendering
# (uses diff_gaussian_rasterization_ms CUDA backend)
# ---------------------------------------------------------------------------

def _ms_raster_settings(viewpoint_camera, pc, pipe):
    """Build GaussianRasterizationSettings for _ms rasterizer."""
    import math as _math
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings
    tanfovx = _math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = _math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )


@torch.no_grad()
def render_imp(viewpoint_camera, pc, pipe):
    """Render with importance scores (accum_weights, area_proj, area_max)."""
    from diff_gaussian_rasterization_ms import GaussianRasterizer
    raster_settings = _ms_raster_settings(viewpoint_camera, pc, pipe)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda")

    rendered_image, radii, accum_weights, area_proj, area_max = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return {
        "render":         rendered_image,
        "radii":          radii,
        "accum_weights":  accum_weights,
        "area_proj":      area_proj,
        "area_max":       area_max,
    }


@torch.no_grad()
def render_depth(viewpoint_camera, pc, pipe):
    """Render depth map for depth reinitialization."""
    from diff_gaussian_rasterization_ms import GaussianRasterizer
    raster_settings = _ms_raster_settings(viewpoint_camera, pc, pipe)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda")

    res = rasterizer.render_depth(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return res


# ---------------------------------------------------------------------------
# LightGaussian: importance score rendering
# (uses compress-diff-gaussian-rasterization CUDA backend)
# ---------------------------------------------------------------------------

@torch.no_grad()
def count_render(viewpoint_camera, pc, pipe):
    """Render with importance score for LightGaussian pruning."""
    import math as _math, sys as _sys, os as _os
    _sys.path.insert(0, _os.path.expanduser("~/LightGaussian/submodules/compress-diff-gaussian-rasterization"))
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)

    tanfovx = _math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = _math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        f_count=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    gaussians_count, important_score, rendered_image, radii = rasterizer(
        means3D=pc.get_xyz,
        means2D=torch.zeros_like(pc.get_xyz, requires_grad=False),
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return {
        "gaussians_count":  gaussians_count,
        "important_score":  important_score,
        "render":           rendered_image,
        "radii":            radii,
    }
