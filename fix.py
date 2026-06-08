import sys

path = '/home/moa/SmokeSeer/scene/gaussian_model_thermal.py'
with open(path, 'r') as f:
    content = f.read()

original_len = len(content)

# 1. Smoke prune_points: direct mask indexing
old = '''        self._opacity = optimizable_tensors["opacity"]
        self._opacity_thermal = optimizable_tensors["opacity_thermal"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._mask = optimizable_tensors["mask"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]'''
new = '''        self._opacity = optimizable_tensors["opacity"]
        self._opacity_thermal = optimizable_tensors["opacity_thermal"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = nn.Parameter(self._mask[valid_points_mask].detach().requires_grad_(True))
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]'''
n = content.count(old); content = content.replace(old, new)
print(f'[1] Smoke prune_points: {n} replacements')

# 2. Surface prune_points: direct mask indexing
old = '''        self._opacity = optimizable_tensors["opacity"]

        self._mask = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]'''
new = '''        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = nn.Parameter(self._mask[valid_points_mask].detach().requires_grad_(True))
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]'''
n = content.count(old); content = content.replace(old, new)
print(f'[2] Surface prune_points: {n} replacements')

# 3. Smoke densification_postfix: direct mask cat
old = '''        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._mask = nn.Parameter(torch.cat([self._mask, torch.ones((self.get_xyz.shape[0] - self._mask.shape[0], 1), device="cuda")], dim=0).requires_grad_(True))'''
new = '''        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._mask = nn.Parameter(torch.cat([self._mask, torch.ones((new_xyz.shape[0], 1), device="cuda")], dim=0).detach().requires_grad_(True))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")'''
n = content.count(old); content = content.replace(old, new)
print(f'[3] Smoke densification_postfix: {n} replacements')

# 4. Surface densification_postfix: direct mask cat
old = '''        self._opacity = optimizable_tensors["opacity"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._mask = optimizable_tensors["mask"]
        self._features_thermal_dc = optimizable_tensors["f_thermal_dc"]
        self._features_thermal_rest = optimizable_tensors["f_thermal_rest"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._mask = nn.Parameter(torch.cat([self._mask, torch.ones((self.get_xyz.shape[0] - self._mask.shape[0], 1), device="cuda")], dim=0).requires_grad_(True))'''
new = '''        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_thermal_dc = optimizable_tensors["f_thermal_dc"]
        self._features_thermal_rest = optimizable_tensors["f_thermal_rest"]
        self._mask = nn.Parameter(torch.cat([self._mask, torch.ones((new_xyz.shape[0], 1), device="cuda")], dim=0).detach().requires_grad_(True))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")'''
n = content.count(old); content = content.replace(old, new)
print(f'[4] Surface densification_postfix: {n} replacements')

# 5. Fix mask_prune double prune_points call
old = '''    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze(-1)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        self.prune_points(prune_mask)'''
new = '''    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze(-1)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()'''
n = content.count(old); content = content.replace(old, new)
print(f'[5] mask_prune double call: {n} replacements')

# 6. cat_tensors_to_optimizer: skip missing keys (so mask key not needed in d)
old = '            extension_tensor = tensors_dict[group["name"]]'
new = '            if group["name"] not in tensors_dict:\n                continue\n            extension_tensor = tensors_dict[group["name"]]'
n = content.count(old); content = content.replace(old, new)
print(f'[6] cat_tensors skip missing: {n} replacements')

with open(path, 'w') as f:
    f.write(content)

print(f'\nDone. Size: {original_len} -> {len(content)} chars')