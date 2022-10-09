import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from torch_scatter import scatter_add, segment_coo

from .devrf import Raw2Alpha, Alphas2Weights, render_utils_cuda, total_variation_cuda, MaskCache


'''Model'''
class DeMPIRF(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0, mpi_depth_motion=0,
                 num_voxels_motion=0, motion_dim=0, timesteps=0, warp_ray=False, world_motion_bound_scale=1.0,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 **kwargs):
        super(DeMPIRF, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.world_motion_bound_scale = world_motion_bound_scale
        self.fast_color_thres = fast_color_thres
        self.act_shift = 0

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)
        self._set_grid_resolution_motion(num_voxels_motion, mpi_depth_motion)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]), requires_grad=False)
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.density[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))
            self.density[..., -1].fill_(10)

        # init motion voxel grid
        self.motion_list = []
        self.motion_dim = motion_dim
        self.timesteps = timesteps      
        self.warp_ray = warp_ray  
        for i in range(self.timesteps):
            init_motion = torch.zeros([1, self.motion_dim, *self.motion_size])
            if self.motion_dim == 9:
                init_motion[:, 0, :, :, :] = 1.0
                init_motion[:, 4, :, :, :] = 1.0
            elif self.motion_dim == 7:
                init_motion[:, 0, :, :, :] = 1.0   
            self.motion_list.append(torch.nn.Parameter(init_motion, requires_grad=False).cpu())     
        # init fwd motion voxel grid for cycle consistency loss, dont need to store it in checkpoint.
        self.fwdmotion_list = []
        for i in range(self.timesteps):
            init_motion = torch.zeros([1, self.motion_dim, *self.motion_size])
            if self.motion_dim == 9:
                init_motion[:, 0, :, :, :] = 1.0
                init_motion[:, 4, :, :, :] = 1.0
            elif self.motion_dim == 7:
                init_motion[:, 0, :, :, :] = 1.0   
            self.fwdmotion_list.append(torch.nn.Parameter(init_motion, requires_grad=False).cpu())       

        if self.timesteps > 1:
            self.timestep = 1.0 / (self.timesteps - 1)
        else:
            self.timestep = 1.0  

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)

        print('dempirf: self.density.shape', self.density.shape)
        print('dempirf: self.k0.shape', self.k0.shape)
        print('dempirf: mlp', self.rgbnet)

        self.mask_cache = None
        self.nonempty_mask = None         

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dempirf: world_size      ', self.world_size)
        print('dempirf: voxel_size_ratio', self.voxel_size_ratio)

    def _set_grid_resolution_motion(self, num_voxels_motion, mpi_depth_motion):
        # Determine grid resolution
        self.num_voxels_motion = num_voxels_motion
        self.mpi_depth_motion = mpi_depth_motion
        r = (num_voxels_motion / self.mpi_depth_motion / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.motion_size = torch.zeros(3, dtype=torch.long)
        self.motion_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.motion_size[2] = self.mpi_depth_motion
        self.motion_voxel_size_ratio = 128. / mpi_depth_motion
        print('dempirf: motion_size      ', self.motion_size)
        print('dempirf: motion_voxel_size_ratio', self.motion_voxel_size_ratio)        

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'mpi_depth_motion': self.mpi_depth_motion,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'num_voxels_motion': self.num_voxels_motion,
            'motion_dim': self.motion_dim,
            'timesteps': self.timesteps,  
            'warp_ray': self.warp_ray,
            'world_motion_bound_scale': self.world_motion_bound_scale,            
            'fast_color_thres': self.fast_color_thres,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dempirf: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dempirf: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))

        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        self_alpha = F.max_pool3d(self.activate_density(self.density), kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache = MaskCache(
                path=None, mask=(self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dempirf: scale_volume_grid finish')

    @torch.no_grad()
    def scale_volume_grid_motion(self, num_voxels_motion, mpi_depth_motion):
        print('dempirf: scale_volume_grid_motion start')
        ori_motion_size = self.motion_size
        self._set_grid_resolution_motion(num_voxels_motion, mpi_depth_motion)
        print('dempirf: scale_volume_grid_motion size from', ori_motion_size, 'to', self.motion_size)
        for i in range(self.timesteps):
            self.motion_list[i] = torch.nn.Parameter(
                F.interpolate(self.motion_list[i].data, size=tuple(self.motion_size), mode='trilinear', align_corners=True), requires_grad=False)  
            self.fwdmotion_list[i] = torch.nn.Parameter(
                F.interpolate(self.fwdmotion_list[i].data, size=tuple(self.motion_size), mode='trilinear', align_corners=True), requires_grad=False)
        print('dempirf: scale_volume_grid_motion finish')                  

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        total_variation_cuda.total_variation_add_grad(
            self.density, self.density.grad, wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        total_variation_cuda.total_variation_add_grad(
            self.k0, self.k0.grad, wxy, wxy, wz, dense_mode)

    def motion_total_variation_add_grad(self, weight, dense_mode, frame_time):
        weight = weight * self.motion_size.max() / 128
        time_id_ = np.round((frame_time / self.timestep).cpu(), decimals=2)
        time_id = time_id_.cuda()
        time_id_prev = time_id.floor().long()
        time_id_next = time_id.ceil().long()
        if time_id_next > 0:
            if time_id_prev == time_id_next and self.motion_list[time_id_next].requires_grad == True:
                total_variation_cuda.total_variation_add_grad(
                    self.motion_list[time_id_next], self.motion_list[time_id_next].grad, weight, weight, weight, dense_mode)                                   
                if self.fwdmotion_list[time_id_next].grad != None:
                    total_variation_cuda.total_variation_add_grad(
                        self.fwdmotion_list[time_id_next], self.fwdmotion_list[time_id_next].grad, weight, weight, weight, dense_mode)                       
            else:
                NotImplementedError
        else:
            return 0.0     

    def cycle_loss(self, ray_pts_ori, ray_pts, frame_time):

        if self.timesteps != 0 and frame_time != 0:
            time_id_ = np.round((frame_time / self.timestep).cpu(), decimals=2)
            time_id = time_id_.cuda()
            time_id_prev = time_id.floor().long()
            time_id_next = time_id.ceil().long()   
            if time_id_prev == time_id_next:
                if time_id_prev == 0:   
                    return 0
                else:
                    if self.motion_dim == 3:      
                        fwdoffset_3D = self.fwdmotion_list[time_id_prev]    # N C X Y Z
                        if self.warp_ray:
                            vox_fwdmotion = self.grid_sampler(ray_pts, fwdoffset_3D)
                            ray_pts_cycled = ray_pts + vox_fwdmotion
                            dis_pts = ray_pts_ori - ray_pts_cycled
                            cycle_loss = torch.mean(torch.sum(dis_pts**2, 1) / 2.0)   
                            return cycle_loss                           
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            return 0       

    def flow_loss(self, grid, pose, frame_time, hwf, weights, rays_pts, ray_id): 
        if self.timesteps != 0 and frame_time != 0:
            time_id_ = np.round((frame_time / self.timestep).cpu(), decimals=3)
            time_id = time_id_.cuda()
            time_id_prev = time_id.floor().long()
            time_id_next = time_id.ceil().long()  
            if time_id_prev == time_id_next:
                if time_id_prev == 0:
                    return 0.0
                else:
                    if time_id_next == 1:
                        pts_transformed_ndc = rays_pts
                    else:
                        fwdmotion_3D = self.fwdmotion_list[time_id_next-1]
                        rays_pts_fwdmotion = self.grid_sampler(rays_pts, fwdmotion_3D)
                        pts_transformed_ndc = rays_pts + rays_pts_fwdmotion      
                    H, W, f = hwf
                    pts_transformed = NDC2world(pts_transformed_ndc, H, W, f)
                    c2w = pose
                    w2c = c2w[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])     
                    # World coordinate to camera coordinate
                    # Translate
                    pts_bwd = pts_transformed - c2w[:3, 3]
                    # Rotate
                    pts_bwd_rotate = torch.sum(pts_bwd[..., None, :] * w2c[:3, :3], -1)
                    # Camera coordinate to 2D image coordinate
                    pts2d_bwd = torch.cat([pts_bwd_rotate[..., 0:1] / (- pts_bwd_rotate[..., 2:]) * f + W * .5,
                                        - pts_bwd_rotate[..., 1:2] / (- pts_bwd_rotate[..., 2:]) * f + H * .5],
                                        -1)   
                    grid_sampts = grid.flatten(0, -2)[ray_id]      
                    induced_flow = pts2d_bwd - grid_sampts[:, :2] - 0.5            
                    flow_loss = img2mae(induced_flow, grid_sampts[:, 2:4], weights, grid_sampts[:, -1].unsqueeze(-1))

                    return flow_loss                                                   
            else:
                raise NotImplementedError
        else:
            return 0.0                                      

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def grid_sampler(self, xyz, grid):
        '''Wrapper for the interp operation'''
        num_ch = grid.shape[1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        ret = ret.reshape(num_ch,-1).T.squeeze(1)
        return ret

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, frame_time, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_pts_ori = ray_pts

        # use absolute motion to canonical space and warp points.
        if self.timesteps != 0 and frame_time != 0:
            time_id_ = np.round((frame_time / self.timestep).cpu(), decimals=2)
            time_id = time_id_.cuda()
            time_id_prev = time_id.floor().long()
            time_id_next = time_id.ceil().long()   
            if time_id_prev == time_id_next:
                if time_id_prev == 0:
                    current_density = self.density
                    current_k0 = self.k0
                else:
                    if self.motion_dim == 7:
                        motion_7D = self.motion_list[time_id_prev]      # N C=7 X Y Z
                        # motion_7D = motion_7D.permute(0, 4, 1, 2, 3)    # N C=7 X Y Z
                        if self.warp_ray:
                            vox_motion = self.grid_sampler(ray_pts, motion_7D)
                            vox_rmtx = pytorch3d.transforms.quaternion_to_matrix(F.normalize(vox_motion[..., :4], dim=-1))      
                            vox_tran = vox_motion[..., 4:]
                            # get transformation mtx
                            proj_matrix = torch.cat((vox_rmtx, vox_tran[..., None]), dim=-1)
                            last_row = torch.zeros(*proj_matrix.shape[:-2], 1, 4)
                            last_row[..., -1] = 1.0        
                            transformation = torch.cat((proj_matrix, last_row), dim=-2)
                            pts_homo = torch.cat((ray_pts, torch.ones_like(ray_pts[..., :1])), dim=-1)
                            # transform pts.
                            pts_homo_transformed = transformation @ pts_homo[..., None]
                            pts_homo_transformed = pts_homo_transformed.squeeze(-1)
                            ray_pts = pts_homo_transformed[..., :3] / pts_homo_transformed[..., -1:]       

                        else:
                            original_pos = self.world_pos.permute(0, 2, 3, 4, 1) # N X Y Z 3
                            vox_motion = self.grid_sampler(original_pos, motion_7D)
                            vox_rmtx = pytorch3d.transforms.quaternion_to_matrix(F.normalize(vox_motion[..., :4], dim=-1))      
                            vox_tran = vox_motion[..., 4:]
                            # get transformation mtx
                            proj_matrix = torch.cat((vox_rmtx, vox_tran[..., None]), dim=-1)
                            last_row = torch.zeros(*proj_matrix.shape[:-2], 1, 4)
                            last_row[..., -1] = 1.0        
                            transformation = torch.cat((proj_matrix, last_row), dim=-2)
                            pts_homo = torch.cat((original_pos, torch.ones_like(original_pos[..., :1])), dim=-1)  
                            # transform pts.
                            pts_homo_transformed = transformation @ pts_homo[..., None]
                            pts_homo_transformed = pts_homo_transformed.squeeze(-1)
                            canonical_pos = pts_homo_transformed[..., :3] / pts_homo_transformed[..., -1:]             
                            
                    elif self.motion_dim == 3:      
                        offset_3D = self.motion_list[time_id_prev]    # N C X Y Z
                        # offset_3D = offset_3D.permute(0, 4, 1, 2, 3)    # N C X Y Z

                        if self.warp_ray:
                            vox_motion = self.grid_sampler(ray_pts, offset_3D)
                            ray_pts = ray_pts + vox_motion
                        else:
                            original_pos = self.world_pos.permute(0, 2, 3, 4, 1) # N X Y Z 3
                            vox_offset = self.grid_sampler(original_pos, offset_3D)
                            canonical_pos = original_pos + vox_offset

                    if self.warp_ray:
                        current_density = self.density
                        current_k0 = self.k0
                    else:
                        # get sampled property
                        current_density = self.grid_sampler(canonical_pos, self.density)
                        current_density = current_density.unsqueeze(0)
                        current_k0 = self.grid_sampler(canonical_pos, self.k0)
                        current_k0 = current_k0.permute(0, 4, 1, 2, 3)  # N C X Y Z          
            else:
                w_next = time_id - time_id_prev    
                w_prev = time_id_next - time_id
                if self.motion_dim == 3:
                    offset_3D = w_prev * self.motion_list[time_id_prev] + w_next * self.motion_list[time_id_next]
                    if self.warp_ray:
                        vox_motion = self.grid_sampler(ray_pts, offset_3D)
                        ray_pts = ray_pts + vox_motion
                    else:
                        original_pos = self.world_pos.permute(0, 2, 3, 4, 1) # N X Y Z 3
                        vox_offset = self.grid_sampler(original_pos, offset_3D)
                        canonical_pos = original_pos + vox_offset
                if self.warp_ray:
                    current_density = self.density
                    current_k0 = self.k0
                else:
                    # get sampled property
                    current_density = self.grid_sampler(canonical_pos, self.density)
                    current_density = current_density.unsqueeze(0)
                    current_k0 = self.grid_sampler(canonical_pos, self.k0)
                    current_k0 = current_k0.permute(0, 4, 1, 2, 3)  # N C X Y Z    
        else:
            current_density = self.density
            current_k0 = self.k0        

        # query for alpha w/ post-activation
        density = self.grid_sampler(ray_pts, current_density)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_pts_ori = ray_pts_ori[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_pts_ori = ray_pts_ori[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]

        # query for color
        vox_emb = self.grid_sampler(ray_pts, current_k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'ray_pts': ray_pts,
            'ray_pts_ori': ray_pts_ori,
            'mask': mask,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict

def img2mae(x, y, weights=None, M=None):
    if weights == None:
        if M == None:
            return torch.mean(torch.abs(x - y))
        else:
            return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]   # M or only the pixels number?           
    else:
        if M == None:
            return torch.mean(torch.abs(x - y) * weights[..., None])
        else:
            return torch.sum(torch.abs(x - y) * weights[..., None] * M) / (torch.sum(M) + 1e-8) / x.shape[-1]   # M or only the pixels number?     


def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world            

@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

