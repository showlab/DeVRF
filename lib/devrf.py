import os
import time
import functools
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms

from torch_scatter import segment_coo

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)     


'''Model'''
class DeVRF(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 num_voxels_motion=0, motion_dim=0, timesteps=0, warp_ray=False, world_motion_bound_scale=1.0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DeVRF, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.world_motion_bound_scale = world_motion_bound_scale
        self.fast_color_thres = fast_color_thres
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('devrf: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        self._set_grid_resolution_motion(num_voxels_motion)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]), requires_grad=False)

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
            self.timestep = 1.0 / (self.timesteps - 1)  # include t=0. so should minus 1.
        else:
            self.timestep = 1.0        

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]), requires_grad=False)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]), requires_grad=False)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            for param in self.rgbnet.parameters():
                param.requires_grad = False                 
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('devrf: feature voxel grid', self.k0.shape)
            print('devrf: mlp', self.rgbnet)

        self.mask_cache = None
        self.nonempty_mask = None            

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        # define the xyz positions of grid
        self.world_pos = torch.zeros([1, 3, *self.world_size])
        xcoord = torch.linspace(start=self.xyz_min[0], end=self.xyz_max[0], steps=self.world_size[0])
        ycoord = torch.linspace(start=self.xyz_min[1], end=self.xyz_max[1], steps=self.world_size[1])
        zcoord = torch.linspace(start=self.xyz_min[2], end=self.xyz_max[2], steps=self.world_size[2])
        grid = torch.meshgrid(xcoord, ycoord, zcoord)
        for i in range(3):
            self.world_pos[0, i, :, :, :] = grid[i]        
        print('devrf: voxel_size      ', self.voxel_size)
        print('devrf: world_size      ', self.world_size)
        print('devrf: voxel_size_base ', self.voxel_size_base)
        print('devrf: voxel_size_ratio', self.voxel_size_ratio)

    def _set_grid_resolution_motion(self, num_voxels_motion):
        self.num_voxels_motion = num_voxels_motion
        self.voxel_size_motion = ((self.xyz_max - self.xyz_min).prod() / num_voxels_motion).pow(1/3)
        self.motion_size = ((self.xyz_max - self.xyz_min) / self.voxel_size_motion).long() 
        
        print('devrf: num_voxels_motion      ', self.num_voxels_motion) 
        print('devrf: voxel_size_motion      ', self.voxel_size_motion)
        print('devrf: world_size_motion      ', self.motion_size)           

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
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
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('devrf: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('devrf: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))

        mask_cache = MaskCache(
                path=self.mask_cache_path,
                mask_cache_thres=self.mask_cache_thres).to(self.xyz_min.device)
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        self_alpha = F.max_pool3d(self.activate_density(self.density), kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache = MaskCache(
                path=None, mask=mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('devrf: scale_volume_grid finish')

    @torch.no_grad()
    def scale_volume_grid_motion(self, num_voxels_motion):
        print('devrf: scale_volume_grid_motion start')
        ori_motion_size = self.motion_size
        self._set_grid_resolution_motion(num_voxels_motion)
        print('devrf: scale_volume_grid_motion scale world_size from', ori_motion_size, 'to', self.motion_size)         
        for i in range(self.timesteps):
            self.motion_list[i] = torch.nn.Parameter(
                F.interpolate(self.motion_list[i].data, size=tuple(self.motion_size), mode='trilinear', align_corners=True), requires_grad=False)  
            self.fwdmotion_list[i] = torch.nn.Parameter(
                F.interpolate(self.fwdmotion_list[i].data, size=tuple(self.motion_size), mode='trilinear', align_corners=True), requires_grad=False)  
        print('devrf: scale_volume_grid finish')          

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('devrf: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('devrf: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.density, self.density.grad, weight, weight, weight, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.k0, self.k0.grad, weight, weight, weight, dense_mode)

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
                        pts_transformed = rays_pts
                    else:
                        fwdmotion_3D = self.fwdmotion_list[time_id_next-1]
                        rays_pts_fwdmotion = self.grid_sampler(rays_pts, fwdmotion_3D)
                        pts_transformed = rays_pts + rays_pts_fwdmotion
                    H, W, f = hwf
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
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

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
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, t_min

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
        ray_pts, ray_id, step_id, t_min = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_pts_ori = ray_pts

        # use absolute motion to canonical space and warp points.
        if self.timesteps != 0 and frame_time != 0:
            time_id_ = np.round((frame_time / self.timestep).cpu(), decimals=3)
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
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_pts_ori = ray_pts_ori[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        if not self.rgbnet_full_implicit:
            k0 = self.grid_sampler(ray_pts, current_k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

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
                stepdist = render_kwargs['stepsize'] * self.voxel_size
                depth = segment_coo(
                        src=(weights * (step_id * stepdist + t_min[ray_id])),    
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super().__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_kwargs']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None

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

''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

@torch.no_grad()
def get_random_rays(random_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_random_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(random_poses) == len(Ks) and len(random_poses) == len(HW)
    H, W = HW[0]
    H = int(H)
    W = int(W)
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(random_poses), H, W, 3], device=random_poses.device)
    rays_d_tr = torch.zeros([len(random_poses), H, W, 3], device=random_poses.device)
    viewdirs_tr = torch.zeros([len(random_poses), H, W, 3], device=random_poses.device)
    imsz = [1] * len(random_poses)
    for i, c2w in enumerate(random_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(random_poses.device))
        rays_d_tr[i].copy_(rays_d.to(random_poses.device))
        viewdirs_tr[i].copy_(viewdirs.to(random_poses.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_random_rays: finish (eps time:', eps_time, 'sec)')
    return rays_o_tr, rays_d_tr, viewdirs_tr, imsz    


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

