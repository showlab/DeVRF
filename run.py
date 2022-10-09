import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, devrf, dempirf
from lib.load_data import load_data


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, frame_times, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []   

    eps_render = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        frame_time = frame_times[i].to(device)
        rays_o, rays_d, viewdirs = devrf.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        # find the neighbored timesteps motion and put them on gpu memory
        if model.timesteps != 0 and frame_time != 0:
            time_id_ = np.round((frame_time / model.timestep).cpu(), decimals=3)
            time_id = time_id_.cuda()
            time_id_prev = time_id.floor().long()
            time_id_next = time_id.ceil().long()  
            if time_id_prev == time_id_next:
                model.motion_list[time_id_prev] = model.motion_list[time_id_prev].to(device)
            else:
                model.motion_list[time_id_prev] = model.motion_list[time_id_prev].to(device)
                model.motion_list[time_id_next] = model.motion_list[time_id_next].to(device)

        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, frame_time, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }

        # put the neighbored timesteps motion to cpu memory
        if model.timesteps != 0 and frame_time != 0:
            if time_id_prev == time_id_next:
                model.motion_list[time_id_prev] = model.motion_list[time_id_prev].cpu()
            else:
                model.motion_list[time_id_prev] = model.motion_list[time_id_prev].cpu()
                model.motion_list[time_id_next] = model.motion_list[time_id_next].cpu()      

        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    eps_render = time.time() - eps_render
    eps_time_str = f'{eps_render//3600:02.0f}:{eps_render//60%60:02.0f}:{eps_render%60:02.0f}'
    print('render: render takes ', eps_time_str)

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)

    return rgbs, depths


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'times', 'render_times', 'random_poses', 'grids'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['times'] = torch.Tensor(data_dict['times'])
    if data_dict['random_poses'] is not None:
        data_dict['random_poses'] = torch.Tensor(data_dict['random_poses'])
    data_dict['grids'] = torch.Tensor(data_dict['grids'])
    data_dict['render_poses'] = torch.Tensor(data_dict['render_poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = devrf.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, data_dict, stage, xyz_min=None, xyz_max=None, coarse_ckpt_path=None):

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times, random_poses, grids, hwf = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'times', 'render_times', 'random_poses', 'grids', 'hwf'#, 'vertices'
        ]
    ]
    frame_times = times[i_train]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None and cfg_train.static_model_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
        # init model
        model_kwargs = copy.deepcopy(cfg_model)
        if cfg.data.ndc:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
            num_voxels = model_kwargs.pop('num_voxels')
            if len(cfg_train.pg_scale) and reload_ckpt_path is None:
                num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
            model = dempirf.DeMPIRF(
                xyz_min=xyz_min, xyz_max=xyz_max,
                num_voxels=num_voxels,
                mask_cache_path=coarse_ckpt_path,
                **model_kwargs)
        else:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
            num_voxels = model_kwargs.pop('num_voxels')
            if len(cfg_train.pg_scale) and reload_ckpt_path is None:
                num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
            model = devrf.DeVRF(
                xyz_min=xyz_min, xyz_max=xyz_max,
                num_voxels=num_voxels,
                mask_cache_path=coarse_ckpt_path,
                **model_kwargs)
            if cfg_model.maskout_near_cam_vox:
                model.maskout_near_cam_vox(poses[i_train,:3,3], near)
        model = model.to(device)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {cfg_train.static_model_path}')
        start = 0
        model_kwargs = copy.deepcopy(cfg_model)
        if cfg.data.ndc:
            model_class = dempirf.DeMPIRF
        else:
            model_class = devrf.DeVRF
        num_voxels_motion = model_kwargs.pop('num_voxels_motion')
        motion_dim = model_kwargs.pop('motion_dim')
        timesteps = model_kwargs.pop('timesteps')   
        warp_ray = model_kwargs.pop('warp_ray')  
        world_motion_bound_scale = model_kwargs.pop('world_motion_bound_scale')  
        mpi_depth_motion = 0
        if cfg.data.ndc:
            mpi_depth_motion = model_kwargs.pop('mpi_depth_motion') 
        if len(cfg_train.pg_motionscale):
            if cfg.data.ndc:
                num_voxels_motion = int(num_voxels_motion / (2**len(cfg_train.pg_motionscale))**3)  # This step will reduce voxels number. numbers in pg_scale in the following.   
                mpi_depth_motion = int(mpi_depth_motion / 2**len(cfg_train.pg_motionscale)) 
            else:
                num_voxels_motion = int(num_voxels_motion / (2**len(cfg_train.pg_motionscale))**3)

        model = utils.load_staticmodel(model_class, num_voxels_motion, timesteps, motion_dim, warp_ray, cfg_train.static_model_path, world_motion_bound_scale, mpi_depth_motion).to(device)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': False,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = devrf.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = devrf.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = devrf.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = devrf.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # init decay factor for first stage
    decay_factor = 1
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    # for fixed grad sampling use
    iters_1step = int(cfg_train.N_iters / timesteps)
    imgs_1step = int(len(i_train) / timesteps)    
    if len(cfg_train.pg_motionscale):
        pg_step = 0
        iters_1step = int(cfg_train.pg_motionscale[pg_step] / timesteps)
    for global_step in trange(start, cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_motionscale:
            # change the momory storage of the last two timesteps of fwd and bwd motion list from gpu to cpu
            model.motion_list[current_step].requires_grad = False
            model.motion_list[current_step] = model.motion_list[current_step].detach()
            model.motion_list[current_step] = model.motion_list[current_step].cpu()
            model.fwdmotion_list[current_step-1] = model.fwdmotion_list[current_step-1].cpu()
            model.fwdmotion_list[current_step].requires_grad = False
            model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].detach()    
            model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].cpu()        

            n_rest_scales = len(cfg_train.pg_motionscale) - cfg_train.pg_motionscale.index(global_step) - 1
            if isinstance(model, devrf.DeVRF):
                cur_motion_voxels = int(cfg_model.num_voxels_motion / (2**n_rest_scales)**3)
                model.scale_volume_grid_motion(cur_motion_voxels)            
            elif isinstance(model, dempirf.DeMPIRF):
                cur_motion_voxels = int(cfg_model.num_voxels_motion / (2**n_rest_scales)**3)
                mpi_depth_motion = int(cfg_model.mpi_depth_motion / (2**n_rest_scales))
                model.scale_volume_grid_motion(cur_motion_voxels, mpi_depth_motion)
            else:
                raise NotImplementedError              
            pg_step +=1
            if pg_step < len(cfg_train.pg_motionscale):
                iters_1step = int((cfg_train.pg_motionscale[pg_step] - cfg_train.pg_motionscale[pg_step-1]) / timesteps)
            else:
                iters_1step = int((cfg_train.N_iters - cfg_train.pg_motionscale[pg_step-1]) / timesteps)
            # update lr
            decay_steps = len(cfg_train.pg_motionscale)
            decay_factor = cfg_train.lrdecay_scale ** (pg_step / decay_steps)
         
        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random_1im':
            # Randomly select one image due to time step.
            if global_step >= cfg_train.precrop_iters_time:
                img_i = np.random.choice(i_train)
            else:
                skip_factor = global_step / float(cfg_train.precrop_iters_time) * len(i_train)
                max_sample = max(int(skip_factor), 3)
                img_i = np.random.choice(i_train[:max_sample])
            # Require i_train order is the same as the above reordering of training images. 
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            frame_time = frame_times[img_i].to(target.device)
        elif cfg_train.ray_sampler == 'sequential_1im_fixed':  
            if len(cfg_train.pg_motionscale):
                if pg_step == 0:
                    current_iters = global_step
                    current_step = int(current_iters / iters_1step)  # 0-49. 
                    if current_iters % iters_1step == 0: 
                        if current_step == 0:
                            model.motion_list[current_step+1] = model.motion_list[current_step+1].to(device)
                            model.motion_list[current_step+1].requires_grad = True             
                            model.fwdmotion_list[current_step+1] = model.fwdmotion_list[current_step+1].to(device)         
                            model.fwdmotion_list[current_step+1].requires_grad = True            
                            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step+1, decay_factor)
                        elif current_step > 1:
                            model.motion_list[current_step] = model.motion_list[current_step].to(device)
                            model.motion_list[current_step].requires_grad = True    
                            model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].to(device)         
                            model.fwdmotion_list[current_step].requires_grad = True      
                            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step, decay_factor)      
                    if current_iters % iters_1step == 0 and current_iters > 0:    
                        if current_step > 1:
                            with torch.no_grad():
                                model.motion_list[current_step].copy_(model.motion_list[current_step-1].data)
                                model.fwdmotion_list[current_step].copy_(model.fwdmotion_list[current_step-1].data)
                                model.motion_list[current_step-1].requires_grad = False 
                                model.motion_list[current_step-1] = model.motion_list[current_step-1].detach()
                                model.fwdmotion_list[current_step-1].requires_grad = False 
                                model.fwdmotion_list[current_step-1] = model.fwdmotion_list[current_step-1].detach()   
                                # store previous step motion in cpu memory
                                model.motion_list[current_step-1] = model.motion_list[current_step-1].cpu()
                                model.fwdmotion_list[current_step-2] = model.fwdmotion_list[current_step-2].cpu()                             

                else:
                    current_iters = global_step - cfg_train.pg_motionscale[pg_step-1]
                    current_step = int(current_iters / iters_1step)  # 0-49. 
                    if current_iters % iters_1step == 0: 
                        if current_step == 0:
                            model.motion_list[current_step+1] = model.motion_list[current_step+1].to(device)
                            model.motion_list[current_step+1].requires_grad = True             
                            model.fwdmotion_list[current_step+1] = model.fwdmotion_list[current_step+1].to(device)         
                            model.fwdmotion_list[current_step+1].requires_grad = True            
                            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step+1, decay_factor)
                        elif current_step > 1:
                            model.motion_list[current_step] = model.motion_list[current_step].to(device)
                            model.motion_list[current_step].requires_grad = True    
                            model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].to(device)         
                            model.fwdmotion_list[current_step].requires_grad = True      
                            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step, decay_factor)                                                                              
                    if current_iters % iters_1step == 0 and current_iters > 0:    
                        if current_step > 1:
                            with torch.no_grad():
                                model.motion_list[current_step-1].requires_grad = False       
                                model.motion_list[current_step-1] = model.motion_list[current_step-1].detach()
                                model.fwdmotion_list[current_step-1].requires_grad = False       
                                model.fwdmotion_list[current_step-1] = model.fwdmotion_list[current_step-1].detach()     
                                # store previous step motion in cpu memory
                                model.motion_list[current_step-1] = model.motion_list[current_step-1].cpu()
                                model.fwdmotion_list[current_step-2] = model.fwdmotion_list[current_step-2].cpu()                                                             
            else:
                current_iters = global_step
                current_step = int(current_iters / iters_1step)  # 0-49. 
                if current_iters % iters_1step == 0: 
                    if current_step == 0:
                        model.motion_list[current_step+1] = model.motion_list[current_step+1].to(device)
                        model.motion_list[current_step+1].requires_grad = True             
                        model.fwdmotion_list[current_step+1] = model.fwdmotion_list[current_step+1].to(device)         
                        model.fwdmotion_list[current_step+1].requires_grad = True            
                        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step+1, decay_factor)
                    elif current_step > 1:
                        model.motion_list[current_step] = model.motion_list[current_step].to(device)
                        model.motion_list[current_step].requires_grad = True    
                        model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].to(device)         
                        model.fwdmotion_list[current_step].requires_grad = True      
                        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, timesteps, current_step, decay_factor)                                                      
                if current_iters % iters_1step == 0 and current_iters > 0:                        
                    if current_step > 1:
                        with torch.no_grad():
                            model.motion_list[current_step].copy_(model.motion_list[current_step-1].data)
                            model.fwdmotion_list[current_step].copy_(model.fwdmotion_list[current_step-1].data)
                            model.motion_list[current_step-1].requires_grad = False
                            model.motion_list[current_step-1] = model.motion_list[current_step-1].detach()
                            model.fwdmotion_list[current_step-1].requires_grad = False
                            model.fwdmotion_list[current_step-1] = model.fwdmotion_list[current_step-1].detach()    
                            # store previous step motion in cpu memory
                            model.motion_list[current_step-1] = model.motion_list[current_step-1].cpu()
                            model.fwdmotion_list[current_step-2] = model.fwdmotion_list[current_step-2].cpu()                            

            img_i = current_iters % imgs_1step + current_step * imgs_1step      
            frame_time_ = frame_times[img_i]
            if frame_time_ == 0:
                img_i = current_iters % imgs_1step + (current_step+1) * imgs_1step
                frame_time_ = frame_times[img_i]
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[img_i, sel_r, sel_c]
            rays_o = rays_o_tr[img_i, sel_r, sel_c]
            rays_d = rays_d_tr[img_i, sel_r, sel_c]
            viewdirs = viewdirs_tr[img_i, sel_r, sel_c]
            frame_time = frame_times[img_i]
            grid = grids[i_train][img_i, sel_r, sel_c]
            pose = poses[i_train][img_i]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, frame_time, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        rgb_loss = loss
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss        
        if cfg_train.weight_motion_cycle>0:
            motion_cycle_loss = cfg_train.weight_motion_cycle * model.cycle_loss(render_result['ray_pts_ori'], render_result['ray_pts'], frame_time)   
            loss += motion_cycle_loss                    
        if cfg_train.weight_flow>0:
            flow_loss = cfg_train.weight_flow * model.flow_loss(grid, pose, frame_time, hwf, render_result['weights'], render_result['ray_pts'], render_result['ray_id'])   
            loss += flow_loss                     
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_motion>0:
                if frame_time > 0:
                    model.motion_total_variation_add_grad(
                        cfg_train.weight_tv_motion/len(rays_o), global_step<cfg_train.tv_dense_before, frame_time)                
        
        optimizer.step()
        psnr_lst.append(psnr.item())            

        # check log & save
        if (global_step+1)%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if (global_step+1)%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            model.motion = torch.nn.Parameter(torch.cat(model.motion_list))     
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    # change the momory storage of the last two timesteps of fwd and bwd motion list from gpu to cpu   
    model.motion_list[current_step].requires_grad = False
    model.motion_list[current_step] = model.motion_list[current_step].detach()
    model.motion_list[current_step] = model.motion_list[current_step].cpu()
    model.fwdmotion_list[current_step-1] = model.fwdmotion_list[current_step-1].cpu()
    model.fwdmotion_list[current_step].requires_grad = False
    model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].detach()    
    model.fwdmotion_list[current_step] = model.fwdmotion_list[current_step].cpu()      

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'motion_list': model.motion_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_fine = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # fine detail reconstruction
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            data_dict=data_dict, stage='fine')
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = devrf.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(devrf.DeVRF, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dempirf.DeMPIRF
        else:
            model_class = devrf.DeVRF
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        imgs_perview = model.timesteps
        num_views = int(len(data_dict['i_train']) / imgs_perview)          
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                frame_times=data_dict['times'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        for i in range(num_views):
            idx = i + num_views * np.array(range(imgs_perview), int)
            rgb_video = 'view' + str(i+1) + '_video.rgb.mp4'
            depth_video = 'view' + str(i+1) + '_video.depth.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs[idx]), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths[idx] / np.max(depths[idx])), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        imgs_perview = model.timesteps
        num_views = int(len(data_dict['i_test']) / imgs_perview)              
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                frame_times=data_dict['times'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        for i in range(num_views):
            idx = i + num_views * np.array(range(imgs_perview), int)
            rgb_video = 'view' + str(i+1) + '_video.rgb.mp4'
            depth_video = 'view' + str(i+1) + '_video.depth.mp4'
            imageio.mimwrite(os.path.join(testsavedir, rgb_video), utils.to8b(rgbs[idx]), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, depth_video), utils.to8b(1 - depths[idx] / np.max(depths[idx])), fps=30, quality=8)                

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                frame_times=data_dict['render_times'],
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    print('Done')

