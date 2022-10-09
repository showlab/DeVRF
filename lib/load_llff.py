import numpy as np
import os, imageio
import torch
import cv2

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original
def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized        

def depthread(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, load_depths=False):     # v1: load poses from static processed poses

    poses = np.load(os.path.join(basedir, 'poses.npy'))
    bds = np.load(os.path.join(basedir, 'bds.npy'))

    num_views = int(basedir[-6])
    basedir_view1 = os.path.join(basedir, 'view1')
    image_dir = os.path.join(basedir_view1, 'images')
    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs_perview = len(image_files)
    num_images = imgs_perview * num_views
    all_imgs = np.array([None] * num_images)
    all_poses = np.array([None] * num_images)
    all_times = np.array([None] * num_images)
    all_bds = np.array([None] * num_images)
    all_flows_b = np.array([None] * num_images)
    all_flow_masks_b = np.array([None] * num_images)
    for i in range(1, num_views+1):
        pose = poses[-(num_views-i+1), ...]
        idx = i-1 + num_views * np.array(range(imgs_perview), int)
        viewimg_dir = 'view' + str(i)
        basedir_view = os.path.join(basedir, viewimg_dir)
        img0 = [os.path.join(basedir_view, 'images', f) for f in sorted(os.listdir(os.path.join(basedir_view, 'images'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape

        sfx = ''

        if height is not None and width is not None:
            _minify(basedir_view, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif factor is not None and factor != 1:
            sfx = '_{}'.format(factor)
            _minify(basedir_view, factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(sh[1] / factor)
            _minify(basedir_view, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(sh[0] / factor)
            _minify(basedir_view, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        else:
            factor = 1

        imgdir = os.path.join(basedir_view, 'images' + sfx)
        flow_dir = os.path.join(basedir_view, 'images' + sfx + '_flow')
        if not os.path.exists(imgdir):
            print( imgdir, 'does not exist, returning' )
            return
        if not os.path.exists(flow_dir):
            print( flow_dir, 'does not exist, returning' )
            return            

        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        imgs = [imread(f)[...,:4]/255. for f in imgfiles]
        all_imgs[idx] = imgs
        all_poses[idx] = [pose] * imgs_perview
        all_bds[idx] = [bds[-(num_views-i+1), ...]] * imgs_perview
        all_times[idx] = np.array(range(imgs_perview)) / (imgs_perview-1.0)

        # get flow 
        H, W = imgs[0].shape[:2]
        for flow_i in range(len(idx)):
            cur_time = all_times[idx[flow_i]]
            if cur_time == 0:
                bwd_flow, bwd_mask = np.zeros((H, W, 2)), np.zeros((H, W))
            else:
                bwd_flow_path = os.path.join(flow_dir, '%03d_bwd.npz'%flow_i)
                bwd_data = np.load(bwd_flow_path)
                bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
                bwd_flow = resize_flow(bwd_flow, H, W)
                bwd_mask = np.float32(bwd_mask)
                bwd_mask = cv2.resize(bwd_mask, (W, H),
                                    interpolation=cv2.INTER_NEAREST)                
            all_flows_b[idx[flow_i]] = bwd_flow
            all_flow_masks_b[idx[flow_i]] = bwd_mask
    
    imgs = np.stack(all_imgs, -1)
    poses = np.stack(all_poses, -1)
    bds = np.stack(all_bds, -1)
    times = np.stack(all_times, -1)
    flows_b = np.stack(all_flows_b, -1)
    flow_masks_b = np.stack(all_flow_masks_b, -1)

    print('Loaded image, poses, times, bds, flows_b, flow_masks_b data', imgs.shape, poses.shape, times.shape, bds.shape, flows_b.shape, flow_masks_b.shape)

    if not load_depths:
        return poses, bds, imgs, times, flows_b, flow_masks_b

    depthdir = os.path.join(basedir, 'stereo', 'depth_maps')
    assert os.path.exists(depthdir), f'Dir not found: {depthdir}'

    depthfiles = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir)) if f.endswith('.geometric.bin')]
    assert poses.shape[-1] == len(depthfiles), 'Mismatch between imgs {} and poses {} !!!!'.format(len(depthfiles), poses.shape[-1])

    depths = [depthread(f) for f in depthfiles]
    depths = np.stack(depths, -1)
    print('Loaded depth data', depths.shape)
    return poses, bds, imgs, times, depths


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds, depths):

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    depths *= sc

    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, new_poses, bds, depths


def load_llff_data(basedir, factor=8, width=None, height=None,
                   recenter=True, bd_factor=.75, spherify=False, path_zflat=False, load_depths=False):

    poses, bds, imgs, times, flows_b, flow_masks_b, *depths = _load_data(basedir, factor=factor, width=width, height=height,
                                           load_depths=load_depths)
    print('Loaded', basedir, bds.min(), bds.max())
    if load_depths:
        depths = depths[0]
    else:
        depths = 0

    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    times = np.moveaxis(times, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if spherify:
        poses, render_poses, bds, depths = spherify_poses(poses, bds, depths)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*1, bds.max()*9
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views) 
    render_poses = torch.Tensor(render_poses)
    render_times = torch.linspace(0., 1., render_poses.shape[0])
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape, times.shape)

    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, times, depths, poses, bds, render_poses, render_times, i_test, flows_b, flow_masks_b

