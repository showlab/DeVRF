import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data


def load_data(args):

    K, depths, random_poses = None, None, None

    if args.dataset_type == 'llff':
        images, times, depths, poses, bds, render_poses, render_times, i_test, flows_b, flow_masks_b = load_llff_data(
                args.datadir, args.factor,
                recenter=True, bd_factor=.75,
                spherify=args.spherify,
                load_depths=args.load_depths)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
        flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)          
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]
        num_views = int(args.datadir[-6])
        imgs_perview = int(images.shape[0] / num_views)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
                images_opacity = images[..., -1]
            else:
                images = images[...,:3]*images[...,-1:]                

        if args.llffhold_view > -1:
            i_test = args.llffhold_view-1 + num_views * np.array(range(imgs_perview), int)
        else:
            i_test = np.arange(images.shape[0])[-num_views:]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, times, render_poses, render_times, hwf, i_split, flows_b, flow_masks_b = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)

        i_train, i_val, i_test = i_split
        flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
        flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)                   

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) 
            else:
                images = images[...,:3] 

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    min_time, max_time = times[i_train[0]], times[i_train[-1]]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    grids = get_grid(int(H), int(W), flows_b.shape[0], flows_b, flow_masks_b) # [N, H, W, 5]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        times=times, render_times=render_times, random_poses=random_poses, grids=grids,
    )
    return data_dict

def get_grid(H, W, num_img, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    grid = np.empty((0, H, W, 5), np.float32)
    for idx in range(num_img):
        grid = np.concatenate((grid, np.stack([i,
                                               j,
                                               flows_b[idx, :, :, 0],
                                               flows_b[idx, :, :, 1],
                                               flow_masks_b[idx, :, :]], -1)[None, ...]))
    return grid


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far


def _generate_random_poses(args, poses, i_test):
    """Calculates random poses for the Blender dataset."""
    n_poses = args.n_random_poses

    if args.random_pose_type == 'renderpath':
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), 100, 0)    # similar to np.max
        radii = np.concatenate([radii, [1.]])
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        random_poses = []
        for _ in range(n_poses):
            t = radii * np.concatenate([
                2 * args.random_pose_radius * (np.random.rand(3) - 0.5), [1,]])
            position = cam2world @ t
            if args.random_pose_focusptjitter:
                z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
            else:
                z_axis_i = z_axis
            random_poses.append(viewmatrix(z_axis_i, up, position, True))
        if args.random_pose_add_test_poses:
            random_poses = random_poses + list(poses[i_test])
    elif args.random_pose_type == 'linearcomb':
        random_poses = list(poses)
        for _ in range(n_poses - poses.shape[0]):
            idx = np.random.choice(poses.shape[0], size=(2,), replace=False)
            w = np.random.rand()
            pose_i = w * poses[idx[0]] + (1 - w) * poses[idx[1]]
            random_poses.append(pose_i)
    elif args.random_pose_type == 'testposes':
        random_poses = list(poses[i_test])
    elif args.random_pose_type == 'allposes':
        random_poses = list(poses)

    random_poses = np.stack(random_poses, axis=0)
    return random_poses

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world    

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)  

def focus_pt_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt    