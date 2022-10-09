
''' Template of data options
'''

data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type='llff',          # blender | llff | None
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    load2gpu_on_the_fly=True,     # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=True,                # to be consistent with baselines

    ndc=True,                     # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    factor=4,                     # the factor of image resolution
    width=None,                   # enforce image width
    height=None,                  # enforce image height
    llffhold_view=4,              # test view
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
)

''' Template of training options
'''

fine_train = dict(
    N_iters=25000,                              # number of optimization steps
    N_rand=8192,                                # batch size (number of random rays per optimization step)
    lrate_density=0,                            # lr of density voxel grid. 1e-1
    lrate_motion_list=1e-3,                     # lr of motion voxel grid.
    lrate_fwdmotion_list=1e-3,                  # lr of forward motion voxel grid.    
    lrate_k0=0,                                 # lr of color/feature voxel grid1e-1
    lrate_rgbnet=0,                             # lr of the mlp to preduct view-dependent color 1e-3
    lrate_decay=20,                             # lr decay by 0.1 after every lrate_decay*1000 steps
    lrdecay_scale=1e-1,                         # the lrate decay scale for the final fine stage
    pervoxel_lr=False,                          # view-count-based lr
    ray_sampler='sequential_1im_fixed',         # ray sampling strategies
    weight_main=1.0,                            # weight of photometric loss
    weight_entropy_last=0.01,                   # weight of background entropy loss
    weight_rgbper=0.1,                          # weight of per-point rgb loss
    tv_every=1,                                 # count total variation loss every tv_every step
    tv_after=0,                                 # count total variation loss from tv_from step
    tv_before=120000,                           # count total variation before the given number of iterations
    tv_dense_before=120000,                     # count total variation densely before the given number of iterations
    weight_tv_density=0.0,                      # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,                           # weight of total variation loss of color/feature voxel grid
    weight_tv_motion=1,                         # weight of total variation loss of motion voxel grid.  
    weight_motion_cycle=100,                    # weight of motion cycle consistency loss.                                  
    weight_flow=5e-3,                           # weight of flow loss  
    pg_scale=[],                                # checkpoints for progressive scaling
    pg_motionscale=[10000, 15000, 20000, 22500],# checkpoints for motion progressive scaling   
    skip_zero_grad_fields=[],                   # the variable name to skip optimizing parameters w/ zero grad in each iteration
)


''' Template of model and rendering options
'''

fine_model_and_render = dict(
    num_voxels=256*3,                           # expected number of voxel 160
    num_voxels_base=256**3,                     # to rescale delta distance 160
    mpi_depth=128,                              # the number of planes in Multiplane Image (work when ndc=True)
    num_voxels_motion=160**3,                   # expected number of motion voxel
    mpi_depth_motion=128,                       # the depth information for motion voxels
    nearest=False,                              # nearest interpolation
    pre_act_density=False,                      # pre-activated trilinear interpolation
    in_act_density=False,                       # in-activated trilinear interpolation
    bbox_thres=1e-3,                            # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,                      # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=9,                               # feature voxel grid dim
    motion_dim=3,                               # motion dim. 3 (x, y, z), 7, 9
    timesteps=50,                               # number of timesteps soccer use 4 views and reduce last step as test.
    warp_ray=True,                              # warp ray or warp voxel
    rgbnet_full_implicit=False,                 # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,                         # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,                             # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=64,                            # width of the colors MLP
    alpha_init=1e-2,                            # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-3,                      # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=False,                 # maskout grid points that between cameras and their near planes
    world_motion_bound_scale=1.0,               # rescale the Motion BBox enclosing the scene
    world_bound_scale=1,                        # rescale the Motion BBox enclosing the scene
    stepsize=0.5,                               # sampling stepsize in volume rendering
)
