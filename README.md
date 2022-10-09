
  
# DeVRF: Fast Deformable Voxel Radiance Fields for Dynamic Scenes

  

[Project page](https://jia-wei-liu.github.io/DeVRF) | [arXiv](https://arxiv.org/abs/2205.15723)

  

> **TL;DR:** A novel representaion and learning paradigm for dynamic radiance fields reconstruction -- 100x faster, no loss in dynamic novel view synthesis quality.

  

<img  src="/figures/DeVRF.png"  alt="DeVRF"  style="zoom:67%;"  />

  

## 📢 News

 - [2022.10.10]  We release the first-version of DeVRF code and dataset!

- [2022.9.15] DeVRF got accepted by [**NeurIPS 2022**](https://nips.cc/)!

- [2022.6.1] We release the arXiv paper.

  

## 📝 Preparation

### Installation
```
git clone https://github.com/showlab/DeVRF.git
cd DeVRF
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.



<details>

<summary> Dependencies (click to expand) </summary>

 

- `PyTorch`, `numpy`, `torch_scatter`, `pytorch3d`: main computation.

- `scipy`, `lpips`: SSIM and LPIPS evaluation.

- `tqdm`: progress bar.

- `mmcv`: config system.

- `opencv-python`: image processing.

- `imageio`, `imageio-ffmpeg`: images and videos I/O.

- `Ninja`: to build the newly implemented torch extention just-in-time.

- `einops`: torch tensor shaping with pretty api.

</details>

### DeVRF dataset

We release all the synthetic and real-world DeVRF dataset on [link](https://drive.google.com/drive/folders/1IuYCTIcUJxPJs6fE9SKSlLPNGiWs3OaQ?usp=sharing). DeVRF dataset consists of 5 inward-facing synthetic scenes (lego|floating_robot|kuka|daisy|glove), 1 inward-facing real-world scene (flower_360), and 3 forward-facing real-world scenes (plant|rabbit|pig_toy). For each scene, we release the static data, dynamic data, and optical flow estimated using [RAFT](https://github.com/princeton-vl/RAFT). Please refer to the following data structure for an overview of DeVRF dataset.

```
    DeVRF dataset
    ├── inward-facing
    │   └── [lego|floating_robot|kuka|daisy|glove|flower_360]
    │       ├── static    
    │       │	├── [train|val|test]
    │       │	└── transforms_[train|val|test].json
    │       └── dynamic_4views    
    │        	├── [train|val|test]
    │        	├── transforms_[train|val|test].json   
    │        	├── train_flow              
    │        	└── train_flow_png  
    │          
    └── forward-facing
        └── [plant|rabbit|pig_toy]
            ├── static    
            │	├── [images|images_4|images_8]
            │	└── poses_bounds.npy
            └── dynamic_4views    
             	├── bds.npy  
             	├── poses.npy                        
             	└── [view1|view2|view3|view4]
	                ├── [images|images_4|images_8]
             	 	├── images_4_flow
             	 	└── images_4_flow_png
```

We additionally provide a light version of DeVRF dataset without optical flow on [link](https://drive.google.com/drive/folders/18-1aRhFd7Z9ugZCOAZ9ZmHcoc9eaRBcg?usp=sharing).

 
## 🏋️‍️ Experiment

### Training

Stage 1: Train the static model using static scene data.
The static model part is almost the same as [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO). The main difference is that we add an accumulated transmittance loss to encourage a clean background for forward-facing scenes. Please refer to [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) for more details.

Note: Please enlarge the world_bound_scale in config file to establish a larger bounding box for dynamic scene modelling in the second stage. For DeVRF dataset, the world_bound_scale parameter is set within [1.05, 2.0].
	
```bash

$ cd static_DirectVoxGO
$ python run.py --config configs/inward-facing/lego.py --render_test

```

Stage 2: Train the dynamic model using dynamic scene data and the trained static model.

```bash

$ python run.py --config configs/inward-facing/lego.py --render_test

```

### Evaluation

To only evaluate the testset `PSNR`, `SSIM`, and `LPIPS` of the trained `lego` without re-training, run:

```bash

$ python run.py --config configs/inward-facing/lego.py --render_only --render_test \

--eval_ssim --eval_lpips_vgg --eval_lpips_alex

```

Use `--eval_lpips_alex` or `--eval_lpips_vgg` to evaluate LPIPS with pre-trained Alex net or VGG net.

### Render video
	
```bash

$ python run.py --config configs/inward-facing/lego.py --render_only --render_video

```

### Reproduction: all config files to reproduce our results.

<details>

<summary> (click to expand) </summary>

```bash
$ ls configs/*

configs/inward-facing:
lego.py floating_robot.py kuka.py daisy.py glove.py flower_360.py

configs/forward-facing:
plant.py rabbit.py pig_toy.py

```
 
</details>

  

## 🎓 Citation

  

If you find our work helps, please cite our paper.

  

```bibtex

@article{liu2022devrf,

title={DeVRF: Fast Deformable Voxel Radiance Fields for Dynamic Scenes},

author={Liu, Jia-Wei and Cao, Yan-Pei and Mao, Weijia and Zhang, Wenqiao and Zhang, David Junhao and Keppo, Jussi and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},

journal={arXiv preprint arXiv:2205.15723},

year={2022}

}

```

  

## ✉️ Contact

  

This repo is maintained by [Jiawei Liu](https://jia-wei-liu.github.io/). Questions and discussions are welcome via jiawei.liu@u.nus.edu.


  

## 🙏 Acknowledgements

  

This codebase is based on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).

  


## LICENSE

  

MIT