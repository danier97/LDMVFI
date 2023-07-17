# LDMVFI: Video Frame Interpolation with Latent Diffusion Models

[**Duolikun Danier**](https://danier97.github.io/), [**Fan Zhang**](https://fan-aaron-zhang.github.io/), [**David Bull**](https://david-bull.github.io/)

[Project](TODO) | [arXiv](https://arxiv.org/abs/2303.09508) | [Video](https://drive.google.com/file/d/1oL6j_l3b2QEqsL0iO7qSZrGUXJaTpRWN/view?usp=share_link)

![Demo gif](assets/ldmvfi.gif)


## Overview
We observe that most existing learning-based VFI models are trained to minimise the L1/L2/VGG loss between their outputs and the ground-truth frames. However, it was shown in previous works that these metrics do not correlate well with the **perceptual quality** of VFI. On the other hand, generative models, especially diffusion models, are showing remarkable results in generating visual content with high perceptual quality. In this work, we leverage the high-fidelity image/video generation capabilities of **latent diffusion models** to perform generative VFI.
<p align="center">
<img src="https://danier97.github.io/LDMVFI/overall.svg" alt="Paper" width="60%">
</p>

## Dependencies and Installation
See [environment.yaml](./environment.yaml) for requirements on packages. Simple installation:
```
conda env create -f environment.yaml
```

## Pre-trained Model
The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1_Xx2fBYQT9O-6O3zjzX76O9XduGnCh_7/view?usp=share_link), and its corresponding config file is [this yaml](./configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml).


## Preparing datasets
### Training sets:
[[Vimeo-90K]](http://toflow.csail.mit.edu/) | [[BVI-DVC quintuplets]](https://drive.google.com/file/d/1i_CoqiNrZ2AU8DKjU8aHM1jIaDGW0fE5/view?usp=sharing)

### Test sets: 
[[Middlebury]](https://vision.middlebury.edu/flow/data/) | [[UCF101]](https://sites.google.com/view/xiangyuxu/qvi_nips19) | [[DAVIS]](https://sites.google.com/view/xiangyuxu/qvi_nips19) | [[SNU-FILM]](https://myungsub.github.io/CAIN/)


To make use of the [evaluate.py](evaluate.py) and the files in [ldm/data/](./ldm/data/), the dataset folder names should be lower-case and structured as follows.
```
└──── <data directory>/
    ├──── middlebury_others/
    |   ├──── input/
    |   |   ├──── Beanbags/
    |   |   ├──── ...
    |   |   └──── Walking/
    |   └──── gt/
    |       ├──── Beanbags/
    |       ├──── ...
    |       └──── Walking/
    ├──── ucf101/
    |   ├──── 0/
    |   ├──── ...
    |   └──── 99/
    ├──── davis90/
    |   ├──── bear/
    |   ├──── ...
    |   └──── walking/
    ├──── snufilm/
    |   ├──── test-easy.txt
    |   ├──── ...
    |   └──── data/SNU-FILM/test/...
    ├──── bvidvc/quintuplets
    |   ├──── 00000/
    |   ├──── ...
    |   └──── 17599/
    └──── vimeo_septuplet/
        ├──── sequences/
        ├──── sep_testlist.txt
        └──── sep_trainlist.txt
```

## Evaluation

To evaluate LDMVFI (with DDIM sampler), for example, on the Middlebury dataset, using PSNR/SSIM/LPIPS, run the following command.
```
python evaluate.py \
--config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
--ckpt <path/to/ldmvfi-vqflow-f32-c256-concat_max.ckpt> \
--dataset Middlebury_others \
--metrics PSNR SSIM LPIPS \
--data_dir <path/to/data/dir> \
--out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max/ \
--use_ddim
```
This will create the directory `eval_results/ldmvfi-vqflow-f32-c256-concat_max/Middlebury_others/`, and store the interpolated frames, as well as a `results.txt` file in that directory. For other test sets, replace `Middlebury_other` with the corresponding class names defined in [ldm/data/testsets.py](ldm/data/testsets.py) (e.g. `Ucf101_triplet`).

\
To evaluate the model on perceptual video metric FloLPIPS, first evaluate the image metrics using the code above (so that the interpolated frames are saved in `eval_results/ldmvfi-vqflow-f32-c256-concat_max`), then run the following code.
```
python evaluate_vqm.py \
--exp ldmvfi-vqflow-f32-c256-concat_max \
--dataset Middlebury_others \
--metrics FloLPIPS \
--data_dir <path/to/data/dir> \
--out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max/ \
```
This will read the interpolated frames previously stored in `eval_results/ldmvfi-vqflow-f32-c256-concat_max/Middlebury_others/` then output the evaluation results to `results_vqm.txt` in the same folder.

\
To interpolate a video (in .yuv format), use the following code.
```
python interpolate_yuv.py \
--net LDMVFI \
--config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
--ckpt <path/to/ldmvfi-vqflow-f32-c256-concat_max.ckpt> \
--input_yuv <path/to/input/yuv> \
--size <spatial res of video, e.g. 1920x1080> \
--out_fps <output fps, should be 2 x original fps> \
--out_dir <desired/output/dir> \
--use_ddim
```

## Training
LDMVFI is trained in two stages, where the VQ-FIGAN and the denoising U-Net are trained separately.
### VQ-FIGAN
```
python main.py --base configs/autoencoder/vqflow-f32.yaml -t --gpus 0,
```
### Denoising U-Net
```
python main.py --base configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml -t --gpus 0,
```
These will create a `logs/` folder within which the corresonding directories are created for each experiment. The log files from training include checkpoints, images and tensorboard loggings.

To resume from a checkpoint file, simply use the `--resume` argument in [main.py](main.py) to specify the checkpoint.


## Citation
```
@article{danier2023ldmvfi,
  title={LDMVFI: Video Frame Interpolation with Latent Diffusion Models},
  author={Danier, Duolikun and Zhang, Fan and Bull, David},
  journal={arXiv preprint arXiv:2303.09508},
  year={2023}
}
```

## Acknowledgement
Our code is adapted from the original [latent-diffusion](https://github.com/CompVis/latent-diffusion) repository. We thank the authors for sharing their code.