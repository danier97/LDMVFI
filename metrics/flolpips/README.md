# FloLPIPS: A bespoke video quality metric for frame interpoation

### Duolikun Danier, Fan Zhang, David Bull


[Project](https://danielism97.github.io/FloLPIPS) | [arXiv](https://arxiv.org/abs/2207.08119)


## Dependencies
The following packages were used to evaluate the model.

- python==3.8.8
- pytorch==1.7.1
- torchvision==0.8.2
- cudatoolkit==10.1.243
- opencv-python==4.5.1.48
- numpy==1.19.2
- pillow==8.1.2
- cupy==9.0.0


## Usage
```python
from flolpips import calc_flolpips
ref_video = '<path to the reference>.mp4'
dis_video = '<path to the distorted>.mp4'
res = calc_flolpips(dis_video, ref_video)
```


## Citation
```
@article{danier2022flolpips,
  title={FloLPIPS: A Bespoke Video Quality Metric for Frame Interpoation},
  author={Danier, Duolikun and Zhang, Fan and Bull, David},
  journal={arXiv preprint arXiv:2207.08119},
  year={2022}
}
```

## Acknowledgement
Lots of code in this repository are adapted/taken from the following repositories:

- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc)

We would like to thank the authors for sharing their code.