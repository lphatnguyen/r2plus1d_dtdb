# Dynamic texture classification with 3D CNN

## Introduction
This repository aims to classify dynamic texture video using the [DTDB dataset](http://vision.eecs.yorku.ca/research/dtdb/). The main purpose of this work is to use a 3D CNN architecture called [R(2+1)D](https://arxiv.org/abs/1711.11248), which is originally used for action recognition, using two type of inputs: RGB frames and DoG(Difference of Gaussian) frames. 

## Description of the code
For the size of frames, we resize the frames spatially to a maximum size of 256 pixels (either the width or height of the frame). Furthermore, for the DoG frames, we first convert the RGB frames to grayscale frames, then we compute the DoG responses using 5 scales with a factor of \sqrt{2}. For temporal axis, we downsample videos with over 30 FPS to 25 FPS. We keep the sampling rate if the its FPS is under 30. 

We trained two models seperately for both kinds of inputs before applying a late fusion after computing the softmax scores. The two pretrained weights for both inputs of RGB and DoG can be found in these following links:
- [Pretrained weights for RGB inputs](https://drive.google.com/file/d/16SB-Qtmvdff4f4BdMS2cujYI8OTLBgm1/view?usp=sharing)
- [Pretrained weights for DoG inputs](https://drive.google.com/file/d/1kbpuZRvP-tp68YLj9eNeCBT0BRB6xPE9/view?usp=sharing)

## Requirements
- Python >= 3.6
- PyTorch >= 1.5.0
- torchvision >= 0.8.1
- OpenCV 
- Numpy >= 1.20.3
- tqdm

## Citation 
If you use this code for publication, please cite:

```bibtex
@misc{lphatnguyen2020,
    author  = {Luong Phat Nguyen and Julien Mille and Dominique Li and Donatello Conte and Nicolas Ragot},
    year    = {2020},
    url     = {https://github.com/lphatnguyen/r2plus1d_dtdb}
```