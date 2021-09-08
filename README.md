# Dynamic texture classification with 3D CNN

## Introduction
This repository aims to classify dynamic texture video using the [DTDB dataset](http://vision.eecs.yorku.ca/research/dtdb/). The main purpose of this work is to use a 3D CNN architecture called [R(2+1)D](https://arxiv.org/abs/1711.11248), which is originally used for action recognition, using two type of inputs: RGB frames and DoG(Difference of Gaussian) frames. 

## Description of the code
For the size of frames, we resize the frames spatially to a maximum size of 256 pixels (either the width or height of the frame). Furthermore, for the DoG frames, we first convert the RGB frames to grayscale frames, then we compute the DoG responses using 5 scales with a factor of \sqrt{2}. 

We trained two models seperately for both kinds of inputs before applying a late fusion after computing the softmax scores. The two models can be found in 