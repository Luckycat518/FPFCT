# FPFCT：For Industrial Image Super Resolution<sup>📌</sup>
<a href="https://github.com/Luckycat518"><img src="https://img.shields.io/badge/GitHub-@Luckycat518-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
<a href="https://charmve.github.io/computer-vision-in-action/" target="_blank"><img src="https://img.shields.io/badge/Computer Vision-000000.svg?logo=GitBook" alt="Computer Vision in Action"></a>
[![License](https://img.shields.io/github/license/Charmve/Surface-Defect-Detection)](LICENSE)

# Table of Contents

- [Introduction](#introduction)
- [Comparison with SOTA methods](#1-Comparison-with-SOTA-methods)
  - [In public datasets](#1-In-public-datasets)
  - [In WCI110](#2-In-WCI110)
- [Our Dataset](#2-our-dataset-for-Industrial-Image-Super-Resolution)
  - [Welding Component Images: WCI110](#1-Welding-Component-Images-WCI110)
- [Notification](#notification)
- [Citation](#citation)

## Notification：Something wrong for uploading files，the authors are trying their best to solve this problem！--20230531

## Introduction


<p>For the smart manufacturing, lots of studies focus on high-quilty surface detection under diverse industrial environments. However, for surface detection, besides the performance of advanced detectors, the image quality/resolution is also the key factor affecting the detection performance. Considering the urgency of the need for high-quality images when performing surface detection in complex and changeable industrial environments, it is necessary to propose a lightweight image processing method for effectively improving the input image quality for surface detection with only a small increase in time and computational cost. This method can reduce the dependence of the detection system on the environment and image collection equipment, leading to significant economical benefits on the industrial detection system design and deployment. Thus, we innovatively proposed a lightweight SR structure by parallelly fusing CNN feature extraction module and Transformer feature extraction module (FPFCT). Different from the serial structure that uses CNN and transformer to extract features successively, the parallel fusion structure can effectively avoid the irreparable information loss caused by the inheritable sampling imperfection of two different methods. Moreover, an industrial image SR dataset called <strong>WCI110</strong>, consisting of 110 typical welding component images with 2040×1524 pixels, is established for the vertification of the proposed method. </p>




## 1. Our dataset for Industrial Image Super Resolution

### 1）Welding Component Images: WCI110

WCI110 can be used for image super resolution tasks.

![image](https://github.com/Luckycat518/FPFCT/blob/main/Cover_Image/dataset_description.jpg)
<div align=center><img src="https://github.com/Luckycat518/FPFCT/blob/main/Cover_Image/dataset_description.jpg"></div>

<p>The WCI110 contains <b>110</b> pictures with a total of <b>110</b> HR welding component surface images under 2040×1524 pixels. For this dataset, there are three typical scenarios, including global view with component structure background, and detail views focusing on the unground weld and ground weld. </p>



## 2. Comparison with SOTA methods

### 1）In public datasets

<p> The PSNR and SSIM of FPFCT are higher than other advanced SR methods and comparable to that of the SOTA SwinIR-light model almost in all the five public standard datasets and the self-established industrial dataset in all three scales. However, the model parameters are reduced by more than 37.5%, and the FLOPs are reduced by more than 25.6% compared to SwinIR-light in all three scales. </p>

![image](https://github.com/Luckycat518/FPFCT/blob/main/Cover_Image/comparison.jpg)
<div align=center><img src="https://github.com/Luckycat518/FPFCT/blob/main/Cover_Image/comparison.jpg"></div>

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)

### 2）In WCI110

<p> For three typical target, including workpiece scratches, surface rust and weld corrugation characteristics, the super-resolution repair ability of FPFCT is comparable to that of SwinIR-light, and it is ahead of other advanced methods. Moreover, it can be noticed that compared with the slimming target in SR image obtained by SwinIR-light, the target of FPFCT is closer to ground truth (GT) in terms of size. </p>

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)


## Notification
<b>The first version of the dataset containing the original images appearing in the paper has been released.</b>
<p>The WCI110 is collected from a commercial company. Therefore, this dataset is limited to academic use！！！Prohibited for any commercial use！！！
<strong>You can only use this dataset for research purpose.</strong>
Before the entire dataset is fully released, if you need this dataset to do some research, please contact us. After review and permission by the commercial company, we will provide the dataset to you.</p>


If you have any questions or idea, please let me know <p>(email: wgq001@csu.edu.cn)</p>

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)

## Citation
This paper is still under review......

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)



