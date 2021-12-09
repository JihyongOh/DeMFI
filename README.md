# DeMFI

**This is the official repository of DeMFI (Deep Joint Deblurring and Multi-Frame Interpolation).**

\[[ArXiv_ver.](https://arxiv.org/abs/2111.09985)\] \[[Demo](https://youtu.be/J93tW1uwRy0)\]

Last Update: 20211206

<!-- **Reference**:   -->
## Reference
> Jihyong Oh and Munchurl Kim "DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting", _arXiv preprint arXiv: 2111.09985_, 2021. 
> 
**BibTeX**
```bibtex
@article{Oh2021DeMFI,
  title={DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting},
  author={Oh, Jihyong and Kim, Munchurl},
  journal={arXiv preprint arXiv:2111.09985},
  year={2021}
}
```

If you find this repository useful, please consider citing our [paper](https://arxiv.org/abs/2111.09985).

**All source codes (train+test) including checkpoints will be provided soon.**

### Examples of the Demo (x8 Multi-Frame Interpolation) videos (240fps) interpolated from blurry videos (30fps)
![gif1](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif1.gif "gif1")
![gif2](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif2.gif "gif2")
![gif3](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif3.gif "gif3")
![gif4](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif4.gif "gif4")
![gif5](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif5.gif "gif5")

The 30fps blurry input frames are interpolated to be 240fps sharp frames. All results are encoded at 30fps to be played as x8 slow motion and spatially down-scaled due to the limit of file sizes. Please watch the full versions of them with this [demo](https://youtu.be/J93tW1uwRy0) including additional scenes.

## Table of Contents
1. [Requirements](#Requirements)
1. [Test](#Test)
1. [Test_Custom](#Test_Custom)
1. [Training](#Training)
1. [Collection_of_Visual_Results](#Collection_of_Visual_Results)
1. [Contact](#Contact)

## Requirements
Our code is implemented using PyTorch1.7, and was tested under the following setting:  
* Python 3.7 
* PyTorch 1.7.1
* CUDA 10.2  
* cuDNN 7.6.5  
* NVIDIA TITAN RTX GPU
* Ubuntu 16.04 LTS

**Caution**: since there is "align_corners=True" option in "nn.functional.interpolate" and "nn.functional.grid_sample" in PyTorch1.7, we recommend you to follow our settings.
Especially, if you use the other PyTorch versions, it may lead to yield a different performance.



## Test
### Quick Start for Evaluations on Test Datasets (Deblurring and Multi-Frame Interpolation (x8) as in Table 2)
1. Download the source codes in a directory of your choice **\<source_path\>**.
2. We follow a blurry formation setting from [BIN (Blurry Video Frame Interpolation)](https://github.com/laomao0/BIN#testing-pre-trained-models) by averaging 11 consecutive frames at a stride of 8 frames over time to synthesize blurry frames captured by a long exposure, which finally generates blurry frames of 30fps with K = 8 and τ = 5 in Eq. 1. We thank the authors for sharing codes for their awesome works.
3. Download datasets from the dropbox links; Adobe240 ([main](https://www.dropbox.com/s/n4uc5tlik96begy/Adobe_240fps_blur.zip?dl=0), [split1](https://www.dropbox.com/s/wmd78jaob2lxpv4/Adobe_240fps_blur.z01?dl=0), [split2](https://www.dropbox.com/s/jlvfl70gs7cyrwn/Adobe_240fps_blur.z02?dl=0), [split3](https://www.dropbox.com/s/xrn52zhftojq6lx/Adobe_240fps_blur.z03?dl=0), [split4](https://www.dropbox.com/s/9xgtg1dfjb5nnyx/Adobe_240fps_blur.z04?dl=0)) (split zip files, 49.7GB), [GoPro(HD)](https://www.dropbox.com/s/x9x36esur8rsvj7/GoPro_blur.zip?dl=0) (14.4GB). Since the copyrights for diverse videos of YouTube240 belong to each creator, we appreciate your understanding that it cannot be distributed. Original copyrights for Adobe240 and GoPro are provided via [link1](https://github.com/shuochsu/DeepVideoDeblurring) and [link2](https://github.com/SeungjunNah/DeepDeblur_release), respectively.
4. Directory formats seem like below:
```
DeMFI
└── Datasets
      ├──── Adobe_240fps_blur
         ├──── test
             ├──── 720p_240fps_1
                 ├──── 00001.png
                 ├──── ...
                 └──── 00742.png
             ...
             ├──── IMG_0183           
         ├──── test_blur
            ├──── ...          
         ├──── train
            ├──── ...          
         ├──── train_blur 
            ├──── ...
``` 
5. Download the pre-trained weight of DeMFI-Net<sub>*rb*</sub>(5,N<sub>*tst*</sub>), which was trained by Adobe240, from [this link](https://www.dropbox.com/s/xj2ixvay0e5ldma/XVFInet_X4K1000FPS_exp1_latest.pt?dl=0) to place in **\<source_path\>/checkpoint_dir/DeMFInet_exp1**.
```
DeMFI
└── checkpoint_dir
   └── DeMFInet_exp1
         ├── DeMFInet_exp1_latest.pt           
```
6. Run **main.py** with the following options in parse_args: 
```bash
# For evaluating on Adobe240
python main.py --gpu 0 --phase 'test' --exp_num 1 --test_data_path './Datasets/Adobe_240fps_blur' --N_tst 3 --multiple_MFI 8 
```
```bash
# For evaluating on GoPro(HD)
python main.py --gpu 0 --phase 'test' --exp_num 1 --test_data_path './Datasets/GoPro_blur' --N_tst 3 --multiple_MFI 8 
```
* The quantitative comparisons (Table2) are attached as belows for a reference.
![Table2](/figures/Table2.PNG "Table2")


### Description
* After running with the above test option, you can get the sharp frames in **\<source_path\>/test_img_dir/DeMFInet_exp1/epoch_07499_final_x8_full_resolution_Ntst3**, then obtain the PSNR/SSIM results per each time index in the screen. 
* Our proposed DeMFI-Net we can properly regulate '--N_tst' by considering R<sub>*t*</sub> (runtime) or computational constraints, even though the training with N<sub>*trn*</sub> is once over. Further details are described in the main paper.
* You can only get Multi-Frame Interpolation (x M) result by regulating '--multiple' as 2 or 8 for evaluation but any M can be chosen for 'test_custom' option, please refer [Test_Custom](#Test_Custom).

## Test_Custom
### Quick Start for your own **blurry** video data ('--custom_path') for any Multi-Frame Interpolation (x M)


## Contact
If you have any question, please send an email to [[Jihyong Oh](https://sites.google.com/view/ozbro)] - jhoh94@kaist.ac.kr.
