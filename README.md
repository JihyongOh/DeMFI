# DeMFI

**This is the official repository of DeMFI (Deep Joint Deblurring and Multi-Frame Interpolation).**

\[[ArXiv_ver.](https://arxiv.org/abs/2111.09985)\] \[[Demo](https://youtu.be/J93tW1uwRy0)\]

Last Update: 20211206

If you find this repository useful, please consider citing our [paper](https://arxiv.org/abs/2111.09985).

**All source codes (train+test) including checkpoints will be provided soon.**

### Examples of the Demo (x8 Multi-Frame Interpolation) videos (240fps) interpolated from blurry videos (30fps)
![gif1](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif1.gif "gif1")
![gif2](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif2.gif "gif2")
![gif3](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif3.gif "gif3")
![gif4](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif4.gif "gif4")
![gif5](/figures/Demo_DeMFI-Net_vs_SOTA_sup_gif5.gif "gif5")

The 30fps blurry input frames are interpolated to be 240fps sharp frames. All results are encoded at 30fps to be played as x8 slow motion and spatially down-scaled due to the limit of file sizes. Please watch the full versions of them with this [demo](https://youtu.be/J93tW1uwRy0) including additional scenes.

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

## Contact
If you have any question, please send an email to [[Jihyong Oh](https://sites.google.com/view/ozbro)] - jhoh94@kaist.ac.kr.
