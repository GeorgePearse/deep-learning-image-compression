# References

A curated collection of research papers and resources on learned/neural image compression.

## Foundational Papers

These papers established the core architectures and techniques used in modern learned image compression.

| Paper | Authors | Venue | Links |
|-------|---------|-------|-------|
| **End-to-end Optimized Image Compression** | Ballé, Laparra, Simoncelli | ICLR 2017 | [arXiv](https://arxiv.org/abs/1611.01704) |
| **Variational Image Compression with a Scale Hyperprior** | Ballé et al. | ICLR 2018 | [arXiv](https://arxiv.org/abs/1802.01436) |
| **Joint Autoregressive and Hierarchical Priors for Learned Image Compression** | Minnen, Ballé, Toderici | NeurIPS 2018 | [arXiv](https://arxiv.org/abs/1809.02736) \| [GitHub](https://github.com/mengab/Joint-Autoregressive-and-Hierarchical-Priors-for-Learned-Image-Compression) |
| **Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules** | Cheng et al. | CVPR 2020 | [arXiv](https://arxiv.org/abs/2001.01568) \| [GitHub](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention) |

## Transformer-Based Methods

| Paper | Authors | Venue | Links |
|-------|---------|-------|-------|
| **Transformer-based Transform Coding (SwinT-ChARM)** | Zhu et al. | ICLR 2022 | [OpenReview](https://openreview.net/forum?id=IDwN6xjHnK8) \| [GitHub](https://github.com/Nikolai10/SwinT-ChARM) |
| **Entroformer: A Transformer-based Entropy Model for Learned Image Compression** | Qian et al. | ICLR 2022 | [OpenReview](https://openreview.net/forum?id=VrjOFfcnSV8) |
| **Transformer-based Image Compression** | Lu et al. | DCC 2022 | [arXiv](https://arxiv.org/abs/2111.06707) \| [GitHub](https://github.com/lumingzzz/TIC) |
| **Learned Image Compression with Mixed Transformer-CNN Architectures** | Liu et al. | CVPR 2023 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learned_Image_Compression_With_Mixed_Transformer-CNN_Architectures_CVPR_2023_paper.pdf) \| [GitHub](https://github.com/jmliu206/LIC_TCM) |

## Generative Methods (GAN / Diffusion)

| Paper | Authors | Venue | Links |
|-------|---------|-------|-------|
| **High-Fidelity Generative Image Compression (HiFiC)** | Mentzer et al. | NeurIPS 2020 | [Project](https://hific.github.io/) \| [PDF](https://papers.neurips.cc/paper_files/paper/2020/file/8a50bae297807da9e97722a0b3fd8f27-Paper.pdf) |
| **Lossy Image Compression with Conditional Diffusion Models (CDC)** | Yang et al. | NeurIPS 2023 | [arXiv](https://arxiv.org/abs/2209.06950) \| [GitHub](https://github.com/buggyyang/CDC_compression) |

## Recent Advances (2024-2025)

| Paper | Venue | Links |
|-------|-------|-------|
| **Neural Image Compression with Text-guided Encoding for both Pixel-level and Perceptual Fidelity** | 2024 | [arXiv](https://arxiv.org/abs/2403.02944) |
| **On Efficient Neural Network Architectures for Image Compression** | 2024 | [arXiv](https://arxiv.org/abs/2406.10361) |
| **Causal Context Adjustment Loss for Learned Image Compression** | NeurIPS 2024 | [GitHub](https://github.com/LabShuHangGU/CCA) |
| **EVC: Towards Real-Time Neural Image Compression with Mask Decay** | 2023 | [arXiv](https://arxiv.org/abs/2302.05071) |
| **WeConvene: Learned Image Compression with Wavelet-Domain Convolution and Entropy Model** | ECCV 2024 | [GitHub](https://github.com/fengyurenpingsheng/WeConvene) |
| **Enhanced Invertible Encoding for Learned Image Compression (InvCompress)** | ACM MM 2021 | [GitHub](https://github.com/xyq7/InvCompress) |

## Key Libraries & Frameworks

| Library | Description | Links |
|---------|-------------|-------|
| **CompressAI** | PyTorch library and evaluation platform for end-to-end compression research (InterDigital) | [GitHub](https://github.com/InterDigitalInc/CompressAI) \| [Docs](https://interdigitalinc.github.io/CompressAI/) |
| **tensorflow/compression** | TensorFlow library for learned compression | [GitHub](https://github.com/tensorflow/compression) |
| **CompressAI-Trainer** | Training platform for end-to-end compression models | [GitHub](https://github.com/InterDigitalInc/CompressAI-Trainer) |

## Paper Collections & Resources

| Repository | Description |
|------------|-------------|
| [Awesome-Deep-Compression](https://github.com/wyf0912/Awesome-Deep-Compression) | Paper list of deep learning based image/video compression |
| [Deep-Learning-Based-Image-Compression](https://github.com/ppingzhang/Deep-Learning-Based-Image-Compression) | Paper list about deep learning based image compression |
| [Learned-Image-Video-Compression](https://github.com/cshw2021/Learned-Image-Video-Compression) | Collection of papers related to data compression |
| [Image-Compression-Papers-and-Code](https://github.com/zhiqiang-zhu/Image-Compression-Papers-and-Code) | Papers with code implementations |

## Review Papers

| Paper | Links |
|-------|-------|
| **Image and Video Compression with Neural Networks: A Review** | [arXiv](https://arxiv.org/abs/1904.03567) |
| **Deep Architectures for Image Compression: A Critical Review** | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165168421003832) |
| **Information Compression in the AI Era: Recent Advances and Future Challenges** | [arXiv](https://arxiv.org/html/2406.10036v1) |

## Related Links

- [Tensorflow compression library](https://github.com/tensorflow/compression) by Ballé et al.
- [Range ANS code](https://github.com/rygorous/ryg_rans) by Fabian 'ryg' Giesen
- [BPG image format](https://bellard.org/bpg) by Fabrice Bellard
- [HEVC HM reference software](https://hevc.hhi.fraunhofer.de)
- [VVC VTM reference software](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
- [AOM AV1 reference software](https://aomedia.googlesource.com/aom)
- [Kodak image dataset](http://r0k.us/graphics/kodak/)

## Citing CompressAI

```bibtex
@article{begaint2020compressai,
    title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
    author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
    year={2020},
    journal={arXiv preprint arXiv:2011.03029},
}
```

For variable bitrate models:

```bibtex
@article{kamisli2024dcc_vbrlic,
    title={Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets},
    author={Kamisli, Fatih and Racap{\'e}, Fabien and Choi, Hyomin},
    year={2024},
    booktitle={2024 Data Compression Conference (DCC)},
    eprint={2402.18930},
}
```
