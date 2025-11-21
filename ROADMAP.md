# Tinify Roadmap

## Completed

- [x] **Rust Extensions (PyO3)** - Ported C++ pybind11 extensions to Rust
  - rANS entropy encoder/decoder
  - pmf_to_quantized_cdf operation
  - Python wrappers with automatic backend selection

- [x] **MkDocs Documentation** - Set up documentation site with GitHub Actions deployment

- [x] **CLI for Training** - Command-line interface for model training

## In Progress

- [ ] **Code Cleanup** - Remove unused legacy code, standardize imports

## Planned

### Triton Kernels (triton-lang)

Custom GPU kernels using [Triton](https://github.com/triton-lang/triton) for performance-critical operations:

- [ ] **Fused Entropy Coding** - GPU-accelerated rANS encode/decode
- [ ] **Fused GDN/IGDN** - Generalized Divisive Normalization as single kernel
- [ ] **Attention Kernels** - Optimized attention for transformer-based models (see [QAttn](https://github.com/IBM/qattn) for mixed-precision ViT kernels)
- [ ] **Quantization Kernels** - Fused quantize + noise injection for training
- [ ] **Convolution Fusions** - Conv + activation + normalization fused ops
- [ ] **Checkerboard Context** - Parallel entropy model decoding kernel

Benefits:
- Eliminate CPU-GPU memory transfers for entropy coding
- Reduce kernel launch overhead
- Custom memory layouts optimized for compression workloads

### Lightning Fabric Integration

Migrate training infrastructure to [Lightning Fabric](https://lightning.ai/docs/fabric/stable/):

- [ ] **Multi-GPU Training** - Simplified DDP/FSDP support
- [ ] **Mixed Precision** - Native FP16/BF16 training with automatic scaling
- [ ] **Checkpointing** - Unified checkpoint format with automatic resume
- [ ] **Logging Integration** - TensorBoard, W&B, MLflow support
- [ ] **Distributed Inference** - Model parallel inference for large models

Benefits:
- Cleaner training code without boilerplate
- Easy switching between single-GPU, DDP, FSDP
- Built-in gradient accumulation and clipping
- Hardware-agnostic code (CUDA, MPS, TPU)

### Model Architectures

#### Variable Rate Compression
Single model supporting multiple bitrates without retraining:

- [ ] **Gain Units** - Channel-wise scaling factors for rate control ([G-VAE](https://arxiv.org/abs/2003.05050))
- [ ] **Multi-Scale Gain** - Combined gain units + quantization step size
- [ ] **Quantization-Reconstruction Offsets** - Improved VBR via QR offsets ([DCC 2024](https://arxiv.org/abs/2402.18930))

#### Asymmetric Encoder-Decoder
Lightweight decoders for mobile/edge deployment:

- [ ] **AsymLLIC Architecture** - Heavy encoder, light decoder ([arXiv 2412.17270](https://arxiv.org/abs/2412.17270))
- [ ] **Progressive Module Substitution** - Gradual decoder simplification during training
- [ ] **Mobile Decoder Targets** - <100 GMACs for 1080p decoding

#### Advanced Entropy Models
Parallel-friendly context models for faster decoding:

- [ ] **Checkerboard Context** - 2-pass parallel decoding ([He et al. CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Checkerboard_Context_Model_for_Efficient_Learned_Image_Compression_CVPR_2021_paper.pdf))
- [ ] **Channel-Wise Autoregressive** - Parallel across spatial dims ([Minnen & Singh](https://arxiv.org/abs/2007.08739))
- [ ] **ELIC Integration** - Unevenly grouped space-channel coding ([CVPR 2022](https://arxiv.org/abs/2203.10886))
- [ ] **Corner-to-Center Context** - Logarithmic decoding order

#### Diffusion-Based Decoders
Perceptual quality via generative models:

- [ ] **CDC Integration** - Conditional diffusion decoder ([Yang et al. NeurIPS 2023](https://arxiv.org/abs/2209.06950))
- [ ] **ResCDC** - Residual diffusion for better fidelity (+2dB PSNR)
- [ ] **Distortion-Perception Tradeoff** - Controllable quality slider

#### Transformer Architectures
State-of-the-art transform coding:

- [ ] **SwinT-ChARM** - Swin transformer with channel-wise ARM ([ICLR 2022](https://openreview.net/forum?id=IDwN6xjHnK8))
- [ ] **LIC-TCM** - Mixed transformer-CNN architecture ([CVPR 2023](https://github.com/jmliu206/LIC_TCM))
- [ ] **Entroformer** - Transformer-based entropy model

### Infrastructure & Deployment

#### PyTorch Optimizations
- [ ] **torch.compile Integration** - Up to 30% inference speedup
- [ ] **Fullgraph Mode** - Single-graph compilation for max performance
- [ ] **Dynamic Shapes** - Handle variable input resolutions

#### Export & Inference
- [ ] **ONNX Export** - Export trained models to ONNX format
- [ ] **TensorRT Integration** - Optimized inference with TensorRT
- [ ] **OpenVINO Support** - Intel hardware optimization
- [ ] **CoreML Export** - Apple Silicon deployment

#### Benchmarking
- [ ] **Benchmark Suite** - Standardized benchmarks (Kodak, CLIC, Tecnick)
- [ ] **BD-Rate Calculator** - Automated Bjontegaard delta metrics
- [ ] **Latency Profiling** - Per-layer timing analysis
- [ ] **Pre-trained Model Hub** - Easy access to pre-trained checkpoints

### Standards Compliance

#### JPEG AI (ISO/IEC 6048)
First neural network-based image compression standard:

- [ ] **Reference Decoder** - Compatible with JPEG AI bitstreams
- [ ] **Profile Support** - Implement core profiles and levels
- [ ] **Conformance Testing** - Validate against reference software
- [ ] **Artifact Detection** - Tools to detect JPEG AI compression artifacts

#### Video Coding for Machines (VCM)
Feature compression for machine vision tasks:

- [ ] **Multi-Scale Feature Compression** - MPEG-VCM compatible
- [ ] **Joint Human-Machine Coding** - Optimize for both perception and analysis
- [ ] **Task-Specific Bitstreams** - Detection, segmentation, classification

## Future Considerations

### Alternative Frameworks

- [ ] **Consider JAX** - Functional transforms, XLA compilation, TPU support
  - Better JIT compilation than PyTorch for some workloads
  - Native support for vectorization (vmap) and parallelization (pmap)
  - Potential for faster entropy model training

### Extended Modalities

- [ ] **Video Compression** - DCVC-series models, temporal context
- [ ] **Point Cloud Compression** - 3D data for autonomous vehicles, AR/VR
- [ ] **Neural Audio Compression** - Codec integration (EnCodec, DAC)
- [ ] **NeRF/Gaussian Compression** - 3D scene representation compression

### Research Directions

- [ ] **Ultra-Low Bitrate** - Semantic compression with LLM guidance (MISC)
- [ ] **One-Step Diffusion** - StableCodec for extreme compression
- [ ] **Content-Adaptive Models** - Per-image model selection
- [ ] **Neural Codec for Features** - Compress intermediate representations

## References

Recent papers to integrate:

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| [ELIC](https://arxiv.org/abs/2203.10886) | CVPR 2022 | Efficient space-channel context |
| [Checkerboard Context](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Checkerboard_Context_Model_for_Efficient_Learned_Image_Compression_CVPR_2021_paper.pdf) | CVPR 2021 | 2-pass parallel decoding |
| [CDC](https://arxiv.org/abs/2209.06950) | NeurIPS 2023 | Diffusion decoder |
| [AsymLLIC](https://arxiv.org/abs/2412.17270) | arXiv 2024 | Asymmetric lightweight codec |
| [VBR with QR Offsets](https://arxiv.org/abs/2402.18930) | DCC 2024 | Variable rate improvements |
| [EVC](https://openreview.net/forum?id=XUxad2Gj40n) | ICLR 2023 | Real-time neural codec |
| [QAttn](https://github.com/IBM/qattn) | CVPRW 2024 | Mixed-precision ViT kernels |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this roadmap.
