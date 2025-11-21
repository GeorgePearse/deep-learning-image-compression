# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.hub import load_state_dict_from_url

from tinify.models import (
    JointAutoregressiveHierarchicalPriorsVbr,
    MeanScaleHyperpriorVbr,
    ScaleHyperpriorVbr,
)

from .pretrained import load_pretrained

__all__ = [
    "bmshj2018_hyperprior_vbr",
    "mbt2018_mean_vbr",
    "mbt2018_vbr",
]

model_architectures: dict[str, type[nn.Module]] = {
    "bmshj2018-hyperprior-vbr": ScaleHyperpriorVbr,
    "mbt2018-mean-vbr": MeanScaleHyperpriorVbr,
    "mbt2018-vbr": JointAutoregressiveHierarchicalPriorsVbr,
}

root_url: str = "https://tinify.s3.amazonaws.com/models/v1"
model_urls: dict[str, dict[str, str]] = {
    "bmshj2018-hyperprior-vbr": {
        "mse": f"{root_url}/bmshj2018-hyperprior-mse-vbr-cddd26be.pth.tar",
        # "ms-ssim": f"{root_url}/bmshj2018-hyperprior-ms-ssim-vbr-HASH.pth.tar",
    },
    "mbt2018-mean-vbr": {
        "mse": f"{root_url}/mbt2018-mean-mse-vbr-45d095e9.pth.tar",
        # "ms-ssim": f"{root_url}/mbt2018-mean-ms-ssim-vbr-HASH.pth.tar",
    },
    "mbt2018-vbr": {
        "mse": f"{root_url}/mbt2018-mse-vbr-f12581a1.pth.tar",
        # "ms-ssim": f"{root_url}/mbt2018-ms-ssim-vbr-HASH.pth.tar",
    },
}

cfgs: dict[str, tuple[int, int]] = {
    "bmshj2018-hyperprior-vbr": (192, 320),
    "mbt2018-mean-vbr": (192, 320),
    "mbt2018-vbr": (192, 320),
}


def _load_model(
    architecture: str,
    metric: str,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if pretrained:
        if architecture not in model_urls or metric not in model_urls[architecture]:
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric]
        state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = load_pretrained(state_dict)
        if architecture in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
            model = model_architectures[architecture].from_state_dict(
                state_dict, vr_entbttlnck=True
            )
        else:
            model = model_architectures[architecture].from_state_dict(state_dict)
        return model

    model = model_architectures[architecture](*cfgs[architecture], **kwargs)
    return model


def bmshj2018_hyperprior_vbr(
    quality: int = 0,
    metric: str = "mse",
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""Variable bitrate (vbr) version of bmshj2018-hyperprior (see tinify/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.

    Args:
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # keep quality here for now for consistency with image models.
    # could be used to indicate which lambda or quality was used to train the base model.
    del quality
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    return _load_model(
        "bmshj2018-hyperprior-vbr", metric, pretrained, progress, **kwargs
    )


def mbt2018_mean_vbr(
    quality: int = 0,
    metric: str = "mse",
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""Variable bitrate (vbr) version of bmshj2018 (see tinify/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.

    Args:
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # keep quality here for now for consistency with image models.
    # could be used to indicate which lambda or quality was used to train the base model.
    del quality
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    return _load_model("mbt2018-mean-vbr", metric, pretrained, progress, **kwargs)


def mbt2018_vbr(
    quality: int = 0,
    metric: str = "mse",
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""Variable bitrate (vbr) version of mbt2018 (see tinify/models/google.py) with variable bitrate components detailed in:
    Fatih Kamisli, Fabien Racape and Hyomin Choi
    `"Variable-Rate Learned Image Compression with Multi-Objective Optimization and Quantization-Reconstruction Offsets`"
    <https://arxiv.org/abs/2402.18930>`_, Data Compression Conference (DCC), 2024.

    Args:
        metric (str): Optimized metric, choose from ('mse', no 'ms-ssim' yet)
        pretrained (bool): If True, returns a pre-trained model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # keep quality here for now for consistency with image models.
    # could be used to indicate which lambda or quality was used to train the base model.
    del quality
    if metric not in ("mse"):  # ("mse", "ms-ssim"): # we have only mse model
        raise ValueError(f'Invalid metric "{metric}"')

    return _load_model("mbt2018-vbr", metric, pretrained, progress, **kwargs)
