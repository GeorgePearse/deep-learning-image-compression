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

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange

try:
    from pointops.functions import pointops
except ImportError:
    pass  # NOTE: Optional dependency.

from tinify.layers.pointcloud.hrtzxf2022 import index_points
from tinify.losses.utils import compute_rate_loss
from tinify.registry import register_criterion


@register_criterion("ChamferPccRateDistortionLoss")
class ChamferPccRateDistortionLoss(nn.Module):
    """Simple loss for regular point cloud compression.

    For compression models that reconstruct the input point cloud.
    """

    LMBDA_DEFAULT: dict[str, float] = {
        # "bpp": 1.0,
        "rec": 1.0,
    }

    lmbda: dict[str, float]

    def __init__(
        self, lmbda: dict[str, float] | None = None, rate_key: str = "bpp"
    ) -> None:
        super().__init__()
        self.lmbda = lmbda or dict(self.LMBDA_DEFAULT)
        self.lmbda.setdefault(rate_key, 1.0)

    def forward(
        self, output: dict[str, Any], target: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {
            **self.compute_rate_loss(output, target),
            **self.compute_rec_loss(output, target),
        }

        loss_sum: Tensor = torch.tensor(0.0, device=list(out.values())[0].device)
        for k in self.lmbda.keys():
            if f"{k}_loss" in out:
                loss_sum = loss_sum + self.lmbda[k] * out[f"{k}_loss"]
        out["loss"] = loss_sum

        return out

    def compute_rate_loss(
        self, output: dict[str, Any], target: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if "likelihoods" not in output:
            return {}
        N, P, _ = target["pos"].shape
        return compute_rate_loss(output["likelihoods"], N, P)

    def compute_rec_loss(
        self, output: dict[str, Any], target: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        dist1, dist2, _, _ = chamfer_distance(
            target["pos"], output["x_hat"], order="b n c"
        )
        loss_chamfer = dist1.mean() + dist2.mean()
        return {"rec_loss": loss_chamfer}


def chamfer_distance(
    xyzs1: Tensor, xyzs2: Tensor, order: str = "b n c"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # idx1, dist1: (b, n1)
    # idx2, dist2: (b, n2)
    xyzs1_bcn: Tensor = rearrange(xyzs1, f"{order} -> b c n").contiguous()
    xyzs1_bnc: Tensor = rearrange(xyzs1, f"{order} -> b n c").contiguous()
    xyzs2_bcn: Tensor = rearrange(xyzs2, f"{order} -> b c n").contiguous()
    xyzs2_bnc: Tensor = rearrange(xyzs2, f"{order} -> b n c").contiguous()
    idx1: Tensor = pointops.knnquery_heap(1, xyzs2_bnc, xyzs1_bnc).long().squeeze(2)
    idx2: Tensor = pointops.knnquery_heap(1, xyzs1_bnc, xyzs2_bnc).long().squeeze(2)
    torch.cuda.empty_cache()
    dist1: Tensor = ((xyzs1_bcn - index_points(xyzs2_bcn, idx1)) ** 2).sum(1)
    dist2: Tensor = ((xyzs2_bcn - index_points(xyzs1_bcn, idx2)) ** 2).sum(1)
    return dist1, dist2, idx1, idx2
