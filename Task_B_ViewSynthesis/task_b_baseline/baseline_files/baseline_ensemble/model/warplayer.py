import torch
import torch.nn.functional as F


backwarp_ten_grid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_ten_grid:
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3], device=tenFlow.device
        ).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1
        )
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2], device=tenFlow.device
        ).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3]
        )
        backwarp_ten_grid[k] = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    grid = (backwarp_ten_grid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
