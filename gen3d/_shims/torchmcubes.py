"""Stand-in for the torchmcubes package (it needs a compiler and the CUDA toolkit to build): marching cubes on the
CPU through PyMCubes, with torchmcubes' conventions (a volume indexed [z, y, x] gives vertices as (x, y, z))."""
import mcubes
import numpy as np
import torch


def marching_cubes(vol: torch.Tensor, thresh: float):
    v, f = mcubes.marching_cubes(vol.detach().float().cpu().numpy(), float(thresh))
    v = np.ascontiguousarray(v[:, ::-1]).astype(np.float32)
    return torch.from_numpy(v).to(vol.device), torch.from_numpy(f.astype(np.int64)).to(vol.device)
