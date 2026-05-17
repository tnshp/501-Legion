import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def get_pos_encoding(pos, time_step, d_model):
    """
    pos: Tensor of shape [N, 2], where pos[:, 0] = x and pos[:, 1] = y
    time_step: scalar, Tensor [N], or Tensor [N, 1]
    d_model: output dimension
    returns: Tensor of shape [N, d_model]
    """
    device = pos.device
    dtype = pos.dtype
    N = pos.shape[0]

    x = pos[:, 0].float()
    y = pos[:, 1].float()

    if not torch.is_tensor(time_step):
        t = torch.tensor(time_step, device=device, dtype=torch.float32).expand(N)
    else:
        t = time_step.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(N)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

    d_x = d_model // 3
    d_y = d_model // 3
    d_t = d_model - d_x - d_y

    def sinusoidal_encoding(coord, dim):
        if dim <= 0:
            return torch.empty(coord.shape[0], 0, device=device, dtype=dtype)

        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half_dim, 1))
        )

        coord = coord.unsqueeze(1)
        pe = torch.zeros(coord.shape[0], dim, device=device, dtype=torch.float32)
        pe[:, 0:2*half_dim:2] = torch.sin(coord * div_term)
        pe[:, 1:2*half_dim:2] = torch.cos(coord * div_term)

        if dim % 2 == 1:
            pe[:, -1] = torch.sin(coord.squeeze(1) * div_term[-1]) if half_dim > 0 else coord.squeeze(1)

        return pe.to(dtype)

    pe_x = sinusoidal_encoding(x, d_x)
    pe_y = sinusoidal_encoding(y, d_y)
    pe_t = sinusoidal_encoding(t, d_t)

    return torch.cat([pe_x, pe_y, pe_t], dim=1)
