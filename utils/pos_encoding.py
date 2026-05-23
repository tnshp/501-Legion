import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def get_pos_encoding(pos, time_step, d_model):
    """
    pos: Tensor of shape [N, 2] or [B, N, 2], where pos[..., 0] = x and pos[..., 1] = y
    time_step: 
        - For single input: scalar, Tensor [N], or Tensor [N, 1]
        - For batch input: scalar, Tensor [B], Tensor [B, N], or Tensor [B, N, 1]
        - [B] shape: each batch instance has different time_step, broadcasted across N positions
    d_model: output dimension
    returns: Tensor of shape [N, d_model] or [B, N, d_model]
    """
    device = pos.device
    dtype = pos.dtype
    
    # Check if batched input
    is_batched = pos.ndim == 3
    
    if is_batched:
        B, N, _ = pos.shape
        x = pos[..., 0].float()  # [B, N]
        y = pos[..., 1].float()  # [B, N]
    else:
        N = pos.shape[0]
        x = pos[:, 0].float()  # [N]
        y = pos[:, 1].float()  # [N]

    # Handle time_step
    if not torch.is_tensor(time_step):
        if is_batched:
            t = torch.tensor(time_step, device=device, dtype=torch.float32).expand(B, N)
        else:
            t = torch.tensor(time_step, device=device, dtype=torch.float32).expand(N)
    else:
        t = time_step.to(device=device, dtype=torch.float32)
        
        if is_batched:
            if t.ndim == 0:
                t = t.expand(B, N)
            elif t.ndim == 1:
                t = t.unsqueeze(1).expand(B, N)  # [B] -> [B, N]
            elif t.ndim == 2 and t.shape[1] == 1:
                t = t.expand(B, N)  # [B, 1] -> [B, N]
            # else: assume [B, N] already
        else:
            if t.ndim == 0:
                t = t.expand(N)
            elif t.ndim == 2 and t.shape[1] == 1:
                t = t.squeeze(1)  # [N, 1] -> [N]

    d_x = d_model // 3
    d_y = d_model // 3
    d_t = d_model - d_x - d_y

    def sinusoidal_encoding(coord, dim):
        """
        coord: [N] or [B, N]
        dim: output dimension
        returns: [..., dim]
        """
        if dim <= 0:
            if coord.ndim == 1:
                return torch.empty(coord.shape[0], 0, device=device, dtype=dtype)
            else:
                return torch.empty(coord.shape[0], coord.shape[1], 0, device=device, dtype=dtype)

        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half_dim, 1))
        )

        is_1d = coord.ndim == 1
        
        if is_1d:
            coord_expanded = coord.unsqueeze(1)  # [N, 1]
            pe = torch.zeros(coord.shape[0], dim, device=device, dtype=torch.float32)
        else:
            coord_expanded = coord.unsqueeze(2)  # [B, N, 1]
            pe = torch.zeros(coord.shape[0], coord.shape[1], dim, device=device, dtype=torch.float32)
        
        pe[..., 0:2*half_dim:2] = torch.sin(coord_expanded * div_term)
        pe[..., 1:2*half_dim:2] = torch.cos(coord_expanded * div_term)

        if dim % 2 == 1:
            if is_1d:
                pe[:, -1] = torch.sin(coord * div_term[-1]) if half_dim > 0 else coord
            else:
                pe[..., -1] = torch.sin(coord * div_term[-1]) if half_dim > 0 else coord

        return pe.to(dtype)

    pe_x = sinusoidal_encoding(x, d_x)
    pe_y = sinusoidal_encoding(y, d_y)
    pe_t = sinusoidal_encoding(t, d_t)

    return torch.cat([pe_x, pe_y, pe_t], dim=-1)

if __name__ == "__main__":
    # Single input (existing)
    pos = torch.randn(240, 2)  # [N, 2]
    encoding = get_pos_encoding(pos, time_step=100, d_model=128)  # [N, 128]

    # Batch input (new)
    pos_batch = torch.randn(10, 240, 2)  # [B, N, 2]
    #time step (B, N)
    time_step = torch.arange(10).unsqueeze(1).expand(-1, 240)  # [B, N ], different time step for each batch instance
    encoding_batch = get_pos_encoding(pos_batch, time_step=time_step, d_model=128)  # [B, N, 128]

    print("Single encoding shape:", encoding.shape)  # Should be [N, 128]
    print("Batch encoding shape:", encoding_batch.shape)  # Should be [B, N, 128]