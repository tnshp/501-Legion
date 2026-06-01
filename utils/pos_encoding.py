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

class LearnedFourierPosEncoding(nn.Module):
    """
    Learnable Fourier-feature positional encoding for 2-D coordinates (x, y),
    plus a normalized scalar for time.

        x -> [sin(x·f_x), cos(x·f_x)]   with LEARNABLE frequencies f_x
        y -> [sin(y·f_y), cos(y·f_y)]   with LEARNABLE frequencies f_y
        t -> t / max_time_step          (single normalized scalar)

    The concatenation [sin/cos for x, sin/cos for y, t] is linearly projected to
    d_model and returned, to be ADDED to the token embedding — the same role the
    old fixed sinusoidal `get_pos_encoding` played, but the spatial frequencies
    are now learned (and tuned to the board scale, not NLP's 10000) while time
    is a plain normalized scalar.

    forward(pos, time_step):
        pos       : [..., 2]   (x = pos[..., 0], y = pos[..., 1])
        time_step : [...]      (broadcasts against pos[..., 0])
        returns   : [..., d_model]
    """

    def __init__(self, d_model, num_freqs=32, coord_scale=100.0,
                 min_wavelength=2.0, max_time_step=500.0):
        super().__init__()
        self.d_model       = d_model
        self.num_freqs     = num_freqs
        self.max_time_step = float(max_time_step)

        # Init wavelengths log-spaced from the whole board (~coord_scale) down to
        # a few units, so attention can resolve both global position and
        # nearby-planet geometry. f = 2π / wavelength.  Learnable thereafter.
        wavelengths = torch.logspace(
            math.log10(coord_scale), math.log10(min_wavelength), num_freqs
        )
        freq_init = 2.0 * math.pi / wavelengths            # [num_freqs]
        self.freq_x = nn.Parameter(freq_init.clone())
        self.freq_y = nn.Parameter(freq_init.clone())

        fourier_dim = 4 * num_freqs + 1                    # sin/cos·(x,y) + t
        self.proj = nn.Linear(fourier_dim, d_model)

    def forward(self, pos, time_step):
        x = pos[..., 0:1]                                  # [..., 1]
        y = pos[..., 1:2]                                  # [..., 1]
        x_proj = x * self.freq_x                           # [..., num_freqs]
        y_proj = y * self.freq_y                           # [..., num_freqs]

        t = time_step.to(x.dtype) / self.max_time_step     # [...]
        t = t.unsqueeze(-1)                                # [..., 1]

        feats = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj),
             torch.sin(y_proj), torch.cos(y_proj), t],
            dim=-1,
        )                                                  # [..., 4·num_freqs + 1]
        return self.proj(feats)                            # [..., d_model]


class LearnedFourierScalarEncoding(nn.Module):
    """
    Encode a non-negative scalar quantity (e.g. ship count) for fine-grained
    magnitude / threshold reasoning.

    Ship counts span ~1..1000 and the game is logarithmic (fleet speed ~
    log(ships)), and the policy acts by choosing a *fraction* of a planet's
    ships — so the decisive quantity is a *ratio*. log1p turns ratios into
    differences and compresses the range; learnable Fourier features then give a
    multi-resolution code so sharp thresholds ("just enough ships to capture")
    are linearly separable — something a single linear projection of the raw,
    unnormalized count cannot resolve.

        s   -> u   = log1p(s) / log1p(max_value)      # ~[0, 1], well-conditioned
        out = [u, sin(u·f), cos(u·f)]                 # learnable frequencies f

    forward(s): s [..., 1] -> [..., 2*num_freqs + 1]
    """

    def __init__(self, num_freqs=16, max_value=1000.0):
        super().__init__()
        self.log_max = math.log1p(float(max_value))
        # Broad log-spaced init from ~1 to ~128 cycles over the normalized range;
        # learnable thereafter so training can sharpen resolution where it pays.
        freq_init = 2.0 * math.pi * torch.logspace(0.0, math.log10(128.0), num_freqs)
        self.freq    = nn.Parameter(freq_init)
        self.out_dim = 2 * num_freqs + 1

    def forward(self, s):
        u    = torch.log1p(s.clamp(min=0)) / self.log_max   # [..., 1]
        proj = u * self.freq                                # [..., num_freqs]
        return torch.cat([u, torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierAngleEncoding(nn.Module):
    """
    Periodic encoding of an angle (radians) via integer harmonics:

        θ -> [sin(kθ), cos(kθ)]  for k = 1..num_harmonics

    A raw angle fed to a linear layer cannot represent the 2π wraparound — 0 and
    2π look maximally different yet are the same heading. Integer harmonics give
    an exact, continuous periodic code, and more harmonics sharpen angular
    resolution so the model can judge whether a fleet's heading will actually
    strike a target planet.

    Harmonics are FIXED integers (a registered buffer, not learned): non-integer
    frequencies would break the exact 2π periodicity.

    forward(theta): theta [..., 1] -> [..., 2*num_harmonics]
    """

    def __init__(self, num_harmonics=8):
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        )
        self.out_dim = 2 * num_harmonics

    def forward(self, theta):
        proj = theta * self.harmonics                       # [..., num_harmonics]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


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