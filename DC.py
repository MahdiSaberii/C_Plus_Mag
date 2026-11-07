import torch
import torch.nn as nn

from Utils import InverseSoftplus, E, EH, MsEHEx

# Tensor layout (T = number of frames: 1 for spatial, R for spatiotemporal):
#   zf, v, u : [B, 2T, H, W]  channels = [real frames..., imag frames...]
#   masks    : [B, T, H, W]
#   r        : [B, T, Nc, H, W]
#   Coil     : [B, Nc, H, W]


def frame(x, t, T):
    """Complex image of frame t from a [B, 2T, H, W] real/imag tensor -> [B, 1, H, W]."""
    return x[:, t:t + 1] + 1j * x[:, T + t:T + t + 1]


def stack_frames(frames):
    """List of T complex [B, 1, H, W] frames -> [B, 2T, H, W] real/imag tensor."""
    return torch.cat([f.real for f in frames] + [f.imag for f in frames], dim=1)


class DC_MI(nn.Module):
    """
    Data consistency with magnitude information (MI), solved per frame with DF_Steps of (momentum) gradient descent on

        ||E_Omega x - y_Omega||^2 + mu ||x - v + u||^2 + lamda || |E_Delta x| - r ||^2

    where r is the magnitude prior on the missing k-space lines (Delta).
    """

    def __init__(self, mu, eta, lamda, DF_Steps, num_of_unrolls, epsilon, Share_params,
                 Momentum_type="None", Smoothing_type="Simple", beta_momentum=0.9,
                 SoftPlus=False, beta_softplus=20):
        super().__init__()
        if Momentum_type not in {"None", "Polyak", "Nesterov"}:
            raise ValueError(f"Unknown Momentum_type: {Momentum_type}")
        if Smoothing_type != "Simple":
            raise ValueError(f"Unknown Smoothing_type: {Smoothing_type}")

        self.SoftPlus_func = nn.Softplus(beta=beta_softplus)
        inv_softplus       = InverseSoftplus(beta=beta_softplus)

        mu_raw = torch.tensor(mu, dtype=torch.float32)
        if SoftPlus:
            mu_raw = inv_softplus(mu_raw)
        if Share_params:
            self.mu    = nn.Parameter(mu_raw.clone())
            self.lamda = nn.Parameter(torch.tensor(lamda, dtype=torch.float32))
        else:
            self.mu    = nn.Parameter(mu_raw.repeat(num_of_unrolls))
            self.lamda = nn.Parameter(torch.full((num_of_unrolls,), lamda, dtype=torch.float32))

        eta_raw = torch.tensor(eta, dtype=torch.float32)
        if SoftPlus:
            eta_raw = inv_softplus(eta_raw)
        self.eta           = nn.Parameter(eta_raw.expand(DF_Steps, num_of_unrolls).contiguous())
        self.beta_momentum = nn.Parameter(torch.tensor([beta_momentum], dtype=torch.float32)
                                          .expand(DF_Steps, num_of_unrolls).contiguous())

        self.SoftPlus      = SoftPlus
        self.DF_Steps      = DF_Steps
        self.epsilon       = epsilon
        self.Share_params  = Share_params
        self.Momentum_type = Momentum_type

    def Smooth_Sgn(self, x):
        """Smoothed complex sign x / |x|."""
        return x / torch.sqrt(x.real ** 2 + x.imag ** 2 + self.epsilon)

    def forward(self, Coil, Omega_Mask, Delta_Mask, r, zf, unroll_ind, v, u):
        mu_val    = self.mu if self.Share_params else self.mu[unroll_ind]
        lamda_val = self.lamda if self.Share_params else self.lamda[unroll_ind]
        if self.SoftPlus:
            mu_val = self.SoftPlus_func(mu_val)

        T       = zf.shape[1] // 2
        outputs = []
        for t in range(T):
            zf_t    = frame(zf, t, T)
            v_t     = frame(v, t, T)
            u_t     = frame(u, t, T)
            Omega_t = Omega_Mask[:, t:t + 1]
            Delta_t = Delta_Mask[:, t:t + 1]
            r_t     = r[:, t]

            x        = zf_t.clone()
            velocity = torch.zeros_like(x)
            for ix in range(self.DF_Steps):
                eta_val  = self.eta[ix, unroll_ind]
                if self.SoftPlus:
                    eta_val = self.SoftPlus_func(eta_val)
                beta_val = self.beta_momentum[ix, unroll_ind]

                x_eval = x - beta_val * velocity if self.Momentum_type == "Nesterov" else x

                Ex    = E(x_eval, Coil, Delta_t)
                term1 = MsEHEx(x_eval, Coil, Omega_t) - zf_t
                term2 = x_eval - v_t + u_t
                term3 = EH(Ex - r_t * self.Smooth_Sgn(Ex), Coil, Delta_t)
                Grad  = term1 + mu_val * term2 + lamda_val * term3

                if self.Momentum_type == "None":
                    x = x - eta_val * Grad
                elif self.Momentum_type == "Polyak":
                    velocity = beta_val * velocity + (1 - beta_val) * Grad
                    x = x - eta_val * velocity
                else:  # Nesterov
                    velocity = beta_val * velocity + Grad
                    x = x - eta_val * velocity
            outputs.append(x)

        return stack_frames(outputs)


class DC_CG(nn.Module):
    """Conjugate-gradient data consistency, per frame: solves (E^H E + mu I) x = zf + mu (v - u)."""

    def __init__(self, mu, DF_Steps, num_of_unrolls, Share_params, SoftPlus=True, beta_softplus=20):
        super().__init__()
        self.SoftPlus_func = nn.Softplus(beta=beta_softplus)

        mu_raw = torch.tensor(mu, dtype=torch.float32)
        if SoftPlus:
            mu_raw = InverseSoftplus(beta=beta_softplus)(mu_raw)
        self.mu = nn.Parameter(mu_raw.clone() if Share_params else mu_raw.repeat(num_of_unrolls))

        self.SoftPlus     = SoftPlus
        self.iterations   = DF_Steps
        self.Share_params = Share_params

    def forward(self, zf, Coil, Mask, unroll_idx, v, u):
        mu_val = self.mu if self.Share_params else self.mu[unroll_idx]
        if self.SoftPlus:
            mu_val = self.SoftPlus_func(mu_val)

        T       = zf.shape[1] // 2
        outputs = []
        for t in range(T):
            Mask_t   = Mask[:, t:t + 1]
            p_now    = frame(zf, t, T) + mu_val * (frame(v, t, T) - frame(u, t, T))
            r_now    = torch.clone(p_now)
            b_approx = torch.zeros_like(p_now)

            for _ in range(self.iterations):
                q        = MsEHEx(p_now, Coil, Mask_t) + mu_val * p_now
                rr_now   = torch.sum(r_now * torch.conj(r_now))
                alpha    = rr_now / torch.sum(q * torch.conj(p_now))       # (r'r) / (p'q)
                b_approx = b_approx + alpha * p_now
                r_next   = r_now - alpha * q
                p_now    = r_next + torch.sum(r_next * torch.conj(r_next)) / rr_now * p_now
                r_now    = r_next
            outputs.append(b_approx)

        return stack_frames(outputs)
