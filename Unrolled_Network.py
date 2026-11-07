import torch
import torch.nn as nn

from DC import DC_MI, DC_CG
from unet.unet import create_model as Unet
from Utils import InverseSoftplus

# All networks take / return images with T frames: zf is [B, 2T, H, W] (real frames, then imag frames)
# and the reconstruction is complex [B, T, H, W]. T = 1 for spatial, T = R for spatiotemporal.


def _admm_beta(init, Unrolls, Share_params, SoftPlus, beta_softplus):
    """Learnable ADMM dual step size (one per unroll unless shared)."""
    beta = torch.tensor(init, dtype=torch.float32)
    if SoftPlus:
        beta = InverseSoftplus(beta=beta_softplus)(beta)
    return nn.Parameter(beta.clone() if Share_params else beta.repeat(Unrolls))


def _unet(n_frames):
    return Unet(use_norm=True, use_time_emb=True, channel_mult=[3, 4, 5], num_channels=32, InOut_Chn=2 * n_frames)


class _UnrolledADMM(nn.Module):
    """Shared helpers for the unrolled ADMM networks."""

    def beta_value(self, k):
        beta = self.beta if self.Share_params else self.beta[k]
        return self.SoftPlus_func(beta) if self.SoftPlus else beta

    def denoise(self, x, k):
        t = torch.full((x.shape[0],), k + 1, dtype=torch.long, device=x.device)
        return self.Network(x, t)

    def to_complex(self, x):
        T = x.shape[1] // 2
        return x[:, :T] + 1j * x[:, T:]

    @torch.no_grad()
    def learned_params(self):
        """Effective (post-SoftPlus) values of the scalar parameters, as numpy arrays, for logging."""
        sp = self.SoftPlus_func if self.SoftPlus else (lambda t: t)
        dc = self.DataConsistency
        params = {"beta": sp(self.beta), "mu": dc.SoftPlus_func(dc.mu) if dc.SoftPlus else dc.mu}
        if isinstance(dc, DC_MI):
            params["lambda"] = dc.lamda
            params["eta"]    = dc.SoftPlus_func(dc.eta) if dc.SoftPlus else dc.eta
        return {k: v.detach().cpu().numpy() for k, v in params.items()}


class UnrolledNet_MI(_UnrolledADMM):
    """Unrolled ADMM with a U-Net proximal step and magnitude-informed data consistency (DC_MI)."""

    def __init__(self, initial_params, Unrolls, DF_Steps, epsilon, Share_params,
                 Momentum_type="None", Smoothing_type="Simple", SoftPlus=False, beta_softplus=20,
                 n_frames=1, zero_dual_init=True):
        super().__init__()
        self.SoftPlus_func   = nn.Softplus(beta=beta_softplus)
        self.Network         = _unet(n_frames)
        self.DataConsistency = DC_MI(initial_params["mu"], initial_params["eta"], initial_params["lamda"],
                                     DF_Steps, Unrolls, epsilon, Share_params, Momentum_type, Smoothing_type,
                                     initial_params["beta_Momentum"], SoftPlus=SoftPlus, beta_softplus=beta_softplus)
        self.beta            = _admm_beta(initial_params["beta_ADMM"], Unrolls, Share_params, SoftPlus, beta_softplus)

        self.SoftPlus       = SoftPlus
        self.Share_params   = Share_params
        self.Unrolls        = Unrolls
        self.zero_dual_init = zero_dual_init   # u^0 = 0 (spatial models) or u^0 = zf (spatiotemporal models)

    def forward(self, zf, Coil, Mask, Delta_Mask, r):
        x = torch.clone(zf)
        u = torch.zeros_like(zf) if self.zero_dual_init else torch.clone(zf)
        for k in range(self.Unrolls):
            v = self.denoise(x + u, k)
            x = self.DataConsistency(Coil, Mask, Delta_Mask, r, zf, k, v, u)
            u = u + self.beta_value(k) * (x - v)
        return self.to_complex(x)


class UnrolledNet_PDDL(_UnrolledADMM):
    """Baseline unrolled ADMM with a U-Net proximal step and conjugate-gradient data consistency."""

    def __init__(self, initial_params, Unrolls, DF_Steps, Share_params, SoftPlus=True, beta_softplus=20,
                 n_frames=1):
        super().__init__()
        self.SoftPlus_func   = nn.Softplus(beta=beta_softplus)
        self.Network         = _unet(n_frames)
        self.DataConsistency = DC_CG(initial_params["mu"], DF_Steps, Unrolls, Share_params,
                                     SoftPlus=SoftPlus, beta_softplus=beta_softplus)
        self.beta            = _admm_beta(initial_params["beta_ADMM"], Unrolls, Share_params, SoftPlus, beta_softplus)

        self.SoftPlus     = SoftPlus
        self.Share_params = Share_params
        self.Unrolls      = Unrolls

    def forward(self, zf, Coil, Mask, Delta_Mask=None, r=None):
        # Delta_Mask and r are unused; they are accepted so both networks share one call signature.
        x = torch.clone(zf)
        u = torch.clone(zf)
        for k in range(self.Unrolls):
            v = self.denoise(x + u, k)
            x = self.DataConsistency(zf, Coil, Mask, k, v, u)
            u = u + self.beta_value(k) * (x - v)
        return self.to_complex(x)


NETWORKS = {"PDDL_Spatial", "MI_Spatial", "MI_Oracle_Spatial",
            "PDDL_Spatiotemporal", "MI_Spatiotemporal", "MI_Oracle_Spatiotemporal"}


def build_model(cfg):
    Network = cfg["Network"]
    if Network not in NETWORKS:
        raise ValueError(f"Unknown Network: {Network}")

    init           = cfg["init"]
    spatiotemporal = Network.endswith("Spatiotemporal")
    n_frames       = cfg["R"] if spatiotemporal else 1

    if Network.startswith("PDDL"):
        return UnrolledNet_PDDL(init, cfg["Unrolls"], cfg["DF_Steps"], cfg["Share_params"],
                                cfg["SoftPlus"], cfg["beta_softplus"], n_frames=n_frames)
    return UnrolledNet_MI(init, cfg["Unrolls"], cfg["DF_Steps"], cfg["epsilon"], cfg["Share_params"],
                          cfg["Momentum_type"], cfg["Smoothing_type"], cfg["SoftPlus"], cfg["beta_softplus"],
                          n_frames=n_frames, zero_dual_init=not spatiotemporal)


def load_checkpoint(model, path, device):
    """Load weights; entries the model no longer has (e.g. from removed experimental modules) are skipped."""
    state = torch.load(path, map_location=device)["model_state_dict"]
    keys = model.state_dict().keys()
    unused = [k for k in state if k not in keys]
    if unused:
        print(f"Skipping {len(unused)} unused checkpoint entries: {sorted(unused)}")
    model.load_state_dict({k: v for k, v in state.items() if k in keys})
    return model
