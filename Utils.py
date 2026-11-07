import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------------------
# Configuration / reproducibility
# ----------------------------------------------------------------------------
def _merge(base, override):
    """Recursive dict update (nested sections such as `paths` are merged key by key)."""
    out = dict(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path="Config.yaml", section=None):
    """
    Load the YAML config. Override order: top-level keys -> `Spatiotemporal` section (only for
    "*_Spatiotemporal" networks) -> `section` (e.g. "Test").
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    base  = {k: v for k, v in raw.items() if k not in {"Spatiotemporal", "Test"}}
    extra = raw.get(section) if section else None

    network = _merge(base, extra)["Network"]
    config  = _merge(base, raw.get("Spatiotemporal")) if network.endswith("Spatiotemporal") else base
    config  = _merge(config, extra)

    config["epsilon"] = float(config["epsilon"])
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_spatiotemporal(cfg):
    return cfg["Network"].endswith("Spatiotemporal")


def experiment_name(cfg):
    """Folder name of an experiment (kept identical to the naming used for the trained models)."""
    R, Network = cfg["R"], cfg["Network"]
    share      = f"Share{int(cfg['Share_params'])}"
    softplus   = "_SoftPlus" if cfg["SoftPlus"] else ""

    if is_spatiotemporal(cfg):
        if Network.startswith("PDDL"):
            return f"{Network}_R{R}_Unet_{share}_CG{softplus}"
        return f"{Network}_R{R}_Unet_{share}{softplus}_{cfg['Momentum_type']}_{cfg['Smoothing_type']}"

    if Network.startswith("PDDL"):
        return f"{Network}_R{R}_Unet_{share}_CG{softplus}_Patience{cfg['patience']}"
    return (f"{Network}_R{R}_{cfg['Smoothing_type']}_epsilon{cfg['epsilon']}_DF{cfg['DF_Steps']}"
            f"_Patience{cfg['patience']}_MagModeNone" + ("" if cfg["SoftPlus"] else "_NoSoftPlus"))


# ----------------------------------------------------------------------------
# Support mask: the original (un-padded) phase-encoding width of each subject
# ----------------------------------------------------------------------------
PE_GROUPS = {
    208: [4, 5, 6],
    180: [83, 84, 85, 89, 90, 91],
    176: [78, 79, 80, 95],
    156: [63, 68],
    150: [16],
    144: [9, 10, 11, 19, 20, 21, 25, 26, 27, 28,
          32, 33, 34, 38, 39, 40, 45, 57, 60, 69, 74],
    136: [107, 108, 109, 111, 112, 113, 157, 160, 161],
    132: [96, 97],
    121: [55],
    120: [53, 56],
    112: [12, 120, 121, 122, 137, 138, 139, 164],
    110: [155, 156],
    104: [146, 150, 151],
}
SUBJECT_TO_PE = {subject: pe for pe, subjects in PE_GROUPS.items() for subject in subjects}


def make_support_mask(subject_number, reference_mask):
    """Binary mask that is 1 on the subject's original PE columns (centred) and 0 on the zero-padded sides."""
    original_pe = SUBJECT_TO_PE[int(subject_number.item())]
    width = reference_mask.shape[-1]
    pad_left = (width - original_pe) // 2
    support = torch.zeros_like(reference_mask)
    support[..., pad_left:pad_left + original_pe] = 1
    return support


# ----------------------------------------------------------------------------
# MRI operators
# ----------------------------------------------------------------------------
class InverseSoftplus(torch.nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, y):
        return (1.0 / self.beta) * torch.log(torch.expm1(self.beta * y))


def FFT(image, axis=(-2, -1)):
    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axis), dim=axis, norm='ortho'), dim=axis)


def IFFT(kspace, axis=(-2, -1)):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axis), dim=axis, norm='ortho'), dim=axis)


def E(x, Coil, Mask):
    """Forward operator: coil sensitivities -> FFT -> sampling mask."""
    return FFT(x * Coil) * Mask


def EH(y, Coil, Mask):
    """Adjoint operator."""
    return torch.sum(IFFT(y * Mask) * torch.conj(Coil), axis=1, keepdim=True)


def MsEHEx(x, Coil, Mask):
    """Normal operator E^H E x."""
    return torch.sum(IFFT(Mask * FFT(x * Coil)) * torch.conj(Coil), axis=1, keepdim=True)


# ----------------------------------------------------------------------------
# Data preparation (shared by train.py and test.py)
# ----------------------------------------------------------------------------
def prepare_batch(batch, Network, device):
    """
    Normalize a DataLoaderSL batch, build the Omega/Delta masks and the coil-combined zero-filled input.

    Shapes (T frames):  ksp, r [B,T,Nc,H,W] | masks, label [B,T,H,W] | zero_filled [B,2T,H,W] | coil [B,Nc,H,W]
    """
    (ksp, r, Omega_Mask, SliceNumber, SubjectNumber, TimeFrame,
     coil, FileName, total_phases, normalization, frames) = batch

    ksp        = (ksp / normalization).to(device)
    r          = (r / normalization).to(device)
    coil       = coil.to(device)
    Omega_Mask = Omega_Mask.to(device)

    # Spatial models: the zero-padded PE sides count as acquired, so Delta only covers missing lines inside the support.
    if not Network.endswith("Spatiotemporal"):
        Support_Mask = make_support_mask(SubjectNumber, Omega_Mask)
        Omega_Mask   = Omega_Mask * Support_Mask + (1.0 - Support_Mask)
    Delta_Mask = 1.0 - Omega_Mask

    # Magnitude prior on the missing lines
    if Network.startswith("MI_Oracle"):
        r = torch.abs(Delta_Mask.unsqueeze(2) * ksp)
    elif Network.startswith("MI"):
        r = torch.abs(Delta_Mask.unsqueeze(2) * r)

    coil_t      = torch.conj(coil.unsqueeze(1))
    zero_filled = torch.sum(IFFT(ksp * Omega_Mask.unsqueeze(2)) * coil_t, dim=2)
    label       = torch.sum(IFFT(ksp) * coil_t, dim=2)
    zero_filled = torch.cat([zero_filled.real, zero_filled.imag], dim=1)
    coil_rss    = torch.sqrt(torch.sum(torch.abs(coil) ** 2, dim=1))

    return dict(ksp=ksp, r=r, coil=coil, coil_rss=coil_rss, Omega_Mask=Omega_Mask, Delta_Mask=Delta_Mask,
                zero_filled=zero_filled, label=label,
                subject=int(SubjectNumber.item()), slice=int(SliceNumber.item()), time=int(TimeFrame.item()),
                n_phases=int(total_phases.item()), frames=[int(t) for t in frames[0]])


def reconstruct(model, b):
    return model(b["zero_filled"], b["coil"], b["Omega_Mask"], b["Delta_Mask"], b["r"])


def kspace_loss(recon, b):
    """L1+L2 loss between the multi-coil k-space of the reconstruction and the reference k-space."""
    return L1_L2_Loss(FFT(recon.unsqueeze(2) * b["coil"].unsqueeze(1)), b["ksp"])


# ----------------------------------------------------------------------------
# Loss / metrics
# ----------------------------------------------------------------------------
def L1_L2_Loss(recon, label):
    return (torch.norm(recon - label, p=2) / torch.norm(label, p=2)
            + torch.norm(recon - label, p=1) / torch.norm(label, p=1))


def getSSIM(ref, rec):
    data_range = np.max(np.abs(ref)) - np.min(np.abs(ref))
    return structural_similarity(ref, rec, data_range=data_range)


def getPSNR(ref, recon):
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    return 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))


# ----------------------------------------------------------------------------
# Saving helpers
# ----------------------------------------------------------------------------
def save_gray_png(image, save_path, vmax=None):
    """Save |image| as a grayscale PNG. Without `vmax` the image is min-max scaled."""
    image = np.abs(np.asarray(image))
    if vmax is None:
        plt.imsave(save_path, image, cmap="gray")
    else:
        plt.imsave(save_path, image, cmap="gray", vmin=0, vmax=vmax)


def save_comparison_pngs(z, y, name, png_dir, diff_scale, window_scale, tags=("Combined", "Recon", "Label", "Error"),
                         recon_scale=None):
    """
    Save windowed PNGs of a reconstruction z vs reference y: a 2x2 panel [y, z; 0, diff_scale*(z-y)] and/or the
    single images. z is divided by `recon_scale` (default: its own maximum), y by its own maximum.
    """
    y_img = np.flipud(y) / np.max(np.abs(y))
    z_img = np.flipud(z) / (np.max(np.abs(z)) if recon_scale is None else recon_scale)
    err   = diff_scale * (z_img - y_img)
    images = {"Combined": np.block([[y_img,                z_img],
                                    [np.zeros_like(y_img), err  ]]),
              "Recon"   : z_img,
              "Label"   : y_img,
              "Error"   : err}
    for tag in tags:
        save_gray_png(images[tag], os.path.join(png_dir, f"{name}_{tag}.png"), vmax=window_scale)


def save_curves(curves, save_path, xlabel="Epoch", ylabel="", title=None, labels=None, x=None, marker=None):
    """Plot one or more 1-D curves (or the columns of a 2-D array) into a PNG."""
    plt.figure()
    curves = np.asarray(curves)
    if curves.ndim == 1:
        curves = curves[:, None]
    x = np.arange(len(curves)) if x is None else x
    for j in range(curves.shape[1]):
        plt.plot(x, curves[:, j], alpha=0.8, marker=marker, label=labels[j] if labels else None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if labels:
        plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
