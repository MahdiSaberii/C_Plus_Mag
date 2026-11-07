import csv
import os

import numpy as np
import scipy.io as sio
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import DataLoaderSL
from Unrolled_Network import build_model, load_checkpoint
from Utils import (experiment_name, getPSNR, getSSIM, is_spatiotemporal, load_config, prepare_batch, reconstruct,
                   save_comparison_pngs, save_curves, set_seed)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics(metrics, out_dir):
    """Per-frame metrics, their mean/std, and per-subject averages (+ plots)."""
    columns = {key: np.array([m[key] for m in metrics]) for key in metrics[0]}
    psnr    = columns["PSNR"]
    ssim    = columns["SSIM"]

    sio.savemat(os.path.join(out_dir, "Metrics_Test.mat"), columns)
    write_csv(os.path.join(out_dir, "Metrics_Test.csv"), metrics)
    with open(os.path.join(out_dir, "Metrics_Test.txt"), "w") as f:
        f.write("=== Metrics Summary ===\n")
        f.write(f"PSNR: mean = {psnr.mean():.4f}, std = {psnr.std():.4f}\n")
        f.write(f"SSIM: mean = {ssim.mean():.4f}, std = {ssim.std():.4f}\n")

    subjects    = np.unique(columns["Subject"])
    per_subject = {"Subject": subjects,
                   "PSNR"   : np.array([psnr[columns["Subject"] == s].mean() for s in subjects]),
                   "SSIM"   : np.array([ssim[columns["Subject"] == s].mean() for s in subjects])}
    rows        = [{"Subject": int(s), "PSNR": p, "SSIM": q} for s, p, q in zip(*per_subject.values())]

    sio.savemat(os.path.join(out_dir, "Metrics_PerSubject.mat"), per_subject)
    write_csv(os.path.join(out_dir, "Metrics_PerSubject.csv"), rows)
    for metric in ["PSNR", "SSIM"]:
        save_curves(per_subject[metric], os.path.join(out_dir, f"{metric}_PerSubject.png"),
                    xlabel="Subject", ylabel=f"Average {metric}", title=f"Average {metric} per Subject",
                    x=subjects, marker="o")


def save_slice_mats(recon_storage, mat_dir):
    """One .mat per subject-slice with all time frames, sorted by time."""
    for (subj_id, slice_id), data in recon_storage.items():
        order = np.argsort(data["Time"])
        sio.savemat(os.path.join(mat_dir, f"subj_{subj_id:04d}_slice_{slice_id:03d}.mat"), {
            "Subject": subj_id,
            "Slice"  : slice_id,
            "Nt"     : data["Nt"],
            "Time"   : np.array(data["Time"])[order],
            "recon"  : np.stack(data["recon"])[order],   # [T, H, W]
            "label"  : np.stack(data["label"])[order],   # [T, H, W]
            "PSNR"   : np.array(data["PSNR"])[order],    # [T]
            "SSIM"   : np.array(data["SSIM"])[order],    # [T]
        })


if __name__ == '__main__':
    set_seed(42)
    cfg      = load_config("Config.yaml", section="Test")
    device   = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    R        = cfg["R"]
    paths    = {k: v.format(R=R) if v else None for k, v in cfg["paths"].items()}
    subjects = set(cfg["subjects"]["test"])
    st       = is_spatiotemporal(cfg)
    exp_name = experiment_name(cfg)
    out_dir  = os.path.join(paths["test_results"], exp_name)
    ckpt     = os.path.join(paths["train_results"], exp_name, "model", f"BestModel_R{R}_Epoch{cfg['model_epoch']}.pth")

    for sub in ["pngs", "files", "mats"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ---------------- Data / model ----------------
    # Spatiotemporal: non-overlapping blocks of R frames (T = 1, R+1, ...)
    test_set    = DataLoaderSL(paths["test_data"], paths["coils"], n_frames=R if st else 1, blocks=st)
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt, device)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    learned          = ", ".join(f"{k}={v.mean():.5f}" for k, v in model.learned_params().items())
    print(f'Total number of trainable parameters: {trainable_params:,}')
    print(f"Learned parameters (mean): {learned}")
    print(f"EXP. Name: {exp_name}    ,, GPU: {device}")

    with open(os.path.join(out_dir, "files", "Config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    with open(os.path.join(out_dir, "files", "trainable_parameters.txt"), "w") as f:
        f.write(f'Total number of trainable parameters: {trainable_params:,}')

    # ---------------- Inference ----------------
    metrics       = []
    recon_storage = {}   # (subject, slice) -> all time frames

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Testing R --> {R}"):
            if int(batch[4]) not in subjects:
                continue
            b     = prepare_batch(batch, cfg["Network"], device)
            recon = reconstruct(model, b) * b["coil_rss"]

            for j, time_id in enumerate(b["frames"]):
                y    = b["label"][0, j].cpu().numpy()
                z    = recon[0, j].cpu().numpy()
                psnr = float(getPSNR(np.abs(y), np.abs(z)))
                ssim = float(getSSIM(np.abs(y), np.abs(z)))
                name = f"Subject_{b['subject']}_Slice_{b['slice']}_Time_{time_id}"

                metrics.append({"Subject": b["subject"], "Slice": b["slice"], "Time": time_id,
                                "PSNR"   : psnr,         "SSIM" : ssim})

                entry = recon_storage.setdefault((b["subject"], b["slice"]),
                                                 {"Nt": b["n_phases"], "Time": [], "recon": [],
                                                  "label": [], "PSNR": [], "SSIM": []})
                entry["Time"].append(time_id)
                entry["recon"].append(z)
                entry["label"].append(y)
                entry["PSNR"].append(psnr)
                entry["SSIM"].append(ssim)

                # Spatial: recon scaled by its own max; spatiotemporal: by the reference max
                save_comparison_pngs(z, y, name, os.path.join(out_dir, "pngs"), cfg["diff_scale"],
                                     cfg["window_scale"], recon_scale=np.max(np.abs(y)) if st else None)

    # ---------------- Save results ----------------
    save_metrics(metrics, out_dir)
    save_slice_mats(recon_storage, os.path.join(out_dir, "mats"))
