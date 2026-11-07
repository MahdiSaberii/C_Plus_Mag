import os

import numpy as np
import scipy.io as sio
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import DataLoaderSL
from Unrolled_Network import build_model
from Utils import (experiment_name, is_spatiotemporal, kspace_loss, load_config, prepare_batch, reconstruct,
                   save_comparison_pngs, save_curves, save_gray_png, set_seed)

PARAM_TITLES = {"beta": "Beta", "mu": "Mu", "lambda": "Lambda", "eta": "Eta"}


def train_one_epoch(model, loader, optimizer, cfg, device, epoch, val_loss, png_dir):
    model.train()
    subjects = set(cfg["subjects"]["train"])
    monitor  = set(cfg.get("monitor_subjects") or [])
    loss_sum = 0.0
    n_train  = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        if int(batch[4]) not in subjects:
            continue
        b = prepare_batch(batch, cfg["Network"], device)

        optimizer.zero_grad()
        recon = reconstruct(model, b)
        loss  = kspace_loss(recon, b)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        n_train  += 1

        # Optional: PNG of the target frame of selected subjects while training
        if b["subject"] in monitor:
            j    = b["frames"].index(b["time"])
            name = f"Subj_{b['subject']:03d}_Slc_{b['slice']:03d}_Time_{b['time']:03d}"
            save_comparison_pngs(recon[0, j].detach().cpu().numpy(), b["label"][0, j].cpu().numpy(), name,
                                 png_dir, cfg["diff_scale"], cfg["window_scale"], tags=["Combined"])

        postfix = {"Average_Loss": f"{loss_sum / n_train:.3f}",
                   "Val_Loss"    : f"{val_loss:.3f}"}
        for key, value in model.learned_params().items():
            postfix[PARAM_TITLES[key]] = f"{value.mean():.3f}"
        progress_bar.set_postfix(postfix)

    return loss_sum / max(n_train, 1)


@torch.no_grad()
def validate(model, loader, cfg, device, png_dir):
    model.eval()
    loss_sum = 0.0

    for batch in loader:
        b     = prepare_batch(batch, cfg["Network"], device)
        recon = reconstruct(model, b)
        name  = f"Subject_{b['subject']:03d}_Slice_{b['slice']}_Time_{b['time']}"

        save_gray_png(recon[0, 0].cpu().numpy(),      os.path.join(png_dir, f"{name}.png"))
        save_gray_png(b["label"][0, 0].cpu().numpy(), os.path.join(png_dir, f"{name}_REFERENCE.png"))

        loss_sum += kspace_loss(recon, b).item()

    return loss_sum / len(loader)


def log_epoch(exp_dir, history, model):
    """Save loss / learning-rate / learned-parameter curves (.png) and values (.mat)."""
    if history["val"]:
        losses, labels = np.stack([history["train"], history["val"]], axis=1), ["Training", "Validation"]
    else:
        losses, labels = np.array(history["train"]), ["Training"]

    save_curves(losses, os.path.join(exp_dir, "Loss_plot.png"), ylabel="Loss", labels=labels)
    sio.savemat(os.path.join(exp_dir, "Loss_List.mat"), {"train_loss": history["train"], "val_loss": history["val"]})

    save_curves(history["lr"], os.path.join(exp_dir, "Learning_Rate.png"), ylabel="Learning Rate")
    sio.savemat(os.path.join(exp_dir, "Learning_Rate.mat"), {"learning_rate": np.array(history["lr"])})

    # One curve per unroll (and per DF step for eta)
    for key, value in model.learned_params().items():
        history["params"].setdefault(key, []).append(value)
        values = np.stack(history["params"][key], axis=0)
        title  = PARAM_TITLES[key]

        save_curves(values.reshape(len(values), -1), os.path.join(exp_dir, f"{title}_Plot.png"),
                    ylabel=title, title=f"{title} Parameter Over Time")
        sio.savemat(os.path.join(exp_dir, f"{title}.mat"), {key: values})


if __name__ == '__main__':
    set_seed(42)
    cfg      = load_config("Config.yaml")
    device   = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    R        = cfg["R"]
    paths    = {k: v.format(R=R) if v else None for k, v in cfg["paths"].items()}
    n_frames = R if is_spatiotemporal(cfg) else 1
    exp_name = experiment_name(cfg)
    exp_dir  = os.path.join(paths["train_results"], exp_name)

    subfolders = ["model", "files"] + (["Val_pngs"] if paths.get("val_data") else []) \
                                    + (["pngs"] if cfg.get("monitor_subjects") else [])
    for sub in subfolders:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    # ---------------- Data ----------------
    train_set    = DataLoaderSL(paths["train_data"], paths["coils"], n_frames=n_frames)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    val_loader   = None
    if paths.get("val_data"):   # without validation data, the best model is selected on the training loss
        val_set    = DataLoaderSL(paths["val_data"], paths["coils"], n_frames=n_frames)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

    # ---------------- Model ----------------
    model = build_model(cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {trainable_params:,}')
    with open(os.path.join(exp_dir, "files", "Config.yaml"), "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    with open(os.path.join(exp_dir, "files", "trainable_parameters.txt"), "w") as f:
        f.write(f'Total number of trainable parameters: {trainable_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=cfg["gamma_scheduler"],
                                                           patience=cfg["patience"])

    # ---------------- Training loop ----------------
    history   = {"train": [], "val": [], "lr": [], "params": {}}
    best_loss = float("inf")
    val_loss  = float("inf")

    for epoch in range(cfg["n_epochs"] + 1):
        print(f"EXP. Name: {exp_name}    ,, GPU: {device}")

        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch, val_loss,
                                     os.path.join(exp_dir, "pngs"))
        if val_loader is not None:
            val_loss = validate(model, val_loader, cfg, device, os.path.join(exp_dir, "Val_pngs"))
            history["val"].append(val_loss)
        monitored = val_loss if val_loader is not None else train_loss

        if monitored < best_loss:
            best_loss = monitored
            torch.save({"model_state_dict": model.state_dict(),
                        "train_loss"      : train_loss,
                        "val_loss"        : val_loss if val_loader is not None else None},
                       os.path.join(exp_dir, "model", f"BestModel_R{R}_Epoch{epoch}.pth"))

        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(monitored)

        history["train"].append(train_loss)
        log_epoch(exp_dir, history, model)
