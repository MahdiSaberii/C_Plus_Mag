import glob
import os
import re

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

FILE_PATTERN = r"subj_(\d+)_slice_(\d+)_T(\d+)"


def frame_number(path):
    return int(re.search(r"_T(\d+)\.mat$", os.path.basename(path)).group(1))


class DataLoaderSL(Dataset):
    """
    One sample per file `subj_XXXX_slice_XXX_TXXX.mat` in `data_path`, containing
    kspace_t, r, mask_t, nPhase, normalization and scanner_id.
    Coil maps are read from `coil_path/fs_XXXX_<scanner>_Slice_<n>.mat` (key "maps").

    n_frames = 1 : spatial, each sample is the single frame T.
    n_frames = R : spatiotemporal, each sample stacks R frames of the same slice:
        blocks=False (training): for every residue mod R, the frame nearest to T (within 1..max_frame).
        blocks=True  (testing) : the contiguous block T..T+R-1, for T = 1, R+1, 2R+1, ...

    Returns arrays with a leading frame axis: ksp/r [T, Nc, H, W], mask [T, H, W], frames [T].
    """

    def __init__(self, data_path, coil_path, n_frames=1, blocks=False, max_frame=25):
        self.root      = data_path
        self.coil_path = coil_path
        self.n_frames  = n_frames
        self.blocks    = blocks
        self.max_frame = max_frame

        files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
        if n_frames > 1:
            last_start = max_frame - n_frames + 1
            files = [f for f in files if frame_number(f) <= last_start]
            if blocks:
                files = [f for f in files if (frame_number(f) - 1) % n_frames == 0]
        self.datapath = files
        print(f"------> # Samples in {data_path}: {len(self.datapath)}")

    def __len__(self):
        return len(self.datapath)

    def nearest_frames(self, T):
        """For each residue mod R, the frame within 1..max_frame closest to T."""
        R, frames = self.n_frames, []
        r0 = (T - 1) % R
        for res in range(R):
            t_plus     = T + (res - r0) % R
            t_minus    = t_plus - R
            candidates = [t for t in [t_plus, t_minus, t_plus + R, t_minus - R] if 1 <= t <= self.max_frame]
            if not candidates:
                raise ValueError(f"No frame found for residue {res} within 1..{self.max_frame}")
            frames.append(min(candidates, key=lambda t: abs(t - T)))
        return frames

    def select_frames(self, T):
        if self.n_frames == 1:
            return [T]
        if self.blocks:
            return list(range(T, T + self.n_frames))
        return self.nearest_frames(T)

    def __getitem__(self, index):
        file_path = self.datapath[index]
        FileName  = os.path.basename(file_path)

        match = re.match(FILE_PATTERN, FileName)
        if not match:
            raise ValueError(f"Unexpected filename format: {FileName}")
        SubjectNumber, SliceNumber, TimeFrame = (int(g) for g in match.groups())

        mat           = sio.loadmat(file_path)
        total_phases  = mat["nPhase"]
        normalization = mat["normalization"]
        frames        = self.select_frames(TimeFrame)

        ksp, r, mask = [], [], []
        for t in frames:
            frame_path = os.path.join(self.root, f"subj_{SubjectNumber:04d}_slice_{SliceNumber:03d}_T{t:03d}.mat")
            if t != TimeFrame and not os.path.exists(frame_path):
                raise FileNotFoundError(f"Adjacent frame not found: {frame_path}")
            frame_mat = mat if t == TimeFrame else sio.loadmat(frame_path)
            ksp.append(frame_mat["kspace_t"].transpose(2, 0, 1))
            r.append(frame_mat["r"].transpose(2, 0, 1))
            mask.append(np.squeeze(frame_mat["mask_t"]))

        # e.g. "1.5 T" -> "1_5T"
        scanner    = str(np.squeeze(mat["scanner_id"])).replace(" ", "").replace(".", "_")
        coil_fpath = os.path.join(self.coil_path, f"fs_{SubjectNumber:04d}_{scanner}_Slice_{SliceNumber}.mat")
        if not os.path.exists(coil_fpath):
            raise FileNotFoundError(f"Coil file not found: {coil_fpath}")
        coil = sio.loadmat(coil_fpath)["maps"].transpose(2, 0, 1)

        return (np.stack(ksp), np.stack(r), np.stack(mask), SliceNumber, SubjectNumber, TimeFrame,
                coil, FileName, total_phases, normalization, np.array(frames))
