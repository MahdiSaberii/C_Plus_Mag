# C+Mag

[![Paper](https://img.shields.io/badge/IEEE%20TBME-Paper-blue)](https://ieeexplore.ieee.org/abstract/document/11735101)
[![arXiv](https://img.shields.io/badge/arXiv-2608.18036-b31b1b)](https://arxiv.org/abs/2608.18036)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Harnessing Magnitude-Only and Complex Measurements for Improved Dynamic MRI Reconstruction with Learned Priors**
Mahdi Saberi, Yaşar Utku Alçalar, Merve Gülle, Chetan Shenoy, Mehmet Akçakaya
*IEEE Transactions on Biomedical Engineering (TBME), 2026*

<p align="center">
  <img src="data/Pipeline.png" width="100%">
</p>

## Abstract

<p align="justify"> MRI reconstruction methods naturally use the acquired complex-valued k-space measurements, while phase retrieval theory suggests that magnitude-only measurements can provide complementary information. We show that in steady-state dynamic MRI, k-space magnitudes are highly consistent across time-frames, so neighboring frames supply informative magnitude measurements at unacquired locations with no additional scan time. Building on this, we propose <b>ℂ+Mag</b>, a magnitude-informed physics-driven deep learning (PD-DL) method that uses ADMM-based unrolling with a novel magnitude-aware data-fidelity formulation. On retrospectively undersampled cine and phase-contrast flow MRI, as well as prospectively undersampled real-time cine, <b>ℂ+Mag</b> improves artifact suppression, anatomical sharpness, and phase preservation over conventional PD-DL, further supported by blinded expert reader evaluations. </p>

## Repository Structure

```
C_Plus_Mag/
├── configs/          # Training and testing configurations
├── data/             # Data loading and preprocessing
├── models/           # Unrolled network, regularizer, data-consistency blocks
├── utils/            # FFT, coil, metric, and visualization helpers
├── train.py
├── test.py
└── assets/           # README figures
```

## Installation

```bash
git clone https://github.com/MahdiSaberii/C_Plus_Mag.git
cd C_Plus_Mag
conda create -n cplusmag python=3.10
conda activate cplusmag
pip install -r requirements.txt
```

## Data

The experiments use [dataset name / source]. Each slice is stored as multi-coil k-space with shape `[frames, coils, H, W]`, with coil sensitivity maps in a separate folder. See [`data/README.md`](data/README.md) for the expected directory layout and preprocessing steps.

## Usage

**Training**
```bash
python train.py --config configs/cine_R8.yaml
```

**Testing**
```bash
python test.py --config configs/cine_R8.yaml --checkpoint checkpoints/cine_R8.pth
```

Pretrained weights for R = 6, 8, 12, and 16 are available [here](LINK).

## Citation

If you find this work useful, please cite:

```bibtex
@article{saberi2026cplusmag,
  title   = {[Paper Title]},
  author  = {Saberi, Mahdi and [Co-authors] and Ak{\c{c}}akaya, Mehmet},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2026},
  doi     = {[DOI]}
}
```

## Contact

For questions, please open an issue or reach out via [mahdisaberii.github.io](https://mahdisaberii.github.io).

This work was done at the [Center for Magnetic Resonance Research (CMRR)](https://cmrr.umn.edu), University of Minnesota.

This work was done at the [Center for Magnetic Resonance Research (CMRR)](https://cmrr.umn.edu), University of Minnesota.
