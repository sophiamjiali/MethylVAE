# MethylVAE

> **Component 1 of:** *Generative Modeling of Tumour Histology from Differential Methylation*

MethylVAE is a β-variational autoencoder (β-VAE) that learns a compact latent representation of DNA methylation profiles. It serves as the upstream model in a two-stage generative pipeline: the learned latent embeddings are subsequently consumed by a downstream conditional diffusion model (CDM) to synthesise tumour histology images.

---

## Overview

DNA methylation is a stable epigenetic mark that varies systematically across cancer types and tumour microenvironments. MethylVAE encodes high-dimensional methylation arrays into a low-dimensional continuous latent space, disentangling biologically meaningful axes of variation via the β penalty on the KL divergence term of the ELBO. The resulting embeddings provide structured, smooth conditioning signals for downstream generative modelling of whole-slide image (WSI) patches.

---

## Repository Structure

```
MethylVAE/
├── config/             # YAML configuration files (pipeline & model hyperparameters)
├── devnotes/           # Development notes & experiment logs
├── notebooks/          # Exploratory Jupyter notebooks
├── plots/              # Evaluation figures (reconstruction quality, latent space)
├── resources/          # Reference materials
├── scripts/            # Pipeline entry-points
├── slurm/              # SLURM batch scripts for HPC execution
├── src/
│   └── MethylCDM/      # Core source modules
├── environment.yaml        # CPU conda environment
├── environment_gpu.yaml    # GPU conda environment (CUDA 12.1)
└── pyproject.toml          # Package metadata
```

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/train_betaVAE.py` | Train the β-VAE |
| `scripts/eval_betaVAE.py` | Evaluate a trained β-VAE checkpoint |
| `scripts/sweep_betaVAE.py` | Run / report an Optuna hyperparameter sweep |

---

## Installation

**GPU environment (recommended):**
```bash
conda env create -f environment_gpu.yaml
conda activate methylcdm-env
pip install -e .
```

**CPU environment:**
```bash
conda env create -f environment.yaml
conda activate methylcdm-env
pip install -e .
```

Key dependencies: Python 3.11, PyTorch, PyTorch Lightning ≥ 2.6, Optuna, Weights & Biases, scikit-learn, h5py.

---

## Usage

### Training

To train a single iteration of the model using constant hyperparameters (as defined in betaVAE_train_config.yaml).

```bash
python scripts/train.py \
  --config_pipeline pipeline.yaml \
  --config_train betaVAE_train.yaml \
  --seed 42 \
  --verbose True
```

```bash
sbatch slurm/train.sh
```

### Hyperparameter Sweep (SLURM)

Launch a sweep on a SLURM cluster:

```bash
sbatch slurm/sweep.sh
```

Report results from a completed sweep:

```bash
python scripts/sweep.py \
  --config_pipeline pipeline.yaml \
  --config_train betaVAE.yaml \
  --study_name <study_name> \
  --report_only
```

### Evaluation

```bash
python scripts/eval_betaVAE.py \
  --config_pipeline pipeline.yaml \
  --config_train betaVAE.yaml
```

### Projection

To generate embeddings.

```bash
sbatch slurm/project.sh
```

```bash
python scripts/project.py \
> --checkpoint <path-to-checkpoint> \
> --data_path <path-to-h5ad> \
> --out_dir <path-to-output-dir> \
> --batch_size 512 \
> --name pancancer \
> --split_projects
```

---

## Configuration

Model and pipeline behaviour is controlled via YAML files in `config/`. Key parameters include the β coefficient (KL weight), latent dimensionality, encoder/decoder architecture, learning rate, and batch size. Refer to the files in `config/` for full documentation of available options.

---

## Pipeline Context

```
Methylation array  ──►  MethylVAE (this repo)  ──►  latent z  ──►  CDM  ──►  WSI patch
```

MethylVAE is **Component 1** of the pipeline. The downstream conditional diffusion model (Component 2) conditions image generation on the latent codes produced here.
