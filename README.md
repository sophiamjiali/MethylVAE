# MethylCDM
Generative Modeling of Tumour Histology from Differential Methylation

To run a sweep: `sbatch slurm/betaVAE_sweep.sh`

# Repository Structure

```
MethylCDM/
├── config/             # Pipeline Configuration Files
├── data/               # Input & Output Data
├── experiments/        # Training Logs, Metrics, & Figures
├── models/             # Model Checkpoints
├── notebooks/          # Exploratory Notebooks for Development
├── scripts/            # Pipeline Entry-points
├── src/                # Core Reusable Modules
└── workflow/           # SnakeMake Pipeline Orchestration
```

## Scripts
The `scripts` folder holds the entry-points to the pipeline, using the logic defined in the `src` folder. Shell wrappers exist to run the pipeline for cluster/HPC.
```
└── script/
    ├── preprocess_methylation.py   # Preprocess methylation data
    ├── preprocess_wsi.py           # Preprocess WSI data
    ├── train_beta_vae.py           # Train β-VAE 
    ├── train_diffusion.py          # Train CDM
    ├── eval_beta_vae.py            # Evaluate β-VAE
    ├── eval_diffusion.py           # Evaluate CDM
```

## SRC
The `src` folder holds the core logic and main stateless implementation of the workflow. The functions defined are used in the `scripts` folder to run the full pipeline.
```
└── src/
    ├── data/               # Dataset & Loader Logic
    ├── preprocessing/      # Methylation & WSI Preprocessing Logic
    ├── models/             # Model Definitions
    ├── training/           # Training Logic
    ├── evaluation/         # Evaluation Logic
    └── utils/              # General Utilities

```
