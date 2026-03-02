# Instructions

Instructions for an oblivious developer (me)

## 1. Download and process DNA methylation data

```
conda activate methylcdm-env
pip install -e .
python ./scripts/process_methylation.py \
  --project {TCGA-###} \
  --config_pipeline pipeline.yaml \
  --config_preproc methylation_preproc.yaml \
  --verbose True
```

## 2. Prepare DNA methylation data for model training
```
conda activate methylcdm-env
pip install -e .
python ./scripts/prepare_data.py \
  --config_pipeline pipeline.yaml \
  --config_betaVAE betaVAE.yaml \
  --verbose True
```

## 3. Train Beta-VAE by initializing a hyperparameter sweep

If running locally:
```
conda activate methylcdm-env
pip install -e .
python ./scripts/train_betaVAE.py \
  --config_pipeline pipeline.yaml \
  --config_train betaVAE.yaml \
  --verbose True
```

If running on HPC:

```
# Import files to HPC; everything except intermediate/processed data other than training data
cd /Volumes/FBI_Drive/MethylCDM-project
scp -r config/ data/training/ devnotes/ sophiali@dev1gpu.mshri.on.ca:/ddn_exa/campbell/sli/methylcdm-project
scp -r experiments/ models/ notebooks/ resources/ scripts/ src/ tools/ wandb/ workflow/
scp -r
```





```
# Import files to HPC
cd /Volumes/FBI_Drive/MethylCDM-project
scp -r {directory} sophiali@dev1gpu.mshri.on.ca:/ddn_exa/campbell/sli/methylcdm-project

# Log into HPC
ssh sophiali@dev1gpu.mshri.on.ca
cd /ddn_exa/campbell/sli/methylcdm-project
```