# ==============================================================================
# Script:           constants.py
# Purpose:          Defines global constants used across the workflow
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             11/22/2025
# ==============================================================================

from pathlib import Path

# =====| GDC API |==============================================================

# Configurations for donwloading data using the GDC API client
CHUNK_SIZE = 25
MAX_RETRIES = 5
RETRY_SLEEP = 15

# =====| Paths |================================================================

# Compute project path relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Compute default paths
CONFIG_DIR = PROJECT_ROOT / "config"

# Compute the GDC-Client path
GDC_CLIENT_PATH = PROJECT_ROOT / "tools/gdc-client"

# =====| Data Directory |=======================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_METHYLATION_DIR = DATA_DIR / "raw" / "methylation"
PROCESSED_METHYLATION_DIR = DATA_DIR / "processed" / "methylation"
METADATA_METHYLATION_DIR = DATA_DIR / "metadata" / "methylation"
TRAINING_METHYLATION_DIR = DATA_DIR / "training" / "methylation"

RAW_WSI_DIR = DATA_DIR / "raw" / "wsi"
INTERMEDIATE_WSI_DIR = DATA_DIR / "intermediate" / "wsi"
PROCESSED_WSI_DIR = DATA_DIR / "processed" / "wsi"
METADATA_WSI_DIR = DATA_DIR / "metadata" / "wsi"
TRAINING_WSI_DIR =  DATA_DIR / "training" / "wsi"

# =====| Model Training |=======================================================

BETAVAE_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "beta_vae"
BETAVAE_SWEEP_DIR = PROJECT_ROOT / "experiments" / "beta_vae"

CDM_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "cdm"
CDM_SWEEP_DIR = PROJECT_ROOT / "experiments" / "cdm"

# =====| Probe Annotations |====================================================

RESOURCES_DIR = PROJECT_ROOT / "resources"
ANNOTATION_27K = RESOURCES_DIR / "illumina27k_annotation_hg19.csv"
ANNOTATION_450K = RESOURCES_DIR / "illumina450k_annotation_hg19.csv"
ANNOTATION_EPIC = RESOURCES_DIR / "illuminaEPIC_annotation_hg19.csv"

# =====| GDC Metadata |=========================================================

# Define DNA Methylation metadata fields
METADATA_METHYLATION = [

    # File-Level Metadata
    "project.project_id",
    "file_id",
    "file_name",
    "data_category",
    "data_type",
    "experimental_strategy",
    "platform",
    "state",
    "data_format",
    "cases.case_id",
    "cases.submitter_id",
    "cases.samples.sample_id",
    "cases.samples.sample_type",

    # Clinical Metadata
    "cases.diagnoses.age_at_diagnosis",
    "cases.demographic.gender",
    "cases.demographic.race",
    "cases.demographic.ethnicity",
    "cases.diagnoses.primary_diagnosis",
    "cases.diagnoses.morphology",
    "cases.diagnoses.tumor_stage",
    "cases.diagnoses.tumor_grade",
    "cases.diagnoses.days_to_death",
    "cases.diagnoses.days_to_last_follow_up",
    "cases.diagnoses.vital_status",
    "cases.prior_malignancy",
    "cases.sites_of_resection_or_biopsy"
]