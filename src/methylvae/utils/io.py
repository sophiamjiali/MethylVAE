# ==============================================================================
# Script:           io.py
# Purpose:          Utility functions for reading and writing
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             11/18/2025
# ==============================================================================

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from methylvae.constants import (
    ANNOTATION_27K,
    ANNOTATION_450K,
    ANNOTATION_EPIC
)

# =====| File I/O Utilities |===================================================

def build_meta_fields(fields):
    meta = []
    for f in fields:
        if '.' in f:
            parts = f.split('.')
            if len(parts) == 1:
                meta.append(parts[0])
            else:
                meta.append(parts)
    return meta


def load_cpg_matrix(files):
    """
    Returns a CpG x Samples matrix given a list of individual parquet files
    holding DNA methylation beta values of a given sample. Columns of the matrix
    are the sample IDs of each sample (file name without the extension).

    Parameters
    ----------
    files (list): list of `pathlib.path` paths to parquet files

    Returns
    -------
    (DataFrame): a CpG x Samples matrix of beta values for the provided dataset
    """

    # Load all beta values in parallel
    with ThreadPoolExecutor() as ex:
        beta_values = list(ex.map(load_beta_file, files))

    # Concatenate on the index to build the matrix
    cpg_matrix = pd.concat(beta_values, axis = 1, join = "outer")
    cpg_matrix = cpg_matrix.sort_index()

    return cpg_matrix


def load_annotation(manifests):
    """
    Loads and returns the dominant Illumina methylation manifest
    (EPIC > 450K > 27K) out of the manifests present in the dataset as
    provided by `manifests`, along with its name.

    Parameters
    ----------
    manifests (list): list of strings of manifests present in the dataset

    Returns 
    -------
    (Tuple): the dominant manifest present in `manifests` and its name

    Raises
    ------
    ValueError: if no viable manifest was provided
    """

    if ("Illumina Human Methylation EPIC" in manifests):
        annotation = pd.read_csv(ANNOTATION_EPIC)
        array_type = "Illumina Human Methylation Epic"
    elif ("Illumina Human Methylation 450" in manifests):
        annotation = pd.read_csv(ANNOTATION_450K)
        array_type = "Illumina Human Methylation 450"
    elif ("Illumina Human Methylation 27" in manifests):
        annotation = pd.read_csv(ANNOTATION_27K)
        array_type = "Illumina Human Methylation 27"
    else:
        raise ValueError ("No valid manifest provided in `manifests`.")
    
    return (annotation, array_type)


def load_beta_file(path):
    """
    Loads a singles-sample beta value .parquet file with the CpG probe ID as
    the index and the sample ID (filename without extension) as the beta value
    column name.

    Parameters
    ----------
    path (str): path to a .parquet file containing beta values

    Returns 
    -------
    beta_values (DataFrame): a dataframe of CpGs x Sample ID
    """

    sample_id = path.stem
    beta_values = pd.read_parquet(path)
    beta_values = beta_values.rename(columns = {"beta_value": sample_id})
    return beta_values

# [END]