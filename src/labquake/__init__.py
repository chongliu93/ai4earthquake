# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: __init__.py -- labquake: forecasting the mechanical state of a fault from one seismic channel
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""labquake: forecasting the mechanical state of a fault from one seismic channel.

Modules follow the three modules of the paper's framework plus the
explanation step:

    pennstate  stage 0  the laboratory records from the Penn State archive
    kilauea    stage 0  the Kīlauea 2018 records, preprocessed as in Johnson & Johnson (2024)
    features   Module 1  rolling variance and kurtosis of the raw signal
    embedding  Module 2  time-delay (Hankel) embedding, PCA(3) and UMAP(2)
    model      Module 3  random-forest regression (BayesSearchCV, Darts)
    explain    feature2target (SHAP), feature2physics (Legendre jets),
               physics2target (correlation with the target)
    io         configurations and the HDF5 files passed between stages
"""
__version__ = "1.0.0"
