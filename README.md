# ai4earthquake

Code for **"From waveforms to fault state across scales with explainable AI"**
(Liu, Elbertsen, Vasconcelos, Hu, Regenauer-Lieb & Niemeijer, submitted,
2026).

A single interpretable framework forecasts the mechanical state of a fault
from one continuous seismic channel and explains what the forecast depends
on. It is applied to three laboratory stick-slip experiments (p4581, p4679,
p5198; shear stress as target) and to the 2018 caldera-collapse sequence of
Kīlauea (ground displacement as target).

```
raw signal ──► Module 1 ──► Module 2 ──► Module 3 ──► forecast
               rolling      delay        random
               variance,    embedding,   forest
               kurtosis     PCA(3)+UMAP(2)
                                 │
                                 └──► explanation: SHAP (feature2target),
                                      Legendre jets (feature2physics),
                                      correlation with target (physics2target)
```

## Repository layout

| Path | Content |
|---|---|
| `src/labquake/` | the library: one module per stage (`features`, `embedding`, `model`, `explain`, `kilauea`, `io`) |
| `scripts/` | numbered entry points, one per stage (`00` raw record … `05` ablation); each reads a config and writes to `outputs/` |
| `configs/` | one YAML per system with W, S, D, the train/test split and the model settings (Supplementary Table 1 of the paper) |
| `data/` | not tracked; where the raw records go (see Data) |
| `tests/` | unit tests on synthetic signals |

## Installation

```
git clone https://github.com/chongliu93/ai4earthquake.git
cd ai4earthquake
conda env create -f environment.yml
conda activate labquake
```

`pip install -e .` works as well.

## Data

Raw records and processed data are not in the repository.

- **Laboratory experiments p4581, p4679, p5198** (Penn State biaxial
  apparatus): acoustic emission and mechanical records from the Penn State
  rock-mechanics data archive,
  http://psudata.s3-website.us-east-2.amazonaws.com/. Expected layout:
  `data/raw/<experiment>/`.
- **Kīlauea 2018 collapse sequence**: seismic waveforms (station RIMD, channel
  HHN) from EarthScope SAGE, HVO network, https://doi.org/10.7914/SN/HV; GNSS
  displacement (station CRIM) from the EarthScope consortium. Preprocessing
  follows Johnson & Johnson (2024), https://github.com/cjohnson-LANL/GRL_Kilauea;
  `src/labquake/kilauea.py` documents which of its outputs stage 0 reads and
  the single departure (one seismic component instead of three). Expected
  layout: `data/raw/kilauea/`.
- **Processed data** (the CSV files behind every figure of the paper): Yoda,
  Utrecht University, https://doi.org/10.24416/UU01-Y5MLH1 (CC BY 4.0).

## Reproducing the computation

The six stages in `scripts/` run in order for one system at a time. Every
stage reads the previous stage's files from `outputs/<system>/` and writes
its own:

```
python scripts/00_build_raw.py         --config configs/p5198.yaml
python scripts/01_compute_features.py  --config configs/p5198.yaml
python scripts/02_embed.py             --config configs/p5198.yaml --target
python scripts/03_train_rf.py          --config configs/p5198.yaml
python scripts/04_explain.py           --config configs/p5198.yaml
python scripts/05_ablation.py          --config configs/p5198.yaml
```

| Stage | Methods section of the paper | Output (`outputs/<system>/`) |
|---|---|---|
| 0 | Double direct shear tests; Kīlauea 2018 collapse sequence | `00_raw/record.h5`: time, signal, target at every sample |
| 1 | Rolling statistics (Module 1) | `01_features/features.h5`: variance and kurtosis over the trailing window W |
| 2 | Time-delay embedding (Module 2); target-side reconstruction (Fig. 3) | `02_embedding/{var,kurt}.h5` (and `target.h5` with `--target`): strided value, PCA1-3, UMAP1-2 for train and test |
| 3 | Random-forest integrator (Module 3); Model training and evaluation | `03_model/`: feature table, forecast, forest, summary with R² |
| 4 | Feature importance; Physics-based interpretation of the learned coordinates; Correlation of the physical quantities with the target | `04_explain/`: SHAP importance, Legendre decomposition, correlations, importance by quantity |
| 5 | Fig. 2f | `05_ablation/`: R² of the instantaneous, raw-lag and twelve-feature forests |

`configs/*.yaml` hold the parameters of Supplementary Table 1 of the paper.

**Parameter study.** The 272 configurations of Fig. 4 and Supplementary
Fig. 7 are stages 1-4 repeated over the grid of window W, stride S and
embedding dimension D listed in Supplementary Data 1 (one row per
configuration). To rerun one of them, copy the system's configuration,
change `window` and the `embedding` entries `S` and `D`, and run stages 1-4;
the per-configuration tables of Supplementary Data 1 and 2 are the
`summary.json` and `04_explain/` outputs of each run. We ran the grid on a
SLURM cluster, one job per configuration; no scheduler scripts are included.

## Tests

```
pytest tests/
```

runs stages 1-5 on a synthetic stick-slip record and checks the file layouts, the row alignment of the target-side embedding,
and that the Darts and scikit-learn paths of stage 3 give the same forest.

## Citation

Paper: Liu, C. *et al.* From waveforms to fault state across scales with
explainable AI. Submitted (2026).

Data: Liu, C. *et al.* Processed data for "From waveforms to fault state across
scales with explainable AI". Yoda, Utrecht University.
https://doi.org/10.24416/UU01-Y5MLH1

Software: this repository, version 1.0.0 (`CITATION.cff` holds the
machine-readable form).

## Licence

MIT (see `LICENSE`). The processed data on Yoda are CC BY 4.0.
