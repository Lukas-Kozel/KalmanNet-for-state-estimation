# 🛰️ KalmanNet & Bayesian Filtering for State Estimation

> **AI-aided and statistical filtering for nonlinear state estimation — from synthetic benchmarks to real-world sensor fusion (NCLT dataset) and Terrain Aided Navigation (TAN).**

This repository contains the full research codebase accompanying the Master's Thesis: **"State Estimation using Statistical and Machine Learning Methods"** (Lukáš Kozel, 2026, University of West Bohemia). 

It provides a comprehensive, unified framework comparing classical statistical filters (Kalman, EKF, UKF, Particle Filters, Auxiliary PF), adaptive methods (Mehra-type, MDM - Measurement Difference Method), and hybrid **model-based deep learning** architectures — namely **KalmanNet** and **Bayesian KalmanNet (BKN)** with epistemic uncertainty quantification via Monte Carlo Dropout.

The core hypothesis: when system noise statistics are unknown or mismatched, **learning the Kalman gain directly from data** (KalmanNet) can match or outperform optimally tuned classical estimators — and Bayesian extensions make the resulting estimates *trustworthy* by exposing predictive uncertainty.

Built in **Python 3.10 + PyTorch** (CUDA-accelerated).

---

## 🗺️ Repository Map

| Path | Description |
|------|-------------|
| 📓 **`main.ipynb`** | **Primary entry point.** Main experimental notebook: system setup, training, and evaluation of filters vs. neural estimators. |
| 📓 **`hyperparams_tests.ipynb`** | Hyperparameter exploration notebook (learning rates, GRU sizes, weight initializations, etc.). |
| 📁 **`Filters/`** | Classical & statistical state estimators. Base classes and many variants at the top level; `Filters/NCLT/` and `Filters/TAN/` contain system-tuned implementations. |
| 🧠 **`NN_models/`** | Legacy neural network estimators (PyTorch). These models process data in **batch mode** (taking the entire dataset trajectory at once). |
| 📊 **`state_NN_models/`** | **Stateful** neural network estimators (KalmanNet, BayesianKalmanNet). These process data step-by-step (online fashion) and explicitly maintain hidden states. *Note: The `KalmanFormer` implementation found here was purely an experimental attempt and its correctness is not guaranteed.* |
| ⚙️ **`Systems/`** | State-space models / plant simulators: `LinearSystem.py`, `NonLinearSystem.py`, `DynamicSystemNCLT.py`, etc. |
| 🧪 **`testcases/main/`** | End-to-end experimental scenarios and quick validation setups. |
| 🧮 **`MDM/`** | Auxiliary model/method implementations for the Measurement Difference Method. |
| 📐 **`TAN/`** | Experimentation area for the Terrain Aided Navigation system model. |
| 🛠️ **`utils/`** | Shared tooling: `trainer.py` (training loop), `losses.py` (MSE/NLL-type objectives), `utils.py` (helpers, metrics). |
| 🚗 **`navigation NCLT dataset/`** | **All real-data navigation work lives here.** Includes downloaders, preprocessors, Optuna tuning, and complex multi-sensor fusion notebooks. |
| 📚 **`literature/`** | Reference papers used during development. |
| 💾 **`data/`** | Location for datasets / data artifacts referenced by the notebooks. |

### 🚗 `navigation NCLT dataset/` — in detail

This folder is the real-world backbone of the project — a full pipeline from raw NCLT logs to trained sensor-fusion estimators:

- **`downloader.py`** — fetches the NCLT dataset sessions used in this project.
- **`preprocess*.py`** — turn raw logs into ground-truth + measurement `.mat`-style trajectories consumed by `DynamicSystemNCLT`.
- **`read_groud_truth.py`** — parsing of GNSS/ground-truth poses.
- **`find_optimal_params*.py`** — offline tuning (incl. Optuna-based search) of filter noise covariances & NN training settings.
- **`NCLT_complex_navigation/`** — the main nonlinear navigation experiments.
- **`NCLT_linear_model_test/`** — simplified linear-model testbench for NCLT trajectories.
- **`linear_velocity_integration-deprecated/`** — *deprecated* earlier iteration (kept for reproducibility/traceability).
- **`NN_weights/`**, **`best_kalmannet_nclt_sensor_fusion.pth`** — saved checkpoints, including our best sensor-fusion KalmanNet.
- **`results/`**, **`NCLT_Graphs/`** — evaluation outputs and generated figures.
- **`preprocessed_NCLT_FULL`**, **`preprocessed_NCLT_trajectory-2012-01-22-angle-update/`** — ready-to-load preprocessed trajectories.
- **`Mapa do diplomky.pdf`**, **`QGIS/`**, **`QGIS_setup.qgz`**, **`generate_ref_trajectory_graf.ipynb`** — map assets & reference-trajectory visualization.

---

## 🚀 Getting Started

**Prerequisites:** Python 3.10, PyTorch (with CUDA if you want GPU training), Jupyter. Install the usual scientific stack (`numpy`, `scipy`, `matplotlib`, `optuna` for the tuning scripts).

Recommended path for a first visit:

1. **Start with `main.ipynb`** — it wires a `Systems/` model to the filters and KalmanNet, trains, and evaluates. This is the canonical usage example.
2. **Explore `testcases/main/`** and **`testcases/sanity_check/`** for focused experiment scenarios and quick correctness checks.
3. **Tune with `hyperparams_tests.ipynb`** once you want to vary architecture/training hyperparameters.
4. **Move to real data:** inside `navigation NCLT dataset/`, use `downloader.py` → `preprocess*.py` → the experiments in `NCLT_complex_navigation/`.
5. Training utilities (`utils/trainer.py`) and loss definitions (`utils/losses.py`) are shared across all experiments.

> Note: checkpoints and trajectories are plain `.pth` / `.mat` files, so everything can also be driven from plain Python scripts — notebooks are not a hard requirement.

---

## 📦 Data (NCLT & TAN)

Experiments use the publicly available **NCLT (North Campus Long-Term) dataset** (University of Michigan) and custom **Terrain Aided Navigation (TAN)** synthetic setups. The raw datasets are **not committed** to this repository due to size.

- Run **`navigation NCLT dataset/downloader.py`** to retrieve the required NCLT sessions.
- Run the **`preprocess*.py`** scripts (same folder) to generate the trajectory/measurement files (`preprocessed_NCLT_*` directories) that `Systems/DynamicSystemNCLT.py` loads at runtime.
- Preprocessed outputs of the reference runs are kept in-tree so results are reproducible without re-downloading.

---

## 📖 Citation & Core References

If you use this codebase in your research, please cite the foundational Master's Thesis:

```bibtex
@mastersthesis{Kozel2026StateEstimation,
  author  = {Luk\'{a}\v{s} Kozel},
  title   = {Odhad stavu pomoc\'{i} statistick\'{y}ch metod a metod zalo\v{z}en\'{y}ch na strojov\'{e}m u\v{c}en\'{i} [State Estimation using Statistical and Machine Learning Methods]},
  school  = {University of West Bohemia, Faculty of Applied Sciences},
  year    = {2026},
  address = {Pilsen, Czech Republic}
}
```

This repository builds upon and implements methodologies from several key publications. Please consider citing the original authors if you utilize their respective architectures or methods:

**1. KalmanNet & Bayesian KalmanNet:**
```bibtex
@article{revach2022kalmannet,
  title   = {KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics},
  author  = {Revach, Guy and Shlezinger, Nir and Ni, Xiaoyong and Escoriza, Adria Lopez and van Sloun, Ruud J. G. and Eldar, Yonina C.},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2022},
  doi     = {10.1109/TSP.2022.3158588}
}

@article{dahan2025bayesian,
  title   = {Bayesian KalmanNet: Quantifying Uncertainty in Deep Learning Augmented Kalman Filter},
  author  = {Dahan, Yehonatan and Revach, Guy and Dunik, Jindrich and Shlezinger, Nir},
  journal = {arXiv preprint arXiv:2309.03058},
  year    = {2025}
}
```

**2. Adaptive Filtering (MDM) & Terrain Aided Navigation (TAN):**
```bibtex
@article{kost2024mdm,
  title   = {Noise Covariances Identification by MDM: Weighting, Recursion, and Implementation},
  author  = {Kost, Oliver and Dun\'{i}k, Jind\v{r}ich and Straka, Ond\v{r}ej},
  journal = {IFAC-PapersOnLine},
  volume  = {58},
  number  = {15},
  year    = {2024},
  doi     = {10.1016/j.ifacol.2024.08.552}
}

@article{matousek2025tan,
  title   = {Lagrangian Grid-Based Filters With Application to Terrain-Aided Navigation},
  author  = {Matou\v{s}ek, Jakub and Dun\'{i}k, Jind\v{r}ich and Straka, Ond\v{r}ej},
  journal = {IEEE Signal Processing Magazine},
  year    = {2025},
  doi     = {10.1109/MSP.2024.3489969}
}
```

---

## 🤝 License & Contributing

Contributions, issues, and feature requests are very welcome — whether it's a new filter variant, an additional benchmark system, or documentation improvements.

1. Fork the repository and create a feature branch.
2. Keep the existing code style; comment non-obvious logic.
3. Verify your change through a relevant notebook in `testcases/` or `main.ipynb` before opening a PR.

For questions and research collaboration, please open an issue.

