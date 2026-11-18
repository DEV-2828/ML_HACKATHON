"""PathSnap Master Technical Report Generator & Compiler.

Generates `docs/PathSnap_Technical_Report.md` and compiles `docs/PathSnap_Technical_Report.pdf`
(Deliverable 2) strictly adhering to Deliverables.pdf Sections 2.2 through 8, using 100%
verified experimental outputs from `data_geo/outputs/`, `results/metrics/`, and `PLOT/`.

To refresh the PDF after replacing `PLOT/ui_demo_screenshot.jpg` or `PLOT/cmd_api_screenshot.png`:
    python scripts/compile_report.py
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_MD = r"""# PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION
## Consolidated Technical Report (Deliverable 2)

**Course:** UE24CS352A Machine Learning — PES University, CSE (RR Campus)  
**Team 3:** Devopam Pal (`PES2UG24CS152`), Deepthi N Reddy (`PES2UG24CS149`), Dwijesh Krishna Ghattamaraju (`PES2UG24CS163`), Hanumantha L (`PES2UG24CS179`)  
**Deliverable:** Consolidated Technical Report (PDF)  
**Public GitHub Repository:** https://github.com/DEV-2828/ML_HACKATHON  
**Global Random Seed:** `42`

---

## Executive Summary

Urban GPS trajectories suffer from multipath reflections, building canyons, signal dropouts, and irregular sampling, causing raw vehicle fixes to drift onto parallel streets or cut across blocks. **PathSnap** is an end-to-end distributed machine learning system that reconstructs realistic, topologically continuous vehicle routes by pairing a **Bayesian-optimised XGBoost candidate classifier** with a **log-space Hidden Markov Model (HMM) and Viterbi decoder** over each trip's directed road-network graph.

Evaluated strictly on the permitted development partition (`10 student_test trips`, `19,160` GPS observations), our complete pipeline achieves:
- **Candidate Classifier (Bayesian-Tuned XGBoost):** `0.8899 ± 0.0103` 5-fold GroupKFold CV Macro F1, `0.8983` Student-Test Macro F1, `0.0567` Binary Log Loss (`-54.0%` vs untuned), `82.89%` Observation-level Top-1 Accuracy (`83.96%` covered), and `92.06%` classifier-only Route Continuity at `12.31 ms/trip` CPU latency.
- **End-to-End Route Reconstruction (Classifier + HMM/Viterbi):** **`87.09%`** Observation-level Top-1 Accuracy (`+4.21` pts), **`88.23%`** Covered Top-1 Accuracy (`+4.26` pts), **`98.34%`** Route Continuity (`+6.29` pts), **`95.05%`** Sequence-Level Arc Precision (`+11.11` pts), and **`91.88%`** Sequence-Level Arc F1 (`+6.49` pts).
- **Two-Laptop Deployment:** Decoupled FastAPI backend (`backend/api.py`) on Laptop 1 serving live predictions and server-side evaluation to the interactive Streamlit + Leaflet Evaluator Console (`ui/app.py`) on Laptop 2.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

*(All exploration and candidate engineering was performed strictly on the permitted `train` (80 trips) and `student_test` (10 trips) partitions in `data_geo/data_pipeline.py`. The 10 reserved `evaluator_test` trips were never accessed during EDA, feature selection, training, or tuning.)*

### 1. Data Preprocessing Methodology
The raw dataset is organised by trip across three partitions: **80 training trips** (`train/`), **10 student-testing trips** (`student_test/`), and **10 reserved evaluator-testing trips** (`evaluator_test/`). Each trip folder contains four headerless whitespace-separated files:
- `.track`: Raw GPS observations (`longitude`, `latitude`, `timestamp_s`; nominally 1 Hz).
- `.nodes`: Road-network vertices (`longitude`, `latitude`; 0-based row index = `node_id`).
- `.arcs`: Directed road segments (`from_node_id`, `to_node_id`; 0-based row index = `arc_id`).
- `.route`: Chronological sequence of reference `arc_id`s traversed by the vehicle (does not map 1:1 to `.track` rows because multiple 1 s GPS points fall along a single road segment).

**Preprocessing & Candidate Generation Steps (`data_geo/data_pipeline.py`):**
1. **Metric CRS Projection:** Because raw longitude/latitude are angular coordinates, ordinary Euclidean distance in degrees is physically invalid. Each trip is projected to its trip-specific **WGS 84 / UTM projected coordinate system** (EPSG:326xx/327xx via `pyproj`) before computing distances and angles.
2. **Spatial Indexing & Candidate Shortlisting:** A 2D `scipy.spatial.cKDTree` is built over projected road-arc midpoints. For each GPS observation, up to **40 nearest midpoints** are shortlisted, exact orthogonal point-to-segment distances are computed, and arcs within an **80.0 m search radius** are retained (always keeping at least the single closest arc if none lie within 80 m).
3. **Reference Label Alignment & Leakage Rules:** Because `.route` has no explicit GPS-row index, each GPS observation is aligned to the geometrically closest arc among that trip's supplied `.route` arcs in UTM meters. Observations whose reference arc is absent from the 80 m candidate set are flagged `label_available = False` (candidate misses), excluded from supervised classifier fitting, and transparently tracked in `data_geo/outputs/data/metadata.json`.
4. **In-Fold Imputation & Scaling:** Missing kinematic values (first observation of each trip and irregular time steps) are imputed strictly inside each CV training fold via `SimpleImputer(strategy="median")` (and `StandardScaler` for linear/neural models) inside `sklearn.pipeline.Pipeline`.

### 2. Exploratory Data Analysis Findings & Feature Selection

| Dataset Metric | Training Partition (`train`, 80 Trips) | Development Partition (`student_test`, 10 Trips) |
| :--- | ---: | ---: |
| **Trips** | 80 | 10 |
| **Total GPS Observations** | 208,417 | 19,160 |
| **Raw Candidate-Arc Rows** | 2,946,671 | 344,479 |
| **Labeled Candidate Rows (`label_available=True`)** | 2,821,038 | 341,116 |
| **Covered GPS Observations** | 185,351 | 18,914 |
| **Candidate Missed Observations** | 23,066 | 246 |
| **Candidate Miss Rate** | **11.07%** (`0.1107`) | **1.28%** (`0.0128`) |
| **Positive Class Rate (`label = 1`)** | **6.57%** (~1:14 imbalance) | **5.54%** (~1:17 imbalance) |
| **Missing Feature Cells (NaN, imputed in-fold)** | 11,101 (0.033% of cells) | 880 (0.021% of cells) |

**Observed Patterns & Data Limitations:**
- **Severe Class Imbalance:** With up to 40 nearby arcs per GPS fix and only 1 positive reference arc, ~93–94% of candidate rows are negative (`0`). Raw binary accuracy alone is therefore misleading; Macro F1, Observation-level Top-1 Accuracy, and Binary Log Loss are essential.
- **Missing-Value Analysis:** Raw coordinate files contain clean coordinates, while derived kinematic features (`gps_heading_sin_delta`, `gps_heading_cos_delta`, `gps_speed_mps`, `gps_step_distance_m`, `gps_time_delta_s`, `gps_turn_angle_rad`) are naturally `NaN` at trip start (`obs_idx = 0`) and on zero-delta timestamps (`11,101` missing cells on `train`, `880` on `student_test`). Median imputation inside each CV fold handles these cleanly without leakage.
- **Reference Route Limitation:** Supplied `.route` files are treated as reference labels rather than error-free physical ground truth; geometric snapping provides a transparent proxy, and candidate misses (`11.07%` train, `1.28%` `student_test`) are never hidden or relabeled.

**12 Engineered Features (`float32`, zero metadata/ID/CRS leakage):**
1. `distance_to_arc_m` — perpendicular distance from GPS fix to candidate road segment (primary proximity cue).
2. `signed_lateral_m` — signed left/right offset relative to directed arc orientation (separates opposite lanes/dual carriageways).
3. `distance_to_arc_midpoint_m` — Euclidean distance to segment midpoint.
4. `distance_to_nearest_endpoint_m` — distance to nearest road node (captures intersection proximity).
5. `arc_length_m` & `point_to_arc_along_fraction` — segment geometry & orthogonal projection position along the arc.
6. `gps_heading_sin_delta`, `gps_heading_cos_delta` — cyclic sine/cosine encoding of heading difference between vehicle motion vector and directed road bearing (critical for rejecting wrong-way / crossing roads).
7. `gps_speed_mps`, `gps_step_distance_m`, `gps_time_delta_s`, `gps_turn_angle_rad` — local vehicle kinematics capturing stops, turns, and irregular sampling gaps.

![Dataset EDA Distributions (train and student_test partitions)](data_geo/outputs/dataset_eda.png)

![Representative EDA Folium Map — train Trip 00000002 (Raw GPS, Road Network, Reference Route)](data_geo/outputs/folium/trip_00000002_report.png)

---

## 3. Model Experimentation and Comparison

We trained and compared **six distinct models spanning five algorithm families** (`src/modeling/train_baselines.py`, `src/modeling/evaluate.py`, `RANDOM_SEED = 42`):
- **Cross-Validation:** **5-fold `GroupKFold` grouped strictly by `trip_id`** across the 80 training trips (zero trip overlap across folds; preprocessing fit strictly per fold).
- **Development Evaluation:** Identical evaluation on all 10 `student_test` trips (`341,116` labeled candidate rows, `Candidate Miss Rate = 0.01284`).
- **Primary Selection Metric:** **5-fold GroupKFold CV Macro F1-score**, jointly assessed with **Observation-level Top-1 Accuracy** (pointwise road snapping), **Binary Log Loss** (probability calibration for HMM emissions), **Route Continuity**, and **per-trip CPU Inference Latency**.

### 3.1 Overall Model Comparison (`results/metrics/model_comparison.csv`)

| Model | Algorithm Family | Key Configuration | CV Macro F1 (± Std) | Top-1 Acc (All / Cov) | Route Cont. | Test Macro F1 | Log Loss | Inference Time (Per Trip) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **MLP** | Neural Network | `layers=(64,32), relu, adam, alpha=1e-3, lr=1e-3, batch=2048, 15 ep, Scaled` | **0.8857 ± 0.0093** | 0.8223 / 0.8330 | **0.9196** | **0.8966** | **0.0556** | 0.1024 s (10.24 ms) |
| **RandomForest** | Ensemble (Bagging) | `n_est=100, depth=16, min_leaf=10, max_samp=0.4, balanced_sub` | 0.8471 ± 0.0122 | 0.8225 / 0.8332 | 0.9122 | 0.8534 | 0.1026 | 0.2246 s (22.46 ms) |
| **XGBoost** | Boosting | `n_est=200, depth=6, lr=0.1, sub=0.8, col=0.8, hist, spw≈14.2` | 0.8270 ± 0.0126 | **0.8260 / 0.8368** | 0.9171 | 0.8327 | 0.1232 | 0.0581 s (5.81 ms) |
| **DecisionTree** | Tree-based | `gini, depth=12, min_leaf=20, balanced` | 0.8184 ± 0.0120 | 0.8032 / 0.8137 | 0.8951 | 0.8247 | 0.1444 | **0.0352 s (3.52 ms)** |
| **ExtraTrees** | Ensemble (Rand.) | `n_est=100, depth=18, min_leaf=10, boot=True, max_samp=0.4` | 0.8142 ± 0.0121 | 0.8148 / 0.8254 | 0.9097 | 0.8170 | 0.1605 | 0.2280 s (22.80 ms) |
| **LogisticReg** | Linear / Prob. | `C=1.0, lbfgs, max_iter=1000, balanced, Scaled` | 0.7161 ± 0.0146 | 0.7450 / 0.7547 | 0.8684 | 0.7104 | 0.2225 | 0.0366 s (3.66 ms) |

### 3.2 Compulsory Analysis for EVERY Model

#### 1. Logistic Regression (`Linear / Probabilistic`)
- **A. Model Understanding:** Selected as a convex linear probabilistic baseline outputting calibrated probabilities directly. Fast to train (`4.26 s`) and infer (`3.66 ms/trip`), but fundamentally limited because GPS map-matching requires nonlinear feature interactions (e.g., short perpendicular distance combined with aligned heading cosine).
- **B. Training & Configuration:** `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `class_weight="balanced"`, `random_state=42` inside `SimpleImputer(median) -> StandardScaler`. Objective: L2-regularised Binary Cross-Entropy (Log Loss).
- **C. Experimental Results:** Student-test Accuracy `0.8855`, Precision `0.3232`, Recall `0.9731`, Top-1 Accuracy `0.7450` (`0.7547` covered), Macro F1 `0.7104`, Log Loss `0.2225`, Route Continuity `0.8684`, CM `[[283660, 38542], [509, 18405]]`. **Clear underfitting:** Train Macro F1 (`0.7178`) ≈ CV Macro F1 (`0.7161 ± 0.0146`, gap `+0.0017`), trailing nonlinear models by 11–18 Macro F1 points with `38,542` false positives.

#### 2. Decision Tree (`Tree-based`)
- **A. Model Understanding:** Learns nonlinear threshold rules over distance, lateral offset, and heading alignment without feature scaling. Delivers a `+5.8` pt Top-1 jump over Logistic Regression (`0.8032`) at `3.52 ms/trip`, but single-tree variance produces coarser leaf probability estimates (`Log Loss = 0.1444`).
- **B. Training & Configuration:** `criterion="gini"`, `max_depth=12`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42`. Objective: Weighted Gini Impurity.
- **C. Experimental Results:** Student-test Accuracy `0.9496`, Precision `0.5252`, Recall `0.9514`, Top-1 `0.8032` (`0.8137` covered), Macro F1 `0.8247`, Log Loss `0.1444`, Continuity `0.8951`, CM `[[305934, 16268], [919, 17995]]`. **Mild overfitting (largest Train–CV gap):** Train F1 (`0.8334`) vs CV F1 (`0.8184 ± 0.0120`, gap `+0.0150`) and highest fold Log-Loss std (`±0.0398`).

#### 3. Random Forest (`Ensemble — Bagging`)
- **A. Model Understanding:** Averages 100 depth-constrained trees on bootstrap subsamples to smooth probabilities and lower variance on noisy GPS points. Cuts false positives to `12,072` and Log Loss to `0.1026`, but is the slowest model to fit (`67.91 s`) and slower to infer (`22.46 ms/trip`) than boosting.
- **B. Training & Configuration:** `n_estimators=100`, `max_depth=16`, `min_samples_leaf=10`, `max_samples=0.4`, `class_weight="balanced_subsample"`, `random_state=42`. Objective: Bagged Gini Impurity.
- **C. Experimental Results:** Student-test Accuracy `0.9611`, Precision `0.5948`, Recall `0.9369`, Top-1 `0.8225` (`0.8332` covered), Macro F1 `0.8534`, Log Loss `0.1026`, Continuity `0.9122`, CM `[[310130, 12072], [1193, 17721]]`. **Well-controlled mild variance:** Train F1 (`0.8602`) vs CV F1 (`0.8471 ± 0.0122`, gap `+0.0131`) generalizes cleanly to `0.8534` on `student_test`.

#### 4. Extra Trees (`Ensemble — Randomized Trees`)
- **A. Model Understanding:** Tests fully randomized split thresholds to further lower variance and speed up bagging (`46.01 s` vs RF `67.91 s`). Random thresholds soften sharp geometric cutoffs near parallel roads, raising Log Loss (`0.1605`) vs exact-split RF.
- **B. Training & Configuration:** `n_estimators=100`, `max_depth=18`, `min_samples_leaf=10`, `bootstrap=True`, `max_samples=0.4`, `class_weight="balanced_subsample"`, `random_state=42`. Objective: Randomized-threshold Gini Impurity.
- **C. Experimental Results:** Student-test Accuracy `0.9462`, Precision `0.5080`, Recall `0.9553`, Top-1 `0.8148` (`0.8254` covered), Macro F1 `0.8170`, Log Loss `0.1605`, Continuity `0.9097`, CM `[[304706, 17496], [846, 18068]]`. **Slightly higher bias:** Train F1 (`0.8174`) closely matches CV (`0.8142 ± 0.0121`, gap `+0.0032`) and Test (`0.8170`).

#### 5. XGBoost (`Boosting`)
- **A. Model Understanding:** Histogram-binned gradient boosting (`tree_method="hist"`) sequentially corrects residual map-matching errors under direct binary log-loss optimisation. Achieved the **highest untuned Top-1 Accuracy (`0.8260` overall / `0.8368` covered)**, fastest nonlinear fit (`13.72 s`), and ultra-fast inference (`5.81 ms/trip`). Its only untuned limitation was setting full `scale_pos_weight ≈ 14.2`, which shifted probabilities upward (Precision `0.5412`, Log Loss `0.1232`).
- **B. Training & Configuration:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, per-fold `scale_pos_weight ≈ 14.2`, `tree_method="hist"`, `random_state=42`. Objective: Binary Logistic Loss (`binary:logistic`).
- **C. Experimental Results:** Student-test Accuracy `0.9526`, Precision `0.5412`, Recall `0.9553`, Top-1 `0.8260` (`0.8368` covered), Macro F1 `0.8327`, Log Loss `0.1232`, Continuity `0.9171`, CM `[[306884, 15318], [845, 18069]]`. **Well-regularised with clear headroom:** Train F1 (`0.8382`) vs CV F1 (`0.8270 ± 0.0126`, gap `+0.0112`); real boosting-round validation log loss (`history_XGBoost.csv`) dropped monotonically from `0.6183` (Round 1) to `0.1495` (Round 200) without plateauing.

#### 6. Multi-Layer Perceptron — MLP (`Neural Network`)
- **A. Model Understanding:** Smooth nonlinear neural representation over standardized geometric/kinematic features trained with unweighted cross-entropy, producing well-calibrated posterior probabilities. Strongest untuned baseline on Macro F1 (`0.8966`), Log Loss (`0.0556`), and Continuity (`0.9196`), but slower to train (`50.72 s`) and slightly behind XGBoost on Top-1 candidate ranking (`0.8223` vs `0.8260`).
- **B. Training & Configuration:** `hidden_layer_sizes=(64, 32)`, `activation="relu"`, `solver="adam"`, `alpha=1e-3`, `learning_rate_init=1e-3`, `batch_size=2048`, `max_iter=15`, `random_state=42` inside `SimpleImputer(median) -> StandardScaler`. Objective: L2-regularised Binary Cross-Entropy.
- **C. Experimental Results:** Student-test Accuracy `0.9787`, Precision `0.8202`, Recall `0.7895`, Top-1 `0.8223` (`0.8330` covered), Macro F1 `0.8966`, Log Loss `0.0556`, Continuity `0.9196`, CM `[[318928, 3274], [3982, 14932]]`. **Well-fit & stable:** Train F1 (`0.8902`) vs CV F1 (`0.8857 ± 0.0093`, gap `+0.0045`); real epoch-wise validation log loss (`history_MLP.csv`) converged smoothly from `0.0850` (Epoch 1) to `0.0742` (Epoch 10) and plateaued at `0.0745` (Epoch 15).

**Selection Justification:** Untuned **XGBoost** led on Top-1 ranking (`0.8260`), was `3.7×` faster to train than MLP, `~2×` faster to infer, and had an un-plateaued validation curve — with its threshold-0.5 Macro F1 / Log Loss held back purely by untuned `scale_pos_weight ≈ 14.2`. Selecting **XGBoost** for Bayesian Hyperparameter Optimisation unlocked superior performance across every metric.

### 3.3 Model Visualisations (Generated from Actual Experimental Logs)

![5-Fold GroupKFold CV Macro F1-score Comparison Across Models](PLOT/cv_macro_f1_comparison.png)

![Student-Test Macro F1-score and Observation-Level Top-1 Accuracy Comparison](PLOT/student_test_top1_and_f1.png)

![Student-Test Binary Log Loss Comparison (Probability Quality for HMM Emissions)](PLOT/log_loss_comparison.png)

![Genuine Iterative Training and Validation Loss Curves (XGBoost and MLP Only)](PLOT/learning_curves_iterative.png)

![Student-Test Confusion Matrices Across All 6 Baseline Models](PLOT/cm_all_models_grid.png)

---

## 4. Bayesian Hyperparameter Optimisation (XGBoost)

Implemented in `src/modeling/tune_bayesian.py` using **Optuna `TPESampler(seed=42)`** strictly over **5-fold `GroupKFold` (`trip_id`) on the 80 training trips** with per-fold median imputation. Neither `student_test` nor `evaluator_test` was accessed during optimisation.

1. **Hyperparameters searched (9):** `scale_pos_weight`, `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
2. **Search ranges & distributions:**
   - `scale_pos_weight`: Uniform `[1.0, 6.0]` (calibrates down from untuned `~14.2`)
   - `max_depth`: Integer uniform `[4, 10]`
   - `learning_rate`: Log-uniform `[0.02, 0.25]`
   - `n_estimators`: Integer uniform `[150, 500]`
   - `min_child_weight`: Integer uniform `[1, 20]`
   - `subsample`: Uniform `[0.6, 1.0]`
   - `colsample_bytree`: Uniform `[0.6, 1.0]`
   - `reg_alpha`: Log-uniform `[1e-4, 10.0]`
   - `reg_lambda`: Log-uniform `[1e-3, 10.0]`
3. **Objective function:** Maximize mean 5-fold `GroupKFold` CV Macro F1-score on the 80 training trips.
4. **Optimisation trials & computational cost:** **30 Bayesian trials** (150 fold fits on `2.82M` rows), completed in **`521.17 s` (`8.69 min`)** on CUDA GPU (`tree_method="hist"`). Final refit on all 80 trips took `6.27 s`, exported to CPU-portable `artifacts/final_model.joblib`.
5. **Best configuration (Trial #23, CV Macro F1 = `0.88993`):** `scale_pos_weight=1.5660`, `max_depth=9`, `learning_rate=0.0539`, `n_estimators=399`, `min_child_weight=13`, `subsample=0.8483`, `colsample_bytree=0.7677`, `reg_alpha=0.000128`, `reg_lambda=1.1579`.
6. **Did tuning improve performance?** **Yes — decisively.** Tuned XGBoost outperformed untuned XGBoost on every quality metric and beat **all 6 untuned baselines** (including MLP) on CV Macro F1 (`0.8899`), Student-Test Macro F1 (`0.8983`), Top-1 Accuracy (`0.8289`), and Continuity (`0.9206`), slashing XGBoost Log Loss by **54.0%** (`0.1232 -> 0.0567`) and false positives by **69.5%** (`15,318 -> 4,670`, CM `[[317532, 4670], [2912, 16002]]`).

### Untuned vs Bayesian-Tuned XGBoost Comparison (`results/metrics/untuned_vs_tuned.csv`)

| Metric | XGBoost (Untuned Baseline) | XGBoost (Bayesian Tuned) | Delta (Tuned − Untuned) |
| :--- | ---: | ---: | ---: |
| **CV Macro F1-score (5-fold GroupKFold)** | 0.82701 ± 0.01262 | **0.88993 ± 0.01025** | **+0.06292** |
| **Student-Test Top-1 Accuracy (Overall)** | 0.82604 | **0.82886** | **+0.00282** |
| **Student-Test Top-1 Accuracy (Covered)** | 0.83679 | **0.83964** | **+0.00285** |
| **Student-Test Macro F1-score** | 0.83265 | **0.89833** | **+0.06568** |
| **Precision / Recall** | 0.54120 / 0.95532 | **0.77409** / 0.84604 | +0.23289 / -0.10928 |
| **Student-Test Log Loss** | 0.12320 | **0.05670** | **-0.06650 (-54.0%)** |
| **Route Continuity (Classifier-Only)** | 0.91705 | **0.92055** | **+0.00350** |
| **CPU Inference Time (10 Trips / Per Trip)** | 0.0581 s (5.81 ms) | 0.1231 s (12.31 ms) | +0.0650 s (+6.50 ms) |

![Optuna Bayesian Optimisation History and Parameter Sensitivity](PLOT/optuna_bayesian_history.png)

![Untuned vs Bayesian-Tuned XGBoost Comparison](PLOT/untuned_vs_tuned_xgboost.png)

![Bayesian-Tuned XGBoost Student-Test Confusion Matrix](PLOT/cm_XGBoost_tuned.png)

---

## 5. HMM and Viterbi Route Reconstruction

Pointwise classifier predictions leave `~7.9%` discontinuous transitions (`Route Continuity = 92.06%`) because each GPS observation is scored independently. We implemented a log-space **Hidden Markov Model + Viterbi decoder** (`src/hmm_decoder.py`, evaluated in `src/hmm/evaluate_hmm.py`, config saved in `artifacts/hmm_config.json`) directly on top of our tuned XGBoost probabilities.

1. **Hidden states:** Candidate road segments (`arc_id`) shortlisted at each GPS observation `t` (up to 40 per observation), preserving exact `(arc_id, from_node, to_node)` graph correspondence.
2. **Emission log-probabilities:** Derived directly from our Bayesian-tuned XGBoost calibrated candidate probability `p(t, i)`: `log E(t, i) = w_clf * log(clip(p(t, i), 1e-6, 1 - 1e-6))` with tuned `w_clf = 0.7`.
3. **Transition log-probabilities & Viterbi recurrence:** Over each trip's directed metric-weighted road graph (`RoadNetworkGraph`):
   - **Same-arc stay (`a_i = a_j`):** `self_transition_bonus = +1.5`.
   - **Direct directed edge (`to_node(a_i) = from_node(a_j)`):** `direct_edge_bonus (+0.8) - 0.25 * d_gps / beta`.
   - **Multi-hop short path:** Bounded local Dijkstra from `to_node(a_i)` to `from_node(a_j)` (capped at `min(600 m, max(120 m, 2.5 * d_gps + 30 m))` and `max_visited = 500`), scored via Newson–Krumm exponential detour penalty `-|d_road - d_gps| / beta` with `beta = 20.0 m`.
   - **Viterbi log-space recurrence:** `V[t, j] = log E(t, j) + max_i (V[t-1, i] + log T(i -> j))` with per-step max-subtraction for numerical stability and backpointer backtracking.
4. **Missing candidates & large GPS gaps:** Observations with `Δt > gap_threshold_s (5.0 s)` or invalid timestamps trigger a **soft transition reset** (`log T = 0`), re-anchoring purely on classifier emissions after dropouts; observations without valid coordinates are skipped cleanly while preserving `obs_idx` alignment.
5. **Disconnected segments / invalid transitions:** Unreachable transitions beyond the bounded Dijkstra cutoff receive a finite `disconnect_penalty = -12.0` (never `-inf`), preventing Viterbi deadlocks while strongly suppressing spurious parallel-street jumps.
6. **Did HMM decoding improve the final route?** **Yes — across every metric and all 10 `student_test` trips.**

### Classifier-Only vs Classifier + HMM/Viterbi (`results/metrics/classifier_vs_hmm.csv`)

| Pipeline | Top-1 Accuracy (Overall) | Top-1 Accuracy (Covered) | Route Continuity | Sequence Arc Precision | Sequence Arc Recall | Sequence Arc F1 | Decode Time / Trip |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Classifier-Only (Bayesian-Tuned XGBoost)** | 0.82886 (82.89%) | 0.83964 (83.96%) | 0.92055 (92.06%) | 0.83940 | 0.87707 | 0.85390 | 0.00 ms |
| **Classifier + HMM/Viterbi** | **0.87093 (87.09%)** | **0.88226 (88.23%)** | **0.98344 (98.34%)** | **0.95049** | **0.89307** | **0.91878** | 564.34 ms |
| **Gain (HMM − Classifier-Only)** | **+0.04207 (+4.21 pts)** | **+0.04261 (+4.26 pts)** | **+0.06290 (+6.29 pts)** | **+0.11109 (+11.11 pts)** | **+0.01601 (+1.60 pts)** | **+0.06488 (+6.49 pts)** | +564.34 ms |

**Improvements & Limitations (Per-Trip Analysis from `results/metrics/hmm_per_trip_breakdown.csv`):**
- **Largest gains:** Dense, noisy trips showed the sharpest gains — Trip `00000077` jumped `+8.36%` Top-1 (`71.24% -> 79.60%`), `+9.37%` Continuity (`90.09% -> 99.46%`), and `+11.16%` Sequence Arc F1 (`79.87% -> 91.04%`); Trip `00000007` achieved `99.41%` Top-1 and `100.00%` Continuity.
- **Where HMM could not fully recover:** (a) Observations where the true arc was absent from the 80 m candidate set (`1.28%` candidate miss rate), (b) closely parallel dual carriageways / service roads in dense urban corridors (`00000056`, `00000065`) where the geometric reference proxy shifted 1–2 s earlier than the HMM's stay-bonus preferred, and (c) bounded Dijkstra runtime (`~0.56 s/trip`, up to `1.4 s` on dense graphs), well within interactive API latency.

![Classifier-Only vs Classifier + HMM/Viterbi Comparison](PLOT/classifier_vs_hmm.png)

---

## 6. Geographical Visualisation Using Folium

Interactive multi-layer Folium maps were generated directly from our real production pipeline (`scripts/generate_folium_maps.py`, using `artifacts/final_model.joblib` + `artifacts/hmm_config.json`) and committed to GitHub:
- `results/plots/folium_map_trip_00000000.html` (`student_test` Trip `00000000`: Clf Top-1 `92.89% ->` HMM Top-1 `94.49%`, Continuity `95.32% -> 99.44%`)
- `results/plots/folium_map_trip_00000007.html` (`student_test` Trip `00000007`: Clf Top-1 `97.93% ->` HMM Top-1 `99.41%`, Continuity `99.78% -> 100.00%`)
- `data_geo/outputs/folium/trip_00000002.html` (`train` Trip `00000002` EDA map)

Each interactive HTML map includes **five distinguishable colour-coded layers**, a fixed on-map **legend**, trip-fitted geographical extent, and a top-right **`LayerControl`** widget:
1. **Nearby Road Network Geometry** (slate gray polylines)
2. **Original Noisy GPS Observations** (blue circular markers with tooltips)
3. **Reference Route (`.route`)** (amber solid line)
4. **Classifier-Only Route (XGBoost)** (red dashed line)
5. **Predicted Route (XGBoost + HMM/Viterbi)** (solid emerald green line with Start/End markers)

![Geographical Route Comparison — student_test Trip 00000000 (GPS, Road Network, Reference, Classifier-Only, HMM Predicted)](PLOT/route_map_trip_00000000.png)

![Geographical Route Comparison — student_test Trip 00000007 (GPS, Road Network, Reference, Classifier-Only, HMM Predicted)](PLOT/route_map_trip_00000007.png)

---

## 7. API, UI and Deployment

### Two-Laptop Distributed Architecture
- **Laptop 1 — Production FastAPI Server (`backend/api.py`):** Loads `artifacts/final_model.joblib`, `artifacts/model_metadata.json`, and `artifacts/hmm_config.json` once on startup. Exposes `GET /health`, `GET /trips`, and `POST /predict` (alias `/evaluate`). On each request, `src/inference_pipeline.py` extracts candidate segments & 12 UTM features from `.track`/`.nodes`/`.arcs` ONLY, scores them with Bayesian-tuned XGBoost, decodes via `HMMRouteDecoder`, computes accuracy strictly server-side using internal `.route` (never exposed to the model or UI), and prints live timestamped telemetry to CMD.  
  **Launch Command:** `.\.venv\Scripts\python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000`
- **Laptop 2 — Evaluator UI (`ui/app.py`):** Streamlit + Folium dashboard connecting over LAN/hotspot (`http://<LAPTOP_1_IP>:8000`). Lists all 10 reserved evaluation trips dynamically from `/trips`, dispatches the evaluator's chosen `trip_id` to Laptop 1, and renders live KPI cards, Classifier vs HMM comparison table, system details, and interactive multi-layer Leaflet map.  
  **Launch Command:** `.\.venv\Scripts\python -m streamlit run ui/app.py`
- **Automated Verification (`backend/test_api.py`):** Verified 10/10 trips passing live HTTP end-to-end (`108 ms – 2,095 ms` per trip).

![Working Evaluator UI (Streamlit + Interactive Folium Route Reconstruction Map)](PLOT/ui_demo_screenshot.jpg)

![Laptop 1 Production FastAPI Server CMD Log (Uvicorn on 0.0.0.0:8000, Model & 10 Evaluation Trips Loaded)](PLOT/cmd_api_screenshot.png)
"""


def _prepare_compact_images(md_text: str, tmp_dir: Path, max_width: int = 1400) -> str:
    """Convert large 300-DPI PNGs to crisp, compact JPEGs in tmp_dir so PDF stays <5 MB."""
    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def _replace(match: re.Match[str]) -> str:
        alt, rel_path = match.group(1), match.group(2)
        src = BASE_DIR / rel_path
        if not src.exists():
            return match.group(0)
        dst = tmp_dir / (src.stem + ".jpg")
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            if rgb.width > max_width:
                new_h = int(rgb.height * (max_width / rgb.width))
                rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
            rgb.save(dst, format="JPEG", quality=86, optimize=True)
        return f"![{alt}]({dst.name})"

    return img_pattern.sub(_replace, md_text)


SIMPLE_CSS = """
body {
    font-family: Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.48;
    color: #111111;
}
h1 { font-size: 15pt; margin-bottom: 6px; color: #111111; }
h2 { font-size: 12.5pt; margin-top: 16px; margin-bottom: 4px; border-bottom: 1px solid #999999; padding-bottom: 2px; color: #111111; }
h3 { font-size: 10.5pt; margin-top: 12px; margin-bottom: 3px; color: #111111; }
h4 { font-size: 9.5pt; margin-top: 8px; margin-bottom: 2px; color: #111111; }
table { border-collapse: collapse; width: 100%; font-size: 7.5pt; margin: 8px 0 12px 0; }
th { background-color: #e8e8e8; color: #111111; font-weight: bold; padding: 4px 4px; border: 1px solid #aaaaaa; text-align: left; }
td { padding: 3px 4px; border: 1px solid #cccccc; vertical-align: top; color: #111111; }
tr:nth-child(even) td { background-color: #f7f7f7; }
code { font-family: "Courier New", monospace; font-size: 8pt; background-color: #f0f0f0; color: #111111; padding: 1px 3px; }
pre { background-color: #f4f4f4; color: #111111; padding: 8px 10px; font-size: 8pt; border-left: 3px solid #888888; }
pre code { background-color: #f4f4f4; color: #111111; padding: 0; }
hr { border-top: 1px solid #cccccc; margin: 12px 0; }
img { display: block; max-width: 100%; margin: 8px auto; }
ul, ol { margin: 4px 0 6px 0; padding-left: 18px; }
li { margin-bottom: 2px; }
"""


def generate_report() -> None:
    out_dir = BASE_DIR / "DOCs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_file = out_dir / "TEAM-3_ML_HACKATHON.pdf"

    with tempfile.TemporaryDirectory(prefix="pathsnap_pdf_") as tmp:
        tmp_dir = Path(tmp)
        compact_md = _prepare_compact_images(REPORT_MD, tmp_dir, max_width=1400)
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(
            Section(compact_md, toc=True, paper_size="A4", root=str(tmp_dir)),
            user_css=SIMPLE_CSS,
        )
        pdf.save(str(pdf_file))

    print(f"Compiled PDF -> {pdf_file} ({pdf_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_report()
