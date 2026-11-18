# PathSnap data and geo pipeline

This folder contains the dataset inspection, preprocessing, candidate generation, feature extraction, EDA artifacts, and report material for PathSnap.

## Run

From the project root, install the listed dependencies and run:

```powershell
python -m pip install -r data_geo/requirements.txt
python data_geo/data_pipeline.py --data-root "C:\path\to\Student folder\Student folder" --output-dir data_geo/outputs --map-trip-id 00000002
```

The data root must contain `train/` and `student_test/`. The script does not modify input trips. It writes inventory and quality CSVs, per-trip feature caches under `outputs/candidate_features/`, the agreed model handoff tables `outputs/data/train_candidates.parquet` and `outputs/data/student_test_candidates.parquet`, an EDA plot, a candidate summary, an EDA JSON, and a multi-layer Folium map. Per-trip output keeps memory bounded.

## Candidate-feature contract for Member 2

The model handoff Parquet has one row per observation/candidate-arc pair and includes `trip_id`, `obs_idx`, `arc_id`, `from_node`, `to_node`, `label`, the engineered feature columns, and label/CRS metadata. Use numeric columns in `FEATURE_COLUMNS` from `data_pipeline.py` as model inputs. IDs, endpoints, labels, route references, and CRS are metadata/targets and must not be fed as predictors. Split validation by `trip_id`; do not split candidate rows randomly across train and validation. See `outputs/data/metadata.json` for GPS observation totals and candidate-miss counts by split.

Rows with `label_available == false` are candidate misses and have no positive candidate. Exclude those observations from candidate-classifier fitting/scoring, but report their rate separately. The target is a transparent geometric proxy: for each GPS point, choose the closest segment from that trip's supplied `.route` arc sequence using projected point-to-segment distance. The route references may be imperfect; this target is not claimed to be independently verified ground truth.

Candidate generation uses a per-trip UTM CRS, a 40-arc nearest-midpoint shortlist, exact point-to-segment distance, and an 80 m default search radius. When no shortlisted arc falls within the radius, it retains the closest shortlisted arc so an observation always has a candidate. These parameters are configurable on the command line and should be tuned using permitted training/development data only.

## Current artifact state

The EDA and data artifacts are generated from the supplied `train` and `student_test` partitions. The map shows raw GPS, nearby road arcs, and the supplied reference route. A predicted-route layer can only be added after Member 2/3 exports predictions; no predicted route is fabricated here. Inference integration likewise depends on the API/classifier contract from the other team members.
