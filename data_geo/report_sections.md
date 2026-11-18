# Draft report: dataset exploration and geographic visualization

These notes summarize the actual data pipeline run; add the team's later model predictions and interpretation before submission. The submitted report must reflect the team's own experiments and interpretation.

## 2.2 Dataset understanding and exploratory analysis

The supplied data is organized by trip: 80 `train` trips and 10 `student_test` trips were found. `.track` contains longitude, latitude, and timestamp in seconds; `.nodes` contains road-node coordinates; `.arcs` contains directed from/to node row IDs, with each arc row index serving as its arc ID; `.route` contains the supplied reference arc sequence. Across these partitions, the pipeline read 227,577 GPS observations. Coordinates are transformed from longitude/latitude to a trip-specific WGS 84 / UTM CRS before distances and road geometry features are computed.

EDA outputs: `outputs/trip_inventory.csv` records file presence and byte sizes; `outputs/trip_quality.csv` records row counts, missing GPS rows, duplicate coordinates, timestamp order/gaps, path length, graph endpoint validity, isolated nodes, and weakly connected components; `outputs/dataset_eda.png` plots trip-level distributions; `outputs/folium/` contains an interactive GPS/road/reference map. State the exact partition for each table/map. Training and student-test are used only as permitted for development. Do not inspect or tune from evaluator-test labels.

Candidate features describe the relationship of a GPS observation to a nearby road segment: projected point-to-segment distance, signed lateral offset, along-segment fraction, arc length, heading difference, nearest endpoint distance, and local GPS speed, step, time gap, and turning angle. UTM coordinates are used because raw angular degrees are not suitable for metre-scale distances. The current configuration uses an 80 m search radius and a shortlist of up to 40 nearest arc midpoints, followed by exact point-to-segment distances.

For a usable supervised target, each observation is associated with the closest segment among the trip's supplied route arcs. This is a geometric association because `.route` is a chronological arc sequence without an explicit observation-to-arc index mapping. Rows where that reference arc is absent from the generated candidates are marked as candidate misses; the model handoff includes `label_available` so Member 2 can exclude these observations from supervised fit/scoring while reporting them separately. The observed candidate miss rates were 11.07% on training (23,066 of 208,417 GPS observations) and 1.28% on student-test (246 of 19,160). Investigate the large train/dev difference and trips with very low route coverage; do not hide or relabel these misses.

The model handoff files are `outputs/data/train_candidates.parquet` and `outputs/data/student_test_candidates.parquet`. Each row contains `trip_id`, `obs_idx`, `arc_id`, `from_node`, `to_node`, binary `label`, `label_available`, route/CRS metadata, and the numeric feature columns. `outputs/data/metadata.json` records observation and miss counts for each split.

## 6. Geographical visualization

The generated Folium map overlays the observed GPS path, nearby road network, and supplied reference route as separate toggleable layers, with distinct colors and a trip-focused extent. `outputs/folium/trip_00000002_report.png` is a static map export for the PDF; `trip_00000002.html` is interactive. After Member 2/3 provide predictions, add predicted-route geometry as a separate layer and compare it with GPS and reference route. The current map contains no predicted route because no model prediction artifact has yet been supplied.

## Inference handoff / verification

The model handoff is `outputs/data/train_candidates.parquet` and `outputs/data/student_test_candidates.parquet`, with `data_pipeline.FEATURE_COLUMNS` defining the model inputs. End-to-end inference should be checked after the classifier and API are integrated: verify the API uses the same CRS, candidate generation, and feature order, and run all available development trips without exposing reference labels to the prediction function or UI. Record the API/inference findings here after that integration; they cannot be verified from the data-pipeline components alone.
