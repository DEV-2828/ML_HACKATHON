# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — Department of Computer Science and Engineering**  
**Course:** UE24CS352A Machine Learning | **Project:** PathSnap Hackathon

---

## 📌 Executive Summary
GPS tracking data in urban environments suffers from multipath reflections, signal interference, and low sampling frequencies. **PathSnap** is a distributed machine learning system designed to reconstruct accurate, continuous vehicle trajectories by matching noisy GPS observations to a directed road network graph.

---

## 🏛️ System Architecture: Two-Laptop Distributed Deployment

The solution follows a strict two-laptop distributed model communicating over a local area network (LAN/Wi-Fi Hotspot):

```
                   [ LAPTOP 1: MODEL & API BACKEND ]
             ┌───────────────────────────────────────────────┐
             │ • Raw Road Network Graph (.nodes, .arcs)      │
             │ • Preprocessing & Candidate Projection        │
             │ • Trained Classifier (Random Forest / GBDT)   │
             │ • Hidden Markov Model & Viterbi Decoder       │
             │ • Server-Side Ground Truth Evaluation Engine  │
             │ • FastAPI REST Service (:8000)                │
             └───────────────────────▲───────────────────────┘
                                     │
                             HTTP / JSON via
                             Local Network (Wi-Fi)
                                     │
             ┌───────────────────────▼───────────────────────┐
             │ • Streamlit / Leaflet Evaluator Console       │
             │ • Dynamic 10-Trip Selector                    │
             │ • Real-time Interactive Folium Route Map      │
             │ • Evaluation Metrics Scorecard (Top-1, etc.)  │
             │ • Classifier vs HMM Ablation Comparison       │
             └───────────────────────────────────────────────┘
                   [ LAPTOP 2: EVALUATOR CONSOLE (UI) ]
```

- **Laptop 1 (Backend Host):** Runs the FastAPI inference server, loads model weights, reconstructs routes via Viterbi decoding, evaluates metrics against server-side ground truth, and outputs real-time inference logs to the terminal.
- **Laptop 2 (Evaluator Client):** Hosts the interactive Evaluator UI. The evaluator selects any of the 10 evaluation trips, triggers inference, and views live KPI metrics and multi-layer Leaflet route maps.

---

## 👥 Team Roles & Responsibilities

| Role | Member Focus | Key Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | Data Preprocessing & EDA | Data parsing, GPS projection, candidate arc generation, feature engineering |
| **Member 2** | Model Experimentation & Opt | Training 5 model families, cross-validation, Bayesian hyperparameter tuning |
| **Member 3** | HMM, Viterbi & FastAPI Backend | Sequential route decoding, transition/emission matrices, FastAPI service |
| **Member 4** | UI, Integration, GitHub & Report | Evaluator UI (Streamlit/Leaflet), 2-laptop networking, repo hygiene, report lead |

---

## 🔌 Two-Laptop Network Setup

### 1. Recommended Network Setup
- Connect both Laptop 1 and Laptop 2 to a **Mobile Hotspot** to avoid campus Wi-Fi AP isolation.
- On **Laptop 1**, find its IP address via PowerShell:
  ```powershell
  ipconfig
  # Look for IPv4 Address, e.g., 192.168.43.120
  ```

### 2. Allow Port 8000 Through Windows Firewall on Laptop 1
Run in PowerShell as Administrator on **Laptop 1**:
```powershell
New-NetFirewallRule -DisplayName "PathSnap FastAPI" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### 3. Verify Connection from Laptop 2
Run in PowerShell on **Laptop 2**:
```powershell
Test-NetConnection -ComputerName <LAPTOP_1_IP> -Port 8000
# Example: Test-NetConnection -ComputerName 192.168.43.120 -Port 8000
```

---

## 🚀 Installation & Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-team>/PathSnap.git
cd PathSnap

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend on Laptop 1
```powershell
# Binds to 0.0.0.0 so external laptops can connect
.\.venv\Scripts\python -m uvicorn backend.mock_api:app --host 0.0.0.0 --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 3. Launch the Evaluator UI on Laptop 2
You have two deployment options:

#### Option A: Streamlit Interactive Dashboard
```powershell
.\.venv\Scripts\python -m streamlit run ui/app.py
```
- In the sidebar, enter Laptop 1's IP address (e.g., `http://192.168.43.120:8000`).
- Select any of the 10 evaluation trips and click **"Reconstruct & Evaluate Route"**.

#### Option B: Standalone Web Dashboard (Zero Dependency)
- Open `ui/index.html` directly in Google Chrome or Microsoft Edge.
- Connects instantly to Laptop 1 and visualizes Leaflet map layers.

---

## 📡 API Contract & JSON Schema

### 1. Trip Evaluation Request
`POST /predict` or `POST /evaluate`
```json
{
  "trip_id": "00000000"
}
```

### 2. Trip Evaluation Response
```json
{
  "trip_id": "00000000",
  "status": "success",
  "metrics": {
    "top_1_accuracy": 0.9269,
    "route_continuity": 0.9624,
    "candidate_miss_rate": 0.0186,
    "inference_time_ms": 121.79,
    "total_observations": 2503,
    "classifier_top_1_accuracy": 0.7439,
    "classifier_continuity": 0.5822
  },
  "raw_gps": [[56.251559, 54.963519], [56.251535, 54.962778]],
  "classifier_only_route": [[56.25155, 54.96350], [56.25180, 54.96410]],
  "hmm_reconstructed_route": [[56.251559, 54.963519], [56.251535, 54.962778]],
  "candidate_segments": [
    [[56.251293, 54.968545], [56.251535, 54.962778]]
  ],
  "details": {
    "nodes_count": 8542,
    "arcs_count": 18196,
    "decoder": "Viterbi (Gaussian Distance Emission & Topological Transition)"
  }
}
```

---

## 📂 Project Repository Structure

```text
PathSnap/
├── README.md                      # Complete system documentation & run instructions
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git hygiene: excludes large datasets, pycache, envs
├── backend/
│   ├── schemas.py                 # Pydantic data contract models
│   ├── mock_api.py                # Working mock API server for Phase 1 / evaluation
│   └── main.py                    # Final production FastAPI backend (M3)
├── ui/
│   ├── app.py                     # Streamlit Evaluator Dashboard
│   └── index.html                 # Standalone Leaflet web console
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data parsing & geospatial calculations (M1)
│   ├── models.py                  # 5 ML model implementations (M2)
│   └── hmm_decoder.py             # HMM transition/emission & Viterbi logic (M3)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_model_experimentation.ipynb
├── artifacts/                     # Serialized model weights & scalers (.pkl)
├── results/                       # Evaluation tables, confusion matrices, logs
└── docs/                          # Technical report draft & architecture diagrams
```

---

## 📈 Evaluation Metrics & Scoring Criteria

1. **Observation-Level Top-1 Accuracy:** Proportion of individual GPS points snapped to the true road arc.
2. **Route Continuity:** Proportion of consecutive road segments in the reconstructed path that share a valid directed topological connection in the `.arcs` road graph.
3. **Candidate Miss Rate:** Percentage of true road segments not captured in candidate generation.
4. **Inference Latency:** End-to-end processing time per evaluation trip.
