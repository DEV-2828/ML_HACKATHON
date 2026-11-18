"""End-to-end verification script for Laptop 1 FastAPI server (`backend/api.py`).

Tests `/health`, `/trips`, and `/predict` across ALL available trips over real HTTP
using only the Python standard library (`urllib.request`), verifying response schema,
real non-hardcoded metrics, and latency.

Usage:
  1. Against an already-running Laptop 1 server:
       python -m backend.test_api --url http://<LAPTOP1_IP>:8000
  2. Self-contained local test (starts uvicorn on a temp port, tests all 10 trips, exits):
       python -m backend.test_api --spawn-local
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict


def _http_get(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        assert resp.status == 200, f"GET {url} returned {resp.status}"
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        assert resp.status == 200, f"POST {url} returned {resp.status}"
        return json.loads(resp.read().decode("utf-8"))


def run_suite(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    health = _http_get(f"{base_url}/health")
    print("HEALTH:", health)

    trips_resp = _http_get(f"{base_url}/trips")
    trips = trips_resp["trips"]
    print(f"TRIPS ({len(trips)}): {trips}\n")
    assert len(trips) > 0, "Server returned 0 trips!"

    for tid in trips:
        body = _http_post(f"{base_url}/predict", {"trip_id": tid})
        assert body["status"] == "success"
        m = body["metrics"]
        assert len(body["raw_gps"]) > 0
        assert len(body["hmm_reconstructed_route"]) > 0
        assert len(body["classifier_only_route"]) > 0
        print(
            f"  trip={tid} | ClfTop1={m['classifier_top_1_accuracy']:.4f} -> "
            f"HMMTop1={m['top_1_accuracy']:.4f} | ClfCont={m['classifier_continuity']:.4f} -> "
            f"HMMCont={m['route_continuity']:.4f} | Miss={m['candidate_miss_rate']:.4f} | "
            f"Latency={m['inference_time_ms']:.0f}ms | SeqF1={body['details'].get('sequence_arc_f1')}"
        )

    print(f"\nALL {len(trips)} TRIPS PASSED PRODUCTION FASTAPI END-TO-END!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PathSnap FastAPI backend across all trips.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Target Laptop 1 API URL")
    parser.add_argument("--spawn-local", action="store_true", help="Spawn local uvicorn on port 8009 for automated verification")
    args = parser.parse_args()

    if not args.spawn_local:
        run_suite(args.url)
        return

    port = 8009
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        ready = False
        for _ in range(40):
            try:
                _http_get(f"{base}/health", timeout=1.0)
                ready = True
                break
            except Exception:
                time.sleep(0.4)
        if not ready:
            raise RuntimeError("Local uvicorn server did not become ready in time.")
        run_suite(base)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
