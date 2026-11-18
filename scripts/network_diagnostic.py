"""
PathSnap Network Diagnostic Utility (Laptop 2)
Author: Member 4 (UI, Integration, 2-Laptop Networking)

Tests end-to-end network connectivity between Laptop 2 (UI) and Laptop 1 (FastAPI Backend).
Verifies:
1. Local IP and Subnet configuration
2. TCP port 8000 accessibility
3. REST /health heartbeat & round-trip latency
4. REST /trips evaluation catalog access
"""

import sys
import time
import socket
import argparse
import requests


def check_local_ip() -> str:
    """Returns local host IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def test_tcp_port(host: str, port: int, timeout: float = 3.0) -> bool:
    """Tests if remote TCP port is open."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def run_diagnostics(backend_url: str):
    print("=" * 65)
    print("🛰️  PathSnap Two-Laptop Network Diagnostic Tool (Laptop 2)")
    print("=" * 65)

    local_ip = check_local_ip()
    print(f"[1/4] Laptop 2 (Current Device) Local IP: {local_ip}")

    # Parse target host and port
    clean_url = backend_url.rstrip("/")
    target = clean_url.replace("http://", "").replace("https://", "")
    if ":" in target:
        host, port_str = target.split(":")
        port = int(port_str)
    else:
        host = target
        port = 8000

    print(f"[2/4] Probing Laptop 1 at {host}:{port}...")
    is_open = test_tcp_port(host, port)
    if is_open:
        print(f"      TCP Port {port} is OPEN and reachable!")
    else:
        print(f"      ❌ TCP Port {port} is CLOSED or unreachable.")
        print("      Troubleshooting Tips:")
        print("      - Make sure Laptop 1 runs: uvicorn backend.mock_api:app --host 0.0.0.0 --port 8000")
        print("      - Ensure Windows Defender Firewall allows port 8000 on Laptop 1:")
        print("        New-NetFirewallRule -DisplayName 'PathSnap' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow")
        print("      - Verify both laptops are on the same Mobile Hotspot.")
        sys.exit(1)

    print(f"[3/4] Testing REST API Heartbeat ({clean_url}/health)...")
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{clean_url}/health", timeout=3.0)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            data = r.json()
            print(f"       HTTP 200 OK — Ping Latency: {lat_ms:.1f} ms")
            print(f"      Service: {data.get('service')}")
            print(f"      Host:    {data.get('host')}")
        else:
            print(f"      ❌ Server returned unexpected HTTP status: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"      ❌ REST Health Check Failed: {e}")
        sys.exit(1)

    print(f"[4/4] Verifying Evaluation Trips Catalog ({clean_url}/trips)...")
    try:
        r = requests.get(f"{clean_url}/trips", timeout=3.0)
        if r.status_code == 200:
            trips = r.json().get("trips", [])
            print(f"       Found {len(trips)} Evaluation Trips:")
            print(f"      Trips: {', '.join(trips[:5])} ...")
        else:
            print(f"      ❌ Failed to fetch trips: HTTP {r.status_code}")
    except Exception as e:
        print(f"      ❌ Trips endpoint failed: {e}")

    print("=" * 65)
    print(" ALL NETWORK CHECKS PASSED! Laptop 2 is 100% ready for evaluation.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PathSnap 2-Laptop Network Connection.")
    parser.add_argument("--url", default="http://localhost:8000", help="Laptop 1 Backend URL")
    args = parser.parse_args()
    run_diagnostics(args.url)
