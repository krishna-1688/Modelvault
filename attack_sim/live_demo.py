"""Sends REAL HTTP traffic to a running gateway (unlike evaluate.py, which
uses FastAPI's in-process TestClient and therefore never touches whatever
server is actually running in another terminal). Use this specifically to
watch the console react live during a demo -- e.g. from a second machine,
pointed at the gateway's Cloudflare tunnel URL via GATEWAY_URL.

Every request carries X-Demo-Label: attacker, the same instrumentation
attack_sim/mixed_demo.py uses for its attacker worker. This is read ONLY by
the console's Service Integrity panel to report a measured false-positive/
recall/precision rate -- detection itself never sees it. Without this header
this traffic still gets caught (or not) exactly the same way; it just
wouldn't be credited as "attacker" in that one panel, which is why this
script and mixed_demo.py must both set it to keep those numbers meaningful
regardless of which script generated the traffic.

Usage (with the gateway already running via `uvicorn modelvault.gateway.api:app`):
    python -m attack_sim.live_demo --attack in_distribution --n 300 --delay 0.03
    python -m attack_sim.live_demo --attack all --n 200
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

from attack_sim import boundary_attack, in_distribution_attack, random_query_attack
from modelvault.model.predict import predict_raw_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")
GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "8000")
# GATEWAY_URL overrides host/port entirely -- needed for tunnels (Cloudflare,
# ngrok) that serve over https with no explicit port, e.g.
# https://xxxx.trycloudflare.com
GATEWAY_URL = os.environ.get("GATEWAY_URL") or f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
N_SYBIL_CLIENTS = 25
# Salts sybil client_ids so re-running the SAME attack type twice (e.g. once
# with defense on, once off, to compare) produces a fresh set of identities
# each time -- without this, "in_distribution-sybil-0..24" collides byte-for-
# byte across runs, and the dashboard's distinct-client counter looks frozen
# even though traffic is genuinely flowing.
RUN_ID = uuid.uuid4().hex[:6]


def generate(attack: str, n: int, seed: int):
    if attack == "random_query":
        from modelvault.model.data_prep import generate_dataset
        n_features = generate_dataset()[1].shape[1]
        return random_query_attack.generate_queries(n, n_features, seed=seed)
    if attack == "in_distribution":
        return in_distribution_attack.generate_queries(n, seed=seed)
    if attack == "boundary":
        return boundary_attack.generate_queries(n, predict_fn=predict_raw_label, seed=seed)
    raise ValueError(f"unknown attack: {attack}")


def run_live_attack(attack: str, n: int, delay: float) -> None:
    print(f"\n== {attack} attack: sending {n} real HTTP requests to {GATEWAY_URL} ==")
    queries = generate(attack, n, seed=hash(attack) % 1000)

    session = requests.Session()
    sent, blocked = 0, 0
    for i, query in enumerate(queries):
        client_id = f"{attack}-{RUN_ID}-sybil-{i % N_SYBIL_CLIENTS}"
        try:
            response = session.post(
                f"{GATEWAY_URL}/predict",
                json={"client_id": client_id, "features": query.tolist()},
                headers={"X-Demo-Label": "attacker"},
                timeout=5,
            )
        except requests.RequestException as exc:
            print(f"Could not reach gateway at {GATEWAY_URL}: {exc}")
            return

        if response.status_code == 429:
            blocked += 1
        else:
            sent += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} sent (blocked so far: {blocked})")

        if delay > 0:
            time.sleep(delay)

    print(f"Done: {sent} accepted, {blocked} rate-limited. Check the dashboard for the live effect.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send real attack traffic to a running ModelVault gateway.")
    parser.add_argument("--attack", choices=["random_query", "in_distribution", "boundary", "all"], default="in_distribution")
    parser.add_argument("--n", type=int, default=300, help="Number of queries to send per attack.")
    parser.add_argument("--delay", type=float, default=0.03, help="Seconds to sleep between requests (0 for max speed).")
    args = parser.parse_args()

    try:
        requests.get(f"{GATEWAY_URL}/health", timeout=3).raise_for_status()
    except requests.RequestException:
        print(f"Gateway not reachable at {GATEWAY_URL}. Start it first with:\n  uvicorn modelvault.gateway.api:app --host 0.0.0.0 --port {GATEWAY_PORT}")
        return

    attacks = ["random_query", "in_distribution", "boundary"] if args.attack == "all" else [args.attack]
    for attack in attacks:
        run_live_attack(attack, args.n, args.delay)


if __name__ == "__main__":
    main()