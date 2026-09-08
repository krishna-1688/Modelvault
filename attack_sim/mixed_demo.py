"""Drives a realistic mixed workload against a running gateway: legitimate
consumers and an extraction attacker, at the same time, on the same endpoint.

This is the scenario the product actually has to survive. A defense that
catches attackers by degrading everyone is worthless -- the demo has to show
the gateway separating the two populations from query CONTENT alone, serving
paying customers at full fidelity while the attacker's reconstruction stalls.

Each request carries an optional X-Demo-Label header ("legitimate" or
"attacker"). The gateway records it for scoring ONLY and never feeds it to
any detection layer -- it exists so the console can show a real, measured
false-positive rate instead of asking the audience to take one on faith.
See the header handling in modelvault/gateway/api.py.

Usage (gateway already running):
    python -m attack_sim.mixed_demo                       # 60s mixed workload
    python -m attack_sim.mixed_demo --duration 120        # longer run
    python -m attack_sim.mixed_demo --attack boundary     # different attack shape
    python -m attack_sim.mixed_demo --legit-only          # baseline: no attacker
"""
from __future__ import annotations

import argparse
import itertools
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from attack_sim import boundary_attack, in_distribution_attack, legit_client, random_query_attack
from modelvault.model.predict import predict_raw_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")
GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "8000")
GATEWAY_URL = os.environ.get("GATEWAY_URL") or f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

N_SYBIL_CLIENTS = 25
RUN_ID = uuid.uuid4().hex[:6]

_stop = threading.Event()
_counters = {"legit_sent": 0, "legit_blocked": 0, "attack_sent": 0, "attack_blocked": 0}
_counter_lock = threading.Lock()


def _post(session: requests.Session, client_id: str, features: np.ndarray, demo_label: str) -> int:
    try:
        response = session.post(
            f"{GATEWAY_URL}/predict",
            json={"client_id": client_id, "features": features.tolist()},
            headers={"X-Demo-Label": demo_label},
            timeout=5,
        )
        return response.status_code
    except requests.RequestException:
        return 0


def legitimate_worker(rate_per_second: float) -> None:
    """Steady, modest, repetitive traffic from a few named consumers."""
    profiles = legit_client.build_consumer_profiles(seed=7)
    rng = np.random.default_rng(11)
    session = requests.Session()
    consumers = itertools.cycle(legit_client.CONSUMER_IDS)
    interval = 1.0 / rate_per_second

    while not _stop.is_set():
        consumer = next(consumers)
        features = legit_client.next_query(profiles, consumer, rng)
        status = _post(session, consumer, features, "legitimate")
        with _counter_lock:
            _counters["legit_sent"] += 1
            if status == 429:
                _counters["legit_blocked"] += 1
        _stop.wait(interval)


def attacker_worker(attack: str, rate_per_second: float) -> None:
    """A systematic extraction sweep, split across sybil identities."""
    if attack == "random_query":
        from modelvault.model.data_prep import generate_dataset
        n_features = generate_dataset()[1].shape[1]
        queries = random_query_attack.generate_queries(4000, n_features, seed=1)
    elif attack == "boundary":
        queries = boundary_attack.generate_queries(4000, predict_fn=predict_raw_label, seed=3)
    else:
        queries = in_distribution_attack.generate_queries(4000, seed=2)

    session = requests.Session()
    interval = 1.0 / rate_per_second

    for i, features in enumerate(queries):
        if _stop.is_set():
            return
        client_id = f"api-user-{RUN_ID}-{i % N_SYBIL_CLIENTS}"
        status = _post(session, client_id, features, "attacker")
        with _counter_lock:
            _counters["attack_sent"] += 1
            if status == 429:
                _counters["attack_blocked"] += 1
        _stop.wait(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed legitimate + attacker traffic against the gateway.")
    parser.add_argument("--duration", type=int, default=60, help="Seconds to run.")
    parser.add_argument("--attack", choices=["random_query", "in_distribution", "boundary"], default="in_distribution")
    parser.add_argument("--legit-rate", type=float, default=6.0, help="Legitimate requests per second.")
    parser.add_argument("--attack-rate", type=float, default=14.0, help="Attacker requests per second.")
    parser.add_argument("--legit-only", action="store_true", help="Baseline mode: no attacker traffic.")
    parser.add_argument("--attack-delay", type=int, default=8, help="Seconds of clean traffic before the attack starts.")
    args = parser.parse_args()

    try:
        requests.get(f"{GATEWAY_URL}/health", timeout=3).raise_for_status()
    except requests.RequestException:
        print(f"Gateway not reachable at {GATEWAY_URL}. Start it first.")
        return

    print(f"Target: {GATEWAY_URL}")
    print(f"Legitimate consumers: {', '.join(legit_client.CONSUMER_IDS)} @ {args.legit_rate}/s")
    if args.legit_only:
        print("Attacker: none (baseline mode -- shows normal traffic is served untouched)")
    else:
        print(f"Attacker: {args.attack} @ {args.attack_rate}/s across {N_SYBIL_CLIENTS} sybil identities, starting in {args.attack_delay}s")
    print(f"Running for {args.duration}s. Watch the console.\n")

    threads = [threading.Thread(target=legitimate_worker, args=(args.legit_rate,), daemon=True)]
    threads[0].start()

    if not args.legit_only:
        def delayed_attack():
            _stop.wait(args.attack_delay)
            if not _stop.is_set():
                print(">>> ATTACK STARTING NOW <<<\n")
                attacker_worker(args.attack, args.attack_rate)
        t = threading.Thread(target=delayed_attack, daemon=True)
        t.start()
        threads.append(t)

    try:
        deadline = time.time() + args.duration
        while time.time() < deadline:
            time.sleep(2)
            with _counter_lock:
                c = dict(_counters)
            print(f"  legit sent={c['legit_sent']} (429s {c['legit_blocked']}) | "
                  f"attack sent={c['attack_sent']} (429s {c['attack_blocked']})")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _stop.set()
        time.sleep(0.4)

    with _counter_lock:
        c = dict(_counters)
    print("\n=== Run complete ===")
    print(f"Legitimate: {c['legit_sent']} sent, {c['legit_blocked']} rate-limited")
    print(f"Attacker:   {c['attack_sent']} sent, {c['attack_blocked']} rate-limited")
    print("Check the console's Service Integrity panel for the measured false-positive rate.")


if __name__ == "__main__":
    main()
