# Demo Script

## Setup (before the room fills up)

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SECRET_SALT and ADMIN_API_KEY with anything
python -m modelvault.model.train        # fetches + caches the real dataset on first run
```

Confirm `artifacts/target/target_classifier.joblib`, `artifacts/target/scaler.joblib`, and
`artifacts/reference/reference_density.joblib` all exist. Note the printed precision/recall/F1/
ROC-AUC/PR-AUC -- these are the real numbers for the real fraud model being protected.

A quick way to get a real transaction's feature vector for pasting into curl commands during the
demo (raw features, 29 values -- too many to type by hand):

```bash
python -c "
from modelvault.model.data_prep import generate_dataset
X_train, _, _, _ = generate_dataset()
print(X_train[0].tolist())
"
```

## Live demo flow

1. **Start the gateway** (terminal 1): `bash scripts/run_gateway.sh`
   - Hit `curl http://localhost:8000/health` (process is up) and
     `curl http://localhost:8000/ready` (model/scaler/reference density actually loaded).
2. **Start the dashboard** (terminal 2): `bash scripts/run_dashboard.sh`
   - Open the browser tab. Point out the (currently empty) threat timeline,
     tier distribution, and alert feed.
3. **Send one real transaction** (terminal 3), using a real feature vector from the snippet above:
   ```bash
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"client_id": "real-user", "features": [<29 real values>]}'
   ```
   Show the dashboard update: one point on the timeline, normal tier.
4. **Run LIVE attack traffic against the running gateway** (terminal 3) --
   this is `attack_sim.live_demo`, not `attack_sim.evaluate`. `evaluate.py` uses an in-process test
   client for speed and doesn't touch whatever gateway is running in another terminal; `live_demo.py`
   sends real HTTP requests, which is what the dashboard needs to see:
   ```bash
   python -m attack_sim.live_demo --attack in_distribution --n 300
   ```
   Flip to the dashboard immediately: watch the threat timeline spike, tier distribution shift
   toward critical, watermark triggers tick up, and the alert feed fire a "HIGH ALERT" banner.
5. **Show the real numbers** (can be run ahead of time, or live if time allows -- takes several
   minutes due to the RandomForest target model's per-query cost): `python -m attack_sim.evaluate`
   - Prints undefended vs. defended clone agreement, averaged over multiple random seeds, for all
     three attack types, plus the ownership-verification result. See `docs/06_benchmark.md` for the
     last recorded numbers and full methodology if you don't want to run it live.
6. **Explain why in-distribution matters most**: it's the attack an identity-based or macro-only
   defense would completely miss -- these queries look legitimate one at a time. It's still
   meaningfully degraded here because of the coverage-density signal.
7. **Toggle defense off** via the dashboard sidebar, send the same real transaction again (or rerun
   `live_demo`), and show the response return full, undegraded probabilities with `threat_index: 0`
   even for an obviously attack-shaped query -- this demonstrates the gateway is actively deciding
   this per request, not reporting a static number.

## One-command version

If time is short: `bash scripts/run_full_demo.sh` trains the model if needed, then runs the full
undefended-vs-defended comparison and prints the ownership verification result in one shot --
useful as a fallback if live-clicking through the dashboard risks running over time. It does not
require the gateway to be running separately (it uses the same in-process test client as
`evaluate.py`).

## Anticipated questions

- **"Doesn't watermarking hurt real users?"** Only elevated/critical-tier
  traffic is ever watermarked, at low, tunable flip rates -- normal traffic is
  never touched.
- **"Can an attacker detect the watermark by resending queries?"** No --
  the decision and the flipped label are both deterministic functions of
  `(client_id, query)`, so identical requests always get identical responses.
- **"What stops an attacker from filtering out watermarked points?"** They
  can't tell which responses were watermarked without the secret salt; from
  their side, the label just looks like normal model uncertainty.
- **"Why fraud detection specifically?"** It's a real, publicly available,
  severely imbalanced (0.17% positive) dataset -- the industry-standard
  benchmark for this class of problem -- so the numbers in this demo are
  real classifier performance and real attack degradation, not a toy
  synthetic task tuned to look good.
- **"Is this actually secure, or just a demo?"** Point at the hardening list
  in `docs/02_architecture.md`: thread-safe shared state, input validation
  (including a real crash bug found and fixed), admin API-key auth, and a
  liveness/readiness split -- these aren't cosmetic, they're the things a
  real deployment would actually need.
