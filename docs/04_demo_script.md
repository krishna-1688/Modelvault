# Demo Script

## Setup (before the room fills up)

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SECRET_SALT with anything
python -m modelvault.model.train
```

Confirm `artifacts/target/target_classifier.joblib` and
`artifacts/reference/reference_density.joblib` both exist.

## Live demo flow

1. **Start the gateway** (terminal 1): `bash scripts/run_gateway.sh`
   - Hit `curl http://localhost:8000/health` to show it's alive.
2. **Start the dashboard** (terminal 2): `bash scripts/run_dashboard.sh`
   - Open the browser tab. Point out the (currently empty) threat timeline,
     tier distribution, and alert feed.
3. **Send one normal-looking request** (terminal 3):
   ```bash
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"client_id": "real-user", "features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}'
   ```
   Show the dashboard update: one point on the timeline, normal/elevated tier.
4. **Run the attack simulation**: `python -m attack_sim.evaluate`
   - While it runs, flip back to the dashboard: watch the threat timeline
     spike, tier distribution shift toward critical, and the alert feed fire
     a "HIGH ALERT" banner.
   - When it finishes, read the printed table: undefended vs. defended
     agreement for all three attack types, and the ownership verification
     result at the bottom.
5. **Explain the number that matters**: the in-distribution attack shows the
   biggest story -- it's the one an identity-based or macro-only defense
   would have completely missed, and it's still meaningfully degraded here
   because of the coverage-density signal.
6. **Toggle defense off** via the dashboard sidebar, rerun a slice of the
   attack, and show the agreement number jump back up -- this demonstrates
   the gateway is actually doing the work, not just reporting a static number.

## One-command version

If time is short: `bash scripts/run_full_demo.sh` starts the gateway, runs
the full comparison, and prints the ownership verification result in one
shot -- useful as a fallback if live-clicking through the dashboard risks
running over time.

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
