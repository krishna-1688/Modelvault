# Trade-offs and Scope Boundaries

## What's explicitly out of scope for this build

- **Differential privacy noise injection.** A natural next step, but it
  changes the output distribution in ways that need careful utility/privacy
  tuning we didn't have time to validate. Left as future work.
- **Confidential computing / TEEs.** Solves a different threat model (server
  compromise), not the API-scraping threat this project targets.
- **FlowGuard-style normalizing flows for density estimation.** Where the
  field is headed for higher-fidelity manifold modeling, but a
  `GaussianMixture` reference density is sufficient to demonstrate the
  detection mechanism at hackathon scope, and is far cheaper to fit and
  reason about.
- **Full FDINet over real deep-network internal activations.** Our target
  model is a `LogisticRegression` classifier over engineered features, not a
  deep network with internal activations to instrument -- the
  density-over-features simplification is the intended scope here, not a
  shortfall.
- **A persistent (e.g. SQLite) telemetry layer.** `gateway/state.py` already
  owns all client state in memory. Adding a database would duplicate that
  responsibility for no benefit at this scale, and state doesn't need to
  survive a restart for a demo.
- **Automated external dataset fetching as the default data path.** Demo day
  has no internet guarantee. `sklearn.make_classification` is fully synthetic,
  deterministic, and requires no network access.

## Known limitations

- **The reservoir is a single shared in-memory window.** In a real
  multi-instance deployment this would need to be a shared store (e.g. Redis)
  so detection works across gateway replicas, not just within one process.
- **The macro signal's calibration is sample-based, not exact.** It draws a
  large sample from the fitted reference density to build an empirical
  percentile-rank distribution, which is an approximation, not a closed-form
  calculation, and needs periodic refitting if the true input distribution
  drifts over time.
- **Watermark flip rates are hand-tuned, not learned.** `flip_rate_elevated`
  and `flip_rate_critical` in `config/settings.yaml` were tuned empirically
  against `attack_sim/evaluate.py`'s output rather than derived from a formal
  utility/detectability trade-off model.
- **The boundary attack simulation is simplified.** It's a basic binary-search
  boundary walk, not a full HopSkipJump implementation -- sufficient to
  exercise `boundary_perturbation.py`, but not a state-of-the-art attack
  implementation in its own right.

## Why these trade-offs are acceptable here

This is a hackathon-scoped proof of concept, not a production security
product. Every simplification above is either (a) something a real
deployment would swap for infrastructure the hackathon environment can't
provide (Redis, GPUs, internet access), or (b) a reduction in model
complexity that doesn't change which defense mechanism is being
demonstrated -- just how faithfully it mirrors a production-scale target
model.
