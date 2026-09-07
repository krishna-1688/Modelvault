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
  model is a `RandomForestClassifier` over engineered features, not a deep
  network with internal activations to instrument -- the density-over-features
  simplification is the intended scope here, not a shortfall.
- **A persistent (e.g. SQLite) telemetry layer.** `gateway/state.py` already
  owns all client state in memory. Adding a database would duplicate that
  responsibility for no benefit at this scale, and state doesn't need to
  survive a restart for a demo.
- **The real dataset as the ONLY data path.** Demo day has no internet
  guarantee, so the real dataset is fetched once and cached locally
  (`data/raw/creditcard.csv`, gitignored -- 142MB, too large to commit), with
  a synthetic fallback (`modelvault/model/data_prep.py`) if neither the cache
  nor a network fetch is available. `MODELVAULT_FORCE_SYNTHETIC_DATA=1`
  forces the fallback explicitly -- used in CI so the pipeline stays fast and
  network-independent rather than re-downloading 150MB on every run.

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
- **"Surrogate agreement with target" is a weaker signal for the boundary
  attack specifically than for the other two.** By construction, its
  training queries sit exactly on the decision boundary -- the region where
  even the true model's own predictions are least stable and a KNN
  surrogate's local smoothness assumption is most violated. That makes
  raw accuracy noisier run-to-run for this attack than for random_query or
  in_distribution. The literature typically evaluates boundary/HopSkipJump-
  style attacks on boundary-reconstruction fidelity or adversarial-example
  transferability, not held-out accuracy -- a more faithful metric we didn't
  implement here. What's unambiguous regardless is the ownership-verification
  result: boundary-attack traffic reliably lands a high fraction of queries
  in the critical tier, generates many watermark triggers, and produces a
  high-confidence `verified=True` result -- which is arguably the more
  important claim for this attack type anyway (can we prove theft?), even
  when the accuracy-degradation number is noisier.

## Why these trade-offs are acceptable here

This is a hackathon-scoped proof of concept, not a production security
product. Every simplification above is either (a) something a real
deployment would swap for infrastructure the hackathon environment can't
provide (Redis, GPUs, internet access), or (b) a reduction in model
complexity that doesn't change which defense mechanism is being
demonstrated -- just how faithfully it mirrors a production-scale target
model.
