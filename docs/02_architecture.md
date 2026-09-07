# Architecture

## Overview

ModelVault is a gateway that sits in front of an existing model API. Every
`/predict` request flows through four layers in sequence:

```
Client query
     |
     v
Layer 1 -- Ingress / Velocity Governor        coarse rate limiting
     |
     v
Layer 2 -- Content-Based Anomaly Detector     scores WHAT is asked, not WHO is asking
     |
     v
Layer 3 -- Graduated Response Controller      degrades the response by tier
     |
     v
Layer 4 -- Dynamic Watermark + Verify         plants and later proves a hidden signal
```

## Layer 1 -- Ingress / Velocity Governor

`modelvault/layer1_ingress/velocity_governor.py`

A per-client token-bucket rate limiter (`rate_limit.requests_per_window` /
`rate_limit.window_seconds` in `config/settings.yaml`). This is intentionally
coarse -- it caps raw request velocity per identity and nothing else. Content
inspection is Layer 2's job.

## Layer 2 -- Content-Based Anomaly Detector

`modelvault/layer2_detection/`

- `reservoir.py`: a fixed-size sliding window (`reservoir.window_size`) of
  recent queries, pooled **across all clients**. This pooling is the whole
  point: an attacker splitting queries across many Sybil accounts still lands
  every query in the same shared window, so the signal below sees the
  systematic pattern regardless of how many identities it's spread across.
- `feature_distortion.py` (macro signal): scores how far a query sits from
  the training data manifold, using a `GaussianMixture` density model fitted
  during training. Because log-likelihood in high-dimensional space is
  heavily left-skewed, scores are calibrated via percentile rank against a
  large sample drawn from the reference model, then passed through a cubic
  decay so only genuinely rare queries score high -- ordinary traffic stays
  low even though its raw likelihood is far from the density's peak.
- `coverage_density.py` (micro signal): scores how sparsely a query's local
  neighborhood is populated relative to the rest of the reservoir window.
  Real usage clusters and repeats; an attacker sweeping feature space to
  maximize information gain per query does neither. This is what catches
  in-distribution extraction attacks that `feature_distortion.py` alone
  cannot, since those queries never leave the training manifold.
- `threat_index.py`: fuses macro (weight 0.6) and micro (weight 0.4) into a
  single 0-100 Threat Index.

## Layer 3 -- Graduated Response Controller

`modelvault/layer3_response/`

- `throttling.py`: maps the Threat Index to one of three tiers
  (`threat_index.tier_normal_max` / `tier_elevated_max` in settings.yaml) and
  degrades the response accordingly -- full precision, rounded confidence, or
  label-only.
- `boundary_perturbation.py`: at the critical tier, queries whose top-two
  class probabilities are within `margin` of each other (i.e. sitting near
  the decision boundary) have their returned label deterministically flipped
  with some probability, seeded from a hash of `(client_id, query, salt)`.
  This targets boundary-search attacks (HopSkipJump-style) that query densely
  around the boundary specifically to map its shape.

Degradation is graduated rather than a hard block because a binary block
either locks out legitimate users on a false positive, or reveals the exact
threshold an attacker needs to stay under.

## Layer 4 -- Dynamic Watermark + Verify

`modelvault/layer4_watermark/`

- `watermark.py`: for elevated/critical tier responses, deterministically
  decides (via `HMAC(secret, client_id + query)` compared against the tier's
  configured flip rate) whether to plant a watermark, and if so flips the
  returned label to a fixed, reproducible alternate class. Determinism means
  a client resending an identical query always gets an identical answer --
  there's no discrepancy to detect the defense by.
- `verification.py`: given a suspect model's predictions on a set of
  previously-watermarked queries, computes what fraction match the expected
  watermark label and reports a statistical confidence (1 - p-value of a
  one-sided exact binomial test against the chance-agreement rate).

## Gateway wiring

`modelvault/gateway/api.py` wires all four layers together behind a FastAPI
app: `/predict`, `/verify-ownership`, `/admin/stats`, `/admin/toggle-defense`,
plus `/health` (liveness) and `/ready` (readiness -- confirms the target
model, scaler, and reference density are actually loadable).
`modelvault/gateway/state.py` is the sole owner of in-memory client state
(reservoir, rate-limit buckets, watermark event log, stats counters) -- there
is deliberately no persistent telemetry database in this build.

Raw features are scaled exactly once, at the gateway boundary
(`transform_features` in `modelvault/model/predict.py`), using the
`StandardScaler` fit during training and saved alongside the model. Every
downstream layer -- reservoir storage, threat scoring, model inference,
watermark hashing -- operates on that same scaled representation. This
matters concretely for the credit card fraud dataset this system is
demonstrated against: raw `Amount` ranges from 0 to tens of thousands while
the `V1`-`V28` PCA features sit roughly in [-50, 50], so anything
distance-based (the reference density model, the reservoir's nearest-neighbor
coverage signal) would otherwise be dominated almost entirely by `Amount`.

### Hardening

- **Input validation** (`gateway/schemas.py`): feature count and finiteness
  are validated before any handler code runs. A custom
  `RequestValidationError` handler in `api.py` sanitizes non-finite values
  in the error response itself -- FastAPI's default error response echoes
  the rejected input back in its `detail`, and Starlette's `JSONResponse`
  refuses to encode `NaN`/`Infinity` even there, which would otherwise crash
  the handler trying to report that the input was invalid.
- **Thread safety**: `VelocityGovernor`, `QueryReservoir`, and `GatewayState`
  all guard their mutable state with locks. FastAPI executes sync `def`
  endpoints in a threadpool, so concurrent requests genuinely race on
  shared, unlocked dict/deque state.
- **Admin authentication** (`gateway/auth.py`): `/admin/*` and
  `/verify-ownership` require a matching `X-API-Key` header when
  `ADMIN_API_KEY` is set in `.env`. Unset, the gateway logs a loud warning
  and allows all requests through -- convenient for local development, never
  safe to leave that way in a real deployment.
