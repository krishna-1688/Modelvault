# ModelVault

A defense system against **model extraction attacks** — where an attacker queries a deployed ML
model's API repeatedly, records input/output pairs, and trains a "surrogate" clone that reproduces
the original model's behavior without ever touching its weights or breaching a server.

ModelVault sits in front of an existing model API and:

1. Detects when incoming queries look like an extraction attempt (not real usage)
2. Gradually gives less information back the more suspicious a client looks
3. Silently plants a hidden, deterministic watermark into responses to sustained-suspicious clients
4. Can later **prove** a suspect model was stolen from it, with a statistical confidence score

Built for Problem Statement 16 (Adversarial ML / AI IP Protection).

## Why this problem is real

- Training a serious ML model costs $1.2M–$200M+ depending on scale. Cloning its behavior via API
  queries can cost under $100.
- Robust Intelligence, Protect AI, and Lakera were acquired for a combined $1.2B+ in the last ~18
  months specifically for this category of defense.
- It's a catalogued attack technique: OWASP LLM10, MITRE ATLAS AML.T0024.

## Architecture

```
Client query
     |
     v
Layer 1 -- Ingress / Velocity Governor        coarse rate limiting
     |
     v
Layer 2 -- Content-Based Anomaly Detector     scores WHAT is asked, not WHO is asking --
  * Macro: feature distortion vs. training     survives an attacker splitting queries
    manifold (density estimation)              across multiple fake accounts
  * Micro: query coverage/redundancy
  * Fused into a 0-100 Threat Index
     |
     v
Layer 3 -- Graduated Response Controller      0-35: full response
  Suspicion score drives information          36-70: rounded confidence
  degradation, never a hard block             71-100: label-only + noise
     |
     v
Layer 4 -- Dynamic Watermark + Verify         deterministic per (client_id, query) via HMAC --
  Plants a hidden, reproducible signal        same query always gets the same response, no
  into high-suspicion responses.              discrepancy leak. /verify-ownership proves theft
  /verify-ownership proves theft with a       with a statistical confidence score.
  statistical confidence score.
```

**Design rationale:**

- **Layer 2 is content-based, not identity-based.** Identity/per-client detection (e.g. PRADA) is
  provably evaded by splitting queries across multiple fake accounts (Sybil attack). Scoring the
  query itself, independent of who sent it, closes that hole.
- **Layer 3 is graduated, not a binary block.** A hard block either locks out real users (false
  positives) or tells the attacker exactly where the detection threshold is. Gradual degradation
  avoids both.
- **Layer 4 watermarks at inference time**, not training time, because a trigger baked into
  training doesn't reliably transfer into an attacker's independently trained surrogate model.
  Watermarking API responses works because the attacker unknowingly trains on our (occasionally
  altered) output.
- **The watermark decision is deterministic** (`HMAC(secret, client_id + query)`), so a client
  can't detect the defense by sending the same query twice and comparing answers.

## Configuration

**Secrets and ports go in `.env`; every tunable threshold goes in `config/settings.yaml`. Never
duplicate a value in both.**

## Getting started

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd
pip install -r requirements.txt
cp .env.example .env            # then fill in SECRET_SALT

python -m modelvault.model.train        # trains target model + reference density
bash scripts/run_gateway.sh             # starts the FastAPI gateway
bash scripts/run_dashboard.sh           # starts the Streamlit dashboard (separate terminal)
python -m attack_sim.evaluate           # undefended vs. defended comparison
```

## Project layout

See `docs/02_architecture.md` for the full breakdown. Core packages:

- `modelvault/model/` -- target classifier + reference density training/inference
- `modelvault/layer1_ingress/` -- rate limiting
- `modelvault/layer2_detection/` -- threat index (macro + micro signals)
- `modelvault/layer3_response/` -- graduated response degradation
- `modelvault/layer4_watermark/` -- watermark injection + ownership verification
- `modelvault/gateway/` -- FastAPI app wiring it all together
- `attack_sim/` -- extraction attack simulations used to measure defense effectiveness
- `dashboard/` -- Streamlit live monitoring UI

## Out of scope

Differential privacy noise injection, confidential computing/TEEs, normalizing-flow-based
detection, full FDINet over deep-network internals, and a persistent telemetry database are all
explicitly out of scope for this build. See `docs/05_tradeoffs.md`.
