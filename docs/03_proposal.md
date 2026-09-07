# Proposal

## What we're building

A drop-in gateway that any team serving a proprietary ML model over an API can
put in front of their existing inference endpoint, with no changes to the
underlying model. It defends against model extraction attacks in three
escalating ways: detect, degrade, and (after the fact) prove theft.

## Why this approach

1. **Content-based detection survives Sybil evasion.** Identity-based
   detection is a known dead end -- split queries across enough accounts and
   any per-client threshold stops firing. Scoring the pooled query stream
   itself removes that escape hatch entirely.
2. **Graduated degradation avoids the false-positive/threshold-leak
   dilemma.** A hard block is a blunt instrument: it either costs you real
   users or it teaches the attacker exactly where to stop. Smooth degradation
   does neither.
3. **Watermarking closes the loop.** Detection and degradation slow an
   attacker down; they don't give you proof after the fact if a competitor's
   product turns out to behave suspiciously like yours. The watermark is
   designed specifically to survive into an attacker's independently trained
   surrogate and to be checkable later with a real statistical confidence
   number, not a guess.

## What we measured

Using `attack_sim/evaluate.py` against three attack types (random-query,
in-distribution, and boundary-search), against a real RandomForest fraud
classifier trained on the Kaggle/ULB Credit Card Fraud dataset (284,807
transactions, 0.17% fraud), comparing an undefended baseline to the full
ModelVault gateway -- see `docs/06_benchmark.md` for the exact numbers and
full methodology (evaluated on a class-balanced set, since raw accuracy is
meaningless at this class imbalance).

The in-distribution result matters most: it's the attack an earlier,
simpler prototype (single flat watermark rate, no reservoir, no
coverage-density signal) had essentially no defense against. The
coverage/redundancy signal in Layer 2 is what closes that gap.

Ownership verification on the boundary-attack surrogate came back
`verified=True` with high statistical confidence, using only responses the
gateway had already returned during the attack -- no special instrumentation
needed after the fact.

## Scope and constraints

The gateway itself runs locally with free, open-source tooling (FastAPI,
scikit-learn, Streamlit, Docker). The target model is trained on real,
publicly available data (fetched once, cached locally, with a synthetic
fallback for offline/CI use), so the whole system is reproducible without
paid infrastructure. See `05_tradeoffs.md` for what was deliberately left
out of this build and why.
