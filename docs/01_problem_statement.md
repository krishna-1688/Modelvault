# Problem Statement

## The attack

Model extraction (a.k.a. model stealing) is a catalogued adversarial ML technique
(OWASP LLM10, MITRE ATLAS AML.T0024): an attacker repeatedly queries a deployed
model's API, records the input/output pairs, and trains a "surrogate" model that
reproduces the original's behavior -- without ever touching its weights or
breaching the serving infrastructure.

This is not theoretical. Training a serious production ML model costs anywhere
from $1.2M to $200M+ depending on scale and domain. Cloning its observable
behavior via API queries can cost under $100 in compute and a few hours of
scripting. The economics are lopsided enough that this has become a primary
threat model for any company whose ML model is also its product.

## Why it matters now

Robust Intelligence, Protect AI, and Lakera -- three companies built
specifically around defending ML systems from attacks like this -- were
acquired for a combined $1.2B+ in the last ~18 months. AI IP protection has
gone from a research curiosity to a funded category.

## What existing defenses miss

Older detection approaches (e.g. PRADA) are identity-based: they build a
suspicion score per client account. This is provably evadable with a Sybil
attack -- split the same query pattern across enough fake accounts and no
single identity ever crosses the detection threshold, even though the
aggregate traffic is clearly systematic extraction.

Hard rate limits and query blocking are the other common response, but a
binary block either locks out legitimate users who trip a false positive, or
hands the attacker a clean signal of exactly where the defense's threshold
sits -- which they then simply stay under.

## What ModelVault does differently

ModelVault scores the *content* of queries pooled across all clients (closing
the Sybil hole), degrades responses gradually instead of blocking (avoiding
both false-positive lockouts and threshold leakage), and plants a
deterministic, hidden watermark into degraded responses so that if a stolen
surrogate does surface, ownership can be proven with a statistical confidence
score -- not just asserted.
