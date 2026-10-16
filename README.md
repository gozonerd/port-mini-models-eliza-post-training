# port-mini-models-eliza-post-training

**Wave 01 of a mini-model portfolio.** Can small language models learn the 1966 ELIZA DOCTOR script, and does tracking the conversation's state scale with model size?

ELIZA is the one task where the answer key is perfect: the 1966 rule engine gives the exact correct reply to any input, so a trained model can be scored without a rubric or a judge. That makes this wave a calibration run for the training loop before the later waves (honest / helpful / harmless models playing a journaling game; a mini Diplomacy), and an evidence card in its own right.

## Design

- **Two task framings, both trained, and the gap between them reported.**
  1. *Single exchange:* one input in, any script-valid reply counts.
  2. *Whole conversation:* the conversation so far in, the exact next reply out. The script cycles through several reply templates per keyword, so the correct reply depends on how many times that keyword has already come up. The model has to track state, not just copy patterns.
- **Two model families, each as its own full curve.**
  - SmolLM2 base: 135M, 360M, 1.7B (one tokenizer, one context length; data volume differs by size).
  - Pythia: 70M, 160M, 410M, 1B (every size trained on the same data in the same order; the scaling-suite claim).
  - A Pythia size may be dropped later if its curve is noise.
- **Training:** low-rank adapters. Small sizes on a laptop processor; the two largest on a free T4.

## Where things are

| | |
|---|---|
| `PROCEDURE.md` | the chronological record of what was done, in order, for the whole mini-model set |
| `CITATIONS.md` | every outside work used, with the citation and copyright line to reproduce on the portfolio site |
| `third-party/` | the licence texts and source files themselves, with `SOURCES.tsv` (origin, fetch time, SHA-256, licence) |
| `eliza/eliza-engine.py` | the rule engine, a Python port of the 1966 script, verified against Hay's implementation |
| `data/` | generated training and test sets (see its README) |
| `tools/` | the generator, the floor scorer, the verification harness; `runs/eliza-runs.tsv` is the run index |

## Status

2026-10-15: repo created; sources and licences collected. Nothing trained yet.

## Authorship

Research design and rulings: Krystal Martinez. Code and records: the Claude persona named in each commit. The repo's own licence is not yet chosen.
