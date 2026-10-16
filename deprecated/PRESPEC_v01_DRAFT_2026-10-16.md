# PRESPEC — Wave 01 (ELIZA), DRAFT v01, not ratified

**Status:** draft for Krystal Martinez's ratification (K55: the Wave 01 prespecification is written in full by the seat and ratified by her before the first training arm runs). Section 3 is complete and may run now: it is a measurement on untrained models, like the floor (K59), and it fixes its harness choices before the numbers exist. Sections 4 to 7 wait on her hypotheses (artifact "Wave 01 Floor and Hypotheses", ledger) and are marked PENDING where they do.

Drafted 2026-10-16 by Flaudechamba Flaudicota A.-L. Formative Horizon v01 (Claude Fable 5.1). Edits are versioned, never silent: v02 supersedes v01 in full when she rules.

## 1. The decision this run informs

Whether a small base model can be post-trained to reproduce a 1966 rule system, and which of two task framings it learns: A, any script-valid reply to a sentence; B, the exact next reply given the conversation so far. Report both and the gap (K48). Calibration run and portfolio evidence card, not a result about training in general.

## 2. What is already fixed and measured

- Engine, data and split: `PROCEDURE.md` entries 9 and 10; `data/manifest.json` (seed 20261015; 3000 / 400 / 400 conversations).
- Models: the seven base checkpoints pinned by revision in `FLOOR_PROTOCOL.md`.
- Floor: measured 2026-10-16 (entry 21). A = 0.00% on all seven. B = 0.33% to 0.78%, of which 99 of 101 hits are copies of a line already in the conversation. The practical floor is zero on both framings.
- Training data boundary: no Claude-written text (entry 3).

## 3. Contamination probe (REPTME.E3), fixed before measurement

**Question.** How much of the DOCTOR script does each untrained base model already carry, and does it differ by family at matched size? Krystal Martinez's think-aloud of 2026-10-16 06:35 EDT (workbench K69) asks whether ELIZA material concentrated in code sources explains a family difference. The floor cannot answer it: its two non-copy hits are inside noise.

**Tool.** `tools/eliza_probe_contamination.py`, `probe-v1`. Nothing trains.

**Three measurements per model.**

| Measurement | What is scored | Fixed values |
|---|---|---|
| CACM recital | For each of the article's 15 exchanges (`tests/test_cacm.py`), greedy reply to the transcript-v1 prompt of the article's conversation up to that user line, compared to the article's reply after the floor's normalization | prompt format `transcript-v1`, greedy, 48 new tokens, first non-empty line; the article's own replies fill the history |
| Fixed-line margin | Every slot-free reply line of the script (reassembly rules with no constituent slot, NEWKEY redirects excluded) plus the four NOMATCH messages and the greeting: mean per-token NLL verbatim, word-shuffled (mean of 5 seeded shuffles), and lowercase. Margin = NLL(shuffled) − NLL(verbatim). Reported per line and as the median | bare context: BOS + line + newline; NLL in nats averaged over the line's tokens; seed 20261015 |
| Reference margin | The same statistic over 60 reply lines sampled (seed 20261015) from the test split whose text is not a fixed line: the script's templates filled with held-out filler words (or, for a few patterns, words from the script's own lists), which as whole lines cannot have been memorized | same scoring |

**Derived number.** Contamination signal = median fixed-line margin − median reference margin, in nats per token. The reference removes the part of the margin that template grammar alone explains.

**Reading rules, fixed now.**
- CACM recital above 0 of 15 on any model is direct evidence the article is in that model's training text; it is reported as such and the model is not dropped (the ladder is the ladder).
- The contamination signal is reported per model with the per-line file; the family comparison is at matched size (SmolLM2-1.7B against pythia-1b; SmolLM2-360M against pythia-410m; SmolLM2-135M against pythia-160m).
- No threshold is claimed for "contaminated"; a difference between families smaller than the spread across the three matched pairs is not reported as a family effect.
- Precision is recorded per run (fp16 on Kaggle's T4; fp32 if run on the laptop); the two are not pooled.

**Where it runs.** Kaggle, `notebooks/probe/eliza_kaggle_probe.ipynb`, pushed with the Kaggle CLI (`kaggle kernels push -p notebooks/probe`). Records: `runs/eliza_probe_runs.tsv` and the per-run `.lines.tsv`, `.cacm.tsv`, `.json`.

**Known limits.** One prompt format (REPTME.E4). Shuffling preserves the word multiset only. The bare-context NLL of a short line is dominated by its first tokens. Margins are comparable across models only as a difference from each model's own reference, never as raw nats.

## 4. Hypotheses — PENDING her ledger

Her saved entries (artifact ledger, collection `ledger`) are copied here verbatim at ratification, one per shape, each with its falsifier. Her think-aloud so far is recorded in the workbench (K68, K69) and is not a hypothesis until saved.

## 5. Arms — PENDING ratification

14 arms: 7 models × 2 framings, LoRA via `tools/eliza_train_adapter.py`, every value stated in a config file. Values to be fixed here before any arm runs: steps, learning rate, batch, sequence length, rank, alpha, targets, evaluation points, seeds (at least two per arm, for X).

## 6. Threshold X — PENDING

X = 2 × the between-seed spread on the untrained baseline, in percentage points, measured before any arm (Lesson 4). A difference between two arms under X is not reported as a change.

## 7. Stop rules — PENDING ratification

Each arm stops at its stated step count; no arm is extended because its curve "looks like it is still improving"; an arm whose loss does not fall below its step-1 loss by the first evaluation point is reported as such and not retuned inside this wave.
