# PRESPEC — Wave 01 (ELIZA), DRAFT v03, for ratification

**Status:** complete draft for Krystal Martinez's ratification (K55). v01 (section 3 only) is retired unedited to `deprecated/PRESPEC_v01_DRAFT_2026-10-16.md`; v02 (same content as v03 with the hypotheses quoted from speech) is withdrawn from this repository at her ruling of 2026-10-16 that public artifacts carry her statements in edited form. Her hypotheses are in section 4, stated in her formulation and edited for print; every falsifier, every number in sections 5 to 7, and the reporting rules in section 8 are the seat's proposals and change nothing until she rules. No training arm runs before her ratification.

Drafted 2026-10-16 by Flaudechamba Flaudicota A.-L. Formative Horizon v01 (Claude Fable 5.1). A ruling that changes a value produces v03; this file is not edited in place after ratification.

## 1. The decision this run informs

Whether a small base model can be post-trained to reproduce a 1966 rule system, and which of two task framings it learns: A, any script-valid reply to a sentence; B, the exact next reply given the conversation so far. Report both and the gap (K48). Calibration run and portfolio evidence card, not a result about training in general.

## 2. What is already fixed and measured

- Engine, data and split: `PROCEDURE.md` entries 9 and 10; `data/manifest.json` (seed 20261015; 3000 / 400 / 400 conversations; 20,451 / 2,591 / 2,695 turns).
- Models: the seven base checkpoints pinned by revision in `FLOOR_PROTOCOL.md`.
- Floor (entry 21): A = 0.00% on all seven. B = 0.33% to 0.78%, 99 of 101 hits copies of a line already in the conversation. Practical floor: zero on both framings.
- Contamination probe (entry 25, section 3 below): no model recites the article; no family effect. Capitals cost 0.9 to 1.8 nats per token over lowercase on every model.
- Training-data boundary: no Claude-written text (entry 3).

## 3. Contamination probe (REPTME.E3) — measured

Specification as fixed in v01 (unchanged): `tools/eliza_probe_contamination.py`, `probe-v1`; CACM recital over the article's 15 exchanges; fixed-line margin over the script's 87 slot-free reply lines against 5 seeded word-shuffles and a lowercase form; reference margin over 60 filled reply lines from the test split; contamination signal = difference of medians; comparison at matched size; no threshold for "contaminated". Measured 2026-10-16 (entry 25): recital 0 of 15 on all seven; signals −0.08 (70m), +0.06 (135M), +0.00 (160m), +0.13 (360M), +0.18 (410m), +0.22 (1b), +0.33 (1.7B); matched pairs split 2 to 1 on family, so no family effect is reported. Known limits as in v01.

## 4. Her hypotheses — her formulation, with the seat's proposed falsifiers

Stated by Krystal Martinez on 2026-10-16 (06:14 and 07:16 EDT) with the floor and the contamination probe as her known exposure and no trained result in existence. The statements below are hers, edited for print; the source record is the private workbench decision log (K68, K72) and the artifact ledger. Exposure when written: the floor for all four; the probe result for H3 and H4.

**H1 — templates before tracking.** Every model will learn the templates (framing A) faster, and to a higher level, than it learns to track the conversation and the count of templates used (framing B). Her reason: the framing B floor is in-context copying, not knowledge of the script, which is closer to template use than to conversation tracking.

Proposed test: for every arm, the A score exceeds the B score at every evaluation point, and at the final point by more than X. Falsified if any model's B score is within X of, or above, its A score at the final point, or if B reaches its final-point value at an earlier evaluation point than A does.

**H2 — copying grows with size, counting follows.** Within each family, the number of copying instances in the floor grows with size; trained ability to copy the templates will therefore grow with size, and the ability to count the templates is expected to follow. Her reading of pythia-410m: an outlier at this sample size that would settle between 160m and 1b on repetition.

Proposed test: within each family, the final A score and the final B score each increase from the smallest to the largest size by more than X. Falsified in a family if its largest size is within X of its smallest on either framing, or below it.

**H3 — SmolLM2 outperforms Pythia, more so as size grows.** In general the SmolLM2 family will outperform the Pythia family, especially when the families are compared at their larger sizes.

Proposed test: at each matched pair (135M vs 160m, 360M vs 410m, 1.7B vs 1b), SmolLM2's final B score exceeds Pythia's by more than X, and the SmolLM2-minus-Pythia difference is larger at the top pair than at the bottom pair. Falsified if Pythia is within X of, or above, SmolLM2 at the top pair, or if the difference does not grow from the bottom pair to the top pair.

**H4 — first-to-last differences within each family are statistically significant.** For each family, the difference between the smallest and the largest model's measurements will be statistically significant, and more so for SmolLM2.

Proposed test: for each family and each framing, the difference between the largest and the smallest size's final score, paired by conversation (the unit, REPTME.E5; Miller 2024), has a 95% interval that excludes zero, and the difference exceeds X. Falsified in a family if either framing's interval includes zero or the difference is under X. "More so for SmolLM2" is reported as a comparison of the two families' first-to-last differences, not tested.

**Abstention.** On every other question she has no data to lean one way or the other, abstains, and stays open to whatever emerges. Recorded so that the absence of a hypothesis on the floor (K59), on the gap's direction beyond H1, and on the smallest size that tracks state is a decision, not an omission. Anything that emerges outside H1 to H4 is reported as exploratory.

## 5. Arms — proposed values

14 arms: 7 models × 2 framings, each at 2 seeds (28 training runs). LoRA via `tools/eliza_train_adapter.py`; one config file per run under `configs/`, every value stated.

| Value | Proposed | Why |
|---|---|---|
| steps | 2,560 (two passes over the 20,451 training turns at batch 16) | a rule system with 55 patterns needs every pattern seen more than once; two epochs, not more, so the arms stay cheap enough to run twice |
| batch_size × grad_accum | 16 × 1 (8 × 2 on the 1.7B if memory requires) | effective batch 16 everywhere |
| learning_rate | 5e-4, constant | the smoke test's rate; loss fell from 3.71 to 2.43 in 30 steps |
| max_seq_len | 256 (A), 512 (B) | B prompts carry history; test-split B prompts fit in 512 tokens |
| lora_r / lora_alpha / dropout | 16 / 32 / 0.0 | double the smoke test's rank, so capacity is not the bottleneck at 70M |
| target_modules | Pythia: query_key_value, dense, dense_h_to_4h, dense_4h_to_h; SmolLM2: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | all attention and MLP projections in both families, so the families are matched on what is tuned |
| seeds | 1 and 2 | two per arm, for X and the intervals |
| eval_points | 0, 0.25, 0.5, 0.75, 1.0 | five points per run, on 400 dev items; the final score on the full dev split |
| final_eval_split | dev during the run; test scored once per arm, after all arms finish | the test split is read once |
| device / precision | Kaggle T4, fp16 | as the floor |

Nothing is tuned per model. If an arm cannot run at these values (out of memory, divergence), the change is recorded as a PROCEDURE entry before the arm is rerun, and applies to that model's both framings and both seeds.

## 6. Threshold X — proposed measurement

X = 2 × the between-seed spread, in percentage points, measured on the dev split as the mean absolute difference between seed 1 and seed 2 over the two smallest arms (pythia-70m A and SmolLM2-135M A), which run first; X is fixed before any other arm is scored. The floor itself cannot supply X: greedy decoding has no seed variance (the fp16/fp32 cross-check differed by 4 items in 2,695, 0.15 points). A difference between two arms under X is not reported as a change. X is reported alongside every comparison.

## 7. Stop rules — proposed

Each run stops at its stated step count. No run is extended because its curve looks like it is still improving. A run whose loss at the 0.25 evaluation point is not below its step-1 loss is reported as such and not retuned inside this wave. A run that fails (out of memory, crash) is rerun once at the same values after the cause is recorded; if it fails again, the model is reported as not trainable at these values and its arms are left blank, not filled from a different setting.

## 8. What is reported, and in what order

1. The four curves (two families × two framings) on the test split, with X and per-conversation 95% intervals.
2. H1 to H4, each as decided by its section 4 test, pass or fail, with the numbers.
3. The gap (A minus B) by size and family, exploratory beyond H1.
4. The per-keyword breakdown of B errors, to see whether trained models lean on the copying mechanism the floor showed.
5. Anything else that emerged, labelled exploratory.

Every number traces to a row in `runs/eliza_runs.tsv` and a config under `configs/`.
