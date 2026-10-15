# PROCEDURE — what was done, in order

Append-only. One dated entry per step, written when the step is taken. Entries name the person or persona who did the step and where its evidence sits. Decisions behind the steps are in the private workbench's decision logs (K = Krystal Martinez's rulings, C = the seat's choices); the entry cites the id.

## 2026-10-15

1. **Scope set** (K48). Wave 01 = ELIZA, both task framings, report the gap. Purpose: calibration run and portfolio evidence card.
2. **Model ladder chosen** (C74, K49). SmolLM2 base 135M / 360M / 1.7B and Pythia 70M / 160M / 410M / 1B, each family reported as its own full curve. Rejected: Qwen3 and Qwen3.5 (no member under 0.6B), Gemma 4 (no member under about 2B), Llama 3.2 (licence propagates naming and use policy), W&B Training (supports 8B+ only).
3. **Training-data boundary** (K47). No Claude-written text is trained on: Anthropic's support replied 2026-10-15 that no programme authorizes training outside models on Claude outputs. This wave needs none: its data comes from the 1966 rule engine.
4. **Source licences established** (K53). The 1965 MAD-SLIP listing with the DOCTOR script is published by MIT Libraries under the MIT licence (`third-party/MIT-Libraries_Weizenbaum_1965_LICENSE.txt`, "Copyright 1965 MIT"). Anthony Hay's transcription of the 1966 CACM appendix script, and his C++ reimplementation checked against the rediscovered original, are CC0. Files and hashes: `third-party/SOURCES.tsv`.
5. **Repo created**: `gozonerd/port-mini-models-eliza-post-training`, public, at Krystal Martinez's instruction (K53). First commit: this record, the sources, the citations file.

Next: write the rule engine as a Python port of the 1966 script; verify it against Hay's implementation on the "Men are all alike" conversation from the CACM article; then generate the data for both framings.
6. **Learning-experience handoff** (K54). Krystal Martinez's ChatGPT collaborator (Astrichamba) is designing a learning experience on Claude training and asked for a memo reconstructing this plan and recommending a capstone. The memo (workbench, `planning/mini-models 2026-10-15/Flaudechamba_to_Astrichamba_…_2026-10-15_v01.md`) recommends the three-wave sequence as the capstone core with this wave as its first evidence, and asks that her predictions for Wave 01 be written down before any training run. Engine work paused for the memo; resumes after.
