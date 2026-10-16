# Floor protocol — fixed before any model is scored

Ruling K59 (2026-10-15): the zero-shot floor is measured on all seven base models, both framings,
before any hypothesis is written, and no hypothesis is made about the floor. These are the harness
choices for that measurement, written down first so the numbers cannot rewrite them.

| Choice | Fixed value | Why |
|---|---|---|
| Models | SmolLM2 135M `93efa2f`, 360M `f8027fd`, 1.7B `effd688`; Pythia 70m `a39f36b`, 160m `50f5173`, 410m `9879c9b`, 1b `f73d7dc` (base checkpoints, revisions pinned 2026-10-15) | the ladder (C74, K49) |
| Data | `data/test.A.jsonl` and `data/test.B.jsonl`, seed 20261015, as hashed in `data/manifest.json` | the held-out set the trained arms will be scored on |
| Prompt format | `transcript-v1`: the script's greeting line, then alternating user line and reply line (replies in capitals), ending with the user line and a newline | the same cue for every model; the greeting makes contamination visible if a model recites ELIZA |
| Decoding | greedy, at most 48 new tokens; the reply is the first non-empty generated line (a model may open with a newline) | deterministic; one trial per item |
| Scoring A | the generated line, uppercased and whitespace-collapsed, is in the allowed set | framing A's definition |
| Scoring B | the generated line equals the exact next reply after the same normalization | framing B's definition |
| Unit | the item for the score; the conversation for any interval (later) | REPTME.E5 |
| Precision | fp32 on the laptop processor; fp16 if a run moves to a GPU, recorded per run | hardware |
| Record | `runs/runs.tsv` row per (model, framing); `runs/<run-id>.items.tsv` per item | REPTME.E8 traceability |

What the floor is for: the contamination measurement (REPTME.E3) and the reference every trained arm's
lift is measured from. It is reported per model and per framing. It is not a hypothesis and it is not
a result about training.

Known limit: the prompt format is one harness choice among many (REPTME.E4). The floor under a
different format would be a different number; the format is held fixed across all arms so that it
cancels in comparisons.
