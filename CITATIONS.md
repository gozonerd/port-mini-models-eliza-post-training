# CITATIONS — for whoever builds the portfolio page

Everything outside this project that it uses, with the line to cite and the notice to reproduce. Read `third-party/SOURCES.tsv` for the files and hashes.

## ELIZA and the DOCTOR script

**The 1966 paper (cite; do not reproduce the article's text):**
Weizenbaum, J. (1966). ELIZA — A computer program for the study of natural language communication between man and machine. *Communications of the ACM, 9*(1), 36–45. https://doi.org/10.1145/365153.365168

**The 1965 source listing, with the DOCTOR script (the licensed source):**
Weizenbaum, J. (1965). *Computer conversations, 1965* [MAD-SLIP source listing of ELIZA with the DOCTOR script]. MIT Libraries, Department of Distinctive Collections. https://hdl.handle.net/1721.3/201699 — "Licensed under an MIT software license." Notice to reproduce: **Copyright 1965 MIT**, with the MIT permission text in `third-party/MIT-Libraries_Weizenbaum_1965_LICENSE.txt`.

**The restoration (cite for the history and the permission):**
Lane, R., Hay, A., Schwarz, A., Berry, D. M., & Shrager, J. (2025). *ELIZA Reanimated: The world's first chatbot restored on the world's first time sharing system.* arXiv:2501.06707. https://arxiv.org/abs/2501.06707 — reports that the Weizenbaum estate granted permission to open-source the recovered code.

**The transcriptions and reference implementation used to check this project's engine:**
Hay, A. (2020–2026). *ELIZA* [C++ reimplementation of 1966 CACM ELIZA; verbatim transcriptions of the 1966 appendix script and the 1965 listing]. https://github.com/anthay/ELIZA — CC0 1.0 (public domain dedication). No notice is required; credit him anyway.

**A nuance to state honestly on the site:** the script this project trains on is the 1966 appendix version (66 keywords), transcribed by Hay. The MIT licence covers the 1965 listing's script (35 keywords); the 1966 article is ACM-copyrighted. The script is data, not article prose, it was open-sourced with the estate's permission, and it has been reimplemented publicly for decades, so the project uses it with the citation above. Say that, rather than claiming a licence the article does not carry.

## Models

**SmolLM2** — Allal, L. B., et al. (2025). *SmolLM2: When Smol goes big — Data-centric training of a small language model.* arXiv:2502.02737. Weights: Hugging Face `HuggingFaceTB/SmolLM2-{135M,360M,1.7B}`, model card `license: apache-2.0` (read 2026-10-15).

**Pythia** — Biderman, S., et al. (2023). *Pythia: A suite for analyzing large language models across training and scaling.* ICML 2023. arXiv:2304.01373. Weights: Hugging Face `EleutherAI/pythia-{70m,160m,410m,1b}`, model card `license: apache-2.0` (read 2026-10-15).

Apache 2.0 text: `third-party/Apache-2.0.txt`. Derived adapters published from this project must carry the Apache 2.0 notice for the base model.

## Word list used for filler words in the generated data

**SCOWL** — Atkinson, K. *Spell Checker Oriented Word Lists*, as shipped in Debian's `wamerican` package (`/usr/share/dict/american-english`). Permissive licence (the SCOWL notice: permission to use, copy, modify, distribute and sell, provided the copyright notice appears). Hash in `data/manifest.json` and `third-party/SOURCES.tsv`. The notice to reproduce: "Copyright 2000-2011 by Kevin Atkinson".

## Software

To be added as each is used: PyTorch, Transformers, PEFT, TRL, Inspect — name, version, licence, at the time of the run.
