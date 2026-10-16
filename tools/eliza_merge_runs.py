"""Merge a downloaded Kaggle results zip into runs/: rows into eliza_runs.tsv (skipping run ids already
present), per-item files and run JSONs copied beside the laptop's.

    python3 tools/eliza_merge_runs.py ~/Downloads/eliza_floor_runs.zip
"""
import os
import shutil
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def main() -> None:
    zpath = sys.argv[1]
    os.makedirs(RUNS, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        src = os.path.join(tmp, "runs")
        if not os.path.isdir(src):
            sys.exit("no runs/ folder in the zip")
        have = set()
        idx = os.path.join(RUNS, "eliza_runs.tsv")
        if os.path.exists(idx):
            for line in open(idx, encoding="utf-8"):
                have.add(line.split("\t")[0])
        added = 0
        incoming = os.path.join(src, "eliza_runs.tsv")
        header_needed = not os.path.exists(idx)
        with open(idx, "a", encoding="utf-8") as out:
            for i, line in enumerate(open(incoming, encoding="utf-8")):
                if i == 0:
                    if header_needed:
                        out.write(line)
                    continue
                if line.split("\t")[0] in have:
                    continue
                out.write(line)
                added += 1
        copied = 0
        for name in os.listdir(src):
            if name == "eliza_runs.tsv":
                continue
            dst = os.path.join(RUNS, name)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src, name), dst)
                copied += 1
        print(f"merged {added} run rows and {copied} files from {os.path.basename(zpath)}")


if __name__ == "__main__":
    main()
