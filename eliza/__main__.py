"""Run the engine on stdin, one user line per input line, printing one reply per line.

    python -m eliza [--script PATH] [--hello] < lines.txt

Mirrors Hay's binary in --nobanner mode so the two can be diffed line for line.
"""
import argparse
import os
import sys

from . import Eliza, load_script

DEFAULT_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "third-party",
                              "ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=DEFAULT_SCRIPT)
    ap.add_argument("--hello", action="store_true", help="print the greeting first")
    args = ap.parse_args()
    e = Eliza(load_script(args.script))
    if args.hello:
        print(e.hello())
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == "":
            break
        print(e.response(line))


if __name__ == "__main__":
    main()
