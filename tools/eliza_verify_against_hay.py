"""Diff this port against Anthony Hay's reference binary on generated conversations.

The sentences here exist only to exercise the engine (keywords, delimiters, memory, cycles,
long words, punctuation). They are NOT training data and never leave this check.

    g++ -O2 -std=c++17 -o tools/hay-eliza-bin third-party/anthay-ELIZA/eliza.cpp
    python3 tools/eliza_verify_against_hay.py [--n 400] [--seed 20261015]
"""
import argparse, random, re, subprocess, sys, os, hashlib, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, 'third-party', 'ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt')
HAY = os.path.join(HERE, 'hay-eliza-bin')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args()
    random.seed(a.seed)
    text = open(SCRIPT, encoding='utf-8').read()
    words = sorted(set(w for w in re.findall(r"[A-Z'][A-Z'\-]*", text)
                       if len(w) > 1 and w not in ('DLIST', 'NEWKEY', 'PRE', 'MEMORY', 'NONE', 'START')))
    extra = ['the', 'a', 'and', 'to', 'of', 'in', 'is', 'it', 'that', 'was', 'for', 'on', 'with', 'he', 'she',
             'they', 'we', 'me', 'my', 'you', 'your', 'i', 'always', 'never', 'everyone', 'nobody', 'computers',
             'dreamed', 'sorry', 'perhaps', 'because', 'why', 'what', 'how', 'yes', 'no', "can't", "don't",
             'mother', 'father', 'sister', 'children', 'wife', 'husband', 'extraordinarily', 'grandchildren']
    vocab = [w.lower() for w in words] + extra

    def sentence():
        s = [random.choice(vocab) for _ in range(random.randint(1, 12))]
        if random.random() < 0.3:
            s.insert(random.randrange(len(s) + 1), random.choice([',', '.', 'but']))
        out = ' '.join(s)
        if random.random() < 0.5:
            out = out[0].upper() + out[1:]
        if random.random() < 0.6:
            out += random.choice(['.', '?', '!', ''])
        return out

    convs = [[sentence() for _ in range(random.randint(1, 14))] for _ in range(a.n)]
    mism = total = 0
    for c in convs:
        inp = '\n'.join(c) + '\n\n'
        mine = subprocess.run([sys.executable, '-m', 'eliza'], input=inp, capture_output=True, text=True, cwd=ROOT).stdout.rstrip('\n').split('\n')
        hay = subprocess.run([HAY, '--nobanner'], input=inp, capture_output=True, text=True).stdout
        hay = [l for l in hay.split('\n') if l != ''][1:]
        total += len(c)
        if mine != hay:
            mism += 1
            if mism <= 3:
                print('MISMATCH:')
                for u, m, h in zip(c, mine + [''] * 20, hay + [''] * 20):
                    print('  U:', u, '\n  mine:', m, '\n  hay: ', h, '' if m == h else '   <<<')
    eng = hashlib.sha256(open(os.path.join(ROOT, 'eliza', 'eliza_engine.py'), 'rb').read()).hexdigest()[:12]
    print(f'{datetime.datetime.utcnow().isoformat()}Z seed={a.seed} conversations={len(convs)} turns={total} '
          f'mismatching_conversations={mism} engine_sha256={eng}')
    sys.exit(1 if mism else 0)


if __name__ == '__main__':
    main()
