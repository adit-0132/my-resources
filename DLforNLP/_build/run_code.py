#!/usr/bin/env python3
"""Execute every chapter's python blocks (concatenated per file, as a reader would)."""
import re, glob, subprocess, sys, os
ok = fail = skipped = 0; fails = []
for f in sorted(glob.glob('notes/**/*.md', recursive=True)):
    blocks = re.findall(r'```python\n(.*?)```', open(f, encoding='utf8').read(), re.S)
    if not blocks: continue
    src = '\n'.join(blocks)
    if 'torch' in src:
        try: import torch  # noqa
        except ImportError: skipped += 1; continue
    r = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True, timeout=300)
    if r.returncode == 0: ok += 1
    else:
        fail += 1; fails.append((f, r.stderr.strip().split('\n')[-1][:120]))
print(f"chapters whose code runs: {ok}   failing: {fail}   skipped (no torch): {skipped}")
for f, e in fails: print(f"  FAIL {f}: {e}")
sys.exit(1 if fail else 0)
