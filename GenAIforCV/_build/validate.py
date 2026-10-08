#!/usr/bin/env python3
"""Check every chapter against the authoring contract. Run from the repo root."""
import os, re, glob, sys

REQUIRED = ["Why this lecture exists", "The ideas", "Worked numericals", "Code",
            "Exam pack", "Beyond the slides", "Cut from the slides"]
EXAM_SUBS = ["Must-memorise", "Numbers worth knowing", "Likely MCQ traps", "Self-test"]

def unquote(p):
    return p.replace('%20', ' ')

issues, stats = [], []
files = sorted(glob.glob('notes/week-*/*.md')) + sorted(glob.glob('notes/*.md'))

for f in files:
    txt = open(f, encoding='utf8').read()
    d = os.path.dirname(f)
    name = os.path.basename(f)
    def bad(msg): issues.append(f"{name}: {msg}")

    # --- section structure
    h2 = re.findall(r'^## (.+)$', txt, re.M)
    h2 = ["Why this lecture exists" if x == "Why this chapter exists" else x for x in h2]
    missing = [s for s in REQUIRED if s not in h2]
    if missing: bad(f"missing H2 section(s): {missing}")
    extra = [s for s in h2 if s not in REQUIRED]
    if extra: bad(f"unexpected H2 section(s): {extra}")
    present = [s for s in h2 if s in REQUIRED]
    if present != [s for s in REQUIRED if s in present]:
        bad("H2 sections out of contract order")

    # --- exam pack subsections
    if "Exam pack" in h2:
        pack = txt.split('## Exam pack')[1].split('\n## ')[0]
        for sub in EXAM_SUBS:
            if sub not in pack: bad(f"Exam pack missing '{sub}'")
        if '<details>' not in pack: bad("Self-test answers not in a <details> block")

    # --- math delimiters (prose only; \( is legal inside Python regexes)
    prose = re.sub(r'```.*?```', '', txt, flags=re.S)
    prose = re.sub(r'`[^`\n]*`', '', prose)
    if re.search(r'\\\(|\\\[', prose): bad("uses forbidden \\( or \\[ math delimiters")

    # --- front matter
    if not re.search(r'^> \*\*(Source|Deck):\*\*', txt, re.M):
        bad("front matter missing **Source:** / **Deck:**")

    # --- image links resolve
    imgs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', txt)
    for src in imgs:
        if src.startswith('http'): bad(f"external image URL not allowed: {src}")
        elif not os.path.isfile(os.path.normpath(os.path.join(d, unquote(src)))):
            bad(f"broken image: {src}")

    # --- internal chapter links resolve
    for tgt in re.findall(r'\]\((\.{1,2}/[^)#]+\.md)\)', txt):
        if not os.path.isfile(os.path.normpath(os.path.join(d, unquote(tgt)))):
            bad(f"broken chapter link: {tgt}")
    for tgt in re.findall(r'\]\((\d\d-[^)#]+\.md)\)', txt):
        if not os.path.isfile(os.path.join(d, tgt)):
            bad(f"broken sibling link: {tgt}")

    # --- figures with no caption or no alt text
    for m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', txt):
        if len(m.group(1)) < 15: bad(f"thin alt text on {os.path.basename(m.group(2))}")

    # --- markdown table column consistency
    for block in re.findall(r'(?:^\|.*\|$\n)+', txt, re.M):
        rows = [r for r in block.strip().split('\n')]
        widths = {len(re.findall(r'(?<!\\)\|', r)) for r in rows}
        if len(widths) > 1:
            bad(f"ragged table near: {rows[0][:55]}")

    stats.append((name, len(txt.split()), len(imgs),
                  len(re.findall(r'^### N\d', txt, re.M)),
                  txt.count('```python')))

print(f"{'chapter':<46}{'words':>7}{'figs':>6}{'nums':>6}{'code':>6}")
print('-' * 71)
for n, w, i, num, c in stats:
    print(f"{n:<46}{w:>7}{i:>6}{num:>6}{c:>6}")
print('-' * 71)
print(f"{'TOTAL ' + str(len(stats)) + ' chapters':<46}"
      f"{sum(s[1] for s in stats):>7}{sum(s[2] for s in stats):>6}"
      f"{sum(s[3] for s in stats):>6}{sum(s[4] for s in stats):>6}")

print(f"\n{len(issues)} issue(s)")
for i in issues: print("  !", i)
sys.exit(1 if issues else 0)
