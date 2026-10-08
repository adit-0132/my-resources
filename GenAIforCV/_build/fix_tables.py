#!/usr/bin/env python3
"""Repair markdown tables: bare | inside $...$ cells, and missing row terminators."""
import sys, re, glob
total = 0
for f in sorted(glob.glob('notes/**/*.md', recursive=True)):
    lines = open(f, encoding='utf8').read().split('\n'); n = 0
    for i, l in enumerate(lines):
        if not l.startswith('|'): continue
        o = l
        l = l.replace(r'\left|', r'\left\lvert ').replace(r'\right|', r'\right\rvert ')
        l = l.replace(r'\|', '\x00')
        l = re.sub(r'\$([^$]*)\$', lambda m: '$' + m.group(1).replace('|', r'\mid ') + '$', l)
        # bare | inside a backtick code span breaks the table; use the HTML entity
        l = re.sub(r'`([^`]*)`', lambda m: '`' + m.group(1).replace('|', '&#124;') + '`', l)
        l = l.replace('\x00', r'\|')
        if not l.rstrip().endswith('|'): l = l.rstrip() + ' |'
        if l != o: lines[i] = l; n += 1
    if n:
        open(f, 'w', encoding='utf8').write('\n'.join(lines)); print(f"  {f}: {n} rows"); total += n
print(f"repaired {total} table rows")
