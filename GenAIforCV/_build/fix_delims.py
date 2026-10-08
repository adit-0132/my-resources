#!/usr/bin/env python3
"""Convert forbidden \\( \\) and \\[ \\] math delimiters to $ and $$."""
import re, glob
total = 0
for f in sorted(glob.glob('notes/**/*.md', recursive=True)):
    s = open(f, encoding='utf8').read(); o = s
    # protect fenced code blocks — converting \( inside code corrupts regexes
    blocks = []
    def stash(m):
        blocks.append(m.group(0)); return f"\x00BLOCK{len(blocks)-1}\x00"
    s = re.sub(r'```.*?```', stash, s, flags=re.S)
    s = re.sub(r'`[^`\n]*`', stash, s)
    s = re.sub(r'\\\[\s*(.*?)\s*\\\]', lambda m: '$$' + m.group(1) + '$$', s, flags=re.S)
    s = re.sub(r'\\\((.*?)\\\)', lambda m: '$' + m.group(1) + '$', s, flags=re.S)
    s = re.sub(r'\x00BLOCK(\d+)\x00', lambda m: blocks[int(m.group(1))], s)
    if s != o:
        n = len(re.findall(r'\\\(|\\\[', o))
        open(f, 'w', encoding='utf8').write(s); print(f"  {f}: {n} converted"); total += n
print(f"converted {total} delimiters")
