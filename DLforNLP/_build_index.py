import re, glob, os, json
SRC='/home/adi/Projects/ideas/DL for NLP'
idx=[]
for tf in sorted(glob.glob('extracted/*.txt')):
    txt=open(tf).read()
    pdf=None
    base=os.path.basename(tf)[:-4]
    for cand in os.listdir(SRC):
        if cand.endswith('.pdf') and re.sub(r'[^\w]+','_',cand[:-4])==base: pdf=cand
    pages=re.split(r'--- page (\d+) ---', txt)
    # pages = ['', '1', body, '2', body, ...]
    pg={}
    for i in range(1,len(pages),2): pg[int(pages[i])]=pages[i+1]
    marks=[]
    for p in sorted(pg):
        m=re.search(r'^\s*Lecture\s*(\d+)\s*:\s*(.*)$', pg[p], re.M)
        if m: marks.append((p,int(m.group(1)),m.group(2).strip()))
    total=max(pg)
    for i,(p,num,title) in enumerate(marks):
        end = marks[i+1][0]-1 if i+1<len(marks) else total
        # title may wrap to the next line
        body=pg[p]
        cont=re.search(r'^\s*Lecture\s*\d+\s*:\s*(.*)\n(.*)$', body, re.M)
        if cont and len(title)<28 and cont.group(2).strip() and not cont.group(2).strip().startswith('Lecture'):
            extra=cont.group(2).strip()
            if len(extra)<40 and extra[0].isupper() or extra.startswith(('Part','Classification','Generation','Encoder','Pruning','Positional','Optimization')):
                title=(title+' '+extra).strip()
        idx.append(dict(lec=num,title=title,pdf=pdf,start=p,end=end,pages=end-p+1))
idx.sort(key=lambda r:r['lec'])
json.dump(idx,open('_build_index.json','w'),indent=1)
print(f"{len(idx)} lectures indexed, {sum(r['pages'] for r in idx)} pages covered")
for r in idx: print(f"  Lec {r['lec']:>2}  p{r['start']:>3}-{r['end']:<3} ({r['pages']:>2}pp)  {r['title'][:62]}")
