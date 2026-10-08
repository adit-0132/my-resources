# Note-Writing Contract — binding on every author

You are writing **one chapter of one book**, not a standalone document. Other authors are writing the
chapter before yours and the chapter after yours, right now, from this same contract. Uniformity is
not a nicety here; it is the deliverable.

Read this whole file before writing a word.

---

## 1. The reader

One specific person. Calibrate everything to them:

- Can use `sklearn` — `fit`, `predict`, `train_test_split`. Knows what overfitting *is* as a word.
- Has **not** done: matrix calculus, backprop by hand, any PyTorch, probability beyond "P(A and B)".
- **Has already studied a companion course** on Generative AI for Computer Vision and has complete
  notes for it (see §5). They know CNNs, backprop, RNN/LSTM, and the Transformer from a *vision*
  angle. Build on that, don't assume it's absent — but see the overlap rule, because the exam is set
  from *these* slides.
- Is sitting an **NPTEL end-term exam** and wants 95+. The exam is MCQ + short numerical. **Exact
  definitions, exact formulas, and the ability to compute small numbers by hand** matter more than vibes.
- Is reading these notes *instead of* watching the lectures. If you leave something out, they never
  see it.

Write to that person. Not to a grad student, and not like a slide deck.

## 2. Non-negotiable file shape

Exactly these seven H2 sections, exactly these names, in exactly this order. No extras, none omitted.
(If a section would be genuinely empty, write one line saying so and why. Do not delete the heading.)

```markdown
# Lec NN — <Title>

> **Source:** `<pdf filename>` pp. <start>–<end> · **Week N** · **Playlist:** Lec NN
> **Prereqs:** [Lec MM — Title](../week-0M/MM-slug.md), ...   (or "none")
> **Feeds into:** [Lec PP — Title](../week-0P/PP-slug.md), ...

## Why this lecture exists

## The ideas

## Worked numericals

## Code

## Exam pack

## Beyond the slides

## Cut from the slides
```

### What goes in each

**Why this lecture exists** — 100–150 words of prose. The *problem* this lecture solves and why it
comes after the previous one. No bullet lists.

**The ideas** — the body and the bulk of the file. Free use of `###`. Every concept on the slides,
taught properly: what it is, why it works, where it breaks. Derive the math, don't assert it. Embed
figures (§4).

**Worked numericals** — 3–6 fully worked examples of the exact kind the exam asks. Every arithmetic
step shown, landing on a concrete number.

> **THE SINGLE MOST IMPORTANT RULE IN THIS CONTRACT.** These slides contain **"Try this problem"**
> pages — the lecturer's own exercises, often with a worked solution on the following page. **Every
> one of them in your page range must appear here, worked in full.** They are the closest thing to
> the actual exam questions that exists. Find them, solve them, show the arithmetic, and say which
> page each came from. If the deck gives a solution, check your answer against it and flag any
> disagreement rather than silently following either.

Then add your own numericals on top until you have 3–6 total. Format each as:

```markdown
### N1. <What is being computed>
**Given:** ...
**Find:** ...
<numbered steps, each showing the arithmetic>
**Answer:** <final line>
```

**Code** — runnable NumPy or PyTorch making an idea concrete. 10–40 lines per block, commented, with
the real printed output shown beneath. Prefer NumPy for mechanism, PyTorch/`transformers`-style
pseudocode only where the real API is the point. No `...` placeholders — it must actually run. If a
lecture is pure concept, say so in one line and skip.

**Exam pack** — four labelled subsections, in this order:

```markdown
### Must-memorise
<table: | Item | Exactly this |>

### Numbers worth knowing
<table of specific figures an MCQ can key on: dataset sizes, model params, years,
 benchmark scores, hyperparameter defaults>

### Likely MCQ traps
<bulleted; each states the confusion AND the correct discrimination>

### Self-test
<6–10 questions, answers inline in <details><summary>Answers</summary> ... </details>>
```

**Beyond the slides** — 2–5 items the deck omits that the reader genuinely needs, each a short
`**Gap:**` / `**Why it matters:**` pair. Real gaps, not adjacent showing-off.

**Cut from the slides** — one paragraph naming what you compressed or dropped and why. The audit trail.

## 3. Notation — identical across all 60 chapters

| Thing | Write | Not |
|---|---|---|
| scalar | $x$, $\eta$ | |
| vector | $\mathbf{x}$, $\mathbf{h}$ | $\vec{x}$ |
| matrix | $\mathbf{W}$, $\mathbf{A}$ | $W$ |
| layer index | $\mathbf{W}^{(l)}$, $a_j^{(l)}$ | $W_l$ |
| time / position | $\mathbf{h}_t$, $w_t$ | $h(t)$ |
| loss | $\mathcal{L}$ | $L$, $J$, $E$ |
| dataset | $\mathcal{D}$ | $D$ |
| expectation | $\mathbb{E}_{p(x)}[\cdot]$ | $E[\cdot]$ |
| learning rate | $\eta$ | $\alpha$ |
| params (generic) | $\theta$ | $\phi$, $\Phi$ |
| vocabulary / its size | $V$ / $\lvert V\rvert$ | |
| sequence length | $T$ (tokens), $n$ (generic) | |
| token / word | $w_t$; embedding $\mathbf{e}_t$ | |
| context window | $c$ | |
| attention | $\mathbf{Q},\mathbf{K},\mathbf{V}$; $d_k,d_v,d_{\text{model}}$; $h$ heads | |
| attention weights | $\alpha_{ij}$ | $a_{ij}$ |
| LSTM cell state | $\mathbf{C}_t$, candidate $\tilde{\mathbf{C}}_t$ | |
| attention context | $\mathbf{c}_t$ — **reserved**, never the cell state | |
| KL | $D_{\mathrm{KL}}(q\,\|\,p)$ | $KL(q,p)$ |
| policy (RL) | $\pi_\theta$; reference $\pi_{\text{ref}}$; reward $r$ | |
| LoRA | rank $r$, $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$, scaling $\alpha$ | |

**Week-2 warning:** Lectures 6–10 follow Prince's *Understanding Deep Learning*, which writes
parameters as $\boldsymbol{\phi}$ and the model as $f[\mathbf{x}, \boldsymbol{\phi}]$. **Translate to
$\theta$** per the table and note the deck's symbol once.

Math delimiters: `$...$` inline, `$$...$$` display on its own lines. **Never** `\(...\)` or `\[...\]`.

## 4. Figures

**Pool A — embedded figures** (`assets/figures/lecNN/f-*.png`): images lifted out of the PDF —
charts, paper screenshots, diagrams. Native resolution.

**Pool B — rendered pages** (`assets/pages/lecNN/p-NN.png`): the whole slide as shown. Use when the
content *is* the page — a built-up equation, an annotated diagram, a table.

Paths are relative to your note's directory (`notes/week-NN/`), so prefix `../../`.

```markdown
![BPE merge table: the five most frequent adjacent pairs and the resulting vocabulary](../../assets/pages/lec02/p-16.png)
*Fig. — Note the merge order is frequency-driven, not alphabetical; this is what makes BPE deterministic given a corpus. Page 40 of Week1.pdf.*
```

Rules:
- **You must open and look at the rendered pages** (`Read` the PNGs) before writing. The extracted
  text is a flat dump with no layout: tables, equations built across a slide, and figure labels are
  unreadable in it. A chapter written from the text dump alone will have holes.
- Pool B page files are numbered by their position **within the source PDF**, not within your
  lecture — `assets/pages/lec02/p-26.png` is PDF page 26. Your front matter gives your page range.
- Every figure gets descriptive alt text and an italic caption saying *what to notice*, ending with
  the page number.
- 4–12 figures per chapter. One is under-illustrated; thirty is a slide dump.
- Verify the file exists before referencing it. Broken image link = defect.
- **No downloads, no external URLs as images.** Local assets only. If a concept needs a picture no
  asset provides, draw a fenced ASCII diagram or a Mermaid block.

## 5. The companion course — overlap rule

The reader has complete notes for *Generative AI for Computer Vision* at
`../../../GenAIforCV/notes/`. Several lectures here cover the same ground from an NLP angle:

| This course | Companion chapter |
|---|---|
| Lec 6–10 (supervised learning → backprop → SGD/Adam) | `week-02/07-perceptron.md`, `08-mlp-and-activations.md`, `09-backpropagation.md`, `supplementary.md` |
| Lec 20 (GRU, LSTM) | `week-05/18-lstm.md`, `week-05/19-gru-seq2seq-attention.md` |
| Lec 18 (seq2seq + attention) | `week-05/19-gru-seq2seq-attention.md` |
| Lec 21–24 (Transformers) | `week-07/25-qkv-and-self-attention.md`, `26-encoder-and-positional-encoding.md`, `27-decoder-and-full-transformer.md` |
| Lec 23 (ViT mention) | `week-08/28-vit-detr-swin.md` |

**The rule: write your chapter standalone anyway.** The NPTEL exam for *this* course is set from
*these* slides, with this lecturer's notation, emphasis and worked examples. A reader who skipped your
chapter because "I did transformers already" would lose marks.

What the overlap buys you is **compression, not omission**: you may move faster through shared
mechanics, say explicitly "you met this in the vision course — here is what changes for text", and
link across with a relative path like
`[the vision course's treatment](../../../GenAIforCV/notes/week-07/25-qkv-and-self-attention.md)`.
Spend the space you save on what is genuinely new here: the NLP framing, this deck's worked problems,
and anything the vision course did not cover.

## 6. Scope discipline

Your assignment names topics you **own** and topics **owned elsewhere**. For owned-elsewhere: one
sentence maximum, then a link. Never re-derive a formula another chapter owns. For what you own: it is
yours completely and no other chapter covers it. If you skimp, there is a hole in the book.

## 7. Voice

- Plain, direct, declarative. Second person for the reader ("you compute", "notice that").
- Define a term the first time it appears in *your* chapter, even if an earlier chapter did — one
  clause is enough.
- Explain *why* before *what* wherever a formula appears.
- No filler: no "in today's fast-paced world", no "it is important to note that", no restating the
  heading as the first sentence, no closing pep talk.
- Don't pad. A tight 1,600 words beats a baggy 3,000. But don't clip real content to hit a number —
  completeness beats brevity when they conflict.
- Bold sparingly, for the term being defined. Tables for anything comparative.

## 8. Done means

- [ ] All seven H2 sections, correctly named and ordered.
- [ ] Front matter filled, with working relative links.
- [ ] You opened the rendered pages and the chapter reflects what is actually on them.
- [ ] **Every "Try this problem" page in your range is worked in full.**
- [ ] Every figure path verified to exist on disk.
- [ ] Notation matches §3 everywhere.
- [ ] Numericals land on concrete numbers; code runs as written.
- [ ] Nothing owned by another chapter is re-taught.
- [ ] Written to the exact path your assignment gives. One file. No other files, anywhere.

Report back: path written, word count, figure count, **how many "Try this problem" pages you found and
worked**, and anything on the pages the ownership map did not anticipate.
