# Note-Writing Contract — binding on every author

You are writing **one chapter of one book**, not a standalone document. Another author is writing the
chapter before yours and the chapter after yours, right now, from this same contract. Uniformity is
not a nicety here; it is the deliverable.

Read this whole file before writing a word.

---

## 1. The reader

One specific person. Calibrate everything to them:

- Can use `sklearn` — `fit`, `predict`, `train_test_split`, a `RandomForestClassifier`. Knows what
  overfitting *is* as a word.
- Has **not** done: matrix calculus, backprop by hand, any PyTorch/TensorFlow, any probability beyond
  "P(A and B)", any eigen-anything since school.
- Is sitting an **NPTEL end-term exam** and wants 95+. The exam is MCQ + short numerical. This means
  **exact definitions, exact formulas, and the ability to compute small numbers by hand** matter more
  than vibes.
- Is reading these notes *instead of* watching the lectures. If you leave something out, they never
  see it. There is no safety net.

Write to that person. Do not write to a grad student, and do not write like a slide deck.

## 2. Non-negotiable file shape

Exactly these H2 sections, exactly these names, in exactly this order. No extras, none omitted.
(If a section would be genuinely empty — e.g. no numerical is possible — write one line saying so and
why. Do not delete the heading.)

```markdown
# Lec NN — <Title>

> **Deck:** `<original pptx filename>` · **Week N** · **Playlist:** Lec NN
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

**Why this lecture exists** — 100–150 words. The *problem* this lecture solves and why it comes
after the previous one. A reader who has just finished the previous chapter should feel a question
being answered. No bullet lists here; prose.

**The ideas** — the body, and the bulk of the file. Free use of H3 (`###`) subheadings. Every
concept on the deck, taught properly: what it is, why it works, where it breaks. Derive the math,
do not assert it. Embed figures (§4). This is where you spend your effort.

**Worked numericals** — 2–5 fully worked numerical examples of the *exact kind the exam asks*.
Show every arithmetic step; land on a concrete number. Conv output sizes, parameter counts, one
backprop step by hand, attention score matrices, KL terms for given means/variances. Invent the
numbers if the slides give none. **This is the highest-value section in the file for a 95+ score** —
treat it that way. Format each as:

```markdown
### N1. <What is being computed>
**Given:** ...
**Find:** ...
<numbered steps, each with the arithmetic shown>
**Answer:** <boxed-looking final line>
```

**Code** — runnable NumPy or PyTorch that makes an idea concrete. 10–40 lines per block, commented,
with the expected printed output shown underneath in a comment or a fenced block. Prefer NumPy for
mechanism (so the reader sees the arithmetic) and PyTorch only where the real-world API is the point.
No `...` placeholders, no pseudo-code — it must actually run. If a lecture is pure concept with
nothing to run, say so in one line and skip.

**Exam pack** — four labelled subsections, in this order:

```markdown
### Must-memorise
<table: | Item | Exactly this |>  — definitions and formulas to know verbatim

### Numbers worth knowing
<table of the specific figures from this lecture an MCQ can key on:
 layer counts, parameter counts, error rates, years, kernel sizes>

### Likely MCQ traps
<bulleted; each trap states the confusion AND the correct discrimination>

### Self-test
<6–10 questions. Answers inline in a collapsed block:
<details><summary>Answers</summary> ... </details> >
```

**Beyond the slides** — things the deck omits that the reader genuinely needs, each as a short
`**Gap:**` / `**Why it matters:**` pair. Keep it tight: 2–5 items. This is for real gaps (a formula
the deck states without the condition under which it holds), not for showing off adjacent knowledge.

**Cut from the slides** — one short paragraph naming what you compressed or dropped and why
(title slides, repeated "Content" slides, prose repeated verbatim across three slides, a figure that
duplicates an earlier one). This is the audit trail that proves nothing was lost by accident.

## 3. Notation — identical across all 30 chapters

Deviating from this table breaks the book. If your deck uses different symbols, **translate to these**
and add one line noting the deck's symbol.

| Thing | Write | Not |
|---|---|---|
| scalar | $x$, $y$, $\eta$ | |
| vector | $\mathbf{x}$, $\mathbf{h}$ | $\vec{x}$, $x$ |
| matrix | $\mathbf{W}$, $\mathbf{A}$ | $W$ |
| layer index | $\mathbf{W}^{(l)}$, $a_j^{(l)}$ | $W_l$, $W[l]$ |
| time step | $\mathbf{h}_t$, $\mathbf{x}_t$ | $h(t)$ |
| loss | $\mathcal{L}$ | $L$, $J$, $E$ |
| dataset | $\mathcal{D}$ | $D$, $S$ |
| expectation | $\mathbb{E}_{p(x)}[\cdot]$ | $E[\cdot]$ |
| learning rate | $\eta$ | $\alpha$, `lr` |
| sigmoid / tanh | $\sigma(\cdot)$ / $\tanh(\cdot)$ | |
| params (generic) | $\theta$ | $w$ |
| conv geometry | in $H\times W\times C$, kernel $K$, stride $S$, pad $P$, filters $F$ | |
| attention | $\mathbf{Q},\mathbf{K},\mathbf{V}$; $d_k,d_v,d_{\text{model}}$; $h$ heads | |
| attention context | $\mathbf{c}_t$ — **reserved for this**, never the LSTM cell state |
| LSTM cell state | $\mathbf{C}_t$ (capital), candidate $\tilde{\mathbf{C}}_t$ |
| VAE | encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$, decoder $p_\theta(\mathbf{x}\mid\mathbf{z})$, prior $p(\mathbf{z})$ | |
| KL | $D_{\mathrm{KL}}(q\,\|\,p)$ | $KL(q,p)$ |

Math delimiters: `$...$` inline, `$$...$$` display on its own lines. **Never** `\(...\)` or
`\[...\]` — the renderer here does not support them.

## 4. Figures

Both asset pools already exist. Use relative paths from your note's directory
(`notes/week-0N/`), which means prefixing `../../`.

**Pool A — extracted original figures** (`assets/figures/<DECK_TAG>/imageNN.png`). The clean,
native-resolution diagrams and photos lifted straight out of the deck. **Prefer these.**

**Pool B — rendered whole slides** (`assets/slides/<DECK_TAG>/sNN.png`). The full slide as the
lecturer showed it. Use when the content *is* the slide — a built-up equation, a shape-drawn diagram,
an annotated architecture — because those are PowerPoint shapes and are not in Pool A.

```markdown
![Inception module: four parallel branches concatenated depth-wise](../../assets/figures/W3_W3L4_P2_CNN_Architectures/image8.png)
*Fig. — 1×1 bottlenecks sit before the 3×3 and 5×5 branches, cutting the multiply count ~10×. Slide 24.*
```

Rules:
- **You must open the rendered slides and look at them** (`Read` the PNGs) before writing. The math
  on these decks is embedded as *images*, so the extracted text is blind to it. A chapter written
  from the text dump alone will be wrong and will have holes. This is the single most common way to
  fail this task.
- Every figure gets descriptive alt text and an italic caption that *says what to notice*, ending
  with the slide number.
- 4–12 figures per chapter. A chapter with one figure is under-illustrated; one with thirty is a
  slide dump.
- Verify the file exists before referencing it. A broken image link is a defect.
- You may **not** download anything or reference an external URL as an image. Local assets only.
  If a concept needs a picture that no asset provides, draw it as a fenced ASCII diagram or a Mermaid
  block instead.

## 5. Scope discipline — read your assignment's ownership block

Your assignment names topics you **own** and topics that are **owned elsewhere**. For anything owned
elsewhere:

- One sentence maximum, then a link. Do not re-teach it.
- Never re-derive a formula another chapter owns.

For anything you own: it is yours completely, and no other chapter will cover it. If you skimp, there
is a hole in the book. This is how 30 parallel authors avoid writing "attention explained from
scratch" five times while leaving positional encoding unwritten.

## 6. Voice

- Plain, direct, declarative. Second person for the reader ("you compute", "notice that").
- Define a term the first time it appears in *your* chapter, even if an earlier chapter defined it —
  one clause is enough: "the receptive field (the input region one unit can see)".
- Explain *why* before *what* wherever a formula appears. The reader should never meet a symbol they
  cannot motivate.
- No filler: no "in today's fast-paced world", no "it is important to note that", no restating the
  heading as the first sentence, no closing pep talk.
- Do not pad. A tight 1,600 words beats a baggy 3,000. But do not clip real content to hit a number —
  completeness beats brevity when they conflict.
- Bold sparingly, for the term being defined. Tables for anything comparative.

## 7. Done means

- [ ] All seven H2 sections present, correctly named and ordered.
- [ ] Front-matter blockquote filled in, with working relative links.
- [ ] You opened the rendered slides and the chapter reflects what is actually on them, including
      the image-only equation slides.
- [ ] Every figure path verified to exist on disk.
- [ ] Notation matches §3 everywhere.
- [ ] Worked numericals land on concrete numbers.
- [ ] Code runs as written.
- [ ] Nothing owned by another chapter is re-taught.
- [ ] Written to the exact path your assignment gives. One file. No other files, anywhere.

Report back: the path you wrote, word count, figure count, and anything you found on the slides that
the ownership map did not anticipate (so the map can be fixed).
