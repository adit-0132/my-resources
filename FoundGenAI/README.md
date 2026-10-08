# Fundamentals of Generative AI and Large Language Models — replacement notes

NPTEL / IISc, 12 weeks. **54 chapters, ~393,000 words, 492 figures, 332 worked numericals.**

These notes are written to be read **instead of** the lectures, by someone with a basic ML background
(comfortable with `sklearn`, no matrix calculus, no PyTorch), targeting 95+ on the end-term exam.

Companion courses, already written: [`../GenAIforCV`](../GenAIforCV) and [`../DLforNLP`](../DLforNLP).
Chapters link across where topics overlap; 31 such links, all resolving.

---

## Read this before you revise

The source decks have **no text layer at all** — every page is an image, so all 805 pages were read
visually. Along the way the chapters catalogued a lot of defects in the slides. These are the ones that
change an answer rather than a word. Each is corrected in place in the relevant chapter, with both the
deck's version and the right one, so you can answer whichever the paper asks for.

**Errors that would cost you marks**

- **Lec 51 p-06 has a sign error in its boxed key equation** — a plus where it must be a minus. The
  deck's own preceding line implies the minus and its own next page implements a difference. Since that
  difference *is* the classifier-free guidance signal, the box as printed describes something that
  cannot work.
- **Lec 47's posterior mean has an ambiguous radical** — typeset so the bar appears to cover
  $\bar\alpha_{t-1}\beta_t$ jointly; it covers $\bar\alpha_{t-1}$ only. Settled numerically: the
  correct reading agrees with the deck's own later formula to 15 digits, the other is **46% off**.
- **Lec 46 p-8's boxed conclusion drops both square roots** and writes $\mathbf{I}$ where $\epsilon$
  belongs. The handwriting above it and the next page are both correct.
- **Lec 38 p-14's objective drops the label from $D$** — contradicting its own p-12 and the whole thesis
  of the lecture. An objective where $D$ never sees the condition is a vanilla GAN.
- **Lec 50 p-11 labels the DDIM update as "the DDPM reverse update".**
- **Lec 04 contradicts itself on AdaGrad across two consecutive pages.** Taken literally, p-6's version
  sends a weight from 5 to −4995 on the first step.
- **Lec 02 p-9's categorical cross-entropy has a broken summation index** ($-\sum_{i=1}^{k} y_k\log\hat y_k$
  — index $i$, summand $k$, and $k$ is both the running index and the limit).
- **Lec 01 p-11 plots a Binomial(10, 0.3) and labels it "Bernoulli Distribution"** — eleven bars where a
  Bernoulli has two.
- **Lec 14 p-8 miscomputes $0.005\times0.221$ as 0.00115** (it is 0.001105).
- **Lec 41 p-13 contradicts p-16**, the paper's own figure on the same deck: 18 convolutions claimed,
  17 actual. **Answer 18** — the deck's text and labels both say so.
- **Lec 54's GloVe slide makes a claim that is false on its own matrix.** Row cosines rank *exactly
  reversed* from what it asserts.
- **Week 11 part-a pp. 6 and 13 drop the square root from the attention scaling** — printing
  $\mathrm{softmax}(\mathbf{A}/d)$ where it must be $\sqrt{d_k}$. At $d_k = 64$ this flattens a real
  attention row to a peak of 0.158 against a uniform 0.143: it destroys the mechanism. **Lec 57 is
  correct and this deck is not.**
- **Week 11 part-a p-16's language-model loss is missing its logarithm** — $-\sum_t P(\cdot)$ where it
  must be $-\sum_t \log P(\cdot)$.
- **Week 11 part-a p-29 writes the LoRA update with a plain $\alpha$** where the paper, every library,
  and **this course's own Week 11 notebook** use $\alpha/r$. With the notebook's `r=8, lora_alpha=16`
  the two disagree by a factor of exactly $r$.
- **Lec 53's `FAST_MODE` is a real bug**: cutting $T$ to 100 leaves $\bar\alpha_T = 0.36$, so $x_T$ still
  holds 60% of the image while the sampler starts from pure noise. Proven with an oracle denoiser.

**Three traps that are nobody's error, just the course's shape**

1. **Three logarithm bases are live.** Lec 14 computes KL in $\log_{10}$, Lec 19 in nats (the book's
   default), Lec 20 in bits. Every numerical answer in these notes states its base. Worth internalising:
   the Gaussian KL's log term in base 10 gives 0.6441 instead of 0.3517 — a **1.83× error, not a clean
   factor of 2.303**, so you cannot spot it by eye. A **negative KL** means you used $\sigma$ for $\sigma^2$.
2. **Symbols collide across adjacent lectures.** $\phi$ is the VAE encoder but the GAN *generator*;
   $y$ is a class label in Lec 38, a target image in Lec 39, and a domain in Lec 40; capital $C$ is the
   LSTM cell state in Lec 55 and the attention context in Lec 56; the guidance scale is $s$ on the slides
   and $w$ in every library, with $w = 1+s$. Each chapter flags its own collisions.
3. **Content is not where the lecture titles say.** The biggest: **Lec 57 "Transformer Encoder" contains
   no attention content at all** — the Q/K/V and multi-head slides are on Lec 56. And **no slide anywhere
   in this course derives the $\sqrt{d_k}$ scaling, works an attention numerical, or states
   $d_k = d_{\text{model}}/h$.**

**What the slides never cover, and these notes supply**

Adam's bias correction · the ELBO's derivation · the closed-form Gaussian KL · the optimal discriminator,
$-\log 4$, Jensen–Shannon and mode collapse · path-length regularisation · the $\sqrt{d_k}$ argument ·
BERT's and GPT's model configurations (so the most likely BERT numerical is unanswerable from the decks)
· zero/one/few-shot in-context learning · GroupNorm · the receptive-field recurrence · multi-channel
convolution · the vanishing-gradient derivation. **Lec 35 "Introduction to DCGAN" has no deck at all** —
a genuine hole in the source material, covered in Lec 34 and Lec 38.

One habit to know about: **the lecturer ends most decks on a bare "Summary" title card with nothing on
it** — confirmed on 14 decks. Summaries were delivered verbally. The *Exam pack* in each chapter is the
replacement.

---

## How to use this

| If you have | Read |
|---|---|
| Weeks | `notes/` in order — each chapter is self-contained |
| A few days | `cram/week-summaries.md`, then `cram/exam-traps.md` |
| A few hours | `cram/formula-sheet.md` and `cram/numbers.md` |
| An hour | `cram/exam-traps.md` — 614 traps, each stating the confusion *and* the discrimination |
| To self-test | `cram/self-test-bank.md` — 1,094 questions with answers |

Every chapter has the same seven sections: *Why this lecture exists · The ideas · Worked numericals ·
Code · Exam pack · Beyond the slides · Cut from the slides*. The last one is the audit trail — what was
compressed or dropped, and why.

---

## Chapters

**Week 1**

| Chapter | Source deck |
|---|---|
| [Lec 01 — Introduction to Generative AI](notes/01-intro-generative-ai.md) | `Lec 01.pdf` |
| [Lec 02 — Activation and Loss Functions](notes/02-activations-and-losses.md) | `Lec 02.pdf` |
| [Lec 03 — Optimizers, Part A](notes/03-optimizers-a.md) | `Lec 03.pdf` |
| [Lec 04 — Optimizers, Part B](notes/04-optimizers-b.md) | `Lec 04.pdf` |
| [Lec 05 — Convolutional Neural Network, Part A](notes/05-cnn-a.md) | `Lec 05.pdf` |
| [Lec 06 — Convolutional Neural Network, Part B](notes/06-cnn-b.md) | `Lec 06.pdf` |

**Week 2**

| Chapter | Source deck |
|---|---|
| [Lec 10 — Introduction to Autoencoders](notes/10-autoencoder-intro.md) | `Lec 10.pdf` |
| [Lec 11 — Reconstruction Loss (MSE, Binary Cross-Entropy)](notes/11-reconstruction-loss.md) | `Lec 11.pdf` |
| [Lec 12 — Types of Autoencoders](notes/12-autoencoder-types.md) | `Lec 12.pdf` |
| [Lec 13 — Denoising Autoencoders](notes/13-denoising-ae.md) | `Lec 13.pdf` |
| [Lec 14 — Sparse Autoencoders](notes/14-sparse-ae.md) | `Lec 14.pdf` |
| [Lec 15 — Contractive Autoencoders](notes/15-contractive-ae.md) | `Lec 15.pdf` |
| [Lec 16 — Numerical Example, Limitations of AE](notes/16-ae-numerical-and-limits.md) | `Lec 16.pdf` |

**Week 3**

| Chapter | Source deck |
|---|---|
| [Lec 19 — KL Divergence, Part A](notes/19-kl-divergence-a.md) | `Lec 19.pdf` |
| [Lec 20 — KL Divergence, Part B](notes/20-kl-divergence-b.md) | `Lec 20.pdf` |
| [Lec 21 — Introduction to VAE and the Encoder](notes/21-vae-encoder.md) | `Lec 21.pdf` |
| [Lec 22 — Probabilistic Decoder, ELBO, VAE Loss](notes/22-elbo-and-vae-loss.md) | `Lec 22.pdf` |
| [Lec 23 — The Reparameterization Trick](notes/23-reparameterization.md) | `Lec 23.pdf` |
| [Lec 24 — VAE Numerical Example](notes/24-vae-numerical.md) | `Lec 24.pdf` |

**Week 4**

| Chapter | Source deck |
|---|---|
| [Lec 26 — Disentanglement and β-VAE](notes/26-beta-vae.md) | `Lec 26.pdf` |
| [Lec 27 — Conditional VAE](notes/27-conditional-vae.md) | `Lec 27.pdf` |
| [Lec 28 — Latent Space Interpolation](notes/28-latent-interpolation.md) | `Lec 28.pdf` |

**Week 5**

| Chapter | Source deck |
|---|---|
| [Lec 31 — Motivation for GANs](notes/31-gan-motivation.md) | `Lec 31.pdf` |
| [Lec 32 — GAN Architecture](notes/32-gan-architecture.md) | `Lec 32.pdf` |
| [Lec 33 — GAN Objective and Loss Functions](notes/33-gan-objective.md) | `Lec 33.pdf` |
| [Lec 34 — GAN Convergence and Nash Equilibrium](notes/34-gan-convergence.md) | `Lec 34.pdf` |

**Week 6**

| Chapter | Source deck |
|---|---|
| [Lec 38 — Conditional GAN](notes/38-conditional-gan.md) | `Lec 38.pdf` |
| [Lec 39 — Image-to-Image Translation: Pix2Pix GAN](notes/39-pix2pix.md) | `Lec 39.pdf` |
| [Lec 40 — CycleGAN](notes/40-cyclegan.md) | `Lec 40.pdf` |
| [Lec 41 — StyleGAN](notes/41-stylegan.md) | `Lec 41.pdf` |
| [Lec 42 — StyleGAN 2](notes/42-stylegan2.md) | `Lec 42.pdf` |

**Week 7**

| Chapter | Source deck |
|---|---|
| [Lec 44 — Introduction to Diffusion Models](notes/44-diffusion-intro.md) | `Lec 44.pdf` |
| [Lec 45 — Mathematical Foundations of Diffusion Models](notes/45-diffusion-math.md) | `Lec 45 .pdf` |
| [Lec 46 — DDPM: The Forward Process](notes/46-ddpm-forward.md) | `Lec 46.pdf` |
| [Lec 47 — DDPM: The Reverse Process](notes/47-ddpm-reverse.md) | `Lec 47.pdf` |
| [Lec 48 — Hands-on: Forward Diffusion](notes/48-forward-diffusion-handson.md) | `Lec 48.pdf` |

**Week 8**

| Chapter | Source deck |
|---|---|
| [Lec 49 — U-Net for Denoising](notes/49-unet.md) | `Lec 49.pdf` |
| [Lec 50 — Classifier-Guided Diffusion](notes/50-classifier-guidance.md) | `Lec 50.pdf` |
| [Lec 51 — Classifier-Free Diffusion Guidance (CFG)](notes/51-classifier-free-guidance.md) | `Lec 51.pdf` |
| [Lec 52 — Stable Diffusion (Latent Diffusion Models)](notes/52-stable-diffusion.md) | `Lec 52.pdf` |
| [Lec 53 — Hands-on: Reverse Diffusion](notes/53-reverse-diffusion-handson.md) | `Lec 53.pdf` |

**Week 9**

| Chapter | Source deck |
|---|---|
| [Lec 54 — Foundations of NLP](notes/54-nlp-foundations.md) | `Lec 54.pdf` |
| [Lec 55 — Sequential Modeling with RNNs and LSTMs](notes/55-rnn-lstm.md) | `Lec 55.pdf` |
| [Lec 56 — Evolution From LSTMs to Transformers](notes/56-lstm-to-transformer.md) | `Lec 56.pdf` |
| [Lec 57 — Transformer Encoder](notes/57-transformer-encoder.md) | `Lec 57.pdf` |

**Week 10**

| Chapter | Source deck |
|---|---|
| [Lec 59 — Transformer Decoder](notes/59-transformer-decoder.md) | `Lec 59.pdf` |
| [Lec 60 — BERT (Bidirectional Encoder Representations from Transformers)](notes/60-bert.md) | `Lec 60.pdf` |
| [Lec 61 — GPT (Generative Pre-trained Transformer)](notes/61-gpt.md) | `Lec 61.pdf` |
| [Lec 62 — Prompt Engineering Basics](notes/62-prompt-engineering.md) | `Lec 62.pdf` |
| [Lec 63 — Hands-on on LLM](notes/63-llm-handson.md) | `Lec 63.pdf` |

**Week 11**

| Chapter | Source deck |
|---|---|
| [Lec 64 — LLM Recap, In-Context Learning, LoRA, and the RAG Principle](notes/64-llm-icl-lora-rag.md) | `Foundations-of-LLM-Lecture1-parta.pdf` |
| [Lec 68 — Retrieval Augmented Generation: Mechanics and Advances](notes/68-rag-advances.md) | `Foundations-of-LLMs-Lecture1-partb.pdf` |

**Week 12**

| Chapter | Source deck |
|---|---|
| [Lec 70–71 — LLMs for Text Generation and Multimodal Alignment](notes/70-llm-generation-multimodal.md) | `Week12-notes.pdf` |
| [Lec 72–73 — Size versus Performance, Benchmarks, Bias and Safety](notes/72-benchmarking-bias-safety.md) | `Week12-notes.pdf` |

---

## Layout

```
notes/            54 chapters, flat (lecture numbering is the spine)
cram/             5 revision sheets, generated from the chapters
assets/pages/     805 rendered slide images, the only form the source exists in
assets/notebooks/ the two companion Jupyter notebooks (Weeks 11, 12)
_build/           CONTRACT.md, OWNERSHIP.md (13 errata batches), and the checkers
```

Rebuild and verify:

```bash
python3 _build/validate.py     # structure, links, figures, tables, delimiters
python3 _build/run_code.py     # executes every chapter's Python
python3 _build/build_cram.py   # regenerates cram/
```

Current state: **0 validation issues · 54/54 chapters' code runs · 0 broken links**
(1,527 internal, 31 cross-course, 492 figures).

## A note on the lecture numbering

Thirteen numbers are missing from the source. Twelve are Practical Exercise / Colab sessions delivered
live with no deck, so nothing is lost: 7, 8, 9, 17, 18, 25, 29, 30, 36, 37, 43, 58. **Lec 35
"Introduction to DCGAN" is the exception** — a content lecture whose slides are genuinely absent.

Weeks 11 and 12 are taught by a **different lecturer** (Sriram Ganapathy, EE, IISc), and their notation
diverges sharply — `z` becomes a *logit* after eight weeks as the VAE latent, `T` becomes *temperature*
after five weeks as the diffusion step count, and `h` becomes an embedding width where it is the head
count everywhere else. Those chapters carry full translation tables, since the exam draws on both
lecturers.

Three topics the syllabus implies are **taught on no slide anywhere**: top-k and top-p sampling (named
once, never defined), beam search (absent entirely), and CLIP / contrastive alignment — which matters
because Lec 52 forwards CLIP's mechanism to Week 12, so without it the book would have no account of how
a text prompt steers a diffusion model. All three are written as clearly-flagged owned content. Likewise
**§2.3 contains no scaling law at all** — no Kaplan, no Chinchilla, no $C \approx 6ND$; its single slide
is a 2026 mixture-of-experts snapshot arguing the size/performance link is *broken* by sparsity. The
classical laws and the reconciliation are both supplied.
