# Ownership map — who teaches what

NPTEL / IISc *Fundamentals of Generative AI and Large Language Models: Theory and Practice*.
53 source PDFs, **805 pages, zero text layer** — see CONTRACT §0.

Chapters live flat in `notes/` (one file per lecture, no week subdirectories), because the lecture
numbering is the organising spine and several weeks map to one PDF.

## The 13 missing lecture numbers

7, 8, 9, 17, 18, 25, 29, 30, 35, 36, 37, 43, 58 have **no slides**. Twelve of them are the playlist's
*Practical Exercise* / *Introduction to Google Colab* / hands-on sessions, which were delivered live
with no deck — so nothing is lost.

**The exception is Lec 35, "Introduction to DCGAN", which is a content lecture whose slides are
genuinely absent.** That is a real hole in the source material. Lec 34 and Lec 38's authors should each
give DCGAN one orienting paragraph (it is the bridge from the vanilla GAN to the conditional variants)
and flag it as not-on-the-slides. Lec 38 owns the flag.

Note that Lec 48 and Lec 53 are titled "Hands on…" but **do** have decks, so they are full chapters.

## Chapter table

| Lec | Week | Title | Chapter file | Pages |
|---|---|---|---|---|
| 1 | 1 | Introduction to Generative AI | `notes/01-intro-generative-ai.md` | 14 |
| 2 | 1 | Activation and Loss Functions | `notes/02-activations-and-losses.md` | 12 |
| 3 | 1 | Optimizers — Part A | `notes/03-optimizers-a.md` | 8 |
| 4 | 1 | Optimizers — Part B | `notes/04-optimizers-b.md` | 11 |
| 5 | 1 | Convolutional Neural Network — Part A | `notes/05-cnn-a.md` | 16 |
| 6 | 1 | Convolutional Neural Network — Part B | `notes/06-cnn-b.md` | 17 |
| 10 | 2 | Introduction to Autoencoders | `notes/10-autoencoder-intro.md` | 10 |
| 11 | 2 | Reconstruction Loss (MSE, BCE) | `notes/11-reconstruction-loss.md` | 11 |
| 12 | 2 | Types of Autoencoders | `notes/12-autoencoder-types.md` | 16 |
| 13 | 2 | Denoising Autoencoders | `notes/13-denoising-ae.md` | 15 |
| 14 | 2 | Sparse Autoencoders | `notes/14-sparse-ae.md` | 11 |
| 15 | 2 | Contractive Autoencoders | `notes/15-contractive-ae.md` | 8 |
| 16 | 2 | Numerical Example, Limitations of AE | `notes/16-ae-numerical-and-limits.md` | 12 |
| 19 | 3 | KL Divergence — Part A | `notes/19-kl-divergence-a.md` | 12 |
| 20 | 3 | KL Divergence — Part B | `notes/20-kl-divergence-b.md` | 10 |
| 21 | 3 | Introduction to VAE and the Encoder | `notes/21-vae-encoder.md` | 12 |
| 22 | 3 | Probabilistic Decoder, ELBO, VAE Loss | `notes/22-elbo-and-vae-loss.md` | 10 |
| 23 | 3 | Reparameterization Trick | `notes/23-reparameterization.md` | 11 |
| 24 | 3 | VAE Numerical Example | `notes/24-vae-numerical.md` | 14 |
| 26 | 4 | Disentanglement and β-VAE | `notes/26-beta-vae.md` | 16 |
| 27 | 4 | Conditional VAE | `notes/27-conditional-vae.md` | 11 |
| 28 | 4 | Latent Space Interpolation | `notes/28-latent-interpolation.md` | 11 |
| 31 | 5 | Motivation for GANs | `notes/31-gan-motivation.md` | 12 |
| 32 | 5 | GAN Architecture | `notes/32-gan-architecture.md` | 13 |
| 33 | 5 | GAN Objective and Loss Functions | `notes/33-gan-objective.md` | 14 |
| 34 | 5 | GAN Convergence and Nash Equilibrium | `notes/34-gan-convergence.md` | 8 |
| 38 | 6 | Conditional GAN | `notes/38-conditional-gan.md` | 18 |
| 39 | 6 | Pix2Pix GAN | `notes/39-pix2pix.md` | 17 |
| 40 | 6 | CycleGAN | `notes/40-cyclegan.md` | 17 |
| 41 | 6 | StyleGAN | `notes/41-stylegan.md` | 18 |
| 42 | 6 | StyleGAN 2 | `notes/42-stylegan2.md` | 20 |
| 44 | 7 | Introduction to Diffusion Models | `notes/44-diffusion-intro.md` | 18 |
| 45 | 7 | Mathematics of Diffusion Models | `notes/45-diffusion-math.md` | 13 |
| 46 | 7 | DDPM — Forward Process | `notes/46-ddpm-forward.md` | 12 |
| 47 | 7 | DDPM — Reverse Process | `notes/47-ddpm-reverse.md` | 13 |
| 48 | 7 | Hands-on: Forward Diffusion | `notes/48-forward-diffusion-handson.md` | 26 |
| 49 | 8 | U-Net for Denoising | `notes/49-unet.md` | 18 |
| 50 | 8 | Classifier-Guided Diffusion | `notes/50-classifier-guidance.md` | 14 |
| 51 | 8 | Classifier-Free Diffusion | `notes/51-classifier-free-guidance.md` | 11 |
| 52 | 8 | Stable Diffusion | `notes/52-stable-diffusion.md` | 18 |
| 53 | 8 | Hands-on: Reverse Diffusion | `notes/53-reverse-diffusion-handson.md` | 24 |
| 54 | 9 | Foundations of NLP | `notes/54-nlp-foundations.md` | 16 |
| 55 | 9 | RNNs and LSTM | `notes/55-rnn-lstm.md` | 12 |
| 56 | 9 | From LSTMs to Transformers | `notes/56-lstm-to-transformer.md` | 15 |
| 57 | 9 | Transformer Encoder | `notes/57-transformer-encoder.md` | 14 |
| 59 | 10 | Transformer Decoder | `notes/59-transformer-decoder.md` | 15 |
| 60 | 10 | BERT | `notes/60-bert.md` | 17 |
| 61 | 10 | GPT | `notes/61-gpt.md` | 14 |
| 62 | 10 | Prompt Engineering Basics | `notes/62-prompt-engineering.md` | 17 |
| 63 | 10 | Hands-on on LLM | `notes/63-llm-handson.md` | 23 |
| 64–67 | 11 | LLM Recap, In-Context Learning, LoRA, RAG | `notes/64-llm-icl-lora-rag.md` | 33 (part a) |
| 68 | 11 | RAG — continued | `notes/68-rag-advances.md` | 19 (part b) |
| 70–71 | 12 | LLMs for Text Generation and Multimodal | `notes/70-llm-generation-multimodal.md` | W12 notes, first half |
| 72–73 | 12 | Size, Benchmarking, Bias and Safety | `notes/72-benchmarking-bias-safety.md` | W12 notes, second half |

> **Weeks 11 and 12 are a different lecturer.** Sriram Ganapathy (EE, IISc / TANUH AI-CoE), with a
> completely different deck design. **The two Week-11 decks do not even resemble each other:** part-a
> is a *black* deck with orange headings and a blue/green rule (Kotak IISc AI–ML Centre and TANUH
> logos); part-b is the purple-and-yellow Google-Slides one. **Week 12 matches part-a**, black with
> orange titles, and switches into a second dark-navy infographic register on pp. 25, 30, 32, 33,
> 35, 36, 37. None carries the NPTEL watermark. Expect different notation and a different register from Lec 01–63. Authors of these
> chapters: translate to CONTRACT §3 notation and **note explicitly where this lecturer's symbols
> differ from the first lecturer's**, because the exam draws on both.
>
> **Week 11 part-b is wholly RAG** — its title slide reads "LLMs - Retrieval Augmented Generation".
> Its page 2 is an "LLMs - Recap" showing the decoder-only transformer stack, which overlaps part-a's
> §1.1; part-b's author should treat that recap as a one-paragraph bridge and link back, not re-teach it.

**Week 11 and 12 authors:** your PDFs are multi-lecture. **Read your deck's "Contents — This Lecture"
page first and report the real section boundaries**, so the map can be corrected. Week 11 part-a's
contents page reads: *1.1 LLMs Recap · 1.2 In-context learning, low-rank adaptation · 1.3 Retrieval
Augmented Generation (1.3.1 Principle, 1.3.2 Recent advances)*. Week 12's reads: *2.1 Large language
models (tokenizers and sampling, text generation) · 2.2 Multimodal alignment · 2.3 Size versus
Performance · 2.5 Evaluation Benchmarks · 2.6 Bias, and Safety* (the deck skips 2.4).

Notebooks for Weeks 11 and 12 are at `assets/notebooks/`.

---

## Contested topics — pre-resolved

### Autoencoders (Lec 10–16) — the course's first major arc
| Aspect | Owner |
|---|---|
| What an AE is, encoder/decoder/bottleneck, undercomplete vs overcomplete, uses | **Lec 10** |
| **Reconstruction loss: MSE vs BCE**, when each applies, their derivatives | **Lec 11** |
| The AE *taxonomy* (vanilla, deep, convolutional…) and code-level architecture | **Lec 12** |
| **Denoising AE** — corruption process, why it regularises | **Lec 13** |
| **Sparse AE** — the sparsity penalty, KL-based sparsity, activation targets | **Lec 14** |
| **Contractive AE** — the Jacobian/Frobenius penalty and what it buys | **Lec 15** |
| A full AE **numerical worked end to end**, and **why an AE is not generative** | **Lec 16** |

Lec 13/14/15 are three parallel regularisers. Each owns its own penalty term completely; **each must
also state in one line how it differs from the other two**, so the reader can tell them apart — that
three-way discrimination is the obvious MCQ.

### KL divergence and the VAE (Lec 19–24) — the mathematical core
| Aspect | Owner |
|---|---|
| **KL divergence**: why Euclidean distance fails, the definition (discrete and continuous), non-negativity, **asymmetry**, **forward vs reverse KL**, the support/zero problem, **mode-covering vs mode-seeking** | **Lec 19** |
| **Where the formula comes from**: self-information → entropy → cross-entropy → $D_{\mathrm{KL}} = H(P,Q) - H(P)$ | **Lec 20** |
| The VAE's motivation, the **probabilistic encoder**, $\mu$ and $\log\sigma^2$ outputs | **Lec 21** |
| The **probabilistic decoder, the ELBO derivation, the two-term VAE loss** | **Lec 22** |
| **The reparameterization trick** — why sampling blocks gradients, $\mathbf{z} = \mu + \sigma\odot\epsilon$ | **Lec 23** |
| A **complete VAE numerical** — forward pass, both loss terms, a gradient step | **Lec 24** |

**Lec 19 owns KL's definition for this whole book.** Lec 22 (ELBO's KL term), Lec 14 (KL sparsity) and
Lec 50/51 all link back rather than redefining it.

### VAE extensions (Lec 26–28)
| Aspect | Owner |
|---|---|
| Entanglement vs disentanglement, **β-VAE** and what $\beta$ trades off | **Lec 26** |
| **Conditional VAE** — conditioning the encoder and decoder on a label | **Lec 27** |
| **Latent space interpolation**, and why a VAE's latent space supports it where an AE's does not | **Lec 28** |

### GANs (Lec 31–42) — this course's deepest arc, far beyond the companions
| Aspect | Owner |
|---|---|
| Why GANs: **six numbered limitations of VAEs** (Gaussian assumption · blurriness · recon-vs-KL trade-off · **posterior collapse** · explicit-likelihood assumption · lack of fine detail), the adversarial framing | **Lec 31** |
| **The architecture** — $G$, $D$, the noise prior, **the alternating training loop**, what each network sees, the gradient paths. Owns its deck's three worked numericals (pp. 4, 10). | **Lec 32** |
| **The loss, derived**: BCE → $\mathcal{L}_{\text{real}}$, $\mathcal{L}_{\text{fake}}$, $\mathcal{L}_D$, $\mathcal{L}_G$; the expectation form; the minimax; non-saturating. **Lec 32 pp. 4/7/8 duplicate Lec 33 pp. 3–6 — assigned to 33.** | **Lec 33** |
| **Convergence, Nash equilibrium**, mode collapse, training instability, the optimal discriminator | **Lec 34** |
| **Conditional GAN** — label conditioning on both networks | **Lec 38** |
| **Pix2Pix** — paired image-to-image, the L1 + adversarial loss, PatchGAN | **Lec 39** |
| **CycleGAN** — *unpaired* translation, **cycle-consistency loss**, the two-generator setup | **Lec 40** |
| **StyleGAN** — the mapping network, $\mathcal{W}$ space, AdaIN, progressive growing, stochastic variation | **Lec 41** |
| **StyleGAN 2** — what was wrong with StyleGAN (droplet artefacts), weight demodulation, path-length regularisation | **Lec 42** |

**Lec 33 owns the GAN objective.** Lec 34, 38, 39, 40 all modify it — each states *its own* loss and
links back for the base. **Pix2Pix vs CycleGAN (paired vs unpaired) is the single most examinable
discrimination in this arc**; Lec 40 owns the contrast table.

### Diffusion (Lec 44–53) — also deeper than the companions
| Aspect | Owner |
|---|---|
| What diffusion is, the forward/reverse intuition, why it beats GANs on diversity, **the Markov property's definition** (its p-16 is a full slide on it) | **Lec 44** |
| **The Markov chain applied to the diffusion schedule**, the noise schedule $\beta_t$, $\alpha_t$, $\bar\alpha_t$ — links back to Lec 44 for the property itself | **Lec 45** |
| **Forward process** in full: $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$, $\alpha_t$, $\bar\alpha_t$, the reparameterised jump to any $t$ | **Lec 46** |
| **Reverse process**: $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$, the noise-prediction objective, the simplified loss | **Lec 47** |
| Hands-on forward diffusion — code, visualisation, the schedule in practice | **Lec 48** |
| **U-Net** — the architecture, skip connections, time-step embedding | **Lec 49** |
| **Classifier guidance** — the gradient term, the guidance scale | **Lec 50** |
| **Classifier-free guidance** — the unconditional/conditional mix, why it replaced classifier guidance | **Lec 51** |
| **Stable Diffusion** — the latent-space move, the VAE + U-Net + text-encoder stack, CLIP conditioning | **Lec 52** |
| Hands-on reverse diffusion — sampling loop, DDPM vs DDIM if shown | **Lec 53** |

**Lec 46 owns the forward-process algebra; Lec 47 owns the reverse.** Lec 45 sets up both and must not
pre-empt either — it owns the general Markov/schedule framing only. **Classifier vs classifier-free
guidance (Lec 50 vs 51) is the key discrimination;** Lec 51 owns the contrast.

### NLP and Transformers (Lec 54–63)
| Aspect | Owner |
|---|---|
| NLP foundations — tokenization, embeddings, the tasks | **Lec 54** |
| **RNNs and LSTM** — recurrence, vanishing gradients, the gates | **Lec 55** |
| **Why Transformers** — the three costs of recurrence, the thought-vector bottleneck, Bahdanau encoder–decoder attention. **Its deck ALSO carries the self-attention/Q-K-V/multi-head/masking slides (pp. 8–11), but Lec 57 owns the teaching** — Lec 56 names and points forward. | **Lec 56** |
| **The encoder** — **self-attention, Q/K/V, $\sqrt{d_k}$, multi-head** (all owned for the book, though *not on Lec 57's own deck*), positional encoding, residual + layer norm, the position-wise FFN, the block | **Lec 57** |
| **The decoder** — masked self-attention, cross-attention, the full architecture | **Lec 59** |
| **BERT** — MLM, NSP, bidirectionality, fine-tuning | **Lec 60** |
| **GPT** — causal LM, the series, zero/few-shot | **Lec 61** |
| **Prompt engineering** — templates, zero/few-shot, CoT if shown | **Lec 62** |
| Hands-on LLM — the practical pipeline | **Lec 63** |

**Lec 57 owns self-attention and Q/K/V for this book.** Lec 56 motivates and stops; Lec 59 applies.
**Lec 60 vs Lec 61 (BERT vs GPT, bidirectional+MLM vs causal+next-token) is a guaranteed exam question**
— Lec 61 owns the contrast table.

### Other pre-resolutions
- **Activation functions** → Lec 2 owns the catalogue and their derivatives. Lec 49 (U-Net) and Lec 57
  reference only.
- **Loss functions** → Lec 2 owns the general catalogue (MSE, CE, BCE). **Lec 11 owns reconstruction
  loss specifically** in the AE context, with its own MSE-vs-BCE discrimination.
- **Optimizers** → Lec 3 (part A: gradient descent, SGD, momentum) and Lec 4 (part B: adaptive methods,
  Adam) split the topic. **Lec 4 owns Adam.** Both own learning-rate behaviour; Lec 4's worked
  learning-rate cases (its page 3) are the canonical ones.
- **CNNs** → Lec 5 (part A: convolution, kernels, stride, padding, pooling) and Lec 6 (part B:
  architectures, feature hierarchy). Lec 5 owns the **output-size formula**; everyone else uses it.
- **KL divergence** → **Lec 19**. Referenced by Lec 14, 20, 22, 24, 50, 51.
- **The ELBO** → **Lec 22**. Lec 21 stops at the intractable posterior; Lec 24 computes it numerically.
- **Mode collapse** → **Lec 34**. Lec 31 may name it as a motivation in one clause.
- **Tokenization** → **Lec 54**, and again in Week 12's part 1 (tokenizers and sampling) — Week 12 owns
  the *sampling* half and links back for tokenization.
- **LoRA** → the **Week 11 part-a** chapter. **RAG** → Week 11 part-a owns the *principle*; part-b owns
  *recent advances*. Both must read their contents pages and report the real split.

## Exercise and worked-example sweep

**No automated sweep is possible — there is no text layer.** Instead, the contract makes it every
author's job: §2 requires reproducing *every* worked example in your page range, and §8 requires
reporting how many you found and whether your arithmetic matched.

Expect many. Two lecture titles are literally "Numerical Example" (Lec 16, Lec 24), and spot-checking
found fully worked arithmetic on ordinary teaching slides too (e.g. Lec 04 page 3 works six
learning-rate cases side by side). **Treat every slide as a potential numerical.**

## Errata and cross-chapter findings

(Appended during authoring.)

### Errata batch 1 — from the Lec 03/04 author

1. **The deck's $\alpha$ is the AdaGrad accumulator, not a learning rate.** This lecturer uses $\eta$
   for the learning rate correctly throughout, and $\alpha_t = \sum_i g_i^2$ for the squared-gradient
   accumulator. Many textbooks use $\alpha$ *for* the learning rate — the opposite convention. If you
   cite an optimiser, keep the deck's meaning and say which you mean.
2. **Lec 04 never shows Adam's bias correction.** Page 9 stops at $w - \eta v_t/(\sqrt{s_t}+\epsilon)$.
   Lec 04 teaches the correction as owned content and flags the omission. **Any chapter that mentions
   "trained with Adam" should not re-explain this** — link to `04-optimizers-b.md`.
3. **Lec 03's momentum is the normalised EWMA** $v = \beta v + (1-\beta)g$, not Polyak/PyTorch's
   $v = \beta v + g$. They differ by exactly $1/(1-\beta)$. Carrying this deck's $\eta$ straight into
   `torch.optim.SGD` would run 10× hot. Relevant to every chapter with training code.
4. **AdaGrad off-by-one, Lec 04 p-6 vs p-7:** p-6 excludes the current gradient from the accumulator,
   p-7 includes it (standard). Taken literally p-6 explodes the first step. Both forms documented.
5. **Three-course disagreement:** `../../DLforNLP` teaches Adam *with* bias correction; this deck
   teaches it without. Flagged in Lec 04 per CONTRACT §5.

### Errata batch 2 — from the Lec 05/06 and Lec 10/12 authors

**Map corrections (binding):**

1. **Pooling lives in Lec 06, not Lec 05.** Lec 05's own closing slide says "Next session: ReLU, Pooling
   Layer, Flatten Layer". Settled split: **Lec 05 owns pooling's *geometry*** (the shared output-size
   formula and the $C_{out}=C_{in}$ depth rule) — **Lec 06 owns the layer itself**, its four types and
   all its worked examples. Both chapters state this.
2. **Lec 06 contains no named architectures.** No LeNet/AlexNet/VGG/ResNet is described anywhere in
   Week 1 — AlexNet and VGG-16 appear only as figure sources for filter visualisations. Readers are
   routed to `../../GenAIforCV/`.
3. **Lec 12 silently owns the entire upsampling arc** (its pp. 5–7), which no map row anticipated:
   nearest-neighbour vs bed-of-nails upsampling, $O = I\times S$, **transposed convolution** with
   $O = I+K-1$ and the full $G_1{+}G_2{+}G_3{+}G_4$ construction, the **checkerboard artifact**, and the
   upsample+convolution fix.
   > **Lec 49 (U-Net) and Lec 52 (Stable Diffusion) authors: link back to `12-autoencoder-types.md`
   > for transposed convolution. Do NOT re-derive it.**
4. **Lec 12's taxonomy is "shallow · deep · convolutional"** — use the deck's own word *shallow*, not
   "vanilla". (Lec 10's outline slide calls the same thing "Simple".) The deck does **not** stray into
   Lec 13/14/15 territory; those appear only on its closing preview slide.
5. **Lec 10's deck has no applications slide.** The map's "uses" row has no slide support; Lec 10 wrote
   it as owned, clearly-sourced beyond-the-slides content.

**Deck defects:**

6. **Lec 12 p-8 is broken code:** `conv_autoencoder` assigns to `latent` but returns `bottleneck`, which
   is never defined — a `NameError` as printed.
7. **Lec 12's convolutional "latent space" is overcomplete and the slide calls it compressed.**
   7×7×256 = 12,544 values against a 784-value input — **16× larger than the input**. Spatial size fell
   16×, channels rose 256×. Directly contradicts Lec 10's bottleneck rule. A likely exam discriminator.
8. **Lec 06 p-7 prints the output-size formula without the floor** and binds `× D` ambiguously; Lec 05
   p-13 insists on the floor. Lec 05's form is canonical.
9. **Lec 06 has a non-standard "Max Avg Min pooling"** found in no library or textbook (output
   $\max-\min$ if $\max-\min > \text{avg}$, else $(\max+\text{avg})/2$). Being non-standard makes it
   *more* examinable, not less. Fully worked in Lec 06 including the tie case.
10. **Neither CNN deck teaches the multi-channel convolution rule**, so a reader can finish Week 1
    believing a 5×5 filter on RGB yields three maps. Stated explicitly in Lec 05 and trapped in both.
11. **Lec 10 coins "loss-free encoding"** for a code from which reconstruction is perfect — this
    lecturer's own phrase, in neither companion course, and examinable. Lec 16 may want it.

### Errata batch 3 — from the Lec 01/02 author

**Map additions:**

1. **Lec 01 owns the "output activation + loss ⇒ implied distribution" rule** (its pp. 8–9, given as
   three Keras lines: linear+mse / sigmoid+binary_crossentropy / softmax+categorical_crossentropy).
   Nobody else teaches it as a rule. Formulas and derivatives still belong to Lec 02, reconstruction
   loss to Lec 11.
2. **Lec 02 p-6 maps activations to architectures**, heavily hand-annotated by the lecturer, and is the
   most exam-predictive page in Week 1: GELU/GeGLU → Transformers & LLMs · SiLU/Swish → diffusion ·
   ReLU family → CNN/AE/VAE/GAN · sigmoid → GAN discriminator · sigmoid-or-tanh → GAN generator & VAE
   decoder. **That table lives in Lec 02.** Lec 49 and Lec 57 reference only.
3. **Lec 02 also owns sparse categorical cross-entropy** ($-\log\hat y_c$, integer labels), which the
   deck ties explicitly to LLM next-token prediction. CE and SCE are numerically identical.
4. **GELU, SiLU/Swish and GeGLU are genuinely new** — absent from both companion courses. Lec 02 spends
   its space there and compresses the shared sigmoid/tanh/ReLU material.

**Deck defects:**

5. **Lec 01 p-11 mislabels a Binomial(10, 0.3) PMF as "Bernoulli Distribution"** — eleven bars where a
   Bernoulli has two.
6. **Lec 01 p-10 writes the Gaussian as $N(\mu;\sigma)$** — semicolon, bare $\sigma$. Per CONTRACT §3
   this is $\mathcal{N}(\mu,\sigma^2)$. Misreading it costs a factor of 2.02 in the density.
7. **Lec 01 p-11 conflates sampling with greedy decoding** ("samples the most likely next word").
   Relevant to Lec 61/62 and Week 12's sampling section.
8. **Lec 02 p-9 categorical CE has an index error**: $-\sum_{i=1}^{k} y_k\log\hat y_k$ — index $i$,
   summand $k$, and $k$ is both the running index and the limit. Correct form $-\sum_{k=1}^{K}$.
9. **Lec 02 p-9 BCE is missing brackets**; as written $-\frac1N$ scopes only the first term.
10. **Lec 02 p-4 prints GELU's tanh approximation but tabulates the exact $z\Phi(z)$ values.** They agree
    to 4 dp, so nothing rounds wrong, but a 5-dp question would need the distinction.

### Tooling note

`validate.py`'s math-delimiter rule now has a `(?<!\\)` lookbehind, so LaTeX row spacing `\\[8pt]`
inside a matrix is no longer flagged as a forbidden `\[` delimiter. Use `\\[8pt]` freely in arrays.

### Errata batch 4 — from the Lec 13/14/15 author

**THE BIG ONE — the course uses three different logarithm bases for KL divergence:**

| Deck | Base |
|---|---|
| Lec 14 (sparse AE penalty) | **$\log_{10}$** |
| Lec 19 (KL definition) | natural log — **the book default** |
| Lec 20 (KL part B) | $\log_2$ / bits |

Lec 14's total penalty is 0.3438 in $\log_{10}$ against 0.7916 in nats — a factor of $\ln 10 = 2.3026$.
**Every author computing a KL, a cross-entropy or an entropy must state the base explicitly in the
answer line**, and should give the natural-log value as primary with the deck's value beside it when
they differ. This is the single most likely way to lose a numerical mark in this course.

**Deck defects:**

1. **Lec 14 p-8, neuron 1, arithmetic slip.** The slide computes $0.005\times0.221 = 0.00115$; it is
   $0.001105$. The slide's 0.000354 should be 0.000309 carrying its own rounded factors, 0.000242
   unrounded in $\log_{10}$, 0.000556 in nats. All three documented in Lec 14's N2.
2. Lec 14 p-8 lists $h_3$ as 0.00001 where exact $\log_{10}$ gives 0.000017.

**Map notes:**

3. **`assets/pages/lec15/` is UNPADDED** (`p-1.png`, not `p-01.png`) where lec13 and lec14 are padded.
   CONTRACT §4 already warns to `ls` first — this deck is why.
4. **Lec 15 has no worked numerical at all** and only 4 content pages of 8. Its six numericals are
   constructed, and the Jacobian closed form it needs is nowhere on the slides.
5. **Matrix convention switches mid-course.** Lec 11 writes $\mathbf{H} = g(\mathbf{X}\mathbf{W}_e)$
   (data left, $\mathbf{W}_e$ is $n\times k$); Lec 13 writes $\mathbf{h} = g(\mathbf{W}\mathbf{x}+\mathbf{b})$
   (data right, $\mathbf{W}_e$ is $k\times n$). **Both appear in this course.** Read shapes off the
   equation in front of you, never from memory. Later authors: flag it wherever your deck switches.
6. **Lec 14's KL sparsity penalty silently requires a sigmoid hidden layer** — undefined for
   $\hat\rho \le 0$ or $\ge 1$, so a ReLU + KL sparse autoencoder cannot run. The deck never says so.
7. **Lec 15's penalty degenerates to weight decay for a linear encoder**, so the CAE is only a distinct
   idea under non-linearity.
8. **Denoising ≈ contractive, quantitatively:** expanding the denoising objective for small Gaussian
   noise of variance $\sigma^2$ yields the contractive penalty with $\lambda \approx \sigma^2$.
9. **Lec 13 is the direct ancestor of the diffusion arc** (denoising score matching → Week 7).
   **Lec 44's author: pick this thread up.** Corrupt-then-restore at one noise level (Lec 13) becomes
   corrupt-then-restore across a schedule of noise levels (DDPM).

### Errata batch 5 — from the Lec 16/19/20 author

**THE LEC 19 / LEC 20 SEAM — map corrected above. The real split is:**
**Lec 19 = what KL *is* and which direction you are in · Lec 20 = where the formula *comes from*.**
Lec 19's own "Next Session" slide defers self-information, entropy and cross-entropy to Lec 20.

1. **Forward vs reverse KL and mode-covering vs mode-seeking belong to Lec 19**, not Lec 20 — three
   full slides of its deck. Any chapter touching KL directionality links to `19-kl-divergence-a.md`.
   > **This matters for the VAE:** the VAE minimises the *reverse* KL, which is mode-seeking — the
   > mathematical root of the blurry-sample complaint Lec 31 uses to motivate GANs.
2. **"Worked discrete KL" is in NEITHER deck.** Lec 20 works self-information, entropy and
   cross-entropy numerically but never computes a KL on numbers. Both chapters supply their own.
3. **The closed-form Gaussian KL is in NEITHER deck.** Lec 20 is wholly discrete (two weather outcomes,
   $\log_2$, bits). Nothing in Week 3 derives $-\tfrac12\sum(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$.
   **Lec 22 and Lec 24 have been messaged mid-flight to own it rather than link for it.** Lec 50/51
   authors: same warning applies to you.
4. **Lec 19 has zero worked numericals** — entirely conceptual. **Lec 20 has 3 blocks / 7 values, all
   verified** (I(Sunny)=0.193 bits, I(Rainy)=3 bits, H=0.544/1/0 bits). No errors in Lec 20 at all.
5. **Lec 16 had 5 blocks / 15 values, all verified** — the full AE forward pass reproduced in NumPy.

**Deck defects:**

6. **Lec 16 p-08 rounding slip:** the five displayed squared residuals sum to 0.36897, but the next line
   reads 0.36896. The *total* is correct (exact sum 0.3689602); the displayed addends are what don't
   reconcile. MSE 0.07379 is right either way.
7. **Lec 16 uses a single shared scalar bias per layer** ($b = 0.01$ for all four neurons), not one bias
   per neuron. Changes the parameter count to 68 where the standard layout gives 80, and diverges from
   PyTorch. Later authors: check whether your deck does the same.
8. **Lec 19 p-04 axis mislabel** — the upper half of the vertical axis reads $-1,-2,-3,-4$. Cosmetic.
9. **Lec 20 presents the cross-entropy example (p-06) before the definition (p-07).** Lec 20's chapter
   reverses the order.
10. Week 3 has **no hands-on session**; the VAE implementation is deferred to Week 4.

### Errata batch 6 — from the Lec 21/22 author

**Mid-flight correction resolved.** Lec 22 now derives the closed-form Gaussian KL itself in 4 steps,
gives both the $\sum$ and $-\tfrac12\sum(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$ forms, names every term,
and supplies the general two-Gaussian expression. **No chapter links upstream for it.** Lec 50/51
authors: it is now available at `22-elbo-and-vae-loss.md` — link there, do not re-derive.

**Seam findings (map boundaries are soft, one is a real duplication):**

1. **The intractable evidence is taught TWICE** — Lec 21 p-7 and Lec 22 p-5 both state Bayes' rule for
   $p(\mathbf{z}\mid\mathbf{x})$ and both write $p(\mathbf{x})=\int p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})d\mathbf{z}$
   with the same "intractable" verdict. **Settled:** Lec 21 owns the intractability argument in full;
   Lec 22 uses p-5 only for the prior-density *weighting* of the integral and the first statement of the
   joint prior $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$.
2. **The decoder is NAMED on Lec 21 p-6** ("Likelihood distribution (Decoder)") though Lec 22 teaches it.
   Kept to one table row in Lec 21.
3. **The KL term is NAMED on Lec 21 p-9**, in bold with the lecturer's handwritten two-Gaussian sketch,
   though Lec 22 owns the loss containing it. Flagged forward, not taught.
4. **Lec 21 → Lec 22 → Lec 23 seams otherwise clean.** Lec 21 genuinely stops at the intractable
   posterior; Lec 22 genuinely ends by naming the reparameterization trick without teaching it.

**Deck defects:**

5. **NEITHER DECK EVER STATES THE COVARIANCE IS DIAGONAL**, though every formula in the VAE arc assumes
   it — the per-dimension KL sum is only legal because of it. Lec 22 states it explicitly.
   **Every later author writing a diagonal-Gaussian formula should do the same.**
6. **Lec 22 p-8 bracket typo:** $L_{KL}=\frac12[\mu^2+\sigma^2-\log\sigma^2-1)$ — opens `[`, closes `)`.
   The formula itself is correct.
7. **Lec 22 p-8 content defect:** the "Reconstruction" arrow points only to *mean squared error*, but
   the deck's own p-3 defines a **Bernoulli** decoder, which gives BCE. Closed in Lec 22
   (Gaussian ⇒ MSE up to a constant; Bernoulli ⇒ BCE exactly), linking to Lec 11.
8. **Lec 22 p-7/p-8 render $\phi$ as $\emptyset$** ($q_\emptyset$) — a font defect, silently corrected.
9. **Lec 22 p-8 gives the KL closed form per-dimension with no summation sign**, and only against
   $\mathcal{N}(0,1)$.
10. **Neither deck derives the ELBO.** p-6 motivates it in prose, p-7 writes it down. Lec 22's 7-step
    derivation and its two-KL discrimination table (slack-KL vs loss-KL) are written from scratch and
    exist nowhere on the slides.

### Errata batch 7 — from the Lec 23/24 author

**RULINGS (binding on all remaining chapters):**

1. **The shared-scalar-bias convention is LECTURER-WIDE, not a Lec 16 quirk.** Both Lec 16 and Lec 24
   add a single scalar bias to every unit in a layer ($b_1 = 0.1$ to all three hidden units, etc.)
   rather than one bias per neuron. It does not change the decks' arithmetic but it changes parameter
   counts — Lec 24: **47 under the deck's reading vs 56 conventional**; Lec 16: 68 vs 80. **If your deck
   does this, work the count both ways and flag it.**
2. **Latent-dimension symbol.** Lec 22 writes the Gaussian KL sum as $\sum_{j=1}^{d}$ with $d$ = latent
   dimension; Lec 24's deck forces $d = 4$ = *input feature count*. **Ruling for chapters not yet
   written: $d$ = input/feature dimension, $k$ = latent dimension.** Lec 24 already carries an explicit
   note reconciling the two; do not re-litigate, just follow the ruling going forward.
3. **The "nothing derives the Gaussian KL" warning is now STALE and must not be repeated.** Correct
   statement: *no deck* derives it; **`22-elbo-and-vae-loss.md` supplies the derivation** as
   beyond-the-slides content. Link there.

**Deck defects:**

4. **Lec 24 mixes reduction conventions inside one total** — reconstruction is a *mean* over 4 features
   (÷4), KL is a *sum* over 2 latent dims (no ÷2). This silently rescales the KL's weight 4×. Total is
   10.063 as written vs **39.196** under a consistent sum-sum convention. A strong MCQ trap, and it is
   the same knob β-VAE turns (relevant to [Lec 26](26-beta-vae.md)).
5. **Lec 23 p-8's "Quick Summary" chain rule omits the KL branch entirely** — it routes
   $\partial\mathcal{L}/\partial\mu_\phi$ only through $\hat{\mathbf{x}}$ and $\mathbf{z}$, but the KL
   term depends on $\mu_\phi$ and $\sigma_\phi$ *directly*. Taken literally the deck's
   $\partial\mathcal{L}/\partial\phi$ is incomplete. Lec 24 N9 computes both branches and shows the
   reconstruction pull and the KL pull **opposing each other** on latent dimension 2.
6. **Lec 23 derives the trick in $\sigma$ but the encoder emits $\log\sigma^2$** — the factor of $\tfrac12$
   in $\sigma = e^{(\log\sigma^2)/2}$ is never reconciled on the slides.
7. **Lec 24 rounds mid-calculation** (z to 3 dp before the decoder, $\hat{\mathbf{x}}$ to 3 dp before the
   loss). Unrounded gives 9.7121 / 10.0637 against the slide's 9.7111 / 10.0631. **Reproduce the deck's
   figures in an exam.**
8. **Lec 24 p-6 writes $\epsilon = \mathcal{N}(0,1)$** with an equals sign where it means $\sim$, and
   introduces $\epsilon^{(2)}, \epsilon^{(3)}$ before $\epsilon^{(1)}$.
9. **Lec 23 p-3 gives the Gaussian KL without the sum over latent dimensions**; Lec 24 p-12 restores it.
10. **No gradient step anywhere in Lec 24** — the deck computes a loss and stops. Supplied as owned
    content ($\partial\mathcal{L}/\partial W_3[1,1] = -1.6335$, loss 9.7111 → 9.6697).
11. **Lec 24's input $\mathbf{x} = [5.1, 3.5, 1.4, 0.2]$ is the first Iris sample** — the deck never says
    so, but it explains the continuous data and the linear output activation.

**Forward pointer:**

12. **>>> Lec 46 author: the DDPM forward jump IS the reparameterization trick. <<<**
    $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ is literally
    $\mu + \sigma\odot\epsilon$. **Link back to [Lec 23](23-reparameterization.md) rather than
    re-motivating it from scratch.** This connection is worth stating explicitly — it is the single
    cleanest bridge between the VAE arc and the diffusion arc.
13. **Log-base trap priced concretely:** evaluating the Gaussian KL's log term in base 10 gives 0.6441
    instead of 0.3517 — a **1.83× error, not a clean factor of 2.303**, so it cannot be spotted by eye.
    Substituting $\sigma$ for $\sigma^2$ gives KL = $-0.0618$; **a negative KL is the giveaway.**

### Errata batch 8 — from the Lec 26/27/28 and Lec 31/32 authors

**RULINGS (binding on remaining GAN chapters):**

1. **This course reverses the VAE's parameter symbols for GANs:** $\phi$ = **generator**, $\theta$ =
   **discriminator** — the opposite of the VAE arc, where $\theta$ was the generative decoder and $\phi$
   the encoder. Lec 31, 32 and 33 all carry the deck's convention and trap the clash. **Remaining GAN
   authors: do the same.**
2. **Loss notation:** use $\mathcal{L}_D$, $\mathcal{L}_G$, $\mathcal{L}_{\text{real}}$,
   $\mathcal{L}_{\text{fake}}$ (CONTRACT §3's script $\mathcal{L}$), not the decks' $L_D$/$L_G$.
3. **Equilibrium numbers, very likely MCQ:** at $D = \tfrac12$, $\mathcal{L}_D = 2\log 2 = 1.3863$ and
   $\mathcal{L}_G = \log 2 = 0.6931$. Not on the Lec 32 slides but derived there (N5), and its code
   converges to exactly those values. **Lec 34 should use them.**
4. **Interpolation parameter is $\alpha$** in Lec 28's deck (not $\lambda$). Note the three-way collision
   with Lec 04's AdaGrad accumulator $\alpha$ and Lec 46's diffusion $\alpha_t$ — all three are live in
   this book and all three are different things.

**Map additions:**

5. **Lec 26 silently owns the capacity / β-annealing arc** (its pp. 12–14): the "6 factors, 3 latent
   dimensions" squeeze with its two mutually exclusive outcomes, and three named fixes — larger latent,
   moderate $\beta$, and **β annealing on the schedule 1 → 2 → 4**. **β goes *up*, unlike a learning
   rate** — a clean MCQ trap. **Lec 51's author:** the guidance scale trades fidelity against diversity
   in exactly this shape; the parallel is worth drawing.
6. **Lec 31 owns posterior collapse** — a full page (p-6) including the decoder split (autoregressive
   decoders collapse easily, CNN decoders do not). No other chapter covers it. **Do not confuse it with
   mode collapse** (Lec 34); both names are now live in this book.
7. **Lec 26 contains no disentanglement metrics** — every claim is made visually from a latent-traversal
   grid. If asked "how is disentanglement measured here?", the answer is *visually, by traversing one
   latent at a time*. The real metrics are named in Lec 26's *Beyond the slides*.
8. **Lec 28's deck never makes the AE-vs-VAE interpolation argument**, covers neither slerp nor latent
   arithmetic. All three are owned, clearly-sourced beyond-the-slides content there.
9. **The lecturer's own outline slides put "GAN Training Procedure" under Lec 33's session**, while the
   map (correctly, pedagogically) gives the loop to Lec 32, whose pp. 3/5/6/9 *are* the loop drawn.
   Recorded so nobody "fixes" it later.
10. **The forger/detective analogy is NOT on the Lec 31 slides** — written as assigned and marked
    off-slide.

**Deck defects:**

11. **Lec 26's $\sigma^2 = 1$ "for simplicity" is load-bearing and never flagged.** All three of its KL
    examples silently use the shortcut $D_{\mathrm{KL}} = \tfrac12\sum\mu^2$. Wrong by a **factor of 12**
    on a case with $\sigma^2 = (0.25, 4)$. High trap value.
12. **Lec 26 pp. 10 and 11 are near-duplicate slides** — identical layout and prose, only the means
    differ. Both arithmetics reproduced.
13. **Lec 27's numerical is unreproducible** — the deck never publishes a weight matrix. Verified for
    internal consistency instead (the cVAE−VAE shift is exactly the row of $W_y$ a one-hot selects).
14. **Lec 27 p-09 reconstructions contain negative values**, so the decoder output is **linear** and the
    loss must be **MSE, not BCE**, by Lec 11's rule. The deck never says so.
15. **Lec 28's endpoints don't hang together:** $\mu_B = [-1.101, -0.993]$ but $z_B = [-0.200, 1.100]$,
    a ~2σ departure, and $z_B$ is suspiciously round. Treat as given data.
16. **Lec 32 p-10 rounds then adds:** $\mathcal{L}_D = 0.328$ where the exact value 0.328505 rounds to
    **0.329**; and 3.218 where exact 3.218876 rounds to **3.219**. A 3-dp MCQ could offer either.
17. **Lec 32 pp. 3/6 render $\phi$ as a slashed empty-set glyph** — the same font defect as Lec 22.
18. **Lec 31 p-7 garbled clause**, silently corrected.
19. **Iris again:** Lec 27's $\mathbf{x} = [5.1, 3.5, 1.4, 0.2]$ is the first Iris sample, as in Lec 24.
    The decks never name the dataset.

**Tooling:**

20. `validate.py`'s figure regexes now allow one level of nested `[]` inside alt text, so vectors like
    `[5.1, 3.5]` in a caption no longer make a figure invisible to the validator.

### Errata batch 9 — from the Lec 38/39 and Lec 33/34 authors

**>>> THE `y` / `X` / `Y` COLLISION — three senses in one week. <<<**

| Deck | Meaning |
|---|---|
| **Lec 38** (cGAN) | `y` is the **condition** — a class label. `D(x,y)` = "image + its label". |
| **Lec 39** (Pix2Pix) | `x` is the **source image** (the condition), `y` the **target image** (ground truth). `D(x,y)` = "source + target". |
| **Lec 40** (CycleGAN) | `X` and `Y` are **domains**. |

All three chapters carry a boxed notation warning and an MCQ trap. Lec 40 was messaged mid-flight.

**Map corrections:**

1. **Lec 34's deck supplies ~15% of what the map assigns it.** Of 8 pages, 4 are content, and those 4
   carry *only* the convergence narrative and the Nash-equilibrium definition. **The optimal
   discriminator, $-\log 4$, Jensen–Shannon, saddle-point geometry, mode collapse and training
   instability are all ABSENT from the slides** — six assigned topics with zero slide support, all
   written as owned content. It is the longest chapter in the book for that reason.
2. **Jensen–Shannon is owned by Lec 34, not Lec 33** (it needs $D^*$, which is Lec 34's).
   `19-kl-divergence-a.md`'s forward reference **has been re-pointed to `34-gan-convergence.md`.**
3. **Neither Lec 38 nor Lec 39 contains a single worked arithmetic step.** Lec 38 is wholly
   conceptual; Lec 39 prints bare hyperparameters only ($\lambda = 100$, $\eta = 2\times10^{-4}$,
   $\beta_1 = 0.5$, patch sizes 1/16/70/286). All 12 numericals in that pair are constructed.
4. **Lec 33 is clean** — 3 blocks, 14 values, all correct, and its BCE is correctly bracketed where
   Lec 02 p-9's is not. **The deck is unambiguously natural log** ($\log 0.1 \approx -2.303$).
5. **`lec33/` is padded, `lec34/` is unpadded** — a second instance of the batch-4 warning, in adjacent
   decks of the same week.

**Content the decks omit that the chapters now own:**

6. **The saturating/non-saturating split breaks the theory–practice link.** The $-\log 4$ / JS theorem is
   proved for the *saturating* minimax loss; every implementation runs the *non-saturating* one. Same
   fixed point, different objective. A reader who memorises "a GAN minimises JS divergence" and then
   opens any GAN codebase will be confused. Stated in both Lec 33 and Lec 34.
7. **Pix2Pix has no noise vector at all** — the deck silently drops the noise box and writes $G(x)$,
   never $G(x,\mathbf{z})$. So it is near-deterministic and models $p(y\mid x)$ as a point, not a
   distribution. A real structural difference from the cGAN one lecture earlier.
8. **PatchGAN's "70×70" is a derived receptive field, not a crop size** — one fully convolutional pass,
   not diced tiles. Derived in Lec 39 from Lec 05's output-size formula, along with the 30×30 decision
   grid. **Lec 39 p-09 also shows the 286×286 whole-image variant scoring *lower* than 70×70** —
   "bigger receptive field is better" is a natural assumption and it is wrong here.
9. **Lec 38 never shows how the condition physically enters either network** — the deck draws an arrow.
   All three mechanisms (one-hot concatenation, label embedding, spatial channel broadcast) are
   reconstructed as owned content with parameter counts.
10. **Lec 38 p-14's objective drops the label from $D$** — it prints $\log(1 - D(G(z,y)))$, contradicting
    its own p-12 and the entire thesis of the lecture. Corrected in the display and trapped.
11. **Lec 33 p-11 writes $V_G(\phi;\theta)$ for the whole value function**, including the real-data term
    the generator cannot touch — its own prose contradicts its notation one line later.
12. Typos: Lec 38 p-05 "MNIT", Lec 39 p-02 "satndard", Lec 33 p-3 "for a real samples". Lec 38 p-17 and
    Lec 39 p-16 are **bare "Summary" title cards with no summary on them**.

**Tooling:**

13. **Deck-transcribed Keras is now fenced as ```keras, not ```python**, in Lec 12 (4 blocks) and Lec 13
    (1 block), so `run_code.py` no longer tries to execute slide transcriptions that need TensorFlow.
    **All 28 chapters' Python now runs clean, 0 failures.** If you transcribe framework code from a
    slide that the reader cannot run, fence it as ```keras — keep ```python for your own runnable
    NumPy/PyTorch.

### Errata batch 10 — from the Lec 40/41 author

**CONTRACT §4 AMENDED — no square brackets inside alt text.** A literal `[` or `]` breaks the Markdown
image *silently*: the figure vanishes from the rendered page and from the validator's count with no
error reported. Two authors have now hit this transcribing matrices like `[[2,4],[6,8]]` into captions.
**Describe matrices and vectors in words.** (Batch 8 item 20 fixed the nested-bracket case in the
regex; a stray single `]` is ambiguous in Markdown itself and cannot be fixed in the checker.)

**Map corrections:**

1. **Lec 40 has no identity loss** — no third loss term at all. Covered as a flagged gap so a stray MCQ
   option is recognisable.
2. **Lec 40 teaches the log-form adversarial loss, but CycleGAN is actually trained with the LSGAN
   least-squares form.** Flagged; **answer with the log form for this exam.**
3. **Lec 41 never names "progressive growing"** — p-13's resolution ladder is presented as a static
   architecture, not a training schedule. Written as owned content and flagged off-slide. **This matters
   for Lec 42:** removing progressive growing only parses if the reader knows it was a *schedule*.
4. **Lec 41 never mentions style mixing or the truncation trick** ($\psi \approx 0.7$). Both owned,
   both flagged off-slide.
5. **Lec 41 never mentions the loss function at all**, which can mislead. Stated explicitly:
   **StyleGAN is a generator redesign — the loss is unchanged from Lec 33.** Clean exam discrimination
   against Lec 40, whose contribution was a new *loss term*.
6. **Lec 40 has zero worked arithmetic**; its only number is $\lambda = 10$. All six numericals
   constructed. **Lec 41 has one example in two stages, 7 values, all verified correct.**

**Substantive correction to the standard story — worth propagating:**

7. **Cycle consistency does NOT make the solution unique.** If $G$ is any bijection and $F = G^{-1}$,
   all $n!$ scrambled pairings satisfy $\mathcal{L}_{\text{cyc}}$ *exactly*. What breaks the tie is the
   convolutional generators' inductive bias, not the loss. Lec 40 counts this ($10! = 3{,}628{,}800$;
   a $2.76\times10^{-7}$ chance at $n=10$) and says **"encourages", never "guarantees"** — mirroring
   Lec 26's language about $\beta$.

**Deck defects:**

8. **Lec 41 p-13 contradicts p-16** (the paper's own figure, on the same deck). p-13 labels Conv 1–18,
   two convolutions at every resolution including 4×4; p-16 shows the 4×4 block with **one** conv,
   because the constant tensor replaces the first. True counts: **17 convolutions, 18 AdaIN/style
   inputs** — not 18/18. **Teach 18 as the examinable answer** (p-12's text and p-13's labels both say
   so); the discrepancy is documented.
9. **Lec 41 pp. 14–15 round mid-calculation:** with unrounded inputs the AdaIN outputs are $-1.6833$,
   $0.1056$, $1.8944$, $3.6833$, so the slide's $0.1$ and $1.9$ are 2-s.f. values — fine at the deck's
   precision, wrong for a 4-dp question.
10. **Lec 40 writes $L_{cyc}$ on p-12 and $L_{cycle}$ on p-13** for the same term. Normalised.
11. **Lec 41 pp. 03/08 write $z \sim N(0,1)$ for a 512-dim vector**; translated to
    $\mathcal{N}(\mathbf{0},\mathbf{I})$ per CONTRACT §3.

**Numericals worth reusing:**

12. **CycleGAN costs exactly 2× Pix2Pix** — 4 networks / 28.4 M params vs 2 networks / 14.2 M. Human
    labelling effort traded for compute.
13. **StyleGAN's disentanglement machinery is cheap:** mapping network 2,101,248 params, affine $A$ at
    $C=512$ 525,312, constant tensor 8,192 — against 2,359,808 for a *single* 3×3 conv 512→512.
14. **Discriminator-arity tell for the `y` collision:** two arguments ⇒ Lec 38 or 39; one argument ⇒
    Lec 40. CycleGAN's PatchGANs are **unconditional**.

### Errata batch 11 — from the Lec 42/44 author

**Map corrections (applied above):**

1. **Lec 44 owns the Markov property's definition** — its p-16 is a full slide titled "Introduction to
   Markov Chain", listed on the deck's own contents page. **Lec 45 owns the chain applied to the
   diffusion schedule** and links back. Lec 45's author was messaged mid-flight.
2. **>>> Lec 42's deck contains NO path-length regularisation. <<<** Not one slide, not on the contents
   page. Its real contents: limitations of StyleGAN · the water-droplet artefact · weight
   modulation/demodulation · **the phase artefact** · generator skip connections and discriminator
   residual connections. Path-length regularisation is written as clearly-labelled beyond-the-slides
   content with its own numerical. **No chapter should assume it was covered in lecture.**
3. **Lec 42 spends 7 of its 16 content pages on the phase artefact**, which the map's one-line row badly
   under-weighted. Phase/texture-sticking, progressive-growing removal, and the skip/residual changes
   are the *larger half* of that deck.
4. **>>> Lec 42 p-18 now OWNS the residual block $y = F(x) + x$ <<<**, taught from scratch with an
   information-preservation justification. No prior chapter owned residual connections.
   **Lec 49 (U-Net) must link here, not re-derive** — Lec 49's author was messaged mid-flight.
   > **Residual *addition* (channel count unchanged) vs U-Net skip *concatenation* (channel count
   > doubles at the join) is now a live MCQ discrimination**, since both are called "skip connections".
5. **Lec 44's deck goes nowhere near Lec 46/47.** It contains no $\beta_t$, $\alpha_t$, $\bar\alpha_t$,
   $q(x_t\mid x_{t-1})$, $p_\theta$, closed-form jump, loss function or ELBO. Just the two chains as
   arrows, $x_T\sim\mathcal{N}(0,\mathbf{I})$, and the Markov property. The noise-prediction objective
   is described in words only.
6. **Lec 42 has one worked example** (p-12's upsampling table); **Lec 44 has zero.**

**Content gap worth propagating to all diffusion authors:**

7. **The decks say "add noise" and never mention the signal is simultaneously SHRUNK.** A reader
   following the slides would expect variance to grow without bound rather than converge to
   $\mathcal{N}(0,\mathbf{I})$. The shrink-and-add structure of
   $\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\epsilon$ must be made explicit.

**Deck defects:**

8. **Lec 44 p-4 self-contradicts on a single slide** — a blue box says autoencoders are "used for image
   generation" while red text on the same page says they "cannot generate realistic new images". **The
   red text is correct** (Lec 16).
9. **Lec 42 p-12 prints 0.66 for $\tfrac23$** — truncated, not rounded; 0.67 is correct to 2 d.p.
10. **Lec 44 p-12 writes the forward chain's terminus as lowercase $x_t$** while the reverse chain
    beside it starts from uppercase $x_T$. The capital is correct.
11. **Lec 44 p-16's Markov formula is mangled by the equation editor** — subscripts rendered as
    subtractions (`P(xt|xt − 1, xt − 2, ….x0)`).
12. **Lec 44 pp. 13/15 render the Gaussian's comma as a raised apostrophe**, $\mathcal{N}(0\,'\,I)$.
13. **Bare "Summary" dividers with no summary on them** now confirmed on Lec 38 p-17, Lec 39 p-16,
    Lec 42 p-19 and Lec 44 p-17. **This lecturer delivers summaries verbally.** A slides-only reader
    gets no recap from any of these lectures — the Must-memorise tables are the replacement.

**Notation:**

14. **StyleGAN's style vector $\mathbf{w}\in\mathcal{W}$ collides with the convolution weight $w_{ijk}$**
    throughout Lec 42. Convention: **style bold, weights plain indexed scalars.**

### Errata batch 12 — from the Lec 45/47, 48/49, 50/52, 53/54, 55/56 and 57/59 authors

**>>> THE ATTENTION PROVENANCE INVERSION — map corrected above. <<<**
**Lec 57's deck contains NO attention content at all.** Its pages are: motivation · encoder stack (N=6) ·
positional encoding (its only numerical) · encoder diagram · residual + layer norm · position-wise FFN.
No Q/K/V, no $\mathrm{softmax}(\mathbf{QK}^\top/\sqrt{d_k})\mathbf{V}$, no multi-head mechanics, no
$d_k = d_{\text{model}}/h$. **The attention slides are on Lec 56 pp. 8–11** (and the formula reappears on
Lec 59 p-5 inside the cross-attention box). **No slide anywhere in this course derives the $\sqrt{d_k}$
scaling, works an attention numerical, or states $d_k = d_{\text{model}}/h$.** Lec 57 owns all of it as
written-from-scratch content; Lec 56 names it and points forward. No duplication — verified both ways.

**New owners the map never anticipated:**

1. **Lec 59 owns BPE and WordPiece** (its pp. 9–11, with a worked merge trace). Lec 54's deck **names no
   tokenization algorithm at all**, so without this they would have fallen through the gap entirely.
   Discrimination: **BPE = frequency, WordPiece = likelihood.**
2. **Lec 59 p-13 is the ONLY place in the entire course that states attention's $O(n^2)$ cost.**
3. **Lec 59 p-7 settles the head count in bold:** 8 heads in the original Transformer, 16 in Transformer
   Big, and **the heads operate in parallel inside a *single* attention sublayer — they are NOT separate
   sublayers.** Near-certain MCQ.
4. **Lec 48 silently owns the cosine schedule** (offset $s=0.008$, the $[10^{-4}, 0.999]$ clamp), **SNR**
   $=\bar\alpha_t/(1-\bar\alpha_t)$, and the **$[0,1]\to[-1,1]$ data-scaling convention**. Lec 52/53 link.
5. **Lec 45 owns the entire Gaussian/covariance review** — five of its ten content pages, and where the
   lecturer puts his one worked example. The map's "closed form for $\mathbf{x}_t$" row was wrong:
   **Lec 45 has no $\alpha_t$, no $\bar\alpha_t$ and no closed form**; those arrive on Lec 46 pp. 7/9.
6. **Lec 52 owns the concatenation-vs-cross-attention switch:** spatial conditions (segmentation maps,
   low-res images) are **concatenated**; text is injected by **cross-attention**.
7. **GroupNorm is owned by nobody** — written into Lec 49 as beyond-the-slides.

**>>> NOTATION COLLISIONS — all confirmed on-slide. <<<**

8. **The $\mathbf{C}$ collision is real and in this lecturer's own decks.** Lec 55 p-9 writes lowercase
   $c_t$ for the cell state and p-10 writes capital $C_t$; **Lec 56 p-6 then writes capital $C_t$ for the
   attention context.** Same mark, two meanings, adjacent lectures. CONTRACT §3's pin
   ($\mathbf{C}_t$ = cell state, $\mathbf{c}_t$ = attention context) holds; both chapters trap it.
9. **Lec 55 SWAPS $\mathbf{V}$ and $\mathbf{W}$ relative to Goodfellow:** this deck has $\mathbf{U}$
   input→hidden, **$\mathbf{V}$ recurrent**, **$\mathbf{W}$ hidden→output**. Goodfellow and `DLforNLP`
   use $\mathbf{W}$ recurrent, $\mathbf{V}$ output. **Answer with the deck's convention.**
10. **The $s$ vs $w$ guidance-scale collision.** Lec 51's deck writes
    $\hat\epsilon = (1+s)\epsilon_c - s\epsilon_\varnothing$; every library writes
    $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$. **Same formula with
    $w = 1+s$.** The deck's $s=3.0$ is the library's `guidance_scale=4.0`; SD's default 7.5 is a $w$.
    **The meaning of zero flips:** $s=0$ is plain conditional, $w=0$ is unconditional.
11. **$\phi$ now carries four meanings** — U-Net feature map (Lec 52), classifier params (Lec 50), VAE
    encoder (Lec 21), GAN generator (Lec 31–34) — plus the recurring font defect rendering it as
    $\varnothing$, which in Lec 51 **genuinely means the null token.**

**Deck defects (high-value):**

12. **Lec 51 p-06 has a SIGN ERROR in its boxed key equation.** Its own preceding line gives
    $\log p(y|x_t) = \log p(x_t|y) + \log p(y) - \log p(x_t)$, but the boxed gradient prints a **plus**
    where it must be a **minus**. The difference *is* CFG's guidance signal, and the deck's own p-07
    implements a difference. **The most consequential defect in Week 8.**
13. **Lec 46 p-8's boxed conclusion is wrong:** $x_2 = \bar\alpha_2 x_0 + (1-\bar\alpha_2)I$ — both square
    roots dropped and $\mathbf{I}$ written where $\epsilon$ belongs. The handwriting above it and p-9 are
    both correct.
14. **Lec 47 pp. 6/8 have an ambiguous radical** in the posterior mean — the bar appears to cover
    $\bar\alpha_{t-1}\beta_t$ jointly; it must cover $\bar\alpha_{t-1}$ only. Settled *numerically*:
    the correct reading agrees with the deck's own p-9 noise form to 15 digits (1.830110); the wrong
    reading gives 2.677689, a **46% error**.
15. **Lec 50 p-11 labels the DDIM update as "the DDPM reverse update"** — it re-noises with the predicted
    $\hat\epsilon$, not a fresh draw.
16. **Lec 53's `FAST_MODE` silently breaks the terminal-noise assumption.** Shortening $T$ from 1000 to
    100 while keeping $\beta \in [10^{-4}, 2\times10^{-2}]$ leaves $\bar\alpha_T = 0.3636$, so $x_T$ still
    carries 60.3% of the image — yet the sampler starts from $\mathcal{N}(\mathbf{0},\mathbf{I})$.
    Proven with an *oracle* predictor so the network cannot be blamed: 12.5% bias at $T{=}100$.
17. **Lec 54's GloVe slide makes a claim that is false on its own matrix.** "Higher co-occurrence values
    indicate stronger semantic relationships" — row cosines on its page-13 numbers rank **exactly
    reversed** (highest count 250 → lowest similarity 0.2177; lowest count 75 → highest 0.7147). The
    analogy also fails on raw counts: (King−Man)+Woman has cosine **−0.0044** with Queen.
18. **Lec 57 p-8 transcribes 0.9988 for 0.9998** in the final-embedding line, though the assembled vector
    above it is correct. Nastily, 0.9988 is a real value from the same table ($\cos 0.05$).
19. **Lec 48 p-13 exports raw Colab DataFrame *metadata* JSON instead of the rendered table**, so which
    method produced which mean is unrecoverable from the PDF. Resolved by re-running at seed 42.
20. **Lec 48's prose says CIFAR-10; its code loads scikit-image's `astronaut`.** Stale text.
21. **Lec 53 and Lec 48 export with broken LaTeX** — display maths renders as literal `[ ... ]`.
22. **The bare "Summary" title card is this lecturer's template**, now confirmed on Lec 38, 39, 42, 44,
    49, 50, 51, 52, 54, 55, 56, 57 and 59. A slides-only reader gets no recap from any of them.

**Seams and conventions:**

23. **0-based vs 1-based timesteps:** the notebooks are 0-based (`reversed(range(T))` → 99…0), the maths
    1-based ($t = T\ldots1$). Lec 48 and Lec 53 both box a warning.
24. **Lec 48 and Lec 49 disagree on time conditioning** — Lec 48's code uses a *learned*
    `nn.Embedding(1000, 64)`; Lec 49 teaches the *sinusoidal* formula. Both flagged; sinusoidal is the
    DDPM paper's. Likely MCQ.
25. **Embeddings are retroactively owned:** Lec 53's `nn.Embedding(11,128)` class-conditioning table is
    *the same object* as a word embedding. The reader has been using embeddings since Week 6.
26. **Static vs contextual embeddings:** Lec 54 p-11 lists BERT/GPT beside Word2Vec/GloVe as if they were
    variants. They are a different *kind* of object — vector per *occurrence*, not per word *type*.
    **>>> Lec 60/61 authors must carry this contrast. <<<**
27. **Carried attention example, verified — Lec 60/61 should reuse it:** 3 tokens, $d_{\text{model}}=4$,
    $d_k=2$; $\boldsymbol{\alpha}$ row 1 $= (0.045388, 0.767918, 0.186694)$; masked row 2
    $= (0.944193, 0.055807, 0)$. Attention sublayer is $4d_{\text{model}}^2$ params **for any $h$** —
    multi-head is free. Encoder block 3,150,336; decoder block 4,199,936; ratio exactly **4/3**.

### Errata batch 13 — from the Lec 60/61 author

1. **>>> Lec 61's deck has NO zero-shot / one-shot / few-shot content and no in-context learning. <<<**
   It also never gives GPT-3 a single number despite putting it on its timeline — no 175 B, no
   2048-token context. The map assigned ICL to Lec 61 with zero slide support; written as clearly
   labelled off-slide content. **Lec 62 and the Week 11 part-a authors were messaged mid-flight**; if
   Week 11's deck teaches ICL properly, that deck is the course's primary source for it.
2. **>>> Neither Lec 60 nor Lec 61 gives any model configuration. <<<** Lec 60 never states $L$, $H$,
   $A$ or the 110 M parameter count — **the most likely numerical MCQ about BERT is unanswerable from
   these slides.** Derived from parts: **BERT-base = 109,482,240** (the exact published figure, built
   from Lec 57's $4d_{\text{model}}^2$), GPT-1 = 116,169,216 against the deck's "117 Million",
   GPT-2 = 1,557,304,000 against its "1.5 Billion".
3. **>>> BERT and GPT use LEARNED position embeddings, not Lec 57's sinusoidal encoding — and neither
   deck says so. <<<** The decks write "Position Embedding" throughout, never "positional encoding".
   This **contradicts Lec 57's only numerical**, and it is load-bearing: a learned table is *the reason*
   for the hard 512-token ceiling — there is no row 512 to look up. Use that as the reason for context
   limits anywhere they come up, not "a safety limit".
4. **Deleting cross-attention makes a decoder-only block parameter-identical to an encoder block.**
   GPT-1 and BERT-base blocks are both exactly **7,084,800** at $d_{\text{model}} = 768$; Lec 59's 4/3
   encoder:decoder ratio returns to 1:1. **The entire architectural difference between BERT and GPT is a
   triangular matrix of $-\infty$ that costs nothing.** Strong MCQ.
5. **Supervision density is a real, unstated asymmetry:** BERT scores 15% of positions per forward pass,
   GPT scores 100% — a ~6.7× difference. It is the thread connecting ELECTRA (on Lec 60 p-15, which
   exists precisely to fix it) to the post-2020 shift to decoder-only models. Neither deck draws it.
6. **Errata 12.26 closed.** Static vs contextual computed on Lec 60's *own* two "bank" sentences:
   cosine **1.000000** under Word2Vec (by construction — there is only one vector) against
   **−0.042329** under BERT. Framed as a **category error, not a quality difference**: Word2Vec *is* a
   table; BERT is a function from (word, sentence) to a vector.
7. **Errata 12.27 extended** — the carried attention example now quantifies the leakage argument:
   bidirectional row 1 puts **0.954612** of its mass on tokens to its right, causal puts **0**; total
   future mass 1.141306 vs 0.
8. **Lec 60 has an unresolved internal contradiction:** p-10 teaches NSP as a core objective, p-15 then
   says RoBERTa *removes* NSP and improves on BERT. The deck leaves it standing. Closed in the chapter
   (NSP is too easy — random negatives differ in topic, so it learns topic matching, not discourse;
   ALBERT's sentence-order prediction is the fix).
9. **Deck defects:** Lec 60 p-1 writes "…from Transformer" (singular) where p-4 and the paper write
   "Transformers". Lec 61's contents page says "Generative Pre-training (unsupervised)" while its own
   p-8 says "(Self supervised)" — *self-supervised* is the precise term. Bare "Summary" title cards on
   Lec 60 p-16 and Lec 61 p-13.
10. **Zero worked arithmetic in either deck.** Lec 60's only numbers: 15% masking, the 80/10/10 split,
    the 50/50 NSP split, DistilBERT's "about 95%". Lec 61's: the seven series years and the GPT-1 vs
    GPT-2 table (117 M / 1.5 B, ~5 GB / ~40 GB, 512 / 1024 tokens, 40K / 50K BPE).

### Errata batch 14 — from the Lec 62/63 author

1. **>>> THE BOOK CONTRADICTED ITSELF ON "HOW MANY IS FEW-SHOT" — now resolved both ways. <<<**
   Lec 62 p-11 says **2–5**; Lec 61 says **10–100** (GPT-3's convention). Both are now in the reader's
   notes with a reciprocal boxed ruling and an MCQ trap in **each** chapter:
   **answer 2–5 for a question sourced from Lec 62 or phrased as prompt engineering; 10–100 if it names
   GPT-3 or in-context learning.** Likely the single most losable numerical mark in Week 10.
2. **Lec 62 is the ONLY place in the course where zero-/one-/few-shot are taught on slides** (Lec 61's
   deck supplies none — errata 13.1). It is the primary source, not revision. This lecturer also treats
   **one-shot as a first-class technique** with its own definition and applications, unlike both
   companion courses, which fold it into few-shot.
3. **Lec 62 owns three techniques the map never named:** *structured output prompting*, *role
   prompting*, and *prompt quality as a 5+3 checklist* with a vague-vs-specific case study.
4. **Lec 63 silently owns the entire decoding arc** — temperature, greedy, top-$k$, nucleus, the
   greedy-degeneration failure and seed determinism. The map's "the practical pipeline" badly
   under-weighted it; it is the larger half of the notebook and its most examinable part.
5. **Parameter-count reconciliation, so nobody "fixes" either figure:** the Lec 60/61 author's
   **7,084,800** per block at $d_{\text{model}}=768$ counts attention + FFN weights only; adding the
   block's two LayerNorms ($2\times2\times768 = 3{,}072$) gives **7,087,872**. Both are correct.

**Deck defects:**

6. **Lec 62 p-15 refutes its own lesson.** The input states three facts; the JSON response emits
   **five fields** — `"Experience": "5 years"` and `"Specialization": "Machine Learning"` appear nowhere
   in the input. The quoted paragraph's opening `"` is never closed. Either truncation or hallucination;
   either way it disproves "structured output prevents hallucination". **Lec 63 §11 independently
   reproduces the same lesson with receipts** — a perfectly shaped numbered list whose item 1 reads
   "Prostate batteries". *Shape yes, truth no.*
7. **Lec 62 p-07 over-claims:** its verdict says the response "matches the length requested". The prompt
   asked 150 words; the printed response is **79** — 52.7%.
8. **Lec 63 Exercise 2 swaps two variables:** it says "change `block_size` from 32 to 128", but
   `block_size` is **64** and **32** is `batch_size`.
9. **Lec 62 p-11 has no LLM Output box** where pp. 9 and 10 both do; reconstructed.
10. **Lec 63's validation loss measures nothing** — the corpus is ten sentences × 80, so the 10%
    validation tail was seen 72 times in training. Train/val never diverge by more than 0.010.
    **Do not read 0.155 as generalisation.**
11. **Bias artefact in a teaching notebook:** Lec 63 §10's nucleus seed-2 output is *"…because the
    internet is dominated by white people and black people."* Flagged forward to Week 12's bias section.
12. Broken LaTeX export confirmed again (`(k)`, `(p)` for `\(k\)`, `\(p\)`); cell outputs truncated
    across page breaks and stitched back.

**Verification note:** Lec 63's `TRAINING_TEXT` was retyped from the page image and re-run — vocabulary
35, 51,680 tokens and `encode("transformer")` all reproduce the notebook's printed values exactly, which
is a hard check that the transcription is character-perfect. Tiny model **413,987** and DistilGPT-2
**81,912,576** parameters both derived from components and matched exactly.

### Errata batch 15 — from the Week 11 author

1. **>>> PART-A HAS NO RAG CONTENT AT ALL. <<<** Its contents page promises §1.3 Retrieval Augmented
   Generation (principle *and* advances); **the deck never gets there.** Its last content page (p-31) is
   LoRA; p-32 is a Q&A divider, p-33 is "Thank You". Zero RAG slides in 33 pages. The map's provisional
   split was wrong: **the whole of §1.3 is in part-b.** Resolved — ch64 writes RAG's principle as
   flagged off-slide content (it alone can write the ICL/LoRA/RAG comparison, owning the other two
   routes); ch68 gets every part-b figure and all mechanics and advances.
2. **Real boundaries.** Part-a: pp. 3–18 §1.1 Recap · pp. 22–27 ICL · pp. 28–31 LoRA. **Part-b has no
   contents page at all**, no section numbers, no summary: pp. 2–12 ≈ §1.3.1, pp. 13–18 ≈ §1.3.2.
3. **Part-b silently owns Agentic AI** (pp. 13–15: the ReAct loop Goal → Agent → Action → Observation,
   tool choice across vector store / web search / SQL-API, a six-row Standard-vs-Agentic table) **and
   GraphRAG** (pp. 16–18, including **Leiden** clustering by name). No map row anticipated agents
   appearing in this course at all.
4. **Part-b p-10 states a training asymmetry worth memorising:** end-to-end backprop runs through the
   query encoder and the generator; **the document encoder is frozen** because the index is precomputed.
5. **Part-b p-11 says the query vector is the *average* of token representations**, not `[CLS]` —
   contradicting plain BERT and DPR. **Answer with the deck.**
6. **The deck classes zero-shot as a *type of* ICL**, which several textbooks do not. Follow the deck.

**Serious deck errors:**

7. **>>> Part-a pp. 6 and 13 drop the square root from the attention scaling. <<<** Both print
   $\mathrm{softmax}(\mathbf{A}/d)$ and $\mathrm{softmax}(\mathbf{A}_k/d_k)$; it must be $\sqrt{d_k}$.
   **Not cosmetic:** at $d_k = 64$ the deck's denominator flattens the p-9 row to a peak of 0.1579
   against a uniform 0.1429 — it destroys the mechanism. **This contradicts [Lec 57](../notes/57-transformer-encoder.md),
   which owns the scaling and is correct.**
8. **>>> Part-a p-16's decoder-only loss is missing its logarithm. <<<** It prints
   $\mathcal{L} = -\sum_t P(\mathbf{x}_t\mid\mathbf{x}_{<t})$; it must be $-\sum_t \log P$. Wrong sign
   behaviour and a vanishing gradient on confidently-wrong tokens. Both forms worked: 3.2189 → 0.5133
   nats correct, against −1.4 → −2.55 as printed.
9. **>>> Part-a p-29 writes $W = W_0 + \alpha\cdot BA$ with a plain $\alpha$ <<<** where the paper, every
   library, **and this course's own `assets/notebooks/week11.ipynb`** use $\alpha/r$. The notebook sets
   `r=8, lora_alpha=16`, so its effective scale is **2** against the slide's **16** — a factor of exactly
   $r$, and the disagreement is $r-1 = 7$ times the size of the intended update. **The first case in this
   book of a companion notebook contradicting its own deck.**
10. **Part-a p-9 truncates 0.236881 to 0.23** where it rounds to **0.24**; the four displayed values sum
    to 0.99, which is the tell. **p-30's LoRA saving is exactly right** (0.0703125 → "0.0703 (7%)").
11. **Part-b p-18 is titled "Agentic RAG" but its table compares Standard RAG with GraphRAG** — p-15 is
    the real agentic table.
12. **Part-b p-09's "$\approx$" hides a 29% spread**: the top-$k$ sum as written under-estimates
    $P(y\mid x)$, renormalising over-estimates (exact 0.6685 vs 0.6493 vs 0.8641). The deck never says
    which it means.
13. **Part-b p-12's $p_\eta(z\mid x)$ cannot be evaluated at inference** — its denominator sums over the
    whole corpus, the exact cost MIPS exists to avoid. A training-time object; nothing on the slide
    says so.
14. **Part-b has ZERO worked arithmetic** across nineteen pages and four equation blocks — a total
    contrast with the Lec 01–63 lecturer.

**>>> WEEK 11 NOTATION DIVERGES SHARPLY FROM CONTRACT §3. <<<** Both chapters carry a full translation
table. The dangerous ones:

| Quantity | Lec 01–63 | Week 11 |
|---|---|---|
| sequence length | $n$ | **$T$** |
| embedding width | $d_{\text{model}}$ | **$D$** |
| per-head width | $d_k$ | **$d$** (p-6), $d_k$ (p-13) |
| head relation | $d_k = d_{\text{model}}/h$ | **$d = h \times d_k$** — same equation, inverted |
| attention output | $\mathrm{head}_i$, $\mathbf{c}_t$ | **$\mathbf{E}$**, $\mathbf{e}_t$ |
| the reals | $\mathbb{R}$ | **$\mathcal{R}$** in part-a, $\mathbb{R}$ in part-b |
| retriever params | — | **$p_\eta$** — collides with $\eta$ = learning rate |
| retriever embedding dim | — | **$h$** — collides with $h$ = number of heads everywhere else |
| a document | — | **$d$** on p-9, then **$z$** on pp. 10/12; corpus $\mathcal{C}$ then $\mathcal{Z}$ |

$D$ and $d$ are reused with different meanings *inside part-a itself* (embedding/head widths on p-6,
matrix rows on p-29), and $\mathbf{A}$ is the attention score matrix on p-6 but a LoRA factor on p-29.
**$\alpha$ now has five live meanings in this book** (AdaGrad accumulator · interpolation · diffusion
schedule · attention weights · LoRA scale).

**Numbers worth propagating:**

15. Notebook FLAN-T5-small + LoRA $r{=}8$ on q,v: **344,064 trainable / 77,305,216 = 0.4451%**, adapter
    1.31 MiB vs 295 MiB checkpoint. LoRA at $r{=}8$ on GPT-1's Q and V across 12 blocks:
    **294,912 / 116,169,216 = 0.2539%**. **"7%" (one matrix) and "0.25%" (one checkpoint) can both be
    correct answers — read which the question asks.**
16. LoRA break-even rank $r^\star = Dd/(D+d)$: **113.78** for the deck's $1024\times128$; **384** ($=d/2$)
    for a square $768\times768$. **Rank of $\mathbf{BA}$ at initialisation is 0** ($\mathbf{A}=\mathbf{0}$),
    so attaching an adapter never degrades a working model.
17. **MIPS ≡ cosine iff vectors are L2-normalised** — a passage and its doubled copy score 1.86 vs 0.93
    under raw inner product but tie at 0.9903 under cosine. Two-stage retrieval is ~**182×** cheaper than
    cross-encoding a 10k-chunk corpus, and **stage-1 recall is a hard ceiling no re-ranker can lift.**

### Errata batch 16 — from the Week 12 author (final batch)

**Page split chosen, and the deck marks it itself:** ch1 = pp. 1–23 (§2.1 + §2.2), ch2 = pp. 24–38
(§2.3 + §2.5 + §2.6). The deck re-shows its contents page four times with covered sections ticked white:
p-02 (2.1), p-19 (2.2), **p-24 (2.3 and 2.5 go white together)**, p-31 (2.6). So p-24 is the lecturer's
own chapter break and §2.3/§2.5 are one delivered unit. Real boundaries: §2.1 pp. 3–18 · **§2.2 pp. 20–23,
four pages only** (titled "LLM pre-training" on the slides but "Multimodal alignment" on the contents) ·
**§2.3 p-25, ONE slide** · §2.5 pp. 26–30 · §2.6 pp. 32–37.

**Three topics the map assigned with ZERO slide support:**

1. **>>> The deck teaches NO top-k and NO top-p. <<<** Both are named once on the p-17 summary slide and
   never defined. Written from scratch with the discrimination (*k* fixes the **count**, *p* fixes the
   **mass**, only *p* adapts to confidence) and the pipeline order (temperature → top-k → top-p → sample).
2. **>>> Beam search is absent from the entire deck. <<<** Written as owned content; N5 shows greedy's
   output is **1.6× less probable** than beam-2's, plus the identity $B=1 \Rightarrow$ greedy.
3. **>>> CLIP and contrastive alignment are absent. <<<** The deck's whole multimodal route is
   encoder → alignment → next-token prediction (LaVIT). **[Lec 52](../notes/52-stable-diffusion.md)
   explicitly forwarded CLIP's mechanism to "the multimodal material in Week 12"** — so without this the
   book would have had no account of how a text prompt steers a diffusion model. Now written as owned
   content (symmetric row/column cross-entropy, free in-batch negatives, the learned temperature).
4. **>>> §2.3 contains no scaling law at all. <<<** No Kaplan, no Chinchilla, no exponents, no
   $C \approx 6ND$. p-25 is a **September-2026 MoE snapshot** that presupposes the literature and argues
   the size/performance link is **broken by sparsity**. The classical laws are written as compressed
   owned content *plus* the reconciliation, which is the real exam question: **the laws predict *loss*
   for *dense* models; the chart plots a *saturating benchmark* for *sparse* ones.**
5. **§2.6 is mostly agentic security, not bias** — four of seven slides are a news wall, agent-swarm
   anatomy, the July-2026 Hugging Face breach and a regulatory-clock slide. "Bias" proper is two slides.
6. **Hallucination falls through every crack** — titled on p-24, taught on no slide, owned by no chapter.
   Covered in ch2's *Beyond the slides* with the three levers (grounding / abstention / verification).

**Deck defects:**

7. **The contents page skips §2.4** (2.1, 2.2, 2.3, **2.5**, 2.6) on all four contents slides. No content
   missing — a numbering slip never corrected.
8. **§2.6's title disagrees with itself:** p-24 reads "Bias, **Safety and Hallucinations**"; pp. 02/31
   read "Bias, and Safety". Hallucinations are never covered.
9. **The deck stops before normalising the softmax** — p-16 prints unnormalised exponentials
   (148.4, 54.6, 2.7) and never divides by the sum, so **the actual probabilities appear on no slide.**
10. **Index reordering between p-11 and p-13:** p-11 lists cat/dog/car/quantum, p-13 relists as
    car/dog/cat/quantum with indices 1–4, so **"index 3" means cat.**
11. **This deck's BPE corpus differs from Lec 59's** — `low/lower/lowest/slow/slower` reporting a *token
    count*, against Lec 59's `low/lower/lowest/new/newer` reporting a *vocabulary*. Two live traces.
12. **No arithmetic error anywhere in this deck** — 3 blocks / 11 values, all verified. Unusual for this
    course and worth recording. **Deck-implicit rule surfaced:** a BPE merge of frequency $f$ removes
    exactly $f$ tokens (29→24→19→16 against stated frequencies 5, 5, 3).

**>>> WEEK 12 NOTATION — the most dangerous collisions in the book. <<<**

13. **`z` is a LOGIT here.** For eight weeks it was the VAE latent, the GAN noise and the Stable
    Diffusion latent. Boxed warning in both chapters.
14. **`T` is TEMPERATURE** — it was the diffusion step count in Lec 44–48, and on p-23's LaVIT figure
    $T_1, T_2$ are **token counts**, a third live meaning.
15. **`x` = input prompt, `y` = output sequence** — a **fourth** sense of $y$ after Lec 38 (class label),
    Lec 39 (target image) and Lec 40 (domain).
16. **`N`/`D`/`C`** (parameters / dataset tokens / compute) collide with sample count, the GAN
    **discriminator**, and the LSTM cell state. **`k`** is the top-$k$ cutoff in ch1 and pass@$k$'s sample
    count in ch2, against batch 7's $k$ = latent dimension.
17. **The deck uses no bold and no vector notation at all** — every quantity is a plain scalar, including
    vocabulary-wide vectors. Translated to CONTRACT §3 throughout.
18. **Perplexity base trap priced three ways:** cross-entropy 1.381551 nats = 1.993157 bits = 0.600000
    digits, **perplexity 3.9811 in all three** — provided you exponentiate in the base you measured in.
    Mixing gives $2^{1.3816} = 2.60$, the wrong answer waiting to be offered.

**The book's honest gap list** (ch2's closing *Beyond the slides*): no distributed training; RLHF and DPO
named but never explained; efficiency (quantisation, pruning, distillation, KV-cache) absent though
p-25's own argument depends on two of them; MoE **routing** never described; and the sharpest — **the
course derives GAN and diffusion objectives across Weeks 5–8 and never computes an FID or an Inception
Score**, so a reader can train a diffusion model with no way to say whether it is good.
