# Ownership map — who teaches what

30 decks → 30 chapters, 1:1 with the playlist's Lec 1–30. Weeks 9–12 (Lec 31–37: Diffusion I/II,
GAN I/II, LLM I/II, VLM) arrive later and will extend this table.

| Lec | Week | Deck | Chapter file |
|---|---|---|---|
| 1 | 1 | `L1P1_Introduction` | `notes/week-01/01-intro-cv-and-genai.md` |
| 2 | 1 | `L1P2_Generative Modelling` | `notes/week-01/02-generative-vs-discriminative.md` |
| 3 | 1 | `L1P3_Generative Vision Models` | `notes/week-01/03-generative-vision-models.md` |
| 4 | 1 | `L2P1_Linear_Algebra` | `notes/week-01/04-linear-algebra.md` |
| 5 | 1 | `L2P2_Probability-1` | `notes/week-01/05-probability-1.md` |
| 6 | 1 | `L2P3_Probability-2` | `notes/week-01/06-probability-2.md` |
| 7 | 2 | `W2L3_P1_NNFundamentals` | `notes/week-02/07-perceptron.md` |
| 8 | 2 | `W2L3_P2_MLP` | `notes/week-02/08-mlp-and-activations.md` |
| 9 | 2 | `W2L3_P3_BackProp` | `notes/week-02/09-backpropagation.md` |
| 10 | 2 | `W2L3_P4_Overfitting` | `notes/week-02/10-overfitting-and-regularization.md` |
| 11 | 3 | `W3L4_P1_CNN_Basics` | `notes/week-03/11-cnn-basics.md` |
| 12 | 3 | `W3L4_P2_CNN_Architectures` | `notes/week-03/12-cnn-architectures.md` |
| 13 | 3 | `W3L4_P3_CNN_Optimization` | `notes/week-03/13-vanishing-gradients-activations.md` |
| 14 | 4 | `W4L5_P1_ResNet` | `notes/week-04/14-resnet.md` |
| 15 | 4 | `W4L5_P2_ModelArch` | `notes/week-04/15-densenet-mobilenet-efficientnet.md` |
| 16 | 4 | `W4L5_P3_SeqLearning` | `notes/week-04/16-sequence-modelling-and-rnn.md` |
| 17 | 5 | `L6P2_BPTT_RNN` | `notes/week-05/17-bptt.md` |
| 18 | 5 | `L6P3_LSTM` | `notes/week-05/18-lstm.md` |
| 19 | 5 | `L6P4_Seq_Model_2` | `notes/week-05/19-gru-seq2seq-attention.md` |
| 20 | 5 | `L7P1_GenAICV_1` | `notes/week-05/20-genai-vision-tasks-1.md` |
| 21 | 5 | `L7P2_GenAICV_2` | `notes/week-05/21-genai-vision-tasks-2.md` |
| 22 | 6 | `L8P1_GenModel_Intro` | `notes/week-06/22-generative-taxonomy-and-mle.md` |
| 23 | 6 | `L8P2_AutoReg` | `notes/week-06/23-autoregressive-pixelrnn-pixelcnn.md` |
| 24 | 6 | `L8P3_Transformer-1` | `notes/week-06/24-captioning-and-spatial-attention.md` |
| 25 | 7 | `L8P4_Transformer-2` | `notes/week-07/25-qkv-and-self-attention.md` |
| 26 | 7 | `L8P5_Transformer-3` | `notes/week-07/26-encoder-and-positional-encoding.md` |
| 27 | 7 | `L8P6_Transformer-4` | `notes/week-07/27-decoder-and-full-transformer.md` |
| 28 | 8 | `L8P7_VIT` | `notes/week-08/28-vit-detr-swin.md` |
| 29 | 8 | `L9P1_VAE-1` | `notes/week-08/29-autoencoders-to-vae.md` |
| 30 | 8 | `L9P2_VAE-2` | `notes/week-08/30-elbo-and-reparameterization.md` |

Asset tags are `W<week>_<deckname with spaces→underscores>`, e.g.
`assets/figures/W1_L1P2_Generative Modelling/`, `assets/slides/W1_L1P2_Generative_Modelling/`.
(Figures keep spaces, slide renders use underscores — check both.)

---

## Contested topics — pre-resolved

These concepts appear on 3–5 decks each. Exactly one chapter **derives** each; everyone else gets
one sentence and a link. This is the part that keeps the book from looping.

### Attention
| Aspect | Owner |
|---|---|
| Additive (Bahdanau) attention over RNN encoder states; alignment scores; context vector $\mathbf{c}_t$ | **Lec 19** |
| Spatial attention over CNN feature grids for captioning (*Show, Attend and Tell*); soft vs hard | **Lec 24** |
| The general attention layer; $\mathbf{Q},\mathbf{K},\mathbf{V}$ projections; scaled dot-product $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$; why $\sqrt{d_k}$; self-attention; masked self-attention; multi-head | **Lec 25** |
| Positional encoding (sin/cos, learned); encoder block assembly | **Lec 26** |
| Cross-attention; Add&Norm; FFN; decoder block; full enc–dec | **Lec 27** |
| Windowed / shifted-window attention (Swin) | **Lec 28** |

Lec 26 and 27 **must not** re-derive scaled dot-product attention. Lec 25 **must not** cover
positional encoding or Add&Norm.

### Vanishing gradients
| Aspect | Owner |
|---|---|
| Canonical derivation: product of Jacobians, why $\sigma'\le 0.25$ kills depth; the full activation-function remedy set (sigmoid, tanh, ReLU, Leaky/PReLU, ELU); dying ReLU | **Lec 13** |
| Skip-connection remedy; identity mapping; gradient highway $1 + \partial\mathcal{F}/\partial x$ | **Lec 14** |
| Recurrent/temporal form: repeated $\mathbf{W}_{hh}$, spectral radius, exploding vs vanishing, gradient clipping | **Lec 17** |
| Gating remedy; constant error carousel; why $\partial C_t/\partial C_{t-1}=\mathbf{f}_t$ | **Lec 18** |

### Generative vs discriminative / the model zoo
| Aspect | Owner |
|---|---|
| Intuition; $p(y\mid\mathbf{x})$ vs $p(\mathbf{x},y)$; decision boundary vs density; worked Bayes comparison | **Lec 2** |
| One-paragraph survey of VAE / GAN / autoregressive / diffusion / flows — explicitly a map, details deferred | **Lec 3** |
| The formal taxonomy: MLE objective, explicit vs implicit density, tractable vs approximate, where each family sits | **Lec 22** |

Lec 3 must stay a survey — no derivations. Lec 22 owns the taxonomy diagram and MLE.

### Other pre-resolutions
- **Backprop**: Lec 9 derives it end to end. Lec 13, 17 reference only.
- **Overfitting / L1 / L2 / dropout / early stopping / k-fold / bias–variance**: Lec 10 owns all of it.
  Lec 14's "skip connections have a regularizing effect" is one sentence + link.
- **1×1 convolution**: Lec 12 owns (GoogLeNet bottleneck, cost arithmetic). Lec 15's pointwise
  convolution links to it.
- **Conv output-size formula** $\lfloor (H-K+2P)/S\rfloor + 1$: Lec 11 owns and drills it. Everyone
  else just uses it.
- **Seq2seq encoder–decoder**: Lec 19 owns. Lec 17's slide 17 teaser is one sentence + link.
- **Plain autoencoder, bottleneck, why AE is not generative**: Lec 29 owns. Lec 3's teaser is a survey line.
- **ELBO derivation, reparameterization trick, KL closed form for diagonal Gaussians**: Lec 30 owns.
  Lec 29 sets up the problem (intractable posterior) and stops there.
- **Softmax**: Lec 8 owns the definition and its derivative. Lec 25 uses it without re-deriving.
- **Activation functions as a catalogue**: Lec 8 introduces sigmoid/tanh/ReLU as *choices*; **Lec 13**
  owns the gradient-behaviour analysis of each. Lec 8 stays short on the gradient story and links forward.
- **ILSVRC / ImageNet error-rate timeline**: Lec 14 owns the table (it is the deck's own framing);
  Lec 1 and Lec 12 may cite single numbers.

## Errata and cross-chapter findings (discovered during authoring — read these)

### CORRECTED: the Lec 5 / Lec 6 seam (the original map was wrong)

Both authors independently verified the decks. The actual split is:

- **Lec 5** (`L2P2_Probability-1`): sample space, axioms, addition rule, mutual exclusivity,
  conditional probability, multiplication rule, independence, law of total probability, Bayes,
  random variables, PMF/PDF/CDF, **the uniform and Gaussian PDFs (slide 16)**, and
  **mean and variance (slides 17–18)**.
- **Lec 6** (`L2P3_Probability-2`): joint and marginal distributions, **the chain rule**, covariance,
  **the covariance matrix**, correlation, likelihood, **MLE and log-likelihood**, Bayes in hypothesis
  form, **MAP / Bayesian learning**, and the **Bayes optimal classifier**.

There is **no** Gaussian, expectation, variance, entropy or KL content on the Lec 6 deck. Both are
written and verified consistent; no gap exists. **Lec 22 must read the MLE consequence below.**

### MLE ownership, revised

Lec 6 now owns **MLE as a general statistical method** — likelihood vs probability, the product form,
the log-likelihood trick, a Bernoulli MLE worked by calculus, and the MLE↔MAP relationship. It also
states NLL ≡ cross-entropy and MLE ≡ minimising KL, and gives KL's definition, non-negativity and
asymmetry.

**Lec 22 therefore owns MLE only as applied to generative modelling**: fitting $p_\theta(\mathbf{x})$
to $p_{\text{data}}$, and the explicit/implicit × tractable/approximate density taxonomy. Lec 22 must
recall Lec 6's result in two sentences and link, NOT re-derive the log-likelihood.

### Notation hazards

- **The deck writes the Gaussian as $\mathcal{N}(\mu, \sigma)$ — second argument is the STANDARD
  DEVIATION.** Lec 5 flags this. **Lec 29 and Lec 30 must use $\mathcal{N}(\mu, \sigma^2)$ (variance)
  and explicitly note the clash**, or the book contradicts itself.
- The deck overloads `D` as both the dataset and the real data distribution. Per the contract,
  $\mathcal{D}$ = dataset; use $p_{\text{data}}$ and $p_g$ for distributions. Lec 2 established this.

### Errors on the source decks — already flagged in the chapters that own them

- `L1P2` slide 22 reads `P(Y = 🐕 | X = Dog) = 0.95` — both sides are the class. The intended
  statement is $P(Y=\text{dog}\mid X=\mathbf{x})=0.95$. Any chapter quoting slide 22 must quote the
  corrected form.
- `L2P2` slide 9 reports 0.00784 / 0.02976 as probabilities; they are **unnormalised numerators** and
  sum to 0.0376, not 1. The true posterior is 0.209. Do not inherit this habit.
- `W2L3_P1` gives **two inconsistent step functions** — slide 8 fires on $z > 0$, slide 9 on
  $z \ge 0$. Lec 7 adopts **slide 8's convention** ($y=1$ iff $z>0$). **Lec 8 must use the same.**
- `W2L3_P1` never states the perceptron learning rule and never mentions $\eta$. Lec 7 derives the
  rule from the deck's own squared error. Do not expect to find it on a slide.

### Already-introduced material (do not present as new)

- **The autoregressive chain-rule factorisation $p(\mathbf{x}) = \prod_i p(x_i \mid x_{<i})$ already
  appears in Week 1** — on `L1P3` slide 14 (Lec 3 shows it as a figure) and as a named slide on the
  Lec 6 deck. **Lec 22 and Lec 23 must frame it as a return to a known identity, not a first
  introduction.**
- Naive Bayes is worked in full in Lec 2 (it is the canonical generative-model MCQ answer). Lec 22
  should not re-work it.
- The deck lists **Vision Transformer as an autoregressive model** (`L1P3` slide 15), which is loose —
  ViT proper is a discriminative encoder. **Lec 28 should clear this up explicitly.**

### Late ownership resolutions

- **Cross-entropy loss** was unassigned. **Lec 8 owns it** — the softmax+CE pairing,
  $\mathcal{L} = -\log p_c$, and the gradient $\partial\mathcal{L}/\partial z_i = p_i - y_i$.
  Lec 9 references it in one line. Anyone needing CE links to Lec 8.
- **Bias gradients** ($\partial\mathcal{L}/\partial b_i = \delta_i$) are derived in **Lec 9**; the deck
  omits them entirely.
- **SGD vs mini-batch vs full-batch** is named by no deck and owned by no chapter. It is covered as a
  gap note in Lec 9. If your deck touches batching, you may claim it — say so in your report.
- **Dropout and data augmentation are NOT on the Lec 10 deck.** Lec 10 covers them only as
  "Beyond the slides" gaps. Do not link to Lec 10 expecting a full dropout treatment; the deck's
  remedy list is exactly three — k-fold CV, regularization, early stopping.
- **Leaky ReLU slope: the deck uses $\alpha = 0.1$, not the more common 0.01.** Lec 8 and Lec 13 must
  both quote 0.1 as the deck's value.
- **The deck's L2 penalty is $\lambda\sum w_j^2$ with no $\tfrac12$**, which gives a decay factor of
  $(1-2\eta\lambda)$. Lec 10 presents $(1-\eta\lambda)$ as canonical (under the $\tfrac{\lambda}{2}$
  convention) and states the deck's variant alongside. Do not contradict this.
- **The XOR weights printed on `W2L3_P2_MLP` slide 3 do not compute XOR** — verified, all four inputs
  give $y \approx 0.80$. The bias terms are wrong. Lec 8 derives and verifies corrected weights
  ($\mathbf{W}^{(1)} = [[-1,1],[1,-1]]$, $\mathbf{b}^{(1)} = [-0.5,-0.5]$,
  $\mathbf{W}^{(2)} = [1,1]$, $b^{(2)} = -0.5$). Any chapter citing this example uses the corrected set.
- **`W2L3_P3_BackProp` indexes weights source-first**: $w_{ei}^{(p)}$ runs **from** $e$ in layer $p-1$
  **to** $i$ in layer $p$ — the transpose of the usual convention. Lec 9 adopts the deck's order and
  flags it. Lec 17 (BPTT) should check this when recalling Lec 9's results.

### Figure-directory naming (corrected)

Pattern is `W<week>_<deckcode>_<Title with original spaces>`. Example: the figures for `L1P3` live in
`assets/figures/W1_L1P3_Generative Vision Models/` — underscore after the deck code, spaces inside
the title. Slide renders replace **all** spaces with underscores:
`assets/slides/W1_L1P3_Generative_Vision_Models/`. URL-encode spaces as `%20` in markdown paths.

**Pool A is not always better.** On several decks the "extracted figures" are just PNG screenshots of
whole slide bodies, no cleaner than the render. Check before preferring Pool A; use whichever is
actually the better image.

### Length calibration

Finished chapters run **4,000–6,000 words by `wc -w`** (which counts LaTeX tokens, table cells and
code). The exemplar is ~4,300. Word targets in assignments refer to this raw measure. Do not clip
real content to hit a number, and do not pad to reach one.

- **The ILSVRC chart on `L1P1_Introduction` slide 5 has a mislabelled y-axis.** It reads
  "Accuracy (Top-5 error)" but the bars are unambiguously *error* (they fall over time, and the
  figure caption says error). Confirmed values: 2010 28.2 · 2011 25.8 · 2012 AlexNet 16.4 ·
  2013 OverFeat 13.6 / Clarifai 11.7 · 2014 VGG 7.3 / GoogLeNet 6.7 · 2015 ResNet 3.6 · human 5.1.
  **Lec 1 flags this mislabel as an MCQ trap. Lec 12 and Lec 14 must flag it identically** so the
  book does not contradict itself. Use the numbers above verbatim.
- **`L1P1_Introduction` slide 4 carries two separate definitions**, "Generative AI" (the field) and
  "Generative Modelling" (the unsupervised task). **Lec 1 owns and quotes both verbatim.** Lec 2 and
  Lec 22 must NOT re-quote slide 4 — build on it instead.
- **The deck groups super-resolution, denoising, colorization and inpainting under the umbrella term
  "image restoration".** Lec 20 and Lec 21 treat several of these as standalone tasks. Those authors
  should note the deck's grouping explicitly so the taxonomy does not appear to contradict Lec 1.

## Chapter-specific notes for authors

- **Lec 4 (Linear Algebra)** — deck is ~90% equation *images*; the text dump is nearly empty. You must
  read the slide renders. Topics: vectors & operations, matrix ops, transpose, identity, singular
  matrices, inverse (2×2 and larger via adjugate/cofactors), eigenvalues & eigenvectors with the
  geometric interpretation, how to solve $\det(\mathbf{A}-\lambda\mathbf{I})=0$. Slide 23 is a
  bonus slide on singular matrices in ML — keep it, it is good. Connect every item forward to where
  the course uses it (covariance, PCA-flavoured latent spaces, weight matrices).
- **Lec 5 / 6 (Probability)** — ✅ WRITTEN. See the corrected seam in the errata section above; the
  original description of this split was wrong and has been replaced.
- **Lec 7** — perceptron, step function, perceptron learning rule with the $d - y$ update, linear
  separability, XOR failure. Own the XOR argument; Lec 8 resolves it with a hidden layer.
- **Lec 11** — slides 7–14 are a build-up animation of one convolution sliding over an input; collapse
  the whole run into one figure plus a hand-computed example. Slides 24–31 likewise animate pooling
  types. Do not reproduce eight near-identical slides.
- **Lec 13** — slides 24–30 overlap Lec 14's opening (deeper-nets-fail framing). Cover the
  *degradation problem as motivation* and hand the ResNet answer to Lec 14.
- **Lec 16** — slides 19–33 animate the RNN unrolling for one-to-many / many-to-one / many-to-many.
  Collapse per pattern, one figure each, and own the five-pattern taxonomy table.
- **Lec 17** — slides 18–28 are an animated character-level RNN ("hello"-style) train/test walkthrough.
  This is excellent exam material: turn it into one fully worked numerical with real one-hot vectors.
- **Lec 20 / 21** — these two are a survey of 13 vision tasks, each slide pair being
  "task → how GenAI helps". Low on math, high on MCQ-able facts. Structure as a task table plus a
  short subsection each; do not inflate. Lec 20: classification, detection, segmentation,
  super-resolution, inpainting (+ CV challenges: viewpoint, scale, deformation, occlusion,
  illumination, clutter, intra-class). Lec 21: image-to-image translation, medical synthesis, anomaly
  detection, 3D scene generation, face synthesis, style transfer, video generation.
- **Lec 25** — slides 3–17 build the attention layer by incrementally generalizing from the captioning
  case to Q/K/V. Preserve that build as the pedagogy; it is the clearest thing on these decks.
- **Lec 29 / 30** — the VAE math is entirely image-based across ~45 slides. Read every render. Lec 29
  ends at "the posterior $p_\theta(\mathbf{z}\mid\mathbf{x})$ is intractable, so we approximate";
  Lec 30 picks up at variational inference and goes through ELBO → two-term loss → reparameterization.

### Errata found later in authoring (batch 2)

- **`W4L5_P2_ModelArch` slide 14 has a bad worked example.** It concludes depthwise-separable
  convolution is "100 times less" multiplications using $p=q=512$ — which would be a 512×512 *kernel*,
  not a thing. The examinable figure is the 3×3 case at **8–9×**. Do not repeat the 100× number.
- **`L7P1_GenAICV_1` slide 12 mislabels intra-class variation as "interclass variation"** in its body
  text; the title and the described concept are both unambiguously intra-class.
- **`L7P1_GenAICV_1` slide 17's classification bars do not sum to 1** (139% and 143%) despite the slide
  saying "softmax". Flagged in Lec 20, not silently repaired.
- **Terminology collision on "diffusion."** `L7P1` slide 32 calls classical inpainting
  "diffusion-based" (PDE intensity propagation); slides 33–35 use "diffusion models" for the generative
  family. Unrelated methods, same word. **Lec 21 and the Week 9–12 diffusion chapters must not conflate
  them.**
- **No deck in Weeks 1–8 names a single evaluation metric.** IoU, mAP, PSNR, SSIM, mIoU and pixel
  accuracy are all absent from all 37 slides of `L7P1`. Lec 20 sources them from outside the deck and
  says so. Lec 21 faces the same wall — do the same and label it.
- **`W3L4_P1_CNN_Basics` uses $N$/$F$/$K$ for input size / kernel size / output size**, which clashes
  with $F$ = number of filters used everywhere else in the book. Lec 11 flags the clash.
- **`W3L4_P1` slide 18 is a typo** ("32 × 32 × 32 image" where the arithmetic says 32×32×3).
- **`W4L5_P3_SeqLearning` has no bias terms anywhere** — its equations are
  $\tanh(W_h h_{t-1} + W_x x_t)$. Lec 16 adds biases and flags the addition. Lec 17/18 should expect
  bias-free source equations.
- **`W3L4_P2` slide 5 is a stray padding explainer** that belongs to Lec 11's territory; Lec 12
  correctly drops it to one clause.
- **The ILSVRC winners chart appears on at least four decks** (`L1P1` s5, `W3L4_P1` s37,
  `W3L4_P2` s9/s10/s18, `W4L5_P1` s3). Lec 14 owns the table; everyone else cites single numbers.

### Errata batch 3 — two book-wide pins

- **PINNED: $\mathbf{c}_t$ is the attention context vector; the LSTM cell state is capital
  $\mathbf{C}_t$.** The decks use lowercase for both. Lec 18 has been normalised to $\mathbf{C}_t$
  and carries the explanatory note; Lec 19 already uses the pinned convention. **Lec 24, 25 and 27
  must use $\mathbf{c}_t$ for context.** Added to CONTRACT §3.
- **The GRU update on `L6P4` slides 25–26 is the COMPLEMENT of Cho et al. (2014).** The deck prints
  $$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1-\mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$
  and is internally consistent with it (slide 24 defines $\mathbf{z}_t$ as how much of the previous
  state is *retained*; slide 26 says $\mathbf{z}_t=1 \Rightarrow \mathbf{h}_t = \mathbf{h}_{t-1}$).
  This also matches PyTorch's `nn.GRU`. Cho's original has the two coefficients swapped. Lec 19 gives
  **both forms**, computes its numerical under both, and tells the reader to answer with the deck's.
  **Any chapter citing the GRU update must use the deck's form.**
- `L6P3_LSTM` slides 18, 23, 24, 26 **render nearly blank** — their equations are embedded images the
  exporter dropped. The content survives in `assets/figures/W5_L6P3_LSTM/image43.png`, `image48.png`,
  `image480.png`. If a render looks suspiciously empty, check Pool A.
- `L6P3_LSTM` physically contains Lec 17's vanishing-gradient derivation **twice** (slides 6–7 and
  23–26). Lec 17 owns it; Lec 18 correctly defers.
- **Landmark system names are largely absent from the Week-5 survey decks.** Lec 21's deck names only
  Pix2Pix and CycleGAN across 31 slides; NeRF, StyleGAN, AnoGAN, Gatys and Sora are editorial
  additions. Lec 21 labels which are the deck's and which are added — **do the same book-wide**.

### Errata batch 4

- **`L6P2_BPTT_RNN` slide 33 is a hidden bonus slide after the "Next lecture" card** and contains the
  best mathematics in the deck — the full recursive expansion of
  $\partial\mathbf{h}_4/\partial\mathbf{W}_{hh}$ collapsing to
  $\sum_i (\partial\mathbf{h}_4/\partial\mathbf{h}_i)(\partial\mathbf{h}_i/\partial\mathbf{W}_{hh})$.
  Do not assume the deck stops at slide 32. Clean extraction: `image2390.png`.
- **The char-RNN softmax columns on `L6P2` slides 21–28 are not the softmax of their own logits.**
  Step 1: scores [0.7, 0.3, 2.5, 1.1] give true softmax [0.109, 0.073, 0.657, 0.162], but the deck
  prints [.02, .03, .92, .03]. **Step 2 self-contradicts**: its scores peak on the third entry while
  its printed softmax peaks on the second. Lec 17 uses honest arithmetic and flags it. Lec 18/19 reuse
  the same CS231n-derived figures — do not inherit the numbers.
- **The deck's one-hot index order is reversed from alphabetical**: e=[0001], a=[0010], g=[0100],
  r=[1000], i.e. index order (r, g, a, e). Any chapter reusing these vectors must keep that order.
- **Gradient clipping and the word "exploding" appear nowhere on the BPTT deck.** Lec 17 owns clipping
  as pure "Beyond the slides" content with no slide to anchor it.
- **RNN weight symbols are pinned as $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$**
  (subscript reads *from → to*). The decks write $W_x, W_h, W_o$. Lec 16 and Lec 17 both use the
  pinned form.
- **`L6P4` slide 5 claims "no parallelisation across time" is an RNN problem that LSTM fixes.** It does
  not — LSTM makes it worse (4 matmuls per step instead of 1). Lec 18 flags this and points forward to
  attention, which is the real fix.

### Errata batch 5 — taxonomy contradictions (matters for Weeks 9-12)

- **The `L8P1_GenModel_Intro` deck contradicts itself on where VAEs sit, three times.**
  Slide 15 calls them "Explicit Density Models: these are *true density estimators*" with no caveat;
  slide 21 puts them in a separate "Latent Variable Models (Approximate Likelihood)" category;
  slide 22 says the likelihood "is generally optimized through a variational lower bound rather than
  computed directly"; slide 16's tree places them at explicit → approximate → variational.
  **Lec 22 adopts the tree/21/22 position (explicit, approximate, variational) and reconciles the
  three in a table.** Lec 29 and Lec 30 must not contradict that placement.
- **The deck classifies diffusion / score-based models as IMPLICIT density** ("implicit density
  estimation via score estimation", slides 15 and 25). This conflicts with the mainstream
  classification, which puts DDPM at explicit-approximate since it maximises an ELBO. Lec 22 follows
  the deck as the examinable answer and flags the conflict. **Lec 31–32 (Diffusion I/II, Weeks 9-12)
  must either follow the deck or explicitly overrule it — pick one and say so, or the book
  self-contradicts.**
- **Slide 16's taxonomy tree predates DDPM and omits diffusion entirely**, so slide 15's three-type
  framing and slide 16's two-type tree cannot be reconciled without saying so. Lec 22 says so.
- **`L8P1` slides 6–13 are label-conditional throughout** — the deck writes $p_\theta(x,y)$ and
  factorises $p_\theta(x,y) = p_\theta(y)p_\theta(x\mid y)$, i.e. the generative-*classifier* setup.
  The unconditional $p_\theta(x)$ only appears from slide 17. Lec 22 teaches both and marks the
  transition. Lec 23 should expect the $p(x)$ framing.
- **`D` is overloaded a third way** on this deck: the real distribution on slide 3
  ($D = I(x)$, $D' = G(Z)$) and the dataset on slides 10–11 ($\mathcal{D} = \{(x_i,y_i)\}$).
- **GMM is named as a generative family on `L8P1` slide 18** — the only mixture-model mention in the
  course. Lec 22 places it at explicit → tractable. GMM/EM is otherwise unowned; acceptable.
- **Pool A is useless on `W6_L8P1`** — all 20 extracted figures are whole-slide screenshots with the
  titles cropped, strictly worse than the renders.

### Errata batch 6 — positional encoding (highest-risk defect found so far)

- **`L8P5_Transformer-3` slide 23 says positional encodings are "CONCATENATED" to the input vectors.
  This is wrong.** They are **ADDED** — as the same deck's own slides 20, 22 and 24 show, and as
  Vaswani et al. specify. The error is inherited from a CS231n slide. **Every chapter touching
  positional encoding must say ADDED.** Lec 26 flags it in a blockquote and twice in MCQ traps.
- **Slide 23 prints the frequency base as 1000, not 10,000** ($\omega_k = 1/1000^{2k/d}$) — a dropped
  zero. The correct base is $10{,}000$. Do not inherit the typo.
- **Slide 23 indexes $k = 1 \ldots d/2$ (one-based)** where Vaswani uses $i = 0 \ldots d/2-1$. Under the
  deck's indexing $PE(\text{pos},0) = \sin(\text{pos})$ fails. Lec 26 adopts the zero-based form.
- **The deck never gives the indexed form $PE(\text{pos},2i)$ / $PE(\text{pos},2i+1)$ at all** — only a
  stacked column vector. The examinable form is supplied from the paper and labelled as such.
- **Slide 13 double-names the first Add&Norm** (it lists "Add & Normalization" *and* a separate
  "LAYER NORMALIZATION" bar for the same component). The canonical order is
  MHSA → Add&Norm → FFN → Add&Norm.
- **Embedding scaling by $\sqrt{d_{\text{model}}}$** (the paper multiplies embeddings by $\sqrt{512}$
  before adding PE) appears on no deck. **Assigned to Lec 26**, covered as a gap.
- **Pre-norm vs post-norm** is unassigned; Lec 26 notes the deck/paper use post-norm and modern models
  use pre-norm. Lec 27 must not contradict this.
- **The $O(L^2)$ cost of the attention score tensor** is on no deck. Lec 26 covers it; **Lec 28 needs
  it** as the reason ViT uses 16x16 patches.

### Errata batch 7 — attention decks

- **`L8P3_Transformer-1` states the attention-weight range as $1 < a < 0$ on slides 16, 17, 19 and 20** —
  operands transposed, unsatisfiable. Correct is $0 < \alpha < 1$. Lec 24 flags it. **No chapter may
  restate the slide's wording.**
- **`L8P3` slide 17 writes $c_1 = \sum a_{1,i,j} z_{1,i,j}$**, giving the CNN *features* a time index.
  The CNN runs once per image; $z_{i,j}$ has no $t$. Inheriting this inflates any cost calculation by
  a factor of the caption length.
- **`L8P4_Transformer-2` uses a TRANSPOSED index convention**: $\alpha_{ij} = \mathbf{q}_j\cdot\mathbf{k}_i/\sqrt{D}$,
  i.e. **first index = key, second = query** — the opposite of the usual row-as-query convention.
  **Lec 25 adopts row = query and flags the deck's order. Lec 26, 27 and 28 use the same deck family
  and must not silently inherit the deck's order.**
- **`L8P4` has a genuine free/summation index typo** repeated on slides 14, 17, 20, 22, 24:
  $y_j = \sum_j a_{ij} v_i$ uses $j$ as both free and summation index. Correct is $\sum_i$.
- **`L8P4` writes $D$, not $d_k$,** for the scaling divisor. Translate when quoting slide 9.
- **The matrix form $\mathrm{softmax}(\mathbf{QK}^\top/\sqrt{d_k})\mathbf{V}$ appears on NO slide of
  `L8P4`** — the deck gives only element-wise rules. Lec 25 assembles and boxes it; later chapters may
  assume it is established there.
- **Permutation invariance is absent from `L8P4` entirely.** Lec 25 states it and hands the solution to
  Lec 26; **Lec 26 cannot assume the deck motivated positional encoding.**
- **$\mathbf{W}^O$ is never named on the deck** (slide 27 says only "a linear layer"). Lec 25
  introduces the symbol; Lec 27 should use it as established.
- **Attention weights are written $\alpha$ book-wide** (Lec 19, 24, 25 agree). The decks use $a$.
- **Pool A is useless across the whole Transformer deck family** (`L8P3`, `L8P4`, `L8P5`) — the
  extracted figures are PNG crops of equation and bullet text boxes; the real diagrams are PowerPoint
  shapes available only as slide renders.

### Errata batch 8 — autoregressive deck

- **`L8P2_AutoReg` slides 3, 8 and 12 have image-only bodies.** The text dump shows titles only. Slide
  3 carries the fully expanded chain-rule product and names WaveNet; slide 8 carries the FVBN
  definition and its product formula; slide 12 carries the **RGB channel sub-factorisation**
  $p(x_i\mid x_{<i}) = p(x_{i,R}\mid x_{<i})\,p(x_{i,G}\mid x_{<i},x_{i,R})\,p(x_{i,B}\mid x_{<i},x_{i,R},x_{i,G})$.
  Clean extractions: `image1.png`, `image4.png`, `image9.png`. **Lec 23 owns the RGB sub-factorisation**
  (it triples the generation step count).
- **The deck writes the product's upper limit as $n^2$ on slide 12 but $n$ on slides 8 and 10.** Lec 23
  states both and flags it.
- **Mask type A vs B IS on the deck** (slide 16), presented as a *channel* connectivity rule. The
  spatial centre-pixel reading is standard but editorial; Lec 23 teaches both and labels which is which.
  The deck's unlabelled 3x3 mask figure (`image13.png`) is specifically a **type-A** mask.
- **`L8P2` slide 5's figure is from Huang et al. 2023 (arXiv:2305.11718), not the PixelRNN paper** — its
  lower row shows a coarse-to-fine variable-length ordering that is anachronistic for this lecture.
  Lec 23 uses it but explains what it actually shows.
- The blind-spot problem, Gated PixelCNN and PixelCNN++ are **not on the deck**; Lec 23 covers them as
  labelled gaps.

### Errata batch 9 — decoder deck (final)

- **`L8P6_Transformer-4` slide 24 prints the CAUSAL MASK inside the CROSS-attention formula** —
  $\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}(\mathbf{QK}^\top/\sqrt{d_k} + M)\mathbf{V}$,
  copied verbatim from slide 22. **Cross-attention is not causally masked.** Lec 27 flags it.
- **The deck's FFN is $\mathbf{W}_2\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$**
  (column convention, "ReLU or GELU"); the paper's is
  $\max(0, \mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2$. Lec 27 gives the paper
  form as primary and notes the deck's.
- **Slide 21 draws the residual joins as $\otimes$, not $\oplus$.** They are additions — compare
  slide 17's explicit `+`.
- **The deck never states $d_{\text{model}}=512$, $d_{ff}=2048$, $h=8$, $N=6$.** Slide 7's worked
  example is a 4-head toy with $d_k=128$ and says "we assume different dimensions". **Do not present
  $h=4$ as the base model.**
- **Transformer parameter count, reconciled** (Lec 27, built sublayer by sublayer): attention sublayer
  1,048,576 · FFN 2,099,712 · LayerNorm 1,024 · encoder block 3,150,336 · decoder block 4,199,936 ·
  both stacks 44,101,632. With a tied 37,000x512 embedding the total is **63.0M**, not the paper's
  stated 65M; the gap is purely vocabulary-dependent ($|V| = 41{,}000$ gives 65.1M). Any chapter
  quoting a Transformer parameter count must use this reconciliation.

### Errata batch 10 — VAE and ViT decks (final)

- **`L9P1_VAE-1` silently swaps $\theta$ and $\phi$ mid-deck.** Slide 4 writes the plain AE as encoder
  $f_\theta$ / decoder $g_\varphi$; slides 7-14 then write the VAE as encoder $q_\phi$ / decoder
  $p_\theta$ — the roles invert. **Book-wide convention: $\phi$ = encoder, $\theta$ = decoder, in both
  halves.** Verified consistent across Lec 29 and Lec 30 (220 occurrences, zero inverted).
- **`L9P1` contradicts itself on the Gaussian within a single slide**: the slide 8/9/10 *diagram labels*
  read $N(\mu_x, \sigma_x)$ (std dev) while the *body text* on slides 7-8 reads
  $N(\mu(x), \sigma^2(x))$ (variance). Book uses variance throughout.
- **`L9P1` slide 19 omits the $d\mathbf{z}$** from $p_\theta(x) = \int p_\theta(x\mid z)p(z)$.
  Slide 17 writes it correctly.
- **`L9P1` slide 13 writes the prior as $p_\theta(\mathbf{z})$**, implying learnable parameters;
  slide 19 correctly writes $p(\mathbf{z})$. Matters because a learned prior would break Lec 30's
  closed-form KL.
- **`L9P1` slides 8/9/10 label the sampled distribution $p(z\mid x)$** — precisely the intractable
  object the lecture then spends six slides proving you cannot sample from. Should be
  $q_\phi(\mathbf{z}\mid\mathbf{x})$.
- **`L9P1` presents its argument backwards**: slide 15 announces the posterior's intractability before
  slide 17 establishes the marginal integral that causes it. Lec 29 reverses the order.
- **`L8P7_VIT` slide 7 projects patches to $D = 512$, not 768.** Lec 28 teaches 768 (ViT-Base) as
  primary and flags the deck's 512 variant. Note $P^2C = d_{\text{model}} = 768$ is a coincidence.
- **`L8P7` slides 19 and 21 are verbatim duplicates**; slides 6 and 10 re-embed the same `.emf` as
  slides 5 and 9, so four slides carry two figures.
- **`L8P7` never mentions the JFT-300M inductive-bias result, the $O(n^2)$ cost, or pre-norm.** All
  three are editorial additions in Lec 28, labelled as such.
- **The ViT-as-autoregressive erratum is CLEARED** — Lec 28 carries a dedicated subsection naming
  `L1P3` slide 15, plus an MCQ trap and a self-test question.
- **Pool A is unusable on `W8_L9P1` and `W8_L8P7`** — screenshots of slide text bodies. Only
  `W8_L8P7/image7.png` (Swin vs ViT) is a genuine figure.

### Errata batch 11 — ELBO deck (last chapter reported)

- **`L9P2_VAE-2` never states the closed-form diagonal-Gaussian KL.** It is on none of the 24 slides,
  which means **the ELBO as presented on the deck is literally uncomputable**. Lec 30 derives it from
  scratch (1-D log-density subtraction, $\mathbb{E}[(z-\mu)^2]=\sigma^2$,
  $\mathbb{E}[z^2]=\sigma^2+\mu^2$, then summed over independent dimensions) and flags the omission.
  This is the single largest *gap* (as opposed to error) found in Weeks 1-8.
- **The deck gives BOTH ELBO routes.** Jensen's inequality via multiply-and-divide on slides 9-10, and
  the KL-decomposition identity on slides 17-18 — but the latter is asserted with the words *"after
  some algebraic manipulation"* and the manipulation is skipped. Lec 30 derives it in four lines and
  uses it as the chapter's spine.
- **`L9P2` contradicts itself on the Gaussian notation**: slide 5 writes $\mathcal{N}(\mu_x, \sigma_x)$
  (std dev) while **slide 19 writes $\mathcal{N}(\mu, \sigma^2)$ (variance)**. The earlier errata
  predicted only the std-dev form. Both VAE chapters use variance throughout.
- **`L9P2` writes the prior as $p_\theta(\mathbf{z})$** with a $\theta$ subscript on slides 12-16 and
  inside the ELBO's KL term. The prior has no learnable parameters; the book writes $p(\mathbf{z})$.
  (Same defect as `L9P1` slide 13 — it is consistent across both VAE decks.)
- **`L9P2` slide 24 is "Next: Diffusion Models — Lecture 10 Part 1"**, confirming the playlist runs
  straight from the VAE into diffusion. This corroborates the Weeks 9-12 mapping (Lec 31-37).
- **Pool A is useless on `W8_L9P2`** — 18 of 21 extracted figures are 1725x714 screenshots of slide
  bodies; the rest are two cat photos and a fragment.

## Unowned topics — to be handled by the editor in a supplementary note

These are **named on slides but never taught** by any deck, and no chapter owns them. They are
near-certain MCQ material, so the editor writes them up in `notes/supplementary.md` rather than
forcing them into a chapter where they do not belong:

- **Batch normalization and internal covariate shift** — listed on `W3L4_P3` slide 26 as a solution,
  never explained. Includes the BatchNorm-vs-LayerNorm contrast (Lec 27 owns the LayerNorm side).
- **Weight initialization (Xavier/Glorot, He)** — named on `W3L4_P3` slide 7. Load-bearing for Lec 13,
  since the real per-layer gradient factor is $f'(z)\cdot\|\mathbf{W}\|$, not $f'(z)$ alone.
- **SGD vs mini-batch vs full-batch gradient descent** — assumed throughout, defined nowhere.
- **Dropout and data augmentation** — absent from the Lec 10 deck; only gap notes exist.
- **Optimizers beyond plain SGD** (Momentum, RMSProp, Adam) — never covered by any deck in Weeks 1–8.

Also note: `W3L4_P3` slide 25 already contains the identity-mapping argument ("use identity mappings
for the additional layers") with a block diagram. Lec 13 uses it only to prove the failure is an
*optimisation* problem, not a representational one, and does not explain skip connections. Lec 14's
cliffhanger is intact.
