# Generative AI for Computer Vision — Study Notes

Complete replacement notes for the NPTEL / IIT Guwahati course
[*Generative AI for Computer Vision*](https://www.youtube.com/playlist?list=PLwdnzlV3ogoUlHWMa5ZnwLuMReY__vHFM),
written from the lecture slides. Built to be read **instead of** watching the lectures, and aimed at a
95+ exam score.

**Covered so far:** Weeks 1–8, Lec 1–30 (30 decks) + 1 supplementary chapter.
~183,000 words · 272 figures · 142 worked numericals · 45 runnable code blocks ·
332 MCQ traps · 616 self-test questions. Every chapter validates clean and every code block runs.
**Pending:** Weeks 9–12, Lec 31–37 — Diffusion I/II, GAN I/II, LLM I/II, Vision-Language Models.
Drop those slides in and they slot into the same structure.

---

## How to read this

Every chapter has the same eight sections, always in this order:

| Section | What it is for |
|---|---|
| **Why this lecture exists** | The problem being solved, and why it comes after the last one |
| **The ideas** | The actual teaching — concepts derived, not asserted |
| **Worked numericals** | Fully computed examples of the kind the exam asks. **The highest-value section.** |
| **Code** | Runnable NumPy / PyTorch that makes a mechanism concrete |
| **Exam pack** | Must-memorise table · numbers worth knowing · MCQ traps · self-test with answers |
| **Beyond the slides** | Gaps in the deck that you genuinely need filled |
| **Cut from the slides** | What was compressed or dropped, and why — the audit trail |

Two reading passes work best. First pass: *Why this lecture exists* → *The ideas* → *Beyond the slides*.
Second pass, closer to the exam: *Worked numericals* → *Exam pack*, with the self-tests answered on
paper before you open the answer block.

### Viewing

The maths is LaTeX inside Markdown. Pick one:

- **VS Code** — open the folder, `Ctrl+Shift+V` to preview. Install *Markdown+Math* for proper rendering.
- **Obsidian** — point a vault at this folder. Renders maths and images natively; best for navigation.
- **PDF** — `pandoc notes/week-0*/[0-9]*.md -o GenAIforCV.pdf --pdf-engine=xelatex --toc`

---

## Chapters

### Week 1 — Foundations
| | Chapter | Covers |
|---|---|---|
| 1 | [Intro: CV and Generative AI](notes/week-01/01-intro-cv-and-genai.md) | What CV is, what GenAI is, why synthetic data, challenges, applications |
| 2 | [Generative vs Discriminative](notes/week-01/02-generative-vs-discriminative.md) | $p(y\mid x)$ vs $p(x,y)$, why synthetic data, Sim2Real gap, the generator framing |
| 3 | [Generative Vision Models](notes/week-01/03-generative-vision-models.md) | The five-family map: VAE, GAN, autoregressive, diffusion, flows |
| 4 | [Linear Algebra](notes/week-01/04-linear-algebra.md) | Vectors, dot products, matrices, determinant, rank, inverse, eigen |
| 5 | [Probability I](notes/week-01/05-probability-1.md) | Axioms, conditional probability, independence, Bayes, random variables |
| 6 | [Probability II](notes/week-01/06-probability-2.md) | Expectation, variance, covariance, the Gaussian, multivariate Gaussian |

### Week 2 — Neural networks
| | Chapter | Covers |
|---|---|---|
| 7 | [The Perceptron](notes/week-02/07-perceptron.md) | Artificial neuron, step function, learning rule, linear separability, XOR |
| 8 | [MLP and Activations](notes/week-02/08-mlp-and-activations.md) | Hidden layers, why non-linearity, sigmoid/tanh/ReLU, softmax |
| 9 | [Backpropagation](notes/week-02/09-backpropagation.md) | Chain rule, forward/backward pass, deltas, complexity, checkpointing |
| 10 | [Overfitting and Regularization](notes/week-02/10-overfitting-and-regularization.md) | Bias–variance, k-fold CV, L1/L2, $\lambda$, early stopping |

### Week 3 — Convolutional networks
| | Chapter | Covers |
|---|---|---|
| 11 | [CNN Basics](notes/week-03/11-cnn-basics.md) | Convolution, stride, padding, pooling, activation maps, normalization |
| 12 | [CNN Architectures](notes/week-03/12-cnn-architectures.md) | LeNet-5, AlexNet, VGG, GoogLeNet, Inception, 1×1 bottlenecks |
| 13 | [Vanishing Gradients](notes/week-03/13-vanishing-gradients-activations.md) | Why depth fails, sigmoid saturation, ReLU and friends, dying ReLU |

### Week 4 — Deeper architectures, and sequences
| | Chapter | Covers |
|---|---|---|
| 14 | [ResNet](notes/week-04/14-resnet.md) | Degradation problem, skip connections, identity mapping, ILSVRC timeline |
| 15 | [DenseNet, MobileNet, EfficientNet](notes/week-04/15-densenet-mobilenet-efficientnet.md) | Dense blocks, depthwise-separable convolution, compound scaling |
| 16 | [Sequence Modelling and RNNs](notes/week-04/16-sequence-modelling-and-rnn.md) | Why sequences, RNN recurrence, the five input/output patterns |

### Week 5 — Recurrence, attention, and vision tasks
| | Chapter | Covers |
|---|---|---|
| 17 | [Backpropagation Through Time](notes/week-05/17-bptt.md) | BPTT, truncated BPTT, character-level modelling, deep and bidirectional RNNs |
| 18 | [LSTM](notes/week-05/18-lstm.md) | Long-term dependencies, cell state, the three gates, constant error carousel |
| 19 | [GRU, Seq2Seq, Attention](notes/week-05/19-gru-seq2seq-attention.md) | GRU, encoder–decoder bottleneck, additive attention, context vector |
| 20 | [GenAI for Vision Tasks I](notes/week-05/20-genai-vision-tasks-1.md) | CV challenges; classification, detection, segmentation, super-res, inpainting |
| 21 | [GenAI for Vision Tasks II](notes/week-05/21-genai-vision-tasks-2.md) | I2I translation, medical synthesis, anomaly detection, 3D, faces, style, video |

### Week 6 — Generative modelling proper
| | Chapter | Covers |
|---|---|---|
| 22 | [Generative Taxonomy and MLE](notes/week-06/22-generative-taxonomy-and-mle.md) | MLE objective, explicit vs implicit density, tractable vs approximate |
| 23 | [Autoregressive: PixelRNN / PixelCNN](notes/week-06/23-autoregressive-pixelrnn-pixelcnn.md) | Chain rule factorisation, FVBN, masked convolution, Row/Diagonal BiLSTM |
| 24 | [Captioning and Spatial Attention](notes/week-06/24-captioning-and-spatial-attention.md) | Show-Attend-and-Tell, attention over CNN feature grids |

### Week 7 — Transformers
| | Chapter | Covers |
|---|---|---|
| 25 | [Q/K/V and Self-Attention](notes/week-07/25-qkv-and-self-attention.md) | Scaled dot-product attention, self-attention, masking, multi-head |
| 26 | [Encoder and Positional Encoding](notes/week-07/26-encoder-and-positional-encoding.md) | Input embedding, sin/cos positional encoding, the encoder block |
| 27 | [Decoder and the Full Transformer](notes/week-07/27-decoder-and-full-transformer.md) | Masked self-attention, cross-attention, Add&Norm, FFN, full architecture |

### Week 8 — Vision transformers and VAEs
| | Chapter | Covers |
|---|---|---|
| 28 | [ViT, DETR, Swin](notes/week-08/28-vit-detr-swin.md) | Patch embedding, ViT encoder, DETR bipartite matching, shifted windows |
| 29 | [Autoencoders to VAE](notes/week-08/29-autoencoders-to-vae.md) | AE bottleneck, why AEs are not generative, the intractable posterior |
| 30 | [ELBO and Reparameterization](notes/week-08/30-elbo-and-reparameterization.md) | Variational inference, ELBO, the two-term loss, the reparameterization trick |

### Supplementary
| | Chapter | Covers |
|---|---|---|
| — | [Topics the slides name but never teach](notes/supplementary.md) | BatchNorm & internal covariate shift, Xavier/He initialization, SGD vs mini-batch, Momentum/RMSProp/Adam, dropout, data augmentation |

Six topics appear in slide bullet lists as "solutions" and are then never explained. That is exactly
the shape of an MCQ, so they are written up properly here. Everything in that chapter is labelled as
coming from outside the decks.

### Cram material

All generated from the chapters by `_build/build_cram.py`, so they can never drift out of sync.
Re-run it after editing any chapter.

| File | What it is |
|---|---|
| [Formula sheet](cram/formula-sheet.md) | Every **Must-memorise** entry in the course, in order |
| [Exam traps](cram/exam-traps.md) | Every MCQ trap, collected — read this the night before |
| [Numbers worth knowing](cram/numbers.md) | Every figure an MCQ can key on |
| [Self-test bank](cram/self-test-bank.md) | Every self-test question, answers collapsed |
| [Week summaries](cram/week-summaries.md) | The one-paragraph version of each lecture |

---

## Repository layout

```
GenAIforCV/
├── README.md              ← you are here
├── notes/week-01..08/     ← 30 chapters, one per lecture
├── cram/                  ← generated: formulas, traps, numbers, self-tests, summaries
├── code/                  ← standalone runnable scripts
├── assets/
│   ├── figures/           ← 366 figures extracted from the decks
│   └── slides/            ← all 769 slides rendered to PNG, by deck
├── extracted/             ← raw text dumps of every deck (provenance)
├── slides_raw/            ← the original .pptx files
└── _build/                ← contract, ownership map, validate.py, build_cram.py
```

`_build/` is the machinery that kept 30 separately-written chapters consistent:

- `CONTRACT.md` — the binding style/structure contract every chapter follows
- `OWNERSHIP.md` — which chapter owns which concept, plus **every error found in the source decks**
- `validate.py` — checks section structure, link integrity, figure existence, table well-formedness
- `build_cram.py` — regenerates `cram/` from the chapters

Run `python3 _build/validate.py` after any edit. Worth reading only if you want to extend the notes.

## Provenance

Every chapter is written from its own deck, with the rendered slides read directly (much of the maths
on these decks is embedded as images, invisible to text extraction). Figures are either lifted from the
deck's own media or are renders of the slide itself; captions give the slide number so you can always
trace a claim back. Nothing is downloaded or external — the whole thing works offline.

Where a chapter adds material the lecturer did not cover, it is quarantined in **Beyond the slides** and
labelled, so you always know what is examinable course content and what is scaffolding.
