# Deep Learning for Natural Language Processing — Study Notes

Complete replacement notes for the NPTEL / IIT Kharagpur course *Deep Learning for Natural Language
Processing* (Pawan Goyal), written from the lecture slides. Built to be read **instead of** watching
the lectures, and aimed at a 95+ exam score.

**60 lectures · 12 weeks · 1,591 source pages.**

Companion volume: [Generative AI for Computer Vision](../GenAIforCV/README.md) — 31 chapters. Several
lectures here overlap it; those chapters compress the shared mechanics and link across, but each stands
alone, because *this* exam is set from *these* slides with this lecturer's notation.

---

## How to read this

Every chapter has the same seven sections, always in this order:

| Section | What it is for |
|---|---|
| **Why this lecture exists** | The problem being solved, and why it follows the last one |
| **The ideas** | The actual teaching — concepts derived, not asserted |
| **Worked numericals** | Fully computed examples. **Every in-deck exercise is worked here.** |
| **Code** | Runnable NumPy/PyTorch that makes a mechanism concrete |
| **Exam pack** | Must-memorise · numbers worth knowing · MCQ traps · self-test with answers |
| **Beyond the slides** | Gaps in the deck you genuinely need filled |
| **Cut from the slides** | What was compressed or dropped, and why — the audit trail |

Two passes work best. First: *Why this lecture exists* → *The ideas* → *Beyond the slides*. Second,
closer to the exam: *Worked numericals* → *Exam pack*, answering the self-tests on paper first.

### Viewing

Maths is LaTeX inside Markdown. **VS Code** (`Ctrl+Shift+V`), **Obsidian** (best for navigation), or
`pandoc notes/week-*/[0-9]*.md -o DLforNLP.pdf --pdf-engine=xelatex --toc`.

---

## Chapters

### Week 1 — Foundations
| | Chapter | Covers |
|---|---|---|
| 1 | [Introduction to NLP](notes/week-01/01-intro-to-nlp.md) | Ambiguity, levels of linguistic structure, the paradigm arc |
| 2 | [Text Processing and Tokenization](notes/week-01/02-text-processing-tokenization.md) | Type/token, `<UNK>`, normalization, **BPE**, WordPiece, SentencePiece |
| 3 | [N-gram Language Models I](notes/week-01/03-ngram-lm-1.md) | The LM task, chain rule, Markov assumption, MLE estimation |
| 4 | [N-gram Language Models II](notes/week-01/04-ngram-lm-2-smoothing-perplexity.md) | Smoothing, backoff, interpolation, **perplexity**, train/dev/test |
| 5 | [NLP Tasks and Paradigms](notes/week-01/05-nlp-tasks-and-paradigms.md) | The paradigms, **precision/recall/F1, macro vs micro** |

### Week 2 — Neural networks
| | Chapter | Covers |
|---|---|---|
| 6 | [Supervised Learning](notes/week-02/06-supervised-learning.md) | Model/loss/training/testing, 1-D linear regression worked end to end |
| 7 | [Shallow Neural Networks](notes/week-02/07-shallow-neural-networks.md) | The piecewise-linear construction, hidden units, **universal approximation** |
| 8 | [Deep Neural Networks](notes/week-02/08-deep-neural-networks.md) | Composition, **regions per parameter**, why depth wins |
| 9 | [Backpropagation](notes/week-02/09-backpropagation.md) | Chain rule, the toy-function trace, **matrix calculus**, autodiff |
| 10 | [Gradient Descent and Initialization](notes/week-02/10-gradient-descent-and-init.md) | He/Xavier init, SGD, momentum, **Adam** with bias correction |

### Week 3 — Word representations
| | Chapter | Covers |
|---|---|---|
| 11 | [Word Representation](notes/week-03/11-word-representation.md) | Distributional hypothesis, **PMI/PPMI**, cosine, analogies |
| 12 | [word2vec Skip-gram](notes/week-03/12-word2vec-skipgram.md) | CBOW vs skip-gram, the softmax objective, the gradient |
| 13 | [Negative Sampling and GloVe](notes/week-03/13-negative-sampling-glove.md) | **Negative sampling**, the 3/4 power, hierarchical softmax, GloVe |
| 14 | [fastText and Beyond Words](notes/week-03/14-fasttext-and-beyond-words.md) | Character n-grams, **OOV handling**, DeepWalk, TransE |
| 15 | [Cross-Lingual Representations](notes/week-03/15-cross-lingual-representations.md) | The supervision taxonomy, mapping, CCA, merge-and-shuffle, IndicFT |

### Week 4 — Recurrence
| | Chapter | Covers |
|---|---|---|
| 16 | [RNN Language Models](notes/week-04/16-rnn-language-models.md) | Fixed-window baseline, the recurrence, **weight sharing** |
| 17 | [RNN Applications](notes/week-04/17-rnn-applications.md) | Generation, **BIO tagging**, classification, bidirectional RNNs |
| 18 | [Seq2Seq and Attention](notes/week-04/18-seq2seq-and-attention.md) | Encoder–decoder, teacher forcing, the bottleneck, **attention** |
| 19 | [Decoding Strategies](notes/week-04/19-decoding-strategies.md) | Greedy, **beam search**, temperature, top-k, **nucleus** |
| 20 | [GRU and LSTM](notes/week-04/20-gru-and-lstm.md) | The recurrent vanishing gradient, gates, the cell state, xLSTM |

### Week 5 — Transformers
| | Chapter | Covers |
|---|---|---|
| 21 | [Introduction to Transformers](notes/week-05/21-intro-to-transformers.md) | Why not recurrence, self-attention intuition, contextual embeddings |
| 22 | [Self-Attention and Multi-Head](notes/week-05/22-self-attention-and-multihead.md) | **Q/K/V, scaled dot-product, √d_k**, multi-head, LayerNorm |
| 23 | [Positional Encoding and the Encoder](notes/week-05/23-positional-encoding-and-encoder.md) | **Sinusoidal PE**, learned embeddings, the encoder block, param counts |
| 24 | [The Decoder and Transformer LMs](notes/week-05/24-decoder-and-transformer-lm.md) | **Masked** self-attention, **cross-attention**, the LM head |
| 25 | [Efficient Transformers](notes/week-05/25-efficient-transformers.md) | **KV cache**, Longformer, BigBird, Linformer, **MQA/GQA**, MoE |

### Week 6 — Pretraining
| | Chapter | Covers |
|---|---|---|
| 26 | [Pretraining and ELMo](notes/week-06/26-pretraining-and-elmo.md) | Transfer learning, The Pile, **ELMo**, the three-architecture taxonomy |
| 27 | [BERT and Masked LM](notes/week-06/27-bert-masked-lm.md) | **MLM, the 80/10/10 recipe**, NSP, fine-tuning, subword BIO |
| 28 | [Span Tasks, T5 and BART](notes/week-06/28-span-tasks-t5-bart.md) | Span heads, **GLUE**, **T5** span corruption, **BART** noising |
| 29 | [GPT and Decoder Pretraining](notes/week-06/29-gpt-decoder-pretraining.md) | The GPT series, zero-shot, **in-context learning** |
| 30 | [Domain and Multilingual Pretraining](notes/week-06/30-domain-and-multilingual-pretraining.md) | SciBERT, mBERT, XLM-R, mT5, **the curse of multilinguality** |

### Week 7 — Applications
| | Chapter | Covers |
|---|---|---|
| 31 | [Question Answering I](notes/week-07/31-question-answering-1.md) | SQuAD, **EM/F1**, retriever–reader, **TF-IDF, BM25**, DPR, ColBERT |
| 32 | [Question Answering II](notes/week-07/32-question-answering-2.md) | **In-batch negatives**, closed-book QA, multi-hop, multilingual, tabular |
| 33 | [Dialogue Systems I](notes/week-07/33-dialogue-systems-1.md) | Open-domain, the genericness problem, **dialogue evaluation** |
| 34 | [Dialogue Systems II](notes/week-07/34-dialogue-systems-2.md) | Frames, **GUS**, slot filling, state tracking, MultiWOZ metrics |
| 35 | [Text Summarization](notes/week-07/35-text-summarization.md) | Extractive vs abstractive, Pegasus, **BLEU, ROUGE**, BARTScore |

### Week 8 — Alignment
| | Chapter | Covers |
|---|---|---|
| 36 | [Instruction Fine-tuning I](notes/week-08/36-instruction-finetuning-1.md) | LM ≠ following instructions, instruction schemas, Super-NaturalInstructions |
| 37 | [Instruction Fine-tuning II](notes/week-08/37-instruction-finetuning-2.md) | Flan, Dolly, **Self-Instruct** |
| 38 | [RLHF I](notes/week-08/38-rlhf-1.md) | Why pairwise, **Bradley-Terry**, reward models, the RL vocabulary |
| 39 | [RLHF II: PPO](notes/week-08/39-rlhf-2-ppo.md) | **KL divergence**, policy gradient, baselines, advantage, **the PPO clip** |
| 40 | [Direct Preference Optimization](notes/week-08/40-dpo.md) | **The DPO objective**, KTO, PPO-vs-DPO, Best-of-N |

### Week 9 — Prompting
| | Chapter | Covers |
|---|---|---|
| 41 | [Prompting I](notes/week-09/41-prompting-1.md) | Templates, output mapping/verbalizers, chat formats, zero/one/few-shot |
| 42 | [Why In-Context Learning Works](notes/week-09/42-why-icl-works.md) | The residual stream, **induction heads**, the random-label result |
| 43 | [Advanced Prompting](notes/week-09/43-advanced-prompting.md) | **Chain-of-thought**, self-consistency, ToT, Self-Ask, PAL/PoT |
| 44 | [Tool-Aided LMs](notes/week-09/44-tool-aided-lms.md) | TALM, **Toolformer** and its self-supervised filter |
| 45 | [Automatic Prompt Engineering](notes/week-09/45-automatic-prompt-engineering.md) | Gradient-based prompt search, EvoPrompt, **prompt-tuning** |

### Week 10 — Efficiency
| | Chapter | Covers |
|---|---|---|
| 46 | [PEFT: Adapters and Prefix-Tuning](notes/week-10/46-peft-adapters-prefix.md) | The three PEFT perspectives, **adapters**, **prefix-tuning** |
| 47 | [LoRA and Variants](notes/week-10/47-lora-and-variants.md) | **LoRA**, merging and zero latency, KronA, VeRA |
| 48 | [Quantization and QLoRA I](notes/week-10/48-quantization-qlora-1.md) | GPU memory accounting, quantization, **NF4** |
| 49 | [QLoRA II](notes/week-10/49-qlora-2.md) | **Double quantization**, paged optimizers, gradient checkpointing |
| 50 | [Pruning and Distillation](notes/week-10/50-pruning-and-distillation.md) | Lottery ticket, magnitude/movement/**Wanda**, DistilBERT, MiniLLM |

### Week 11 — Modern LLMs
| | Chapter | Covers |
|---|---|---|
| 51 | [Scaling Laws](notes/week-11/51-scaling-laws.md) | Kaplan, **C = 6ND**, **Chinchilla** and the ~20 tokens/param rule |
| 52 | [Modern LLMs and Activations](notes/week-11/52-modern-llms-and-activations.md) | The model roll-call, GELU/Swish, **SwiGLU**, **RMSNorm** |
| 53 | [Positional Embeddings: RoPE and ALiBi](notes/week-11/53-positional-embeddings-rope-alibi.md) | T5 bias, **ALiBi**, **RoPE**, position interpolation |
| 54 | [Long Sequence Modeling](notes/week-11/54-long-sequence-modeling.md) | **Linear attention**, fixed-size cache, Memorizing Transformers |
| 55 | [Retrieval Augmented Generation](notes/week-11/55-retrieval-augmented-generation.md) | **RAG**, REALM, **kNN-LM**, long-context vs RAG |

### Week 12 — Interpretability and trust
| | Chapter | Covers |
|---|---|---|
| 56 | [Interpretability: Probing](notes/week-12/56-interpretability-probing.md) | **Probing** and its caveats, BERT's pipeline, **logit/tuned lens** |
| 57 | [Interpretability: Multilingual](notes/week-12/57-interpretability-multilingual.md) | The latent-language hypothesis, **LAPE**, language-specific neurons |
| 58 | [Interpretability: FFN and Causal Tracing](notes/week-12/58-interpretability-ffn-and-causal-tracing.md) | **FFN as key-value memories**, **causal tracing**, ROME |
| 59 | [Trustworthy LLMs: Taxonomy](notes/week-12/59-trustworthy-llms-taxonomy.md) | The taxonomy, hallucination vs misinformation, **red-teaming** |
| 60 | [Machine Unlearning](notes/week-12/60-machine-unlearning.md) | Gradient ascent, random mismatch, KL utility, **and the course wrap-up** |

### Cram material

Generated from the chapters by `_build/build_cram.py`, so they cannot drift. Re-run after any edit.

| File | What it is |
|---|---|
| [Formula sheet](cram/formula-sheet.md) | Every **Must-memorise** entry, in course order |
| [Exam traps](cram/exam-traps.md) | Every MCQ trap, collected — read this the night before |
| [Numbers worth knowing](cram/numbers.md) | Every figure an MCQ can key on |
| [Self-test bank](cram/self-test-bank.md) | Every self-test question, answers collapsed |
| [Week summaries](cram/week-summaries.md) | The one-paragraph version of each lecture |

---

## Read this before you revise

**The slides contain a large number of errors, and they are catalogued.** Every chapter flags the ones
in its own range; the full list is in `_build/OWNERSHIP.md` under "Errata". A sample of what was found
and verified numerically:

- **BLEU's brevity penalty is non-standard.** The deck uses the linear $\min(1, c/r)$, not Papineni's
  $e^{1-r/c}$. The lecturer's own worked 52% only reproduces with the linear form. **Use the deck's.**
- **Cross-entropy is computed in base-10 logs** in one worked solution (2.37; in nats it is 5.458).
- **Two Program-of-Thought slides print code that does not produce their own printed answer** — one
  Fibonacci loop starts at index 3, one interest calculation solves a different question.
- **Several slides mis-round or mis-sum their own examples** — an Int8 quantization table, a
  "70% reduction" that is 66%, a bit-budget that sums to 5.6 but prints 5.2.
- **A prefix-tuning formula omits the ×2 for keys and values**, contradicting the deck's own results table.
- **Two sign errors in the RLHF derivations**, and a slide calling KL divergence a "distance".

Where a deck is self-consistently non-standard, the notes teach **the deck's version** (the exam is set
from these slides) and flag the discrepancy. Where it is simply wrong, they give the correct answer and
say so.

**The in-deck exercises are worked in full.** The lecturer poses problems throughout; most decks give no
solution, so those answers are derived from first principles and independently verified in code. A few
decks *do* answer their own exercises in handwriting — those are checked against the lecturer's working
and any disagreement is flagged rather than silently resolved.

## Repository layout

```
DLforNLP/
├── README.md              ← you are here
├── notes/week-01..12/     ← 60 chapters, one per lecture
├── cram/                  ← generated: formulas, traps, numbers, self-tests, summaries
├── assets/
│   ├── pages/lecNN/       ← all 1,591 source pages rendered to PNG
│   └── figures/lecNN/     ← images extracted from the PDFs
├── extracted/             ← raw text dumps per week (provenance)
└── _build/                ← contract, ownership map + errata, validate.py, build_cram.py
```

**A warning if you extend these notes:** the text layer of these PDFs is nearly useless for most
lectures — whole ranges extract as titles and URLs only, because the slides are images. Several
chapters' central content (a four-column ablation table, a one-layer control experiment, the lecturer's
handwritten solutions) exists *only* in the rendered pages. Read `assets/pages/`, not `extracted/`.

Run `python3 _build/validate.py` and `_build/run_code.py` after any edit.
