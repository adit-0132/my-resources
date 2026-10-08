# Formula Sheet

Every **Must-memorise** entry from all chapters, in course order. Generated from the chapters by `_build/build_cram.py` — edit the chapters, not this file.

## [Lec 1 — Introduction to NLP](../notes/week-01/01-intro-to-nlp.md)

| Item | Exactly this |
|---|---|
| Natural language | a language that **evolved naturally through human use** |
| NLP, the deck's definition | making computers **understand** what we write/speak, and making computers **write/speak**; systems that **design, implement and test** processing of natural languages for **practical applications** |
| Why NLP is hard | **language ambiguity** |
| The deck's slogan | *"In natural languages, ambiguity is the rule, not an exception"* |
| Levels of linguistic structure (deck's list, bottom → top) | **Characters → Morphology → Words → Syntax: Part of Speech → Syntax: Constituents → Semantics → Discourse** (7 levels) |
| Deck's example for the levels | `Alice talked to Bob.`; morphology `talk + -ed → [VerbPast]`; semantics `CommunicationEvent(e), Agent(e, Alice), Recipient(e, Bob)`; discourse `SpeakerContext(s), TemporalBefore(e, s)` |
| The three ML paradigms | sentiment/news grouping → **text classification**; NER/code-mixing → **sequence labelling**; MT/summarisation/chatbots → **text generation** |
| Five historical eras | 1950s–80s grammar/rule-based → 1980s–2000s expert systems + statistical → 2000s–2010s neural + dense representations → 2010s–2020s deep learning revolution → 2020s–now era of LLMs |
| First-era systems | Chomsky's *Syntactic Structures*, **ELIZA**, **SHRDLU** |
| What deep learning changed | **feature engineering → representation learning**: dense vectors replace hand-written sparse indicator features |
| Sparse vs dense, key property | one-hot `dog` and `cat` are **orthogonal** (similarity 0); dense embeddings are close, so evidence generalises |
| Page 21's thesis | **"Just use generation!"** — every task phrased as a text prompt, answered by one model; held-out tasks work via **zero-shot task generalization** (T0, Sanh et al., ICLR 2022) |
| PP-attachment parse count | $C_{n+1}$ for $n$ PPs, where $C_k = \frac{1}{k+1}\binom{2k}{k}$ |

## [Lec 2 — Text Processing Basics and Tokenization](../notes/week-01/02-text-processing-tokenization.md)

| Item | Exactly this |
|---|---|
| Type / Token | an element of the vocabulary / an instance of that type in running text |
| Heaps' = Herdan's law | $\lvert V\rvert = kN^{\beta}$, often $0.67 < \beta < 0.75$; grows with **more than** $\sqrt{N}$ |
| A text is produced by | a specific writer(s), at a specific time, in a specific variety, of a specific language, for a specific function |
| Corpora vary by | language, variety, code switching, genre, author demographics |
| OOV / closed vocabulary | seen very rarely or not at all in training / unable to produce unseen word forms |
| `<UNK>` limits | never **generate** it; no features for novel words; infeasible for morphologically rich languages; destroys rare-word-heavy text |
| Why tokenization was redefined | (1) the 2016 result that subword segmentation improves MT; (2) neural LMs need a **fixed-size vocabulary** |
| New definition | segmenting into **non-typographically and non-linguistically motivated** units, usually smaller than classical tokens; the old units are now **pre-tokens**, the old step **pre-tokenization** |
| Morpheme | smallest meaning-bearing unit; `unlikeliest` = {un-, likely, -est} |
| BPE main idea | **use data to automatically tell what the tokens should be** |
| BPE two parts | **token learner** (corpus ⇒ vocabulary) and **token segmenter** (sentences ⇒ tokens) |
| BPE learner step 3 | choose the **2 most frequently adjacent** tokens, **respecting word boundaries** |
| BPE vocabulary size | $\lvert V\rvert = \lvert\text{initial characters}\rvert + k$ |
| BPE segmenter | replays the learned merges **greedily, in the order they were learned** |
| Choice of $k$ | an **open research question** |
| BPE vs WordPiece | BPE merges the most **frequent** pair; WordPiece merges the pair that most increases the **likelihood** of the training data |
| SentencePiece | subword tokenization **without pre-tokenization**, for languages without spaces |
| After normalization | most tokenizers are **irreversible**; with pretrained LMs we do **no** normalization except casing |

## [Lec 3 — N-gram Language Models I: The Task, the Chain Rule and Counting](../notes/week-01/03-ngram-lm-1.md)

| Item | Exactly this |
|---|---|
| Language model | a model that assigns a probability to the next word **or** to a whole word sequence |
| LM goal | $P(W) = P(w_1, w_2, \ldots, w_T)$ |
| Related task | $P(w_n \mid w_1 \ldots w_{n-1})$ |
| Conditional probability | $P(B\mid A) = P(A,B)/P(A) \Rightarrow P(A,B) = P(A)P(B\mid A)$ |
| Chain rule | $P(w_1\ldots w_T) = \prod_{t=1}^{T} P(w_t \mid w_1 \ldots w_{t-1})$ — **exact, no assumption** |
| Markov assumption | $P(w_n \mid w_{1:n-1}) \approx P(w_n \mid w_{n-1})$ (bigram) |
| General n-gram | $P(w_i \mid w_1\ldots w_{i-1}) \approx P(w_i \mid w_{i-n+1}\ldots w_{i-1})$ |
| Bigram model | $P(w_1\ldots w_T) \approx \prod_{t=1}^{T} P(w_t \mid w_{t-1})$ |
| MLE for a bigram | $P(w_i \mid w_{i-1}) = \dfrac{C(w_{i-1}, w_i)}{C(w_{i-1})}$ |
| MLE in words | the value that makes the observed data most probable = the **relative frequency** |
| Parameters of an n-gram model | $\lvert V\rvert^{n}$ |
| Context length | an $n$-gram model conditions on $n-1$ words |
| Markov order | bigram = 1st order, trigram = 2nd order |
| Log space | $\log P(W) = \sum_t \log P(w_t \mid \text{context})$; avoids underflow, adding beats multiplying |
| Self-supervised | labels come from the input itself — here, the next word |
| Boundary tokens | `<s>` gives word 1 a context; `</s>` lets the model end a sentence |

## [Lec 4 — N-gram Language Models II: Smoothing and Perplexity](../notes/week-01/04-ngram-lm-2-smoothing-perplexity.md)

| Item | Exactly this |
|---|---|
| Add-1 (Laplace) bigram | $P_{\text{add-1}}(w_i \mid w_{i-1}) = \dfrac{C(w_{i-1},w_i)+1}{C(w_{i-1})+\lvert V\rvert}$ |
| Add-k | $P_{\text{add-}k} = \dfrac{C(w_{i-1},w_i)+k}{C(w_{i-1})+k\lvert V\rvert}$ |
| Why $+\lvert V\rvert$ | you added 1 to each of $\lvert V\rvert$ continuations, so the total must grow by $\lvert V\rvert$ |
| Interpolation | $\hat{P} = \lambda_1 P_{\text{tri}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{uni}}$, $\sum\lambda_j = 1$ |
| Backoff vs interpolation | backoff uses **one** order; interpolation **mixes all**; interpolation usually works better |
| Perplexity | $PP(W) = P(w_1\ldots w_N)^{-1/N} = \sqrt[N]{\dfrac{1}{P(w_1\ldots w_N)}}$ |
| Perplexity (bigram) | $PP(W) = \sqrt[N]{\prod_{i=1}^{N}\dfrac{1}{P(w_i \mid w_{i-1})}}$ |
| Direction | **lower perplexity is better**; minimising $PP$ ≡ maximising $P$ |
| Ranges | probability $[0,1]$; perplexity $[1,\infty]$ |
| Interpretation | weighted average **branching factor** — effective number of choices per word |
| Uniform model | $PP = \lvert V\rvert$ exactly |
| Cross-entropy link | $PP(W) = 2^{H(W)}$, $H$ in bits per word |
| Extrinsic vs intrinsic | extrinsic = in a real task; intrinsic = perplexity, and it need not track task performance |
| Dev set exists because | repeated testing on the test set implicitly tunes to it |

## [Lec 5 — NLP Tasks and Paradigms](../notes/week-01/05-nlp-tasks-and-paradigms.md)

| Item | Exactly this |
|---|---|
| The paradigms | text classification · sequence labeling · text generation · structured prediction |
| Classification, formal | input: a document $d$ and a **fixed** class set $C = \{c_1,\ldots,c_J\}$; output: a predicted class $c \in C$ |
| Classification, ML version | add a training set of $m$ hand-labelled docs $(d_1,c_1)\ldots(d_m,c_m)$; output a learned classifier $\gamma : d \to c$ |
| Named classifiers | Naïve Bayes, SVM, neural networks, $k$-NN |
| 2×2 matrix orientation (deck) | **rows = system output, columns = gold standard** (`sklearn` is the transpose) |
| Accuracy | $\dfrac{TP+TN}{TP+FP+FN+TN}$ |
| Precision | $\dfrac{TP}{TP+FP}$ — along the system-positive **row**; guards against false alarms |
| Recall | $\dfrac{TP}{TP+FN}$ — down the gold-positive **column**; guards against misses |
| $F_\beta$ | $\dfrac{(\beta^2+1)PR}{\beta^2 P + R}$ |
| $F_1$ | $\dfrac{2PR}{P+R} = \dfrac{2}{1/P + 1/R}$ — the **harmonic** mean |
| Why harmonic | it sits near the *smaller* of $P,R$, so you cannot game it by maximising one; $F_1 \le$ arithmetic mean, equal only when $P=R$ |
| $\beta$ direction | recall is weighted $\beta^2$ times as heavily as precision; $\beta>1$ favours **recall**, $\beta<1$ favours **precision** |
| Multi-class $P_c$, $R_c$ | $P_c$ = diagonal ÷ **row** sum; $R_c$ = diagonal ÷ **column** sum (deck orientation) |
| Macro-averaging | compute the metric per class, then **average over classes** — all classes count equally |
| Micro-averaging | pool all decisions into **one** confusion matrix, then compute — dominated by frequent classes |
| Single-label identity | pooled FP = pooled FN ⇒ micro-$P$ = micro-$R$ = micro-$F_1$ = **accuracy** |
| Open class | **content** words; gains new members — nouns, main verbs, adjectives, adverbs, interjections, numbers |
| Closed class | **function** words; fixed inventory — determiners, conjunctions, pronouns, prepositions, particles, **auxiliary** verbs |
| POS tagging | assign a part of speech to **each word**; words often have more than one POS (`book` = VERB or NOUN) |
| POS methods | HMM → MEMM → CRF → RNNs, Transformers |
| POS evaluation | **accuracy** and **macro-F1** (equal importance to each tag) |
| Conditional LM for dialogue | $\hat{r}_t = \arg\max_{w \in V} P(w \mid q, r_1 \ldots r_{t-1})$ |
| Dependency parsing | input: sentence $x = w_1,\ldots,w_n$; output: a dependency **graph** $G$ |

## [Lec 6 — Supervised Learning](../notes/week-02/06-supervised-learning.md)

| Item | Exactly this |
|---|---|
| Supervised learning | define a mapping from input to output; **learn it from paired input/output examples** |
| Unsupervised learning | construct a model from input data **without corresponding output labels** |
| Reinforcement learning | an **agent** in the world learning to choose actions leading to high **reward** |
| Machine learning | subset of AI that learns by **fitting mathematical models to observed data** |
| Deep learning's place | a band **across** supervised / unsupervised / RL — "DNNs contribute to each of the areas" |
| Model | a **family of equations**, $\mathbf{y} = f(\mathbf{x};\theta)$ (deck: $\mathbf{f}[\mathbf{x},\boldsymbol{\phi}]$) |
| Inference | computing the outputs from the inputs, with $\theta$ fixed |
| Training | finding parameters that predict outputs well on a training dataset |
| Training set | $\mathcal{D} = \{\mathbf{x}_i, \mathbf{y}_i\}_{i=1}^{I}$, $I$ pairs |
| Loss function (= cost function) | $\mathcal{L}(\theta)$ — returns a **scalar** that is **smaller when the model maps inputs to outputs better** |
| Training, formally | $\hat{\theta} = \operatorname{argmin}_{\theta}\big[\mathcal{L}(\theta)\big]$ |
| Testing | run on a **separate** test dataset of input/output pairs; measure **generalization** |
| 1-D linear regression model | $y = \theta_0 + \theta_1 x$; $\theta_0$ = y-offset, $\theta_1$ = slope; 2 parameters |
| Least-squares loss | $\mathcal{L}(\theta) = \sum_{i=1}^{I}\left(\theta_0 + \theta_1 x_i - y_i\right)^2$ |
| Closed-form slope | $\hat{\theta}_1 = \dfrac{I\sum x_iy_i - \sum x_i\sum y_i}{I\sum x_i^2 - (\sum x_i)^2}$, $\;\hat{\theta}_0 = \bar{y} - \hat{\theta}_1\bar{x}$ |
| Training (the picture) | walking downhill on the loss surface — **gradient descent** |
| Why generalization fails | model **too simple**; or model **too complex** → fits statistical peculiarities → **overfitting** |
| Objection 1 + answer | closed form exists — but not for more complex models |
| Objection 2 + answer | exhaustive search works — but not with a million parameters |
| Linear-layer parameter count | $D_{\text{in}}D_{\text{out}} + D_{\text{out}}$ |

## [Lec 7 — Shallow Neural Networks](../notes/week-02/07-shallow-neural-networks.md)

| Item | Exactly this |
|---|---|
| Example shallow network (deck form) | $y = \phi_0 + \phi_1 a[\theta_{10}+\theta_{11}x] + \phi_2 a[\theta_{20}+\theta_{21}x] + \phi_3 a[\theta_{30}+\theta_{31}x]$ |
| Two-stage form | $h_d = a(\theta_{d0}+\theta_{d1}x)$; $\;y = \phi_0 + \sum_d \phi_d h_d$ |
| ReLU | $a(z) = \mathrm{ReLU}(z) = 0$ if $z<0$, $z$ if $z \ge 0$; equivalently $\max(0,z)$ |
| General form | $h_d = a\big(\theta_{d0}+\sum_{i=1}^{D_i}\theta_{di}x_i\big)$, $\;y_j = \phi_{j0}+\sum_{d=1}^{D}\phi_{jd}h_d$ |
| Matrix form | $\mathbf{h} = a(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})$, $\;\mathbf{y} = \mathbf{W}^{(2)}\mathbf{h}+\mathbf{b}^{(2)}$ |
| Parameter count | $D(D_i+1) + D_o(D+1)$ |
| Joint location of unit $d$ | $x_d^{\ast} = -\,b^{(1)}_d / W^{(1)}_d$ |
| Slope in a region | $\sum_{d \in \text{on}} W^{(2)}_d W^{(1)}_d$ |
| Regions in 1-D | at most $H+1$ for $H$ hidden units — **1 joint per ReLU** |
| Universal Approximation Theorem | "with enough hidden units, a shallow neural network can describe any **continuous** function on a **compact subset of $\mathbb{R}^D$** to **arbitrary precision**" |
| What UAT does not give | a width bound, a guarantee that training finds it, or any generalisation claim |
| Shallow vs deep | one hidden layer = shallow; more than one = deep |
| Pre-activation / activation | before / after the activation function |
| Fully connected / feedforward | everything in a layer connects to everything in the next / no loops |
| Capacity | $\approx$ number of hidden units |
| Biases / weights | Y-offsets / slopes |
| Diagram reading rule | each parameter multiplies its source and adds to its target |

## [Lec 8 — Deep Neural Networks](../notes/week-02/08-deep-neural-networks.md)

| Item | Exactly this |
|---|---|
| Deep network | a network with **more than one hidden layer** |
| ReLU networks describe | **piecewise linear** mappings — shallow *and* deep |
| Why deep | **many more linear regions for the same number of parameters** (Montufar et al., 2014) |
| Composition | $f(\mathbf{x};\theta) = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{x})$ |
| Layer equations | $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$, $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$; output layer has **no** $\sigma$ |
| Deck's form | $\mathbf{h}_1 = \mathrm{a}[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0\mathbf{x}]$, $\mathbf{h}_2 = \mathrm{a}[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\mathbf{h}_1]$, $\mathbf{y} = \boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2\mathbf{h}_2$ |
| $\boldsymbol{\beta}_k$ / $\boldsymbol{\Omega}_k$ | **bias vector** / **weight matrix** of layer $k$ |
| Combining two nets | $\psi_{j0} = \theta'_{j0} + \theta'_{j1}\phi_0$ and $\psi_{jk} = \theta'_{j1}\phi_k$ (so $\boldsymbol{\Psi}$ is rank 1) |
| Depth / width | $K$ = number of layers / $D_k$ = hidden units per layer |
| Hyperparameters | depth, width, $\eta$, batch size — **chosen before training**; tuned by hyperparameter search |
| Parameters | weights and biases — learned **during** training |
| Max regions, $D_i{=}1$ | shallow: $H + 1$; deep: $(D+1)^K$ |
| Params, $D_i{=}1$ | shallow: $3H + 1$; layer $l$: $D_l D_{l-1}$ weights $+\ D_l$ biases |
| Universal approximation | obeyed by **both** — so it is *not* the reason to go deep |
| Deeper models | fit **faster** (fewer epochs) at matched parameter count |

## [Lec 9 — Backpropagation](../notes/week-02/09-backpropagation.md)

| Item | Exactly this |
|---|---|
| Who and when | **Rumelhart, Hinton, Williams (1986)** |
| What backprop is | an efficient algorithm for *computing the gradient*; **not** an optimiser |
| Forward pass | compute **and store** $\mathbf{f}_0 = \mathbf{b}^{(0)}+\mathbf{W}^{(0)}\mathbf{x}$, $\mathbf{h}_k = \mathrm{a}[\mathbf{f}_{k-1}]$, $\mathbf{f}_k = \mathbf{b}^{(k)}+\mathbf{W}^{(k)}\mathbf{h}_k$ |
| Backward recursion | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{f}_{k-1}} = \mathbb{I}[\mathbf{f}_{k-1}>0]\odot\left(\mathbf{W}^{(k)\top}\dfrac{\partial\mathcal{L}}{\partial\mathbf{f}_k}\right)$ |
| Bias gradient | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{b}^{(k)}} = \dfrac{\partial\mathcal{L}}{\partial\mathbf{f}_k}$ (pass-through) |
| Weight gradient | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{W}^{(k)}} = \dfrac{\partial\mathcal{L}}{\partial\mathbf{f}_k}\mathbf{h}_k^{\top}$ (outer product) |
| Derivative of a linear map | $\dfrac{\partial(\mathbf{W}\mathbf{h})}{\partial\mathbf{h}} = \mathbf{W}^{\top}$ — **transposed** |
| Derivative w.r.t. bias | $\partial(\mathbf{b}+\mathbf{W}\mathbf{h})/\partial\mathbf{b} = \mathbf{I}$ |
| Shape rule | the gradient always has the **same shape** as what you differentiate w.r.t. |
| Jacobian | matrix of all $\partial f_j/\partial a_k$; the deck lays it out $(\dim\mathbf{a})\times(\dim\mathbf{f})$ |
| ReLU derivative | 1 if $z>0$, 0 if $z<0$, undefined at $z=0$; **convention: 0** |
| ReLU backward | elementwise mask $\mathbb{I}[\mathbf{f}>0]\odot$ (its Jacobian is diagonal) |
| Toy function | $\mathrm{f} = \beta_3+\omega_3\cos[\beta_2+\omega_2\exp[\beta_1+\omega_1\sin[\beta_0+\omega_0 x]]]$; chain $f_0,h_1{=}\sin,f_1,h_2{=}\exp,f_2,h_3{=}\cos,f_3,\mathcal{L}{=}(f_3{-}y)^2$ |
| $\partial\mathcal{L}/\partial f_3$ | $2(f_3 - y_i)$ |
| Cost | finite diff: $P+1$ (one-sided) or $2P$ (central) forward passes; backprop: $\approx 2$, **independent of $P$** |
| Algorithmic differentiation | each component knows its own derivative; you specify the order; works on any **acyclic** graph |
| Backprop = | **reverse-mode** automatic differentiation |
| Why reverse mode | one sweep per *output*; deep learning has one output (the scalar loss) and $10^8$ inputs |

## [Lec 10 — Gradient Descent and Initialization](../notes/week-02/10-gradient-descent-and-init.md)

| Item | Exactly this |
|---|---|
| Gradient descent update | $\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}(\theta)$ |
| What $\eta$ controls | the **magnitude** of the step, not its direction |
| GD stability (quadratic, curvature $c$) | $\eta < 2/c$ |
| GD guarantee | global minimum **only if the loss is convex**; otherwise only a stationary point |
| SGD update | $\theta_{t+1} \leftarrow \theta_t - \eta\sum_{i\in\mathcal{B}_t}\partial\ell_i/\partial\theta$ |
| Epoch | one full pass through the dataset, sampling **without replacement** |
| Momentum | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + (1-\beta)\nabla\mathcal{L}$; $\;\theta \leftarrow \theta - \eta\mathbf{v}_t$; $\beta \approx 0.9$ |
| Momentum unrolled | $\mathbf{v}_t = (1-\beta)\sum_{k\ge0}\beta^k\nabla\mathcal{L}(\theta_{t-k})$ — infinite weighted sum, weights decay backwards in time |
| Normalized gradients | $\theta \leftarrow \theta - \eta\,\mathbf{m}/(\sqrt{\mathbf{v}}+\epsilon)$ with $\mathbf{m}=\mathbf{g}$, $\mathbf{v}=\mathbf{g}^2$ — **will not converge** |
| Adam 1st moment | $\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$ |
| Adam 2nd moment | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ (elementwise square) |
| Adam bias correction | $\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^{\,t})$, $\;\hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^{\,t})$ |
| Why bias correction | both moments start at $\mathbf{0}$, so early estimates are biased **toward zero** |
| Adam update | $\theta_t \leftarrow \theta_{t-1} - \eta\,\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$ |
| Adam defaults | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| Adam's first step | exactly $\eta\cdot\mathrm{sign}(g)$ in magnitude $\eta$ |
| Initialisation aim | **keep the variance the same between two layers** |
| Deck's variance recursion | $\sigma^2_{f'} = D_h\,\sigma_W^2\,\sigma_f^2/2$ (ReLU) |
| **He** init (ReLU) | $\mathrm{Var}(W) = 2/n_{\text{in}}$ — the 2 compensates for ReLU zeroing half its inputs |
| **Xavier/Glorot** init (tanh, sigmoid) | $\mathrm{Var}(W) = 2/(n_{\text{in}}+n_{\text{out}})$ |
| Biases | initialise to **0** |
| Zero weight init | fatal — all units compute the same thing and get the same gradient; **symmetry is never broken** |
| Training hyperparameters | learning algorithm, batch size, learning-rate schedule, momentum coefficients → **hyper-parameter search** |

## [Lec 11 — Word Representation](../notes/week-03/11-word-representation.md)

| Item | Exactly this |
|---|---|
| One-hot vector | length $\lvert V\rvert$, a single 1 at the word's index, 0 elsewhere |
| One-hot dot product | $\mathbf{e}_u^\top\mathbf{e}_v = 0$ for all $u \neq v$ — all one-hots are **orthogonal** |
| Distributional hypothesis | words occurring in similar **contexts** have similar meanings |
| Wittgenstein | "The meaning of a word is its use in the language" |
| Zellig Harris (1954) | if A and B have almost identical **environments**, they are **synonyms** |
| Embedding | a vector representing word meaning; "embedded" into a continuous space |
| Co-occurrence / distributional matrix | **targets × contexts**, cell = count of co-occurrences in a window / sentence / document |
| Cosine similarity | $\cos(\mathbf{u},\mathbf{v}) = \dfrac{\mathbf{u}^\top\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}$ |
| Why cosine | it normalises away vector **length**, i.e. word frequency, comparing only the context *pattern* |
| PMI | $\mathrm{PMI}(w_1,w_2) = \log_2 \dfrac{P(w_1,w_2)}{P(w_1)P(w_2)}$, base 2, in **bits** |
| PMI estimators | $P(w_1,w_2) = \mathrm{freq}(w_1,w_2)/N$, $P(w) = \mathrm{freq}(w)/N$ |
| PMI fast form | $\log_2 \dfrac{N \cdot C_{wc}}{(\text{row sum})(\text{col sum})}$ |
| PMI meaning | how many times **more than chance** the pair co-occurs; $\mathrm{PMI}=k$ means $2^k\times$ |
| PPMI | $\mathrm{PPMI}(w,c) = \max(\mathrm{PMI}(w,c), 0)$ |
| Why clamp | $\log_2 0 = -\infty$; and negative PMI needs an infeasibly large corpus to be reliable |
| tf-idf | $f_{ij}\cdot\log(N/N_j)$, then $L_2$-normalise within the document |
| tf-idf's three factors | word frequency $\propto f_{ij}$; document length $\propto 1/\lvert D_i\rvert$; document frequency $\propto 1/N_j$ |
| Sparse vs dense | sparse: long ($\lvert V\rvert$), mostly zero, counted. Dense: short (50–1000), mostly non-zero, learned |
| Three reasons for dense | fewer parameters to tune; generalise better than explicit counts; capture **synonymy** (car/automobile are distinct dimensions in a count matrix) |
| Vector offset | $\mathbf{x}_{\text{apple}} - \mathbf{x}_{\text{apples}} \approx \mathbf{x}_{\text{car}} - \mathbf{x}_{\text{cars}}$ |
| Analogy arithmetic | $a{:}b::c{:}d \Rightarrow \mathbf{x}_d \approx \mathbf{x}_b - \mathbf{x}_a + \mathbf{x}_c$, then nearest by cosine |
| Deck's analogy formula | $d = \arg\max_x \dfrac{(\mathbf{w}_b - \mathbf{w}_a + \mathbf{w}_c)^\top\mathbf{w}_x}{\lVert\mathbf{w}_b - \mathbf{w}_a + \mathbf{w}_c\rVert}$ |
| Static embeddings | **one vector per word type**, context-independent — the limitation ELMo/BERT remove |

## [Lec 12 — Learning Word Representation I: word2vec Skip-gram](../notes/week-03/12-word2vec-skipgram.md)

| Item | Exactly this |
|---|---|
| Skip-gram direction | **centre → context**: predict the outside words from the centre word |
| CBOW direction | **context → centre**: predict the centre word from the summed context |
| Two vectors per word | $\mathbf{v}_w$ when $w$ is **centre**, $\mathbf{u}_w$ when $w$ is **context** |
| Likelihood | $\mathcal{L}'(\theta) = \prod_{t=1}^{T}\prod_{-m\le j\le m,\;j\ne0} P(w_{t+j}\mid w_t;\theta)$ — **maximise** |
| Loss | $\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{-m\le j\le m,\;j\ne0}\log P(w_{t+j}\mid w_t;\theta)$ — **minimise** |
| Why log | monotone, so the argmax is unchanged; turns the product into a sum and stops underflow |
| Why $1/T$ | makes it a per-token average, comparable across corpus sizes |
| $P(o\mid c)$ | $\dfrac{\exp(\mathbf{u}_o^\top\mathbf{v}_c)}{\sum_{w\in V}\exp(\mathbf{u}_w^\top\mathbf{v}_c)}$ |
| Role of the dot product | similarity score (cosine $\times$ the two lengths) |
| Role of $\exp$ | maps $\mathbb{R}\to(0,\infty)$, order-preserving, amplifies score gaps |
| Role of the denominator | normalises over the **whole vocabulary** so the $\lvert V\rvert$ probabilities sum to 1 |
| Gradient | $\dfrac{\partial}{\partial\mathbf{v}_c}\log P(o\mid c) = \mathbf{u}_o - \sum_{x\in V}P(x\mid c)\mathbf{u}_x$ |
| Its interpretation | **observed − expected** |
| Output-vector gradient | $\partial\mathcal{L}/\partial\mathbf{u}_w = \left(P(w\mid c) - \mathbb{1}[w{=}o]\right)\mathbf{v}_c$ |
| Update rule | $\theta^{\text{new}} = \theta^{\text{old}} - \eta\,\partial\mathcal{L}/\partial\theta^{\text{old}}$ (deck writes $\alpha$) |
| Parameter count | $2\lvert V\rvert d$ — two embedding matrices, no biases |
| Pairs from $T$ tokens, radius $m$ | $2mT - m(m+1)$ (with truncation at both ends) |
| The issue | the softmax denominator costs $O(\lvert V\rvert d)$ **per update** |
| The two fixes | negative sampling, hierarchical softmax — and **both change the objective function** |

## [Lec 13 — Negative Sampling, Hierarchical Softmax and GloVe](../notes/week-03/13-negative-sampling-glove.md)

| Item | Exactly this |
|---|---|
| The problem | skip-gram's softmax denominator sums over all of $V$ → $O(\lvert V\rvert)$ per update |
| Negative sampling reframes it as | **binary classification**: did $(w,c)$ come from the corpus? |
| Classifier | $P(D{=}1 \mid w,c) = \sigma(\mathbf{v}_c^\top\mathbf{u}_w) = 1/(1 + e^{-\mathbf{v}_c^\top\mathbf{u}_w})$ |
| NS objective (maximise) | $\log\sigma(\mathbf{u}_o^\top\mathbf{v}_c) + \sum_{k=1}^{K}\mathbb{E}_{w_k\sim P_n}\big[\log\sigma(-\mathbf{u}_{w_k}^\top\mathbf{v}_c)\big]$ |
| The identity behind the minus sign | $1 - \sigma(z) = \sigma(-z)$ |
| Noise distribution | $P_n(w) \propto U(w)^{3/4}$ — unigram raised to the **3/4** power, renormalised |
| Effect of the 3/4 | **flattens**: rare words sampled more often, frequent words less |
| NS gradient (positive) | $\partial\mathcal{L}/\partial\mathbf{c}_{\text{pos}} = [\sigma(\mathbf{c}_{\text{pos}}^\top\mathbf{w}) - 1]\mathbf{w}$ |
| NS gradient (negative) | $\partial\mathcal{L}/\partial\mathbf{c}_{\text{neg}} = [\sigma(\mathbf{c}_{\text{neg}}^\top\mathbf{w})]\mathbf{w}$ |
| NS cost | $O(Kd)$ instead of $O(\lvert V\rvert d)$ |
| Hierarchical softmax structure | vocabulary = **leaves of a binary tree**; $\lvert V\rvert - 1$ internal nodes, each with a learnable vector; **no per-word output vector** |
| HS probability | $P(w\mid w_i) = \prod_{j=1}^{L(w)}\sigma\big([\![n(w,j{+}1)=ch(n(w,j))]\!]\cdot\mathbf{v}_{n(w,j)}^\top\mathbf{v}_{w_i}\big)$ |
| The bracket | $+1$ if true (left), $\mathbf{-1}$ otherwise (right) — **not** 0/1 |
| $L(w)$ | number of **internal** (non-leaf) nodes on the root-to-leaf path |
| HS cost | $O(d\log_2\lvert V\rvert)$ |
| Huffman | frequent words get short codes → sit shallower → cheaper |
| Two embedding sets | target $\mathbf{W}$, context $\mathbf{C}$; the deck says **add them**: $\mathbf{w}_i + \mathbf{c}_i$ |
| GloVe notation | $X_{ij}$ counts, $X_i = \sum_k X_{ik}$, $P_{ij} = X_{ij}/X_i$ |
| GloVe's insight | **ratios** $P_{ik}/P_{jk}$ encode meaning; raw probabilities do not |
| Log-bilinear | $\mathbf{w}_i^\top\tilde{\mathbf{w}}_k = \log P_{ik}$; hence $\mathbf{w}_x^\top(\mathbf{w}_a - \mathbf{w}_b) = \log\frac{P(x\mid a)}{P(x\mid b)}$ |
| Symmetric form | $\mathbf{w}_i^\top\tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}$ ($\log X_i$ absorbed into $b_i$) |
| GloVe objective | $\mathcal{L} = \sum_{i,j} f(X_{ij})\big(\mathbf{w}_i^\top\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2$ |
| Why $f$ | $f(0)=0$ skips zero cells (avoids $\log 0$); saturates so frequent pairs are not over-weighted |
| $f$ | $(x/x_{\max})^{\alpha}$ for $x<x_{\max}$, else 1; $x_{\max}=100$, $\alpha=3/4$ |
| Skip-gram ↔ GloVe | skip-gram $= \sum_i X_i H(P_i, Q_i)$, a count-weighted cross-entropy; GloVe swaps cross-entropy for weighted least squares on $\log X$ |
| One-line contrast | word2vec = **prediction**-based, local windows; GloVe = **count**-based, global matrix |

## [Lec 14 — Word Vectors: Other Extensions](../notes/week-03/14-fasttext-and-beyond-words.md)

| Item | Exactly this |
|---|---|
| Phrase score | $\text{score}(w_i,w_j) = \dfrac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i)\times\text{count}(w_j)}$, merge if above threshold |
| What $\delta$ does | discounting constant; forces bigrams with count $< \delta$ to score negative |
| What the phrase score is | a PMI-style association ratio without the $\log$ and the corpus-size factor |
| fastText citation | **Bojanowski et al., 2017**; library from **Facebook AI Research** |
| fastText core idea | a word is a **bag of character n-grams**, plus the whole-word token |
| fastText n-gram range | **$n = 3$ to $6$**, all lengths used |
| Boundary markers | `<` and `>` pad the word, so `her` in `where` ≠ `<he` in `herself` |
| Word vector | $\mathbf{e}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$ — the **sum** of n-gram vectors |
| fastText score | $s(w,c) = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c$ |
| fastText loss | skip-gram with negative sampling: $\log(1+e^{-s(w_t,w_c)}) + \sum_{n\in\mathcal{N}_{t,c}}\log(1+e^{s(w_t,n)})$ |
| OOV handling | sum (deck: average) the n-gram vectors of the unseen word; **word2vec/GloVe return nothing** |
| Hashing | n-grams hashed into $B$ buckets to bound memory; the paper uses $B = $ **2 million** |
| doc2vec | Le and Mikolov **2014**; adds a **paragraph matrix** $\mathbf{D}$, one vector per document |
| Sentence baselines | mean embedding; tf-idf-weighted mean |
| Poincaré embeddings | Nickel and Kiela; **hyperbolic** space for hierarchies; distance from origin = generality |
| Node embedding goal | $\text{similarity}(u,v) \approx \mathbf{z}_u^\top\mathbf{z}_v$ |
| DeepWalk | random walks = "sentences", nodes = "words", then **skip-gram** |
| DeepWalk loss | $\mathcal{L} = \sum_{u\in V}\sum_{v\in N_R(u)} -\log\dfrac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\top\mathbf{z}_n)}$ |
| KG triple | $(h, r, t)$ = (head entity, relation, tail entity) |
| KG completion task | given (head, relation), **predict the missing tail** — not the same as link prediction |
| TransE constraint | $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ iff the triple is true |
| TransE score | $f_r(h,t) = -\lVert \mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ — **negated** distance |
| TransE loss | $\sum \left[\gamma + d(\mathbf{h}+\mathbf{r},\mathbf{t}) - d(\mathbf{h}'+\mathbf{r},\mathbf{t}')\right]_+$ |
| Corruption | replace the **head or the tail** with a random entity; never the relation |
| TransE normalisation | entity embeddings renormalised to **unit norm every iteration** |

## [Lec 15 — Cross-Lingual Representations](../notes/week-03/15-cross-lingual-representations.md)

| Item | Exactly this |
|---|---|
| Why cross-lingual embeddings | data availability across languages is very rarely the same; transfer from high- to low-resource |
| The payoff | **zero-shot transfer**: train the classifier on English labels, apply it to another language's vectors |
| 4 model types | **monolingual mapping**, **pseudo-cross-lingual**, **cross-lingual training**, **joint optimization** |
| 5 data tiers, dearest→cheapest | **word-aligned → sentence-aligned → document-aligned → lexicon → no parallel data** |
| Document-aligned subtypes | **topic**-aligned (Wikipedia) or **label/class**-aligned (sentiment / multi-class datasets) |
| Mikolov's observation | geometric relations between words are **similar across languages** |
| Mapping objective | $\min_{\mathbf{W}} \sum_{i=1}^{n} \lVert \mathbf{W}\mathbf{x}_i - \mathbf{z}_i \rVert^2$ |
| Closed form | $\mathbf{W} = \mathbf{Z}\mathbf{X}^{\top}(\mathbf{X}\mathbf{X}^{\top})^{-1}$; the deck says SGD |
| CCA (Faruqui & Dyer) | **two** matrices, one per language, projecting **both** into a **shared third space** of dimension $d$, maximising **correlation** |
| Merge and shuffle (Vulić & Moens) | **concatenate** aligned documents, **randomly permute** the words, train an off-the-shelf monolingual model |
| Bilingual compositional sentence model (Hermann & Blunsom) | sentence vector = **sum of word embeddings**; objective = **distance between the two sentence representations** |
| BAE (Lauly et al.) | bag-of-words autoencoder, **tree-based decoder like hierarchical softmax**, **four reconstruction losses** per aligned pair |
| Bilingual skip-gram (Luong et al.) | skip-gram objective as **both** the monolingual and the cross-lingual loss; source words also predict their **aligned** target words — needs **word alignments** |
| BilBOWA (Gouws et al.) | minimises distance between the **means** of the word representations in aligned sentences; **no word alignments**; uses extra **monolingual** data |
| IndicFT | fastText on AI4Bharat's IndicNLP Corpora, **11 Indian languages**; subwords matter because Indic languages are **highly agglutinative** |
| Translation accuracy @$k$ | fraction of source test words whose gold translation is in the top $k$ retrieved candidates |

## [Lec 16 — RNN Language Models](../notes/week-04/16-rnn-language-models.md)

| Item | Exactly this |
|---|---|
| RNN recurrence | $\mathbf{h}_t = \sigma(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{e}_t + \mathbf{b}_h)$ |
| RNN output | $\hat{\mathbf{y}}_t = \mathrm{softmax}(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$ |
| Deck's symbols | $h_t = g(Uh_{t-1} + Wx_t)$, $y_t = \mathrm{softmax}(Vh_t)$ — $W$ input, $U$ recurrent, $V$ output |
| Matrix shapes | $W : d_h\times d_{\text{in}}$, $U : d_h\times d_h$, $V : d_{\text{out}}\times d_h$ |
| Initial state | $\mathbf{h}_0$, normally $\mathbf{0}$ |
| Fixed-window LM | $\mathbf{e}=[\mathbf{e}_1;\ldots;\mathbf{e}_c]$, $\mathbf{h}=f(\mathbf{W}\mathbf{e}+\mathbf{b}_1)$, $\hat{\mathbf{y}}=\mathrm{softmax}(\mathbf{U}\mathbf{h}+\mathbf{b}_2)$ |
| Fixed-window flaws | window too small; enlarging window enlarges $\mathbf{W}$; window never large enough; **no symmetry / no weight sharing across positions** |
| Fixed-window gains | no sparsity problem; no need to store observed n-grams |
| RNN advantages | any-length input; model size independent of length; (in theory) long-range context; weights shared across time steps |
| Composition functions | element-wise (sum), concatenation, FFN, CNN, RNN, Transformer |
| Why sum fails | order-blind — commutative |
| Per-step loss | $\mathcal{L}_t = -\log \hat{y}_{t,\,w_{t+1}}$ |
| Sequence loss | $\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T}\mathcal{L}_t$ |
| Training regime | **self-supervision** — the next word in the corpus is the label |
| Loss ↔ perplexity | $PP = e^{\mathcal{L}}$ (nats); see [Lec 4](../notes/week-01/04-ngram-lm-2-smoothing-perplexity.md) |

## [Lec 17 — RNN Applications: Text Generation, Sequence Labeling, Text Classification](../notes/week-04/17-rnn-applications.md)

| Item | Exactly this |
|---|---|
| Autoregressive generation | sample a word from $\mathbf{y}_t$, feed its embedding in as input at $t{+}1$, repeat until `</s>` or max length |
| Generation start | special begin-of-sentence token `<s>`, $\mathbf{h}_0 = \mathbf{0}$, parameters frozen |
| Train vs test input | training uses the **true** previous word; test uses the **model's own** output → **exposure bias** |
| Sequence-labeling task | assign a label from a **small fixed set** to **each** element; inputs = word embeddings, outputs = softmax tag probabilities |
| Named entity | anything that can be referred to with a **proper name**; often multi-word |
| Four core NER types | **PER, LOC, ORG, GPE** (+ extensions: dates, times, prices) |
| NER = two jobs | find **spans** that constitute proper names; **tag the type** |
| Why NER is hard | (1) **segmentation** — POS has none, each word gets one tag; (2) **type ambiguity** |
| Why NER matters | sentiment analysis, question answering, information extraction |
| **B** | token that **begins** a span |
| **I** | token **inside** a span |
| **O** | token **outside** any span |
| BIO tag count | $2n+1$ for $n$ entity types: $n$ B tags, $n$ I tags, 1 O tag |
| IO / BIOES counts | $n+1$ / $4n+1$ |
| Why B is needed | IO cannot separate **two adjacent entities of the same type** |
| Sentence encoding, basic | the **final hidden state** $\mathbf{h}_T$ |
| Sentence encoding, better | **element-wise max or mean** over all hidden states |
| Bidirectional RNN | $\mathbf{h}_t^f = \text{RNN}_{\text{fwd}}(\mathbf{x}_1..\mathbf{x}_t)$, $\mathbf{h}_t^b = \text{RNN}_{\text{bwd}}(\mathbf{x}_n..\mathbf{x}_t)$, $\mathbf{h}_t = [\mathbf{h}_t^f;\mathbf{h}_t^b]$ |
| Bi-RNN limitation | needs the **entire sequence up front** → **unusable for language modelling or real-time generation** |
| Bi-RNN size | hidden state $2h$; parameter count exactly **doubles** |
| RNN tagger params (no bias) | $d\,d_h + d_h^2 + d_h K$, independent of sentence length |

## [Lec 18 — RNN for Sequence to Sequence, and Attention](../notes/week-04/18-seq2seq-and-attention.md)

| Item | Exactly this |
|---|---|
| Seq2seq innovation | the lengths $n_x$ and $n_y$ **can vary from each other** |
| Three components | **encoder**, **context vector**, **decoder** |
| Basic context vector | $\mathbf{c} = \mathbf{h}^e_n$, the **final** encoder hidden state |
| Encoder / decoder shape | encoder = many-to-one; decoder = one-to-many |
| MT objective | $p(y \mid x) = \prod_{i=1}^{m} p(y_i \mid y_1,\ldots,y_{i-1}, x)$ |
| Its name | **Conditional Language Model** |
| Training loss | $\mathcal{L} = \frac{1}{T}\sum_{i=1}^{T}\mathcal{L}_i$, $\;\mathcal{L}_i = -\log P(y_i)$ — average cross-entropy **per target word** |
| Teacher forcing | at training, feed the **gold** $y_{i-1}$ as the decoder's input, not the model's $\hat{y}_{i-1}$ |
| Its cost | **exposure bias** — the model never trains on its own (imperfect) prefixes |
| Context at every step | $\mathbf{h}^d_i = g(\hat{y}_{i-1}, \mathbf{h}^d_{i-1}, \mathbf{c})$ — same $\mathbf{c}$ each step |
| Multi-layer rule | hidden states from RNN layer $i$ are the **inputs** to RNN layer $i{+}1$ |
| Bottleneck | $\mathbf{c}$ must represent **absolutely everything** about the source; it is the **only** thing the decoder knows about it |
| Attention score (deck's) | $\operatorname{score}(\mathbf{h}^d_{i-1}, \mathbf{h}^e_j) = \mathbf{h}^d_{i-1}\cdot\mathbf{h}^e_j$ — **dot-product attention**, zero parameters |
| Attention weights | $\alpha_{ij} = \operatorname{softmax}_j\!\left(\operatorname{score}(\mathbf{h}^d_{i-1},\mathbf{h}^e_j)\right)$, $\;\sum_j \alpha_{ij} = 1$ |
| Context vector | $\mathbf{c}_i = \sum_j \alpha_{ij}\,\mathbf{h}^e_j$ — **generated anew at each decoding step $i$** |
| Decoder with attention | $\mathbf{h}^d_i = g(\hat{y}_{i-1}, \mathbf{h}^d_{i-1}, \mathbf{c}_i)$ |
| Two benefits | improves NMT performance; gives **alignment for free**, never explicitly trained |
| LSTM params (no bias) | $4\left(d_h d_{\text{in}} + d_h^2\right)$ per direction |
| BIO tag count | $2 \times (\text{\# entity types}) + 1$ |

## [Lec 19 — Decoding Strategies](../notes/week-04/19-decoding-strategies.md)

| Item | Exactly this |
|---|---|
| What decoding is | choosing which token to emit from the distribution the model outputs; separate from the model |
| Greedy decoding | $\hat{w}_t = \arg\max_{w \in V} P(w \mid w_{<t})$ |
| Greedy's failure | a locally optimal token can doom every continuation, and there is **no backtracking** |
| Beam search core idea | at each step keep the $k$ most probable **partial hypotheses** |
| Beam size $k$ | the *beam width*; in practice **5 to 10** |
| Beam search per step | expand all $k$ hypotheses by all $\lvert V\rvert$ words → score $k\lvert V\rvert$ candidates → keep top $k$ |
| Hypothesis score | $\text{score}(y_1\ldots y_t) = \sum_{i=1}^{t}\log P_{\text{LM}}(y_i \mid y_{<i}, x)$ |
| Why logs | products underflow; $\log$ is monotonic so the ranking is unchanged |
| Stopping criterion | a hypothesis emitting `</s>` is **complete** → set aside, keep searching; stop at timestep $T$ **or** $n$ completed hypotheses |
| Length-normalised score | $\frac{1}{t}\sum_{i=1}^{t}\log P_{\text{LM}}(y_i \mid y_{<i}, x)$ |
| Why normalise | every $\log p$ is negative, so longer hypotheses always score lower — the bias is arithmetic, not quality |
| Deterministic methods | greedy and beam search. Everything else samples |
| Quality vs diversity | high-probability words → coherent but repetitive/boring; mid-probability words → creative but incoherent/less factual |
| Temperature | $P_\tau(w) = \dfrac{\exp(u_w/\tau)}{\sum_j \exp(u_j/\tau)}$; deck writes $\tau \in (0,1]$ |
| Temperature limits | $\tau \to 0$ → greedy; $\tau = 1$ → raw distribution; $\tau \to \infty$ → uniform |
| Lower $\tau$ | **sharpens**. Higher $\tau$ **softens/flattens** |
| Top-$k$ sampling | keep the $k$ most probable tokens, **renormalise**, sample |
| Nucleus / top-$p$ | $V^{(p)} =$ the **smallest** set with $\sum_{w \in V^{(p)}} P(w \mid w_{<t}) \geq p$; renormalise; sample |
| Top-$p$'s advantage | candidate-set size **adapts to the shape** of the distribution; a fixed $k$ cannot |
| Greedy as a special case | beam search with $k=1$ **and** top-$k$ sampling with $k=1$ |

## [Lec 20 — Better RNN Units: GRU and LSTM](../notes/week-04/20-gru-and-lstm.md)

| Item | Exactly this |
|---|---|
| Vanilla RNN recurrence (deck p. 114) | $\mathbf{h}_t = \tanh(\mathbf{U}\mathbf{h}_{t-1} + \mathbf{W}\mathbf{x}_t)$ |
| Gradient chain | $\dfrac{\partial\mathcal{L}^{(t)}}{\partial\mathbf{h}_k} = \Big(\prod_{j=k+1}^{t}\dfrac{\partial\mathbf{h}_j}{\partial\mathbf{h}_{j-1}}\Big)^{\!\top}\dfrac{\partial\mathcal{L}^{(t)}}{\partial\mathbf{h}_t}$ |
| One Jacobian | $\dfrac{\partial\mathbf{h}_j}{\partial\mathbf{h}_{j-1}} = \operatorname{diag}(1-\mathbf{h}_j^2)\,\mathbf{U}$ |
| Vanishing condition | $\sigma_{\max}(\mathbf{U}) < 1$ ⟹ gradient vanishes geometrically in $(t-k)$ |
| Exploding condition | $\sigma_{\max}(\mathbf{U}) > 1$ is **necessary** (not sufficient) for explosion |
| Why worse than depth | the **same** matrix is reused every step; a deep net has a different $\mathbf{W}^{(l)}$ per layer |
| Forget gate | $\mathbf{f}_t = \sigma(\mathbf{U}_f\mathbf{h}_{t-1} + \mathbf{W}_f\mathbf{x}_t)$ |
| Input gate | $\mathbf{i}_t = \sigma(\mathbf{U}_i\mathbf{h}_{t-1} + \mathbf{W}_i\mathbf{x}_t)$ |
| Candidate | $\tilde{\mathbf{C}}_t = \tanh(\mathbf{U}_g\mathbf{h}_{t-1} + \mathbf{W}_g\mathbf{x}_t)$ |
| **Cell update** | $\mathbf{C}_t = \mathbf{f}_t\odot\mathbf{C}_{t-1} + \mathbf{i}_t\odot\tilde{\mathbf{C}}_t$ |
| Output gate | $\mathbf{o}_t = \sigma(\mathbf{U}_o\mathbf{h}_{t-1} + \mathbf{W}_o\mathbf{x}_t)$ |
| Hidden state | $\mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{C}_t)$ |
| Gates use | $\sigma$ — a gate is a fraction in $[0,1]$: "how much to let through" |
| Candidates use | $\tanh$ — a candidate is a **value** in $[-1,1]$, must be able to be negative |
| Why LSTM fixes it | $\partial\mathbf{C}_t/\partial\mathbf{C}_{t-1} = \operatorname{diag}(\mathbf{f}_t)$ — a **learned, per-coordinate** gate the net can hold near 1, not a fixed matrix raised to a power |
| $\mathbf{C}_t$ vs $\mathbf{h}_t$ | $\mathbf{C}_t$ = long-term, private, carried; $\mathbf{h}_t$ = short-term, exposed, gated view of $\mathbf{C}_t$ |
| LSTM params | $4\big(n_h(n_h+n_x)+n_h\big)$ |
| GRU params | $3\big(n_h(n_h+n_x)+n_h\big)$ = exactly $0.75\times$ LSTM |
| Two residual RNN problems | $O(n)$ path length for distant pairs; **no parallelism across time** |

## [Lec 21 — Introduction to Transformers](../notes/week-05/21-intro-to-transformers.md)

| Item | Exactly this |
|---|---|
| General definition of attention | given a set of vector **values** and a vector **query**, attention computes a **weighted sum of the values, dependent on the query** |
| What the weighted sum is | a **selective summary** of the values; the query decides what to focus on |
| Attention's output size | a **fixed-size** representation of an **arbitrary set** of representations |
| Terminology | "the **query attends to** the values" |
| Self-attention | queries, keys and values all drawn from the **same** sequence |
| RNN issue 1 | $O(n)$ steps for distant word pairs → long-distance dependencies are hard (gradient problems); linear order is "baked in" |
| RNN issue 2 | **lack of parallelizability** — future hidden states can't be computed before past ones; inhibits training on very large datasets |
| Self-attention, sequential ops | $O(1)$ |
| Self-attention, max path length | $O(1)$ |
| Transformer's original form | **encoder-decoder**; encoder-only and decoder-only come later |
| Encoder block | **self-attention sublayer + feed-forward network** |
| Sharing | self-attention is shared across positions; the FFN is applied **separately per position** |
| Block I/O width | same size as the original token embeddings (so blocks can stack) |
| First layer's input | **static** token embeddings |
| Static vs contextual embedding | static = one vector per word type; contextual = a different vector per occurrence, depending on surrounding words |
| Simplified attention | $\text{score}(\mathbf{x}_i,\mathbf{x}_j) = \mathbf{x}_i\cdot\mathbf{x}_j$; $\alpha_{ij} = \mathrm{softmax}(\text{score})$ for $j\le i$; $\mathbf{a}_i = \sum_{j\le i}\alpha_{ij}\mathbf{x}_j$ |
| The three roles | **query** = current element being compared; **key** = prior input compared against it; **value** = prior element's content that gets weighted and summed |
| Hash-table analogy | hashtable: one query → exactly one key-value pair. Attention: each query matches **each** key to varying degrees, returning a weighted sum of values |
| Retrieval analogy | YouTube: search text = query, video titles/descriptions = keys, videos = values |
| The parallelism claim | self-attention over **all** words is computed **in parallel** |
| The leftover problem | self-attention is order-agnostic → positional information must be added ([Lec 23](../notes/week-05/23-positional-encoding-and-encoder.md)) |

## [Lec 22 — Self-Attention and Multi-Head Attention](../notes/week-05/22-self-attention-and-multihead.md)

| Item | Exactly this |
|---|---|
| The three projections | $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, $\mathbf{V} = \mathbf{X}\mathbf{W}^V$ |
| Per-position form | $\mathbf{q}_i = \mathbf{x}_i\mathbf{W}^Q$, $\mathbf{k}_j = \mathbf{x}_j\mathbf{W}^K$, $\mathbf{v}_j = \mathbf{x}_j\mathbf{W}^V$ |
| Score | $\mathrm{score}(\mathbf{x}_i,\mathbf{x}_j) = \dfrac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}}$ |
| Weights, output | $\alpha_{ij} = \mathrm{softmax}_j(\mathrm{score})$; $\;\mathbf{a}_i = \sum_j \alpha_{ij}\mathbf{v}_j$ |
| **The central equation** | $\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\dfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ |
| Why $\sqrt{d_k}$ | $\mathrm{Var}[\mathbf{q}\cdot\mathbf{k}] = d_k$ for unit-variance components; dividing by $\sqrt{d_k}$ restores variance 1 and keeps softmax out of saturation, where gradients vanish |
| Multi-head | $\mathbf{a}_i = (\mathrm{head}^1 \oplus \cdots \oplus \mathrm{head}^h)\mathbf{W}^O$, $\;\mathrm{head}^c_i = \sum_j \alpha^c_{ij}\mathbf{v}^c_j$ |
| Head width | $d_k = d_v = d_{\text{model}}/h$ |
| $\mathbf{W}^O$ shape | $hd_v \times d_{\text{model}}$ |
| FFN | $\mathrm{FFN}(\mathbf{x}_i) = \mathrm{ReLU}(\mathbf{x}_i\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$ |
| "Position-wise" | the **same** FFN weights applied **independently** at every position; no cross-position mixing |
| LayerNorm (deck) | $\mu = \frac1H\sum_i a_i$, $\sigma = \sqrt{\frac1H\sum_i(a_i-\mu)^2}$, $x' = \dfrac{x-\mu}{\sigma+\epsilon}$ |
| LayerNorm (full) | $\boldsymbol{\gamma}\odot\dfrac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \boldsymbol{\beta}$ |
| Layer vs batch norm | layer norm is **across features of one token**; batch norm is **across the batch per feature** |
| Residual | $\mathbf{x}_\ell = F(\mathbf{x}_{\ell-1}) + \mathbf{x}_{\ell-1}$ |
| The block | $\mathbf{z} = \mathrm{LayerNorm}(\mathbf{x} + \mathrm{SelfAttention}(\mathbf{x}))$; $\;\mathbf{y} = \mathrm{LayerNorm}(\mathbf{z} + \mathrm{FFN}(\mathbf{z}))$ |
| The three training tricks, in order | #1 residual connections, #2 LayerNorm, #3 scaled dot-product attention |
| Division of labour | attention **mixes across positions**; FFN **transforms within a position** |

## [Lec 23 — Positional Encodings and the Encoder Block](../notes/week-05/23-positional-encoding-and-encoder.md)

| Item | Exactly this |
|---|---|
| Sinusoidal PE (deck's form, p. 67) | $P(k,2i) = \sin\!\big(k/n^{2i/d}\big)$, $P(k,2i{+}1) = \cos\!\big(k/n^{2i/d}\big)$ |
| Same in this book's notation | $PE(\text{pos},2i) = \sin\!\big(\text{pos}/10000^{2i/d_{\text{model}}}\big)$, $PE(\text{pos},2i{+}1) = \cos(\cdot)$ |
| $i$'s range | $0 \le i < d_{\text{model}}/2$ — $i$ indexes **pairs**, not single dimensions |
| Even / odd | even dimension → **sine**; odd dimension → **cosine** |
| Combination rule | PE is **ADDED** to the embedding, never concatenated; **once**, before block 1 |
| Range of each entry | $[-1, 1]$ |
| Wavelengths | geometric from $2\pi$ to $10000\cdot 2\pi$ |
| Row 0 of the PE matrix | $(0,1,0,1,\ldots)$ for any $d$ and any $n$ |
| Relative-position property | $PE(\text{pos}{+}k) = \mathbf{M}_k\,PE(\text{pos})$, $\mathbf{M}_k$ block-diagonal rotations, independent of $\text{pos}$ |
| Why self-attention needs it | self-attention is **permutation-equivariant** — it sees a set |
| Matrix self-attention | $\mathbf{Z} = \text{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}\big)\mathbf{V}$, with $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$ etc. |
| Attention-map shape | $T \times T$ per head; $h$ heads → $h \times T \times T$ |
| Head width | $d_k = d_v = d_{\text{model}}/h$, so $h\,d_v = d_{\text{model}}$ |
| Encoder block order | MHSA → **Add & Norm** → position-wise FFN → **Add & Norm**, $\times N$ |
| Add & Norm means | $\text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$ — residual **then** norm (post-norm) |
| Params per block (weights) | $4d_{\text{model}}^2 + 2 d_{\text{model}} d_{ff} = 12 d_{\text{model}}^2$ when $d_{ff}=4d_{\text{model}}$ |
| Does $h$ change the count? | **No** |
| ViT patch count | $N = HW/P^2$; project each flattened $P^2C$ patch by $\mathbf{E} \in \mathbb{R}^{(P^2C)\times D}$ |

## [Lec 24 — The Decoder and Transformers as Language Models](../notes/week-05/24-decoder-and-transformer-lm.md)

| Item | Exactly this |
|---|---|
| Decoder block sublayers | **3**: masked self-attention → encoder-decoder attention → feed-forward (each + residual + LayerNorm) |
| Encoder block sublayers | **2**: self-attention → feed-forward |
| Masked self-attention | $\mathbf{Z} = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$, $M_{ij} = -\infty$ for $j>i$, else $0$ |
| Which triangle is masked | **strictly upper** ($j > i$); the diagonal stays — a token attends to itself |
| Why $-\infty$ | masking is **pre-softmax**, and $e^{-\infty}=0$; it also removes the term from the denominator so the row still sums to 1 |
| Why masking is needed | the autoregressive property — position $t$ must not see $t{+}1$, or training is cheating |
| Cross-attention $\mathbf{Q}$ | from the **decoder** |
| Cross-attention $\mathbf{K},\mathbf{V}$ | from the **encoder output** (top encoder layer) |
| Cross-attention matrix shape | $n \times m$ (target length × source length) — **rectangular, unmasked** |
| LM head | $\mathbf{u} = \mathbf{h}^L_N\mathbf{E}^\top$ (logits, $1\times\mid V\mid $); $\mathbf{y} = \mathrm{softmax}(\mathbf{u})$ |
| Unembedding layer shape | $d \times \lvert V\rvert$ |
| Weight tying | the same matrix is used as input embedding $\mathbf{E}$ and output projection $\mathbf{E}^\top$ |
| Decoder-only LM | **keep** masked self-attention, **drop** encoder-decoder attention → 2 sublayers |
| Attention sublayer params | $4d^2$ ($\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V,\mathbf{W}^O$), independent of $h$ |
| FFN params | $2d\,d_{ff}$ ($= 8d^2$ when $d_{ff}=4d$) |
| Encoder / decoder block | $12d^2$ / $16d^2$ (bias-free); decoder-only block $12d^2$ |
| Encoder passes vs decoder passes | encoder **1** per sentence; decoder **1** at training, **$T$** at inference |

## [Lec 25 — Efficient Transformers](../notes/week-05/25-efficient-transformers.md)

| Item | Exactly this |
|---|---|
| Self-attention complexity | $O(n^2 d)$ time, $O(n^2)$ memory — **quadratic in sequence length** |
| What the KV cache stores | key and value vectors of all previous tokens, per layer per head |
| Why K and V are cacheable | they depend only on past tokens, which causal masking freezes |
| Why Q is not cached | the query is for the *new* token, different every step |
| KV cache saving | per-step projection $O(td^2) \to O(d^2)$; ratio over $n$ steps $=(n+1)/2$ |
| KV cache size | $2 \cdot L \cdot B \cdot n \cdot h_{kv} \cdot d_{\text{head}} \cdot b$ — **linear in $n$ and in batch** |
| Local/windowed attention | each token attends to $w$ neighbours; $O(nwd)$ |
| Windowed receptive field | $L \times w$ after $L$ layers (deck, p. 111) |
| Dilated receptive field | $L \times d \times w$ for dilation $d$ (deck, p. 112) |
| Dilation's selling point | larger receptive field **without increasing computation** |
| Sparse Transformer | **half the heads local, half fixed-strided**; local $\lfloor i/N\rfloor=\lfloor j/N\rfloor$, strided $(i-j)\bmod N=0$ |
| Longformer = | dilated sliding window **+** symmetric global attention on a few tokens |
| Longformer global tokens | classification → `[CLS]`; QA → **all question tokens** |
| "Symmetric" global attention | the token attends to all **and** all attend to it |
| BigBird = | **window + global + random** ($r$ random keys per query) |
| BigBird's theoretical claim | universal approximator of sequence functions, **Turing complete**, still $O(n)$ |
| BIGBIRD-ITC vs ETC | ITC makes **existing** tokens global; ETC **adds new** global tokens such as `[CLS]` |
| Linformer idea | project the $n$ keys and values to $k \ll n$ via learned $\mathbf{E}_i,\mathbf{F}_i$ |
| Linformer complexity | $O(nk)$ time **and** space |
| Linformer's assumption | the attention matrix is approximately **low-rank** |
| MQA | all query heads share **one** K head and **one** V head |
| GQA | one K/V head **per group** of query heads; interpolates MHA ($g=h$) ↔ MQA ($g=1$) |
| What MQA/GQA reduce | **KV-cache memory and bandwidth** — *not* FLOPs |
| MoE layer | $N$ expert FFNs + router; token goes to top-$k$ experts |
| MoE gate | $p_i(\mathbf{x}) = e^{h(\mathbf{x})_i}/\sum_j e^{h(\mathbf{x})_j}$, $h(\mathbf{x})=\mathbf{W}_r\mathbf{x}$ |
| MoE output | $\mathbf{y} = \sum_{i\in\mathcal{T}} p_i(\mathbf{x}) E_i(\mathbf{x})$, $\mathcal{T}$ = top-$k$ set |
| Switch layer | MoE with **$k = 1$** |
| MoE decoupling | parameters $\times N$, compute $\times k$; total/active $= N/k$ |
| MoE routing granularity | **per token**, independently, at each MoE layer |
| What experts learn | token-level categories (punctuation, verbs, conjunctions, visual descriptions) in context — **not** topics |

## [Lec 26 — Pretraining and ELMo](../notes/week-06/26-pretraining-and-elmo.md)

| Item | Exactly this |
|---|---|
| ELMo equation | $\mathbf{ELMo}^{task}_k = \gamma^{task} \sum_{j=0}^{L} s^{task}_j \mathbf{h}^{LM}_{k,j}$ |
| $s^{task}_j$ | **softmax-normalised mixture-model weights**; $\sum_j s_j = 1$ |
| $\gamma^{task}$ | scales the **overall usefulness of ELMo to the task** (one scalar) |
| Number of layer vectors | $L + 1$, because $j$ starts at **0** (the token embedding) |
| $\mathbf{h}^{LM}_{k,j}$, $j \geq 1$ | $[\overrightarrow{\mathbf{h}}^{LM}_{k,j}; \overleftarrow{\mathbf{h}}^{LM}_{k,j}]$ — concatenation of the two directions |
| ELMo's architecture | **stacked biLSTM** (2 layers), **not** a Transformer |
| ELMo's "bidirectional" | a forward LM and a backward LM, trained **separately**, concatenated. **Not** joint bidirectional conditioning |
| ELMo vs TagLM | TagLM used only the **top** layer; ELMo learns a **task-specific combination of all** layers |
| How ELMo is used | **freeze** the biLM, concatenate its output into the task model's input, train the task model |
| Lower biLSTM layer | lower-level **syntax** → POS, syntactic dependencies, NER |
| Higher biLSTM layer | higher-level **semantics** → sentiment, SRL, QA, SNLI |
| Pretraining is self-supervised | the label is the next/masked token, already in the raw text — **no human annotation** |
| Fine-tuning, in one phrase | "pretraining … serving as **parameter initialization**" |
| Pretrain stage 1 vs 2 | random weights + lots of text + LM objective; **then** pretrained weights + text&labels + supervised objective |
| The three architectures | **encoder-only** (BERT family), **encoder-decoder** (Flan-T5, BART), **decoder-only** (GPT, Claude, Llama, Mixtral) |
| ELMo full name / paper | *Embeddings from Language Models*; Peters et al., **NAACL 2018**, "Deep contextualized word representations" |

## [Lec 27 — BERT and Masked Language Modelling](../notes/week-06/27-bert-masked-lm.md)

| Item | Exactly this |
|---|---|
| Why not next-word prediction for an encoder | bidirectional self-attention means position $t$ already sees $w_t$ — the model would trivially copy |
| MLM objective | $\mathcal{L}_{\text{MLM}} = -\sum_{t \in M} \log p_\theta(w_t \mid \tilde{\mathbf{x}})$, $\tilde{\mathbf{x}}$ = masked input, $M$ = masked positions |
| MLM output layer | $\mathbf{y}_t = \operatorname{softmax}(\mathbf{W}_V \mathbf{h}_t)$, $\mathbf{W}_V \in \mathbb{R}^{\lvert V\rvert \times d_h}$ |
| Loss comes from | **masked positions only** |
| Selection rate | **15%** of (sub)word tokens |
| The split | **80%** → `[MASK]`, **10%** → random token, **10%** → unchanged (still predicted) |
| Why the split | `[MASK]` is never seen at fine-tuning time (train/test mismatch); and it forces strong representations of *non*-masked tokens too |
| BERT stands for | **B**idirectional **E**ncoder **R**epresentations from **T**ransformers |
| BERT vs ELMo | BERT is **deeply** bidirectional (one network, both directions at every layer); ELMo **concatenates two independent unidirectional** LMs |
| Input embedding | token + **segment** + position, summed; positions are **learned**, not sinusoidal |
| `[CLS]` | prepended to every sequence; its final-layer output is the sequence representation |
| `[SEP]` | **between** the two sentences **and after** the second |
| NSP data | **50%** true adjacent pairs, **50%** random second sentence |
| NSP head | $\mathbf{y} = \operatorname{softmax}(\mathbf{W}_{\text{NSP}} C)$, $\mathbf{W}_{\text{NSP}} \in \mathbb{R}^{2 \times d_h}$ |
| NSP later verdict | contributes little; **RoBERTa dropped it** |
| Static vs contextual | static = word **types** (dictionary entries); contextual = word **instances** (one per occurrence) |
| Classification head | $\mathbf{y} = \operatorname{softmax}(\mathbf{W}_C C)$, $\mathbf{W}_C \in \mathbb{R}^{K \times d_h}$ — the only new parameters |
| Sequence-labeling head | $\mathbf{y}_t = \operatorname{softmax}(\mathbf{W}_K \mathbf{h}^L_t)$, $\mathbf{W}_K \in \mathbb{R}^{k \times d_h}$, $k$ = tag count |
| Subword BIO — training | copy the word's gold tag to **all** its subword pieces |
| Subword BIO — decoding | take the argmax tag of the **first** subword piece |
| Best feature extraction (CoNLL NER) | **concat of last four hidden layers**, 96.1 F1 |

## [Lec 28 — Span Tasks, T5 and BART (Pretraining Encoders and Encoder-Decoders)](../notes/week-06/28-span-tasks-t5-bart.md)

| Item | Exactly this |
|---|---|
| Span, formally | contiguous tokens with start $i$, end $j$, $1 \le i \le j \le T$ |
| Number of spans | $\dfrac{T(T+1)}{2}$ |
| Length limit | legal spans satisfy $j - i < L$; count $= LT - \dfrac{L(L-1)}{2}$ |
| SQuAD head | two new vectors only: $\mathbf{S}, \mathbf{E} \in \mathbb{R}^{d_h}$ |
| Start probability | $P_i = \dfrac{e^{\mathbf{S}\cdot\mathbf{h}_i}}{\sum_j e^{\mathbf{S}\cdot\mathbf{h}_j}}$, softmax over **positions** |
| Span score | $\mathbf{S}\cdot\mathbf{h}_i + \mathbf{E}\cdot\mathbf{h}_j$, maximised subject to $j \ge i$ |
| Span representation | $\text{spanRep}_{ij} = [\mathbf{h}_i; \mathbf{h}_j; \mathbf{g}_{ij}]$, dimension $3d_h$ |
| Span summary | $\mathbf{g}_{ij} = \dfrac{1}{(j-i)+1}\sum_{k=i}^{j}\mathbf{h}_k$ — divisor is the span **length** |
| Span-based NER extra label | **NULL**, because most spans are not entities |
| GLUE | 9 corpora, 3 groups: single-sentence (CoLA, SST-2); similarity/paraphrase (MRPC, STS-B, QQP); inference (MNLI, QNLI, RTE, WNLI) |
| RoBERTa | train longer, more data, **drop NSP** |
| SpanBERT | mask **contiguous spans** |
| MASS | mask a $k$-token fragment in the encoder, decode exactly it; $k{=}1\Rightarrow$ BERT, $k{=}m\Rightarrow$ LM |
| T5 framing | **text-to-text**: every task is string in → string out, selected by a **task prefix** |
| T5 objective | **span corruption** — one sentinel per span; target = sentinel + span, repeated, then one extra sentinel |
| T5's winning config | **encoder-decoder + denoising (replace corrupted spans)** |
| BART | bidirectional encoder + autoregressive decoder, trained as a **denoising autoencoder** |
| BART noise types | token masking, token **deletion**, **text infilling**, sentence **permutation**, document **rotation** |
| BART's best combination | **text infilling + sentence permutation** |
| T5 target vs BART target | T5 emits **only the missing spans**; BART emits the **entire original document** |
| Fine-tuning an enc-dec | **no additional parameters**; decoder loss updates encoder *and* decoder |

## [Lec 29 — Pretraining Transformer Decoder: GPT, Zero-shot and In-context Learning](../notes/week-06/29-gpt-decoder-pretraining.md)

| Item | Exactly this |
|---|---|
| GPT stands for | **Generative Pretrained Transformer** |
| Decoder pretraining objective | plain **next-token prediction**, $\mathcal{L} = -\sum_t \log P_\theta(w_t \mid w_{<t})$ |
| Deck's phrasing | *"It's natural to pretrain decoders as language models"* |
| GPT-1 architecture | Transformer **decoder**, 12 layers, 117M params, $d=768$, FFN 3072 |
| GPT-1 tokenizer | **BPE with 40,000 merges**; vocabulary **larger than BERT's** |
| GPT-1 corpus | **BooksCorpus, over 7,000 unique books** — long contiguous spans |
| GPT-1 adaptation | pretrain → add a **linear + softmax head**, fine-tune per task |
| GPT-1 NLI format | `[START] premise [DELIM] hypothesis [EXTRACT]`; classifier reads **`[EXTRACT]`** (the **last** token) |
| GPT-1 similarity task | run **both orderings**, add the outputs |
| GPT-1 multiple choice | one forward pass **per candidate answer**, softmax over the scores |
| "Beyond language modeling" idea | *"The decoder can work with a prompt!"* — any NLP task is $p(\text{output}\mid\text{input})$ |
| GPT-2 claim | LMs do tasks *"without any parameter or architecture modification"* — zero-shot |
| GPT-2 summarization cue | append **`TL;DR:`** after the article, generate **100 tokens** |
| GPT-2 translation cue | condition on pairs `english sentence = french sentence`, then `english sentence =`, **greedy** decoding |
| In-context learning (deck) | *"A **frozen** LM performs a task **only by conditioning on the prompt text**"* |
| Few-shot ICL (deck) | (1) prompt **includes examples**; (2) **no examples of the behaviour seen in training** |
| Zero-shot ICL (deck) | (1) prompt includes **no examples** (instructions allowed); (2) none seen in training |
| The caveat the deck states | *"We are unlikely to be able to verify (2)"* |
| The disambiguation | ICL "few-shot" ≠ supervised "few-shot" (train on few examples) — *"The above is different"* |
| GPT-3 one-liner | *"applied **without any gradient updates or fine-tuning**, with tasks and few-shot demonstrations specified purely via text"* |
| Gradient updates in 0/1/few-shot | **zero, always** |
| Prompting pipeline | input $x$ → **template** `[x] Overall, it was [z]` → prompt $x'$ → LM fills $[z]$ → **answer mapping** to a label |
| Params shortcut | $\approx 12Nd^2$; exact per block $= 12d^2 + 13d$ |

## [Lec 30 — Domain-Specific and Multilingual Pretraining](../notes/week-06/30-domain-and-multilingual-pretraining.md)

| Item | Exactly this |
|---|---|
| Data distribution | $p_l = \dfrac{D_l}{\sum_k D_k}$ |
| Temperature sampling | $q_l \propto p_l^{1/T}$, renormalised; $T\ge 1$ |
| $T=1$ / $T\to\infty$ | proportional / uniform over languages |
| mC4 / mT5 notation | $p^\alpha$ with $\alpha = 1/T$; mT5 uses $\alpha = 0.3$ |
| Same trick elsewhere | word2vec $P_n(w)\propto U(w)^{3/4}$ ([Lec 13](../notes/week-03/13-negative-sampling-glove.md)) |
| SCIVOCAB ∩ BASEVOCAB | **42%** overlap |
| SciBERT's main source of gain | the **scientific corpus**, not the vocabulary |
| Vocabulary replacement requires | pretraining **from scratch** (continual pretraining cannot do it) |
| Catastrophic forgetting | continual pretraining on a new domain degrades the old domain |
| Forgetting remedies | mix general pretrain data; mix parameters (**mix-out**) |
| mBERT | BERT architecture, Wikipedia, **104 languages**, shared **110k WordPiece**, 110M params |
| XLM objectives | alternates **MLM** and **TLM**; TLM sees both sentences of a parallel pair |
| XLM input embeddings | token + position + **language** |
| XLM-R | encoder-only, **CC-100** CommonCrawl, 100 languages, 270M–550M |
| mT5 | encoder-decoder, **mC4**, **101 languages**, 300M–13B |
| mC4 | **107 languages**, low-resource upsampled |
| XNLI | cross-lingual NLI, entailment/neutral/contradiction; deck misspells it **"XLNI"** |
| Zero-shot cross-lingual | labelled data for task X only in language A; none in B (unlabelled B may exist) |
| Translate-train | MT the English training set into targets; **not** zero-shot |
| IndicBERT | **22 Indian languages + English**; language-family-specific |
| **Curse of multilinguality** | for a **fixed-size** model, **per-language capacity decreases** as the number of languages grows (Conneau et al., 2020) |
| Curse, full form | helps low-resource up to a point (deck: peak at **15** languages), then degrades all |
| Fertility | **average number of tokens per word**; high fertility → more memory, slower decoding, shorter usable sequence |
| Vocabulary extension pipeline | train tokenizer on monolingual data → **concat** with existing vocab → initialise new embeddings from base LLM → continually pretrain |
| Continual pretraining objective | causal LM, $p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t\mid\mathbf{x}_{<t})$ |
| Avoid forgetting English | include English in the pretraining data |
| Align the new language | parallel data; romanized data |
| **Uptraining** | adapt a pretrained checkpoint to a **slightly different architecture**, then briefly re-pretrain |
| MHA → MQA/GQA conversion | key and value heads are **mean-pooled**; further pretrain for **α = 0.05** of original steps |
| Sparse upcycling | copy all dense weights; experts = **identical copies** of the replaced MLP; only the **router** is new |

## [Lec 31 — Question Answering I](../notes/week-07/31-question-answering-1.md)

| Item | Exactly this |
|---|---|
| QA goal (deck's words) | systems that **automatically answer questions posed by humans in a natural language** |
| QA taxonomy | information **source** / question **type** / answer **type** |
| Reading comprehension | $(P, Q) \rightarrow A$ — the passage is **given** |
| Open-domain QA | input is $Q$ alone; the passage must be **found** in a large collection |
| Retriever | $f(\mathcal{D}, Q) \rightarrow P_1,\ldots,P_K$, $K$ pre-defined |
| Reader | $g(Q, \{P_1,\ldots,P_K\}) \rightarrow A$ — "a reading comprehension problem!" |
| Joint score | $S(b,s,q) = S_{\text{retr}}(b,q) + S_{\text{read}}(b,s,q)$; $a^* = \text{TEXT}(\arg\max_{b,s} S)$ |
| SQuAD EM | 0/1 string equality after normalisation, **max over gold answers**, then averaged |
| SQuAD F1 | token-level $F_1 = \dfrac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}}$, **max over gold answers**, then averaged |
| SQuAD normalisation | remove **a, an, the** and **punctuation** |
| Term frequency (p. 18) | $\text{tf}_{t,d} = \log_{10}(1 + \text{count}(t,d))$ |
| Term frequency (p. 20 table) | $\text{tf}_{t,d} = 1 + \log_{10}\text{count}(t,d)$ — **the one the worked example uses** |
| IDF | $\text{idf}_t = \log_{10}\dfrac{N}{\text{df}_t}$ |
| tf-idf | $\text{tf-idf}(t,d) = \text{tf}_{t,d}\cdot\text{idf}_t$ |
| Document score | $\text{score}(q,d) = \cos(\mathbf{q},\mathbf{d}) = \frac{\mathbf{q}}{\mid \mathbf{q}\mid }\cdot\frac{\mathbf{d}}{\mid \mathbf{d}\mid }$ |
| BM25 | $\sum_{t\in q}\log\!\left(\frac{N}{\text{df}_t}\right)\dfrac{\text{tf}_{t,d}}{k\left(1-b+b\frac{\mid d\mid }{\mid d_{\text{avg}}\mid }\right)+\text{tf}_{t,d}}$ |
| $k_1$ controls | **term-frequency saturation** (balance between tf and IDF) |
| $b$ controls | **document-length normalisation** |
| Bi-encoder (DPR) | $\mathbf{h}_q = \text{BERT}_Q(q)[\texttt{CLS}]$, $\mathbf{h}_d = \text{BERT}_D(d)[\texttt{CLS}]$, score $= \mathbf{h}_q\cdot\mathbf{h}_d$ |
| ColBERT | $\text{score}(q,d) = \sum_{i=1}^{N}\max_{j=1}^{m} \mathbf{E}_{q_i}\cdot\mathbf{E}_{d_j}$ — **MaxSim, late interaction** |
| SpanBERT's two ideas | mask **contiguous spans**; predict masked tokens from the **two span endpoints** (SBO) |
| SBO | $\mathbf{y}_i = f(\mathbf{x}_{s-1}, \mathbf{x}_{e+1}, \mathbf{p}_{i-s+1})$; $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{SBO}}$ |

## [Lec 32 — Question Answering II: Training Retrievers, Generative QA, Multilingual and Tabular QA](../notes/week-07/32-question-answering-2.md)

| Item | Exactly this |
|---|---|
| Dense retriever score | $\text{sim}(q,p) = \mathbf{h}_q^\top \mathbf{h}_p$, two separate BERT towers |
| DPR training set | $\mathcal{D} = \{\langle q_i, p_i^{+}, p_{i,1}^{-}, \ldots, p_{i,n}^{-}\rangle\}_{i=1}^{m}$ — **one** positive, $n$ negatives |
| DPR loss | $-\log \dfrac{e^{\text{sim}(q_i,p_i^{+})}}{e^{\text{sim}(q_i,p_i^{+})} + \sum_{j=1}^{n} e^{\text{sim}(q_i,p_{i,j}^{-})}}$ |
| Max-margin alternative | $\sum_{d_{\text{pos}}}\sum_{d_{\text{neg}}} \max(0, s(q,d_{\text{neg}}) - s(q,d_{\text{pos}}))$ |
| Positives | dataset-provided gold passage, **or** high-BM25 passage **containing the answer string** (*relevance-guided supervision*) |
| Negatives (three) | **random** corpus passages; **BM25 hard** = top BM25 without the answer; **in-batch** = other questions' positives |
| In-batch loss | $\mathbf{S} = \mathbf{Q}\mathbf{P}^\top$, diagonal = positives, $\mathcal{L} = \frac1B\sum_i -\log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}$ |
| In-batch economics | $2B$ encodings → $B(B-1)$ negatives; linear cost, quadratic supervision |
| Closed-book / retrieval-free | query the LM directly; knowledge lives in the **weights**, no corpus at inference |
| Generative QA | fine-tune **T5**; encoder gets question (+passage), decoder emits the answer |
| UnifiedQA's four formats | **EX**tractive, **AB**stractive, **M**ultiple-**C**hoice, **Y**es/**N**o — all one input/output string |
| RAG (one line) | retrieved passages + question as the LLM's **prompt**, answer generated token by token → [Lec 55](../notes/week-11/55-retrieval-augmented-generation.md) |
| Natural Questions | real Google queries **≥ 8 words**; Wikipedia page from **top-5** results; **long** answer (paragraph) + **short** answer (entities), or **null** |
| HotpotQA | **multi-hop**: the answer needs **two or more** passages; 42% bridge-entity |
| Multi-hop failure mode | neither required passage individually matches the question well, so single-pass top-$k$ misses both |
| Multilingual QA | questions, answers **and** corpus in language $L$ |
| Crosslingual QA | questions and answers in $L$, corpus in **English** |
| Three multilingual approaches | (1) zero-shot transfer, (2) translation-based (Translate-Test / Translate-Train), (3) multilingual retriever-generator (mDPR + mGEN, Asai et al. 2021) |
| Translate-Test flaw | MT + QA error propagation; answers must exist in English → **anglocentric** |
| Translate-Train flaw | must translate the **entire corpus**; corpus and training data noisy from MT |
| Table linearisation | `[HEAD] c1 \| c2 ... [ROW] 1 v1 \| v2 ... [ROW] 2 ...` |
| TAPEX pretraining | sample a table, synthesize SQL, run a real **SQL executor**, supervise on its output |
| Structure-aware embeddings | row id + column id (+ numeric rank) embeddings added to the token embedding |

## [Lec 33 — Dialogue Systems I: Open-Domain Chatbots and Dialogue Evaluation](../notes/week-07/33-dialogue-systems-1.md)

| Item | Exactly this |
|---|---|
| The two kinds of conversational agent | **Chatbots** (open-domain, chit-chat, for fun or therapy) vs **task-based dialogue agents** ([Lec 34](../notes/week-07/34-dialogue-systems-2.md)) |
| Why open-domain uses representation learning | schema design and annotation for belief states/intents is too complicated; the two levers are **encoder/decoder architecture design** and **loss-function design** |
| Four challenges in dialogue modelling | interpretation of the context; handling background knowledge; extracting useful features from context; learning from limited data |
| Two corpus-based chatbot architectures | **response by retrieval** and **response by generation** |
| Neural IR retrieval | $\mathbf{h}_q = \text{BERT}_Q(q)[\texttt{CLS}]$, $\mathbf{h}_r = \text{BERT}_R(r)[\texttt{CLS}]$, $\text{response}(q,C) = \arg\max_{r \in C} \mathbf{h}_q \cdot \mathbf{h}_r$ |
| Retrieval: strength / weakness | always fluent and human-sounding / **can never say anything not already in the corpus** |
| Generation | $\hat{r}_t = \arg\max_{w \in V} P(w \mid q, r_1 \ldots r_{t-1})$ |
| Generation's common problem | **generic, bland, repetitive responses** ("I don't know", "See you later") — they are high-probability under the model; same phenomenon as [Lec 19](../notes/week-04/19-decoding-strategies.md) |
| The *other* common problem | **no consistent personality** — trained on many speakers, so a "mish-mash of personalities" (Li et al. 2016) |
| Fix for inconsistency | **PersonaChat** — condition on an explicit persona (5 profile sentences per speaker) |
| Third strategy | **response by retrieving and refining** — retrieve informative text (Wikipedia), concatenate to the dialogue context with a **separator token**, let the encoder-decoder weave it in |
| Three evaluation families | content-overlap, model-based, human |
| Content-overlap metrics | BLEU, ROUGE, METEOR, CIDEr — lexical similarity to a gold reference; fast, widely used |
| **Why they fail for dialogue** | many valid responses exist for any utterance, so a good reply can share **zero** words with the reference (**false negative**) while a contradictory reply shares most of them (**false positive**); *n*-gram overlap has **no concept of semantic relatedness** |
| BERTScore | $R_{\text{BERT}} = \frac{1}{\lvert x\rvert}\sum_{x_i} \max_{\hat{x}_j} \mathbf{x}_i^\top\hat{\mathbf{x}}_j$; $P$ flips the direction; $F_{\text{BERT}} = 2PR/(P+R)$; optional **idf** importance weighting |
| Why model-based metrics help | embeddings, so **no n-gram bottleneck**; embeddings **pretrained**, distance metric can be **fixed** |
| Why even they are not enough | still anchored to one ground-truth response; a good reply can be relevant but not similar to the GT → need **reference-free**, context-aware scoring |
| DMI | Discourse Mutual Information Maximization: shared encoder $g_\phi$ on context and response, $\mathbf{z}_R = \mathbf{W}\mathbf{E}_r$, maximise MI between $\mathbf{z}_C$ and $\mathbf{z}_R$ |
| Human-evaluation dimensions (8) | fluency; coherence/consistency; factuality and correctness; commonsense; style/formality; grammaticality; typicality; redundancy |
| Cohen's $\kappa$ | $\kappa = \dfrac{p_o - p_e}{1 - p_e}$ |

## [Lec 34 — Dialogue Systems II: Task-Oriented Dialogue](../notes/week-07/34-dialogue-systems-2.md)

| Item | Exactly this |
|---|---|
| Task-oriented agent | helps the user solve a task (travel reservation, buying a product) by filling a structured frame and querying a database |
| **Frame** | a set of **slots**, each to be filled with information of a given **type**, each associated with a **question** to the user |
| Domain ontology | another name for the frame/slot structure of a domain |
| **GUS** | **Genial Understander System**, Bobrow, Kaplan, Kay, Norman, Thompson & Winograd, 1977, *Artificial Intelligence* 8(2):155–173 |
| GUS control structure | ask questions → fill any slots the user specifies (possibly many at once) → when the frame is full, do the database query |
| Condition-action rules | rules attached to a slot that fire when it is filled — e.g. DESTINATION city becomes default *StayLocation* in the hotel frame; DESTINATION DAY copied to ARRIVAL DAY on short trips |
| Frame detection | deciding which slot of **which frame** the user is filling, and switching control to it |
| Two architectures | **GUS / frame-based** (1977, industrial) and **dialogue-state / belief-state** (extension of GUS, research) |
| GUS NLU, three steps | 1. domain classification 2. intent determination 3. slot filling |
| Dialogue-state components | **NLU → dialogue state tracker → dialogue policy → NLG** |
| Dialogue state tracker | maintains the user's most recent dialogue act and the **accumulated** set of slot-filler constraints |
| GUS policy | ask questions until the frame is full, then report back |
| Slot filling as sequence labeling | **BIO tagging** — one `B-` and one `I-` per slot type plus `O`; $2n+1$ tags ([Lec 17](../notes/week-04/17-rnn-applications.md)) |
| Deck's tagged example | `O O O O O B-DES I-DES O B-DEPTIME I-DEPTIME O` over "I want to fly to San Francisco on Monday afternoon please" |
| Contextual-embedding slot filling | BERT encoder → per-token classifier+softmax → BIO tag; **domain+intent generated over `<EOS>`** as a joint label like `AIRLINE_TRAVEL + SEARCH_FLIGHT` |
| After tagging | extract filler string, then **normalise to the ontology** via homonym dictionaries (SF = SFO = San Francisco) |
| Dialogue act interpretation | **1-of-N supervised classification**, from encodings of the current sentence **+ prior dialogue acts** |
| Simple state tracker | run a slot-filler after each sentence and merge into the running state |
| Generative DST | generate the state as text with a seq2seq LM; schema-driven prompting (EMNLP 2021, T5) |
| NLG two stages | **content planning** (what to say) → **sentence realization** (how to say it) |
| **Delexicalization** | replace slot values in the training set with a generic placeholder token, to improve generalisation when training data is scarce |
| Relexicalization | substitute the real values back into the generated delexicalized sentence |
| **Combined Score** | $\text{Score} = \text{BLEU} + 0.5(\text{Inform} + \text{Success})$ |
| Inform | how often **all** entities provided by the system are correct |
| Success | how often the system **correctly answers all requested attributes** |
| Entity-F1 | F1 for the entities returned to the user in the generated response |
| Joint goal accuracy | fraction of turns in which **every** slot of the state is correct |

## [Lec 35 — Text Summarization](../notes/week-07/35-text-summarization.md)

| Item | Exactly this |
|---|---|
| Summarization | automatically condensing the given document(s) to present the most relevant information in a concise and quickly-readable format |
| Taxonomy by output | **extractive** vs **abstractive** vs **hybrid** |
| Taxonomy by input | **single-document** vs **multi-document** |
| Taxonomy by genre | **generic** vs **domain-specific** |
| Extractive trade-off | always grammatical and faithful; rigid and redundant |
| Abstractive trade-off | fluent and compressive; can hallucinate facts not in the source |
| BERTSum | extractive summarization as **sentence classification**: a `[CLS]` before **every** sentence; its output scores that sentence; **1 = include, 0 = exclude** |
| BERTSum segment embeddings | **interval** — alternate $E_A, E_B$ per sentence |
| Abstractive framing | seq2seq; input = document(s), output = summary |
| Pegasus objective | **Gap Sentence Generation (GSG)** + MLM, applied **simultaneously** |
| GSG | mask whole **sentences** with `[MASK1]`, generate them as the decoder target; other sentences stay in the input with `[MASK2]` token masking |
| Gap-sentence selection | **Random**, **Lead** (first $m$), **Principal** (top-$m$ by $s_i = \mathrm{rouge}(x_i, D\setminus\{x_i\})$, ROUGE-1-F1) |
| Pegasus's lesson | make the pretraining objective **resemble the downstream task** |
| Compression ratio | $\lvert D\rvert / \lvert S\rvert$ |
| Coverage / density | how derivative a summary is of the source; extractive ⇒ coverage $=1$ |
| Long-document fix | **extract-then-paraphrase** (cheap selector under the context limit, then a generator) |
| Book-length fix | **hierarchical merging** or **incremental updating** — both recursive summarization |
| BLEU | $\min\!\left(1, \frac{\text{output-length}}{\text{reference-length}}\right)\left(\prod_{n=1}^{4} p_n\right)^{1/4}$ |
| BLEU full name | **Bilingual Evaluation Understudy** (Papineni et al., 2002); built for MT |
| Modified precision | $p_n = \dfrac{\sum_g \min(\text{Count}_c(g), \text{Count}_r(g))}{\sum_g \text{Count}_c(g)}$ — clip by the reference count |
| BLEU orientation | **precision** |
| ROUGE-$n$ | $\dfrac{\text{overlapping } n\text{-grams}}{\text{total } n\text{-grams in the reference summary}}$ |
| ROUGE orientation | **recall** — "a recall-based counterpart to BLEU" |
| ROUGE-L | longest common **subsequence**; $R_{\text{lcs}} = \text{LCS}/\lvert X\rvert$, $P_{\text{lcs}} = \text{LCS}/\lvert Y\rvert$, $F_{\text{lcs}}$ their weighted harmonic mean |
| BARTScore | $\sum_{t=1}^{m}\omega_t \log p(\mathbf{y}_t \mid \mathbf{y}_{<t}, \mathbf{x}, \theta)$; $\omega_t$ IDF or uniform |
| BARTScore directions | $s\to h$ **faithfulness**; $r\to h$ **precision**; $h\to r$ **recall** |

## [Lec 36 — Instruction Fine-tuning I](../notes/week-08/36-instruction-finetuning-1.md)

| Item | Exactly this |
|---|---|
| Alignment (Askell et al.) | AI is "aligned" if it is **helpful, honest, and harmless** — and the definition is not tied to language |
| The failure the lecture opens with | **Language Modeling ≠ Following Human Instructions**; LMs are not aligned with **user intents** (Ouyang et al. 2022) |
| The second failure | **Language Modeling ≠ Incorporating Human Values** (Zhao et al. 2021) |
| Deck's PROMPT example | *"Explain the moon landing to a 6 year old in a few sentences."* → GPT-3 emits **four more instructions**, not an answer |
| Instruction fine-tuning | supervised fine-tuning on **(instruction, output)** pairs across **many tasks**, with the **same next-token objective** |
| Alignment recipe | **step 1 = instruction tuning (SFT); step 2 = RLHF** (reward model + RL), initialised from the SFT model |
| Basic premise | NLP tasks can be described via natural language instructions |
| Why natural language | task-specific supervision "can't generalize to unseen tasks"; a model that understands instructions "should be able to generalize to any task that can be defined in terms of natural language" |
| Three paradigms | **(A)** pretrain–finetune: train on A, infer on A · **(B)** prompting: no weight update · **(C)** instruction tuning: train on B,C,D…, infer on **unseen** A |
| Instruction schema — Instructions block | **Title · Definition · Things to Avoid · Emphasize/Caution** |
| Schema — positive example | Input · Output · **Explanation** (3 fields) |
| Schema — negative example | Input · Output · Explanation · **Suggestion** (4 fields) |
| Schema — the rest | Prompt message; Input/outputs (Input, Output) per instance; counters: # examples for task, # task instances, # tasks |
| Instructions are written | **once per task**; input/output repeats **per instance** |
| Natural Instructions finding 1 | instructions improve cross-task generalization significantly |
| Natural Instructions finding 2 | **excluding negative examples helps** — "negative instructions are surprisingly difficult for the models to learn from" |
| Super-NI subtitle | **Generalization via *declarative* instructions** on 1600+ tasks |
| Models trained | **Tk-INSTRUCT** = instruction-tuned **T5**; **mTk-INSTRUCT** = instruction-tuned **mT5** |
| Super-NI data scaling | performance ↑ with **training tasks** and **model parameters**; **NOT** with instances per task |
| The scaling-law warning | Lec 36's "Scaling Laws" = **instruction-tuning data** scaling. Kaplan/Chinchilla **compute** scaling is [Lec 51](../notes/week-11/51-scaling-laws.md) |

## [Lec 37 — Instruction Fine-tuning II: Flan and Self-Instruct](../notes/week-08/37-instruction-finetuning-2.md)

| Item | Exactly this |
|---|---|
| Flan paper | Wei et al., *Finetuned Language Models Are Zero-Shot Learners*, **ICLR'22** |
| Flan's data trick | **generate multiple templates from the same dataset** — one supervised dataset → $k$ instruction phrasings |
| Flan: dataset / task category / task | dataset = original source (SQuAD); task category = unique setup (extractive QA); **task = ⟨dataset, category⟩ pair, with any number of templates** |
| Flan evaluation metric | **few-shot prompted accuracy (exact match)** on all four benchmark suites |
| Flan held-out suites | **MMLU, BBH, TyDiQA, MGSM** — not in the finetuning data |
| Self-Instruct paper | Wang et al., *SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions*, **ACL'23** |
| Self-Instruct's 4 steps | 1 instruction generation → 2 classification-task identification → 3 instance generation → 4 filtering (then loop) |
| Step 1 sampling | 8 in-context instructions per call: **6 human-written + 2 model-generated** |
| "Classification task" definition | a task with a **small, limited output label space** |
| Input-first | non-classification; generate **input then output** |
| Output-first | classification; generate **class label then input** |
| Why output-first | input-first biases inputs toward one label (grammar-error detection → grammatical inputs); conditioning on the label balances the class distribution |
| Filter 1 | add an instruction only if **ROUGE-L similarity with any existing instruction $< 0.7$** |
| ROUGE-L, $\beta=1$ | $2\lvert\text{LCS}(X,Y)\rvert / (\lvert X\rvert + \lvert Y\rvert)$ — see [Lec 35](../notes/week-07/35-text-summarization.md) |
| Other filters | keyword blacklist (image/picture/graph); drop identical instances and same-input-different-output; heuristics (too long/short, output repeats input) |
| Dolly | 15K examples, **Databricks employees**, 7 task types, contest for the top 20 labellers |
| Human-data limitations | **too much human effort**; **hard to increase task diversity and complexity** |
| Benefits of instruction tuning | task semantics as an alternative to examples; cheap to collect; cross-task generalization; user-friendly |

## [Lec 38 — Reinforcement Learning from Human Feedback I](../notes/week-08/38-rlhf-1.md)

| Item | Exactly this |
|---|---|
| Bradley–Terry (year) | **1952**, paired comparison model |
| Preference probability | $P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$ |
| Derivation | $P(i\succ j) = \lambda_i/(\lambda_i+\lambda_j)$ with $\lambda = e^{r}$ gives $e^{r_i}/(e^{r_i}+e^{r_j}) = \sigma(r_i-r_j)$ |
| Reward-model loss | $\mathcal{L}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\big[\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\big]$ |
| Equivalent stable form | $\mathcal{L} = \log\big(1+e^{-\Delta}\big)$, the softplus of the negative margin |
| Gradient | $\partial\mathcal{L}/\partial r_w = -(1-\sigma(\Delta))$, $\partial\mathcal{L}/\partial r_l = +(1-\sigma(\Delta))$ |
| RM architecture | base **instruction-tuned** LM with the LM head replaced by a $d_{\text{model}}\times 1$ scalar head |
| RM input / output | input = prompt **+** completion concatenated; output = **one scalar per response**, not per token |
| Three phases | (1) SFT, (2) reward-model training on preferences, (3) RL optimisation of the policy against the RM |
| Problem 1 | human-in-the-loop is expensive → learn a model of preference and query that |
| Problem 2 | human judgements are noisy and miscalibrated → ask for pairwise comparisons |
| RL goal | $\max_\theta \mathbb{E}\big[\sum_t \gamma^t r_t\big]$, $\gamma$ = discount factor |
| Trajectory | $\tau = (s_0,a_0,r_0,\ldots,s_T,a_T,r_T)$ |
| Trajectory distribution | $P(\tau\mid\pi) = p(s_0)\prod_{t=0}^{T}\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$ |
| Advantage | $A(s,a) = Q(s,a) - V(s)$ |
| RL ↔ LM map | policy = LM; state = prompt + tokens so far; action = next token; reward = RM score **at the end only**; trajectory = one response |
| Three RLHF differences | reward **model** not function; **no state transitions**; **response-level** rewards |

## [Lec 39 — RLHF II: Policy Gradient and PPO](../notes/week-08/39-rlhf-2-ppo.md)

| Item | Exactly this |
|---|---|
| RLHF objective | $\max_\theta \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}[r_\phi(x,y)] - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})$ |
| KL divergence | $D_{\mathrm{KL}}(q\|p) = \mathbb{E}_q[\log(q/p)] = \sum_x q(x)\log\frac{q(x)}{p(x)}$ |
| KL properties | $\ge 0$, $=0$ iff $q=p$; **not symmetric**, **not a distance/metric** |
| Forward vs reverse | forward $D_{\mathrm{KL}}(p\|\pi_\theta)$ = mode-**covering**; reverse $D_{\mathrm{KL}}(\pi_\theta\|p)$ = mode-**seeking**. RLHF uses reverse |
| Per-sample KL reward | $\tilde r = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ |
| Why the KL penalty | stops **reward hacking / reward-model over-optimisation** — a leash on drift from $\pi_{\text{ref}}$ |
| Log-derivative trick | $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$ |
| REINFORCE | $\nabla_\theta J = \mathbb{E}[\,R\,\nabla_\theta\log\pi_\theta(a\mid s)\,]$ |
| Trajectory | $\tau = (s_0,a_0,s_1,a_1,\ldots)$; $P(\tau\mid\pi) = \rho_0(s_0)\prod_t P(s_{t+1}\mid s_t,a_t)\pi(a_t\mid s_t)$ |
| LM trajectory | state = prompt + tokens so far; action = next token; transition deterministic |
| Discounted return | $R(\tau) = \sum_{t\ge0}\gamma^t r_t$ |
| Vanilla PG problem | **unbiased but high variance** |
| Reward-to-go | $\sum_{t'=t}^{T} r_{t'}$ — drop rewards earned before the action |
| Baseline unbiasedness | $\mathbb{E}_\pi[\nabla\log\pi(a\mid s)]b(s) = b(s)\nabla\sum_a\pi = b(s)\nabla 1 = 0$ |
| Value head | "an additional linear layer on top of the LM" estimating $V(s_t)$ |
| $Q$ | $Q^\pi(s,a) = \mathbb{E}_{s'}[r(s,a) + \gamma\,\mathbb{E}_{a'}[Q^\pi(s',a')]]$ |
| $V$ | $V^\pi(s) = \mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$ |
| Advantage | $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$; $\mathbb{E}_{a\sim\pi}[A] = 0$ |
| Importance sampling | $\mathbb{E}_{x\sim p}[f] = \mathbb{E}_{x\sim q}[\tfrac{p}{q}f]$ |
| PPO ratio | $\rho_t = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$ |
| **PPO clipped loss** | $\mathcal{L} = \mathbb{E}[\min(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]$ |
| What the $\min$ does | makes the bound **pessimistic**: clips only when clipping *lowers* the objective |
| Full PPO loss | $\mathcal{L}_{\mathrm{PPO}} = \mathcal{L}_{\text{POLICY}} + c_1\mathcal{L}_{\mathrm{VF}} + c_2\mathcal{L}_{\mathrm{ENTROPY}}$ (deck's signs; paper subtracts $c_1\mathcal{L}_{\mathrm{VF}}$) |
| Four models in memory | policy, reference, reward model, value head |

## [Lec 40 — Aligning to User Preferences via Direct Preference Optimization](../notes/week-08/40-dpo.md)

| Item | Exactly this |
|---|---|
| RLHF closed-form optimum | $\pi^*(\mathbf{y}\mid\mathbf{x}) = \frac{1}{Z(\mathbf{x})}\pi_{\text{ref}}(\mathbf{y}\mid\mathbf{x})\exp\!\big(\tfrac{1}{\beta}r(\mathbf{x},\mathbf{y})\big)$ |
| Partition function | $Z(\mathbf{x}) = \sum_{\mathbf{y}}\pi_{\text{ref}}(\mathbf{y}\mid\mathbf{x})\exp\!\big(\tfrac{1}{\beta}r(\mathbf{x},\mathbf{y})\big)$ |
| The inversion | $r(\mathbf{x},\mathbf{y}) = \beta\big(\log\frac{\pi_\theta(\mathbf{y}\mid\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}\mid\mathbf{x})} + \log Z(\mathbf{x})\big)$ |
| Implicit reward | $\hat r_\theta(\mathbf{x},\mathbf{y}) = \beta\log\frac{\pi_\theta(\mathbf{y}\mid\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}\mid\mathbf{x})}$ |
| Preference probability | $\Pr_\theta(\mathbf{y}_a\succ\mathbf{y}_b\mid\mathbf{x}) = \sigma\big(\hat r_\theta(\mathbf{x},\mathbf{y}_a) - \hat r_\theta(\mathbf{x},\mathbf{y}_b)\big)$ |
| **DPO loss** | $-\mathbb{E}\big[\log\sigma\big(\beta\log\frac{\pi_\theta(\mathbf{y}_w\mid\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_w\mid\mathbf{x})} - \beta\log\frac{\pi_\theta(\mathbf{y}_l\mid\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l\mid\mathbf{x})}\big)\big]$ |
| DPO gradient weight | $\sigma(-u)$, large when the implicit reward model is wrong |
| Why $Z$ vanishes | both completions share the prompt $\mathbf{x}$, so $\log Z(\mathbf{x})$ cancels in the difference |
| DPO is | **offline** RL; a supervised binary-classification loss |
| KTO loss | $\mathbb{E}[1 - \hat h(x,y;\beta)]$, $\hat h$ a sigmoid of the log-ratio against the dataset-mean $\beta D_{\mathrm{KL}}$, sign flipped for undesirable $y$ |
| BoN | $\hat{\mathbf{y}}_{\text{best}} = \arg\max_i r(\mathbf{x},\hat{\mathbf{y}}_i)$ over $N$ sampled candidates |
| BoN for training | = **rejection sampling** (RAFT) |
| DPO paper title | *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* |

## [Lec 41 — Prompting I](../notes/week-09/41-prompting-1.md)

| Item | Exactly this |
|---|---|
| Prompting (deck's definition) | encouraging a pre-trained model to make particular predictions by providing a textual "prompt" specifying the task to be done |
| Basic prompting | append a textual string to the beginning of the sequence and complete |
| Prompt template | a string with `[X]` (you fill) and `[Z]` (the LM fills); a function $x \mapsto x'$ |
| The three stages, in order | **answer prediction → output selection → output mapping** |
| Answer prediction | given a prompt, predict the answer |
| Output selection | from a longer response, select the information indicative of an answer |
| Output mapping | given an answer, map it into a class label or continuous value |
| Verbalizer | the label → token-set map; "often map many extracted words onto a single class" |
| Extraction by task | classification → keywords; regression/numerical → numbers; code → triple-backtick snippets |
| Basic prompt | no instruction; the LLM simply completes the sentence |
| Instruction prompt | two components: the **instruction** and the **data** it refers to |
| Output indicator | the extra component (`Text:` / `Sentiment:`) that forces a specific output format |
| Chat prompts | messages are converted to **token strings**; roles are special tokens, format is model-specific |
| Phi-3 markers | `<s>` BOS, `<\|user\|>` start of prompt, `<\|end\|>` end, `<\|assistant\|>` start of output |
| Zero-shot | prompting **without** examples |
| One-shot | prompting with **a single** example |
| Few-shot | prompting with **more than one** example |
| Few-shot learning | a pattern mapping inputs to outputs that the LLM follows — **with no weight updates** |
| Seven components of a complex prompt | Persona, Instruction, Context, Format, Audience, Tone, Data |
| Prompt-engineering rule | describe the task as clearly as possible; add scope, audience, length |

## [Lec 42 — Prompting: Why Does In-Context Learning Work?](../notes/week-09/42-why-icl-works.md)

| Item | Exactly this |
|---|---|
| In-context learning | learning to do new tasks / better predict tokens / reduce loss, **during the forward pass at inference time**, **without any gradient-based updates to the parameters** |
| Induction-head pattern | **`[A][B] … [A] → [B]`**, instantiating the completion rule $A\,B \ldots A\,B$ |
| Induction head, property 1 | **Prefix matching** — attends back to previous tokens that were followed by the current and/or recent tokens |
| Induction head, property 2 | **Copying** — its output **increases the logit of the attended-to token** |
| Residual stream | the **sum** of the outputs of all previous layers and the original embedding |
| Residual-stream update | $\mathbf{x}^{(i+1)} = \mathbf{x}^{(i)} + \sum_{h\in H_i} h(\mathbf{x}^{(i)})$; $\mathbf{x}^{(i+2)} = \mathbf{x}^{(i+1)} + m(\mathbf{x}^{(i+1)})$ |
| Embed / unembed | $\mathbf{x}^{(0)} = \mathbf{W}_E\mathbf{t}$; $T(\mathbf{t}) = \mathbf{W}_U\mathbf{x}^{(-1)}$ |
| Reads vs writes | $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ **read from** the stream; $\mathbf{W}_O$ **writes to** it |
| QK circuit | $\mathbf{W}_{QK} = \mathbf{W}_Q\mathbf{W}_K^\top$ — **which tokens** information moves to and from |
| OV circuit | $\mathbf{W}_{OV} = \mathbf{W}_V\mathbf{W}_O$ — **what information** is moved from an attended token |
| $K$-composition | one head's output is used to build a later head's **key** vector — the induction circuit |
| The two heads | layer-$l$ **previous-token head** writes "I follow $X$"; later **induction head** matches on it and copies |
| Attention vs MLP | attention moves info **across** residual streams; MLPs move it **across dimensions within** one |
| ICL score | $\mathcal{L}_{500} - \mathcal{L}_{50}$, averaged over dataset examples; **negative = better** |
| Why 500 and 50 | 500 is near the end of a **length-512** context; 50 is far enough in that language and document type are established, yet near the start |
| Phase change | induction heads form and ICL improves **simultaneously**, early in training |
| One-layer control | a one-layer attention-only model shows **no sudden improvement** — it cannot do $K$-composition |
| Order sensitivity | the same demonstrations in a different order can swing between **state-of-the-art and random** performance |
| Random-label result | replacing demonstration labels with random ones drops performance **only marginally** |
| The four ingredients | **Format, Label space, Input distribution, Input–Label mapping** ($F, L, I, M$) |
| Deck's verdict | "**the format is likely easier to exploit than the input–label mapping**" |

## [Lec 43 — Advanced Prompting Techniques](../notes/week-09/43-advanced-prompting.md)

| Item | Exactly this |
|---|---|
| Demonstration selection, deck's rule | retrieve demonstrations **similar to the current input**, dynamically, per input — top-$T$ by embedding similarity |
| Learned selection | one-layer NN over **frozen** BERT embeddings, trained by **policy gradient**; reward = answer correct |
| Selection policy | $\pi_\theta(e_i \mid p_i) = \dfrac{\exp[\mathbf{h}(e_i)\cdot\mathbf{h}(p_i)]}{\sum_{e'_i \in E_{\text{cand}}}\exp[\mathbf{h}(e'_i)\cdot\mathbf{h}(p_i)]}$ |
| Chain-of-thought | instruct the LLM **to generate reasoning steps**, or **to learn from demonstrations of detailed reasoning processes** |
| Why CoT works | (i) more forward-pass computation per problem; (ii) decomposition into individually easy steps |
| CoT's four stated advantages | decomposition · transparency/interpretability · trust · it is in-context learning, so works off-the-shelf |
| Few-shot CoT | demonstrations contain worked rationales |
| **Zero-shot CoT** | append **"Let's think step-by-step."** — no demonstrations at all |
| Self-consistency | sample $m$ diverse chains at $\tau > 0$, then **majority vote over the final answers** |
| Self-consistency slogan | *marginalize out reasoning paths to aggregate final answers* |
| Self-consistency vs beam search | beam search = low diversity, searches for one high-probability sequence; self-consistency = **diversity of reasoning paths is the key** |
| ToT components | thought decomposition · thought generation (propose prompt) · state evaluation (value prompt) · search |
| ToT search | **BFS or DFS**; each state evaluated by a **classifier via a prompt** or by **majority vote** |
| ToT taxonomy order | IO → CoT → CoT-SC → ToT |
| Self-Ask | the model emits its own **follow-up sub-questions** and intermediate answers, then "So the final answer is:" |
| Self-Ask + search | each `Follow up:` is routed to a **search engine**; its response becomes the `Intermediate answer:` |
| PAL | generates intermediate steps **and Python code**; *"shifts the role of running the reasoning steps from the language model to the Python interpreter"* |
| PoT | generates text + programming-language statements + an answer; **decouples complex computation from reasoning and language understanding**; mainly **Codex** |

## [Lec 44 — Tool-aided Language Models](../notes/week-09/44-tool-aided-lms.md)

| Item | Exactly this |
|---|---|
| Deck's definition of a tool | "a **function interface to a computer program that runs external to the LM**, where the LM generates the function calls and input arguments in order to use the tool" |
| Why tools (deck's two) | complex reasoning → **Struggle**; real-world information → **Fundamentally unable** |
| Tool Use | switching between **text-generation mode** and **tool-execution mode** |
| Tool Learning | **inference-time prompting** (Lec 43) vs **learning by training** (Lec 44) |
| TALM pipeline | input → tool input → *call* → tool result (appended) → output |
| TALM mechanism | generate a **delimiter** (e.g. `&#124;result`); on detection, call the API and append its result |
| API-call tuple | $c = (a_c, i_c)$: API name, input |
| Linearization | $e(c) = \texttt{<API>}\,a_c(i_c)\,\texttt{</API>}$; $e(c,r) = \texttt{<API>}\,a_c(i_c) \to r\,\texttt{</API>}$ |
| Toolformer's four pipeline steps | **sample** calls → **execute** → **filter** → **fine-tune** (then inference) |
| Sampling positions | $p_i = p_M(\texttt{<API>} \mid P(\mathbf{x}), x_{1:i-1})$; keep $I = \{i \mid p_i > \tau_s\}$, top $k$ |
| Filter: positive term | $L_i^{+} = L_i(e(c_i, r_i))$ |
| Filter: negative term | $L_i^{-} = \min\big(L_i(\varepsilon),\, L_i(e(c_i,\varepsilon))\big)$ |
| **Keep rule** | $L_i^{-} - L_i^{+} \ge \tau_f$ |
| Inference | decode until `→`, **interrupt**, call API, insert response **and** `</API>`, continue |
| No vocabulary change | `<API>`, `</API>`, `→` are implemented as `[`, `]`, `->` |

## [Lec 45 — Automatic Prompt Engineering](../notes/week-09/45-automatic-prompt-engineering.md)

| Item | Exactly this |
|---|---|
| Automated prompt design | search or generation for the most effective prompt template **in a predefined search space** |
| Three approaches, in deck order | prompt paraphrasing · gradient-based discrete prompt search · prompt tuning |
| Token space vs continuous space | first two search **token space**; prompt tuning searches **continuous space** (soft prompts) |
| AutoPrompt Eq. 2 | $V_{\text{cand}} = \text{top-}k_{\,w\in V}\left[\mathbf{e}_{w}^{\top}\nabla\log p(y\mid\mathbf{x}_{\text{prompt}})\right]$ |
| Gradient taken w.r.t. | the **input embedding** of the $j$-th trigger token |
| First-order approximation | $f(\mathbf{x}+\epsilon\mathbf{v}) - f(\mathbf{x}) \approx \epsilon\,\mathbf{v}^{\top}\nabla f(\mathbf{x})$ |
| Loss change on a swap | $\Delta\mathcal{L} \approx (\mathbf{e}_{w'}-\mathbf{e}_w)^{\top}\nabla_{\mathbf{e}_w}\mathcal{L}$ |
| Why Eq. 2 drops $\mathbf{e}_w$ | $\mathbf{e}_w^{\top}\mathbf{g}$ is constant across candidates — it cannot change the top-$k$ |
| Class probability from an MLM | $p(y\mid\mathbf{x}_{\text{prompt}}) = \sum_{w\in V_y} p([\text{MASK}]=w\mid\mathbf{x}_{\text{prompt}})$ |
| Beam search in AutoPrompt | top-$k$ candidates per position, searched **left to right** across positions |
| Label-token step 1 | logistic classifier on the contextualized `[MASK]` embedding: $p(y\mid\mathbf{h})\propto\exp(\mathbf{h}\cdot\mathbf{y}+\beta_y)$ |
| Label-token step 2 | substitute $\mathbf{w}_{\text{out}}$ for $\mathbf{h}$: $s(y,w)=p(y\mid\mathbf{w}_{\text{out}})$, $V_y = \text{top-}k\,[s(y,w)]$ |
| EvoPrompt | LLM does **crossover** then **mutation**; fitness = dev-set accuracy; survival of the fittest |
| Prompt-tuning slogan | "Optimize the embeddings of a prompt, instead of the words" |
| Prompt-tuning shapes | $\mathbf{X}_e\in\mathbb{R}^{n\times d}$, $\mathbf{P}_e\in\mathbb{R}^{p\times d}$, $[\mathbf{P}_e;\mathbf{X}_e]\in\mathbb{R}^{(p+n)\times d}$ |
| What is updated | **only $\mathbf{P}_e$** — the backbone is frozen |
| Trainable count | $p \times d$ |
| Prompt- vs prefix-tuning | prompt-tuning = soft prompts at the **input only**; prefix-tuning = learned prefixes at **every layer** (Lec 46) |
| Power-of-Scale finding | prompt tuning **matches** model tuning **as model size increases** |

## [Lec 46 — Parameter-Efficient Fine-Tuning I: Adapters and Prefix-Tuning](../notes/week-10/46-peft-adapters-prefix.md)

| Item | Exactly this |
|---|---|
| Four downsides of prompt-based learning | Inefficiency · Poor performance · Sensitivity (wording, example order) · Lack of clarity (random labels work) |
| Full fine-tuning vs PEFT | Update **all** model parameters vs update a **small subset** |
| The three perspectives | **Functional** → adapters · **Input** → prefix-tuning · **Parameter** → LoRA |
| Adapter function | $f_\phi(\mathbf{x}) = \mathbf{W}^U(\sigma(\mathbf{W}^D\mathbf{x}))$, used as $\mathbf{x} + f_\phi(\mathbf{x})$ |
| Adapter shapes | $\mathbf{W}^D \in \mathbb{R}^{r\times d}$ (down), $\mathbf{W}^U \in \mathbb{R}^{d\times r}$ (up), $r \ll d$ |
| Adapters per layer | **2** — after the projection following multi-head attention, and after the two feed-forward layers |
| Adapter parameter count | $2L\,(2d_{\text{model}}r + d_{\text{model}} + r)$; $\approx r/(3d)$ of the model |
| Adapter initialisation | $\mathbf{W}^U \approx \mathbf{0}$ so the module starts as the **identity** |
| Parallel adapters (PALs) | $\text{TS}(\mathbf{h}) = \mathbf{V}^Dg(\mathbf{V}^E\mathbf{h})$; $\mathbf{h}^{(l+1)} = \text{LN}(\mathbf{h}^{(l)} + \text{SA}(\mathbf{h}^{(l)}) + \text{TS}(\mathbf{h}^{(l)}))$ |
| Prefix-tuning | Learned continuous vectors prepended **at every layer**; all pretrained weights frozen |
| Prefix mechanism | $\mathbf{h}_i = \mathbf{P}_\theta[i]$ for $i \in P_{\text{idx}}$ (read, not computed); elsewhere the Transformer computes as usual |
| Prefix parameter count | **All layers:** $d_{\text{model}}\lvert P_{\text{idx}}\rvert L$ · **Embeddings only:** $d_{\text{model}}\lvert P_{\text{idx}}\rvert$ |
| Prompt vs prefix | Prompt-tuning = **input layer only** ([Lec 45](../notes/week-09/45-automatic-prompt-engineering.md)) · Prefix-tuning = **every layer** · ratio $= L$ |
| Infix | $\{\mathbf{x};\text{INFIX};\mathbf{y}\}$; in decoder-only models affects **only $\mathbf{y}$**; slightly underperforms prefix; combinable |
| Adapter downside | **New functions increase the number of operations** at inference |
| Prefix downside | **Extends the model's context window**; **requires large models to perform well** |

## [Lec 47 — LoRA and Its Variants](../notes/week-10/47-lora-and-variants.md)

| Item | Exactly this |
|---|---|
| LoRA decomposition | $\mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$, $\mathbf{B}\in\mathbb{R}^{d\times r}$, $\mathbf{A}\in\mathbb{R}^{r\times k}$, $r \ll \min(d,k)$ |
| LoRA forward pass | $\mathbf{h} = \mathbf{W}_0\mathbf{x} + \Delta\mathbf{W}\mathbf{x} = \mathbf{W}_0\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$ (scaled by $\alpha/r$ in practice) |
| The hypothesis | the **update** $\Delta\mathbf{W}$ has low intrinsic rank during adaptation — **not** $\mathbf{W}_0$ (Aghajanyan et al. 2020) |
| Initialisation | $\mathbf{A} \sim \mathcal{N}(0,\sigma^2)$, $\mathbf{B} = \mathbf{0}$, so $\Delta\mathbf{W} = \mathbf{0}$ at step 0 |
| What is frozen | $\mathbf{W}_0$; only $\mathbf{A}$ and $\mathbf{B}$ receive gradients |
| At inference | merge: $\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$ → **no additional latency** |
| Parameters, one $d\times k$ matrix | $r(d+k)$; square case $2dr$ |
| Compression ratio (square) | $d^2 / 2dr = d/(2r)$ |
| LoRA over a stack | $\lvert\Theta\rvert = 2 \times L_{\text{tuned}} \times d_{\text{model}} \times r$ |
| Best matrices to adapt | $\mathbf{W}_q$ and $\mathbf{W}_v$ |
| Kronecker product | $\mathbf{A}\otimes\mathbf{B}$: replace each $a_{ij}$ by $a_{ij}\mathbf{B}$; shape $(a_1b_1)\times(a_2b_2)$ |
| Kronecker rank | $\operatorname{rank}(\mathbf{A}\otimes\mathbf{B}) = \operatorname{rank}(\mathbf{A})\cdot\operatorname{rank}(\mathbf{B})$ — **not** low-rank |
| KronA update | $\mathbf{W}_{\text{tuned}} = \mathbf{W} + s[\mathbf{A}_k \otimes \mathbf{B}_k]$ |
| KronA params / constraint | $a_1a_2 + b_1b_2$ subject to $a_1b_1 = a_2b_2 = d_h$ |
| KronA fast product | $(\mathbf{A}\otimes\mathbf{B})\mathbf{x} = \gamma(\mathbf{B}\,\eta_{b_2\times a_2}(\mathbf{x})\,\mathbf{A}^\top)$ |
| VeRA update | $\mathbf{h} = \mathbf{W}_0\mathbf{x} + \boldsymbol\Lambda_b\mathbf{B}\boldsymbol\Lambda_d\mathbf{A}\mathbf{x}$; $\mathbf{A},\mathbf{B}$ **frozen random, shared across layers** |
| VeRA params | $\lvert\Theta\rvert = L_{\text{tuned}} \times (d_{\text{model}} + r)$ |
| VeRA init | $\mathbf{d} = \mathbf{1}$, $\mathbf{b} = \mathbf{0}$ |
| Mergeable (zero latency) | full FT, **LoRA**, **plain KronA**, BitFit, VeRA |
| Not mergeable | adapters, Compacter, KronA$^{\mathrm B}$, KronA$^{\mathrm B}_{\text{res}}$ |

## [Lec 48 — Quantization and QLoRA I](../notes/week-10/48-quantization-qlora-1.md)

| Item | Exactly this |
|---|---|
| Model state (Adam, mixed precision) | $2\theta$ (fp16 weights) $+\,2\theta$ (fp16 grads) $+\,12\theta$ (fp32 optimizer) $= \mathbf{16\theta}$ bytes |
| The 3 slots inside the optimizer state | fp32 **master weights** + Adam **momentum** + Adam **variance**, 4 bytes each |
| Deck's conservative budget | weight 16 bit + grad 16 bit + optimizer 64 bit $= 96$ bit $= \mathbf{12}$ **bytes/param** |
| Why an fp32 master copy | $\eta\nabla$ is too small to be representable in FP16 — the update vanishes |
| Activation memory per layer | $sbh\left(34 + 5\dfrac{as}{h}\right)$ bytes; $\times L$ for the stack |
| LoRA per-parameter cost | 16 + 0.4 + 0.8 + 0.4 $= \mathbf{17.6}$ bit $\approx 2.2$ byte |
| QLoRA per-parameter cost | weight **4 bit** + 0.4 + 0.8 + 0.4; deck prints **5.2 bit** |
| What LoRA removes / does not | removes **gradients and optimizer state** for frozen weights; **does not** remove the weights |
| Absmax quantization | $\mathbf{X}^{\text{Int8}} = \operatorname{round}\!\big(c^{\text{FP32}}\mathbf{X}^{\text{FP32}}\big)$, $c^{\text{FP32}} = \dfrac{127}{\operatorname{absmax}(\mathbf{X}^{\text{FP32}})}$ |
| Dequantization | $\mathbf{X}^{\text{FP32}} = \mathbf{X}^{\text{Int8}} / c^{\text{FP32}}$ |
| Quantization constant | the scale $c$; must be stored with the tensor |
| PTQ vs QAT | PTQ = convert after training, may degrade; QAT = convert during training, better — **QLoRA is QAT** |
| QLoRA's 3 innovations | **4-bit NormalFloat**, **Double Quantization**, **Paged Optimizers** |
| QLoRA storage / compute types | **4-bit storage**, **bfloat16 computation** |
| NF4 idea | weights are ~normally distributed → place the 16 levels at **normal quantiles**, so buckets hold **equal probability mass** |
| NF4 values | $-1.0,\,-0.7,\,-0.53,\,-0.39,\,-0.28,\,-0.18,\,-0.09,\,0,\,0.08,\,0.16,\,0.25,\,0.34,\,0.44,\,0.56,\,0.72,\,1.0$ |
| NF4 split | **7 negative + exact 0 + 8 positive** = 16 |
| Quantization recipe | normalize by absmax → nearest level (binary search) → **store the index** → dequantize = lookup + denormalize |

## [Lec 49 — QLoRA II: Double Quantization and Paged Optimizers](../notes/week-10/49-qlora-2.md)

| Item | Exactly this |
|---|---|
| QLoRA's three innovations | 4-bit NormalFloat · Double Quantization · Paged Optimizers |
| Storage vs compute type | 4-bit **storage**, BFloat16 **computational** |
| Block-wise quantization | one quantization constant $c = 127/\operatorname{absmax}(\text{block})$ per block, not per tensor |
| Why block-wise | "if there are any outliers in a block, they won't affect the quantisation in the other blocks" |
| Double quantization, in words | quantize the **quantization constants** (first level 32-bit → 8-bit) |
| Second-level equation | $c_1^{\text{Int8}} = \operatorname{round}(c_2^{\text{FP32}} c_1^{\text{FP32}})$ |
| Constant overhead, before | $32/64 = 0.5$ bits/parameter |
| Constant overhead, after | $(8 + 32/256)/64 = 0.127$ bits/parameter |
| Saving | $\approx 0.373$ bits/parameter |
| Paged optimizer | NVIDIA unified memory doing automatic page-to-page transfers of **optimizer state** between GPU and CPU |
| What paging solves | **memory spikes** (long sequences / gradient-checkpointing boundaries), not average memory |
| Gradient checkpointing | discard activations, recompute in the backward pass; checkpoint every $\sqrt{n}$ layers |
| QLoRA forward pass | $\mathbf{Y}^{\text{BF16}} = \mathbf{X}\,\text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{\text{NF4}}) + \tfrac{\alpha}{r}\mathbf{X}\mathbf{B}\mathbf{A}$ |
| The dequantized $\mathbf{W}^{\text{BF16}}$ | is **deleted** after the layer's matmul |

## [Lec 50 — Other Parameter Efficient Methods: Pruning, Distillation](../notes/week-10/50-pruning-and-distillation.md)

| Item | Exactly this |
|---|---|
| Quantization (deck's words) | "no parameters are changed, up to $k$ bits of precision" |
| Pruning (deck's words) | "a number of parameters are set to zero, the rest are unchanged" |
| Distillation (deck's words) | "~all parameters are changed"; "train one model (the student) to replicate the behavior of another model (the teacher)" |
| Lottery Ticket Hypothesis | a randomly-initialised dense net contains a sparse subnetwork that, trained in isolation **from the same initialisation**, matches the full net's accuracy |
| Magnitude pruning | zero out the $X\%$ of parameters with least $\mid W_{ij}\mid $; a type of **unstructured** pruning |
| Movement pruning | keep weights that **move the most away from 0** during fine-tuning; $S_{ij} = -\sum_t (\partial\mathcal{L}/\partial W_{ij})W_{ij}$ |
| Wanda score | $S_{ij} = \lvert W_{ij}\rvert \cdot \lVert\mathbf{X}_j\rVert_2$, compared **per output row**, not globally |
| Structured pruning | learn masks that turn off whole components; coarse = whole MHA/FFN, fine = heads and hidden dims |
| Soft targets | the teacher's full distribution; the relative sizes of the non-argmax classes are the **dark knowledge** |
| Temperature | $p_i^{(T)} = \exp(z_i/T)/\sum_j\exp(z_j/T)$; $T>1$ flattens, ranking unchanged, odds ratio $= e^{(z_i-z_j)/T}$ |
| Distillation loss | $\lambda T^2 D_{\mathrm{KL}}(p^{(T)}\|q^{(T)}) + (1-\lambda)\mathcal{L}_{\text{CE}}(y, q^{(1)})$ |
| Word-level KD | match the teacher's per-step vocabulary distribution at gold-forced positions |
| Sequence-level KD | $\mathcal{L}_{\text{SEQ-KD}} \approx -\log p(\mathbf{t} = \hat{\mathbf{y}} \mid \mathbf{s})$; combined $\mathcal{L} = (1-\alpha)\mathcal{L}_{\text{SEQ-NLL}} + \alpha\mathcal{L}_{\text{SEQ-KD}}$ |
| Forward KL | $D_{\mathrm{KL}}(p_{\text{teacher}}\|q_{\text{student}})$ — mode-**covering**, used by standard KD |
| Reverse KL (MiniLLM) | $D_{\mathrm{KL}}(q_{\text{student}}\|p_{\text{teacher}})$ — mode-**seeking**; "prevents the student from overestimating the low-probability regions of the teacher distribution" |

## [Lec 51 — Scaling Laws of LLMs](../notes/week-11/51-scaling-laws.md)

| Item | Exactly this |
|---|---|
| The three factors | $N$ = parameters (**excluding embeddings**), $D$ = dataset size in tokens, $C$ = compute in PF-days |
| PF-day | $10^{15}$ ops/s for one day $= 8.64\times10^{19}$ FLOPs ("roughly $10^{20}$"); $\approx$ 3 A100-days |
| A100 | 312 TFLOP/s $= 2.7\times10^{19}$ FLOPs/day |
| $L(N)$ | $(N_c/N)^{\alpha_N}$, $\alpha_N \approx 0.076$, $N_c \approx 8.8\times10^{13}$ |
| $L(D)$ | $(D_c/D)^{\alpha_D}$, $\alpha_D \approx 0.095$, $D_c \approx 5.4\times10^{13}$ tokens |
| $L(C)$ | $(C_c/C)^{\alpha_C}$, $\alpha_C \approx 0.050$, $C_c \approx 3.1\times10^{8}$ PF-days |
| Power law on log-log | a straight line; **slope $= -\alpha$** |
| Forward FLOPs/parameter/token | **2** — one multiply, one add |
| Backward | $\approx 2\times$ forward $= 4N$ |
| **Training compute** | $\boxed{C \approx 6ND}$, equivalently $C = 6NBS$ at batch $B$ for $S$ steps |
| Kaplan's full forward count | $C_{\text{forward}} = 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$; second term negligible when $d_{\text{model}} \gg n_{\text{ctx}}/12$ |
| Kaplan's recipe | 10× compute → **5×** model, **2×** data; 100× → **25×** model, **4×** data |
| Chinchilla's recipe | 10× compute → **3.1×** both; 100× → **10×** both |
| Chinchilla TLDR | **~20 tokens per parameter** |
| Chinchilla-optimal split | $N = \sqrt{C/120}$, $D = 20N$ |
| Kaplan's bug | one **learning-rate schedule** for all runs → small/long runs under-trained → bias toward bigger models |
| Papers | Kaplan et al. **2020**, arXiv **2001.08361**; Hoffmann et al. **2022**, *Training Compute-Optimal Large Language Models* (DeepMind) |

## [Lec 52 — Modern LLMs and Architecture Variations I](../notes/week-11/52-modern-llms-and-activations.md)

| Item | Exactly this |
|---|---|
| Leaky ReLU | $f(x) = \max(\alpha x, x)$, **$\alpha$ a hyperparameter, $\alpha = 0.1$ on this deck** |
| PReLU | **same formula**, $\alpha$ **learned via backprop** |
| ELU | $x$ if $x>0$, else $\alpha(e^x - 1)$; drawback: **requires `exp()`** |
| SELU | $\lambda x$ if $x>0$, else $\lambda\alpha(e^x-1)$; $\alpha = 1.67326\ldots$, $\lambda = 1.05070\ldots$ |
| SELU's selling point | **self-normalising — train deep nets without BatchNorm** |
| GELU | $x\,P(X \le x) = x\Phi(x) = \tfrac{x}{2}\big(1 + \text{erf}(x/\sqrt2)\big) \approx x\sigma(1.702x)$ |
| GELU's intuition | multiply by 0 or 1 at random, large values more likely by 1; **take the expectation** = data-dependent dropout |
| GELU used in | **BERT, GPT, GPT-2, GPT-3** (Hendrycks & Gimpel, 2016) |
| Swish | $x\cdot\sigma(\beta x)$; **$\beta=1.72 \Rightarrow$ GELU**, **$\beta\to\infty \Rightarrow$ ReLU**; $\beta$ fixed or trainable |
| Swish$_1$ | also called **SiLU** |
| FFN (original) | $\max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$ |
| FFN (modern) | **no bias terms**: $\max(\mathbf{x}\mathbf{W}_1, 0)\mathbf{W}_2$ |
| GLU | $\sigma(\mathbf{x}\mathbf{W}+\mathbf{b}) \odot (\mathbf{x}\mathbf{V}+\mathbf{c})$ — **component-wise product of two linear projections, one through a sigmoid** |
| Bilinear | GLU with **no activation at all** |
| ReGLU / GEGLU / SwiGLU | same shape, gate activation ReLU / GELU / Swish$_\beta$ |
| SwiGLU used in | **LLaMA, Mistral, Qwen, PaLM** |
| GLU parameter cost | **three** matrices not two → 1.5× at equal $d_{ff}$ |
| The fix | $d_{ff}' = \tfrac{2}{3}d_{ff}$, i.e. $\tfrac{8}{3}d_{\text{model}}$ when $d_{ff} = 4d_{\text{model}}$ |
| LayerNorm (standard) | $\gamma\odot\dfrac{\mathbf{h}-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ — re-centre **and** re-scale |
| LayerNorm (this deck) | $\alpha\cdot\dfrac{\mathbf{h}-\mu}{\sigma+\epsilon} + \beta$ — **$\epsilon$ outside the root** |
| RMSNorm | $\gamma\odot\dfrac{\mathbf{h}}{\sqrt{\frac1n\sum_k h_k^2 + \epsilon}}$, $\sigma_{\text{rms}} = \big(\tfrac1n\sum_k h_k^2\big)^{1/2}$ |
| RMSNorm drops | the **mean-centering** (and the $\beta$ shift); "only re-scales, does not re-center" |
| Identity | $\sigma_{\text{rms}}^2 = \sigma^2 + \mu^2$ |
| The five "what's new" | SwiGLU · RMSNorm · bias in attention layer · RoPE · Mixture-of-experts |

## [Lec 53 — Modern Positional Embeddings: RoPE and ALiBi](../notes/week-11/53-positional-embeddings-rope-alibi.md)

| Item | Exactly this |
|---|---|
| Relative-attention form (deck) | $\alpha(i,j) = \text{Softmax}\!\left(\frac{\mathbf{q}_i\mathbf{k}_j^\top + \text{PE}(i,j)}{\sqrt{d}} + \text{Mask}(i,j)\right)$ |
| T5 offset | $d(i,j) = i - j$; simple design $\text{PE}(i,j) = u_{i-j}$ |
| T5 bias score | $\mathbf{q}_i\mathbf{k}_j^\top + u_{b(i-j)}$, with $b(\cdot)$ the bucket index |
| T5 bucketing | first $\frac{n_b+1}{2}$ buckets exact (one offset each); rest widen **logarithmically**; last bucket is a catch-all |
| ALiBi bias | $\text{PE}(i,j) = -\beta(i-j) = \beta(j-i)$ |
| ALiBi slope (deck) | $\beta_k = 1/2^{8/k}$ |
| ALiBi slope (paper, correct) | $\beta_k = 2^{-8k/n_{\text{head}}}$; for $n_{\text{head}}=8$ that is $2^{-k}$ |
| ALiBi position embeddings | **none at all** |
| RoPE, additive vs multiplicative | sinusoidal $\mathbf{e}_i = \mathbf{x}_i + \text{PE}(i)$; RoPE $\mathbf{e}_i = \mathbf{x}_i\mathbf{R}(i)$ |
| 2-D rotation (deck's row form) | $\text{Ro}(\mathbf{x},\theta)=[x_1\;x_2]\begin{bmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{bmatrix}$ |
| RoPE frequency | $\theta_k = 10000^{-2(k-1)/d}$, $k = 1,\ldots,d/2$ |
| RoPE $d$-dim form | **block-diagonal**, $d/2$ independent $2\times2$ rotations |
| **The key property** | $\mathbf{q}_m^\top\mathbf{k}_n = \mathbf{x}_m^\top\mathbf{W}_q\mathbf{R}^d_{\Theta,n-m}\mathbf{W}_k\mathbf{x}_n$ — depends on $n-m$ only |
| Why it cancels | $\mathbf{R}_a^\top\mathbf{R}_b = \mathbf{R}_{b-a}$ |
| RoPE slogan | **absolute encoding applied, relative behaviour obtained** |
| RoPE applied to | $\mathbf{q}$ and $\mathbf{k}$ only (**not** $\mathbf{v}$), after projection, at **every** layer |
| RoPE preserves | the **magnitude** of the embedding |
| Position interpolation | $\mathbf{f}'(\mathbf{x},m) = \mathbf{f}(\mathbf{x}, mL/L')$ |
| Adjusted base frequency | base $b$: $10{,}000 \to 500{,}000$, which **reduces** every rotation angle |
| Extrapolate vs interpolate | extrapolate = apply outside the fitted range; interpolate = map back inside it (**preferred**) |

## [Lec 54 — Long Sequence Modeling](../notes/week-11/54-long-sequence-modeling.md)

| Item | Exactly this |
|---|---|
| Attention complexity | $O(n^2 d)$ time, $O(n^2)$ memory; $2n^2d$ MACs — **doubling $n$ quadruples the cost** |
| Sparse attention | $\text{Att}_{\text{sparse}}(\mathbf{q}_i,\mathbf{K}_{\le i},\mathbf{V}_{\le i}) = \sum_{j\in G}\alpha'_{ij}\mathbf{v}_j$, $G \subseteq \{0,\dots,i\}$, weights normalised **over $G$** |
| Sparse attention's limit | reduces computation but **still keeps the whole KV cache** |
| Generalized attention | $\mathbf{v}'_i = \dfrac{\sum_j \text{sim}(\mathbf{q}_i,\mathbf{k}_j)\mathbf{v}_j}{\sum_j \text{sim}(\mathbf{q}_i,\mathbf{k}_j)}$ — $O(n^2)$ |
| Kernel substitution | $\text{sim}(\mathbf{q},\mathbf{k}) = \phi(\mathbf{q})^\top\phi(\mathbf{k})$ |
| The re-ordering | $(\phi(\mathbf{Q})\phi(\mathbf{K})^\top)\mathbf{V} = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{V})$ — **exact**, by associativity |
| Linear complexity | $O(nd^2)$ time, $O(d^2)$ state; crossover at $n = d$ |
| Deck's feature map | $\phi(x) = \text{elu}(x) + 1$ |
| Running sums | $\mathbf{S}_i = \sum_{j\le i}\phi(\mathbf{k}_j)\mathbf{v}_j^\top$ ($d\times d$), $\mathbf{Z}_i = \sum_{j\le i}\phi(\mathbf{k}_j)$; $\mathbf{v}'_i = \phi(\mathbf{q}_i)^\top\mathbf{S}_i / \phi(\mathbf{q}_i)^\top\mathbf{Z}_i$ |
| Why causal masking is still linear | $\mathbf{S}_i, \mathbf{Z}_i$ computable from $\mathbf{S}_{i-1},\mathbf{Z}_{i-1}$ in **constant time** |
| Cost of linear attention | no softmax ⇒ **flatter, less selective** attention distribution |
| Mem = KV cache | $\text{Att}(\mathbf{q}_i,\text{Mem}) = \text{Att}_{\text{qkv}}(\mathbf{q}_i,\mathbf{K}_{\le i},\mathbf{V}_{\le i})$, $\text{Mem} = (\mathbf{K}_{\le i},\mathbf{V}_{\le i})$ |
| Four fixed-size caches | (a) window, (b) moving average, (c) recurrent network, (d) hybrid (compressed + local) |
| Window cache | $\text{Mem} = (\mathbf{K}_{[i-n_c+1,i]}, \mathbf{V}_{[i-n_c+1,i]})$, $n_c$ = window size |
| Moving-average cache | unweighted $\frac{1}{n_c}\sum_j \mathbf{k}_j$; weighted version uses $\beta_1 < \dots < \beta_{n_c}$ (recent gets more) |
| RNN-like cache | $\text{Mem} = f((\mathbf{k}_i,\mathbf{v}_i), \text{Mem}_{\text{pre}})$; deck's instance $\text{Mem}_i = \frac{(\mathbf{k}_i,\mathbf{v}_i)+i\,\text{Mem}_{i-1}}{i+1}$ |
| Hybrid principle | local context explicit with minimal loss; long-range context more compressed |
| Internal vs external memory | internal = updates to the KV cache; external = independent model accessing large-scale context |
| Memorizing Transformer layer | **one layer near the top** = kNN-augmented: dense local self-attention **+** approximate kNN search into external memory |
| Memory write rule | after each training step, the local context's (key, value) pairs are **appended**; oldest dropped; size $M$ per head |
| Merge gate | $g = \sigma(b_g)$, $\mathbf{V}_a = \mathbf{V}_m \odot g + \mathbf{V}_c \odot (1-g)$; $b_g$ = **learned per-head scalar** |
| kNN softmax scope | over the **top-$k$ retrieved keys only** |
| Why memory can be huge | the index is **non-differentiable** — no backprop, no optimiser state |

## [Lec 55 — Retrieval Augmented Generation](../notes/week-11/55-retrieval-augmented-generation.md)

| Item | Exactly this |
|---|---|
| Parametric knowledge | information encoded in the model's **parameters/weights during training**, used for tasks needing it |
| Hallucination | "a response that is **not faithful to the facts of the world**" |
| Four motivations for retrieval | long tail · outdated/hard to update · hard to interpret and verify · (editorial) private data |
| RAG datastore | **raw text corpus**, billions~trillions of tokens; **not** labeled data, **not** a knowledge base |
| Query | the **retrieval input** — *not necessarily* the input to the LM |
| Retrieval goal | find a small subset of datastore elements most similar to the query |
| TF-IDF sim | $\mathrm{sim}(i,j) = \mathrm{tf}_{i,j}\times\log\frac{N}{\mathrm{df}_i}$ |
| Dense sim | $\mathrm{sim}(i,j) = \mathrm{Encoder}(i)\cdot\mathrm{Encoder}(j)$, into $h$ dimensions |
| Index | $\operatorname*{argTop-}k_{d\in\mathcal{D}} \mathrm{sim}(q,d)$ by fast NN search; FAISS / Distributed FAISS / SCaNN |
| Three design axes | **What** to retrieve · **How** to use it · **When** to retrieve |
| REALM retrieval | $z_1,\dots,z_k = \operatorname*{argTop-}k(\mathbf{x}\cdot\mathbf{z})$ over 13M Wikipedia chunks |
| REALM read | `[MASK]` $z_i$ `[SEP]` $x$ → LM → $P(y\mid x,z_i)$, weighted average |
| REALM marginal | $\sum_{z\in\mathcal{D}} P(z\mid x)P(y\mid x,z)$, approximated by top $k$ (0 otherwise) |
| kNN-LM datastore entry | $(k_i, v_i) = (f(c_i),\ \text{next token})$ |
| kNN-LM normalization | $p(k_i) \propto \exp(-d_i)$ — softmax over **negative** distance |
| kNN-LM aggregation | $p_{\text{kNN}}(y) = \sum_i \mathbf{1}[y = v_i]\,p(k_i)$ |
| kNN-LM interpolation | $p(y) = \lambda\,p_{\text{kNN}}(y) + (1-\lambda)\,p_{\text{LM}}(y)$ |
| Same rule, p. 88's notation | $\Pr(\cdot\mid\mathbf{h}_i) = \lambda\Pr_{k\text{nn}}(\cdot\mid\mathbf{h}_i) + (1-\lambda)\Pr_{\text{lm}}(\cdot\mid\mathbf{h}_i)$ |
| Memory-based taxonomy (pp. 86–89) | (a) kNN-search-augmented **attention** · (b) **kNN-LM** · (c) **RAG** |
| kNN-LM training | **none** — datastore built from a frozen LM, $\lambda$ tuned on dev |
| Training challenge | updating the index = **recomputing dense vectors for all documents** |
| Four training options | independent · sequential · joint w/ async index update (every $T$ steps) · joint w/ in-batch approximation |
| Self-Route | step 1 RAG-and-Route ("write unanswerable if…"), step 2 long-context prediction for the declined queries |

## [Lec 56 — Model Interpretability: Probing and the Logit Lens](../notes/week-12/56-interpretability-probing.md)

| Item | Exactly this |
|---|---|
| Interpretability (deck's definition) | *the study of understanding the decisions that AI systems make and putting them into easily human-understandable terms* |
| Why interpretability (deck) | to iteratively design systems that are more **performant** and **human-understandable** |
| Probe (deck's definition) | *a classifier specifically trained to predict some property from a pretrained model's representations* |
| Probing: what trains | **the classifier only**; the pretrained model is **frozen, no further fine-tuning** |
| Probing's inference | if a classifier can predict the property from the representation, the property is **encoded** in it |
| Edge probing | probing over **spans and pairs of spans**; span pooling → MLP → **independent binary** classifiers |
| Three layer-combination options | **lexical baseline** / **cat** (embeddings ⊕ top layer) / **mix** (learned scalar combination of all layers, ELMo-style) |
| Scalar mixture | $\mathbf{h}_{i,\tau} = \gamma_\tau \sum_{\ell=0}^{L} s_\tau^{(\ell)} \mathbf{h}_i^{(\ell)}$, $s$ softmax-normalised |
| Extra parameters for the mixture | $L+2$ (one scalar per layer, $\ell = 0 \ldots L$, plus $\gamma$) |
| Centre of gravity | $\bar{E}_s[\ell] = \sum_{\ell=0}^{L} \ell \cdot s_\tau^{(\ell)}$ — weighted by **mixture scalars** |
| Differential score | $\Delta_\tau^{(\ell)} = \text{Score}(P_\tau^{(\ell)}) - \text{Score}(P_\tau^{(\ell-1)})$ |
| Expected layer | $\bar{E}_\Delta[\ell] = \dfrac{\sum_{\ell=1}^{L}\ell\,\Delta_\tau^{(\ell)}}{\sum_{\ell=1}^{L}\Delta_\tau^{(\ell)}}$ — weighted by **score gains** |
| Higher CoG means | the information is captured by **higher layers** |
| Control task | same probe, same inputs, **random labels fixed per word type** |
| Selectivity | accuracy(real task) − accuracy(control task); **higher is better** |
| Logit lens | $\mathrm{softmax}(\mathbf{W}_U \mathbf{h}^{(\ell)})$ — the model's **pretrained unembedding matrix** applied to an **intermediate** hidden state |
| Rank view | the rank of the **final layer's top-1 token** in each intermediate distribution (not the true token) |
| KL view | $D_{\mathrm{KL}}(p^{(\ell)} \| p^{(L)})$ — intermediate w.r.t. final; 0 at the last layer |
| Tuned lens | $\text{TunedLens}_\ell(\mathbf{h}_\ell) = \text{LogitLens}(\mathbf{A}_\ell \mathbf{h}_\ell + \mathbf{b}_\ell)$ |
| Translator | the per-layer **affine** pair $(\mathbf{A}_\ell, \mathbf{b}_\ell)$ |
| Tuned-lens training objective | $\arg\min\ \mathbb{E}_{\mathbf{x}}[D_{\mathrm{KL}}(f_{>\ell}(\mathbf{h}_\ell) \,\|\, \text{TunedLens}_\ell(\mathbf{h}_\ell))]$ |
| Probing is… | **correlational**, not causal — see [Lec 58](../notes/week-12/58-interpretability-ffn-and-causal-tracing.md) |

## [Lec 57 — Model Interpretability: Multilingual](../notes/week-12/57-interpretability-multilingual.md)

| Item | Exactly this |
|---|---|
| The deck's three topics | How do multilingual transformers work? · Language-specific neurons · Multilingual workflow |
| The latent-language paper | *Do Llamas Work in English? On the Latent Language of Multilingual Transformers* (arXiv 2402.10588) |
| The two tasks | **translation** (French word → Chinese) and **repetition** (Chinese word → same Chinese word) |
| Tool used | the **logit lens** of [Lec 56](../notes/week-12/56-interpretability-probing.md) — unembed an intermediate layer |
| The finding | English dominates the **middle** layers; the target language takes over in the **last few** |
| Phase I | building good token representations; high entropy; neither language has mass |
| Phase II | **concept space with higher probability to tokens in English**; entropy collapses |
| Phase III | predict concepts in the chosen target language |
| What it does **not** mean | not literal translation into English — the **representation space** is English-*aligned* |
| Activation probability | $p^k_{i,j} = \mathbb{E}(\mathbb{I}(\text{act\_fn}(\tilde{\mathbf{h}}^{(i)}\mathbf{W}_1^{(i)})_j > 0) \mid \text{language } k)$ |
| LAPE | $\text{LAPE}_{i,j} = -\sum_{k=1}^{l} p'^k_{i,j}\log p'^k_{i,j}$, with $\mathbf{p}'$ the **L1-normalised** $\mathbf{p}$ |
| LAPE direction | **LOW LAPE = language-specific.** High = language-agnostic |
| LAPE bounds | $0$ (one language) to $\log l$ (uniform); uniform over $m$ of $l$ gives $\log m$ |
| LAPE paper | *Language-Specific Neurons: The Key to Multilingual Capabilities in LLMs* (arXiv 2402.16438) |
| Second method | $\text{Imp}(N^{(i)}\mid c) = \lVert T_i\backslash N^{(i)}(\mathbf{h}^{(i)}) - T_i(\mathbf{h}^{(i)})\rVert_2$, require $\ge\epsilon$ for **all** $c_l \in \mathcal{C}$ (arXiv 2402.18815) |
| Workflow, one line | understand the non-English query, interpret in English, solve in English, generate in the original language |
| Where each happens | reasoning in English via **self-attention**; multilingual knowledge via **feed-forward** |
| Intervention payoff | clamping a few hundred neurons changes the **output language**, no retraining |

## [Lec 58 — Interpretability III: FFN as Key-Value Memories and Causal Tracing](../notes/week-12/58-interpretability-ffn-and-causal-tracing.md)

| Item | Exactly this |
|---|---|
| FFN as memory | $\mathrm{FF}(\mathbf{x}) = f(\mathbf{x}\mathbf{K}^\top)\,\mathbf{V}$, $f$ = ReLU |
| Shapes | $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{d_m \times d}$; $\mathbf{x} \in \mathbb{R}^{d}$; row $i$ = $\mathbf{k}_i$ / $\mathbf{v}_i$ |
| Neural memory | $\mathrm{MN}(\mathbf{x}) = \mathrm{softmax}(\mathbf{x}\mathbf{K}^\top)\mathbf{V} = \sum_i p(\mathbf{k}_i\mid \mathbf{x})\mathbf{v}_i$ — "almost identical" |
| The one difference | softmax (normalised) vs ReLU (**unnormalized, non-negative**) |
| Memory coefficient | $m_i = f(\mathbf{x}\cdot\mathbf{k}_i)$; $d_m$ = **number of memories in the layer** |
| The conjecture | $\mathbf{k}_i$ captures a **pattern** in the input; $\mathbf{v}_i$ is the **distribution of tokens that follow** that pattern |
| Trigger example | a training **prefix** $x_1\ldots x_j$ with the highest memory coefficient for $\mathbf{k}_i^{\ell}$ |
| Layer trend (keys) | layers **1–9 shallow** (surface n-grams, shared last word); layers **10–16 semantic** (shared topic, no surface similarity) |
| Values as distributions | $\mathbf{p}_i^{\ell} = \mathrm{softmax}(\mathbf{v}_i^{\ell}\cdot \mathbf{E})$ |
| Fact tuple | $t = (s, r, o)$ — subject, relation, object |
| Residual update | $\mathbf{h}_i^{(l)} = \mathbf{h}_i^{(l-1)} + \mathbf{a}_i^{(l)} + \mathbf{m}_i^{(l)}$ |
| MLP term | $\mathbf{m}_i^{(l)} = \mathbf{W}_{proj}^{(l)}\sigma(\mathbf{W}_{fc}^{(l)}\gamma(\mathbf{a}_i^{(l)}+\mathbf{h}_i^{(l-1)}))$ |
| The three runs | **clean** (record all states) → **corrupted** (noise on subject-token *embeddings*) → **corrupted-with-restoration** (patch **one** clean state) |
| Corruption | $\mathbf{h}_i^{(0)} := \mathbf{h}_i^{(0)} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(0;\nu)$, **subject indices only**, **layer 0** |
| Indirect effect | $\mathbb{P}_{\text{restore}}[o] - \mathbb{P}_*[o]$; averaged = **AIE** |
| The result | **early-site MLP at the LAST SUBJECT TOKEN** stores the fact; **late-site attention at the last token** transports it |
| Severing ablation | severing **MLP** after restoration kills the early-site effect; severing **attention** does not |
| Why it matters | correlational probing cannot support editing; a causal localisation can |

## [Lec 59 — Trustworthy LLMs: the Taxonomy](../notes/week-12/59-trustworthy-llms-taxonomy.md)

| Item | Exactly this |
|---|---|
| The seven dimensions, in deck order | **Reliability, Safety, Fairness, Resistance to Misuse, Explainability & Reasoning, Social Norm, Robustness** |
| Reliability's five leaves | Misinformation, Hallucination, Inconsistency, **Miscalibration**, Sycophancy |
| Safety's six leaves | Violence, Unlawful Conduct, Harms to Minor, Adult Content, Mental Health Issues, Privacy Violation |
| Fairness's four leaves | Injustice, Stereotype Bias, Preference Bias, Disparate Performance |
| Resistance to Misuse's four leaves | Propagandistic Misuse, Cyberattack Misuse, Social-engineering Misuse, **Leaking Copyrighted Content** |
| Explainability & Reasoning's three | Lack of Interpretability, Limited Logical Reasoning, Limited Causal Reasoning |
| Social Norm's three | Toxicity, Unawareness of Emotions, Cultural Insensitivity |
| Robustness's four | Prompt Attacks, Paradigm & Distribution Shifts, Interventional Effect, Poisoning Attacks |
| Reliability defined | generating correct, truthful, and consistent outputs **with proper confidence** |
| Safety defined | avoiding unsafe and illegal outputs, and leaking private information |
| Fairness defined | avoiding bias and ensuring no disparate performance |
| Resistance to Misuse defined | prohibiting the misuse **by malicious attackers** to do harm |
| Explainability & Reasoning defined | the ability to explain the outputs to users and reason correctly |
| Social Norm defined | reflecting the universally shared human values |
| Robustness defined | resilience against adversarial attacks and distribution shift |
| Misinformation vs hallucination | misinformation = a **false statement about a real thing**; hallucination = **fabricated** content with no grounding |
| Inconsistency | **different answers to the same question** across phrasings, sessions, users, or turns of one conversation |
| Sycophancy | model **changes its answer to agree with the user** — flattering, reconfirming misconceptions — rather than to be correct |
| Sycophancy trigger | users **challenge** the output or **repeatedly force** compliance |
| Unlawful conduct caveat | obey the laws of the **country where the model operates** — jurisdiction-dependent |
| Copyright leakage cause | the **memorization effect** on training data |
| Five common alignment tasks | removing harmful responses · erasing copyrighted contents · reducing hallucinations · adapting to change of user consent on data usage · adapting to policy change |
| The pivot question | "But can we do something quickly?" → **How to quickly remove the impact of certain training samples on LLMs?** |
| LLM unlearning defined | unlearn pretraining misbehaviours using samples that represent them — **with negative samples only** |
| Unlearning pipeline | pretrained model → **user-reported or red-teaming failed cases** → LLM Unlearning → unlearned model |
| Red-teaming defined | **a form of evaluation that elicits model vulnerabilities that might lead to undesirable behaviors** |
| Jailbreaking (deck's wording) | "another term for red-teaming wherein the LLM is manipulated to break away from its guardrails" |
| Why negative samples are easy | user reporting + internal red-teaming; **"highly automatable"** |
| Why positive samples are hard | must hire humans to write helpful outputs · required in RLHF · **not a direct treatment** |
| Unlearning's use case | limited resources, dynamic environment, negatives only, quick fix; **priority is to stop harmful outputs**, not to maximise helpfulness |
| ASR | $k/N$; $\mathrm{SE} = \sqrt{p(1-p)/N}$; 95% CI $= p \pm 1.96\,\mathrm{SE}$ |
| Rule of three | 0 hits in $N$ trials ⇒ 95% upper bound on the rate $\approx 3/N$ |

## [Lec 60 — Trustworthy LLMs: Machine Unlearning](../notes/week-12/60-machine-unlearning.md)

| Item | Exactly this |
|---|---|
| The four goals | **unlearn effectiveness · utility preservation · generalization · low cost (no retraining)** |
| GA update | $\theta_{t+1} \leftarrow \theta_t + \eta\,\nabla_{\theta_t}\ell(h_\theta(x), y)$ — the deck writes $\lambda$ for $\eta$ |
| Practical Lesson 1 | *"Unlearning longer"* — **loss on negative samples is not a good indicator**; it rises for **3×–5× more steps** |
| Why ascent diverges | $-\log p \to \infty$ as $p \to 0$: no optimum, gradient never vanishes, no stopping point |
| Practical Lesson 2 | *"**Cross-entropy is limited for preserving utility**"* — CE on a normal dataset does **not** maintain normal performance |
| Practical Lesson 3 | **format of the normal dataset matters** — same format as the forget set, to block formatting shortcuts |
| Solution 1 | $\mathcal{L}_{\text{rdn}} = \sum_{(x^{\text{fgt}},\cdot)} \frac{1}{\lvert\mathcal{Y}^{\text{rdn}}\rvert}\sum_{y^{\text{rdn}}} L(x^{\text{fgt}}, y^{\text{rdn}}; \theta_t)$ — bounded below by $\ln\lvert\mathcal{Y}^{\text{rdn}}\rvert$ |
| Solution 2 | $\mathcal{L}_{\text{nor}} = \sum_{(x^{\text{nor}},y^{\text{nor}})}\sum_{i} D_{\mathrm{KL}}\big(h_{\theta^{o}} \,\|\, h_{\theta_t}\big)$, per token |
| KL direction | **forward** (original first) = supervised; RLHF uses **backward** = requires sampling |
| The full update | $\theta_{t+1} \leftarrow \theta_t - \epsilon_1\nabla\mathcal{L}_{\text{fgt}} - \epsilon_2\nabla\mathcal{L}_{\text{rdn}} - \epsilon_3\nabla\mathcal{L}_{\text{nor}}$ |
| Term names | $\epsilon_1$ **Unlearn Harm** · $\epsilon_2$ **Random Mismatch** · $\epsilon_3$ **Maintain Performance** |
| $\mathcal{L}_{\text{fgt}}$ | $:= -\sum L(x^{\text{fgt}}, y^{\text{fgt}}; \theta_t)$ — the **minus** is where ascent lives |
| Sequence loss | $L(x,y;\theta) := \sum_{i=1}^{\lvert y\rvert}\ell(h_\theta(x, y_{<i}), y_i)$ |
| Where the losses apply | **the $y$ part only**, not $(x,y)$ as in RLHF |
| Copyright metric | **leak rate** = BLEU between completion and ground truth, flagged above a threshold |
| Harmfulness metric | **harmful rate**, judged by the **PKU team's moderation model** |
| Unlearning vs RLHF | unlearning **removes** behaviour, needs only the forget data; RLHF teaches **better** behaviour, needs preference pairs |
