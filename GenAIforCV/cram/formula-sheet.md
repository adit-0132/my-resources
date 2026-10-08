# Formula Sheet

Every **Must-memorise** entry from all chapters, in course order. Generated from the chapters by `_build/build_cram.py` — edit the chapters, not this file.

## [Lec 1 — Introduction: Computer Vision and Generative AI](../notes/week-01/01-intro-cv-and-genai.md)

| Item | Exactly this |
|---|---|
| Computer vision | Extracting interpretation from visual data; camera + computer mirrors eye + brain |
| Slide-3 interpretation | "Colour Image, Human, Lady, Monalisa" |
| **Generative AI** | A **subset of AI** that creates **new, original content** — text, images, code, audio, video — by **learning patterns from large amounts of existing data** |
| GenAI vs traditional AI | Traditional AI is **rule-based**; GenAI uses **foundation models** |
| Core technologies (4) | **GANs, VAEs, LLMs, diffusion models** |
| **Generative modelling** | An **unsupervised** learning task; learns patterns of input data from a given distribution to generate new examples belonging to a **similar** distribution |
| Data observation | Performance improves with **size, quality and diversity** of the annotated dataset |
| The Problem | Collecting and labelling large datasets is **expensive, time-consuming, often impractical** |
| The Solution | **Synthetic data generation** creates realistic, **labelled** data **at scale** |
| Challenges (7) | fidelity · artifacts/distortions · **mode collapse** · scene consistency · bias · compute cost & training time · ethical misuse (deepfakes, misinformation) |
| Mode collapse | The model generates **limited variations** of outputs — a **diversity** failure, not a realism failure |
| Industries transformed (4) | healthcare, autonomous driving, entertainment, surveillance |

## [Lec 2 — Generative Modelling: Generative vs Discriminative Learning](../notes/week-01/02-generative-vs-discriminative.md)

| Item | Exactly this |
|---|---|
| Generative modelling (definition) | learning the underlying probability distribution of data so that new, realistic samples can be generated |
| Discriminative model learns | $p(y\mid\mathbf{x})$ directly, or a decision function $f(\mathbf{x})\to y$ |
| Discriminative model does **not** model | $p(\mathbf{x})$, the distribution of the predictors |
| Generative model learns | $p(\mathbf{x})$, or the joint $p(\mathbf{x}, y)$ |
| Generative objective | find $\theta$ with $p_\theta(\mathbf{x},y) \approx p_{\text{data}}(\mathbf{x},y)$ |
| Joint factorisation | $p_\theta(\mathbf{x},y) = p_\theta(y)\,p_\theta(\mathbf{x}\mid y)$ — prior × class-conditional |
| Bayes' theorem | $p(y\mid\mathbf{x}) = \dfrac{p(\mathbf{x}\mid y)\,p(y)}{p(\mathbf{x})}$ |
| Evidence | $p(\mathbf{x}) = \sum_{y'} p(\mathbf{x}\mid y')\,p(y')$ |
| Discriminative objective | maximise $\sum_i \log p_\theta(y_i \mid \mathbf{x}_i)$ |
| Logistic-regression boundary | $\beta_0 + \boldsymbol{\beta}^\top\mathbf{x} = 0$ |
| Generator | $\mathbf{z}\sim p(\mathbf{z})$ → $G$ → $D' = G(\mathbf{z})$, minimise statistical distance to $D$ |
| Distance metrics the deck names | L1 norm, L2 norm |
| Sim2Real mitigations (4) | domain randomization; photorealistic rendering; generative AI augmentation; fine-tuning with real samples |
| Domain-adaptation workflow | train on large synthetic set → fine-tune on small real set |

## [Lec 3 — Generative Vision Models: The Landscape](../notes/week-01/03-generative-vision-models.md)

| Item | Exactly this |
|---|---|
| The five families | VAE, GAN, Autoregressive, Diffusion, Normalizing Flows |
| AE vs VAE | AE maps input to a **single point** in latent space; VAE maps it to a **probability distribution** (typically Gaussian) |
| GAN generator | $G$ = forger; takes noise $\mathbf{z}$ (Gaussian or uniform), outputs $G(\mathbf{z})$ |
| GAN discriminator | $D$ = detective; outputs $D(\mathbf{x})$ = probability the sample is **real**; ≈1 real, ≈0 fake |
| Why VAEs/AEs blur | MSE-style pixel reconstruction **averages multiple plausible outputs** |
| Why GANs are sharp | They optimise **perceptual realism**, not pixel error |
| Autoregressive factorization | $p(x_1,\dots,x_n) = p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)\cdots$ |
| Diffusion forward process | Gradually **add Gaussian noise** over many time steps until pure noise; fixed, not learned |
| Diffusion backward process | Noise **gradually removed by sampling using annealed Langevin dynamics** |
| Normalizing flow definition | Sequence of **invertible and differentiable** transformations of a simple (Gaussian) base distribution |
| Flow's unique property | **Exact** likelihood, via the change-of-variables formula $p_x(x) = p_z(z)\,\lvert dz/dx\rvert$ |
| Flow training objective | Minimise the **negative log-likelihood** |

## [Lec 4 — Mathematical Preliminaries I: Linear Algebra](../notes/week-01/04-linear-algebra.md)

| Item | Exactly this |
|---|---|
| Dot product | $\mathbf{x}\cdot\mathbf{y} = \sum_i x_i y_i$ — a **scalar** |
| Geometric dot product | $\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$ |
| Cosine similarity | $\cos\theta = \dfrac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ |
| Magnitude | $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ |
| Multiplication rule | $[m\times n]\cdot[n\times p] = [m\times p]$; inner dims must match |
| Product entry | $c_{ij} = \sum_k a_{ik}b_{kj}$ = (row $i$ of $\mathbf{A}$) · (col $j$ of $\mathbf{B}$) |
| Transpose of a product | $(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$ — **order reverses** |
| $2\times2$ determinant | $\det = ad - bc$ |
| $2\times2$ inverse | $\mathbf{A}^{-1} = \frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ |
| General inverse | $\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})}\mathrm{adj}(\mathbf{A})$, $\mathrm{adj} = $ cofactor matrix **transposed** |
| Singular | $\det = 0 \iff$ rank $< n \iff$ no inverse |
| Rank | max number of linearly independent rows (= columns) |
| Eigen definition | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{v} \neq \mathbf{0}$ |
| Characteristic equation | $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ |
| Trace identity | $\sum_i \lambda_i = \mathrm{tr}(\mathbf{A})$ |
| Determinant identity | $\prod_i \lambda_i = \det(\mathbf{A})$ |

## [Lec 5 — Mathematical Preliminaries II: Basic Probability I](../notes/week-01/05-probability-1.md)

| Item | Exactly this |
|---|---|
| Classical probability | $P(E) = \dfrac{\text{favourable outcomes}}{\text{total outcomes}}$, **equally likely only** |
| Frequentist probability | $P(A) = \#(A)/\#(\Omega)$ |
| Range | $0 \le P(E) \le 1$; $P(S)=1$; $P(\emptyset)=0$ |
| Complement | $P(\bar{A}) = 1 - P(A)$ |
| Addition rule | $P(A\cup B) = P(A)+P(B)-P(A\cap B)$ |
| Mutually exclusive | $P(A\cap B)=0 \Rightarrow P(A\cup B)=P(A)+P(B)$ |
| Conditional probability | $P(B\mid A) = \dfrac{P(A\cap B)}{P(A)}$, $P(A)\neq 0$ |
| Multiplication / product theorem | $P(A\cap B)=P(A)P(B\mid A)=P(B)P(A\mid B)$ |
| Independence | $P(A\cap B)=P(A)P(B)$, equivalently $P(B\mid A)=P(B)$ |
| $n$-event independence | $P(A_1\cap\cdots\cap A_n)=\prod_i P(A_i)$ |
| Law of total probability | $P(A)=\sum_{i} P(B_i)P(A\mid B_i)$, $B_i$ exhaustive **and** mutually exclusive |
| Bayes' theorem | $P(B_i\mid A)=\dfrac{P(B_i)P(A\mid B_i)}{\sum_j P(B_j)P(A\mid B_j)}$ |
| Bayes vocabulary | posterior $=\dfrac{\text{likelihood}\times\text{prior}}{\text{evidence}}$ |
| Random variable | a **function** from sample space to $\mathbb{R}$ |
| PMF validity | $p_i\ge 0$ and $\sum_i p_i = 1$ |
| PDF validity | $f(x)\ge 0$ and $\int_{R_x} f(x)dx = 1$ |
| Interval probability | $P(a\le X\le b)=\int_a^b f(x)dx$ |
| CDF | $F(x)=P(X\le x)$; $=\sum_{x_j\le x}p_j$ or $\int_{-\infty}^x f(t)dt$ |
| CDF ↔ PDF | $f(x)=dF/dx$; $P(a\le X\le b)=F(b)-F(a)$ |
| Uniform pdf | $f(x)=\dfrac{1}{b-a}$ on $[a,b]$ |
| Gaussian pdf | $f(x)=\dfrac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ |
| Expectation | $\mathbb{E}[X]=\sum_i x_ip_i$ or $\int x f(x)dx$ |
| Variance | $\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]=\mathbb{E}[X^2]-\mu^2$ |

## [Lec 6 — Mathematical Preliminaries II: Basic Probability II](../notes/week-01/06-probability-2.md)

| Item | Exactly this |
|---|---|
| Joint, general | $P(A,B) = P(A)P(B\mid A) = P(B)P(A\mid B)$ |
| Joint, independent | $P(A,B) = P(A)P(B)$ — **only** under independence |
| Marginal (discrete) | $P(X{=}x) = \sum_y P(X{=}x, Y{=}y)$ |
| Marginal (continuous) | $f_X(x) = \int f_{X,Y}(x,y)\,dy$ |
| Chain rule | $P(X_1,\dots,X_N) = \prod_{i=1}^{N} P(X_i \mid X_1,\dots,X_{i-1})$ |
| Linearity of expectation | $\mathbb{E}[aX+bY+c] = a\mathbb{E}[X]+b\mathbb{E}[Y]+c$, always |
| Variance identity | $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ |
| Affine variance | $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$ |
| Covariance | $\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)] = \mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$ |
| Sample covariance | $\sum_i (x_i-\bar{x})(y_i-\bar{y}) / (n-1)$ |
| Correlation | $\rho_{XY} = \mathrm{Cov}(X,Y)/(\sigma_X\sigma_Y) \in [-1,+1]$ |
| Covariance matrix | $\Sigma_{ij} = \mathrm{Cov}(X_i,X_j)$; **symmetric**; eigenvectors = principal directions |
| Likelihood | $L(\theta) = P(X\mid\theta)$ — a function of $\theta$, not a distribution over $\theta$ |
| Log-likelihood | $\ell(\theta) = \sum_{i=1}^{n}\log P(x_i\mid\theta)$ |
| MLE | $\hat\theta = \arg\max_\theta \ell(\theta)$ |
| Bayes (hypothesis form) | $P(h\mid\mathcal{D}) = P(\mathcal{D}\mid h)P(h)/P(\mathcal{D})$ |
| MAP | $h_{\mathrm{MAP}} = \arg\max_h P(\mathcal{D}\mid h)P(h)$ |
| ML hypothesis | $h_{\mathrm{ML}} = \arg\max_h P(\mathcal{D}\mid h)$ (equal priors) |
| Bayes optimal | $\arg\max_{v_j\in V}\sum_{h_i\in H} P(v_j\mid h_i)P(h_i\mid\mathcal{D})$ |
| KL divergence | $D_{\mathrm{KL}}(q\,\|\,p) = \sum_x q(x)\log\frac{q(x)}{p(x)} \ge 0$, not symmetric |

## [Lec 7 — Neural Network Fundamentals: the Perceptron](../notes/week-02/07-perceptron.md)

| Item | Exactly this |
|---|---|
| Biological ↔ artificial map | dendrite→input, synapse→weight, soma→summation, axon→output |
| Three neuron operations | multiplication, summation, activation (in that order) |
| Weighted sum | $z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top\mathbf{x} + b$ |
| Bias-as-input form | $z = \sum_{i=0}^{n} w_i x_i$ with $x_0 = 1$, $w_0 = b$ |
| Step activation (deck, slide 8) | $y = 1$ if $z > 0$; $y = 0$ if $z \le 0$ |
| Step activation (deck, slide 9) | $f(x) = 0$ if $x < 0$; $1$ if $x \ge 0$ — other name: **Heaviside** / threshold |
| Deck's error | $\mathcal{L} = \tfrac12(A - d)^2$, $A$ = predicted, $d$ = desired (target) |
| **Perceptron learning rule** | $w_i \leftarrow w_i + \eta(d - y)x_i$, $\;b \leftarrow b + \eta(d-y)$ |
| Threshold reading of the bias | fires when $\mathbf{w}^\top\mathbf{x} > -b$; threshold is $-b$ |
| Decision boundary | $\mathbf{w}^\top\mathbf{x} + b = 0$ — a line in 2-D, hyperplane in $n$-D |
| Convergence theorem | linearly separable $\Rightarrow$ converges in finite updates; otherwise loops forever |
| Linear separability | classes separable by one straight line / hyperplane |

## [Lec 8 — Multi-Layer Perceptron and Activation Functions](../notes/week-02/08-mlp-and-activations.md)

| Item | Exactly this |
|---|---|
| Pre-activation | $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ |
| Activation | $\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$, applied element-wise, $\mathbf{a}^{(0)} = \mathbf{x}$ |
| Weight shape | $\mathbf{W}^{(l)}$ is $n_l \times n_{l-1}$ (rows = this layer) |
| Params in layer $l$ | $n_l n_{l-1} + n_l$ |
| Linear collapse | $\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})+\mathbf{b}^{(2)} = \tilde{\mathbf{W}}\mathbf{x}+\tilde{\mathbf{b}}$ |
| Sigmoid | $\sigma(z) = \dfrac{1}{1+e^{-z}}$, range $(0,1)$ |
| tanh | $\tanh(z) = \dfrac{e^z-e^{-z}}{e^z+e^{-z}} = 2\sigma(2z)-1$, range $(-1,1)$ |
| ReLU | $\max(0,z)$, range $[0,\infty)$ |
| Leaky ReLU | $\max(0.1z, z)$ (deck's slope) |
| Softmax | $\sigma(\mathbf{z})_i = \dfrac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$ |
| Softmax shift invariance | $\sigma(\mathbf{z} + c\mathbf{1}) = \sigma(\mathbf{z})$; use $c = -\max_i z_i$ |
| Softmax Jacobian | $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$ |
| Cross-entropy | $\mathcal{L} = -\sum_i y_i\log p_i = -\log p_c$ |
| Softmax + CE gradient | $\partial\mathcal{L}/\partial z_i = p_i - y_i$ |
| Universal approximation | one hidden layer, enough units, non-linear $f$ ⟹ approximates any continuous function on a compact set |

## [Lec 9 — Backpropagation](../notes/week-02/09-backpropagation.md)

| Item | Exactly this |
|---|---|
| Forward (one layer) | $z_i^{(p)} = \sum_e w_{ei}^{(p)}a_e^{(p-1)} + b_i^{(p)}$, $\ a_i^{(p)} = \sigma(z_i^{(p)})$ |
| Deck's weight index | $w_{ei}^{(p)}$ = **from** $e$ in layer $p-1$ **to** $i$ in layer $p$ |
| Loss | $\mathcal{L} = \tfrac12\sum_i(a_i^{(k)} - d_i)^2$ |
| Loss derivative | $\partial\mathcal{L}/\partial a_i = a_i - d_i$ (prediction − target) |
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ |
| Sigmoid derivative | $\sigma'(z) = \sigma(z)(1-\sigma(z)) = a(1-a)$ |
| Output delta | $\delta_i^{(k)} = (a_i^{(k)} - d_i)\,a_i^{(k)}(1-a_i^{(k)})$ |
| Hidden delta | $\delta_e^{(k)} = a_e^{(k)}(1-a_e^{(k)})\sum_i \delta_i^{(k+1)}w_{ei}^{(k+1)}$ |
| Weight gradient | $\partial\mathcal{L}/\partial w_{ei}^{(k)} = \delta_i^{(k)}\,a_e^{(k-1)}$ |
| Bias gradient | $\partial\mathcal{L}/\partial b_i^{(k)} = \delta_i^{(k)}$ |
| Update | $w_{ei}^{(k)}(t+1) = w_{ei}^{(k)}(t) - \eta\,\delta_i^{(k)}a_e^{(k-1)}$ |
| Chain rule | multiply **along** a path, add **across** paths |
| Parameter count | $W = \sum_{l=1}^{L} n_{l-1}n_l$ (+ $\sum_l n_l$ biases) |

## [Lec 10 — Overfitting, Bias–Variance, and Regularization](../notes/week-02/10-overfitting-and-regularization.md)

| Item | Exactly this |
|---|---|
| Overfitting (formal) | $\exists h'$: $\text{Error}_T(h) < \text{Error}_T(h')$ but $\text{Error}_\mathcal{D}(h) > \text{Error}_\mathcal{D}(h')$ |
| Bias | Error from overly simplistic assumptions → **underfitting**, high train *and* test error |
| Variance | Error from sensitivity to the training set → **overfitting**, low train error, high test error |
| Error decomposition | $\text{Total} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$ |
| Complexity rule | As complexity ↑: bias ↓, variance ↑ |
| Regularized loss | $\mathcal{L} = \mathcal{L}_{\text{original}} + \lambda\times\text{Penalty}$ |
| L1 (Lasso) | $\mathcal{L}_0 + \lambda\sum_{j}\lvert\phi_j\rvert$ — weights go to **exactly zero**, sparse, feature selection |
| L2 (Ridge) | $\mathcal{L}_0 + \lambda\sum_{j}\phi_j^2$ — weights shrink, **never exactly zero**, no feature selection |
| Weight decay update | $w \leftarrow (1-\eta\lambda)w - \eta\,\partial\mathcal{L}_0/\partial w$ |
| λ direction | λ ↑ → simpler model, bias ↑, variance ↓. λ ↓ → more flexible, bias ↓, variance ↑ |
| k-fold average | $\bar{s} = \frac{1}{k}\sum_{i=1}^{k}s_i$; round $i$ trains on $S - S_i$, validates on $S_i$ |
| LOOCV | $k = n$, one sample per fold |
| Early stopping | Halt at the epoch where **validation** loss stops falling and starts rising |

## [Lec 11 — Convolutional Neural Networks I: Basics](../notes/week-03/11-cnn-basics.md)

| Item | Exactly this |
|---|---|
| Output size (conv **and** pool) | $H_{\text{out}} = \left\lfloor \dfrac{H - K + 2P}{S} \right\rfloor + 1$ |
| Conv parameter count | $(K \times K \times C_{\text{in}} + 1)\times F$ |
| Output depth | $C_{\text{out}} = F$, the number of filters |
| Kernel depth | always $C_{\text{in}}$ — a kernel spans the full input depth |
| Weights in one $3\times3$ kernel on 3 channels | $3\times3\times3 = 27$, plus 1 bias $= 28$ |
| `same` padding (odd $K$, $S=1$) | $P = \dfrac{K-1}{2}$ |
| `valid` padding | $P = 0$ |
| Pooling output depth | $C_{\text{out}} = C_{\text{in}}$ — pooling never mixes channels |
| Pooling parameters | **0** |
| Receptive field growth ($S=1$) | $R^{(l)} = R^{(l-1)} + (K^{(l)} - 1)$ |
| What DL calls convolution | cross-correlation — no kernel flip |
| Three CNN principles | local receptive fields, parameter sharing, translation equivariance |
| LCN | $I^{k+1}(x,y) = \dfrac{I^k(x,y) - m^k(N(x,y))}{\sigma^k(N(x,y))}$ |
| Pipeline | CONV → ReLU → NORM → POOL (repeat) → FLATTEN → FC → SOFTMAX |

## [Lec 12 — CNN Model Architectures: LeNet → GoogLeNet](../notes/week-03/12-cnn-architectures.md)

| Item | Exactly this |
|---|---|
| Receptive field growth (stride 1) | $r_l = r_{l-1} + (K_l - 1)$, $r_0 = 1$; $n$ identical layers → $n(K-1)+1$ |
| Three $3\times3$ ≡ one $7\times7$ | same effective receptive field, **three ReLUs instead of one** |
| VGG parameter argument | $3(3^2C^2) = 27C^2$ vs $7^2C^2 = 49C^2$ → ~45% fewer |
| $1\times1$ convolution | linear combination **across channels** at each spatial position; changes depth, **never** $H\times W$ |
| Bottleneck layer | a $1\times1$ conv placed before an expensive conv to cut $C_{\text{in}}$ |
| Conv multiply count | ops $= H \cdot W \cdot F \cdot K^2 \cdot C_{\text{in}}$ |
| Conv parameter count | $K^2 \cdot C_{\text{in}} \cdot F$ weights $+\,F$ biases |
| Inception module | $1\times1$, $3\times3$, $5\times5$, $3\times3$ max-pool **in parallel**, concatenated **depth-wise** |
| Depth after concatenation | sum of all four branch depths; pooling branch keeps $C_{\text{in}}$ |
| Global average pooling | average each channel over all $H\times W$ → one number per channel; replaces the FC head |
| Auxiliary classifiers | extra softmax heads at intermediate layers, **training only**, for gradient flow |

## [Lec 13 — CNN Optimization: Vanishing Gradients and Activation Functions](../notes/week-03/13-vanishing-gradients-activations.md)

| Item | Exactly this |
|---|---|
| Backward recursion | $\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)\top}\boldsymbol{\delta}^{(l+1)}\right)\odot f'(\mathbf{z}^{(l)})$ |
| Gradient as a product | $\boldsymbol{\delta}^{(l)} = \left[\prod_{k=l}^{L-1}\mathbf{D}^{(k)}\mathbf{W}^{(k+1)\top}\right]\boldsymbol{\delta}^{(L)}$, $\mathbf{D}^{(k)}=\mathrm{diag}(f'(\mathbf{z}^{(k)}))$ |
| Decay law | $\left\lvert \partial\mathcal{L}/\partial w^{(1)}\right\rvert  \sim \gamma^{L-1}$; $\gamma<1$ vanishes, $\gamma>1$ explodes |
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$, range $(0,1)$ |
| Sigmoid derivative | $\sigma'(z) = \sigma(z)[1-\sigma(z)]$, **max $=0.25$ at $z=0$** |
| tanh derivative | $1-\tanh^2 z = \mathrm{sech}^2 z$, max $=1$ at $z=0$ |
| tanh ↔ sigmoid | $\tanh(z) = 2\sigma(2z)-1$ |
| ReLU | $f(z)=\max(0,z)$; $f'=1$ for $z>0$, $f'=0$ for $z<0$ |
| Leaky ReLU | $f(z)=\max(\alpha z, z)$; $f'=1$ for $z>0$, $f'=\alpha$ for $z<0$; **$\alpha$ is any small positive constant** |
| ELU | $f(z)=z$ for $z>0$, $\alpha(e^{z}-1)$ for $z\le0$ |
| Dying ReLU | input always negative $\Rightarrow$ output 0 **and** gradient 0 $\Rightarrow$ weights frozen permanently |
| Clip by norm | if $\|\mathbf{g}\|>\tau$ then $\mathbf{g}\leftarrow\tau\mathbf{g}/\|\mathbf{g}\|$ — direction preserved |
| Degradation | deeper plain net has **higher training error** than shallower — an optimisation failure, not overfitting |
| Identity-mapping argument | a deeper net can emulate a shallower one with identity layers, so its optimum is no worse; it still trains worse |

## [Lec 14 — CNN with Residual Connections (ResNet)](../notes/week-04/14-resnet.md)

| Item | Exactly this |
|---|---|
| Residual mapping | $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$ |
| Residual block output | $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$ |
| Gradient through a block | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{x}} = \dfrac{\partial\mathcal{L}}{\partial\mathbf{y}}\left(\dfrac{\partial F}{\partial\mathbf{x}} + 1\right)$ |
| Gradient through $N$ blocks | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{x}^{(L)}}\prod_{i}\left(1 + \dfrac{\partial F_i}{\partial\mathbf{x}^{(i)}}\right)$ |
| Why the $+1$ matters | Even if $\partial F/\partial\mathbf{x}\to 0$, the factor $\to 1$, so the gradient has a direct unobstructed path and does **not** vanish |
| Identity mapping | $\mathbf{a}^{(l+2)} = g(\mathbf{W}^{(l+2)}\mathbf{a}^{(l+1)} + \mathbf{b}^{(l+2)} + \mathbf{a}^{(l)})$; with $\mathbf{W}^{(l+2)}=\mathbf{0},\ \mathbf{b}^{(l+2)}=\mathbf{0}$ this is $g(\mathbf{a}^{(l)}) = \mathbf{a}^{(l)}$ |
| Why residual is easier | If optimal $H \approx$ identity, the block only needs $F \to \mathbf{0}$ — far easier than fitting an identity through non-linear layers |
| Degradation problem | Deeper plain nets have higher **training** error — an optimization failure, **not** overfitting |
| Basic block | $3\times3$, $3\times3$ — ResNet-18/34 |
| Bottleneck block | $1\times1 \to 3\times3 \to 1\times1$ — ResNet-50/101/152 |
| Projection shortcut | $\mathbf{y} = F(\mathbf{x}) + \mathbf{W}_s\mathbf{x}$; a $1\times1$ conv matching the block's stride and output channels, used **only** on dimension mismatch |
| Stem | $7\times7$ conv, 64 filters, stride 2 → BN → ReLU → $3\times3$ max-pool stride 2 |
| Head | global average pooling → one FC(1000) → softmax |
| Identity shortcut cost | **zero** trainable parameters |

## [Lec 15 — CNN Model Architectures II: DenseNet, MobileNet, EfficientNet](../notes/week-04/15-densenet-mobilenet-efficientnet.md)

| Item | Exactly this |
|---|---|
| DenseNet layer rule | $\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$ — **concatenation** |
| ResNet layer rule (contrast) | $\mathbf{x}_\ell = H_\ell(\mathbf{x}_{\ell-1}) + \mathbf{x}_{\ell-1}$ — **addition** |
| DenseNet connections | $\dfrac{L(L+1)}{2}$ for $L$ layers; plain CNN has $L$ |
| Growth rate | each layer emits exactly $k$ feature maps; layer $\ell$ input $= k_0 + k(\ell-1)$ |
| DenseNet $H_\ell$ | BN → ReLU → $3\times3$ Conv |
| Transition layer | BN → $1\times1$ Conv → $2\times2$ average pooling |
| Standard conv cost | $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$ |
| Depthwise cost | $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$ |
| Pointwise cost | $M \cdot N \cdot D_F \cdot D_F$ |
| Cost ratio | $\dfrac{1}{N} + \dfrac{1}{D_K^2}$ — independent of $D_F$ |
| Depthwise output channels | **equal to input channels**, always |
| Pointwise = | a $1\times1$ convolution |
| Compound scaling | $d = \alpha^\varphi$, $w = \beta^\varphi$, $r = \gamma^\varphi$ |
| Compound constraint | $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, with $\alpha,\beta,\gamma \ge 1$ |
| FLOP scaling | $\propto d \cdot w^2 \cdot r^2 = 2^\varphi$ |

## [Lec 16 — Sequential Modelling and Recurrent Neural Networks](../notes/week-04/16-sequence-modelling-and-rnn.md)

| Item | Exactly this |
|---|---|
| Sequence modelling | Learning **temporal dependencies** from **ordered** data (text, speech, time-series) |
| Sequential models named | **RNN, LSTM, GRU, Transformer** |
| RNN, one sentence | A network that maintains an **internal state** updated as each element of the sequence is processed |
| Abstract recurrence | $\mathbf{h}_t = f_{\mathbf{W}}(\mathbf{h}_{t-1}, \mathbf{x}_t)$ |
| State update | $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$ |
| Output | $\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$ |
| The structural fact | **The same parameters and function are used at every time step** |
| Parameter count | $d_h d_x + d_h^2 + d_h + d_y d_h + d_y$ — **independent of $T$** |
| Five patterns | one-to-one · one-to-many · many-to-one · many-to-many (same length) · many-to-many (different length / seq2seq) |
| Non-recurrent pattern | **one-to-one** |
| Unfolding | Rewriting the loop as a feed-forward chain of $T$ copies sharing one weight set |
| Reason RNNs beat MLPs on sequences | Memory (state), order sensitivity, **variable-length** input |

## [Lec 17 — Training RNNs: Backpropagation Through Time](../notes/week-05/17-bptt.md)

| Item | Exactly this |
|---|---|
| BPTT, in one line | Unfold the RNN over $T$ steps into a feed-forward graph; apply the chain rule backwards |
| Total loss | $\mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_t$ |
| **The defining equation** | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{W}} = \sum_{t=1}^{T}\dfrac{\partial\mathcal{L}_t}{\partial\mathbf{W}}$ — gradients are **accumulated** over time steps, then one update |
| Generic per-step form | $\dfrac{\partial\mathcal{L}_t}{\partial\mathbf{W}_{hh}} = \dfrac{\partial\mathcal{L}_t}{\partial\mathbf{h}_t}\displaystyle\sum_{k=1}^{t}\dfrac{\partial\mathbf{h}_t}{\partial\mathbf{h}_k}\dfrac{\partial\mathbf{h}_k}{\partial\mathbf{W}_{hh}}$ |
| One-step Jacobian | $\partial\mathbf{h}_j/\partial\mathbf{h}_{j-1} = \mathrm{diag}(1-\mathbf{h}_j^2)\,\mathbf{W}_{hh}$ |
| Gradient through time | $\dfrac{\partial\mathbf{h}_t}{\partial\mathbf{h}_k} = \prod_{j=k+1}^{t}\mathrm{diag}(\tanh'_j)\mathbf{W}_{hh}$ — a product of $(t-k)$ Jacobians |
| Vanish / explode criterion | $\gamma\lambda_{\max} < 1 \Rightarrow$ vanish; $> 1 \Rightarrow$ explode ($\lambda_{\max}$ = largest singular value / spectral radius of $\mathbf{W}_{hh}$) |
| Gradient clipping | if $\|\mathbf{g}\| > \theta$: $\mathbf{g} \leftarrow (\theta/\|\mathbf{g}\|)\mathbf{g}$ — direction preserved |
| Truncated BPTT | Forward over the whole sequence carrying $\mathbf{h}$ across chunks; backward **within one chunk only** |
| Teacher forcing | Feed the **true** previous token during training |
| Free-running / exposure bias | At test time feed the model's **own** output; it never trained on its own mistakes, so errors compound |
| Bidirectional RNN | Two chains (forward + backward), outputs concatenated; **needs the whole sequence up front** |

## [Lec 18 — Long Short-Term Memory (LSTM)](../notes/week-05/18-lstm.md)

| Item | Exactly this |
|---|---|
| Forget gate | $\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f)$ |
| Input gate | $\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i)$ |
| Candidate | $\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c)$ |
| **Cell state update** | $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$ |
| Output gate | $\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o)$ |
| Hidden state | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$ |
| Gradient along cell state | $\partial\mathbf{C}_t/\partial\mathbf{C}_{t-1} = \mathrm{diag}(\mathbf{f}_t)$ |
| Vanilla RNN comparison | $\partial\mathbf{h}_t/\partial\mathbf{h}_i = \prod_m \mathbf{W}_h^\top\mathrm{diag}(\tanh'(\cdot))$ |
| CEC | Constant Error Carousel — the near-lossless gradient loop through $\mathbf{C}_t$ |
| Parameter count | $4\big(n_h(n_h+n_x)+n_h\big)$ |
| $\odot$ | element-wise (Hadamard) product |

## [Lec 19 — GRU, Seq2Seq, and Attention](../notes/week-05/19-gru-seq2seq-attention.md)

| Item | Exactly this |
|---|---|
| Alignment / energy score | $e_{t,i} = f_{\text{att}}(\mathbf{s}_{t-1}, \mathbf{h}_i)$ — $f_{\text{att}}$ is a learned MLP |
| Additive (Bahdanau) form | $e_{t,i} = \mathbf{v}^\top\tanh(\mathbf{W}_s\mathbf{s}_{t-1} + \mathbf{W}_h\mathbf{h}_i)$ |
| Attention weights | $\alpha_{t,i} = \dfrac{\exp(e_{t,i})}{\sum_{k=1}^{T}\exp(e_{t,k})}$ |
| Normalisation constraint | $\sum_{i=1}^{T}\alpha_{t,i} = 1$, $\alpha_{t,i} > 0$ — summed over **encoder** positions |
| Context vector | $\mathbf{c}_t = \sum_{i=1}^{T}\alpha_{t,i}\mathbf{h}_i$ — weighted sum of **all** encoder states |
| Decoder update | $\mathbf{s}_t = \text{decoder}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}_t)$ |
| Vanilla seq2seq context | $\mathbf{c} = \mathbf{h}_T$, the encoder's **final** hidden state, fixed for all $t$ |
| GRU reset gate | $\mathbf{r}_t = \sigma(\mathbf{W}_x^{r}\mathbf{x}_t + \mathbf{W}_h^{r}\mathbf{h}_{t-1})$ |
| GRU update gate | $\mathbf{z}_t = \sigma(\mathbf{W}_x^{u}\mathbf{x}_t + \mathbf{W}_h^{u}\mathbf{h}_{t-1})$ |
| GRU candidate | $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_x\mathbf{x}_t + \mathbf{r}_t\odot\mathbf{W}_h\mathbf{h}_{t-1})$ |
| GRU state (deck) | $\mathbf{h}_t = \mathbf{z}_t\odot\mathbf{h}_{t-1} + (1-\mathbf{z}_t)\odot\tilde{\mathbf{h}}_t$ |
| GRU state (Cho 2014) | $\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t$ |
| Cell parameter count | blocks $\times\,(nm + n^2 + n)$; RNN 1, GRU 3, LSTM 4 blocks |
| The bottleneck, in words | the entire source is compressed into **one fixed-length context vector**; loss grows with sequence length |

## [Lec 20 — Generative AI for Vision Tasks I](../notes/week-05/20-genai-vision-tasks-1.md)

| Item | Exactly this |
|---|---|
| CV challenges (7) | viewpoint variation · scale variation · **deformation** · occlusion · illumination changes · background clutter · intra-class variation |
| Viewpoint variation | An object looks very different from different **angles or positions**; alters shape, size, visible features, perspective |
| Scale variation | Same object at different **sizes**, depending on **distance from the camera** |
| Deformation | Shape/appearance changes when the object **bends, stretches, moves, changes posture** |
| Occlusion | One object **partially or completely blocks** another from view |
| Background clutter | Background contains many objects/patterns/**textures** that hide the main object |
| Intra-class variation | Objects of the **same category** have very different appearances |
| Image classification | Assigns an input image to **one of several predefined categories**; classification layer + **softmax** |
| Object detection | Predicts **bounding boxes + class labels + confidence scores**; = localization **+** classification |
| Detector names (deck) | **YOLO, Faster R-CNN, SSD** |
| **IoU** | $\text{area}(\cap)/\text{area}(\cup)$ of predicted and ground-truth boxes; TP if $\ge$ threshold (usually 0.5) |
| **mAP** | Mean over classes of AP; AP = area under that class's precision–recall curve |
| Precision / Recall / $F_1$ | $TP/(TP{+}FP)$ · $TP/(TP{+}FN)$ · $2PR/(P{+}R)$ |
| **Semantic segmentation** | Class label to **every pixel**, grouping all pixels of a category **without distinguishing individual objects** |
| **Instance segmentation** | Identifies and **separates each individual object** within the same class |
| Segmentation models (deck) | **FCN, U-Net, DeepLab, Mask R-CNN** |
| **mIoU** | Mean over classes of $TP/(TP+FP+FN)$ computed on pixels |
| Super-resolution | Reconstructs a **high-resolution** image from a **low-resolution** input by recovering lost spatial detail |
| SR models (deck) | **SRCNN, EDSR, ESRGAN, SwinIR**; Real-ESRGAN, latent/Stable-Diffusion SR |
| **PSNR** | $10\log_{10}(L^2/\text{MSE})$, $L=255$ for 8-bit; in **dB**, higher is better |
| **SSIM** | Structural similarity — luminance, contrast, structure; range $[-1,1]$, 1 = identical |
| Image inpainting | Reconstructs **missing/damaged/unwanted** regions with content that blends with surrounding pixels; preserves **structure, texture, semantics** |
| Traditional inpainting | **Diffusion- and patch-based**; good for **small** holes, fails on complex scenes |
| GenAI's four contributions | synthetic training data · data augmentation · representation/feature learning · direct generation of the output |
| Prompt-driven segmentation | **SAM (Segment Anything Model)** — zero-shot and few-shot |
| "Image restoration" | The deck's umbrella term covering **super-resolution, denoising, colorization, inpainting** |

## [Lec 21 — Generative AI for Vision Tasks II](../notes/week-05/21-genai-vision-tasks-2.md)

| Item | Exactly this |
|---|---|
| Image-to-image translation | Transforms an image from one visual domain to another **while preserving its underlying semantic content and structural information** |
| I2I vs text-to-image | I2I **conditions on an existing image**; text-to-image conditions on a prompt |
| **Pix2Pix** | **Paired** I2I; conditional GAN; adversarial loss **+ $\ell_1$ to the ground-truth pair** |
| **CycleGAN** | **Unpaired** I2I; **two** generators $G,F$ + **two** discriminators; adversarial **+ cycle-consistency** |
| Cycle-consistency loss | $\mathbb{E}_{\mathbf{x}}\|F(G(\mathbf{x}))-\mathbf{x}\|_1 + \mathbb{E}_{\mathbf{y}}\|G(F(\mathbf{y}))-\mathbf{y}\|_1$ — **both directions** |
| Medical synthesis motives | limited annotated data · patient privacy · class imbalance · cross-modality (MRI↔CT) |
| Anomaly detection mechanism | Train on **normal data only**; at test time **reconstruction error = anomaly score**; threshold it |
| Classical anomaly baselines | Isolation Forest, One-Class SVM, Local Outlier Factor |
| Anomaly difficulties | rarity · class imbalance · high dimensionality · evolving distributions · **no labelled anomalies** |
| NeRF | A network mapping (3-D point, view direction) → (colour, density); the scene *is* the weights |
| Latent attribute edit | $\mathbf{z}' = \mathbf{z} + \alpha\hat{\mathbf{d}}$ along a learned attribute direction |
| Style transfer separation | Deep features = **content**; feature-channel correlations (colour, texture, brushstrokes) = **style** |
| Gatys method | Conv weights **frozen**; the **synthesized image's pixels** are the optimised variables |
| Why video is harder | Must model spatial appearance **and** temporal dynamics across frames — coherence + cost |
| Classical video methods | optical flow, frame interpolation, RNNs |
| Responsible-deployment trio | watermarking · provenance tracking · synthetic media detection |

## [Lec 22 — Generative Modelling: Taxonomy and Maximum Likelihood](../notes/week-06/22-generative-taxonomy-and-mle.md)

| Item | Exactly this |
|---|---|
| The modelling goal | $p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$; $p_{\text{data}}$ is **unknown** and only sampled |
| Dataset | $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$, drawn i.i.d. from $p_{\text{data}}$ |
| MLE objective (product) | $\theta^{*} = \arg\max_\theta \prod_{i=1}^{N} p_\theta(\mathbf{x}_i, y_i)$ |
| MLE objective (log) | $\theta^{*} = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i, y_i)$ |
| Factorised form | $\theta^{*} = \arg\max_\theta \sum_i [\log p_\theta(y_i) + \log p_\theta(\mathbf{x}_i \mid y_i)]$ |
| Generative decomposition | $p_\theta(\mathbf{x}, y) = p_\theta(y)\,p_\theta(\mathbf{x}\mid y)$ = class prior × class-conditional density |
| NLL loss (slide 17) | $\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(\mathbf{x}_i)$ |
| MLE $\equiv$ KL | maximising likelihood $\equiv$ minimising $D_{\mathrm{KL}}(p_{\text{data}}\,\|\,p_\theta)$ ([Lec 6](../notes/week-01/06-probability-2.md)) |
| Autoregressive factorisation | $p(\mathbf{x}) = \prod_{i=1}^{d} p(x_i \mid x_{<i})$ — an **exact identity**, each conditional a neural net |
| Explicit density | model **defines and can evaluate** $p_\theta(\mathbf{x})$ |
| Implicit density | model **never computes** $p_\theta(\mathbf{x})$; provides a sampling procedure only |
| Tractable vs approximate | exact likelihood vs a bound/approximation |
| Implicit GAN form | $\mathbf{x} = G(\mathbf{z})$ where $\mathbf{z}\sim p(\mathbf{z})$ |
| Score function | $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ — gradient w.r.t. the **input**, not the parameters |
| Gaussian MLE | $\hat\mu = \frac1N\sum x_i$, $\hat\sigma^2 = \frac1N\sum(x_i-\hat\mu)^2$ (divisor $N$) |
| Why it is "generative" | you can draw $y\sim p_\theta(y)$, $\mathbf{x}\sim p_\theta(\mathbf{x}\mid y)$, $\mathbf{x}\sim p_\theta(\mathbf{x})$ — the model *is* the data-generating process |

## [Lec 23 — Autoregressive Generative Models: FVBN, PixelRNN, PixelCNN](../notes/week-06/23-autoregressive-pixelrnn-pixelcnn.md)

| Item | Exactly this |
|---|---|
| Chain-rule factorisation | $p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i \mid x_1,\dots,x_{i-1})$ |
| Same, deck's $n\times n$ image form | $p(\mathbf{x}) = \prod_{i=1}^{n^2} p(x_i \mid x_{<i})$ — upper limit is $n^2$ |
| RGB sub-factorisation | $p(x_i\mid x_{<i}) = p(x_{i,R}\mid x_{<i})\,p(x_{i,G}\mid x_{<i},x_{i,R})\,p(x_{i,B}\mid x_{<i},x_{i,R},x_{i,G})$ |
| FVBN definition | autoregressive generative model, **all variables visible** (no latents), joint factorised into a sequence of conditionals |
| FVBN training objective | maximise $\sum_{\mathbf{x}\in\mathcal{D}}\sum_i \log p_\theta(x_i\mid x_{<i})$ — plain MLE |
| Where it sits in the taxonomy | explicit density → **tractable** density |
| Masked convolution | ordinary conv whose kernel is multiplied by a fixed binary mask zeroing all weights pointing **below or to the right** of the centre |
| Why mask | to avoid seeing the future context — otherwise the autoregressive property is violated and the model cheats |
| Mask A | centre **excluded**; **first layer only** |
| Mask B | centre **included**; **all later layers** |
| Surviving weights, $k\times k$ | A: $(k^2-1)/2$ · B: $(k^2+1)/2$ |
| The asymmetry | training **parallel** (masking), generation **strictly sequential** |
| Generation cost | $n^2$ forward passes for an $n\times n$ image ($3n^2$ with channel factorisation) |
| Diagonal BiLSTM's purpose | capture the **entire available global context** |
| Row LSTM's weakness | roughly triangular context — **cannot** capture the entire context |

## [Lec 24 — Image Captioning and Spatial Attention (Transformers I)](../notes/week-06/24-captioning-and-spatial-attention.md)

| Item | Exactly this |
|---|---|
| Feature grid | CNN output $H \times W \times D$; $HW$ locations, each a $D$-dim **annotation vector** $\mathbf{z}_{i,j}$ |
| Alignment score | $e_{t,i,j} = f_{\text{att}}(\mathbf{h}_{t-1}, \mathbf{z}_{i,j})$, $f_{\text{att}}$ = an MLP; output is a **scalar** |
| Attention weights | $\alpha_{t,i,j} = \dfrac{\exp(e_{t,i,j})}{\sum_{i',j'}\exp(e_{t,i',j'})}$ — softmax over **all** $HW$ cells |
| Normalisation | $\sum_{i,j} \alpha_{t,i,j} = 1$ and $0 < \alpha_{t,i,j} < 1$ |
| Context vector | $\mathbf{c}_t = \sum_{i,j} \alpha_{t,i,j}\,\mathbf{z}_{i,j}$ — dimension $D$, same as one $\mathbf{z}_{i,j}$ |
| Output rule (deck) | $y_t = g_v(y_{t-1}, \mathbf{h}_{t-1}, \mathbf{c}_t)$ |
| Soft attention | weighted average over all locations; **differentiable**; plain backprop |
| Hard attention | sample one location from $\mathrm{Multinoulli}(\boldsymbol{\alpha}_t)$; **non-differentiable**; needs **REINFORCE** |
| Soft = expectation of hard | $\mathbb{E}[\mathbf{z}_{i^\star,j^\star}] = \sum_{i,j}\alpha_{t,i,j}\mathbf{z}_{i,j}$ |
| Baseline failure | one fixed $\mathbf{c}$ for all steps $\Rightarrow$ **information bottleneck** |
| Uniform-weight case | $\alpha_{t,i,j} = 1/(HW)$ everywhere $\Rightarrow$ $\mathbf{c}_t$ = global average pool |
| Paper | Xu et al., *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"*, **ICML 2015** |

## [Lec 25 — Transformers II: Q/K/V and Self-Attention](../notes/week-07/25-qkv-and-self-attention.md)

| Item | Exactly this |
|---|---|
| Scaled dot-product attention | $\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\dfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ |
| The three projections | $\mathbf{Q}=\mathbf{X}\mathbf{W}^Q$, $\mathbf{K}=\mathbf{X}\mathbf{W}^K$, $\mathbf{V}=\mathbf{X}\mathbf{W}^V$ |
| Role of each | Q = what I seek · K = what I advertise · V = what I hand over |
| Why $\sqrt{d_k}$ | $\mathrm{Var}(\mathbf{q}\!\cdot\!\mathbf{k}) = d_k$ for unit-variance components; dividing by $\sqrt{d_k}$ restores unit variance and stops softmax saturation |
| Shapes | $[M\times d_k][d_k\times N]\to[M\times N]$, then $[M\times N][N\times d_v]\to[M\times d_v]$ |
| Softmax axis | row-wise, over the **keys**; each row sums to 1 |
| Self-attention | $\mathbf{Q},\mathbf{K},\mathbf{V}$ all from the **same** sequence; score matrix is $N\times N$ |
| Cross-attention | $\mathbf{Q}$ from one sequence, $\mathbf{K},\mathbf{V}$ from another ([Lec 27](../notes/week-07/27-decoder-and-full-transformer.md)) |
| Causal mask | $e_{ij} \leftarrow -\infty$ for $j>i$, applied **before** softmax, since $e^{-\infty}=0$ |
| Multi-head | $\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\mathbf{W}^O$, $\ \mathrm{head}_i = \mathrm{Attention}(\mathbf{X}\mathbf{W}^Q_i,\mathbf{X}\mathbf{W}^K_i,\mathbf{X}\mathbf{W}^V_i)$ |
| Per-head width | $d_k = d_v = d_{\text{model}}/h$ |
| Permutation | self-attention is permutation-equivariant — **no notion of order** |
| Complexity | self-attention $\Theta(n^2 d)$ per layer, $O(1)$ sequential steps; recurrence $\Theta(n d^2)$, $O(n)$ sequential |

## [Lec 26 — Transformers III: The Encoder and Positional Encoding](../notes/week-07/26-encoder-and-positional-encoding.md)

| Item | Exactly this |
|---|---|
| Sinusoidal PE, even dims | $PE_{(pos,2i)} = \sin\!\big(pos / 10000^{2i/d_{\text{model}}}\big)$ |
| Sinusoidal PE, odd dims | $PE_{(pos,2i+1)} = \cos\!\big(pos / 10000^{2i/d_{\text{model}}}\big)$ |
| How PE enters | $\mathbf{x} = \mathbf{e}_{\text{token}} + \mathbf{p}_{\text{pos}}$ — **ADDED**, same width |
| Frequency | $\omega_i = 1/10000^{2i/d_{\text{model}}}$, $i = 0,\dots,\tfrac{d}{2}-1$ |
| Wavelength | $\lambda_i = 2\pi\cdot 10000^{2i/d_{\text{model}}}$; geometric from $2\pi$ to $\approx 10000\cdot 2\pi$ |
| Linear-shift property | $\mathbf{p}(pos+k) = \mathbf{M}_k\,\mathbf{p}(pos)$, $\mathbf{M}_k$ a rotation depending on $k$ alone |
| Encoder block order | MHSA → **Add & Norm** → position-wise FFN → **Add & Norm** |
| Residual form | $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$ |
| Encoder's job | input sequence → **contextualised representations**, same length |
| Permutation property | self-attention alone is permutation-equivariant; PE is what breaks it |
| Input embedding | learnable $\mathbf{E}\in\mathbb{R}^{V\times d_{\text{model}}}$; row lookup = one-hot × $\mathbf{E}$ |
| Encoder as a map | $\mathbf{c} = T_w(\mathbf{z})$ (the deck's notation) |

## [Lec 27 — Transformers IV: The Decoder and the Full Architecture](../notes/week-07/27-decoder-and-full-transformer.md)

| Item | Exactly this |
|---|---|
| Residual (Add) | $\text{Output} = \mathbf{x} + \text{Sublayer}(\mathbf{x})$ |
| Add & Norm (post-norm, the paper's) | $\mathrm{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$ |
| Pre-norm (modern) | $\mathbf{x} + \text{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$ |
| Layer norm statistics | $\mu = \frac1d\sum_i a_i$, $\sigma^2 = \frac1d\sum_i (a_i-\mu)^2$ — **over features, one token** |
| Layer norm output | $y_i = \gamma_i \dfrac{a_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i$ |
| LayerNorm parameter count | $2d_{\text{model}}$ ($\gamma$ and $\beta$) |
| LN vs BN, one line | LN normalises across **features per token**; BN across the **batch per feature** |
| Position-wise FFN | $\mathrm{FFN}(\mathbf{x}) = \max(0,\ \mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2$ |
| Deck's FFN form | $\mathbf{W}_2\,\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$ (column convention, ReLU or GELU) |
| "Position-wise" | The **same** FFN weights applied **independently** to every token position |
| Masked attention | $\mathrm{softmax}\!\big(\tfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\big)\mathbf{V}$, $\mathbf{M}=-\infty$ on future positions |
| **Cross-attention** | $\mathbf{Q}$ from the **decoder**; $\mathbf{K},\mathbf{V}$ from the **encoder output** |
| Cross-attention score shape | $T \times S$ (target rows × source columns) — **rectangular** |
| Decoder block order | masked self-attn → A&N → cross-attn → A&N → FFN → A&N |
| Decoder output head | Linear $d_{\text{model}} \to \mid \mathcal{V}\mid $, then softmax |
| Teacher forcing | Feed the ground-truth previous tokens during training; one parallel pass |

## [Lec 28 — Vision Transformers: ViT, DETR, Swin](../notes/week-08/28-vit-detr-swin.md)

| Item | Exactly this |
|---|---|
| Number of patches | $N = \dfrac{HW}{P^2}$; for $224{\times}224$, $P{=}16$: $N = 14^2 = 196$ |
| Sequence length into the ViT encoder | $N + 1$ (patches **plus** [CLS]) = 197 |
| Flattened patch dimension | $P^2 C$; for $P{=}16, C{=}3$: $768$ |
| Patch projection | $\mathbf{z} = \mathbf{x}\mathbf{E} + \mathbf{b}$, $\mathbf{E}\in\mathbb{R}^{(P^2C)\times d_{\text{model}}}$, shared across patches |
| Patch embedding ≡ convolution | kernel $K = P$, stride $S = P$, padding 0, $F = d_{\text{model}}$ filters |
| ViT input | $\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1\mathbf{E};\cdots;\mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$ |
| ViT block (pre-norm) | $\mathbf{z}'_\ell = \mathrm{MSA}(\mathrm{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$; $\mathbf{z}_\ell = \mathrm{MLP}(\mathrm{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$ |
| ViT position embeddings | **learned 1-D**, added not concatenated; not sinusoidal |
| ViT architecture class | **encoder-only, discriminative** — not autoregressive, not generative |
| ViT's missing inductive biases | locality and translation equivariance |
| Attention complexity | $O(n^2 d)$ in tokens; $O((HW)^2)$ in image area for global, $O(M^2 HW)$ for windowed |
| DETR pipeline | CNN backbone → Transformer encoder → decoder with $N$ object queries → per-query FFN (class + box) |
| DETR matching | Hungarian algorithm, optimal **one-to-one** bipartite assignment, $O(N^3)$ |
| DETR set objective | $\hat{\sigma} = \arg\min_{\sigma}\sum_j \mathcal{L}_{\text{match}}(\mathbf{y}_j, \hat{\mathbf{y}}_{\sigma(j)})$ |
| DETR's headline claim | **no anchor boxes, no NMS** — fully end-to-end |
| Swin cost formulas | $\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$; $\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$ |
| W-MSA / SW-MSA | regular windows then windows shifted by $\lfloor M/2 \rfloor$, in **alternating** blocks |
| Patch merging | concat $2{\times}2$ tokens ($4C$) → linear to $2C$; resolution halves, width doubles |
| Swin's two wins | **linear** complexity in image size + **hierarchical** multi-scale features |

## [Lec 29 — From Autoencoders to the Variational Autoencoder](../notes/week-08/29-autoencoders-to-vae.md)

| Item | Exactly this |
|---|---|
| Autoencoder | encoder $\mathbf{z} = f_\phi(\mathbf{x})$, decoder $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$ |
| AE loss | reconstruction only: MSE or binary cross-entropy between $\mathbf{x}$ and $\hat{\mathbf{x}}$ |
| Undercomplete | $\dim(\mathbf{z}) < \dim(\mathbf{x})$ — the useful case |
| Overcomplete | $\dim(\mathbf{z}) \ge \dim(\mathbf{x})$ — identity map available, needs another constraint |
| Why AE is not generative | the latent space has no probabilistic structure, so there is nothing to sample from |
| VAE encoder | $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}^2(\mathbf{x})))$ |
| VAE decoder | $p_\theta(\mathbf{x}\mid\mathbf{z})$ — the likelihood |
| Prior | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ — fixed, not learned |
| Joint | $p_\theta(\mathbf{x},\mathbf{z}) = p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})$ |
| Marginal (the objective) | $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$ |
| MLE objective | $\theta^{*} = \arg\max_\theta \sum_{i=1}^{n}\log p_\theta(\mathbf{x}^{(i)})$ |
| True posterior | $p_\theta(\mathbf{z}\mid\mathbf{x}) = \dfrac{p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$ |
| Why the posterior is intractable | its denominator is the intractable marginal |
| The fix (this chapter's endpoint) | approximate it with a diagonal-Gaussian $q_\phi(\mathbf{z}\mid\mathbf{x})$ |
| Encoder head size | $2d$ outputs for latent dim $d$: $d$ for $\boldsymbol{\mu}$, $d$ for $\log\boldsymbol{\sigma}^2$ |
| Recovering variance | $\sigma_j^2 = \exp(\log\sigma_j^2) > 0$ for any real network output |
| Diagonal Gaussian density | $\prod_j \frac{1}{\sqrt{2\pi\sigma_j^2}}\exp\!\big(-\frac{(z_j-\mu_j)^2}{2\sigma_j^2}\big)$ |
| Taxonomy slot | explicit density, **approximate** (variational) |

## [Lec 30 — The ELBO and the Reparameterization Trick](../notes/week-08/30-elbo-and-reparameterization.md)

| Item | Exactly this |
|---|---|
| The evidence | $\log p_\theta(\mathbf{x}) = \log\int p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ — intractable |
| Jensen's inequality (concave) | $\log\mathbb{E}[Y] \ge \mathbb{E}[\log Y]$ |
| ELBO (compact form) | $\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right]$ |
| **The key identity** | $\log p_\theta(\mathbf{x}) = \mathcal{L}_{\mathrm{ELBO}} + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x}))$ |
| Consequence | $D_{\mathrm{KL}}\ge 0 \Rightarrow \mathcal{L}_{\mathrm{ELBO}} \le \log p_\theta(\mathbf{x})$; tight iff $q_\phi = p_\theta(\mathbf{z}\mid\mathbf{x})$ |
| ELBO (two-term form) | $\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ |
| VAE loss | $\mathcal{L}_{\mathrm{VAE}} = -\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}(q_\phi\,\|\,p(\mathbf{z}))$ — minimise |
| **Closed-form KL** | $D_{\mathrm{KL}} = -\dfrac{1}{2}\sum_{j=1}^{J}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$ |
| Prior | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Encoder | $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$ |
| **Reparameterization** | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Its gradients | $\partial z_j/\partial\mu_j = 1$, $\partial z_j/\partial\sigma_j = \epsilon_j$ |
| sigma from logvar | $\sigma = \exp(\tfrac12\log\sigma^2)$ |
| Recon term, Gaussian decoder | $-\log p_\theta(\mathbf{x}\mid\mathbf{z}) \propto \sum_i(x_i-\hat{x}_i)^2$ → MSE |
| Recon term, Bernoulli decoder | $-\log p_\theta(\mathbf{x}\mid\mathbf{z}) = -\sum_i[x_i\log\hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$ → BCE |

## [Supplementary — Topics the Slides Name But Never Teach](../notes/supplementary.md)

| Item | Exactly this |
|---|---|
| Batch norm | $\hat{x} = \dfrac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$, then $y = \gamma\hat{x} + \beta$ |
| BN learned parameters | $\gamma$ (scale) and $\beta$ (shift) — **two per feature**, both learned |
| BN at test time | uses **running averages** of mean/variance, not batch statistics |
| Internal covariate shift | the shifting distribution of a layer's inputs as earlier layers update |
| Layer norm | normalises across **features within one sample**; batch-size independent |
| Xavier variance | $2/(n_{\text{in}} + n_{\text{out}})$ — for **sigmoid/tanh** |
| He variance | $2/n_{\text{in}}$ — for **ReLU** |
| Zero initialisation | fatal — breaks symmetry breaking; all units stay identical |
| Dropout | zero each unit with probability $1-p_{\text{keep}}$ during training only |
| Inverted dropout | divide by $p_{\text{keep}}$ at **train** time; no test-time change |
| Momentum | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + (1-\beta)\nabla\mathcal{L}$, $\beta \approx 0.9$ |
| Adam | momentum + RMSProp, bias-corrected; $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| Mini-batch size | typically 32–256; "SGD" in practice means mini-batch |
