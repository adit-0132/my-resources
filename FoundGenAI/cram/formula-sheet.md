# Formula Sheet

Every **Must-memorise** entry from all chapters, in course order. Generated from the chapters by `_build/build_cram.py` — edit the chapters, not this file.

## [Lec 01 — Introduction to Generative AI](../notes/01-intro-generative-ai.md)

| Item | Exactly this |
|---|---|
| Discriminative model | $P(y\mid \mathbf{x})$ |
| Generative model | $P(\mathbf{x})$ **or** $P(\mathbf{x},y)$ |
| Discriminative models do not | mainly learn *how the data was generated* |
| Unsupervised generation | learn $p_\theta(\mathbf{x})$ from unlabelled $\mathbf{x}_1\ldots\mathbf{x}_n$ |
| After training | $p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$ |
| Generation step | $\mathbf{x}_{\text{new}} \sim p_\theta(\mathbf{x})$ |
| Joint modelling | $p_\theta(\mathbf{x},y)$ |
| Bayes inversion | $p(y\mid\mathbf{x}) = \dfrac{p(\mathbf{x}\mid y)\,p(y)}{p(\mathbf{x})}$ |
| Conditional generation | $p_\theta(\mathbf{x}\mid y)$ or $p_\theta(\mathbf{x}\mid c)$ |
| Continuous output → | Gaussian → linear (or sigmoid) activation + MSE |
| Binary output → | Bernoulli → sigmoid activation + binary cross-entropy |
| Token output → | Categorical → softmax activation + categorical cross-entropy |
| How a distribution is specified in code | implicitly, by **output activation + loss function** |
| Gaussian parameters | mean $\mu$ (centre), standard deviation $\sigma$ (spread); write $\mathcal{N}(\mu,\sigma^2)$ |
| Bernoulli | two outcomes, success (1) / failure (0), one parameter $p$ |
| Categorical | probability over $k$ distinct categories |
| Generative modelling aims to | learn structure and distribution so **new, realistic, diverse** samples can be generated |
| Course spine | AE → VAE → GAN → Diffusion → Sequence models → LLMs → RAG → Multimodal AI and Ethics |

## [Lec 02 — Activation and Loss Functions](../notes/02-activations-and-losses.md)

| Item | Exactly this |
|---|---|
| Neuron | $z=\sum_i x_i w_i + b$, then $a = f(z)$ |
| Why an activation | to introduce **non-linearity**; without it the network collapses to one linear map |
| ReLU | $f(z)=\max(0,z)$; $f'=1$ if $z>0$, else $0$ |
| Leaky ReLU | $f(z)=\max(0.01z,z)$; $f'=1$ if $z>0$, else $0.01$ |
| PReLU | $\max(az,z)$ with $a$ **learned** |
| ELU | $z$ if $z>0$, else $\alpha(e^{z}-1)$; $f' = 1$ or $\alpha e^{z}$ |
| GELU | $z\,\Phi(z)$; slide's form $0.5z\big(1+\tanh[\sqrt{2/\pi}(z+0.044715z^3)]\big)$ |
| SiLU / Swish | $z\,\sigma(z) = z/(1+e^{-z})$ |
| Sigmoid | $\sigma(z)=\dfrac{1}{1+e^{-z}}$; $\sigma'=\sigma(1-\sigma)$ |
| tanh | $\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$; $\tanh' = 1-\tanh^2$ |
| Softmax | $s(z_i)=\dfrac{e^{z_i}}{\sum_{j=1}^{k}e^{z_j}}$; $\partial s_i/\partial z_j = s_i(\delta_{ij}-s_j)$ |
| Average loss | $\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\ell(\hat y_i,y_i)$, $\hat y_i=f(\mathbf{x}_i;\theta)$ |
| MSE | $\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat x_i)^2$ |
| BCE | $-\frac{1}{N}\sum_{i=1}^{N}[x_i\log\hat x_i+(1-x_i)\log(1-\hat x_i)]$ |
| Categorical CE | $-\sum_{k=1}^{K}y_k\log(\hat y_k)$ |
| Sparse CE | $-\log(\hat y_c)$, $c$ an integer index |
| Sigmoid + BCE gradient | $\partial\ell/\partial z = \hat x - x$ |
| Softmax + CE gradient | $\partial\mathcal{L}/\partial z_i = \hat y_i - y_i$ |
| Transformers / LLMs use | **GELU** (and GeGLU) in hidden layers, **softmax** at the output |
| Diffusion models use | **SiLU / Swish** |
| GAN discriminator output | **sigmoid** |
| GAN generator / VAE decoder output | **sigmoid or tanh** |

## [Lec 03 — Optimizers, Part A](../notes/03-optimizers-a.md)

| Item | Exactly this |
|---|---|
| Gradient-descent update | $W_{\text{new}} = W_{\text{old}} - \eta\,\dfrac{\partial\mathcal{L}}{\partial W}$ |
| Why minus | gradient points **uphill**; you want to minimise $\mathcal{L}$ |
| Gradient's two roles | sign → direction; magnitude → amount of influence |
| Gradient on 1 point | **Stochastic** Gradient Descent |
| Gradient on $k$ points | **Mini-batch** Stochastic Gradient Descent |
| Gradient on all $n$ points | **(Batch) Gradient Descent** |
| Momentum velocity | $v_t = \beta v_{t-1} + (1-\beta)g_t$ |
| Momentum update | $W_{\text{new}} = W_{\text{old}} - \eta\,v_t$ |
| $v_t$ is an | Exponentially Weighted Moving Average (EWMA) of gradients |
| Momentum's attribution | Polyak, 1964 |
| EWMA memory length | $\dfrac{1}{1-\beta}$ steps |
| Weight on gradient $k$ steps old | $(1-\beta)\beta^k$ |
| Updates per epoch, mini-batch | $\lceil n/k \rceil$ |

## [Lec 04 — Optimizers, Part B](../notes/04-optimizers-b.md)

| Item | Exactly this |
|---|---|
| SGD update | $W_{\text{new}} = W_{\text{old}} - \eta g$ |
| What $\eta$ does | for the same gradient, sets **how much the weight actually changes** |
| AdaGrad accumulator | $\alpha_t = \sum_{i=1}^{t} g_i^2$ |
| AdaGrad adaptive rate | $\eta'_t = \dfrac{\eta}{\sqrt{\alpha_t + \varepsilon}}$ |
| AdaGrad update | $w_{\text{new}} = w_{\text{old}} - \dfrac{\eta}{\sqrt{\alpha_t+\varepsilon}}g_t$ |
| AdaGrad's flaw | $\alpha_t$ only grows $\Rightarrow$ $\eta'_t \to 0$, learning stops |
| RMSProp statistic | $s_t = \beta s_{t-1} + (1-\beta)g_t^2$ |
| RMSProp update | $w_{\text{new}} = w_{\text{old}} - \dfrac{\eta}{\sqrt{s_t}+\varepsilon}g_t$ |
| Adam in words | **Adam = Momentum + RMSProp**; *Adaptive Moment Estimation* |
| Adam first moment | $v_t = \beta_1 v_{t-1} + (1-\beta_1)g_t$ |
| Adam second moment | $s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2$ |
| Adam bias correction | $\hat v_t = \dfrac{v_t}{1-\beta_1^{\,t}}$, $\hat s_t = \dfrac{s_t}{1-\beta_2^{\,t}}$ |
| Adam update | $w_{\text{new}} = w_{\text{old}} - \eta\dfrac{\hat v_t}{\sqrt{\hat s_t}+\epsilon}$ |
| Adam defaults | $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| Adam's first corrected step | exactly $\eta$, for any $g_1$ |

## [Lec 05 — Convolutional Neural Network, Part A](../notes/05-cnn-a.md)

| Item | Exactly this |
|---|---|
| **Output size (the formula)** | $H_{\text{out}} = \left\lfloor \frac{H + 2P - K}{S}\right\rfloor + 1$, same for $W$ |
| Output depth, convolution | $C_{\text{out}} = F$ (number of filters) |
| Output depth, pooling | $C_{\text{out}} = C_{\text{in}}$ (unchanged) |
| No padding, stride 1 | $H_{\text{out}} = H - K + 1$ |
| Padding only, stride 1 | $H_{\text{out}} = H + 2P - K + 1$ |
| `same` padding, $S=1$, odd $K$ | $P = \dfrac{K-1}{2}$ |
| `valid` padding | $P = 0$ |
| Shrinkage per unpadded layer | $K - 1$ pixels per axis |
| Convolution operation | element-wise multiply the window by the filter, then **sum** |
| Filter depth | a filter is always as deep as its input: $K \times K \times C$ |
| Floor function | greatest integer $\le$ the number; $\lfloor 3.1\rfloor = \lfloor 3.9\rfloor = 3$ |
| Receptive field growth | $R_l = R_{l-1} + (K_l-1)\,j_{l-1}$, $j_l = j_{l-1}S_l$ |
| Three failures of flattening | destroys spatial relationships · pixels become independent features · local dependencies lost |

## [Lec 06 — Convolutional Neural Network, Part B](../notes/06-cnn-b.md)

| Item | Exactly this |
|---|---|
| **Conv layer parameters** | $(K_h \times K_w \times C_{\text{in}} + 1)\times F$ |
| Bias count | one per **filter**, hence the single $+1$ inside the bracket |
| Conv layer output | $H_{\text{out}}\times W_{\text{out}}\times F$ — see [Lec 05](../notes/05-cnn-a.md) for $H_{\text{out}}$ |
| Dense layer parameters | $n_{\text{in}}\times n_{\text{out}} + n_{\text{out}}$ |
| ReLU | $f(x)=\max(0,x)$; shape unchanged; **0 parameters** |
| Pooling | downsampling; **0 parameters**; depth unchanged |
| Pooling output size | same formula as convolution, with pool size as $K$ |
| Keras pooling stride default | `strides = pool_size` |
| Max-Avg-Min rule | $\text{Max}-\text{Min}$ if $\text{Max}-\text{Min} > \text{Avg}$, else $\frac{\text{Max}+\text{Avg}}{2}$ |
| Flatten | $H\times W\times C \to$ vector of length $HWC$; 0 parameters |
| Flatten's position | after the last pooling layer, before the first dense layer |
| Classification head | $k$ classes → $k$ softmax units + categorical cross-entropy; 2 classes → 1 sigmoid + BCE |
| What does **not** affect parameter count | stride, padding, input height and width |
| Layer order | conv → ReLU → pool (repeat) → flatten → dense → softmax |

## [Lec 10 — Introduction to Autoencoders](../notes/10-autoencoder-intro.md)

| Item | Exactly this |
|---|---|
| Autoencoder | a feed-forward neural network that learns to reconstruct its input |
| Learning paradigm | **unsupervised** (a special category of unsupervised deep learning) |
| The target | **the input itself** |
| Three components | 1. encoder, 2. latent code, 3. decoder |
| Two objectives | 1. **representation learning** (encoder), 2. **reconstruction learning** (decoder) |
| Encoder | maps $\mathbf{x}_i \to \mathbf{h}$, lower-dimensional |
| Decoder | maps $\mathbf{h} \to \hat{\mathbf{x}}_i$ |
| The pipeline | $\mathbf{x}_i \to \mathbf{h} \to \hat{\mathbf{x}}_i$ |
| Undercomplete | $\dim(\mathbf{h}) < \dim(\mathbf{x}_i)$ |
| Overcomplete | $\dim(\mathbf{h}) \geq \dim(\mathbf{x}_i)$ — note the $\geq$ |
| Loss-free encoding | an $\mathbf{h}$ from which $\hat{\mathbf{x}}_i$ is reconstructed **perfectly** |
| Overcomplete failure | learns a **trivial encoding**: copy $\mathbf{x}_i$ into $\mathbf{h}$, copy $\mathbf{h}$ into $\hat{\mathbf{x}}_i$ |
| Why the bottleneck | "forces the network to preserve only the most important information" |
| Output layer width | must equal the input dimension — not a free choice |
| Training objective | reconstruction loss — see [Lec 11](../notes/11-reconstruction-loss.md) |

## [Lec 11 — Reconstruction Loss (MSE, Binary Cross-Entropy)](../notes/11-reconstruction-loss.md)

| Item | Exactly this |
|---|---|
| Encoder | $\mathbf{H} = g(\mathbf{X}\mathbf{W}_e)$ |
| Decoder | $\hat{\mathbf{X}} = f(\mathbf{H}\mathbf{W}_d)$ |
| Tied weights | $\mathbf{W}_d = \mathbf{W}_e^\top$ |
| Per-sample MSE | $l_i = \lVert\mathbf{x}_i - \hat{\mathbf{x}}_i\rVert^2$ |
| Total MSE | $\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m} l_i$ |
| Reduced linear objective | $f(\mathbf{W}) = \min_\mathbf{W}\lVert\mathbf{X} - \mathbf{X}\mathbf{W}\mathbf{W}^\top\rVert^2 + \lambda\lVert\mathbf{W}\rVert^2$ |
| Per-sample BCE | $l_i = -\sum_{j=1}^{d}[x_{ij}\log\hat{x}_{ij} + (1-x_{ij})\log(1-\hat{x}_{ij})]$ |
| Total BCE | $\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{d}[\cdots]$ |
| Sigmoid | $\sigma(a) = \dfrac{1}{1+e^{-a}}$ |
| Binary data → | sigmoid decoder + BCE |
| Real data → | linear decoder + MSE |
| Trained model | $\mathbf{X}\mathbf{W}^* = \mathbf{H}^*$, then $\mathbf{H}^*\mathbf{W}^{*\top} = \hat{\mathbf{X}}$ |

## [Lec 12 — Types of Autoencoders](../notes/12-autoencoder-types.md)

| Item | Exactly this |
|---|---|
| Shallow autoencoder | **only one hidden layer** between input and output |
| Shallow architecture | $\mathbf{x}\in\mathbb{R}^n \to \mathbf{h}\in\mathbb{R}^m\,(m<n) \to \hat{\mathbf{x}}\in\mathbb{R}^n$ |
| Deep autoencoder | **multiple hidden layers**; learns **hierarchical and abstract** representations |
| Deep architecture | $\mathbf{x}\in\mathbb{R}^n \to \mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}\in\mathbb{R}^m \to \hat{\mathbf{x}}\in\mathbb{R}^n$ |
| Convolutional AE, encoder | convolutional layers **and pooling** to reduce spatial dimension |
| Convolutional AE, latent | the **output of the final convolutional layer** of the encoder |
| Convolutional AE, decoder | **transposed convolutions** (a.k.a. deconvolution) to upsample back to the original dimension |
| Upsampling output size | $\text{Output} = \text{Input} \times S$ |
| Upsampling parameters | **none — not learnable** |
| Nearest neighbour | each element **copied** into an $S\times S$ block |
| Bed of nails | each element written once, the other $S^2-1$ cells **zero** |
| Transposed conv output size | $(I + K - 1) \times (I + K - 1)$ (stride 1, no padding) |
| Transposed conv method | scale the kernel by each input cell, paste at that cell's offset, **sum the overlaps** |
| Checkerboard artifacts | grid-like patterns from transposed convolution; **systematic, not noise** |
| The fix | **upsampling + convolution** — spreads values uniformly, kernel applies evenly |
| Latent width rule | too small → poor reconstruction; larger → better; **beyond a point, diminishing returns** |
| DNN AE weakness | **flattens** the image, **fails to preserve spatial relationships** (edges, contours, shapes) |
| CNN AE strength | preserves local spatial structure; learns **edges, textures, shapes** |
| Output activation | sigmoid for $[0,1]$ data (BCE loss); linear for real-valued (MSE) — [Lec 11](../notes/11-reconstruction-loss.md) |

## [Lec 13 — Denoising Autoencoders](../notes/13-denoising-ae.md)

| Item | Exactly this |
|---|---|
| Corruption map | $\mathbf{x} \xrightarrow{\text{corruption}} \tilde{\mathbf{x}}$, random, unlearned, training-time only |
| Gaussian noise | $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$ |
| Masking noise | $\tilde{x}_{i,j} = 0$ w.p. $q$; $\ = x_{i,j}$ w.p. $1-q$ |
| Salt-and-pepper | chosen pixels replaced by the **extreme** values (0 or max) |
| Encoder (corrupted input) | $\mathbf{h} = g(\mathbf{W}_e\tilde{\mathbf{x}} + \mathbf{b})$ |
| Decoder | $\hat{\mathbf{x}} = f(\mathbf{W}_d\mathbf{h} + \mathbf{c})$ |
| Objective | $\min\ \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}})$ — **clean** $\mathbf{x}$ as target |
| Full pipeline | $\mathbf{x} \to \tilde{\mathbf{x}} \to \mathbf{h} \to \hat{\mathbf{x}}$, learns $\tilde{\mathbf{x}} \to \hat{\mathbf{x}} \approx \mathbf{x}$ |
| Why it regularises | $\tilde{\mathbf{x}} \neq \mathbf{x}$, so copying no longer minimises the loss |
| What is learned instead | $p(x_{i,j}\mid\mathcal{N}_{i,j})$ / $\hat{x}_{i,j}\approx F(\mathcal{N}_{i,j})$ |
| Loss choice | MSE for continuous data, BCE for binary or normalised ([Lec 11](../notes/11-reconstruction-loss.md)) |
| Neighbourhood | the 8 pixels around $(i,j)$, written $\mathcal{N}_{i,j}$ |
| Expected masked count | $\mathbb{E}[M] = dq$ for $d$ components |

## [Lec 14 — Sparse Autoencoders](../notes/14-sparse-ae.md)

| Item | Exactly this |
|---|---|
| Average activation | $\hat\rho_j = \dfrac{1}{m}\sum_{i=1}^{m} a_j^{(2)}(\mathbf{x}^{(i)})$ — **column mean** |
| Target | $\rho$ = desired average activation; the SAE enforces $\hat\rho_j = \rho$ |
| KL sparsity, per neuron | $D_{\mathrm{KL}}(\rho\,\|\,\hat\rho_j) = \rho\log\dfrac{\rho}{\hat\rho_j} + (1-\rho)\log\dfrac{1-\rho}{1-\hat\rho_j}$ |
| Total sparsity penalty | $\displaystyle\sum_{j=1}^{n} D_{\mathrm{KL}}(\rho\,\|\,\hat\rho_j)$ |
| Full objective | $\mathcal{L}_{\text{sparse}}(\mathbf{W},\mathbf{b}) = \mathcal{L}(\mathbf{x}_i,\hat{\mathbf{x}}_i) + \beta\sum_{j} D_{\mathrm{KL}}(\rho\,\|\,\hat\rho_j)$ |
| Generic form | $\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \beta\,\mathcal{L}_{\text{sparsity}}$ |
| Zero-penalty condition | $\hat\rho_j = \rho \Rightarrow D_{\mathrm{KL}} = 0$ (because $\log 1 = 0$) |
| Two terms | **active part** $\rho\log(\rho/\hat\rho_j)$ · **inactive part** $(1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}$ |
| What is penalised | the **activations** $\mathbf{h}$ — *not* the weights, *not* the input |
| Alternative penalty | L1 on activations (named by the deck, not developed) |
| Architecture allowed | **overcomplete**, $\dim(\mathbf{h}) > \dim(\mathbf{x})$ |
| $\beta$ | controls how much sparsity matters relative to reconstruction |
| Hidden activation required | sigmoid, so $\hat\rho_j\in(0,1)$ reads as a firing rate |

## [Lec 15 — Contractive Autoencoders](../notes/15-contractive-ae.md)

| Item | Exactly this |
|---|---|
| CAE goal | $\mathbf{h}(\mathbf{x}) \approx \mathbf{h}(\mathbf{x}+\boldsymbol{\delta})$ for small $\boldsymbol{\delta}$ |
| Regularisation term | $\Omega(\theta) = \lVert \mathbf{J}_\mathbf{x}(\mathbf{h}) \rVert_F^2$ |
| Full loss | $\mathcal{L}_{\text{CAE}} = \mathcal{L}_{\text{rec}}(\mathbf{x},\hat{\mathbf{x}}) + \lambda\lVert \mathbf{J}_\mathbf{x}(\mathbf{h}) \rVert_F^2$ |
| Squared Frobenius norm | $\displaystyle\lVert\mathbf{J}_\mathbf{x}(\mathbf{h})\rVert_F^2 = \sum_{j=1}^{n}\sum_{l=1}^{k}\left(\frac{\partial h_l}{\partial x_j}\right)^2$ |
| Jacobian shape | $k\times n$ — hidden dimension $\times$ input dimension |
| A Jacobian entry | $\partial h_l/\partial x_j$: how much hidden unit $l$ moves when input feature $j$ moves |
| Column $j$ of $\mathbf{J}$ | all hidden units' sensitivity to input feature $x_j$ |
| Large entry | $h_l$ is **sensitive** to $x_j$ |
| Entry near 0 | $h_l$ is **insensitive** to $x_j$ |
| The trade-off | preserve important variations, suppress unimportant ones |
| $\lambda$ | higher → more stable features; lower → more detailed reconstruction |
| Sigmoid-encoder Jacobian | $\partial h_l/\partial x_j = h_l(1-h_l)W_{lj}$ |
| Linear-encoder Jacobian | $\mathbf{J} = \mathbf{W}_e$ (so the penalty becomes weight decay) |
| vs the other two | corrupt **input** (denoising) · penalise **activations** (sparse) · penalise **Jacobian** (contractive) |

## [Lec 16 — Numerical Example, Limitations of AE](../notes/16-ae-numerical-and-limits.md)

| Item | Exactly this |
|---|---|
| The deck's architecture | $5 \to 4 \to 3 \to 4 \to 5$ |
| Layer rule | $\mathbf{h} = \text{act}(\mathbf{x}\mathbf{W} + b)$, $\mathbf{x}$ a **row** vector, neuron $j$ reads **column** $j$ |
| Hidden activation | ReLU, $\max(0,a)$ |
| Output activation | **linear**, $f(a)=a$, because the inputs are continuous |
| Loss used | MSE, $\frac{1}{5}\sum_{i=1}^{5}(x_i-\hat{x}_i)^2$ (divide by **features**, one sample) |
| Limitation 1 | low reconstruction error $\ne$ meaningful latent features |
| Limitation 2 | the code is not guaranteed useful for classification / clustering / anomaly detection |
| Limitation 3 | the decoder only approximates the input; **the model cannot generate new samples** |
| Limitation 4 | latent too small → information loss → poor reconstruction; latent too large → low error but **weak feature learning** (memorisation) |
| The fix the deck names | VAEs add **regularization on the latent space**, via a **KL divergence term** in the loss |
| Linear AE objective | $\min_\mathbf{W}\lVert\mathbf{X}-\mathbf{X}\mathbf{W}\mathbf{W}^\top\rVert^2$ |
| Linear AE $=$ | PCA — the same rank-$k$ subspace |

## [Lec 19 — KL Divergence, Part A](../notes/19-kl-divergence-a.md)

| Item | Exactly this |
|---|---|
| KL, discrete | $D_{\mathrm{KL}}(P\,\|\,Q) = \sum_x P(x)\log\dfrac{P(x)}{Q(x)}$ |
| KL, continuous | $D_{\mathrm{KL}}(P\,\|\,Q) = \displaystyle\int P(x)\log\dfrac{P(x)}{Q(x)}\,dx$ |
| As an expectation | $\mathbb{E}_{P(x)}\!\left[\log\dfrac{P(x)}{Q(x)}\right]$ — averaged under the **first** argument |
| Forward KL | $D_{\mathrm{KL}}(P\,\|\,Q)$ — truth first; **mode-covering**; penalises $Q$ for missing $P$'s mass |
| Reverse KL | $D_{\mathrm{KL}}(Q\,\|\,P)$ — model first; **mode-seeking**; penalises $Q$ for mass where $P$ has none |
| Non-negativity | $D_{\mathrm{KL}}(P\,\|\,Q) \ge 0$, $=0$ **iff** $P=Q$ everywhere |
| Asymmetry | $D_{\mathrm{KL}}(P\,\|\,Q) \ne D_{\mathrm{KL}}(Q\,\|\,P)$ in general — **it is a divergence, not a distance** |
| Infinite when | $P(x)>0$ and $Q(x)=0$ for forward KL |
| Convention | $0\log 0 = 0$ |
| Proof tool | $\log t \le t-1$ for $t>0$, equality only at $t=1$ |
| Units | natural log → **nats**; $\log_2$ → **bits**; $1\text{ nat} = 1.442695$ bits |
| Applies to | discrete **and** continuous distributions |

## [Lec 20 — KL Divergence, Part B](../notes/20-kl-divergence-b.md)

| Item | Exactly this |
|---|---|
| Self-information | $I(x) = \log_2\dfrac{1}{P(x)} = -\log_2 P(x)$ |
| Why the log | independent events multiply probabilities; $\log(ab)=\log a+\log b$ makes information **add** |
| Why $1/P$ | surprise must fall as probability rises |
| Entropy | $H(P) = \sum_x P(x) I(x) = -\sum_x P(x)\log_2 P(x)$ |
| Cross-entropy | $H(P,Q) = \sum_x P(x) I_Q(x) = -\sum_x P(x)\log_2 Q(x)$ |
| **The identity** | $D_{\mathrm{KL}}(P\,\|\,Q) = H(P,Q) - H(P)$ |
| Equivalently | $H(P,Q) = H(P) + D_{\mathrm{KL}}(P\,\|\,Q)$ |
| KL from the identity | $\sum_x P(x)\log_2\dfrac{P(x)}{Q(x)}$ |
| Entropy is cross-entropy with itself | $H(P,P) = H(P)$, so $D_{\mathrm{KL}}(P\,\|\,P)=0$ |
| One-hot target | $H(P)=0$, so $D_{\mathrm{KL}}(P\,\|\,Q) = H(P,Q) = -\log_2 Q(\text{correct class})$ |
| Why minimising CE $=$ minimising KL | $H(P)$ has no $\theta$ in it, so it does not affect $\arg\min_\theta$ |
| Units | $\log_2$ → **bits** (this deck); $\ln$ → **nats** (the rest of this book) |
| Convention | $0\log 0 = 0$ |

## [Lec 21 — Introduction to VAE and the Encoder](../notes/21-vae-encoder.md)

| Item | Exactly this |
|---|---|
| Generative goal | $p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$, then $\mathbf{x}_{\text{new}} \sim p_\theta(\mathbf{x})$ |
| Encoder (approximate posterior) | $q_\phi(\mathbf{z}\mid\mathbf{x})$ — parameters $\phi$ |
| Decoder (likelihood) | $p_\theta(\mathbf{x}\mid\mathbf{z})$ — parameters $\theta$ |
| Bayes' rule for the posterior | $p(\mathbf{z}\mid\mathbf{x}) = \dfrac{p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x})}$ |
| The evidence | $p(\mathbf{x}) = \displaystyle\int p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$ |
| Why it is intractable | integration over **all possible values** of $\mathbf{z}$ |
| The VAE's response | learn $q_\phi(\mathbf{z}\mid\mathbf{x}) \approx p(\mathbf{z}\mid\mathbf{x})$ |
| Encoder's assumed family | $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}\big(\mu_\phi(\mathbf{x}),\ \sigma^2_\phi(\mathbf{x})\big)$ |
| Prior | $p(z_i) = \mathcal{N}(0,1)$, jointly $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Encoder outputs | $\mu_\phi(\mathbf{x})$ and $\log\sigma^2_\phi(\mathbf{x})$ — **$2d$ numbers for latent dim $d$** |
| Variance recovery | $\sigma^2 = \exp(\log\sigma^2)$, $\ \sigma = \exp(\tfrac12\log\sigma^2)$ |
| Sampling the latent | $\mathbf{z} \sim \mathcal{N}\big(\mu_\phi(\mathbf{x}),\ \sigma^2_\phi(\mathbf{x})\big)$ |
| What keeps $q_\phi$ near the prior | $D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$ |
| The four Bayes names | posterior / likelihood / prior / evidence |

## [Lec 22 — Probabilistic Decoder, ELBO, VAE Loss](../notes/22-elbo-and-vae-loss.md)

| Item | Exactly this |
|---|---|
| Decoder, real-valued data | $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathcal{N}\big(\mu_\theta(\mathbf{z}),\ \sigma^2_\theta(\mathbf{z})\big)$ |
| Decoder, binary data | $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathrm{Bernoulli}\big(\pi_\theta(\mathbf{z})\big)$ |
| What we want to maximise | $\log p_\theta(\mathbf{x})$ |
| The exact decomposition | $\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}}(\mathbf{x}) + D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big)$ |
| Why it is a lower bound | KL $\geq 0$, so $\mathcal{L}_{\text{ELBO}}(\mathbf{x}) \leq \log p_\theta(\mathbf{x})$ |
| When the bound is tight | $q_\phi(\mathbf{z}\mid\mathbf{x}) = p_\theta(\mathbf{z}\mid\mathbf{x})$ |
| **The ELBO** | $\mathcal{L}_{\text{ELBO}}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$ |
| Max → min | $\max(f) \iff \min(-f)$ |
| **The VAE loss** | $\mathcal{L}_{\text{VAE}}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$ |
| In words | Reconstruction Loss **+** KL Divergence Loss |
| Gaussian decoder ⟹ | reconstruction term = MSE (up to a constant and a scale) |
| Bernoulli decoder ⟹ | reconstruction term = BCE, exactly |
| **Closed-form KL, one dim** | $\tfrac12\big(\mu^2 + \sigma^2 - \log\sigma^2 - 1\big)$ |
| **Closed-form KL, $d$ dims** | $-\tfrac12\sum_{j=1}^{d}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$ |
| General two-Gaussian KL | $\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$ |
| Monte Carlo estimate | $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] \approx \frac1L\sum_l \log p_\theta(\mathbf{x}\mid\mathbf{z}^{(l)})$, usually $L=1$ |

## [Lec 23 — The Reparameterization Trick](../notes/23-reparameterization.md)

| Item | Exactly this |
|---|---|
| The trick | $\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon$ |
| The noise | $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, parameter-free, **treated as a constant** in backprop |
| What it replaces | $\mathbf{z} \sim q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x}))$ |
| Derivative w.r.t. mean | $\dfrac{\partial z}{\partial\mu} = 1$ |
| Derivative w.r.t. std | $\dfrac{\partial z}{\partial\sigma} = \epsilon$ |
| Why it is needed | sampling is a random operation; $\mathbf{z}$ is not a differentiable function of $\mu_\phi,\sigma_\phi$, so backprop cannot pass the sampling node |
| Loss-to-latent chain | $\dfrac{\partial\mathcal{L}}{\partial\mathbf{z}} = \dfrac{\partial\mathcal{L}}{\partial\hat{\mathbf{x}}}\cdot\dfrac{\partial\hat{\mathbf{x}}}{\partial\mathbf{z}}$ (works even *without* the trick) |
| Mean branch | $\dfrac{\partial\mathcal{L}}{\partial\mu_\phi} = \dfrac{\partial\mathcal{L}}{\partial\hat{\mathbf{x}}}\cdot\dfrac{\partial\hat{\mathbf{x}}}{\partial\mathbf{z}}$ |
| Scale branch | $\dfrac{\partial\mathcal{L}}{\partial\sigma_\phi} = \dfrac{\partial\mathcal{L}}{\partial\hat{\mathbf{x}}}\cdot\dfrac{\partial\hat{\mathbf{x}}}{\partial\mathbf{z}}\cdot\epsilon$ |
| Into the encoder weights | $\dfrac{\partial\mathcal{L}}{\partial\phi} = \dfrac{\partial\mathcal{L}}{\partial\mu_\phi}\dfrac{\partial\mu_\phi}{\partial\phi} + \dfrac{\partial\mathcal{L}}{\partial\sigma_\phi}\dfrac{\partial\sigma_\phi}{\partial\phi}$ |
| Distribution check | $\mathbb{E}[z] = \mu$, $\mathrm{Var}(z) = \sigma^2$, affine-of-Gaussian is Gaussian |
| log-variance to $\sigma$ | $\sigma = e^{\frac{1}{2}\log\sigma^2} = \sqrt{e^{\log\sigma^2}}$ |
| Graph language | before: random node **in** the path; after: random node **beside** the path |

## [Lec 24 — VAE Numerical Example](../notes/24-vae-numerical.md)

| Item | Exactly this |
|---|---|
| Encoder hidden layer | $\mathbf{a} = \mathrm{ReLU}(\mathbf{x}\mathbf{W}_1 + b_1)$ |
| Mean head | $\mu = \mathbf{a}\mathbf{W}_\mu + b_\mu$, **no activation** |
| Log-variance head | $\log\sigma^2 = \mathbf{a}\mathbf{W}_\sigma + b_\sigma$, **no activation** |
| Variance | $\sigma^2 = e^{\log\sigma^2}$ (base $e$) |
| Standard deviation | $\sigma = \sqrt{\sigma^2} = e^{\frac12\log\sigma^2}$ |
| Reparameterization | $z_j = \mu_j + \sigma_j\epsilon_j$, $\epsilon\sim\mathcal{N}(0,1)$ |
| Decoder hidden | $\mathbf{a}_{\text{dec}} = \mathrm{ReLU}(\mathbf{z}\mathbf{W}_2 + b_2)$ |
| Decoder output | $\hat{\mathbf{x}} = f(\mathbf{a}_{\text{dec}}\mathbf{W}_3 + b_3)$; $f$ linear for continuous, sigmoid for binary/$[0,1]$ |
| What $\hat{\mathbf{x}}$ is | the **mean** $\mu_x$ of $p_\theta(\mathbf{x}\mid\mathbf{z})$, not a draw from it |
| Reconstruction loss (this deck) | $\mathcal{L}_{\text{rec}} = \dfrac{1}{d}\sum_{i=1}^{d}(x_i - \mu_{x,i})^2$ |
| **Gaussian KL, closed form** | $D_{\mathrm{KL}}(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,\mathbf{I})) = \dfrac12\sum_{j=1}^{k}\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$ |
| Same formula, flipped | $= -\dfrac12\sum_{j=1}^{k}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$ |
| Log base in that formula | **natural log, nats** |
| KL is zero when | $\mu_j = 0$ and $\sigma_j^2 = 1$ for every $j$ — i.e. $q_\phi$ equals the prior |
| Total | $\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\mathrm{KL}}$ |
| $\partial\mathcal{L}_{\mathrm{KL}}/\partial\mu_j$ | $\mu_j$ |
| $\partial\mathcal{L}_{\mathrm{KL}}/\partial(\log\sigma_j^2)$ | $\tfrac12(\sigma_j^2 - 1)$ |

## [Lec 26 — Disentanglement and β-VAE](../notes/26-beta-vae.md)

| Item | Exactly this |
|---|---|
| Standard VAE loss | $\mathcal{L}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ |
| β-VAE loss | $\mathcal{L}_{\beta\text{-VAE}}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + \boldsymbol{\beta}\, D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ |
| What changed | **only** a scalar weight on the KL term |
| $\beta = 1$ | **the standard VAE, exactly** |
| $\beta > 1$ | β-VAE: stronger KL pressure, encourages disentanglement |
| Entangled representation | changing one latent variable may change **multiple** properties together |
| Disentangled representation | each latent dimension **mainly** captures one meaningful factor of variation |
| Disentangled space is | **interpretable** and **controllable** |
| $\beta$ too large $\Rightarrow$ | **reconstruction quality may decrease** |
| β-VAE guarantees | nothing — it *encourages*, does **not guarantee** perfect disentanglement |
| What $\beta\uparrow$ encourages | $\mu(\mathbf{x}) \approx 0$, $\sigma^2(\mathbf{x}) \approx 1$ |
| Prior | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| Per-dim KL | $\frac{1}{2}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$ |
| Per-dim KL when $\sigma_j^2 = 1$ | $\frac{1}{2}\mu_j^2$ |
| The three fixes | 1. increase latent size · 2. tune $\beta$ (moderate) · 3. $\beta$ annealing |
| $\beta$ annealing direction | **up**: $\beta: 1 \to 2 \to 4$ (or from 0) |

## [Lec 27 — Conditional VAE](../notes/27-conditional-vae.md)

| Item | Exactly this |
|---|---|
| cVAE encoder | $q_\phi(\mathbf{z}\mid\mathbf{x}, \mathbf{y})$ |
| cVAE decoder | $p_\theta(\mathbf{x}\mid\mathbf{z}, \mathbf{y})$ |
| cVAE prior | $p(\mathbf{z})$ — **not** $p(\mathbf{z}\mid\mathbf{y})$ |
| What cVAE learns | $(\mathbf{x},\mathbf{y}) \to \mathbf{z} \to \hat{\mathbf{x}}$ |
| What plain VAE learns | $\mathbf{x} \to \mathbf{z} \to \hat{\mathbf{x}}$ |
| Reparameterization in a cVAE | $\mathbf{z} = \mu_\phi(\mathbf{x},\mathbf{y}) + \sigma_\phi(\mathbf{x},\mathbf{y}) \odot \epsilon$, $\;\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| VAE loss | $\mathcal{L}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ |
| **cVAE loss** | $\mathcal{L}_{\text{cVAE}}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{y})}[\log p_\theta(\mathbf{x}\mid\mathbf{z},\mathbf{y})] + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{y})\,\|\,p(\mathbf{z}))$ |
| What changed in the loss | **only** $\mathbf{y}$ added to three conditioning bars; no new term, no new hyperparameter |
| How $\mathbf{y}$ enters | **concatenation** ($\oplus$) at two places: with $\mathbf{x}$ at the encoder, with $\mathbf{z}$ at the decoder |
| What $\mathbf{y}$ is | a condition — a class label or attribute, usually **one-hot** |
| What cVAE buys | **controlled generation** — ask for digit 5, get digit 5 |
| Encoder input length | $D_x + D_y$ |
| Decoder input length | $D_z + D_y$ |
| Extra parameters | $D_y \times D_h$ in each network |

## [Lec 28 — Latent Space Interpolation](../notes/28-latent-interpolation.md)

| Item | Exactly this |
|---|---|
| Interpolation formula | $\mathbf{z}_\alpha = (1-\alpha)\mathbf{z}_A + \alpha\mathbf{z}_B$ |
| Per-component form | $\mathbf{z}_\alpha = [(1-\alpha)z_{A1} + \alpha z_{B1},\ (1-\alpha)z_{A2} + \alpha z_{B2}]$ |
| $\alpha = 0$ gives | $\mathbf{z}_A$ — **exactly image A** |
| $\alpha = 1$ gives | $\mathbf{z}_B$ — **exactly image B** |
| $\alpha$ weights | $\alpha$ on the **destination** $\mathbf{z}_B$, $(1-\alpha)$ on the source $\mathbf{z}_A$ |
| Deck's $\alpha$ grid | $0,\ 0.25,\ 0.5,\ 0.75,\ 1$ |
| The three steps | encode both inputs → interpolate the latents → decode every $\mathbf{z}_\alpha$ |
| Why a VAE can do this | the **KL term** pulls every $q_\phi(\mathbf{z}\mid\mathbf{x})$ toward $p(\mathbf{z})$, making the latent space **continuous and dense** |
| Why an AE cannot | reconstruction-only loss leaves **holes**; a midpoint can land where the decoder was never trained → garbage |
| What makes it an interpolation | $(1-\alpha) + \alpha = 1$, both non-negative — a **convex combination** |
| $\alpha < 0$ or $\alpha > 1$ | **extrapolation**, not interpolation |
| What is interpolated | the **latent vectors**, never the pixels |

## [Lec 31 — Motivation for GANs](../notes/31-gan-motivation.md)

| Item | Exactly this |
|---|---|
| Limitation 1 | Gaussian assumption: $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x}))$, prior fixed at $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Limitation 2 | Blurry generated images, caused by a **pixel-wise** reconstruction loss averaging plausible outputs |
| Limitation 3 | Trade-off: "Improving one may reduce the performance of the other" |
| Limitation 4 | Posterior collapse: $\mu(\mathbf{x})\to 0$, $\sigma(\mathbf{x})\to 1$, so $q_\phi(\mathbf{z}\mid\mathbf{x})\approx p(\mathbf{z})$ and $D_{\mathrm{KL}}\approx 0$ |
| Limitation 5 | Explicit **likelihood assumption**: Gaussian ⇒ MSE, Bernoulli ⇒ BCE |
| Limitation 6 | Samples lack sharp edges, fine textures, small facial details, high-frequency information |
| VAE loss | $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ |
| The blur theorem | $\arg\min_{\hat{x}}\ \mathbb{E}[(x-\hat{x})^2] = \mathbb{E}[x]$, with minimum $\operatorname{Var}(x)$ |
| The motivation | "To generate highly realistic data without explicitly defining a probability distribution or reconstruction loss" |
| Generator | $\mathbf{z} \to G(\mathbf{z}) = \hat{\mathbf{x}}$; goal: generate samples that look real |
| Discriminator | $D(\mathbf{x}) \to$ real or fake; goal: correctly distinguish real from generated |
| GAN, one line | two neural networks that learn by **competing with each other** |
| Collapse by decoder type | text ⇒ autoregressive decoder (collapse is easy); images ⇒ CNN non-autoregressive decoder |

## [Lec 32 — GAN Architecture](../notes/32-gan-architecture.md)

| Item | Exactly this |
|---|---|
| Generator map | $G: \mathbf{z} \mapsto \hat{\mathbf{x}}$, with $\mathbf{z}\in\mathbb{R}^d$, $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Discriminator map | $D: \text{sample} \mapsto \hat{y}\in(0,1)$, the probability the input is **real** |
| Output-shape rule | "Generated image size is same as the real image" |
| Real data | $\mathbf{x} \sim p_{\text{data}}(\mathbf{x})$; the generator's induced distribution is $p_g$ |
| Step A | train $D$: real → label 1, fake → label 0; **$\phi$ frozen, $\theta$ updated** |
| Step B | train $G$: fake → label **1**; **$\theta$ frozen, $\phi$ updated** |
| Discriminator loss, structure | $\mathcal{L}_D = \mathcal{L}_{\text{real}} + \mathcal{L}_{\text{fake}}$ (derivation: [Lec 33](../notes/33-gan-objective.md)) |
| Discriminator targets | $D(\mathbf{x}) \to 1$ and $D(G(\mathbf{z})) \to 0$ |
| Generator target | $D(G(\mathbf{z})) \to 1$ |
| Generator update | $\phi \leftarrow \phi - \eta\,\partial \mathcal{L}_G/\partial\phi$ |
| Discriminator update | $\theta \leftarrow \theta - \eta\,\partial \mathcal{L}_D/\partial\theta$ |
| Generator gradient path | $\mathcal{L}_G \to \hat{y} \to \hat{\mathbf{x}} \to \phi$, i.e. $\mathcal{L}_G \to D_\theta \to G_\phi \to \phi$ |
| Discriminator gradient split | $\partial \mathcal{L}_D/\partial\theta = \partial \mathcal{L}_{\text{real}}/\partial\theta + \partial \mathcal{L}_{\text{fake}}/\partial\theta$ |
| The padlock rule | gradient passes **through** $D$ in Step B, but $\theta$ is not updated |
| Structural absence | a GAN has **no encoder** — nothing maps $\mathbf{x}$ to $\mathbf{z}$ |

## [Lec 33 — GAN Objective and Loss Functions](../notes/33-gan-objective.md)

| Item | Exactly this |
|---|---|
| **The GAN objective** | $\min_G\max_D V(D,G) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}[\log(1-D(G(\mathbf{z})))]$ |
| Deck's parameterised form | $\min_\phi\max_\theta V(\theta;\phi)$, with $\theta \to D$, $\phi \to G$ |
| Per-sample BCE | $\mathcal{L} = -[y\log\hat y + (1-y)\log(1-\hat y)]$ |
| Real-sample loss | $\mathcal{L}_{\text{real}} = -\log D(\mathbf{x})$ — from $y=1$ |
| Fake-sample loss | $\mathcal{L}_{\text{fake}} = -\log(1-D(G(\mathbf{z})))$ — from $y=0$ |
| Discriminator loss | $\mathcal{L}_D = \mathcal{L}_{\text{real}} + \mathcal{L}_{\text{fake}} = -[\log D(\mathbf{x}) + \log(1-D(G(\mathbf{z})))]$ |
| Loss ↔ value | $V = -\mathcal{L}_D$; $\ \min_D[-A] \iff \max_D A$ |
| Discriminator step | $\max_\theta V_D(\theta;\phi)$, $\phi$ frozen |
| Generator step (minimax) | $\min_\phi \mathbb{E}_{\mathbf{z}}[\log(1-D(G(\mathbf{z})))]$, $\theta$ frozen |
| **Generator step (non-saturating)** | $\max_G \mathbb{E}_{\mathbf{z}}[\log D(G(\mathbf{z}))]$ — BCE with the fake labelled **1** |
| Saturating gradient (w.r.t. $D$'s logit) | magnitude $= D(G(\mathbf{z}))$ |
| Non-saturating gradient | magnitude $= 1 - D(G(\mathbf{z}))$ |
| What $\max_D$ wants | $D(\mathbf{x})\to1$ **and** $D(G(\mathbf{z}))\to0$ |
| What $\min_G$ wants | $D(G(\mathbf{z}))\to1$; the first term is constant in $\phi$ |
| Monte Carlo form | $\mathbb{E}_{p_{\text{data}}}[\log D(\mathbf{x})] \approx \frac1m\sum_{i=1}^m \log D(\mathbf{x}^{(i)})$ |

## [Lec 34 — GAN Convergence and Nash Equilibrium](../notes/34-gan-convergence.md)

| Item | Exactly this |
|---|---|
| **Optimal discriminator** | $D^*(\mathbf{x}) = \dfrac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$ |
| Derived by maximising | $f(y) = a\log y + b\log(1-y)$, $\ f'(y) = \frac{a}{y}-\frac{b}{1-y} = 0 \Rightarrow y = \frac{a}{a+b}$ |
| Second-order check | $f''(y) = -\frac{a}{y^2}-\frac{b}{(1-y)^2} < 0$ — concave, so it is a maximum |
| **At convergence** | $p_g = p_{\text{data}} \Rightarrow D^*(\mathbf{x}) = \tfrac12$ for every $\mathbf{x}$ |
| **Global optimum value** | $C(G) = V(D^*,G) = -\log 4 = -1.3863$ nats $= -2$ bits |
| Objective in divergence form | $C(G) = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g)$ |
| What a GAN minimises | the **Jensen–Shannon divergence** between $p_g$ and $p_{\text{data}}$ |
| $D^*$ from the ratio | $D^* = \dfrac{1}{1+r}$, $\ r = p_g/p_{\text{data}}$ |
| **Nash equilibrium (deck)** | neither network can obtain a better outcome by changing its own parameters **alone**, with the other's fixed |
| Game type | zero-sum: $D$'s payoff $+V$, $G$'s payoff $-V$ |
| Geometry of the solution | a **saddle point** — maximum along $D$'s axes, minimum along $G$'s |
| Saddle instability | for $V=uv$, simultaneous steps give $u'^2+v'^2 = (1+\eta^2)(u^2+v^2)$ |
| **Mode collapse** | $G$ maps many $\mathbf{z}$ to one (or few) outputs; $p_g$ covers part of $p_{\text{data}}$ |
| Deck's convergence definition | $p_g$ matches $p_{\text{data}}$ and $D$ can no longer distinguish real from generated |

## [Lec 38 — Conditional GAN](../notes/38-conditional-gan.md)

| Item | Exactly this |
|---|---|
| Generator | $G(\mathbf{z},\mathbf{y})$ — noise **and** condition |
| Discriminator | $D(\mathbf{x},\mathbf{y})$ — sample **and** condition |
| What $D$ outputs | $D(\mathbf{x},\mathbf{y}) = P(\text{real}\mid \mathbf{x},\mathbf{y})$, a scalar in $(0,1)$ |
| Noise distribution | $\mathbf{z}\sim\mathcal{N}(0,1)$ component-wise, mean 0, **variance** 1 |
| $D$'s targets | $D(\mathbf{x},\mathbf{y})\to1$ and $D(G(\mathbf{z},\mathbf{y}),\mathbf{y})\to0$ |
| $G$'s target | $D(G(\mathbf{z},\mathbf{y}),\mathbf{y})\to1$ |
| Real loss | $\mathcal{L}_D^{\text{real}} = -\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x},\mathbf{y})]$ |
| Fake loss | $\mathcal{L}_D^{\text{fake}} = -\mathbb{E}_{\mathbf{z}\sim p_\mathbf{z}}[\log(1-D(G(\mathbf{z},\mathbf{y}),\mathbf{y}))]$ |
| Total $D$ loss | $\mathcal{L}_{\mathcal{D}} = \mathcal{L}_D^{\text{real}} + \mathcal{L}_D^{\text{fake}}$ |
| Saturating $G$ loss | $\mathcal{L}_G = \mathbb{E}[\log(1-D(G(\mathbf{z},\mathbf{y}),\mathbf{y}))]$ — vanishing gradient |
| Non-saturating $G$ loss | $\mathcal{L}_G = -\mathbb{E}[\log D(G(\mathbf{z},\mathbf{y}),\mathbf{y})]$ — what you implement |
| cGAN objective | $\min_G\max_D V = \mathbb{E}_{p_{\text{data}}}[\log D(\mathbf{x},\mathbf{y})] + \mathbb{E}_{p_\mathbf{z}}[\log(1-D(G(\mathbf{z},\mathbf{y}),\mathbf{y}))]$ |
| $D$ learns two things | does the image match the condition? · is it real? |
| $G$ learns two things | looks real · matches $\mathbf{y}$ |
| Original paper | Mirza & Osindero, *Conditional Generative Adversarial Nets*, arXiv:1411.1784 |

## [Lec 39 — Image-to-Image Translation: Pix2Pix GAN](../notes/39-pix2pix.md)

| Item | Exactly this |
|---|---|
| What Pix2Pix is | a specialised cGAN for **paired** image-to-image translation |
| Paper | Isola, Zhu, Zhou & Efros, *Image-to-Image Translation with Conditional Adversarial Networks*, arXiv:1611.07004 |
| Generator | **U-Net** — encoder (contracting) + bottleneck + decoder (expanding) + **skip connections** |
| What skips do | "preserve low-level spatial details lost during encoding" |
| Discriminator | **PatchGAN** — classifies whether each $N\times N$ patch is real or fake |
| PatchGAN output rule | the final output is the **average of all patch decisions** |
| Best patch size | $70\times70$ — best visual and FCN performance, best tradeoff |
| Adversarial term | $\mathcal{L}_{\text{cGAN}}(G,D) = \mathbb{E}_{\mathbf{x},\mathbf{y}}[\log D(\mathbf{x},\mathbf{y})] + \mathbb{E}_{\mathbf{x},\mathbf{y}}[\log(1-D(\mathbf{x},G(\mathbf{x})))]$ |
| Reconstruction term | $\mathcal{L}_{L1}(G) = \mathbb{E}_{\mathbf{x},\mathbf{y}}[\lVert\mathbf{y} - G(\mathbf{x})\rVert_1]$ |
| Total generator loss | binary cross-entropy loss $+\ \lambda\,L_1$ loss |
| Full objective | $G^{*} = \arg\min_G\max_D\ \mathcal{L}_{\text{cGAN}}(G,D) + \lambda\mathcal{L}_{L1}(G)$ |
| Why $L_1$ not $L_2$ | $L_1$ → conditional **median**, $L_2$ → conditional **mean**; averaging is blur, so $L_1$ blurs less |
| At inference | the **discriminator is not used** |

## [Lec 40 — CycleGAN](../notes/40-cyclegan.md)

| Item | Exactly this |
|---|---|
| Generators | $G: X \to Y$ and $F: Y \to X$ |
| Discriminators | $D_X$ on domain $X$, $D_Y$ on domain $Y$ |
| Networks trained | **four** (2 generators + 2 discriminators) |
| Adversarial loss, $X\to Y$ | $\mathcal{L}_{\text{GAN}}(G,D_Y,X,Y) = \mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}[\log D_Y(\mathbf{y})] + \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}[\log(1-D_Y(G(\mathbf{x})))]$ |
| Adversarial loss, $Y\to X$ | $\mathcal{L}_{\text{GAN}}(F,D_X,Y,X) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}[\log D_X(\mathbf{x})] + \mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}[\log(1-D_X(F(\mathbf{y})))]$ |
| Forward cycle | $\mathbf{x} \to G(\mathbf{x}) \to F(G(\mathbf{x})) \approx \mathbf{x}$ |
| Backward cycle | $\mathbf{y} \to F(\mathbf{y}) \to G(F(\mathbf{y})) \approx \mathbf{y}$ |
| Cycle-consistency loss | $\mathcal{L}_{\text{cyc}}(G,F) = \mathbb{E}_{\mathbf{x}}[\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1] + \mathbb{E}_{\mathbf{y}}[\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1]$ |
| Full objective | $\mathcal{L} = \mathcal{L}_{\text{GAN}}(G,D_Y,X,Y) + \mathcal{L}_{\text{GAN}}(F,D_X,Y,X) + \lambda\mathcal{L}_{\text{cyc}}(G,F)$ |
| Solution | $G^*,F^* = \arg\min_{G,F}\max_{D_X,D_Y}\mathcal{L}(G,F,D_X,D_Y)$ |
| Norm in the cycle loss | **L1**, $\lVert\cdot\rVert_1$ — not L2 |
| The deck's key sentence | adversarial loss ensures the **output looks realistic**; it does **not** ensure the output **corresponds to the input** |
| Who minimises | generators $G, F$ (fool $D$ **and** stay cycle-consistent) |
| Who maximises | discriminators $D_X, D_Y$ |

## [Lec 41 — StyleGAN](../notes/41-stylegan.md)

| Item | Exactly this |
|---|---|
| Mapping network | $f: \mathcal{Z}\to\mathcal{W}$, $\mathbf{w} = f(\mathbf{z})$, an **8-layer MLP** |
| Dimensions | $\mathbf{z}\in\mathbb{R}^{512}$, $\mathbf{w}\in\mathbb{R}^{512}$ — **equal**, not a bottleneck |
| Prior | $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ (deck: $z\sim N(0,1)$) |
| Why $\mathcal{W}$ exists | $\mathcal{Z}$ must follow the prior's density ⇒ forced curvature ⇒ **unavoidable entanglement**; $\mathcal{W}$ is never sampled from, so it is unconstrained and free to disentangle |
| Synthesis input | a **fixed learned tensor, $4\times4\times512$**, trainable, shared by all generated images |
| Full pipeline | $\mathbf{z}\to$ **Mapping Network** $\to \mathbf{w}\to$ **Styles** $\to$ **Synthesis Network** $\to$ **Image** |
| Affine transform | $\mathbf{y} = \mathbf{A}\mathbf{w} + \mathbf{b}$, $\mathbf{y} = (y_s, y_b)$ — $y_s$ learned **scale**, $y_b$ learned **bias** |
| AdaIN | $\mathrm{AdaIN}(\mathbf{x}_i,\mathbf{y}) = y_{s,i}\dfrac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + y_{b,i}$ |
| AdaIN, in words | normalize the feature map (**removes original statistics**), then apply a new style using **learned scale and bias** |
| AdaIN's guarantee | output per-channel mean $= y_b$, output per-channel sd $= \lvert y_s\rvert$ |
| Normalization scope | **per channel, independently**, over spatial positions of **this image** (instance norm) |
| Upsampling rule | after **2 conv layers in one resolution**, upsample |
| Conv layers | **18**, over **9** resolutions, $4\times4 \to 1024\times1024$ |
| Coarse layers ($4$–$16$) | pose, face shape, coarse structure |
| Middle layers ($32$–$128$) | eyes, nose, hairstyle |
| Fine layers ($256$–$1024$) | pores, wrinkles, hair strands, texture |
| Noise injection | per-pixel $\mathcal{N}(0,1)$ image, scaled by **$B$, a learned per-channel scaling factor**; adds **stochastic variation** |
| Where variation comes from | AdaIN **and** noise injection — **not** from the initial input tensor |
| Paper | Karras, Laine, Aila (Nvidia), *A Style-Based Generator Architecture for GANs* |

## [Lec 42 — StyleGAN 2](../notes/42-stylegan2.md)

| Item | Exactly this |
|---|---|
| The two StyleGAN artefacts | water-droplet artefact · phase artefact (blob / phase-coherence artefact) |
| Droplet's cause | AdaIN's **instance normalisation** |
| Droplet's mechanism | generator makes one strong localised spike that dominates the normalisation statistics |
| Where the droplet appears | **all feature maps from $64\times64$ resolution upward**, systemic to every StyleGAN image |
| Droplet's fix | remove AdaIN; **weight modulation + demodulation** |
| Core idea of the fix | "the style modifies the **convolution weights**", not the feature maps |
| Weight modulation | $w'_{ijk} = s_i \cdot w_{ijk}$ |
| Weight demodulation | $w''_{ijk} = \dfrac{w'_{ijk}}{\sqrt{\sum_{i,k}(w'_{ijk})^2 + \epsilon}}$ |
| Index convention | $i$ = input channel · $j$ = output channel · $k$ = position inside the filter |
| What the sum runs over | $i$ and $k$ — **not** $j$; "normalizes one output channel at a time" |
| AdaIN (for the contrast only) | $\text{AdaIN}(x_i,\mathbf{y}) = y_{s,i}\frac{x_i-\mu(x_i)}{\sigma(x_i)} + y_{b,i}$ |
| Phase artefact's cause | **progressive growing** |
| "Phase", defined | signal-processing term for spatial alignment / positional consistency of patterns |
| Phase artefact's fix | remove progressive growing; fixed architecture; **skip connections in $G$**, **residual connections in $D$** |
| Residual block | $y = F(x) + x$ |
| Why residual, per the deck | original information + new learned refinements; without it "repeated convolutions may distort feature locations" |
| Why skip, per the deck | early layers directly influence the final image; spatial shifts stop accumulating over upsampling |
| StyleGAN 2 pipeline order | $\mathbf{z}$ → mapping network → $\mathbf{w}$ → affine transform → weight modulation → convolution → weight demodulation → noise injection → activation |
| The paper | Karras, Laine, Aittala, Hellsten, Lehtinen, Aila, *Analyzing and Improving the Image Quality of StyleGAN*, arXiv:1912.04958 |

## [Lec 44 — Introduction to Diffusion Models](../notes/44-diffusion-intro.md)

| Item | Exactly this |
|---|---|
| The motivation, in the deck's words | *"Need models which are Stable like VAE, but image quality of GANs"* |
| Diffusion's physics origin | **nonequilibrium thermodynamics** (2015) — "how ink spreads in water or how perfume spreads in air" |
| The principle | particles spread from regions of **high concentration to low concentration** |
| Timeline | AE **1987** · VAE **2013** · GAN **2014** · diffusion **2015** |
| AE's limitation | cannot generate realistic new images |
| VAE's limitation | generated images are often **blurry** |
| GAN's limitations | **training instability** and **mode collapse** |
| **Tractable**, defined | can easily (i) compute $p(\mathbf{x})$, (ii) train the model, (iii) sample from it |
| The dichotomy | tractable but not expressive · expressive but intractable |
| Diffusion's resolution | *"breaking a hard problem into many easy steps"* — both flexible **and** tractable |
| Forward process | **known, fixed, not learned**; gradually add noise; complex data → simple Gaussian |
| Forward chain | $\mathbf{x}_0 \to \mathbf{x}_1 \to \cdots \to \mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Reverse process | **learned**; $\mathbf{x}_T \to \mathbf{x}_{T-1} \to \cdots \to \mathbf{x}_0$ |
| Where the learning is | *"In Forward process only noise is added. In Reverse process the actual learning happens."* |
| The core idea | learning $p(\mathbf{x})$ directly is hard; learning $p(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ is much easier |
| Time-varying probability | $p_t(\mathbf{x}) = p(\mathbf{x}\mid t)$ — probability of data $\mathbf{x}$ at time $t$ |
| Image vs noise | image = signal with all of structured information; noise = unstructured data |
| What the network predicts | **the added noise**, not the clean image |
| Training loop | real $\mathbf{x}_0$ → add noise → $\mathbf{x}_t$ → network → predict noise → loss vs true noise → update weights |
| Sampling loop | $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ → repeated denoising $\mathbf{x}_T\to\cdots\to\mathbf{x}_0$ |
| **Markov property** | $P(\mathbf{x}_t\mid \mathbf{x}_{t-1},\mathbf{x}_{t-2},\ldots,\mathbf{x}_0) = P(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ |
| Markov in words | the next state depends only on the current state, not on the sequence before it |
| What is Markov | **both** the forward noising and the reverse denoising processes |
| The paper | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, arXiv:1503.03585 |

## [Lec 45 — Mathematical Foundations of Diffusion Models](../notes/45-diffusion-math.md)

| Item | Exactly this |
|---|---|
| Forward transition | $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ — **fixed and known**, no parameters |
| Reverse transition | $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ — $\theta$ = weights of the denoising network |
| What is learned | **only** the reverse process |
| Noise variance schedule | $\{\beta_1,\beta_2,\ldots,\beta_T\}$, with $\beta_t \in (0,1)$, **not constant**, increasing |
| Signal retention (one step) | $\alpha_t = 1-\beta_t$ |
| Signal retention (cumulative) | $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ |
| Forward Markov property (defined in [Lec 44](../notes/44-diffusion-intro.md)) | $q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\ldots,\mathbf{x}_0) = q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ |
| Forward joint | $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0) = \prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ |
| Terminal requirement | $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Gaussian density | $p(x) = \dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$ |
| Standard Gaussian | $\mathcal{N}(0,1)$ — mean 0, **variance** 1 |
| Multivariate Gaussian | $p(\mathbf{x}) = \mathcal{N}(\mathbf{x};\boldsymbol{\mu},\boldsymbol{\Sigma})$, $\mathbf{x}\in\mathbb{R}^d$ |
| Variance | $\operatorname{Var}(x_1) = \mathbb{E}[(x_1-\mu_1)^2]$ |
| Covariance | $\operatorname{Cov}(x_1,x_2) = \mathbb{E}[(x_1-\mu_1)(x_2-\mu_2)]$ |
| Covariance matrix (2-D) | $\boldsymbol{\Sigma} = \begin{bmatrix}\operatorname{Var}(x_1) & \operatorname{Cov}(x_1,x_2)\\ \operatorname{Cov}(x_2,x_1) & \operatorname{Var}(x_2)\end{bmatrix}$ |
| Zero covariance means | no relationship between those dimensions |

## [Lec 46 — DDPM: The Forward Process](../notes/46-ddpm-forward.md)

| Item | Exactly this |
|---|---|
| One-step forward | $q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t\mathbf{I}\big)$ |
| Same, in $\alpha$ | $q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t;\ \sqrt{\alpha_t}\,\mathbf{x}_{t-1},\ (1-\alpha_t)\mathbf{I}\big)$ |
| One-step sample | $\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$, $\epsilon_t\sim\mathcal{N}(0,1)$ |
| Definitions | $\alpha_t = 1-\beta_t$, $\ \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ |
| **Closed form (sample)** | $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| **Closed form (distribution)** | $q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}\big)$ |
| Rule 1 | $\epsilon\sim\mathcal{N}(0,1)\Rightarrow a\epsilon\sim\mathcal{N}(0,a^2)$ |
| Rule 2 | independent $u,v$: $\operatorname{Var}(u+v) = \operatorname{Var}(u)+\operatorname{Var}(v)$ |
| The cancellation | $\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = 1-\bar\alpha_t$ |
| Variance preservation | $\alpha_t\cdot 1 + (1-\alpha_t) = 1$ |
| Unscaled recursion blows up | $\operatorname{Var}(\mathbf{x}_t) = \operatorname{Var}(\mathbf{x}_{t-1}) + (1-\alpha_t)$ |
| Reparameterization | $\mathbf{z} = \mu + \sigma\odot\epsilon$ — see [Lec 23](../notes/23-reparameterization.md) |
| SNR (owned by [Lec 48](../notes/48-forward-diffusion-handson.md)) | $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ |
| Covariance structure | $\beta_t\mathbf{I}$ — diagonal, equal entries, i.e. **isotropic** |

## [Lec 47 — DDPM: The Reverse Process](../notes/47-ddpm-reverse.md)

| Item | Exactly this |
|---|---|
| Reverse Markov property | $P(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\ldots,\mathbf{x}_T) = P(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ |
| Reverse joint | $p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$, $\ \mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Model family | $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t) = \mathcal{N}\big(\mu_\theta(\mathbf{x}_t,t),\ \Sigma_\theta(\mathbf{x}_t,t)\big)$ |
| Tractable posterior | $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}\big(\tilde\mu_t(\mathbf{x}_t,\mathbf{x}_0),\ \tilde\beta_t\mathbf{I}\big)$ |
| Posterior mean | $\tilde\mu_t = \dfrac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0 + \dfrac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t$ |
| Posterior variance | $\tilde\beta_t = \dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ |
| Why it can't be used | it depends on $\mathbf{x}_0$, unknown during generation |
| Invert the forward form | $\mathbf{x}_0 = \dfrac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$ |
| Mean without $\mathbf{x}_0$ | $\tilde\mu_t = \dfrac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$ |
| Learned mean | $\mu_\theta = \dfrac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)\right)$ |
| **Sampling equation** | $\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t,t) + \sigma_t\mathbf{z}$, $\ \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| $\sigma_t^2$ | $\beta_t$, or $\tilde\beta_t$ — the deck never says |
| Noise dropped at | $t=1$ (the final step returns $\mu_\theta$) |
| **Simplified loss** | $\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0,\epsilon,t}\big[\lVert\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\rVert^2\big]$ |
| Expectation is over | images $\mathbf{x}_0$, noise $\epsilon$, timesteps $t$ |
| Why Gaussian reverse steps | valid only because $\beta_t$ is small |
| ELBO terms | $\mathcal{L}_T$ (constant) $+ \sum\mathcal{L}_{t-1}$ (Gaussian KLs) $+\ \mathcal{L}_0$ |

## [Lec 48 — Hands-on: Forward Diffusion](../notes/48-forward-diffusion-handson.md)

| Item | Exactly this |
|---|---|
| Number of steps | $T = 1000$ |
| Linear schedule endpoints | $\beta_1 = 10^{-4}$, $\beta_T = 0.02$ |
| Linear spacing | $\Delta\beta = (0.02-10^{-4})/999 = 1.992\times10^{-5}$ |
| Cosine clip range | $[10^{-4},\ 0.999]$, offset $s = 0.008$ |
| $\alpha_t$ | $\alpha_t = 1 - \beta_t$ |
| $\bar\alpha_t$ | $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ — `torch.cumprod` |
| The jump (owned by [Lec 46](../notes/46-ddpm-forward.md)) | $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| One Markov step | $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\epsilon$ — std is $\sqrt{\beta_t}$ |
| Markov property | $q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\ldots,\mathbf{x}_0) = q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ |
| Signal-to-noise ratio | $\mathrm{SNR}(t) = \dfrac{\bar\alpha_t}{1-\bar\alpha_t}$ |
| Simplified objective | $\mathcal{L}_{\text{simple}} = \mathbb{E}\big[\lVert\epsilon - \epsilon_\theta(\mathbf{x}_t,t)\rVert_2^2\big]$ |
| Data scaling | $[0,1] \to [-1,1]$ via `x * 2 - 1` |
| One-step clean estimate | $\hat{\mathbf{x}}_0 = \dfrac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ |
| Six training steps | sample $\mathbf{x}_0$ → sample $t$ → sample $\epsilon$ → build $\mathbf{x}_t$ → feed $(\mathbf{x}_t, t)$ → predict $\epsilon$ |

## [Lec 49 — U-Net for Denoising](../notes/49-unet.md)

| Item | Exactly this |
|---|---|
| What the U-Net computes | $\hat\epsilon = \epsilon_\theta(\mathbf{x}_t, t)$ |
| Shape requirement | output shape $=$ input shape, $H\times W\times C$ |
| Three regions | Encoder (contracting) · Bottleneck · Decoder (expanding) |
| Encoder trend | $H,W$ halve; $C$ doubles |
| Decoder trend | $H,W$ double; $C$ halves |
| U-Net skip (owned here) | **concatenation**, encoder → decoder, $C_{\text{out}} = C_{\text{dec}} + C_{\text{enc}}$; doubles at an equal-width join |
| Residual skip ([Lec 42](../notes/42-stylegan2.md)) | **addition**, inside a block, $\mathbf{H}_{\text{out}} = \mathbf{H} + F(\mathbf{H})$; channels unchanged |
| Decoder stage order | **upsample first, then concatenate**, then blocks |
| Why U-Net (deck's two reasons) | captures global context; preserves fine detail via skip connections |
| Sinusoidal time embedding | $\mathrm{TE}(t)_{2i} = \sin\!\big(t/10000^{2i/d}\big)$, $\mathrm{TE}(t)_{2i+1} = \cos\!\big(t/10000^{2i/d}\big)$ |
| Typical embedding dim | 128 to 512 |
| Time injection | $\mathbf{b}_t = \mathbf{W}_t\,\mathrm{TE}(t)$, then $\mathbf{h}_2 = \mathbf{h}_1 + \mathbf{b}_t$, broadcast over $H\times W$ |
| ResBlock order | GN → SiLU → Conv1 → $+\mathbf{b}_t$ → GN → SiLU → Conv2 → $+\mathbf{H}$ |
| Activation / normalisation | SiLU, $\mathrm{SiLU}(x)=x\sigma(x)$; GroupNorm |
| DDPM's three additions to the original U-Net | timestep embeddings · residual blocks · attention layers |
| Self-attention | $\mathbf{A} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})$, $\mathbf{Y} = \mathbf{A}\mathbf{V}$, on $HW$ flattened tokens |
| Full path | $\mathbf{x}_t \to$ Encoder $\to$ Bottleneck $\to$ Decoder $\to \epsilon$ |

## [Lec 50 — Classifier-Guided Diffusion](../notes/50-classifier-guidance.md)

| Item | Exactly this |
|---|---|
| The key equation | $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid y) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p_\phi(y\mid\mathbf{x}_t)$ |
| In words | conditional score $=$ unconditional score $+$ classifier gradient |
| Bayes step it comes from | $p(\mathbf{x}_t\mid y) = p(y\mid\mathbf{x}_t)p(\mathbf{x}_t)/p(y)$ |
| Why $p(y)$ drops out | it does not depend on $\mathbf{x}_t$, so $\nabla_{\mathbf{x}_t}\log p(y) = \mathbf{0}$ |
| Score ↔ noise | $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) = -\dfrac{1}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t)$ |
| Deck's gloss of it | Score $=$ Constant $\times$ Noise Prediction |
| **Guided noise prediction** | $\hat\epsilon(\mathbf{x}_t,t,y) = \epsilon_\theta(\mathbf{x}_t,t) - s\sqrt{1-\bar\alpha_t}\,\nabla_{\mathbf{x}_t}\log p_\phi(y\mid\mathbf{x}_t)$ |
| The classifier | $p_\phi(y\mid\mathbf{x}_t)$ — probability that the **noisy** image $\mathbf{x}_t$ belongs to class $y$ |
| Guidance scale | $s$; small $s$ = diverse but weak conditioning, large $s$ = strong conditioning but less diversity |
| $\bar\alpha_t$ | $\prod_{i=1}^{t}\alpha_i$ — cumulative signal surviving $t$ forward steps ([Lec 46](../notes/46-ddpm-forward.md)) |
| Sampling rule | the guided $\hat\epsilon$ **replaces** $\epsilon_\theta$ in the reverse update; nothing else changes |
| The two limitation headings | *Requirement for a Separate Model* · *Adversarial Vulnerability* |
| The paper | Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021 |

## [Lec 51 — Classifier-Free Diffusion Guidance (CFG)](../notes/51-classifier-free-guidance.md)

| Item | Exactly this |
|---|---|
| **The CFG noise prediction** (deck's form) | $\hat\epsilon(\mathbf{x}_t,t,y) = (1+s)\,\epsilon_\theta(\mathbf{x}_t,t,y) - s\,\epsilon_\theta(\mathbf{x}_t,t,\varnothing)$ |
| **Equivalent form** (code's form) | $\tilde\epsilon_\theta = \epsilon_\theta(\mathbf{x}_t,\varnothing) + w\big(\epsilon_\theta(\mathbf{x}_t,c) - \epsilon_\theta(\mathbf{x}_t,\varnothing)\big)$ |
| Relation between them | $w = 1 + s$ |
| Core idea | **one** diffusion model trained to do conditional *and* unconditional generation simultaneously |
| The training trick | drop the condition with fixed probability (deck: **10 to 20 percent**) and replace it with the **null token $\varnothing$** |
| $\varnothing$ | the null token — "represents absence of conditioning information" |
| The Bayes identity | $\nabla_{\mathbf{x}_t}\log p(y\mid\mathbf{x}_t) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid y) - \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$ (**minus**; the deck's box prints $+$) |
| The deck's one-liner | "CFG approximates this classifier gradient using the diffusion model itself" |
| Guidance score, the two cases | classifier guidance: **classifier gradient**; CFG: **conditional − unconditional difference** |
| Sampling cost, the two cases | classifier guidance: **U-Net + classifier**; CFG: **two U-Net passes** |
| $w=0$ | unconditional sampling |
| $w=1$ ($s=0$) | plain conditional sampling — **guidance off** |
| $w>1$ | guidance: fidelity ↑, diversity ↓ |
| The paper | Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop |

## [Lec 52 — Stable Diffusion (Latent Diffusion Models)](../notes/52-stable-diffusion.md)

| Item | Exactly this |
|---|---|
| Core idea | compress the image into a compact latent, and **perform diffusion in latent space** |
| LDM vs Stable Diffusion | LDM is the **algorithm** (Rombach et al., CVPR 2022); Stable Diffusion is the **open-source implementation** for text-to-image |
| Deck's pixel count | a $512\times512\times3$ image = **786,432** values |
| Latent size | **$64\times64$** for a $512\times512$ image (4 channels → 16,384 values, **48×** fewer) |
| Encoder / decoder | $\mathbf{z} = \mathcal{E}(\mathbf{x})$ and $\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$ |
| Training order | the **autoencoder is trained before diffusion and then frozen**; only the U-Net is trained in stage 2 |
| Objective | $\mathcal{L} = \mathbb{E}\big[\lVert\epsilon - \epsilon_\theta(\mathbf{z}_t,t)\rVert^2\big]$ — the DDPM loss on latents |
| Text embedding | $\mathbf{c} = \tau_\theta(y)$, from a **pretrained CLIP text encoder** |
| **Where text enters** | **cross-attention, inside the U-Net, at several resolutions** |
| Cross-attention projections | $\mathbf{Q}$ from the **U-Net feature map**; $\mathbf{K}$ and $\mathbf{V}$ from the **text embedding** |
| Attention formula | $\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\big(\mathbf{Q}\mathbf{K}^\top/\sqrt{d}\big)\mathbf{V}$ |
| Method 1 | **concatenation** — for conditions with spatial information (segmentation map, low-resolution image) |
| Method 2 | **cross-attention** — for text embeddings, which have no spatial dimensions |
| Conditioning modalities | text prompts · semantic maps · images |
| The three (four) components | VAE encoder + decoder · U-Net denoiser · text encoder |
| Advantage over DDPM | noise is added to a compact **"blueprint"** instead of the photograph; "much more computationally efficient" |

## [Lec 53 — Hands-on: Reverse Diffusion](../notes/53-reverse-diffusion-handson.md)

| Item | Exactly this |
|---|---|
| Reverse mean | $\mu_\theta(\mathbf{x}_t,t) = \dfrac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t,t,y)\right)$ |
| Reverse sample | $\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t,t) + \sqrt{\tilde\beta_t}\,\mathbf{z}$, $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Posterior variance | $\tilde\beta_t = \beta_t\dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$, with $\bar\alpha_{-1}=1$ |
| Last step | **no extra noise is added**; equivalently $\tilde\beta$ is 0 there |
| Loop direction | $t = T-1, T-2, \ldots, 1, 0$ (`reversed(range(T))`) |
| Starting point | $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ |
| Training loss | $\mathcal{L}_{\text{simple}} = \mathbb{E}\big[\lVert\epsilon - \epsilon_\theta(\mathbf{x}_t,t,y)\rVert_2^2\big]$ |
| Classifier-free mix | $\hat\epsilon = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ |
| Null class index | `num_classes` — one extra embedding row |
| Display map | $(\text{clamp}(\mathbf{x},-1,1)+1)/2$ |
| DDIM step | $\hat{\mathbf{x}}_0 = \dfrac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}}$, then $\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar\alpha_{t-1}}\hat\epsilon$ — **no $\mathbf{z}$** |

## [Lec 54 — Foundations of NLP](../notes/54-nlp-foundations.md)

| Item | Exactly this |
|---|---|
| NLP, the deck's definition | a technology that allows computers to **interpret, manipulate and comprehend** human language |
| The pipeline | raw text → text preprocessing → text representation → deep learning model → prediction/output |
| Preprocessing chain | lowercasing → remove punctuation & special characters → tokenization → stop-word removal → lemmatization → clean text |
| Tokenization | splitting text into smaller units called **tokens** |
| The four granularities | word · sentence · subword · character |
| Subword tokenization is used by | **modern LLMs such as BERT and GPT** |
| Stop words | frequently occurring words (*the, is, are, of, and*) carrying little semantic meaning |
| Lemma | a word's base or dictionary form, meaning preserved |
| Traditional representations | one-hot encoding, bag of words, TF-IDF |
| Modern representations | Word2Vec, GloVe (static); BERT, GPT (contextual) |
| Word2Vec core idea | words in similar contexts get similar vectors; learning from **local context windows** |
| The analogy | $(\text{King}-\text{Man})+\text{Woman}\approx\text{Queen}$ |
| GloVe core idea | words frequently co-occurring in a large corpus likely share meaning; dense embeddings from a **word co-occurrence matrix** |
| TF-IDF | $\text{tf}(t,d)\cdot\ln(N/\text{df}(t))$ |
| Cosine similarity | $\cos(\mathbf{v}_a,\mathbf{v}_b) = \dfrac{\mathbf{v}_a\cdot\mathbf{v}_b}{\lVert\mathbf{v}_a\rVert\lVert\mathbf{v}_b\rVert}$ |
| The four challenges | discrete nature · evolving language · ambiguity/vagueness · long-range dependency |

## [Lec 55 — Sequential Modeling with RNNs and LSTMs](../notes/55-rnn-lstm.md)

| Item | Exactly this |
|---|---|
| RNN hidden state | $\mathbf{h}_t = \tanh(\mathbf{V}\mathbf{h}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b}_h)$ |
| RNN output | $\hat{\mathbf{y}}_t = g(\mathbf{W}\mathbf{h}_t + \mathbf{b}_y)$ |
| This deck's matrices | $\mathbf{U}$ input→hidden, $\mathbf{V}$ hidden→hidden, $\mathbf{W}$ hidden→output |
| Weight sharing | the same $\mathbf{U},\mathbf{V},\mathbf{W}$ at every timestep |
| Gradient path | $\partial\mathbf{h}_T/\partial\mathbf{h}_1 = \prod_t \operatorname{diag}(1-\mathbf{h}_t^2)\mathbf{V}$ |
| Forget gate | $\mathbf{f}_t = \sigma(\mathbf{W}_f\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f)$ |
| Input gate | $\mathbf{i}_t = \sigma(\mathbf{W}_i\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i)$ |
| Candidate cell | $\widehat{\mathbf{C}}_t = \tanh(\mathbf{W}_C\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_C)$ |
| **Cell-state update** | $\mathbf{C}_t = \mathbf{f}_t\odot\mathbf{C}_{t-1} + \mathbf{i}_t\odot\widehat{\mathbf{C}}_t$ |
| Output gate | $\mathbf{o}_t = \sigma(\mathbf{W}_o\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o)$ |
| Hidden state out | $\mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{C}_t)$ |
| Cell-state gradient | $\partial\mathbf{C}_t/\partial\mathbf{C}_{t-1} = \operatorname{diag}(\mathbf{f}_t)$ |
| Gate activations | three sigmoids $(\mathbf{f},\mathbf{i},\mathbf{o})$, two $\tanh$s (candidate, and the squash in $\mathbf{h}_t$) |
| The deck's gate verbs | **retain** (forget), **update** (input), **expose** (output) |
| Feed-forward verdict | "Feed-forward networks have no memory of previous inputs" |

## [Lec 56 — Evolution From LSTMs to Transformers](../notes/56-lstm-to-transformer.md)

| Item | Exactly this |
|---|---|
| Attention weight meaning | $\alpha_{t,j}$ = attention paid to the $j$-th **input** word while generating the $t$-th **output** word |
| Normalisation | $\sum_{j=1}^{T}\alpha_{t,j} = 1$ for every output step $t$ |
| How the weights are formed | softmax over alignment scores, $\alpha_{t,j} = \exp(e_{t,j}) / \sum_k \exp(e_{t,k})$ |
| **Context vector** | $\mathbf{c}_t = \sum_{j=1}^{T}\alpha_{t,j}\,\mathbf{h}_j$ — a fresh one per output word |
| Softmax (deck's page 11) | $\sigma(\mathbf{z})_i = e^{z_i} / \sum_{j=1}^{K} e^{z_j}$ |
| The three limitations | sequential dependency · memory constraints · long-range dependency |
| Path length claim | information between distant words travels $O(n)$ sequential steps; 1000 words ⇒ 1000 operations |
| The bottleneck's name here | the **thought vector** — "information about the input sequence in compressed form" |
| Attention's motivating question | "do we really need every previous hidden state equally?" — **No** |
| The landmark paper | *Attention Is All You Need*, Vaswani et al., Google Brain, **2017** |
| Attention's own date | **2014** — three years before the Transformer |
| The core idea of the Transformer | "Instead of reading one word after another, it looks at all words simultaneously" |
| Self-attention, in one line | the same weighted-sum machinery with source = target; built in [Lec 57](../notes/57-transformer-encoder.md) |

## [Lec 57 — Transformer Encoder](../notes/57-transformer-encoder.md)

| Item | Exactly this |
|---|---|
| Projections | $\mathbf{Q}=\mathbf{X}\mathbf{W}^{Q}$, $\mathbf{K}=\mathbf{X}\mathbf{W}^{K}$, $\mathbf{V}=\mathbf{X}\mathbf{W}^{V}$ |
| Scaled dot-product attention | $\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ |
| One attention weight | $\alpha_{ij}=\dfrac{\exp(\mathbf{q}_i\cdot\mathbf{k}_j/\sqrt{d_k})}{\sum_{j'}\exp(\mathbf{q}_i\cdot\mathbf{k}_{j'}/\sqrt{d_k})}$ |
| One output | $\mathbf{c}_i=\sum_{j=1}^{n}\alpha_{ij}\mathbf{v}_j$ |
| Why $\sqrt{d_k}$ | $\text{Var}(\mathbf{q}\cdot\mathbf{k})=d_k$ for unit-variance components; dividing by $\sqrt{d_k}$ restores unit variance and keeps softmax out of saturation |
| Multi-head | $\text{Concat}(\text{head}_1,\ldots,\text{head}_h)\mathbf{W}^{O}$, $\text{head}_m=\text{Attention}(\mathbf{X}\mathbf{W}^{Q}_m,\mathbf{X}\mathbf{W}^{K}_m,\mathbf{X}\mathbf{W}^{V}_m)$ |
| Head width | $d_k=d_v=d_{\text{model}}/h$ |
| Positional encoding, even | $\text{PE}(pos,2i)=\sin\!\left(\dfrac{pos}{10000^{2i/d_{\text{model}}}}\right)$ |
| Positional encoding, odd | $\text{PE}(pos,2i+1)=\cos\!\left(\dfrac{pos}{10000^{2i/d_{\text{model}}}}\right)$ |
| How PE enters | **added** to the embedding, once, before layer 1 |
| Sublayer wrapper | $\text{LayerNorm}(\mathbf{x}+\text{Sublayer}(\mathbf{x}))$ |
| Layer norm | $\gamma\odot\dfrac{\mathbf{z}-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, with $\mu,\sigma^2$ over the $d_{\text{model}}$ features of **one token** |
| FFN | $\text{FFN}(\mathbf{x})=\max(0,\mathbf{x}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2$ |
| Encoder block order | self-attention → Add & Norm → FFN → Add & Norm |
| Q, K, V in one line each | Query = what I am looking for · Key = what I advertise · Value = what I contribute |

## [Lec 59 — Transformer Decoder](../notes/59-transformer-decoder.md)

| Item | Exactly this |
|---|---|
| Masked attention | $\text{softmax}\!\left(\dfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$ |
| The mask | $M_{ij} = 0$ if $j\le i$, $-\infty$ if $j > i$ — strictly upper triangle is $-\infty$ |
| Why it works | $e^{-\infty}=0$, so masked weights are exactly 0 and the rest renormalise to sum to 1 |
| Resulting $\boldsymbol{\alpha}$ | lower triangular; row 1 is always $(1,0,\ldots,0)$; last row unmasked |
| Cross-attention sources | $\mathbf{Q}$ from the **decoder**; $\mathbf{K}$ and $\mathbf{V}$ from the **encoder output** |
| Cross-attention shape | $\mathbf{Q}: m\times d_k$, $\mathbf{K}: n\times d_k$ → scores $m \times n$ (rectangular) |
| Cross-attention masking | **none** — the whole source is visible |
| Decoder sublayers, in order | masked multi-head self-attention → Add & Norm → encoder–decoder attention → Add & Norm → position-wise FFN → Add & Norm |
| Output head | Linear ($d_{\text{model}}\times\lvert V\rvert$) then softmax over the vocabulary |
| Heads | 8 original, 16 Transformer-Big; all heads live in **one** sublayer |
| Training | teacher forcing, all $m$ positions in **1** parallel pass |
| Inference | $m$ sequential passes, one token each |
| BPE merge rule | merge the most **frequent** adjacent pair, repeat |
| WordPiece merge rule | merge the pair giving the greatest **likelihood** gain; `##` marks a continuation |
| Encoder vs decoder sublayer count | 2 vs 3 |

## [Lec 60 — BERT (Bidirectional Encoder Representations from Transformers)](../notes/60-bert.md)

| Item | Exactly this |
|---|---|
| BERT expands to | **B**idirectional **E**ncoder **R**epresentations from **T**ransformers |
| Origin | Google, **2018** (paper published NAACL **2019**, Devlin et al.) |
| Architecture | **encoder-only** — the [Lec 57](../notes/57-transformer-encoder.md) stack, no decoder, no cross-attention |
| Primary goal | understand language, **not** generate language |
| Pre-training objectives | **MLM + NSP**, trained jointly |
| MLM selection rate | **15%** of tokens |
| MLM 80/10/10 | 80% → `[MASK]`, 10% → random word, 10% → unchanged |
| Reason for 80/10/10 | `[MASK]` never appears at fine-tuning or inference time — reduce the train/test mismatch |
| MLM loss | $\mathcal{L}_{\text{MLM}} = -\frac{1}{\lvert M\rvert}\sum_{i\in M}\log P(x_i \mid \mathbf{x}_{\setminus M})$ |
| NSP split | **50% IsNext / 50% NotNext** |
| NSP head reads | the final vector of `[CLS]` |
| Joint loss | $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$, no weighting |
| `[CLS]` | first token of every input; its final embedding represents the whole sequence |
| `[SEP]` | segment separator; appears **twice** in a two-sentence input |
| Input embedding | **token + position + segment**, summed |
| Tokenizer | **WordPiece** ([Lec 59](../notes/59-transformer-decoder.md)) |
| Pre-training corpus | **BooksCorpus + Wikipedia** |
| Pre-training vs fine-tuning | pre-training is **self-supervised**, fine-tuning is **supervised** |
| What fine-tuning updates | a small new output layer **and the entire BERT model**, together |
| Static vs contextual | one vector per word **type** vs one per word **occurrence** |

## [Lec 61 — GPT (Generative Pre-trained Transformer)](../notes/61-gpt.md)

| Item | Exactly this |
|---|---|
| GPT expands to | **G**enerative **P**re-trained **T**ransformer |
| Architecture | **decoder-only** Transformer |
| What decoder-only deletes | the **cross-attention** sublayer; masked self-attention and the FFN survive |
| Training objective | **next-token prediction** (causal / autoregressive language modelling) |
| The factorisation | $P(x_1,\ldots,x_n) = \prod_{t=1}^{n} P(x_t \mid x_1,\ldots,x_{t-1})$ |
| The loss | $\mathcal{L} = -\frac{1}{n}\sum_{t}\log P(x_t \mid x_{<t})$, natural log |
| Why not an encoder | bidirectional attention lets the model see the token it must predict — **information leakage** |
| Attention type | **causal masked** self-attention |
| Stage 1 | generative pre-training, **self-supervised**, on unlabelled text |
| Stage 2 | **supervised** fine-tuning; **the pre-trained weights are updated** |
| GPT-1's delimiter tokens | `Start`, `Delim`, `Extract`; the prediction is read off **`Extract`** at the **end** |
| GPT-1 reference | Radford, Narasimhan, Salimans, Sutskever, *Improving Language Understanding by Generative Pre-Training*, OpenAI, **2018** |
| In-context learning | task performed from examples **in the prompt**, with **no weight update** |
| Zero / one / few-shot | $k = 0$ / $k = 1$ / $k$ typically 10–100 demonstrations in the prompt |
| The series | GPT-1 2018 · GPT-2 2019 · GPT-3 2020 · GPT-3.5 2022 · GPT-4 2023 · GPT-4o 2024 · GPT-5 2025 |
| Seven limitations | hallucinations · bias · computational cost · context window · prompt sensitivity · knowledge cutoff · safety |

## [Lec 62 — Prompt Engineering Basics](../notes/62-prompt-engineering.md)

| Item | Exactly this |
|---|---|
| Prompt (deck's definition) | "the input given to an AI model that guides its response" |
| Prompt engineering (deck's definition) | "the process of designing effective inputs for LLMs" |
| What a prompt mechanically is | a **prefix** that conditions $P(x_t \mid x_{<t};\theta)$ — weights $\theta$ never change |
| Generation | $P(\mathbf{y}\mid\mathbf{p}) = \prod_{i=1}^{m} P(y_i \mid \mathbf{p}, y_{<i})$ |
| A prompt may include (5) | instructions · questions · context · constraints · examples |
| The four slots | instruction · context · input · output indicator |
| Good prompt (5) | clear · context-rich · unambiguous · goal-oriented · specific |
| Avoid (3) | vague instructions · missing context · multiple unrelated questions |
| Why prompts matter (5) | define the task · provide context · specify the audience · control the output format · reduce ambiguity |
| Six techniques, deck's order | zero-shot · one-shot · few-shot · chain-of-thought · structured output · role |
| Zero-shot | **no** examples; instruction + pretrained knowledge only |
| One-shot | **exactly one** input–output example |
| Few-shot | **2–5** examples (deck's number) |
| Chain-of-thought | model reasons step by step **before** the final answer; trigger "Think step by step" |
| Role prompting | assign a role/profession/persona; adapts **tone, detail, vocabulary, perspective** |
| Structured output | user specifies the **exact format** (e.g. JSON) of the response |

## [Lec 63 — Hands-on on LLM](../notes/63-llm-handson.md)

| Item | Exactly this |
|---|---|
| Causal LM objective | predict $x_{t+1}$ at every position $t$; loss is cross-entropy, **natural log** |
| Target construction | target $=$ input **shifted left by one token** |
| Causal mask | `torch.triu(ones, diagonal=1)` → $-\infty$ **before** the softmax |
| Why `diagonal=1` | position $t$ may attend to itself; only strictly future positions are hidden |
| Encoder layer, decoder model | `TransformerEncoderLayer` + causal mask + no cross-attention $=$ decoder-only |
| Attention sublayer cost | $4d_{\text{model}}^2$ weights $+\,4d_{\text{model}}$ biases, **independent of $h$** |
| Context-window mechanism | learned `nn.Embedding(block_size, d)` — **no row beyond `block_size`**, so it raises |
| Temperature | $P_T(v) = \exp(z_v/T)\big/\sum_u \exp(z_u/T)$, applied to **logits**, **before** truncation |
| $T<1$ / $T>1$ | sharper, more repetitive / flatter, more diverse |
| Greedy | $\arg\max$ over logits; the $T\to0$ limit; **deterministic** |
| Top-$k$ | keep the $k$ largest logits, set the rest to $-\infty$, renormalise, sample |
| Top-$p$ (nucleus) | keep the **smallest set whose cumulative probability reaches $p$**, renormalise, sample |
| top-$k$ vs top-$p$ | fixed **count**, floating coverage · fixed **coverage**, floating count |
| Generation uses | `logits[:, -1, :]` — the **last** position only |
| Context window | `ids[:, -block_size:]` — a sliding window; older tokens are dropped |
| FFN width | $d_{\text{ff}} = 4\,d_{\text{model}}$ |
| $d_k$ | $d_{\text{model}}/h = 128/4 = 32$ |
| Perplexity | $e^{\mathcal{L}}$ when $\mathcal{L}$ is in nats |
| Uniform-guess loss | $\log V$ nats |
| Byte-level BPE marker · WordPiece marker | `Ġ` = leading space · `##` = word continuation |
| BPE vs WordPiece merge criterion | **frequency** vs **likelihood** (see [Lec 59](../notes/59-transformer-decoder.md)) |

## [Lec 64 — LLM Recap, In-Context Learning, LoRA, and the RAG Principle](../notes/64-llm-icl-lora-rag.md)

| Item | Exactly this |
|---|---|
| Deck's self-attention | $\mathbf{A} = \mathbf{Q}\mathbf{K}^\top$, $\mathbf{E} = \mathrm{softmax}(\mathbf{A}/d)\mathbf{V}$ (**slide**); correct is $/\sqrt{d_k}$ |
| Deck's head relation | $d = h\times d_k$ |
| Deck's decoder-only loss | $\mathcal{L} = -\sum_t P(\mathbf{x}_t\mid\mathbf{x}_{<t})$ (**slide**); correct is $-\sum_t \log P(\mathbf{x}_t\mid\mathbf{x}_{<t})$ |
| Encoder-only vs decoder-only | "the difference is in the loss function" (plus the causal mask) |
| Layer norm | normalises **across the feature dimension(s)** |
| Residual connections | "increases the gradient flow and stability" |
| The two frozen-weight routes | parameters frozen → ICL; small number of new parameters → LoRA |
| ICL input form | $(\mathcal{X}_1,\mathcal{Y}_1),\ldots,(\mathcal{X}_n,\mathcal{Y}_n),(\mathcal{X}_{n+1},\cdots)$; model learns $f(\mathcal{X})\to\mathcal{Y}$; **weights frozen** |
| Three ICL types | zero-shot (instruction only) · one-shot (1 example) · few-shot (several) |
| Why ICL works | each layer builds an internal representation of the context; self-attention "dynamically retrieves" relevant examples |
| ICL ≈ which algorithm | ridge regression, $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| LoRA decomposition | $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$, $\mathbf{A}\in\mathcal{R}^{r\times d}$, $\mathbf{B}\in\mathcal{R}^{D\times r}$, $r\ll\min(D,d)$ |
| LoRA update | $\mathbf{W} = \mathbf{W}_0 + \alpha\cdot\mathbf{B}\mathbf{A}$ (**slide**); $\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ (paper, notebook) |
| LoRA parameter counts | full $Dd$; LoRA $r(D+d)$; fraction $r(1/d + 1/D)$ |
| Where LoRA goes | **Query and Value** projections of self-attention; rarely feed-forward |
| Why LoRA works | downstream tasks need "specific but limited shifts in representation space" |
| LoRA's category | parameter efficient fine-tuning (**PEFT**) |
| RAG principle | retrieve top-$k$ → augment the prompt → generate grounded; non-parametric memory beside frozen parametric memory |

## [Lec 68 — Retrieval Augmented Generation: Mechanics and Advances](../notes/68-rag-advances.md)

| Item | Exactly this |
|---|---|
| Parametric memory | "knowledge that's baked into a model's weights during training"; frozen at training time, can't be updated without retraining or fine-tuning |
| Its four limitations | not aware of constant happenings in the world · unable to locate the source of information · hallucinations · capacity is fixed and finite |
| Non-parametric memory buys | allowing information update · increase capacity · reduce hallucinations |
| The pipeline | Input Prompt → Retriever ↔ Knowledge Base → Retrieved Documents → Augmented Prompt → LLM → Generated Response |
| Augmentation is | "retrieved snippets **appended** to the input query" |
| RAG, exactly | $P(y\mid x) = \sum_{d\in\mathcal{C}} P(y\mid x,d)\cdot P(d\mid x)$ |
| RAG, in practice | $P(y\mid x) \approx \sum_{i=1}^{k} P(y\mid x,d_i)\cdot P(d_i\mid x)$ |
| Which factor is the retriever | $P(d\mid x)$ — the **second** one |
| MIPS | Maximum Inner Product Search |
| Encoders | $q(x) = \mathrm{Enc}_q(x)\in\mathbb{R}^h$, $d(z) = \mathrm{Enc}_d(z)\in\mathbb{R}^h$ |
| Score | $\mathrm{score}(x,z) = q(x)^\top d(z)$ |
| Top-$k$ set | $\mathcal{Z}_k(x) = \operatorname{top-\mathit{k}}_{z\in\mathcal{Z}} q(x)^\top d(z)$ |
| Retriever distribution | $p_\eta(z\mid x) = \exp(q(x)^\top d(z)) \big/ \sum_{z'\in\mathcal{Z}}\exp(q(x)^\top d(z'))$ |
| Query encoding rule | a BERT encoder; the query is the **average** of the token representations |
| What is trained | "end-to-end backprop through $q$ and $p_\theta$" — **not** the document encoder |
| Retriever / generator labels | retriever $p_\eta$ is **non-parametric**; generator $p_\theta$ is **parametric** |
| Agentic AI | "pursues a goal through its own loop of reasoning and action — instead of answering one prompt and stopping" |
| The agent loop | Goal → Agent → Action (call a tool) → Observation → loop → Final Output |
| Agentic RAG, the one difference | the model decides: how many passes, how to phrase the query, which tool, when to stop |
| Graph RAG | "retrieves from a knowledge graph built out of the corpus — connected entities and summaries of whole clusters of them" |
| Graph RAG indexing | Documents → Chunks → Extract entities & relationships → Knowledge graph → Communities (**Leiden**) → Community summaries |
| Graph RAG retrieval modes | a graph **neighbourhood** (local) or the **community summaries** (global) |
| Where the money goes | GraphRAG at **index** time; agentic RAG at **query** time |

## [Lec 70–71 — LLMs for Text Generation and Multimodal Alignment](../notes/70-llm-generation-multimodal.md)

| Item | Exactly this |
|---|---|
| Decoding | generating outputs from an LLM |
| Greedy decoding | $y_t = \arg\max_i p(y_t = i \mid y_{<t}, x)$ |
| Beam-search score | $\sum_{s=1}^{t}\log p(y_s \mid y_{<s}, x)$ |
| Beam search at $B=1$ | **is greedy decoding** |
| Temperature softmax | $p(i) = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$ |
| $T<1$ / $T=1$ / $T>1$ | sharper / unchanged / flatter |
| $T \to 0^+$ | becomes greedy decoding |
| Probability ratio under $T$ | $p(i)/p(j) = \exp\big((z_i - z_j)/T\big)$ |
| Multinomial sampling | draw $U \sim \mathrm{Uniform}[0,1)$, return the first token whose **cumulative** probability exceeds $U$ |
| Top-$k$ | keep the $k$ most probable tokens, renormalise, sample |
| Top-$p$ (nucleus) | keep the **smallest** set whose cumulative probability **reaches** $p$, renormalise, sample |
| Pipeline order | temperature → top-$k$ → top-$p$ → sample |
| BPE merge saving | a pair of frequency $f$ removes exactly $f$ tokens |
| English token length | **4–5 characters** |
| Multimodal three steps | modality-specific encoder → modality alignment → pre-training by **next-token prediction** |
| Multimodal objective | unchanged next-token cross-entropy — no new loss term |
| CLIP loss | symmetric cross-entropy over an $N\times N$ cosine-similarity matrix, both rows and columns |

## [Lec 72–73 — Size versus Performance, Benchmarks, Bias and Safety](../notes/72-benchmarking-bias-safety.md)

| Item | Exactly this |
|---|---|
| Active vs total parameters | **active** = what a token's compute costs; **total** = what the memory costs |
| Why the link broke | every frontier open model is a **sparse mixture of experts** |
| Three explanations on p. 25 | sparsity · distillation and data · **test-time compute** |
| Compute identity | $C \approx 6ND$ ($N$ params, $D$ tokens; 2 forward + 4 backward) |
| Chinchilla rule | $D \approx 20N$ tokens per parameter |
| Kaplan vs Chinchilla | Kaplan: scale $N$ hardest. Chinchilla: scale $N$ and $D$ **equally** |
| Four benchmark families | static datasets · human judgement · agentic · safety/adversarial |
| Static-benchmark cons | **saturated, contaminated, no interactive behaviour** |
| Human-judgement instruments | **pairwise comparison** and the **1–7 Likert scale** |
| Agentic metrics | task success rate · steps / tool calls · cost, latency, error recovery |
| Safety domains | toxicity/hate · jailbreak / prompt injection / tool misuse · privacy leaks |
| Safety approaches | curated adversarial sets · automated attack generators · human red-teaming |
| pass@$k$ | $1 - \binom{n-c}{k}\big/\binom{n}{k}$ |
| Perplexity | $\exp(\mathcal{L})$ with $\mathcal{L}$ the **mean** cross-entropy in nats |
| Contamination inflation | $a_{\text{obs}} = a_{\text{true}} + f(1 - a_{\text{true}})$ |
| Five bias stages | corpus · annotation · alignment · inference · deployment |
| The two harms | **representational** (stereotype, erasure) vs **allocational** (unequal service where something is at stake) |
| The three-step order | **Measure → Mitigate → Govern** |
| The one-line Measure rule | publish the **worst-group** score beside the mean |
| The one-line Govern rule | fairness is a **value judgement**; name the definition |
| Contamination defence | a **private held-out split** |
| An agent | model + tools + memory + a loop: observe → plan → act → new state |
| Why an agent is a security problem | holds credentials · **data channel = instruction channel** · no human approves each step |
| A swarm | many agents, one goal, a shared channel; behaviour is **emergent** |
