# Supplementary — Topics the Slides Name But Never Teach

> **Deck:** none — assembled by the editor · **Covers:** gaps found across Weeks 1–8
> **Prereqs:** [Lec 9 — Backpropagation](week-02/09-backpropagation.md), [Lec 13 — Vanishing Gradients](week-03/13-vanishing-gradients-activations.md)
> **Feeds into:** everything from Week 3 onward

## Why this chapter exists

While writing the thirty lecture chapters, six topics kept surfacing in the same way: a slide names
them in a bullet list of "solutions" or "challenges", the lecturer moves on, and no deck ever explains
them. `W3L4_P3` slide 26 is the worst offender — it lists batch normalization, parameter
initialisation, dropout, weight sharing and data augmentation as the answers to deep-network problems,
and teaches none of them.

That is a problem for you specifically, because a bulleted list on a slide is exactly the shape of an
NPTEL multiple-choice question. "Which of the following addresses internal covariate shift?" is a fair
question against this syllabus, and nothing in the lectures prepares you for it. Everything below is
flagged as **not derivable from the decks** — it is here because the exam can reach it, not because
the lecturer covered it.

## The ideas

### Batch normalization and internal covariate shift

**Where it appears:** `W3L4_P3` slide 26 (as a solution), `W4L5_P1` slide 4 (internal covariate shift
as a challenge). Taught nowhere.

The problem it claims to solve: as training updates the weights of layer $l$, the *distribution* of
inputs arriving at layer $l+1$ keeps shifting. Every layer is chasing a moving target, so you must use
a small learning rate and training is slow. Ioffe and Szegedy (2015) named this **internal covariate
shift**.

Batch normalization normalises each feature across the **mini-batch**, then rescales with two learned
parameters:

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta$$

where $\mu_{\mathcal{B}}$ and $\sigma^2_{\mathcal{B}}$ are the mean and variance of that feature over
the batch, $\epsilon$ is a small constant preventing division by zero, and $\gamma, \beta$ are
**learned**. Those two parameters matter: without them, normalisation would force every layer's output
to be zero-mean unit-variance, which throws away representational capacity. With them, the network can
learn to undo the normalisation if that is what it wants — including recovering the identity by setting
$\gamma = \sigma_{\mathcal{B}}$ and $\beta = \mu_{\mathcal{B}}$.

What it buys you: much higher learning rates, faster convergence, reduced sensitivity to
initialisation, and a mild regularising effect (each sample's normalisation depends on which other
samples landed in its batch, which injects noise).

**Train versus test behaviour is the examinable part.** At training time BN uses the current batch's
statistics. At test time there may be no batch — you might be classifying one image — so BN instead
uses a **running average** of the mean and variance accumulated during training. Forgetting to switch
modes (`model.eval()` in PyTorch) is the classic bug, and "what does BN use at inference time" is the
classic question.

> **Honesty note:** the "internal covariate shift" explanation is the one the slides gesture at and the
> one an exam will mark correct. Later work (Santurkar et al., 2018) showed BN's real benefit is
> smoothing the optimisation landscape, and that reducing covariate shift explains little. Answer the
> exam with covariate shift; know that the story is contested.

**BatchNorm vs LayerNorm** — you need this contrast because [Lec 27](week-07/27-decoder-and-full-transformer.md)
explains why Transformers use layer norm:

| | Batch norm | Layer norm |
|---|---|---|
| Normalises across | the **batch**, per feature | the **features**, per sample |
| Depends on batch size | yes — breaks at batch size 1 | no |
| Variable-length sequences | awkward | fine |
| Train/test behaviour | **differs** (running averages) | identical |
| Standard in | CNNs | Transformers, RNNs |

### Weight initialization: Xavier and He

**Where it appears:** `W3L4_P3` slide 7 ("parameter initialisation"). Taught nowhere.

This is load-bearing for [Lec 13](week-03/13-vanishing-gradients-activations.md), which shows the
gradient decaying as a product of per-layer factors. That chapter writes the factor as the activation
derivative, but the honest version is

$$\text{per-layer factor} \approx f'(z) \cdot \|\mathbf{W}\|$$

so the **weights' scale matters just as much as the activation's derivative**. Initialise too small and
the signal dies going forward; too large and it explodes. Both break training before it starts.

Why you cannot just use $\mathcal{N}(0, 1)$: the pre-activation $z = \sum_{i=1}^{n} w_i x_i$ sums $n$
terms, so its variance grows with $n$. A layer with 1000 inputs produces pre-activations roughly
$\sqrt{1000} \approx 32$ times larger than a layer with one input, which drives sigmoid and tanh
straight into saturation.

The fix is to scale the initial variance by the fan-in:

- **Xavier / Glorot initialization** — for **sigmoid and tanh**:
  $$\mathrm{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \quad\text{or, fan-in only,}\quad \frac{1}{n_{\text{in}}}$$
  Derived by requiring that variance be preserved in both the forward and backward passes.

- **He initialization** — for **ReLU**:
  $$\mathrm{Var}(w) = \frac{2}{n_{\text{in}}}$$
  The factor of 2 compensates for ReLU zeroing out roughly half its inputs, halving the variance.

**Matching the initialiser to the activation is the whole point, and is the examinable fact: Xavier
with tanh, He with ReLU.** Also note that initialising all weights to **zero** is fatal — every unit in
a layer then computes the same thing and receives the same gradient forever, so they never
differentiate. This is the **symmetry-breaking** problem, and it is why initialisation must be random.
Biases, by contrast, are safely initialised to zero.

### Gradient descent: batch, stochastic, and mini-batch

**Where it appears:** assumed throughout — [Lec 9](week-02/09-backpropagation.md) quotes a complexity
of $O(B \cdot W)$ for an undefined batch size $B$. Defined nowhere.

Three variants, distinguished only by how many samples you use per update:

| Variant | Samples per update | Updates per epoch | Character |
|---|---|---|---|
| **Batch (full-batch)** | all $N$ | 1 | smooth, slow, exact gradient, needs all data in memory |
| **Stochastic (SGD)** | 1 | $N$ | very noisy, fast updates, can escape shallow minima |
| **Mini-batch** | $B$ (typically 32–256) | $N/B$ | the practical compromise — what everyone actually uses |

"SGD" in modern usage almost always means mini-batch gradient descent. The noise in stochastic updates
is not purely a cost: it helps escape saddle points, which dominate high-dimensional loss surfaces far
more than local minima do.

### Optimizers beyond plain SGD

**Where they appear:** nowhere in Weeks 1–8. Included because any question about training deep networks
can reach them, and because you will meet them the moment you write real code.

Plain SGD updates $\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}$. Three refinements:

- **Momentum** accumulates an exponentially-weighted average of past gradients:
  $$\mathbf{v}_t = \beta\mathbf{v}_{t-1} + (1-\beta)\nabla\mathcal{L}, \qquad \theta \leftarrow \theta - \eta\mathbf{v}_t$$
  with $\beta \approx 0.9$. It damps oscillation across steep ravine walls and accelerates along the
  consistent direction — the ball-rolling-downhill picture.

- **RMSProp** divides the step by a running root-mean-square of recent gradients, giving each parameter
  its own effective learning rate. Parameters with consistently large gradients get smaller steps.

- **Adam** (Adaptive Moment Estimation) combines both: a first moment (momentum) and a second moment
  (RMSProp), each bias-corrected for their zero initialisation. Defaults $\beta_1 = 0.9$,
  $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ are near-universal. **Adam is the default optimizer for
  essentially every model in this course** — Transformers and VAEs in particular are trained with it.

### Dropout

**Where it appears:** `W3L4_P3` slide 26, and as one of AlexNet's innovations in
[Lec 12](week-03/12-cnn-architectures.md). The [Lec 10](week-02/10-overfitting-and-regularization.md)
deck — the regularization lecture — does not mention it at all, so it is covered here.

During training, randomly zero each unit's output with probability $p$ (commonly 0.5 for fully
connected layers, lower for convolutional ones). At test time, use all units.

Why it regularises: no unit can rely on any particular other unit being present, so the network cannot
build fragile co-adapted chains of neurons. Each forward pass effectively trains a different thinned
sub-network, and testing with all units approximates averaging an exponentially large ensemble of them.

**The scaling detail is the examinable part.** Training with dropout and testing without it would make
test-time activations systematically larger. Two fixes: scale activations by $p$ at test time
(original formulation), or — what every framework actually does — divide by $p$ during training
("inverted dropout") so that test time needs no adjustment at all.

Like batch norm, **dropout behaves differently at train and test time**, which is the other half of why
`model.eval()` exists.

### Data augmentation

**Where it appears:** `W3L4_P3` slide 26. Taught nowhere, despite being conceptually central to a
course about generating synthetic training data.

Apply label-preserving transformations to training images to enlarge the effective dataset: random
crops, horizontal flips, rotations, scaling, colour jitter, random erasing. The point is to bake known
invariances into the model — if a cat is still a cat when mirrored, show the network both.

**Connect this to the course's thesis.** Classical augmentation applies hand-written transformations to
images you already have. Generative augmentation — the subject of
[Lec 2](week-01/02-generative-vs-discriminative.md) and [Lec 20](week-05/20-genai-vision-tasks-1.md) —
synthesises genuinely new images. The first cannot add information that was not in your dataset; the
second can, if the generator has learned something true about the data distribution. That distinction
is worth a sentence in any written answer about why generative models matter.

One trap: **horizontal flips are not always label-preserving.** Flip a digit and 2 is no longer 2; flip
a road scene and the traffic drives on the wrong side. Augmentation choices encode assumptions.

## Worked numericals

### N1. Batch normalization applied by hand
**Given:** one feature's activations across a batch of 4: $[2, 4, 6, 8]$, with $\gamma = 2$,
$\beta = 1$, $\epsilon = 0$.
**Find:** the normalised and rescaled outputs.

1. $\mu_{\mathcal{B}} = (2+4+6+8)/4 = 20/4 = 5$.
2. Deviations: $-3, -1, 1, 3$. Squares: $9, 1, 1, 9$, summing to 20.
3. $\sigma^2_{\mathcal{B}} = 20/4 = 5$, so $\sigma_{\mathcal{B}} = \sqrt{5} = 2.2361$.
4. $\hat{x} = [-3, -1, 1, 3]/2.2361 = [-1.3416, -0.4472, 0.4472, 1.3416]$.
5. Check: $\hat{x}$ has mean 0 and variance 1. ✓
6. $y = 2\hat{x} + 1 = [-1.6833, 0.1056, 1.8944, 3.6833]$.

**Answer:** $y = [-1.683, 0.106, 1.894, 3.683]$. Note the output has mean 1 ($=\beta$) and standard
deviation 2 ($=\gamma$) — the learned parameters set the output distribution exactly.

### N2. Xavier and He initialization ranges
**Given:** a layer with $n_{\text{in}} = 500$, $n_{\text{out}} = 100$.
**Find:** the initialisation standard deviation under Xavier and under He.

1. Xavier (both fans): $\mathrm{Var}(w) = 2/(500+100) = 2/600 = 0.003333$.
2. $\sigma_{\text{Xavier}} = \sqrt{0.003333} = 0.05774$.
3. He (fan-in): $\mathrm{Var}(w) = 2/500 = 0.004$.
4. $\sigma_{\text{He}} = \sqrt{0.004} = 0.06325$.
5. Ratio: $0.06325/0.05774 = 1.095$ — He is about 10% wider here.
6. Compare naive $\mathcal{N}(0,1)$: $\sigma = 1$, which is **17×** too large, driving
   pre-activations to roughly $\sqrt{500} \approx 22$ and saturating any sigmoid instantly.

**Answer:** Xavier $\sigma = 0.0577$, He $\sigma = 0.0632$. Both are far smaller than unit variance,
which is the entire point.

### N3. Counting updates under the three gradient-descent variants
**Given:** $N = 50{,}000$ training images, 10 epochs, mini-batch size $B = 64$.
**Find:** parameter updates performed by each variant.

1. Full-batch: 1 update per epoch $\times$ 10 = **10 updates**.
2. SGD (one sample): $50{,}000 \times 10 = $ **500,000 updates**.
3. Mini-batch: $\lceil 50{,}000/64 \rceil = 782$ per epoch, $\times 10 = $ **7,820 updates**.
4. Gradient computations are identical in all three ($500{,}000$ sample-gradients); only the *grouping*
   differs.

**Answer:** 10 / 500,000 / 7,820. Full-batch takes too few steps to converge in reasonable time;
pure SGD pays enormous overhead per sample and cannot use vectorised hardware. Mini-batch wins on both.

### N4. Inverted dropout preserves expected activation
**Given:** a layer output of $[4, 4, 4, 4]$ with dropout $p_{\text{keep}} = 0.5$; suppose units 1 and 3
survive.
**Find:** the output with and without inverted scaling, and the expected value in each case.

1. Naive dropout (train): $[4, 0, 4, 0]$. Mean $= 2$.
2. At test time with no dropout: $[4,4,4,4]$, mean $= 4$. **Mismatch by a factor of 2.**
3. Inverted dropout (train): divide survivors by $p_{\text{keep}} = 0.5$, giving $[8, 0, 8, 0]$. Mean $= 4$. ✓
4. Test time: no change needed, $[4,4,4,4]$, mean $= 4$. ✓
5. In expectation: $\mathbb{E}[\text{output}] = p_{\text{keep}} \cdot (x/p_{\text{keep}}) = x$. Scale preserved exactly.

**Answer:** inverted dropout gives train and test the same expected activation, which is why frameworks
use it and why no test-time rescaling appears in modern code.

## Code

```python
import numpy as np
rng = np.random.default_rng(0)

# --- batch norm, matching N1 -------------------------------------------
def batch_norm(x, gamma, beta, eps=1e-8):
    mu  = x.mean(axis=0)
    var = x.var(axis=0)                     # population variance (÷N), as BN uses
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, xhat

x = np.array([[2.], [4.], [6.], [8.]])
y, xhat = batch_norm(x, gamma=2.0, beta=1.0)
print("normalised:", np.round(xhat.ravel(), 4))
print("output    :", np.round(y.ravel(), 4))
print("out mean/std:", round(y.mean(), 4), round(y.std(), 4))
# normalised: [-1.3416 -0.4472  0.4472  1.3416]
# output    : [-1.6833  0.1056  1.8944  3.6833]
# out mean/std: 1.0 2.0                       <- exactly beta and gamma

# --- why initialisation scale matters ----------------------------------
n_in = 500
for name, sigma in [("N(0,1) naive", 1.0),
                    ("Xavier", np.sqrt(2/(n_in+100))),
                    ("He",     np.sqrt(2/n_in))]:
    W = rng.normal(0, sigma, size=(n_in, 100))
    z = rng.normal(0, 1, size=(1, n_in)) @ W       # one layer's pre-activation
    print(f"{name:12s} sigma={sigma:.4f}  |z| std={z.std():7.3f}  "
          f"tanh saturated={(np.abs(np.tanh(z))>0.99).mean():.1%}")
# N(0,1) naive sigma=1.0000  |z| std= 22.887  tanh saturated=100.0%
# Xavier       sigma=0.0577  |z| std=  1.322  tanh saturated=4.0%
# He           sigma=0.0632  |z| std=  1.448  tanh saturated=6.0%

# --- inverted dropout preserves the mean, matching N4 ------------------
def inverted_dropout(x, p_keep, rng):
    mask = (rng.random(x.shape) < p_keep) / p_keep   # scale during TRAINING
    return x * mask

a = np.full(100_000, 4.0)
out = inverted_dropout(a, p_keep=0.5, rng=rng)
print(f"kept {np.mean(out>0):.1%} of units, train mean={out.mean():.4f}, test mean={a.mean():.4f}")
# kept 50.0% of units, train mean=4.0003, test mean=4.0000
```

## Exam pack

### Must-memorise

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

### Numbers worth knowing

| Quantity | Value |
|---|---|
| Typical dropout rate (FC layers) | 0.5 |
| Adam defaults | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| Momentum coefficient | 0.9 |
| He's factor vs Xavier | 2 in the numerator, fan-in only — compensates ReLU zeroing half the inputs |
| BN paper | Ioffe & Szegedy, 2015 |
| Dropout paper | Srivastava et al., 2014 |
| Adam paper | Kingma & Ba, 2014 |

### Likely MCQ traps

- **"Batch norm normalises across features."** No — that is *layer* norm. BN normalises across the
  batch, per feature. The two are frequently swapped in options.
- **"Batch norm behaves identically at train and test."** No. It uses batch statistics in training and
  running averages at inference. Layer norm *does* behave identically — that is part of why
  Transformers prefer it.
- **"$\gamma$ and $\beta$ are hyperparameters."** No, they are **learned** parameters, updated by
  backprop like any weight.
- **"Dropout is applied at test time too."** No — it is training-only. Applying it at test time makes
  predictions random.
- **"Xavier is for ReLU."** No. Xavier for sigmoid/tanh, **He for ReLU**. This exact swap is the most
  common trap in this group.
- **"Initialise weights to zero for a neutral start."** Fatal. All units in a layer become identical
  and stay identical. Biases to zero is fine; weights must be random.
- **"SGD means full-batch gradient descent."** Opposite — stochastic means one sample at a time, and in
  practice the term means mini-batch.
- **"More augmentation is always better."** No. Transformations must be label-preserving; flipping
  digits or text destroys the label.
- **"Adam always beats SGD."** Not always — well-tuned SGD with momentum still wins on some vision
  benchmarks in final generalisation. Adam converges faster and is more forgiving of learning rate.

### Self-test

1. A BN layer sits on a feature map with 64 channels. How many learned parameters does it add?
2. Why can batch norm not be used with a batch size of 1 at training time?
3. Which initialiser pairs with ReLU, and what is the factor of 2 compensating for?
4. State the two equivalent ways of handling dropout's train/test scale mismatch.
5. Dataset of 60,000 samples, batch size 128, 5 epochs. How many parameter updates?
6. Why does initialising all weights to zero fail, when initialising all biases to zero is fine?
7. Give two reasons Transformers use layer norm rather than batch norm.
8. What distribution does a BN layer's output have, in terms of $\gamma$ and $\beta$?
9. Which augmentation would you avoid for a handwritten-digit dataset, and why?
10. What does the "adaptive" in Adam refer to?

<details><summary>Answers</summary>

1. 128 — one $\gamma$ and one $\beta$ per channel, $2 \times 64$.
2. The batch variance of a single sample is zero, so normalisation divides by $\sqrt{\epsilon}$ and the output is meaningless.
3. He initialization, $\mathrm{Var}(w) = 2/n_{\text{in}}$. The 2 compensates for ReLU zeroing roughly half its inputs, which otherwise halves the variance.
4. Either scale activations by $p_{\text{keep}}$ at test time (original), or divide by $p_{\text{keep}}$ during training (inverted dropout — what frameworks do).
5. $\lceil 60{,}000/128 \rceil = 469$ per epoch, $\times 5 = 2{,}345$ updates.
6. Zero weights make every unit in a layer compute the same output and receive the same gradient, so they never differentiate — symmetry is never broken. Biases are fine because the *weights* already differ, so the units are already distinct.
7. (a) Sequences have variable length, which makes batch statistics ill-defined; (b) layer norm is batch-size independent and behaves identically at train and test, important for autoregressive generation one token at a time.
8. Mean $\beta$ and standard deviation $\gamma$, per feature.
9. Horizontal flipping — a mirrored 2 or 5 is not a valid digit, so the transformation is not label-preserving.
10. Per-parameter learning rates, derived from the running second moment of each parameter's gradients — parameters with large recent gradients take smaller steps.

</details>

## Beyond the slides

Everything in this chapter is beyond the slides by construction; that is its purpose. Two further
items worth knowing but not worth a section:

**Gap:** Learning-rate schedules — step decay, cosine annealing, warmup — are absent everywhere.
**Why it matters:** Transformer training in particular is famously unstable without a warmup phase, and
"why does the Transformer paper use warmup" is a reasonable question against a syllabus that covers the
Transformer in four lectures.

**Gap:** Weight sharing is listed on `W3L4_P3` slide 26 as a solution but never explained.
**Why it matters:** You have actually met it twice — convolutional kernels reused across spatial
positions ([Lec 11](week-03/11-cnn-basics.md)) and recurrent weights reused across time steps
([Lec 16](week-04/16-sequence-modelling-and-rnn.md)). It is not a separate technique; recognising it as
the same idea in two costumes is the insight.

## Cut from the slides

Nothing — there are no slides to cut from. This chapter exists precisely because the material has no
deck. Every item here was flagged during authoring as named-but-untaught: batch normalization and
parameter initialisation from `W3L4_P3` slides 7 and 26, internal covariate shift from `W4L5_P1` slide
4, dropout and data augmentation from `W3L4_P3` slide 26 and `W3L4_P2` slide 7 (AlexNet's innovations),
and SGD/mini-batch from the undefined batch size $B$ in `W2L3_P3` slide 18. Optimizers beyond plain
SGD appear on no slide in Weeks 1–8 and are included only because they are unavoidable in practice and
plausibly examinable. If your exam is strictly deck-scoped, this chapter is insurance rather than
core — but it is cheap insurance.
