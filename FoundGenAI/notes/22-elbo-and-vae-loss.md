# Lec 22 — Probabilistic Decoder, ELBO, VAE Loss

> **Source:** `Lec 22.pdf` (10 pages) · **Week 3** · **Playlist:** Lec 22
> **Prereqs:** [Lec 11 — Reconstruction Loss (MSE, BCE)](11-reconstruction-loss.md), [Lec 19 — KL Divergence, Part A](19-kl-divergence-a.md), [Lec 21 — Introduction to VAE and the Encoder](21-vae-encoder.md)
> **Feeds into:** [Lec 23 — The Reparameterization Trick](23-reparameterization.md), [Lec 24 — VAE Numerical Example](24-vae-numerical.md), [Lec 26 — Disentanglement and β-VAE](26-beta-vae.md), [Lec 27 — Conditional VAE](27-conditional-vae.md), [Lec 47 — DDPM Reverse Process](47-ddpm-reverse.md)

## Why this lecture exists

[Lec 21](21-vae-encoder.md) left you stuck. The encoder wants the true posterior $p(\mathbf{z}\mid\mathbf{x})$; computing it needs the evidence $p(\mathbf{x}) = \int p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$; that integral is intractable; so the encoder settles for an approximation $q_\phi(\mathbf{z}\mid\mathbf{x})$. What nobody has said yet is how you *train* an approximation to something you cannot compute.

This lecture answers that, and it is the mathematical pivot of the whole course. The trick is to stop chasing $\log p_\theta(\mathbf{x})$ and instead maximise a quantity that is always **below** it and is computable — the Evidence Lower Bound. Out of that single move falls the entire VAE loss: one term that is literally the reconstruction loss of [Lec 11](11-reconstruction-loss.md), and one term that is the KL divergence of [Lec 19](19-kl-divergence-a.md) between the encoder and the prior. The same bound reappears, barely disguised, in the diffusion derivation in Week 7.

## The ideas

> **Notation note.** The deck writes plain $x$, $z$; this book bolds them. The deck's font renders $\phi$ as $\emptyset$ on pages 7 and 8 — it means $\phi$, the encoder's parameters, throughout. The deck writes $KL(\cdot\|\cdot)$; this book writes $D_{\mathrm{KL}}(\cdot\,\|\,\cdot)$. **All logarithms here are natural logs**, so every KL and every loss is in **nats**; the course is not consistent about this (Lec 14's deck uses $\log_{10}$, Lec 20's uses $\log_2$ and quotes bits), so read the base off the question before you compute.

### The decoder is a distribution, not a function

![Slide titled VAEs: Decoder. Left column: the encoder's q_phi(z|x) = N(mu_phi(x), sigma²_phi(x)), z sampled from it, then the decoder outputs the parameters of p_theta(x|z); for real-valued data it emits mu_theta(z), sigma²_theta(z) defining a Gaussian. Right: for binary data it emits pi_theta(z), defining a Bernoulli. Top right shows the x → Encoder → z → Decoder → x̂ diagram with both output branches drawn](../assets/pages/lec22/p-03.png)
*Fig. — Notice the decoder's output arrow **forks**, exactly as the encoder's did. What comes out of a VAE decoder is never the reconstruction itself; it is the **parameters of a distribution over reconstructions**, and $\hat{\mathbf{x}}$ is then a draw from that distribution. Which parameters depends only on the data type. Page 3.*

An autoencoder's decoder is a function: latent in, reconstruction out, deterministically. A VAE's decoder is a **conditional distribution** $p_\theta(\mathbf{x}\mid\mathbf{z})$ — "the probability of generating data $\mathbf{x}$ given the latent $\mathbf{z}$" — and the network emits the *parameters* of that distribution.

The pipeline on the slide, step by step:

1. The encoder gives $\mu_\phi(\mathbf{x})$ and $\sigma^2_\phi(\mathbf{x})$, defining $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}\big(\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x})\big)$.
2. A latent vector is **sampled**: $\mathbf{z} \sim \mathcal{N}\big(\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x})\big)$.
3. That $\mathbf{z}$ goes to the decoder network.
4. The decoder outputs the parameters of $p_\theta(\mathbf{x}\mid\mathbf{z})$. Which parameters depends on the data type — exactly the Case 1 / Case 2 split from [Lec 11](11-reconstruction-loss.md), now stated probabilistically:

| Data type | Decoder emits | Distribution | Sampling the output |
|---|---|---|---|
| **continuous / real-valued** | $\mu_\theta(\mathbf{z})$, $\sigma^2_\theta(\mathbf{z})$ | $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathcal{N}\big(\mu_\theta(\mathbf{z}), \sigma^2_\theta(\mathbf{z})\big)$ | $\hat{\mathbf{x}} \sim \mathcal{N}\big(\mu_\theta(\mathbf{z}), \sigma^2_\theta(\mathbf{z})\big)$ |
| **binary** | $\pi_\theta(\mathbf{z})$ — one probability per feature or pixel | $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathrm{Bernoulli}\big(\pi_\theta(\mathbf{z})\big)$ | $\hat{\mathbf{x}} \sim \mathrm{Bernoulli}\big(\pi_\theta(\mathbf{z})\big)$ |

A **Bernoulli** distribution is the coin-flip distribution: one parameter $\pi \in (0,1)$, and the outcome is $1$ with probability $\pi$ and $0$ with probability $1-\pi$. A binary decoder emits one such $\pi$ per pixel — which is why its output layer ends in a sigmoid, exactly as in [Lec 11](11-reconstruction-loss.md).

**The symbols $\mu, \sigma^2$ now appear twice with different meanings.** $\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x})$ are the **encoder's**, parameters of a distribution over *latents*. $\mu_\theta(\mathbf{z}), \sigma^2_\theta(\mathbf{z})$ are the **decoder's**, parameters of a distribution over *data*. Read the subscript, every time.

In practice almost every implementation fixes the decoder's $\sigma^2_\theta$ to a constant and emits only $\mu_\theta(\mathbf{z})$, which is then what you look at and call "the reconstruction". The deck shows the general two-output form; the consequence of fixing the variance is the whole point of the MSE connection below.

### One input, many latents

![Slide titled VAEs: Decoder. Left: x = [0.80, 0.10, 0.60, 0.30, 0.90] with a table of many sampled z's; three samples z¹=[0.56,−0.11], z²=[0.43,−0.32], z³=[0.61,−0.24] each decoded to a slightly different reconstruction. Right: mu(x)=[0.50,−0.20], sigma²(x)=[0.04,0.09], the two latent Gaussians, and the red question "Given this real training sample x, which latent representation z produce this sample through the decoder?"](../assets/pages/lec22/p-04.png)
*Fig. — The lecture's hinge. Three different latent vectors drawn from the **same** $q_\phi(\mathbf{z}\mid\mathbf{x})$ decode to three slightly different reconstructions, all close to the input. Then the red text turns the picture around and asks the reverse question — which $\mathbf{z}$ *produced* this $\mathbf{x}$? — and that reverse question is what forces the ELBO. Page 4.*

For one input $\mathbf{x}$ there is not one $\mathbf{z}$. The deck: *"There can be many possible $z$ values that can represent the same input"*, and *"there are infinitely many possible $z$'s"*. With $\mathbf{x} = [0.80, 0.10, 0.60, 0.30, 0.90]$, the encoder produces $\mu(\mathbf{x}) = [0.50, -0.20]$ and $\sigma^2(\mathbf{x}) = [0.04, 0.09]$, so

$$z_1 \sim \mathcal{N}(0.50,\ 0.04), \qquad z_2 \sim \mathcal{N}(-0.20,\ 0.09)$$

(variance second, as always: $\sigma_1 = 0.2$ and $\sigma_2 = 0.3$). Three draws from that distribution, each passed through the decoder:

| Sample | $\mathbf{z}$ | $\hat{\mathbf{x}}$ |
|---|---|---|
| 1 | $[0.56,\ -0.11]$ | $[0.79,\ 0.12,\ 0.58,\ 0.32,\ 0.88]$ |
| 2 | $[0.43,\ -0.32]$ | $[0.76,\ 0.14,\ 0.61,\ 0.28,\ 0.86]$ |
| 3 | $[0.61,\ -0.24]$ | $[0.82,\ 0.09,\ 0.57,\ 0.34,\ 0.91]$ |

*"The reconstructed outputs are slightly different because different latent vectors were sampled. However, all of them remain similar to the original input."* N1 measures exactly how similar.

This is the behaviour a plain autoencoder cannot produce, and it is the mechanism behind every VAE advantage: a whole neighbourhood of latent space decodes to the same thing, so the latent space has no holes.

### The reverse question, and the same wall again

![Slide titled VAEs: Decoder. Left: Bayes' rule p(z|x) = p(x|z)p(z)/p(x) with the denominator p(x) = ∫p(x|z)p(z)dz, and the note that this requires considering all possible latent variables — "for neural network decoders, this integral is usually intractable (difficult)". Right: with prior p(z) = N(0, I), each candidate z contributes p(x|z)p(z); z¹=[0.2,−0.1] near the origin has high prior density, z²=[4.0,−3.5] far away has very low density](../assets/pages/lec22/p-05.png)
*Fig. — The same intractability [Lec 21](21-vae-encoder.md) hit on its page 7, now with the integral's **weighting** explained. Each candidate latent contributes $p(\mathbf{x}\mid\mathbf{z}) \times p(\mathbf{z})$: how well it explains the data, times how plausible it was to begin with. Notice the second factor penalises latents far from the origin — that is the prior doing its job. Page 5.*

Turn the picture around: *given this real training sample $\mathbf{x}$, which latent representation $\mathbf{z}$ produced it?* That is $p(\mathbf{z}\mid\mathbf{x})$, and Bayes' rule says

$$p(\mathbf{z}\mid\mathbf{x}) = \frac{p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x})}, \qquad p(\mathbf{x}) = \int p(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}$$

[Lec 21](21-vae-encoder.md) owns this wall and quantifies it (its N5: about $10^{25}$ decoder evaluations for a 25-dimensional latent). What this deck adds is *what the integral is summing*. With the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ — note this slide writes the **joint** prior, which page 8 of Lec 21 only wrote per-coordinate — each candidate latent contributes a product of two factors:

- $p(\mathbf{x}\mid\mathbf{z}^k)$ — the decoder's verdict: could *this* latent have generated the observed $\mathbf{x}$?
- $p(\mathbf{z}^k)$ — the prior's verdict: was this latent plausible in the first place? The deck's example: $\mathbf{z}^1 = [0.2, -0.1]$ sits near the origin and has **high** prior density; $\mathbf{z}^2 = [4.0, -3.5]$ is far out and has **very low** density.

So the integral is $\sum_{\text{all } \mathbf{z}}$ (fit) $\times$ (plausibility), and the deck's verdict is blunt: *"For neural network decoders, this integral is usually intractable (difficult)."*

### What we would like to maximise, and cannot

![Slide titled VAEs: Evidence Lower Bound (ELBO). Text: after training the model should give high probability to digit-like images and low probability to noise; ideally we want to maximise log p_theta(x); computing p_theta(x) is difficult because it requires checking all possible latent variables z; so the VAE maximises the Evidence Lower Bound instead, a computable lower bound of log p_theta(x)](../assets/pages/lec22/p-06.png)
*Fig. — The statement of the problem in one slide, and the name of the answer. "ELBO is a computable lower bound of $\log p_\theta(\mathbf{x})$" is the sentence the rest of this chapter earns. Page 6.*

The training objective you *want* is **maximum likelihood**: make the model assign high probability to the data you actually observed. The deck's framing — after training on handwritten digits, the model should give high probability to digit-like images and low probability to noise — is the right intuition. Formally, maximise

$$\log p_\theta(\mathbf{x})$$

**Why the logarithm.** Probabilities of high-dimensional data are astronomically small (a product of hundreds of per-pixel factors), so $p_\theta(\mathbf{x})$ underflows to zero in floating point. The log turns products into sums, turns a number like $10^{-400}$ into $-921$, and — because $\log$ is increasing — whatever maximises $\log p_\theta(\mathbf{x})$ also maximises $p_\theta(\mathbf{x})$. Maximising the log is the same problem, numerically survivable.

But $\log p_\theta(\mathbf{x})$ needs $p_\theta(\mathbf{x})$, which is the intractable integral. So, says the deck, *"directly maximizing $\log p_\theta(\mathbf{x})$ is usually not possible. Therefore, VAE maximizes a simpler objective called Evidence Lower Bound (ELBO)."*

Three words, each load-bearing. **Evidence** — $p_\theta(\mathbf{x})$ is called the evidence ([Lec 21](21-vae-encoder.md)'s Bayes table). **Lower bound** — the ELBO is always $\leq \log p_\theta(\mathbf{x})$, never above. **Computable** — unlike the thing it bounds.

### Deriving the ELBO

The deck does not derive this. It states the result on its next slide and moves on. Derive it once, properly, and the VAE loss stops being three memorised symbols.

**Two facts about expectations, first, in words.** For a distribution $q(\mathbf{z})$ and any function $f(\mathbf{z})$:

$$\mathbb{E}_{q(\mathbf{z})}[f(\mathbf{z})] = \int q(\mathbf{z})\,f(\mathbf{z})\,d\mathbf{z}$$

— *"the average value of $f$, when $\mathbf{z}$ is drawn according to $q$"*. And since $q$ is a probability distribution, $\int q(\mathbf{z})\,d\mathbf{z} = 1$: **the average of a constant is that constant.** That second fact is the only trick in step 1.

---

**Step 1 — turn $\log p_\theta(\mathbf{x})$ into an expectation.**

$\log p_\theta(\mathbf{x})$ contains no $\mathbf{z}$ at all. It is a constant as far as $\mathbf{z}$ is concerned. So averaging it over *any* distribution of $\mathbf{z}$ leaves it unchanged — and we choose the encoder's distribution:

$$\log p_\theta(\mathbf{x}) \;=\; \int q_\phi(\mathbf{z}\mid\mathbf{x})\,\log p_\theta(\mathbf{x})\,d\mathbf{z} \;=\; \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x})\big]$$

Nothing has been approximated. We have only written the same number in a form with $q_\phi$ in it, which is what lets $q_\phi$ into the algebra.

**Step 2 — replace $p_\theta(\mathbf{x})$ using Bayes' rule.**

The joint distribution factorises two ways: $p_\theta(\mathbf{x},\mathbf{z}) = p_\theta(\mathbf{z}\mid\mathbf{x})\,p_\theta(\mathbf{x})$. Rearranged,

$$p_\theta(\mathbf{x}) = \frac{p_\theta(\mathbf{x},\mathbf{z})}{p_\theta(\mathbf{z}\mid\mathbf{x})} \qquad\Longrightarrow\qquad \log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x},\mathbf{z})}{p_\theta(\mathbf{z}\mid\mathbf{x})}\right]$$

Still exact. Note this introduces the true posterior $p_\theta(\mathbf{z}\mid\mathbf{x})$ — the thing we cannot compute. It will not survive.

**Step 3 — multiply and divide by $q_\phi(\mathbf{z}\mid\mathbf{x})$.**

This is the whole move. Insert $\dfrac{q_\phi(\mathbf{z}\mid\mathbf{x})}{q_\phi(\mathbf{z}\mid\mathbf{x})} = 1$ inside the fraction and regroup:

$$
\begin{aligned}
\log p_\theta(\mathbf{x})
&= \mathbb{E}_{q_\phi}\!\left[\log \left( \frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}\mid\mathbf{x})}{p_\theta(\mathbf{z}\mid\mathbf{x})} \right)\right] \\[4pt]
&= \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right]}_{\textstyle \mathcal{L}_{\text{ELBO}}(\mathbf{x})}
\;+\; \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(\mathbf{z}\mid\mathbf{x})}{p_\theta(\mathbf{z}\mid\mathbf{x})}\right]}_{\textstyle D_{\mathrm{KL}}\left(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\right)}
\end{aligned}
$$

using $\log(ab) = \log a + \log b$ and the fact that the average of a sum is the sum of the averages. The second bracket is, term for term, [Lec 19](19-kl-divergence-a.md)'s definition of the KL divergence from $q_\phi(\mathbf{z}\mid\mathbf{x})$ to $p_\theta(\mathbf{z}\mid\mathbf{x})$.

**Step 4 — this is an exact identity, so read it.**

$$\boxed{\;\log p_\theta(\mathbf{x}) \;=\; \mathcal{L}_{\text{ELBO}}(\mathbf{x}) \;+\; D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big)\;}$$

No approximation has been made anywhere. The log-evidence splits *exactly* into the ELBO plus a KL divergence.

**Step 5 — why it is a *lower* bound.**

KL divergence is **non-negative** — [Lec 19](19-kl-divergence-a.md) establishes this, and it is the only property we need. So the second term can only add, never subtract:

$$\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}}(\mathbf{x}) + (\text{something} \geq 0) \qquad\Longrightarrow\qquad \mathcal{L}_{\text{ELBO}}(\mathbf{x}) \;\leq\; \log p_\theta(\mathbf{x})$$

That is the bound, and that is where the name comes from. Equality holds **if and only if** the KL term is zero, i.e. if and only if $q_\phi(\mathbf{z}\mid\mathbf{x}) = p_\theta(\mathbf{z}\mid\mathbf{x})$ exactly — the encoder has nailed the true posterior.

**Step 6 — the bound's slack is the encoder's error, and that is a bonus.**

Rearranging step 4: $D_{\mathrm{KL}}\big(q_\phi\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big) = \log p_\theta(\mathbf{x}) - \mathcal{L}_{\text{ELBO}}(\mathbf{x})$. The **gap between the bound and the truth is exactly how wrong the encoder's approximate posterior is.** So pushing the ELBO up does two jobs at once:

- it raises $\log p_\theta(\mathbf{x})$ (better generative model, via $\theta$), **and**
- it shrinks the gap, dragging $q_\phi$ toward the true posterior (better encoder, via $\phi$).

One objective, both networks, no separate inference step. That is why the VAE works.

**Step 7 — split the ELBO into its two usable halves.**

Factor the joint the *other* way, $p_\theta(\mathbf{x},\mathbf{z}) = p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})$ — decoder times prior, both of which we have:

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}}(\mathbf{x})
&= \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right] \\[2pt]
&= \mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] \;+\; \mathbb{E}_{q_\phi}\!\left[\log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right] \\[2pt]
&= \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] \;-\; D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)
\end{aligned}
$$

The last line flips the fraction inside the log, which costs a minus sign, and leaves [Lec 19](19-kl-divergence-a.md)'s KL again — but now against the **prior** $p(\mathbf{z})$, not against the intractable posterior. Every quantity in that expression is computable. That is the deck's page-7 formula, now earned.

> **The two KLs are different and the exam will try this on you.**
>
> | | $D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big)$ | $D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$ |
> |---|---|---|
> | Second argument | the **true posterior** | the **prior** |
> | Computable? | no — contains $p(\mathbf{x})$ | yes — closed form, below |
> | Where it lives | the **slack** of the bound; never evaluated | **inside the loss**; evaluated every step |
> | If it were zero | the bound is tight, ELBO $=\log p_\theta(\mathbf{x})$ | the encoder ignores the input entirely |

> **The alternative one-line route (Jensen's inequality).** Some exams ask for this instead. $\log$ is a concave function, so $\log \mathbb{E}[u] \geq \mathbb{E}[\log u]$. Then
> $$\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z}\mid\mathbf{x})\,\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\,d\mathbf{z} = \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right] \geq \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right] = \mathcal{L}_{\text{ELBO}}$$
> Shorter, but it tells you only that a gap exists. The seven-step derivation tells you **what the gap is**, which is the more examinable fact.

### From ELBO to the VAE loss

![Slide: the ELBO written as E_{q_phi}[log p_theta(x|z)] boxed in blue and labelled "Reconstruction", minus D_KL(q_phi(z|x) ‖ p(z)) boxed in red and labelled "KL regularization term"; below, max(f) ⇔ min(−f), giving L_VAE(x) = −E[log p_theta(x|z)] + D_KL(...), i.e. Reconstruction Loss + KL Divergence Loss](../assets/pages/lec22/p-07.png)
*Fig. — The slide to photograph. Two boxes, two questions. Blue: can the decoder rebuild $\mathbf{x}$ from a latent the encoder sampled? Red: is the encoder's distribution close to the prior? Notice the sign flip at the bottom — the minus in front of the expectation and the **plus** in front of the KL are both consequences of turning a maximisation into a minimisation. Page 7.*

The deck's statement of the ELBO, which step 7 just derived:

$$\mathcal{L}_{\text{ELBO}}(\mathbf{x}) \;=\; \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]}_{\text{Reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)}_{\text{KL regularization term}}$$

with the deck's own reading of each box:

- **Reconstruction:** *"If the encoder samples $\mathbf{z}$ from $q_\phi(\mathbf{z}\mid\mathbf{x})$, can the decoder reconstruct the original input $\mathbf{x}$ with high probability?"*
- **KL regularization:** *"Is the encoder distribution $q_\phi(\mathbf{z}\mid\mathbf{x})$ close to the prior distribution $p(\mathbf{z})$?"*

Now the bookkeeping that trips people up. We want to **maximise** the ELBO. Neural networks are trained by **minimising** a loss. The deck's one-liner:

$$\max(f) \iff \min(-f)$$

So negate the whole ELBO. Negating a difference flips **both** signs:

$$\mathcal{L}_{\text{VAE}}(\mathbf{x}) = -\,\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] \;+\; D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$$

$$\mathcal{L}_{\text{VAE}} = \text{Reconstruction Loss} + \text{KL Divergence Loss}$$

**Minus** on the reconstruction term, **plus** on the KL term. Memorise the signs together with the sentence that produces them: *the ELBO has a minus before the KL and we are minimising its negative.* A sign error here is the most common way to fail a VAE numerical.

### The reconstruction term *is* the loss from Lec 11

![Slide titled VAEs: Loss Function. L(x) = −E_{q_phi}[log p_theta(x|z)] + KL(q_phi(z|x) ‖ p(z)); a green arrow from "Reconstruction" points to "Mean squared error", a green arrow from "Regularization" points to the closed form L_KL = ½[μ² + σ² − log σ² − 1]; at the bottom, sampling breaks gradient flow, motivating the reparameterization trick](../assets/pages/lec22/p-08.png)
*Fig. — The abstract objective cashed out into two things you can compute. Notice the deck labels the reconstruction arrow **"Mean squared error"** and stops there — it never mentions the Bernoulli/BCE case even though its own page 3 defined that decoder. That omission is closed below. Page 8.*

The deck draws a green arrow from "Reconstruction" to "Mean squared error" and says nothing more. That arrow deserves a derivation, because it is the single most useful bridge in the course: **the probabilistic reconstruction term and the ordinary reconstruction losses of [Lec 11](11-reconstruction-loss.md) are the same thing written two ways.**

**Case 1 — Gaussian decoder gives MSE.**

Take the real-valued decoder $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathcal{N}\big(\mu_\theta(\mathbf{z}),\,\sigma_{\text{dec}}^2\big)$ with the decoder variance **fixed** (not learned), and $D$ data dimensions, independent across dimensions. The Gaussian density in one dimension is $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$, so taking logs and summing over the $D$ dimensions:

$$\log p_\theta(\mathbf{x}\mid\mathbf{z}) = -\frac{1}{2\sigma_{\text{dec}}^2}\sum_{j=1}^{D}\big(x_j - \mu_{\theta,j}(\mathbf{z})\big)^2 \;-\; \frac{D}{2}\log\big(2\pi\sigma_{\text{dec}}^2\big)$$

Negate it to get the reconstruction *loss*:

$$-\log p_\theta(\mathbf{x}\mid\mathbf{z}) = \underbrace{\frac{1}{2\sigma_{\text{dec}}^2}\big\lVert \mathbf{x} - \hat{\mathbf{x}} \big\rVert^2}_{\text{scaled squared error}} \;+\; \underbrace{\frac{D}{2}\log\big(2\pi\sigma_{\text{dec}}^2\big)}_{\text{constant in }\theta,\ \phi}$$

writing $\hat{\mathbf{x}} = \mu_\theta(\mathbf{z})$. The second term does not depend on the network's parameters at all, so it contributes nothing to any gradient and can be dropped. What remains is $\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2$ times a constant — **[Lec 11](11-reconstruction-loss.md)'s MSE**. Choosing $\sigma_{\text{dec}}^2 = \tfrac12$ makes the constant exactly $1$ and the match exact.

And notice what $\sigma^2_{\text{dec}}$ is doing: it is the **weight of the reconstruction term against the KL term**. A small decoder variance means the model is confident, which scales the squared error up and lets reconstruction dominate. That knob is [Lec 26](26-beta-vae.md)'s $\beta$ in disguise.

**Case 2 — Bernoulli decoder gives BCE.**

With binary data and $p_\theta(\mathbf{x}\mid\mathbf{z}) = \mathrm{Bernoulli}(\pi_\theta(\mathbf{z}))$, the probability of one feature is $\pi_j^{x_j}(1-\pi_j)^{1-x_j}$ — which equals $\pi_j$ when $x_j = 1$ and $1-\pi_j$ when $x_j = 0$, the switch you met in [Lec 11](11-reconstruction-loss.md). Features are independent given $\mathbf{z}$, so the probabilities multiply and the logs add:

$$\log p_\theta(\mathbf{x}\mid\mathbf{z}) = \sum_{j=1}^{D}\Big[x_j\log \pi_{\theta,j}(\mathbf{z}) + (1-x_j)\log\big(1-\pi_{\theta,j}(\mathbf{z})\big)\Big]$$

$$-\log p_\theta(\mathbf{x}\mid\mathbf{z}) = -\sum_{j=1}^{D}\Big[x_j\log \hat{x}_j + (1-x_j)\log(1-\hat{x}_j)\Big]$$

That is **[Lec 11](11-reconstruction-loss.md)'s binary cross-entropy, exactly** — with no dropped constant at all. BCE is not an analogue of the negative log-likelihood; it *is* the negative log-likelihood of a Bernoulli decoder.

| Decoder distribution | $-\log p_\theta(\mathbf{x}\mid\mathbf{z})$ equals | Data | Relationship |
|---|---|---|---|
| $\mathcal{N}\big(\mu_\theta(\mathbf{z}),\ \sigma^2_{\text{dec}}\big)$ | $\frac{1}{2\sigma^2_{\text{dec}}}\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2 + \text{const}$ | real-valued | **MSE**, up to a constant and a scale |
| $\mathrm{Bernoulli}\big(\pi_\theta(\mathbf{z})\big)$ | $-\sum_j[x_j\log\hat{x}_j + (1-x_j)\log(1-\hat{x}_j)]$ | binary | **BCE**, exactly |

So the answer to "why is MSE the right reconstruction loss?" is no longer "because it measures distance". It is: **MSE is the negative log-likelihood of a Gaussian decoder, and the ELBO's reconstruction term is a negative log-likelihood.** The loss was never a design choice; it was a distributional assumption. [Lec 11](11-reconstruction-loss.md) flagged this as a gap it was leaving open — this is the closure.

### What the expectation means, and how it is actually computed

$\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]$ says: *average the decoder's log-probability of $\mathbf{x}$ over **every** latent the encoder might have emitted, weighted by how likely the encoder was to emit it.* Page 4's three samples were a picture of exactly this average, over three of the infinitely many $\mathbf{z}$'s.

You cannot compute that average exactly — it is another integral over $\mathbf{z}$. You estimate it by **Monte Carlo**: draw $L$ samples from $q_\phi$ and average.

$$\mathbb{E}_{q_\phi}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big] \approx \frac{1}{L}\sum_{l=1}^{L} \log p_\theta\big(\mathbf{x}\mid\mathbf{z}^{(l)}\big), \qquad \mathbf{z}^{(l)} \sim q_\phi(\mathbf{z}\mid\mathbf{x})$$

In practice **$L = 1$**. One sample per data point per training step. It is a noisy estimate, but it is *unbiased*, and across a minibatch of hundreds of points and thousands of steps the noise averages out. This is why a VAE training loop looks so much like an autoencoder's: encode, draw one $\mathbf{z}$, decode, score.

And that single draw is the problem the deck ends on: *"Since sampling is a random operation, randomness breaks gradient flow, making backpropagation impossible. So, the question is: how do we compute gradients when part of the network involves sampling?"* The answer is the reparameterization trick, which [Lec 23](23-reparameterization.md) owns completely.

### The KL term in closed form — derived

The deck states the result and says only *"when both distributions are Gaussian, the KL divergence has a closed-form expression"*:

$$L_{\mathrm{KL}} = \tfrac{1}{2}\big[\mu^2 + \sigma^2 - \log\sigma^2 - 1\big]$$

Nothing earlier in this book derives it — [Lec 19](19-kl-divergence-a.md) gives KL's definition and its properties, [Lec 20](20-kl-divergence-b.md) works it for *discrete* distributions in bits. So the derivation is owned here. It needs the Gaussian density and two facts about averages, and nothing else.

**Setup.** One latent dimension. $q = \mathcal{N}(\mu, \sigma^2)$ (the encoder's, for this dimension), $p = \mathcal{N}(0, 1)$ (the prior). [Lec 19](19-kl-divergence-a.md)'s definition, written as an expectation:

$$D_{\mathrm{KL}}(q\,\|\,p) = \mathbb{E}_{q}\big[\log q(z) - \log p(z)\big]$$

**Step 1 — write both log-densities.** From $\mathcal{N}(m, v)$'s density $\frac{1}{\sqrt{2\pi v}}e^{-(z-m)^2/2v}$:

$$\log q(z) = -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}, \qquad \log p(z) = -\tfrac{1}{2}\log(2\pi) - \frac{z^2}{2}$$

**Step 2 — subtract.** The $\log 2\pi$ halves cancel, leaving

$$\log q(z) - \log p(z) = -\tfrac{1}{2}\log\sigma^2 - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}$$

**Step 3 — average over $z \sim q$, term by term.** Two facts, both just the definition of variance:

- $\mathbb{E}_q\big[(z-\mu)^2\big] = \sigma^2$ — the average squared distance from the mean **is** the variance.
- $\mathbb{E}_q\big[z^2\big] = \sigma^2 + \mu^2$ — because $\mathrm{Var}(z) = \mathbb{E}[z^2] - (\mathbb{E}[z])^2$, so $\mathbb{E}[z^2] = \mathrm{Var}(z) + \mu^2$.

The first term has no $z$ in it, so it averages to itself. Therefore

$$D_{\mathrm{KL}}(q\,\|\,p) = -\tfrac{1}{2}\log\sigma^2 - \frac{\sigma^2}{2\sigma^2} + \frac{\sigma^2+\mu^2}{2} = -\tfrac{1}{2}\log\sigma^2 - \tfrac{1}{2} + \tfrac{1}{2}\sigma^2 + \tfrac{1}{2}\mu^2$$

**Step 4 — collect.**

$$\boxed{\;D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big) = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - \log\sigma^2 - 1\big)\;}$$

which is the deck's formula exactly. **Summing over the $d$ latent dimensions** — legal because the encoder's covariance is diagonal, so $q_\phi$ factorises and KL adds across independent coordinates — gives the form you will see in every implementation:

$$D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big) = \frac{1}{2}\sum_{j=1}^{d}\Big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\Big) = -\frac{1}{2}\sum_{j=1}^{d}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)$$

The two expressions are identical — the second is the first with the minus sign pulled outside, and it is the form written in the original VAE paper and in most PyTorch code. **Every term, named:**

| Term | Is | Pushes toward | Why |
|---|---|---|---|
| $\mu_j^2$ | squared posterior mean, dimension $j$ | $\mu_j \to 0$ | the prior is centred at zero |
| $\sigma_j^2$ | posterior variance, dimension $j$ | $\sigma_j^2$ small | penalises over-spread codes |
| $-\log\sigma_j^2$ | negative log-variance (**what the encoder emits directly**) | $\sigma_j^2$ large | penalises collapsed, near-zero variance |
| $-1$ | the constant that makes the minimum exactly $0$ | — | so a perfect match costs nothing |
| $\tfrac12\sum_j$ | half, summed over latent dimensions | — | diagonal covariance ⟹ KL adds across dimensions |

**Sanity-check it before you trust it.** Set $\mu = 0$, $\sigma^2 = 1$ — the encoder exactly matching the prior. Then $\tfrac12(0 + 1 - \log 1 - 1) = \tfrac12(0 + 1 - 0 - 1) = 0$. Zero, as a divergence from a distribution to itself must be. The $\sigma^2$ and the $-1$ exist precisely to make that happen.

**The general two-Gaussian form**, for completeness, since a question may give a non-standard prior:

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

Put $\mu_2 = 0$, $\sigma_2^2 = 1$: $\log(1/\sigma_1) = -\tfrac12\log\sigma_1^2$, and the middle term becomes $\tfrac12(\sigma_1^2 + \mu_1^2)$, recovering the boxed formula. (Verified numerically in *Code*.)

> **Direction matters, and this one is "reverse".** The VAE minimises $D_{\mathrm{KL}}(q\,\|\,p)$, with the *approximation* in the first slot — what [Lec 19](19-kl-divergence-a.md) calls the **reverse** KL, and KL is not symmetric. Reverse KL is **mode-seeking**: it is heavily penalised wherever $q$ puts mass that $p$ does not, so $q$ prefers to cover one region of $p$ well rather than spread thinly over all of it. Forward KL, $D_{\mathrm{KL}}(p\,\|\,q)$, is mode-covering and would give a different model. [Lec 19](19-kl-divergence-a.md) owns that contrast; the point to carry forward is that mode-seeking behaviour is one standard explanation for why VAE samples are **blurry and under-diverse** — the complaint that opens [Lec 31](31-gan-motivation.md).

### The tug of war between the two terms

The two terms of $\mathcal{L}_{\text{VAE}}$ want opposite things, and understanding the conflict explains almost every VAE pathology.

| | Reconstruction term wants | KL term wants |
|---|---|---|
| $\sigma^2_\phi(\mathbf{x})$ | **tiny** — a precise, repeatable code the decoder can rely on | **exactly 1** — matching the prior |
| $\mu_\phi(\mathbf{x})$ | **spread far apart** — distinct inputs, distinct codes | **all near 0** — matching the prior |
| In the limit | an ordinary autoencoder, with all of [Lec 16](16-ae-numerical-and-limits.md)'s problems | the encoder ignores $\mathbf{x}$ entirely (**posterior collapse**) |

Neither extreme is a usable generative model. Reconstruction alone gives you back the autoencoder whose latent space you cannot sample. KL alone gives you $q_\phi(\mathbf{z}\mid\mathbf{x}) = p(\mathbf{z})$ for every input, a decoder fed pure noise, and reconstructions that are the dataset average. The balance between them is what makes the latent space both informative *and* samplable, and deliberately re-weighting that balance with a coefficient is exactly what [Lec 26](26-beta-vae.md)'s $\beta$-VAE does.

## Worked numericals

> All logarithms below are **natural** logs; every KL and loss is in **nats**. Multiply by $1/\ln 2 = 1.4427$ for bits, or by $1/\ln 10 = 0.4343$ for base-10 units.

### N1. The slide's three reconstructions, scored (page 4)

The deck shows three reconstructions and never measures them. Here is the arithmetic.

**Given:** $\mathbf{x} = [0.80,\ 0.10,\ 0.60,\ 0.30,\ 0.90]$ and the three decoder outputs from page 4.
**Find:** the squared error for each, and the Monte Carlo estimate of the reconstruction term with $L=3$ samples.

1. Sample 1, $\hat{\mathbf{x}}^1 = [0.79, 0.12, 0.58, 0.32, 0.88]$. Residuals $\mathbf{x}-\hat{\mathbf{x}}^1 = [0.01, -0.02, 0.02, -0.02, 0.02]$; squares $[0.0001, 0.0004, 0.0004, 0.0004, 0.0004]$; sum $= 0.0017$.
2. Sample 2, $\hat{\mathbf{x}}^2 = [0.76, 0.14, 0.61, 0.28, 0.86]$. Residuals $[0.04, -0.04, -0.01, 0.02, 0.04]$; squares $[0.0016, 0.0016, 0.0001, 0.0004, 0.0016]$; sum $= 0.0053$.
3. Sample 3, $\hat{\mathbf{x}}^3 = [0.82, 0.09, 0.57, 0.34, 0.91]$. Residuals $[-0.02, 0.01, 0.03, -0.04, -0.01]$; squares $[0.0004, 0.0001, 0.0009, 0.0016, 0.0001]$; sum $= 0.0031$.
4. Monte Carlo average over the three samples: $\frac{1}{3}(0.0017 + 0.0053 + 0.0031) = \frac{0.0101}{3} = 0.00336\overline{6}$.

**Answer:** squared errors $0.0017$, $0.0053$, $0.0031$; reconstruction term estimate $\approx 0.003367$ (sum-over-features convention; $0.000673$ if you also divide by the $D=5$ features). **The slide's three reconstructions are all consistent with the stated input** — the worst is sample 2, off by at most $0.04$ in any coordinate. Note how much the estimate moves between samples ($0.0017$ to $0.0053$, a factor of three): that spread is the Monte Carlo noise you accept when you use $L=1$ in training.

### N2. The KL term for the slide's encoder output (pages 4 and 8)

**Given:** the page-4 encoder output $\mu = [0.50,\ -0.20]$ and $\sigma^2 = [0.04,\ 0.09]$, prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$.
**Find:** $D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)$ using the page-8 closed form.

1. Formula, per dimension: $\tfrac12\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$.
2. **Dimension 1:** $\mu_1^2 = 0.50^2 = 0.25$; $\sigma_1^2 = 0.04$; $\log 0.04 = -3.218876$, so $-\log\sigma_1^2 = +3.218876$.
   Sum inside: $0.25 + 0.04 + 3.218876 - 1 = 2.508876$. Half: $1.254438$.
3. **Dimension 2:** $\mu_2^2 = (-0.20)^2 = 0.04$; $\sigma_2^2 = 0.09$; $\log 0.09 = -2.407946$, so $-\log\sigma_2^2 = +2.407946$.
   Sum inside: $0.04 + 0.09 + 2.407946 - 1 = 1.537946$. Half: $0.768973$.
4. Add the two dimensions: $1.254438 + 0.768973 = 2.023411$.

**Answer:** $D_{\mathrm{KL}} = 2.0234$ **nats** ($= 2.919$ bits, $= 0.8788$ in base-10 units). Almost all of it comes from the $-\log\sigma^2$ terms: the encoder has squeezed its variances to $0.04$ and $0.09$ against the prior's $1$, and the KL punishes exactly that collapse. Dimension 1 costs more than dimension 2 because it is both tighter ($0.04 < 0.09$) and further from the origin ($0.5 > 0.2$).

### N3. The complete VAE loss for that data point

**Given:** N1's reconstruction estimate and N2's KL, with a Gaussian decoder of fixed variance $\sigma^2_{\text{dec}} = 0.5$ (so the scale factor $\frac{1}{2\sigma^2_{\text{dec}}} = 1$ and the constant is dropped).
**Find:** $\mathcal{L}_{\text{VAE}}(\mathbf{x})$ and the share of each term.

1. Reconstruction loss: $\frac{1}{2\sigma^2_{\text{dec}}}\cdot 0.003367 = 1 \times 0.003367 = 0.003367$.
2. KL loss: $2.023411$.
3. Total: $\mathcal{L}_{\text{VAE}} = 0.003367 + 2.023411 = 2.026778$.
4. Shares: reconstruction $0.003367/2.026778 = 0.166\%$; KL $99.83\%$.

**Answer:** $\mathcal{L}_{\text{VAE}} = 2.0268$ nats, of which the KL term is **99.8%**. This model reconstructs beautifully and matches the prior terribly — the classic signature of a VAE drifting toward autoencoder behaviour. Gradient descent will respond by widening $\sigma^2_\phi$ and pulling $\mu_\phi$ toward zero, which will make reconstruction worse. That trade is the whole of VAE training.

### N4. KL sanity checks you should be able to do in your head

**Given:** the closed form $\tfrac12(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$, one dimension at a time.
**Find:** the KL in four diagnostic cases.

1. **$\mu = 0$, $\sigma^2 = 1$** (exactly the prior): $\tfrac12(0 + 1 - 0 - 1) = \mathbf{0}$. A distribution's divergence from itself is zero.
2. **$\mu = 2$, $\sigma^2 = 1$** (right shape, wrong place): $\tfrac12(4 + 1 - 0 - 1) = \tfrac12(4) = \mathbf{2.0}$ nats. With $\sigma^2 = 1$ the formula collapses to $\mu^2/2$.
3. **$\mu = 0$, $\sigma^2 = 4$** (right place, too wide): $\log 4 = 1.386294$, so $\tfrac12(0 + 4 - 1.386294 - 1) = \tfrac12(1.613706) = \mathbf{0.8069}$ nats.
4. **$\mu = 0$, $\sigma^2 = 0.01$** (right place, collapsed): $\log 0.01 = -4.605170$, so $\tfrac12(0 + 0.01 + 4.605170 - 1) = \tfrac12(3.615170) = \mathbf{1.8076}$ nats.

**Answer:** $0$, $2.0$, $0.8069$, $1.8076$ nats. The asymmetry in cases 3 and 4 is the lesson: a variance **4× too large** costs $0.81$ nats, while a variance **100× too small** costs $1.81$. The $-\log\sigma^2$ term grows without bound as $\sigma^2 \to 0$ but only logarithmically as $\sigma^2$ grows, so the KL fights collapse much harder than it fights over-spreading.

### N5. The Bernoulli decoder's reconstruction term is Lec 11's BCE

**Given:** binary $\mathbf{x} = [1, 0, 1, 1, 0]$ and a decoder emitting $\pi_\theta(\mathbf{z}) = [0.92,\ 0.08,\ 0.81,\ 0.76,\ 0.12]$ — the exact numbers from [Lec 11](11-reconstruction-loss.md)'s page 7.
**Find:** $-\log p_\theta(\mathbf{x}\mid\mathbf{z})$ by multiplying Bernoulli probabilities, and check it against the BCE formula.

1. The Bernoulli probability of each feature, $\pi_j^{x_j}(1-\pi_j)^{1-x_j}$:
   $j=1$ ($x=1$): $0.92$. $j=2$ ($x=0$): $1-0.08 = 0.92$. $j=3$: $0.81$. $j=4$: $0.76$. $j=5$ ($x=0$): $1-0.12 = 0.88$.
2. Features are independent given $\mathbf{z}$, so the joint probability is the product:
   $p_\theta(\mathbf{x}\mid\mathbf{z}) = 0.92 \times 0.92 \times 0.81 \times 0.76 \times 0.88 = 0.458518$.
3. Negative log: $-\log(0.458518) = 0.779754$.
4. Now [Lec 11](11-reconstruction-loss.md)'s BCE on the same numbers: $-\log 0.92 - \log 0.92 - \log 0.81 - \log 0.76 - \log 0.88 = 0.083382 + 0.083382 + 0.210721 + 0.274437 + 0.127833 = 0.779754$.

**Answer:** both routes give $0.779754$ **nats**, identically. $-\log$ of a product is the sum of $-\log$s, which is all the equivalence is — but it means the BCE you were taught as a distance measure in [Lec 11](11-reconstruction-loss.md) was a negative log-likelihood the entire time. No constants are dropped in the Bernoulli case, unlike the Gaussian one.

### N6. Reading the bound: ELBO, true log-likelihood, and the gap

**Given:** for one data point, $\mathcal{L}_{\text{ELBO}}(\mathbf{x}) = -102.4$ nats, and an oracle tells you $D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big) = 3.1$ nats.
**Find:** $\log p_\theta(\mathbf{x})$, and what happens to each quantity if the encoder improves so that the posterior KL falls to $0.4$ while $\theta$ is held fixed.

1. The exact identity from step 4 of the derivation: $\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}} + D_{\mathrm{KL}}(q_\phi\|p_\theta(\mathbf{z}\mid\mathbf{x}))$.
2. $\log p_\theta(\mathbf{x}) = -102.4 + 3.1 = -99.3$ nats.
3. With $\theta$ fixed, $\log p_\theta(\mathbf{x})$ cannot change — it does not involve $\phi$ at all. It stays at $-99.3$.
4. So the new ELBO is $-99.3 - 0.4 = -99.7$ nats.

**Answer:** $\log p_\theta(\mathbf{x}) = -99.3$ nats; improving the encoder raises the ELBO from $-102.4$ to $-99.7$ while the true log-likelihood is unmoved. **Training the encoder does not make the model better; it makes the bound tighter.** And note the sign: an ELBO of $-99.3$ corresponds to $p_\theta(\mathbf{x}) \approx e^{-99.3} \approx 7\times10^{-44}$ — log-likelihoods of real data are large negative numbers, and a "bigger" ELBO means closer to zero.

## Code

Four claims from this chapter, each checked against the deck's own numbers: the Gaussian reconstruction term is MSE plus a constant, the Bernoulli one is exactly BCE, the closed-form KL agrees with brute-force sampling, and the two assemble into the loss.

```python
import numpy as np

# ---------- 1. The reconstruction term with a GAUSSIAN decoder is MSE + a constant
x  = np.array([0.80, 0.10, 0.60, 0.30, 0.90])          # deck page 4, the input
xh = np.array([[0.79, 0.12, 0.58, 0.32, 0.88],         # mu_theta(z) for the deck's
               [0.76, 0.14, 0.61, 0.28, 0.86],         # three sampled latent vectors
               [0.82, 0.09, 0.57, 0.34, 0.91]])

D, s2 = x.size, 0.5                                    # fixed decoder variance
sse        = ((x - xh) ** 2).sum(axis=1)
nll_gauss  = sse / (2 * s2) + 0.5 * D * np.log(2 * np.pi * s2)   # -log p_theta(x|z)
print("squared error per z   =", np.round(sse, 6))
print("-log p(x|z), Gaussian =", np.round(nll_gauss, 6))
print("difference            =", np.round(nll_gauss - sse, 6), " <- same constant")

# ---------- 2. With a BERNOULLI decoder it is exactly Lec 11's BCE, no constant
xb = np.array([1., 0., 1., 1., 0.])                    # Lec 11's binary input
pi = np.array([0.92, 0.08, 0.81, 0.76, 0.12])          # pi_theta(z), the decoder output
pmf       = pi ** xb * (1 - pi) ** (1 - xb)            # Bernoulli pmf, feature by feature
nll_bern  = -np.log(pmf).sum()                         # -log of their product
bce       = -(xb * np.log(pi) + (1 - xb) * np.log(1 - pi)).sum()   # Lec 11's formula
print("\n-log p(x|z), Bernoulli =", round(nll_bern, 6))
print("Lec 11 BCE             =", round(bce, 6))

# ---------- 3. The KL term: closed form against brute-force Monte Carlo
mu, var = np.array([0.50, -0.20]), np.array([0.04, 0.09])        # deck page 4 encoder
kl_closed = 0.5 * np.sum(mu ** 2 + var - np.log(var) - 1)
rng  = np.random.default_rng(0)
z    = mu + np.sqrt(var) * rng.standard_normal((400000, 2))      # samples from q
logq = (-0.5 * np.log(2 * np.pi * var) - (z - mu) ** 2 / (2 * var)).sum(1)
logp = (-0.5 * np.log(2 * np.pi)       -  z        ** 2 / 2     ).sum(1)
print("\nKL closed form =", round(kl_closed, 6))
print("KL Monte Carlo =", round((logq - logp).mean(), 6))

# ---------- 4. The whole loss for this data point
print("\nVAE loss = recon + KL =", round(sse.mean(), 6), "+", round(kl_closed, 6),
      "=", round(sse.mean() + kl_closed, 6))
```

```
squared error per z   = [0.0017 0.0053 0.0031]
-log p(x|z), Gaussian = [2.863525 2.867125 2.864925]
difference            = [2.861825 2.861825 2.861825]  <- same constant

-log p(x|z), Bernoulli = 0.779754
Lec 11 BCE             = 0.779754

KL closed form = 2.023411
KL Monte Carlo = 2.022788

VAE loss = recon + KL = 0.003367 + 2.023411 = 2.026777
```

Read block 1's third line: the difference between the Gaussian negative log-likelihood and the plain squared error is $2.861825$ for *every* sample — a constant, therefore invisible to the gradient, therefore droppable. That is the MSE equivalence, demonstrated rather than asserted. Block 2 shows the Bernoulli case needs no such caveat: $0.779754$ both ways, and that is [Lec 11](11-reconstruction-loss.md)'s own worked example reappearing as a log-likelihood. Block 3 checks the closed-form KL against 400,000 brute-force samples of $\log q - \log p$ and agrees to three decimals — the residual is Monte Carlo noise, and it is precisely the noise the closed form lets you avoid in training.

## Exam pack

### Must-memorise

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

### Numbers worth knowing

| Quantity | Value |
|---|---|
| Deck's input (page 4) | $\mathbf{x} = [0.80, 0.10, 0.60, 0.30, 0.90]$ |
| Deck's encoder output (page 4) | $\mu = [0.50, -0.20]$, $\sigma^2 = [0.04, 0.09]$, so $\sigma = [0.2, 0.3]$ |
| Deck's three latents | $[0.56,-0.11]$, $[0.43,-0.32]$, $[0.61,-0.24]$ |
| Their squared errors | $0.0017$, $0.0053$, $0.0031$ (mean $0.003367$) |
| KL for that encoder output | $2.0234$ nats $=2.919$ bits |
| Full loss for that point | $2.0268$ nats, 99.8% of it KL |
| $D_{\mathrm{KL}}$ when $\mu=0,\sigma^2=1$ | exactly $0$ |
| $D_{\mathrm{KL}}$ when $\sigma^2=1$ | $\mu^2/2$ |
| $D_{\mathrm{KL}}$ when $\mu=0,\sigma^2=4$ | $0.8069$ nats |
| $D_{\mathrm{KL}}$ when $\mu=0,\sigma^2=0.01$ | $1.8076$ nats |
| $\log 0.04$ / $\log 0.09$ | $-3.2189$ / $-2.4079$ |
| Lec 11's BCE example as a log-likelihood | $-\log(0.458518) = 0.779754$ nats |
| Typical $L$ in training | $1$ sample |
| Prior | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ |

### Likely MCQ traps

- **Sign of the KL term.** It is **minus** in the ELBO and **plus** in the loss. Writing $\mathcal{L}_{\text{VAE}} = -\mathbb{E}[\log p_\theta] - D_{\mathrm{KL}}$ is the single commonest error; negating the ELBO flips *both* terms.
- **Confusing the two KLs.** $D_{\mathrm{KL}}(q_\phi\|p_\theta(\mathbf{z}\mid\mathbf{x}))$ is the bound's slack and is never computed. $D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}))$ is in the loss and has a closed form. The first has the **posterior** in slot two; the second has the **prior**.
- **"ELBO is an approximation of $\log p_\theta(\mathbf{x})$."** It is a **lower bound** — always $\leq$, never above. "Approximation" loses the direction, which is the whole content of the word *bound*.
- **"Maximising the ELBO maximises the likelihood exactly."** Only if the bound is tight. In general you raise a floor; the ceiling may not move. See N6.
- **$\mathcal{N}(0.50, 0.04)$ read as $\sigma = 0.04$.** Variance second. $\sigma = 0.2$.
- **Forgetting the $-1$ in the closed-form KL.** Without it, a perfect match $\mu=0, \sigma^2=1$ would score $0.5$ instead of $0$. The $-1$ is what makes zero achievable.
- **Using $\sigma$ where the formula wants $\sigma^2$.** The closed form is $\tfrac12(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$ — variance in both the second and third slots, and $\log\sigma^2$, not $\log\sigma$. ($\log\sigma^2 = 2\log\sigma$.)
- **Thinking MSE and BCE are "chosen" losses.** They are the negative log-likelihoods of a Gaussian and a Bernoulli decoder respectively. Choosing the loss *is* choosing the decoder's distribution.
- **Log base.** Natural logs here; the answers are in **nats**. [Lec 20](20-kl-divergence-b.md)'s discrete KL worked example is in $\log_2$ and reports **bits**, and [Lec 14](14-sparse-ae.md)'s deck uses $\log_{10}$. A distractor that is your answer $\times 1.443$ or $\times 0.434$ is a base trap.
- **"The decoder outputs $\hat{\mathbf{x}}$."** It outputs the **parameters** of $p_\theta(\mathbf{x}\mid\mathbf{z})$ — $\mu_\theta(\mathbf{z})$ and $\sigma^2_\theta(\mathbf{z})$, or $\pi_\theta(\mathbf{z})$. $\hat{\mathbf{x}}$ is a sample from that distribution (in practice, usually just $\mu_\theta(\mathbf{z})$ itself).
- **Thinking the expectation is computed exactly.** It is a one-sample Monte Carlo estimate in essentially every implementation.
- **Treating KL as symmetric.** $D_{\mathrm{KL}}(q\|p) \neq D_{\mathrm{KL}}(p\|q)$. The VAE uses the **reverse** direction, with the approximation first — see [Lec 19](19-kl-divergence-a.md).

### Self-test

1. Write the exact identity relating $\log p_\theta(\mathbf{x})$, the ELBO, and a KL divergence. Which KL is it?
2. Why is the ELBO a *lower* bound, and under what condition is it tight?
3. State the ELBO and then the VAE loss, and explain why the KL term's sign differs between them.
4. A decoder outputs $\pi_\theta(\mathbf{z})$ for each pixel. What distribution is that, and what does the reconstruction term become?
5. Compute $D_{\mathrm{KL}}\big(\mathcal{N}(1.0,\ 0.25)\,\|\,\mathcal{N}(0,1)\big)$ in nats.
6. An encoder outputs $\mu = [0, 0]$ and $\sigma^2 = [1, 1]$ for every input. What is the KL term, what has gone wrong, and what is the condition called?
7. Name every term in $-\tfrac12\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ and say which direction each pushes $\sigma_j^2$.
8. Why can the $\frac{D}{2}\log(2\pi\sigma^2_{\text{dec}})$ term be dropped from a Gaussian decoder's reconstruction loss, but no constant can be dropped from a Bernoulli decoder's?
9. $\mathcal{L}_{\text{ELBO}} = -215.6$ nats and the posterior KL is $4.2$ nats. What is $\log p_\theta(\mathbf{x})$?
10. The expectation $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})]$ cannot be computed exactly. How is it handled, and what problem does that create?

<details><summary>Answers</summary>

1. $\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}}(\mathbf{x}) + D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big)$ — the KL from the **approximate posterior to the true posterior**, not to the prior.
2. Because KL is non-negative ([Lec 19](19-kl-divergence-a.md)), so the ELBO equals $\log p_\theta(\mathbf{x})$ minus something $\geq 0$. It is tight exactly when $q_\phi(\mathbf{z}\mid\mathbf{x}) = p_\theta(\mathbf{z}\mid\mathbf{x})$, i.e. the encoder recovers the true posterior.
3. ELBO $= \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}))$; loss $= -\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] + D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}))$. The loss is the **negated** ELBO (because networks minimise), and negation flips the sign of both terms.
4. A **Bernoulli** distribution, one per pixel. The reconstruction term becomes **binary cross-entropy**, exactly as in [Lec 11](11-reconstruction-loss.md), with no dropped constant.
5. $\tfrac12(1.0^2 + 0.25 - \log 0.25 - 1) = \tfrac12(1 + 0.25 + 1.386294 - 1) = \tfrac12(1.636294) = \mathbf{0.8181}$ nats.
6. $D_{\mathrm{KL}} = \tfrac12(0+1-0-1) \times 2 = \mathbf{0}$ — perfect, and useless. The encoder is ignoring the input entirely, so the latent carries no information and the decoder is fed pure prior noise. This is **posterior collapse**, and it is what the loss looks like when the KL term wins outright.
7. $1$: the constant that makes a perfect match cost zero. $\log\sigma_j^2$: enters with a $+$ inside the bracket and hence a $-$ overall, penalising **small** $\sigma_j^2$ (pushes it up). $\mu_j^2$: penalises means away from the origin. $\sigma_j^2$: penalises **large** variance (pushes it down). The two variance terms balance at $\sigma_j^2 = 1$. The $-\tfrac12\sum_j$ sums over latent dimensions, which is legal because the covariance is diagonal.
8. Because that term depends only on $D$ and the **fixed** decoder variance, not on $\theta$ or $\phi$, so its gradient is zero and dropping it changes nothing. The Bernoulli negative log-likelihood has no parameter-free additive piece at all — every term involves $\pi_\theta(\mathbf{z})$ — so there is nothing to drop; BCE equals the negative log-likelihood exactly.
9. $-215.6 + 4.2 = \mathbf{-211.4}$ nats.
10. By **Monte Carlo**: sample $\mathbf{z}^{(l)} \sim q_\phi(\mathbf{z}\mid\mathbf{x})$ and average $\log p_\theta(\mathbf{x}\mid\mathbf{z}^{(l)})$, in practice with $L=1$. The problem is that sampling is a random, non-differentiable operation, so gradients cannot flow back through it into $\phi$ — solved by the reparameterization trick, [Lec 23](23-reparameterization.md).

</details>

## Beyond the slides

**Gap: the deck asserts the ELBO and never derives it.**
**Why it matters:** page 6 says "ELBO is a computable lower bound of $\log p_\theta(\mathbf{x})$" and page 7 writes the formula down. Without the derivation you cannot answer *why* it is a lower bound (KL $\geq 0$), *what the gap is* (the posterior KL), *when it is tight* ($q_\phi = p_\theta(\mathbf{z}\mid\mathbf{x})$), or *why training $\phi$ helps at all* — and all four are standard exam questions. The seven steps in *The ideas* are the whole content of the lecture; the slide is the summary.

**Gap: page 8 labels the reconstruction term "Mean squared error" and never mentions BCE.**
**Why it matters:** the deck's own page 3 defines a Bernoulli decoder, which gives cross-entropy, not squared error. Using MSE on binary data costs you the gradient advantage [Lec 11](11-reconstruction-loss.md) measured (13× weaker on confidently-wrong units). The correct rule is the one from [Lec 11](11-reconstruction-loss.md), now with a reason underneath it: **the decoder's distribution determines the reconstruction loss** — Gaussian gives MSE, Bernoulli gives BCE — and on MNIST-style $[0,1]$ data, BCE is the standard choice.

**Gap: the closed-form KL is stated for one dimension with no sum, and only against $\mathcal{N}(0,1)$.**
**Why it matters:** real latents have $d$ dimensions, and the formula you need in code is $-\tfrac12\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$. The sum is legal only because the encoder's covariance is **diagonal** (independent coordinates), a fact neither deck states. And it collapses to the simple form only for a standard-normal prior — [Lec 27](27-conditional-vae.md)'s conditional VAE and some diffusion derivations use a non-standard one, which needs the general two-Gaussian expression given above. The deck also has a bracket typo, opening with `[` and closing with `)`.

**Gap: nothing says the expectation is estimated from a single sample.**
**Why it matters:** read literally, $\mathbb{E}_{q_\phi}[\cdot]$ looks like something you evaluate. It is not; it is an integral over $\mathbf{z}$, just as intractable as the one that started the lecture, and training replaces it with **one draw per data point per step**. That single draw is also precisely why [Lec 23](23-reparameterization.md) is needed, and it is why N1's estimate swings by a factor of three across three samples. Any question asking "how many samples of $\mathbf{z}$ does standard VAE training use per input per step?" wants the answer **one**.

**Gap: the loss is presented as a settled formula, with no mention that the balance between its two terms is a choice.**
**Why it matters:** the decoder's fixed variance $\sigma^2_{\text{dec}}$ already acts as a weight on the reconstruction term ($\frac{1}{2\sigma^2_{\text{dec}}}$), so the "plain" VAE loss implicitly contains a tuning knob that nobody labels. Making the knob explicit — $\mathcal{L} = \text{recon} + \beta\,D_{\mathrm{KL}}$ — is [Lec 26](26-beta-vae.md)'s $\beta$-VAE, and N3's 99.8%-KL split is a concrete picture of why anyone would want to turn it. Also unmentioned: *KL annealing*, ramping $\beta$ from 0 upward over early epochs, which is the standard fix for posterior collapse.

## Cut from the slides

Pages 1, 2, 9 and 10 are the course title card, the three-line agenda, the next-session preview and the thank-you; nothing is lost. Page 4's left-hand table of many sampled $\mathbf{z}$'s (the grid of hand-drawn tick marks under $z_1, z_2$) is a sketch of "infinitely many latents" rather than data, so it is described in prose and not tabulated. Pages 5's Bayes block duplicates [Lec 21](21-vae-encoder.md)'s page 7 almost exactly; it is embedded for its *new* content — the prior-density weighting with $\mathbf{z}^1 = [0.2,-0.1]$ against $\mathbf{z}^2 = [4.0,-3.5]$, and the explicit $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ — with the intractability argument itself left to Lec 21 by the ownership map. The handwritten red marks (the arrow to $\log p_\theta(x)$ on page 6, the brace joining the three reconstructions on page 4, the underline under $q_\phi(\mathbf{z}\mid\mathbf{x})$ on page 8) are carried into the prose as emphasis. The deck's rendering of $\phi$ as $\emptyset$ on pages 7 and 8 is silently corrected throughout. The reparameterization trick is named on page 8 and deliberately not taught here: one sentence and a forward link, because [Lec 23](23-reparameterization.md) owns it in full. Everything else on pages 3 through 8 is reproduced, and the two derivations the deck omits — the ELBO and the Gaussian closed-form KL — are supplied in full because no earlier chapter has them.
