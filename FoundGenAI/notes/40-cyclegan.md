# Lec 40 — CycleGAN

> **Source:** `Lec 40.pdf` (17 pages) · **Week 6** · **Playlist:** Lec 40
> **Prereqs:** [Lec 33 — GAN Objective and Loss Functions](33-gan-objective.md), [Lec 34 — GAN Convergence and Nash Equilibrium](34-gan-convergence.md), [Lec 38 — Conditional GAN](38-conditional-gan.md), [Lec 39 — Pix2Pix GAN](39-pix2pix.md)
> **Feeds into:** [Lec 41 — StyleGAN](41-stylegan.md)

## Why this lecture exists

Pix2Pix can turn a sketch into a handbag, but only because somebody photographed the same handbag twice — once as an edge map, once in colour. Every training example is a *pair*, and the L1 term in its loss compares the generator's output against *that specific* target image. Remove the pairing and Pix2Pix has nothing to compute.

That is a crippling restriction. There is no photograph of a particular horse wearing stripes, no Monet painting of a specific Yosemite waterfall, no winter and summer shot of the same instant. For most interesting translations the two domains exist only as two unaligned piles of pictures.

This lecture answers the deck's own question — *can a model learn translation between two domains using only separate image collections, without paired examples?* — and the answer turns out to need one new idea. Drop the pairing and the adversarial loss alone becomes hopelessly underdetermined. **Cycle consistency** is what puts the determination back.

## The ideas

### First, defuse the symbol collision

> **Week 6 uses $x$ and $y$ in three incompatible ways in three consecutive lectures. Fix this before reading another line.**
>
> | Lecture | What $x$ is | What $y$ is | What $D(\cdot)$ is fed |
> |---|---|---|---|
> | [Lec 38 — cGAN](38-conditional-gan.md) | the image | the **condition**: a *class label* | $D(\mathbf{x}, y)$ = image **+ its label** |
> | [Lec 39 — Pix2Pix](39-pix2pix.md) | the **source image** (the condition) | the **target image** (the ground truth) | $D(\mathbf{x}, \mathbf{y})$ = source **+ target**, a stacked pair |
> | **Lec 40 — CycleGAN (here)** | a sample from **domain $X$** | a sample from **domain $Y$** | $D_X(\mathbf{x})$ or $D_Y(\mathbf{y})$ = **one image, one domain** |
>
> Two consequences. First, CycleGAN's discriminators take **one argument**, not two — they are *unconditional*. Any formula you see here with $D(\mathbf{x},\mathbf{y})$ in it is Lec 38's or Lec 39's, not this lecture's. Second, capital $X$ and $Y$ here are **sets of images**, not individual images: $X$ is the pile of horse photographs, $\mathbf{x}$ is one of them. The deck writes all four symbols in plain italics and relies on you to tell them apart from context; this chapter bolds the images ($\mathbf{x}$, $\mathbf{y}$) and leaves the domains plain ($X$, $Y$), per CONTRACT §3.

### What breaks when you remove the pairs

![Slide headed Motivation for CycleGAN: a "Limitation for Conditional GAN — Paired dataset required" box, with three Input→Target pairs (grey bag→green bag, day street→night street, photo→Van Gogh painting) and the bullets "Pix2Pix requires paired datasets", "But collecting such paired datasets is difficult", "Can a model learn translation between two domains using only separate image collections, without paired examples?"](../assets/pages/lec40/p-03.png)
*Fig. — The arrow between each pair is the thing being removed. In Pix2Pix those two images are one training example; here they will become two images in two unrelated folders. Page 3.*

A **paired** dataset is a set of tuples $\{(\mathbf{x}_i, \mathbf{y}_i)\}$ in which $\mathbf{y}_i$ is the known correct translation of $\mathbf{x}_i$. An **unpaired** dataset is two separate collections, $\{\mathbf{x}_i\}_{i=1}^{m}$ from domain $X$ and $\{\mathbf{y}_j\}_{j=1}^{n}$ from domain $Y$, with **no correspondence whatever** between the indices — and typically $m \neq n$.

Pix2Pix's loss has a term $\lVert \mathbf{y} - G(\mathbf{x}) \rVert_1$ in it. Unpaired, there is no $\mathbf{y}$ to put in that slot. You are left with the adversarial loss only. And the adversarial loss, on its own, is not enough — the lecturer says so explicitly on page 11, and that sentence is the hinge of the whole lecture.

> **Adversarial loss only ensures "Output looks realistic". It does NOT ensure output corresponds to input (any horse could become any zebra).** — Lec 40, page 11

Sit with how badly that fails. $D_Y$ looks at one image at a time and asks *is this a plausible zebra?* It never sees the horse that produced it. So a generator that ignores its input entirely — that memorises one convincing zebra and emits it for every horse — scores perfectly on the adversarial loss. So does a generator that maps horses to zebras at random, as long as the *collection* of zebras it produces looks like a real collection of zebras. Nothing in the objective says "this zebra must be *this* horse's zebra", because without pairs nothing in the data says it either.

The problem is **underdetermined**: an enormous family of mappings all achieve the same, optimal adversarial loss, and only a vanishing fraction of them preserve the content of the input. N4 counts them.

### The fix in one sentence

If you cannot check the translation against a target, check it against **itself**. Translate the horse to a zebra, translate that zebra back to a horse, and demand that you get the original horse back. A generator that threw away the input's content cannot possibly do this — the information it would need to rebuild the horse is gone. That is **cycle consistency**, and it is the only genuinely new idea in CycleGAN.

![Slide headed CycleGAN: "CycleGAN learns image-to-image translation between two domains without requiring paired examples", with three bidirectional Source Domain / Target Domain rows — Horse↔Zebra, Winter↔Summer, Photo↔Painting — and the Zhu-Park-Isola-Efros reference](../assets/pages/lec40/p-04.png)
*Fig. — Notice every arrow is **double-headed**. Unlike Pix2Pix, CycleGAN learns both directions at once, and it has to: the backward map is what makes the forward map checkable. The deck's reference is Zhu, Park, Isola and Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", BAIR, UC Berkeley. Page 4.*

### Four networks

![Slide headed Components of CycleGAN Architecture: Domain X (horse) and Domain Y (zebra) photographs, then three labelled boxes — Two Generators (Generator G: X→Y, Generator F: Y→X), Two Discriminators (D_Y checks whether zebra images are real or generated, D_X checks whether horse images are real or generated), and Cycle Consistency Mechanism marked "Key idea introduced in CycleGAN"](../assets/pages/lec40/p-05.png)
*Fig. — The whole architecture on one slide: two generators, two discriminators, one new loss. Count them, because "how many networks does CycleGAN train?" is an exam question with the answer **four**. Page 5.*

| Network | Signature | Job |
|---|---|---|
| $G$ | $X \to Y$ | turn a horse into a zebra |
| $F$ | $Y \to X$ | turn a zebra into a horse |
| $D_X$ | $X \to [0,1]$ | is this horse image real or generated? |
| $D_Y$ | $Y \to [0,1]$ | is this zebra image real or generated? |

The deck writes the generator maps as $G: X \to Y$ and $F: Y \to X$ — note that $F$, not $G^{-1}$, is the name of the backward map, and that it is a *separate network with its own weights*. Nothing constrains $F$ to be the mathematical inverse of $G$; the cycle loss merely pressures it toward being one.

![Slide headed Architecture of CycleGAN-Generators: "Learn Two Generators G and F", G transforms images from domain X (Horse) to domain Y (Zebra), F transforms images from domain Y back to domain X, with a horse and zebra photo, a circular X⇄Y diagram with G on top and F underneath and D_X, D_Y arrows, and the boxed mapping functions G: X→Y and F: Y→X](../assets/pages/lec40/p-06.png)
*Fig. — The circular diagram at the right is the mental model to keep. $G$ goes clockwise, $F$ goes anticlockwise, and the two discriminators sit on the two nodes, not on the arrows. Page 6.*

![Slide headed Architecture of Cycle GAN-Discriminators: "Learn two Adversarial Discriminators D_X and D_Y", D_X checks whether the horse image is real or fake, D_Y checks whether the zebra image is real or fake, with the lower box reading "D_Y: Distinguishes real images from domain Y and generated images G(X). D_X: Distinguishes real images from domain X and generated images F(Y)."](../assets/pages/lec40/p-07.png)
*Fig. — Read the bottom box carefully: $D_Y$'s fakes are $G(\mathbf{x})$ and $D_X$'s fakes are $F(\mathbf{y})$. Each discriminator polices **its own domain** and sees only one generator's output. Getting the subscripts backwards is the classic slip. Page 7.*

Each discriminator is an ordinary real-or-fake classifier of the kind built in [Lec 32](32-gan-architecture.md). In the original paper both are 70×70 PatchGANs — the patch-wise discriminator taught in [Lec 39](39-pix2pix.md), where the 70×70 is derived as a *receptive field* (not a crop size) and the output is a 30×30 grid of decisions from one fully convolutional pass. This deck does not mention PatchGAN at all; go to Lec 39 for it, and note the one structural change CycleGAN makes: **its PatchGANs are unconditional.** Pix2Pix stacks the source image onto the target before judging; CycleGAN's $D_Y$ sees $G(\mathbf{x})$ alone, with no $\mathbf{x}$ attached — because unpaired, there is no meaningful pair to stack.

### Cycle consistency, drawn

![Slide headed Cycle Consistency in CycleGAN: top row — a horse, box "G: Horse → Zebra", a zebra, box "F: Zebra → Horse", a horse labelled "This horse should be same as the original horse". Bottom row — a zebra, box "F: Zebra → Horse", a horse, box "G: Horse → Zebra", a zebra labelled "This zebra should be same as the original zebra". Caption: "The translated image should preserve the underlying content/structure of the original image."](../assets/pages/lec40/p-08.png)
*Fig. — Two cycles, not one. Top row starts in $X$ and must return to $X$; bottom row starts in $Y$ and must return to $Y$. Both are enforced, every step, in the same loss. Page 8.*

Write the two round trips as compositions:

$$\mathbf{x} \;\xrightarrow{\;G\;}\; G(\mathbf{x}) \;\xrightarrow{\;F\;}\; F(G(\mathbf{x})) \;\approx\; \mathbf{x} \qquad\text{(forward cycle)}$$

$$\mathbf{y} \;\xrightarrow{\;F\;}\; F(\mathbf{y}) \;\xrightarrow{\;G\;}\; G(F(\mathbf{y})) \;\approx\; \mathbf{y} \qquad\text{(backward cycle)}$$

The slide's phrasing of *why* this is the right constraint is worth memorising: **"The translated image should preserve the underlying content/structure of the original image."** Cycle consistency is a *proxy* for content preservation. You cannot measure content directly, but you can measure whether enough of it survived the round trip to rebuild the original, and that is almost as good.

Here is the intuition in one line. $F(G(\mathbf{x})) = \mathbf{x}$ forces $G$ to be *injective in effect*: distinct horses must produce distinct zebras, and the zebra must carry the horse's pose, background, lighting and body shape in a form $F$ can read back out. The only thing $G$ is free to change is whatever $F$ can put back for free — which is exactly the domain-defining attribute, the stripes.

### The adversarial loss — two copies of it

The base minimax objective belongs to [Lec 33](33-gan-objective.md). CycleGAN's only modification is to instantiate it **twice**, once per direction, with the deck's symbol $L_{GAN}$ (written $\mathcal{L}_{\text{GAN}}$ here, per CONTRACT §3).

![Slide headed Adversarial Loss: "Adversarial Loss from G, D_Y" with the boxed formula L_GAN(G, D_Y, X, Y) = E_{y~p_data(y)}[log(D_Y(y))] + E_{x~p_data(x)} log[(1 − D_Y(G(x)))], annotated "Trains discriminator D_Y to classify real zebra images as real" and "Trains discriminator D_Y to classify generated zebra images as fake". Desired output: D_Y(fake zebra)→0, G tries to fool D_Y, G wants D_Y(G(x))→1. Generator G tries to minimize the loss, Discriminator D_Y tries to maximize the loss.](../assets/pages/lec40/p-10.png)
*Fig. — Term by term: the left expectation is over **real** $\mathbf{y}$ drawn from domain $Y$, the right over **real** $\mathbf{x}$ pushed through $G$. The two "desired output" lines look contradictory until you notice the first is $D_Y$'s wish and the second is $G$'s. Page 10.*

$$\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) \;=\; \mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}\big[\log D_Y(\mathbf{y})\big] \;+\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}\big[\log\big(1 - D_Y(G(\mathbf{x}))\big)\big]$$

![Slide headed Adversarial Loss: "Adversarial Loss from F, D_X" with the boxed formula L_GAN(F, D_X, X, Y) = E_{x~p_data(x)}[log(D_X(x))] + E_{y~p_data(y)}[log(1 − D_X(F(y)))], the same two annotations for horses, the desired outputs D_X(fake horse)→0 and D_X(F(y))→1, and in red "Adversarial loss only ensures 'Output looks realistic'. It does NOT ensure Output corresponds to input (any horse could become any zebra)"](../assets/pages/lec40/p-11.png)
*Fig. — The two red ticks at the bottom are the most important sentences in the deck. Everything after this page exists to repair the defect they name. Page 11.*

$$\mathcal{L}_{\text{GAN}}(F, D_X, Y, X) \;=\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}\big[\log D_X(\mathbf{x})\big] \;+\; \mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}\big[\log\big(1 - D_X(F(\mathbf{y}))\big)\big]$$

The structure is identical; only the subscripts and the roles of $\mathbf{x}$ and $\mathbf{y}$ swap. Both are the **minimax (saturating) form** from [Lec 33](33-gan-objective.md): the discriminator maximises, the generator minimises, and the value is always $\le 0$ because both logs take arguments in $(0,1)$.

Three facts to carry into N1 and the exam:

- $\mathcal{L}_{\text{GAN}} \le 0$ always; its maximum, $0$, is reached only by a perfect discriminator ($D_Y(\mathbf{y}) = 1$ on every real, $D_Y(G(\mathbf{x})) = 0$ on every fake).
- At the Nash equilibrium of [Lec 34](34-gan-convergence.md) the discriminator is helpless and outputs $0.5$ on everything, giving $\log 0.5 + \log 0.5 = -2\ln 2 = -1.3863$ nats (= $-2$ bits).
- In practice CycleGAN is trained with the **least-squares** variant rather than these logs — a genuine gap in the deck, covered in *Beyond the slides*.

### The cycle-consistency loss

![Slide headed Cycle Consistency Losses in CycleGAN with the red formula L_cyc(G,F) = E_{x~pdata(x)}[‖F(G(x)) − x‖_1] + E_{y~pdata(y)}[‖G(F(y)) − y‖_1], the left term labelled "Forward Cycle Loss — Original horse vs reconstructed horse", the right "Backward Cycle Loss — Original zebra vs reconstructed zebra", "These two errors should be minimized", and two chain diagrams: Horse x → G → Zebra G(x) → F → Horse F(G(x)), and Zebra y → F → Horse F(y) → G → Zebra G(F(y))](../assets/pages/lec40/p-12.png)
*Fig. — The two chains at the bottom are the formula redrawn. Note the subscript 1 on both norms — this is **L1**, absolute differences, not squared. Page 12.*

$$\mathcal{L}_{\text{cyc}}(G, F) \;=\; \underbrace{\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}\big[\lVert F(G(\mathbf{x})) - \mathbf{x} \rVert_1\big]}_{\text{forward cycle loss}} \;+\; \underbrace{\mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}\big[\lVert G(F(\mathbf{y})) - \mathbf{y} \rVert_1\big]}_{\text{backward cycle loss}}$$

Unpack the pieces:

- $\lVert \mathbf{a} \rVert_1 = \sum_k |a_k|$ — the **L1 norm**, the sum of absolute values over every pixel and channel. It is the same norm Pix2Pix uses, for the same reason (see [Lec 39](39-pix2pix.md)): L1 penalises large errors less harshly than L2 and so produces visibly sharper images.
- The **forward** term compares a real horse with a reconstructed horse. Both live in $X$. Nothing in it touches domain $Y$'s data.
- The **backward** term compares a real zebra with a reconstructed zebra. Both live in $Y$.
- Each term uses only **one** real image — $\mathbf{x}$, or $\mathbf{y}$. That is precisely why it works unpaired: it never needs to know which $\mathbf{y}$ corresponds to which $\mathbf{x}$, because it compares $\mathbf{x}$ with itself.

That last point is the heart of the lecture. **The cycle loss manufactures a target out of the input.** Pix2Pix's L1 term needs a ground-truth $\mathbf{y}$ supplied by a human; CycleGAN's L1 term uses $\mathbf{x}$, which you already have.

### The complete objective

![Slide headed Objective Function of CycleGAN: the "Complete CycleGAN loss" L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λL_cycle(G, F), with the three terms annotated "Adversarial Loss for horse to zebra", "Adversarial Loss for zebra to horse", "Cycle consistency Loss — In original paper λ=10"; below, the objective G*, F* = arg min_{G,F} max_{D_X,D_Y} L(G, F, D_X, D_Y), split into "Find the values of G and F that minimize the loss" and "Find D_X and D_Y that maximize the loss", with boxes "Generators (G,F) try to: fool discriminators, maintain cycle consistency → minimize loss" and "Discriminators (D_X, D_Y) try to correctly classify real vs fake → maximize loss"](../assets/pages/lec40/p-13.png)
*Fig. — Three terms, one hyperparameter. $\lambda = 10$ is a number to memorise: it is on the slide and it is exactly the kind of thing an MCQ keys on. Page 13.*

$$\mathcal{L}(G, F, D_X, D_Y) \;=\; \mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) \;+\; \mathcal{L}_{\text{GAN}}(F, D_X, Y, X) \;+\; \lambda\,\mathcal{L}_{\text{cyc}}(G, F)$$

$$G^*, F^* \;=\; \arg\min_{G,F}\;\max_{D_X, D_Y}\; \mathcal{L}(G, F, D_X, D_Y)$$

Read the min–max carefully — the deck lays out exactly who does what:

| Who | Direction | Wants |
|---|---|---|
| $G$ and $F$ | **minimise** $\mathcal{L}$ | fool the discriminators **and** maintain cycle consistency |
| $D_X$ and $D_Y$ | **maximise** $\mathcal{L}$ | correctly classify real vs fake |

Two consequences of the single $\lambda$ sitting on only the third term:

1. **$\lambda$ is asymmetric on purpose.** The cycle term is a sum of pixel-wise absolute differences over an entire image — potentially thousands of numbers — whereas each adversarial term is a pair of logs bounded in $[-\infty, 0]$ and typically of order 1. Without a large $\lambda$ the cycle constraint is numerically present but practically ignored. $\lambda = 10$ is the paper's choice.
2. **The discriminators do not see $\lambda\mathcal{L}_{\text{cyc}}$ at all.** It contains no $D$, so $\partial(\lambda\mathcal{L}_{\text{cyc}})/\partial D = 0$ and it is invisible to the max player. Cycle consistency is purely a generator-side constraint.

### Pix2Pix vs CycleGAN — the contrast table

This is the single most examinable discrimination in the GAN arc. Learn it as a table, not as prose.

| | **Pix2Pix** ([Lec 39](39-pix2pix.md)) | **CycleGAN** (this lecture) |
|---|---|---|
| **Data requirement** | **paired** — aligned $(\mathbf{x}_i, \mathbf{y}_i)$ tuples | **unpaired** — two independent collections, $m \neq n$ allowed |
| **Generators** | **1** ($G: X \to Y$) | **2** ($G: X\to Y$ and $F: Y\to X$) |
| **Discriminators** | **1** ($D$, on domain $Y$) | **2** ($D_X$ on $X$, $D_Y$ on $Y$) |
| **Networks trained** | 2 | **4** |
| **Loss terms** | adversarial (conditional) $+\ \lambda\lVert \mathbf{y} - G(\mathbf{x})\rVert_1$ | $\mathcal{L}_{\text{GAN}}(G,D_Y) + \mathcal{L}_{\text{GAN}}(F,D_X) + \lambda\mathcal{L}_{\text{cyc}}$ |
| **What the L1 term compares** | output vs the **ground-truth target** $\mathbf{y}_i$ | reconstruction vs **its own input** $\mathbf{x}$ |
| **Why the L1 term had to change** | — | Pix2Pix's L1 needs a *specific* ground-truth image; unpaired there is none, so the target is manufactured by the round trip ([Lec 39](39-pix2pix.md) names this mechanism) |
| **Where supervision comes from** | the human who built the pairs | the round trip (self-supervision) |
| **Noise vector $\mathbf{z}$** | **none** — the deck writes $G(\mathbf{x})$, never $G(\mathbf{x},\mathbf{z})$; near-deterministic, models $p(\mathbf{y}\mid\mathbf{x})$ as a point | **none either** — $G(\mathbf{x})$, $F(\mathbf{y})$; both are deterministic image-to-image maps, unlike the $G(\mathbf{z})$ of [Lec 32](32-gan-architecture.md) |
| **Discriminator conditioning** | conditional — sees $(\mathbf{x}, \mathbf{y})$ stacked together | **unconditional** — sees one image, one domain |
| **Directions learned** | one, $X \to Y$ | **both**, simultaneously |
| **$\lambda$ in the paper** | 100 | **10** |
| **Typical use case** | edges→photo, map→aerial, labels→façade, sketch→colour, B&W→colour where aligned data exists | horse↔zebra, summer↔winter, photo↔Monet/Van Gogh/Cézanne, apple↔orange |
| **Fails at** | anything you cannot photograph twice | large geometric change (shape stays, texture moves) |

The one-sentence version: **Pix2Pix compares the output to a target; CycleGAN compares the round trip to the input.** Everything else in the table follows from that.

### What the method actually achieves

![Slide headed Transformations taken from original paper: left, a grid of input/output pairs for horse→zebra, zebra→horse, winter Yosemite→summer Yosemite, summer→winter, apple→orange, orange→apple. Right, Figure 15 from the paper comparing CycleGAN against Gatys et al. neural style transfer on Photo→Van Gogh, Photo→Ukiyo-e, Photo→Cézanne](../assets/pages/lec40/p-14.png)
*Fig. — Look at what changes and what does not: the horse's pose, the grass, the sky and the camera angle all survive; only the coat changes. That invariance is cycle consistency doing its job, visible. The right panel is the paper's comparison against classical neural style transfer, where CycleGAN learns the artist's whole *collection* rather than copying one painting. Page 14.*

The results also show the method's limit. Every transformation on that slide is a **texture or colour** change on a fixed geometry. Horse→zebra keeps the horse's silhouette exactly; it does not turn a horse into a giraffe. Cycle consistency is partly responsible: a large shape change destroys the information $F$ would need to rebuild the original, so the cycle loss actively resists it.

## Worked numericals

> **The slides contain no worked arithmetic at all.** Lec 40 is entirely conceptual — formulas and diagrams, no numbers except $\lambda = 10$. All six numericals below are constructed, and all are the kind an NPTEL paper actually sets on this material.

### N1. Both adversarial losses from discriminator outputs

**Given:** one real zebra with $D_Y(\mathbf{y}) = 0.9$ and one generated zebra with $D_Y(G(\mathbf{x})) = 0.3$; one real horse with $D_X(\mathbf{x}) = 0.85$ and one generated horse with $D_X(F(\mathbf{y})) = 0.25$. Batch size 1 in each direction, so each expectation is just the single value.
**Find:** both adversarial losses, in nats.

1. Horse→zebra direction:
$$\mathcal{L}_{\text{GAN}}(G, D_Y) = \log(0.9) + \log(1 - 0.3) = \log(0.9) + \log(0.7)$$
2. $\ln 0.9 = -0.105361$, $\ln 0.7 = -0.356675$.
3. Sum: $-0.105361 - 0.356675 = -0.462036$.
4. Zebra→horse direction:
$$\mathcal{L}_{\text{GAN}}(F, D_X) = \log(0.85) + \log(1 - 0.25) = \log(0.85) + \log(0.75)$$
5. $\ln 0.85 = -0.162519$, $\ln 0.75 = -0.287682$.
6. Sum: $-0.162519 - 0.287682 = -0.450201$.

**Answer:** $\mathcal{L}_{\text{GAN}}(G, D_Y) = -0.4620$ nats and $\mathcal{L}_{\text{GAN}}(F, D_X) = -0.4502$ nats. **Natural logs** throughout, per CONTRACT §3; in bits divide by $\ln 2 = 0.6931$, giving $-0.6666$ and $-0.6495$ bits. Both are negative, as every adversarial loss in this form must be, and both are *far* from the maximum of $0$ — these discriminators are doing a mediocre job.

### N2. Forward and backward cycle loss on a 2×2 patch

**Given:** a 2×2 greyscale horse patch and its round-trip reconstruction, and a 2×2 zebra patch and its round-trip reconstruction:

$$\mathbf{x} = \begin{bmatrix} 0.80 & 0.20 \\ 0.50 & 0.90 \end{bmatrix},\quad F(G(\mathbf{x})) = \begin{bmatrix} 0.70 & 0.35 \\ 0.55 & 0.60 \end{bmatrix},\quad \mathbf{y} = \begin{bmatrix} 0.40 & 0.60 \\ 0.10 & 0.30 \end{bmatrix},\quad G(F(\mathbf{y})) = \begin{bmatrix} 0.45 & 0.50 \\ 0.20 & 0.30 \end{bmatrix}$$

**Find:** the forward cycle loss, the backward cycle loss, and $\mathcal{L}_{\text{cyc}}$.

1. Forward residuals $F(G(\mathbf{x})) - \mathbf{x}$: $\;-0.10,\; +0.15,\; +0.05,\; -0.30$.
2. Absolute values: $0.10,\ 0.15,\ 0.05,\ 0.30$.
3. Forward cycle loss $= 0.10 + 0.15 + 0.05 + 0.30 = 0.60$.
4. Backward residuals $G(F(\mathbf{y})) - \mathbf{y}$: $\;+0.05,\; -0.10,\; +0.10,\; 0.00$.
5. Absolute values: $0.05,\ 0.10,\ 0.10,\ 0.00$.
6. Backward cycle loss $= 0.05 + 0.10 + 0.10 + 0.00 = 0.25$.
7. $\mathcal{L}_{\text{cyc}} = 0.60 + 0.25 = 0.85$.

**Answer:** forward $= 0.60$, backward $= 0.25$, $\mathcal{L}_{\text{cyc}} = \mathbf{0.85}$ (in pixel-intensity units, $\lVert\cdot\rVert_1$ as a **sum**, exactly as the deck writes it).

> **The sum-vs-mean trap.** The deck's $\lVert\cdot\rVert_1$ is a sum over all elements. PyTorch's `nn.L1Loss()` defaults to `reduction='mean'`, which divides by the element count — here 4. That would give $0.15$, $0.0625$ and $0.2125$. The *ranking* is unchanged but every number is 4× smaller, and $\lambda = 10$ then means something completely different. Always check which convention a question is using; the deck's is the sum.

### N3. The complete objective, assembled

**Given:** the two adversarial losses from N1 and the cycle loss from N2, with the paper's $\lambda = 10$.
**Find:** $\mathcal{L}(G, F, D_X, D_Y)$, and the share each term contributes.

1. Adversarial contribution: $-0.462036 + (-0.450201) = -0.912237$.
2. Cycle contribution: $\lambda\,\mathcal{L}_{\text{cyc}} = 10 \times 0.85 = 8.50$.
3. Total: $-0.912237 + 8.50 = 7.587763$.
4. Magnitudes: $|{-0.912237}| = 0.912$ against $8.50$. Ratio $8.50 / 0.912 = 9.32$.

**Answer:** $\mathcal{L} = \mathbf{7.5878}$ (adversarial terms in nats, cycle term in pixel units — the sum is dimensionally mixed, which is normal for a weighted multi-task loss). The cycle term is **9.3× larger** than the two adversarial terms combined. That asymmetry is the point of $\lambda = 10$: it guarantees the generators care more about not destroying content than about fooling the discriminator.

Now rerun with $\lambda = 1$: total $= -0.912237 + 0.85 = -0.062$. The cycle term no longer dominates, and the generators would happily sacrifice the round trip for a slightly better adversarial score. **Dropping $\lambda$ from 10 to 1 is how you reproduce the failure mode the lecture exists to prevent.**

### N4. How many mappings satisfy the adversarial loss alone?

**Given:** a toy world with $n$ horses in domain $X$ and $n$ zebras in domain $Y$, both uniformly distributed. $G$ must be a function $X \to Y$ whose output distribution is exactly $p_{\text{data}}(Y)$ — this is precisely what a perfectly-fooled $D_Y$ demands and nothing more.
**Find:** how many such $G$ exist, for $n = 3$ and $n = 10$, and the chance of landing on the content-preserving one at random.

1. To emit each of the $n$ zebras with probability exactly $1/n$ while taking each of the $n$ horses as input once, $G$ must be a **bijection** — a one-to-one pairing.
2. The number of bijections on $n$ elements is $n!$.
3. $n = 3$: $3! = 3\times2\times1 = 6$.
4. $n = 10$: $10! = 3{,}628{,}800$.
5. Exactly one of them is the correct content-preserving pairing, so the chance of hitting it is $1/n!$: $1/6 = 0.1667$ for $n=3$, and $1/3{,}628{,}800 = 2.76\times10^{-7}$ for $n=10$.

**Answer:** $6$ and $3{,}628{,}800$ optimal-but-mostly-wrong mappings; a $2.76\times10^{-7}$ chance at $n=10$. This is the deck's "any horse could become any zebra", counted. With a realistic $n$ in the thousands, $n!$ exceeds the number of atoms in the observable universe — the adversarial loss, alone, tells you essentially nothing about which translation you got.

> **The honest footnote, which the deck does not give you.** Cycle consistency does **not** reduce this count to 1. If $G$ is any bijection and $F = G^{-1}$, then $F(G(\mathbf{x})) = \mathbf{x}$ exactly, so *all* $n!$ pairings satisfy the cycle loss perfectly too. What actually breaks the tie is that $G$ and $F$ are small convolutional networks with finite capacity: encoding an arbitrary permutation is enormously expensive for them, whereas "change the coat texture, keep everything else" is cheap. Cycle consistency makes the right answer *reachable*; the architecture makes it *preferred*. See *Beyond the slides*.

### N5. Counting networks and parameters

**Given:** a generator with $11.4$ million parameters and a discriminator with $2.8$ million, the same sizes used for both directions.
**Find:** the trainable-parameter count and network count for Pix2Pix and for CycleGAN.

1. Pix2Pix: 1 generator + 1 discriminator $= 2$ networks.
2. Pix2Pix parameters: $11.4 + 2.8 = 14.2$ M.
3. CycleGAN: 2 generators + 2 discriminators $= 4$ networks.
4. CycleGAN parameters: $2(11.4) + 2(2.8) = 22.8 + 5.6 = 28.4$ M.
5. Ratio: $28.4 / 14.2 = 2.0$.

**Answer:** $2$ networks / $14.2$ M against $4$ networks / $28.4$ M — **exactly double**. Removing the pairing requirement is not free; it costs you a second generator, a second discriminator, two extra forward passes per step (the two round trips) and roughly twice the memory. That trade — *human labelling effort for compute* — is the real content of the Pix2Pix/CycleGAN choice.

### N6. L1 against L2 on the same cycle residual

**Given:** the forward residuals from N2: $0.10,\ 0.15,\ 0.05,\ 0.30$.
**Find:** the L1 cycle loss, the squared-error alternative, and which pixel dominates each.

1. L1: $0.10 + 0.15 + 0.05 + 0.30 = 0.60$.
2. Squares: $0.01,\ 0.0225,\ 0.0025,\ 0.09$. Sum of squares $= 0.125$. L2 norm $= \sqrt{0.125} = 0.35355$.
3. Share of L1 owned by the worst pixel: $0.30 / 0.60 = 50.0\%$.
4. Share of the sum of squares owned by the worst pixel: $0.09 / 0.125 = 72.0\%$.

**Answer:** L1 $= 0.60$, sum of squares $= 0.125$, L2 norm $= 0.3536$. Under squared error the single worst pixel claims **72%** of the gradient against **50%** under L1. Squared error therefore drives the generator to hedge — to output a blurry average that avoids any large error anywhere — which is exactly the blurriness complaint from [Lec 31](31-gan-motivation.md). **CycleGAN uses L1 for sharpness**, the same reason Pix2Pix does.

## Code

The deck asserts that the adversarial loss cannot tell a content-preserving translation from a scrambled one. Twenty lines make that concrete: build two generators that emit *the identical set* of outputs, confirm the adversarial loss is bit-for-bit identical, then watch the cycle loss separate them.

```python
import numpy as np

# Domain X: four "horses", each a 2x2 greyscale patch flattened to 4 numbers.
X = np.array([[0.9, 0.1, 0.1, 0.9],
              [0.1, 0.9, 0.9, 0.1],
              [0.8, 0.8, 0.2, 0.2],
              [0.2, 0.2, 0.8, 0.8]])

stripe = lambda v: 1.0 - v            # the "horse -> zebra" edit, exactly invertible

# Two candidate generators G. They emit the SAME FOUR zebras as a set;
# G_bad just hands each horse somebody else's zebra.
G_good = stripe(X)
G_bad  = stripe(X)[[2, 3, 0, 1]]

Yset = stripe(X)                      # the real zebra collection D_Y was trained on
def d_y(img):                         # a PERFECT D_Y: 0.5 on anything in the zebra set
    return 0.5 if any(np.allclose(img, r) for r in Yset) else 0.0

def adv(out):                         # generator's half of L_GAN: E[log(1 - D_Y(G(x)))]
    return float(np.mean([np.log(1.0 - d_y(r)) for r in out]))

def cyc(out, F=stripe):               # forward cycle loss, L1, averaged per image
    return float(np.sum(np.abs(F(out) - X)) / len(X))

lam = 10
for name, out in [("G_good", G_good), ("G_bad ", G_bad)]:
    a, c = adv(out), cyc(out)
    print(f"{name}  adversarial={a: .6f}   cycle L1={c: .6f}   "
          f"total={a + lam * c: .6f}")
```

```
G_good  adversarial=-0.693147   cycle L1= 0.000000   total=-0.693147
G_bad   adversarial=-0.693147   cycle L1= 1.600000   total= 15.306853
```

Read the two lines. The adversarial term is **exactly equal** — $-0.693147 = \ln 0.5$ — for the generator that translates each horse correctly and the one that shuffles the assignments. $D_Y$ sees the same four zebras either way and has no mechanism to object. The cycle loss is $0$ against $1.6$, and with $\lambda = 10$ the totals are $-0.69$ against $+15.31$: a gap of 16. **The adversarial loss cannot see the difference; the cycle loss sees nothing else.** That is the entire lecture, in two printed lines.

## Exam pack

### Must-memorise

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

### Numbers worth knowing

| Quantity | Value |
|---|---|
| $\lambda$ in the original paper | $\mathbf{10}$ |
| Generators | 2 |
| Discriminators | 2 |
| Total networks | 4 |
| Terms in the complete objective | 3 (two adversarial + one cycle) |
| Terms inside $\mathcal{L}_{\text{cyc}}$ | 2 (forward + backward) |
| Terms inside one $\mathcal{L}_{\text{GAN}}$ | 2 (real log + fake log) |
| Paired images CycleGAN needs | **0** |
| $\mathcal{L}_{\text{GAN}}$ at Nash equilibrium ($D\equiv0.5$) | $2\ln 0.5 = -1.3863$ **nats** $= -2$ **bits** |
| Maximum possible $\mathcal{L}_{\text{GAN}}$ | $0$ (perfect discriminator) |
| Range of $\mathcal{L}_{\text{GAN}}$ | $(-\infty,\ 0]$ |
| Minimum possible $\mathcal{L}_{\text{cyc}}$ | $0$ (perfect round trip, attainable) |
| Pix2Pix's $\lambda$, for contrast | 100 |
| Optimal-but-wrong mappings for $n$ images | $n!$ ($10! = 3{,}628{,}800$) |
| Parameter cost vs Pix2Pix | exactly $2\times$ |

### Likely MCQ traps

- **The three-way $x$/$y$ collision across Lec 38, 39 and 40.** In [Lec 38](38-conditional-gan.md) $y$ is a **class label** and $D(\mathbf{x},y)$ means image-plus-label. In [Lec 39](39-pix2pix.md) $\mathbf{x}$ is the **source image** and $\mathbf{y}$ the **target image**, and $D(\mathbf{x},\mathbf{y})$ means source-plus-target. Here $X$ and $Y$ are **domains** (whole collections), $\mathbf{x}$ and $\mathbf{y}$ are single images from them, and CycleGAN's discriminators are **unconditional** — $D_X$ and $D_Y$ each take **one** argument. Read the discriminator's arity first: two arguments means Lec 38 or 39, one argument means Lec 40. See the [boxed warning](#first-defuse-the-symbol-collision) above.
- **"CycleGAN's generator takes a noise vector $\mathbf{z}$."** It does not — neither does Pix2Pix. Both are deterministic image-to-image maps written $G(\mathbf{x})$. Only the vanilla GAN of [Lec 32](32-gan-architecture.md) and StyleGAN of [Lec 41](41-stylegan.md) start from $\mathbf{z}$.
- **Paired vs unpaired — the single most examinable line in the arc.** Pix2Pix *requires* $(\mathbf{x}_i,\mathbf{y}_i)$ tuples; CycleGAN requires *none*, only two separate collections of possibly different sizes. See the [contrast table](#pix2pix-vs-cyclegan--the-contrast-table) above and memorise it whole: 1 vs 2 generators, 1 vs 2 discriminators, 2 vs 4 networks, $\lambda=100$ vs $\lambda=10$.
- **"CycleGAN has one generator and two discriminators."** No. **Two of each.** Two discriminators without two generators would be incoherent — there would be nothing generating domain-$X$ images for $D_X$ to judge.
- **Thinking $F = G^{-1}$ by construction.** $F$ is an independent network with its own weights. The cycle loss *encourages* it toward being $G$'s inverse; nothing enforces it, and in a half-trained model $F(G(\mathbf{x})) \neq \mathbf{x}$.
- **Using L2 in the cycle loss.** The deck writes $\lVert\cdot\rVert_1$ twice, with a visible subscript 1. L1 is chosen for sharpness (N6).
- **Putting $\lambda$ on the adversarial terms.** $\lambda$ multiplies **only** $\mathcal{L}_{\text{cyc}}$. An option showing $\lambda\mathcal{L}_{\text{GAN}} + \mathcal{L}_{\text{cyc}}$ is wrong.
- **Forgetting the backward cycle.** $\mathcal{L}_{\text{cyc}}$ has **two** expectations. An option listing only $\mathbb{E}_{\mathbf{x}}[\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1]$ is half the formula.
- **Swapping the discriminator subscripts.** $D_Y$ judges **zebras** ($Y$) and sees fakes $G(\mathbf{x})$; $D_X$ judges **horses** ($X$) and sees fakes $F(\mathbf{y})$. The discriminator's subscript is its *domain*, not the generator that feeds it.
- **"Cycle consistency makes the adversarial loss unnecessary."** The reverse of the real relationship. Without $\mathcal{L}_{\text{GAN}}$, the identity map $G = F = \text{id}$ drives $\mathcal{L}_{\text{cyc}}$ to zero and produces no translation at all. Each loss plugs the other's hole: adversarial gives realism, cycle gives correspondence.
- **"CycleGAN needs fewer images than Pix2Pix."** It needs *unaligned* images, not fewer. It typically wants more, because it has four networks to fit.
- **Mixing up which image each cycle term compares.** Forward compares a **horse with a horse** ($\mathbf{x}$ vs $F(G(\mathbf{x}))$), backward a **zebra with a zebra**. Neither ever compares a horse with a zebra — that comparison is exactly what unpaired data makes impossible.
- **Log base.** Natural logs throughout this book. $2\ln 0.5 = -1.3863$ nats but $2\log_2 0.5 = -2$ bits. State the base in your answer (errata batch 4).

### Self-test

1. State the two generator mappings and the two discriminators of CycleGAN, with the domain each discriminator polices.
2. Write $\mathcal{L}_{\text{cyc}}(G,F)$ in full and name both of its terms.
3. In one sentence, why is the adversarial loss alone insufficient for unpaired translation?
4. Give the complete CycleGAN objective and the value of $\lambda$ used in the original paper.
5. $D_Y(\mathbf{y}) = 0.8$ on a real zebra and $D_Y(G(\mathbf{x})) = 0.4$ on a fake. Compute $\mathcal{L}_{\text{GAN}}(G, D_Y)$ in nats.
6. A 2×2 patch $\mathbf{x} = \begin{bmatrix}0.5 & 0.5\\0.5 & 0.5\end{bmatrix}$ reconstructs as $\begin{bmatrix}0.6 & 0.4\\0.5 & 0.8\end{bmatrix}$. Compute the forward cycle loss, and its contribution to the objective at $\lambda = 10$.
7. Which of the four networks receives gradient from the $\lambda\mathcal{L}_{\text{cyc}}$ term, and why?
8. Give four differences between Pix2Pix and CycleGAN.
9. If you set $\lambda = 0$, what degenerate solution becomes optimal? If you delete both adversarial terms instead, what becomes optimal?
10. Why can CycleGAN change a horse's coat but not turn a horse into a giraffe?

<details><summary>Answers</summary>

1. $G: X \to Y$ (horse→zebra) and $F: Y \to X$ (zebra→horse). $D_Y$ polices **domain $Y$** — it separates real zebras from $G(\mathbf{x})$. $D_X$ polices **domain $X$** — it separates real horses from $F(\mathbf{y})$.
2. $\mathcal{L}_{\text{cyc}}(G,F) = \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}(\mathbf{x})}[\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1] + \mathbb{E}_{\mathbf{y}\sim p_{\text{data}}(\mathbf{y})}[\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1]$. First term: **forward cycle loss** (original horse vs reconstructed horse). Second: **backward cycle loss** (original zebra vs reconstructed zebra).
3. Because $D_Y$ judges each output in isolation against domain $Y$ and never sees the input, so any mapping whose outputs look like plausible zebras is optimal — the output need have nothing to do with the particular horse that produced it ("any horse could become any zebra").
4. $\mathcal{L} = \mathcal{L}_{\text{GAN}}(G,D_Y,X,Y) + \mathcal{L}_{\text{GAN}}(F,D_X,Y,X) + \lambda\mathcal{L}_{\text{cyc}}(G,F)$, with $G^*,F^* = \arg\min_{G,F}\max_{D_X,D_Y}\mathcal{L}$. $\lambda = 10$.
5. $\ln 0.8 + \ln 0.6 = -0.223144 - 0.510826 = \mathbf{-0.73397}$ nats ($= -1.0589$ bits).
6. Residuals $+0.1, -0.1, 0.0, +0.3$; absolute values $0.1, 0.1, 0, 0.3$; forward cycle loss $= \mathbf{0.5}$. Contribution at $\lambda=10$: $\mathbf{5.0}$.
7. Only $G$ and $F$. The term contains no $D_X$ or $D_Y$, so its derivative with respect to the discriminators' parameters is identically zero. Cycle consistency is a generator-side constraint.
8. Any four of: paired vs unpaired data; 1 vs 2 generators; 1 vs 2 discriminators; L1 against a ground-truth target vs L1 against the round-trip reconstruction; conditional vs unconditional discriminator; one direction vs both directions; $\lambda = 100$ vs $\lambda = 10$.
9. $\lambda = 0$ removes cycle consistency, so any content-destroying mapping that produces plausible zebras is optimal — including the one that emits the same zebra for every horse. Deleting the adversarial terms instead makes $G = F = \text{identity}$ optimal: $\mathcal{L}_{\text{cyc}} = 0$ exactly, and no translation happens at all.
10. A large geometric change would destroy the spatial information $F$ needs to rebuild the original horse, so the cycle loss penalises it heavily. Cycle consistency therefore biases the model toward texture and colour changes on a preserved geometry — visible in every example on page 14.

</details>

## Beyond the slides

**Gap: the deck never says cycle consistency fails to make the solution unique.**
**Why it matters:** as N4 shows, if $G$ is any bijection and $F = G^{-1}$ then the cycle loss is exactly zero — all $n!$ scrambled pairings satisfy it perfectly. The constraint narrows the solution set from "all mappings" to "invertible mappings", which is a huge reduction but not down to one. What actually selects the content-preserving map is the *inductive bias of the convolutional generators*: a near-identity, locally-acting transformation is cheap to represent, while an arbitrary permutation of the dataset is astronomically expensive. Any question phrased as "cycle consistency *guarantees* correct translation" is false; the right word is *encourages*, exactly as [Lec 26](26-beta-vae.md) says of $\beta$ and disentanglement.

**Gap: the deck shows the log-based adversarial loss, but CycleGAN is trained with the least-squares form.**
**Why it matters:** the original paper replaces $\log D$ and $\log(1-D)$ with squared errors, $\mathbb{E}[(D_Y(\mathbf{y})-1)^2] + \mathbb{E}[D_Y(G(\mathbf{x}))^2]$ (the LSGAN objective), because the log form saturates and destabilises training — the vanishing-gradient problem of [Lec 33](33-gan-objective.md) and [Lec 34](34-gan-convergence.md). The deck teaches the log form, so answer exam questions with the log form; but know that a real implementation you read will not match the slide, and that the *structure* (two terms, real pushed to 1, fake pushed to 0) is identical either way.

**Gap: identity loss is not mentioned at all.**
**Why it matters:** the paper adds an optional third term, $\mathcal{L}_{\text{identity}} = \mathbb{E}_{\mathbf{y}}[\lVert G(\mathbf{y})-\mathbf{y}\rVert_1] + \mathbb{E}_{\mathbf{x}}[\lVert F(\mathbf{x})-\mathbf{x}\rVert_1]$ — feed a *zebra* to the horse→zebra generator and demand it come back unchanged. Without it, CycleGAN tends to shift the colour palette of the whole image (famously, photo→painting translations drift warm or cold even in regions that should not change). It is used for the Monet/photo task and omitted for horse→zebra. Not on these slides, so unlikely to be examined — but if an option mentions a third loss term, this is what it means.

**Gap: the deck never names the training dynamics or the replay buffer.**
**Why it matters:** CycleGAN trains all four networks in alternation, and uses an **image pool** — the discriminators are shown a buffer of 50 previously-generated images rather than only the current batch. This damps the oscillation that [Lec 34](34-gan-convergence.md) warns about, by stopping the discriminator from over-fitting to the generator's most recent quirk. It is the single most common reason a hand-rolled CycleGAN trains worse than the reference implementation.

**Gap: no evaluation metric is given.**
**Why it matters:** the lecture ends at "here are some nice pictures". The paper evaluates with AMT perceptual studies and with **FCN-score** — run a pre-trained semantic segmentation network on the generated image and check whether it recovers the input's label map. That is a direct, quantitative test of exactly the property cycle consistency is a proxy for: *did the content survive?* Knowing that a content-preservation metric exists makes the whole design rationale click.

## Cut from the slides

Pages 1, 2, 15, 16 and 17 are the title, the contents list, an empty "Summary" divider, the "Next Session: StyleGAN" card and a bare YouTube URL; nothing teachable is in them. Page 9 is an overview slide that re-states, in box form, the four networks from page 5 and announces that the loss splits into adversarial and cycle-consistency parts — it is pure signposting and is absorbed into the two sections that follow it rather than embedded as a figure. Pages 6 and 7 both carry the same circular $X \rightleftarrows Y$ diagram; both are reproduced because the surrounding text differs (generators on 6, discriminators on 7). The deck's plain italic $x$, $y$, $G(x)$, $F(y)$ have been set in bold as $\mathbf{x}$, $\mathbf{y}$ per CONTRACT §3 since they are images, and the deck's $L_{GAN}$, $L_{cyc}$, $L_{cycle}$ (it uses the last two interchangeably on pages 12 and 13) are written uniformly as $\mathcal{L}_{\text{GAN}}$ and $\mathcal{L}_{\text{cyc}}$. The horse/zebra photographs decorating pages 4–8 are illustrative repeats of the same two images and are not reproduced more than once each. Everything substantive on pages 3–14 is in this chapter.
