# Lec 41 — StyleGAN

> **Source:** `Lec 41.pdf` (18 pages) · **Week 6** · **Playlist:** Lec 41
> **Prereqs:** [Lec 26 — Disentanglement and β-VAE](26-beta-vae.md), [Lec 32 — GAN Architecture](32-gan-architecture.md), [Lec 33 — GAN Objective and Loss Functions](33-gan-objective.md)
> **Feeds into:** [Lec 42 — StyleGAN 2](42-stylegan2.md)

## Why this lecture exists

Every GAN so far has had one input: a noise vector $\mathbf{z}$, shoved into the first layer of the generator. It works — the pictures are sharp — but you have no handle on anything. Nudge $z_3$ and the face changes age, hair colour and head angle at once. You cannot ask for "the same person, smiling", because no coordinate means "smiling".

[Lec 26](26-beta-vae.md) named that failure — **entanglement** — and attacked it inside the VAE by reweighting the KL term. StyleGAN attacks the same failure from the opposite end, and its diagnosis is sharper: the latent space is entangled because **you forced it to be**. The distribution you sample $\mathbf{z}$ from is fixed in advance, so $\mathcal{Z}$'s geometry has no freedom to match the data's factor structure.

The fix is almost embarrassingly simple: stop feeding $\mathbf{z}$ to the generator. Pass it through a small network first, into a second latent space nobody constrained, and feed *that* to the image-maker — not once at the bottom, but at every layer, as a style.

## The ideas

### The problem, as the deck states it

![Slide headed Motivation for StyleGAN: a column of latent-vector cells z1…zn with arrows pointing at three different generated faces, and the red text "Researchers wanted finer control over generated image attributes" and "That was not possible in traditional GAN"](../assets/pages/lec41/p-04.png)
*Fig. — Three arrows leave three different parts of the latent vector and each produces a wholly different face. Nothing about which cell you touched predicts what changes. Page 4.*

In a traditional GAN ([Lec 32](32-gan-architecture.md)) you sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ — the deck writes it $z \sim N(0,1)$, with variance second per CONTRACT §3 — and compute $G(\mathbf{z}) \to$ image. One vector in, one picture out, and no structure in between.

![Slide headed Latent Space Entanglement: the latent-vector column with arrows to four faces that differ in hair colour, hair length and skin tone simultaneously. Bullets: "One latent variable affects multiple image attributes simultaneously", "changing one latent dimension affects multiple semantic attributes". Then "WHY?" and the quote "The input latent space must follow the probability density of the training data, and this leads to unavoidable entanglement."](../assets/pages/lec41/p-05.png)
*Fig. — The quoted sentence at the bottom right is the most important line on the slide and the deck spends only that one line on it. The next two sections unpack it, because it is the whole justification for StyleGAN's architecture. Page 5.*

**Entanglement**: one latent coordinate controls several semantic attributes at once, so no coordinate is a usable control. Formally stated by the deck: *changing one latent dimension affects multiple semantic attributes*.

### Why $\mathcal{Z}$ is *forced* to be entangled — the central idea

This is the argument the deck compresses into one sentence. Take it slowly, because it is the reason every later piece of the architecture exists.

You, the designer, choose the prior. You decided $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ — a round, symmetric, factorised Gaussian cloud. That choice is made *before* you see any data, and it never changes during training. The generator must therefore satisfy a hard constraint:

$$\text{push } \mathcal{N}(\mathbf{0},\mathbf{I}) \text{ through } G \;\;\Longrightarrow\;\; \text{get } p_{\text{data}}$$

Now look at what $p_{\text{data}}$ is actually shaped like. Real datasets have **correlated attributes and forbidden combinations**. In a face dataset, "male" and "long hair" are both common on their own but rare together. The data cloud is not round; it has holes.

![Slide headed Why latent Space Entanglement Happens: a 2×5 grid of face photographs labelled "Dataset", with the bullets "Multiple facial attributes become correlated" and "Latent space becomes curved because the generator avoids producing unlikely combinations"](../assets/pages/lec41/p-06.png)
*Fig. — The second bullet is the mechanism. The generator must avoid emitting combinations the dataset does not contain, and the only way to do that, starting from a round Gaussian, is to **curve** the mapping. Page 6.*

So $G$ has to take a round cloud and bend it onto a lumpy, holed distribution. A straight line in $\mathcal{Z}$ — "walk along $z_3$" — gets bent by that warping into a curve in image space that crosses several attribute boundaries at once. **That is entanglement, and it is not a training failure. It is a geometric necessity, imposed by fixing the prior.** No amount of extra training or extra capacity removes it, because the constraint is on the input distribution, not on the network.

Here is the move. **Do not feed $\mathbf{z}$ to the generator. Feed $\mathbf{w} = f(\mathbf{z})$.**

$$\mathbf{z} \;\xrightarrow{\;f\;}\; \mathbf{w} \;\xrightarrow{\;\text{synthesis}\;}\; \text{image}$$

$\mathcal{W}$ is **never sampled from**. You never draw a $\mathbf{w}$ from a prior; you only ever get one by pushing a $\mathbf{z}$ through $f$. So **nothing constrains $\mathcal{W}$'s density to be anything in particular.** It is free to be as lumpy, curved and hole-ridden as $p_{\text{data}}$ demands — and if $f$ absorbs *all* of the warping, then the synthesis network sees a space where the warping is already done and straight lines can stay straight.

The one-line summary to memorise:

> **$\mathcal{Z}$ must match the shape of the prior you chose, so it is forced to be entangled. $\mathcal{W}$ is unconstrained, so it is free to disentangle. The mapping network $f$ is where the unavoidable warping gets quarantined.**

N4 turns this into arithmetic.

### How this compares with β-VAE

[Lec 26](26-beta-vae.md) and this lecture attack the same enemy with opposite weapons.

| | **β-VAE** ([Lec 26](26-beta-vae.md)) | **StyleGAN** (here) |
|---|---|---|
| **Model family** | VAE (encoder + decoder, explicit likelihood) | GAN (generator + discriminator, implicit) |
| **Where disentanglement is sought** | in the **inference** latent $q_\phi(\mathbf{z}\mid\mathbf{x})$ | in a new **intermediate** latent $\mathcal{W}$ |
| **Mechanism** | **loss change** — multiply the KL term by $\beta > 1$ | **architecture change** — insert a mapping network; loss untouched |
| **What it pushes toward** | the factorised prior $\mathcal{N}(\mathbf{0},\mathbf{I})$ — *toward* a fixed shape | away from any fixed shape — $\mathcal{W}$ has **no** prior |
| **Cost** | reconstruction quality falls as $\beta$ rises | extra parameters (~2.1 M) and a deeper forward pass |
| **Extra networks** | none | one 8-layer MLP, plus one affine map per layer |
| **Guarantee** | none — "encourages, does not guarantee" | none either — $\mathcal{W}$ is *freer*, not provably disentangled |
| **Image quality** | blurry (reverse-KL, mode-seeking — [Lec 19](19-kl-divergence-a.md)) | state-of-the-art sharpness |

The deepest contrast is in the third and fourth rows. **β-VAE disentangles by pushing the latent *toward* a factorised Gaussian; StyleGAN disentangles by removing the requirement to be a Gaussian at all.** One adds a constraint, the other deletes one. Both are honest attempts at the same problem and neither comes with a proof.

### The key idea, drawn

![Slide headed Introduction to StyleGAN: left of a dashed divider, "Traditional GAN" as Latent vector (z) → Generator. Right, "Style GAN" as Latent vector (z) → Mapping Network (f) → Generator, annotated "Maps Z-space into intermediate latent space W" and "This allows different image attributes to vary independently". Below: z → Mapping Network → w → Styles → Synthesis Network → Image. Reference: Karras, Laine, Aila (Nvidia), "A Style-Based Generator Architecture for Generative Adversarial Networks"](../assets/pages/lec41/p-07.png)
*Fig. — One box inserted into the pipeline. The bottom line is the five-stage flow to memorise verbatim: **z → Mapping Network → w → Styles → Synthesis Network → Image**. Note "Styles" is a separate stage between $\mathbf{w}$ and the synthesis network — that is the affine transform. Page 7.*

### The mapping network $f: \mathcal{Z} \to \mathcal{W}$

![Slide headed Mapping Network and W Space in StyleGAN with three boxes. Latent vector (z): z ~ N(0,1), typically z ∈ R^512, "This vector contains compressed information for generating the image". Mapping Network (f): 8-layer MLP, f: Z → W, w = f(z), "Transform the entangled latent space Z into a more disentangled intermediate latent space W", example "one direction controls pose, another controls hairstyle, another controls eyeglasses". Generator: "In traditional GANs the generator begins image creation directly from z. In styleGAN it starts with a fixed learned tensor of size 4 × 4 × 512", initialized with constant or learned values and updated during backpropagation](../assets/pages/lec41/p-08.png)
*Fig. — Three facts to lift straight off this slide for the exam: **8-layer MLP**, **$\mathbf{z} \in \mathbb{R}^{512}$**, **fixed learned tensor $4\times4\times512$**. Page 8.*

![Slide headed Mapping Network in StyleGAN: a vertical flow from "Latent z ∈ Z (Dimension = 512)" through a "Normalize" box into a Multilayer Perceptron drawn as eight stacked FC blocks, out to "w ∈ W (Dimension = 512)", then sideways to "Affine transformation (A)" and on to "Synthesis Network". Red callouts: "W-space becomes disentangled", "Untangles attributes (pose, smile, hair color, etc.)"](../assets/pages/lec41/p-09.png)
*Fig. — Count the FC blocks: there are exactly **eight**. And read the two dimensions: 512 in, 512 out. The mapping network is a change of coordinates, **not** a compression. Page 9.*

$$f: \mathcal{Z} \to \mathcal{W}, \qquad \mathbf{w} = f(\mathbf{z}), \qquad \mathbf{z}\in\mathbb{R}^{512},\ \ \mathbf{w}\in\mathbb{R}^{512}$$

Three details the slides give you and one they do not:

1. **The "Normalize" box comes first.** Before the MLP, $\mathbf{z}$ is normalised — in the paper, divided by its own root-mean-square so it lands on the unit hypersphere. The deck shows the box but not the formula; it is $\mathbf{z} \mapsto \mathbf{z}/\sqrt{\frac{1}{n}\sum_j z_j^2 + \varepsilon}$.
2. **Eight fully-connected layers**, all $512 \to 512$. No convolution, no spatial structure — $\mathbf{z}$ is just a vector of 512 numbers.
3. **Input and output dimensions are equal.** This matters: $f$ is not an encoder and $\mathbf{w}$ is not a compressed code. It is a learned, invertible-in-practice re-coordinatisation whose only job is to absorb curvature.
4. *(Not on the slides.)* Depth is what buys the warping. A one-layer $f$ would be an affine map and could only rotate and stretch the Gaussian — it would stay Gaussian, and stay entangled. Eight non-linear layers can bend it arbitrarily. **"Why eight layers?" has the answer "enough non-linearity to absorb an arbitrary warp", not "because 8 is a nice number".**

The deck's example of what disentangled directions in $\mathcal{W}$ look like: *one direction controls pose, another controls hairstyle, another controls eyeglasses.*

### The synthesis network starts from a constant

Here is the second architectural shock. In a traditional GAN, $\mathbf{z}$ *is* the first layer's input. In StyleGAN, the synthesis path begins from a **fixed learned tensor of size $4\times4\times512$** — the same tensor for every image the model will ever produce. It is a trainable parameter (8,192 numbers), updated by backpropagation like any weight, and then frozen at inference.

So where does an image's identity come from, if the starting block is identical for everyone? **Entirely from the styles.**

![Slide headed "Analogy in StyleGAN": two photographs of the same clay block being worked, labelled "Same starting Clay Block", and two photographs of different finished horse sculptures labelled "Final style is different". Right-hand bullets: "Styles injected through AdaIN, Noise injection, Not from the initial input tensor."](../assets/pages/lec41/p-10.png)
*Fig. — The deck's own analogy, and it is a good one. The learned tensor is the lump of clay — identical every time. The styles are the sculptor's instructions. The last bullet is the examinable one: variation comes from **AdaIN and noise**, explicitly **not** from the input tensor. Page 10.*

![Slide headed Full Architecture Flow of StyleGAN: "Two parallel parts" branching into "Mapping Path (Z → f(z) = w)" which produces a disentangled latent representation, and "Synthesis Path" which starts from the fixed learned tensor 4 × 4 × 512 (trainable and shared for all generated images) and actually generates the image. Below, "How are the Mapping and Synthesis Networks Connected?": the learned tensor flows through convolution layers (18 conv layers); w → Affine Transform → Style parameters; at every convolution layer the affine transformation produces style parameters that are injected through AdaIN](../assets/pages/lec41/p-12.png)
*Fig. — "Two parallel parts" is the right mental model: the mapping path never touches pixels, the synthesis path never touches $\mathbf{z}$, and the only wire between them is the style injection at every layer. Memorise **18 conv layers** and **trainable, shared across all generated images**. Page 12.*

### The affine transformation: $\mathbf{w}$ becomes styles

The deck's page 11 gives it as one line. A general affine map is $y = \mathbf{W}x + b$; in StyleGAN it is

$$\mathbf{y} = \mathbf{A}\mathbf{w} + \mathbf{b}, \qquad \mathbf{y} = (y_s,\, y_b)$$

where $y_s$ is a **learned scaling parameter** and $y_b$ a **learned bias parameter**. The deck's phrasing: *the affine transformation converts latent vector $w$ into style parameters $y$*; *the learned tensor provides spatial feature maps (think of it as a lump of clay)*; **styles = sculpting instructions**.

Two things to hold onto:

- **There is one $\mathbf{A}$ per injection point, not one globally.** The slide's "$\mathbf{A}$" appears beside every AdaIN box in the architecture diagram. The same $\mathbf{w}$ enters 18 different affine maps and comes out as 18 different style pairs. That is how one 512-vector controls eighteen layers differently.
- **$y_s$ and $y_b$ are per-channel vectors, not scalars.** If a layer has $C$ feature-map channels, $\mathbf{A}$ outputs $2C$ numbers: a scale and a bias for each channel. N6 counts them.

### Adaptive instance normalization (AdaIN)

$$\boxed{\;\mathrm{AdaIN}(\mathbf{x}_i, \mathbf{y}) \;=\; y_{s,i}\,\frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} \;+\; y_{b,i}\;}$$

Here $\mathbf{x}_i$ is the $i$-th **channel** of the feature map, $\mu(\mathbf{x}_i)$ and $\sigma(\mathbf{x}_i)$ are that channel's mean and standard deviation computed over its own spatial positions, and $y_{s,i},y_{b,i}$ are that channel's style scale and bias.

Read it as two moves, exactly as the deck does:

1. **Normalize** — $\dfrac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)}$ gives the channel zero mean and unit variance. The deck's gloss: *"removes the original statistics of the feature map"*, so that *"after normalization old style information is removed; only structural information remains."* Dividing by $\sigma$ destroys contrast and magnitude; subtracting $\mu$ destroys offset. **What survives is the spatial pattern — where the bright bits are relative to each other.**
2. **Re-style** — multiply by $y_{s,i}$ and add $y_{b,i}$: *"apply a new style using learned scale and bias."*

The crucial, slide-free consequence:

$$\text{mean}\big(\mathrm{AdaIN}(\mathbf{x}_i,\mathbf{y})\big) = y_{b,i}, \qquad \text{sd}\big(\mathrm{AdaIN}(\mathbf{x}_i,\mathbf{y})\big) = |y_{s,i}|$$

because the normalised channel has mean 0 and sd 1, so scaling by $y_{s,i}$ and shifting by $y_{b,i}$ sets the output's statistics to exactly those two numbers. **AdaIN completely overwrites each channel's first and second moments with values the style chooses, no matter what they were before.** Whatever the previous layer "wanted" the brightness and contrast of channel $i$ to be, the style decides. That total override is why the deck can say identity comes from the styles and not from the input tensor. N3 verifies it numerically.

Three discriminations worth bolting down:

| | Normalised over | Statistics depend on |
|---|---|---|
| **Batch norm** | batch × spatial, per channel | other images in the batch |
| **Instance norm** | spatial only, per channel, **per image** | this image only |
| **AdaIN** | same as instance norm, then **scale/shift by style** | this image, then overwritten by $\mathbf{w}$ |

AdaIN is instance normalization with the usual *learned-constant* $\gamma,\beta$ replaced by *input-dependent* $y_s,y_b$. The word **adaptive** means exactly that: the scale and bias adapt to the latent, instead of being fixed parameters. And the deck is explicit on the red line of page 14: **"AdaIN first normalizes every channel independently."**

![Slide headed Adaptive Instance Normalization (AdaIN): a 2-by-2 feature-map channel with entries 2, 4 on the top row and 6, 8 on the bottom; Mean μ(x_i) = (2+4+6+8)/4 = 5; Standard deviation σ(x_i) = sqrt(((2−5)² + (4−5)² + (6−5)² + (8−5)²)/4) = sqrt((9+1+1+9)/4) = sqrt(5) ≈ 2.236; a boxed Normalize operation (x_i − μ)/σ producing the matrix with entries −1.34, −0.45 on the top row and 0.45, 1.34 on the bottom, annotated "zero mean, unit variance". Bullets: "After normalization old style information is removed", "Only structural information remains", and in red "AdaIN first normalizes every channel independently."](../assets/pages/lec41/p-14.png)
*Fig. — The deck's worked example, stage 1. Note the divisor in $\sigma$ is **4**, the number of elements — this is the population standard deviation, not the $n-1$ sample version. Using $n-1$ gives $\sqrt{20/3} = 2.582$ and every later number changes. Page 14.*

![Slide headed Style Injection in AdaIN: the flow z → Mapping Network → w → Affine Transformation, box "y = Aw + b", producing y = (y_s, y_b) with "y_s: Learned scaling parameter" and "y_b: Learned Bias Parameter", "These are used by AdaIN". Boxed formula AdaIN of x_i and y equals y_s,i times (x_i − μ(x_i))/σ(x_i) plus y_b,i, leading to "Styled feature maps ↓ Generated image". Worked below: if y_s = 2 and y_b = 1, then 2 times the normalised matrix (entries −1.34, −0.45 on the top row, 0.45, 1.34 on the bottom) plus 1 gives entries −1.68, 0.1 on the top row and 1.9, 3.68 on the bottom](../assets/pages/lec41/p-15.png)
*Fig. — Stage 2, and the only place the deck writes the full AdaIN formula. Check the output's own statistics: mean $= 1 = y_b$, sd $= 2 = y_s$. The style has completely replaced the channel's original mean of 5 and sd of 2.236. Page 15.*

### The synthesis network, layer by layer

![Slide headed Synthesis Network: "Fixed Learned Tensor (4 x 4 x 512)" → "Convolution layers" → "Generates" → "Feature Maps" (containing edges, textures, face structure, hair pattern etc) → "AdaIN". AdaIN bullets: "Normalize the feature map (removes the original statistics of the feature map)", "Apply a new style using learned scale and bias", and in red "After 2 Conv layers in one resolution, there is upsampling". Below, a chain of nine blocks — Conv 1/Conv 2 at 4×4, Conv 3/Conv 4 at 8×8, Conv 5/Conv 6 at 16×16, Conv 7/Conv 8 at 32×32, Conv 9/Conv 10 at 64×64, Conv 11/Conv 12 at 128×128, Conv 13/Conv 14 at 256×256, Conv 15/Conv 16 at 512×512, Conv 17/Conv 18 at 1024×1024 — with arrows marked "Upsampling". Lower resolution learns pose, face shape, coarse structure; middle layers learn eyes, nose, hairstyle; higher-resolution layers learn pores, wrinkles, hair strands, texture.](../assets/pages/lec41/p-13.png)
*Fig. — The single most numerically examinable page in the deck. Nine resolution blocks, two convs each, eight upsamplings, $4 \to 1024$. And the three-band attribute map — coarse/middle/fine — is a guaranteed MCQ. Page 13.*

| Block | Resolution | Convs | What it controls (deck's words) |
|---|---|---|---|
| 1 | $4\times4$ | 1, 2 | **pose, face shape, coarse structure** |
| 2 | $8\times8$ | 3, 4 | " |
| 3 | $16\times16$ | 5, 6 | " |
| 4 | $32\times32$ | 7, 8 | **eyes, nose, hairstyle** |
| 5 | $64\times64$ | 9, 10 | " |
| 6 | $128\times128$ | 11, 12 | " |
| 7 | $256\times256$ | 13, 14 | **pores, wrinkles, hair strands, texture** |
| 8 | $512\times512$ | 15, 16 | " |
| 9 | $1024\times1024$ | 17, 18 | " |

The rule governing the whole ladder: **after 2 conv layers in one resolution, there is upsampling.** Nine resolutions, eight upsamplings, each doubling: $4 \times 2^8 = 1024$. ✔

Why the attribute bands fall out this way is worth one sentence, because it is not arbitrary. A $4\times4$ feature map has sixteen spatial positions for the *entire face* — nothing finer than "head tilted left" can be represented at that scale. A $1024\times1024$ map has a position per output pixel, so it can only sensibly carry things that vary pixel-to-pixel: pores, stray hairs, skin texture. **Resolution determines semantic scale.** Injecting a style at layer $k$ therefore changes the image at the scale layer $k$ can represent, and nothing else.

### Progressive growing

The deck draws the ladder but never names the training procedure that produced it. **Progressive growing** is inherited from StyleGAN's predecessor (ProGAN, Karras et al. 2017) and works like this:

1. Train generator and discriminator at $4\times4$ only, until stable.
2. **Fade in** the next block: add the $8\times8$ layers to both networks, blending their contribution from 0 to 1 over a few thousand iterations so no sudden shock hits the already-trained weights.
3. Repeat up to $1024\times1024$.

Two reasons it matters. **Stability:** at $4\times4$ there are 16 positions and the real/fake decision is nearly trivial, so the discriminator never overwhelms the generator in the way [Lec 34](34-gan-convergence.md) warns about; the hard high-resolution problem is only attempted once the easy structure is already solved. **Speed:** the early epochs run on tiny images, so most of training is cheap.

On upsampling: StyleGAN uses **bilinear upsampling followed by a $3\times3$ convolution**, not transposed convolution. [Lec 12](12-autoencoder-types.md) owns transposed convolution, derives $O = I + K - 1$ and explains the **checkerboard artefact** it produces — go there for the mechanism. StyleGAN's upsample-then-convolve is precisely Lec 12's named fix for that artefact, applied at scale.

### Noise injection and stochastic variation

![Slide: the paper's Figure 1, "(a) Traditional" generator — Latent z ∈ Z → Normalize → Fully-connected → PixelNorm → Conv 3×3 → PixelNorm (4×4), then Upsample → Conv 3×3 → PixelNorm → Conv 3×3 → PixelNorm (8×8) — against "(b) Style-based generator" — Latent z ∈ Z → Normalize → Mapping network f of eight FC layers → w ∈ W, with A boxes feeding "style" into AdaIN blocks; the Synthesis network g starts from Const 4×4×512, with ⊕ nodes taking B-scaled Noise before each AdaIN, through Conv 3×3 and Upsample. Annotations: "Noise adds stochastic variation to the synthesis network"; "B: Learned scaling factor; Decides how strongly the noise should affect each feature channel."](../assets/pages/lec41/p-16.png)
*Fig. — Put (a) beside (b) and count the differences: $\mathbf{z}$ enters (a) at the bottom of the image stack and enters (b) nowhere near it; (b) gains a mapping network, a constant input, an A at every AdaIN, and a noise input at every $\oplus$. PixelNorm in (a) is replaced by AdaIN in (b). Page 16.*

The $\oplus$ nodes are **noise injection**, and they do a different job from the styles:

- At each injection point a **single-channel** Gaussian noise image of the current resolution is drawn fresh, $\mathbf{n} \sim \mathcal{N}(0, 1)$ per pixel.
- It is broadcast to every channel and scaled by **$B$, a learned per-channel scaling factor** — the deck's words: *"decides how strongly the noise should affect each feature channel."*
- The result is added to the feature map *before* AdaIN: $\mathbf{x}_i \leftarrow \mathbf{x}_i + B_i\,\mathbf{n}$.

**Why have it at all?** Real faces contain genuinely random detail — the exact placement of each hair, the pattern of freckles, pore distribution. None of it is semantic; none of it should cost a dimension of $\mathbf{w}$. Without a noise input the generator would have to manufacture that randomness out of $\mathbf{w}$, burning latent capacity on things nobody wants to control and re-entangling the space. Giving it a dedicated per-pixel noise source **frees $\mathbf{w}$ to carry only what a human would name**.

The clean discrimination, and an obvious exam question:

| | **Style** ($\mathbf{w} \to$ AdaIN) | **Noise** ($\mathbf{n} \to \oplus$) |
|---|---|---|
| Source | $\mathbf{w} = f(\mathbf{z})$, via affine $\mathbf{A}$ | fresh $\mathcal{N}(0,1)$ sample per layer |
| Granularity | **global** — one scale + one bias per *channel* | **local** — one value per *pixel* |
| Controls | semantic attributes: pose, identity, hairstyle | stochastic detail: hair placement, freckles, pores |
| Changing it | changes who the person is | same person, different strands of hair |
| Learned part | $\mathbf{A}$ (and $f$) | $B$, the per-channel scaling factor |
| Entering at | every AdaIN (18 points) | every $\oplus$ (one before each AdaIN) |

### Style mixing — not on these slides

Because each layer gets its own style from its own affine map, nothing forces all eighteen to come from the same $\mathbf{w}$. **Style mixing** feeds $\mathbf{w}_1 = f(\mathbf{z}_1)$ to the coarse layers and $\mathbf{w}_2 = f(\mathbf{z}_2)$ to the fine ones, producing person 1's pose and face shape wearing person 2's hair texture and skin detail. It is used in the paper as a *regulariser* during training — randomly switching $\mathbf{w}$ mid-stack discourages the network from assuming adjacent layers' styles are correlated, which strengthens the per-layer separation described above. **The deck does not mention it**; it is included here because it is the most direct demonstration that the per-layer style design does what it claims.

> **Droplet artefacts.** StyleGAN's outputs contain characteristic blob-like artefacts, and AdaIN's normalisation step is the culprit. [Lec 42](42-stylegan2.md) owns the diagnosis and the weight-demodulation fix — do not expect them here.

## Worked numericals

> **The slides contain one worked example, split across pages 14 and 15** — the AdaIN computation, in two stages. Both stages are reproduced below (N1, N2) with independent verification. N3–N6 are constructed.

### N1. The deck's AdaIN example, stage 1 — normalization (page 14)

**Given:** a single channel of a feature map, $\mathbf{x}_i = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}$.
**Find:** $\mu(\mathbf{x}_i)$, $\sigma(\mathbf{x}_i)$, and the normalised channel.

1. Mean: $\mu = \dfrac{2+4+6+8}{4} = \dfrac{20}{4} = 5$.
2. Deviations: $2-5 = -3$, $4-5 = -1$, $6-5 = 1$, $8-5 = 3$.
3. Squared deviations: $9,\ 1,\ 1,\ 9$. Sum $= 20$.
4. Variance (divisor $N = 4$, the **population** form the slide uses): $\sigma^2 = \dfrac{20}{4} = 5$.
5. $\sigma = \sqrt{5} = 2.2360680$.
6. Normalise each entry, $(x - 5)/2.236068$:
   - $-3/2.236068 = -1.3416408$
   - $-1/2.236068 = -0.4472136$
   - $\;\;1/2.236068 = \;\;0.4472136$
   - $\;\;3/2.236068 = \;\;1.3416408$

**Answer:** $\mu = 5$, $\sigma = \sqrt{5} \approx 2.236$, normalised channel $\begin{bmatrix} -1.3416 & -0.4472 \\ 0.4472 & 1.3416 \end{bmatrix}$. **This matches the slide exactly** (it prints $-1.34$, $-0.45$, $0.45$, $1.34$ — the $-0.45$ is $-0.4472$ rounded up, correctly). Mean of the result: $(-1.3416-0.4472+0.4472+1.3416)/4 = 0$ ✔. Variance: $(1.8+0.2+0.2+1.8)/4 = 1$ ✔ — "zero mean, unit variance", as the slide claims.

> **The divisor trap.** The slide divides by $N = 4$. If you use the sample standard deviation with $N-1 = 3$ you get $\sigma = \sqrt{20/3} = 2.5820$, and the normalised entries become $-1.1619, -0.3873, 0.3873, 1.1619$ — all four answers wrong. Normalization layers always use the population form. NumPy's `np.std` defaults to $N$ (correct here); `pandas.Series.std` defaults to $N-1$ (wrong here).

### N2. The deck's AdaIN example, stage 2 — style injection (page 15)

**Given:** the normalised channel from N1, with style parameters $y_s = 2$ and $y_b = 1$.
**Find:** $\mathrm{AdaIN}(\mathbf{x}_i, \mathbf{y})$, and the output's own mean and standard deviation.

1. Apply $\mathrm{AdaIN} = y_s \cdot (\text{normalised}) + y_b$ entrywise, using the slide's rounded inputs:
   - $2(-1.34) + 1 = -2.68 + 1 = -1.68$
   - $2(-0.45) + 1 = -0.90 + 1 = \;\;0.10$
   - $2(\;\;0.45) + 1 = \;\;0.90 + 1 = \;\;1.90$
   - $2(\;\;1.34) + 1 = \;\;2.68 + 1 = \;\;3.68$
2. Output mean: $(-1.68 + 0.10 + 1.90 + 3.68)/4 = 4.00/4 = 1.00 = y_b$ ✔
3. Output sd: deviations $-2.68, -0.90, 0.90, 2.68$; squares $7.1824, 0.81, 0.81, 7.1824$; sum $15.9848$; $/4 = 3.9962$; $\sqrt{3.9962} = 1.99905 \approx 2 = y_s$ ✔

**Answer:** $\begin{bmatrix} -1.68 & 0.10 \\ 1.90 & 3.68 \end{bmatrix}$ — **matches the slide exactly.** With unrounded inputs the entries are $-1.683282,\ 0.105573,\ 1.894427,\ 3.683282$, so the slide's $0.1$ and $1.9$ are $0.1056$ and $1.8944$ rounded to 2 significant figures. Nothing is wrong, but a 4-dp exam question would need the unrounded values.

The channel went in with $(\mu, \sigma) = (5,\ 2.236)$ and came out with $(\mu, \sigma) = (1,\ 2)$ — exactly $(y_b, y_s)$. **The style completely replaced the channel's statistics.**

### N3. AdaIN's two guarantees, on a channel with different statistics

**Given:** a second channel $\mathbf{x}_2 = \begin{bmatrix} 10 & 10 \\ 11 & 9 \end{bmatrix}$ of the same feature map, with its own style $y_s = 0.5$, $y_b = -3$.
**Find:** the AdaIN output, and confirm its mean and sd.

1. $\mu = (10+10+11+9)/4 = 40/4 = 10$.
2. Deviations: $0,\ 0,\ 1,\ -1$. Squares: $0,0,1,1$, sum $2$.
3. $\sigma^2 = 2/4 = 0.5$, $\sigma = \sqrt{0.5} = 0.7071068$.
4. Normalised: $0,\ 0,\ 1/0.7071068 = 1.4142136,\ -1.4142136$.
5. AdaIN $= 0.5(\text{normalised}) + (-3)$:
   - $0.5(0) - 3 = -3.0$
   - $0.5(0) - 3 = -3.0$
   - $0.5(1.4142136) - 3 = 0.7071068 - 3 = -2.2928932$
   - $0.5(-1.4142136) - 3 = -0.7071068 - 3 = -3.7071068$
6. Mean: $(-3 - 3 - 2.2928932 - 3.7071068)/4 = -12/4 = -3 = y_b$ ✔
7. Sd: deviations $0,0,0.7071068,-0.7071068$; squares sum $= 1.0$; $/4 = 0.25$; $\sqrt{0.25} = 0.5 = y_s$ ✔

**Answer:** $\begin{bmatrix} -3.0000 & -3.0000 \\ -2.2929 & -3.7071 \end{bmatrix}$, with mean exactly $-3$ and sd exactly $0.5$.

Two readings. Channel 2 started at $(\mu,\sigma) = (10, 0.707)$ — utterly different from channel 1's $(5, 2.236)$ — and both channels now carry **exactly the statistics their style dictated**. And the two channels got **different** styles from the same $\mathbf{w}$, because $\mathbf{A}$ outputs a separate $(y_s, y_b)$ pair per channel. That is the per-channel independence the deck's red line asserts.

### N4. Why the prior forces entanglement — a probability calculation

**Given:** a face dataset of 10,000 images in which $P(\text{male}) = 0.5$ and $P(\text{long hair}) = 0.4$, but only 200 images show a long-haired male, so $P(\text{male and long hair}) = 0.02$.
**Find:** how far the data departs from independence, and what that costs the latent space.

1. Under independence, $P(\text{male}) \times P(\text{long hair}) = 0.5 \times 0.4 = 0.20$, i.e. **2,000** of the 10,000 images.
2. Observed: $0.02$, i.e. **200** images. Ratio $0.02/0.20 = 0.1$ — the combination is **10× under-represented**.
3. Conditional check: $P(\text{long hair} \mid \text{male}) = 0.02/0.5 = 0.04$, against the marginal $P(\text{long hair}) = 0.40$. Again a factor of **10**.
4. Now the latent side. $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ has **independent** coordinates by construction, so if $z_1$ meant "maleness" and $z_2$ meant "hair length" and both were used straight, the generator would emit the (male, long hair) corner with probability $0.20$ — **ten times too often**.
5. To emit it at $0.02$ instead, $G$ must **shrink** the region of $\mathcal{Z}$ that lands in that corner by a factor of 10, which means bending the mapping there.
6. Consequence: the straight line "increase $z_2$, hold $z_1$ fixed" no longer traces "lengthen hair, hold sex fixed" — it gets deflected around the shrunken region, dragging maleness along with it.

**Answer:** the data is **10× away from independence** on this one pair of attributes, and that factor of 10 is exactly the warping $G$ must apply — which is entanglement. The deck's sentence, *"the input latent space must follow the probability density of the training data, and this leads to unavoidable entanglement"*, is this calculation in words. A real dataset has thousands of such correlated pairs. The mapping network's job is to perform all of that warping **before** the synthesis network, so the synthesis network's input space is already bent to fit and its own directions can stay straight.

### N5. Counting the synthesis network

**Given:** the ladder on page 13 — start at $4\times4$, end at $1024\times1024$, two conv layers per resolution, upsample between resolutions.
**Find:** the number of resolutions, conv layers, upsampling steps, AdaIN operations and distinct style vectors.

1. Resolutions double from 4 to 1024. Number of doublings: $1024/4 = 256 = 2^8$, so **8 doublings**.
2. Number of resolution blocks $= 8 + 1 = \mathbf{9}$ (namely 4, 8, 16, 32, 64, 128, 256, 512, 1024).
3. Conv layers $= 2 \times 9 = \mathbf{18}$ ✔ (the deck states 18 on page 12 and labels Conv 1–18 on page 13 — the two slides agree).
4. Upsampling steps $= 9 - 1 = \mathbf{8}$ (between blocks, not after the last).
5. AdaIN operations: one after each conv $= \mathbf{18}$.
6. Distinct style vectors: one affine $\mathbf{A}$ per AdaIN $= \mathbf{18}$, all computed from the **same** $\mathbf{w}$.
7. Sanity check on the resolution: $4 \times 2^8 = 4 \times 256 = 1024$ ✔.

**Answer:** **9** resolutions, **18** conv layers, **8** upsamplings, **18** AdaIN operations, **18** style pairs from **1** latent $\mathbf{w}$. Final image $1024\times1024$, which is $1{,}048{,}576$ pixels, or $3{,}145{,}728$ numbers in RGB — grown from a starting tensor of $4\times4\times512 = 8{,}192$ numbers, a factor of **384**.

### N6. Parameter counts

**Given:** $\mathbf{z},\mathbf{w}\in\mathbb{R}^{512}$; mapping network = 8 fully-connected $512\to512$ layers with bias; constant input tensor $4\times4\times512$; the first AdaIN sits on a layer with $C = 512$ channels.
**Find:** parameters in the mapping network, in the constant tensor, in one affine $\mathbf{A}$, and in one $3\times3$ conv at that layer.

1. One FC layer: $512 \times 512$ weights $+\ 512$ biases $= 262{,}144 + 512 = 262{,}656$.
2. Mapping network: $8 \times 262{,}656 = \mathbf{2{,}101{,}248} \approx 2.10$ M.
3. Constant tensor: $4 \times 4 \times 512 = \mathbf{8{,}192}$ trainable numbers.
4. Affine $\mathbf{A}$ must output one $y_s$ and one $y_b$ per channel, so $2C = 1024$ numbers from a 512-vector: $512 \times 1024 + 1024 = 524{,}288 + 1{,}024 = \mathbf{525{,}312}$.
5. One $3\times3$ conv, 512 in, 512 out ([Lec 05](05-cnn-a.md)'s counting rule): $3\times3\times512\times512 + 512 = 2{,}359{,}296 + 512 = \mathbf{2{,}359{,}808}$.
6. Compare: $2{,}359{,}808 / 525{,}312 = 4.49$.

**Answer:** mapping network **2.10 M**, constant tensor **8,192**, one affine **525,312**, one conv **2,359,808**. Three readings for the exam. The constant input is **tiny** — 8,192 numbers against millions elsewhere, which is why it can afford to be the same for every image. One affine map costs about **a fifth** of one conv layer, so eighteen of them are a real but affordable overhead. And the mapping network, at 2.1 M, is a small fraction of a full StyleGAN (≈26 M in the generator) — the disentanglement comes almost free in parameter terms.

## Code

The deck asserts that AdaIN "removes the original statistics" and "applies a new style". Twenty lines prove both halves exactly, on two channels with deliberately different statistics — and reproduce the slide's own numbers as channel 0.

```python
import numpy as np

# A 2-channel feature map, 2x2 each. Channel 0 is the deck's page-14 example.
x = np.array([[[2., 4.], [6., 8.]],          # channel 0: mean 5,   sd sqrt(5)
              [[10., 10.], [11., 9.]]])      # channel 1: mean 10,  sd sqrt(0.5)

def instance_norm(x):                 # per channel, over the H x W positions only
    mu = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True)          # population sd, divisor N
    return (x - mu) / sd, mu.ravel(), sd.ravel()

xn, mu, sd = instance_norm(x)
print("per-channel mu  =", np.round(mu, 4))
print("per-channel sd  =", np.round(sd, 4))
print("normalised ch0  =\n", np.round(xn[0], 4))
print("normalised ch1  =\n", np.round(xn[1], 4))

# Styles: one (y_s, y_b) pair PER CHANNEL, produced by the affine map y = A w + b.
y_s = np.array([2.0, 0.5]).reshape(2, 1, 1)
y_b = np.array([1.0, -3.0]).reshape(2, 1, 1)
out = y_s * xn + y_b
print("AdaIN ch0 (y_s=2,  y_b= 1) =\n", np.round(out[0], 4))
print("AdaIN ch1 (y_s=0.5,y_b=-3) =\n", np.round(out[1], 4))
print("output mean per channel =", np.round(out.mean(axis=(1, 2)), 10))
print("output sd   per channel =", np.round(out.std(axis=(1, 2)), 10))
```

```
per-channel mu  = [ 5. 10.]
per-channel sd  = [2.2361 0.7071]
normalised ch0  =
 [[-1.3416 -0.4472]
 [ 0.4472  1.3416]]
normalised ch1  =
 [[ 0.      0.    ]
 [ 1.4142 -1.4142]]
AdaIN ch0 (y_s=2,  y_b= 1) =
 [[-1.6833  0.1056]
 [ 1.8944  3.6833]]
AdaIN ch1 (y_s=0.5,y_b=-3) =
 [[-3.     -3.    ]
 [-2.2929 -3.7071]]
output mean per channel = [ 1. -3.]
output sd   per channel = [2.  0.5]
```

Three readings. Channel 0 reproduces the slide: $\mu = 5$, $\sigma = 2.2361$, normalised $[-1.3416, -0.4472, 0.4472, 1.3416]$, styled $[-1.6833, 0.1056, 1.8944, 3.6833]$ — the deck's $-1.68, 0.1, 1.9, 3.68$ to its printed precision. The `axis=(1,2)` in `instance_norm` is where "**independently per channel**" lives; change it to `axis=(0,1,2)` and you have a single global normalization, which would couple the channels and break the whole design. And the last two printed lines are the guarantee, to ten decimal places: **output mean is exactly $y_b$ and output sd is exactly $|y_s|$, for both channels, regardless of what went in.**

## Exam pack

### Must-memorise

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

### Numbers worth knowing

| Quantity | Value |
|---|---|
| Mapping-network depth | **8** FC layers |
| $\dim\mathcal{Z}$, $\dim\mathcal{W}$ | **512**, **512** |
| Constant input tensor | $4\times4\times512 = 8{,}192$ values |
| Starting resolution | $4\times4$ |
| Final resolution | $1024\times1024$ |
| Resolution blocks | **9** |
| Conv layers | **18** |
| Upsampling steps | **8** ($4\times2^8 = 1024$) |
| AdaIN operations | **18** |
| Affine maps $\mathbf{A}$ | **18**, all fed by one $\mathbf{w}$ |
| Style numbers per layer | $2C$ (one $y_s$ + one $y_b$ per channel); $1{,}024$ at $C = 512$ |
| Mapping-network parameters | $8(512^2{+}512) = 2{,}101{,}248 \approx 2.10$ M |
| One affine $\mathbf{A}$ at $C{=}512$ | $525{,}312$ |
| One $3\times3$ conv, $512\to512$ | $2{,}359{,}808$ |
| Deck's AdaIN example | $\mathbf{x}=[2,4,6,8]$ → $\mu = 5$, $\sigma = \sqrt5 = 2.236$ |
| Normalised | $[-1.3416,\ -0.4472,\ 0.4472,\ 1.3416]$ |
| Styled, $y_s{=}2$, $y_b{=}1$ | $[-1.68,\ 0.1056,\ 1.8944,\ 3.68]$, mean $1$, sd $2$ |

### Likely MCQ traps

- **"The mapping network compresses $\mathbf{z}$."** It does not. $512 \to 512$, same dimension in and out. It is a **re-coordinatisation**, not an encoder. Any option implying a bottleneck or a latent-size reduction is wrong.
- **"$\mathbf{w}$ is sampled from a prior."** Never. You sample $\mathbf{z}$; $\mathbf{w}$ is only ever *computed* as $f(\mathbf{z})$. That is precisely why $\mathcal{W}$ is unconstrained and can disentangle — the single most important idea in the lecture.
- **"Entanglement in $\mathcal{Z}$ is a training failure, curable with more data/epochs."** No. It is **geometric and unavoidable**, because the prior's shape is fixed before training and $G$ must bend it onto $p_{\text{data}}$ (N4). The deck's word is *unavoidable*.
- **"StyleGAN's generator takes $\mathbf{z}$ as input."** The **synthesis network takes a constant**, $4\times4\times512$, identical for every image. $\mathbf{z}$ enters only the mapping network. An option showing $\mathbf{z}$ feeding the first conv is describing a traditional GAN.
- **"The input tensor is what makes each image different."** The opposite — it is the *same* for every image. Variation comes from **AdaIN styles and noise injection** (the deck's page-10 bullet says exactly this).
- **AdaIN is not batch norm.** AdaIN normalises **per channel of this one image** over spatial positions; batch norm uses statistics pooled across the batch. And AdaIN's scale/bias are **computed from $\mathbf{w}$ at inference**, where batch norm's $\gamma,\beta$ are fixed learned constants. "Adaptive" names that difference.
- **Getting the sd divisor wrong.** $\sigma$ uses $N$, not $N-1$: $\sqrt{20/4} = 2.236$, not $\sqrt{20/3} = 2.582$. Every downstream digit changes.
- **"One $\mathbf{A}$ for the whole network."** There are **18**, one per injection point. One $\mathbf{w}$, eighteen different style pairs — that is how a single latent controls coarse and fine detail separately.
- **"$y_s$ and $y_b$ are scalars."** They are **vectors of length $C$**, one entry per feature channel. The deck's example uses one channel, so they look scalar.
- **Confusing styles with noise.** **Style = global, per channel, from $\mathbf{w}$, semantic.** **Noise = local, per pixel, freshly sampled, stochastic detail.** $B$ is the learned scaling factor on the *noise*, not on the style.
- **Coarse/fine inversion.** Low resolution ($4$–$16$) → **pose and face shape**; high resolution ($256$–$1024$) → **pores and hair strands**. A question offering "the $4\times4$ layer controls skin texture" has them backwards.
- **Counting 9 conv layers or 16.** Nine *resolutions*, **18** conv layers (two per resolution), **8** upsamplings. Three different numbers, all plausible-looking options.
- **Attributing droplet artefacts or weight demodulation to StyleGAN 1.** Those belong to [Lec 42](42-stylegan2.md).
- **β-VAE vs StyleGAN.** β-VAE changes the **loss** (a $\beta$ on the KL term, [Lec 26](26-beta-vae.md)); StyleGAN changes the **architecture** and leaves the adversarial loss of [Lec 33](33-gan-objective.md) untouched. Neither *guarantees* disentanglement.

### Self-test

1. State, in two sentences, why $\mathcal{Z}$ is forced to be entangled and why $\mathcal{W}$ is not.
2. Write the five-stage StyleGAN pipeline from $\mathbf{z}$ to the image.
3. Give the AdaIN formula and name every symbol in it.
4. A channel is $\begin{bmatrix}1 & 3 \\ 5 & 7\end{bmatrix}$ with $y_s = 3$, $y_b = -1$. Compute $\mu$, $\sigma$, the normalised channel and the AdaIN output.
5. Without computing anything, state the mean and standard deviation of the output in question 4. Justify in one line.
6. How many conv layers, resolution blocks, upsamplings and AdaIN operations are in the synthesis network, and what final resolution do they reach?
7. What does the synthesis network take as its input, what is its size, and is it trainable?
8. Distinguish style injection from noise injection on four axes.
9. Why does injecting a style at the $8\times8$ block change pose rather than skin texture?
10. The mapping network is 8 fully-connected $512\to512$ layers with biases. How many parameters is that?
11. Name one thing β-VAE and StyleGAN have in common and two ways they differ.

<details><summary>Answers</summary>

1. $\mathbf{z}$ is sampled from a prior you fixed in advance ($\mathcal{N}(\mathbf{0},\mathbf{I})$), so the generator must bend that round, factorised cloud onto the lumpy, correlated density of the training data — and that bending is entanglement. $\mathcal{W}$ is never sampled from, only computed as $f(\mathbf{z})$, so no prior constrains its density and it is free to take whatever (curved) shape makes the synthesis network's directions straight.
2. $\mathbf{z} \to$ Mapping Network $\to \mathbf{w} \to$ Styles $\to$ Synthesis Network $\to$ Image.
3. $\mathrm{AdaIN}(\mathbf{x}_i,\mathbf{y}) = y_{s,i}\frac{\mathbf{x}_i-\mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + y_{b,i}$. $\mathbf{x}_i$ = the $i$-th channel of the feature map; $\mu(\mathbf{x}_i),\sigma(\mathbf{x}_i)$ = that channel's own mean and sd over its spatial positions; $y_{s,i}$ = learned scaling parameter for channel $i$, $y_{b,i}$ = learned bias parameter, both from $\mathbf{y} = \mathbf{A}\mathbf{w}+\mathbf{b}$.
4. $\mu = (1+3+5+7)/4 = 4$. Deviations $-3,-1,1,3$; squares $9,1,1,9$, sum $20$; $\sigma^2 = 5$, $\sigma = 2.2360680$. Normalised: $-1.3416, -0.4472, 0.4472, 1.3416$. AdaIN $= 3(\cdot) - 1$: $-5.0249,\ -2.3416,\ 0.3416,\ 3.0249$.
5. Mean $= y_b = -1$, sd $= |y_s| = 3$. Because the normalised channel has mean 0 and sd 1, a linear map $v \mapsto y_s v + y_b$ sends those to $y_b$ and $|y_s|$ exactly.
6. **18** conv layers, **9** resolution blocks, **8** upsamplings, **18** AdaIN operations, final resolution $1024\times1024$.
7. A **fixed learned tensor** of size $4\times4\times512$ ($8{,}192$ values). **Yes, trainable** — updated by backpropagation — but shared by all generated images, so it carries no per-image information.
8. Style: global per channel / derived from $\mathbf{w}$ via the affine $\mathbf{A}$ / controls semantic attributes (pose, identity, hairstyle) / learned part is $\mathbf{A}$. Noise: per pixel / freshly sampled $\mathcal{N}(0,1)$ each forward pass / controls stochastic detail (hair placement, freckles, pores) / learned part is $B$, the per-channel scaling factor.
9. Because an $8\times8$ feature map has only 64 spatial positions for the whole face. Nothing finer than coarse geometry — head angle, face outline — can be represented at that scale, so a style change there can only alter coarse structure. Texture needs a position per pixel and therefore lives in the $256$–$1024$ blocks.
10. $8 \times (512\times512 + 512) = 8 \times 262{,}656 = \mathbf{2{,}101{,}248}$ (≈ 2.10 M).
11. In common: both target **disentanglement**, and neither guarantees it. Differences (any two): β-VAE changes the **loss** (a $\beta$ weight on the KL term) while StyleGAN changes the **architecture** and leaves the loss alone; β-VAE pushes the latent **toward** a fixed factorised prior while StyleGAN **removes** the prior constraint by introducing an unsampled space; β-VAE pays in reconstruction blur while StyleGAN pays in parameters and depth.

</details>

## Beyond the slides

**Gap: the deck gives the "$\mathcal{Z}$ must follow the data density" argument as a single quoted sentence and never unpacks it.**
**Why it matters:** it is the entire justification for the architecture, and without it StyleGAN looks like an arbitrary pile of tricks. The unpacking is in *Why $\mathcal{Z}$ is forced to be entangled* and quantified in N4: real attributes are correlated, a factorised Gaussian prior is not, and the generator must warp the gap away. Put the warping in a cheap 8-layer MLP and the expensive convolutional stack gets a pre-straightened input. Any exam question asking "why does StyleGAN need an intermediate latent space?" wants that argument, not "to make it disentangled" — which merely restates the goal.

**Gap: progressive growing is drawn but never named or explained.**
**Why it matters:** page 13's resolution ladder *is* progressive growing's skeleton, but the deck presents it as a static architecture rather than a training schedule. StyleGAN trains $4\times4$ first, then fades in each higher-resolution block over thousands of iterations. That is what makes a $1024\times1024$ GAN trainable at all, given the instability of [Lec 34](34-gan-convergence.md). It is also what StyleGAN 2 later removes — so knowing that it is a *schedule*, not a fixed structure, is what makes [Lec 42](42-stylegan2.md) make sense.

**Gap: page 13 says two convolutions at every resolution including $4\times4$; the paper's own figure on page 16 shows only one there.**
**Why it matters:** look carefully at the $4\times4$ block of the style-based generator on page 16: `Const 4×4×512 → ⊕B → AdaIN → Conv 3×3 → ⊕B → AdaIN`. There is **one** $3\times3$ convolution, not two — the constant tensor takes the first conv's place, which is exactly why it exists. That makes the true count **17 convolutions and 18 AdaIN/style inputs**, not 18 and 18. The two slides contradict each other. **For this exam, answer 18 conv layers**, since both page 12's text and page 13's labels say so; but if a question counts *style inputs* or *AdaIN layers*, 18 is right for a different reason.

**Gap: no mention of style mixing or truncation.**
**Why it matters:** both are standard parts of using a trained StyleGAN. **Style mixing** (two $\mathbf{w}$'s, one for coarse layers, one for fine) is the demonstration that the per-layer design works, and is used as a regulariser during training. **The truncation trick** pulls $\mathbf{w}$ toward the average $\bar{\mathbf{w}}$, $\mathbf{w}' = \bar{\mathbf{w}} + \psi(\mathbf{w}-\bar{\mathbf{w}})$ with $\psi < 1$, trading diversity for quality — every impressive StyleGAN sample you have seen was generated with $\psi \approx 0.7$. Neither appears on these slides.

**Gap: the loss function is never mentioned, which can mislead.**
**Why it matters:** a reader could finish this deck thinking StyleGAN introduced a new objective. It did not. The discriminator, the adversarial loss and the training game are unchanged from [Lec 33](33-gan-objective.md) — the paper's own framing is that *all* of its improvements are generator-side architecture. **StyleGAN is a generator redesign, full stop.** That is a clean, likely exam discrimination against CycleGAN ([Lec 40](40-cyclegan.md)), whose contribution was a new *loss term*.

**Gap: no quantitative measure of disentanglement is given.**
**Why it matters:** the paper introduces two — **perceptual path length** (how much the image changes per unit step in latent space; smoother means less entangled) and **linear separability** (can a linear classifier find an attribute boundary in the space?). Both score $\mathcal{W}$ better than $\mathcal{Z}$, which is the empirical evidence for the whole design. Without them, "W-space becomes disentangled" is an assertion rather than a result.

## Cut from the slides

Pages 1, 2, 17 and 18 are the title, the contents list, an empty "Summary" divider and the "Next Session: StyleGAN 2" card. Page 3 is a plain traditional-GAN recap — latent vector into $G$, $G(z) \to$ image, $z \sim N(0,1)$ — wholly owned by [Lec 32](32-gan-architecture.md), so it is compressed into one sentence of the opening section rather than embedded. Page 11 (Affine Transformation) is taught in full in prose but not shown as a figure, because its content — $y = \mathbf{W}x+b$, $z\to w\to y = Aw+b$, "style parameters injected into AdaIN at every layer", and the clay/sculpting-instructions analogy — is reproduced in its entirety by page 15's richer slide, which also carries the AdaIN formula and the worked numbers. The thispersondoesnotexist.com face grids on pages 4, 5 and 6 are the same stock of generated faces reused; page 5 and 6 are embedded because their text differs and matters, page 4's for the motivation statement. The deck's plain $z$, $w$, $x_i$ are set as $\mathbf{z}$, $\mathbf{w}$, $\mathbf{x}_i$ per CONTRACT §3, its $N(0,1)$ as $\mathcal{N}(\mathbf{0},\mathbf{I})$ with variance second, and its $Z$/$W$ as the spaces $\mathcal{Z}$/$\mathcal{W}$. Everything substantive on pages 3–16 is in this chapter.
