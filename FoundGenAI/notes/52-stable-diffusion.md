# Lec 52 — Stable Diffusion (Latent Diffusion Models)

> **Source:** `Lec 52.pdf` (18 pages) · **Week 8** · **Playlist:** Lec 52
> **Prereqs:** [Lec 46 — DDPM Forward Process](46-ddpm-forward.md), [Lec 47 — DDPM Reverse Process](47-ddpm-reverse.md), [Lec 49 — U-Net for Denoising](49-unet.md), [Lec 51 — Classifier-Free Diffusion Guidance](51-classifier-free-guidance.md), [Lec 21 — Introduction to VAE and the Encoder](21-vae-encoder.md)
> **Feeds into:** [Lec 53 — Hands-on: Reverse Diffusion](53-reverse-diffusion-handson.md), [Lec 57 — Transformer Encoder](57-transformer-encoder.md)

## Why this lecture exists

Everything in Week 7 and Week 8 so far has run diffusion on pixels. That is why it is slow: a single 512×512 colour image is 786,432 numbers, and the U-Net must process all of them at *every one* of hundreds of denoising steps. Train a model that way and you need a research lab's hardware budget.

This lecture is one structural move that removes the problem. Train an autoencoder first, freeze it, and run the entire diffusion process on its **compressed latent** instead of on pixels. A 512×512×3 image becomes a 64×64×4 latent — 16,384 numbers, 48× fewer — and the U-Net, the schedule, the loss and the sampler are all unchanged except that they now operate on latents. At the end, one pass of the decoder turns the finished latent back into an image.

Add a text encoder and a cross-attention layer so a prompt can steer the U-Net, and you have Stable Diffusion: the model that put image generation on consumer graphics cards.

## The ideas

### LDM is the algorithm, Stable Diffusion is the implementation

The deck is unusually careful about this on page 4 and it is a free exam mark:

- **Latent Diffusion Models (LDMs)** are the *algorithm*, introduced by Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", **CVPR 2022**.
- **Stable Diffusion** is "the practical, open-source implementation that uses LDM for text-to-image generation".

So "Stable Diffusion is a latent diffusion model" is true; "latent diffusion model" and "Stable Diffusion" are not synonyms.

### The problem: pixel-space diffusion is enormous

![Motivation slide with a box 'Problem with Pixel-Space Diffusion', the statement that DDPM performs forward and reverse diffusion directly on high-dimensional pixel-space images, x in R^(HxWx3), a green example box computing that a 512x512x3 RGB image contains 786,432 pixel values, and three bulleted problems](../assets/pages/lec52/p-03.png)
*Fig. — The only arithmetic in the entire deck, and it is the argument for the whole model: **786,432**, and every one of those numbers goes through the network at every denoising step. Page 3.*

The deck's example: a $512\times512\times3$ RGB image contains $512\times512\times3 = 786{,}432$ pixel values, and "these must be processed during every denoising step, making diffusion computationally expensive." Three consequences are listed:

- high memory consumption;
- high computational cost;
- slow inference due to **hundreds of denoising steps**.

The third is the multiplier that makes the other two unbearable. A classifier is run once. A diffusion sampler is run $T$ times, so every inefficiency is paid $T$ times over.

### The move: do the diffusion somewhere smaller

![Latent Diffusion slide with a Core Idea box: compress the image into a compact latent representation and perform diffusion in latent space; the Rombach CVPR 2022 citation; and a blue box distinguishing LDMs the algorithm from Stable Diffusion the implementation, ending 'Diffusion occurs in a 64x64 latent space for a 512x512 image'](../assets/pages/lec52/p-04.png)
*Fig. — "Compress the image into a compact latent representation and perform diffusion in latent space." Nine words, and the entire lecture follows from them. Page 4.*

The deck's last bullet is the one to hold onto: **diffusion occurs in a $64\times64$ latent space for a $512\times512$ image.**

Now do the arithmetic the deck leaves out. Stable Diffusion's autoencoder produces a latent with **4 channels**, so the latent tensor is $64\times64\times4$:

$$\underbrace{512\times512\times3}_{786{,}432} \quad\longrightarrow\quad \underbrace{64\times64\times4}_{16{,}384}, \qquad \frac{786{,}432}{16{,}384} = \boxed{48\times\ \text{fewer values}}$$

Spatially the reduction is a factor of $f = 512/64 = 8$ in each direction, so $8^2 = 64\times$ fewer spatial positions; the channel count goes *up* slightly, from 3 to 4, which gives back a factor of $4/3$ and lands on $64 \times 3/4 = 48$.

**That number is the whole model.** Every per-step cost — memory for activations, convolution FLOPs, attention cost — scales with the number of values the network has to touch. Divide that by 48 and a model that needed a cluster fits on one GPU. The deck's page 15 says the same thing in words: "This makes Stable Diffusion much more computationally efficient than pixel-space diffusion."

> **The deck never states the 4 channels, and never computes the ratio.** It says "64×64 latent space" and stops. If an MCQ says "a 512×512 image is compressed 48×", that is this calculation; if it says "64×", someone assumed 3 latent channels. Both numbers are defensible depending on the autoencoder; Stable Diffusion v1 uses 4, giving 48.

A second reason for the move, which the deck implies but does not spell out: the latent is not just smaller, it is **perceptually cleaner**. The autoencoder has already thrown away the imperceptible high-frequency detail that a pixel-space diffusion model would otherwise waste most of its capacity modelling. The diffusion model gets to spend all of its capacity on semantics.

### The three components

Stable Diffusion is three networks, and knowing which does what is the most examinable structural fact in the lecture.

| Component | Symbol | What it does | Trained? | Owned by |
|---|---|---|---|---|
| **VAE encoder** | $\mathcal{E}$ | image → latent, $\mathbf{z} = \mathcal{E}(\mathbf{x})$ | trained first, then **frozen** | [Lec 21](21-vae-encoder.md), [Lec 22](22-elbo-and-vae-loss.md) |
| **VAE decoder** | $\mathcal{D}$ | latent → image, $\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$ | frozen with the encoder | [Lec 21](21-vae-encoder.md), [Lec 12](12-autoencoder-types.md) for the upsampling |
| **U-Net denoiser** | $\epsilon_\theta$ | predicts the noise in a **noisy latent** | **this is what diffusion training trains** | [Lec 49](49-unet.md) |
| **Text encoder** | $\tau_\theta$ | prompt → embedding, $\mathbf{c} = \tau_\theta(y)$ | pretrained **CLIP**, frozen | this chapter; [Lec 57](57-transformer-encoder.md) for the architecture |

The autoencoder is an ordinary encoder/decoder pair of the kind [Lec 21](21-vae-encoder.md) builds; the decoder's upsampling uses transposed convolution, derived in [Lec 12](12-autoencoder-types.md) — do not re-derive it. Its loss is the reconstruction loss of [Lec 11](11-reconstruction-loss.md) plus a very lightly weighted KL term (so lightly that the latent is essentially an autoencoder code, not a sampled VAE latent — see *Beyond the slides*).

![Latent Diffusion Model Training Pipeline: a flow from Input Image x through AutoEncoder to Latent z = E(x), Forward Diffusion, Noisy latent, U-Net, Reverse Diffusion, Predicted noise, Clean latent, Decoder, Generate Image, with a green note that the autoencoder is trained before diffusion and then frozen](../assets/pages/lec52/p-05.png)
*Fig. — Follow the arrows and notice there is no path from the U-Net back to the autoencoder. The green annotation is the reason: **"The autoencoder is trained before diffusion and then frozen."** Two separate training stages, never joint. Page 5.*

**The two-stage training is examinable.** Stage 1 trains $\mathcal{E}$ and $\mathcal{D}$ on images alone, with no diffusion involved. Stage 2 freezes both and trains only $\epsilon_\theta$ on latents. The text encoder is pretrained CLIP and is never trained here at all. So of the four components, exactly **one** is trained by this lecture's procedure.

### The objective: DDPM's loss, with $\mathbf{z}$ in place of $\mathbf{x}$

![Latent Diffusion Objective Function slide: 'Instead of predicting noise in pixel space, the network predicts noise in latent space', the boxed training loss L = E[|| eps - eps_theta(z_t, t) ||^2], definitions of z_t, eps and eps_theta, and the note 'This is the same DDPM objective, but applied to latent vectors'](../assets/pages/lec52/p-06.png)
*Fig. — Compare this with [Lec 47](47-ddpm-reverse.md)'s simplified loss and spot the difference: there is exactly one, $\mathbf{z}_t$ where $\mathbf{x}_t$ used to be. That is the measure of how little the latent move changes. Page 6.*

$$\mathcal{L} = \mathbb{E}\big[\lVert \epsilon - \epsilon_\theta(\mathbf{z}_t, t)\rVert^2\big]$$

with $\mathbf{z}_t$ the **noisy latent**, $\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ the true Gaussian noise, and $\epsilon_\theta$ the prediction. The deck's own summary: **"This is the same DDPM objective, but applied to latent vectors."**

Everything from [Lec 46](46-ddpm-forward.md) carries over symbol for symbol. The forward jump is

$$\mathbf{z}_t = \sqrt{\bar\alpha_t}\,\mathbf{z}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \mathbf{z}_0 = \mathcal{E}(\mathbf{x}_0)$$

with the same $\beta_t$, $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_s \alpha_s$. The reverse process is [Lec 47](47-ddpm-reverse.md)'s, unchanged. **Nothing about diffusion is re-derived; only the space it runs in has changed.** For the conditional version, $\epsilon_\theta(\mathbf{z}_t,t)$ becomes $\epsilon_\theta(\mathbf{z}_t,t,\mathbf{c})$ and [Lec 51](51-classifier-free-guidance.md)'s CFG is applied exactly as written there.

### The text-to-image pipeline

![Stable Diffusion Pipeline (Text to Image Generation): Text Prompt into a Text Encoder annotated 'pretrained CLIP text encoder to understand its semantic meaning', producing a Text Embedding c = tau_theta(y), which feeds a Cross Attention box annotated 'This is where the text enters', into a U-Net that also takes the Noisy latent z_t, then Clean Latent, Decoder, Generate Image, with 'Frozen' beneath the decoder](../assets/pages/lec52/p-07.png)
*Fig. — The black callout is the sentence to remember: **"This is where the text enters."** Not at the input, not concatenated to the latent — through cross-attention, inside the U-Net. Page 7.*

Reading the pipeline left to right:

1. The user's **text prompt** $y$.
2. A **pretrained CLIP text encoder** $\tau_\theta$ converts it to a **text embedding** $\mathbf{c} = \tau_\theta(y)$. The deck: these embeddings "represent semantic meaning of the text". For Stable Diffusion v1 this is CLIP ViT-L/14, which emits **77 token vectors of 768 dimensions each** — a sequence, not one vector, which is why cross-attention is possible at all.
3. A **noisy latent** $\mathbf{z}_t$ (pure Gaussian noise at $t=T$).
4. The **U-Net** denoises $\mathbf{z}_t$, with $\mathbf{c}$ injected at several layers by **cross-attention**.
5. Repeat for every $t$ until a **clean latent** $\mathbf{z}_0$ remains.
6. The frozen **decoder** maps it to pixels.

> **CLIP** (Contrastive Language–Image Pretraining) is a model trained on hundreds of millions of image–caption pairs to put an image and its caption *near each other* in a shared embedding space. That is exactly the property Stable Diffusion needs: an embedding of "a yellow butterfly" that already lives near pictures of yellow butterflies. The deck names CLIP and goes no further; the mechanism belongs to the multimodal material in Week 12.

### Cross-attention: how text reaches the pixels

[Lec 57](57-transformer-encoder.md) owns self-attention, $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ and the scaled dot-product formula for this book. Those chapters are later in the course, so here is exactly enough to use the mechanism, and no more.

**The problem cross-attention solves.** The latent is a $64\times64$ grid of feature vectors; the prompt is a sequence of 77 token vectors. They have different shapes, different lengths and different meanings. You cannot add them or concatenate them. What you *can* do is let every position in the latent grid **look up** which words are relevant to it.

**Attention in one paragraph.** Attention is a soft dictionary lookup. You have a **query** (what am I looking for?), a set of **keys** (what is each entry about?), and a set of **values** (what each entry contains). You score every query against every key, softmax the scores into weights that sum to 1, and return the weighted average of the values. **Self**-attention takes all three from the same sequence. **Cross**-attention takes the query from one sequence and the keys and values from another — which is precisely "latent asks, text answers".

![Cross Attention slide with three boxes: Q = W_Q^(i) · phi_i(z_t) with notes that the query comes from intermediate U-Net feature maps and phi_i(z_t) is in R^(N x d); K = W_K^(i) tau_theta(y) from text embeddings; and V = W_V^(i) tau_theta(y) also from text embeddings](../assets/pages/lec52/p-08.png)
*Fig. — The asymmetry is the entire point. **One** of the three comes from the image side; **two** come from the text side. Memorise which: $\mathbf{Q}$ from the latent, $\mathbf{K}$ and $\mathbf{V}$ from the text. Page 8.*

$$\mathbf{Q} = \mathbf{W}_Q^{(i)}\,\phi_i(\mathbf{z}_t), \qquad \mathbf{K} = \mathbf{W}_K^{(i)}\,\tau_\theta(y), \qquad \mathbf{V} = \mathbf{W}_V^{(i)}\,\tau_\theta(y)$$

where $\phi_i(\mathbf{z}_t) \in \mathbb{R}^{N\times d}$ is the flattened intermediate feature map produced by the $i$-th layer of the U-Net ($N$ spatial positions, feature dimension $d$), and $\mathbf{W}_Q^{(i)}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)}$ are learnable projection matrices — a fresh set per layer $i$, which is why cross-attention can mean different things at different resolutions.

> **Two notation warnings on this slide.** (1) The deck reuses $\phi$ for the U-Net's intermediate feature map. In [Lec 50](50-classifier-guidance.md) $\phi$ was the classifier's parameters, in [Lec 51](51-classifier-free-guidance.md) the similar-looking $\varnothing$ was the null token, and in the VAE arc $\phi$ was the encoder. Four meanings in one course. (2) The deck writes $\mathbf{W}_Q^{(i)}\cdot\phi_i(\mathbf{z}_t)$ with $\phi_i(\mathbf{z}_t)$ declared as $N\times d$, which does not compose as left multiplication — you would need $\phi_i(\mathbf{z}_t)^\top$, or the row convention $\phi_i(\mathbf{z}_t)\mathbf{W}_Q^{(i)}$. This is the same data-left/data-right switch flagged for Lec 11 versus Lec 13 (errata batch 4). **Read shapes off the equation in front of you, never from memory.** Throughout this chapter $\mathbf{Q}$ is $N\times d_k$, and $\mathbf{K}, \mathbf{V}$ are $M\times d_k$ with $M$ the number of text tokens.

![Cross Attention slide with the boxed formula Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V, a 'Why Cross attention is required' box, the example prompt 'A yellow butterfly sitting on a purple flower', and a three-row table mapping latent regions Butterfly, Flower and Background to the words they attend to](../assets/pages/lec52/p-09.png)
*Fig. — The table is the clearest statement in the deck of what attention buys: **different regions of one image attend to different words of one prompt**. A single global text vector could not do this. Page 9.*

$$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

(The deck writes $\sqrt{d}$; CONTRACT §3's symbol for this quantity is $d_k$, the key dimension. Same thing.)

Four things to be able to say about this formula:

- $\mathbf{Q}\mathbf{K}^\top$ is $N\times M$ — **one score for every (latent position, text token) pair**. With $N = 4096$ and $M = 77$ that is 315,392 scores.
- $\sqrt{d}$ is a **scaling factor**, not a probability. Dot products of $d$-dimensional vectors grow like $\sqrt{d}$, and feeding large numbers to a softmax saturates it into a one-hot; dividing keeps the scores in a range where the softmax has useful gradients.
- The **softmax is taken over the text tokens** (each row), so every latent position distributes a total weight of exactly 1 across the words. A position cannot attend to "everything a lot".
- Multiplying by $\mathbf{V}$ returns, for each latent position, a **weighted blend of word content** — which is then added back into the U-Net's feature map and carried forward.

The deck's example makes it concrete with the prompt *"A yellow butterfly sitting on a purple flower."*

| Latent region | Attends to |
|---|---|
| Butterfly | "yellow", "butterfly" |
| Flower | "purple", "flower" |
| Background | remaining context |

This is also the mechanism behind a famous failure: nothing *forces* the butterfly region to pick "yellow" rather than "purple". When a model paints a purple butterfly on a yellow flower, the cross-attention weights bound the wrong adjective to the wrong noun. N5 below computes a clean version of this lookup by hand.

### The full architecture

The deck walks one diagram — the figure from Rombach et al. — three times over pages 10, 11 and 12, outlining a different block in red dashes each time: **pixel space** (page 10), **latent space** (page 11), **conditioning** (page 12). Page 10's lesson is that the model enters and leaves pixel space exactly once: the encoder at training time, the decoder at the end of sampling. Everything in between is latent.

![Architecture diagram from Rombach et al. with the Latent Space block outlined in red dashes, beneath four bullets stating that diffusion occurs entirely in latent space, Gaussian noise is progressively added to z, the U-Net learns to remove it, and Cross Attention is applied at several resolutions](../assets/pages/lec52/p-11.png)
*Fig. — Count the $\frac{Q}{KV}$ boxes inside the U-Net: four, two on the way down and two on the way up. The red bullet says it explicitly — **"Cross Attention is applied at several resolutions"** — which is why the conditioning can affect both coarse layout and fine texture. Page 11.*

The four bullets on page 11:

- Diffusion occurs **entirely** in the latent space.
- Gaussian noise is progressively added to the latent representation $\mathbf{z}$ (the forward process of [Lec 46](46-ddpm-forward.md)).
- The U-Net learns to remove this noise during reverse diffusion ([Lec 47](47-ddpm-reverse.md), [Lec 49](49-unet.md)).
- **Cross-attention is applied at several resolutions.**

That last point matters. The U-Net of [Lec 49](49-unet.md) has a contracting path, a bottleneck and an expanding path; cross-attention blocks are inserted at several levels of both paths. At coarse resolution, text steers *composition* ("butterfly on flower"); at fine resolution it steers *appearance* ("yellow"). One injection point at the bottleneck would give you the first and not the second.

Page 12 highlights the **conditioning** block instead, and its box lists what can go in: "Stable Diffusion supports multiple conditioning modalities — text prompts, semantic maps, images." Note that $\tau_\theta$ is drawn as a *generic* condition encoder. Text is one input to it, not the only one; the same architecture does segmentation-map-to-image and image-to-image with a different $\tau_\theta$.

### Two ways a condition can enter, and the switch between them

![Conditioning slide showing three examples: a Semantic Map to Image pair with a segmentation mask of a dog and two cats becoming a photograph, a Text Prompting example 'The cat is sitting on a red carpet' with the generated image, and an Image to Image translation pair colourising a greyscale portrait](../assets/pages/lec52/p-13.png)
*Fig. — Three different conditioning modalities, one architecture. Only the middle one needs cross-attention; the other two carry spatial structure and can be concatenated. Page 13.*

![Architecture diagram with the 'switch' legend circled in blue and 'concat' boxed in red, above Method 1 Concatenation (used when conditioning has spatial information: segmentation map, low-resolution image, concatenated with the latent feature maps) and Method 2 Cross Attention (used for text embeddings, no spatial dimensions), and a Switch summary box](../assets/pages/lec52/p-14.png)
*Fig. — The little "switch" symbol in the original paper's legend turns out to carry real content: it selects between the two injection routes. This slide is a strong MCQ target because it gives a clean rule. Page 14.*

| | **Method 1 — Concatenation** | **Method 2 — Cross-attention** |
|---|---|---|
| Used when | the condition **has spatial information** | the condition has **no spatial dimensions** |
| Examples | segmentation map, low-resolution image | text embeddings |
| How | the condition is **concatenated with the latent feature maps** (extra input channels) | $\mathbf{K},\mathbf{V}$ from the condition, $\mathbf{Q}$ from the U-Net features |
| Why that way | spatial conditions already align pixel-for-pixel with the latent, so position $i$ can simply be stacked on position $i$ | text has no position in the image, so every image position must *decide* which words it needs |

The deck's summary box: "Spatial conditioning (segmentation maps, low-resolution images) uses **concatenation**. Text conditioning uses **cross-attention**."

The reasoning behind the rule is worth stating because an exam can ask *why*. A segmentation map is a grid of the same shape as the image — "this pixel is dog" lines up with "this latent position" — so a channel-wise stack carries the information with zero machinery. A prompt has no such alignment: the word "yellow" does not belong at any particular coordinate, so the model has to learn *where* it belongs. That learning is what the attention weights are.

### What you actually gained

![Advantage of Stable Diffusion over DDPM slide: DDPM adds Gaussian noise to a high-resolution photograph until it becomes pure noise; Stable Diffusion first compresses the photograph into a compact 'blueprint' (the latent), then adds noise to that blueprint, cleans it step by step during generation and reconstructs the full image, making it much more computationally efficient](../assets/pages/lec52/p-15.png)
*Fig. — The deck's "blueprint" metaphor is a good one: you do not redraw a building to redesign it, you edit the plan and then build. Page 15.*

| | DDPM ([Lec 46](46-ddpm-forward.md)/[Lec 47](47-ddpm-reverse.md)) | Stable Diffusion |
|---|---|---|
| Diffusion runs on | the **pixels** of a high-resolution photograph | the **latent** "blueprint" |
| Values per step (512×512) | 786,432 | 16,384 |
| Noise is added to | the image | the latent |
| Extra networks needed | none | a trained-and-frozen autoencoder, plus a text encoder if conditioning on text |
| Output path | the denoised image *is* the output | the clean latent is **decoded** to an image |
| Cost | high memory, high compute, slow inference | **~48× fewer values per step** |

![Grid of LAION text-to-image samples from the 1.45B model: a street sign reading 'Latent Diffusion', a zombie in the style of Picasso, a half-mouse-half-octopus animal, an illustration of a conscious neural network, a squirrel eating a burger, a watercolour octopus-chair, and a shirt reading 'I love generative models'](../assets/pages/lec52/p-16.png)
*Fig. — Seven prompts, two samples each, from the 1.45-billion-parameter LAION model. Notice the street sign: the upper sample spells "LATENT DIFFUSION" and the lower one spells "LATETEN DIFFUSION". Text rendering is where cross-attention's bag-of-concepts behaviour shows through most clearly. Page 16.*

## Worked numericals

**The deck contains exactly one worked number** — page 3's $512\times512\times3 = 786{,}432$ — reproduced and verified as N1. The other five are constructed, using Stable Diffusion v1's actual dimensions where the deck gives none.

### N1. The deck's pixel count, and the compression ratio it does not compute

**Given:** a $512\times512\times3$ RGB image, and the deck's statement that diffusion runs in a $64\times64$ latent space. Stable Diffusion v1's latent has 4 channels.
**Find:** the pixel count (the deck's number), the latent count, and the ratio.

1. Pixels: $512\times512 = 262{,}144$ spatial positions. Times 3 colour channels: $262{,}144\times3 = 786{,}432$. **Matches the slide exactly.**
2. Latent: $64\times64 = 4{,}096$ spatial positions. Times 4 channels: $4{,}096\times4 = 16{,}384$.
3. Ratio: $786{,}432 / 16{,}384$. Simplify: $786{,}432/16{,}384 = 48$ exactly (check: $16{,}384\times48 = 16{,}384\times50 - 16{,}384\times2 = 819{,}200 - 32{,}768 = 786{,}432$). ✓
4. Decompose the 48: spatial factor $(512/64)^2 = 8^2 = 64$; channel factor $3/4 = 0.75$; product $64\times0.75 = 48$.

**Answer:** 786,432 pixel values against 16,384 latent values — a **48× reduction**, made of a 64× spatial reduction partly given back by the rise from 3 to 4 channels.

Two variants an exam could use. With a 3-channel latent the ratio would be $(512/64)^2 = 64$. For $256\times256$ images with the same $f=8$ encoder, pixels $= 196{,}608$ and latent $= 32\times32\times4 = 4{,}096$, ratio again **48** — the ratio depends on the autoencoder, not the image size.

### N2. The encoder's downsampling stages

**Given:** the autoencoder reduces $512\times512$ to $64\times64$ using stride-2 convolutional stages, each halving both spatial dimensions (the convolution geometry of [Lec 05](05-cnn-a.md)).
**Find:** how many stages, and the spatial size after each.

1. The overall downsampling factor is $f = 512/64 = 8$.
2. Each stride-2 stage halves, so $2^n = 8 \Rightarrow n = 3$.
3. The chain: $512 \to 256 \to 128 \to 64$.
4. Values at each stage, if the channel count were held at 3: $786{,}432 \to 196{,}608 \to 49{,}152 \to 12{,}288$. In practice channels *rise* through the encoder (the usual CNN trade of space for depth, [Lec 06](06-cnn-b.md)) before a final projection to 4 channels.

**Answer:** **3** stride-2 stages, $512\to256\to128\to64$, downsampling factor $f=8$. The decoder mirrors this with 3 upsampling stages, using the transposed convolution of [Lec 12](12-autoencoder-types.md).

Why $f=8$ and not 16? Because compression is not free: the autoencoder must still be able to reconstruct. Push $f$ higher and the decoder starts inventing detail; push it lower and you lose the speed-up. $f=8$ with 4 channels is the setting Rombach et al. found to be the best trade, and it is why "64×64" is a number worth memorising.

### N3. Why attention in particular becomes affordable

**Given:** convolution cost scales **linearly** with the number of spatial positions, while self-attention cost scales with the **square** of it.
**Find:** the saving on each, going from $512\times512$ to $64\times64$.

1. Spatial positions: pixel space $512\times512 = 262{,}144$; latent space $64\times64 = 4{,}096$. Ratio $262{,}144/4{,}096 = 64$.
2. Convolution: cost $\propto N$, so the saving is $\mathbf{64\times}$ (48× once you count the extra channel, as in N1).
3. Self-attention: cost $\propto N^2$. Pixel space $262{,}144^2 = 6.872\times10^{10}$; latent $4{,}096^2 = 1.678\times10^{7}$.
4. Ratio: $6.872\times10^{10}/1.678\times10^{7} = 4{,}096 = 64^2$.

**Answer:** convolutions get **64×** cheaper; self-attention gets **4,096×** cheaper.

This is the hidden reason the latent move matters so much. A U-Net with attention layers at full 512×512 resolution is not merely expensive, it is infeasible — a $262{,}144\times262{,}144$ attention matrix is $6.9\times10^{10}$ entries, about 275 GB in 32-bit floats for a *single* layer of a *single* image. In the latent space the same matrix is 67 MB. **Latent diffusion did not just make attention cheaper; it made attention possible**, and attention is how text conditioning works.

### N4. Cross-attention shapes at Stable Diffusion v1 scale

**Given:** at the U-Net's top level the feature map is $64\times64$ with $d = 320$ channels. The CLIP text encoder emits $M = 77$ token vectors of 768 dimensions, projected to the same $d = 320$ for keys and values.
**Find:** the shapes of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, the attention matrix, and its entry count; then the same at the $8\times8$ bottleneck.

1. Flatten the spatial grid: $N = 64\times64 = 4{,}096$ query positions. So $\mathbf{Q}$ is $4{,}096\times320$.
2. $\mathbf{K}$ and $\mathbf{V}$ are both $77\times320$.
3. $\mathbf{Q}\mathbf{K}^\top$ is $(4{,}096\times320)(320\times77) = 4{,}096\times77$.
4. Entries: $4{,}096\times77 = 315{,}392$.
5. Softmax is applied along the second axis, so there are 4,096 independent softmaxes of length 77, and **every row sums to exactly 1**.
6. Output $= \mathrm{softmax}(\cdot)\mathbf{V}$ is $(4{,}096\times77)(77\times320) = 4{,}096\times320$ — the same shape as $\mathbf{Q}$, so it can be added straight back into the U-Net's feature map.
7. At the $8\times8$ bottleneck, $N = 64$: the attention matrix is $64\times77 = 4{,}928$ entries.

**Answer:** $\mathbf{Q}: 4096\times320$; $\mathbf{K},\mathbf{V}: 77\times320$; attention $4096\times77 = 315{,}392$ entries at the top level, $64\times77 = 4{,}928$ at the bottleneck; output $4096\times320$.

Notice how modest 315,392 is — about 1.2 MB. Compare with N3's pixel-space self-attention. Cross-attention is cheap precisely because the text side is short ($M=77$), so the matrix is a thin rectangle, not a square.

### N5. Cross-attention computed by hand

**Given:** the deck's prompt *"A yellow butterfly sitting on a purple flower"*, reduced to $M=4$ tokens with idealised orthonormal keys in $d=4$ dimensions: $\mathbf{K} = \mathbf{I}_4$ with rows ("yellow", "butterfly", "purple", "flower"), and $\mathbf{V} = \mathbf{I}_4$. Three query positions:
$$\mathbf{Q} = \begin{bmatrix} 4 & 4 & 0 & 0\\ 0 & 0 & 4 & 4\\ 1 & 1 & 1 & 1\end{bmatrix} \quad \begin{matrix}\text{(butterfly region)}\\ \text{(flower region)}\\ \text{(background)}\end{matrix}$$
**Find:** the attention weights, and verify the deck's table.

1. Scores: $\mathbf{Q}\mathbf{K}^\top = \mathbf{Q}$ (since $\mathbf{K}=\mathbf{I}$). Scale by $\sqrt{d} = \sqrt{4} = 2$:
 row 1 → $[2, 2, 0, 0]$; row 2 → $[0, 0, 2, 2]$; row 3 → $[0.5,0.5,0.5,0.5]$.
2. Row 1 softmax. Exponentiate: $e^2 = 7.389056$, $e^2 = 7.389056$, $e^0 = 1$, $e^0 = 1$. Sum $= 16.778112$.
 Weights: $7.389056/16.778112 = 0.440399$ (twice), $1/16.778112 = 0.059601$ (twice).
 Check: $2(0.440399) + 2(0.059601) = 0.880798 + 0.119202 = 1.000000$. ✓
3. Row 2 is row 1 with the halves swapped, by symmetry: $[0.059601,\ 0.059601,\ 0.440399,\ 0.440399]$.
4. Row 3: all four scores equal, so the softmax is uniform: $[0.25, 0.25, 0.25, 0.25]$.

| Latent region | yellow | butterfly | purple | flower | mass on the right two words |
|---|---|---|---|---|---|
| Butterfly | $0.440399$ | $0.440399$ | $0.059601$ | $0.059601$ | $88.08\%$ |
| Flower | $0.059601$ | $0.059601$ | $0.440399$ | $0.440399$ | $88.08\%$ |
| Background | $0.250000$ | $0.250000$ | $0.250000$ | $0.250000$ | $-$ |

**Answer:** as tabulated. The butterfly position puts **88.08%** of its attention on "yellow" and "butterfly"; the flower position puts the same on "purple" and "flower"; the background spreads evenly, which is the deck's "remaining context" row. Because $\mathbf{V}=\mathbf{I}$, the context vector written back into each latent position *is* its attention row.

Note what controls the sharpness: the magnitude of $\mathbf{Q}$. Halve the queries to 2 instead of 4 and the scores become $[1,1,0,0]$, giving weights $0.365529/0.134471$ — only $73.1\%$ on the right words. **The $\sqrt{d}$ division exists to stop this knob from running away**: without it, a 320-dimensional dot product would produce scores large enough to make every row effectively one-hot.

### N6. End-to-end sampling cost, with CFG

**Given:** 50 denoising steps, [Lec 51](51-classifier-free-guidance.md)'s classifier-free guidance (two U-Net passes per step), and one decoder pass at the end. Take one latent-space U-Net pass as 1.00 unit; the decoder, which works at full pixel resolution but runs only once, costs 15 units; the CLIP text encoder costs 0.5 units and runs once.
**Find:** the total, and how it splits.

1. Text encoding: $1 \times 0.5 = 0.5$ units — done **once**, before the loop, since the prompt does not change.
2. Denoising: $50$ steps $\times\ 2$ passes $\times\ 1.00 = 100.0$ units.
3. Decoding: $1 \times 15 = 15.0$ units — done **once**, after the loop.
4. Total: $0.5 + 100.0 + 15.0 = 115.5$ units.
5. Split: denoising $100/115.5 = 86.6\%$, decoding $13.0\%$, text encoding $0.4\%$.

**Answer:** $115.5$ units, of which **86.6% is the denoising loop**.

Two readings. First, the expensive, pixel-resolution decoder is affordable *because it runs once* while the U-Net runs 100 times — exactly the asymmetry the latent move exploits. Second, compare with the pixel-space counterfactual: at 48× the per-step cost, step 2 alone would be $100\times48 = 4{,}800$ units, making the total $\approx 4{,}800$ instead of $115.5$ — a **41× end-to-end speed-up** even after paying for the encoder and decoder.

## Code

Two things the deck asserts and never computes: the compression ratio, and what the cross-attention table on page 9 actually looks like as numbers. Both fit in thirty lines of NumPy.

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---- 1. the compression argument, in numbers (deck page 3 + page 4)
pix = 512 * 512 * 3
lat = 64 * 64 * 4
print(f"pixel-space values  : {pix:,}")
print(f"latent-space values : {lat:,}")
print(f"reduction factor    : {pix/lat:.0f}x   (spatial downsample f = {512//64})")

# ---- 2. cross-attention, the deck's butterfly/flower prompt, by hand (page 9)
# 3 latent query positions x 4 CLIP text tokens, feature dimension d = 4.
tokens = ["yellow", "butterfly", "purple", "flower"]
K = np.eye(4)                       # one key per token (idealised, orthonormal)
V = np.eye(4)                       # values = the token's own content
Q = np.array([[4., 4., 0., 0.],     # latent region that is becoming the butterfly
              [0., 0., 4., 4.],     # latent region that is becoming the flower
              [1., 1., 1., 1.]])    # background: no strong preference
d = K.shape[1]

scores = Q @ K.T / np.sqrt(d)       # 3 x 4
A = np.exp(scores - scores.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)        # softmax over the 4 text tokens, rows sum to 1

print("\nattention weights (rows sum to 1):", tokens)
for name, row in zip(["butterfly region", "flower region   ", "background      "], A):
    print(f"  {name} {row}")
print("row sums:", A.sum(1))
print("\ncontext written into the latent (A @ V):\n", A @ V)
```

```
pixel-space values  : 786,432
latent-space values : 16,384
reduction factor    : 48x   (spatial downsample f = 8)

attention weights (rows sum to 1): ['yellow', 'butterfly', 'purple', 'flower']
  butterfly region [0.4404 0.4404 0.0596 0.0596]
  flower region    [0.0596 0.0596 0.4404 0.4404]
  background       [0.25 0.25 0.25 0.25]
row sums: [1. 1. 1.]

context written into the latent (A @ V):
 [[0.4404 0.4404 0.0596 0.0596]
 [0.0596 0.0596 0.4404 0.4404]
 [0.25   0.25   0.25   0.25  ]]
```

The first block reproduces the deck's 786,432 and supplies the 48× it omits. The second reproduces page 9's table as actual weights: the butterfly position gives 88% of its attention to "yellow" and "butterfly", the flower position gives 88% to "purple" and "flower", and the background spreads uniformly. The `row sums` line is the check worth running on any attention implementation — **the softmax is over the text axis, so each latent position's weights must sum to 1**; if your rows sum to 77 you softmaxed the wrong axis.

The last block shows why the two matrices are identical here: with $\mathbf{V} = \mathbf{I}$, the context vector handed back to the U-Net *is* the attention distribution. In a real model $\mathbf{V}$ is a learned projection of CLIP embeddings, so the output is a blend of word *meanings*, not a list of weights.

## Exam pack

### Must-memorise

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

### Numbers worth knowing

| Quantity | Value |
|---|---|
| $512\times512\times3$ | $786{,}432$ |
| $64\times64\times4$ | $16{,}384$ |
| Compression ratio | $\mathbf{48\times}$ (would be $64\times$ with a 3-channel latent) |
| Spatial downsample factor $f$ | $8$ — three stride-2 stages, $512\to256\to128\to64$ |
| Spatial positions, pixel vs latent | $262{,}144$ vs $4{,}096$ (ratio 64) |
| Self-attention cost ratio, pixel vs latent | $64^2 = \mathbf{4{,}096\times}$ |
| CLIP ViT-L/14 text output | $77$ tokens × $768$ dimensions |
| Cross-attention matrix at $64\times64$ | $4{,}096\times77 = 315{,}392$ entries |
| Cross-attention matrix at the $8\times8$ bottleneck | $64\times77 = 4{,}928$ entries |
| Cross-attention blocks in the deck's U-Net diagram | 4 (two down, two up) |
| N5: attention on the two correct words | $2\times0.440399 = 88.08\%$ |
| N6: share of sampling cost in the denoising loop | $86.6\%$ |
| Components trained by this lecture's procedure | **1** of 4 (the U-Net) |
| Paper and venue | Rombach et al., CVPR **2022**; samples shown from a **1.45B** parameter LAION model |

### Likely MCQ traps

- **"Stable Diffusion runs diffusion on the image and then compresses it."** Backwards. Compress **first** ($\mathbf{z}=\mathcal{E}(\mathbf{x})$), diffuse in latent space, decode **last**. The model enters pixel space exactly twice: encoder at training, decoder at the end of sampling.
- **"The autoencoder and the U-Net are trained together."** No. Page 5 is explicit: the autoencoder is trained **before** diffusion and then **frozen**. Two stages, never joint.
- **Getting $\mathbf{Q}$/$\mathbf{K}$/$\mathbf{V}$ the wrong way round.** $\mathbf{Q}$ comes from the **image side** (U-Net features); $\mathbf{K}$ and $\mathbf{V}$ come from the **text side**. One from the image, two from the text. Swapping them would have the text asking the image what it should mean.
- **"Text is concatenated to the latent."** That is Method 1, and it is for **spatial** conditions. Text uses **cross-attention** because it has no spatial dimensions. The deck gives this as a clean rule and it is the most likely single MCQ on page 14.
- **Compression ratio confusion.** $48\times$ with 4 latent channels; $64\times$ is the purely *spatial* reduction $(512/64)^2$. If a question says "the number of values falls by a factor of 64", it has ignored the channel change.
- **"The latent is 64×64×3."** Stable Diffusion v1's latent has **4** channels. The deck never says so, which is exactly why it is worth knowing.
- **"$\sqrt{d}$ is there to normalise the probabilities."** No — the **softmax** normalises. $\sqrt{d}$ rescales the dot products so the softmax does not saturate; it is a variance fix, not a probability fix.
- **Softmaxing the wrong axis.** The softmax runs over the **text tokens**, so each latent position's weights sum to 1. Rows, not columns.
- **"Cross-attention is applied once, at the bottleneck."** Page 11 says "at several resolutions", and the diagram shows four blocks. One injection point would give composition control without appearance control.
- **"LDM and Stable Diffusion are the same thing."** LDM is the algorithm/paper; Stable Diffusion is one open-source text-to-image implementation of it.
- **$\phi$ overload.** On page 8 $\phi_i(\mathbf{z}_t)$ is the U-Net's **intermediate feature map**. In [Lec 50](50-classifier-guidance.md) $\phi$ was classifier parameters, in [Lec 51](51-classifier-free-guidance.md) the similar glyph $\varnothing$ was the null token, and in [Lec 21](21-vae-encoder.md) $\phi$ was the VAE encoder. Four meanings; read it from context.
- **"Stable Diffusion uses classifier guidance."** It uses **classifier-free** guidance ([Lec 51](51-classifier-free-guidance.md)) — a classifier over free-form prompts does not exist. That is also why each sampling step costs two U-Net passes.
- **Assuming the latent is a sampled VAE latent.** The encoder's KL term is weighted so weakly that the latent behaves as a deterministic autoencoder code; the diffusion model operates on $\mathcal{E}(\mathbf{x})$ directly. See *Beyond the slides*.

### Self-test

1. Compute the number of values in a $512\times512\times3$ image and in a $64\times64\times4$ latent, and the ratio.
2. Which of the four components does the diffusion training stage actually update?
3. Where exactly does the text enter the model, and in what form?
4. In cross-attention, which of $\mathbf{Q},\mathbf{K},\mathbf{V}$ comes from the latent and which from the text? Why that way round?
5. Write the latent-diffusion training loss and say what each symbol is.
6. A segmentation map is used as the condition. Which injection method, and why?
7. Give the shape of the cross-attention matrix for a $32\times32$ feature map and 77 text tokens, and say which axis the softmax runs over.
8. State the difference between a Latent Diffusion Model and Stable Diffusion.
9. Why is the expensive full-resolution decoder not a problem for inference cost?
10. Self-attention at $512\times512$ against self-attention at $64\times64$: by what factor does the cost fall, and why is that factor not 64?

<details><summary>Answers</summary>

1. $512\times512\times3 = 786{,}432$; $64\times64\times4 = 16{,}384$; ratio $= 48$. (Spatially $8^2=64$, times $3/4$ for the channel change.)
2. **Only the U-Net denoiser $\epsilon_\theta$.** The VAE encoder and decoder were trained in a previous stage and are frozen; the CLIP text encoder is pretrained and frozen.
3. Through **cross-attention layers inside the U-Net**, at several resolutions, as the embedding $\mathbf{c} = \tau_\theta(y)$ produced by a pretrained CLIP text encoder (77 token vectors for SD v1). It is not an input to the first layer and it is not concatenated with the latent.
4. $\mathbf{Q}$ from the latent (the U-Net's intermediate feature map $\phi_i(\mathbf{z}_t)$); $\mathbf{K}$ and $\mathbf{V}$ from the text embedding $\tau_\theta(y)$. That direction makes every *image position* the one asking the question — "which words apply to me?" — and the text the thing being looked up, which is what lets different regions attend to different words.
5. $\mathcal{L} = \mathbb{E}[\lVert\epsilon-\epsilon_\theta(\mathbf{z}_t,t)\rVert^2]$. $\mathbf{z}_t$ is the noisy latent, $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ the noise actually added by the forward process, $\epsilon_\theta$ the U-Net's prediction. It is [Lec 47](47-ddpm-reverse.md)'s simplified DDPM loss with $\mathbf{z}$ in place of $\mathbf{x}$.
6. **Concatenation** (Method 1). A segmentation map has spatial information that already aligns with the latent grid position-for-position, so it can simply be stacked on as extra channels; no learned alignment is needed.
7. $N = 32\times32 = 1{,}024$, so the matrix is $1{,}024\times77 = 78{,}848$ entries. The softmax runs over the **text axis** (length 77), so each of the 1,024 rows sums to 1.
8. **LDM** is the algorithm — diffusion performed in the latent space of a pretrained autoencoder — introduced by Rombach et al. at CVPR 2022. **Stable Diffusion** is the practical open-source implementation of that algorithm for text-to-image generation.
9. Because it runs **once per image**, after the whole denoising loop, whereas the U-Net runs once (or twice, with CFG) per step for 50+ steps. In N6's accounting the decoder is 13% of the total and the loop is 86.6%.
10. By $64^2 = 4{,}096$. Attention cost scales with the **square** of the number of positions, and the number of positions falls by 64, so the cost falls by $64^2$. The factor is 64 only for costs that scale linearly, such as convolution.

</details>

## Beyond the slides

**Gap: the deck calls it a VAE but never says how its latent differs from [Lec 21](21-vae-encoder.md)'s.**
**Why it matters:** a textbook VAE's encoder outputs $\mu$ and $\log\sigma^2$ and you *sample* $\mathbf{z}$ via the reparameterisation trick of [Lec 23](23-reparameterization.md), with a KL term pulling the posterior toward $\mathcal{N}(\mathbf{0},\mathbf{I})$ ([Lec 22](22-elbo-and-vae-loss.md) owns that closed form — do not re-derive it). Stable Diffusion's autoencoder has the same shape but a **KL weight so small** (around $10^{-6}$) that the regularisation barely bites: the latent is effectively deterministic and is *not* distributed as a standard normal. That is deliberate. If the latent were forced to $\mathcal{N}(\mathbf{0},\mathbf{I})$ you could sample it directly and would not need a diffusion model at all; the whole point is a latent rich enough to need one. The deck's page 5 even calls the block "AutoEncoder", not "VAE", which is the more honest label.

**Gap: the deck never says the latent must be rescaled.**
**Why it matters:** the DDPM schedule of [Lec 46](46-ddpm-forward.md) assumes data of roughly unit variance — that is what makes $\bar\alpha_t$ interpolate sensibly between signal and noise. The raw encoder output is not unit-variance, so real implementations multiply by a scaling factor (0.18215 in Stable Diffusion v1) on the way in and divide on the way out. Get it wrong and the forward process either destroys the signal immediately or never destroys it. It is one constant, it appears in every implementation, and nothing in the deck hints at it.

**Gap: cross-attention is presented, but not the fact that it is a bag of concepts.**
**Why it matters:** the attention weights in N5 are a soft assignment with no grammar in it. Nothing in $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})$ encodes that "yellow" modifies "butterfly" rather than "flower", or that "two cats" means a count. This is the mechanism behind the model's best-known failures — attribute binding, counting, negation, and the misspelt street sign on page 16 — and the reason later systems replaced or augmented CLIP with a language model that carries more structure. If asked "what can cross-attention conditioning not do", the answer is *bind attributes reliably*, and the reason is that the weights are a similarity lookup, not a parse.

**Gap: the deck shows CFG nowhere, although Stable Diffusion cannot work without it.**
**Why it matters:** every sample on page 16 was generated with [Lec 51](51-classifier-free-guidance.md)'s classifier-free guidance, at a scale around 7.5. Set $w=1$ — plain conditional sampling — and Stable Diffusion's output is recognisably prompt-related but washed out and loosely obedient. The two lectures are halves of one system: Lec 51 supplies the steering, Lec 52 supplies the space to steer in. It also explains the doubled per-step cost in N6, which no slide in this deck accounts for.

**Gap: no mention of what the latent move costs you.**
**Why it matters:** the compression is lossy and the loss is permanent. Whatever the encoder discards, no amount of diffusion can recover — the decoder is a fixed function and the best possible output is $\mathcal{D}(\mathcal{E}(\mathbf{x}))$, not $\mathbf{x}$. This shows up as a characteristic inability to render small faces, fine text and regular high-frequency patterns, and it is why later versions ship improved autoencoders rather than only bigger U-Nets. Framed in this course's language: the model inherits the reconstruction floor of [Lec 11](11-reconstruction-loss.md), and the bottleneck argument of [Lec 10](10-autoencoder-intro.md) applies — a code that is 48× smaller *must* have thrown something away.

## Cut from the slides

Pages 1, 2, 17 and 18 are the title card, the contents list, a bare "Summary" title card with no summary on it, and the "Next Session: Foundations of NLP" card — nothing lost. Pages 10, 11, 12 and 14 all reproduce the *same* architecture figure from Rombach et al. with a different block outlined in red dashes — pixel space, latent space, conditioning, and the switch/concat legend respectively. Only pages 11 and 14 are embedded, since the diagram is identical in all four and re-showing it would spend four figures on one picture; pages 10 and 12 are given in prose instead, with their captions quoted. Page 10's "Pixel Space" box adds only $\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$, already given on page 5, so it is not restated. The U-Net's internal structure, its skip connections and its timestep embedding are visible in the architecture figure but belong to [Lec 49](49-unet.md); the decoder's transposed convolutions belong to [Lec 12](12-autoencoder-types.md); the forward and reverse processes belong to [Lec 46](46-ddpm-forward.md) and [Lec 47](47-ddpm-reverse.md); and self-attention, $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ and the softmax scaling belong to [Lec 57](57-transformer-encoder.md) — each is linked rather than re-taught, and only the cross-attention specialisation is developed here. Page 13's three example images are embedded as one figure; the individual photographs carry no information beyond the three modalities. Page 16's sample grid is embedded for the street-sign observation. Everything else on pages 3 through 16 is reproduced in full.
