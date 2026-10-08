# Week Summaries

The one-paragraph version of every lecture — its **Why this lecture exists** section. Use this to rebuild the thread of the course quickly, or to decide what to re-read.

### [Lec 01 — Introduction to Generative AI](../notes/01-intro-generative-ai.md)

Everything you have been taught to build so far answers one question: *given this input, which output?* A classifier, a regressor, a time-series forecaster — all of them learn $P(y\mid x)$ and stop there. That model can tell a cat from a dog but cannot draw either.

This lecture performs the pivot the rest of the course depends on. It replaces the target $P(y\mid x)$ with $P(x)$ — the distribution the data itself came from — and shows that once you have $P(x)$ you can *sample* from it, which is what "generating" means. It then fixes the vocabulary used for the next sixty lectures: the three generation modes (unsupervised, joint, conditional), the three output distributions (Gaussian, Bernoulli, categorical) and the rule that an output distribution is never written down in code but *implied* by an activation plus a loss. Get that rule now and Lec 02 onward is bookkeeping.

### [Lec 02 — Activation and Loss Functions](../notes/02-activations-and-losses.md)

[Lec 01](../notes/01-intro-generative-ai.md) ended on a rule it did not explain: you never write "Gaussian" or "Bernoulli" in code — you write an **output activation** and a **loss**, and those two together declare the distribution. This lecture supplies both halves of that pair.

It is also the only place in the course where the activation catalogue is laid out properly. Every architecture that follows picks from this list and assumes you know why: CNNs and autoencoders use ReLU, Transformers use GELU, diffusion U-Nets use SiLU, a GAN discriminator ends in a sigmoid, an LLM ends in a softmax. Those are not arbitrary brand preferences, and the reason in each case is a property of the function's **derivative** — which the slides show tables of values for but never differentiate. Supplying the derivatives is what turns this from a lookup table into something you can reason with.

### [Lec 03 — Optimizers, Part A](../notes/03-optimizers-a.md)

Lec 02 gave you a loss — a single number saying how wrong the network is. It did not say what to *do* with that number. Backpropagation turns the loss into a gradient $\partial\mathcal{L}/\partial W$ for every weight; the **optimizer** is the rule that turns those gradients into actual new weight values. That rule is the entire difference between a network that converges in ten epochs and one that oscillates forever.

This lecture builds that rule from nothing. It starts with the one-line update $W_{\text{new}} = W_{\text{old}} - \eta\,\partial\mathcal{L}/\partial W$, shows by arithmetic why the minus sign is there, then asks the two questions that generate every optimizer in this course: *how much data do you look at before each update?* (batch, stochastic, mini-batch) and *should the step depend only on the current gradient?* (momentum). [Lec 04](../notes/04-optimizers-b.md) asks the third — *should every weight share one learning rate?* — and answers it with Adam.

### [Lec 04 — Optimizers, Part B](../notes/04-optimizers-b.md)

[Lec 03](../notes/03-optimizers-a.md) left one hyperparameter unexamined: $\eta$. It appeared in every update, was set to $0.1$ every time, and nothing explained how that number was chosen or what happens when it is wrong. This lecture opens with six side-by-side cases that answer the first question arithmetically, then uses them to expose a defect that no single value of $\eta$ can fix.

The defect is this: a real network has millions of weights, and they do not all receive gradients of the same size. One $\eta$ shared by all of them is simultaneously too big for the busy weights and too small for the rare ones. The fix is to give every weight its *own* learning rate, derived from its own gradient history. That idea produces AdaGrad, then RMSProp, then — combined with Lec 03's momentum — **Adam**, the optimizer that trains essentially every model in the rest of this course.

### [Lec 05 — Convolutional Neural Network, Part A](../notes/05-cnn-a.md)

Everything you have met so far treats an input as a flat list of numbers. An image is not a flat list. It is a grid, and the fact that pixel $(4,7)$ sits beside pixel $(4,8)$ is *information*. Flatten it and that information is destroyed before training even starts.

This lecture fixes that. It replaces the dense layer's "every input touches every neuron" with a small window that slides across the grid, reusing one set of weights everywhere. That change kills the parameter explosion, preserves spatial structure, and gives you feature detectors for free. The price is that you must now track *geometry*: how big is the output, and how does it move with kernel size, stride and padding? This chapter owns that arithmetic for the whole book — every later chapter that stacks convolutions ([U-Net](../notes/49-unet.md), the DCGAN family, [Stable Diffusion](../notes/52-stable-diffusion.md)) links back here rather than restating it.

### [Lec 06 — Convolutional Neural Network, Part B](../notes/06-cnn-b.md)

[Lec 05](../notes/05-cnn-a.md) gave you one layer: a filter slides, multiplies, sums, and a feature map comes out. A feature map is not a prediction. Something has to squash the negatives, shrink the maps so the next layer is affordable, and eventually turn a three-dimensional block of numbers into a vector of class scores.

This lecture supplies the other four components — ReLU, pooling, flatten, fully connected — and then assembles them into the repeating **conv → ReLU → pool** block that is the architecture of every classical CNN. It also delivers the two numbers an exam will actually ask you for: the **parameter count** of a convolutional layer, and the **output shape** of a layer stack. Those two, plus Lec 05's output-size formula, are the complete numerical content of the CNN topic in this course.

### [Lec 10 — Introduction to Autoencoders](../notes/10-autoencoder-intro.md)

Everything in Week 1 was supervised. You had an input $\mathbf{X}$ and a label $\mathbf{Y}$, and the network learned the map between them. That arrangement has a hard dependency: somebody had to label the data. Most data is not labelled, and labelling is the expensive part.

This lecture opens Week 2 by removing the label entirely. If you only have $\mathbf{X}$, what can a neural network still learn? The answer this course builds on for the next six lectures is: make the network reconstruct its own input through a deliberately narrow channel. Nothing is given away by that target — you already have $\mathbf{X}$ — yet the narrowness forces the network to find structure, because a narrow channel cannot carry everything. That model is the **autoencoder**, and it is the architectural ancestor of the VAE, and through the VAE of Stable Diffusion's latent space. This lecture gives you its three parts, its two objectives, and the single dimension choice that decides whether it learns anything at all.

### [Lec 11 — Reconstruction Loss (MSE, Binary Cross-Entropy)](../notes/11-reconstruction-loss.md)

Lec 10 gave you the autoencoder's shape: squeeze the input through a bottleneck, push it back out, demand that what comes out matches what went in. "Matches" was left vague. This lecture makes it a number you can differentiate.

The whole of autoencoder training is one question — *how wrong is $\hat{\mathbf{x}}$ compared to $\mathbf{x}$?* — and the answer depends entirely on what kind of data you have. Real-valued pixels and binary pixels want different loss functions, and choosing wrong costs you training speed and sometimes convergence. This lecture derives both choices, shows the decoder activation each one forces, and reduces the whole linear autoencoder to a single compact objective $\min_W \|\mathbf{X} - \mathbf{X}\mathbf{W}\mathbf{W}^\top\|^2$ that will reappear in Lec 16 when you discover why this model cannot generate anything.

### [Lec 12 — Types of Autoencoders](../notes/12-autoencoder-types.md)

Lec 10 gave you one autoencoder: an input, a narrow middle, an output. Lec 11 gave you the number it minimises. Neither said how many layers to use, or what to do when the input is an image rather than a row of a spreadsheet — and the answer to the second question is not "flatten it", because flattening throws away the fact that neighbouring pixels are neighbours.

This lecture is the deck's own "architecture and code-level understanding" session. It builds three concrete autoencoders — **shallow**, **deep** and **convolutional** — in Keras, with the layer sizes written out, and it spends over a third of its pages on the one operation none of the earlier lectures covered: how a decoder makes an image *bigger*. Convolution and pooling only shrink. Getting from a $7\times7$ feature map back to $28\times28$ needs upsampling or transposed convolution, and the choice between them has a visible consequence — the checkerboard artifact. That machinery reappears in the U-Net ([Lec 49](../notes/49-unet.md)) and in Stable Diffusion's decoder ([Lec 52](../notes/52-stable-diffusion.md)), so it is worth the pages.

### [Lec 13 — Denoising Autoencoders](../notes/13-denoising-ae.md)

Lec 10 and 12 gave the autoencoder a bottleneck and relied on it: make the code narrower than the input and the model *has* to compress. That argument collapses the moment the hidden layer is wide. An **overcomplete** autoencoder — more hidden units than input features — can route each input straight through to the output and score a perfect reconstruction loss while learning nothing at all. The deck calls this the **trivial identity mapping**, and it is overfitting in its purest form.

This lecture gives the first of three fixes. Rather than shrinking the model, you damage the data: feed the network a deliberately corrupted copy of the input and still demand the *clean* original at the output. Copying is now the wrong answer, so the network must use whatever structure survives the corruption — chiefly, what each pixel's neighbours say about it. The payoff is a code that is robust rather than memorised.

### [Lec 14 — Sparse Autoencoders](../notes/14-sparse-ae.md)

[Lec 13](../notes/13-denoising-ae.md) broke the identity shortcut by damaging the input. This lecture breaks it a second way, without touching the data at all: leave the hidden layer as wide as you like, but forbid most of it from switching on.

The motivating observation is that a narrow bottleneck is a very blunt instrument. Forcing $\dim(\mathbf{h}) < \dim(\mathbf{x})$ limits *how many* numbers the code may use, which also limits how expressive the features can be. A sparse autoencoder separates those two things. It keeps a wide hidden layer — hundreds of possible features — but requires that any single input activates only a handful of them. Capacity stays high; usage stays low. The constraint is imposed by adding a **sparsity penalty** to the loss, measured by how far each neuron's *average* activation across the training set has drifted from a small target $\rho$. That penalty is a KL divergence, which is the first appearance in this course of the quantity the whole VAE arc is built on.

### [Lec 15 — Contractive Autoencoders](../notes/15-contractive-ae.md)

[Lec 13](../notes/13-denoising-ae.md) made the code robust by showing the network damaged inputs. That works, but it is indirect: you hope that training on noisy copies teaches the encoder to ignore noise. The contractive autoencoder asks for the same thing directly. Instead of corrupting the data and letting the loss sort it out, it writes "be insensitive to small input changes" as an explicit, differentiable term in the objective.

The term is built from the **Jacobian** — the full table of partial derivatives of every hidden unit with respect to every input feature. A large entry means a hidden unit twitches when one input nudges; penalising the sum of the squares of all those entries forces the encoder to flatten out. The result is a code that *contracts* a neighbourhood of inputs to nearly a single point, which is where the name comes from. This is the last of Week 2's three regularisers, so it also carries the table that tells them apart.

### [Lec 16 — Numerical Example, Limitations of AE](../notes/16-ae-numerical-and-limits.md)

Six lectures of autoencoder theory have given you matrices, losses and three regularisers, but you have never once watched a number travel from the input layer to the output layer. This lecture does nothing else: it takes one 5-dimensional vector, pushes it through a $5\to4\to3\to4\to5$ network by hand, and lands on a reconstruction error. Every multiplication is on the slide. If you can redo this page without the slide, you can answer any autoencoder numerical an exam can set.

Then the lecture turns and knocks the whole model down. A low reconstruction error does not mean the latent code is meaningful; the code is not guaranteed useful for any downstream task; and — the sentence the rest of the course is built on — **the model cannot generate new samples**. The four limitations on pages 9 and 10 are the reason Lec 19 through Lec 24 exist.

### [Lec 19 — KL Divergence, Part A](../notes/19-kl-divergence-a.md)

[Lec 16](../notes/16-ae-numerical-and-limits.md) ended with a diagnosis and a prescription. The diagnosis: an autoencoder's latent space has no structure, so there is no distribution to sample from and the model cannot generate. The prescription, in the deck's own words, was to add "a KL divergence regularization term to the loss function" — a penalty that measures how far the encoder's output distribution has drifted from a prior you *can* sample from.

That penalty is a distance between two probability distributions, and distance between distributions is not a thing you already know how to compute. Euclidean distance works on *points*; a distribution is not a point, and the two distributions you care about may not even live on the same support. This lecture builds the right tool from scratch: what it is, which direction you are measuring in, and what each direction does to a model that cannot fit the target exactly.

**This chapter owns the definition of KL divergence for the whole book.** [Lec 14](../notes/14-sparse-ae.md), [Lec 20](../notes/20-kl-divergence-b.md), [Lec 22](../notes/22-elbo-and-vae-loss.md), [Lec 24](../notes/24-vae-numerical.md), [Lec 50](../notes/50-classifier-guidance.md) and [Lec 51](../notes/51-classifier-free-guidance.md) all link back here rather than redefining it.

### [Lec 20 — KL Divergence, Part B](../notes/20-kl-divergence-b.md)

[Lec 19](../notes/19-kl-divergence-a.md) handed you $\sum_x P(x)\log\frac{P(x)}{Q(x)}$ and showed you what it does, but never said where the formula comes from. Why a logarithm? Why that weight? Why a ratio? Without an answer those are three arbitrary choices you have to memorise, and memorised formulas do not survive an exam that rearranges them.

This lecture derives KL divergence from scratch out of one idea — **surprise** — in four steps: the surprise of a single outcome (self-information), the average surprise of a distribution (entropy), the average surprise when you use the wrong distribution to predict (cross-entropy), and the difference between the last two, which *is* the KL divergence. The payoff is the identity $D_{\mathrm{KL}}(P\,\|\,Q) = H(P,Q) - H(P)$, which makes KL mean something concrete: **the extra bits you waste by modelling with $Q$ when the truth is $P$.** It also quietly explains why [Lec 11](../notes/11-reconstruction-loss.md)'s cross-entropy loss has the shape it does.

The deck works in $\log_2$ throughout, so every number in it is in **bits**.

### [Lec 21 — Introduction to VAE and the Encoder](../notes/21-vae-encoder.md)

Lec 16 ended with a verdict: an autoencoder reconstructs, it does not generate. Feed it a latent vector it has never seen and you get garbage, because nothing in its training ever told the latent space what shape to be. This lecture names the fix in one move — **stop emitting a point, emit a distribution** — and then builds the encoder that does it.

The deck's own one-line statement of the motive is worth memorising: *"Autoencoder learned reconstruction. VAE is introduced because we want generation."* Everything in this chapter follows from taking that seriously. If you want to sample new data, you need a latent space you can sample from, which means you need a known distribution over latents, which means the encoder must produce distributions rather than codes. The lecture stops at the point where you discover that the distribution you actually want is impossible to compute; [Lec 22](../notes/22-elbo-and-vae-loss.md) picks up there.

### [Lec 22 — Probabilistic Decoder, ELBO, VAE Loss](../notes/22-elbo-and-vae-loss.md)

[Lec 21](../notes/21-vae-encoder.md) left you stuck. The encoder wants the true posterior $p(\mathbf{z}\mid\mathbf{x})$; computing it needs the evidence $p(\mathbf{x}) = \int p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$; that integral is intractable; so the encoder settles for an approximation $q_\phi(\mathbf{z}\mid\mathbf{x})$. What nobody has said yet is how you *train* an approximation to something you cannot compute.

This lecture answers that, and it is the mathematical pivot of the whole course. The trick is to stop chasing $\log p_\theta(\mathbf{x})$ and instead maximise a quantity that is always **below** it and is computable — the Evidence Lower Bound. Out of that single move falls the entire VAE loss: one term that is literally the reconstruction loss of [Lec 11](../notes/11-reconstruction-loss.md), and one term that is the KL divergence of [Lec 19](../notes/19-kl-divergence-a.md) between the encoder and the prior. The same bound reappears, barely disguised, in the diffusion derivation in Week 7.

### [Lec 23 — The Reparameterization Trick](../notes/23-reparameterization.md)

[Lec 22](../notes/22-elbo-and-vae-loss.md) finished the VAE's loss function, and the model looked ready to train. It is not. There is a sampling step buried in the middle of the forward pass — the latent vector $\mathbf{z}$ is *drawn* from $q_\phi(\mathbf{z}\mid\mathbf{x})$ — and backpropagation cannot cross it. Gradients flow happily from the loss back to $\mathbf{z}$, and then they stop, because $\mathbf{z}$ was produced by a random number generator rather than by an arithmetic expression in $\mu$ and $\sigma$. The encoder would never receive a gradient and would never learn.

This lecture fixes it with one line of algebra: $\mathbf{z} = \mu + \sigma \odot \epsilon$. The randomness is pushed out into a parameter-free variable $\epsilon$, and $\mu$ and $\sigma$ are left on an ordinary differentiable path. That rewrite is what makes the VAE trainable at all, and it reappears in diffusion models in Week 7. It is the single most examinable idea in Week 3.

### [Lec 24 — VAE Numerical Example](../notes/24-vae-numerical.md)

Weeks 2 and 3 have built the VAE one abstraction at a time: a probabilistic encoder, an intractable posterior, an ELBO, a two-term loss, a trick to get gradients past a sampler. Nothing so far has produced a number.

This lecture pushes one real input — a four-feature vector — through a complete VAE with explicit weight matrices, and computes every intermediate value to four or five decimal places: the hidden activations, $\mu$, $\log\sigma^2$, $\sigma$, three different reparameterized samples $\mathbf{z}$, the decoder's hidden layer, the reconstruction $\hat{\mathbf{x}}$, the reconstruction loss, the KL term and the total. It is the most directly examinable deck in Week 3, because an NPTEL numerical question is almost certainly one slice of exactly this pipeline. The slides stop at the loss value; this chapter carries it one step further and takes a gradient step, which is what the loss was for.

### [Lec 26 — Disentanglement and β-VAE](../notes/26-beta-vae.md)

[Lec 22](../notes/22-elbo-and-vae-loss.md) gave you a VAE that reconstructs well and samples legally. It never promised the latent vector would be *readable*. Train a VAE on MNIST with a 4-dimensional code and you get four numbers that jointly determine the digit — but no single one of them is "thickness" or "slant". Each is a smear of everything.

That costs you control. If you want a generative model you can steer — thicker stroke, same digit; more smile, same face — you need the code's axes to line up with the factors a human would name. This lecture names the failure (**entanglement**), names the goal (**disentanglement**), and gives you the one-character fix: put a weight $\beta$ in front of the KL term. Everything else about the VAE stays exactly as it was. The rest of the lecture is about what that one character costs you.

### [Lec 27 — Conditional VAE](../notes/27-conditional-vae.md)

A trained VAE can generate digits. It cannot generate a **5**. Sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, decode it, and you get whatever that region of the latent space happens to mean — a 3, a 9, something in between. You have a generator with no steering wheel.

[Lec 26](../notes/26-beta-vae.md) attacked this by hoping a latent axis would learn to mean something, and paid for it in reconstruction quality. This lecture takes the direct route: if you already know the label at training time, **give it to the model**. Feed the label into the encoder alongside the image, and into the decoder alongside the latent. At generation time you choose the label, and the decoder must honour it.

The architectural change is a concatenation. The mathematical change is adding $\mathbf{y}$ to the conditioning bar of every distribution in the loss. That really is all of it, and readers consistently expect something harder.

### [Lec 28 — Latent Space Interpolation](../notes/28-latent-interpolation.md)

Take two images, encode them to two points in the latent space, walk in a straight line from one point to the other, and decode every point you pass through. If the model is any good you get a smooth morph: a 3 that bends and closes until it is an 8, with every frame a plausible digit.

That walk is the single most convincing demonstration that a VAE learned *structure* rather than a lookup table. It is also the experiment a plain autoencoder fails. [Lec 16](../notes/16-ae-numerical-and-limits.md) told you an autoencoder's latent space has holes — regions no training code ever occupied, where the decoder was never trained and will output noise. This lecture completes that argument from the other side: the VAE's KL term is what fills the holes, and interpolation is how you see that it worked. The arithmetic itself is a weighted average and takes one line.

### [Lec 31 — Motivation for GANs](../notes/31-gan-motivation.md)

You have just built a generative model that works: the VAE encodes, samples, decodes, and its latent space interpolates smoothly. Then you look at what it draws: the faces are soft, the digits smudged, the textures gone. The samples are *plausible* and *unconvincing* at once.

This lecture is the diagnosis. The lecturer lists six things a VAE does badly, most of them trace to one choice: a VAE is trained to maximise an explicitly chosen likelihood, evaluated pixel by pixel, and that objective rewards hedging. Where the model is unsure, the cheapest answer is the average of the possibilities — and the average of several sharp images is a blurry image. The lecture then asks the question the GAN arc answers: can you generate realistic data *without* writing down a probability distribution or a reconstruction loss at all? The answer: let a second network invent the loss.

### [Lec 32 — GAN Architecture](../notes/32-gan-architecture.md)

[Lec 31](../notes/31-gan-motivation.md) left you with a slogan — two networks compete, one forges, one detects — and a four-step loop too loose to implement. This lecture turns it into wiring.

Three questions have to be answered before a GAN is a program rather than an analogy. What exactly does each network take in and emit? Where does the input to the generator come from, given that there is no encoder anywhere? And when both networks share a computation graph, which parameters move on which step? The third is where readers go wrong, because the same forward pass — noise, generator, fake image, discriminator, probability — is run in both halves of the training loop, with different things frozen and *different labels on the identical image*. Get the freezing or the labels backwards and the model trains itself into the ground. This chapter spells the loop out line by line.

### [Lec 33 — GAN Objective and Loss Functions](../notes/33-gan-objective.md)

[Lec 32](../notes/32-gan-architecture.md) gave you two networks and a training loop: a generator that turns noise into a sample, a discriminator that scores samples as real or fake, and an alternating schedule that updates one while the other is frozen. What it did not give you is the number each network is actually trying to move. "Fool the discriminator" is a slogan, not a gradient.

This lecture turns the slogan into one equation. It builds that equation the honest way — from binary cross-entropy on a single real sample, then a single fake sample, then both together, then averaged into expectations — and arrives at the minimax value function that the whole rest of the GAN arc modifies. [Lec 34](../notes/34-gan-convergence.md), [Lec 38](../notes/38-conditional-gan.md), [Lec 39](../notes/39-pix2pix.md), [Lec 40](../notes/40-cyclegan.md), [Lec 41](../notes/41-stylegan.md) and [Lec 42](../notes/42-stylegan2.md) all state *their own* loss and link back here for the base.

**This chapter owns the GAN objective for the whole book.**

### [Lec 34 — GAN Convergence and Nash Equilibrium](../notes/34-gan-convergence.md)

[Lec 33](../notes/33-gan-objective.md) left you with a quantity to optimise and no idea what optimising it achieves. You can compute $V(D,G)$ for any pair of networks, but you cannot yet say which pair is the right answer, what value $V$ takes there, or whether gradient descent will ever find it.

This lecture answers all three, and the third answer is uncomfortable. The solution is $p_g = p_{\text{data}}$ with the discriminator reduced to coin-flipping at $D = \tfrac12$ everywhere. The value there is $-\log 4$. And the solution is a **saddle point** of a two-player game rather than the minimum of anything, which is why GAN training oscillates, why the loss curve tells you nothing, and why the generator can collapse onto a single output and stay there. This chapter owns the optimal discriminator, the Nash-equilibrium framing, **mode collapse** and **training instability** for the whole book.

### [Lec 38 — Conditional GAN](../notes/38-conditional-gan.md)

A GAN trained on MNIST will hand you beautiful digits. It will not hand you a *seven*. You draw $\mathbf{z}$, you get whatever that particular $\mathbf{z}$ happens to decode to, and nothing in the model tells you which $\mathbf{z}$ produces which digit. For a lab demo that is charming; for anything useful it is fatal.

The fix is almost embarrassingly small: feed the class label into the network alongside the noise. The lecture's real content is the part that is *not* obvious — that the label must go into the **discriminator** as well, or the generator is never penalised for ignoring it. Everything in Week 6 is built on this one modification. Pix2Pix, CycleGAN and classifier-free guidance are all conditional generation wearing different clothes.

### [Lec 39 — Image-to-Image Translation: Pix2Pix GAN](../notes/39-pix2pix.md)

[Lec 38](../notes/38-conditional-gan.md) left you with a generator that takes a condition. It also left the condition abstract: a ten-way class label on MNIST. This lecture makes the condition an entire image, and the output an entire image — sketch to photograph, map to satellite view, segmentation mask to street scene.

Two things break when you do that. A generator built as an encoder–decoder throws away the spatial detail that an image-to-image task needs preserved, and a discriminator that returns one verdict for a whole $256\times256$ image is both enormous and bad at local texture. Pix2Pix fixes the first with a **U-Net** and the second with a **PatchGAN**, then adds an $L_1$ term to the generator's loss to pin the output to the ground truth. Each of those three choices is separately examinable.

### [Lec 40 — CycleGAN](../notes/40-cyclegan.md)

Pix2Pix can turn a sketch into a handbag, but only because somebody photographed the same handbag twice — once as an edge map, once in colour. Every training example is a *pair*, and the L1 term in its loss compares the generator's output against *that specific* target image. Remove the pairing and Pix2Pix has nothing to compute.

That is a crippling restriction. There is no photograph of a particular horse wearing stripes, no Monet painting of a specific Yosemite waterfall, no winter and summer shot of the same instant. For most interesting translations the two domains exist only as two unaligned piles of pictures.

This lecture answers the deck's own question — *can a model learn translation between two domains using only separate image collections, without paired examples?* — and the answer turns out to need one new idea. Drop the pairing and the adversarial loss alone becomes hopelessly underdetermined. **Cycle consistency** is what puts the determination back.

### [Lec 41 — StyleGAN](../notes/41-stylegan.md)

Every GAN so far has had one input: a noise vector $\mathbf{z}$, shoved into the first layer of the generator. It works — the pictures are sharp — but you have no handle on anything. Nudge $z_3$ and the face changes age, hair colour and head angle at once. You cannot ask for "the same person, smiling", because no coordinate means "smiling".

[Lec 26](../notes/26-beta-vae.md) named that failure — **entanglement** — and attacked it inside the VAE by reweighting the KL term. StyleGAN attacks the same failure from the opposite end, and its diagnosis is sharper: the latent space is entangled because **you forced it to be**. The distribution you sample $\mathbf{z}$ from is fixed in advance, so $\mathcal{Z}$'s geometry has no freedom to match the data's factor structure.

The fix is almost embarrassingly simple: stop feeding $\mathbf{z}$ to the generator. Pass it through a small network first, into a second latent space nobody constrained, and feed *that* to the image-maker — not once at the bottom, but at every layer, as a style.

### [Lec 42 — StyleGAN 2](../notes/42-stylegan2.md)

StyleGAN produced the best faces anyone had seen, and every single one of them had a bubble in it. Not always visible in the final image — but look at the activations inside the generator and the blob is there, in every feature map, from the 64×64 stage onward. The deck's own words: *"It is a systemic problem that plagues all StyleGAN images."*

This lecture is a forensic exercise. It takes two specific visual defects — the **water-droplet artefact** and the **phase artefact** — traces each back to a named architectural choice, and replaces that choice. The droplet traces to AdaIN; the fix is weight demodulation. The phase artefact traces to progressive growing; the fix is a fixed-size network with skip and residual connections. Nothing else about StyleGAN changes. That is what makes it worth studying: it is the clearest worked example in the course of *diagnosing* a generative model rather than scaling it.

### [Lec 44 — Introduction to Diffusion Models](../notes/44-diffusion-intro.md)

You have already built a diffusion model. In [Lec 13](../notes/13-denoising-ae.md) you took a clean image, deliberately corrupted it with noise, and trained a network to restore it. Corrupt, then restore. That is the entire idea of diffusion — the only thing Week 7 adds is a **schedule**: instead of one fixed noise level, you corrupt across a thousand of them, from barely-touched to pure static, and train one network to undo any single step.

That sounds like more work for the same result. It is not, and the reason is the thread running through this whole lecture. Undoing *all* the corruption at once is a hard problem that no architecture solves cleanly — it is what makes autoencoder outputs bland and VAE outputs blurry. Undoing a 3% perturbation is nearly trivial. Diffusion buys quality by replacing one impossible step with a thousand easy ones, and in doing so it throws away the adversarial game entirely: there is no discriminator, no two-player equilibrium, and therefore no mode collapse.

### [Lec 45 — Mathematical Foundations of Diffusion Models](../notes/45-diffusion-math.md)

[Lec 44](../notes/44-diffusion-intro.md) sold you the picture: destroy an image with noise step by step, then learn to walk the destruction backwards and you have a generator. The picture is not yet a model. Two things have to be nailed down before any of it can be written as code — *how much* noise enters at each step, and *what kind* of distribution the noise comes from.

This lecture supplies both. It names the sequence $\{\beta_1,\beta_2,\ldots,\beta_T\}$ that controls the noise injected per step, argues that getting that sequence wrong breaks training in two opposite ways, and then spends half its pages reviewing the Gaussian distribution — scalar, standard, and multivariate with a covariance matrix — because every single distribution in the next three lectures is Gaussian. It derives nothing about the chain itself. That is deliberate: [Lec 46](../notes/46-ddpm-forward.md) does the forward algebra, [Lec 47](../notes/47-ddpm-reverse.md) the reverse. This lecture is the vocabulary you need to read them.

### [Lec 46 — DDPM: The Forward Process](../notes/46-ddpm-forward.md)

[Lec 45](../notes/45-diffusion-math.md) gave you the schedule $\{\beta_t\}$ and the Gaussian vocabulary, and stopped. This lecture writes the forward process down as an equation and then does the one piece of algebra the whole method rests on.

The problem it solves is practical. Training a diffusion model means showing the network pairs $(\mathbf{x}_t, \epsilon)$ at *randomly chosen* timesteps. If the only way to reach $\mathbf{x}_{500}$ were to run 500 noising steps, every training example would cost 500 operations and training would be hopeless. This lecture proves you can jump straight from $\mathbf{x}_0$ to $\mathbf{x}_t$ for any $t$ in **one** step:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

The lecturer derives this by hand on the slides, by composing two Gaussians and watching their variances add. And when you see the final form, you should recognise it — it is $\mu + \sigma\epsilon$, the reparameterization trick from [Lec 23](../notes/23-reparameterization.md), wearing a diffusion hat.

### [Lec 47 — DDPM: The Reverse Process](../notes/47-ddpm-reverse.md)

[Lec 46](../notes/46-ddpm-forward.md) finished the half of diffusion that requires no learning. You can now destroy an image in one operation and know the exact distribution of the result. None of that generates anything.

This lecture builds the generator. It asks what $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ should look like, discovers that the *exactly computable* answer depends on $\mathbf{x}_0$ — which is precisely the thing you do not have at generation time — and then performs the manoeuvre that makes DDPM work: rewrite the unknown $\mathbf{x}_0$ in terms of the noise that produced $\mathbf{x}_t$, and train a network to predict that noise instead. The target collapses from "an image" to "the $\epsilon$ you added", the loss collapses to a mean squared error, and generation becomes a loop you can write in six lines.

### [Lec 48 — Hands-on: Forward Diffusion](../notes/48-forward-diffusion-handson.md)

[Lec 46](../notes/46-ddpm-forward.md) proved that you can jump straight from a clean image to any noise level in one line of algebra. That proof leaves four practical questions unanswered: what numbers actually go into $\beta_t$, how you store the $\bar\alpha_t$ table so a training loop can index it, what a real image tensor looks like after the jump, and whether the one-shot formula really agrees with grinding the Markov chain forward step by step.

This session answers all four by running code. It is a Colab notebook — "Diffusion Tutorial 1" — not a slide deck, so every claim arrives as a printed number or a rendered image. It also builds a small timestep-conditioned CNN and trains it to predict the noise, which is the first time in the course you see the DDPM objective move a loss. It deliberately stops before reverse sampling; that is [Lec 53](../notes/53-reverse-diffusion-handson.md).

### [Lec 49 — U-Net for Denoising](../notes/49-unet.md)

[Lec 47](../notes/47-ddpm-reverse.md) reduced the whole reverse process to one learned function: given a noisy image $\mathbf{x}_t$ and the timestep $t$, predict the noise $\epsilon$ that was added. [Lec 48](../notes/48-forward-diffusion-handson.md) trained a stand-in — four convolutions in a row — and got it working on MNIST. This lecture supplies the architecture that is actually used.

Two requirements fix the design. The output is a noise map, so it must be **exactly the same size and shape as the input**: $\epsilon_\theta(\mathbf{x}_t,t)$ lives in the same $H\times W\times C$ as $\mathbf{x}_t$. And one network must handle **every** noise level, so $t$ has to enter the computation. The U-Net answers the first with a contracting-then-expanding path tied together by skip connections; sinusoidal time embeddings injected into every block answer the second. Everything else here — residual blocks, group normalisation, self-attention — is in service of those two, and is owned elsewhere in this book.

### [Lec 50 — Classifier-Guided Diffusion](../notes/50-classifier-guidance.md)

A trained DDPM is a machine that turns noise into *a* realistic image. Not *your* image. Feed it fresh Gaussian noise and you get a cat, or a car, or a bird — whatever the training set contained, sampled in proportion to how common it was. The model learned $p(\mathbf{x})$, the distribution of everything, and sampling from $p(\mathbf{x})$ is by definition sampling at random.

That is useless for the thing people actually want to do, which is ask for a dog and get a dog. This lecture adds the first steering mechanism: train a second network, a **classifier** that can read a *noisy* image and say which class it is, and at every reverse step push the denoising trajectory in the direction that makes that classifier more confident. One extra term in the score, one knob $s$ controlling how hard you push.

It works, and it is the method that first beat GANs on ImageNet. It is also clumsy enough that the next lecture deletes it.

### [Lec 51 — Classifier-Free Diffusion Guidance (CFG)](../notes/51-classifier-free-guidance.md)

[Lec 50](../notes/50-classifier-guidance.md) bought you steering at the price of a second network — a classifier trained from scratch on noisy images at every point of the schedule, queried forward and backward at every reverse step. The slides of this lecture open by reprinting that bill, unchanged, as the motivation.

The fix is almost embarrassingly simple. The U-Net can already take a condition $y$ as an input; so train it to accept $y$ *and* to accept nothing, by randomly throwing the label away during training. Then at sampling time run it twice — once with the condition, once without — and **extrapolate** along the line from the unconditional prediction through the conditional one. The difference between the two outputs is the steering signal the classifier used to provide, and it comes free from a model you were training anyway.

This is the method that every text-to-image system you have used actually runs, which is why it gets its own lecture and its own comparison table.

### [Lec 52 — Stable Diffusion (Latent Diffusion Models)](../notes/52-stable-diffusion.md)

Everything in Week 7 and Week 8 so far has run diffusion on pixels. That is why it is slow: a single 512×512 colour image is 786,432 numbers, and the U-Net must process all of them at *every one* of hundreds of denoising steps. Train a model that way and you need a research lab's hardware budget.

This lecture is one structural move that removes the problem. Train an autoencoder first, freeze it, and run the entire diffusion process on its **compressed latent** instead of on pixels. A 512×512×3 image becomes a 64×64×4 latent — 16,384 numbers, 48× fewer — and the U-Net, the schedule, the loss and the sampler are all unchanged except that they now operate on latents. At the end, one pass of the decoder turns the finished latent back into an image.

Add a text encoder and a cross-attention layer so a prompt can steer the U-Net, and you have Stable Diffusion: the model that put image generation on consumer graphics cards.

### [Lec 53 — Hands-on: Reverse Diffusion](../notes/53-reverse-diffusion-handson.md)

[Lec 47](../notes/47-ddpm-reverse.md) gave you $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ and the noise-prediction objective. That is a *distribution*, written once. It is not yet a program. Turning it into a program raises questions the derivation never had to answer: which index does the loop run over, in which direction, what shape does $t$ have when a batch carries a different timestep per image, where exactly does the extra Gaussian $\mathbf{z}$ get added, and what do you do at the very last step where adding it would be wrong.

This lecture is the whole of Week 7 and Week 8 compiled into one Colab notebook: schedule, U-Net, training loop, sampler, guidance sweep. The sampler is the part that is genuinely new, and it is fifteen lines long. Everything else you have already met as mathematics. By the end you should be able to write `p_sample` from memory and say, for any line of it, which symbol in [Lec 47](../notes/47-ddpm-reverse.md)'s derivation it implements.

### [Lec 54 — Foundations of NLP](../notes/54-nlp-foundations.md)

Eight weeks of this course have been about images. An image arrives as a grid of numbers; you can subtract two of them, interpolate between them, add Gaussian noise to them. Language arrives as *characters*. You cannot subtract "king" from "queen", you cannot add noise to "the", and there is no sense in which "cat" is 0.3 of the way from "dog" to "car".

Every architecture in the second half of the course — RNN, LSTM, Transformer, BERT, GPT — assumes its input is already a sequence of vectors. This lecture is the adapter that makes that true. It does two things: **tokenization**, which cuts a string into discrete units, and **embedding**, which maps each unit to a vector in $\mathbb{R}^d$ where arithmetic means something again. Everything after Week 9 is built on top of the output of this lecture, and if you are hazy about which step produces what, every later architecture diagram will be hard to read.

### [Lec 55 — Sequential Modeling with RNNs and LSTMs](../notes/55-rnn-lstm.md)

Lec 54 turned text into numbers: tokenize, embed, and you have a list of vectors. The list is the problem. "Dog bites man" and "Man bites dog" contain identical vectors and mean opposite things, so whatever consumes the list must care about its *order*. A feed-forward network does not — it sees a fixed-size input and has no memory of anything it processed before.

This lecture introduces the one architectural idea that fixes that: a **hidden state** that is written at every timestep and read at the next, so information flows forward through time. It then shows the price. The same hidden state that carries information also carries gradients, and because the identical weight matrix multiplies it at every step, the gradient is raised to a power. Long sequences make that power large, and a number raised to a large power either vanishes or explodes. The LSTM is the repair: a second state with a gated, mostly-additive update path, so the network can *choose* what to keep.

### [Lec 56 — Evolution From LSTMs to Transformers](../notes/56-lstm-to-transformer.md)

[Lec 55](../notes/55-rnn-lstm.md) left you with a working model and an unsolved problem. The LSTM's gates make long-range memory *possible*, but recurrence itself imposes two costs no amount of gating removes. First, $\mathbf{h}_t$ cannot be computed until $\mathbf{h}_{t-1}$ exists, so a 1000-token sentence needs 1000 steps in strict order — a dependency chain that no number of GPUs can shorten. Second, in a translation system the entire source sentence is squeezed into one fixed-width vector before the decoder sees a single word, and that vector is the same size for a five-word sentence and a five-hundred-word one.

This lecture names both costs and then introduces the idea that removes them: **attention**. Instead of forcing the decoder to work from one compressed summary, let it look back at *every* encoder state and weight them differently for every word it produces. That single change — a weighted sum instead of a bottleneck — is what the next three lectures are built on.

### [Lec 57 — Transformer Encoder](../notes/57-transformer-encoder.md)

[Lec 56](../notes/56-lstm-to-transformer.md) ended on a question the slides print in red: *can we remove recurrence entirely?* Recurrence is what makes an LSTM slow — token $t$ cannot be computed until token $t-1$ has been, so a 500-word sentence is 500 sequential steps no GPU can shorten, and information from word 1 must survive 499 hops to reach word 500.

This lecture builds the replacement. Every token gets a representation informed by every other token, and all of them are computed *at once*, by one matrix multiplication. The mechanism is self-attention; its moving parts are three learned projections called Query, Key and Value. Because the new operation is blind to word order, position has to be injected deliberately — the deck spends four of its ten content pages on exactly that. The result is the encoder block: attention, add and normalise, feed-forward, add and normalise, stacked six deep.

Everything in Weeks 10, 11 and 12 is this block, rearranged.

### [Lec 59 — Transformer Decoder](../notes/59-transformer-decoder.md)

[Lec 57](../notes/57-transformer-encoder.md) built a machine that reads. Given $n$ tokens it returns $n$ contextual vectors, all at once, every token free to look at every other. That is exactly the wrong machine for writing. A generator emits one token at a time, and when it is deciding what word comes third it must not be allowed to look at the fourth — the fourth does not exist yet at inference, and if it is visible during training the model learns to copy the answer instead of predicting it.

So the decoder changes two things and keeps everything else. It **masks** its self-attention, so position $t$ sees only positions $1$ through $t$. And it adds a second attention sublayer, **cross-attention**, whose queries come from the decoder but whose keys and values come from the encoder — the join between reading and writing. Three sublayers instead of two, one new matrix of $-\infty$, and the architecture is complete.

### [Lec 60 — BERT (Bidirectional Encoder Representations from Transformers)](../notes/60-bert.md)

[Lec 57](../notes/57-transformer-encoder.md) built an encoder block and [Lec 59](../notes/59-transformer-decoder.md) built a decoder, but neither said what to *train* them on. The original Transformer was trained on translation, which needs paired sentences in two languages — expensive, scarce, and useless for the other hundred things you want a language model to do.

BERT answers a narrower question: if you only want to *understand* text, not produce it, you can throw the decoder away, keep the encoder, and let every word see every other word. That freedom is also the problem. A model that can see the whole sentence cannot be trained to predict the next word, because the next word is already on its input. So bidirectionality forces a different training objective, and most of this lecture is that objective: mask some words out and make the model guess them. The rest is the two-stage habit — pre-train once on raw text, fine-tune cheaply per task — that the whole LLM half of the course now runs on.

### [Lec 61 — GPT (Generative Pre-trained Transformer)](../notes/61-gpt.md)

[Lec 60](../notes/60-bert.md) threw the decoder away and kept the encoder, because understanding text does not require writing any. This lecture does the mirror-image thing: throw the *encoder* away and keep the decoder, because writing text does not require anything else.

That sounds like a smaller idea than it turned out to be. A decoder trained to predict the next token learns grammar, facts, translation, arithmetic and style, all from the same objective and all without a single human label — and, as the models got larger, it stopped needing to be fine-tuned at all. You could just *ask*. The arc from "a 117 M-parameter model you fine-tune per task" to "a model you talk to" is the arc of this lecture and of the rest of the course.

This chapter also owns the **BERT versus GPT contrast**, the discrimination that the whole LLM half of the exam is built on.

### [Lec 62 — Prompt Engineering Basics](../notes/62-prompt-engineering.md)

[Lec 61](../notes/61-gpt.md) left you with a model that can do a task it was never trained on, provided you phrase the task in text. That is a *capability*. It says nothing about what you should actually type.

This lecture is the technique. It names the parts a prompt is built from, grades prompts against five concrete levers, and lays out six named prompting patterns with a worked example each. The thing it never says — and the thing that makes every one of those patterns obvious instead of arbitrary — is what a prompt *mechanically is*: a string of tokens prepended to the model's input, which shifts the probability distribution over the next token. Not a command, not a configuration, not an API call. A prefix. Once you hold that, "why does showing two examples help?" stops being magic and becomes arithmetic, and you can predict which techniques will work without memorising a list.

### [Lec 63 — Hands-on on LLM](../notes/63-llm-handson.md)

Weeks 9 and 10 built a decoder-only transformer out of mathematics: masked self-attention, positional embeddings, a next-token objective, a prompt that conditions the continuation. Every one of those is a formula. None of them is a program, and several questions that the mathematics can ignore become unavoidable the moment you type code — what exactly is the training target, how long is the context, what does the model emit at the final position, and how do you turn one probability vector into one actual token.

This notebook answers them twice: once on a 414-thousand-parameter model you train from scratch in under four seconds, and once on an 82-million-parameter pretrained DistilGPT-2. The genuinely new content is the last part — **decoding**. Greedy, top-$k$ and nucleus sampling, with temperature on top, are the knobs that turn a probability distribution into text, and this is where the course teaches them with running code.

### [Lec 64 — LLM Recap, In-Context Learning, LoRA, and the RAG Principle](../notes/64-llm-icl-lora-rag.md)

You have a pretrained decoder-only model with billions of frozen parameters and a new task it was never trained on. There are exactly three things you can do, and this lecture names all three. You can leave the weights alone and put examples in the prompt — **in-context learning**. You can leave the weights alone and train a tiny bolt-on — **low-rank adaptation**. Or you can leave the weights alone and hand the model facts it does not contain — **retrieval augmented generation**.

Notice what every option has in common: nobody retrains the base model. Full fine-tuning of a 70-billion-parameter model is out of reach for almost everyone, so the entire practical craft of using LLMs is the craft of *not* touching $\mathbf{W}_0$. This lecture is the course's first systematic treatment of that craft, and it is delivered by a different lecturer with different notation, which is itself examinable.

### [Lec 68 — Retrieval Augmented Generation: Mechanics and Advances](../notes/68-rag-advances.md)

[Lec 64](../notes/64-llm-icl-lora-rag.md) left you with a three-box sketch — retrieve, augment, generate — and a reason to want it. Sketches do not pass exams. This lecture is the machinery: what a retriever actually computes, how "relevant" becomes a number you can sort on, what probability a RAG system is really estimating, and what you lose by keeping only the top $k$ documents.

It then goes somewhere the first deck never hints at. Two named variants extend the basic one-shot pipeline in opposite directions: **Agentic RAG** lets the model retrieve repeatedly and decide for itself when it has enough, and **GraphRAG** replaces the flat list of passages with a knowledge graph built by an LLM over the whole corpus. Both are on the slides, both have a comparison table the lecturer clearly wrote for an exam, and neither is in the companion courses.

### [Lec 70–71 — LLMs for Text Generation and Multimodal Alignment](../notes/70-llm-generation-multimodal.md)

Every architecture lecture so far stopped at the softmax. [Lec 59](../notes/59-transformer-decoder.md) built the decoder, [Lec 61](../notes/61-gpt.md) made it causal, and both ended with a probability vector over the vocabulary — and then said nothing about what you *do* with it. That gap is the whole of this lecture.

A probability distribution is not a sentence. Turning one into the other is called **decoding**, and the rule you choose changes the output more visibly than any architectural detail: the same frozen model is a deterministic question-answerer or a chaotic poet depending on one scalar. This lecture makes that scalar numerical. It also closes the input side — what a token actually *is*, and how a picture becomes one — which is the bridge to the multimodal models that [Lec 52](../notes/52-stable-diffusion.md) kept deferring.

### [Lec 72–73 — Size versus Performance, Benchmarks, Bias and Safety](../notes/72-benchmarking-bias-safety.md)

Everything before this point taught you to *build* a generative model. Nothing taught you to *judge* one. The course has reported exactly one number as evidence of quality — the training loss — and a training loss cannot tell you whether a model is useful, whether it is honest, whether it works equally well for everyone, or whether deploying it is safe.

This lecture supplies the missing half. It asks three questions in order. **How much does size actually buy?** — the answer in 2026 is far less than the 2020 scaling literature promised, and for a reason worth understanding. **How do you measure capability at all?** — four families of benchmark, each with a failure mode that invalidates the others' results. And **what breaks when you deploy?** — bias, which enters at five distinct stages, and safety, which this lecturer argues has stopped being a research topic and become an operational one.
