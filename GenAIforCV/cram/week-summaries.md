# Week Summaries

The one-paragraph version of every lecture — its **Why this lecture exists** section. Use this to rebuild the thread of the course quickly, or to decide what to re-read.

## Week 1

### [Lec 1 — Introduction: Computer Vision and Generative AI](../notes/week-01/01-intro-cv-and-genai.md)

This is the first chapter, so nothing precedes it — but something does motivate it. A course called
"Generative AI for Computer Vision" has to answer an obvious objection before it can start: computer
vision already works. Networks classify images better than people do. So why spend twelve weeks
learning to *make* images instead of *reading* them?

The lecture answers that in one move. It shows you a chart of ImageNet error rates falling from 28.2%
to 3.6% in five years, points out that there is almost nothing left to win there, and then names the
thing that is actually scarce: not models, not compute, but **labelled data**. Generation is how you
manufacture the data that supervised learning is starving for. Everything else in the course is
machinery for that claim.

### [Lec 2 — Generative Modelling: Generative vs Discriminative Learning](../notes/week-01/02-generative-vs-discriminative.md)

Lecture 1 told you that generative AI for vision is a field. This lecture tells you what the word
*generative* actually means, and it does so by contrast: there is another way to learn from labelled
data, and almost everything you have met before — the classifier you trained with `sklearn`, logistic
regression, a CNN that outputs "cat" — belongs to that other way. The split between the two is the
single most examinable idea in the course, because every model family in weeks 6 to 12 sits on one
side of it. The lecture also answers a question you should be asking: why bother *making* images at
all, when the point of computer vision is to understand them? The answer is that labelled real data is
expensive, biased, dangerous to collect and legally fraught, and synthetic data fixes all four.

### [Lec 3 — Generative Vision Models: The Landscape](../notes/week-01/03-generative-vision-models.md)

The previous lecture established *what* a generative model is: something that learns $p(\mathbf{x})$
rather than $p(y \mid \mathbf{x})$, so that you can sample new images instead of only labelling
existing ones. That leaves an obvious question hanging — nobody can actually write down the
probability distribution of all 196,608 numbers in a 256×256 colour photograph, so how does anyone
build such a thing in practice?

The answer is that there are five families of trick, and they are genuinely different tricks, not
variations on one. This lecture is the map. It names all five, says what each one's core move is,
and shows where each one shines and where it falls over. Nothing here is derived — every family gets
its own chapter later. Read this so that when Week 6 opens with autoregressive models and Week 8 with
VAEs, you already know which country you are standing in.

### [Lec 4 — Mathematical Preliminaries I: Linear Algebra](../notes/week-01/04-linear-algebra.md)

Everything after this point in the course is linear algebra wearing a costume. A neural network layer
is a matrix multiply. An image is a vector. Attention is a pile of dot products divided by a square
root. A latent space is a low-dimensional vector space you are trying to map onto a high-dimensional
one. If you can multiply matrices and say what an eigenvector is without hesitating, the rest of the
course is about *which* matrices and *why* — which is the interesting part. If you cannot, every
later lecture will feel like memorisation.

The deck is a refresher, so it moves fast and states results without proof. This chapter restates them
with the reasoning attached, because an exam will ask you to *apply* them, not recite them.

### [Lec 5 — Mathematical Preliminaries II: Basic Probability I](../notes/week-01/05-probability-1.md)

Linear algebra gave you the containers — vectors, matrices, the spaces data lives in. It cannot tell
you which points in those spaces are *likely*. That is the entire job of a generative model: not to
draw a boundary between cats and dogs, but to say how probable any particular image is, and then to
sample new ones from that judgement. Every object in the second half of this course is a probability
distribution wearing a neural network. $p(\mathbf{x})$, $p(y \mid \mathbf{x})$,
$q_\phi(\mathbf{z} \mid \mathbf{x})$, $p(\mathbf{z})$ — you will not survive a single slide of the VAE
lectures if the vertical bar in those expressions is not completely automatic.

The deck is a refresher and states results without motivation. This chapter supplies the motivation,
because the exam asks you to compute with these rules, not recite them.

### [Lec 6 — Mathematical Preliminaries II: Basic Probability II](../notes/week-01/06-probability-2.md)

Lec 5 gave you one random variable at a time: its PMF, its PDF, its mean, its variance. That is not
enough to do anything. An image is not one random variable, it is hundreds of thousands of them, all
tangled together, and a *generative model* is precisely a claim about how they are tangled. This
lecture supplies the two things you need to make that claim. First, the language of several random
variables at once — joint, marginal, chain rule, covariance — which is how you say "these pixels are
related". Second, the machinery for *fitting* such a claim to data: likelihood, maximum likelihood,
and the Bayesian alternatives MAP and the Bayes optimal classifier. Every objective function in the
remaining 24 lectures is one of these two things wearing a different name.

## Week 2

### [Lec 7 — Neural Network Fundamentals: the Perceptron](../notes/week-02/07-perceptron.md)

You now know that a dot product measures similarity and that $\mathbf{W}\mathbf{x}+\mathbf{b}$ is the
only operation a linear layer performs. This lecture spends that knowledge: it builds the smallest
object in the whole course that *learns* — one neuron, a handful of weights, and a rule for nudging
them when the answer comes out wrong. Everything later is this unit repeated, stacked, and
differentiated.

It also fails, on purpose. The perceptron cannot learn XOR, and the failure is not a bug or a
tuning problem — it is a theorem. Seeing exactly *why* a single line cannot carve up four points is
what makes the hidden layer of [Lec 8](../notes/week-02/08-mlp-and-activations.md) feel inevitable rather than
arbitrary. The deck's last content slide is that failure. Treat it as the setup for the next chapter.

### [Lec 8 — Multi-Layer Perceptron and Activation Functions](../notes/week-02/08-mlp-and-activations.md)

The previous lecture ended on a wall. A single perceptron draws exactly one straight line through the
input space, and XOR's four points cannot be split by any straight line. That is not a small failure
— it is a proof that a single-layer model is permanently blind to any pattern that is not linearly
separable, and almost nothing in vision is.

This lecture pays off that cliffhanger. Insert one layer of units *between* the input and the output,
give each a non-linear squashing function, and the network can draw several lines and then combine
the regions they carve out. XOR falls immediately; so does, in principle, any continuous function.
What follows is the architecture, the exact weights that solve XOR, the proof that the non-linearity
is doing all the work, and softmax — the output activation every classifier in the rest of the course
ends with.

### [Lec 9 — Backpropagation](../notes/week-02/09-backpropagation.md)

Lec 8 gave you a network that *can* represent a complicated function, but said nothing about how to
find the weights that make it represent the *right* one. A perceptron was easy: one layer, and the
update rule $w \leftarrow w + \eta(d-y)x$ falls straight out of "if the answer is too small, push the
weight up". A hidden layer destroys that argument, because nobody tells you what a hidden unit was
*supposed* to output. There is no target for $h_1$. The deck puts it as two questions on slide 4:
given an input, how do we compute the output — and *how do we get the weights?*

Backpropagation is the answer, and it is the mechanism every remaining lecture in this course runs
on. CNNs, RNNs, transformers, VAEs: all of them are trained by exactly the procedure below. Learn it
once, properly, by hand.

### [Lec 10 — Overfitting, Bias–Variance, and Regularization](../notes/week-02/10-overfitting-and-regularization.md)

You can now build a network and train it: forward pass, loss, backprop, update. Backprop is a machine
for driving the *training* loss to zero, and it is very good at its job. That is exactly the problem.
Nobody pays you for low training loss — the training labels are already known. What you are actually
buying is performance on data the model has never seen, and those two things come apart. Push training
loss hard enough and the network starts memorising the noise in your particular sample rather than the
structure in the world.

This lecture is the whole of generalization in this book: how to recognise that failure, how to name
its two components (bias and variance), how to measure it honestly (cross-validation), and how to fight
it (regularization, early stopping). Every later architecture — ResNet, dropout-laden transformers,
augmented VAE training — is partly an answer to this lecture.

## Week 3

### [Lec 11 — Convolutional Neural Networks I: Basics](../notes/week-03/11-cnn-basics.md)

You can already build a multi-layer perceptron and train it with backpropagation. Point it at an image
and it falls over. Not because the idea is wrong but because of a counting problem: a dense layer
connects every input to every unit, and an image has a lot of inputs. One modest layer on one modest
photograph needs more weights than there are pixels in the whole training set. The network also learns
each position independently — an edge detector painstakingly learned in the top-left corner is useless
in the bottom-right, because those are different weights.

Convolution fixes both problems with one mechanism: a small bank of weights, slid across the image, so
the same detector is applied everywhere. This lecture is the arithmetic of that mechanism, and every
architecture from [Lec 12](../notes/week-03/12-cnn-architectures.md) to [Lec 28](../notes/week-08/28-vit-detr-swin.md) is
built from the four numbers you learn to juggle here: kernel size, stride, padding, and depth.

### [Lec 12 — CNN Model Architectures: LeNet → GoogLeNet](../notes/week-03/12-cnn-architectures.md)

[Lec 11](../notes/week-03/11-cnn-basics.md) gave you the pieces — convolution, stride, padding, pooling — but a pile
of pieces is not a network. Someone has to decide how many filters, of what size, in what order, and
every one of those choices costs memory, compute, and parameters. This lecture is the history of four
people making those choices well, over sixteen years: LeNet-5 in 1998, AlexNet in 2012, and then VGGNet
and GoogLeNet in the same year, 2014. Each contributes exactly one durable idea, and all four ideas are
still in the models you will meet in week 4 and beyond. Read this chapter as a sequence of answers to
one question — *given a fixed compute budget, what shape should a convolutional network be?* — because
that is the question the next three lectures also answer, just with newer tricks.

### [Lec 13 — CNN Optimization: Vanishing Gradients and Activation Functions](../notes/week-03/13-vanishing-gradients-activations.md)

You have just seen a decade of architectures get deeper — LeNet 5 layers, AlexNet 8, VGG 19, GoogLeNet
22 — with accuracy improving each time. The obvious next move is to keep stacking. It does not work.
Past a certain depth, plain networks get *worse*, and worse in a way that has nothing to do with the
usual culprit: they fail on the **training** set, where a bigger model should always win.

This chapter explains why. Backpropagation sends the error signal backwards as a *product* of one
factor per layer, and each factor carries an activation derivative. If those derivatives sit below 1,
the product collapses geometrically and the early layers stop learning. The whole activation-function
story — sigmoid, tanh, ReLU, Leaky ReLU — is really a story about keeping that one number near 1.
The deck's punchline is that even the best activation is not enough, which is the problem
[Lec 14](../notes/week-04/14-resnet.md) exists to solve.

## Week 4

### [Lec 14 — CNN with Residual Connections (ResNet)](../notes/week-04/14-resnet.md)

The previous chapter left you with a debt. It showed that backpropagating through $N$ layers
multiplies $N$ derivatives together, that each is smaller than 1, and that the product therefore dies
exponentially with depth. Then it offered ReLU and Leaky ReLU as repairs, and admitted they are
partial: a Leaky ReLU slope of $\alpha = 0.01$ is still less than 1, so fifty in series still
annihilate the signal. Worse, deep plain networks were observed to get *worse* training error than
shallow ones, which no activation-function tuning explains.

This lecture pays the debt. One architectural change — adding the block's input to its output —
turns the product of derivatives into a product of terms that each contain an explicit $+1$. That
single term is why a 152-layer network could be trained at all, and why it won ILSVRC 2015 at 3.6%
top-5 error, below the 5.1% a human scores.

### [Lec 15 — CNN Model Architectures II: DenseNet, MobileNet, EfficientNet](../notes/week-04/15-densenet-mobilenet-efficientnet.md)

ResNet settled one question: how do you train a network 150 layers deep without the gradient dying on
the way back? Once that was answered, the field stopped asking "can we go deeper?" and started asking
three sharper questions. Are we *wasting* parameters by making every layer re-learn features an earlier
layer already found? Can a convolution run on a phone, where you have a watt of power and no GPU? And
when you are handed more compute, *which* dimension of the network should you grow — layers, channels,
or input pixels? Those three questions produce DenseNet, MobileNet and EfficientNet. They are not
competitors; they optimise different things. Keep that framing and the three architectures stop being a
list to memorise and become three answers to three questions.

### [Lec 16 — Sequential Modelling and Recurrent Neural Networks](../notes/week-04/16-sequence-modelling-and-rnn.md)

Everything in the course so far has assumed a fixed-size input and no memory. An MLP takes a vector
of length $n$; a CNN takes an $H\times W\times C$ tensor. Hand either one a sentence of 7 words and
then a sentence of 40, and the architecture does not exist to accept both. Worse, flatten a sentence
into a bag of word vectors and you throw away order — "dog bites man" and "man bites dog" become the
same input, with opposite meanings.

Real data is full of order: video, speech, text, DNA, stock prices, ECG traces. This lecture
introduces the machinery that handles it. One cell, applied repeatedly, carrying a hidden state
forward. That single structural idea — **the same weights at every time step** — is what makes a
sequence model work, and it is also what makes it hard to train, which is Lec 17's subject.

## Week 5

### [Lec 17 — Training RNNs: Backpropagation Through Time](../notes/week-05/17-bptt.md)

[Lec 16](../notes/week-04/16-sequence-modelling-and-rnn.md) built an RNN and ran it forward: a hidden state
carried along a sequence, one shared weight matrix applied at every step. It never said how those
weights get learned. That is this lecture. The answer sounds easy — unroll the recurrence into a
feed-forward graph and run ordinary backprop — and it almost is. But one fact changes everything:
the same matrix $\mathbf{W}_{hh}$ appears at every time step, so its gradient is not one number but a
**sum of contributions from every step**, and each contribution reaches back through a chain of
multiplications by that same matrix. Repeated multiplication by a fixed matrix either collapses to
zero or blows up. That single observation explains why vanilla RNNs cannot remember anything for long,
why we truncate training, why we clip gradients, and why the next lecture has to invent LSTM.

### [Lec 18 — Long Short-Term Memory (LSTM)](../notes/week-05/18-lstm.md)

Lec 17 ended on a diagnosis, not a cure. Backpropagation through time multiplies the same recurrent
weight matrix once per time step, so the gradient reaching step 1 from step 50 is a 50-fold product;
if the factors sit below 1 the gradient is annihilated, and the early steps simply stop learning. The
deck's own arithmetic says it: $0.9^{50} \approx 0.005$. That is a training failure, but it is also a
*modelling* failure — a vanilla RNN overwrites its entire hidden state at every step, so there is no
place for information to sit still and wait. This lecture introduces the 1997 architecture that fixes
both at once by adding a second state vector that is *added to* rather than transformed, and three
learned gates that decide what gets erased, written and exposed. Everything in the rest of the course
that remembers something over a long range descends from this idea.

### [Lec 19 — GRU, Seq2Seq, and Attention](../notes/week-05/19-gru-seq2seq-attention.md)

Gating fixed the gradient. [Lec 18](../notes/week-05/18-lstm.md) gave the LSTM a cell state that lets error flow
backwards across hundreds of steps without dying. But gating did not fix *architecture*. The moment
you ask an RNN to map a sentence in one language to a sentence in another — different length,
different word order — you need two networks, an encoder and a decoder, joined by a single vector.
And that single vector is a wall: everything the source sentence says has to fit through it, no
matter how long the source is.

This lecture does three things. It builds the encoder–decoder (seq2seq) model, shows exactly why its
fixed-length context vector is a bottleneck, and then removes the bottleneck with **attention** — the
idea that every chapter from here to the end of the course is built on. It also introduces the
**GRU**, the LSTM's cheaper sibling.

### [Lec 20 — Generative AI for Vision Tasks I](../notes/week-05/20-genai-vision-tasks-1.md)

Nineteen lectures of machinery have gone past — perceptrons, CNNs, ResNets, RNNs, attention — and
every one of them was a *mechanism*. None of them said what the mechanism is for. This lecture stops
and takes inventory: here are the tasks computer vision actually gets asked to do, here is why each
one is hard, and here is the specific place a generative model earns its keep in each.

It comes after the sequence-modelling block because that block ends the "how do we build a predictor"
arc. Before the course turns to building generators properly (Week 6 onwards), it needs you to
already know what you would point one at. The lecture carries almost no mathematics; its value is a
map, a vocabulary, and a set of distinctions — semantic versus instance segmentation, detection
versus classification — that the exam tests relentlessly.

### [Lec 21 — Generative AI for Vision Tasks II](../notes/week-05/21-genai-vision-tasks-2.md)

[Lec 20](../notes/week-05/20-genai-vision-tasks-1.md) surveyed the tasks where generation *assists* an existing
discriminative pipeline — it augments a classifier, fills a hole, upsamples a patch. The output is
still judged against a ground truth that already exists somewhere. This half of the survey crosses a
line. Here the generated image **is** the deliverable: there is no ground-truth winter photograph of
that summer valley, no real CT scan for that MRI, no true next frame of a video that was never shot.
Once there is no target to regress to, supervised loss functions stop working, and the lecture's whole
job is to show you what replaces them — cycle consistency, reconstruction error as a score, adversarial
realism, temporal coherence. Seven tasks, each a case study in generating without a label.

## Week 6

### [Lec 22 — Generative Modelling: Taxonomy and Maximum Likelihood](../notes/week-06/22-generative-taxonomy-and-mle.md)

Week 1 gave you two things that have not yet been joined up. [Lec 3](../notes/week-01/03-generative-vision-models.md)
named five families of generative model and described what each one *does*;
[Lec 6](../notes/week-01/06-probability-2.md) taught maximum likelihood as a piece of statistics, with coins.
Neither said why those five families exist, or what any of them is actually optimising when it trains
on photographs. This lecture supplies the missing spine. One objective — maximise the likelihood the
model assigns to the training set — turns out to be computable exactly for some model designs,
computable only up to a bound for others, and not computable at all for a third group that sidesteps
it entirely. That single fork is the taxonomy, and the taxonomy is the table of contents for the
remaining eighteen lectures. Everything from here to diffusion is one of its leaves.

### [Lec 23 — Autoregressive Generative Models: FVBN, PixelRNN, PixelCNN](../notes/week-06/23-autoregressive-pixelrnn-pixelcnn.md)

Lec 22 left you with a taxonomy and an empty box. It said generative models split into *explicit
density* (you write down $p_\theta(\mathbf{x})$ and maximise it) and *implicit density* (you only
learn to sample), and that explicit density splits again into *tractable* and *approximate*. The
tractable box was named but not filled. This lecture fills it.

The filling is one idea: impose an **order** on the pixels of an image, and the chain rule of
probability turns the impossibly high-dimensional joint $p(\mathbf{x})$ into a product of ordinary
one-variable conditionals that a neural network can model. Nothing is approximated. You get the
exact likelihood of any image, which is what "tractable density" means. The price — and it is a
brutal one — is that generation must then happen one pixel at a time, and this chapter makes that
cost numerical so you can see exactly why the field moved on.

### [Lec 24 — Image Captioning and Spatial Attention (Transformers I)](../notes/week-06/24-captioning-and-spatial-attention.md)

[Lec 19](../notes/week-05/19-gru-seq2seq-attention.md) diagnosed a failure in sequence-to-sequence
translation: the encoder squeezes an entire sentence into one fixed-length context vector, the decoder
has nothing else to look at, and long inputs degrade. Attention over the encoder's hidden states fixed
it. This lecture shows that the *identical* failure appears in vision. Swap the source sentence for an
image and the encoder RNN for a CNN, and you get image captioning — where the standard recipe pools the
whole picture into one vector before a word is ever emitted. A caption like "boy waving flag" needs the
boy for one word and the flag for another, and one vector cannot serve both.

The fix is the same mechanism, pointed at a different kind of thing: instead of attending over a
*sequence of hidden states indexed by time*, you attend over a *grid of feature vectors indexed by
space*. That single substitution is the whole lecture, and it is the last step before attention gets
written down in its general form.

## Week 7

### [Lec 25 — Transformers II: Q/K/V and Self-Attention](../notes/week-07/25-qkv-and-self-attention.md)

You have now seen attention twice: once bolted onto an RNN decoder to pick encoder states
([Lec 19](../notes/week-05/19-gru-seq2seq-attention.md)), and once bolted onto a CNN feature grid to pick
image regions while generating a caption ([Lec 24](../notes/week-06/24-captioning-and-spatial-attention.md)).
Both times it was an accessory, and both times the mechanism was identical — only the *things being
attended to* changed. That repetition is the clue: if the mechanism does not care whether its inputs
are RNN states or CNN cells, you can strip the RNN and the CNN away and keep only the mechanism. What
is left is a layer that takes a set of vectors in, gives a set of vectors out, and does nothing but
similarity-weighted lookup.

This lecture performs that stripping. Everything in the rest of the course — the encoder, the decoder,
ViT, DETR, Swin, every large language model — is this one layer, repeated.

### [Lec 26 — Transformers III: The Encoder and Positional Encoding](../notes/week-07/26-encoder-and-positional-encoding.md)

Lec 25 left you with a debt. Self-attention lets every token look at every other token in one step,
which is wonderful — but it is **permutation-invariant**. Shuffle the input tokens and the output
tokens shuffle with them, unchanged. A self-attention layer sees a *bag* of vectors, not a *sequence*
of them. So "dog bites man" and "man bites dog" come out of the layer carrying exactly the same
information, and no amount of stacking fixes it.

This lecture pays the debt. You add position back in explicitly, as a vector, before the first
attention layer ever runs. Then you assemble the pieces from Lec 25 into the thing they were always
heading towards: the **encoder block**, and a stack of six of them. By the end you can draw the
encoder from memory, count its parameters, and compute a positional-encoding vector by hand.

### [Lec 27 — Transformers IV: The Decoder and the Full Architecture](../notes/week-07/27-decoder-and-full-transformer.md)

Lec 26 left the encoder with two boxes drawn but not explained: **Add & Norm** and the
**feed-forward network**. You were told where they sit and promised the reasoning later. This is
later. And the encoder on its own only *reads* — it turns a source sentence into a pile of context
vectors and stops. Nothing so far can *write*. Translation, captioning and every generative use of a
Transformer need a second stack that emits one token at a time, consults the encoder's output while
doing it, and is forbidden from peeking at the token it has not produced yet. That stack is the
**decoder**, and the one new mechanism inside it — **cross-attention** — is the hinge the whole
encoder–decoder architecture turns on. By the end of this chapter you can draw the complete 2017
Transformer from memory and count its parameters.

## Week 8

### [Lec 28 — Vision Transformers: ViT, DETR, Swin](../notes/week-08/28-vit-detr-swin.md)

Four lectures built the Transformer on sentences. This is a computer vision course, so the obvious
question is: point it at an image. The obvious answer — treat every pixel as a token — dies on
arithmetic, because self-attention costs $O(n^2)$ in the number of tokens and a 224×224 image has
50,176 pixels. That single number is the hinge of the whole lecture. Everything here is an answer to
"what do you make a token, and which tokens are allowed to talk to each other?" ViT says: make a
16×16 patch a token, and let them all talk. DETR says: keep a CNN, and make the *outputs* tokens.
Swin says: let tokens talk only inside a local window, and slide the window. Three answers, three
architectures, one constraint.

### [Lec 29 — From Autoencoders to the Variational Autoencoder](../notes/week-08/29-autoencoders-to-vae.md)

You already know how to squeeze an image into a short vector and rebuild it: that is an autoencoder,
and it is just two MLPs glued back to back. The obvious next thought is "so make up a short vector
yourself, push it through the decoder, and you have generated an image." That thought is wrong, and
*why* it is wrong is the single most important idea in this chapter. A plain autoencoder scatters its
training images across the latent space as isolated points with no distributional structure, so almost
every vector you invent lands in a hole where the decoder has never been trained and produces noise.

The variational autoencoder fixes this by refusing to map an input to a point at all. It maps each
input to a *distribution*. This chapter builds that idea, then recasts it as a formal latent-variable
model — at which point you will be able to see for yourself that the quantity you want to maximise is
an integral nobody can compute. Lec 30 solves that problem; this one earns it.

### [Lec 30 — The ELBO and the Reparameterization Trick](../notes/week-08/30-elbo-and-reparameterization.md)

[Lec 29](../notes/week-08/29-autoencoders-to-vae.md) built a latent-variable generative model and then walked into a
wall. You want to maximise $\log p_\theta(\mathbf{x})$, but computing it means integrating over every
possible latent code, and the posterior $p_\theta(\mathbf{z}\mid\mathbf{x})$ you would need to do that
efficiently is itself unavailable. The chapter ended on the proposal: give up on the exact posterior
and approximate it with something you *can* compute.

This lecture cashes that proposal in. It turns "approximate the posterior" into a concrete, trainable
objective — the **ELBO** — shows that maximising it does two useful things at once, splits it into the
two terms you actually code up, and then fixes the one remaining obstacle: a random sampling step
sitting in the middle of the network, blocking gradients. Everything in the VAE that works, works
because of what is on these slides.

## Supplementary

### [Supplementary — Topics the Slides Name But Never Teach](../notes/supplementary.md)

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
