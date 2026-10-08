# Week Summaries

The one-paragraph version of every lecture — its **Why this lecture exists** section. Use this to rebuild the thread of the course quickly, or to decide what to re-read.

## Week 1

### [Lec 1 — Introduction to NLP](../notes/week-01/01-intro-to-nlp.md)

Every other lecture in this course is a technique. This one is the problem statement, and without it
the techniques look arbitrary.

Two things get established. First, *why natural language resists computation at all*: not because text
is long or noisy, but because almost every utterance has more than one licensed reading, and the
machinery that picks the right one is world knowledge the computer does not have. Ambiguity is the
reason NLP is a research field rather than a parsing exercise. Second, *how the field has repeatedly
changed its mind* about what a solution looks like — hand-written grammars, then counting, then
learned representations, then pretrain-and-fine-tune, then pretrain-and-prompt, and now "phrase every
task as text generation and let one model do it".

That last arc is the spine of the remaining 59 chapters. Fix it now and everything that follows has a
place to sit.

### [Lec 2 — Text Processing Basics and Tokenization](../notes/week-01/02-text-processing-tokenization.md)

Lecture 1 promised that computers would process natural language. Before any model can touch a
sentence, something has to chop that sentence into discrete pieces and give each piece an integer ID,
because neural networks consume index vectors, not characters. That chopping step is **tokenization**,
and it happens to every piece of text in every NLP system you will build.

The lecture's real content is that tokenization is *not obvious*. "Split on spaces" fails immediately
on `isn't`, `prize-winning` and `great movie!`. Splitting into characters throws away everything a word
means. And whatever you choose, a test sentence will eventually contain a word your vocabulary has
never seen. The resolution — learn the units from the data itself, as **subwords**, using **byte-pair
encoding** — is what every modern model does, and it is this chapter's centrepiece.

### [Lec 3 — N-gram Language Models I: The Task, the Chain Rule and Counting](../notes/week-01/03-ngram-lm-1.md)

Lecture 2 turned a stream of characters into a sequence of tokens. That is the input format, not a
model — nothing so far assigns a *score* to a piece of text. This lecture introduces the one task that
the remaining 57 lectures all come back to: **given some words, predict the next one**, and
equivalently, assign a probability to a whole sentence.

It matters far beyond spell-checking. Every large language model in this course — GPT, BERT, LLaMA —
is trained on exactly this objective, which is why the deck's own title slide shouts "LM in LLMs!!".
The task is also **self-supervised**: the correct answer is the next word, which is already sitting in
the corpus, so no human has to label anything and the method scales to the entire internet.

The lecture then builds the simplest honest model of it — count n-grams and divide — and shows both
why that works and where it falls apart.

### [Lec 4 — N-gram Language Models II: Smoothing and Perplexity](../notes/week-01/04-ngram-lm-2-smoothing-perplexity.md)

Lecture 3 built an n-gram language model by counting. That model has a fatal flaw and no way to tell
whether it is any good — and this lecture fixes both.

The flaw is arithmetic. A single unseen bigram makes one factor of the chain-rule product zero, which
makes the probability of the *entire sentence* zero, no matter how ordinary the rest of it is. Since a
test set always contains word pairs the training corpus never showed, an unsmoothed n-gram model
assigns probability zero to essentially everything. Smoothing is the repair.

The second half answers "is model A better than model B?" without running a translation system to find
out. That answer is **perplexity**, and it is the most reliably examined numerical in the first half
of this course.

### [Lec 5 — NLP Tasks and Paradigms](../notes/week-01/05-nlp-tasks-and-paradigms.md)

Weeks 1–4 of the lecture list read like a catalogue — sentiment, NER, translation, summarization,
chatbots, parsing. Treated as a catalogue they are unlearnable. This lecture replaces the catalogue
with a **map**: almost every NLP problem is one of three or four machine-learning shapes, and once you
know the shape you know what the input is, what the output is, what a model must emit, and which
number you report.

That last part is why this chapter is the metrics reference for the rest of the book. An n-gram model
had perplexity ([Lec 4](../notes/week-01/04-ngram-lm-2-smoothing-perplexity.md)); a classifier does not. The deck spends
eight pages on accuracy, precision, recall, F-measure and the macro-vs-micro distinction because every
later chapter — BERT fine-tuning, QA, dialogue, instruction tuning — reports those numbers and assumes
you can compute them. The lecturer's own arithmetic on page 132 is the single most mechanically
examinable thing in Week 1.

## Week 2

### [Lec 6 — Supervised Learning](../notes/week-02/06-supervised-learning.md)

Week 1 built language models by counting. Counting works for n-grams and stops working almost
immediately after: you cannot count your way to a sentiment classifier, a translator or a captioning
system. Everything from here to the end of the course is instead a *parameterised function fitted to
labelled data*, and this lecture is where that recipe is stated once, in full, so the remaining 54
lectures can assume it.

The lecture does two things. It walks a gallery of wildly different tasks — house prices, review
sentiment, image labels, French translation, photo captions — and extracts the single shape they all
share. Then it instantiates that shape in the smallest possible case, 1-D linear regression, and runs
it end to end: model, data, loss, training, testing. Every network in this course is that same
skeleton with a bigger $f$.

### [Lec 7 — Shallow Neural Networks](../notes/week-02/07-shallow-neural-networks.md)

Lecture 6 fitted a straight line. A straight line can only express one relationship between input and
output — proportionality — and it takes exactly one input and produces exactly one output. Almost
nothing you care about in NLP looks like that.

This lecture builds the smallest model that escapes all three limits at once. The trick is startling
in how little it takes: compute a few *different* straight lines of the input, bend each one with a
single rule (negatives become zero), then add the bent lines back up with weights. The result is a
function made of straight pieces joined end to end, and you can place those joints anywhere and give
each piece any slope you like. With enough pieces you can trace any curve. That is the shallow neural
network, and the geometry of how it is assembled is the whole content of this lecture.

### [Lec 8 — Deep Neural Networks](../notes/week-02/08-deep-neural-networks.md)

[Lec 7](../notes/week-02/07-shallow-neural-networks.md) ended on an awkward note. The universal approximation theorem
says a network with **one** hidden layer can approximate any continuous function to any accuracy, given
enough hidden units. If one layer is enough in principle, every deep network ever trained is a waste of
effort — you could flatten it and lose nothing.

This lecture answers that. The answer is not about what a network *can* represent; both shallow and
deep networks represent everything. It is about what a network can represent **per parameter**. A deep
ReLU network carves its input space into exponentially more linear regions than a shallow one with the
same parameter budget, because each layer folds the space the previous layer produced, so regions
*multiply* with depth where they only *add* with width. That single argument is the whole lecture, and
it is the most examinable thing in Week 2.

### [Lec 9 — Backpropagation](../notes/week-02/09-backpropagation.md)

Lecture 8 built a deep network and showed it is a composition of linear maps and activations. Training
it means descending the loss, and descending the loss means knowing $\partial\mathcal{L}/\partial\theta$
for every one of the parameters. Nobody told you how to get those numbers.

You could get them by brute force: nudge one parameter, re-run the network, see how the loss moved.
That costs one forward pass *per parameter*, and a real model has $10^8$ of them. It is not slow, it
is impossible.

Backpropagation is the algorithm that computes the **entire** gradient — every partial derivative, all
at once — for roughly the cost of **one** extra forward pass. That single fact is why deep learning
works at all. This lecture derives it on the deck's toy function, generalises it to a deep network,
and builds the matrix calculus you need to write it in vector form.

### [Lec 10 — Gradient Descent and Initialization](../notes/week-02/10-gradient-descent-and-init.md)

[Lecture 9](../notes/week-02/09-backpropagation.md) showed you how to compute $\partial\mathcal{L}/\partial\theta$ for
every parameter in a deep network. That is a number, not a trained model. This lecture is the other
half: what you *do* with that number, and where the parameters were sitting when you started.

Both halves can fail on their own. A learning rate ten times too large makes the loss explode no
matter how exact your gradient was. A weight initialisation off by an order of magnitude kills the
signal before the first gradient is ever computed. And on a non-convex surface — which is every
network you will ever train — plain gradient descent carries no guarantee of finding anything good.

So this lecture builds the training loop that the remaining fifty chapters quietly assume: random
initialisation, mini-batches, momentum, and Adam. Everything from word2vec to a 70-billion-parameter
LLM is trained by the machinery on these thirty slides.

## Week 3

### [Lec 11 — Word Representation](../notes/week-03/11-word-representation.md)

Everything so far has treated a word as an *index*. Tokenization produced a vocabulary, the n-gram
models counted strings, and nowhere did any of it know that "hotel" and "motel" are nearly the same
thing. That ignorance has a cost you can measure: a search engine that cannot match "Baltimore motel"
against "Baltimore hotel", a classifier whose feature "the previous word was *terrible*" fires on
nothing when the test set says *dreadful*.

This lecture replaces the index with a **vector**, and it does so without any supervision — purely by
counting what each word appears next to. That one move is the foundation of the whole rest of the
course: word2vec, GloVe, fastText, and eventually BERT are all refinements of the idea introduced
here. Get the counting, the weighting and the similarity measure right now and the next four lectures
are variations on a theme you already understand.

### [Lec 12 — Learning Word Representation I: word2vec Skip-gram](../notes/week-03/12-word2vec-skipgram.md)

Lecture 11 argued that a word's meaning lives in the company it keeps, and built dense vectors by
*counting* co-occurrences and reweighting them with PMI. That works, but it is a two-stage pipeline:
build a $\lvert V\rvert \times \lvert V\rvert$ matrix, then factor or truncate it. The matrix is
enormous, re-estimating it when new text arrives means starting over, and nothing in the procedure is
*learning* — there are no parameters and no objective.

This lecture replaces counting with prediction. You declare up front that every word *is* a short
dense vector, you write down a task those vectors have to solve — guess the words around me — and you
let gradient descent do the rest. The vectors become the parameters of a tiny classifier. That single
reframing is word2vec, and it is the template every embedding method after it follows.

### [Lec 13 — Negative Sampling, Hierarchical Softmax and GloVe](../notes/week-03/13-negative-sampling-glove.md)

Lecture 12 left an unpaid bill. Skip-gram's probability of a context word given a centre word is a
softmax whose denominator sums $\exp(\mathbf{u}_w^\top\mathbf{v}_c)$ over **every** word in the
vocabulary. With $|V| = 50{,}000$ and 300-dimensional vectors, one gradient step touches 15 million
numbers — and a real corpus has billions of (centre, context) pairs. The model is correct and
untrainable.

This lecture pays the bill twice over, with two different tricks that **change the objective** rather
than approximate the old one: negative sampling turns the $|V|$-way softmax into a handful of binary
classifications, and hierarchical softmax replaces the flat vocabulary with a binary tree so the cost
drops to $\log_2|V|$. It then asks a question that reopens everything: if co-occurrence statistics are
what skip-gram is implicitly learning, why not just count them? That question is GloVe.

### [Lec 14 — Word Vectors: Other Extensions](../notes/week-03/14-fasttext-and-beyond-words.md)

By the end of Lecture 13 you can train a word embedding and you know two ways to make the objective
cheap. But the whole construction rests on an assumption nobody has questioned yet: that the unit you
embed is a **word**, and that you have seen every word you will ever need.

Both halves of that assumption are wrong. "New York" is one thing, not two. "running" and "runs" share
almost everything, and word2vec gives them unrelated vectors. A word the training corpus never
contained gets no vector at all — not a bad one, *none*. And the same machinery that embeds words
turns out to embed sentences, taxonomies, social networks and factual databases with almost no change.

This lecture is five extensions in one go: phrases below the word boundary of interest, **fastText**
below the word, doc2vec above it, and then the jump out of text entirely into graphs and knowledge
graphs. The breadth is the point; so is the fact that every one of them is still skip-gram underneath.

### [Lec 15 — Cross-Lingual Representations](../notes/week-03/15-cross-lingual-representations.md)

Everything in this week has trained embeddings for *one* language, and the implicit language has been
English — because that is where the data is. Labelled NLP datasets, treebanks, sentiment corpora and
QA benchmarks are overwhelmingly English; Marathi, Oriya and Assamese have almost none. Training a
sentiment classifier for Oriya the normal way is not hard, it is impossible, because the labels do
not exist.

The fix this lecture builds is a **shared embedding space**: one vector space in which the English
word *dog* and the Hindi word *kutta* land in nearly the same place. Once you have that, you train a
classifier on English labels, feed it Hindi vectors at test time, and it works — **zero-shot transfer**,
with no Hindi labels at all. The whole lecture is a catalogue of ways to build that shared space,
organised by how much bilingual supervision each one demands.

## Week 4

### [Lec 16 — RNN Language Models](../notes/week-04/16-rnn-language-models.md)

Weeks 1 and 3 each solved half of a problem. Week 1 built a language model that could score a sentence
but could only count: it saw "students opened their" and "pupils opened their" as two unrelated
symbols, and it ran out of evidence the moment you asked for more than three words of context. Week 3
built word vectors in which *students* and *pupils* sit next to each other, but it never used them to
predict anything beyond a window of neighbours.

This lecture joins them. You replace the count table with a *learned function over embeddings*, so
statistical strength is shared automatically across words that mean similar things. Then you discover
that the obvious first attempt — concatenate a fixed window of embeddings and run them through a
hidden layer — inherits the n-gram model's worst habit: a window that cannot grow, and a separate set
of weights for every position in it. Fixing that gives you the recurrent neural network, the first
architecture in this course that can read a sequence of any length.

### [Lec 17 — RNN Applications: Text Generation, Sequence Labeling, Text Classification](../notes/week-04/17-rnn-applications.md)

Lecture 16 built one thing: an RNN trained to predict the next word. This lecture shows that the same
machine, with the output layer rewired, solves three different families of NLP task. Feed the model's own prediction back in as the next input and you have a text
generator. Put a softmax over tags instead of over the vocabulary and you have a part-of-speech tagger
or a named-entity recogniser. Throw away all outputs except the last hidden state and you have a
sentence classifier. Three tasks, one recurrence, one set of weight matrices.

Two new ideas arrive with them. **BIO tagging** is the trick that converts a
span-finding problem into a per-token labelling problem, and it reappears in later chapters. **Bidirectional RNNs** fix a left-to-right model's blindness to the right
half of the sentence — at the cost of ruling out generation entirely.

### [Lec 18 — RNN for Sequence to Sequence, and Attention](../notes/week-04/18-seq2seq-and-attention.md)

Every RNN architecture you have met so far produces exactly one output per input token, or one output
for the whole sequence. Neither shape can translate. "Il convient de noter" has four words; "It should
be noted that" has five — and they do not line up in order. Translation needs a model that reads a
sequence of one length and writes a sequence of a *different* length, in a *different* order.

This lecture builds that model: the encoder–decoder, or sequence-to-sequence, architecture. It then
immediately finds the flaw in it. The encoder squeezes the entire source sentence into one
fixed-length vector, and the decoder sees nothing else. The fix is **attention**, and attention is the
single most consequential idea in this course — Lectures 21–24 take this lecture's mechanism, throw
away the RNN around it, and get the Transformer.

### [Lec 19 — Decoding Strategies](../notes/week-04/19-decoding-strategies.md)

Every lecture up to here has been about building a model. Training is finished; the weights are fixed.
At each generation step that trained model hands you one thing: a probability distribution over the
whole vocabulary. Nothing in the model tells you which token to actually emit.

That gap is the subject of this lecture. **Decoding** is the algorithm that turns a sequence of
distributions into a sequence of tokens, and it is a completely separate object from the model. The
same weights, the same input sentence, decoded two different ways, produce two different outputs —
one fluent and bland, the other surprising and occasionally incoherent. No retraining involved.

This matters now because [Lec 18](../notes/week-04/18-seq2seq-and-attention.md) finished the encoder-decoder machine
and [Lec 17](../notes/week-04/17-rnn-applications.md) described the generation *loop* while explicitly deferring the
*choice*. This chapter is that choice. It is also the last piece of decoding you will be taught: the
Transformer chapters, GPT and the prompting chapters all reuse exactly these five algorithms.

### [Lec 20 — Better RNN Units: GRU and LSTM](../notes/week-04/20-gru-and-lstm.md)

Lectures 16 to 19 built the recurrent stack: an RNN language model, sequence labelling, seq2seq with
attention, and decoding. Every one of them assumed the RNN could actually *learn* from what happened
many steps ago. It cannot.

The reason is arithmetic, not engineering. An RNN applies the **same** recurrent weight matrix at
every time step, so the gradient flowing from step $t$ back to step $k$ is multiplied by that one
matrix $t-k$ times. Anything raised to a large power either collapses to zero or blows up; there is
no middle. Collapse is the common case, and it means the model's weights get updated only by what
happened recently.

This lecture diagnoses that failure precisely and then fixes it with **gating**: replace the
"rewrite the state every step" recurrence with "keep the state unless a learned gate says otherwise".
That is the LSTM. It also closes with the two problems gating does *not* fix — which is the whole
reason Lecture 21 exists.

## Week 5

### [Lec 21 — Introduction to Transformers](../notes/week-05/21-intro-to-transformers.md)

Week 4 gave you a recurrent encoder-decoder and bolted attention onto it so the decoder could look
back at every source word instead of squeezing the sentence through one vector. That fixed the
bottleneck but left the recurrence untouched — and the recurrence is the real problem. An RNN computes
$\mathbf{h}_t$ only after $\mathbf{h}_{t-1}$ exists, so a length-$n$ sentence costs $n$ strictly
ordered steps no GPU can shorten, and information from word 1 reaching word $n$ must survive $n$ hops
through the same weight matrix. Gating ([Lec 20](../notes/week-04/20-gru-and-lstm.md)) softens the second
problem and does nothing about the first.

This lecture asks the question that produced the Transformer: *if attention already lets any position
read any other position directly, why keep the recurrence at all?* Everything here is motivation and
intuition. The machinery arrives in [Lec 22](../notes/week-05/22-self-attention-and-multihead.md).

### [Lec 22 — Self-Attention and Multi-Head Attention](../notes/week-05/22-self-attention-and-multihead.md)

Lecture 21 told you *what* self-attention does — every token looks at every other token and builds a
contextual representation of itself — and left the mechanism as a picture. This lecture turns that
picture into arithmetic you can do on paper, and it is the single most load-bearing lecture in the
course. Eight later chapters assume it.

Three things get settled here. First, how one input vector becomes a query, a key and a value, and
why it takes three separate learned projections instead of reusing the raw vector. Second, the exact
formula — scaled dot-product attention — including *why* the $\sqrt{d_k}$ is there, which is the kind
of question that separates a 70 from a 95. Third, everything that has to be bolted around attention
before a deep stack of these layers will train at all: multiple heads, a position-wise feed-forward
network, residual connections and layer normalization. By the end you can compute a self-attention
output by hand and draw the complete Transformer block from memory.

### [Lec 23 — Positional Encodings and the Encoder Block](../notes/week-05/23-positional-encoding-and-encoder.md)

Lecture 21 bought parallelism by throwing away recurrence, and Lecture 22 turned that trade into
equations. The bill comes due here. A self-attention layer computes every output position from every
input position with the same weight matrices and no notion of order, so the thing it reads is a **bag
of vectors**, not a sentence. The deck says it bluntly on its second slide: *"There is still a major
problem! Order does not matter!!"*

This lecture pays that bill. It injects position back into the input as a fixed sinusoidal signal that
is **added** to the token embeddings, then re-presents self-attention in matrix form — the version a
GPU actually executes — and finally assembles the complete encoder block and asks you to count its
parameters. It closes by noticing that nothing in the block was ever specific to text, and feeding it
image patches instead.

### [Lec 24 — The Decoder and Transformers as Language Models](../notes/week-05/24-decoder-and-transformer-lm.md)

Lectures 22 and 23 built half a machine. You have a mechanism (scaled dot-product attention,
multi-head) and a stack that consumes a sentence and emits one contextual vector per token. That stack
*reads*. It cannot *write*, and nothing so far has stopped any position from looking at any other.

For generation that permissiveness is fatal. A model trained to predict the next word while being
allowed to see the next word learns nothing — it copies. So the decoder needs a second mechanism: a
way to forbid looking forward. And a translation decoder needs a third: a way to consult the source
sentence the encoder just read. This lecture adds exactly those two things, bolts a vocabulary-sized
softmax on top, and then does the move that defines the last decade of NLP — throws the encoder away
and keeps the decoder as a standalone language model.

### [Lec 25 — Efficient Transformers](../notes/week-05/25-efficient-transformers.md)

The previous four lectures built the Transformer and showed it works. This one is the bill.

Self-attention compares every token with every other token, so its cost grows with the *square* of the
sequence length. At 512 tokens that is affordable. At 32,768 it is 4,096 times worse, and the score
matrix alone no longer fits on a GPU. Separately, generation is slow for a different reason: a decoder
emits one token at a time, and a naive implementation redoes all the arithmetic for every previous
token at every step.

So there are two distinct problems — **the quadratic cost of attention in sequence length** and **the
wasted recomputation during autoregressive decoding** — and this lecture gives you the standard fixes
for both. Almost every architectural choice in a modern LLM is one of the answers on these slides.

## Week 6

### [Lec 26 — Pretraining and ELMo](../notes/week-06/26-pretraining-and-elmo.md)

Weeks 4 and 5 gave you architectures — RNNs, LSTMs, Transformers — and the course's first slide here
admits something blunt: *"Transformers are not always better than RNNs by themselves."* The
architecture was never the whole story. What made the last seven years happen is **how you train**,
not **what you train**.

The problem is economic. There are thousands of NLP applications — legal clause extraction, biomedical
NER, customer-support QA — and almost none of them has enough labelled data to teach a randomly
initialised network what English *is*, let alone what the task is. So you split the job: learn language
once, from raw unlabelled text, at enormous scale; then adapt that model cheaply to each task. This
lecture is that idea, the first model that made it work for NLP (**ELMo**), and the three-way
architectural taxonomy that the next four lectures fill in.

### [Lec 27 — BERT and Masked Language Modelling](../notes/week-06/27-bert-masked-lm.md)

Lecture 26 ended with a taxonomy and a problem. Encoders give you bidirectional context — every
position sees every other position — and that is exactly what you want for understanding tasks. But
the only self-supervised objective you have met so far is next-word prediction, and you cannot train
an encoder that way. A bidirectional encoder predicting $w_t$ already has $w_t$ in its input; the
loss would go to zero through a one-hop copy and the model would learn nothing. The lecture's one-word
answer is on the slide title: **masks**. Hide some of the input, predict what you hid, and suddenly
bidirectional conditioning becomes a legitimate training signal. That objective is **masked language
modelling**, the model built on it is **BERT**, and the pretrain-once-fine-tune-many-times recipe that
follows is the single most examined idea in this course.

### [Lec 28 — Span Tasks, T5 and BART (Pretraining Encoders and Encoder-Decoders)](../notes/week-06/28-span-tasks-t5-bart.md)

Lecture 27 gave you BERT and two fine-tuning recipes: one label for the whole sentence, or one label
per token. A large family of NLP tasks fits neither. "Which part of this passage answers the
question?" and "which word sequences in this sentence are named entities?" are both questions about
*contiguous stretches of tokens* — spans — and a span is not a token and not a sentence. The first
half of this lecture builds the machinery for spans, then stops to ask how you would even know BERT
was working, which is what GLUE is for.

The second half closes the taxonomy. Lecture 26 named three architectures; Lecture 27 pretrained the
encoder; Lecture 29 will pretrain the decoder. That leaves the encoder-decoder, which has a genuine
problem of its own: masked language modelling trains an encoder but gives the decoder nothing to do,
while language modelling trains a decoder but wastes the encoder's bidirectionality. MASS, T5 and
BART are three answers.

### [Lec 29 — Pretraining Transformer Decoder: GPT, Zero-shot and In-context Learning](../notes/week-06/29-gpt-decoder-pretraining.md)

[Lec 26](../notes/week-06/26-pretraining-and-elmo.md) laid out three things you can pretrain: an encoder, an
encoder-decoder, or a decoder. Lectures 27 and 28 did the first two, and both needed an *invented*
objective — masking tokens, corrupting spans — because a bidirectional model cannot simply be asked
to predict the next word. The decoder needs no invention. It already is a language model, so you
pretrain it by doing exactly what [Lec 3](../notes/week-01/03-ngram-lm-1.md) defined: predict the next
token. That is the whole objective.

This lecture is where that unglamorous choice turns out to win. Scaling the same next-token model
three times produced GPT-1, GPT-2 and GPT-3, and somewhere along the way the model stopped needing
a task-specific head at all. You describe the task in text, optionally show a few examples in the
prompt, and it answers — with no gradient updates. Everything in Weeks 8 to 12 is about models of
this shape.

### [Lec 30 — Domain-Specific and Multilingual Pretraining](../notes/week-06/30-domain-and-multilingual-pretraining.md)

Everything in Week 6 so far has been pretrained on general English — news, books, Wikipedia — and
evaluated on general English. Two assumptions are buried in that sentence, and this lecture breaks
both.

The first is *general*. Scientific papers, clinical notes and financial filings use a different
vocabulary and give ordinary words different senses, so a model pretrained on Wikipedia arrives in
those domains half-blind. The second is *English*. Most of the world's text is not English, and the
languages with the least text are exactly the ones a model trained by sampling text proportionally
will ignore.

The repairs look superficially alike — pretrain on different data — but each has a cost the deck is
careful to name. Specialising destroys general ability (catastrophic forgetting). Adding languages
helps until you run out of model capacity, and then it hurts (the curse of multilinguality). Those two
trade-offs are what you must leave with.

## Week 7

### [Lec 31 — Question Answering I](../notes/week-07/31-question-answering-1.md)

Lecture 28 taught you to point a BERT span head at a passage and read an answer out of it. That
assumes somebody already handed you the right passage. In the real world nobody does. You type a
question into a search box and the system has to find the paragraph *before* it can read it, out of
five million Wikipedia articles.

This lecture splits the problem in two. **Reading comprehension** is answering from a given passage —
solved, more or less, by the previous lecture. **Open-domain QA** adds a search step in front, and
that step is the whole difficulty, because it has to touch the entire corpus for every question.
Everything you learn here about scoring a document against a query — TF-IDF, BM25, dense bi-encoders —
is the machinery that Week 11's retrieval-augmented generation bolts onto a generative model. The
retriever is the part of RAG that is not the LM.

### [Lec 32 — Question Answering II: Training Retrievers, Generative QA, Multilingual and Tabular QA](../notes/week-07/32-question-answering-2.md)

[Lec 31](../notes/week-07/31-question-answering-1.md) built the retriever-reader pipeline and replaced sparse BM25
scoring with a dense bi-encoder, then stopped on a question it did not answer: *where does the
training data for that retriever come from, and what objective do you train it with?* A dense
retriever needs supervision saying "this passage answers this question and that one does not", and
nobody hands you such a file.

This lecture answers it. The answer is **contrastive**: you never score a passage in isolation, only
relative to competitors, and the whole art is choosing which competitors. From there the lecture asks
the opposite question — can you skip retrieval entirely and read the answer out of a language model's
weights? — and then widens the setting twice: to languages other than English, and to data that is not
prose at all.

### [Lec 33 — Dialogue Systems I: Open-Domain Chatbots and Dialogue Evaluation](../notes/week-07/33-dialogue-systems-1.md)

Every task so far had a right answer. Question answering has a gold span; translation has a reference
sentence; classification has a label. Dialogue does not. To "What are your plans for the weekend?"
there are thousands of good replies and they need share no words with each other, which breaks both
halves of the usual recipe: you cannot train by matching one reference, and you cannot *score* by
matching one reference either.

So this lecture does two things. First it lays out the three ways a chit-chat agent can produce a
turn — retrieve one, generate one, or retrieve-then-refine — and names the two characteristic failure
modes of the generative route: bland responses and inconsistent personality. Then it spends its second
half on evaluation, which is where the real intellectual content is. The argument that n-gram overlap
metrics are *invalid* for dialogue is the most examinable idea in this chapter, and this is the only
chapter in the book that owns it.

### [Lec 34 — Dialogue Systems II: Task-Oriented Dialogue](../notes/week-07/34-dialogue-systems-2.md)

[Lec 33](../notes/week-07/33-dialogue-systems-1.md) built chatbots that converse. A chatbot that produces a fluent,
engaging, entirely plausible reply has succeeded. A system that is supposed to book your flight and
produces a fluent, engaging reply while booking the wrong city has failed completely, and the failure
costs money.

That difference in what counts as success forces a different architecture. A task-oriented agent has
to end up with a *structured* object — a destination, a date, an airline — that it can hand to a
database or an API. Fluent text is not an object. So this lecture builds the machinery that turns
utterances into filled-in structures: the frame, the slot, the tagger that finds slot values in
running text, and the tracker that keeps the structure consistent while the user changes their mind.
The design dates from 1977 and is still what ships, because when real money is involved,
reliability beats fluency.

### [Lec 35 — Text Summarization](../notes/week-07/35-text-summarization.md)

Every application so far has had a short answer: a span, a label, a reply turn. Summarization is the
first task where the *output* is long free text that no gold string uniquely determines — there are
many acceptable summaries of one document, and none of them is "the" answer. That breaks everything
downstream. You cannot train by exact match, you cannot score by exact match, and you cannot even
fit the input into a Transformer when the document is a 3,000-word earnings call.

So this lecture does three jobs at once. It gives the task taxonomy and the two model families
(extractive classification, abstractive generation). It shows what happens when the document is longer
than the context window. And it closes Week 7 with the evaluation machinery — **BLEU**, **ROUGE**,
**BARTScore** — that every generation task in this course, including the dialogue systems of
[Lec 33](../notes/week-07/33-dialogue-systems-1.md), is actually graded by.

## Week 8

### [Lec 36 — Instruction Fine-tuning I](../notes/week-08/36-instruction-finetuning-1.md)

[Lec 29](../notes/week-06/29-gpt-decoder-pretraining.md) left you with GPT-3: a 175-billion-parameter model
trained to predict the next token, which you steer by writing a prompt. What that lecture never said
is that the thing you get out of pretraining is **not an assistant**. It is a text continuer. Ask it
to explain the moon landing to a six-year-old and it will cheerfully produce four more questions,
because on web text a question is more often followed by another question than by an answer. The model
is doing exactly what it was trained to do; what it was trained to do is not what you want.

This lecture names that gap — **alignment** — and gives the first and cheapest repair:
collect `(instruction, output)` pairs across hundreds of tasks and keep training the model on them with
the ordinary language-modelling loss. Nothing about the architecture or the objective changes. Only
the data changes, and the behaviour that falls out is "follow instructions".

### [Lec 37 — Instruction Fine-tuning II: Flan and Self-Instruct](../notes/week-08/37-instruction-finetuning-2.md)

[Lec 36](../notes/week-08/36-instruction-finetuning-1.md) told you *what* instruction tuning is: take a pretrained LM,
fine-tune it on (instruction, input, output) triples drawn from many tasks, and it starts following
instructions on tasks it never saw. It did not tell you where those triples come from, and that turns
out to be the whole problem. Super-NaturalInstructions took NLP researchers across dozens of
institutions to assemble 1,616 tasks. Dolly took a company-wide contest at Databricks to produce
15,000 examples. Neither scales, and neither gives you the *diversity* that the scaling results in
Lec 36 said you need. This lecture gives three answers in increasing order of automation: multiply
each existing supervised dataset by writing several templates for it (Flan), pay humans to write
instructions from scratch (Dolly), or have the model write its own (Self-Instruct). The third is the
one worth remembering.

### [Lec 38 — Reinforcement Learning from Human Feedback I](../notes/week-08/38-rlhf-1.md)

Instruction tuning taught the model to *imitate*. You collected demonstrations — ideal assistant
responses written by contractors — and ran the ordinary next-token objective on them. That gets a base
model to answer rather than ramble, and it stops there, because imitation can only reproduce what a
human was willing to write down.

Two things it cannot do. It cannot say "this answer is **worse**" — the cross-entropy loss only ever
says "this answer is right", so every piece of training signal must be a positive example. And for
open-ended generation — summarise this, explain this, be helpful here — writing the ideal response is
expensive and slow, while *comparing* two candidate responses is fast and most people agree on the
answer. RLHF is the machinery for turning that cheap comparison signal into gradients. This lecture
builds the signal; [Lec 39](../notes/week-08/39-rlhf-2-ppo.md) builds the optimiser.

### [Lec 39 — RLHF II: Policy Gradient and PPO](../notes/week-08/39-rlhf-2-ppo.md)

Lecture 38 left you with a reward model: a network that reads a prompt and a response and returns one
number saying how much a human would like it. It also left you with the obvious next question and no
answer to it — *how do you actually change the language model's weights to make that number go up?*

You cannot backpropagate through it. The reward is computed on a string of **sampled** tokens, and
sampling is not differentiable; nor is the reward model guaranteed to be differentiable with respect to
anything you control. So this lecture builds the optimiser from scratch: the policy-gradient estimator
that gets a gradient out of a non-differentiable reward, the variance reductions that make it usable,
and **PPO**, the algorithm that actually ships. It also installs the leash — the KL penalty — without
which the whole procedure produces high-reward gibberish.

### [Lec 40 — Aligning to User Preferences via Direct Preference Optimization](../notes/week-08/40-dpo.md)

Lecture 39 left you with a working but ugly machine. To run PPO you need four networks resident at
once — policy, frozen reference, reward model, value head — a sampling loop that regenerates
completions every few updates, and a clipping hyperparameter that, tuned badly, collapses the model.
Worse, the whole thing rests on a reward model that is itself a noisy fit to human labels; a bad
reward model quietly corrupts everything downstream.

This lecture removes the reward model. The observation is that the RLHF objective you already wrote
down has a **closed-form optimal policy** expressed in terms of the reward. Read that relation
backwards and you get the reward expressed in terms of the optimal policy. Substitute that into the
Bradley-Terry preference likelihood and every reward term cancels. What survives is an ordinary
supervised classification loss on preference pairs — no RL, no sampling, two models instead of four.
That is Direct Preference Optimization, and it closes the alignment arc.

## Week 9

### [Lec 41 — Prompting I](../notes/week-09/41-prompting-1.md)

Everything up to Week 8 assumed that using a pretrained model meant *changing* it. BERT got a
classification head and a fine-tuning run; instruction tuning updated every weight on a mixture of
tasks. Each new task cost you a training job, a labelled dataset, and a separate copy of the model.

This lecture removes all three. Once a model is large enough and has been instruction-tuned, you can
specify a task by *writing it down* — in the same text channel the model reads anyway — and read the
answer out of the generated continuation. The weights never move. That turns "add a task" from an
engineering project into an editing task, and it turns the task interface into a design problem: what
exactly do you write, how do you format demonstrations, and how do you convert free-running text back
into a label? Those three questions are this lecture.

### [Lec 42 — Prompting: Why Does In-Context Learning Work?](../notes/week-09/42-why-icl-works.md)

[Lec 29](../notes/week-06/29-gpt-decoder-pretraining.md) and [Lec 41](../notes/week-09/41-prompting-1.md) told you that
in-context learning happens: paste $k$ solved examples into the prompt and accuracy goes up. Neither
said *how*. That gap should bother you, because nothing about the model changed. The weights are
frozen. No gradient was computed, no optimiser step was taken, nothing was stored between the examples
and the question. Whatever "learning" means here, it is happening inside a single forward pass, and it
has to be implemented by matrix multiplications that were already there.

This lecture gives the leading mechanistic answer — the **induction head**, a two-attention-head
circuit that completes repeated patterns — and then undermines your confidence in the word "learning"
with two empirical results: shuffling the demonstrations can swing accuracy from state-of-the-art to
chance, and replacing every demonstration label with a *random* one barely hurts at all.

### [Lec 43 — Advanced Prompting Techniques](../notes/week-09/43-advanced-prompting.md)

Lecture 41 gave you a prompt template and a few demonstrations. Lecture 42 then showed that the thing
is fragile: reorder the demonstrations and accuracy swings, and the model is leaning on *format* more
than on the input–label mapping. That leaves two open questions, and this lecture answers both.

First: if which demonstrations you pick matters, how do you pick them? Second, and far more
consequential: a frozen model asked to answer a multi-step question in one shot has exactly one
forward pass per output token in which to do all the work, and it fails on problems a child could do.
The fix is not a bigger model or a fine-tune — it is to make the model *write its working out*. That
single idea, chain-of-thought, spawns the whole family of methods here: sampling many chains and
voting, searching a tree of partial chains, decomposing into sub-questions, and handing the arithmetic
to a Python interpreter. None of them touch a single weight.

### [Lec 44 — Tool-aided Language Models](../notes/week-09/44-tool-aided-lms.md)

Everything so far has tried to make the language model itself better — more parameters, better
prompts, chain-of-thought, instruction tuning, alignment. This lecture takes the opposite route: stop
asking the model to do things it is structurally bad at, and let it **call a program that is good at
them**.

The lecturer's framing is blunt. For complex reasoning the LM *struggles*; for real-world information
it is *fundamentally unable*. No amount of scale fixes the second one — the weights were frozen on a
particular day, and today's weather is not in them. So the lecture defines what a "tool" is for an LM,
shows the text-to-text interface that makes tool calls just more tokens, and then spends seven pages
on **Toolformer**, whose claim is that a model can teach *itself* where tool calls belong, with no
human labelling the positions.

### [Lec 45 — Automatic Prompt Engineering](../notes/week-09/45-automatic-prompt-engineering.md)

Everything in Week 9 so far has been a human writing a prompt by hand. Lec 41 gave you templates and
verbalizers, Lec 42 explained why in-context learning works at all, Lec 43 added chain-of-thought, Lec
44 bolted on tools. In all four, the prompt itself came out of someone's head.

That is the weak link. Hand-written prompts are brittle — rewording a template can move accuracy by
twenty points — unprincipled, and they do not transfer: a prompt tuned for BERT is worth little on
RoBERTa. If prompt wording is just another thing that gets *fit* to a task, it should be fit by
search, not by intuition.

This lecture replaces the human. It escalates through three levels: paraphrase the prompt and keep the
best; use *gradients* to search discrete token space; and finally abandon tokens entirely and learn
continuous **soft prompts** by backpropagation. That last step trains a tiny set of parameters while
the model stays frozen — which is exactly the premise of all of Week 10.

## Week 10

### [Lec 46 — Parameter-Efficient Fine-Tuning I: Adapters and Prefix-Tuning](../notes/week-10/46-peft-adapters-prefix.md)

Weeks 8 and 9 gave you two ways to make a pretrained model do your task. Full fine-tuning
([Lec 36](../notes/week-08/36-instruction-finetuning-1.md)) updates every weight and works well — but it
hands you a complete private copy of the model for each task. At 7 billion parameters that is 14 GB
per customer per task, and the copies cannot share a GPU. Prompting
([Lec 41](../notes/week-09/41-prompting-1.md)–[45](../notes/week-09/45-automatic-prompt-engineering.md)) touches
no weights at all, but pays the prompt's token cost on every single forward pass and is brittle to
wording.

This lecture is the synthesis. Freeze the backbone, train a *tiny* new set of parameters, and get
full-fine-tuning accuracy with roughly 1% of the storage. The deck organises the whole family into
three perspectives — function, input, parameter — and develops the first two. The third, LoRA, is
[Lec 47](../notes/week-10/47-lora-and-variants.md).

### [Lec 47 — LoRA and Its Variants](../notes/week-10/47-lora-and-variants.md)

Lecture 46 gave you two of the three ways to make fine-tuning cheap: bolt small **functions** into the
network (adapters), or push learned vectors into the **input** (prefix-tuning). Both work, and both
have the same structural defect — they change the shape of the computation. An adapter is an extra
sequential module every token must pass through, so it costs latency at inference forever, and a
prefix eats context length.

The third perspective asks a different question. Do not add anything to the network; instead ask what
the *weight update itself* looks like, and parameterise that cheaply. If $\Delta\mathbf{W}$ can be
written as the product of two thin matrices, you can train the thin matrices, then **add the product
back into the original weights** and ship a model that is byte-for-byte the same shape as the one you
started with. That is LoRA. It is the most widely deployed PEFT method in existence, the one every
downstream lecture in this week builds on, and the most examinable idea in Week 10.

### [Lec 48 — Quantization and QLoRA I](../notes/week-10/48-quantization-qlora-1.md)

Lecture 47 made fine-tuning cheap in *trainable parameters*: freeze $\mathbf{W}$, learn a rank-$r$
update $\mathbf{B}\mathbf{A}$, and you update a thousandth of the weights. But a frozen weight still
occupies memory. You still have to hold the whole base model on the GPU to run a forward pass through
it, and that is what actually stops you fine-tuning a 70-billion-parameter model on the card in your
desk.

So this lecture does the accounting properly. It asks where every byte of GPU memory goes during
fine-tuning, shows that LoRA deletes three of the four buckets but not the largest one, and then
attacks the remaining bucket directly: store the frozen base model at **4 bits per weight** instead of
16. That is QLoRA. The machinery that makes 4 bits work without destroying the model is
**quantization**, and in particular a data type called **NF4** built out of the quantiles of a normal
distribution.

### [Lec 49 — QLoRA II: Double Quantization and Paged Optimizers](../notes/week-10/49-qlora-2.md)

Lecture 48 shrank the frozen base model from 16 bits per weight to 4 with NF4, and stopped there. Two
things were left unsaid, and both are the difference between a method that works on a slide and a
method that works on your GPU.

The first is that quantization is not free. Every quantized block carries a **scale factor** stored at
full precision, and at realistic block sizes that bookkeeping costs you half a bit on every single
parameter — more than a tenth of your 4-bit budget. The second is that peak memory, not average
memory, is what crashes a training run: a single spike on one long batch kills a job that has been
running for three days. This lecture fixes both, then assembles the whole of QLoRA into one picture.

### [Lec 50 — Other Parameter Efficient Methods: Pruning, Distillation](../notes/week-10/50-pruning-and-distillation.md)

Week 10 has spent four lectures making a large model *cheaper to adapt*: adapters, LoRA, quantization.
Every one of those leaves the model the size it was. The deployed network still has 110 million or 70
billion parameters, and you still pay for all of them at every forward pass.

This lecture closes the week with the two remaining ways to make the model itself smaller. The deck
opens by putting all three side by side, and that comparison is the frame for everything that follows:
quantization makes each weight **cheaper to store**, pruning **deletes weights outright**, and
distillation **trains a different, smaller model** to behave like the big one. They are not
alternatives to each other — they compose — but they fail in different ways, and the exam tests
whether you can tell which is which.

## Week 11

### [Lec 51 — Scaling Laws of LLMs](../notes/week-11/51-scaling-laws.md)

By Lecture 29 you knew that GPT-3 has 175 billion parameters and that bigger models are better. What
you did not have was a way to *decide* anything. Given a fixed amount of GPU time — which is the only
situation anyone is ever actually in — should you train a big model on a little data, or a small model
on a lot? Until 2020 the honest answer was "nobody knows, try it". This lecture is the answer.

It turns out that test loss falls as a **power law** in model size, dataset size and compute, which
means the curves are *straight lines* on a log-log plot and can be extrapolated years ahead of the
hardware. The lecture gives you Kaplan et al.'s 2020 measurement of those lines, the arithmetic that
converts a GPU budget into a number of parameters and tokens, the methodological bug that made
Kaplan's conclusion wrong, and Hoffmann et al.'s 2022 correction — Chinchilla — which is the version
the field actually uses.

### [Lec 52 — Modern LLMs and Architecture Variations I](../notes/week-11/52-modern-llms-and-activations.md)

Lecture 51 told you *how big* to make a model and *how much* data to feed it. It said nothing about
what to actually build. This lecture answers that, in two halves that look unrelated and are not.

The first half is a census: which models exist, who made them, how many parameters, how many training
tokens, open or closed. That census is where the second half comes from — once you line up LLaMA,
Mistral, Qwen and PaLM side by side, the same five deviations from Vaswani et al.'s 2017 Transformer
appear in every row. None of them is a new idea about language. They are all cheapness: a gating
scheme that buys quality per FLOP, a normalisation that drops half its arithmetic, an activation that
is smooth where ReLU is not. A 2024 decoder block is the 2017 block with those substitutions made.
This chapter teaches the census so you can recognise the models, then the substitutions so you can
explain why a modern block looks the way it does.

### [Lec 53 — Modern Positional Embeddings: RoPE and ALiBi](../notes/week-11/53-positional-embeddings-rope-alibi.md)

You train a model on sequences of at most 2048 tokens because attention costs $O(n^2)$ and you cannot
afford more. Then you deploy it and someone pastes in a 30,000-token contract. What happens at
position 5000?

With the schemes [Lec 23](../notes/week-05/23-positional-encoding-and-encoder.md) taught, something bad. A
*learned* position embedding table has 2048 rows; row 5000 does not exist, so there is literally
nothing to add. A *sinusoidal* encoding is a closed-form function of position, so it technically
returns a vector for position 5000 — but the model has never seen that vector and attention scores
blow up. Both failures have the same root: they encode **absolute** position, while attention only
ever cares about how *far apart* two tokens are. This lecture fixes the root. It gives you four
modern schemes — T5's learned bias, ALiBi, RoPE, and the two ways to stretch RoPE past its training
length — and RoPE is the one every current open LLM uses.

### [Lec 54 — Long Sequence Modeling](../notes/week-11/54-long-sequence-modeling.md)

[Lec 53](../notes/week-11/53-positional-embeddings-rope-alibi.md) made *positions* extrapolate: RoPE and ALiBi let a
model index token 100,000 without the embedding table falling apart. That fixes the addressing. It does
not fix the arithmetic. Self-attention still compares every token with every other token, so the work
and the memory grow with the square of the length, and the KV cache still grows without bound as you
decode. A model that can *name* position 100,000 may still be unable to *afford* it.

This lecture is the architectural half of the answer. [Lec 25](../notes/week-05/25-efficient-transformers.md)
already thinned the attention matrix by making it sparse; here the deck goes further and removes the
$n \times n$ matrix altogether (linear attention), then removes the unbounded cache (fixed-size
caches), then adds an external, non-differentiable memory you can search (memorizing transformers).
The last of these quietly reinvents the RNN.

### [Lec 55 — Retrieval Augmented Generation](../notes/week-11/55-retrieval-augmented-generation.md)

Every lecture since Week 6 has made the model bigger and the training corpus larger, on the assumption
that knowledge should live in the weights. This lecture says that assumption is wrong for a large class
of facts, and it closes the retrieval arc that opened in Lec 31.

A model's weights are a lossy compression of its training corpus, frozen at the moment training
stopped. It cannot tell you where an answer came from, it cannot learn that Twitter changed CEO last
June, and it has seen rare facts too few times to store them. Lec 54 made contexts longer; this lecture
asks the complementary question — instead of carrying everything, why not *fetch* the few passages you
need, at query time, from a datastore you can edit? That is Retrieval Augmented Generation, and the
deck spends its last four pages on whether long contexts have made it unnecessary.

## Week 12

### [Lec 56 — Model Interpretability: Probing and the Logit Lens](../notes/week-12/56-interpretability-probing.md)

Fifty-five lectures have been spent making models work. This one asks what they are *doing*.

The question is forced on us by scale. A logistic regression has one weight per feature and you can
read the weights; a decision tree is a flowchart you can print. A 175-billion-parameter Transformer is
a stack of identical blocks, each shuffling a vector that means nothing to a human. The model gets the
answer right, and you have no account of why — which matters when you want to debug it, trust it, or
edit it.

So this lecture introduces the two oldest and most portable tools for looking inside. **Probing** asks
*what information is sitting in a layer* by training a tiny classifier on its frozen activations.
**The logit lens** asks *what the model would say if it stopped here* by applying the output
unembedding to intermediate states. Probing is the lens pointed sideways at representations; the logit
lens is the same model's own decoder pointed inward. Both are correlational, which is exactly the
limitation [Lec 58](../notes/week-12/58-interpretability-ffn-and-causal-tracing.md) exists to fix.

### [Lec 57 — Model Interpretability: Multilingual](../notes/week-12/57-interpretability-multilingual.md)

[Lec 30](../notes/week-06/30-domain-and-multilingual-pretraining.md) established *that* a single transformer
trained on a hundred languages can answer in all of them — mBERT, XLM-R, mT5, plus the curse of
multilinguality. It never said *how*. One model, one hundred languages: is there a separate French
machine hiding inside, or one machine that French is routed through?

This lecture answers with two pieces of evidence. The first, from the paper *Do Llamas Work in
English?*, takes a model prompted in French, answering in Chinese, and uses the logit lens of
[Lec 56](../notes/week-12/56-interpretability-probing.md) to read out what every intermediate layer is "thinking".
English shows up in the middle — in a prompt that contains no English at all. The second isolates
**neurons** that fire for one language and almost nothing else, and shows that switching off a few
hundred of them makes a 70B model answer a Chinese question in English.

### [Lec 58 — Interpretability III: FFN as Key-Value Memories and Causal Tracing](../notes/week-12/58-interpretability-ffn-and-causal-tracing.md)

Probing ([Lec 56](../notes/week-12/56-interpretability-probing.md)) tells you that a property is *decodable* from a
representation. It cannot tell you that the model *uses* it. A probe is a correlation: you fit a
classifier on frozen activations and report its accuracy. If the accuracy is high, the information is
present — but it might be a side effect the network never reads.

This lecture takes the next step. Instead of reading the model, you **intervene** on it: you break a
part, or repair a part, and watch what happens to the output. That is a causal claim, and it is the
only kind of claim that supports editing. The lecture does this twice. First it reinterprets the
feed-forward sublayer — half of every Transformer's parameters, and the half nobody looked at — as a
**key-value memory**. Then it localises a single stored fact to a specific layer and token with
**causal tracing**. Both results are the machinery [Lec 60](../notes/week-12/60-machine-unlearning.md) needs: you
cannot remove a fact until you know where it lives.

### [Lec 59 — Trustworthy LLMs: the Taxonomy](../notes/week-12/59-trustworthy-llms-taxonomy.md)

Weeks 8 through 11 built a capable model and then made it follow instructions and prefer what humans
prefer. None of that machinery ever named the *failures* it is supposed to prevent. "Alignment" was
treated as a single undifferentiated good, measured by a reward model that was itself a black box.

This lecture supplies the missing vocabulary. It names seven dimensions of trustworthiness and
twenty-nine concrete failure modes underneath them, and it walks through eight of those failures with real
transcripts from ChatGPT, GPT-3, GPT-4 and `text-davinci-003`. That taxonomy is the content: once you
can say which category a failure belongs to, you can say which tool fixes it.

It then poses the question that drives the final lecture. RLHF is the standard fix, but a full
preference-collection and PPO cycle takes weeks and needs human-written *good* answers. The deck's own
pivot — **"But can we do something quickly?"** — opens the door to unlearning, which needs only the
behaviour you want removed.

### [Lec 60 — Trustworthy LLMs: Machine Unlearning](../notes/week-12/60-machine-unlearning.md)

[Lec 59](../notes/week-12/59-trustworthy-llms-taxonomy.md) catalogued what can go wrong with a deployed LLM and named
unlearning as one of the repairs. It did not say how to do it. That is this lecture.

The problem is sharp. A model has memorised something it must not have — a copyrighted novel, a
person's private data, a recipe for harm. Retraining from scratch without that data is the only
guaranteed fix, and for a 7B model it costs roughly 184,000 GPU-hours. Unlearning asks whether you
can instead *edit* the trained model: run a few thousand gradient steps that remove the target
behaviour while leaving everything else intact. The lecture's answer is a single method built in
three attempts, each one a repair for a failure of the last, and the failures are the examinable
part. It closes the course, so the final pages step back to survey what comes next.
