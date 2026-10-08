# Lec 14 — Word Vectors: Other Extensions

> **Source:** `Week3.pdf` pp. 87–108 · **Week 3** · **Playlist:** Lec 14
> **Prereqs:** [Lec 13 — Negative Sampling and GloVe](13-negative-sampling-glove.md), [Lec 12 — word2vec Skip-gram](12-word2vec-skipgram.md)
> **Feeds into:** [Lec 15 — Cross-lingual Representations](15-cross-lingual-representations.md), [Lec 2 — Text Processing and Tokenization](../week-01/02-text-processing-tokenization.md)

## Why this lecture exists

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

## The ideas

A one-line reminder of where you are. [Lec 13](13-negative-sampling-glove.md) ended with **GloVe**,
which fits $\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$ to a global
co-occurrence matrix; the deck's page 89 is just the download page for Stanford's pre-trained GloVe
vectors, and the numbers on it are in the Exam pack because they are exactly the sort of thing an MCQ
keys on.

### 1. Phrase embeddings

Tokenize "New York Times" and word2vec sees three independent types. It will learn a vector for `New`
dominated by its use in `New Delhi`, `new year`, `brand new`; a vector for `York` pulled by `Yorkshire`
and `New York`; and nothing at all that denotes *the newspaper*. The meaning of the whole is not the
sum, the average, or any simple function of the parts, so no amount of post-hoc combination recovers it.

The fix is embarrassingly direct: **detect the phrase before training and replace it with a single
token**. `New_York_Times` then enters the vocabulary as one type and word2vec learns one vector for it,
in exactly the usual way. Nothing about the model changes — only the tokenization.

That shifts the whole problem onto detection. The deck's criterion:

![Slide "How to Learn Phrase Embeddings?": frequent pairs like "New York Times" are replaced by unique tokens while "this is" is left alone; candidate phrases are scored from unigram and bigram counts and thresholded, score(wi,wj) = (count(wi wj) − δ) / (count(wi) × count(wj))](../../assets/pages/lec14/p-090.png)
*Fig. — The discount $\delta$ in the numerator is what stops rare accidental pairs scoring highly; the product of unigram counts in the denominator is what stops frequent-but-meaningless pairs like "this is" scoring highly. Page 90 of `Week3.pdf`.*

$$\text{score}(w_i, w_j) = \frac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i) \times \text{count}(w_j)}$$

Read the two parts separately, because each is doing a distinct job.

- **The denominator makes this a pointwise-mutual-information-style statistic.** Recall PMI from
  [Lec 11](11-word-representation.md): $\text{PMI}(w_i,w_j) = \log \frac{P(w_i w_j)}{P(w_i)P(w_j)}$.
  Substituting counts over a corpus of $N$ tokens, $\frac{P(w_i w_j)}{P(w_i)P(w_j)} = \frac{N \cdot
  \text{count}(w_i w_j)}{\text{count}(w_i)\text{count}(w_j)}$. The deck's score is that ratio without
  the $\log$ and without the constant $N$ — neither of which changes the *ranking*, so thresholding
  this score and thresholding PMI select the same phrases. It asks "do these two words co-occur far
  more than their individual frequencies would predict?", which is why `this is` fails: both words are
  so common that the product in the denominator is enormous.
- **The discount $\delta$ kills low-count noise.** A bigram seen three times in a corpus has a tiny
  numerator but possibly a tiny denominator too, and the ratio can spike. Subtracting $\delta$ makes
  the score *negative* for any bigram with $\text{count}(w_i w_j) < \delta$, so it can never clear the
  threshold. $\delta$ is effectively a minimum-frequency floor smuggled into the numerator.

Phrases above a chosen threshold are merged into single tokens; everything else is left alone. The
procedure is usually run **two to four times** over the corpus with a decreasing threshold, so that
`New_York` forms on pass one and `New_York_Times` on pass two — that is how trigram and longer phrases
appear even though the score only ever looks at bigrams.

### 2. fastText: a word is a bag of character n-grams

The deck's framing (page 91): fastText comes from **Bojanowski et al., 2017**, and `fasttext.cc` is also a library for word embeddings and text classification from **Facebook AI Research**. Its stated motivation is that word2vec gives *a distinct vector representation for each word*, so it **cannot handle out-of-vocabulary (OOV) tokens**, and the extension is to *take subword information into account*.

word2vec's input matrix has one row per vocabulary type. That is both its simplicity and its two
failures:

1. **It wastes morphology.** `eat`, `eats`, `eating`, `eaten` get four unrelated rows. Each must learn
   its meaning from its own occurrences, so rare inflections are badly estimated even when the lemma
   is common.
2. **It cannot represent an out-of-vocabulary word at all.** There is no row to look up. The usual
   fallback is to map it to `<UNK>` or return a zero vector, neither of which carries any information
   about the actual word.

fastText's repair keeps the entire skip-gram-with-negative-sampling training loop from
[Lec 13](13-negative-sampling-glove.md) and changes only *what a word's vector is made of*.

![Slide "So, how does fastText work?": the binary-logistic negative log-likelihood log(1+e^{−s(wt,wc)}) + Σ_{n∈N_{t,c}} log(1+e^{s(wt,n)}), with the sum labelled "Negative samples"; this formulation ignores the internal structure of the word, so fastText denotes each word as a bag of character n-grams, shown as <eating> → <ea, eat, ati, tin, ing, ng>](../../assets/pages/lec14/p-092.png)
*Fig. — The loss is exactly skip-gram with negative sampling, written in its binary-logistic form. Everything fastText adds is hidden inside $s(w_t, w_c)$. Note the angle brackets `<` and `>` padding the word. Page 92.*

The objective on the slide is the negative log-likelihood for one (target, context) pair:

$$\log\!\left(1 + e^{-s(w_t, w_c)}\right) + \sum_{n \in \mathcal{N}_{t,c}} \log\!\left(1 + e^{\,s(w_t, n)}\right)$$

with $\mathcal{N}_{t,c}$ the sampled negatives. This is $-\log\sigma(s) - \sum \log\sigma(-s)$ rewritten
using $-\log\sigma(x) = \log(1+e^{-x})$ — the identical objective [Lec 13](13-negative-sampling-glove.md)
derives. **fastText does not introduce a new loss.** It redefines the score $s$.

#### Building the subword set

Pad the word with boundary symbols — `eating` becomes `<eating>` — then take every character
substring of length $n$ for $n = 3, 4, 5, 6$, and finally add the padded whole word as one more item.

![Slide table "So, how does fastText work?": all character n-grams between 3 and 6 characters are used. For "eating": n=3 → <ea, eat, ati, tin, ing, ng>; n=4 → <eat, eati, atin, ting, ing>; n=5 → <eati, eatin, ating, ting>; n=6 → <eatin, eating, ating>. We also include the word itself, <eating>](../../assets/pages/lec14/p-093.png)
*Fig. — Count them: 6 + 5 + 4 + 3 = 18 n-grams, plus the whole-word token `<eating>` = **19 items**. Also notice the 6-gram `eating` (no brackets) is a *different* entry from the whole-word token `<eating>` (with brackets). Page 93.*

The **boundary markers earn their place**. Without them, the trigram `her` extracted from `where`
would be identical to the trigram `her` extracted from `herself`, and the model could not tell a
word-initial `her` from a mid-word one. With padding, `herself` yields `<he`, `her`, `ers`, … while
`where` yields `<wh`, `whe`, `her`, `ere`, `re>` — the brackets let prefixes, suffixes and interiors
occupy distinct slots. This is why `<er>` (the standalone word *er*) never collides with the suffix
`er>` or the stem-internal `er`.

The fastText paper's own worked example is `where` at $n=3$: `<wh`, `whe`, `her`, `ere`, `re>`, plus
the special sequence `<where>`. Six items. The deck uses `eating` instead; both appear below.

#### The score, and why the sum is the whole trick

![Slide "So, how does fastText work?" asking how to calculate s(w,c); a quote from the fastText paper defines G_w as the set of n-grams in w, assigns a vector z_g to each n-gram g, represents a word by the sum of its n-gram vectors, and gives s(w,c) = Σ_{g∈G_w} z_g^T v_c](../../assets/pages/lec14/p-094.png)
*Fig. — The paper's own wording: "We represent a word by the **sum** of the vector representations of its n-grams." The context side $\mathbf{v}_c$ is an ordinary per-word vector — only the target side is decomposed. Page 94.*

Let $\mathcal{G}_w \subset \{1,\dots,G\}$ be the set of n-gram indices occurring in $w$, and let each
n-gram $g$ carry a learned vector $\mathbf{z}_g$. Then a word's representation is

$$\mathbf{e}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g
\qquad\text{and}\qquad
s(w, c) = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c$$

Those two expressions are the same statement, because the dot product is linear:
$\left(\sum_g \mathbf{z}_g\right)^\top \mathbf{v}_c = \sum_g \mathbf{z}_g^\top \mathbf{v}_c$. Compare
with plain skip-gram, where $s(w,c) = \mathbf{e}_w^\top \mathbf{v}_c$ and $\mathbf{e}_w$ is a single
looked-up row. **The only change is that the target vector is now assembled, not looked up.**

Three consequences follow immediately and all three are examinable:

- **Parameters are shared across words.** Every word containing `ing` updates $\mathbf{z}_{\texttt{ing}}$.
  A rare word gets gradient signal from the frequent words it shares substrings with, so its vector is
  far better estimated than word2vec could manage.
- **Morphologically rich languages benefit most.** Turkish, Finnish, Hungarian, Tamil and Sanskrit
  generate dozens or hundreds of surface forms per lemma by agglutination; word2vec must learn each
  form from its own sparse counts, while fastText sees them all share a stem n-gram. The original
  paper's biggest gains are on German, Czech, Russian and Arabic, and the smallest on English — which
  has almost no inflection.
- **The word's own token is still there.** Because `<eating>` is in $\mathcal{G}_w$, a frequent word
  can still learn an idiosyncratic, non-compositional component that its substrings do not explain.
  Without it, `New_York` would be forced to equal the sum of its character n-grams.

#### Hashing: bounding the n-gram table

The number of distinct character n-grams over a large corpus is far larger than the word vocabulary —
millions of them, most seen a handful of times. Storing a vector for each is unaffordable.

![Slide "So, how does fastText work?": the unique n-grams can be huge, so hashing bounds memory — instead of one embedding per unique n-gram, learn B embeddings where B is the bucket size; the paper used 2 million buckets. A diagram maps a K-entry unique dictionary to a B-entry hashed dictionary, with "ing" hashed to bucket index 10](../../assets/pages/lec14/p-095.png)
*Fig. — $B = 2{,}000{,}000$ buckets is the paper's setting and a very likely MCQ number. Two different n-grams hashing to the same bucket **share one vector** — an accepted collision cost, not a bug. Page 95.*

Each n-gram string is passed through a hash function (FNV-1a in the released code) and reduced modulo
$B$. The embedding table has exactly $B$ rows regardless of how many distinct n-grams the corpus
contains, so memory is fixed in advance. Collisions mean two unrelated subwords are forced to share a
vector; with $B$ in the millions this is rare enough that it costs little accuracy. Note that **word
types are stored separately** from the hashed buckets — the whole-word tokens for in-vocabulary words
get their own rows.

#### OOV handling — the headline property

![Slide "How does fastText handle OOV words?": a heatmap of cosine similarity between the character n-grams of the OOV word "interlink" (x axis) and those of "connect" (y axis), red for positive and blue for negative, with the caption "simply average the vector representation of its n-grams"](../../assets/pages/lec14/p-096.png)
*Fig. — The figure shows that even for words never seen in training, the subword vectors carry real signal: `erlink` / `link` in "interlink" aligns strongly with `connec` / `nnect` in "connect", because both subwords occurred in other training words with related meanings. Page 96.*

This is the property to be able to state cold:

| Model | A word seen in training | A word **never** seen in training |
|---|---|---|
| **word2vec** | look up its row in the input matrix | **nothing** — no row exists; fall back to `<UNK>` or a zero vector |
| **GloVe** | look up its row | **nothing** — it has no row in the co-occurrence matrix either |
| **fastText** | sum its n-gram vectors **plus** its own whole-word vector | **sum its n-gram vectors** — the whole-word token is missing, but the subwords are not |

The mechanism in one sentence: **an unseen word is still made of seen character n-grams, and fastText
has a vector for every n-gram, so it can build a vector for any string whatsoever.** Confronted with
`tensorflowify`, word2vec returns `<UNK>`; fastText returns a vector built from `<te`, `ten`, `ens`,
… , `ify`, `fy>` and lands somewhere sensible near other `-ify` verbs.

> **Sum or average?** The paper (and page 94) says **sum**. The deck's OOV slide (page 96) says
> "average". They differ only by a constant factor $1/|\mathcal{G}_w|$, which changes vector *norms*
> but not cosine similarities, so rankings are unaffected. If an exam question quotes the formula,
> answer **sum**; if it quotes the OOV slide's wording, answer average and know they are the same up
> to scaling. This book uses **sum**.

#### fastText's character n-grams versus BPE

Both exist to handle rare and unseen words, and they are **different mechanisms**. One sentence each,
because [Lec 2](../week-01/02-text-processing-tokenization.md) owns subword tokenization:

| | fastText character n-grams | BPE / WordPiece ([Lec 2](../week-01/02-text-processing-tokenization.md)) |
|---|---|---|
| What it does | **represents** a word; the token sequence is untouched | **segments** a word; it replaces the token sequence |
| Chosen by | fixed rule: every substring of length 3–6 | learned merges driven by corpus frequency |
| Pieces per word | many, heavily **overlapping** (19 for `eating`) | few, **disjoint**, covering the word exactly once |
| Combined by | summed into one word vector | kept as separate tokens the model consumes in order |
| Reversible? | no — you cannot read the word off its n-gram bag | yes — concatenating the pieces rebuilds the word |

The MCQ discrimination: **BPE changes the tokenization, fastText changes the embedding.** A BPE-based
Transformer never produces a single vector for `eating` at all; fastText always does.

### 3. Beyond words: sentence and document embeddings

![Slide "Beyond words: Sentence and documents": mean embeddings, then tf-idf-weighted averages, then the question of learning larger units directly; side-by-side diagrams of word2Vec (word matrix feeding "the", "cat", "sat" into average/concatenate then a classifier predicting "on") and doc2Vec, which adds an orange paragraph matrix D indexed by paragraph id](../../assets/pages/lec14/p-097.png)
*Fig. — The only architectural difference is the orange box: one extra vector, looked up by paragraph ID, concatenated alongside the word vectors. Everything else is word2vec. Le and Mikolov, 2014. Page 97.*

Two approaches, in the order the deck gives them:

**Compose the word vectors.** The default after word2vec was the **mean embedding**: average the
vectors of the words in the sentence. Cheap, surprisingly strong, and completely blind to word order —
*"dog bites man"* and *"man bites dog"* get identical vectors. A refinement weights the average by
**tf-idf** so that content words count more than function words, since `the` contributes to every
sentence equally and therefore carries no discriminative information.

**Learn the larger unit directly — doc2vec** (Le and Mikolov, *Distributed Representations of
Sentences and Documents*, 2014). Add a **paragraph matrix** $\mathbf{D}$ with one row per document,
and when predicting a word from its context, concatenate (or average) the paragraph vector in with the
context word vectors. The paragraph vector acts as a **memory of what the document is about** —
information the local context window does not have. Training is unchanged otherwise.

The practical wrinkle worth knowing: at test time a *new* document has no row in $\mathbf{D}$, so you
must **infer** its vector — freeze all word vectors and run gradient descent on that one new row until
it converges. Unlike a word2vec lookup, embedding an unseen document costs an optimisation.

### 4. Hierarchical representations

![Slide "Hierarchical Representations": a WordNet taxonomy embedded in a Poincaré disc, with general concepts (Entity, Abstraction, Matter) near the centre and specific ones (Sperm Whale, Soft Pretzel, Penicillium) near the rim, beside the title page of Nickel and Kiela, "Poincaré Embeddings for Learning Hierarchical Representations"](../../assets/pages/lec14/p-098.png)
*Fig. — Look at the layout: "Entity" sits near the centre, "Soft Pretzel" and "Sperm Whale" at the rim. Depth in the taxonomy maps to distance from the origin. Page 98.*

Some vocabularies are *trees*: `mammal` is a `vertebrate` is an `animal` is an `entity`. A tree with
branching factor $b$ has $O(b^\ell)$ nodes at depth $\ell$, and you cannot pack exponentially many
roughly-equidistant points into a low-dimensional Euclidean ball — so representing a deep taxonomy in
$\mathbb{R}^d$ needs $d$ to grow with the depth.

**Poincaré embeddings** (Nickel and Kiela, Facebook AI Research) embed into **hyperbolic** space
instead — the open unit ball with a metric that makes distance grow without bound as you approach the
boundary. Volume grows exponentially with radius there, which is exactly the growth rate a tree needs.
The practical payoff: hierarchies that need hundreds of Euclidean dimensions fit in **5 to 10
hyperbolic dimensions**, and the geometry encodes the hierarchy for free — *distance from the origin
is generality*, so a node's norm tells you its depth in the taxonomy.

### 5. Node embeddings and graph representation learning

The deck's stated goal (page 99): **efficient, task-independent feature learning for machine learning with graphs** — a single learned map $f: u \to \mathbb{R}^d$ from a node to a $d$-dimensional feature representation.

Machine learning on graphs traditionally meant hand-engineering features per node — degree,
clustering coefficient, PageRank — and re-engineering them for every new task. Node embeddings replace
that with one learned map $f: u \to \mathbb{R}^d$.

$$\text{similarity}(u,v) \;\approx\; \mathbf{z}_u^\top \mathbf{z}_v$$

Everything hinges on what "similarity in the network" means. Adjacency is too crude — two nodes with
no edge between them but twenty shared neighbours are obviously similar. Shortest-path distance is
expensive and still arbitrary.

#### DeepWalk: random walks turn a graph into a corpus

![Slide "Graph Representation Learning": a 12-node graph with a start node and red arrows for Steps 1–5 tracing 5 → 8 → 9 → 8 → 11; text explains that repeatedly moving to a uniformly random neighbour produces a random walk on the graph, and that z_u^T z_v should approximate the probability that u and v co-occur on a random walk](../../assets/pages/lec14/p-101.png)
*Fig. — Follow the red arrows. Note Step 3 goes 8 → 9 and Step 4 comes straight back 9 → 8: a walk may revisit nodes, exactly as a sentence may repeat a word. Page 101.*

**DeepWalk's idea, and the reason this subsection is in a *natural language processing* course:**

1. Start at a node. Repeatedly jump to a uniformly random neighbour. Record the sequence.
2. **Call that sequence a sentence** and the nodes in it **words**.
3. Run skip-gram on the resulting "corpus".

That is the whole algorithm. The similarity the red box asked for is defined implicitly: two nodes are
similar if they **co-occur on random walks**, which is the graph-theoretic analogue of
*distributional similarity* — [Lec 11](11-word-representation.md)'s "you shall know a word by the
company it keeps", with neighbours standing in for context words. Random walks capture both immediate
neighbourhood and higher-order structure at a controllable cost, and you never have to materialise an
$|V| \times |V|$ similarity matrix.

![Slide "Graph Representation Learning: DeepWalk": the loss L = Σ_{u∈V} Σ_{v∈N_R(u)} −log( exp(z_u^T z_v) / Σ_{n∈V} exp(z_u^T z_n) ), annotated as a sum over all nodes u, a sum over nodes v seen on random walks from u, and the predicted probability of u and v co-occurring on a random walk](../../assets/pages/lec14/p-102.png)
*Fig. — Put this beside the skip-gram objective from [Lec 12](12-word2vec-skipgram.md) and they are the same formula: a softmax over the whole vocabulary of a dot product, summed over (target, context) pairs. Only the names changed — $N_R(u)$ replaces the context window. Page 102.*

$$\mathcal{L} = \sum_{u \in V} \; \sum_{v \in N_R(u)} -\log \frac{\exp(\mathbf{z}_u^\top \mathbf{z}_v)}{\sum_{n \in V} \exp(\mathbf{z}_u^\top \mathbf{z}_n)}$$

where $N_R(u)$ is the multiset of nodes seen on random walks started from $u$. Here $V$ is the node
set, playing the role the vocabulary plays in text.

The denominator sums over every node in the graph, so a naïve implementation is $O(|V|^2)$ per epoch —
the identical blow-up skip-gram has, with the identical fix. DeepWalk uses **hierarchical softmax**;
its successor **node2vec** uses **negative sampling**. Both are derived in
[Lec 13](13-negative-sampling-glove.md) and neither needs re-deriving here: *the graph problem has
been reduced to the word2vec you already know.* That reduction is the elegance, and it is the single
thing to remember about DeepWalk.

(node2vec, the standard follow-up, keeps everything but replaces the uniform random walk with a
**biased** second-order walk whose return parameter $p$ and in-out parameter $q$ interpolate between
breadth-first exploration — structural equivalence — and depth-first — community membership.)

### 6. Knowledge graph embeddings and TransE

A **knowledge graph** stores facts as **triples** $(h, r, t)$ — head entity, relation, tail entity.
`(J.K. Rowling, genre, Science Fiction)`. Nodes are entities, edges are typed relations.

The deck names five public KGs — **FreeBase, Wikidata, DBpedia, YAGO, NELL** — and two shared characteristics that are the entire motivation for embedding them (page 103): they are **massive** (millions of nodes and edges) and **incomplete** (many true edges are missing). Enumerating all possible facts over a massive KG is intractable, so the question becomes: *can we predict plausible but missing links?*

Page 104 states the task precisely: given a $(\text{head}, \text{relation})$ pair, **predict the
missing tail**. The deck's example is the triple `(J.K. Rowling, genre, ?)` with the answer
`Science Fiction` — an edge the graph does not contain but should. The slide adds a parenthetical that
is a classic trap: this is **slightly different from link prediction**, because link prediction asks
*is there an edge between these two nodes*, while KG completion asks *which of the millions of
entities belongs in this slot, for this specific relation type*. It is a ranking problem over entities,
not a binary decision over pairs.

#### TransE: relations are translations

![Slide "Knowledge Graph Embeddings: TransE": for a triple (h,r,t) with embeddings h, r, t in R^k, TransE requires h + r ≈ t if the link exists and h + r ≠ t otherwise; entity scoring function f_r(h,t) = −||h + r − t||. Two plots show an arrow r carrying h to t, and an arrow "Nationality" carrying "Obama" to "U.S.A"](../../assets/pages/lec14/p-105.png)
*Fig. — The geometry is the definition. The relation is a **fixed displacement vector**: the same arrow "Nationality" added to any person should land on their country. Page 105.*

Embed entities **and relations** in the same $\mathbb{R}^k$ (the deck writes the dimension $k$; this
book writes $d$ elsewhere). TransE's constraint:

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t} \quad \text{if } (h,r,t) \text{ is true,} \qquad
\mathbf{h} + \mathbf{r} \not\approx \mathbf{t} \quad \text{otherwise}$$

and the **scoring function**

$$f_r(h,t) = -\lVert \mathbf{h} + \mathbf{r} - \mathbf{t} \rVert$$

**Watch the sign — this is the single most common TransE error.** The *distance*
$d(\mathbf{h}+\mathbf{r},\mathbf{t}) = \lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ is **low** for a
true triple. The *score* is the **negated** distance, so it is **high** for a true triple. An exam
question asking "what does TransE optimise?" wants: *minimise the distance
$\lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ for true triples and maximise it for corrupted ones.*

The appeal is that it recovers the analogy structure of [Lec 11](11-word-representation.md) on a
*database*: $\mathbf{Obama} + \mathbf{nationality} \approx \mathbf{USA}$ is the same kind of statement
as $\mathbf{king} - \mathbf{man} + \mathbf{woman} \approx \mathbf{queen}$. A missing tail is then
predicted by computing $\mathbf{h} + \mathbf{r}$ and returning the **nearest entity embedding**.

#### TransE: how to learn

![Slide "TransE: How to Learn?": Algorithm 1 initialises relation and entity vectors uniformly in (−6/√k, 6/√k) and normalises them; each loop renormalises every entity embedding, samples a minibatch, samples a corrupted triple (h',r,t') for each (h,r,t), and updates on the gradient of Σ [γ + d(h+r, t) − d(h'+r, t')]_+, annotated as a contrastive loss favouring low distance for valid triples and high distance for corrupted ones](../../assets/pages/lec14/p-106.png)
*Fig. — Read line 9: the corrupted triple replaces the head **or** the tail with a random entity, never the relation. Read line 5: entity embeddings are **renormalised to unit norm every iteration**. Both are examinable details. Page 106.*

The loss is a **margin-based ranking loss** over positive/corrupted pairs:

$$\mathcal{L} = \sum_{\left((h,r,t),\,(h',r,t')\right) \in T_{\text{batch}}}
\left[\, \gamma + d(\mathbf{h}+\mathbf{r},\, \mathbf{t}) \;-\; d(\mathbf{h}'+\mathbf{r},\, \mathbf{t}') \,\right]_+$$

where $[x]_+ = \max(0, x)$ and $\gamma > 0$ is the **margin** hyperparameter. Unpack it:

- The **positive** term $d(\mathbf{h}+\mathbf{r},\mathbf{t})$ enters with a $+$, so minimising
  $\mathcal{L}$ pushes true triples' distances **down**.
- The **negative** term $d(\mathbf{h}'+\mathbf{r},\mathbf{t}')$ enters with a $-$, so it pushes
  corrupted triples' distances **up**.
- The hinge $[\cdot]_+$ means the loss is **exactly zero** once
  $d_{\text{neg}} \ge d_{\text{pos}} + \gamma$. Already-well-separated triples contribute no gradient
  at all; training concentrates entirely on the ones still violating the margin. This is what "margin"
  buys you: without it ($\gamma = 0$) the model would be satisfied by an arbitrarily small gap, and
  with no hinge it would waste capacity pushing already-correct triples further apart forever.

The rest of the algorithm, line by line:

| Line | What it does | Why |
|---|---|---|
| 1, 3 | initialise $\mathbf{r}, \mathbf{e} \sim \text{uniform}(-\tfrac{6}{\sqrt{k}}, \tfrac{6}{\sqrt{k}})$ | Xavier-style scaling; variance stays sane as $k$ grows ([Lec 10](../week-02/10-gradient-descent-and-init.md)) |
| 2 | normalise relation vectors once | puts all relations on a comparable scale at the start |
| 5 | **renormalise every entity to unit norm each iteration** | without it the model trivially shrinks all embeddings toward $\mathbf{0}$, driving every distance to zero and the loss to $\gamma$ with no learning |
| 9 | sample a **corrupted** triple $(h', r, t')$ not in the KG | KGs store only positive facts; negatives must be manufactured — the same problem negative sampling solves in [Lec 13](13-negative-sampling-glove.md) |
| 12 | gradient step on the hinge loss | — |

**TransE's known weakness**, which the deck does not state but an exam might: because $\mathbf{r}$ is a
single fixed displacement, a **one-to-many** relation is impossible to satisfy. If
`(Rowling, genre, Fantasy)` and `(Rowling, genre, Tragicomedy)` are both true, then
$\mathbf{h}+\mathbf{r}$ must equal two different tails at once, forcing
$\mathbf{t}_{\text{Fantasy}} = \mathbf{t}_{\text{Tragicomedy}}$. Symmetric relations fail too: $(h,r,t)$
and $(t,r,h)$ both true forces $\mathbf{r} = \mathbf{0}$. TransH, TransR and RotatE exist to fix this.

### Everything here is still static

One forward-pointing sentence, since it frames all of Week 3: every model in this chapter assigns a
word **one vector, independent of the sentence it appears in** — `bank` gets a single blend of
riverbank and savings-bank no matter the context. fastText softens the *vocabulary* problem but not
the *polysemy* problem. The fix is **contextual embeddings**, where the vector is computed from the
whole sentence at inference time, and that begins with ELMo and BERT in
[Lec 26](../week-06/26-pretraining-and-elmo.md) and [Lec 27](../week-06/27-bert-masked-lm.md).

## Worked numericals

No "Try this problem" page appears in `Week3.pdf` pp. 87–108; the six below are built from the deck's
own examples and formulas.

### N1. Enumerate fastText's character n-grams and count them
**Given:** the deck's word `eating` (page 93), and the paper's word `where`.
**Find:** (a) the 3-grams of `eating`; (b) all n-grams for $n = 3$–$6$ and the total item count;
(c) the 3-grams of `where`; (d) a general formula.

1. Pad with boundary symbols: `eating` → `<eating>`, which has $6 + 2 = 8$ characters.
2. $n = 3$: slide a 3-character window over 8 characters → $8 - 3 + 1 = 6$ substrings:
   `<ea`, `eat`, `ati`, `tin`, `ing`, `ng>`. ✓ matches the deck's row.
3. $n = 4$: $8 - 4 + 1 = 5$ → `<eat`, `eati`, `atin`, `ting`, `ing>`. ✓
4. $n = 5$: $8 - 5 + 1 = 4$ → `<eati`, `eatin`, `ating`, `ting>`. ✓
5. $n = 6$: $8 - 6 + 1 = 3$ → `<eatin`, `eating`, `ating>`. ✓
6. Subtotal: $6 + 5 + 4 + 3 = 18$ n-grams.
7. Add the whole-word token `<eating>`: $18 + 1 = 19$ items in $\mathcal{G}_w$.
8. For `where`: padded `<where>` has $5 + 2 = 7$ characters, so $7 - 3 + 1 = 5$ trigrams —
   `<wh`, `whe`, `her`, `ere`, `re>` — plus `<where>` = **6 items** at $n=3$ only.
9. General: for a word of length $L$, the padded form has $L+2$ characters and contributes
   $\max(0,\, L + 3 - n)$ n-grams of length $n$. Total over $n = 3 \ldots 6$ plus one whole-word token:
   $$|\mathcal{G}_w| = 1 + \sum_{n=3}^{6} \max(0,\, L + 3 - n)$$
10. Check on `eating`, $L=6$: $1 + (9{-}3) + (9{-}4) + (9{-}5) + (9{-}6) = 1 + 6+5+4+3 = 19$. ✓
11. **Trap — short words.** For `cat`, $L=3$, padded `<cat>` is 5 characters: $n=3$ gives
    `<ca`,`cat`,`at>` (3); $n=4$ gives `<cat`,`cat>` (2); $n=5$ gives `<cat>` (1); $n=6$ gives none.
    The single 5-gram `<cat>` **is** the whole-word token, so the union has
    $3 + 2 + 1 = 6$ distinct items, not 7. The formula over-counts by 1 whenever $L + 2 \le 6$.

**Answer:** `eating` → 6 trigrams; **19 items** in total for $n=3$–$6$ including the whole-word token.
`where` at $n=3$ → **6 items**. $\mathbf{e}_{\texttt{eating}} = \sum_{g \in \mathcal{G}_w}\mathbf{z}_g$,
a sum of 19 vectors.

### N2. Embedding an OOV word with fastText, and what word2vec returns
**Given:** a fastText model trained with $n = 3$ only, $d = 2$, whose n-gram table contains

| $g$ | `<be` | `bea` | `eat` | `ati` | `tin` | `ing` | `ng>` |
|---|---|---|---|---|---|---|---|
| $\mathbf{z}_g$ | $(0.2, -0.1)$ | $(0.4, 0.3)$ | $(1.0, 0.5)$ | $(-0.3, 0.8)$ | $(0.6, -0.4)$ | $(0.1, 1.2)$ | $(-0.5, 0.2)$ |

The word vocabulary contains `eating` but **not** `beating`.
**Find:** fastText's vector for the OOV word `beating`, and word2vec's.

1. Pad: `beating` → `<beating>`, length $7 + 2 = 9$.
2. 3-grams: $9 - 3 + 1 = 7$ → `<be`, `bea`, `eat`, `ati`, `tin`, `ing`, `ng>`.
3. The whole-word token `<beating>` has **no** trained vector (the word is OOV), so it is dropped.
   All seven 3-grams **do** have vectors — five of them were trained on `eating` itself.
4. Sum the first coordinates: $0.2 + 0.4 + 1.0 + (-0.3) + 0.6 + 0.1 + (-0.5)$
   $= 0.6 + 1.0 = 1.6;\; 1.6 - 0.3 = 1.3;\; 1.3 + 0.6 = 1.9;\; 1.9 + 0.1 = 2.0;\; 2.0 - 0.5 = \mathbf{1.5}$.
5. Sum the second coordinates: $-0.1 + 0.3 + 0.5 + 0.8 + (-0.4) + 1.2 + 0.2$
   $= 0.2 + 0.5 = 0.7;\; 0.7 + 0.8 = 1.5;\; 1.5 - 0.4 = 1.1;\; 1.1 + 1.2 = 2.3;\; 2.3 + 0.2 = \mathbf{2.5}$.
6. $\mathbf{e}_{\texttt{beating}} = (1.5,\, 2.5)$.
7. Under the deck's "average" wording: $(1.5, 2.5)/7 = (0.214,\, 0.357)$ — same direction, norm scaled
   by $1/7$, so every cosine similarity is identical.
8. **word2vec**: `beating` has no row in the input matrix. There is no vector to return. The model
   maps it to `<UNK>` or emits $\mathbf{0}$ — it has learned nothing about this word.

**Answer:** fastText gives $\mathbf{e}_{\texttt{beating}} = (1.5,\, 2.5)$ (or $(0.214, 0.357)$
averaged); word2vec and GloVe give **nothing**. Note that five of the seven subwords were trained on
`eating`, so the OOV vector automatically lands near it.

### N3. A TransE score and the margin loss
**Given:** $d = 2$, Euclidean ($L_2$) distance, margin $\gamma = 2$. True triple $(h, r, t)$ with
$\mathbf{h} = (1,2)$, $\mathbf{r} = (3,1)$, $\mathbf{t} = (4,2)$. Two corruptions: a tail corruption
$\mathbf{t}' = (1,-1)$, and a head corruption $\mathbf{h}' = (3,3)$.
**Find:** $f_r(h,t)$, both corrupted distances, and the loss term for each pair.

1. $\mathbf{h} + \mathbf{r} = (1+3,\; 2+1) = (4, 3)$.
2. $\mathbf{h} + \mathbf{r} - \mathbf{t} = (4-4,\; 3-2) = (0, 1)$.
3. $d_{\text{pos}} = \lVert(0,1)\rVert = \sqrt{0^2 + 1^2} = \mathbf{1.0}$.
4. Score: $f_r(h,t) = -d_{\text{pos}} = \mathbf{-1.0}$ (negative distance — high score means plausible).
5. **Tail corruption** $(h, r, t')$: $\mathbf{h}+\mathbf{r}-\mathbf{t}' = (4-1,\; 3-(-1)) = (3, 4)$,
   so $d_{\text{neg}} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5.0}$.
6. Loss: $[\gamma + d_{\text{pos}} - d_{\text{neg}}]_+ = [2 + 1.0 - 5.0]_+ = [-2.0]_+ = \mathbf{0}$.
   The margin constraint $d_{\text{neg}} \ge d_{\text{pos}} + \gamma$ reads $5.0 \ge 3.0$ ✓ —
   **satisfied, no gradient.**
7. **Head corruption** $(h', r, t)$: $\mathbf{h}'+\mathbf{r} = (3+3,\; 3+1) = (6,4)$;
   subtract $\mathbf{t}$: $(6-4,\; 4-2) = (2,2)$; $d_{\text{neg}} = \sqrt{4+4} = \sqrt{8} = \mathbf{2.828}$.
8. Loss: $[2 + 1.0 - 2.828]_+ = [0.172]_+ = \mathbf{0.172}$. Constraint reads $2.828 \ge 3.0$ ✗ —
   **violated**, so this pair contributes gradient that pushes $\mathbf{h}'+\mathbf{r}$ away from
   $\mathbf{t}$ and $\mathbf{h}+\mathbf{r}$ toward $\mathbf{t}$.
9. Batch loss for these two pairs: $0 + 0.172 = \mathbf{0.172}$.

**Answer:** $f_r(h,t) = -1.0$; tail-corrupted loss $= 0$ (margin satisfied), head-corrupted loss
$= 0.172$ (margin violated). Only the second pair trains the model — the hinge is what makes training
focus on hard negatives.

### N4. A DeepWalk random walk turned into skip-gram pairs
**Given:** an undirected graph on $\{1,\dots,6\}$ with edges
$1\!-\!2,\; 1\!-\!3,\; 2\!-\!3,\; 3\!-\!4,\; 4\!-\!5,\; 4\!-\!6,\; 5\!-\!6$. Context window $c = 1$.

```
    2 ──── 3 ──── 4 ──── 5
    │    ╱        │  ╲   │
    │   ╱         │   ╲  │
    1 ─╯          └──── 6
```

**Find:** the degree of each node, the probability of the specific walk
$1 \to 3 \to 4 \to 5 \to 6 \to 4$, and the skip-gram training pairs it produces.

1. Adjacency and degrees: $N(1)=\{2,3\}$, $\deg 1 = 2$; $N(2)=\{1,3\}$, $\deg 2 = 2$;
   $N(3)=\{1,2,4\}$, $\deg 3 = 3$; $N(4)=\{3,5,6\}$, $\deg 4 = 3$; $N(5)=\{4,6\}$, $\deg 5 = 2$;
   $N(6)=\{4,5\}$, $\deg 6 = 2$.
2. A uniform random walk picks the next node with probability $1/\deg(\text{current})$.
3. $P(\text{walk}) = \tfrac{1}{\deg 1}\cdot\tfrac{1}{\deg 3}\cdot\tfrac{1}{\deg 4}\cdot\tfrac{1}{\deg 5}\cdot\tfrac{1}{\deg 6}
   = \tfrac12 \cdot \tfrac13 \cdot \tfrac13 \cdot \tfrac12 \cdot \tfrac12$.
4. $\tfrac12 \cdot \tfrac13 = \tfrac16$; $\;\tfrac16 \cdot \tfrac13 = \tfrac1{18}$;
   $\;\tfrac1{18}\cdot\tfrac12 = \tfrac1{36}$; $\;\tfrac1{36}\cdot\tfrac12 = \tfrac{1}{72} \approx \mathbf{0.0139}$.
5. Treat the walk as the "sentence" `1 3 4 5 6 4` and emit (target, context) pairs within $c=1$:

| Position | Target | Contexts | Pairs |
|---|---|---|---|
| 1 | 1 | 3 | (1,3) |
| 2 | 3 | 1, 4 | (3,1), (3,4) |
| 3 | 4 | 3, 5 | (4,3), (4,5) |
| 4 | 5 | 4, 6 | (5,4), (5,6) |
| 5 | 6 | 5, 4 | (6,5), (6,4) |
| 6 | 4 | 6 | (4,6) |

6. Count: $1 + 2 + 2 + 2 + 2 + 1 = \mathbf{10}$ pairs.
7. General rule for a walk of length $T$ with window $c$: the interior $T - 2c$ positions give $2c$
   pairs each and the $2c$ edge positions give fewer; here $T=6$, $c=1$ → $2(T-1) = 10$. ✓
8. These pairs feed the **identical** skip-gram objective from [Lec 12](12-word2vec-skipgram.md),
   trained with hierarchical softmax (DeepWalk) or negative sampling (node2vec).

**Answer:** $P(\text{walk}) = 1/72 \approx 0.0139$; the walk yields **10 skip-gram pairs**. Node 4
appears as a target three times, so it accumulates the most gradient — exactly as a frequent word does.

### N5. Parameter count: fastText vs word2vec at the same vocabulary
**Given:** $|V| = 200{,}000$ word types, embedding dimension $d = 300$, fastText bucket size
$B = 2{,}000{,}000$ (the paper's value, page 95). Both models keep separate input and output matrices
([Lec 13](13-negative-sampling-glove.md)).
**Find:** total parameters for each, and the ratio.

1. **word2vec** input matrix: $|V| \times d = 200{,}000 \times 300 = 60{,}000{,}000$.
2. word2vec output (context) matrix: another $60{,}000{,}000$.
3. word2vec total $= 120{,}000{,}000 = \mathbf{120\text{M}}$.
4. **fastText** input side holds word rows **plus** hashed n-gram buckets:
   $(|V| + B) \times d = (200{,}000 + 2{,}000{,}000) \times 300 = 2{,}200{,}000 \times 300
   = 660{,}000{,}000$.
5. fastText output matrix is per-word as usual: $|V| \times d = 60{,}000{,}000$.
6. fastText total $= 660\text{M} + 60\text{M} = \mathbf{720\text{M}}$.
7. Ratio $= 720/120 = \mathbf{6\times}$.
8. The buckets alone are $600\text{M}/720\text{M} = 83\%$ of fastText's parameters, and $B$ is a
   **fixed** budget — it does not grow with vocabulary or corpus. At $|V| = 2{,}000{,}000$ word2vec
   would need 1.2B parameters while fastText's bucket term stays at 600M.

**Answer:** word2vec **120M**, fastText **720M**, a **6×** increase — the price of OOV capability. The
hashing trick is what caps it: without buckets you would need one row per distinct n-gram, which for a
large corpus runs to tens of millions.

### N6. Phrase detection with the deck's score
**Given:** corpus counts $\text{count}(\texttt{New}) = 1000$, $\text{count}(\texttt{York}) = 800$,
$\text{count}(\texttt{New York}) = 600$; $\text{count}(\texttt{this}) = 5000$,
$\text{count}(\texttt{is}) = 20000$, $\text{count}(\texttt{this is}) = 1200$. Discount $\delta = 5$,
threshold $\tau = 10^{-4}$.
**Find:** which bigrams are merged into phrase tokens.

1. $\text{score}(\texttt{New},\texttt{York}) = \dfrac{600 - 5}{1000 \times 800} = \dfrac{595}{800{,}000}$.
2. $= 7.4375 \times 10^{-4}$.
3. $7.4375\times10^{-4} > 10^{-4}$ → **merge** into the single token `New_York`.
4. $\text{score}(\texttt{this},\texttt{is}) = \dfrac{1200 - 5}{5000 \times 20000} = \dfrac{1195}{100{,}000{,}000}$.
5. $= 1.195 \times 10^{-5}$.
6. $1.195\times10^{-5} < 10^{-4}$ → **do not merge**. This reproduces the deck's own claim on page 90
   that "this is" remains unchanged, even though it occurs **twice as often** as "New York" — the
   product of unigram counts is 125× larger.
7. A rare accidental pair, $\text{count}(w_i w_j) = 4$, $\text{count}(w_i) = 50$,
   $\text{count}(w_j) = 30$: $\dfrac{4 - 5}{50 \times 30} = \dfrac{-1}{1500} = -6.67\times10^{-4}$ —
   **negative**, so it can never clear any positive threshold. That is $\delta$ doing its job.

**Answer:** `New York` is merged ($7.44\times10^{-4}$), `this is` is not ($1.20\times10^{-5}$), and
any bigram with count below $\delta$ scores negative and is automatically rejected. Raw frequency is
**not** the criterion — the ratio is.

## Code

```python
import numpy as np

# ---------- fastText subword extraction (matches N1) ------------------------
def char_ngrams(word, nmin=3, nmax=6):
    """fastText's G_w: boundary-padded n-grams, plus the whole-word token."""
    padded = "<" + word + ">"
    grams = []
    for n in range(nmin, nmax + 1):
        grams += [padded[i:i + n] for i in range(len(padded) - n + 1)]
    grams.append(padded)                       # the word itself counts as one
    return list(dict.fromkeys(grams))          # dedupe, preserve order

for n in (3, 4, 5, 6):
    print(f"eating n={n}: {char_ngrams('eating', n, n)[:-1]}")
print("eating 3-6 ->", len(char_ngrams("eating")), "items")
print("where  n=3 ->", char_ngrams("where", 3, 3))
print("cat    3-6 ->", char_ngrams("cat"), len(char_ngrams("cat")), "items")
# eating n=3: ['<ea', 'eat', 'ati', 'tin', 'ing', 'ng>']
# eating n=4: ['<eat', 'eati', 'atin', 'ting', 'ing>']
# eating n=5: ['<eati', 'eatin', 'ating', 'ting>']
# eating n=6: ['<eatin', 'eating', 'ating>']
# eating 3-6 -> 19 items
# where  n=3 -> ['<wh', 'whe', 'her', 'ere', 're>', '<where>']
# cat    3-6 -> ['<ca', 'cat', 'at>', '<cat', 'cat>', '<cat>'] 6 items

# ---------- embedding an OOV word (matches N2) ------------------------------
Z = {"<be": np.array([0.2, -0.1]), "bea": np.array([0.4, 0.3]),
     "eat": np.array([1.0,  0.5]), "ati": np.array([-0.3, 0.8]),
     "tin": np.array([0.6, -0.4]), "ing": np.array([0.1, 1.2]),
     "ng>": np.array([-0.5, 0.2])}
WORD_VECS = {"eating": np.array([1.4, 2.1])}            # word2vec's whole table

def fasttext_vec(word):
    subs = char_ngrams(word, 3, 3)[:-1]                 # n=3 only in this toy
    known = [Z[s] for s in subs if s in Z]              # OOV word token dropped
    return np.sum(known, axis=0), len(known)

def word2vec_vec(word):
    return WORD_VECS.get(word)                          # None if unseen

v, k = fasttext_vec("beating")
print(f"\nbeating: {k} known 3-grams -> sum {v}  mean {v / k}")
print("word2vec('beating') ->", word2vec_vec("beating"))
# beating: 7 known 3-grams -> sum [1.5 2.5]  mean [0.21428571 0.35714286]
# word2vec('beating') -> None          <- the headline difference

# ---------- TransE score and margin loss (matches N3) -----------------------
d = lambda a, b: float(np.linalg.norm(a - b))
h, r, t = np.array([1., 2.]), np.array([3., 1.]), np.array([4., 2.])
gamma = 2.0
d_pos = d(h + r, t)
print(f"\nf_r(h,t) = -{d_pos:.3f} = {-d_pos:.3f}   (score = minus distance)")
for name, hc, tc in [("tail-corrupt", h, np.array([1., -1.])),
                     ("head-corrupt", np.array([3., 3.]), t)]:
    d_neg = d(hc + r, tc)
    loss = max(0.0, gamma + d_pos - d_neg)
    ok = "satisfied" if loss == 0 else "VIOLATED"
    print(f"{name}: d_neg={d_neg:.3f}  loss={loss:.3f}  margin {ok}")
# f_r(h,t) = -1.000 = -1.000   (score = minus distance)
# tail-corrupt: d_neg=5.000  loss=0.000  margin satisfied
# head-corrupt: d_neg=2.828  loss=0.172  margin VIOLATED
```

Every printed number matches N1, N2 and N3 by hand. The two lines worth staring at are
`word2vec('beating') -> None` against `fasttext -> [1.5 2.5]`, and the fact that the tail-corrupted
TransE pair contributes **exactly zero** loss while the head-corrupted one does not.

## Exam pack

### Must-memorise

| Item | Exactly this |
|---|---|
| Phrase score | $\text{score}(w_i,w_j) = \dfrac{\text{count}(w_i w_j) - \delta}{\text{count}(w_i)\times\text{count}(w_j)}$, merge if above threshold |
| What $\delta$ does | discounting constant; forces bigrams with count $< \delta$ to score negative |
| What the phrase score is | a PMI-style association ratio without the $\log$ and the corpus-size factor |
| fastText citation | **Bojanowski et al., 2017**; library from **Facebook AI Research** |
| fastText core idea | a word is a **bag of character n-grams**, plus the whole-word token |
| fastText n-gram range | **$n = 3$ to $6$**, all lengths used |
| Boundary markers | `<` and `>` pad the word, so `her` in `where` ≠ `<he` in `herself` |
| Word vector | $\mathbf{e}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$ — the **sum** of n-gram vectors |
| fastText score | $s(w,c) = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c$ |
| fastText loss | skip-gram with negative sampling: $\log(1+e^{-s(w_t,w_c)}) + \sum_{n\in\mathcal{N}_{t,c}}\log(1+e^{s(w_t,n)})$ |
| OOV handling | sum (deck: average) the n-gram vectors of the unseen word; **word2vec/GloVe return nothing** |
| Hashing | n-grams hashed into $B$ buckets to bound memory; the paper uses $B = $ **2 million** |
| doc2vec | Le and Mikolov **2014**; adds a **paragraph matrix** $\mathbf{D}$, one vector per document |
| Sentence baselines | mean embedding; tf-idf-weighted mean |
| Poincaré embeddings | Nickel and Kiela; **hyperbolic** space for hierarchies; distance from origin = generality |
| Node embedding goal | $\text{similarity}(u,v) \approx \mathbf{z}_u^\top\mathbf{z}_v$ |
| DeepWalk | random walks = "sentences", nodes = "words", then **skip-gram** |
| DeepWalk loss | $\mathcal{L} = \sum_{u\in V}\sum_{v\in N_R(u)} -\log\dfrac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^\top\mathbf{z}_n)}$ |
| KG triple | $(h, r, t)$ = (head entity, relation, tail entity) |
| KG completion task | given (head, relation), **predict the missing tail** — not the same as link prediction |
| TransE constraint | $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ iff the triple is true |
| TransE score | $f_r(h,t) = -\lVert \mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ — **negated** distance |
| TransE loss | $\sum \left[\gamma + d(\mathbf{h}+\mathbf{r},\mathbf{t}) - d(\mathbf{h}'+\mathbf{r},\mathbf{t}')\right]_+$ |
| Corruption | replace the **head or the tail** with a random entity; never the relation |
| TransE normalisation | entity embeddings renormalised to **unit norm every iteration** |

### Numbers worth knowing

| Quantity | Value |
|---|---|
| fastText n-gram lengths | 3 to 6 |
| `<eating>` padded length / 3-grams / total items | 8 chars / 6 trigrams / **19** items for $n=3$–$6$ |
| `<where>` at $n=3$ | 5 trigrams + 1 whole-word token = **6** |
| fastText hash buckets $B$ | **2,000,000** |
| fastText year / authors | 2017 / Bojanowski, Grave, Joulin, Mikolov |
| doc2vec year / authors | 2014 / Le and Mikolov |
| Named public KGs | FreeBase, Wikidata, DBpedia, YAGO, NELL |
| KG characteristics | **massive** (millions of nodes/edges) and **incomplete** |
| TransE init range | $\text{uniform}(-6/\sqrt{k},\, +6/\sqrt{k})$ |
| GloVe Common Crawl (uncased) | 42B tokens, 1.9M vocab, 300d, 1.75 GB |
| GloVe Common Crawl (cased) | 840B tokens, 2.2M vocab, 300d, 2.03 GB |
| GloVe Wikipedia 2014 + Gigaword 5 | 6B tokens, 400K vocab, 300d, 822 MB |
| GloVe Twitter | 2B tweets, 27B tokens, 1.2M vocab, **200d**, 1.42 GB |

### Likely MCQ traps

- **"Which model can embed an out-of-vocabulary word?"** → **fastText** only. word2vec and GloVe both
  need a row that does not exist. This is the single most likely question in the chapter.
- **"fastText averages its n-gram vectors."** The paper and page 94 say **sum**; page 96 says
  *average*. They differ by a constant and give identical cosine similarities. Prefer *sum* unless the
  question quotes the OOV slide.
- **"TransE maximises $\lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ for true triples."** Backwards. It
  **minimises** the distance for true triples; the *score* is the negated distance, so the **score** is
  maximised. Watch for the sign in $f_r(h,t) = -\lVert\cdot\rVert$.
- **"TransE corrupts the relation."** No — line 9 corrupts the **head or the tail** entity. The
  relation is held fixed, which is what makes the pair a meaningful contrast.
- **"fastText uses BPE."** No. BPE ([Lec 2](../week-01/02-text-processing-tokenization.md)) *segments*
  a word into a few disjoint learned pieces; fastText *represents* a word by summing many overlapping
  fixed-length character n-grams. Same motivation, different mechanism, different place in the pipeline.
- **Forgetting the whole-word token.** $\mathcal{G}_w$ includes `<eating>` itself. A question asking
  "how many vectors are summed for `eating` at $n=3$–$6$?" wants **19**, not 18.
- **Dropping the boundary markers when counting.** `eating` has 6 letters but `<eating>` has 8
  characters, giving $8-3+1 = 6$ trigrams, not $6-3+1 = 4$.
- **"Hashing is used to compress the word vocabulary."** No — hashing bounds the **n-gram** table at
  $B$ buckets. Word types keep their own rows.
- **"DeepWalk needs a new objective."** It is skip-gram, unchanged, on random walks. The only new
  ingredient is the walk.
- **"A random walk cannot revisit a node."** It can — the deck's own figure goes $8 \to 9 \to 8$.
- **KG completion vs link prediction.** The deck flags this explicitly: completion predicts the
  **tail** given (head, relation); link prediction asks whether *any* edge exists between two nodes.
- **"Phrases are detected by raw bigram frequency."** No — by the **ratio** to the product of unigram
  frequencies. `this is` is more frequent than `New York` and is still rejected.
- **"Poincaré embeddings use Euclidean space with more dimensions."** The point is the opposite:
  **hyperbolic** geometry lets 5–10 dimensions do what hundreds of Euclidean dimensions cannot.
- **"fastText is contextual."** It is **static** — one vector per word regardless of sentence.
  Contextual embeddings start at [Lec 26](../week-06/26-pretraining-and-elmo.md).

### Self-test

1. List every character n-gram fastText extracts from `where` at $n = 3$, including the whole-word token. How many are there?
2. How many items are in $\mathcal{G}_w$ for `playing` at $n = 3$–$6$?
3. Why do the boundary symbols `<` and `>` matter?
4. Give word2vec's and fastText's outputs for a word that never appeared in training.
5. Write TransE's scoring function and say whether a *high* value means the triple is likely true or likely false.
6. Compute the TransE margin loss for $\mathbf{h}=(0,1)$, $\mathbf{r}=(2,2)$, $\mathbf{t}=(2,3)$, corrupted $\mathbf{t}'=(0,0)$, $\gamma=1$.
7. In DeepWalk, what plays the role of a sentence, and what plays the role of a word?
8. $\text{count}(a)=400$, $\text{count}(b)=250$, $\text{count}(ab)=105$, $\delta=5$. Compute the phrase score. Is it merged at threshold $10^{-3}$?
9. What single component does doc2vec add to word2vec, and what does it represent?
10. Name two mechanisms that bound fastText's memory and TransE's embedding scale respectively.

<details><summary>Answers</summary>

1. `<wh`, `whe`, `her`, `ere`, `re>`, plus `<where>` — **6** items. (`<where>` is padded to 7 characters, giving $7-3+1=5$ trigrams.)
2. `playing` has $L=7$, padded length 9. $n=3$: 7; $n=4$: 6; $n=5$: 5; $n=6$: 4 → 22, plus the whole-word token = **23**.
3. They distinguish position: `her` at the start of `herself` becomes `<he`/`her`, while `her` inside `where` stays `her`. Without padding a prefix, suffix and interior substring would be the same symbol.
4. word2vec returns nothing — no row exists, so it falls back to `<UNK>` or $\mathbf{0}$. fastText sums the vectors of the word's character n-grams (the whole-word token is absent, the subwords are not) and returns a usable vector.
5. $f_r(h,t) = -\lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$. High (i.e. close to 0, since it is always $\le 0$) means **likely true** — the score is the negated distance.
6. $\mathbf{h}+\mathbf{r}=(2,3)$; $d_{\text{pos}} = \lVert(0,0)\rVert = 0$. $\mathbf{h}+\mathbf{r}-\mathbf{t}' = (2,3)$, $d_{\text{neg}} = \sqrt{4+9} = \sqrt{13} = 3.606$. Loss $=[1 + 0 - 3.606]_+ = \mathbf{0}$ — margin comfortably satisfied.
7. A **random walk** over the graph is the sentence; each **node** visited is a word. Skip-gram then runs unchanged.
8. $(105-5)/(400\times250) = 100/100{,}000 = 1.0\times10^{-3}$. That equals the threshold, so it is merged only under a $\ge$ rule — a deliberately borderline case; strictly above $10^{-3}$ it is not.
9. A **paragraph matrix** $\mathbf{D}$ with one vector per document, concatenated or averaged in with the context word vectors; it acts as a memory of the document's overall topic, which the local window cannot supply.
10. fastText: **hashing n-grams into $B = 2$ million buckets**. TransE: **renormalising every entity embedding to unit norm each iteration** (line 5 of Algorithm 1), which stops the model shrinking all embeddings to zero.

</details>

## Beyond the slides

**Gap:** The deck never says what fastText does with a word it *has* seen versus one it has not, and
the asymmetry matters.
**Why it matters:** For an in-vocabulary word the sum includes the whole-word token `<w>`, which is
free to learn a non-compositional meaning. For an OOV word that term is simply missing, so the vector
is **purely compositional**. That is why fastText's OOV vectors are good for morphological variants
(`beating` from `eating`) and poor for arbitrary novel words whose meaning is not in their spelling —
a brand name like `Zylox` gets a vector, but a meaningless one. An exam asking "is fastText's OOV
vector as good as a trained one?" wants: no, and this is why.

**Gap:** The subword-vs-tokenization boundary is never drawn, and modern NLP sits on the other side
of it.
**Why it matters:** Every model from BERT onward uses **BPE or WordPiece**
([Lec 2](../week-01/02-text-processing-tokenization.md)) — the word is *split* and the pieces stay
separate tokens through the network, so OOV disappears as a concept without anyone summing n-grams.
fastText's approach is the last major embedding method to keep the word as the unit. Knowing which
camp a model is in ("does it have an OOV problem?") resolves a whole family of questions.

**Gap:** TransE's failure on 1-to-N, N-to-1 and symmetric relations is not mentioned.
**Why it matters:** It is the obvious follow-up question to "what does TransE optimise", and the
argument is two lines (given in the TransE section above). It also explains the existence of TransH,
TransR, DistMult, ComplEx and RotatE — if a question lists "TransE variants", the thing they vary is
exactly this expressiveness limit.

**Gap:** The deck shows DeepWalk but not **node2vec**, its standard successor.
**Why it matters:** node2vec is the same algorithm with a **biased second-order random walk**
controlled by a return parameter $p$ and an in-out parameter $q$, which lets you dial between
BFS-like walks (capturing structural roles) and DFS-like walks (capturing communities). It is more
commonly used than DeepWalk in practice and is a natural distractor in an MCQ about random-walk
embeddings.

**Gap:** Nothing is said about how the phrase-merging pass interacts with everything downstream.
**Why it matters:** Merging phrases **changes the vocabulary**, which changes
$\lvert V\rvert$, the softmax cost, the negative-sampling distribution and the perplexity of any
language model built on the same tokenization ([Lec 4](../week-01/04-ngram-lm-2-smoothing-perplexity.md)
warns that perplexity is incomparable across tokenizations). The original word2vec phrase release
added roughly 3 million phrase types on top of its word vocabulary — a very large change, not a
cosmetic preprocessing step.

## Cut from the slides

Page 87 is the title slide and page 88 the agenda; both are folded into the section ordering above.
Page 89 is a screenshot of the Stanford GloVe download page — GloVe itself belongs to
[Lec 13](13-negative-sampling-glove.md), so it gets one sentence here, but the four corpus
configurations on it are extracted into *Numbers worth knowing* because they are precisely the kind of
detail an MCQ keys on. Page 107 is the Jurafsky and Martin reference (SLP 3rd edition, August 2024
manuscript, Chapter 6) and page 108 the closing "Thank you" slide; neither carries content.
Page 104's knowledge-graph-completion illustration is described in prose rather than embedded, to keep
the figure count in range. Pages 98 (Poincaré) and 99–100 (node embeddings) are single conceptual
slides with no equations; they are taught at the depth they are given plus the geometric reasoning the
deck leaves implicit. Nothing in the range is dropped. The deck contains **no "Try this problem" page**
in pp. 87–108, consistent with the ownership map.
