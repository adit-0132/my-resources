# Lec 45 — Automatic Prompt Engineering

> **Source:** `Week9.pdf` pp. 93–114 · **Week 9** · **Playlist:** Lec 45
> **Prereqs:** [Lec 41 — Prompting I](41-prompting-1.md), [Lec 43 — Advanced Prompting](43-advanced-prompting.md), [Lec 9 — Backpropagation](../week-02/09-backpropagation.md)
> **Feeds into:** [Lec 46 — PEFT: Adapters and Prefix-Tuning](../week-10/46-peft-adapters-prefix.md)

## Why this lecture exists

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

## The ideas

![Slide "Need for Automatic Prompt Engineering": automated prompt design uses search or generation over a predefined search space and usually outperforms manual engineering; three approaches listed — prompt paraphrasing, gradient-based discrete prompt search, prompt tuning, annotated in red as token space, token space, continuous space / soft prompts](../../assets/pages/lec45/p-095.png)
*Fig. — The whole lecture in one slide. Notice the lecturer's handwritten split: the first two methods search **token space** (the prompt stays a real string); prompt tuning searches **continuous space** and produces "soft prompts" that are not strings at all. Page 95 of `Week9.pdf`.*

### Why automate at all

The deck's definition is worth memorising verbatim: **automated prompt design is using some form of
search or generation to find the most effective prompt template within a predefined search space.**
Two clauses carry the weight. *Search or generation* — you either enumerate candidates or have a model
produce them. *Predefined search space* — you must fix what is being searched over (how many tokens,
drawn from what vocabulary, in what template slots) before any of this is well posed.

The deck's claim for why it wins: automated design "will usually outperform manual prompt engineering,
as it is generally more complete in its search of parameter space". A human tries maybe ten phrasings;
a search tries thousands. The cost is implementation complexity, and — for the gradient method —
access to the model's internals.

### Prompt paraphrasing — the cheap automation

![Slide "Prompt Paraphrasing": the seed prompt "[X] shares a border with [Y]." is fed through a box labelled Paraphrasing Model, which emits "[X] has a common border with [Y].", "[X] adjoins [Y].", and more; a red arrow loops the outputs back to the input](../../assets/pages/lec45/p-096.png)
*Fig. — Jiang et al. 2019. The red loop the lecturer drew is the important part: paraphrases can be fed back in to generate second-generation candidates. Everything stays in natural language. Page 96.*

Take one seed prompt that a human wrote. Push it through a paraphrasing model — a round-trip
translation system, or a seq2seq paraphraser, or just an LLM asked for rewordings. You now have a pool
of semantically equivalent templates. Score each on a **development set** and keep the winner.

This is the simplest thing that counts as automation and it needs nothing but black-box access to the
target model. Its limits are equally plain: the search never leaves the neighbourhood of your seed, and
"semantically equivalent" paraphrases can score wildly differently, which is itself the evidence that
manual prompting is unreliable.

### Gradient-based discrete prompt search (AutoPrompt)

This is the mathematical centrepiece of the lecture. The method is **AutoPrompt** (Shin et al., 2020).

![Slide "Gradient-based discrete prompt search": left column shows Original Input x_inp = "a real joy.", Trigger Tokens x_trig = atmosphere, alot, dialogue, Clone..., and Template lambda(x_inp, x_trig) = {sentence}[T][T][T][T][T][P]; right shows the assembled AUTOPROMPT input fed to a Masked LM, producing p([MASK]|x_prompt) over words like Cris, marvelous, philanthrop (positive) and worse, incompetence, Worse (negative), summed into p(y|x_prompt)](../../assets/pages/lec45/p-097.png)
*Fig. — The three moving parts. `[T]` slots hold **trigger tokens** (searched for); `[P]` is the prediction slot where `[MASK]` goes. The probabilities of several label tokens are **summed** into a class probability — that summation is the verbalizer, and it is also searched for. Page 97.*

**The setup.** Fix a template $\lambda(\mathbf{x}_{\text{inp}}, \mathbf{x}_{\text{trig}})$ — in the
deck's sentiment example, `{sentence}[T][T][T][T][T][P].` The number of trigger tokens is a
**hyperparameter** (the deck uses 5), and they are all **initialized to `[MASK]`**. You then take a
batch of examples from the training set and ask: which real token should go in slot $j$?

**The obstacle.** Prompt tokens are **discrete**. There is no gradient to descend on "which word sits
in slot 3" — the loss is a function of a vocabulary index, not of a continuous variable. You cannot
nudge the word "dialogue" 0.01 towards "Rating".

**The fix.** The token is discrete, but its *embedding* is a vector, and the model is differentiable
with respect to that vector. So use the gradient not to *move* anything, but to **score candidate
substitutions**.

![Slide: formal statement that at each step we compute a first-order approximation of the change in log-likelihood produced by swapping the jth trigger token with another token w in V, giving V_cand = top-k over w of [w_in^T grad log p(y|x_prompt)], Equation 2; alongside a diagram of candidate triggers [MASK][MASK][MASK] becoming man/##s/cameo and Rating/fiennes/go, with the gradient arrow annotated](../../assets/pages/lec45/p-099.png)
*Fig. — Equation (2). The gradient is taken **with respect to the input embedding of the $j$-th trigger token**, and it is dotted with the *input* embedding $\mathbf{w}_{\text{in}}$ of each candidate word. The deck's own question — "Why this particular form of Eq 2?" — is answered on the next slide. Page 99.*

In the deck's notation (we translate to the course's symbols below):

$$V_{\text{cand}} = \underset{w \in V}{\text{top-}k}\left[\, \mathbf{w}_{\text{in}}^{\top} \nabla \log p(y \mid \mathbf{x}_{\text{prompt}}) \,\right]$$

Writing $\mathbf{e}_w$ for the input embedding of token $w$ and
$\mathbf{g} = \nabla_{\mathbf{e}_{w_j}} \log p(y \mid \mathbf{x}_{\text{prompt}})$ for the gradient at
the current occupant $w_j$ of slot $j$, this is

$$V_{\text{cand}} = \underset{w' \in V}{\text{top-}k}\ \left[\mathbf{e}_{w'}^{\top}\mathbf{g}\right]$$

#### Why that form — directional derivatives and first-order approximation

![Slide "Directional Derivatives and 1st order approx.": definition of the directional derivative of f at x in direction v as the limit of (f(x+eps*v) - f(x))/eps; dropping the limit gives (f(x+eps*v)-f(x))/eps approx grad_v f(x); rearranged, f(x+eps*v) - f(x) approx eps * grad_v f(x); in words, moving from x to x+eps*v changes f by about eps*grad_v f(x)](../../assets/pages/lec45/p-100.png)
*Fig. — The justification for the whole method, from David Rosenberg's notes. The red annotation in the top right spells out the application: $\mathbf{w}_{\text{in}}^{\top}\nabla_{\text{trig}}\log p(y\mid x)$. Page 100.*

For a differentiable $f : \mathbb{R}^d \to \mathbb{R}$, the directional derivative at $\mathbf{x}$ in
direction $\mathbf{v}$ is

$$\nabla_{\mathbf{v}} f(\mathbf{x}) = \lim_{\epsilon \to 0} \frac{f(\mathbf{x} + \epsilon\mathbf{v}) - f(\mathbf{x})}{\epsilon}$$

Drop the limit for small $\epsilon$ and rearrange:

$$f(\mathbf{x} + \epsilon\mathbf{v}) - f(\mathbf{x}) \approx \epsilon\,\nabla_{\mathbf{v}} f(\mathbf{x}) = \epsilon\,\mathbf{v}^{\top}\nabla f(\mathbf{x})$$

In words, as the slide says: start at $\mathbf{x}$, move to $\mathbf{x} + \epsilon\mathbf{v}$, and $f$
changes by about $\epsilon \nabla_{\mathbf{v}} f(\mathbf{x})$.

Now apply it. Swapping token $w$ for $w'$ in slot $j$ moves that slot's embedding from $\mathbf{e}_w$ to
$\mathbf{e}_{w'}$ — a step of exactly $\mathbf{v} = \mathbf{e}_{w'} - \mathbf{e}_w$ with $\epsilon = 1$.
So the first-order change in the log-likelihood is

$$\Delta \log p \;\approx\; (\mathbf{e}_{w'} - \mathbf{e}_w)^{\top}\mathbf{g}$$

and since the loss is $\mathcal{L} = -\log p(y \mid \mathbf{x}_{\text{prompt}})$,

$$\Delta\mathcal{L} \;\approx\; -(\mathbf{e}_{w'} - \mathbf{e}_w)^{\top}\mathbf{g} \;=\; (\mathbf{e}_{w'} - \mathbf{e}_w)^{\top}\nabla_{\mathbf{e}_w}\mathcal{L}$$

**Why the deck's Eq. 2 drops the $\mathbf{e}_w$ term:** the subtracted quantity
$\mathbf{e}_w^{\top}\mathbf{g}$ is the *same constant* for every candidate $w'$. It shifts all scores
equally and therefore cannot change the ranking. So the top-$k$ of $\mathbf{e}_{w'}^{\top}\mathbf{g}$
is identical to the top-$k$ of $\Delta \log p$. This is why Eq. 2 looks like a bare dot product rather
than a difference — and it is a favourite exam point.

Note also that $\mathbf{e}_{w'}^{\top}\mathbf{g}$ for all $w' \in V$ is a **single matrix–vector
product** $\mathbf{E}\mathbf{g}$ against the embedding matrix. One backward pass plus one matmul scores
the entire vocabulary. That is the whole efficiency argument.

**The approximation is only first order**, and $\epsilon = 1$ is not small — the step from one
embedding to another is a full-sized jump, not an infinitesimal nudge. So the score is a *heuristic for
shortlisting*, never a verdict.

Page 101 then closes the loop: *"for each candidate in this set, we then re-evaluate the equation on
the updated prompt, and retain the prompt with the highest probability in the next step."* The shortlist
is scored **for real** by a forward pass, not by the linear approximation. And the quantity being
re-evaluated is the class probability, which an MLM produces by **summing** over a set of label tokens:

$$p(y \mid \mathbf{x}_{\text{prompt}}) = \sum_{w \in V_y} p([\text{MASK}] = w \mid \mathbf{x}_{\text{prompt}})$$

where $V_y$ is the set of label tokens corresponding to label $y$ — the verbalizer, which §*Automating
label-token selection* below shows how to find.

#### The loop, assembled

1. Fix the template and the number of trigger tokens; initialize them all to `[MASK]`.
2. Take a batch of training examples. Forward, then backward to get
   $\mathbf{g} = \nabla_{\mathbf{e}_{w_j}}\log p(y\mid\mathbf{x}_{\text{prompt}})$ for slot $j$.
3. Score every vocabulary item with $\mathbf{e}_{w'}^{\top}\mathbf{g}$; keep the top-$k$ as
   $V_{\text{cand}}$.
4. For each candidate in $V_{\text{cand}}$, actually substitute it and re-evaluate
   $p(y \mid \mathbf{x}_{\text{prompt}}) = \sum_{w \in V_y} p([\text{MASK}] = w \mid \mathbf{x}_{\text{prompt}})$
   on the batch.
5. Retain the prompt with the highest probability. Move to the next slot and repeat.

The pattern — *cheap linear proxy proposes, expensive exact evaluation disposes* — is the thing to
remember. The gradient never updates a single model weight; the model is frozen throughout.

#### Beam search over positions

![Slide "Gradient-based discrete prompt search: beam search": the candidate-trigger table evolves from [MASK][MASK][MASK] with batch probabilities 0.01/0.05/0.03, through "Rating [MASK][MASK]" at 0.18/0.11/0.08, to "Rating ##omi #!!" at 0.95/0.89/0.77; text explains that top-k token candidates are considered for each position, searching left to right, scoring each beam with the summed label-token probability](../../assets/pages/lec45/p-102.png)
*Fig. — Watch the batch probabilities climb 0.01 → 0.18 → 0.95 as slots are filled left to right. The final trigger, "Rating ##omi #!!", is not a phrase any human would write. Page 102.*

Greedily fixing each slot in turn is short-sighted: the best token for slot 1 depends on what will land
in slots 2 and 3. So AutoPrompt wraps the substitution step in **beam search** — the same algorithm you
met for decoding in [Lec 19](../week-04/19-decoding-strategies.md), but over *prompt positions* rather
than over output time steps. Keep the top-$k$ candidates per position, search **left to right** across
positions, and score each beam with
$p(y\mid\mathbf{x}_{\text{prompt}}) = \sum_{w\in V_y} p([\text{MASK}]=w\mid\mathbf{x}_{\text{prompt}})$.

The two "top-$k$"s in this lecture are different objects and get confused constantly: one is the
gradient shortlist $V_{\text{cand}}$ *inside* a position; the other is the beam width *across* positions.

### Automating label-token selection

[Lec 41](41-prompting-1.md) established that the mapping from the model's output word to a class label
— the **verbalizer** — is a design choice. Guessing "positive"/"negative" is exactly as unprincipled as
guessing a template. AutoPrompt automates it too, in two steps.

![Slide "Automating label token selection": step one trains a logistic classifier on the contextualized [MASK] embedding h = Transformer_enc(x~), written p(y|h) proportional to exp(h . y + beta_y) with learned weight and bias terms for label y; step two substitutes h with the MLM's output word embeddings w_out to obtain s(y,w) = p(y|w_out) and V_y = top-k over w in V of s(y,w); handwritten annotations show 768x1 vectors and a 768x2 weight matrix](../../assets/pages/lec45/p-103.png)
*Fig. — Both steps, with the lecturer's dimension annotations: $\mathbf{h}^{(i)}$ is $768 \times 1$, and the learned label weights form a $768 \times 2$ matrix for two classes. Step 2 learns nothing — it reuses the frozen output embeddings. Page 103.*

**Step 1.** Run the prompted input through the encoder, $\mathbf{h} = \text{Transformer}_{\text{enc}}(\tilde{\mathbf{x}})$,
and take the **contextualized embedding of the `[MASK]` token**. Train a logistic classifier on it:

$$p(y \mid \mathbf{h}^{(i)}) \;\propto\; \exp\!\left(\mathbf{h}^{(i)} \cdot \mathbf{y} + \beta_y\right)$$

where $\mathbf{y}$ and $\beta_y$ are the learned weight vector and bias **for label $y$**. These are
the only learnable parameters in the whole procedure.

**Step 2.** Now substitute $\mathbf{h}^{(i)}$ with the MLM's **output word embeddings**
$\mathbf{w}_{\text{out}}$ — i.e. ask the classifier "how positive would it be if the `[MASK]` slot
resolved to word $w$?":

$$s(y, w) = p(y \mid \mathbf{w}_{\text{out}}), \qquad V_y = \underset{w \in V}{\text{top-}k}\,[\,s(y,w)\,]$$

The resulting $V_y$ is the label-token set. Size $|V_y|$ is a hyperparameter — the deck's exercise uses
3. The trick is that the classifier was trained in *contextual* embedding space but is applied to
*output-vocabulary* embedding space; it works because an MLM scores words by the dot product of the
`[MASK]` representation with $\mathbf{w}_{\text{out}}$, so the two spaces are already aligned.

### Huge gains in performance

![Slide "Huge gains in performance": a results table with rows BiLSTM 82.8, BiLSTM+ELMo 89.3, BERT (linear probing) dev 85.2 test 83.4, BERT (finetuned) 93.5, RoBERTa (linear probing) 87.9/88.8, RoBERTa (finetuned) 96.7; then BERT (manual) 63.2/63.2, BERT (AUTOPROMPT) 80.9/82.3, RoBERTa (manual) 85.3/85.2, RoBERTa (AUTOPROMPT) 91.2/91.4](../../assets/pages/lec45/p-105.png)
*Fig. — SST-2 sentiment accuracy. The lower block is the prompting block: AutoPrompt lifts BERT from 63.2 to 82.3 test, and RoBERTa from 85.2 to 91.4. Both beat **linear probing** of the same model; neither beats full fine-tuning. Page 105.*

Three readings of that table, all examinable:

- **AutoPrompt beats manual prompting by a lot** — +19.1 test points on BERT, +6.2 on RoBERTa.
- **AutoPrompt is competitive with linear probing** while training far fewer parameters: for RoBERTa it
  wins outright (91.4 vs 88.8 test, 91.2 vs 87.9 dev); for BERT it is a near-tie on test (82.3 vs 83.4)
  and behind on dev (80.9 vs 85.2). RoBERTa + AutoPrompt also beats BiLSTM (82.8) and BiLSTM+ELMo (89.3).
- **It does not beat fine-tuning** (93.5 / 96.7). Prompting is not yet a replacement for updating
  weights — which is precisely the gap Week 10 is about.

![Slide "Example Prompts": a table of task, prompt template, prompt found by AUTOPROMPT, and label tokens; Sentiment Analysis yields "unflinchingly bleak and desperate Writing academicswhere overseas will appear [MASK]." with pos: partnership, extraordinary, ##bla and neg: worse, persisted, unconstitutional; NLI, Fact Retrieval and Relation Extraction rows follow with similarly disfluent strings](../../assets/pages/lec45/p-106.png)
*Fig. — The counterintuitive payoff. "unflinchingly bleak and desperate Writing academicswhere overseas will appear [MASK]" is not English, yet it drives sentiment classification to 82.3. Note the positive label tokens include "partnership" and the subword "##bla". Page 106.*

**The discovered prompts are often nonsense** — ungrammatical strings, subword fragments like `##omi`,
`#!!`, `##bla`. They still work. This is the lecture's most memorable and most examinable fact: a prompt
is not an instruction the model "understands", it is a **key into the model's learned conditional
distribution**. Human readability and effectiveness are different axes. The label tokens are equally
odd — "partnership" and "extraordinary" for positive; "worse", "persisted", "unconstitutional" for
negative. Only some of them look sentiment-bearing to a human.

### EvoPrompt — evolutionary search over prompt text

![Slide "EvoPrompt": a Genetic Algorithm implemented by LLMs; the query asks the LLM to cross over Prompt 1 ("Now you are a categorizer, your mission is to ascertain the sentiment of the provided text, either favorable or unfavourable.") and Prompt 2 ("Assign a sentiment label to the given sentence from ['negative','positive'] and return only the label without any other text."), then mutate the result; the response shows the crossover prompt mixing orange and blue fragments, then the mutated final prompt bracketed in <prompt> tags](../../assets/pages/lec45/p-107.png)
*Fig. — The LLM itself performs crossover and mutation; the orange and blue colouring shows which fragments were inherited from which parent. The output is wrapped in `<prompt></prompt>` so it can be parsed automatically. Page 107.*

AutoPrompt needs gradients, so it needs white-box access. **EvoPrompt** (Guo et al., 2023) needs none.
It runs a genetic algorithm in which the *LLM is the genetic operator*:

| GA concept | EvoPrompt realisation |
|---|---|
| Individual | one natural-language prompt |
| Population | a pool of candidate prompts |
| **Crossover** | ask the LLM to combine two parent prompts into one |
| **Mutation** | ask the LLM to reword the combined prompt |
| **Fitness** | accuracy on a **development set** |
| **Selection** | "retain those with superior performance, similar to the survival of the fittest in nature" |

The deck shows two variants: a plain **Genetic Algorithm (GA)** version (page 107) and a
**Differential Evolution (DE)** version (page 108), in which the LLM is asked to identify the
*different parts* between two prompts, mutate only those, and combine them with a third prompt and a
basic prompt — the textual analogue of DE's $a + F(\mathbf{b} - \mathbf{c})$ update, which the slide
annotates directly on the diagram.

Unlike AutoPrompt, EvoPrompt's outputs stay **fluent and human-readable**, because every candidate is
generated by an LLM asked to write a prompt. The bootstrapping instinct — use the model to improve its
own inputs — is the same one behind Self-Instruct in [Lec 37](../week-08/37-instruction-finetuning-2.md).

### Prompt-tuning — from discrete to continuous

![Slide "Prompt-tuning": heading "Optimize the embeddings of a prompt, instead of the words."; left, Model Tuning makes a separate 11B-parameter copy per task (Task A/B/C models); right, Prompt Tuning keeps one frozen Pre-trained Model (11B params) and a mixed-task batch where each example carries its own task prompt, with Task Prompts at 20K params each](../../assets/pages/lec45/p-109.png)
*Fig. — The operational payoff. Model tuning stores a full 11B copy per task; prompt tuning stores one frozen model plus a 20K-parameter prompt per task, so a **mixed-task batch** can be served by a single model. The lecturer's annotation computes $10 \times 768$ for a 10-token prompt. Page 109.*

Discrete search is still hunting through a vocabulary. Prompt-tuning (Lester et al., 2021) asks the
obvious next question: why restrict yourself to vectors that happen to be real words? **Optimize the
embeddings of a prompt instead of the words.**

**The mechanism.** Given an input of $n$ tokens $\{x_1, \ldots, x_n\}$, the model first embeds them into
a matrix $\mathbf{X}_e \in \mathbb{R}^{n \times d}$, where $d$ is the embedding dimension (the deck
writes this $e$ and calls the matrix $X_e$; we use $d$ per the course's notation). The soft prompt is a
free parameter matrix

$$\mathbf{P}_e \in \mathbb{R}^{p \times d}$$

where $p$ is the **prompt length**. Concatenate it in front of the embedded input:

$$[\mathbf{P}_e ; \mathbf{X}_e] \in \mathbb{R}^{(p+n)\times d}$$

and push that through the encoder–decoder exactly as normal. Train to maximise $p(Y)$, but **only
$\mathbf{P}_e$ is updated**. Every weight of the backbone is frozen.

Three consequences to hold onto:

1. **Soft prompts need not correspond to any real token.** $\mathbf{P}_e$ lives anywhere in
   $\mathbb{R}^d$; the rows almost never coincide with a row of the embedding matrix. You cannot print
   a soft prompt. The gain is a strictly larger search space — the discrete methods were restricted to
   the $|V|$ points of the embedding table, prompt tuning gets the whole continuous space between them.
2. **The gradient is the ordinary one.** Backpropagation ([Lec 9](../week-02/09-backpropagation.md))
   runs all the way back through the frozen network to the input embeddings; you just discard every
   weight gradient and keep $\partial\mathcal{L}/\partial\mathbf{P}_e$. Nothing new is needed
   mathematically — only gradients *with respect to embeddings*, which is also what AutoPrompt needed.
3. **You train $p \times d$ parameters**, not $|\theta|$. For $p=10$, $d=1024$ that is 10,240 numbers
   against a model of hundreds of millions. See N2 and N4.

#### How effective?

![Slide "Prompt-tuning: How effective?": a log-x plot of SuperGLUE score against model parameters from 1e8 to 1e11, with four curves — Model Tuning, Model Tuning (Multi-task), Prompt Design, Prompt Tuning; Prompt Tuning starts below Model Tuning at 1e8 and converges with it near 1e10 at about 90; Prompt Design (GPT-3 few-shot) trails far below throughout](../../assets/pages/lec45/p-111.png)
*Fig. — The key empirical finding of the whole lecture: the green Prompt Tuning curve is clearly below the red Model Tuning curve at $10^8$ parameters and **meets it at around $10^{10}$**. The gap closes as scale grows — hence the paper's title, "The Power of Scale". Page 111.*

The deck's own summary: standard model tuning of T5 is strong but "requires storing separate copies of
the model for each end task"; prompt tuning "matches the quality of model tuning as size increases,
while enabling the reuse of a single frozen model for all tasks", and "significantly outperforms
few-shot prompt design using GPT-3". Results are mean and standard deviation over 3 runs.

**The scale dependence is the examinable point.** Prompt tuning is *not* uniformly as good as fine
tuning. At small scale it loses by several SuperGLUE points. It becomes competitive only in the
$10^{10}$-parameter range. An MCQ that says "prompt tuning always matches fine-tuning" is false; one
that says "prompt tuning matches fine-tuning as model scale grows" is the deck's claim.

Note also the fourth curve, **Prompt Design** — GPT-3-style few-shot prompting with no training at all.
It trails everything, even at $10^{11}$ parameters. Learning beats hand-designing, which is the thesis
of the lecture restated as a graph.

### Prompt-tuning vs prefix-tuning — the boundary with Lec 46

| | **Prompt-tuning** (this lecture) | **Prefix-tuning** ([Lec 46](../week-10/46-peft-adapters-prefix.md)) |
|---|---|---|
| What is learned | one matrix $\mathbf{P}_e \in \mathbb{R}^{p\times d}$ | a learned prefix at **every** layer |
| Where it is injected | the **input embedding layer only** | the key/value activations of all $L$ layers |
| Parameter count | $p \times d$ | roughly $L$ times larger |

Prompt-tuning prepends learnable vectors at the **input only**; prefix-tuning prepends learned vectors
at **every layer**. Prefix-tuning is strictly more expressive and strictly more expensive, and
[Lec 46](../week-10/46-peft-adapters-prefix.md) owns it, alongside adapters and the three PEFT
perspectives. [Lec 47](../week-10/47-lora-and-variants.md) then covers LoRA.

This chapter is the first in the course to **train something small while freezing the backbone**. The
deck's own closing note — "Will talk about more parameter-efficient methods next week" — is the handoff:
Week 10 generalises exactly this move, asking where else a small trainable module can be inserted into
a frozen model, and how small it can get.

## Worked numericals

### N1. AutoPrompt label-token selection with BERT — the deck's exercise (page 104)
**Given:** AutoPrompt with BERT for text classification, 2 classes (positive, negative), $|V_y| = 3$.
**Find:** the two steps for deciding the label-word sets, and the number of learnable parameters.

1. **Step 1 — fit a logistic classifier on the `[MASK]` representation.** Run the prompted input
   through the encoder, take the contextualized embedding of `[MASK]`,
   $\mathbf{h}^{(i)} = \text{Transformer}_{\text{enc}}(\tilde{\mathbf{x}})^{[\text{MASK}]}$. For
   BERT-base, $\mathbf{h}^{(i)} \in \mathbb{R}^{768}$. Fit
   $p(y\mid\mathbf{h}^{(i)}) \propto \exp(\mathbf{h}^{(i)}\cdot\mathbf{y} + \beta_y)$.
2. **Step 2 — transfer the classifier to vocabulary space.** Replace $\mathbf{h}^{(i)}$ with the MLM's
   output word embedding $\mathbf{w}_{\text{out}}$ of each candidate word, giving
   $s(y,w) = p(y\mid\mathbf{w}_{\text{out}})$, and set
   $V_y = \text{top-}k_{\,w\in V}\,[s(y,w)]$ with $k = 3$.
3. **Count the parameters.** Only Step 1 learns anything. Per label $y$ you learn one weight vector
   $\mathbf{y}\in\mathbb{R}^{768}$ and one scalar bias $\beta_y$.
4. Weights: $768 \times 2 = 1536$.
5. Biases: $1 \times 2 = 2$.
6. Total $= 1536 + 2 = \mathbf{1538}$.
7. **Step 2 adds zero parameters** — $\mathbf{w}_{\text{out}}$ is the frozen MLM output-embedding
   matrix, and top-$k$ is a selection, not a fit. Likewise **$|V_y| = 3$ does not enter the count**;
   changing it to 10 would leave the answer at 1538. That is the trap in the question.

**Answer:** **1538 learnable parameters** ($768\times2$ weights $+\,2$ biases). The slide's handwritten
working reads "$768\times2 + 2$", so the deck agrees. *(If you adopt the redundant-free binary
parameterisation with a single weight vector and one bias, you would get $768+1 = 769$; the deck's
per-label form is what is asked for.)*

### N2. Extra storage: full fine-tuning vs prompt-tuning on T5 — the deck's exercise (page 112)
**Given:** T5 with model dimensionality $d = 1024$ and 24 layers, for 3 tasks (summarization, QA, text
simplification). Either full fine-tuning per task, or prompt-tuning with soft prompts of length
$p = 10$ per task.
**Find:** the extra parameters stored at inference time in each scenario, beyond the T5 parameters.

1. **Per encoder layer.** Self-attention $\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V,\mathbf{W}^O$ give
   $4d^2$; the FFN with $d_{ff}=4d$ gives $2 \times 4d^2 = 8d^2$. Total $\mathbf{12d^2}$.
2. **Per decoder layer.** Masked self-attention $4d^2$ + cross-attention $4d^2$ + FFN $8d^2 =
   \mathbf{16d^2}$.
3. Encoder + decoder per layer index: $12d^2 + 16d^2 = 28d^2$. (These are the deck's handwritten
   "$12d^2L$" and "$16d^2L$".)
4. $d^2 = 1024^2 = 1{,}048{,}576$.
5. Per layer index: $28 \times 1{,}048{,}576 = 29{,}360{,}128$.
6. Over 24 layers: $29{,}360{,}128 \times 24 = 704{,}643{,}072 \approx 705$M — a sane number, since
   T5-Large ($d=1024$, 24+24 layers) really is $\approx770$M including embeddings.
7. **Full fine-tuning, 3 tasks:** $704{,}643{,}072 \times 3 = \mathbf{2{,}113{,}929{,}216}$
   $\approx 2.11$ **billion** extra parameters — one whole copy of the model per task.
8. **Prompt-tuning, per task:** $p \times d = 10 \times 1024 = 10{,}240$.
9. **Prompt-tuning, 3 tasks:** $10{,}240 \times 3 = \mathbf{30{,}720}$.
10. Ratio: $2{,}113{,}929{,}216 / 30{,}720 = \mathbf{68{,}812.8\times}$, i.e. prompt-tuning stores
    $0.001453\%$ of what full fine-tuning stores.

**Answer:** full fine-tuning $\approx 2.11\times10^{9}$ extra parameters; prompt-tuning $30{,}720$ —
a factor of about **68,800**. The deck's handwriting gives exactly $28\times(1024)^2\times24\times3$ and
$(1024\times10)\times3$, so our arithmetic matches its setup. *Two caveats the slide leaves implicit:
"24 layers" is read as 24 encoder **and** 24 decoder layers (that is what $12d^2L + 16d^2L$ assumes);
and the count excludes embeddings, biases, LayerNorms and T5's relative-position biases.*

### N3. Ranking candidate substitutions by the first-order score
**Given:** slot $j$ currently holds token $w$ with input embedding $\mathbf{e}_w = (0.5, -1.0, 0.25, 0.0)$.
The gradient of the log-likelihood w.r.t. that slot's embedding is
$\mathbf{g} = (0.8, -0.6, 0.20, 0.4)$. Three candidate replacements:
$\mathbf{e}_A = (1.0, 0.0, -0.5, 0.5)$, $\mathbf{e}_B = (0.2, -1.6, 0.8, 0.5)$,
$\mathbf{e}_C = (-0.4, -0.5, 1.5, 1.0)$.
**Find:** $\Delta\mathcal{L}$ for each, and the ranking.

1. Baseline: $\mathbf{e}_w^{\top}\mathbf{g} = (0.5)(0.8) + (-1.0)(-0.6) + (0.25)(0.20) + (0)(0.4)
   = 0.40 + 0.60 + 0.05 + 0 = 1.05$.
2. $\mathbf{e}_A^{\top}\mathbf{g} = 0.80 + 0 - 0.10 + 0.20 = 0.90$.
3. $\mathbf{e}_B^{\top}\mathbf{g} = 0.16 + 0.96 + 0.16 + 0.20 = 1.48$.
4. $\mathbf{e}_C^{\top}\mathbf{g} = -0.32 + 0.30 + 0.30 + 0.40 = 0.68$.
5. $\Delta\log p \approx \mathbf{e}_{w'}^{\top}\mathbf{g} - 1.05$:
   A $= -0.15$, B $= +0.43$, C $= -0.37$.
6. $\Delta\mathcal{L} = -\Delta\log p$: A $= +0.15$, B $= \mathbf{-0.43}$, C $= +0.37$.
7. We want the **largest decrease in loss**, i.e. the most negative $\Delta\mathcal{L}$: **B**, then A,
   then C.
8. Check the ranking shortcut: ranking by the raw dot product gives $1.48 > 0.90 > 0.68$ — the same
   order. Subtracting the constant 1.05 cannot reorder anything, which is why Eq. 2 omits it.

**Answer:** $\Delta\mathcal{L}_A = +0.15$, $\Delta\mathcal{L}_B = -0.43$, $\Delta\mathcal{L}_C = +0.37$;
ranking **B > A > C**, so with $k=1$ the candidate set is $\{B\}$. Only B is predicted to help — and
that prediction must still be confirmed by an actual forward pass.

### N4. Prompt-tuning parameter fraction
**Given:** T5-XXL, 11B parameters, $d = 4096$, soft prompt of length $p = 5$ (the deck's "20K params
each"). Also BERT-base, 110M parameters, $d = 768$, $p = 10$ (the lecturer's $10\times768$ annotation).
**Find:** trainable parameters and the fraction of the model they represent.

1. T5-XXL: $p \times d = 5 \times 4096 = \mathbf{20{,}480}$ — which is the slide's "20K params each".
2. Fraction: $20{,}480 / 11{,}000{,}000{,}000 = 1.862\times10^{-6} = \mathbf{0.000186\%}$.
3. Equivalently $11\times10^{9}/20{,}480 \approx \mathbf{537{,}000\times}$ fewer parameters.
4. BERT-base: $10 \times 768 = \mathbf{7{,}680}$.
5. Fraction: $7{,}680 / 110{,}000{,}000 = 6.98\times10^{-5} = \mathbf{0.00698\%}$, about
   $\mathbf{14{,}300\times}$ fewer.

**Answer:** 20,480 trainable parameters (0.000186% of T5-XXL) and 7,680 (0.00698% of BERT-base).
Note the fraction gets **smaller** as the model grows, because $p\times d$ scales linearly in $d$ while
the model scales roughly as $d^2 L$ — which is part of why prompt tuning becomes relatively more
attractive at scale.

### N5. Why exhaustive discrete prompt search is impossible
**Given:** BERT's WordPiece vocabulary $|V| = 30{,}522$; $m = 5$ trigger-token positions.
**Find:** the size of the search space, and compare with AutoPrompt's cost.

1. Each of the $m$ slots can hold any of $|V|$ tokens independently, so the space is $|V|^m$.
2. $|V|^2 = 30{,}522^2 = 931{,}592{,}484 \approx 9.32\times10^8$.
3. $|V|^3 \approx 2.843\times10^{13}$; $|V|^4 \approx 8.679\times10^{17}$;
   $|V|^5 \approx \mathbf{2.65\times10^{22}}$.
4. At an optimistic $10^6$ prompt evaluations per second, that is
   $2.65\times10^{22}/10^6 = 2.65\times10^{16}$ seconds $= 8.4\times10^{8}$ years — about **840 million
   years**.
5. AutoPrompt's cost instead: per iteration, one backward pass per position plus $k$ real evaluations
   per position. With $k=100$, $m=5$, 10 iterations: $5 \times 100 \times 10 = \mathbf{5{,}000}$ real
   evaluations.
6. Reduction factor: $2.65\times10^{22}/5{,}000 = 5.3\times10^{18}$.

**Answer:** $|V|^m \approx 2.65\times10^{22}$ candidate prompts — unsearchable by any means. AutoPrompt
visits about 5,000, roughly $5\times10^{18}$ times fewer, because the gradient tells it which 100 of
30,522 tokens are worth actually trying. *That* is what "gradient-based" buys you.

### N6. The real cost of EvoPrompt
**Given:** population size $N = 10$ prompts, $T = 10$ generations, development set of 200 examples. Each
generation produces $N$ new candidates via one LLM crossover+mutation call each, and every candidate is
scored on the full dev set.
**Find:** total LLM calls.

1. Scoring the initial population: $10 \times 200 = 2{,}000$ inference calls.
2. Per generation, evolution calls: $10$ (one per new candidate).
3. Per generation, scoring calls: $10 \times 200 = 2{,}000$.
4. Over 10 generations: evolution $= 10\times10 = 100$; scoring $= 10 \times 2{,}000 = 20{,}000$.
5. Total scoring calls $= 2{,}000 + 20{,}000 = \mathbf{22{,}000}$.
6. Total LLM calls $= 22{,}000 + 100 = \mathbf{22{,}100}$.
7. At 0.4 s per call: $22{,}100 \times 0.4 = 8{,}840$ s $= \mathbf{2.46}$ hours (serially).

**Answer:** 22,100 LLM calls, dominated 221:1 by **dev-set scoring**, not by the clever evolutionary
operators. The dev-set size is the cost driver, which is why EvoPrompt implementations subsample it.
Compare N5: 22,100 evaluations to explore a space of fluent English prompts is cheap — but it is also
blind, which is the trade-off against AutoPrompt's 5,000 gradient-guided ones.

## Code

Two blocks. The first reproduces N3 — scoring the whole vocabulary against an embedding gradient and
taking the top-$k$. The second reproduces N2 and N4.

```python
import numpy as np

# A 6-word toy vocabulary with d = 4 input embeddings (rows of E).
vocab = ["cinematic", "Rating", "##omi", "terrible", "the", "whatsoever"]
E = np.array([
    [ 1.0,  0.0, -0.5,  0.5],   # cinematic
    [ 0.2, -1.6,  0.8,  0.5],   # Rating
    [-0.4, -0.5,  1.5,  1.0],   # ##omi
    [ 0.9, -0.2,  0.1, -0.7],   # terrible
    [ 0.0,  0.1,  0.0,  0.1],   # the
    [-1.0,  0.3, -0.2,  0.6],   # whatsoever
])

e_cur = np.array([0.5, -1.0, 0.25, 0.0])        # embedding of the token now in slot j
g     = np.array([0.8, -0.6, 0.20, 0.4])        # grad of log p(y|x_prompt) wrt that slot

scores   = E @ g                                 # AutoPrompt Eq. 2: w_in^T grad log p
delta_lp = scores - e_cur @ g                    # first-order change in log-likelihood
delta_L  = -delta_lp                             # change in the loss  L = -log p

order = np.argsort(-scores)                      # top-k, largest score first
print(f"e_cur . g = {e_cur @ g:+.4f}\n")
print(f"{'rank':<5}{'token':<12}{'w_in.g':>9}{'D log p':>10}{'D loss':>9}")
for r, i in enumerate(order, 1):
    print(f"{r:<5}{vocab[i]:<12}{scores[i]:>9.4f}{delta_lp[i]:>10.4f}{delta_L[i]:>9.4f}")

k = 3
print("\nV_cand (top-k, k=3):", [vocab[i] for i in order[:k]])
```

```
e_cur . g = +1.0500

rank token          w_in.g   D log p   D loss
1    Rating         1.4800    0.4300  -0.4300
2    cinematic      0.9000   -0.1500   0.1500
3    ##omi          0.6800   -0.3700   0.3700
4    terrible       0.5800   -0.4700   0.4700
5    the           -0.0200   -1.0700   1.0700
6    whatsoever    -0.7800   -1.8300   1.8300

V_cand (top-k, k=3): ['Rating', 'cinematic', '##omi']
```

The whole vocabulary was scored by one matrix–vector product `E @ g`. Note that `argsort` on `scores`
and on `delta_lp` give the identical order, because they differ by the constant 1.0500 — the point made
in §*Why that form*. Only `Rating` has a positive predicted gain, matching N3.

```python
def t5_layer_params(d):
    """Weight matrices per layer, ignoring biases/LayerNorm, with d_ff = 4d."""
    enc = 4*d*d + 2*(d*4*d)          # self-attn QKVO + FFN  = 12 d^2
    dec = 4*d*d + 4*d*d + 2*(d*4*d)  # self + cross + FFN    = 16 d^2
    return enc, dec

def compare(d, n_layers, p, n_tasks, model_total=None, name=""):
    enc, dec = t5_layer_params(d)
    full_per_task = (enc + dec) * n_layers if model_total is None else model_total
    soft_per_task = p * d
    print(f"--- {name}: d={d}, layers={n_layers}, p={p}, tasks={n_tasks}")
    print(f"  per-layer: encoder {enc//(d*d)}d^2, decoder {dec//(d*d)}d^2, sum {(enc+dec)//(d*d)}d^2")
    print(f"  full fine-tuning, extra storage : {full_per_task*n_tasks:,}")
    print(f"  prompt tuning,   extra storage : {soft_per_task*n_tasks:,}")
    print(f"  ratio                          : {full_per_task/soft_per_task:,.1f} x")
    print(f"  soft prompt as % of full       : {100*soft_per_task/full_per_task:.6f} %")

compare(1024, 24, 10, 3, name="deck p.112 - T5 (d=1024, 24 layers)")
compare(4096, 24, 5, 1, model_total=11_000_000_000, name="T5-XXL 11B, 20K-param prompt")
compare(768, 12, 10, 1, model_total=110_000_000,   name="BERT-base 110M, p=10")
```

```
--- deck p.112 - T5 (d=1024, 24 layers): d=1024, layers=24, p=10, tasks=3
  per-layer: encoder 12d^2, decoder 16d^2, sum 28d^2
  full fine-tuning, extra storage : 2,113,929,216
  prompt tuning,   extra storage : 30,720
  ratio                          : 68,812.8 x
  soft prompt as % of full       : 0.001453 %
--- T5-XXL 11B, 20K-param prompt: d=4096, layers=24, p=5, tasks=1
  per-layer: encoder 12d^2, decoder 16d^2, sum 28d^2
  full fine-tuning, extra storage : 11,000,000,000
  prompt tuning,   extra storage : 20,480
  ratio                          : 537,109.4 x
  soft prompt as % of full       : 0.000186 %
--- BERT-base 110M, p=10: d=768, layers=12, p=10, tasks=1
  per-layer: encoder 12d^2, decoder 16d^2, sum 28d^2
  full fine-tuning, extra storage : 110,000,000
  prompt tuning,   extra storage : 7,680
  ratio                          : 14,322.9 x
  soft prompt as % of full       : 0.006982 %
```

Matches N2 and N4 exactly.

## Exam pack

### Must-memorise

| Item | Exactly this |
|---|---|
| Automated prompt design | search or generation for the most effective prompt template **in a predefined search space** |
| Three approaches, in deck order | prompt paraphrasing · gradient-based discrete prompt search · prompt tuning |
| Token space vs continuous space | first two search **token space**; prompt tuning searches **continuous space** (soft prompts) |
| AutoPrompt Eq. 2 | $V_{\text{cand}} = \text{top-}k_{\,w\in V}\left[\mathbf{e}_{w}^{\top}\nabla\log p(y\mid\mathbf{x}_{\text{prompt}})\right]$ |
| Gradient taken w.r.t. | the **input embedding** of the $j$-th trigger token |
| First-order approximation | $f(\mathbf{x}+\epsilon\mathbf{v}) - f(\mathbf{x}) \approx \epsilon\,\mathbf{v}^{\top}\nabla f(\mathbf{x})$ |
| Loss change on a swap | $\Delta\mathcal{L} \approx (\mathbf{e}_{w'}-\mathbf{e}_w)^{\top}\nabla_{\mathbf{e}_w}\mathcal{L}$ |
| Why Eq. 2 drops $\mathbf{e}_w$ | $\mathbf{e}_w^{\top}\mathbf{g}$ is constant across candidates — it cannot change the top-$k$ |
| Class probability from an MLM | $p(y\mid\mathbf{x}_{\text{prompt}}) = \sum_{w\in V_y} p([\text{MASK}]=w\mid\mathbf{x}_{\text{prompt}})$ |
| Beam search in AutoPrompt | top-$k$ candidates per position, searched **left to right** across positions |
| Label-token step 1 | logistic classifier on the contextualized `[MASK]` embedding: $p(y\mid\mathbf{h})\propto\exp(\mathbf{h}\cdot\mathbf{y}+\beta_y)$ |
| Label-token step 2 | substitute $\mathbf{w}_{\text{out}}$ for $\mathbf{h}$: $s(y,w)=p(y\mid\mathbf{w}_{\text{out}})$, $V_y = \text{top-}k\,[s(y,w)]$ |
| EvoPrompt | LLM does **crossover** then **mutation**; fitness = dev-set accuracy; survival of the fittest |
| Prompt-tuning slogan | "Optimize the embeddings of a prompt, instead of the words" |
| Prompt-tuning shapes | $\mathbf{X}_e\in\mathbb{R}^{n\times d}$, $\mathbf{P}_e\in\mathbb{R}^{p\times d}$, $[\mathbf{P}_e;\mathbf{X}_e]\in\mathbb{R}^{(p+n)\times d}$ |
| What is updated | **only $\mathbf{P}_e$** — the backbone is frozen |
| Trainable count | $p \times d$ |
| Prompt- vs prefix-tuning | prompt-tuning = soft prompts at the **input only**; prefix-tuning = learned prefixes at **every layer** (Lec 46) |
| Power-of-Scale finding | prompt tuning **matches** model tuning **as model size increases** |

### Numbers worth knowing

| Quantity | Value |
|---|---|
| AutoPrompt trigger-token count in the deck's example | **5**, all initialized to `[MASK]` |
| SST-2: BERT manual → AutoPrompt | 63.2 → **82.3** test (dev 63.2 → 80.9) |
| SST-2: RoBERTa manual → AutoPrompt | 85.2 → **91.4** test (dev 85.3 → 91.2) |
| SST-2: BERT linear probing / finetuned | 83.4 / **93.5** test |
| SST-2: RoBERTa linear probing / finetuned | 88.8 / **96.7** test |
| SST-2: BiLSTM / BiLSTM+ELMo | 82.8 / 89.3 test |
| Discovered sentiment prompt | "unflinchingly bleak and desperate Writing academicswhere overseas will appear [MASK]." |
| Discovered label tokens (sentiment) | **pos:** partnership, extraordinary, ##bla · **neg:** worse, persisted, unconstitutional |
| Deck's beam-search probability trace | 0.01/0.05/0.03 → 0.18/0.11/0.08 → **0.95/0.89/0.77** |
| Prompt-tuning slide: model size | **11B** params per task copy under model tuning |
| Prompt-tuning slide: prompt size | **20K** params per task |
| Scale at which prompt tuning catches model tuning | $\approx 10^{10}$ parameters, SuperGLUE $\approx 90$ |
| Power-of-Scale runs reported | mean and s.d. over **3 runs** |
| Deck exercise p.112 | $d=1024$, 24 layers, $p=10$, 3 tasks → 2.11B vs 30,720 |
| Deck exercise p.104 | BERT, 2 classes, $\mid V_y\mid =3$ → **1538** learnable parameters |
| Key papers | AutoPrompt (Shin et al., 2020, arXiv 2010.15980) · EvoPrompt (2023, arXiv 2309.08532) · The Power of Scale (Lester et al., EMNLP 2021) · paraphrasing (Jiang et al., 2019) |

### Likely MCQ traps

- **"Gradient-based prompt search updates the prompt embeddings by gradient descent."** No. The prompt
  stays a **discrete token sequence**; the gradient only *ranks candidate substitutions*. The method
  that actually descends on embeddings is **prompt tuning**.
- **"Gradient-based prompt search fine-tunes the model."** No. No model weight is ever updated in
  AutoPrompt or in prompt tuning. The model is frozen in both.
- **Confusing the two top-$k$'s.** $V_{\text{cand}}$ is the gradient shortlist **within** a position;
  the beam width is **across** positions. Also distinct from $k$ in $V_y$ (label-set size).
- **"The top-$k$ candidates are the final prompt."** No — they are re-evaluated by a **real forward
  pass**, and only the best actually survives. The linear score is a proposal mechanism.
- **"$\mathbf{e}_w^{\top}\mathbf{g}$ must be subtracted or the ranking is wrong."** It is a constant
  across candidates; including it changes the values but not the order.
- **Gradient w.r.t. the output embedding.** It is w.r.t. the **input** embedding $\mathbf{w}_{\text{in}}$
  of the trigger slot. $\mathbf{w}_{\text{out}}$ appears only in label-token selection.
- **"Automatically found prompts are more fluent than human ones."** The opposite for AutoPrompt — they
  are typically ungrammatical token salad. EvoPrompt's *are* fluent, because an LLM writes them.
- **"$|V_y|$ changes the learnable-parameter count."** It does not. Only the per-label weight vector and
  bias are learned (N1).
- **"Prompt tuning always matches full fine-tuning."** Only **as scale grows**; at $10^8$ parameters it
  loses by several SuperGLUE points. The claim is scale-conditional.
- **"Prompt tuning and prompt design are the same."** Prompt *design* is hand-written few-shot
  prompting with no learning (the bottom curve on page 111). Prompt *tuning* learns $\mathbf{P}_e$.
- **Prompt-tuning vs prefix-tuning.** Input layer only vs every layer. The single most likely
  discrimination question in this pair of lectures.
- **"A soft prompt can be printed out as text."** No — $\mathbf{P}_e$'s rows are arbitrary points in
  $\mathbb{R}^d$ and generally correspond to no vocabulary item.
- **"EvoPrompt needs model gradients."** It needs only black-box API access; that is its main advantage
  over AutoPrompt.

### Self-test

1. State AutoPrompt's Eq. 2 and say what the gradient is taken with respect to.
2. Why can't you simply run gradient descent on the trigger tokens themselves?
3. Give the first-order approximation of $\Delta\mathcal{L}$ when token $w$ is swapped for $w'$, and
   explain why the deck's score drops one of its terms.
4. With $\mathbf{g}=(1,-2,0.5)$, $\mathbf{e}_w=(0,1,2)$ and candidates
   $\mathbf{e}_A=(2,0,0)$, $\mathbf{e}_B=(0,-1,1)$, which candidate is ranked first and what is its
   $\Delta\mathcal{L}$?
5. BERT, 3 classes, $|V_y|=5$. How many learnable parameters in label-token selection?
6. In what order does AutoPrompt's beam search move across trigger positions, and what scores a beam?
7. Give the shapes of $\mathbf{X}_e$, $\mathbf{P}_e$ and their concatenation in prompt tuning, and say
   which is trained.
8. A model has $d=2048$ and you use a 20-token soft prompt for 4 tasks. How many trainable parameters
   in total?
9. What exactly does the "Power of Scale" result claim, and what does it *not* claim?
10. One sentence each: how does prompt-tuning differ from prefix-tuning, and from prompt design?

<details><summary>Answers</summary>

1. $V_{\text{cand}} = \text{top-}k_{\,w\in V}\left[\mathbf{e}_w^{\top}\nabla\log p(y\mid\mathbf{x}_{\text{prompt}})\right]$; the gradient is w.r.t. the **input embedding of the $j$-th trigger token**.
2. Because the prompt is a sequence of discrete vocabulary indices. There is no continuous variable to descend on — a gradient step would land between tokens, which is not a token.
3. $\Delta\mathcal{L} \approx (\mathbf{e}_{w'}-\mathbf{e}_w)^{\top}\nabla_{\mathbf{e}_w}\mathcal{L}$. The $\mathbf{e}_w$ term is identical for every candidate, so it shifts all scores by a constant and cannot alter the top-$k$.
4. $\mathbf{e}_w^{\top}\mathbf{g} = 0 - 2 + 1 = -1$. $\mathbf{e}_A^{\top}\mathbf{g} = 2$; $\mathbf{e}_B^{\top}\mathbf{g} = 2 + 0.5 = 2.5$. B ranks first; $\Delta\log p = 2.5-(-1) = 3.5$, so $\Delta\mathcal{L} = -3.5$.
5. Step 1 only: $768\times3$ weights $+\,3$ biases $= 2304 + 3 = \mathbf{2307}$. $|V_y|=5$ is irrelevant.
6. **Left to right** across positions, keeping the top-$k$ candidates per position; each beam is scored by $p(y\mid\mathbf{x}_{\text{prompt}}) = \sum_{w\in V_y}p([\text{MASK}]=w\mid\mathbf{x}_{\text{prompt}})$ on the batch.
7. $\mathbf{X}_e\in\mathbb{R}^{n\times d}$ (frozen embeddings of the real tokens), $\mathbf{P}_e\in\mathbb{R}^{p\times d}$ (**trained**), concatenation $[\mathbf{P}_e;\mathbf{X}_e]\in\mathbb{R}^{(p+n)\times d}$. Only $\mathbf{P}_e$ is updated.
8. $20\times2048 = 40{,}960$ per task; $\times 4 = \mathbf{163{,}840}$.
9. It claims prompt tuning **converges to** model-tuning quality **as model size increases** (meeting it around $10^{10}$ parameters on SuperGLUE) while needing one frozen model for all tasks. It does **not** claim parity at small scale — there prompt tuning is clearly worse.
10. Prompt-tuning learns vectors at the **input embedding layer only**, prefix-tuning learns them at **every layer** ([Lec 46](../week-10/46-peft-adapters-prefix.md)). Prompt design is **hand-written** few-shot prompting with **no learned parameters at all**.

</details>

## Beyond the slides

**Gap:** The deck never says how soft prompts are **initialized**, yet Lester et al. show it matters
enormously at small scale.
**Why it matters:** Random initialization is much worse than initializing $\mathbf{P}_e$ from the
embeddings of real vocabulary tokens (or, for classification, from the embeddings of the class label
strings). The difference vanishes at $10^{10}$ parameters — which is part of *why* the curves converge
with scale, not just an incidental detail. Likewise, prompt **length** $p$ matters at small scale
(longer is better up to ~100 tokens) and barely at all at XXL.

**Gap:** The deck shows AutoPrompt only in the **masked-LM** setting, with a `[MASK]` slot.
**Why it matters:** The same machinery works for autoregressive models by putting the trigger tokens
before the input and scoring the next-token distribution — and this is exactly the recipe behind
**adversarial trigger** attacks and many jailbreak-style prompt searches (GCG and its descendants).
Recognising AutoPrompt as the ancestor of that literature is worth a sentence in an essay answer.

**Gap:** Prompt tuning's **inference-time cost** is silently non-zero.
**Why it matters:** Prepending $p$ soft tokens lengthens every sequence from $n$ to $p+n$, and
self-attention is quadratic in sequence length. A 100-token soft prompt on a 128-token input nearly
doubles the attention cost of every forward pass, forever. "Parameter-efficient" does not mean
"compute-efficient" — a distinction Week 10 returns to with adapters (which add depth, not length) and
LoRA (which adds neither at inference, once merged).

**Gap:** No discussion of **why** a nonsense prompt works.
**Why it matters:** The deck presents "unflinchingly bleak and desperate Writing academicswhere
overseas will appear [MASK]" as a curiosity. The mechanistic reading is that the trigger tokens are not
communicating meaning; they are steering the residual stream into a region where the `[MASK]`
distribution is already sentiment-separable — which is the same picture [Lec 42](42-why-icl-works.md)
builds for in-context learning. It also implies these prompts are **model-specific**: a BERT trigger
does nothing for RoBERTa, which is the honest limitation of the whole discrete-search line.

## Cut from the slides

Page 93 is the title card, page 94 the two-item "Concepts Covered" list (Discrete Prompt Search, Prompt
Tuning), page 113 the single reference (Kamath et al., *Large Language Models: A Deep Dive*, 2024) and
page 114 the "Thank You" card — none carry content, and I have folded their substance into the section
ordering above. Page 101 largely repeats page 99's equation and figure, so I quoted its new material
(the re-evaluation step and the $V_y$ summation) in prose rather than embedding a near-duplicate
image. Page 108's Differential
Evolution variant of EvoPrompt is summarised in prose rather than shown, because its diagram adds a DE
mutation schema ($a + F(\mathbf{b}-\mathbf{c})$) that is a detail of genetic algorithms rather than of
NLP; the GA version on page 107 carries the examinable idea. Manual prompting, templates and
verbalizers belong to [Lec 41](41-prompting-1.md); chain-of-thought and its relatives to
[Lec 43](43-advanced-prompting.md); beam search itself to
[Lec 19](../week-04/19-decoding-strategies.md); backpropagation to
[Lec 9](../week-02/09-backpropagation.md); and prefix-tuning, adapters and LoRA to
[Lec 46](../week-10/46-peft-adapters-prefix.md) and [Lec 47](../week-10/47-lora-and-variants.md) — each
gets a sentence and a link here, no more.
