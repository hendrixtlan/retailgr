# RetailGR

Open, cloud-native generative recommendations for retail, end to end:

```
events ─▶ Kafka ─▶ lakehouse ─▶ sequences ─▶ HSTU ─▶ retrieval ─▶ policy ─▶ API
          topics   Iceberg      per variant  PyTorch  + ranking   stock,
          schemas  or Parquet   time split            embeddings  price, promos
```

Everything is an open component — Kafka, Spark on Kubernetes, Apache Iceberg,
PyTorch — so the whole thing runs on any cloud or on a laptop.

It exists to answer three questions with data instead of opinion:

> **1. At what granularity should the model see an item — SKU, style-colour, or
> product — and does the answer differ by category?**
>
> **2. Does HSTU actually beat SASRec on this data, and which of its parts earn
> their place — with intervals, not point estimates?**
>
> **3. Does the whole request path fit inside a 100 ms latency budget?**

The SKU stays the system of record everywhere. Only the *model token* changes,
and `configs/granularity.yaml` decides it per category.

---

## Quick start

```bash
make install        # virtualenv + dependencies
make test           # unit tests, no Spark, no broker

# the model questions
make stage1         # build the lakehouse, run every granularity variant
make ablate         # switch off HSTU components one at a time, with intervals
make movielens      # the same questions on real timestamps

# the serving question
make bundle         # train and export a serving bundle
make bench          # measure the latency budget, stage by stage
make serve          # run the API on http://127.0.0.1:8080
```

Nothing above needs Docker. The in-process broker and online store are
faithful about the guarantees the pipeline depends on, so the full path runs
with no infrastructure; `make up` swaps in real Kafka, Redis, MinIO and an
Iceberg catalog when you want them.

`make stage1` takes about four minutes on a laptop and writes
`artifacts/latest.md`. Here is that report from an actual run on the default
synthetic dataset (3,000 users / 3,081 SKUs), so you can check your own run
against it:

| Variant | Tokens | Distinct SKUs | SKUs per token | Tokens < 5 events |
| --- | --- | --- | --- | --- |
| `sku` | 3033 | 3081 | 1.016 | 858 (28.3%) |
| `style_color` | 809 | 3081 | 3.808 | 16 (2.0%) |
| `product` | 499 | 3081 | 6.174 | 8 (1.6%) |
| `config` | 944 | 3081 | 3.264 | 36 (3.8%) |

Every metric carries a 95% bootstrap interval over users, because a mean over
~1,000 users has an interval wide enough to swallow most of the differences
worth arguing about. (The table below predates the consent filter, which cut
the test split from 1,101 users to 943 — see *The dataset moved under every
number* further down.)

| Variant | Model | Recall@200 | NDCG@200 | Coverage@200 |
| --- | --- | --- | --- | --- |
| `sku` | popularity | 0.3261 [0.3109, 0.3412] | 0.1259 [0.1185, 0.1337] | 0.0748 |
| `sku` | sasrec_small | 0.2926 [0.2744, 0.3108] | 0.1304 [0.1205, 0.1409] | 0.8042 |
| `sku` | hstu_small | 0.2828 [0.2643, 0.3011] | 0.1268 [0.1169, 0.1371] | 0.8583 |
| `product` | popularity | 0.5326 [0.5128, 0.5524] | 0.1875 [0.1787, 0.1962] | 0.4640 |
| `product` | sasrec_small | 0.5294 [0.5109, 0.5479] | 0.2227 [0.2115, 0.2338] | 0.9980 |
| `product` | hstu_small | 0.4736 [0.4542, 0.4924] | 0.2080 [0.1963, 0.2196] | 0.9980 |
| `config` | popularity | 0.4095 [0.3923, 0.4276] | 0.1509 [0.1429, 0.1589] | 0.2402 |
| `config` | sasrec_small | 0.4219 [0.4031, 0.4414] | 0.1797 [0.1682, 0.1909] | 0.9979 |
| `config` | hstu_small | 0.4135 [0.3941, 0.4323] | 0.1758 [0.1647, 0.1868] | 0.9989 |

Three things worth noticing, all of them the harness doing its job:

- At SKU level, **28% of tokens have fewer than five training events** and both
  sequence models fall *below* the popularity baseline on recall. That is the
  fragmentation cost of per-variant tokens, visible as a number.
- The popularity baseline covers 7% of the catalog at SKU level; the sequence
  models cover ~100%. Coverage is not a vanity metric — it is what keeps the
  long tail reachable.
- **HSTU and SASRec are a tie here.** The point estimates differ by 2–6% and it
  is tempting to read that as a result; the paired test says only the
  `product` variant's gap survives adjustment. Those intervals above overlap
  heavily, which is exactly why every comparison in this repository is paired
  rather than eyeballed — see below.

**Read all of it as machinery, not as an answer.** This generator gives sizes
no preference of their own, so collapsing them is bound to win here. The answer
for your catalog has to come from your data, which is the whole point of having
the harness.

---

## What runs, in order

```
raw dataset ──▶ bronze ──▶ silver ──▶ gold sequences ──▶ models ──────▶ report
  adapter       as-is      cleaned     per variant       popularity     Markdown
                           + hierarchy  + time split      SASRec         + JSON
                                                          HSTU
```

| Stage | Command | Output |
| --- | --- | --- |
| Synthetic data | `retailgr generate-data` | `data/raw/synthetic/*.csv` |
| Bronze | `retailgr ingest` | `bronze.interactions`, `bronze.catalog` |
| Silver | `retailgr silver` | `silver.interactions`, `silver.item_hierarchy` |
| Gold | `retailgr sequences --granularity sku` | `gold.sequences_sku`, `gold.vocab_sku` |
| Train + score | `retailgr evaluate --variant sku` | metrics as JSON |
| Everything | `retailgr experiment` | `artifacts/latest.md` and `.json` |
| HSTU ablation | `retailgr ablate --seeds 13 17 23` | `artifacts/ablation_config.md` |
| Ranker negatives | `retailgr rank-experiment` | `artifacts/ranker_negatives.md` |
| Platform audit | `retailgr audit` | `artifacts/platform_audit.md`, exits non-zero |
| Iceberg, no JVM | `retailgr sequences --set warehouse.backend=iceberg --set warehouse.iceberg.engine=pyiceberg` | the lakehouse, no Spark jars |
| Pod sizing | `retailgr sizing` | `artifacts/sizing.md` |
| Mutation check | `make mutants` | breaks each claim, reports what nothing caught |
| Export for serving | `retailgr export-model` | `artifacts/bundle/` |
| Stream → online store | `retailgr bootstrap` | user tails, item state, fallbacks |
| Serve | `retailgr serve --bootstrap` | the API on `:8080` |
| Latency | `retailgr bench` | `artifacts/latency.md` |

Run the CLI as `python -m retailgr.cli <command>` or, after `make install`, as
`retailgr <command>`. Any value in `pipeline.yaml` can be overridden per run
with `--set key.path=value`.

---

## The pieces that matter

### Granularity is config, not code

```yaml
# configs/granularity.yaml
default: sku
by_category:
  apparel:     {level: style_color}
  footwear:    {level: style_color}
  electronics: {level: product, plus_attributes: [capacity]}
  grocery:     {level: sku}
```

`src/retailgr/granularity.py` implements this twice — once in pure Python, once
as a Spark expression — and `tests/test_pipeline.py` asserts the two agree on
real catalog rows. That test is there because training/serving skew usually
starts exactly here.

### The time split has no leakage

```
|<----------- train ----------->|<-- val -->|<-- test -->|
0                              T1          T2          Tmax
```

Global cutoffs, not per-user leave-one-out. A per-user split lets one user's
held-out event sit before another user's training events, which inflates every
number it touches. Two tests assert that no evaluation history crosses its
cutoff.

The vocabulary is built **from training events only**. A token first seen after
T1 is a cold-start item that an id-based model cannot retrieve; counting it
would flatter the metrics.

### Metrics

Recall@K, NDCG@K, HitRate@K and catalog coverage, normalised by
`min(len(targets), K)` so a user with many targets is not capped by the window
length. Reported overall *and by the user's dominant category*, because the
granularity decision is per category.

`k` values at or above the vocabulary size are **dropped, not clamped** — at
that k the model returns the whole catalog and the metric stops discriminating.
The report says which were dropped.

### How every comparison is tested

A mean over a thousand users is not a fact, and two models that differ by 2%
on such a mean usually do not differ at all. Three pieces of machinery, in
`evaluation/stats.py`:

**Bootstrap intervals** on every metric, over users. Reported everywhere a
mean appears.

**Paired comparison** for model-vs-model, which is what makes these
comparisons possible at all. Both models score the same users, so the test
runs on the per-user *difference* — removing the between-user spread, which is
several times larger than any gap between two models. `tests/test_stats.py`
pins the size of that effect: on data with a consistent 0.01 improvement, two
independent bootstrap intervals overlap completely while the paired test finds
it at p<0.01. The p-value is a sign-flip permutation test, which is the right
null for a paired design and assumes no normality.

**Holm-Bonferroni** across each table of tests. A report that shows eight
comparisons and reads the ones under 0.05 is taking eight chances to be
fooled, not one; with eight true nulls the odds of at least one false positive
are about 34%. The verdict column uses the adjusted value, and it changes its
mind about exactly the marginal rows that matter.

None of that covers **training variance**, and no bootstrap can: resampling
users says nothing about what another seed would have produced.
`ablate --seeds 13 17 23` trains each configuration once per seed and reports
the spread, which is the only way to tell a 2% architectural difference from
a 2% difference between two runs of the same architecture. On the synthetic
set that spread is 0.0002–0.0013 — small enough that user sampling, not the
seed, is the binding constraint.

Writing this changed the results. Two claims in an earlier draft of this
README did not survive it, and they are called out where they were made.

### The models

| Model | Reads | Role |
| --- | --- | --- |
| **Popularity** | nothing per-user | The floor. Nothing ships until it beats this. |
| **SASRec** | items | The reference. Causal self-attention, tied item embeddings. |
| **HSTU** | items **+ actions + timestamps** | Retrieval: the candidate. |
| **HSTU ranker** | items and actions interleaved, **+ candidates** | Ranking: four heads, target-aware. Currently blocked by the gate — see below. |

The two sequence presets are matched on hidden size, depth, heads, dropout,
sequence length, loss, batch size, epochs and seed, so a gap between them is
architecture and modalities, not capacity.

### HSTU, and why it is not a Transformer

`src/retailgr/models/hstu.py` follows Equations (1)–(3) of
[Zhai et al., ICML 2024](https://arxiv.org/abs/2402.17152) and was checked
against the authors'
[reference code](https://github.com/meta-recsys/generative-recommenders)
(Apache-2.0). Three details are load-bearing and easy to get wrong:

1. **No softmax.** Attention is `silu(QKᵀ + rab) / N` — normalised by a
   constant sequence length, not a data-dependent denominator, so the *number*
   of prior related events survives as signal. The paper's ablation puts HR@10
   at .0617 with softmax against .0893 without.
2. **The causal mask is multiplicative and applied after the activation.**
   `silu(0) = 0`, but an additive `-inf` mask before the activation would leak
   a constant from masked positions. `tests/test_hstu.py` asserts that changing
   the last event cannot alter any earlier position.
3. **Actions are a modality.** Per Table 1 of the paper, each position fuses
   `(item, action)`, and a position supervises the next item only when that
   item's action was positive.

### What the ablation found — and at which cutoff

`make ablate --seeds 13 17 23` switches off one component at a time, trains
each configuration three times, and compares every one against the full model
*paired on the same users*.

**This section used to report a result that was true and misleading.** Its
table said the temporal bias costs 3.7% and the action modality is worth
2.5%, and it did not say the metric was **NDCG@200** — the report rendered
only the deepest cutoff, under a column called "Difference". At **NDCG@10**,
the ten items a customer is actually shown, the same comparison was +0.0003
at p=0.740. The report now renders both cutoffs, with the metric in every
title and every significant verdict, so a verdict cannot be copied without
its cutoff again.

Re-running it on the current data, with models trained to their own best
epoch:

| vs `hstu_full` | Δ NDCG@10 | p adj. | Δ NDCG@200 | p adj. | at @200 |
| --- | ---: | ---: | ---: | ---: | --- |
| `sasrec (baseline)` | −0.0026 | 1.000 | +0.0073 | 0.030 | 4.0% better |
| `hstu_no_temporal_bias` | −0.0017 | 1.000 | +0.0068 | 0.029 | **3.7% better** |
| `hstu_no_rab` | +0.0013 | 1.000 | +0.0067 | 0.007 | **3.7% better** |
| `hstu_no_actions` | −0.0012 | 1.000 | −0.0113 | 0.003 | **6.2% worse** |
| `hstu_seq_normalised` | −0.0007 | 1.000 | +0.0039 | 0.024 | 2.1% better |
| `hstu_all_targets` | −0.0002 | 1.000 | +0.0007 | 0.212 | tie |

**At the top ten, nothing matters.** No component of HSTU changes NDCG@10,
and HSTU ties SASRec — in this run, in the fixed-schedule run on the same
data, and in the old run on the old data. Every one of those eighteen
comparisons is a tie. Since the ranker is gated off, retrieval's top ten is
what the service actually serves, so this is the result that describes the
product today.

**At depth, three effects are robust.** They reproduce across a change of
dataset and a change of training schedule, and the direction never moves:

| @200, vs `hstu_full` | old data, 8 epochs | current data, 8 epochs | current data, converged |
| --- | ---: | ---: | ---: |
| no temporal bias | 3.7% better | 4.2% better | 3.7% better |
| no relative bias | 3.5% better | 5.3% better | 3.7% better |
| no actions | 2.5% worse | 3.8% worse | 6.2% worse |

Depth is not irrelevant here: retrieval hands the ranker its top 300, so
recall at 200 is the candidate generator's job. These are findings about
**candidate generation**, not about what a customer sees. The bias terms'
learned time buckets have no real signal to fit — this generator scatters
events uniformly inside a session — and the action modality carries real
information about which items get engagement.

**Two @200 verdicts are fragile, and they are the interesting ones.**
SASRec against HSTU and per-sequence normalisation were ties at eight epochs
and became significant once each model stopped at its own best epoch. A
verdict that flips with the stopping rule is a statement about the stopping
rule. The reason is visible in the seed spread: early stopping watches
NDCG@10, which plateaus by epoch 3–8, and each seed stops somewhere
different on that plateau (`hstu_full` stopped at 6, 8 and 3) — so the
deeper cutoff, which is still moving, picks up the variance. `hstu_full`'s
spread at @200 went from ≤0.0025 across seeds to 0.0090. What survives every
schedule and both cutoffs is narrower and still worth stating: **HSTU never
beats SASRec here.**

### Training to convergence, which changed nothing that was served

`fit(train, val)` was the signature of all four models, and nine call sites
passed `data.val`. **None of the implementations read it.** Every model
trained a fixed eight epochs while every call site suggested validation was
steering training, and for a long time this README listed "train to
convergence" as a precondition for believing anything about HSTU.

`retailgr convergence` measured it. Three seeds, both schedules scored on
test, paired per user and pooled across seeds:

| Model | best epoch per seed | converged − 8 epochs, NDCG@10 | p |
| --- | --- | ---: | ---: |
| `sasrec_small` | 3, 3, 7 | −0.0012 [−0.0040, +0.0015] | 0.401 |
| `hstu_small` | 6, 8, 3 | −0.0001 [−0.0016, +0.0013] | 0.845 |

Validation NDCG@10 peaks by epoch 3–8 and then sits flat while the training
loss keeps falling from ~6.4 to ~3.5. **The models were not undertrained;
they run out of data long before they run out of epochs.** The lever was
never more training — it is more data, or less capacity. The HSTU–SASRec
comparison at NDCG@10 is a tie at both schedules (p=0.172 at eight epochs,
p=0.779 converged).

Early stopping is on in every shipped retrieval config anyway, and the
reason is not this dataset: eight epochs was tuned to nothing and happened to
land on this dataset's plateau, and the next dataset will not be this one.
The details are where early stopping goes wrong quietly, so each is a test:
selection on validation and never on test (checked structurally across every
call site), the validation metric scored exactly as test is, dropout switched
back on after every check — `evaluate` calls `net.eval()`, and forgetting to
undo it trains every later epoch as a different network — and the best
weights kept as a copy rather than a live reference the next optimiser step
would overwrite. The ranker still trains a fixed schedule: its natural
criterion is slate AUC, which needs retrieval's candidates for every
validation user on every epoch, and a test fails the day that changes.

### The dataset moved under every number, and nothing noticed

The ablation above has **943** test users. The run this README used to quote
had **1,101**. The difference is the consent filter added in the privacy
work, which removes about 15% of users before silver — and every model
number in this README had been computed on the dataset from before it, while
the pipeline had been producing a different one since.

Directions survived; magnitudes did not. The relative-bias effect moved from
3.5% to 5.3%, the action effect from 2.5% to 3.8%, before convergence moved
them again. The documentation-drift test checks the test count, not the
numbers, so it could not see this. The fix is in the reports rather than the
tests: every ablation report now states its test-user count in its header,
so a quoted number carries the dataset it came from.

---

## The streaming path

One event definition, two consumers:

```
producers ─▶ interactions.v1 ─┬─▶ BronzeSink ─────────▶ bronze.interactions
                              └─▶ SessionStateConsumer ▶ online store
catalog/pricing/inventory CDC ───▶ SessionStateConsumer ▶ item state (compacted)
```

- **The schema is written down once.** `streaming/schemas.py` holds the Avro
  contract, and the Spark DDL for bronze is *derived* from it, so a field added
  to the topic cannot be silently missing from the lakehouse. Validation runs
  at the producer, not the consumer: a malformed event that reaches the topic
  is already everyone's problem.
- **`interactions.v1` is keyed by `user_id`**, which is what keeps one user's
  events ordered within a partition — the assumption the sequence builder
  makes. Catalog, pricing and inventory are compacted and keyed by SKU, so
  they are current state rather than a log.
- **The session-state consumer resolves the model token** with the same
  `GranularityResolver` the Spark job uses, and writes the token into the
  user's tail. Resolving it at request time from a different config is the
  classic origin of training/serving skew.
- **`recs.served.v1` is not optional.** Without an exposure log you cannot
  tell a product the customer rejected from one they never saw, and every
  label you train on afterwards is biased.

`retailgr bootstrap` replays the *same* silver table that trained the model
through the broker and materialises the online store from it. That is the join
between the two paths: if the replay produces events the consumer cannot turn
back into equivalent state, they have drifted.

## Serving

```
context ──▶ retrieval ──▶ policy filter ──▶ ranking ──▶ policy re-rank ──▶ API
tail from   user vector   stock, region,    4 heads,    promos, price,     + exposure
the store   → top-K ANN   already bought    M-FALCON    diversity, pins    log
```

Measured, not claimed (`make bench`, 300 requests over real user histories,
480-item catalog). Both columns are real runs — the ranker exists, and the
second column is it forced on:

| Stage | p99 without ranker | p99 with ranker | Budget |
| --- | --- | --- | --- |
| context | 0.100 | 0.098 | 10 |
| retrieval | 2.770 | 2.797 | 25 |
| filter | 0.336 | 0.483 | 5 |
| ranking | 0.002 | **19.726** | 35 |
| re-rank | 1.428 | 1.937 | 5 |
| **end to end** | **4.776** | **25.560** | **100** |

233 req/s single-threaded without the ranker, and every response came from the
model path — no fallbacks, which matters because a fallback is cheap to serve
and would flatter the numbers.

The ranker lands at 19.7 ms p99 against the 35 ms the architecture allocated
it, scoring 300 candidates in one forward pass. It fits. **It is still not
served by default**, and the reason is below.

Two caveats about the table:

- **480 items is not a catalog.** Exact retrieval is one BLAS matmul, which is
  the right choice up to a few hundred thousand items and the wrong one at ten
  million; `serving.retrieval_index: faiss` is there for that, behind the same
  interface.
- **In-process measurement.** It covers context assembly, retrieval, the model
  forward pass and the policy layer, but not HTTP, TLS or network.

### The ranker, and why it is blocked

Retrieval answers "which few hundred items are plausible". Ranking answers "of
these, which will this customer click, buy, and *keep*". `models/ranker.py`
implements the paper's ranking formulation: items and actions **interleaved**
as separate positions, candidates appended so the interaction happens inside
the encoder, and four heads — click, cart, purchase, and return given a
purchase.

Two pieces of it are worth calling out:

- **M-FALCON is verified, not asserted.** Scoring 300 candidates in one
  forward pass is only legitimate if it gives the same answer as 300 separate
  passes. `tests/test_ranker.py` asserts that equivalence, and getting there
  required two real fixes: candidates must not attend to each other, and every
  candidate must sit at the *same* relative position, or its score depends on
  its slot in the batch. A third — the attention normaliser must be a constant
  rather than the sequence length — was caught by that test failing.
- **Return labels are linked, and matured.** A return arrives days after the
  purchase as its own event, so the sequence builder joins them on
  `user_id + order_id + sku`. A purchase newer than the return window is
  marked *immature* and excluded from the head, because labelling it "kept"
  would teach the model that recent purchases are safe. On the synthetic set:
  2,159 training purchases, 1,923 matured, 16.1% returned, 236 correctly
  withheld.

**And then it lost.** `retailgr export-model` runs the offline gate and blocks
it:

```
## Gate: BLOCKED
the ranker's MRR is 0.17x retrieval's on the same task,
so re-ranking would be a regression with extra latency
```

The heads calibrate well — AUC 0.96 click, 0.98 purchase, 0.80 return — and
re-ranking still made NDCG@10 **82% worse** than retrieval order. Three
measurements, in the order they were needed, untangled why:

| Signal | hit@1 | MRR | Median rank of the true next item (of 100) |
| --- | --- | --- | --- |
| random | 0.010 | 0.052 | 50 |
| ranker, best head | 0.060 | 0.134 | 27 |
| ranker, blended | 0.045 | 0.100 | 38 |
| **retrieval** | **0.415** | **0.595** | **2** |

1. **The ranker has signal** — 6x better than random. It is not broken.
2. **The blend was destroying some of it.** The heads do not share a scale and
   should not: P(purchase) spans 0.0001–0.007 while P(cart) reaches 0.11,
   because a purchase genuinely is rarer. Weighting them as if comparable
   (purchase 1.0, cart 0.3) made the blended score *equal to* 0.3 × cart and
   threw the purchase head away. The fix is an expected-value formulation
   where the weights are relative business value, so rarity and worth cancel.
3. **Retrieval is simply 6x better at ordering.** That is the real finding,
   and the cause is a training-signal asymmetry: retrieval learns against the
   full softmax — every item in the vocabulary, at every position, every step.
   The ranker learns against sampled negatives at *one* position per user per
   epoch. Roughly two orders of magnitude less discriminative signal.

So the ranker is exported to the bundle directory but not wired into the
request path, and the API serves retrieval order. `--force-ranker` overrides
the gate; the gate exists so that overriding is a decision rather than an
accident, and it **fails closed** — no measurement means no ship.

### Calibration, and the number that explained the rest

The blend is an expected value, so it multiplies each head's probability by a
business figure. AUC says nothing about whether those probabilities are on the
probability scale — it is invariant to exactly the transform that breaks them —
so `evaluation/calibration.py` measures it directly: Brier score against the
base-rate constant, expected calibration error over quantile bins, the Murphy
decomposition, and a reliability table per head.

The measurement has to happen on the right distribution, and that turned out to
be most of the work. The probabilities the blend consumes come from *candidate*
positions, where training saw one true item against 256 uniform-random ones — a
base rate of 1/257 by construction, a property of the sampler. Serving shows
retrieval's top 300 instead. So everything here is fitted on validation slates
built the way the request path builds them and measured on test slates, never
on the training mixture.

Building that set surfaced a defect nothing else had:

> **On this dataset, 100% of click, cart and purchase events land on an item
> the customer has already interacted with.** The serving policy excludes seen
> items, so the first version of this measurement — which used the customer's
> immediate next event as the positive, the way `evaluate_discrimination` does —
> came back with a base rate of exactly **zero**, for every head, on every
> slate. Not a bug in the measurement: under `exclude_seen`, the next
> engagement is not something this system can be scored on at all. The fix was
> to label against the held-out future window, which needed target *actions*
> in the gold tables — "did they engage" and "did they buy" are different
> labels on the same slate, and only one of them is what the purchase head
> predicts.

With that in place, the heads are 1.4–1.6x under-confident and calibration
repairs it — ECE falls from 1.2x the base rate to about 0.27x. But the column
worth reading is the last one:

| Head | Resolution on history | Resolution on the slate |
| --- | --- | --- |
| `click` | 40.7% | **0.045%** |
| `cart` | 40.6% | **0.043%** |
| `purchase` | 24.7% | **0.015%** |
| `return` | 19.5% | – |

> These are the numbers after `history_loss_weight` dropped to 0.1 — see
> [the section below](#the-number-that-was-never-about-ranking), which is
> where the left-hand column stopped being believed. It was 81.4% / 0.02%,
> and the fall in it is the *intended* effect of down-weighting a loss that
> a five-entry lookup table already solved. The gap narrowed from ~4000x to
> ~900x, and both halves moved for the right reason.

Resolution is the share of the outcome's variance a head's predictions
actually explain, and calibration cannot change it — a monotone rescaling
moves where probabilities sit, never how well they separate outcomes.

*Given an item is in the customer's history, which action did they take on
it?* the heads explain tens of percent of the variance. *Given an item is in
a slate we are about to show, will they act on it?* they explain hundredths
of one percent. Every AUC in this report, and in the paper this design
follows, is measured on the left-hand question. The ranker is deployed on the
right-hand one.

That gap looked like it subsumed every earlier ranker finding, and for a
while this README said so. It does not, for two reasons found later and
documented below: resolution share is quadratic in the deviation from the
base rate, so ~144x of the gap is the metric comparing a 0.2% base rate with
a 24% one — and the left-hand number was measuring the data generator, not
ranking.

**Calibrating made the blend worse, and that is reported rather than buried.**
Per-head MRR is unchanged to four decimals (0.2031 → 0.2031 — the monotonicity
invariant, confirmed on the real model), but blended MRR fell from 0.1908 to
0.1413. The mechanism is measurable: on a real slate the raw heads span
0.000011–0.0011 for `purchase` and **0.0196–0.8113** for `return`, because the
return head is calibrated on history positions — the only place a return
outcome exists — where the base rate is 14.6%. Compressing the other three
heads onto the slate's scale left `(1 − P(return))` as the blend's dominant
source of variation, and the return head orders at MRR 0.058 against random's
0.052. The gate had already refused to ship precisely that:

```
## Gate: BLOCKED
| Criterion                            | Measured        | Threshold | Verdict  |
| Orders at least as well as retrieval | 0.31x           | >= 1.00x  | **fail** |
| Probabilities on the right scale     | 1.2x off (cart) | <= 4.0x   | **fail** |

calibration could not be measured on the serving distribution for `return`
(it is calibrated on history positions, which is a different distribution
from the slate it is applied to), so the expected-value blend would be
multiplying money by numbers of unknown scale
```

The gate now has two criteria because a two-stage ranker has two independent
ways to be wrong, and both fail closed. A calibrator that cannot be validated
on the distribution it will be applied to counts as unmeasured, not as
approximately fine.

Three things the fitting refuses to do, each of which it was caught doing
first: install a correction fitted on fewer than ten outcomes; install one
that inverts a head's ordering (a signal-free head's slope is pure noise, and
flipping it genuinely improves calibration); and install one whose improvement
does not survive being measured out-of-sample — a two-parameter family always
shaves something off the in-sample ECE, including on a head that is already
calibrated. Each returns the identity and says so in the report.

### Testing the obvious explanation, and watching it fail

The cheap hypothesis for that collapse was the negative sampler. The ranker
learns its candidate positions against 256 items drawn uniformly from the
vocabulary, almost none of which retrieval would ever surface — so "positive
versus 256 uniform" is solvable from an item prior, which is precisely what
retrieval already provides. At serving it is handed retrieval's top 300, where
every candidate is plausible and the prior says nothing. Train on the easy
question, deploy on the hard one.

`retailgr rank-experiment` tests that instead of assuming it. One retrieval
model is trained and shared, its own candidates become the pool, and three
mixes are compared — with the paired test and the Holm adjustment the rest of
the project uses:

| Variant | click | cart | purchase | blended MRR | vs `uniform`, paired | p adj |
| --- | --- | --- | --- | --- | --- | --- |
| **retrieval, on its own slate** | **0.17%** | **0.17%** | **0.09%** | 0.6828 | – | – |
| `uniform` _(baseline)_ | 0.03% | 0.02% | 0.01% | 0.1389 | – | – |
| `half_hard` | 0.01% | 0.01% | 0.01% | 0.0960 | **30.8% worse** | **0.001** |
| `all_hard` | 0.02% | 0.03% | 0.01% | 0.1312 | no difference detected | 0.611 |

**Which mix comes out worst is not stable, and that is worth saying.** On the
previous run of this same experiment it was `all_hard` that was significantly
worse (36.3%, p adj 0.001) and `half_hard` that tied; here they have swapped.
What reproduced across both runs is the only thing being claimed: hard
negatives never *helped*, slate resolution never improved, and retrieval stayed
5–9x more informative than the ranker on its own slate. Treat the ordering
among the mixes as noise and the null result as the finding.

**The hypothesis is refuted, and not narrowly.** Harder negatives moved
resolution the wrong way, monotonically, and `all_hard` is significantly worse
at ordering than the baseline it was meant to beat. Meanwhile the heads' 82%
and 73% on history positions barely moved — the model did not get worse, it
got worse *at the serving task specifically*.

The first row is the control, and adding it is what turned a confusing result
into a clear one. "The ranker explains 0.02% of the outcome variance" only
condemns the ranker if something else can do better on the same slate; if
nothing can, it condemns the task. So retrieval's own score is measured on the
slate retrieval itself produced, against exactly the same per-head labels —
`slate_head_labels` is shared between the two measurements so the comparison
cannot drift, which matters because a first version of this control labelled
"any target" against the ranker's "a target they purchased" and made the gap
look three times worse than it is.

With the control in place the three facts fit together:

- **The task is learnable.** Retrieval reaches 0.19% with a 4.9x lift from its
  bottom decile to its top one. Every number here is small because most of
  *which one of 300 items a customer picks next* is irreducible; what matters
  is the ratio.
- **The ranker has 4–6x less of it than the model it exists to improve on.**
  On the slate retrieval built, retrieval's own ordering is several times more
  informative than the ranker's.
- **Harder negatives make that worse, and the reason is consistent with the
  remaining explanation.** Retrieval's top 300 is a third of this 945-token
  catalogue, so it is thick with items the customer would have liked and never
  saw. Labelling them 0 adds noise. With only one true positive per user per
  epoch there is no extra positive signal to absorb that noise — you cannot
  fix a shortage of signal by making the negatives harder.

So what is left is the expensive explanation, which was always the more likely
one: retrieval trains against the full softmax at every position of every
sequence, and the ranker against a handful of candidates at one position per
user per epoch. Roughly two orders of magnitude less discriminative signal,
and no sampling trick substitutes for it.

One caveat this dataset cannot argue away: a uniform negative drawn from 945
tokens already has about a one-in-three chance of being something retrieval
would have shown, so hard negatives had far less room to help here than they
would on a catalogue of millions. A null result on this data is weaker
evidence against the hypothesis than a positive result would have been for it.
The experiment is a command, so it re-runs on a real catalogue unchanged.

### The number that was never about ranking

Everything above rests on the heads scoring 73–82% resolution on history
positions and ~0% on the serving slate — a gap this project described as a
collapse and spent two experiments trying to close. The diagnosis, when it
finally came, was that **one of those two numbers was never measuring
ranking.**

The control is embarrassingly cheap. Take how many times the same item has
just repeated, cap it at four, and look up the answer in a five-entry table
fitted on train:

| head | run-length table | + item identity | **the ranker** |
| --- | ---: | ---: | ---: |
| click | 0.9654 | 0.9848 | 0.9861 |
| cart | 0.9654 | 0.9848 | 0.9858 |
| purchase | 0.9853 | **0.9935** | 0.9909 |

AUC, on the same positions. For the purchase head **the lookup table wins.**
The table is `{run=0: 0%, run=1: 79%, run=2: 84%, run=3: 39%}` — the
generator's funnel, `view → add_to_cart → purchase` on consecutive events.
No user modelling, no attention, no embeddings. An earlier control using item
marginals scored AUC 0.445, i.e. nothing, which is why the artifact went
unnoticed: the obvious control was the wrong one.

So there was no collapse. There was one real number and one that measured the
data generator.

**On the slate there is real signal**, which rules out the comfortable
explanation that the task is impossible: retrieval's own ordering of its own
candidates reaches **AUC 0.714**, against the ranker's 0.53. And of the
~7000x "collapse", about 144x is the metric — resolution share is quadratic in
the deviation from the base rate, and the slate's base rate is 0.2% against
the teacher-forced set's 24%. The remaining ~50x was real.

### Two hypotheses, one right, and a third that failed instructively

**The one that looked strongest was wrong.** 52.5% of the ranker's training
positives are items already in the prefix, and split by action it is stark:

| last action | already in the prefix |
| --- | ---: |
| view | 34.4% |
| add_to_cart | **100%** |
| purchase | **100%** |
| return | **100%** |

A customer cannot buy something they never looked at, so a purchase positive
is a repeat by construction — and `exclude_seen` removes every already-seen
item *before* the ranker is consulted. Three of four heads were trained
entirely on a class of item that never reaches them, which is a clean
explanation for the purchase head ordering the slate **below chance**.
Restricting positives to unseen items (`candidate_positive: unseen`) did not
help. It was slightly worse. The likely reason is a tension worth keeping: an
unseen item is the right *class* but the weakest *label* — a first view says
far less about intent than a purchase.

**Dropping the teacher-forced loss tripled the ordering signal.** Five seeds:

| head | `control` | `no_history` | over chance | control scale | `no_history` scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| click | 0.5455 ±0.0134 | **0.6312** ±0.0110 | **2.88x** | 1.97x | 0.51x |
| cart | 0.5463 ±0.0151 | **0.6315** ±0.0109 | **2.84x** | 2.03x | 0.59x |
| purchase | 0.5196 ±0.0375 | 0.5786 ±0.0065 | 4.01x *(inside noise)* | 2.35x | 0.92x |

Calibration does not break, which was the obvious thing to check next: the
gate measures `max(bias, 1/bias)`, so it is two-sided, and `no_history`'s
worst distance is 1.96x against the control's 2.35x. It moves from
over-confident to under-confident and slightly closer to correct.

**And then the check that stopped it becoming the default.** The `return`
head has no slate-shaped label — a slate yields at most one matured purchase
per user — so it is invisible to every table above. Measured where it does
exist:

```
control     return: n=1602 base=0.1873 AUC=0.8729
no_history  return: n=1602 base=0.1873 AUC=0.5112     <- chance
```

Structurally inevitable in hindsight: in `_candidate_batch` the return head's
negatives are fully masked (something never bought cannot be returned) and
its positive carries a label only when matured, so the candidate block gives
it about one label per user per epoch. With the history loss off, that is all
it gets. And unlike the other three, the return head's teacher-forced task is
**not** a funnel artifact — "given a matured purchase, will it come back" is a
real, causal prediction with no leak.

So `history_loss_weight` became per-head, and a fourth arm was designed from
that measurement: off for click/cart/purchase, on for `return`. **It failed,
and not in the expected direction:**

| Arm | click | cart | purchase | return head |
| --- | ---: | ---: | ---: | ---: |
| `control` | 0.5437 | 0.5399 | 0.5025 | 0.8500 |
| `no_history` | **0.6367** | **0.6377** | **0.5774** | 0.5001 |
| `return_only` | 0.5130 | 0.5059 | 0.4920 | 0.8455 |
| *retrieval* | *0.7138* | *0.7138* | *0.6953* | – |

`return_only` keeps the return head and orders the slate **worse than the
control**, beyond noise. That looked non-monotonic, and calling it so was a
mistake worth recording: the two arms differ in the *magnitude* of the
teacher-forced loss **and** in which heads carry it, so nothing could be
attributed to either.

A scalar sweep separates them — all four heads scaled together, composition
held fixed:

| weight | ordering (over chance) | gain | return head | cost | share of available gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | +0.0437 | 1.00x | 0.8500 | – | 0% |
| 0.5 | +0.0690 | 1.58x | 0.8273 | 2.7% | 27% |
| 0.25 | +0.0775 | 1.77x | 0.8279 | 2.6% | 36% |
| **0.1** | **+0.1287** | **2.95x** | **0.7195** | 15.4% | **91%** |
| 0.0 | +0.1367 | 3.13x | 0.5001 | 41.2% | 100% |

**Monotone.** There is no interior optimum in magnitude, so the choice is a
point on a frontier rather than a maximum. And because it is monotone, the
`return_only` result is attributable after all: that arm's total weight is
2.0 and it scores +0.0130, while `scale_25` at total weight 1.25 scores
+0.0775. At comparable magnitude, keeping all four heads is far better than
keeping only `return` — so **the return head's own teacher-forced loss is
specifically harmful to slate ordering**, and the other three had been
partly counteracting it. Mechanically that is not strange: return propensity
is roughly anti-correlated with purchase propensity, since an item that comes
back is one the customer engaged with enough to buy.

The default is now `history_loss_weight: 0.1` — 91% of the ordering gain for
15% of the return head, and 1.0 is dominated. It changes nothing about
shipping: **no arm clears the gate.** The best reaches 0.6377 against
retrieval's 0.7138, which is 61% of retrieval's ordering signal, up from 21%.
Real progress, still not a second stage worth serving.
`retailgr slate-experiment` is the command.

### What happens when a dependency dies

The module docstring of `service.py` has always claimed:

> Every stage has a fallback… a failure in retrieval **or ranking** falls back
> to the precomputed list in the online store rather than erroring.

Read from the code it looked true — three fallbacks, one instrumented exit, an
AST test proving every `return` goes through it. Then each dependency was
taken down and the caller's actual response recorded:

| Failure injected | Before | After |
| --- | --- | --- |
| Redis: `user_tail` down | **500** | 200 · `store_error` |
| Redis: `item_states` down | **500** | 200 · `model` (fails open) |
| Redis: `fallback` down | **500** | 200 · `store_error` |
| Redis: entirely down | **500** | 200 · `store_error` |
| ANN index throws | 200 · `retrieval_error` | unchanged |
| Encoder / GPU throws | 200 · `retrieval_error` | unchanged |
| Ranker throws | **500** | 200 · `model`, `ranker_used=false` |
| Kafka (exposure log) down | 200, log lost | 200, log lost **and counted** |
| Iceberg / catalog down | serving unaffected | unchanged |

**Two thirds of the sentence was false, and the store one was circular.**
Three store reads in the request path were unprotected — and the fallback for
*everything else* was itself a store read, so the recovery route ran through
the component that had failed. The elaborate fallback machinery could not
have run in the one outage it most needed to.

The fix is the three guards plus a **last-resort list held in the process**: a
copy of the global cold-start list, read at startup and refreshed on every
successful store read. A pod that has been up serves a stale popular list; a
pod that *started* against a dead store has nothing cached and serves an empty
200 — which is honest, which `RetailGREmptyResponses` sees, and which is not a
500.

A broken ranker now degrades to **retrieval order**, not to the popular list.
A bundle with no ranker serves retrieval order and that is a supported
deployment, so a broken one should land in the same place: losing the second
stage is not losing the model. `ranker_used=false` goes out with it, so the
exposure log does not record a ranking that never happened.

The Kafka path was already correct — `_log_exposure` swallows everything, so a
dead broker costs the log and not the customer. But its comment claimed the
gap "shows up as a drop in log volume, which is alertable", and **nothing
counted the writes**, so it was not. `retailgr_exposure_log_total` by outcome
now exists, with an alert expressed as a *ratio* so it stays silent in the
default deployment where no logger is configured at all.

Iceberg is a batch dependency by construction: the serving pod reads a bundle
from disk. Checked two ways — an AST sweep over the request-path modules, and
a subprocess that imports `serving.service` and `serving.api` and asserts
`pyspark` never entered `sys.modules`, which catches the lazy import an AST
walk would miss. The real consequence of a dead catalog is a bundle that stops
being refreshed, and `RetailGRBundleStale` is what reports that.

`tests/test_failure_modes.py` pins all of it by **injecting failures**, not by
asserting handlers exist — fallback code only runs in the outage it was
written for, so reading it proves nothing. One of the tests is structural: a
store call added to `recommend()` without a guard fails the suite, because
that defect looks fine in every test, every staging environment, and in
production right up until the store blips.

### The policy layer

The model proposes; the policy layer disposes, deterministically and from
config. Filters are vetoes — out of stock at the customer's store, not
eligible, already bought — and adjustments are bounded, so relevance still
decides most of the order. `tests/test_serving.py` pins the behaviour that
matters: an item scoring nine times higher than another but out of stock must
not appear at all, and a diversity cap must never return a short list.

Every response carries a `PolicyTrace` of what was dropped and why, because
"why did it show me this" is a question merchandisers ask daily.

---

## Backends: nothing here is load-bearing infrastructure

| Layer | Default (no infrastructure) | Real | Verified here |
| --- | --- | --- | --- |
| Lakehouse | Parquet on local disk | Apache Iceberg, via Spark **or** pyiceberg | yes — pyiceberg engine, no JVM |
| Broker | in-process, partitioned and offset-tracked | Kafka (KRaft) + Schema Registry | partitioner only; no broker reachable |
| Online store | in-process, thread-safe | Redis | yes — real `redis-server` via redislite |
| Retrieval | exact BLAS matmul | FAISS | yes — both the flat and IVF paths |

```bash
make up              # MinIO :9000, Iceberg :8181, Kafka :9092, Redis :6379
make stack-iceberg   # the lakehouse on Iceberg
make stack-kafka     # the streaming path on real Kafka and Redis
```

The defaults are test doubles, not toys: the in-process broker partitions by
key hash, preserves order within a partition and only advances offsets on
commit, which are the three guarantees the pipeline actually depends on. A
test that passes against it is testing something real.

### Swappable, checked rather than claimed

Four components are advertised as replaceable, each behind a `Protocol` and a
factory — and Python enforces neither at runtime. A backend can be missing a
method, or have one with an incompatible signature, and nothing notices until
someone selects it in production. An integration test cannot catch that
without the infrastructure it is testing, which is exactly the infrastructure
that is absent.

So conformance is checked **statically**, by comparing every registered
backend's methods and signatures against the protocol it claims to satisfy.
`retailgr audit` runs it, and it exits non-zero:

| Check | What it would catch |
| --- | --- |
| No cloud vendor SDK under `src/retailgr` | one `import boto3` in the request path, and "any cloud" is false |
| No endpoint configuration cannot reach | an address that is written down is fine; one nobody can change is not |
| Optional clients imported lazily | a top-level `import redis` turns an optional backend into a required dependency of the whole package |
| Every backend conforms to its protocol | the method that is missing until the day you switch |

The endpoint rule is the one worth explaining, because the naive version of
it is useless. `cfg.get("streaming.bootstrap_servers", "localhost:9092")` is
*allowed* — so is a parameter default — because the property being checked is
overridability, not absence. The same string written anywhere a caller cannot
replace it fails the build. `tests/test_platform.py` exercises every check
twice: once on a violation it must catch, once on the legitimate form it must
let through.

Behaviour is checked separately, by **one contract per swap point run against
every implementation of it** — not one suite per backend, which would make
"swappable" mean "both exist" rather than "either will do". Backends whose
infrastructure is absent skip out of the *same* test, so `make up` covers them
with identical assertions and no new code. The suite prints what it could not
verify, because a contract that quietly skips half the backends and shows a
row of green dots is the platform equivalent of reporting AUC on the easy
distribution:

```
broker:          verified ['memory']
    kafka        NOT verified — `confluent_kafka` is not installed
online_store:    verified ['memory']
    redis        PARTLY verified — ran against fakeredis in-process; protocol and
                 key layout exercised, a real server's connection behaviour was not
retrieval_index: verified ['exact', 'faiss_flat', 'faiss_ivf']
```

**FAISS needed no infrastructure, so it was verified — and it was broken.**

The index never returned the items it was supposed to. `nprobe` was set to a
tenth of the cells, so the search visited a tenth of the catalogue; on 50,000
items it returned **41% of the true top-20**. Nothing had ever caught it
because nothing had ever run it: the obvious fixture is a few dozen vectors,
and below `4 * nlist` the class falls back to a flat index — an exact search
wearing FAISS's name. Recall lost at retrieval is unrecoverable, since the
ranker reorders what it is handed and cannot ask for a candidate that was
never returned.

| nprobe | 10 _(was the default)_ | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- | --- |
| recall@20 | **41%** | 68% | 89% | 98% | 100% |
| p50 vs exact | 0.17x | 0.37x | 0.68x | 1.00x | 1.36x |

Reading the second row was the more uncomfortable part. At the point where
FAISS finally keeps 98% of the results, it is **exactly as fast as the exact
BLAS matmul** — 0.367 ms against 0.366 ms. So the measurement was extended to
find where it starts paying at all, recall held at or above 90%:

| Catalogue | 50,000 | 200,000 | 500,000 | 1,000,000 |
| --- | --- | --- | --- | --- |
| speedup vs exact | **0.99x** | 1.15x | 1.35x | 1.60x |

Below roughly 100,000 items FAISS costs recall for nothing. This project's
vocabulary is **945 tokens**. The fix was to make `nprobe` a parameter with a
recall-preserving default, add `measure_recall()` so the trade is tuned on
your embeddings rather than inherited from mine, and put a recall floor in the
contract so it cannot come back — but the honest conclusion is that `exact`
stays the default and FAISS is for a catalogue this repository does not have.

### Executing the three that had never run

The list above used to end with Iceberg, Kafka and Redis marked *written,
reviewed, never executed* — no Docker daemon, Maven Central answering 403,
the Apache and GitHub release hosts blocked. That paragraph was honest and it
was also the most interesting thing left in the repository, because the last
time an unexecuted path here was finally run — FAISS — it turned out to be
returning 41% of the results it should have.

Two of the three can be executed after all, by routes that were there all
along:

| Backend | Route | Status |
| --- | --- | --- |
| **Redis** | `redislite` ships a compiled `redis-server` in a PyPI wheel | **verified** — real server, real TCP, the project's own client |
| **Iceberg** | `pyiceberg` is the format in a wheel, no JVM | **verified** — and now a supported engine, not a workaround |
| **Kafka** | needs the broker distribution; Maven, Apache and GitHub releases all blocked | still unverified, for a reason that is now stated precisely |

Running them found three defects. None of them were visible to any test that
existed, because all three live where data actually moves.

**1. The time split was not deterministic.** This is the serious one. The
train/val cutoffs came from `approxQuantile(..., relativeError=0.001)`, which
summarises per partition and merges — so its answer depends on how the data
is *laid out*, not only on what the data is. Parquet partitions by
`event_date`; the Iceberg tables do not. Same events, same pipeline:

```
parquet   cutoff_train_end_unix = 1771376010   val_users = 1127
iceberg   cutoff_train_end_unix = 1771374424   val_users = 1131
```

1,586 seconds apart, four users moved between splits. Every metric downstream
shifts a little and nothing anywhere says why — two runs on identical data
had quietly stopped being comparable because the storage changed. The fix is
an exact percentile, which is a function of the values alone; `approxQuantile`
is still available behind `sequences.quantile_error` for datasets too large
to sort, as a decision rather than a default. Afterwards both backends choose
the same cutoffs to the second and produce **byte-identical gold tables** —
which `tests/test_warehouse_equivalence.py` now asserts by hashing them.

**2. A `MapType` degraded in transit.** `item_hierarchy.attributes` is a map,
and the granularity resolver calls `element_at(attributes, 'capacity')` on it.
Routed through pandas it arrived as `list<struct<_1,_2>>` — still readable,
still writable, no longer a map. It failed two stages later with a SQL error
naming neither pandas nor Iceberg. On a simpler frame it does something worse
than fail: a `struct` silently becomes a `map`.

**3. The Kafka backend was being skipped for a reason that did not exist.**
The contract suite and the platform audit both recorded its dependency as
`confluent_kafka`. The code imports `kafka-python`. So on a machine with the
right client installed *and a broker running*, the Kafka contract would still
have skipped — the verification would have reported green and never run. A
wrong skip reason is worse than a missing test, because it looks like
coverage.

**And the in-process broker hashed differently from Kafka.** Not a defect in
the sense of breaking anything: `partition_for` used `zlib.crc32`, which
satisfies the guarantee the pipeline depends on — one key always lands in one
partition, so a customer's events never overtake each other. It fails a second
thing the double is used for. "Will my keys hot-spot across twelve
partitions?" is an operational question people ask of the local run and then
act on, and a hash Kafka does not use makes that answer unrelated to the
cluster's. It now uses murmur2, the same hash, checked byte-for-byte against
`kafka-python`'s implementation — which needs no broker, so it is verified
here.

> **What is still not executed.** Real Kafka. The broker is a Java
> distribution and every route to it from this environment is closed: Maven
> Central 403, `downloads.apache.org` unreachable, GitHub *releases* blocked
> (raw files are not), `proxy.golang.org` refused — which also rules out
> building a Kafka-protocol server in Go. `KafkaBroker`'s admin client,
> producer configuration and consumer-group handling have therefore never
> run. What *has* been verified without a broker is the partitioner, against
> the real client's hash. `make up` on a machine with a daemon runs the
> identical contract against the real thing, and the suite prints which
> backends it could not reach rather than implying it reached them.

---

## Deploying it

`Dockerfile` and `deploy/k8s/` — a Deployment, Service, HPA, PodDisruptionBudget
and two CronJobs for the batch side. Nothing in them names a cloud: every
external address lives in one ConfigMap, which is the portability claim
expressed as a file you would edit when moving.

Every manifest is validated against the **real Kubernetes API schemas** in
`tests/test_deploy.py`, in strict mode, against 1.29 rather than the newest
version — so a field an older cluster rejects fails here. That needs no
cluster, no kubectl and no network. Schema validity is the floor; the tests
above it assert the things a schema is happy to let you omit, and two of them
check across artefacts:

- **Every command a manifest runs must be a real CLI subcommand.** An earlier
  draft had an init container running `retailgr fetch-bundle`. It validated
  cleanly, read plausibly, and referred to a command that has never existed —
  it would have failed on first apply, in a cluster, at the worst moment.
- **Every probe path must be a route the API serves.** The same draft pointed
  at `/readyz`, which did not exist either. Writing it turned out to be worth
  doing on its own: liveness and readiness answer different questions, and a
  liveness probe that fails when Redis blips restarts every replica at once,
  turning a dependency wobble into an outage.

### The numbers in those manifests are measured

`retailgr sizing` runs the real service in fresh interpreters and reports what
a pod actually needs:

| | |
| --- | --- |
| Bundle on disk | 1.3 MiB |
| Bare interpreter | 11 MiB |
| After importing the serving stack | 29 MiB |
| **After loading the bundle** | **510 MiB** — PyTorch initialises here |
| Warm, after 25 requests | 523 MiB |
| Cold start | 1.03–1.26 s (most of it loading) |
| Throughput | 196–233 req/s per core |

**97% of the pod is framework.** The bundle is 1.3 MiB and loading it costs
481 MiB, so shrinking the model to fit a pod would be optimising the 0.25%.
That is the kind of thing worth knowing before a sprint gets spent on it.

Re-measuring these gave 523 MiB again to the tenth of a megabyte, and 196–233
req/s per core across runs. So the memory figure is a property of the system
and the throughput figure is partly a property of the machine that ran it —
which is why the manifests are sized from the first and an autoscaler is told
to watch CPU rather than trust the second.

Those figures set `memory: 654Mi` (warm + 25%) and `limits: 1047Mi` (2x warm —
a memory limit is a kill switch, not a target), and a test fails if the
manifest and the measurement drift more than 15% apart. There is deliberately
**no CPU limit**: a CPU limit throttles rather than kills, and throttling a
latency-budgeted request path produces p99 spikes that get misread as a model
problem. The HPA scales on CPU for the same reason the memory table gives —
memory here barely moves with load, so a memory target would never fire.

### Observability: the numbers the request path already had

Every response has always carried per-stage timings and a policy trace
explaining why the list looks the way it does. All of it was computed,
returned in the body, and thrown away — so the 100 ms budget was verified
*offline* by `retailgr bench` and completely unobservable *online*. Knowing
p99 on the machine that ran the benchmark is not knowing it in a pod.

`/metrics` now exposes it in Prometheus text format, alongside `/healthz` and
`/readyz`. Three decisions, each the opposite of the obvious one:

- **No dependency.** `serving/metrics.py` renders the format itself, for the
  same reason the broker and the online store have in-process
  implementations: the default has to work with nothing installed, and a
  metric that only exists when an optional package is present is a metric
  nobody can rely on. `prometheus-client` is a *dev* dependency — the tests
  parse this module's output with the reference parser, so the format is
  checked against the real implementation without being tied to it.
- **Buckets come from the budget.** `DURATION_BUCKETS` is built around
  `bench.CLAIMED_BUDGET_MS`, so every per-stage budget and the 100 ms total
  land *on* a boundary rather than inside one. The SLO is then readable
  straight off the metric: `..._bucket{stage="total",le="0.1"}` **is** the
  count of requests inside budget, with nothing to interpolate.
- **No user labels, ever.** A `user_id` label is one time series per
  customer — it kills the Prometheus server and puts identifiers in a system
  with no retention policy and a far wider audience than the lakehouse.
  Cardinality is asserted to be bounded by construction.

The metric that matters most is the dullest: `retailgr_requests_total` split
by `served_from`. Latency tells you the system is slow. **The fallback rate
tells you it has stopped answering with the model and started answering with
a list of popular items** — which looks completely healthy from outside, and
is the characteristic failure of this design.

Every one of the four exits from `recommend()` goes through a single
`_finish()`, and a test parses the AST to prove it: an uncounted fallback is
invisible in exactly the situation the fallback rate exists to reveal.

**Tracing** was almost finished and stopped one step short. `recommend` took
a `request_id`, the response carried it, the exposure topic had a field for
it — and the HTTP handler called `recommend()` with no id at all, so every
request minted a fresh uuid4 and whatever id the caller arrived with went in
the bin. A trace crossing three services stopped here, and "why did this
customer get this list at 14:32" was answerable only by timestamp and luck.

`serving/tracing.py` reads W3C `traceparent` (falling back to
`X-Request-Id`, then to a new id) and echoes both headers back. It is thirty
lines of parsing rather than OpenTelemetry on purpose: the SDK, exporter and
configuration surface are a lot of serving image to extract a substring, and
the ids produced are the standard's, so a collector downstream joins on them
without knowing this module exists. What it deliberately does **not** do is
emit spans — propagating an id is what makes the logs and the exposure topic
join up; span export is a collector, a protocol and an operational
commitment, and claiming half of it under the name "tracing" is worse than
having none.

### Alerts, and the one that was silently inverted

`deploy/k8s/50-alerts.yaml` is a `PrometheusRule`: three recording rules and
ten alerts, split into latency SLO (multi-window multi-burn-rate), the
correctness set this project earned the hard way, and deployment freshness.

An alert rule is the only artefact in a system that is exercised solely when
everything else has already failed. It is never run in development, produces
no output when correct, and produces no output when broken either. So
`tests/test_alerts.py` closes three loops:

1. **Every metric name in a PromQL expression is one `metrics.py` emits** —
   name, `_total`/`_bucket`/`_count` suffix and label names, read off the
   registry rather than listed by hand.
2. **Every label *value* in a selector is one the code can produce.**
   `served_from="retrieval_error"` matches nothing if the service spells it
   `retrieval_failed`, and matching nothing is silence, not an error.
3. **Every `le` is a real boundary, spelled the way the exporter spells it.**
   `le` is a string label compared as a string: `le="0.10"` against a bucket
   exposed as `le="0.1"` selects an empty vector for ever.

The expressions are parsed with `promql-parser`, a binding over Prometheus's
own grammar — "it parses" means it parses, not that it looked plausible to a
regex. A final test drives the real exporter and checks every selector
against the *rendered* scrape, so the render path is covered too and not just
the declarations.

What a parser cannot tell you is whether an expression asks a sensible
question, and writing these tests found one that did not. Bundle staleness
was:

```promql
changes(sum by (model_version) (retailgr_build_info)[7d:1h]) == 0
```

It parses. It reads convincingly. It is nonsense: `build_info` is pinned at
the constant `1`, so `changes()` over it measures **replica scaling**, not
deployments — and a version deployed an hour ago has zero changes, so the
alert fired an hour after every *successful* deploy and stayed silent when
the export pipeline died. Worse than missing, because it looked like
coverage.

The fix was to export the number the question is actually about:
`retailgr_build_timestamp_seconds`, taken from the bundle's `created_at` and
not from process start — using the process would reset freshness on every
restart, which is the event most likely to happen while the pipeline is
broken. `time() - max(...) > 7d` then means what it says, and a structural
test now rejects `rate`/`changes`/`increase` applied to any registered gauge,
so the class of mistake cannot come back in a different rule.

Ten deliberate mutations of the alerts file — a renamed metric, a respelled
`le`, a capitalised boolean label, a removed `for:`, the inverted staleness
rule put back — are **all ten caught**.

`kubernetes-validate` has no schema for `PrometheusRule`: it is a custom
resource whose schema ships with the operator, not with Kubernetes. So
`test_deploy.py` exempts that kind from its schema sweep — and asserts the
exempted kind is validated by `test_alerts.py`, because otherwise adding a
kind to the exemption list would be a one-line change that turns a failing
test green, reads like housekeeping, and leaves a manifest checked by
nothing.

---

## Privacy, starting with the endpoint that was leaking

The first thing this pass found was live. `GET /v1/model` returned the bundle
manifest verbatim; the manifest carried `ranker_metrics`; the discrimination
diagnostic had kept the `user_ids` it needs to pair its samples. **200 real
customer ids, 600 entries, 65 KB, on an unauthenticated route** — since the
ranker shipped.

The cause is more useful than the bug. `experiment.py` already strips
`per_user` and `user_ids` before writing a run record, in two places, by
hand. The rule was known and applied in two of the three places that needed
it, because *"remember to strip identifiers at each call site"* is not a rule
a codebase can keep. So the scrub now happens at the **serialisation
boundary** — where a structure stops being an in-process object and becomes a
file or a response body — and `tests/test_privacy.py` walks the AST of every
module to assert that no payload reaches `write_text` without crossing it.
There is deliberately no allowlist of files judged harmless: two of the
writes it catches are item data the scrub cannot change, and they go through
it anyway, because an allowlist is the structure that rots.

Scrubbing replaces rather than deletes: `user_ids: [200 ids]` becomes
`n_user_ids: 200`. The diagnostic ran over 200 users and that belongs in the
record; which 200 does not. And the inverse assertion gets its own tests —
the online store and the event stream **must** keep `user_id`, because a
cache of user tails with the user scrubbed out is not a safer cache, it is a
broken one.

### What consent costs, measured

Every write-up of consent in a recommender stops at "we honour the flag".
The second half — *how much worse do the recommendations get* — is the half a
product owner asks about, and without a number the conversation is a
negotiation between someone who says privacy is free and someone who says it
is ruinous.

Consent removes whole users, not events: one decision covers an account, so
the cost of an opt-in rate is the cost of training on that share of users.
`retailgr consent-cost` sweeps it, three seeds per point:

| Opt-in rate | Train users | Recall@10 uniform | vs 100% | most-active-out | vs 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 2501 | 0.0955 ±0.0013 | 1.000x | 0.0955 | 1.000x |
| 90% | 2251 | 0.0952 ±0.0005 | 0.997x | 0.0957 | 1.002x |
| 75% | 1876 | 0.0945 ±0.0013 | 0.989x | 0.0954 | 0.999x |
| 50% | 1250 | 0.0927 ±0.0012 | **0.971x** | 0.0905 | **0.947x** |
| 25% | 625 | 0.0810 ±0.0060 | **0.849x** | 0.0758 | **0.794x** |

**At realistic opt-in rates consent is free.** Down to 75% the drop is inside
this model's seed-to-seed spread, which means it is indistinguishable from
re-running the same configuration with a different random seed. Consent is
not the thing to argue about at 85%.

**Who opts out only starts to matter once half of them do.** The right-hand
columns drop the *most active* users instead of a random sample — a
deliberately pessimistic stand-in for the fact that consent is not
independent of behaviour. At 90% and 75% it makes no difference at all, and
at 25% it turns a 15.1% loss into a 20.6% loss. So the uniform estimate is
usable for planning in the range anyone actually operates in, and the
correction is worth modelling only if opt-in collapses.

That second run exists because the uniform number is a lower bound and
saying so in a footnote is cheap. Measuring how wrong the assumption can be
is the part that makes the first number trustworthy.

### Consent as a field, in two implementations that are checked against each other

The Avro schema carries a `consent` string; `privacy.consented` **fails
closed**, so an absent or unparseable value denies everything except
`service` — a producer that has not been updated yet must not become a
silent opt-in. `service` is always granted, because an event that arrived
cannot be un-processed for the request it arrived in, and recording that
honestly beats a flag that pretends otherwise.

The rule is written in Python and again as a Spark expression, for the same
reason `granularity.py` is: a UDF would serialise every row of the largest
table in the warehouse through the interpreter. Writing it twice is the
hazard, so `tests/test_privacy_spark.py` runs both over a table of awkward
values and fails if they ever disagree — and **it caught one immediately**:

```python
F.transform(F.split(consent, ","), F.trim)   # silently does nothing
```

`F.trim` takes an optional second argument, so `transform` reads its arity as
the `(element, index)` form and hands it the index. No error, untrimmed
array. The value it broke on was `"service, analytics"` — what a producer
that joins with `", "` sends, which is most of them. The symptom would have
been every such customer silently denied analytics, for ever, with a green
test suite.

Silver **refuses to run** when enforcement is on and every consent value is
null. Both silent alternatives are wrong in a way that is hard to notice
later: keeping everything is a blanket opt-in, dropping everything is an
empty warehouse that reads like a broken join. A public research dataset
genuinely has no consent signal, so that is configuration
(`privacy.consent.enforce: false`) rather than an exception to swallow.

Pseudonymisation is real HMAC-SHA256 with the key blocks precomputed so
Spark evaluates it natively, not `sha2(key || value)` — the prefix
construction is what people reach for when the engine has no HMAC builtin,
and it is length-extendable. There is **no default key**, because a default
committed here would make every pseudonym in every deployment reversible by
anyone who can read this repository while looking exactly as safe as a real
one. And the module says plainly what it does not buy: a pseudonym attached
to a fifty-event purchase history is re-identifiable by anyone holding a
second copy of those purchases, which for a retailer is every payment
processor they use.

### Erasure, and the two ways it silently does not happen

`retailgr forget --user-id U000048` clears the raw file, bronze, silver, all
four gold sequence tables and the online store, in that order — forced,
because the pipeline re-derives everything from `data/raw/` and the cache is
rebuilt from silver, so the wrong order is undone by the next scheduled job.

The deliverable is not `forget`, it is `verify`. Both defects below produce
the same symptom — a green result for a deletion that did not occur — and
that is worse than a red one, because a confident record of an erasure is
exactly what an audit relies on.

**The verifier was blind.** It searched each file's bytes for the identifier.
Spark writes Parquet with Snappy, so the string `U000068` appears nowhere in
the bytes of a file holding nineteen of that customer's rows: **130 files
searched, zero hits, customer present.** The first real erasure run reported
"no occurrences" and was only correct because the raw CSV is plain text and
caught it. Fixed by decoding Parquet rather than grepping it — every string
column, not just `user_id`, because two adapters build `session_id` by
concatenating the user id into it.

The test that would have caught it is the one that gives every other result
meaning: **assert the verifier finds a user who was not erased.** Measured
both ways on the real warehouse:

```
U000048 (erased)     → survived: []                    clean: true   exit 0
U000068 (not erased) → survived: raw CSV (19), bronze  clean: false  exit 1
                                 parquet ×N
```

**Iceberg keeps the deleted rows readable.** `Table.delete()` writes a new
snapshot without them; every earlier snapshot still resolves by id, and this
repository's own `read_arrow(..., snapshot_id=...)` hands them back. A delete
without snapshot expiry removes the customer from the current view of the
table and from nowhere else. `tests/test_erasure.py` demonstrates exactly
that against the real format, then asserts the erasure expires the snapshot —
so "expiry is required" is a shown fact, not a line in a docstring.

The receipt gets the same treatment, and a test found that too: the first
version wrote `user_id` into a permanent artifact, which is an erasure that
ends by creating a new record of the person who asked to be forgotten. It now
carries a digest, and says whether that digest is keyed — unkeyed it is a
correlation handle over a million-candidate space and not a protection, and
which of the two you have is the difference between a safeguard and the
appearance of one.

### What cannot be erased, named every time

A report listing only what it did invites the reader to assume the rest was
nothing, so all three appear in every receipt with a bound attached:

- **Kafka.** A log is append-only. `interactions.v1` is keyed by `user_id`
  but is `cleanup_policy: delete`, so a tombstone removes nothing, and
  `recs.served.v1` carries `user_id` in the value while being keyed by
  `request_id`, so no key-targeted delete can reach it at all. *Bound: both
  topics have 7-day retention.*
- **The trained model.** Erasure removes training data, not the parameters
  fitted to it. The encoder has item embeddings and no per-user vectors, so
  nothing in the weights is *about* one customer — but their behaviour shaped
  them, and removing that contribution is machine unlearning, which is a
  research area and not a function call. *Bound: the next export trains on
  data they are no longer in; the currently-serving bundle still carries it.*
- **Backups and object-store versioning.** Out of scope for this code, in
  scope for whoever operates it. *Bound: unknown to this code — named so it
  is not assumed to be nothing.*

### Retention, and four settings that enforced nothing

`privacy.retention` set a window per layer with a paragraph explaining why
bronze is shortest — it is the only layer holding raw, unfiltered,
un-pseudonymised events. **Nothing read any of it.** `bronze_days`,
`silver_days`, `gold_days` and `snapshot_expiry_days` were read by no code at
all. The Redis TTL beside them *was* wired, which is exactly what made the
block look finished.

That is worse than the settings being absent: someone reading
`configs/pipeline.yaml` would reasonably conclude bronze is pruned at thirty
days, and the only way to find out otherwise is to go looking for the job.

`retailgr retention` is now that job, and two of its decisions are the
non-obvious ones:

- **The clock is the wall clock.** Retention means "older than N days from
  now". Measuring from the newest event in the table instead is the reading
  that makes a fixed sample dataset work and would silently retain everything
  for ever in production, because the newest event is always today. `clock:
  data` exists for the sample case and has to be asked for.
- **A pass that would remove nearly everything refuses.** Running the shipped
  config against the sample data removes 100% of bronze — its events are
  months old — and the job exits non-zero with a message naming the clock as
  the likely cause. Removing a whole table is far more likely to be a wrong
  clock, days confused with hours, or a backfill of historical data than a
  policy, and an empty warehouse is an expensive way to find that out. With
  `clock: data` the same pass prunes 46.3% of bronze and proceeds, so the job
  is not merely a refusal machine.

Gold is pruned per user, not per row: `gold.sequences_*` has no scalar date,
only an `input_ts` array per customer, and half a sequence is not a smaller
training example but a corrupted one.

The general fix is `tests/test_config_is_wired.py`: **every key in every
config file must be read by some code in `src/`.** It scans the 91 leaves,
allows for blocks consumed wholesale (`PolicyConfig.from_dict`,
`SyntheticConfig(**params)`) and for blocks whose keys are data rather than
names (surfaces, product categories), and carries one documented exemption —
`streaming.schema_registry_url`, which points at a service nothing contacts
because the wire format is JSON. An exemption needs a reason about the key;
the test rejects one containing "TODO" or "for now". Writing it also caught a
self-referential bug in its own matcher: scanning `tests/` too meant a key
named only in a test counted as read, and the test naming it was that one.

Retention elsewhere: a TTL on the serving tails. Those tails were
bounded by **count** and not by time, so a customer who stopped shopping kept
their last fifty events indefinitely. The shared cold-start list moved out of
the per-user keyspace at the same time: it used to live at
`{ns}:fallback:__global__`, one wildcard away from being deleted by any
cleanup sweep, and the symptom would have been empty recommendations for
every new customer while every per-user path kept working.

---

## Real datasets

| Dataset | How to get it | What it gives you |
| --- | --- | --- |
| `ml1m` | `make movielens` | 1,000,209 ratings, 6,040 users, **real timestamps over ~3 years**. No item variants, so the granularity question does not apply. Ratings become actions (≥4 purchase, 3 view, ≤2 rejection), which is what exercises HSTU's action modality. |
| `ml100k` | `make movielens` | 100,000 ratings **with genres**, so the per-category breakdown is real. |
| `rees46` | Kaggle, into `data/raw/rees46/` | views, cart, purchases, price, brand — closest to a real retail event stream. No variants: SKU = product. |
| `hm` | Kaggle, into `data/raw/hm/` | purchases only, but a genuine hierarchy: `article_id` is product+colour, `product_code` is the product. The cleanest public test of style-colour vs product. |

```bash
make movielens                                   # fetch and run
retailgr ablate --dataset ml1m --model-config hstu_ml1m.yaml
retailgr experiment --dataset hm --model-configs sasrec_base.yaml hstu_base.yaml
```

MovieLens is fetched from public GitHub mirrors rather than
`files.grouplens.org`, so it works from networks that only allow GitHub; the
canonical source and its terms of use are at
[grouplens.org](https://grouplens.org/datasets/movielens/). REES46 and H&M need
a Kaggle account, so their adapters ship untested against real files — the
schemas are from the datasets' published column lists.

Only MovieLens carries real timestamps among the datasets that can be fetched
automatically, and none of the public ones carry returns. Return events are in
the canonical schema and the synthetic generator produces them with reasons, so
the return-risk head has something to train on before real data arrives.

Adding a retailer's own export means writing one adapter in
`src/retailgr/datasets/adapters.py`; nothing downstream changes.

### What MovieLens said

Running the same harness on real timestamps (`make movielens`, 1,850 s):

| Model | Recall@10 | NDCG@10 | Recall@200 | NDCG@200 |
| --- | --- | --- | --- | --- |
| popularity | **0.1793** | **0.1851** | **0.3273** | **0.2310** |
| sasrec_ml1m | 0.1594 | 0.1621 | 0.3062 | 0.2063 |
| hstu_ml1m | 0.1603 | 0.1617 | 0.2991 | 0.2038 |

**Popularity wins, and both sequence models lose to it.** Three reasons, and
none of them is that sequence models do not work:

1. **The split is deliberately hard.** A global time cutoff means the test
   window is the last slice of *calendar* time, and MovieLens users are
   concentrated in time: someone joins, rates a few hundred films in a week,
   and leaves. So the 1,175 test users are disproportionately users with thin
   training history — and for them, popular films genuinely are the best
   guess. A per-user leave-one-out split, which is what the HSTU paper's
   public numbers use, is a much easier and less production-like protocol.
   **These numbers are therefore not comparable with the paper's.**
2. **Eight epochs is not convergence.** Training loss was still falling when
   it stopped (4.34 → 2.85 for SASRec, 4.28 → 2.70 for HSTU). The paper's
   reproduction configs train far longer.
3. **HSTU reached a lower training loss than SASRec and a slightly worse test
   score** — the same mild overfitting the synthetic ablation showed.

The useful conclusion is not "HSTU is bad", it is: **on this protocol and this
data the sequence models are not yet worth their cost, and the harness says so
before anything reaches production.** That is the harness working.

### The temporal bias fails on real timestamps too

The open question after the synthetic run was whether HSTU's temporal bias only
looked bad because the synthetic generator scatters events at random. MovieLens
answers it: the timestamps are real and span three years, and the direction
replicates.

| Model | Recall@200 | NDCG@200 | Final loss |
| --- | --- | --- | --- |
| `sasrec` | 0.3090 | 0.2087 | 2.850 |
| `hstu_full` | 0.2990 | 0.2049 | 2.688 |
| **`hstu_no_temporal_bias`** | **0.3106** | **0.2143** | 2.837 |

Note the loss column: `hstu_full` fits the training data *better* (2.688 vs
2.837) and generalises *worse*. Those 129 learned time-bucket parameters are
spending capacity on noise. Two things plausibly explain why real timestamps
still do not help here:

- **MovieLens timestamps are rating times, not consumption times.** A user
  who rates two hundred films in one sitting produces near-zero gaps that say
  nothing about the order they watched them in. "Real" is not the same as
  "informative".
- **Both models are undertrained**, so any component that adds parameters is
  penalised twice.

> **These MovieLens numbers are point estimates from a single seed**, run
> before the interval machinery existed. The direction agrees with the
> synthetic result, which *is* interval-tested, but the magnitudes here have
> not been through a paired test or a multi-seed run. Re-run with
> `retailgr ablate --dataset ml1m --model-config hstu_ml1m.yaml --seeds 13 17 23`
> before quoting them. That costs about an hour on a laptop, which is why it
> has not been done here rather than because it does not matter.

What this does *not* license is deleting the temporal bias from the
architecture. It licenses one sentence, and the cutoff is part of it: **on the synthetic
data, turning the temporal bias off improves HSTU's NDCG@200 by 3.7–4.2%
across two datasets and two training schedules, and MovieLens points the same
way — while at NDCG@10 it changes nothing.** A retrieval model that feeds a
ranker its top 300 does care about depth; a customer looking at ten items
does not see the difference. The
defaults stay faithful to the paper so that `make ablate` keeps measuring this
rather than assuming it, and a retail event stream — where a gap of ten
seconds and a gap of ten days really are different intents — is where the
paper's version should get its fair test.

---

## Layout

```
configs/            pipeline.yaml, granularity.yaml, model presets
docker/             MinIO, Iceberg catalog, Kafka, Schema Registry, Redis
scripts/            fetch_movielens.sh
src/retailgr/
  config.py         YAML loading, path resolution, --set overrides
  compute.py        thread budget for torch and BLAS
  granularity.py    SKU -> model token, in Python and in Spark
  privacy.py        consent, pseudonymisation, the identifier scrub
  erasure.py        retailgr forget: delete a customer, then prove it
  actions.py        the action vocabulary (HSTU's second modality)
  online_store.py   user tails, item state, fallbacks (memory | Redis)
  spark_session.py  one builder for both lakehouse backends
  io/               table read/write, in-memory loaders
  datasets/         synthetic generator + dataset adapters
  jobs/             ingest, silver, sequences, export, bootstrap, retention
  models/           popularity, SASRec, HSTU, HSTU ranker, early stopping
  evaluation/       metrics, bootstrap + paired tests + Holm, calibration
                    (Brier/ECE/Murphy, Platt + isotonic), ranker AUC + gate
  streaming/        schemas, broker (memory | Kafka), producer, consumers, replay
  serving/          bundle, retrieval, policy, service, api, bench,
                    metrics (Prometheus, no dependency), tracing (W3C)
  experiment.py     the experiments, the ablation, the reports
  ranker_experiment.py  negative-sampling comparison, with its retrieval control
  consent_experiment.py what honouring consent costs the model, swept
  convergence.py    fixed epochs against early stopping, scored on test
  slate_experiment.py   can the ranker order retrieval's own candidates?
  platform.py       the portability audit: vendor SDKs, endpoints, conformance
  serving/sizing.py measured pod footprint: memory, cold start, throughput
deploy/k8s/         manifests, schema-validated in tests/test_deploy.py
                    50-alerts.yaml: PrometheusRule, checked in test_alerts.py
Dockerfile          the serving image
  cli.py
tests/              685 tests: granularity, metrics, statistics, calibration,
                    HSTU architecture, ranker + M-FALCON equivalence,
                    streaming, serving, policy, HTTP, end-to-end pipeline,
                    platform audit, backend contracts, deployment manifests,
                    Iceberg format, cross-backend equivalence, doc drift,
                    observability + tracing, alert rules vs. real metrics,
                    consent (Python vs Spark), erasure + its verifier,
                    injected dependency failures, slate ordering,
                    retention + its guard, every config key is wired,
                    early stopping and the ablation's cutoffs
scripts/
  mutation_check.py breaks each load-bearing claim; checks a test notices
```

---

## Verifying the whole thing

Everything above was measured, but not all at the same time and not all under
the same code. That is its own failure mode: a repository full of real numbers
that quietly stopped describing the thing it ships. So every artifact here was
re-derived from scratch — `generate-data`, `experiment`, `ablate --seeds 13 17
23`, `export-model`, `bench`, `sizing`, `audit`, `rank-experiment` — and the
old numbers compared against the new ones.

**The re-derivation was necessary, not ceremonial.** Making the time split
deterministic changed the split, so every model metric in this README had been
computed on a partition of the data the code no longer produces.

| Claim | After re-deriving |
| --- | --- |
| The temporal bias costs ~4% | **holds at NDCG@200 only** — 3.7% converged on current data; at NDCG@10 a tie in every run |
| The relative bias costs ~3.5% | **holds at NDCG@200 only** — 3.7% converged; tie at NDCG@10 |
| *The action modality is not demonstrably helping* | **reversed at @200** — 6.2% worse without it converged; still a tie at NDCG@10 |
| HSTU vs SASRec is a tie | **holds at NDCG@10** under every schedule; at @200 it depends on the stopping rule |
| The models are undertrained at 8 epochs | **refuted** — validation peaks at epoch 3–8; converged vs fixed is p=0.401 (SASRec), p=0.845 (HSTU) |
| Heads: 73–82% resolution on history, ~0% on the slate | **reframed** — the history number was the generator's funnel; a five-entry lookup table scores AUC 0.965 on it |
| Hard negatives do not help | **holds**, but *which* mix is worst changed between runs |
| The ranker is blocked by its gate | **holds** — now 0.31x retrieval's MRR, up from 0.21x |
| Serving p99 inside its budget | **holds** |
| Warm memory 523 MiB | **holds** — 523 MiB again, to the tenth |
| Throughput 233 req/s per core | **softened** — 196–233 across runs; it is partly the machine |

One reversal, one softened number, one instability found, everything else
reproduced. The reversal is the interesting one: *actions earn their place*
had been written off as unsupported on the strength of a split that depended
on how Parquet happened to lay the data out.

### Checking the checkers

A green suite proves the tests pass. It does not prove they would notice if
the thing they describe stopped being true, so `make mutants` breaks each
load-bearing claim on purpose and checks that something fails:

```
claim broken on purpose                                result
M-FALCON normaliser depends on the candidate count     caught
the in-process broker partitions with crc32 again      caught
a calibrator may reverse a head's ranking              caught
the offline gate passes when it cannot be evaluated    caught
FAISS probes a tenth of the cells again                caught
Iceberg reaches Spark through pandas again             caught
the paired test calls everything significant           caught
the time split goes back to an approximate quantile    caught
the bundle manifest is written without scrubbing       caught
/v1/model returns the manifest verbatim                caught
consent fails open instead of closed                   caught
the erasure verifier greps Parquet instead of decoding caught
erasure leaves the raw file everything derives from    caught
the Iceberg erasure skips snapshot expiry              caught
the cold-start list moves back into the user keyspace  caught
serving tails lose their TTL                           caught
the Spark consent filter stops trimming                caught
a dead online store raises instead of degrading        caught
the fallback needs the store it is recovering from     caught
a throwing ranker becomes a 500 again                  caught
an inventory read failure empties the page             caught
a head left out of the history-loss map is switched off caught
exclude_seen goes back to one global answer            caught
the retention guard stops refusing and empties a table  caught
early stopping leaves dropout off after a check         caught
the best weights are kept as a live reference           caught
early stopping returns the last weights, not the best   caught
the ablation harness selects epochs on test             caught
the ablation report renders only the deepest cutoff     caught
```

29 of 29 — and one of the new ones was *not* caught on the first run.
"Early stopping returns the last weights instead of the best" survived: the
test for it trained a tiny model on a fixture small enough to learn
completely, the validation metric saturated, and the last epoch scored
exactly what the best one did, so returning either looked the same. The test
was asserting something true about a fixture that could not tell the
difference. It now drives the monitor directly — a best epoch, then
different weights and a worse score, then restore — and fails the moment
`restore` stops restoring.

The earlier history of this list: it was 7 of 8 the first time. The miss then was instructive too: the test written to
guard the map-type defect exercised `io/iceberg.py`, and the defect had been
in `io/tables.py` — adjacent code, different path, and reverting the fix left
the test green. The full-pipeline test did catch it, in seventy seconds. There
is now a millisecond one, which is the difference between a guard that runs
and a guard that runs nightly.

### Installing it from nothing

Every number above came from an environment with packages accumulated across
weeks of work, which says nothing about whether the *declared* dependencies
are enough. A clean virtualenv and `pip install -e ".[dev,serving]"` found two
things:

- **`pydantic` was imported at module level and never declared.** It worked
  because FastAPI happens to pull it in, which is a fact about FastAPI's
  packaging rather than about this project.
- **`pandas` likewise** — and this one was reachable. `pyproject.toml`
  supports `pyspark>=3.5`, PySpark 3.5 cannot convert Arrow directly, and the
  fallback that exists for exactly that case calls `.to_pandas()`. On a fresh
  install it raised `ModuleNotFoundError` instead of falling back.

`tests/test_platform.py` now compares every third-party import against
`pyproject.toml`, so an inherited dependency fails the build rather than a pod.

The same install also showed that a documented `make install` leaves the
partitioner check skipped, because comparing against the real Kafka client
needs a client that extra does not install. The core claim now has hardcoded
test vectors from the Java implementation and is checked everywhere; the
comparison against the live client remains as a bonus when it is present.

### Documentation drift

The README carries the measured numbers, so it is part of the system.
`tests/test_docs.py` asserts that every command it names exists, every make
target it shows is defined, every path it points at is there, the test count
has not drifted more than 10%, and the *unverified* list has not kept naming
backends that now run. What no test can check is whether a number in the prose
is still the number a fresh run produces — that needs the run, and
`artifacts/` is where it lands.

---

## Where this stands

Verified here, end to end:

- Raw data → lakehouse → sequences → popularity, SASRec and HSTU, across four
  token granularities, per category, on a leak-free time split.
- A component ablation of HSTU with bootstrap intervals, paired tests,
  multiple-comparison adjustment and three training seeds — which found that
  no component changes the top ten, that three effects at depth survive a
  change of dataset and of training schedule, and that the report had been
  hiding which cutoff it measured.
- Early stopping on the validation split that every model accepted and none
  read, and a measurement showing the models were never undertrained: they
  plateau by epoch 3–8 and eight epochs changed nothing at NDCG@10.
- The same questions on real timestamps (MovieLens-1M, 1M ratings).
- The streaming path: schemas, producer, replay, bronze sink, session state.
- The serving path: retrieval → policy → API, p99 measured at 4.8 ms.
- The ranker: interleaved sequences, four heads, linked return labels,
  M-FALCON equivalence asserted, 19.7 ms p99 inside its 35 ms budget — and an
  offline gate that blocks it from serving because it orders worse than
  retrieval.
- Calibration of the ranker's heads on serving-shaped slates, which found
  73–82% of the outcome variance explained on the distribution they train on
  and under 0.03% on the one they are deployed on — **and then found that the
  first of those two numbers was not measuring ranking at all.** A five-entry
  lookup table on how many times the same item just repeated scores AUC 0.965
  against the model's 0.986, and beats it outright on the purchase head. The
  81.4% was the synthetic generator's funnel.
- A controlled test of the obvious fix for that — negatives drawn from
  retrieval's own candidates — which refuted it: harder negatives moved slate
  resolution the wrong way and made ordering significantly worse.
- Four more arms after the diagnosis, of which the reasoned one failed and an
  unreasonable-looking one worked: restricting training positives to items
  `exclude_seen` would actually show did not help, and dropping the
  teacher-forced loss nearly tripled slate ordering while taking the `return`
  head to chance. The trade is real and no arm here escapes it.
- Consent as an enforced field with the cost measured rather than assumed
  (free at realistic opt-in rates, and the damage depends on *who* opts out
  only once half of them do), pseudonymisation as real HMAC verified equal
  between Python and Spark, and an erasure path whose verifier was caught
  reporting success on data it could not see inside.
- Every dependency taken down in turn and the caller's actual response
  recorded, which found that a dead Redis was a 500 on every request and that
  the fallback for everything else was itself a Redis call.
- The portability claim, as a test that exits non-zero: no vendor SDKs, no
  unreachable endpoints, every backend conforming to its protocol.
- The FAISS backend, verified for the first time and found returning 41% of
  the true top-20 — and measured to be no faster than the exact matmul below
  roughly 100,000 items.
- The deployment: manifests validated against the real Kubernetes schemas,
  carrying a measured 523 MiB warm footprint and a 1.03 s cold start.
- Two of the three backends that had never been executed: Redis against a real
  `redis-server`, and Iceberg through a JVM-free engine. Running them found
  that the time split was not deterministic across storage layouts, that a
  `MapType` was degrading in transit, and that the Kafka contract had been
  skipping for a dependency that does not exist.

Also not done, and worth saying because the config implies otherwise: the
exposure logger is built and tested and `api.serve` passes `None` for it, so
`recs.served.v1` is never written in the default deployment. The wire format
is JSON rather than the Avro-with-Schema-Registry the config's endpoint
implies. And `privacy.pseudonymisation` is off by default, so the pipeline's
pseudonymised path is exercised by tests rather than by any run reported here.

Not verified here: real Kafka — the broker is a Java distribution and Maven,
the Apache download hosts, GitHub releases and the Go module proxy are all
blocked from this environment, so there is no route to one. Its partitioner is
verified against the real client's hash; its admin, producer and consumer-group
code is not. Also unverified: the REES46 and H&M adapters, which need the
datasets.

Next, in rough order of value:

1. **Give the ranker a way out of the trade it is stuck in.** The measured
   position: dropping the teacher-forced loss nearly triples slate ordering
   and takes the `return` head to chance; keeping it only for `return` saves
   that head and orders *worse than the control*. No arm escapes, and the
   best reaches 61% of retrieval's ordering signal, which is not enough to
   serve. The structural cause is still the one the config comment named
   first — retrieval trains against the full softmax at every position of
   every sequence, the ranker against a handful of candidates at one position
   per user per epoch. What has *not* been tried is a candidate loss at every
   position rather than one cut per user, which needs the candidate block to
   attend per position rather than to the whole prefix, and exposure-log
   negatives, which carry real "shown and not taken" information instead of
   manufactured false negatives. The number to watch is slate AUC against
   retrieval's 0.714 — not resolution share, which is quadratic in the
   deviation from a 0.2% base rate, and not AUC on history positions, which a
   five-entry lookup table already scores 0.965 on.
2. **Give the return head a slate-shaped label.** Now the highest-value fix
   rather than the fourth, because it is what would make the trade above
   measurable in one place. The head is invisible to every slate measurement
   for a structural reason — a slate yields at most one matured purchase per
   user — so an arm can win on the slate and silently give the head away. It
   needs return outcomes attached to targets, which means carrying the
   maturity window into the target window.
3. **Run the Kafka contract against a real broker, and swap JSON for Avro
   while you are there.** The contract is written and skips here for a reason
   the suite states correctly. The second half is smaller than it sounds: the
   two `_serialise`/`_deserialise` hooks in `broker.py`, a Schema Registry
   client pointed at the endpoint the config already carries, and registering
   each schema under `<topic>-value`. Until then the payloads carry no schema
   id, so an out-of-date producer is caught by a consumer raising on a missing
   field rather than by the registry refusing the write.
4. **Settle `exclude_seen` with something the metric cannot settle.** It is
   now per-surface and the offline numbers are in: turning it off doubles
   recall@10 on non-overlapping intervals. But recall counts a repeat as a
   hit, and whether re-showing what the customer already found is worth
   anything needs an A/B test, not another evaluation.
5. **Re-run MovieLens with intervals and seeds.** Its numbers predate the
   statistics module and are the last point estimates left in this README.
6. **Less capacity, or more data — not more epochs.** Training to
   convergence was measured and changed nothing served: validation plateaus
   by epoch 3–8 while training loss keeps falling, so these models run out of
   data long before they run out of epochs. Dropout, weight decay and a
   smaller model are the cheap experiments; real retail data is the one that
   matters.
7. **Close the loop**: wire the exposure logger into `api.serve` — it is built,
   tested and passed `None` in the default deployment, so
   `retailgr_exposure_log_total` currently counts nothing — then feed
   `recs.served.v1` into the label builder and run the first A/B against the
   popularity baseline.
8. **Put it on Kubernetes for real**: Strimzi for Kafka, the Spark operator for
   the batch jobs, the serving image behind the HPA that is already written.

Reference for the model:
[paper](https://arxiv.org/abs/2402.17152) ·
[code](https://github.com/meta-recsys/generative-recommenders) (Apache-2.0).
