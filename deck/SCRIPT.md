# NeonNull — finale script, 9:00 for a 10:00 slot

Rehearse to 9:00. The 60 seconds of slack absorbs the "can you hear me" at the
start and one judge interrupting mid-way. Never plan to use all ten.

**You are online and they are in a room.** That costs you presence and buys you
two things: these notes are invisible to them, and your slides arrive at full
resolution instead of through a projector. Use both. Nobody in that room can be
word-perfect. You can.

Three rules for the delivery:

1. **Pause after every number.** Online audio compresses; a number spoken into a
   continuous stream is lost. Say it, stop for one beat, continue.
2. **Say "next slide" out loud is not needed, but do say the slide's idea in your
   first sentence** so anyone whose video lagged is re-anchored.
3. **Slow down 15% from what feels natural.** Every remote presenter speeds up.

Bold text is where the emphasis goes. `//` is a deliberate pause.

---

## 1 — Cover · 0:00 → 0:25

Good morning. I'm Sahil, Team NeonNull, joining from Vellore.

Seven polymer properties, one model. // The hard part of this problem was never
accuracy. It was that **five of the seven targets have about two hundred and
twenty training rows each**, and the score weights all seven equally.

> *Own the remote thing in one clause and never mention it again. Do not
> apologise for not being in the room.*

## 2 — The imbalance · 0:25 → 1:20

So here is the shape of it. **Seventy one percent of the metric is carried by
fourteen percent of the data.** //

Fit those five independently and you are fitting a few hundred rows with a
thousand features. A per-target LightGBM reaches R-squared **zero point seven
five** on dielectric constant. That is not a model anyone would use.

The usual answers are regularisation and feature selection. We tried both.
Neither moves this, because the problem is not variance in the fit. It is that
there is genuinely not enough signal in two hundred rows.

The only way to get more signal is to bring in rows from somewhere else.

## 3 — We measured the targets first · 1:20 → 2:30  ★

*(Slow down. This is the slide the whole talk hangs on.)*

So the question becomes: which somewhere else. And we decided to **measure that
before choosing a model**. //

Four hundred and fifteen molecules in this dataset carry more than one label.
That overlap lets you compute the correlation between targets directly, and the
structure is unambiguous.

Chain band gap against bulk band gap: **plus zero point nine three**. //
Dielectric constant against refractive index: **plus zero point nine two**,
which is the textbook relation, n-squared is approximately epsilon, turning up
in the data on its own. // Chain band gap against refractive index: **minus zero
point eight five**.

All six are frontier-orbital quantities. Physically, they are **one property
looked at six ways**. //

Glass transition is the exception. It is thermal, it shares no molecule with any
of the six, and it has four thousand rows of its own. So it trains alone.

That slide decided the architecture, before we ran a single model.

## 4 — Architecture · 2:30 → 3:20

Which gives us this. Two paths and one shared encoder.

The tree path is RDKit descriptors, fingerprints, and eighteen polymer-specific
features we compute from the **backbone**, the chain between the two attachment
points.

The graph path is a message-passing network trained from scratch. No pretrained
weights, because the rules don't allow them.

The shared encoder is the piece that matters. One network reads the molecule for
all six electronic targets, and six small heads read off the individual
properties. Each row only ever trains its own head, but **every row trains the
encoder**. So the two hundred and twenty refractive-index rows are learning from
all three thousand two hundred electronic rows.

## 5 — Where the gain lands · 3:20 → 4:20  ★

And this is the result I'd point you to first.

Refractive index gains **zero point zero five six**. Dielectric constant, **zero
point zero four seven**. Both have two hundred and twenty nine rows. // Chain
band gap, the one with two thousand rows, gains **zero point zero one zero**. //

So the gain **concentrates where the rows are scarce**. That is the signature of
transfer actually happening, rather than a number that merely went up. If the
big targets had gained the most, I would not trust it.

There's one exception, and I'd rather name it than have you find it. Electron
affinity has two hundred and twenty one rows and gains only zero point zero one
zero. It was already at R-squared zero point nine three on its own. The gain
needs **scarce data and remaining headroom**, and that target had no headroom
left.

Across the cluster it's plus zero point zero one five seven. Then moving from
five folds to ten, **at identical compute**, added another zero point zero zero
five eight. That was the single largest gain of the round.

## 6 — Results · 4:20 → 5:10

Full numbers. These are out-of-fold, and the blend weights are fitted by nested
cross-validation, so **no weight ever sees the row it scores**. That matters,
because with two hundred rows it is very easy to fit weights that flatter you.

Read the multi-task column against LightGBM and GNN. It wins on all six.
Dielectric constant goes from zero point seven five to zero point eight six.

Official metric **zero point nine one four**, against the organiser's ridge
baseline of zero point five four. And the progression underneath is the honest
history: zero point eight nine per-target, zero point nine zero four with the
shared encoder, zero point nine one four at ten folds.

## 7 — Predictions · 5:10 → 5:35

All seven targets against truth. The orange panels are the small ones. They're
noisier, which is what two hundred rows should look like.

What I'm checking here is that there's no structured bias. No curvature, no fan,
no drift at the extremes. There isn't.

## 8 — Invariance · 5:35 → 6:45  ★

*(Drop your pace here. This is the one judges will remember, and it's a theme
the round explicitly scored.)*

Now the part I think matters most, and it isn't a score. //

A polymer can be written as several different SMILES strings. Same molecule,
different spelling. **A model that answers differently for two spellings of the
same molecule isn't measuring chemistry. It's measuring text.**

So we tested it. We re-spelled every molecule at random and re-predicted.
Predictions moved by up to **three percent of the target's standard deviation**. //

Cross-validation would never have shown that, because in train and in test each
molecule is written one particular way. **You only see it if you go looking.**

The cause was atom ordering. Tied ring paths, Gasteiger charges on the dummy
atoms, and Ipc rounding all depend on the order atoms arrive in. The fix was to
re-parse every molecule from its canonical SMILES and drop the one descriptor we
couldn't stabilise.

It cost zero point zero zero zero one. Deviation afterwards is **exactly zero**.
Not small. Zero. // And sixty five percent of the supplied SMILES were not
canonical, so this was live in the competition data.

## 9 — What didn't work · 6:45 → 7:30

Nineteen experiments. Four survived. Three are worth naming.

**PI1M pretraining.** We masked atoms and learned to reconstruct them across a
hundred thousand unlabelled repeat units. It reached **ninety nine percent**
reconstruction accuracy, and it still **lost zero point zero zero three eight**
in the blend. Good at the pretext task, not better for our targets. We're
reporting it because it was the auxiliary dataset you highlighted, and that's
the honest finding.

And multi-task including glass transition: **minus zero point zero two three**.

## 10 — Round 2 to Round 3 · 7:30 → 8:15

Which is the point of this slide. That minus zero point zero two three is from
Round Two, where **the same technique made things worse**. One encoder over Tg
and band gap. They share no physics, so the shared encoder just diluted both. //

Same technique. Opposite result. The difference isn't the model. It's **whether
the targets belong together**.

That's why this round we measured the correlations first and let them choose the
architecture, instead of trying multi-task and seeing what happened.

## 11 — Particulars · 8:15 → 8:40

Briefly, the engineering. Three hundred and eighty seven thousand parameters.
Four and three quarter hours on Kaggle CPU, no GPU. Under a millisecond per
molecule at inference.

And it's deterministic. Parallel message passing isn't, and we measured two
identical runs differing by **zero point one eight per prediction** before we
pinned it. Kaggle reproduced our local score within zero point zero zero three
five on every run of the competition.

## 12 — Close · 8:40 → 9:05

To close, what this says about polymers. Band gap falls as conjugation runs
further along the backbone. Refractive index and dielectric constant rise with
the same polarisable, aromatic-rich backbones that narrow that gap.

Which is **one story, not three**. And it's the same story the correlation
matrix told us before we built anything.

Everything is in the notebook and the repo. Thank you. I'm happy to take
questions.

---

## If you are running long

Cut in this order, and only these:

1. **Slide 7** to one sentence: "All seven against truth, no structured bias."  (saves 20s)
2. **Slide 11** to one sentence: "387k parameters, under five hours on CPU, deterministic and reproduced on Kaggle." (saves 20s)
3. **Slide 2** second paragraph, the regularisation one. (saves 15s)

Never cut slides 3, 5, 8 or 10. They are the argument.

## If you are cut off at the time limit

Stop mid-sentence and say: "I'll stop there. The one line I'd leave you with is
that we measured which targets belong together before choosing a model, and that
decision is worth more than any tuning we did."
