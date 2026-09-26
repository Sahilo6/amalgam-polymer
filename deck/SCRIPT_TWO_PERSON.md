# NeonNull — finale script for two speakers

Same 12 slides, same argument, two voices. It runs a shade over nine minutes in
a ten-minute slot.

**One checkpoint to rehearse against:** slide 8 should be starting at around the
six-minute mark. If it's later than that, take the cuts at the end of this file.

## The division

| Slides | Speaker |
|---|---|
| 1 – 5 | **Sahil** |
| 6 – 10 | **Neerav** |
| 11 – 12 | **Sahil** |

Two handoffs, not twelve. Every handoff online costs a few seconds of dead air
and a moment where nobody is sure who is talking, so we do it twice and both
times at a natural break in the argument.

Sahil opens with the problem and the insight that decides the architecture.
Neerav carries the evidence: the numbers, the invariance finding, and the
negative results. Sahil closes.

## Mechanics, agreed before you join

- **Sahil shares the screen for the whole ten minutes and advances every slide**,
  including during Neerav's section. Switching who shares mid-talk costs ten to
  fifteen seconds each way and sometimes just fails. Not worth it.
- Neerav's section below has `[→ ADVANCE]` markers. Sahil follows along on the
  same script. If an advance gets missed, Neerav just says "next slide" in a
  normal voice and carries on. It is not a disaster and the judges will not care.
- **Both cameras on, the whole time.** In a gallery of faces the judges need to
  see who is speaking. The person not speaking should look attentive, not bored,
  and should not be visibly reading.
- **Both unmuted**, assuming two quiet rooms. Unmuting lag clips the first two or
  three words of a handoff, which is exactly where clarity matters most. If one
  room is noisy, that person mutes and accepts the delay.
- Never talk over each other. If both start, **Sahil stops**, every time. Agree
  this now so neither of you has to decide in the moment.

## Handoff technique

The person handing off names the other person **and** what is coming next. The
person picking up says thanks and goes straight in. No "over to you", no dead
air, no "can you hear me".

---

# SAHIL — slides 1 to 5

## 1 — Cover
Good morning. I'm Sahil, this is Neerav, and we're Team NeonNull, joining from
Vellore.

Seven polymer properties, one model. // The hard part of this problem was never
accuracy. It was that **five of the seven targets have about two hundred and
twenty training rows each**, and the score weights all seven equally.

> *Ask "can everyone hear us" ONCE, before this. Then never again. Do not
> apologise for being remote.*

`[→ ADVANCE]`

## 2 — The imbalance
So here is the shape of it. **Seventy one percent of the metric is carried by
fourteen percent of the data.** //

Fit those five independently and you are fitting a few hundred rows with a
thousand features. A per-target LightGBM reaches R-squared **zero point seven
five** on dielectric constant. That is not a model anyone would use.

The usual answers are regularisation and feature selection. We tried both.
Neither moves this, because the problem is not variance in the fit. It is that
there is genuinely not enough signal in two hundred rows.

The only way to get more signal is to bring in rows from somewhere else.

`[→ ADVANCE]`

## 3 — We measured the targets first  ★
*(Slow down. The whole talk hangs on this slide.)*

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

`[→ ADVANCE]`

## 4 — Architecture
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

`[→ ADVANCE]`

## 5 — Where the gain lands  ★
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

> ### HANDOFF 1
> **Sahil:** "Neerav will take you through the numbers, and the finding I think
> matters most, which is what happens when you re-spell a molecule."
>
> *Stop. Do not add anything. Advance the slide.*

`[→ ADVANCE]`

---

# NEERAV — slides 6 to 10

## 6 — Results
Thanks Sahil.

Full numbers. These are out-of-fold, and the blend weights are fitted by nested
cross-validation, so **no weight ever sees the row it scores**. That matters,
because with two hundred rows it is very easy to fit weights that flatter you.

Read the multi-task column against LightGBM and GNN. It wins on all six.
Dielectric constant goes from zero point seven five to zero point eight six.

The official metric is **zero point nine one four**, against the organiser's
ridge baseline of zero point five four. And the progression underneath is the
honest history: zero point eight nine per-target, zero point nine zero four with
the shared encoder, zero point nine one four at ten folds.

`[→ ADVANCE]`

## 7 — Predictions
All seven targets against truth. The orange panels are the small ones. They're
noisier, which is what two hundred rows should look like.

What we're checking here is that there's no structured bias. No curvature, no
fan, no drift at the extremes. There isn't.

`[→ ADVANCE]`

## 8 — Invariance  ★
*(This is your slide. Drop your pace, and give the two-spellings line room.)*

Now the part we think matters most, and it isn't a score. //

A polymer can be written as several different SMILES strings. Same molecule,
different spelling. **A model that answers differently for two spellings of the
same molecule isn't measuring chemistry. It's measuring text.**

So we tested it. We re-spelled every molecule at random and re-predicted.
Predictions moved by up to **three percent of the target's standard
deviation**. //

Cross-validation would never have shown that, because in train and in test each
molecule is written one particular way. **You only see it if you go looking.**

The cause was atom ordering. Tied ring paths, Gasteiger charges on the dummy
atoms, and Ipc rounding all depend on the order the atoms arrive in. The fix was
to re-parse every molecule from its canonical SMILES and drop the one descriptor
we couldn't stabilise.

It cost zero point zero zero zero one. Deviation afterwards is **exactly zero**.
Not small. Zero. // And sixty five percent of the supplied SMILES were not
canonical, so this was live in the competition data.

`[→ ADVANCE]`

## 9 — What didn't work
Nineteen experiments. Four survived. Three are worth naming.

**PI1M pretraining.** We masked atoms and learned to reconstruct them across a
hundred thousand unlabelled repeat units. It reached **ninety nine percent**
reconstruction accuracy, and it still **lost zero point zero zero three eight**
in the blend. Good at the pretext task, not better for our targets. We're
reporting it because it was the auxiliary dataset you highlighted, and that's
the honest finding.

And multi-task including glass transition: **minus zero point zero two three**.

`[→ ADVANCE]`

## 10 — Round 2 to Round 3  ★
Which is the point of this slide. That minus zero point zero two three is from
Round Two, where **the same technique made things worse**. One encoder over
glass transition and band gap. They share no physics, so the shared encoder just
diluted both. //

Same technique. Opposite result. The difference isn't the model. It's **whether
the targets belong together**.

That's why this round we measured the correlations first and let them choose the
architecture, instead of trying multi-task and seeing what happened.

> ### HANDOFF 2
> **Neerav:** "Sahil will close on how it runs, and what it actually says about
> polymers."
>
> *Stop talking. Sahil advances.*

`[→ ADVANCE]`

---

# SAHIL — slides 11 and 12

## 11 — Particulars
Thanks Neerav.

Briefly, the engineering. Three hundred and eighty seven thousand parameters.
Four and three quarter hours on Kaggle CPU, no GPU. Under a millisecond per
molecule at inference.

And it's deterministic. Parallel message passing isn't, and we measured two
identical runs differing by **zero point one eight per prediction** before we
pinned it. Kaggle reproduced our local score within zero point zero zero three
five on every run of the competition.

`[→ ADVANCE]`

## 12 — Close
To close, what this says about polymers. Band gap falls as conjugation runs
further along the backbone. Refractive index and dielectric constant rise with
the same polarisable, aromatic-rich backbones that narrow that gap.

Which is **one story, not three**. And it's the same story the correlation
matrix told us before we built anything.

Everything is in the notebook and the repo. Thank you. We're happy to take
questions.

---

## If you are running long

Cut in this order, and only these. Whoever owns the slide makes the cut.

1. **Slide 7** (Neerav) to one sentence: "All seven against truth, no structured
   bias." — saves 20s
2. **Slide 11** (Sahil) to one sentence: "387k parameters, under five hours on
   CPU, deterministic and reproduced on Kaggle." — saves 20s
3. **Slide 2** (Sahil), drop the regularisation paragraph. — saves 15s

Never cut 3, 5, 8 or 10. They are the argument.

## If you are cut off at the time limit

Whoever is speaking stops mid-sentence and says:

> "I'll stop there. The one line we'd leave you with is that we measured which
> targets belong together before choosing a model, and that decision was worth
> more than any tuning we did."

## Rehearsal plan

You need three run-throughs together, not one.

1. **Read-through, untimed.** Neerav reads his sections aloud once to find the
   words that don't sit right in his mouth. Change them. A script you fight is
   worse than one you wrote.
2. **Timed, cameras on, screen shared.** Full dress. Sahil practises advancing
   during Neerav's section. Run a single clock for the whole thing and check it
   against the slide-8 checkpoint, rather than timing anyone separately.
3. **Handoffs only.** Run just the last twenty seconds of slide 5 and the first
   twenty of slide 6, then the same around slide 10. Five times each. Handoffs
   are the only thing in a two-person talk that a solo talk doesn't have, so
   they're the only thing that needs isolated practice.
