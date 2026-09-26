# Finale — Q&A bank and online run sheet

> **One thing to settle before you rehearse.** This Q&A was written when the
> presentation was solo, and it now has two presenters. Judges at a finale
> routinely ask "who did what". Decide the honest answer between you, write it
> into the "Team size / who did what" row below, and make sure both of you give
> the *same* answer. Two presenters describing the split differently is the one
> thing in this document that could actually cost you.

## Part 1 — The run sheet

Your competition here is not the other teams' models. It is that you are a
window on a projector and they are people in a room. Everything below exists to
close that gap.

### The single highest-value thing you can do

**Email Aryan the PDF the night before**, with one line: *"In case my connection
drops, here are my slides so the room can still follow."* If you lose the call
at minute four, someone advances slides while you dial in by phone and keep
talking. Without that, a dropped connection ends your presentation.

### Two people, two connections

Everything below doubles. Both of you need the audio setup, both need the
network fallback, both need to be on camera.

- **If you're in separate rooms:** never in the same room on two devices without
  headphones, or the feedback howl will end the call. Separate rooms is safer.
- **If you're in the same room:** one device, one mic, both of you in frame,
  second device fully off (not muted, off).
- Whoever is not speaking **mutes only if their room is noisy**. Otherwise stay
  unmuted, because unmute lag clips the first words of every handoff.
- Exchange phone numbers and keep them to hand. If one of you drops, the other
  keeps presenting and covers both sections from the script. Agree in advance
  that this is the plan, so the remaining person doesn't freeze.

### Audio, in priority order

Judges forgive a frozen camera. They do not forgive audio they have to strain
through, and straining is what makes a remote presenter forgettable.

- **Wired earphones with an inline mic.** Not the laptop mic, which picks up
  room echo and your keyboard. **Not Bluetooth earbuds** — the moment the mic
  opens, the codec drops to narrowband mono and you sound like a phone call from
  2005. This is the most common self-inflicted wound in online presentations.
- Test on the actual platform once the day before, and again 20 minutes before.
  Record 30 seconds and listen back. You are checking for echo and for whether
  your plosives pop.
- Have the phone dial-in number saved. If audio dies, dial in and keep going.

### Where your script lives

You have one screen and you are sharing it, so PowerPoint presenter view is not
available unless you have a second monitor. If you don't:

- Open `SCRIPT.md` **on your phone, propped behind the laptop at eye level.**
  Not flat on the desk. Eyes down at a desk reads as reading; eyes near the
  camera reads as talking.
- Phone on Do Not Disturb and screen timeout set to 10 minutes or Never.

If you do have a second monitor, use presenter view and put the notes there.

### Network

- Ethernet if you can get it. Otherwise sit next to the router.
- **Phone hotspot already paired and tested**, so switching is two taps, not a
  password hunt.
- Close everything syncing: Drive, Dropbox, backups, other browser tabs.

### Frame and room

- Camera at eye level. Stack books under the laptop.
- Light in **front** of you, not behind. A window behind you makes you a
  silhouette.
- Plain wall behind you. No virtual background — they chew CPU and smear your
  edges when you gesture.
- Water off-camera. Do Not Disturb on the laptop.

### Timeline on the day

| When | Do |
|---|---|
| Night before | Email PDF to Aryan. Full run-through out loud, timed. |
| 60 min before | Restart the laptop. Nothing else running. |
| 20 min before | Join the test link, check audio, check screen share. |
| 10 min before | Join the real call. Camera on. Mute until called. |
| Your slot | Unmute, share the **Slideshow window** only. Not the desktop, not the editing view. |
| After Q&A | Ask to stay on the call for the rest of the session. |

That last row matters more than it looks. Teams that present and disappear are
teams that are not in the room when the judges talk it over. Stay visible.

### The first fifteen seconds

Say exactly this and nothing else about logistics:

> "Can everyone hear me clearly? ... Great. Good morning, I'm Sahil, Team
> NeonNull."

**Ask once.** Do not ask again mid-talk, do not say "sorry, can you still hear
me", do not apologise for being remote. One check, then behave as if you are in
the room.

---

## Part 2 — Q&A

### Who answers what

Agree this before you join. Two people fielding questions is worse than one
unless you have a rule, because the failure mode is both starting at once and
then both stopping at once.

**The rule: whoever presented the slide the question is about answers it first.**
Slides 1-5 and 11-12, Sahil. Slides 6-10, Neerav.

- If the question is about a modelling decision, why an architecture, or a number
  that isn't on a slide, **Sahil takes it** regardless of which slide it came
  from.
- If you're handed a question you can't fully answer, don't bluff and don't
  trail off. Hand it over by name and in one clause: *"Sahil has the detail on
  that one."* That reads as a team that knows its own division of labour, not as
  a gap.
- **Never both start.** If it happens, Neerav stops. Fixed rule, decided now, so
  neither of you has to negotiate it live in front of judges.
- The one who isn't answering stays on camera and stays engaged. Do not look at
  your phone. In gallery view everyone can see you.

### How to answer, online

- **Paraphrase the question before you answer it.** "So the question is whether
  the multi-task gain could be a cross-validation artefact." This buys thinking
  time, confirms you heard it over bad audio, and helps everyone in the room who
  also didn't hear it.
- **30 to 45 seconds, then stop.** Silence feels twice as long to you as to
  them. Stop talking and let them come back.
- If you didn't hear it: *"Sorry, the audio cut on the last part, could you
  repeat it?"* Once is fine. If it happens twice, ask them to type it in chat.
- If you don't know: say so, then say what you would measure to find out. That
  answer scores better than a guess, and judges can always tell.

---

### The ones you will almost certainly get

**"Why did the PI1M pretraining fail?"**

Masked-atom reconstruction got to 99% accuracy, so the model learned the pretext
task well. It learned local valence and connectivity rules, which our descriptors
and the supervised GNN already encode explicitly. So it spent capacity on
something we weren't missing. What it did not learn is anything about frontier
orbitals, because nothing in an unlabelled repeat unit tells you where the HOMO
is. With 3,266 labelled electronic rows, supervised multi-task transfer from
*correlated labels* beat unsupervised transfer from a million unlabelled
molecules. We report it as minus 0.0038 rather than omitting it, because it was
the auxiliary dataset you highlighted.

**"You came second, not first."**

On the public split, by 0.002. I can tell you what that's worth: four of the
seven targets have between 54 and 82 rows in the public subset, and bootstrapping
our own predictions at that size gives the public score a standard deviation of
0.0074. So the gap is about a quarter of one standard deviation. That's not a
claim that we're really first. It's the reason we never tuned against the public
board. Both final submissions were selected on out-of-fold score across 7,409
labelled rows, not on rank.

**"What's actually novel? Anyone can blend LightGBM and a GNN."**

Agreed, and the model family isn't the contribution. Three things are. One, the
correlation structure is an **input to the architecture**, not a result: we
measured which targets belong together on the 415 multi-label molecules and let
that decide what shares an encoder. Two, the eighteen backbone features, which
read the chain between the two attachment points as a polymer rather than
treating the molecule as a bag of substructures. They beat 1,024 Avalon bits.
Three, we treated invariance as a measurement, and it found a real defect that
cross-validation could not have.

**"How do you know the multi-task gain isn't a CV artefact?"**

Three checks. Each target keeps its own fold split, so a validation row's label
never reaches the encoder in the fold that scores it. The blend weights are
fitted by nested cross-validation, so no weight sees the row it scores. And the
gain scales inversely with the number of training rows a target has, which is
what transfer should look like. If the gain had been uniform, or largest on the
big targets, I'd have assumed it was leakage. It held on the public split too.

**"Your own chart contradicts you. EEA has 221 rows and only gains 0.010."**

*(You name this on the slide yourself, so it should not come as an attack. If it
does:)* That's right, and it's the one exception. EEA was already at R-squared
0.931 as a per-target model, which is the highest of the five small targets. The
gain needs two things, scarce data and remaining headroom, and EEA only had the
first. The targets that gained most, NC and EPS, started at 0.86 and 0.81. So the
mechanism is consistent: the encoder can only give a target what that target was
still missing.

**"Why not use a pretrained chemistry model?"**

The rules prohibit pretrained weights, so everything trains from scratch in one
notebook. But we also tested the in-domain version of that idea ourselves on
PI1M, and it measured negative. So I don't think we left much on the table.

---

### The ones that separate a good answer from a great one

**"Would this work on a polymer class that isn't in your training set?"**

Honestly, I don't know, and I'd want to say that clearly rather than guess. We
validated with random splits, not scaffold splits, so nothing we measured speaks
to that. My expectation is that the backbone features extrapolate better than the
fingerprints, because they're continuous physical quantities like conjugation run
length rather than indicators for substructures that may simply be absent. But
before anyone acted on a prediction for a new class I'd want two things: a
scaffold-split evaluation, and a conformal prediction interval so the model
reports when it's out of its depth. That's the first thing I'd build next.

**"Can you explain an individual prediction?"**

Yes, and at two levels. Globally, feature importance concentrates on the backbone
block, and the single strongest signal is dimer conjugated-bond fraction, which
correlates minus 0.76 with chain band gap. Locally, for a given polymer we can
report which structural motifs are driving the value: how far conjugation runs
along the backbone, whether the attachment atoms are sp2, how much mass sits in
side chains. So the output is a number plus the reason for it, which is what a
chemist actually needs to act.

**"If you had one more week, what would you do?"**

Not more models. The blend saturated. I'd spend it on labels. The model's
uncertainty tells you which polymers it's least sure about, so I'd use it to
select fifty molecules to measure next, and add them to the 220-row targets.
That's the highest-value thing this model can do for a lab: not predict
everything, but tell you which experiments are worth running.

**"What was the hardest part?"**

Not the modelling. It was that RDKit isn't on the Kaggle image and internet is
off, so two two-hour runs failed before we understood why. After that we built a
three-minute pre-flight kernel that checks the environment before committing
compute, and a deploy script that refuses to push if the deployed code doesn't
match local. Two submissions had already been wasted on stale code. The lesson
I'd carry is that shipping discipline cost us more score than any hyperparameter.

---

### Quick answers, for the ones that come fast

| Question | Answer |
|---|---|
| Team size / who did what? | **[FILL THIS IN BEFORE THE DAY — see the note at the top of this file.]** Answer it plainly and without hedging; judges ask this to understand the team, not to catch anyone out. |
| Why ten folds? | Not more compute. We halved the seeds and doubled the folds. Each model sees 90% of the data instead of 80%. Worth +0.0058. |
| Why does a GNN help with only 3,266 rows? | It isn't used alone. Its errors decorrelate from the trees, so it earns 0.5 to 0.85 blend weight. On Tg by itself it's actually worse than LightGBM, 0.905 against 0.913. |
| Isn't canonicalising SMILES standard? | Canonicalising the *string* is. Our pipeline did that and was still order-dependent underneath, in the ring paths, the Gasteiger charges and Ipc. The contribution isn't the fix, it's that we measured instead of assuming. |
| Could you add Tg to the shared encoder with gating? | Possibly, but Tg has 4,143 rows. It doesn't need help, and in Round 2 including it cost 0.023. |
| Runtime? | Four hours forty three minutes on Kaggle CPU, no GPU. Under a millisecond per molecule at inference. |
| Is it reproducible? | Single-threaded with per-fold seeds. Kaggle reproduced our local out-of-fold score within 0.0035 on every run. |

---

## Part 3 — The one line to land

If you remember nothing else under pressure, this is the talk:

> **We measured which targets belong together before choosing a model. Six of the
> seven are one physical quantity, so they share an encoder. The same technique
> lost 0.023 in Round 2, where the two targets shared no physics.**

Any question you can't answer, you can bridge back to that sentence and still be
saying something true and specific.
