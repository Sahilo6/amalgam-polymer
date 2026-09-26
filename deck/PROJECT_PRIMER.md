# What this project actually is

A from-scratch explainer. No chemistry background assumed, no Kaggle background
assumed. By the end of it you should be able to present half of it and defend
all of it.

---

## 1. The one-paragraph version

A polymer's useful properties (how hot it can get before it softens, how much it
bends light, whether it conducts) are expensive to measure in a lab and slow to
compute from physics. If you could predict them from the molecule's structure
alone, you could screen thousands of candidate polymers before synthesising any.
This competition gave us 6,565 polymer structures written as text, with seven
different properties measured across them, and asked for a model that predicts
all seven. We came second of the shortlisted teams on the public score, and the
part we are actually proud of is not the score.

---

## 2. What a polymer looks like as data

A polymer is a **repeat unit** joined to itself over and over, like a chain made
of identical links. You describe one link and note where the joins go.

The data gives each molecule as a **SMILES string**: a way of writing a molecule
as text. Real examples from the training set:

```
*OC(=O)c1ccc(cc1)C(=O)OCC(C(C*)CCC)CCC
*C(C*)c1ccc(cc1)C(O)C(F)(F)F
*CCCCCCCCCCCC(=O)N*
```

Reading it: letters are atoms (`C` carbon, `O` oxygen, `N` nitrogen), lowercase
means aromatic (a benzene-style ring), brackets are branches, digits open and
close rings. **The two `*` symbols are the attachment points** — that is where
this unit joins the next one to form the chain. That detail matters more than it
looks, and section 6 explains why.

### The seven properties

| Code | What it is | Units | Rows | Range |
|---|---|---|---|---|
| `tg` | **Glass transition temperature.** Below it the polymer is rigid and glassy; above it, rubbery. The single most quoted number about a plastic. | °C | 4,143 | −110 to 495 |
| `egc` | **Band gap, isolated chain.** Energy needed to kick an electron loose. Big gap = insulator, small gap = semiconductor. | eV | 2,028 | 0.02 to 9.86 |
| `egb` | **Band gap, bulk.** Same idea, measured in the packed solid rather than one chain. | eV | 337 | 0.51 to 10.11 |
| `eps` | **Dielectric constant.** How much the material stores electrical energy. Matters for capacitors and insulation. | — | 229 | 2.61 to 9.09 |
| `nc` | **Refractive index.** How much it bends light. Matters for lenses and coatings. | — | 229 | 1.56 to 2.76 |
| `ei` | **Ionisation energy.** Energy to remove an electron entirely. | eV | 222 | 4.03 to 9.84 |
| `eea` | **Electron affinity.** Energy released when it gains an electron. | eV | 221 | 0.39 to 5.14 |

**7,409 rows across 6,565 molecules.** The data is "long format": one row is one
molecule paired with one property, not one molecule with seven columns. Most
molecules have exactly one property measured. Only **415** have more than one.

Hold onto that 415. It is the hinge of the whole project.

---

## 3. The metric, and why it is the entire story

Score = the average of seven R² values, one per property, **unweighted**.

**R²** ("R squared") runs from 1.0 (perfect) down through 0 (no better than
always guessing the average) and can go negative (worse than guessing). It is
the fraction of the variation in the true values your model explains.

Unweighted is the whole game. Glass transition has 4,143 rows and counts for one
seventh. Electron affinity has 221 rows and also counts for one seventh.

> **Five of the seven properties have around 220 rows each. Together they are
> 71% of the score and 14% of the data.**

So this is not really a chemistry-prediction problem. It is a **small-data**
problem wearing a chemistry costume. Anyone can get a decent score on the two big
properties. The competition is decided entirely on the five tiny ones.

Fitting 220 rows against a thousand features is hopeless in the ordinary way. We
tried the ordinary ways: stronger regularisation, feature selection, simpler
models. None of them moved it, because the problem is not that the model is
overfitting. It is that **220 rows genuinely do not contain enough information**.
You cannot regularise your way to information that isn't there. You have to bring
in rows from somewhere else.

---

## 4. The central insight

Which rows? The obvious answer is "the other properties' rows", and the obvious
objection is that they measure different things.

So we checked, instead of assuming. Those **415 molecules with more than one
label** let you correlate the properties against each other directly. The result:

- chain band gap vs bulk band gap: **+0.93**
- dielectric constant vs refractive index: **+0.92**
- chain band gap vs refractive index: **−0.85**

Six of the seven properties are tightly coupled. That is not a coincidence in the
data, it is physics: all six depend on the same thing, **how easily electrons
move around in the molecule**. The dielectric/refractive-index pair at +0.92 is
literally a textbook relation (n² ≈ ε) reappearing in measured data.

Glass transition is the odd one out. It is about how chains slide past each
other, not about electrons. It correlates with nothing, shares no molecule with
the other six, and has 4,143 rows of its own. So it trains alone.

**That measurement decided the architecture before any model was trained.** The
six coupled properties share one neural network; glass transition gets its own.

### Why this is the interesting part

In Round 2 of this same competition we tried the same technique on two properties
(glass transition and chain band gap) and **it made things worse by 0.023**. Same
code, same idea, negative result — because those two share no physics, so one
network serving both just pulled it in two directions.

Same technique, opposite outcome. The difference is not the model. It is whether
the things you are sharing a model between actually belong together. That is the
sentence the whole presentation is built around.

---

## 5. What we actually built

Three models, blended.

**a) LightGBM (the "tree path").** Gradient-boosted decision trees over ~1,000
hand-computed numbers per molecule: standard chemistry descriptors from RDKit
(molecular weight, ring counts, polarity), plus **fingerprints** (a fingerprint
asks "does this molecule contain substructure #417?" a thousand times and writes
down a thousand yes/nos), plus 18 polymer-specific features we wrote ourselves.

**b) A message-passing graph neural network (the "graph path"), from scratch.**
Treats the molecule as a graph: atoms are nodes, bonds are edges. Each atom
repeatedly exchanges information with its neighbours, so after three rounds every
atom knows about its local neighbourhood, and the whole thing is pooled into one
vector describing the molecule. 387,000 parameters. Trained from random
initialisation because **the rules forbid pretrained weights**.

**c) The multi-task encoder.** This is the one that won it. A *single* GNN reads
any molecule and produces one shared representation. On top sit six tiny
"heads", one per electronic property. A dielectric-constant row trains the
dielectric head and the shared encoder; it never touches the refractive-index
head. But because the encoder is shared, **all 3,266 electronic rows contribute
to the representation that the 229-row properties use.**

Then the three models' predictions are **blended** per property, with weights
fitted to minimise error.

### Two pieces of methodology worth understanding

**Cross-validation and "out-of-fold".** Split the data into ten parts. Train on
nine, predict the tenth. Rotate. Every row ends up with a prediction made by a
model that never saw it. That collection is the **out-of-fold (OOF)**
prediction set, and scoring it is an honest estimate of performance on new data.

**Nested cross-validation for the blend.** Subtle and important. If you fit blend
weights on all the OOF predictions and then score those same predictions, the
weights have seen the answers and your score is inflated. With 220 rows that
inflation is large. So we fit the weights inside an inner loop that excludes the
rows being scored. Slower, less flattering, correct.

**Why this matters for you:** "how do you know your gain is real and not
overfitting?" is the most likely technical question, and nested CV is the answer.

---

## 6. The invariance finding

The part worth being proudest of, and it is not a score.

The same molecule can be written as several different valid SMILES strings —
same atoms, same bonds, different starting point and traversal order. Like
writing the same address as "12 Main St, Apt 4" versus "Apartment 4, 12 Main
Street".

A model should give identical answers for all of them. If it doesn't, it is not
reading chemistry. It is reading text.

Nobody was asking us to check this, and **cross-validation cannot reveal it**,
because in both training and test data each molecule appears written exactly one
way. So the model looks perfectly consistent while being silently
inconsistent. You only find it if you go looking.

We went looking: re-spelled every molecule randomly and re-predicted.
**Predictions moved by up to 3% of the property's standard deviation.**

The cause was atom ordering leaking into supposedly structural features. Three
specific culprits: when two paths through a ring tie, which one gets picked
depends on atom order; the charge calculation on the `*` dummy atoms depends on
atom order; and one descriptor (`Ipc`) produces numbers so large that rounding
differs with order.

Fix: re-parse every molecule from its **canonical** SMILES (a single agreed
spelling, so atom order is deterministic) before computing anything, and drop
`Ipc` entirely. Cost to accuracy: 0.0001. Deviation afterwards: **exactly zero.**

And it was not hypothetical — **65% of the SMILES the organisers supplied were
not canonical.** The defect was live in the competition data.

---

## 7. Results

Official metric **0.9139** out-of-fold, against the organisers' baseline of
0.537. Second on the public leaderboard at 0.900.

How it got there:

| Step | Score |
|---|---|
| Per-property models, blended | 0.889 |
| + shared multi-task encoder | 0.904 |
| + seed averaging | 0.908 |
| + ten folds instead of five | 0.914 |

The shared encoder is worth **+0.0157**, and it lands where the data is thinnest:
refractive index +0.056, dielectric constant +0.047, both with 229 rows, against
+0.010 for chain band gap with 2,028. Gains concentrating in the small properties
is the signature of transfer genuinely happening.

One honest exception: electron affinity has 221 rows but gains only +0.010. It
was already at 0.931 on its own. Transfer needs *both* scarce data and room to
improve, and that one had no room.

### About being second

The gap to first is 0.002. The public leaderboard is scored on 37% of the test
set, and four of the seven properties have only 54 to 82 rows in it. We
bootstrapped our own predictions at that size and found the public score has a
**standard deviation of 0.0074**. So 0.002 is about a quarter of one standard
deviation — noise, not a difference in skill.

That is not a claim that we are secretly first. It is why we never tuned against
the leaderboard, and selected both final submissions on out-of-fold score over
7,409 rows instead of on rank.

---

## 8. What didn't work

Nineteen experiments, four survived. Report the failures — judges trust people
who do.

- **Pretraining on PI1M (−0.0038).** PI1M is a million unlabelled polymer
  structures. We hid random atoms and trained a model to guess them, hoping it
  would learn chemistry it could reuse. It hit **99% accuracy at guessing the
  hidden atoms** and still made the final blend *worse*. It learned local bonding
  rules that our descriptors already encoded explicitly, and nothing about
  electron behaviour, because an unlabelled structure doesn't tell you that.
  Supervised transfer from 3,266 correlated labels beat unsupervised transfer
  from a million molecules.
- **Multi-task including glass transition (−0.023).** Section 4.
- **Multi-task LightGBM (+0.0004).** Too weak alone to earn blend weight.
- Extra fingerprints, hyperparameter tuning, more model families: all ≈ 0.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **SMILES** | A molecule written as a text string. |
| **Canonical SMILES** | The one agreed spelling for a molecule, so the same molecule always produces the same string. |
| **Repeat unit** | The link that repeats to form the polymer chain. The `*` marks where it joins. |
| **Backbone** | The main chain running between the two `*` attachment points. Side chains hang off it. |
| **Descriptor** | A single computed number about a molecule (weight, ring count, polarity). |
| **Fingerprint** | A long vector of yes/no answers to "does this molecule contain substructure X?". |
| **RDKit** | The standard open-source chemistry library. Computes all of the above. |
| **R²** | Fraction of variation explained. 1.0 perfect, 0 no better than the mean, negative worse. |
| **Fold / cross-validation** | Splitting data into parts, training on some and predicting the rest, rotating. |
| **Out-of-fold (OOF)** | Predictions for a row made by a model that never trained on it. Our honest score. |
| **Nested CV** | Cross-validation inside cross-validation, so blend weights never see the rows they're scored on. |
| **Blend** | Weighted average of several models' predictions. |
| **GNN / MPNN** | Graph neural network. Atoms pass messages to bonded neighbours, repeatedly. |
| **Encoder** | The part of a network that turns input into a representation. Ours is shared across six properties. |
| **Head** | A small output layer on top of a shared encoder, one per property. |
| **Multi-task learning** | One model trained on several related tasks so they share what they learn. |
| **Transfer** | Knowledge learned on data-rich tasks improving a data-poor one. |
| **Pretraining** | Training on an unlabelled task first, hoping the representation transfers. Ours didn't. |
| **Band gap** | Energy needed to free an electron. Determines insulating vs semiconducting. |
| **Frontier orbitals** | The outermost electron energy levels. Six of our seven properties depend on them. |
| **Determinism** | Same input, same code, same output every time. Required: the hosts re-run our notebook. |

---

## 10. The constraints we worked under

Worth knowing, because they explain choices that otherwise look odd.

- **Everything must run inside one Kaggle notebook**, end to end. No uploading a
  model trained elsewhere.
- **No external data and no pretrained weights.** This bans every off-the-shelf
  chemistry transformer. The GNN had to be built and trained from scratch.
- **No GPU**, so ~4h45m on CPU.
- **The hosts re-run the notebook after the deadline.** If it doesn't reproduce,
  the submission is void. So it has to be deterministic — which took real work,
  because the parallel operation at the heart of message passing sums numbers in
  nondeterministic order. Two identical runs differed by 0.18 per prediction
  until we forced single-threading. Kaggle then reproduced our local score within
  0.0035 every time.
- Three submissions a day, two final selections.

---

## 11. Where everything lives

| What | Where |
|---|---|
| Code | `github.com/Sahilo6/amalgam-polymer` |
| Notebook | `kaggle.com/code/sahilsadhwani25/neonnull-round-3` |
| Report (5pp) | `report/r3/NeonNull_R3_Report.pdf` |
| Deck (12 slides) | `deck/NeonNull_Amalgam_Finale.pptx` |
| Two-person script | `deck/SCRIPT_TWO_PERSON.md` |
| Q&A prep + run sheet | `deck/QA_AND_RUNSHEET.md` |
| The insight | `src/mtgnn.py` (shared encoder), `src/polymer.py` (backbone features) |
| The invariance test | `src/invariance.py` |

---

## 12. If you remember one thing

> **We measured which properties belong together before choosing a model. Six of
> the seven are one physical quantity viewed six ways, so they share an encoder.
> The same technique lost 0.023 in Round 2, where the two properties shared no
> physics.**

Almost any question can be answered by walking back to that sentence.
