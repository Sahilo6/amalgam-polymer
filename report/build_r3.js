const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, LevelFormat, ExternalHyperlink } = require("docx");

const FIG = (n) => fs.readFileSync(`report/r3/${n}`);
const INK = "0B0B0B", MUTED = "52514E", ACCENT = "EB6834", LIGHT = "F3F2EE";
const t = (x, o = {}) => new TextRun({ text: x, size: 20, color: INK, ...o });
const b = (x) => t(x, { bold: true });
const i = (x) => t(x, { italics: true });
const p = (x, o = {}) => new Paragraph({ spacing: { after: o.after ?? 80, line: 264 },
  children: Array.isArray(x) ? x : [t(x)] });
const h1 = (x) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 190, after: 55 },
  children: [new TextRun({ text: x, size: 26, bold: true, color: INK })] });
const h2 = (x) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 45 },
  children: [new TextRun({ text: x, size: 22, bold: true, color: ACCENT })] });
const bl = (x) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 45, line: 252 },
  children: Array.isArray(x) ? x : [t(x)] });
const fig = (n, w, ar, cap) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 70, after: 35 },
    children: [new ImageRun({ type: "png", data: FIG(n), transformation: { width: w * 96, height: (w / ar) * 96 } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [new TextRun({ text: cap, size: 17, italics: true, color: MUTED })] })];
const cell = (x, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" } : undefined,
  margins: { top: 45, bottom: 45, left: 85, right: 85 },
  children: [new Paragraph({ spacing: { after: 0 }, alignment: o.align,
    children: [new TextRun({ text: x, size: 17, bold: !!o.head, color: INK })] })] });
const table = (ws, rows, leftAlign = false) => new Table({
  width: { size: ws.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: ws,
  borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "C9C8C2" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9C8C2" },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E3E2DC" }, insideVertical: { style: BorderStyle.NONE } },
  rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) =>
    cell(c, ws[ci], { head: ri === 0, align: (!leftAlign && ci > 0 && ri > 0) ? AlignmentType.RIGHT : undefined })) })) });

const c = [];
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 35 },
  children: [new TextRun({ text: "Amalgam Hackathon — Round 3 Report", size: 22, color: MUTED })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 55 },
  children: [new TextRun({ text: "One Encoder for Shared Physics", size: 38, bold: true, color: INK })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
  children: [new TextRun({ text: "Multi-task learning across correlated polymer properties, with measured representation invariance", size: 21, color: MUTED })] }));
c.push(table([2300, 7000], [["Field", "Value"],
  ["Project Title", "One Encoder for Shared Physics: Multi-Task Polymer Property Prediction"],
  ["Team Name", "NeonNull"],
  ["Primary Domain", "Materials Science (polymer informatics)"],
  ["Round 3 targets", "Tg, Egc, Egb, Ei, Eea, EPS, Nc — mean R² across all seven"]], true));

c.push(h1("1. Proposed Strategy and Technical Novelty"));
c.push(p([b("The central observation: six of the seven targets are the same physics measured six ways, and five of them are starved of data.")]));
c.push(p("Round 3 keeps the long format of Round 2 but adds five properties with roughly 220 training rows each. Because the metric is an unweighted mean of seven per-target R² values, those five carry 5/7 of the score while contributing 14% of the data. Fitting them independently is hopeless: a per-target LightGBM reaches R² 0.75 on the dielectric constant. Our approach is built entirely around that imbalance."));

c.push(h2("AI methodology"));
c.push(p("Supervised deep learning and gradient boosting in combination, trained from scratch inside a single Kaggle notebook: the rules prohibit pretrained models and external data. Two views of each molecule feed a per-target blend. The first is gradient-boosted trees (LightGBM) over RDKit descriptors, Morgan, MACCS and Avalon fingerprints plus our polymer backbone block. The second is a message-passing graph neural network on the molecular graph, in two forms: one per target, and one multi-task encoder shared across the six electronic properties with a linear head each. Blend weights are fitted per target on out-of-fold predictions by constrained least squares. Self-supervised pretraining on the auxiliary PI1M corpus was also evaluated, and rejected on measurement (Section 3)."));

c.push(h2("Novel contribution: a physically motivated multi-task cluster"));
c.push(p("Before training anything we measured how the targets relate on the 415 molecules carrying more than one label. The structure is unambiguous: chain and bulk band gap correlate +0.93, dielectric constant and refractive index +0.92 (the textbook n² ≈ ε relation), and chain band gap against refractive index −0.85. Every one of these is a frontier-orbital quantity. Glass transition, a thermal property, shares no molecule with any of them and belongs to a different physical family."));
c.push(...fig("fig1_target_correlation.png", 4.5, 1.22, "Figure 1. Target correlations on molecules carrying both labels. The six electronic/optical properties form one cluster; Tg stands apart."));
c.push(p([t("That measurement, not a hyperparameter search, dictated the architecture: "), b("one shared graph encoder across the six electronic targets with a separate output head per target"), t(", and Tg trained alone. Each row contributes loss only through its own head, so the 220-row targets inherit a representation learned from all 3,266 electronic rows.")]));
c.push(p([i("We had direct evidence this could fail. In Round 2 we tested multi-task learning across Tg and Egc and it lost 0.023 R² — those two properties share no physics, and forcing one encoder to serve both diluted each. The difference here is not the technique but whether the targets are physically related, which is exactly what Figure 1 establishes.")]));
c.push(...fig("fig6_architecture.png", 5.6, 1.94, "Figure 2. Pipeline. Orange marks the polymer-specific features carried over from Round 2; blue marks the graph models. Blend weights are fitted per target on out-of-fold predictions."));

c.push(h2("Featurization"));
c.push(p([t("The polymer-aware block carried over from Round 2 is central: the repeat unit's backbone is the shortest path between its two attachment points, ring-complete so a phenylene in the chain counts as chain, and 18 scalars plus a full descriptor set are computed on that extracted chain alone. Backbone length alone correlates +0.47 with Tg. "), b("Its effect on the small targets is visible in the blend: "), t("the multi-task encoder carries 0.70 to 0.91 of the weight on all five, and only 0.29 on Tg — the shared representation does the work precisely where the data is thin.")]));

c.push(h2("Polymer invariance, measured rather than assumed"));
c.push(p("A model that gives different answers for two spellings of the same molecule cannot be trusted by a materials scientist. We tested this rather than asserting it: for 120 test molecules per target we generated five randomised SMILES each and measured the spread in predictions."));
c.push(p([b("The initial model was not invariant. "), t("Predictions moved by up to 3.02% of a target standard deviation on Tg and 1.22% on Egc. Tracing it found three causes with one root: atom ordering. Tied shortest paths around a ring resolve differently, Gasteiger charges on dummy atoms converge differently, and RDKit's Ipc descriptor exceeds 10¹³ where floating-point rounding is not order-stable. Because 65% of the supplied SMILES are not in canonical form, this was affecting real predictions, not a hypothetical.")]));
c.push(p([b("The fix is one line and a dropped descriptor: "), t("re-parse every molecule from its canonical SMILES before featurizing, and exclude Ipc. After it, the maximum deviation across equivalent SMILES is "), b("exactly zero"), t(" at both the feature level (all 14,221 features) and the prediction level. The graph models were already invariant to 6×10⁻⁷ eV, floating-point noise. The fix cost 0.0001 in score.")]));
c.push(...fig("fig5_invariance.png", 5.2, 2.3, "Figure 3. Maximum prediction change across five randomised SMILES of the same molecule, before and after canonicalisation."));

c.push(h1("2. Preliminary Salient Results"));
c.push(h2("Key EDA findings"));
c.push(bl([b("Severe target imbalance. "), t("Tg has 4,143 rows and Egc 2,028; the other five have 221 to 337 each. Each is 1/7 of the metric regardless.")]));
c.push(bl([b("The targets cluster by physics "), t("(Figure 1), and only 415 of 6,565 molecules carry more than one label, so the transfer has to happen through the representation rather than through shared rows.")]));
c.push(bl([b("Clean polymer SMILES. "), t("All 10,605 unique structures parse, and every one carries exactly two attachment points, which is what makes backbone extraction well defined.")]));
c.push(bl([b("Most SMILES are non-canonical (65%), "), t("which is why the invariance defect above had real effect.")]));
c.push(...fig("fig2_row_imbalance.png", 5.2, 2.25, "Figure 4. Training rows per target. Five of seven are data-starved."));

c.push(h2("Results"));
c.push(table([1250, 1050, 1250, 1250, 1400, 1400, 1700], [
  ["Target", "rows", "LightGBM", "GNN", "multi-task", "blend", "multi-task wt"],
  ["EGC", "2,028", "0.908", "0.913", "0.930", "0.933", "0.71"],
  ["EGB", "337", "0.902", "0.940", "0.960", "0.961", "0.82"],
  ["EI", "222", "0.798", "0.826", "0.873", "0.873", "0.88"],
  ["EEA", "221", "0.891", "0.930", "0.936", "0.941", "0.71"],
  ["EPS", "229", "0.746", "0.805", "0.858", "0.855", "0.70"],
  ["NC", "229", "0.810", "0.860", "0.918", "0.916", "0.91"],
  ["TG", "4,143", "0.913", "0.905", "—", "0.920", "—"]]));
c.push(new Paragraph({ spacing: { after: 90 }, children: [] }));
c.push(p([b("Official metric: 0.9139"), t(" (mean of seven per-target R², out-of-fold, blend weights fitted by nested cross-validation so they never see the rows they score). The progression was 0.8888 for per-target models, 0.9038 adding the multi-task encoder, 0.9085 with seed averaging, and 0.9139 on moving the shared encoder from five folds to ten. The organizer's Ridge baseline scores 0.537.")]));
c.push(...fig("fig3_multitask_gain.png", 5.3, 2.05, "Figure 5. Effect of adding the shared encoder, per target. Gains concentrate where rows are scarce: +0.056 on refractive index (229 rows) against +0.010 on chain band gap (2,028 rows)."));
c.push(...fig("fig4_pred_vs_true.png", 6.0, 2.03, "Figure 6. Out-of-fold predictions against ground truth for all seven targets."));

c.push(h1("3. Technical Challenges and Pivots"));
c.push(bl([b("220-row targets. "), t("The central difficulty. Ten-fold cross-validation would leave 22-row validation folds, so we use five folds throughout and rely on the shared encoder rather than per-target capacity. Seed averaging matters far more than it did for the tree models: averaging six initialisations of the multi-task network is worth +0.0047, while seed averaging gradient-boosted trees measured exactly zero in Round 2.")]));
c.push(bl([b("Pretraining on the auxiliary data did not work. "), t("We pretrained the encoder on 100,000 PI1M repeat units with a masked-atom objective, reaching 99% reconstruction accuracy, then fine-tuned. It scored −0.0038 against the same folds. The corpus is in-domain and the task was learned; the representation simply did not transfer to property regression better than supervised training on 3,266 labelled rows. We report it as a negative result rather than omitting it.")]));
c.push(bl([b("The invariance defect was invisible until measured. "), t("Nothing in cross-validation reveals it, because the training and test SMILES are each written one particular way. It only appears when you deliberately re-spell the molecules, which is why we built the test.")]));
c.push(bl([b("Reproducibility. "), t("The submitted notebook is single-threaded with per-fold seeds; parallel index_add_ in the message-passing step is otherwise non-deterministic. Kaggle reproduced our local out-of-fold score within 0.0035 on every run.")]));

c.push(h1("4. Final Sprint Roadmap"));
c.push(h2("Immediate and refinement"));
c.push(p([t("The model is frozen and reproducible. The last change we made was the largest single gain of the round and is worth stating precisely: moving the shared encoder from five folds to ten, "), b("at identical compute"), t(" (thirty fold-runs either way, three seeds instead of six), was worth "), b("+0.0058"), t(" on the official metric. Each model then trains on 90% of the 3,266 electronic rows rather than 80%, and the gain lands almost entirely on the data-starved targets: +0.018 on dielectric constant, +0.013 on refractive index, +0.009 on ionisation energy. Seed averaging by contrast has reached clear diminishing returns (+0.0036 for the second and third seed, +0.0012 for the fourth through sixth), so reallocating that same compute to folds was strictly better. With more time the obvious next step is more folds still, since the trend has not turned.")]));

c.push(h2("Explainability: electronic applications"));
c.push(p("For a newly proposed polymer the model's reasoning about electronic behaviour is legible through the backbone block, which is where its feature importance concentrates. Band gap falls as conjugation runs along the chain lengthen, as backbone aromatic atom count rises, and as the attachment atoms become sp² or aromatic, which lets conjugation propagate across repeat units rather than terminating at the unit boundary. We quantified this directly: the conjugated-bond fraction of an explicitly constructed dimer correlates −0.76 with chain band gap. The motifs the model associates with low, tunable gaps are fused and linked aromatic backbones — phenylene, fluorene and thiophene-like units — with low side-chain mass fraction; sp³ spacers and bulky pendant groups raise the gap. Because ionisation energy and electron affinity share the encoder, the same structural signals explain where the frontier orbitals sit, not just their separation."));
c.push(h2("Explainability: optical applications"));
c.push(p("Two of the seven targets are directly optical, and this is where the shared encoder pays off most visibly: refractive index gains +0.056 R² from multi-task training, the largest gain of any target. The model's optical reasoning follows the same structural axis, because refractive index and dielectric constant correlate +0.92 with each other and −0.85 with band gap: polarisable, conjugated, aromatic-rich backbones raise both the refractive index and the dielectric constant while narrowing the gap, and the absorption edge follows at roughly 1240/Egc nm. Backbone rigidity and planarity — high ring-atom count, low rotatable-bond count — concentrate that effect, while side-chain bulk dilutes it and acts as a blue-shifting, solubilising trade-off. For a candidate polymer the model can therefore report predicted refractive index, dielectric constant and absorption edge together with the backbone motifs driving all three, which is the combination an optical-materials decision actually needs."));
c.push(h2("Final output"));
c.push(p("A single self-contained Kaggle notebook: data loading, canonicalisation, featurization, backbone extraction, per-target LightGBM and GNN, the multi-task encoder, blending and submission, running end to end on Kaggle CPU with no accelerator and no uploaded artifacts."));

c.push(h1("Appendix"));
c.push(p([b("Notebook: "), new ExternalHyperlink({ children: [new TextRun({ text: "kaggle.com/code/sahilsadhwani25/neonnull-round-3", size: 20, color: "2A78D6", underline: {} })], link: "https://www.kaggle.com/code/sahilsadhwani25/neonnull-round-3" }),
  t("   ·   "), b("Code: "), new ExternalHyperlink({ children: [new TextRun({ text: "github.com/Sahilo6/amalgam-polymer", size: 20, color: "2A78D6", underline: {} })], link: "https://github.com/Sahilo6/amalgam-polymer" })]));
c.push(p([b("Model particulars: "), t("LightGBM 4,000 trees with early stopping, 5 folds. Message-passing GNN 383k parameters, hidden 128, three layers with GRU updates, mean+max pooling. Multi-task variant 387k parameters, shared encoder with six linear heads, ten folds × three seeds. Inference under 1 ms per molecule.")]));
c.push(p([b("References: "), t("RDKit (rdkit.org); Gilmer et al., Neural Message Passing for Quantum Chemistry, ICML 2017; Ke et al., LightGBM, NeurIPS 2017; Gedeck et al., Avalon fingerprints, JCIM 2006; Ma & Luo, PI1M: A Benchmark Database for Polymer Informatics, JCIM 2020.")]));
c.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
c.push(p([b("Every Round 3 experiment, with its measured effect on the official metric:")]));
c.push(table([4400, 1900, 3000], [["Idea", "Effect", "Verdict"],
  ["Multi-task GNN over the electronic cluster", "+0.0157", "shipped"],
  ["Shared encoder at ten folds, not five", "+0.0058", "shipped: largest single gain"],
  ["Multi-task seed averaging (1→3)", "+0.0036", "shipped"],
  ["Seed averaging beyond three", "+0.0012", "superseded by folds at equal cost"],
  ["Canonical atom order (invariance fix)", "−0.0001", "shipped: invariance 3.0% → 0"],
  ["Multi-task LightGBM", "+0.0004", "too weak to justify"],
  ["PI1M masked-atom pretraining", "−0.0038", "rejected"],
  ["Multi-task including Tg (Round 2 test)", "−0.0230", "rejected: no shared physics"]]));

const doc = new Document({ creator: "NeonNull", title: "One Encoder for Shared Physics",
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 340, hanging: 230 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 20 } } } },
  sections: [{ properties: { page: { margin: { top: 900, bottom: 900, left: 1060, right: 1060 } } }, children: c }] });
Packer.toBuffer(doc).then((x) => { fs.writeFileSync("report/r3/NeonNull_R3_Report.docx", x); console.log("written"); });
