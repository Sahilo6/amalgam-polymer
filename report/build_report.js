const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, LevelFormat, PageBreak, ExternalHyperlink,
} = require("docx");

const FIG = (n) => fs.readFileSync(`report/figures/${n}`);
const INK = "0B0B0B", MUTED = "52514E", ACCENT = "EB6834", LIGHT = "F3F2EE";

// ---- helpers ---------------------------------------------------------------
const p = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 100, line: 276 },
  alignment: opts.align,
  children: Array.isArray(text) ? text : [new TextRun({ text, size: 20, color: INK, ...opts.run })],
});
const t = (text, o = {}) => new TextRun({ text, size: 20, color: INK, ...o });
const b = (text) => t(text, { bold: true });
const i = (text) => t(text, { italics: true });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 80 }, children: [new TextRun({ text, size: 26, bold: true, color: INK })] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 60 }, children: [new TextRun({ text, size: 22, bold: true, color: ACCENT })] });
const bullet = (children) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 },
  children: Array.isArray(children) ? children : [t(children)] });
const fig = (name, widthIn, aspect, caption) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: FIG(name), transformation: { width: widthIn * 96, height: (widthIn / aspect) * 96 } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 }, children: [new TextRun({ text: caption, size: 17, italics: true, color: MUTED })] }),
];
const cell = (text, w, o = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" } : undefined,
  margins: { top: 50, bottom: 50, left: 90, right: 90 },
  children: [new Paragraph({ spacing: { after: 0 }, alignment: o.align,
    children: [new TextRun({ text, size: 17, bold: !!o.head, color: INK })] })],
});
const table = (widths, rows, leftAlign=false) => new Table({
  width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
  borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "C9C8C2" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9C8C2" },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E3E2DC" }, insideVertical: { style: BorderStyle.NONE } },
  rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, widths[ci], { head: ri === 0, align: (!leftAlign && ci > 0 && ri > 0) ? AlignmentType.RIGHT : undefined })) })),
});
const gap = (after = 120) => new Paragraph({ spacing: { after }, children: [] });

// ---- content ---------------------------------------------------------------
const children = [];

// header block
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Amalgam Hackathon Report", size: 22, color: MUTED })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Chains, Not Molecules", size: 40, bold: true, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "Backbone-aware featurization and graph learning for polymer Tg and band gap prediction", size: 22, color: MUTED })] }));
children.push(table([2400, 6900], [
  ["Field", "Value"],
  ["Project Title", "Chains, Not Molecules: Backbone-Aware Learning for Polymer Property Prediction"],
  ["Team Name", "NeonNull"],
  ["Primary Domain", "Materials Science (polymer informatics)"],
  ["Competition", "Amalgam 2026, Round 1: Polymer Property Prediction (Kaggle)"],
  ["Public leaderboard", "0.911 (2nd of the field; organizer baseline 0.537)"],
], true));
children.push(gap(160));

// ---- 1 ---------------------------------------------------------------------
children.push(h1("1. Proposed Strategy and Technical Novelty"));
children.push(p([
  b("The core idea in one sentence: "),
  t("every row in this dataset is a polymer repeat unit, a chain segment with two attachment points, and the single most valuable thing we did was stop treating it as an ordinary molecule."),
]));
children.push(p("Off-the-shelf polymer ML computes molecular descriptors and fingerprints on the repeat-unit SMILES and hands them to a regressor. That was our starting point too, and it plateaued at 0.896. Every gain after that came from changing what the model could see, never from changing the model. Of nineteen hypotheses we tested, three worked, and all three were representational."));

children.push(h2("AI methodology"));
children.push(p("A two-view ensemble, trained entirely from scratch inside a single Kaggle notebook (the rules prohibited pretrained models and external data). View one is gradient boosting (LightGBM, 10-fold) over RDKit descriptors, Morgan, MACCS and Avalon fingerprints, plus our polymer-aware features. View two is a message-passing graph neural network that reads the molecular graph directly. Tg and Egc are disjoint row sets, so each target gets its own models and its own blend weights, fitted on out-of-fold predictions."));
children.push(...fig("fig6_architecture.png", 5.2, 1.99, "Figure 1. Two views of every molecule, blended per target. Orange marks the novel feature block; blue marks the from-scratch GNN."));

children.push(h2("Novel contribution: chain decomposition"));
children.push(p("Each repeat unit carries exactly two dummy atoms (*) marking where the chain continues. The shortest path between them is the polymer backbone; everything else is a pendant side group. We extract that backbone explicitly and describe it as an object in its own right, which generic featurization never does."));
children.push(p([
  t("Two details make this work. First, "),
  b("ring completion"),
  t(": if the backbone passes through a ring, the whole ring is backbone. This is chemically right (a phenylene in the chain is a chain unit, not a side group) and it is also what makes extraction robust: path-only extraction leaves broken aromatic fragments and fails to sanitize on 59% of molecules; ring-complete extraction succeeds on 98%. Second, "),
  b("18 hand-built scalars"),
  t(" are computed on the extracted chain: length, rotatable bonds, ring and aromatic atom counts, heteroatoms, conjugated bond count and longest conjugation run, sp3 fraction, branch points, and side-chain count, maximum size and mass fraction. A second block computes the full RDKit descriptor set on the backbone fragment alone."),
]));
children.push(...fig("fig5_backbone_extraction.png", 4.2, 1.51, "Figure 2. Backbone extraction on a training molecule. Orange is the ring-complete chain (30 of 36 atoms); the nitro and methoxy groups are side chains. Backbone length alone correlates +0.47 with Tg."));

children.push(h2("How featurization moved the score"));
children.push(p("We measured every block by greedy forward selection on the same 3-fold split, keeping only what improved the official metric once earlier blocks were present. The 18 backbone scalars gave +0.0062, more than the 1,024-bit Avalon fingerprint (+0.0056) and more than the four other fingerprint families combined, which proved redundant once Avalon was in. The physics agrees with the numbers: glass transition is governed by chain stiffness, and chain stiffness is exactly what the backbone block describes. The gain landed almost entirely on Tg (0.8845 to 0.8935)."));
children.push(...fig("fig2_feature_ablation.png", 5.4, 2.11, "Figure 3. Marginal gain of each feature block over the baseline (descriptors + Morgan + MACCS). The two polymer-aware blocks are the largest and third-largest contributors."));
children.push(p([
  t("The GNN is the second representational win. Its 383k parameters read the graph directly (22 atom features, three message-passing layers with GRU updates, mean and max pooling), so its errors correlate only 0.74 to 0.78 with the tree model, against 0.999 between the tree models themselves. It earns 0.41 of the blend weight on Tg and 0.61 on Egc, and adds +0.0053 alone and a further +0.0037 with 10-fold training and two-seed averaging. "),
  i("We tested a polymer-aware variant of the GNN with backbone flags and bond features; it added nothing, because three rounds of message passing can already infer chain position from topology. The trees needed the chain spelled out because they cannot traverse a graph."),
]));

children.push(h2("SMILES invariance"));
children.push(p("Every feature in the final model is computed on the RDKit molecular graph, which is canonicalised on parse. Descriptors, fingerprints, the backbone extraction and the GNN's adjacency all depend only on that graph, so any valid SMILES for the same repeat unit yields identical inputs and identical predictions. We tested the one representation that is not invariant, a character-level CNN over the SMILES string, and it added only +0.0005 to the blend; it was not shipped."));

// ---- 2 ---------------------------------------------------------------------
children.push(h1("2. Preliminary Salient Results"));
children.push(h2("Key EDA findings"));
children.push(bullet([b("Long format. "), t("6,171 training rows are 4,143 Tg and 2,028 Egc measurements on 6,158 distinct polymers; only 7 molecules carry both labels. The targets differ in scale by ~70x (Tg: 140 ± 109 °C; Egc: 4.53 ± 1.56 eV) and the metric averages per-target R², so the two properties carry equal weight and separate models per target are the natural design.")]));
children.push(bullet([b("Clean polymer SMILES. "), t("All 10,264 unique SMILES parse under RDKit and every one carries exactly two attachment points, which is what makes backbone extraction well defined.")]));
children.push(bullet([b("No distribution shift. "), t("A classifier trained to tell train rows from test rows scores AUC 0.48 to 0.49, i.e. chance. Local cross-validation is therefore a faithful proxy for the hidden set.")]));
children.push(bullet([b("The error lives in a tail. "), t("The worst 5% of rows carry 52% (Tg) and 46% (Egc) of total squared error. For Egc those rows have a clear chemical signature: short, rigid, conjugated backbones (mean length 7.4 vs 13.4 atoms, rotatable bonds 2.1 vs 6.5) with bulky side chains (mass fraction 0.61 vs 0.44). These are the molecules whose band gap is set by conjugation that extends across repeat units, which a single unit under-represents.")]));

children.push(h2("Results"));
children.push(table([3300, 1500, 1500, 1500, 1500], [
  ["Model (10-fold OOF)", "R² Tg", "R² Egc", "Official", "Notes"],
  ["Organizer baseline (Ridge)", "0.464", "0.611", "0.537", "provided"],
  ["LightGBM, generic features", "0.890", "0.902", "0.896", "day 1"],
  ["+ Avalon + backbone scalars", "0.904", "0.913", "0.908", ""],
  ["+ backbone descriptors", "0.911", "0.916", "0.914", ""],
  ["GNN alone (10-fold, 2 seeds)", "0.910", "0.925", "0.918", "383k params"],
  ["LightGBM + GNN blend (final)", "0.918", "0.928", "0.925", "shipped"],
]));
children.push(gap(80));
children.push(p("Blend numbers are nested: weights are fitted on folds that never see the rows they score, so they are not inflated by the weight fit. The final notebook reproduced its local score on Kaggle within 0.0012 on every one of seven runs."));
children.push(...fig("fig1_pred_vs_true.png", 5.2, 2.14, "Figure 4. Out-of-fold predictions of the final blend against ground truth for both targets."));
children.push(...fig("fig3_oof_vs_lb.png", 5.0, 2.11, "Figure 5. Six submissions: the local estimate and the public leaderboard moved together, with a stable offset of 0.0139 ± 0.0012. We optimised the local estimate throughout and never the board, which is a 37% sample with noise of about ±0.006."));

// ---- 3 ---------------------------------------------------------------------
children.push(h1("3. Technical Challenges and Pivots"));
children.push(h2("Roadblocks"));
children.push(bullet([b("The metric was not published in a machine-readable place. "), t("We built the pipeline assuming pooled RMSE; the first submission scored 0.888 and revealed it was mean per-target R². Because every model was already per-target, the switch cost a reporting change and no retraining. That accidental robustness became a design rule: never let anything trade one target against the other.")]));
children.push(bullet([b("Notebook-only execution. "), t("RDKit is absent from the Kaggle image, internet is gated on account verification, and the data mounts under a different path than documented. Three multi-hour runs died on these before we built a three-minute pre-flight kernel and a five-gate deployment script (staleness check, pre-flight, push, out-of-fold score match, file validation). No run failed after that.")]));
children.push(bullet([b("Reproducibility. "), t("The hosts re-run the pinned notebook and void any submission that does not reproduce. Our GNN initially differed by up to 0.18 per prediction between identical runs, traced to parallel index_add_ atomics summing in varying order. Pinning to a single thread made it bit-exact at no cost, because the graphs are small.")]));

children.push(h2("Pivots"));
children.push(bullet([b("From more models to different views. "), t("Adding XGBoost, CatBoost, ExtraTrees, SVR and kernel ridge to the blend measured +0.0000 each: they all read the same features and made the same mistakes. CatBoost cost 91% of a five-hour notebook for +0.0006 and was dropped. The gain came from a model that reads a different representation.")]));
children.push(bullet([b("From assumed gains to measured ones. "), t("A conjugation feature correlated -0.76 with Egc and added +0.0002 to the model, because the information was already present. Dimer descriptors, seed averaging on trees, multi-task learning across targets (-0.023 on Egc), pseudo-labelling, feature pruning and a SMILES CNN all measured at or below zero. We called the ceiling twice and were wrong both times; the GNN and 10-fold training came after. The lesson we kept: measure with nested validation, promote only on that number.")]));

// ---- 4 ---------------------------------------------------------------------
children.push(h1("4. Final Sprint Roadmap"));
children.push(h2("Immediate tasks and refinement"));
children.push(p("The model is final and reproducible; remaining effort goes to the presentation. With more compute we would pursue the two levers that showed signal but did not fit Kaggle's nine-hour limit: wider and deeper GNNs (+0.0018 solo, but 82 CPU-hours at the size tested) and a stronger sequence model as a genuinely third view (our CNN decorrelated well, error correlation 0.64 to 0.74, but was too weak alone to carry blend weight)."));

children.push(h2("Explainability: electronic applications"));
children.push(p("For a newly proposed polymer, the model's chemical reading of band gap comes from the backbone block. Egc falls as conjugation runs along the chain lengthen (longest backbone conjugation run, conjugated-bond fraction), as aromatic atoms on the backbone increase, and as the attachment atoms are sp2 or aromatic, which lets conjugation propagate across repeat units. Our dimer analysis quantifies this: the conjugated fraction of a two-unit chain correlates -0.76 with Egc. Motifs the model associates with low, tunable gaps are fused and linked aromatic backbones (phenylene, fluorene, thiophene-like units) with low side-chain mass fraction; bulky pendants and sp3 spacers in the chain raise the gap. LightGBM feature importances on the backbone block, and per-atom GNN pooling contributions, give these as a ranked list for any input SMILES."));
children.push(h2("Explainability: optical applications"));
children.push(p("The same quantity sets optical behaviour: the absorption edge sits near 1240/Egc nm, so the motifs that narrow the electronic gap red-shift absorption and emission. The model's structural signals for optical tuning are therefore backbone rigidity and planarity (high backbone ring-atom count, low rotatable-bond count) and extended conjugation, against which side-chain bulk acts as a blue-shifting, solubilising trade-off. For a candidate polymer the model can report predicted Egc, the implied absorption edge, and which backbone motifs drive it, which is the information a synthesis decision actually needs."));

children.push(h2("Final output"));
children.push(p("A single self-contained Kaggle notebook (data loading, featurization, backbone extraction, LightGBM and GNN training, blending and submission, 6h 04m on Kaggle CPU, no accelerator) plus the local research codebase with the full ablation record. Both are linked below."));

// ---- appendix --------------------------------------------------------------
children.push(h1("Appendix"));
children.push(p([b("Kaggle notebook (final, pinned): "), new ExternalHyperlink({ children: [new TextRun({ text: "kaggle.com/code/sahilsadhwani25/amalgam-2026-fast", size: 20, color: "2A78D6", underline: {} })], link: "https://www.kaggle.com/code/sahilsadhwani25/amalgam-2026-fast" })]));
children.push(p([b("Codebase: "), new ExternalHyperlink({ children: [new TextRun({ text: "github.com/Sahilo6/amalgam-polymer", size: 20, color: "2A78D6", underline: {} })], link: "https://github.com/Sahilo6/amalgam-polymer" })]));
children.push(p([b("Model particulars: "), t("LightGBM 4,000 trees max with early stopping, 10 folds; MPNN 383,489 parameters, hidden 128, 3 layers, 120 epochs max with early stopping, 10 folds × 2 seeds, single-threaded for determinism. Inference is under 1 ms per molecule.")]));
children.push(p([b("References: "), t("RDKit (rdkit.org); Gilmer et al., Neural Message Passing for Quantum Chemistry, ICML 2017; Gedeck et al., Avalon fingerprints, J. Chem. Inf. Model. 2006; Ke et al., LightGBM, NeurIPS 2017.")]));
children.push(gap(60));
children.push(p([b("Every hypothesis we tested, "), t("with its measured effect on the official metric (nested where a blend is involved):")]));
children.push(table([4300, 2300, 2700], [
  ["Idea", "Effect", "Verdict"],
  ["Polymer backbone scalars (18)", "+0.0062", "kept"],
  ["Avalon fingerprint", "+0.0056", "kept"],
  ["Message-passing GNN in blend", "+0.0053", "kept"],
  ["Backbone descriptors", "+0.0025", "kept"],
  ["GNN: 10-fold + 2 seeds", "+0.0037", "kept"],
  ["GNN seed averaging (1 to 3 seeds)", "+0.0020", "kept"],
  ["10-fold instead of 5-fold (trees)", "+0.0019", "kept"],
  ["Morgan r=3, RDKit FP, atom-pair, torsion", "≤ +0.0005", "redundant with Avalon"],
  ["Hyperparameter search, 20 trials", "+0.005 Tg, 0 Egc", "dropped"],
  ["Conjugation features (corr -0.76 with Egc)", "+0.0002", "already encoded"],
  ["Polymer-aware GNN (backbone + bond feats)", "+0.0000", "GNN infers it"],
  ["Extra model families (ET, SVR, KRR)", "+0.0000", "same view"],
  ["Tree seed averaging", "+0.0000", "no variance to remove"],
  ["Blend weighting (NNLS, stacking)", "+0.0000", "at optimum"],
  ["SMILES character CNN, 3 seeds", "+0.0005", "too weak alone"],
  ["Pseudo-labelling test rows", "-0.0002 to -0.0023", "reinforces errors"],
  ["Feature pruning (top 200 to 2000)", "0 to -0.0103", "features all carry signal"],
  ["Multi-task (share Tg and Egc rows)", "-0.008 Tg, -0.023 Egc", "different physics"],
  ["Dimer descriptors; bare / dimer backbone", "-0.001 to -0.0014", "linear rescalings"],
]));

// ---- document --------------------------------------------------------------
const doc = new Document({
  creator: "NeonNull", title: "Chains, Not Molecules",
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 20 } } } },
  sections: [{ properties: { page: { margin: { top: 1080, bottom: 1080, left: 1150, right: 1150 } } }, children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report/NeonNull_Amalgam_Report.docx", buf); console.log("written"); });
