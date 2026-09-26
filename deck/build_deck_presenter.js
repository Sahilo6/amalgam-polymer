const pptx = require("pptxgenjs");
const fs = require("fs");
const P = new pptx();
P.layout = "LAYOUT_WIDE";                       // 13.3 x 7.5
P.author = "NeonNull"; P.title = "Amalgam Round 3 — NeonNull";

const NAVY="21295C", BLUE="2A78D6", ORANGE="EB6834", INK="151515", MUTED="5B5B5B", LIGHT="F4F6FA", WHITE="FFFFFF";
const HEAD="Cambria", BODY="Calibri";
const FIG=(n)=>`report/r3/${n}`;

const title = (s, txt, sub) => {
  s.addText(txt, { x:0.6, y:0.32, w:12.1, h:0.72, fontSize:32, bold:true, color:INK, fontFace:HEAD, isTextBox:true, margin:0 });
  if (sub) s.addText(sub, { x:0.6, y:1.02, w:12.1, h:0.42, fontSize:15, color:MUTED, fontFace:BODY, isTextBox:true, margin:0 });
};
const dark = () => { const s=P.addSlide(); s.background={color:NAVY}; return s; };
const light = () => { const s=P.addSlide(); s.background={color:WHITE}; return s; };
const card = (s,x,y,w,h,fill) => s.addShape(P.ShapeType.roundRect,{x,y,w,h,fill:{color:fill||LIGHT},rectRadius:0.08,line:{color:fill||LIGHT}});
const stat = (s,x,y,w,big,lab,col) => {
  s.addText(big,{x,y,w,h:0.78,fontSize:44,bold:true,color:col||BLUE,fontFace:HEAD,align:"center",isTextBox:true,margin:0});
  s.addText(lab,{x,y:y+0.76,w,h:0.62,fontSize:12,color:MUTED,fontFace:BODY,align:"center",isTextBox:true,margin:0});
};

/* 1 — cover */
let s = dark();
s.addText("One Encoder for Shared Physics",{x:0.9,y:2.15,w:11.5,h:0.95,fontSize:42,bold:true,color:WHITE,fontFace:HEAD,isTextBox:true,margin:0});
s.addText("Multi-task learning across correlated polymer properties, with measured representation invariance",
  {x:0.9,y:3.12,w:11.0,h:0.55,fontSize:17,color:"C7D2F0",fontFace:BODY,isTextBox:true,margin:0});
s.addShape(P.ShapeType.rect,{x:0.9,y:3.95,w:1.5,h:0.035,fill:{color:ORANGE},line:{color:ORANGE}});
s.addText([{text:"Team NeonNull",options:{bold:true,color:WHITE}},{text:"   ·   Sahil Sadhwani   ·   VIT",options:{color:"C7D2F0"}}],
  {x:0.9,y:4.3,w:11,h:0.38,fontSize:15,fontFace:BODY,isTextBox:true,margin:0});
s.addText("Amalgam Hackathon — Round 3 Finale   ·   Polymer Property Prediction   ·   September 2026",
  {x:0.9,y:4.75,w:11,h:0.36,fontSize:12,color:"8FA3D9",fontFace:BODY,isTextBox:true,margin:0});

/* 2 — the problem */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nHOOK. Ask 'can everyone hear me' ONCE, then never again.\n\nGood morning. I'm Sahil, Team NeonNull, joining from Vellore.\n\nSeven polymer properties, one model. The hard part of this problem was never accuracy. It was that FIVE OF THE SEVEN targets have about 220 training rows each, and the score weights all seven equally.");
s = light(); title(s,"The problem is not accuracy. It is imbalance.","Seven targets, one unweighted mean of per-target R². Five targets have ~220 training rows each.");
s.addImage({path:FIG("fig2_row_imbalance.png"),x:0.75,y:1.6,w:7.5,h:3.33});
card(s,8.6,1.7,4.1,1.35);
stat(s,8.6,1.82,4.1,"5 of 7","targets under 500 rows",ORANGE);
card(s,8.6,3.2,4.1,1.35);
stat(s,8.6,3.32,4.1,"71%","of the metric carried by\n14% of the data",ORANGE);
s.addText("A per-target LightGBM reaches only R² 0.75 on dielectric constant. Fitting these independently is hopeless.",
  {x:0.75,y:5.3,w:11.8,h:0.5,fontSize:14,italic:true,color:MUTED,fontFace:BODY,isTextBox:true,margin:0});

/* 3 — the insight */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nHere is the shape of it. 71% of the metric is carried by 14% of the data.\n\nFit those five independently and you are fitting a few hundred rows with a thousand features. Per-target LightGBM reaches R2 0.75 on dielectric constant. Not a model anyone would use.\n\nThe usual answers are regularisation and feature selection. We tried both. Neither moves this, because the problem is not variance in the fit. There is genuinely not enough signal in 200 rows.\n\nThe only way to get more signal is to bring in rows from somewhere else.\n\n[CUT THE REGULARISATION PARA IF RUNNING LONG]");
s = light(); title(s,"We measured the targets before modelling them","On the 415 molecules carrying more than one label, the structure is unambiguous.");
s.addImage({path:FIG("fig1_target_correlation.png"),x:0.75,y:1.5,w:6.2,h:5.06});
const facts=[["+0.93","chain vs bulk band gap"],["+0.92","dielectric constant vs refractive index — the textbook n² ≈ ε"],["−0.85","chain band gap vs refractive index"]];
facts.forEach(([v,t],i)=>{
  const y=1.75+i*1.15; card(s,7.4,y,5.3,0.95);
  s.addText(v,{x:7.6,y:y+0.17,w:1.3,h:0.6,fontSize:26,bold:true,color:BLUE,fontFace:HEAD,isTextBox:true,margin:0});
  s.addText(t,{x:8.95,y:y+0.2,w:3.6,h:0.6,fontSize:12.5,color:INK,fontFace:BODY,isTextBox:true,margin:0});
});
s.addText([{text:"All six are frontier-orbital quantities. ",options:{bold:true}},
  {text:"Glass transition is thermal and shares no molecule with any of them — so it trains alone.",options:{}}],
  {x:7.4,y:5.3,w:5.3,h:0.9,fontSize:13,color:INK,fontFace:BODY,isTextBox:true,margin:0});

/* 4 — architecture */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nKEY SLIDE. SLOW DOWN. Pause after every number.\n\nSo: which somewhere else. We decided to MEASURE that before choosing a model.\n\n415 molecules carry more than one label. That overlap lets you compute target-to-target correlation directly, and the structure is unambiguous.\n\nChain vs bulk band gap: plus 0.93. // Dielectric constant vs refractive index: plus 0.92, the textbook n-squared approx epsilon turning up in the data on its own. // Chain band gap vs refractive index: minus 0.85.\n\nAll six are frontier-orbital quantities. Physically, ONE PROPERTY LOOKED AT SIX WAYS.\n\nGlass transition is the exception. Thermal, shares no molecule with any of the six, has 4,000 rows of its own. So it trains alone.\n\nThat slide decided the architecture, before we ran a single model.");
s = light(); title(s,"Architecture","Tree path, graph path, and one encoder shared across the six electronic targets.");
s.addImage({path:FIG("fig6_architecture.png"),x:0.95,y:1.5,w:11.4,h:11.4/1.99});

/* 5 — what worked */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nTwo paths and one shared encoder.\n\nTree path: RDKit descriptors, fingerprints, and 18 polymer-specific features from the BACKBONE, the chain between the two attachment points.\n\nGraph path: message-passing network trained from scratch. No pretrained weights, the rules don't allow them.\n\nThe shared encoder is the piece that matters. One network reads the molecule for all six electronic targets; six small heads read off the individual properties. Each row trains only its own head, but EVERY ROW TRAINS THE ENCODER. So the 220 refractive-index rows learn from all 3,200 electronic rows.");
s = light(); title(s,"The shared encoder pays where data is thinnest","Gains concentrate where rows are scarce. EEA is the exception: it was already at R² 0.93, with little left to borrow.");
s.addImage({path:FIG("fig3_multitask_gain.png"),x:0.75,y:1.60,w:7.6,h:3.71});
[["+0.056","refractive index\n229 rows"],["+0.047","dielectric constant\n229 rows"],["+0.010","chain band gap\n2,028 rows"]]
 .forEach(([v,t],i)=>{ const y=1.7+i*1.45; card(s,8.75,y,3.95,1.25);
   s.addText(v,{x:8.9,y:y+0.14,w:1.55,h:0.55,fontSize:24,bold:true,color:ORANGE,fontFace:HEAD,isTextBox:true,margin:0});
   s.addText(t,{x:10.5,y:y+0.16,w:2.1,h:0.85,fontSize:11.5,color:MUTED,fontFace:BODY,isTextBox:true,margin:0}); });
s.addText("Multi-task over the cluster: +0.0157. Moving it from five folds to ten, at identical compute: a further +0.0058.",
  {x:0.75,y:5.45,w:11.8,h:0.45,fontSize:15,bold:true,color:INK,fontFace:BODY,isTextBox:true,margin:0});

/* 6 — results */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nKEY SLIDE. The result I'd point you to first.\n\nRefractive index gains 0.056. Dielectric constant, 0.047. Both have 229 rows. // Chain band gap, the one with 2,000 rows, gains 0.010.\n\nSo the gain CONCENTRATES WHERE THE ROWS ARE SCARCE. That is the signature of transfer actually happening, rather than a number that merely went up. If the big targets had gained most, I would not trust it.\n\nThere is one exception and I want to name it before you do. Electron affinity has 221 rows and gains only 0.010. It was already at R2 0.93 single-task. The gain depends on scarce data AND remaining headroom, and EEA had none.\n\nAcross the cluster, plus 0.0157. Then five folds to ten, AT IDENTICAL COMPUTE, another 0.0058. Single largest gain of the round.\n\n>>> HANDOFF 1. SAHIL SAYS: \"Neerav will take you through the numbers, and the finding I think matters most, which is what happens when you re-spell a molecule.\" Then STOP and advance.");
s = light(); title(s,"Results","Out-of-fold, blend weights fitted by nested cross-validation so they never see the rows they score.");
const rows=[[{text:"Target",options:{bold:true}},{text:"Rows",options:{bold:true}},{text:"LightGBM",options:{bold:true}},{text:"GNN",options:{bold:true}},{text:"Multi-task",options:{bold:true}},{text:"Blend",options:{bold:true}}],
 ["EGC","2,028","0.908","0.913","0.930","0.933"],["EGB","337","0.902","0.940","0.960","0.961"],
 ["EI","222","0.798","0.826","0.873","0.873"],["EEA","221","0.891","0.930","0.936","0.940"],
 ["EPS","229","0.746","0.805","0.858","0.855"],["NC","229","0.810","0.860","0.918","0.916"],
 ["TG","4,143","0.913","0.905","—","0.920"]];
s.addTable(rows,{x:0.75,y:1.6,w:7.6,fontSize:12,fontFace:BODY,color:INK,border:{type:"solid",color:"DDE3EE",pt:0.5},
  fill:{color:WHITE},rowH:0.33,align:"right",valign:"middle",colW:[1.15,1.05,1.4,1.2,1.5,1.3]});
card(s,8.75,1.7,3.95,1.5,NAVY);
s.addText("0.9139",{x:8.75,y:1.86,w:3.95,h:0.72,fontSize:40,bold:true,color:WHITE,fontFace:HEAD,align:"center",isTextBox:true,margin:0});
s.addText("official metric, mean R² over 7 targets",{x:8.75,y:2.6,w:3.95,h:0.45,fontSize:11,color:"C7D2F0",fontFace:BODY,align:"center",isTextBox:true,margin:0});
card(s,8.75,3.4,3.95,1.15);
s.addText("0.537",{x:8.75,y:3.52,w:3.95,h:0.5,fontSize:24,bold:true,color:MUTED,fontFace:HEAD,align:"center",isTextBox:true,margin:0});
s.addText("organizer Ridge baseline",{x:8.75,y:4.02,w:3.95,h:0.4,fontSize:11,color:MUTED,fontFace:BODY,align:"center",isTextBox:true,margin:0});
card(s,0.75,4.95,11.9,1.05);
s.addText([{text:"Progression:  ",options:{bold:true}},
  {text:"0.889 per-target  →  0.904 shared encoder  →  0.908 seed-averaged  →  0.914 at ten folds",options:{}}],
  {x:1.05,y:5.25,w:11.3,h:0.5,fontSize:14,color:INK,fontFace:BODY,isTextBox:true,margin:0});

/* 7 — predictions */
s.addNotes("===== NEERAV SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\n[NEERAV OPENS WITH: \"Thanks Sahil.\"]\n\nOut-of-fold. Blend weights fitted by nested cross-validation, so NO WEIGHT EVER SEES THE ROW IT SCORES. That matters because with 200 rows it is very easy to fit weights that flatter you.\n\nRead the multi-task column against LightGBM and GNN. Wins on all six. Dielectric constant, 0.75 to 0.86.\n\nOfficial metric 0.914 against the organiser's ridge baseline of 0.54. The progression underneath is the honest history: 0.889 per-target, 0.904 shared encoder, 0.914 at ten folds.");
s = light(); title(s,"Predictions against ground truth","All seven targets, out-of-fold.");
s.addImage({path:FIG("fig4_pred_vs_true.png"),x:0.9,y:1.6,w:11.5,h:11.5/2.03});

/* 8 — invariance */
s.addNotes("===== NEERAV SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nSHORT. Cut to one sentence if running long.\n\nAll seven against truth. Orange panels are the small targets. Noisier, which is what 200 rows should look like.\n\nWhat I'm checking is that there's no structured bias. No curvature, no fan, no drift at the extremes. There isn't.");
s = light(); title(s,"Polymer invariance: we measured it, then fixed it","A model that answers differently for two spellings of the same molecule cannot be trusted.");
s.addImage({path:FIG("fig5_invariance.png"),x:0.75,y:1.7,w:7.3,h:7.3/2.3});
const inv=[["Found","predictions moved up to 3.02% of target sd across randomised SMILES"],
           ["Cause","atom ordering — tied ring paths, Gasteiger charges, Ipc rounding"],
           ["Why it mattered","65% of the supplied SMILES are not canonical"],
           ["Fix","re-parse from canonical SMILES, drop Ipc. Cost: 0.0001"]];
inv.forEach(([h,t],i)=>{ const y=1.75+i*1.05;
  s.addText(h,{x:8.4,y:y,w:4.3,h:0.3,fontSize:12,bold:true,color:ORANGE,fontFace:BODY,isTextBox:true,margin:0});
  s.addText(t,{x:8.4,y:y+0.29,w:4.3,h:0.68,fontSize:11.5,color:INK,fontFace:BODY,isTextBox:true,margin:0}); });
card(s,8.4,5.95,4.3,0.72,NAVY);
s.addText("deviation after the fix: exactly 0",{x:8.4,y:6.13,w:4.3,h:0.38,fontSize:14,bold:true,color:WHITE,fontFace:BODY,align:"center",isTextBox:true,margin:0});

/* 9 — what didn't work */
s.addNotes("===== NEERAV SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nKEY SLIDE. This is the one they'll remember. Drop your pace.\n\nThe part I think matters most, and it isn't a score.\n\nA polymer can be written as several different SMILES strings. Same molecule, different spelling. A MODEL THAT ANSWERS DIFFERENTLY FOR TWO SPELLINGS OF THE SAME MOLECULE ISN'T MEASURING CHEMISTRY. IT'S MEASURING TEXT.\n\nSo we tested it. Re-spelled every molecule at random, re-predicted. Predictions moved by up to 3% of the target's standard deviation. //\n\nCross-validation would never have shown that, because in train and in test each molecule is written one particular way. YOU ONLY SEE IT IF YOU GO LOOKING.\n\nCause: atom ordering. Tied ring paths, Gasteiger charges on the dummy atoms, Ipc rounding. Fix: re-parse from canonical SMILES, drop the one descriptor we couldn't stabilise.\n\nCost 0.0001. Deviation afterwards EXACTLY ZERO. Not small. Zero. // And 65% of the supplied SMILES were not canonical, so this was live in the competition data.");
s = light(); title(s,"What did not work","Nineteen experiments across the competition. Four survived.");
const neg=[["PI1M pretraining","−0.0038","Masked-atom pretraining on 100k unlabeled repeat units reached 99% reconstruction accuracy. The representation still did not beat supervised training on 3,266 labelled rows."],
 ["Multi-task including Tg","−0.0230","Tested in Round 2. Tg and Egc share no physics, so one encoder serving both diluted each. Same technique, opposite result — which is why we measured the correlations first this time."],
 ["Multi-task LightGBM","+0.0004","Too weak alone (R² 0.79) to earn blend weight."]];
neg.forEach(([h,v,t],i)=>{ const y=1.65+i*1.55; card(s,0.75,y,11.9,1.4);
  s.addText(h,{x:1.0,y:y+0.17,w:3.5,h:0.35,fontSize:15,bold:true,color:INK,fontFace:BODY,isTextBox:true,margin:0});
  s.addText(v,{x:1.0,y:y+0.6,w:3.5,h:0.5,fontSize:22,bold:true,color:i<2?"B3261E":MUTED,fontFace:HEAD,isTextBox:true,margin:0});
  s.addText(t,{x:4.7,y:y+0.22,w:7.7,h:1.0,fontSize:12.5,color:MUTED,fontFace:BODY,isTextBox:true,margin:0}); });

/* 10 — paradigm shift */
s.addNotes("===== NEERAV SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\n19 experiments. Four survived. Three worth naming.\n\nPI1M pretraining. Masked atoms, reconstructed them across 100k unlabelled repeat units. 99% reconstruction accuracy, and it STILL LOST 0.0038 in the blend. Good at the pretext task, not better for our targets. We're reporting it because it was the auxiliary dataset you highlighted, and that's the honest finding.\n\nAnd multi-task including glass transition: minus 0.023.");
s = light(); title(s,"What changed from Round 2 to Round 3","The same technique, applied where the physics justified it.");
card(s,0.75,1.7,5.85,4.0);
s.addText("Round 2  ·  two targets",{x:1.05,y:1.95,w:5.2,h:0.35,fontSize:15,bold:true,color:MUTED,fontFace:BODY,isTextBox:true,margin:0});
s.addText([{text:"Polymer backbone features",options:{bullet:true,bold:true,breakLine:true}},
  {text:"18 scalars from the chain between the two attachment points. Worth +0.0056, more than 1,024 Avalon bits.",options:{breakLine:true,fontSize:12,color:MUTED}},
  {text:"Message-passing GNN",options:{bullet:true,bold:true,breakLine:true}},
  {text:"Reads topology directly; errors decorrelated from the trees. Worth +0.0053.",options:{breakLine:true,fontSize:12,color:MUTED}},
  {text:"Multi-task across Tg and Egc",options:{bullet:true,bold:true,breakLine:true}},
  {text:"Rejected: −0.0230.",options:{fontSize:12,color:"B3261E"}}],
  {x:1.05,y:2.4,w:5.25,h:3.1,fontSize:13.5,color:INK,fontFace:BODY,isTextBox:true,margin:0,paraSpaceAfter:5});
card(s,6.9,1.7,5.75,4.0,NAVY);
s.addText("Round 3  ·  seven targets",{x:7.2,y:1.95,w:5.1,h:0.35,fontSize:15,bold:true,color:"8FA3D9",fontFace:BODY,isTextBox:true,margin:0});
s.addText([{text:"Measure the targets first",options:{bullet:true,bold:true,breakLine:true,color:WHITE}},
  {text:"Correlation structure decided the architecture before any model ran.",options:{breakLine:true,fontSize:12,color:"C7D2F0"}},
  {text:"Multi-task across the electronic cluster",options:{bullet:true,bold:true,breakLine:true,color:WHITE}},
  {text:"Same technique that failed in Round 2, now +0.0157 — because these six targets are one physical quantity. Ten folds instead of five added +0.0058 more.",options:{breakLine:true,fontSize:12,color:"C7D2F0"}},
  {text:"Invariance treated as a measurement",options:{bullet:true,bold:true,breakLine:true,color:WHITE}},
  {text:"Not an assumption. Found a 3% defect and removed it.",options:{fontSize:12,color:"C7D2F0"}}],
  {x:7.2,y:2.4,w:5.15,h:3.1,fontSize:13.5,color:WHITE,fontFace:BODY,isTextBox:true,margin:0,paraSpaceAfter:5});

/* 11 — particulars */
s.addNotes("===== NEERAV SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nKEY SLIDE. The payoff. Don't rush it.\n\nThat minus 0.023 is from Round 2, where THE SAME TECHNIQUE MADE THINGS WORSE. One encoder over Tg and band gap. They share no physics, so it just diluted both. //\n\nSame technique. Opposite result. The difference isn't the model. It's WHETHER THE TARGETS BELONG TOGETHER.\n\nThat's why this round we measured the correlations first and let them choose the architecture, instead of trying multi-task and seeing what happened.\n\n>>> HANDOFF 2. NEERAV SAYS: \"Sahil will close on how it runs, and what it actually says about polymers.\" SAHIL: \"Thanks Neerav.\" Advance.");
s = light(); title(s,"Model particulars","Everything runs from scratch inside one Kaggle notebook. No pretrained weights, no uploaded artifacts.");
const pc=[["387k","parameters\nmulti-task encoder"],["10 × 3","folds × seeds\nmulti-task encoder"],["4h 43m","training\nKaggle CPU, no GPU"],["<1 ms","inference\nper molecule"]];
pc.forEach(([v,t],i)=>{ const x=0.75+i*3.06; card(s,x,1.8,2.85,1.9); stat(s,x,1.98,2.85,v,t,BLUE); });
card(s,0.75,4.15,11.9,1.75);
s.addText([{text:"Reproducibility.  ",options:{bold:true}},
  {text:"Single-threaded with per-fold seeds — parallel index_add_ in message passing is otherwise non-deterministic, and we measured two identical runs differing by 0.18 per prediction before pinning it. Kaggle reproduced our local out-of-fold score within 0.0035 on every run of the competition.",options:{}}],
  {x:1.05,y:4.42,w:11.3,h:1.3,fontSize:13.5,color:INK,fontFace:BODY,isTextBox:true,margin:0});

/* 12 — close */
s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\n[SAHIL OPENS WITH: \"Thanks Neerav.\"]\n\nCut to one sentence if running long.\n\nBriefly, the engineering. 387k parameters. Four and three quarter hours on Kaggle CPU, no GPU. Under a millisecond per molecule at inference.\n\nAnd it's deterministic. Parallel message passing isn't, and we measured two identical runs differing by 0.18 per prediction before we pinned it. Kaggle reproduced our local score within 0.0035 on every run.");
s = dark();
s.addText("What we would tell a materials scientist",{x:0.9,y:1.5,w:11.5,h:0.7,fontSize:32,bold:true,color:WHITE,fontFace:HEAD,isTextBox:true,margin:0});
const close=[["Electronic","Band gap falls as conjugation runs along the backbone lengthen and as attachment atoms turn sp² — conjugation then propagates across repeat units instead of stopping at the unit boundary. Dimer conjugated-bond fraction correlates −0.76 with chain band gap."],
 ["Optical","Refractive index and dielectric constant rise with the same polarisable, aromatic-rich backbones that narrow the gap (+0.92 with each other, −0.85 with band gap). The absorption edge follows at ≈1240/Egc nm. Side-chain bulk blue-shifts and solubilises."]];
close.forEach(([h,t],i)=>{ const y=2.5+i*1.75;
  s.addText(h,{x:0.9,y:y,w:2.2,h:0.4,fontSize:17,bold:true,color:ORANGE,fontFace:HEAD,isTextBox:true,margin:0});
  s.addText(t,{x:3.2,y:y,w:9.2,h:1.5,fontSize:13,color:"D6DEF5",fontFace:BODY,isTextBox:true,margin:0}); });
s.addText("References:  Gilmer et al., Neural Message Passing for Quantum Chemistry, ICML 2017  ·  Ke et al., LightGBM, NeurIPS 2017  ·  Gedeck et al., Avalon fingerprints, JCIM 2006  ·  Ma & Luo, PI1M, JCIM 2020  ·  RDKit",
  {x:0.9,y:6.05,w:11.5,h:0.6,fontSize:10.5,color:"8FA3D9",fontFace:BODY,isTextBox:true,margin:0});
s.addText("Team NeonNull   ·   github.com/Sahilo6/amalgam-polymer   ·   kaggle.com/code/sahilsadhwani25/neonnull-round-3",
  {x:0.9,y:6.75,w:11.5,h:0.4,fontSize:11,color:"C7D2F0",fontFace:BODY,isTextBox:true,margin:0});

s.addNotes("===== SAHIL SPEAKS =====\n(Sahil drives the deck throughout, including Neerav's slides.)\n\nCLOSE. Land it, then STOP. Don't trail off.\n\nTo close, what this says about polymers. Band gap falls as conjugation runs further along the backbone. Refractive index and dielectric constant rise with the same polarisable, aromatic-rich backbones that narrow that gap.\n\nONE STORY, NOT THREE. And it's the same story the correlation matrix told us before we built anything.\n\nEverything is in the notebook and the repo. Thank you. I'm happy to take questions.\n\n--- IF CUT OFF AT TIME: \"I'll stop there. The one line I'd leave you with is that we measured which targets belong together before choosing a model, and that decision is worth more than any tuning we did.\"");
P.writeFile({fileName:"deck/NeonNull_Finale_PRESENTER.pptx"}).then(()=>console.log("presenter deck written"));
