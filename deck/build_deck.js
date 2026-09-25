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
s.addNotes("Seven polymer properties from SMILES. Our approach is built around one measured fact: six of the seven are the same physics, and five of them have almost no data.");

/* 2 — the problem */
s = light(); title(s,"The problem is not accuracy. It is imbalance.","Seven targets, one unweighted mean of per-target R². Five targets have ~220 training rows each.");
s.addImage({path:FIG("fig2_row_imbalance.png"),x:0.75,y:1.6,w:7.5,h:3.33});
card(s,8.6,1.7,4.1,1.35);
stat(s,8.6,1.82,4.1,"5 of 7","targets under 500 rows",ORANGE);
card(s,8.6,3.2,4.1,1.35);
stat(s,8.6,3.32,4.1,"71%","of the metric carried by\n14% of the data",ORANGE);
s.addText("A per-target LightGBM reaches only R² 0.75 on dielectric constant. Fitting these independently is hopeless.",
  {x:0.75,y:5.3,w:11.8,h:0.5,fontSize:14,italic:true,color:MUTED,fontFace:BODY,isTextBox:true,margin:0});
s.addNotes("Because the metric averages per-target R2 without weighting, the five small targets carry five sevenths of the score while contributing a seventh of the data. Everything we did follows from that.");

/* 3 — the insight */
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
s.addNotes("This measurement, not a hyperparameter search, dictated the architecture. Six targets share an encoder; Tg does not.");

/* 4 — architecture */
s = light(); title(s,"Architecture","Tree path, graph path, and one encoder shared across the six electronic targets.");
s.addImage({path:FIG("fig6_architecture.png"),x:0.95,y:1.5,w:11.4,h:11.4/1.99});
s.addNotes("Tree path: RDKit descriptors, fingerprints, and our polymer backbone features. Graph path: a message-passing network. The multi-task encoder serves the six electronic targets through six heads; each row trains only its own head. Blend weights fitted per target on out-of-fold predictions.");

/* 5 — what worked */
s = light(); title(s,"The shared encoder pays where data is thinnest","Gains scale inversely with training rows — exactly what transfer should look like.");
s.addImage({path:FIG("fig3_multitask_gain.png"),x:0.75,y:1.55,w:7.6,h:3.70});
[["+0.041","refractive index\n229 rows"],["+0.029","dielectric constant\n229 rows"],["+0.007","chain band gap\n2,028 rows"]]
 .forEach(([v,t],i)=>{ const y=1.7+i*1.45; card(s,8.75,y,3.95,1.25);
   s.addText(v,{x:8.9,y:y+0.14,w:1.55,h:0.55,fontSize:24,bold:true,color:ORANGE,fontFace:HEAD,isTextBox:true,margin:0});
   s.addText(t,{x:10.5,y:y+0.16,w:2.1,h:0.85,fontSize:11.5,color:MUTED,fontFace:BODY,isTextBox:true,margin:0}); });
s.addText("Multi-task over the cluster: +0.0157 on the official metric.",
  {x:0.75,y:5.45,w:11.8,h:0.45,fontSize:15,bold:true,color:INK,fontFace:BODY,isTextBox:true,margin:0});
s.addNotes("The smallest targets gain most. Refractive index, with 229 rows, gains 0.041 R2 purely from borrowing a representation learned on 3,266 electronic rows.");

/* 6 — results */
s = light(); title(s,"Results","Out-of-fold, blend weights fitted by nested cross-validation so they never see the rows they score.");
const rows=[[{text:"Target",options:{bold:true}},{text:"Rows",options:{bold:true}},{text:"LightGBM",options:{bold:true}},{text:"GNN",options:{bold:true}},{text:"Multi-task",options:{bold:true}},{text:"Blend",options:{bold:true}}],
 ["EGC","2,028","0.908","0.913","0.927","0.930"],["EGB","337","0.902","0.940","0.958","0.959"],
 ["EI","222","0.798","0.826","0.866","0.864"],["EEA","221","0.891","0.930","0.942","0.945"],
 ["EPS","229","0.746","0.805","0.834","0.837"],["NC","229","0.810","0.860","0.906","0.903"],
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
s.addText([{text:"Progression across submissions:  ",options:{bold:true}},
  {text:"0.889 per-target  →  0.904 adding the shared encoder  →  0.908 seed-averaged  →  0.914 at ten folds",options:{}}],
  {x:1.05,y:5.25,w:11.3,h:0.5,fontSize:14,color:INK,fontFace:BODY,isTextBox:true,margin:0});
s.addNotes("The multi-task column is the story: it beats the per-target models on every one of the six, and the blend adds a little more.");

/* 7 — predictions */
s = light(); title(s,"Predictions against ground truth","All seven targets, out-of-fold.");
s.addImage({path:FIG("fig4_pred_vs_true.png"),x:0.9,y:1.6,w:11.5,h:11.5/2.03});
s.addNotes("Orange panels are the 220-row targets. They are noisier, as expected, but no longer unusable.");

/* 8 — invariance */
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
s.addNotes("Nothing in cross-validation reveals this, because train and test SMILES are each written one particular way. It only appears if you deliberately re-spell the molecules.");

/* 9 — what didn't work */
s = light(); title(s,"What did not work","Nineteen experiments across the competition. Four survived.");
const neg=[["PI1M pretraining","−0.0038","Masked-atom pretraining on 100k unlabeled repeat units reached 99% reconstruction accuracy. The representation still did not beat supervised training on 3,266 labelled rows."],
 ["Multi-task including Tg","−0.0230","Tested in Round 2. Tg and Egc share no physics, so one encoder serving both diluted each. Same technique, opposite result — which is why we measured the correlations first this time."],
 ["Multi-task LightGBM","+0.0004","Too weak alone (R² 0.79) to earn blend weight."]];
neg.forEach(([h,v,t],i)=>{ const y=1.65+i*1.55; card(s,0.75,y,11.9,1.4);
  s.addText(h,{x:1.0,y:y+0.17,w:3.5,h:0.35,fontSize:15,bold:true,color:INK,fontFace:BODY,isTextBox:true,margin:0});
  s.addText(v,{x:1.0,y:y+0.6,w:3.5,h:0.5,fontSize:22,bold:true,color:i<2?"B3261E":MUTED,fontFace:HEAD,isTextBox:true,margin:0});
  s.addText(t,{x:4.7,y:y+0.22,w:7.7,h:1.0,fontSize:12.5,color:MUTED,fontFace:BODY,isTextBox:true,margin:0}); });
s.addNotes("We report the pretraining result as a negative rather than omitting it. The auxiliary dataset was the one the organizers highlighted, and the honest finding is that it did not transfer.");

/* 10 — paradigm shift */
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
  {text:"Same technique that failed in Round 2, now +0.0157 — because these six targets are one physical quantity.",options:{breakLine:true,fontSize:12,color:"C7D2F0"}},
  {text:"Invariance treated as a measurement",options:{bullet:true,bold:true,breakLine:true,color:WHITE}},
  {text:"Not an assumption. Found a 3% defect and removed it.",options:{fontSize:12,color:"C7D2F0"}}],
  {x:7.2,y:2.4,w:5.15,h:3.1,fontSize:13.5,color:WHITE,fontFace:BODY,isTextBox:true,margin:0,paraSpaceAfter:5});
s.addNotes("The paradigm shift is that we stopped asking which model is best and started asking which targets belong together.");

/* 11 — particulars */
s = light(); title(s,"Model particulars","Everything runs from scratch inside one Kaggle notebook. No pretrained weights, no uploaded artifacts.");
const pc=[["387k","parameters\nmulti-task encoder"],["10 × 3","folds × seeds\nmulti-task"],["3h 20m","training\nKaggle CPU, no GPU"],["<1 ms","inference\nper molecule"]];
pc.forEach(([v,t],i)=>{ const x=0.75+i*3.06; card(s,x,1.8,2.85,1.9); stat(s,x,1.98,2.85,v,t,BLUE); });
card(s,0.75,4.15,11.9,1.75);
s.addText([{text:"Reproducibility.  ",options:{bold:true}},
  {text:"Single-threaded with per-fold seeds — parallel index_add_ in message passing is otherwise non-deterministic, and we measured two identical runs differing by 0.18 per prediction before pinning it. Kaggle reproduced our local out-of-fold score within 0.0035 on every run of the competition.",options:{}}],
  {x:1.05,y:4.42,w:11.3,h:1.3,fontSize:13.5,color:INK,fontFace:BODY,isTextBox:true,margin:0});
s.addNotes("Determinism matters because the hosts re-run the pinned notebook. Single-threading cost nothing because the graphs are small.");

/* 12 — close */
s = dark();
s.addText("What we would tell a materials scientist",{x:0.9,y:1.5,w:11.5,h:0.7,fontSize:32,bold:true,color:WHITE,fontFace:HEAD,isTextBox:true,margin:0});
const close=[["Electronic","Band gap falls as conjugation runs along the backbone lengthen and as attachment atoms turn sp² — conjugation then propagates across repeat units instead of stopping at the unit boundary. Dimer conjugated-bond fraction correlates −0.76 with chain band gap."],
 ["Optical","Refractive index and dielectric constant rise with the same polarisable, aromatic-rich backbones that narrow the gap (+0.92 with each other, −0.85 with band gap). The absorption edge follows at ≈1240/Egc nm. Side-chain bulk blue-shifts and solubilises."]];
close.forEach(([h,t],i)=>{ const y=2.5+i*1.75;
  s.addText(h,{x:0.9,y:y,w:2.2,h:0.4,fontSize:17,bold:true,color:ORANGE,fontFace:HEAD,isTextBox:true,margin:0});
  s.addText(t,{x:3.2,y:y,w:9.2,h:1.5,fontSize:13,color:"D6DEF5",fontFace:BODY,isTextBox:true,margin:0}); });
s.addText("Team NeonNull   ·   kaggle.com/code/sahilsadhwani25/neonnull-round-3",
  {x:0.9,y:6.55,w:11.5,h:0.4,fontSize:12,color:"8FA3D9",fontFace:BODY,isTextBox:true,margin:0});
s.addNotes("The model is legible: its feature importance concentrates on the backbone block, so for a new polymer it reports not just the predicted value but which structural motifs drive it.");

P.writeFile({fileName:"deck/NeonNull_Amalgam_Finale.pptx"}).then(()=>console.log("deck written"));
