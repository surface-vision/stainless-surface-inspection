// Round 2 deck: Stainless Spark, PS1, AI surface defect detection
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5
pres.title = 'Stainless Spark Round 2 - AI Surface Defect Detection';

const C = {
  navy: '143A5A', navy2: '2B5B80', onNavy: 'BFD2E2', accent: '0086C3',
  ink: '141C24', slate: '334150', muted: '5F6B78', neutral: 'F1F4F7', edge: 'D5DCE3',
  white: 'FFFFFF', green: '2E7D32', amber: 'B77900', red: 'C62828',
  h1: 'DCEAF5', h2: 'A9CBE6', h3: '5E9CCB', h4: '1F6FA8',
};
const F = 'Calibri';
const W = 13.333;
// ---------------------------------------------------------------------------
// EDIT BEFORE SUBMITTING: these four lines are the only placeholders.
const TEAM = '[TEAM NAME]';
const MEMBERS = ['[Member 1]', '[Member 2]', '[Member 3]', '[Member 4]'];
const INSTITUTION = '[INSTITUTION]';
// ---------------------------------------------------------------------------
const SECTIONS = ['Problem', 'Insights', 'Solution', 'Implementation', 'Impact'];

function T(slide, text, o) {
  slide.addText(text, Object.assign({ fontFace: F, isTextBox: true, margin: 0, valign: 'top', color: C.ink, fontSize: 10 }, o));
}
function R(slide, x, y, w, h, fill, o) {
  slide.addShape(pres.shapes.RECTANGLE, Object.assign({ x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 } }, o || {}));
}
function RR(slide, x, y, w, h, fill, o) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: fill, width: 0 } }, o || {}));
}
function dot(slide, x, y, d, col) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { color: col, width: 0 } });
}
function chrome(slide, active, title, band, source) {
  slide.background = { color: C.white };
  // navigation tabs
  const x0 = 0.5, tw = 1.62, gap = 0.06;
  SECTIONS.forEach((s, i) => {
    const on = s === active;
    R(slide, x0 + i * (tw + gap), 0.22, tw, 0.3, on ? C.navy : C.neutral);
    T(slide, `${i + 1}  ${s}`, { x: x0 + i * (tw + gap), y: 0.22, w: tw, h: 0.3, align: 'center', valign: 'middle', fontSize: 10, bold: on, color: on ? C.white : C.muted });
  });
  T(slide, `STAINLESS SPARK ROUND 2  |  PS1  |  ${TEAM}`, { x: 9.0, y: 0.22, w: 3.83, h: 0.3, align: 'right', valign: 'middle', fontSize: 9, color: C.muted });
  T(slide, title, { x: 0.5, y: 0.62, w: 12.33, h: 0.62, fontSize: 20, bold: true, color: C.navy, valign: 'middle' });
  // decision band
  R(slide, 0.5, 6.82, 12.33, 0.46, C.navy);
  T(slide, band, { x: 0.68, y: 6.82, w: 12.0, h: 0.46, fontSize: 12, bold: true, color: C.white, valign: 'middle' });
  if (source) T(slide, source, { x: 0.5, y: 6.55, w: 12.33, h: 0.24, fontSize: 9, color: C.muted, valign: 'middle' });
}
function ph(slide, x, y, w, text, h) {
  R(slide, x, y, w, h || 0.3, C.navy2);
  T(slide, text, { x: x + 0.1, y, w: w - 0.2, h: h || 0.3, fontSize: 11, bold: true, color: C.white, valign: 'middle' });
}
function tag(slide, x, y, text, col, w) {
  RR(slide, x, y, w || 0.78, 0.2, col);
  T(slide, text, { x, y, w: w || 0.78, h: 0.2, fontSize: 9, bold: true, color: C.white, align: 'center', valign: 'middle' });
}
function table(slide, rows, o) {
  slide.addTable(rows, Object.assign({ fontFace: F, fontSize: 9.5, color: C.ink, border: { type: 'solid', pt: 0.5, color: C.edge }, margin: [0.03, 0.05, 0.03, 0.05], valign: 'middle' }, o));
}
const hd = (t, o) => ({ text: t, options: Object.assign({ bold: true, color: C.white, fill: { color: C.navy } }, o || {}) });
const cl = (t, o) => ({ text: t, options: o || {} });

// ============ SLIDE 1: COVER ============
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  T(s, 'JINDAL STAINLESS  |  STAINLESS SPARK: ENGINEERING INNOVATION, BUILDING FUTURES  |  ROUND 2  |  PROBLEM STATEMENT 1', { x: 0.6, y: 0.45, w: 12, h: 0.3, fontSize: 11, color: C.onNavy, bold: true, charSpacing: 1 });
  T(s, 'Catch every strip defect where it is born, not where it is shipped', { x: 0.6, y: 1.0, w: 8.6, h: 1.4, fontSize: 34, bold: true, color: C.white });
  T(s, 'AI surface inspection for stainless strip that detects, classifies and scores every defect at line speed, then routes it to the process that caused it. Built, validated on three public steel datasets, and running live today.', { x: 0.6, y: 2.6, w: 8.2, h: 0.95, fontSize: 15, color: C.onNavy });
  const stats = [
    ['0.764', 'detection accuracy on unseen steel\nimages (published range 0.70-0.80)'],
    ['93.7% to 32.5%', 'clean steel wrongly flagged,\nbefore vs after clean-strip training'],
    ['INR 2.5-4.2 cr', 'per line, estimated, vs USD 1-3M\nfor a commercial system'],
    ['~2.2 years', 'payback for one line at a\nconservative 1% downgrade rate'],
  ];
  stats.forEach(([n, l], i) => {
    const x = 0.6 + i * 2.1;
    T(s, n, { x, y: 3.85, w: 2.0, h: 0.5, fontSize: [24, 19, 18, 22][i], bold: true, color: C.white, valign: 'bottom' });
    T(s, l, { x, y: 4.4, w: 2.0, h: 0.7, fontSize: 10, color: C.onNavy });
  });
  // demo card
  RR(s, 9.55, 1.0, 3.2, 4.1, C.white);
  s.addImage({ path: 'qr.png', x: 10.2, y: 1.2, w: 1.9, h: 1.9 });
  T(s, 'TRY THE LIVE DEMO', { x: 9.7, y: 3.2, w: 2.9, h: 0.3, fontSize: 11, bold: true, color: C.accent, align: 'center' });
  T(s, 'surface-vision.github.io', { x: 9.7, y: 3.5, w: 2.9, h: 0.35, fontSize: 15, bold: true, color: C.navy, align: 'center' });
  T(s, 'The trained detector runs in your browser. Upload a strip image to get box, type, confidence and severity. Nothing leaves the page.', { x: 9.8, y: 3.9, w: 2.7, h: 1.0, fontSize: 10, color: C.slate, align: 'center' });
  // storyline
  SECTIONS.forEach((sec, i) => {
    const q = ['Where is value lost?', 'What did we learn?', 'What do we build?', 'How does JSL deploy it?', 'What is it worth?'][i];
    RR(s, 0.6 + i * 2.45, 5.45, 2.35, 0.62, C.navy2);
    T(s, `${i + 1}  ${sec}`, { x: 0.72 + i * 2.45, y: 5.49, w: 2.2, h: 0.28, fontSize: 12, bold: true, color: C.white });
    T(s, q, { x: 0.72 + i * 2.45, y: 5.76, w: 2.2, h: 0.26, fontSize: 10, color: C.onNavy });
  });
  T(s, `TEAM  ${TEAM}   |   ${MEMBERS.join('  ·  ')}   |   ${INSTITUTION}`, { x: 0.6, y: 6.45, w: 12.1, h: 0.4, fontSize: 13, bold: true, color: C.white, valign: 'middle' });
  s.addNotes('Cover. Open by pointing at the QR: the detector is live at surface-vision.github.io and runs in the judge\'s browser. The four numbers are the story in one line: it is accurate within the published band, we fixed the false-alarm problem on real clean steel, it costs a fraction of a commercial system, and one line pays back in about two years even at a conservative downgrade rate.');
}

// ============ SLIDE 2: PROBLEM ============
{
  const s = pres.addSlide();
  chrome(s, 'Problem',
    'A defect born at the caster is paid for at every later stage, but today it is found at the end of the line',
    'The problem is not that defects exist. They are found late and by sampling, and nobody is told which process caused them.',
    'Sources: AMETEK Surface Vision, Ternium case study (2020); IISE, Measuring the Cost of Quality; ASTM A240 (304: 8.0-10.5% Ni); LME nickel USD 16,402/t and INR 95.8/USD, 22 Sep 2026; JSL FY26 results.');
  // value chain
  ph(s, 0.5, 1.35, 8.35, 'Where stainless surface defects are born, and where they are found without automated inspection');
  const st = [
    ['Melt & cast', 'SMS', 'Inclusions, slivers', 'none'],
    ['Hot rolling', 'HSM', 'Rolled-in scale, pits, edge cracks', 'none'],
    ['Anneal & pickle', 'HRAP', 'Pickling patches, over / under-pickle', 'spot check'],
    ['Cold rolling', 'CRM', 'Scratches, roll marks, crazing', 'spot check'],
    ['Final anneal / BA', 'CRAP / BA', 'Stains, BA marks, handling scratches', 'visual sample'],
    ['Finish & despatch', 'Slit, cut', 'Found here or by the customer', 'visual + claims'],
  ];
  const bw = 1.3, bx0 = 0.55, by = 1.75;
  st.forEach(([n, code, born, seen], i) => {
    const x = bx0 + i * (bw + 0.08);
    T(s, 'BORN HERE', { x, y: by, w: bw, h: 0.18, fontSize: 9, bold: true, color: C.muted });
    T(s, born, { x, y: by + 0.18, w: bw, h: 0.5, fontSize: 9.5, color: C.ink });
    RR(s, x, by + 0.72, bw, 0.62, i === 5 ? C.accent : C.navy);
    T(s, n, { x: x + 0.05, y: by + 0.76, w: bw - 0.1, h: 0.3, fontSize: 11, bold: true, color: C.white, align: 'center' });
    T(s, code, { x: x + 0.05, y: by + 1.04, w: bw - 0.1, h: 0.24, fontSize: 9.5, color: C.onNavy, align: 'center' });
    const col = seen === 'none' ? C.red : (seen === 'spot check' ? C.amber : C.amber);
    T(s, 'SEEN TODAY', { x, y: by + 1.42, w: bw, h: 0.18, fontSize: 9, bold: true, color: C.muted });
    T(s, seen, { x, y: by + 1.6, w: bw, h: 0.22, fontSize: 10, bold: true, color: col });
  });
  T(s, 'Value added keeps rising left to right: melt, rolling, annealing and pickling are all paid for before the coil is graded. "Seen today" is typical practice without automated surface inspection; JSL\'s own inspection coverage by line is a P0 input.', { x: 0.55, y: 3.72, w: 8.25, h: 0.45, fontSize: 9.5, color: C.slate, italic: true });

  // PAF framework
  ph(s, 0.5, 4.3, 8.35, 'Cost-of-quality lens (PAF): where AI inspection moves the money');
  table(s, [
    [hd(''), hd('Prevention'), hd('Appraisal'), hd('Internal failure'), hd('External failure')],
    [cl('Today', { bold: true }), cl('Process fixes happen after complaints'), cl('Manual, sampled, inspector-dependent'), cl('Downgrade, rework, re-pickle, scrap'), cl('Customer claims, returns, lost orders')],
    [cl('With AI', { bold: true, color: C.accent }), cl('Typed defect sent to its owner in minutes'), cl('100% of the surface, the same rule on every coil'), cl('Caught upstream, before more value is added'), cl('Fewer escapes to exposed-finish customers')],
  ], { x: 0.5, y: 4.62, w: 8.35, colW: [0.95, 1.85, 1.85, 1.85, 1.85], rowH: [0.26, 0.5, 0.5] });
  T(s, 'Output: shift spend from failure costs (right) to appraisal and prevention (left). Cost of poor quality averages ~15% of sales in manufacturing, range 5-35% (IISE).', { x: 0.5, y: 5.95, w: 8.35, h: 0.45, fontSize: 9.5, color: C.slate });

  // right column stats
  ph(s, 9.1, 1.35, 3.73, 'Why it hurts more on stainless');
  const facts = [
    ['300 m/min', '"...it is impossible for the human eye to inspect the whole surface." AMETEK / Ternium, cold mill'],
    ['INR 1.26-1.65 lakh', 'of nickel in every tonne of 304 (8-10.5% Ni). Downgrading it gives away far more value than on carbon steel'],
    ['2.57 Mt', 'JSL FY26 sales at INR 1,67,407/t blended realisation, so every 0.1% downgraded is ~2,570 t'],
  ];
  facts.forEach(([n, l], i) => {
    const y = 1.75 + i * 1.15;
    T(s, n, { x: 9.15, y, w: 3.65, h: 0.42, fontSize: 22, bold: true, color: C.accent });
    T(s, l, { x: 9.15, y: y + 0.42, w: 3.65, h: 0.65, fontSize: 9.5, color: C.slate });
  });
  RR(s, 9.1, 5.25, 3.73, 1.15, C.neutral);
  T(s, [
    { text: 'Problem statement\n', options: { bold: true, color: C.navy, fontSize: 11 } },
    { text: 'How might JSL detect, classify and score every surface defect at line speed, early enough to act, and tell the right process owner what caused it?', options: { fontSize: 10, color: C.ink } },
  ], { x: 9.2, y: 5.3, w: 3.55, h: 1.05 });
  s.addNotes('Problem. Walk the chain left to right. The key point: a sliver born at the caster is only seen at finishing, after hot rolling, annealing, pickling and cold rolling have all been paid for. Nickel is why this hurts more on stainless: at today\'s LME price (USD 16,402/t, INR 95.8/USD) there is INR 1.26-1.65 lakh of nickel in every tonne of 304. The PAF table is the framework: AI inspection is an appraisal cost that moves spend out of internal and external failure. The "seen today" row is typical practice; we found no public statement of JSL\'s inspection coverage, so we ask for it in P0 instead of assuming.');
}

// ============ SLIDE 3: INSIGHTS I (stainless) ============
{
  const s = pres.addSlide();
  chrome(s, 'Insights',
    'On stainless the surface is the product, so grade, finish and root cause decide which defect matters',
    'Classify, don\'t just alarm: a typed defect becomes a work order for a named owner. The pilot belongs on a 300-series cold-rolled line.',
    'Sources: ASSDA, 2B/2D/BA finishes; JSL Q1 FY27 earnings call and results (Aug 2026); JSL cold-rolling expansion to 2.67 MTPA; Steel in Translation 2023 (rolled-in scale); Leão et al. 2021 (tundish slivers). Matrix: team hypothesis.');
  // left: why different
  ph(s, 0.5, 1.35, 3.9, 'Three facts that change the brief');
  const f = [
    ['The finish cannot be repaired', 'For mill finishes like 2B and BA, "surface damage such as scratches, grinding marks or spatter cannot be matched by polishing" (ASSDA).'],
    ['JSL is moving toward surface-critical tonnes', 'Cold rolling grows 2.05 to 2.67 MTPA by FY28. The 300-series mix held margins in Q1 FY27. Exports are 11% (Japan, Korea, Brazil).'],
    ['End markets see the surface', 'Automotive, railways, metros and white goods drive demand (Q1 FY27 call). BA goes into appliance interiors.'],
  ];
  f.forEach(([h, b], i) => {
    const y = 1.75 + i * 0.78;
    dot(s, 0.55, y + 0.03, 0.28, C.accent);
    T(s, String(i + 1), { x: 0.55, y: y + 0.03, w: 0.28, h: 0.28, fontSize: 11, bold: true, color: C.white, align: 'center', valign: 'middle' });
    T(s, h, { x: 0.95, y, w: 3.45, h: 0.28, fontSize: 11, bold: true, color: C.navy });
    T(s, b, { x: 0.95, y: y + 0.27, w: 3.45, h: 0.5, fontSize: 9.5, color: C.slate });
  });

  // right: grade x defect matrix
  ph(s, 4.65, 1.35, 8.18, 'Framework: grade-by-defect risk matrix (which defects to train for first)');
  const lv = { H: [C.h4, C.white], M: [C.h2, C.ink], L: [C.h1, C.muted] };
  const g = [
    ['200 series (Cr-Mn)', 'M', 'M', 'H', 'M', 'L', 'Kitchenware, tubes'],
    ['300 series (304, 316)', 'H', 'M', 'H', 'L', 'L', 'Appliances, auto, rail'],
    ['400 series (409, 430)', 'H', 'M', 'M', 'M', 'H', 'Exhaust, white goods'],
    ['Duplex (2205)', 'M', 'H', 'M', 'H', 'L', 'Process, infrastructure'],
  ];
  const rows = [[hd('Grade family'), hd('Slivers, inclusions'), hd('Scale, pickling'), hd('Scratches, roll marks'), hd('Edge cracks'), hd('Ridging, roping'), hd('Typical exposed use')]];
  g.forEach(r => {
    rows.push([cl(r[0], { bold: true })].concat(r.slice(1, 6).map(v => cl(v, { align: 'center', bold: true, fill: { color: lv[v][0] }, color: lv[v][1] }))).concat([cl(r[6], { fontSize: 9 })]));
  });
  table(s, rows, { x: 4.65, y: 1.7, w: 8.18, colW: [1.7, 1.0, 1.0, 1.08, 0.95, 0.95, 1.5], rowH: 0.27 });
  T(s, 'H / M / L = relative surface risk. Team hypothesis from metallurgy literature (e.g. TiN-cluster slivers in Ti-stabilised grades, ridging in ferritic 430, poor hot ductility in duplex); to be re-scored in P0 against JSL\'s defect records. Output: 300 series carries the highest risk on exposed finishes and the most nickel per tonne, so it is the pilot grade.', { x: 4.65, y: 3.13, w: 8.18, h: 0.6, fontSize: 9.5, color: C.slate });

  // bottom: routing table
  ph(s, 0.5, 4.15, 12.33, 'Framework: root-cause routing (condensed fishbone). Each class the model outputs maps to a process owner');
  table(s, [
    [hd('Defect family'), hd('Born at'), hd('Likely cause (evidence)'), hd('Owner'), hd('First corrective action')],
    [cl('Inclusion, sliver', { bold: true }), cl('Caster'), cl('Low tundish level, mould-flux entrainment, reoxidation (Leão 2021)'), cl('SMS'), cl('Hold tundish level, review flux; tag the heat ID')],
    [cl('Rolled-in scale, pits', { bold: true }), cl('Reheat, HSM'), cl('Descaler pressure below spec, blocked nozzles (Steel in Translation 2023)'), cl('HSM'), cl('Keep descaler above ~180 bar, clear nozzles')],
    [cl('Patches, pickling marks', { bold: true }), cl('HRAP / CRAP'), cl('Uneven scale removal; acid concentration, temperature, line speed'), cl('Pickling line'), cl('Check acid concentration and speed across width')],
    [cl('Scratches, roll marks', { bold: true }), cl('CRM, handling'), cl('Guides, roller tables, roll surface condition, coil handling'), cl('CRM / logistics'), cl('Inspect guides, trigger roll change')],
    [cl('Edge cracks', { bold: true }), cl('HSM'), cl('Edge cooling, low hot ductility (duplex, ferritic)'), cl('HSM'), cl('Edge heating and trim practice; needs edge cameras')],
  ], { x: 0.5, y: 4.47, w: 12.33, colW: [1.9, 1.3, 4.5, 1.4, 3.23], rowH: [0.24, 0.31, 0.31, 0.31, 0.31, 0.31] });
  s.addNotes('Insights, stainless-specific. First, the finish is the product and cannot be polished back. Second, JSL is growing cold rolling by about 30% by FY28 and leaning on the 300 series, so the share of surface-critical tonnes rises. The matrix is our framework for choosing what to train first. It is a hypothesis from the metallurgy literature and we say so; P0 re-scores it with JSL\'s defect records. Its output is the pilot grade: 300 series. The routing table turns detection into action: every class maps to a process owner and a first corrective action. Crazing and patches attributions are reasoned, not sourced, and need a JSL metallurgist\'s sign-off.');
}

// ============ SLIDE 4: INSIGHTS II (evidence) ============
{
  const s = pres.addSlide();
  chrome(s, 'Insights',
    'Our Round 1 build produced four findings, and each one changed the design',
    'The model is the cheap part. JSL\'s own strip, the resolution spec, and trustworthy confidence decide whether the system works.',
    'All numbers measured by us on three public steel-surface datasets, on images the model never saw in training.');
  const qx = [0.5, 6.77], qy = [1.35, 3.97], qw = 6.06, qh = 2.5;
  const quad = (i, j, head) => { ph(s, qx[i], qy[j], qw, head); RR(s, qx[i], qy[j] + 0.3, qw, qh - 0.3, C.neutral); };
  const so = (i, j, txt) => T(s, [{ text: 'So we: ', options: { bold: true, color: C.accent } }, { text: txt, options: { color: C.ink } }], { x: qx[i] + 0.12, y: qy[j] + qh - 0.42, w: qw - 0.24, h: 0.38, fontSize: 10 });

  // Q1 domain shift
  quad(0, 0, '1  A model that has never seen clean steel raises false alarms on it');
  s.addChart(pres.charts.BAR, [
    { name: 'Trained on defects only', labels: ['Clean strip flagged %', 'Defects caught %', 'Clean vs defect separation'], values: [93.7, 72.0, 60.8] },
    { name: '+ clean strip added', labels: ['Clean strip flagged %', 'Defects caught %', 'Clean vs defect separation'], values: [32.5, 86.6, 95.8] },
  ], {
    x: 0.6, y: 1.7, w: 3.7, h: 1.75, barDir: 'bar', barGrouping: 'clustered', chartColors: ['9AA7B4', C.accent],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 9, dataLabelColor: C.ink, dataLabelFontFace: F,
    catAxisLabelFontSize: 9, catAxisLabelColor: C.slate, catAxisLabelFontFace: F, valAxisHidden: true, valAxisMaxVal: 110,
    valGridLine: { style: 'none' }, catGridLine: { style: 'none' }, catAxisOrientation: 'maxMin', dataLabelFormatCode: '0.0', showLegend: true, legendPos: 'b', legendFontSize: 9, legendFontFace: F,
  });
  T(s, 'On 505 certified-clean steel images, a model trained only on defects flagged 93.7% as defective. Adding real clean strip to training cut that to 32.5%, and accuracy went up, not down. Training on clean strip alone fails: it stops finding defects.', { x: 4.4, y: 1.72, w: 2.1, h: 1.8, fontSize: 9.5, color: C.slate });
  so(0, 0, 'train on JSL\'s own clean and defective strip before any alarm is trusted.');

  // Q2 bigger not better
  quad(1, 0, '2  A bigger model or sharper images did not help');
  table(s, [
    [hd('Option tested'), hd('Accuracy'), hd('Extra cost'), hd('Verdict')],
    [cl('Compact model'), cl('0.752'), cl('-'), cl('Chosen', { color: C.green, bold: true })],
    [cl('Compact model + clean strip'), cl('0.764'), cl('none'), cl('Ready', { color: C.green, bold: true })],
    [cl('Model 3.7x larger'), cl('0.734'), cl('3.5x compute'), cl('No gain', { color: C.amber, bold: true })],
    [cl('4x sharper input images'), cl('0.734'), cl('4x pixels'), cl('No gain', { color: C.amber, bold: true })],
    [cl('Extra processing per image'), cl('0.006 lower'), cl('2.4x slower'), cl('Rejected', { color: C.red, bold: true })],
  ], { x: 6.87, y: 1.72, w: 5.86, colW: [2.5, 0.9, 1.36, 1.1], rowH: 0.245 });
  T(s, 'Accuracy on held-out images. The gap between compact and large models is within noise.', { x: 6.87, y: 3.22, w: 5.86, h: 0.2, fontSize: 9, color: C.muted });
  so(1, 0, 'keep the model compact and spend the budget on camera coverage instead.');

  // Q3 throughput
  quad(0, 1, '3  Line speed, not the model, sets the hardware bill');
  T(s, '~18', { x: 0.65, y: 4.35, w: 2.7, h: 0.6, fontSize: 36, bold: true, color: C.navy });
  T(s, 'processors needed to see every 0.2 mm of a 1.28 m wide strip moving at 250 m/min', { x: 0.65, y: 4.95, w: 2.75, h: 0.95, fontSize: 9.5, color: C.slate });
  T(s, '1', { x: 3.6, y: 4.35, w: 2.7, h: 0.6, fontSize: 36, bold: true, color: C.muted });
  T(s, 'processor is enough if images are coarsened to 1.28 mm, but fine scratches and crazing then disappear', { x: 3.6, y: 4.95, w: 3.0, h: 0.95, fontSize: 9.5, color: C.slate });
  so(0, 1, 'fix the image resolution with JSL at the start, and size the hardware from it.');

  // Q4 trust
  quad(1, 1, '4  Operators only act on alarms they can trust');
  const tr = [
    ['68% lower', 'confidence error after calibration. Before, an alarm marked "60% sure" was right 84% of the time; now stated confidence is within ~5 points of reality'],
    ['0 of 3,200', 'real steel images wrongly rejected by our "is this steel?" check, which stops the system judging anything else (unchecked, it called a plain white square a defect)'],
  ];
  tr.forEach(([n, l], k) => {
    T(s, n, { x: 6.92, y: 4.35 + k * 0.78, w: 2.2, h: 0.6, fontSize: 20, bold: true, color: C.navy, valign: 'middle' });
    T(s, l, { x: 9.15, y: 4.35 + k * 0.78, w: 3.55, h: 0.72, fontSize: 9.5, color: C.slate, valign: 'middle' });
  });
  so(1, 1, 'give every alarm a trustworthy confidence and a "this is steel" check.');
  s.addNotes('Four findings from our own tests, each of which changed the design. One: a model trained only on defect images treats almost all clean steel as defective, 93.7%. Adding real clean strip cut that to 32.5% and accuracy went up. Two: a bigger model and sharper images did not improve accuracy, so the money goes into cameras, not model size. Three: at full detail a 250 m/min line needs about 18 processors; resolution is therefore a spec decision we make with JSL on day one. Four: operators act only on alarms they trust, so every alarm carries a calibrated confidence and passes an is-this-steel check.');
}

// ============ SLIDE 5: SOLUTION ============
{
  const s = pres.addSlide();
  chrome(s, 'Solution',
    'Detect on the strip, decide at the coil, route to the owner, and learn from every operator correction',
    'Buy proven optics, own the model. JSL gets vendor-grade capture with a detector trained on its own grades and wired to its own process owners.',
    'BUILT = working in our prototype today, live at surface-vision.github.io. Unlabelled-learning result: J. Mater. Inf. 2025. Commercial price range: industry figure, unverified.');
  ph(s, 0.5, 1.35, 12.33, 'Architecture: six stages from camera to corrective action');
  const stg = [
    ['1 Capture', 'Cameras on both faces of the strip, lit to reveal fine scratches, down to 0.2 mm', 'TO BUILD', C.muted],
    ['2 Check', 'Confirms the image is usable steel before judging it', 'BUILT', C.green],
    ['3 Detect', 'Finds and classifies every defect across the full strip width', 'BUILT', C.green],
    ['4 Score', 'Trustworthy confidence, severity 0-100, defect type and position on the coil map', 'BUILT', C.green],
    ['5 Decide', 'Coil ACCEPT / DOWNGRADE / HOLD, with limits set per grade and finish', 'PARTIAL', C.amber],
    ['6 Act & learn', 'Operator alert, coil hold, owner work order; corrections retrain it in under an hour', 'TO BUILD', C.muted],
  ];
  const sw = 1.95;
  stg.forEach(([h, b, t, col], i) => {
    const x = 0.5 + i * (sw + 0.126);
    RR(s, x, 1.75, sw, 1.45, i < 5 ? C.neutral : 'E4EEF6');
    T(s, h, { x: x + 0.1, y: 1.8, w: sw - 0.2, h: 0.3, fontSize: 12, bold: true, color: C.navy });
    T(s, b, { x: x + 0.1, y: 2.1, w: sw - 0.2, h: 0.8, fontSize: 9.5, color: C.slate });
    tag(s, x + 0.1, 2.92, t, col, 0.85);
    if (i < 5) T(s, '>', { x: x + sw, y: 2.25, w: 0.126, h: 0.3, fontSize: 14, bold: true, color: C.accent, align: 'center' });
  });

  // innovations
  ph(s, 0.5, 3.4, 5.6, 'What is new beyond a standard detector');
  const inn = [
    ['Trained on clean strip, not just defects', 'Fixes the false-alarm problem that kills most pilots. Measured: 93.7% to 32.5%.'],
    ['Root-cause routing', 'Each class maps to an owner (slide 3), so an alarm becomes a work order, not just a red light.'],
    ['Catches defects it was never taught', 'A second check flags unusual surface patterns that were never labelled. Proposed.'],
    ['Learns from JSL\'s unlabelled coils', 'Published steel research shows unlabelled images train as well as the standard approach, so JSL\'s image archive becomes an asset.'],
  ];
  inn.forEach(([h, b], i) => {
    const y = 3.8 + i * 0.66;
    dot(s, 0.55, y + 0.02, 0.26, C.accent);
    T(s, String(i + 1), { x: 0.55, y: y + 0.02, w: 0.26, h: 0.26, fontSize: 10, bold: true, color: C.white, align: 'center', valign: 'middle' });
    T(s, h, { x: 0.9, y, w: 5.2, h: 0.26, fontSize: 10.5, bold: true, color: C.navy });
    T(s, b, { x: 0.9, y: y + 0.26, w: 5.2, h: 0.38, fontSize: 9.5, color: C.slate });
  });

  // build vs buy
  ph(s, 6.35, 3.4, 6.48, 'Framework: build vs buy (options scored against JSL\'s needs)');
  const crit = ['Capex per line', 'Proven at line speed', 'Tuned to JSL grades', 'Root-cause routing', 'JSL owns data and model', 'Time to first line'];
  const opts = [
    ['Commercial (ISRA, AMETEK)', ['r', 'g', 'a', 'a', 'r', 'g'], 'USD 1-3M (unverified)'],
    ['Fully in-house', ['g', 'r', 'g', 'g', 'g', 'r'], 'INR 2-3 cr (est.)'],
    ['Hybrid: buy optics, own model', ['g', 'a', 'g', 'g', 'g', 'a'], 'INR 2.5-4.2 cr (est.)'],
  ];
  const cx0 = 8.35, cwid = 0.745;
  crit.forEach((c, k) => T(s, c, { x: cx0 + k * cwid, y: 3.76, w: cwid - 0.04, h: 0.42, fontSize: 9, bold: true, color: C.slate, align: 'center', valign: 'bottom' }));
  const colmap = { g: C.green, a: C.amber, r: C.red };
  opts.forEach(([n, sc, cost], j) => {
    const y = 4.28 + j * 0.56;
    if (j === 2) RR(s, 6.38, y - 0.07, 6.42, 0.54, 'E4EEF6');
    T(s, n, { x: 6.45, y: y - 0.02, w: 1.9, h: 0.24, fontSize: 9.5, bold: true, color: j === 2 ? C.accent : C.ink });
    T(s, cost, { x: 6.45, y: y + 0.22, w: 1.85, h: 0.2, fontSize: 9, color: C.muted });
    sc.forEach((v, k) => dot(s, cx0 + k * cwid + (cwid - 0.04) / 2 - 0.1, y + 0.08, 0.2, colmap[v]));
  });
  dot(s, 6.45, 5.99, 0.14, C.green); T(s, 'strong', { x: 6.63, y: 5.96, w: 0.6, h: 0.2, fontSize: 9, color: C.muted });
  dot(s, 7.25, 5.99, 0.14, C.amber); T(s, 'partial', { x: 7.43, y: 5.96, w: 0.6, h: 0.2, fontSize: 9, color: C.muted });
  dot(s, 8.05, 5.99, 0.14, C.red); T(s, 'weak', { x: 8.23, y: 5.96, w: 0.6, h: 0.2, fontSize: 9, color: C.muted });
  T(s, 'Output: hybrid wins on 4 of 6 criteria and is never weak.', { x: 8.9, y: 5.94, w: 3.9, h: 0.24, fontSize: 9.5, bold: true, color: C.navy });
  s.addNotes('Solution. Stages 2 to 4 are built and running today, including in the browser demo. Stage 5 has coil rules built, but grade- and finish-specific limits need JSL input. Stages 1 and 6 are plant-side and are the pilot work. What makes this more than a standard detector: it is trained on clean strip as well as defects, every defect is routed to the process that caused it, a second check catches defects it was never taught, and it can learn from JSL\'s unlabelled image archive. Build vs buy: commercial systems are proven but cost USD 1-3M per line and keep the model closed. Fully in-house is cheapest but slowest to prove. Hybrid, meaning industrial cameras plus our model, is never weak on any criterion.');
}

// ============ SLIDE 6: IMPLEMENTATION PLAN ============
{
  const s = pres.addSlide();
  chrome(s, 'Implementation',
    'Seven months to shadow mode on one line, and nothing touches the mill until trust is earned',
    'Commit INR 2.5-4.2 cr and one line for seven months. The month-7 go/no-go is decided on JSL\'s own alarm and recall numbers.',
    'Costs are team estimates for a 1.3-1.6 m line inspected on both faces, at INR 95.8/USD; to be replaced by vendor quotes in P0.');
  // gantt
  ph(s, 0.5, 1.35, 7.6, 'Stage-gate roadmap (months)');
  const gx = 2.75, gw = 3.6;
  [0, 4, 8, 12, 16, 20].forEach(m => T(s, `m${m}`, { x: gx + gw * m / 20 - 0.2, y: 1.7, w: 0.4, h: 0.2, fontSize: 9, color: C.muted, align: 'center' }));
  const P = [
    ['P0', 'Instrument one line', 0, 1.5, 'Frames at line speed; resolution and grade matrix fixed'],
    ['P1', 'Fine-tune on JSL strip', 1.5, 4, 'Clean-strip alarms <= 25%, recall >= 90%'],
    ['P2', 'Shadow mode: advises only', 4, 7, '>= 90% operator agreement over 4 weeks'],
    ['P3', 'Auto-hold on severe defects', 7, 12, 'Holds trusted; downgrade rate measured'],
    ['P4', 'Second line and grade', 12, 20, 'Fleet rollout decision on real payback'],
  ];
  P.forEach(([c, n, a, b, gate], i) => {
    const y = 1.95 + i * 0.5;
    T(s, c, { x: 0.55, y, w: 0.35, h: 0.4, fontSize: 11, bold: true, color: C.accent, valign: 'middle' });
    T(s, n, { x: 0.9, y, w: 1.8, h: 0.4, fontSize: 10, color: C.ink, valign: 'middle' });
    R(s, gx, y + 0.08, gw, 0.24, C.neutral);
    R(s, gx + gw * a / 20, y + 0.08, gw * (b - a) / 20, 0.24, i < 3 ? C.accent : C.navy2);
    T(s, gate, { x: 6.45, y, w: 1.65, h: 0.42, fontSize: 9, color: C.slate, valign: 'middle' });
  });
  R(s, gx + gw * 7 / 20 - 0.01, 1.95, 0.02, 2.45, C.red);
  T(s, 'pilot go / no-go', { x: gx + gw * 7 / 20 + 0.05, y: 4.4, w: 1.4, h: 0.2, fontSize: 9, bold: true, color: C.red });
  T(s, 'EXIT GATE', { x: 6.55, y: 1.7, w: 1.6, h: 0.2, fontSize: 9, bold: true, color: C.muted });

  // BOM
  ph(s, 8.35, 1.35, 4.48, 'Cost per line, estimated (INR lakh)');
  table(s, [
    [hd('Item'), hd('INR lakh')],
    [cl('Cameras and lighting'), cl('85-145')],
    [cl('Computing hardware'), cl('45-75')],
    [cl('Installation and plant integration'), cl('90-150')],
    [cl('Training on JSL\'s own strip'), cl('30-50')],
    [cl('Total, one line', { bold: true }), cl('250-420', { bold: true, color: C.accent })],
    [cl('Running cost, per year'), cl('~50')],
  ], { x: 8.35, y: 1.7, w: 4.48, colW: [3.38, 1.1], rowH: 0.36, fontSize: 10.5 });

  // plant connection, in rollout order
  ph(s, 0.5, 4.7, 12.33, 'Plant connection: read-only first, control last');
  const L = [
    ['Process data', 'Coil ID, grade, gauge and speed, so each defect is tied to the heat it came from', 'from P0'],
    ['Quality records', 'Coil disposition and a defect map that can travel with the customer certificate', 'from P2'],
    ['Line control', 'Slow or hold the line on severe defects. The only write path, switched on last', 'from P3'],
  ];
  const lw = (12.33 - 2 * 0.3) / 3;
  L.forEach(([h, b, p], i) => {
    const x = 0.5 + i * (lw + 0.3);
    RR(s, x, 5.08, lw, 1.3, i === 2 ? 'E4EEF6' : C.neutral);
    T(s, h, { x: x + 0.15, y: 5.16, w: lw - 1.3, h: 0.32, fontSize: 12, bold: true, color: C.navy, valign: 'middle' });
    tag(s, x + lw - 1.05, 5.22, p, i === 2 ? C.navy : C.accent, 0.9);
    T(s, b, { x: x + 0.15, y: 5.55, w: lw - 0.3, h: 0.7, fontSize: 10.5, color: C.slate });
    if (i < 2) T(s, '>', { x: x + lw, y: 5.55, w: 0.3, h: 0.3, fontSize: 16, bold: true, color: C.accent, align: 'center' });
  });
  s.addNotes('Implementation. Five phases, 20 months, with a hard go/no-go at month 7 after shadow mode. The system never acts on the mill until P3; before that it only reads data and advises. Cost is our estimate, INR 2.5-4.2 crore per line, against a commercial range of USD 1-3M, about INR 9.6-28.7 crore. Vendor quotes replace the estimate in P0. The plant connection goes in order: process data first, quality records next, line control last.');
}

// ============ SLIDE 7: FEASIBILITY & RISK ============
{
  const s = pres.addSlide();
  chrome(s, 'Implementation',
    'Every high risk is retired in P0-P2, before the system is allowed to slow the mill',
    'Accuracy, speed and trust are already proven off-site. What only JSL\'s line can prove is measured in shadow mode first.',
    'Evidence from our own tests on public steel datasets. KPI targets are proposed, to be agreed with JSL Quality in P0.');
  // risk heat grid
  ph(s, 0.5, 1.35, 3.6, 'Risk map (likelihood x impact)');
  const gx = 1.1, gy = 1.8, cs = 0.74;
  const heat = [[C.h2, C.h3, C.h4], [C.h1, C.h2, C.h3], [C.neutral, C.h1, C.h2]];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R(s, gx + c * cs, gy + r * cs, cs - 0.04, cs - 0.04, heat[r][c]);
  T(s, 'LIKELIHOOD', { x: 0.05, y: 2.8, w: 1.0, h: 0.2, fontSize: 9, bold: true, color: C.muted, rotate: 270 });
  ['High', 'Med', 'Low'].forEach((l, r) => T(s, l, { x: 0.66, y: gy + r * cs + 0.26, w: 0.4, h: 0.2, fontSize: 9, color: C.muted }));
  ['Low', 'Med', 'High'].forEach((l, c) => T(s, l, { x: gx + c * cs, y: gy + 3 * cs, w: cs, h: 0.2, fontSize: 9, color: C.muted, align: 'center' }));
  T(s, 'IMPACT', { x: gx, y: gy + 3 * cs + 0.2, w: 3 * cs, h: 0.2, fontSize: 9, bold: true, color: C.muted, align: 'center' });
  // placements [id,row,col,offset]
  const pl = [[1, 0, 2, 0], [2, 1, 2, 0], [3, 0, 1, 0], [4, 1, 1, 0], [5, 1, 1, 1], [6, 2, 1, 0], [7, 1, 0, 0]];
  pl.forEach(([id, r, c, o]) => {
    const x = gx + c * cs + 0.06 + o * 0.34, y = gy + r * cs + 0.19;
    dot(s, x, y, 0.3, C.navy);
    T(s, String(id), { x, y, w: 0.3, h: 0.3, fontSize: 11, bold: true, color: C.white, align: 'center', valign: 'middle' });
  });

  // risk table
  ph(s, 4.35, 1.35, 8.48, 'Risks, evidence and how each is retired');
  table(s, [
    [hd('#'), hd('Risk'), hd('Evidence'), hd('Mitigation'), hd('Retired in')],
    [cl('1', { bold: true }), cl('Model does not transfer to JSL strip'), cl('93.7% false alarms before the fix'), cl('Train on JSL clean and defect images'), cl('P1')],
    [cl('2', { bold: true }), cl('Low-contrast defects missed'), cl('Rolled-in scale is our weakest class'), cl('Angled lighting; sharper images'), cl('P0-P1')],
    [cl('3', { bold: true }), cl('Rare defects under-learnt'), cl('Few roll-mark or edge-crack examples'), cl('Edge cameras; targeted labelling'), cl('P1-P3')],
    [cl('4', { bold: true }), cl('Operators ignore alarms'), cl('Raw confidence was over-confident'), cl('Calibrated scores; shadow mode; override log'), cl('P2')],
    [cl('5', { bold: true }), cl('Not enough computing at line speed'), cl('~18 processors needed at full detail'), cl('Test target hardware on real frames'), cl('P0')],
    [cl('6', { bold: true }), cl('Heat, dust, acid fumes'), cl('Pickling and HSM environments'), cl('Sealed enclosures, air knives, cooling'), cl('P0')],
    [cl('7', { bold: true }), cl('Plant IT integration slips'), cl('Plant IT change control'), cl('Read-only until P3; one write path'), cl('P3')],
  ], { x: 4.35, y: 1.67, w: 8.48, colW: [0.3, 2.2, 2.45, 2.73, 0.8], rowH: 0.3 });

  // KPIs
  ph(s, 0.5, 4.65, 12.33, 'Pilot KPIs: how JSL will know it works');
  const k = [
    ['>= 90%', 'defective strip flagged', 'measured 92.2% in our tests'],
    ['<= 25%', 'clean strip wrongly flagged', 'measured 32.5% in our tests'],
    ['<= 5 pts', 'gap between stated and real confidence', 'measured 4.6 pts'],
    ['< 0.2 s', 'from defect to operator screen', 'design target'],
    ['>= 90%', 'operator agreement, shadow mode', 'measured in P2'],
    ['< 8 h', 'to retrain for a new line or grade', 'measured 0.6 h'],
  ];
  k.forEach(([n, l, m], i) => {
    const x = 0.5 + i * 2.07;
    RR(s, x, 5.02, 1.97, 1.43, C.neutral);
    T(s, n, { x: x + 0.1, y: 5.08, w: 1.8, h: 0.48, fontSize: 22, bold: true, color: C.navy });
    T(s, l, { x: x + 0.1, y: 5.56, w: 1.8, h: 0.45, fontSize: 9.5, color: C.ink });
    T(s, m, { x: x + 0.1, y: 6.03, w: 1.8, h: 0.36, fontSize: 9, color: i === 1 ? C.amber : C.muted, bold: i === 1 });
  });
  s.addNotes('Feasibility and risk. The top-right of the map is transfer to JSL strip and low-contrast defects; both are retired in P0 and P1, before money is spent on integration. The honest gap is clean strip wrongly flagged: we measured 32.5% against a 25% target, which is exactly why P1 trains on JSL\'s own strip. Rare defects like roll marks and edge cracks need JSL data and edge cameras; the unseen-defect check covers what is never labelled.');
}

// ============ SLIDE 8: IMPACT ============
{
  const s = pres.addSlide();
  chrome(s, 'Impact',
    'At a conservative 1% downgrade rate one line pays back in ~2.2 years; the fleet protects ~INR 13 cr a year',
    'Recommendation: approve a 7-month pilot on one 300-series cold-rolled line. The number that decides fleet rollout is JSL\'s own downgrade rate.',
    'Inputs: JSL FY26 (INR 1,67,407/t; 25,65,902 t). Swept assumptions: 0.4 Mt line, 30% recovered, INR 3.35 cr capex, INR 0.5 cr/yr run cost, 12% rate. Commercial: USD 2M at INR 95.8. MD quote: ET, 16 Jul 2024.');
  // heatmap
  ph(s, 0.5, 1.35, 4.2, 'Loss pool per line, INR cr/yr');
  T(s, '0.4 Mt x downgrade rate x price discount x INR 1,67,407/t', { x: 0.5, y: 1.7, w: 4.2, h: 0.25, fontSize: 9.5, italic: true, color: C.slate });
  const grid = [[3.3, 5.0, 6.7], [6.7, 10.0, 13.4], [13.4, 20.1, 26.8]];
  const shade = v => v < 5 ? [C.h1, C.ink] : v < 10 ? [C.h2, C.ink] : v < 20 ? [C.h3, C.white] : [C.h4, C.white];
  const hrow = [hd('Downgraded'), hd('10% off', { align: 'center' }), hd('15% off', { align: 'center' }), hd('20% off', { align: 'center' })];
  const trows = [hrow];
  ['0.5% of volume', '1.0% of volume', '2.0% of volume'].forEach((lab, r) => {
    trows.push([cl(lab, { bold: true })].concat(grid[r].map((v, c) => {
      const [f, t] = shade(v);
      const base = r === 1 && c === 0;
      return cl(v.toFixed(1), { align: 'center', bold: true, fill: { color: f }, color: t, fontSize: base ? 12 : 10.5, border: base ? { type: 'solid', pt: 2, color: C.red } : undefined });
    })));
  });
  table(s, trows, { x: 0.5, y: 2.0, w: 4.2, colW: [1.35, 0.95, 0.95, 0.95], rowH: 0.34 });
  T(s, 'Outlined: base case, 6.7 cr pool. Recovering 30% = INR 2.0 cr/yr, 1.5 cr net of run cost. Fleet at the same rates (2.57 Mt, ~6.4 lines): INR 43 cr pool, INR 12.9 cr/yr recovered.', { x: 0.5, y: 3.4, w: 4.2, h: 0.75, fontSize: 9.5, color: C.slate });

  // tornado
  ph(s, 4.95, 1.35, 4.35, 'What moves payback (years, base 2.2)');
  const tor = [
    ['Downgrade rate 0.5-2%', 0.95, 6.64],
    ['Share of loss recovered 15-45%', 1.33, 6.64],
    ['Line volume 0.3-0.5 Mt', 1.67, 3.33],
    ['Capex INR 2.5-4.2 cr', 1.66, 2.78],
    ['Price discount 10-20%', 0.95, 2.22],
  ];
  const ax = 7.0, aw = 2.15, max = 7, base = 2.22;
  const X = v => ax + aw * v / max;
  tor.forEach(([n, lo, hi], i) => {
    const y = 1.85 + i * 0.44;
    T(s, n, { x: 5.0, y, w: 1.95, h: 0.34, fontSize: 9, color: C.ink, valign: 'middle' });
    R(s, X(lo), y + 0.07, X(base) - X(lo), 0.2, C.accent);
    R(s, X(base), y + 0.07, Math.max(X(hi) - X(base), 0.01), 0.2, '9AA7B4');
    T(s, `${lo.toFixed(1)}-${hi.toFixed(1)}`, { x: X(hi) + 0.04 > 8.9 ? 8.62 : X(hi) + 0.04, y: y - 0.08, w: 0.6, h: 0.16, fontSize: 9, color: C.muted });
  });
  R(s, X(base) - 0.01, 1.8, 0.02, 2.25, C.navy);
  T(s, 'Downgrade rate and recovery share dominate. Neither is public, so both are measured in P2-P3 before fleet spend.', { x: 5.0, y: 4.05, w: 4.3, h: 0.45, fontSize: 9.5, color: C.slate });

  // hybrid vs commercial
  ph(s, 9.55, 1.35, 3.28, 'Base-case payback');
  const pb = [['Hybrid (ours)', 2.2, C.accent], ['Commercial system', 12.7, '9AA7B4']];
  pb.forEach(([n, v, col], i) => {
    const y = 1.85 + i * 0.72;
    T(s, n, { x: 9.6, y, w: 3.2, h: 0.22, fontSize: 9.5, bold: true, color: C.ink });
    R(s, 9.6, y + 0.25, 2.5 * v / 12.7, 0.26, col);
    T(s, `${v} yr`, { x: 9.6 + 2.5 * v / 12.7 + 0.05, y: y + 0.24, w: 0.7, h: 0.28, fontSize: 11, bold: true, color: C.navy, valign: 'middle' });
  });
  T(s, [
    { text: '+INR 2.1 cr ', options: { bold: true, color: C.navy, fontSize: 13 } },
    { text: '5-year NPV per line at 12%\n', options: { fontSize: 9.5, color: C.slate } },
    { text: '>= 0.11% ', options: { bold: true, color: C.navy, fontSize: 13 } },
    { text: 'of line revenue lost to downgrade (rate x discount) is enough for a 2-year payback', options: { fontSize: 9.5, color: C.slate } },
  ], { x: 9.6, y: 3.35, w: 3.2, h: 1.15 });

  // non-financial
  ph(s, 0.5, 4.6, 7.9, 'Impact beyond the P&L');
  const nf = [
    ['Coverage', '100% of both faces, the same rule on every coil, instead of sampled grading'],
    ['Customers', 'Defect map per coil for auto, rail and appliance buyers; fewer claims on exposed finishes'],
    ['Process', 'Typed defects tied to heat and coil IDs give the SMS and HSM root-cause data they lack today'],
    ['Safety and assets', 'Severe-defect holds guard against strip breaks, the Ternium business case'],
    ['Sustainability', 'Less remelt and re-pickling of downgraded coils saves energy, acid and nickel'],
    ['People', '"AI will not replace; it will instead renew and re-energise." Inspectors move to decisions (JSL MD)'],
  ];
  nf.forEach(([h, b], i) => {
    const c = i % 2, r = Math.floor(i / 2);
    const x = 0.55 + c * 3.95, y = 4.98 + r * 0.52;
    T(s, h, { x, y, w: 1.2, h: 0.46, fontSize: 10, bold: true, color: C.accent });
    T(s, b, { x: x + 1.2, y, w: 2.65, h: 0.48, fontSize: 9, color: C.slate });
  });

  // the ask
  RR(s, 8.65, 4.6, 4.18, 1.88, C.navy);
  T(s, 'WHAT WE ASK OF JSL', { x: 8.8, y: 4.68, w: 3.9, h: 0.28, fontSize: 11, bold: true, color: C.onNavy });
  T(s, [
    { text: 'One 300-series cold-rolled line for 7 months', options: { bullet: true, breakLine: true } },
    { text: 'INR 2.5-4.2 cr pilot budget', options: { bullet: true, breakLine: true } },
    { text: 'Downgrade rate by line and grade from the MIS', options: { bullet: true, breakLine: true } },
    { text: 'A metallurgist at 20% to sign off root causes', options: { bullet: true, breakLine: true } },
    { text: 'In return: measured downgrade rate, payback and a go / no-go at month 7', options: { bold: true, color: C.onNavy } },
  ], { x: 8.8, y: 5.0, w: 3.95, h: 1.4, fontSize: 10.5, color: C.white, paraSpaceAfter: 4 });
  s.addNotes('Impact. We scoped the business case to one line, which is where the capex is spent. At a conservative 1% of volume downgraded at a 10% discount, one 0.4 Mt line loses INR 6.7 crore a year. Recovering 30% of it, less INR 0.5 crore run cost, pays back INR 3.35 crore in about 2.2 years; a commercial system at the same benefit takes about 12.7 years. That is why the hybrid build matters. The tornado shows the two numbers that dominate are the downgrade rate and how much we recover, and neither is public. So the ask is not only budget; it is JSL\'s own downgrade data, and the pilot is designed to measure it. Across the fleet, the same rates protect about INR 13 crore a year.');
}

pres.writeFile({ fileName: '../out/JSW_Round2_Surface_Defect_Detection.pptx' }).then(f => console.log('wrote', f));
