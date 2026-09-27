// Round 2 deck (v3, compact): Stainless Spark, PS1, AI surface defect detection
// Assets first:  node gen_assets.js  and  ../../.venv/bin/python gen_detections.py  and  ../../.venv/bin/python gen_qr.py
const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

// Our own stainless photos (Round2/stainless_test). The proof slide gains a fourth dataset card
// only when run_stainless_test.py has produced results; otherwise the slide is unchanged.
const ST_DIR = process.env.STAINLESS_DIR || path.join(__dirname, '..', 'stainless_test', 'out');
const STAINLESS = fs.existsSync(path.join(ST_DIR, 'summary.json')) ? (() => {
  const sm = JSON.parse(fs.readFileSync(path.join(ST_DIR, 'summary.json'), 'utf8'));
  const rs = JSON.parse(fs.readFileSync(path.join(ST_DIR, 'results.json'), 'utf8'));
  const pick = rs.find((r) => r.defective && r.flagged) || rs[0];
  return { sm, img: path.join(ST_DIR, 'annotated', pick.file.replace(/\.[^.]+$/, '.jpg')) };
})() : null;

// Interview quotes (Round2/interviews/quotes.json). Only entries with consent and a quote are used.
// With quotes, slide 2 gains a "Voices from the plant floor" band and slide 3 marks what they validated.
const QUOTES_FILE = process.env.QUOTES_FILE || path.join(__dirname, '..', 'interviews', 'quotes.json');
const QUOTES = (() => {
  try { return JSON.parse(fs.readFileSync(QUOTES_FILE, 'utf8')).filter((q) => q.consent && q.quote && q.quote.trim()); } catch (e) { return []; }
})();
const who = (q) => (q.consent === 'name' && q.name ? `${q.name}, ${q.role}` : q.role);
const validatedBy = (what) => QUOTES.find((q) => (q.validates || []).includes(what));
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5
pres.title = 'Stainless Spark Round 2 - AI Surface Defect Detection';

// ---------------------------------------------------------------------------
// Team details. Leave TEAM empty if there is no team name: the header then shows the institution.
const TEAM = 'Futuristic';
const MEMBERS = ['Krishna Kanta Mondal', 'Prathmesh Walimbe'];
const INSTITUTION = 'IIT Bombay';
// ---------------------------------------------------------------------------

// Jindal Stainless scheme, sampled from the team's Round 1 slide: orange band and panel headers,
// red active tab, charcoal rail, tables and decision band. 'navy'/'accent' keep their roles
// (dark and emphasis); green, amber and red are reserved for status.
const C = {
  navy: '2B2B2B', navy2: '5C5C5C', onNavy: 'D6D6D6', accent: 'D2651F',
  ink: '1A1A1A', slate: '404040', muted: '6E6E6E', neutral: 'F3F3F3', edge: 'D9D9D9',
  white: 'FFFFFF', green: '2E7D32', amber: 'C99700', red: 'C62828', grey: '9E9E9E',
  h1: 'FCE5D3', h2: 'F5BE92', h3: 'EC9352', h4: 'C4561A', tint: 'FDEFE4',
  band: 'E4803A', head: 'E4803A', tab: 'B63831', tabOff: 'F2F2F2', rail: '2B2B2B',
};
const RAIL = ['PROBLEM  ·  WHY NOW', 'INSIGHTS  ·  OWNERS  ·  PRECEDENT', 'INSIGHTS  ·  DESIGN CHOICES', 'SOLUTION  ·  HOW IT WORKS',
  'SOLUTION  ·  PROOF', 'IMPLEMENTATION  ·  ROADMAP  ·  RISK', 'IMPACT  ·  VALUE  ·  ASK'];
let railIdx = 0;
const F = 'Calibri';
const SECTIONS = ['Problem', 'Insights', 'Solution', 'Implementation', 'Impact'];
const ICON = (n, c) => `img/icons/${n}-${c || 'navy'}.png`;

function T(s, text, o) {
  s.addText(text, Object.assign({ fontFace: F, isTextBox: true, margin: 0, valign: 'top', color: C.ink, fontSize: 10 }, o));
}
function R(s, x, y, w, h, fill, o) {
  s.addShape(pres.shapes.RECTANGLE, Object.assign({ x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 } }, o || {}));
}
function RR(s, x, y, w, h, fill, o) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: fill, width: 0 } }, o || {}));
}
function dot(s, x, y, d, col) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { color: col, width: 0 } });
}
// icon inside a filled circle
function badge(s, name, x, y, d, fill) {
  dot(s, x, y, d, fill || C.navy);
  const p = d * 0.22;
  s.addImage({ path: ICON(name, 'white'), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}
function photo(s, pth, x, y, w, h) {
  s.addImage({ path: pth, x, y, w, h, sizing: { type: 'cover', w, h } });
}
function icon(s, name, x, y, d, col) { s.addImage({ path: ICON(name, col), x, y, w: d, h: d }); }
// orange header band with both logos; used by the cover and every content slide
function brandBand(s) {
  R(s, 0, 0, 13.333, 0.86, C.band);
  s.addImage({ path: 'img/logo_jsl.jpg', x: 0.1, y: 0.08, w: 2.38, h: 0.7 });
  s.addImage({ path: 'img/logo_spark.png', x: 2.56, y: 0.08, w: 1.36, h: 0.7 });
}
function chrome(s, active, title, band, source) {
  s.background = { color: C.white };
  brandBand(s);
  const x0 = 4.14, tw = 1.43, gap = 0.06;
  SECTIONS.forEach((sec, i) => {
    const on = sec === active;
    R(s, x0 + i * (tw + gap), 0.2, tw, 0.46, on ? C.tab : C.tabOff);
    T(s, `${i + 1}  ${sec}`, { x: x0 + i * (tw + gap), y: 0.2, w: tw, h: 0.46, align: 'center', valign: 'middle', fontSize: 10.5, bold: on, color: on ? C.white : C.navy2 });
  });
  T(s, [{ text: TEAM ? `Team ${TEAM}` : '', options: { bold: true, fontSize: 11 } }, { text: `\n${INSTITUTION}`, options: { fontSize: 9.5 } }],
    { x: 11.62, y: 0.16, w: 1.62, h: 0.54, align: 'right', valign: 'middle', color: C.white });
  T(s, title, { x: 0.5, y: 0.9, w: 12.33, h: 0.42, fontSize: 20, bold: true, color: C.ink, valign: 'middle' });
  // left rail with the slide's mode, rotated
  R(s, 0.1, 1.35, 0.3, 5.43, C.rail);
  T(s, RAIL[railIdx++] || active.toUpperCase(), { x: 0.25 - 5.43 / 2, y: 1.35 + 5.43 / 2 - 0.15, w: 5.43, h: 0.3, rotate: 270, fontSize: 10, bold: true, color: C.white, align: 'center', valign: 'middle', charSpacing: 1 });
  R(s, 0.1, 6.82, 12.73, 0.46, C.rail);
  T(s, band, { x: 0.28, y: 6.82, w: 12.4, h: 0.46, fontSize: 12, bold: true, color: C.white, valign: 'middle' });
  if (source) T(s, source, { x: 0.5, y: 6.53, w: 12.33, h: 0.26, fontSize: 9, color: C.muted, valign: 'middle' });
}
function ph(s, x, y, w, text) {
  R(s, x, y, w, 0.3, C.head);
  T(s, text, { x: x + 0.1, y, w: w - 0.2, h: 0.3, fontSize: 11, bold: true, color: C.white, valign: 'middle' });
}
function tag(s, x, y, text, col, w) {
  RR(s, x, y, w || 0.85, 0.22, col);
  T(s, text, { x, y, w: w || 0.85, h: 0.22, fontSize: 9, bold: true, color: C.white, align: 'center', valign: 'middle' });
}
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ fontFace: F, fontSize: 9.5, color: C.ink, border: { type: 'solid', pt: 0.5, color: C.edge }, margin: [0.03, 0.05, 0.03, 0.05], valign: 'middle' }, o));
}
const hd = (t, o) => ({ text: t, options: Object.assign({ bold: true, color: C.white, fill: { color: C.rail } }, o || {}) });
const cl = (t, o) => ({ text: t, options: o || {} });
const DET = [
  ['inclusion', 'Inclusion'], ['crazing', 'Crazing'], ['rolled-in_scale', 'Rolled-in scale'],
  ['pitted_surface', 'Pitted surface'], ['patches', 'Patches'], ['scratches', 'Scratches'],
];

// ============ SLIDE 1: COVER ============
{
  const s = pres.addSlide();
  s.background = { color: C.rail };
  s.addImage({ path: 'img/cover_coil.jpg', x: 8.55, y: 0.86, w: 4.783, h: 6.64, sizing: { type: 'cover', w: 4.783, h: 6.64 } });
  R(s, 8.55, 0.86, 4.783, 6.64, C.rail, { fill: { color: C.rail, transparency: 40 } });
  R(s, 8.55, 0.86, 0.06, 6.64, C.band);
  brandBand(s);
  T(s, [{ text: 'ROUND 2  ·  PROBLEM STATEMENT 1', options: { bold: true, fontSize: 12 } }, { text: '\nAI surface defect detection', options: { fontSize: 10.5 } }],
    { x: 8.3, y: 0.14, w: 4.9, h: 0.6, align: 'right', valign: 'middle', color: C.white });
  T(s, 'Catch every strip defect where it is born, not where it is shipped', { x: 0.6, y: 1.12, w: 7.7, h: 1.4, fontSize: 34, bold: true, color: C.white });
  T(s, 'AI surface inspection that finds, grades and routes every stainless strip defect to the process that caused it.', { x: 0.6, y: 2.58, w: 7.5, h: 0.7, fontSize: 15, color: C.onNavy });

  const stats = [
    ['90%', 'of defects caught on real mill\nstrip it had never seen', 24],
    ['4.7%', 'of clean strip wrongly flagged,\nfewer than 1 in 20', 24],
    ['~₹6.7 cr', 'a year at stake per line, at a\n1% downgrade rate', 22],
    ['7 months', 'to shadow mode on one line,\ngo / no-go at month 7', 22],
  ];
  stats.forEach(([n, l, fs], i) => {
    const x = 0.6 + i * 1.92;
    R(s, x, 3.5, 0.05, 1.02, C.band);
    T(s, n, { x: x + 0.14, y: 3.46, w: 1.75, h: 0.46, fontSize: fs, bold: true, color: C.band, valign: 'bottom' });
    T(s, l, { x: x + 0.14, y: 3.96, w: 1.78, h: 0.6, fontSize: 9.5, color: C.onNavy });
  });

  T(s, 'REAL OUTPUT FROM OUR SYSTEM, ON STEEL IMAGES IT HAD NEVER SEEN', { x: 0.6, y: 4.82, w: 7.6, h: 0.25, fontSize: 9.5, bold: true, color: C.onNavy, charSpacing: 1 });
  DET.forEach(([k], i) => s.addImage({ path: `img/det_${k}.png`, x: 0.6 + i * 1.27, y: 5.12, w: 1.17, h: 1.17 }));

  T(s, `${TEAM ? `TEAM  ${TEAM}   |   ` : ''}${MEMBERS.join('  ·  ')}   |   ${INSTITUTION}`, { x: 0.6, y: 6.6, w: 7.8, h: 0.4, fontSize: 12, bold: true, color: C.white, valign: 'middle' });

  // live demo card on the photo
  RR(s, 9.55, 4.2, 2.85, 2.55, C.white);
  s.addImage({ path: 'qr.png', x: 10.3, y: 4.3, w: 1.35, h: 1.35 });
  T(s, 'TRY THE LIVE DEMO', { x: 9.6, y: 5.68, w: 2.75, h: 0.25, fontSize: 10, bold: true, color: C.accent, align: 'center' });
  T(s, 'surface-vision.github.io', { x: 9.6, y: 5.93, w: 2.75, h: 0.3, fontSize: 13, bold: true, color: C.navy, align: 'center' });
  T(s, 'Runs in your browser. Nothing leaves the page.', { x: 9.65, y: 6.24, w: 2.65, h: 0.4, fontSize: 9, color: C.slate, align: 'center' });
  T(s, 'Photo: Methem, Wikimedia Commons (public domain)', { x: 8.75, y: 7.12, w: 4.4, h: 0.25, fontSize: 9, color: C.onNavy, align: 'right' });
  s.addNotes('Cover. The six images are real output from our system on steel images it had never seen. The QR opens the live demo in the judge\'s browser. Four numbers tell the story: on real mill strip it catches 9 in 10 defects and wrongly flags fewer than 1 in 20 clean images; each line has about INR 6.7 crore a year at stake from downgrades even at a conservative 1%; and one line reaches shadow mode in seven months, with a go/no-go before anything touches the mill.');
}

// ============ SLIDE 2: PROBLEM ============
{
  const s = pres.addSlide();
  chrome(s, 'Problem',
    'A defect born at the caster is paid for at every later stage, but today it is found only at the end',
    'The problem is not that defects exist. They are found late, by sampling, and nobody is told which process caused them.',
    'Sources: AMETEK/Ternium; ASSDA; IISE; ASTM A240; LME nickel USD 16,402/t at INR 95.8 (22 Sep 2026); ISSDA; DGTR (Sep 2025); BIS QCO 2024. Photos: Wikimedia Commons (notes).');
  ph(s, 0.5, 1.35, 12.33, 'Where stainless surface defects are born, and where they are seen today');
  // value-added wedge (the ribbon compresses when there are quotes to show under it)
  const Q = QUOTES.length > 0;
  const g2 = Q ? { wy: 1.72, wh: 0.26, by: 2.03, bd: 0.5, ny: 2.57, cy: 2.8, ry: 2.99, ty: 3.35 }
               : { wy: 1.74, wh: 0.34, by: 2.19, bd: 0.66, ny: 2.93, cy: 3.18, ry: 3.4, ty: 3.83 };
  s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: 0.6, y: g2.wy, w: 12.13, h: g2.wh, flipH: true, fill: { color: C.h1 }, line: { color: C.h1, width: 0 } });
  T(s, 'VALUE ADDED KEEPS RISING  →', { x: 9.6, y: g2.wy + 0.03, w: 3.05, h: g2.wh - 0.04, fontSize: 9, bold: true, color: C.navy, align: 'right', valign: 'middle' });
  const st = [
    ['flame', 'Melt & cast', 'SMS', 'Inclusions, slivers', 'NOT SEEN', C.red],
    ['factory', 'Hot rolling', 'HSM', 'Scale, pits, edge cracks', 'NOT SEEN', C.red],
    ['droplets', 'Anneal & pickle', 'HRAP', 'Pickling patches', 'SPOT CHECK', C.amber],
    ['layers', 'Cold rolling', 'CRM', 'Scratches, roll marks', 'SPOT CHECK', C.amber],
    ['sparkles', 'Final anneal / BA', 'CRAP / BA', 'Stains, BA marks', 'SAMPLED', C.amber],
    ['truck', 'Finish & despatch', 'Slit, cut', 'Found here, or by the customer', 'CLAIMS', C.red],
  ];
  const cw = 1.95, cg = 0.126;
  st.forEach(([ic, n, code, born, seen, col], i) => {
    const x = 0.5 + i * (cw + cg), cx = x + cw / 2;
    if (i < 5) s.addShape(pres.shapes.LINE, { x: cx + g2.bd / 2 + 0.07, y: g2.by + g2.bd / 2, w: cw + cg - g2.bd - 0.14, h: 0, line: { color: C.h3, width: 1.5, dashType: 'dash', endArrowType: 'triangle' } });
    badge(s, ic, cx - g2.bd / 2, g2.by, g2.bd, i === 5 ? C.accent : C.navy);
    T(s, n, { x, y: g2.ny, w: cw, h: 0.24, fontSize: 11, bold: true, color: C.navy, align: 'center' });
    T(s, code, { x, y: g2.cy, w: cw, h: 0.2, fontSize: 9, color: C.muted, align: 'center' });
    T(s, born, { x, y: g2.ry, w: cw, h: 0.36, fontSize: 9.5, color: C.ink, align: 'center' });
    tag(s, cx - 0.5, g2.ty, seen, col, 1.0);
  });
  if (Q) {
    QUOTES.slice(0, 3).forEach((q, i) => {
      const n = Math.min(QUOTES.length, 3), qw = (12.33 - (n - 1) * 0.12) / n, x = 0.5 + i * (qw + 0.12);
      RR(s, x, 3.66, qw, 0.56, C.tint);
      R(s, x, 3.66, 0.05, 0.56, C.accent);
      T(s, [{ text: `“${q.quote.trim()}”`, options: { italic: true, color: C.navy } }, { text: `  — ${who(q)}`, options: { color: C.muted, fontSize: 9 } }],
        { x: x + 0.14, y: 3.68, w: qw - 0.22, h: 0.52, fontSize: 9.5, valign: 'middle' });
    });
  }

  // why stainless: three photo tiles, one number each
  ph(s, 0.5, 4.3, 3.95, 'Why it hurts more on stainless');
  [['img/photos/mill.jpg', '300 m/min', 'too fast for the eye to see'],
   ['img/cover_coil.jpg', '₹126k-165k', 'of nickel in a tonne of 304'],
   ['img/photos/lift.jpg', 'No repair', '2B/BA damage can\'t be polished out'],
  ].forEach(([img, n, l], i) => {
    const x = 0.5 + i * 1.335;
    photo(s, img, x, 4.66, 1.28, 0.92);
    T(s, n, { x, y: 5.6, w: 1.28, h: 0.26, fontSize: i === 1 ? 11 : 12, bold: true, color: C.accent });
    T(s, l, { x, y: 5.86, w: 1.28, h: 0.56, fontSize: 9, color: C.slate });
  });

  // why now: demand trend + three drivers
  ph(s, 4.6, 4.3, 4.25, 'Why now');
  T(s, [{ text: 'India demand, Mt', options: { bold: true, color: C.navy } }, { text: '  +84% in 5 yrs', options: { color: C.accent, bold: true } }], { x: 4.65, y: 4.64, w: 2.1, h: 0.22, fontSize: 9 });
  const yrs = ['FY21', 'FY22', 'FY23', 'FY24', 'FY25', 'FY30F'];
  s.addChart(pres.charts.BAR, [
    { name: 'Actual', labels: yrs, values: [2.61, 3.46, 3.94, 4.49, 4.80, 0] },
    { name: 'Forecast', labels: yrs, values: [0, 0, 0, 0, 0, 6.8] },
  ], {
    x: 4.6, y: 4.84, w: 2.15, h: 1.6, barDir: 'col', barGrouping: 'stacked', barGapWidthPct: 30, chartColors: [C.accent, C.h2],
    showValue: true, dataLabelFormatCode: '0.0;-0.0;;', dataLabelFontSize: 9, dataLabelFontBold: true, dataLabelColor: 'FFFFFF', dataLabelPosition: 'inEnd', dataLabelFontFace: F,
    catAxisLabelFontSize: 9, catAxisLabelColor: C.slate, catAxisLabelFontFace: F, catAxisLineShow: false, valAxisHidden: true, valAxisMinVal: 0,
    valGridLine: { style: 'none' }, catGridLine: { style: 'none' }, showLegend: false,
  });
  [['ship', '1.73 Mt imports', 'FY25; anti-dumping probe on cold-rolled 300/400, Sep 2025'],
   ['badge-check', 'IS 6911 mandatory', 'BIS certification for sheet and strip, since Aug 2024'],
   ['factory', '2.67 MTPA', 'JSL cold rolling by FY28, up from 2.05'],
  ].forEach(([ic, n, l], i) => {
    const y = 4.66 + i * 0.59;
    icon(s, ic, 6.8, y + 0.02, 0.26, 'accent');
    T(s, n, { x: 7.12, y, w: 1.72, h: 0.24, fontSize: 10, bold: true, color: C.navy });
    T(s, l, { x: 7.12, y: y + 0.23, w: 1.72, h: 0.34, fontSize: 9, color: C.slate });
  });

  // cost of quality, compact PAF
  ph(s, 9.0, 4.3, 3.83, 'Cost of quality: where AI moves the money');
  s.addText('spend moves from failure to prevention', {
    shape: pres.shapes.LEFT_ARROW, x: 9.05, y: 4.66, w: 3.73, h: 0.3, fontFace: F, fontSize: 9, bold: true, color: C.white, align: 'center', valign: 'middle',
    fill: { color: C.accent }, line: { color: C.accent, width: 0 }, margin: 0,
  });
  [['Prevention', 'defect sent to its owner in minutes', 0, 0, C.tint],
   ['Internal failure', 'caught before more value is added', 1, 0, C.neutral],
   ['Appraisal', '100% of the surface, every coil', 0, 1, C.tint],
   ['External failure', 'fewer escapes to customers', 1, 1, C.neutral],
  ].forEach(([h, b, cx, cy, f]) => {
    const x = 9.05 + cx * 1.89, y = 5.03 + cy * 0.6;
    RR(s, x, y, 1.84, 0.55, f);
    T(s, h, { x: x + 0.07, y: y + 0.03, w: 1.72, h: 0.22, fontSize: 9.5, bold: true, color: C.navy });
    T(s, b, { x: x + 0.07, y: y + 0.25, w: 1.74, h: 0.28, fontSize: 9, color: C.slate });
  });
  T(s, 'Poor quality costs ~15% of sales in manufacturing (IISE).', { x: 9.05, y: 6.23, w: 3.78, h: 0.22, fontSize: 9, color: C.slate, italic: true });
  s.addNotes('Problem. Walk the ribbon left to right: a sliver born at the caster is only seen at finishing, after hot rolling, annealing, pickling and cold rolling have been paid for. Why stainless hurts more: every tonne of 304 carries INR 126,000-165,000 of nickel, and a 2B or BA mill finish cannot be polished back. Why now: Indian stainless demand grew 84% in five years to 4.8 Mt and is heading to 6.8 Mt by FY30, led by rail and metro coaches; imports hit 1.73 Mt in FY25 and DGTR opened an anti-dumping probe on cold-rolled 300/400 series in September 2025, so quality is the domestic producer\'s moat; BIS certification to IS 6911 has been mandatory for sheet and strip since August 2024; and JSL is adding cold-rolling capacity. The cost-of-quality frame is the output: inspection moves spend from failure to prevention. Photo credits: cold-rolling mill, ThyssenKrupp Steel USA (Free Art License); coil, Methem (public domain); stainless lift interior, Shwun Vounwun3 (CC BY-SA 4.0); all via Wikimedia Commons.');
}

// ============ SLIDE 3: INSIGHTS I (stainless, owners) ============
{
  const s = pres.addSlide();
  chrome(s, 'Insights',
    'On stainless the surface is the product, so every defect type must reach the process that caused it',
    'Classify, don\'t just alarm: a typed defect becomes a work order for a named owner. The pilot belongs on 300-series cold-rolled strip.',
    'Images: our system on held-out frames. Sources: Nortal/Outokumpu 2024; Tata Steel; Parsytec; Leão 2021; Steel in Translation. Photos: Commons (notes). ' + [validatedBy('owners') ? `Owners checked with ${validatedBy('owners').role}.` : 'Owners: team hypothesis.', validatedBy('grades') ? `Grade risks checked with ${validatedBy('grades').role}.` : 'Grade risks: team hypothesis.'].join(' '));
  ph(s, 0.5, 1.35, 12.33, validatedBy('owners') ? `Six defect families, each traced to the process that caused it (owners checked with ${validatedBy('owners').role})` : 'Six defect families, each traced to the process that caused it (real output from our system)');
  const own = {
    inclusion: ['Caster', 'SMS', 'Tundish level, mould flux'],
    crazing: ['Reheat, hot rolling', 'HSM', 'Reheat and coiling temperature'],
    'rolled-in_scale': ['Hot rolling', 'HSM', 'Descaler pressure, nozzles'],
    pitted_surface: ['Reheat, pickling', 'HSM / pickling', 'Furnace atmosphere, over-pickling'],
    patches: ['Anneal & pickle', 'Pickling line', 'Acid strength, line speed'],
    scratches: ['Cold rolling, handling', 'CRM / logistics', 'Guides, rolls, coil handling'],
  };
  const cw = 1.95, cg = 0.126;
  DET.forEach(([k, name], i) => {
    const x = 0.5 + i * (cw + cg);
    const [born, owner, fix] = own[k];
    RR(s, x, 1.72, cw, 2.66, C.neutral);
    s.addImage({ path: `img/det_${k}.png`, x: x + (cw - 1.52) / 2, y: 1.78, w: 1.52, h: 1.52 });
    T(s, [{ text: 'Born: ', options: { color: C.muted } }, { text: born, options: { bold: true, color: C.navy } }], { x: x + 0.1, y: 3.35, w: cw - 0.2, h: 0.21, fontSize: 9.5 });
    T(s, [{ text: 'Owner: ', options: { color: C.muted } }, { text: owner, options: { bold: true, color: C.accent } }], { x: x + 0.1, y: 3.56, w: cw - 0.2, h: 0.21, fontSize: 9.5 });
    T(s, [{ text: 'Fix: ', options: { color: C.muted } }, { text: fix, options: { color: C.slate } }], { x: x + 0.1, y: 3.78, w: cw - 0.2, h: 0.44, fontSize: 9.5 });
  });
  T(s, 'Stainless-only types to add with JSL data:', { x: 0.5, y: 4.43, w: 2.85, h: 0.26, fontSize: 9.5, bold: true, color: C.navy, valign: 'middle' });
  let chx = 3.38;
  ['Roping', 'Ridging (ferritic 430)', 'Orange peel', 'Anneal colour', 'BA stains', 'Lamination'].forEach((c) => {
    const w = 0.24 + c.length * 0.06;
    RR(s, chx, 4.45, w, 0.22, C.tint);
    T(s, c, { x: chx, y: 4.45, w, h: 0.22, fontSize: 9, color: C.navy, align: 'center', valign: 'middle' });
    chx += w + 0.08;
  });

  // precedent: photo cards, one line each
  ph(s, 0.5, 4.78, 5.0, 'Precedent: who already does this, and what we copy');
  [['img/photos/okp_mill.jpg', 'Outokumpu', 'AI inspection live since 2024', 'offline, multi-site'],
   ['img/photos/tata_view.jpg', 'Tata Steel', 'video analytics, cold-rolled strip', 'in-house team owns it'],
   ['img/proof_gc10.png', 'Parsytec', 'bright + dark-field light on stainless', 'optics spec'],
  ].forEach(([img, n, what, copy], i) => {
    const x = 0.5 + i * 1.69;
    photo(s, img, x, 5.13, 1.62, 0.6);
    T(s, [{ text: n, options: { bold: true, color: C.navy } }, { text: `  ${what}`, options: { color: C.slate, fontSize: 9 } }], { x, y: 5.76, w: 1.64, h: 0.4, fontSize: 9.5 });
    T(s, [{ text: 'Copy: ', options: { bold: true, color: C.accent } }, { text: copy, options: { color: C.accent } }], { x, y: 6.16, w: 1.64, h: 0.24, fontSize: 9 });
  });

  // grade matrix
  ph(s, 5.75, 4.78, 7.08, 'Grade-by-defect risk: which defects to train for first');
  const lv = { H: [C.h4, C.white], M: [C.h2, C.ink], L: [C.h1, C.muted] };
  const g = [
    ['200 series (Cr-Mn)', 'M', 'M', 'H', 'M', 'L'],
    ['300 series (304, 316)', 'H', 'M', 'H', 'L', 'L'],
    ['400 series (409, 430)', 'H', 'M', 'M', 'M', 'H'],
    ['Duplex (2205)', 'M', 'H', 'M', 'H', 'L'],
  ];
  const rows = [[hd('Grade'), hd('Slivers', { align: 'center' }), hd('Scale', { align: 'center' }), hd('Scratches', { align: 'center' }), hd('Edge cracks', { align: 'center' }), hd('Ridging', { align: 'center' })]];
  g.forEach(r => rows.push([cl(r[0], { bold: true, fill: { color: r[0].startsWith('300') ? C.tint : C.white } })].concat(r.slice(1).map(v => cl(v, { align: 'center', bold: true, fill: { color: lv[v][0] }, color: lv[v][1] })))));
  table(s, rows, { x: 5.75, y: 5.13, w: 5.0, colW: [1.6, 0.68, 0.68, 0.68, 0.68, 0.68], rowH: 0.255 });
  photo(s, 'img/photos/vb_coaches.jpg', 10.9, 5.13, 1.93, 1.28);
  R(s, 10.9, 5.79, 1.93, 0.62, C.rail, { fill: { color: C.rail, transparency: 15 } });
  T(s, [{ text: 'Pilot grade: 300 series', options: { bold: true, color: C.white } }, { text: '\nexposed surfaces, most nickel', options: { color: C.onNavy, fontSize: 9 } }],
    { x: 10.98, y: 5.8, w: 1.8, h: 0.6, fontSize: 10, valign: 'middle' });
  s.addNotes('Insights, stainless-specific. The six images are real output from our system on images it had never seen; each family is born at a different stage, so a typed detection goes straight to the right owner as a work order. The owner mapping needs a JSL metallurgist\'s sign-off in P0. Precedent: Outokumpu, Europe\'s largest stainless producer, switched on an AI surface inspection system at Tornio in October 2024, built to run without internet and scale across sites; Tata Steel Kalinganagar, India\'s first WEF Lighthouse, uses video analytics for surface defects on cold-rolled products; Parsytec systems on stainless lines use 170 micron cameras with bright and dark field light and make a decision per coil. We copy the offline edge design, an in-house team that owns the system, and the optics spec. The grade matrix output is the pilot grade: 300 series. Photo credits: Outokumpu Tornio mill, Methem (CC BY 3.0); Tata Steel plant, Kharbaan Ghaltaan (CC BY-SA 4.0); Vande Bharat coaches, Ravi Dwivedi (CC BY-SA 4.0); all via Wikimedia Commons. The Vande Bharat photo illustrates a 300-series exposed-surface use; JSL supplies stainless for these coaches.');
}

// ============ SLIDE 4: INSIGHTS II (our findings) ============
{
  const s = pres.addSlide();
  chrome(s, 'Insights',
    'Four design choices that make the system work on a real line',
    'Success is decided on the line, not in the lab: JSL\'s own strip, the right image detail, coil-level rules and operator trust.',
    'Results measured by us on public steel data the system had never seen. Hardware sizing derived from our line-speed benchmark.');
  const qx = [0.5, 6.77], qy = [1.35, 3.97], qw = 6.06, qh = 2.47;
  const quad = (i, j, head) => { ph(s, qx[i], qy[j], qw, head); RR(s, qx[i], qy[j] + 0.3, qw, qh - 0.3, C.neutral); };
  const so = (i, j, txt) => T(s, [{ text: 'So we: ', options: { bold: true, color: C.accent } }, { text: txt, options: { color: C.ink } }], { x: qx[i] + 0.15, y: qy[j] + qh - 0.36, w: qw - 0.3, h: 0.3, fontSize: 10, valign: 'middle' });

  // Q1: two rings, each beside a real mill-strip image
  quad(0, 0, '1  Alarms you can act on');
  const ring = (x, val, big, img, cap, sub) => {
    s.addChart(pres.charts.DOUGHNUT, [{ name: big, labels: ['a', 'b'], values: [val, 100 - val] }], {
      x, y: 1.72, w: 1.3, h: 1.3, holeSize: 70, chartColors: [C.accent, 'E3E3E3'], showLegend: false, showValue: false, showPercent: false, dataBorder: { pt: 0, color: 'FFFFFF' },
    });
    T(s, big, { x, y: 2.16, w: 1.3, h: 0.42, fontSize: 16, bold: true, color: C.navy, align: 'center', valign: 'middle' });
    T(s, sub, { x: x - 0.05, y: 3.05, w: 1.45, h: 0.22, fontSize: 9, color: C.slate, align: 'center' });
    s.addImage({ path: img, x: x + 1.38, y: 1.76, w: 1.12, h: 1.12 });
    T(s, cap, { x: x + 1.38, y: 2.92, w: 1.12, h: 0.3, fontSize: 9, bold: true, color: C.accent, align: 'center' });
  };
  ring(0.62, 89.6, '90%', 'img/proof_severstal.png', 'defect: flagged', '258 of 288 caught');
  ring(3.62, 4.7, '4.7%', 'img/mill_clean.jpg', 'clean: no alarm', '14 of 300 flagged');
  so(0, 0, 'prove it on JSL\'s own clean and defective strip before any alarm is trusted.');

  // Q2: one verdict per coil
  quad(1, 0, '2  One verdict per coil, not a stream of alarms');
  R(s, 6.95, 1.8, 4.2, 0.36, C.h1);
  [[0.35, C.navy], [1.1, C.accent], [1.9, C.navy], [2.75, C.accent], [3.6, C.navy]].forEach(([dx, c]) => R(s, 6.95 + dx, 1.84, 0.12, 0.28, c));
  T(s, 'one coil, both faces: every defect with its type, size and position', { x: 6.95, y: 2.18, w: 4.25, h: 0.2, fontSize: 9, color: C.muted, italic: true });
  s.addShape(pres.shapes.CHEVRON, { x: 11.22, y: 1.87, w: 0.16, h: 0.24, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  [['ACCEPT', C.green], ['DOWNGRADE', C.amber], ['HOLD', C.red]].forEach(([t, c], k) => tag(s, 11.47, 1.72 + k * 0.26, t, c, 1.22));
  T(s, [{ text: 'Limits per grade and finish: ', options: { color: C.slate } }, { text: 'a light scratch passes on 2B tubes, downgrades a BA panel.', options: { color: C.navy, bold: true } }],
    { x: 6.95, y: 2.5, w: 4.25, h: 0.5, fontSize: 9.5 });
  photo(s, 'img/photos/lift.jpg', 11.4, 2.58, 1.3, 0.78);
  so(1, 0, 'agree accept, downgrade and hold limits per grade and finish with JSL Quality in P0.');

  // Q3: image detail sets the hardware bill -- three settings
  quad(0, 1, '3  Image detail sets what is seen, and the hardware bill');
  [['0.2 mm per pixel', 18, '~18 processors', 'sees defects from ~0.5 mm'],
   ['0.5 mm per pixel', 3, '~3 processors', 'sees defects from ~1.3 mm'],
   ['1.28 mm per pixel', 1, '1 processor', 'fine scratches vanish (~3 mm)'],
  ].forEach(([res, n, proc, see], i) => {
    const x = 0.64 + i * 1.98;
    if (i) R(s, x - 0.1, 4.36, 0.01, 1.42, C.edge);
    T(s, res, { x, y: 4.32, w: 1.85, h: 0.24, fontSize: 10, bold: true, color: i === 0 ? C.navy : C.slate });
    for (let k = 0; k < n; k++) icon(s, 'cpu', x + (k % 6) * 0.26, 4.6 + Math.floor(k / 6) * 0.25, 0.21, 'navy');
    T(s, proc, { x, y: 5.37, w: 1.85, h: 0.22, fontSize: 11, bold: true, color: C.ink });
    T(s, see, { x, y: 5.59, w: 1.9, h: 0.22, fontSize: 9, color: C.slate });
  });
  T(s, '1.28 m strip at 250 m/min. Processors scale with pixels; a defect needs 2-3 pixels.', { x: 0.64, y: 5.83, w: 5.8, h: 0.22, fontSize: 9, italic: true, color: C.muted });
  so(0, 1, 'fix the image resolution with JSL first, then size the hardware.');

  // Q4: what the operator actually sees, plus how trust is earned
  quad(1, 1, '4  Operators act only on alarms they can trust');
  s.addImage({ path: 'img/demo_result.png', x: 6.9, y: 4.34, w: 2.05, h: 1.63 });
  [['scan-search', 'Evidence on every alarm'], ['user-check', '4 weeks of shadow mode'], ['refresh-cw', 'Every override reviewed']].forEach(([ic, h], k) => {
    const y = 4.42 + k * 0.56;
    badge(s, ic, 9.4, y, 0.42, C.navy);
    T(s, h, { x: 9.92, y, w: 2.85, h: 0.42, fontSize: 11, bold: true, color: C.navy, valign: 'middle' });
  });
  so(1, 1, 'earn trust on the shift floor before the system is allowed to act.');
  s.addNotes('Four design choices. One: alarms you can act on. On real mill strip it catches 258 of 288 defects and wrongly flags only 14 of 300 clean images; the pilot repeats this on JSL\'s own strip before any alarm is trusted. Two: one verdict per coil. Every defect rolls up into accept, downgrade or hold, with limits agreed per grade and finish with JSL Quality. Three: image detail decides what can be seen and what the hardware costs; full detail at 250 m/min needs about 18 processing units, so resolution is a day-one decision with JSL. Four: trust is earned on the shift floor, with evidence on every alarm, four weeks of shadow mode and a weekly review of every override. Images: a real mill-strip defect our system flagged, and a clean mill-strip image that raised no alarm, both from the held-out test set; the screenshot is the live demo\'s alarm view. Lift photo: Shwun Vounwun3 (CC BY-SA 4.0).');
}

// ============ SLIDE 5: SOLUTION ============
{
  const s = pres.addSlide();
  chrome(s, 'Solution',
    'Detect on the strip, decide at the coil, route to the owner, and learn from every correction',
    'Buy proven cameras, own the software: vendor-grade capture, tuned to JSL\'s grades and wired to JSL\'s process owners.',
    'BUILT = working in our prototype today, live at surface-vision.github.io.');
  ph(s, 0.5, 1.35, 12.33, 'How it works: six stages from camera to corrective action');
  const stg = [
    ['camera', 'Capture', 'Both faces, bright + dark-field light, 0.2 mm detail', 'TO BUILD', C.muted],
    ['shield-check', 'Check', 'Is it usable steel? If not, skip it', 'BUILT', C.green],
    ['scan-search', 'Detect', 'Find and classify every defect', 'BUILT', C.green],
    ['gauge', 'Score', 'Confidence, severity 0-100, coil position', 'BUILT', C.green],
    ['split', 'Decide', 'Accept, downgrade or hold, per grade', 'PARTIAL', C.amber],
    ['refresh-cw', 'Act & learn', 'Alert, hold, work order; operator feedback improves it', 'TO BUILD', C.muted],
  ];
  const cw = 1.95, cg = 0.126;
  stg.forEach(([ic, h, b, t, col], i) => {
    const x = 0.5 + i * (cw + cg);
    RR(s, x, 1.72, cw, 1.3, i === 5 ? C.tint : C.neutral);
    badge(s, ic, x + 0.12, 1.78, 0.46, C.navy);
    T(s, h, { x: x + 0.66, y: 1.78, w: cw - 0.7, h: 0.46, fontSize: 12, bold: true, color: C.navy, valign: 'middle' });
    T(s, b, { x: x + 0.12, y: 2.28, w: cw - 0.2, h: 0.44, fontSize: 9, color: C.slate });
    tag(s, x + 0.12, 2.74, t, col, 0.9);
    if (i < 5) s.addShape(pres.shapes.CHEVRON, { x: x + cw + 0.015, y: 2.2, w: 0.1, h: 0.22, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  });
  // where each stage runs
  const zone = (i0, i1, label, fill) => {
    const x0 = 0.5 + i0 * (cw + cg), x1 = 0.5 + i1 * (cw + cg) + cw;
    R(s, x0, 3.08, x1 - x0, 0.05, fill);
    T(s, label, { x: x0, y: 3.14, w: x1 - x0, h: 0.24, fontSize: 9, bold: true, color: fill, align: 'center', valign: 'middle' });
  };
  zone(0, 0, 'At the line: cameras, lights', C.muted);
  zone(1, 3, 'Edge server beside the line, runs offline', C.accent);
  zone(4, 4, 'Plant network (IEC 62443)', C.navy2);
  zone(5, 5, 'People: screen, work orders', C.navy);

  // running today
  ph(s, 0.5, 3.52, 3.75, 'Running today');
  RR(s, 0.5, 3.82, 3.75, 2.62, C.neutral);
  s.addImage({ path: 'img/det_scratches.png', x: 0.62, y: 3.93, w: 2.0, h: 2.0 });
  T(s, [
    { text: 'Upload a strip image, get:\n', options: { bold: true, color: C.navy } },
    { text: '•  box on the defect\n•  defect type\n•  confidence\n•  severity score', options: { color: C.slate } },
  ], { x: 2.72, y: 3.95, w: 1.5, h: 1.5, fontSize: 9.5 });
  T(s, [{ text: '16 samples built in', options: { bold: true, color: C.navy } }, { text: ', including clean mill strip. Or drop your own photo.', options: { color: C.slate } }], { x: 2.72, y: 5.3, w: 1.5, h: 0.66, fontSize: 9 });
  T(s, [{ text: 'surface-vision.github.io', options: { bold: true, color: C.accent } }, { text: '  runs in the browser', options: { color: C.muted } }], { x: 0.62, y: 6.02, w: 3.55, h: 0.3, fontSize: 9.5, valign: 'middle' });

  // what is new
  ph(s, 4.42, 3.52, 4.1, 'What JSL gets that a standard system does not');
  [['target', 'Alarms you can act on', '4.7% false alarms'],
   ['split', 'Root-cause routing', 'a work order per alarm'],
   ['layers', 'One verdict per coil', 'accept · downgrade · hold'],
   ['shield-check', 'JSL owns the system', 'no vendor lock-in'],
  ].forEach(([ic, h, b], i) => {
    const x = 4.42 + (i % 2) * 2.08, y = 3.9 + Math.floor(i / 2) * 1.28;
    RR(s, x, y, 2.0, 1.2, C.neutral);
    badge(s, ic, x + 0.72, y + 0.1, 0.56, C.accent);
    T(s, h, { x: x + 0.05, y: y + 0.68, w: 1.9, h: 0.26, fontSize: 10.5, bold: true, color: C.navy, align: 'center' });
    T(s, b, { x: x + 0.05, y: y + 0.92, w: 1.9, h: 0.24, fontSize: 9, color: C.slate, align: 'center' });
  });

  // build vs buy
  ph(s, 8.7, 3.52, 4.13, 'Build vs buy');
  const crit = ['Cost per line', 'Proven at speed', 'Tuned to JSL grades', 'JSL owns the system'];
  const opts = [
    ['Commercial', 'proven; the vendor owns it', ['r', 'g', 'a', 'r']],
    ['Fully in-house', 'JSL owns it; slow to prove', ['g', 'r', 'g', 'g']],
    ['Hybrid (recommended)', 'vendor cameras, JSL software', ['g', 'a', 'g', 'g']],
  ];
  const cx0 = 10.28, cwid = 0.63;
  crit.forEach((c, k) => T(s, c, { x: cx0 + k * cwid, y: 3.86, w: cwid - 0.03, h: 0.42, fontSize: 9, bold: true, color: C.slate, align: 'center', valign: 'bottom' }));
  const colmap = { g: C.green, a: C.amber, r: C.red };
  opts.forEach(([n, cost, sc], j) => {
    const y = 4.4 + j * 0.52;
    if (j === 2) RR(s, 8.72, y - 0.06, 4.09, 0.5, C.tint);
    T(s, n, { x: 8.8, y, w: 1.5, h: 0.2, fontSize: 10, bold: true, color: j === 2 ? C.accent : C.ink });
    T(s, cost, { x: 8.8, y: y + 0.2, w: 1.5, h: 0.2, fontSize: 9, color: C.muted });
    sc.forEach((v, k) => dot(s, cx0 + k * cwid + (cwid - 0.03) / 2 - 0.1, y + 0.08, 0.2, colmap[v]));
  });
  [['strong', C.green], ['partial', C.amber], ['weak', C.red]].forEach(([l, c], k) => {
    dot(s, 8.82 + k * 0.85, 5.99, 0.14, c);
    T(s, l, { x: 9.0 + k * 0.85, y: 5.94, w: 0.65, h: 0.22, fontSize: 9, color: C.muted, valign: 'middle' });
  });
  T(s, 'Output: hybrid is never weak.', { x: 8.8, y: 6.17, w: 4.0, h: 0.24, fontSize: 10, bold: true, color: C.accent, valign: 'middle' });
  s.addNotes('Solution. Stages 2 to 4 are built and running today in the browser demo. Stage 5 has coil rules built, but grade- and finish-specific limits need JSL input. Stages 1 and 6 are the plant-side pilot work. The strip underneath shows where each stage runs: cameras at the line, an edge server beside it that works offline, the plant network behind IEC 62443 zones, and people on screens and work orders. What JSL gets that a standard system does not: alarms it can act on, routing to the owning process, one verdict per coil, and ownership of its own system. Build vs buy: commercial systems are proven but closed and vendor-owned; fully in-house is slowest to prove; hybrid is never weak.');
}

// ============ SLIDE 6: PROOF (what the data says) ============
{
  const s = pres.addSlide();
  chrome(s, 'Solution',
    'It already works on public steel data; the pilot\'s job is to prove it on JSL\'s own strip',
    'Strong on four of six defect families today. Every weak spot is known, and each has a fix in the pilot plan.',
    'Datasets: NEU-DET (Northeastern Univ.); Severstal steel defects (Kaggle); GC10-DET (Lv et al. 2020, CC BY 4.0). All scores on held-out images the system had never seen.');

  // datasets (plus our own stainless photos, when they exist)
  ph(s, 0.5, 1.35, 6.06, STAINLESS ? 'Tested on three public datasets and our own stainless photos' : 'Tested on three public steel datasets');
  const ds = [
    ['img/det_patches.png', 1.1, 1.1, 'NEU-DET', '1,800 images, 6 types', 'University benchmark, hot-rolled strip'],
    ['img/proof_severstal.png', 1.1, 1.1, 'Severstal', '12,568 real mill frames', 'Production strip from a steel mill'],
    ['img/proof_gc10.png', 1.75, 1.18, 'GC10-DET', '2,294 frames, 10 types', 'Full-width line-scan camera frames (roll marks shown)'],
  ];
  if (STAINLESS) {
    const m = STAINLESS.sm;
    ds.push([STAINLESS.img, 0, 0, 'Stainless (ours)', `${m.photos} photos, ${m.finishes.length} finish${m.finishes.length === 1 ? '' : 'es'}`,
      `defects flagged ${m.defective_flagged}/${m.defective}; clean flagged ${m.clean_flagged}/${m.clean}`]);
  }
  const n4 = ds.length === 4, cwd = n4 ? 1.455 : 1.95, gap = n4 ? 0.08 : 0.105, ib = n4 ? 1.27 : 1.75, ih0 = n4 ? 0.98 : 1.18;
  ds.forEach(([img, iw, ih, n, cnt, d], i) => {
    const x = 0.5 + i * (cwd + gap);
    RR(s, x, 1.72, cwd, 2.35, i === 3 ? C.tint : C.neutral);
    R(s, x + 0.09, 1.8, ib, ih0, C.white);
    if (i === 3) s.addImage({ path: img, x: x + 0.09, y: 1.8, w: ib, h: ih0, sizing: { type: 'cover', w: ib, h: ih0 } });
    else {
      const k = Math.min(1, ib / Math.max(iw, 0.01), ih0 / Math.max(ih, 0.01));
      s.addImage({ path: img, x: x + 0.09 + (ib - iw * k) / 2, y: 1.8 + (ih0 - ih * k) / 2, w: iw * k, h: ih * k });
    }
    T(s, n, { x: x + 0.09, y: 1.84 + ih0, w: cwd - 0.14, h: 0.26, fontSize: n4 ? 10 : 11, bold: true, color: i === 3 ? C.accent : C.navy });
    T(s, cnt, { x: x + 0.09, y: 2.1 + ih0, w: cwd - 0.12, h: 0.24, fontSize: 9, bold: true, color: C.accent });
    T(s, d, { x: x + 0.09, y: 2.35 + ih0, w: cwd - 0.12, h: 0.66, fontSize: 9, color: C.slate });
  });

  // per-defect accuracy
  ph(s, 6.77, 1.35, 6.06, 'How well it finds each defect type (accuracy, held-out images)');
  const pc = [['Patches', 0.942], ['Inclusion', 0.795], ['Scratches', 0.794], ['Pitted surface', 0.763], ['Rolled-in scale', 0.597], ['Crazing', 0.581]];
  const ax = 8.3, aw = 3.9;
  R(s, ax + aw * 0.70, 1.74, aw * 0.10, 2.2, C.tint);
  T(s, 'published range', { x: ax + aw * 0.70 - 0.3, y: 3.92, w: aw * 0.10 + 0.6, h: 0.18, fontSize: 9, color: C.muted, align: 'center' });
  pc.forEach(([n, v], i) => {
    const y = 1.8 + i * 0.35;
    T(s, n, { x: 6.87, y, w: 1.4, h: 0.28, fontSize: 10, color: C.ink, valign: 'middle', bold: v < 0.7 });
    R(s, ax, y + 0.05, aw * v, 0.19, v >= 0.7 ? C.accent : C.grey);
    T(s, v.toFixed(2), { x: ax + aw * v + 0.06, y, w: 0.5, h: 0.28, fontSize: 10, bold: true, color: v >= 0.7 ? C.ink : C.slate, valign: 'middle' });
  });

  // headline results
  ph(s, 0.5, 4.25, 6.06, 'Headline results');
  const hr = [
    ['target', '178 of 180', 'test images: the right defect type named'],
    ['scan-search', 'Live', 'anyone can test it now at surface-vision.github.io'],
  ];
  hr.forEach(([ic, n, l], i) => {
    const x = 0.5 + i * 2.055;
    RR(s, x, 4.62, 1.95, 1.8, C.neutral);
    badge(s, ic, x + 0.12, 4.72, 0.44, C.navy);
    T(s, n, { x: x + 0.12, y: 5.22, w: 1.8, h: 0.44, fontSize: 20, bold: true, color: C.navy, valign: 'middle' });
    T(s, l, { x: x + 0.12, y: 5.68, w: 1.75, h: 0.66, fontSize: 9.5, color: C.slate });
  });
  {
    const x = 0.5 + 2 * 2.055;
    RR(s, x, 4.62, 1.95, 1.8, C.neutral);
    badge(s, 'archive', x + 0.12, 4.72, 0.44, C.navy);
    T(s, '~10,100', { x: x + 0.12, y: 5.22, w: 1.8, h: 0.44, fontSize: 20, bold: true, color: C.navy, valign: 'middle' });
    const parts = [['Lab', 1800, C.h4], ['Mill strip', 6001, C.accent], ['Line-scan', 2294, C.h2]];
    let px = x + 0.12;
    parts.forEach(([, v, c]) => { const w = 1.7 * v / 10095; R(s, px, 5.7, w, 0.18, c); px += w; });
    parts.forEach(([n, v, c], k) => {
      const ly = 5.93 + k * 0.155;
      R(s, x + 0.12, ly + 0.035, 0.1, 0.1, c);
      T(s, `${n} ${v.toLocaleString('en-US')}`, { x: x + 0.28, y: ly, w: 1.6, h: 0.16, fontSize: 9, color: C.slate, valign: 'middle' });
    });
  }

  // weak spots
  ph(s, 6.77, 4.25, 6.06, 'Known weak spots, and the fix for each');
  table(s, [
    [hd('Weak spot'), hd('Today (0-1)', { align: 'center' }), hd('Fix in the pilot')],
    [cl('Low-contrast texture: crazing, rolled-in scale', { bold: true }), cl('0.58-0.60', { align: 'center', bold: true, color: C.amber }), cl('Angled lighting; JSL labels in P1')],
    [cl('Real mill frames are harder than the lab benchmark', { bold: true }), cl('0.57 vs 0.75', { align: 'center', bold: true, color: C.amber }), cl('Prove it on JSL\'s own strip in P1')],
    [cl('Roll marks and edge cracks: few examples', { bold: true }), cl('0.20', { align: 'center', bold: true, color: C.red }), cl('Edge cameras; targeted labelling with JSL')],
  ], { x: 6.77, y: 4.62, w: 6.06, colW: [2.75, 1.05, 2.26], rowH: [0.3, 0.5, 0.5, 0.5] });
  s.addNotes('Proof. Three public datasets, about 10,100 images, all scored on images the system had never seen. Per defect, four of six families sit at or above the published range; crazing and rolled-in scale are low-contrast textures and are weaker. The honest headline is the second weak spot: on real mill frames the score is 0.57 against 0.75 on the lab benchmark, which is exactly why the pilot starts on JSL\'s own strip before anything is trusted. Roll marks score 0.20 because public data has only a few dozen examples; edge cameras and targeted labelling with JSL fix that.');
}

// ============ SLIDE 6: IMPLEMENTATION ============
{
  const s = pres.addSlide();
  chrome(s, 'Implementation',
    'Seven months to shadow mode on one line, and nothing touches the mill until trust is earned',
    'Commit one line for seven months. The month-7 go/no-go is decided on JSL\'s own alarm and recall numbers.',
    'Pragati: JSL\'s digitalisation programme; Phase 2 brings Level-2 process data at Jajpur. KPI targets to be agreed with JSL Quality in P0.');
  // gantt
  ph(s, 0.5, 1.35, 7.85, 'Stage-gate roadmap (months)');
  const gx = 2.75, gw = 3.55;
  [0, 4, 8, 12, 16, 20].forEach(m => T(s, `m${m}`, { x: gx + gw * m / 20 - 0.2, y: 1.7, w: 0.4, h: 0.2, fontSize: 9, color: C.muted, align: 'center' }));
  T(s, '◆ EXIT GATE', { x: 6.5, y: 1.7, w: 1.8, h: 0.2, fontSize: 9, bold: true, color: C.muted });
  const P = [
    ['P0', 'Instrument one line', 0, 1.5, 'Optics fixed; Pragati L2 data linked'],
    ['P1', 'Learn JSL strip', 1.5, 4, 'Clean alarms ≤ 25%, recall ≥ 90%'],
    ['P2', 'Shadow mode: advises only', 4, 7, '≥ 90% operator agreement'],
    ['P3', 'Auto-hold on severe defects', 7, 12, 'Holds trusted; downgrade measured'],
    ['P4', 'Second line and grade', 12, 20, 'Fleet decision on measured value'],
  ];
  P.forEach(([c, n, a, b, gate], i) => {
    const y = 1.95 + i * 0.38;
    icon(s, ['camera', 'layers', 'scan-search', 'shield-check', 'factory'][i], 0.56, y + 0.05, 0.24, 'accent');
    T(s, c, { x: 0.84, y, w: 0.32, h: 0.34, fontSize: 11, bold: true, color: C.accent, valign: 'middle' });
    T(s, n, { x: 1.16, y, w: 1.58, h: 0.34, fontSize: 9.5, color: C.ink, valign: 'middle' });
    R(s, gx, y + 0.07, gw, 0.2, C.neutral);
    R(s, gx + gw * a / 20, y + 0.07, gw * (b - a) / 20, 0.2, i < 3 ? C.accent : C.navy2);
    s.addShape(pres.shapes.DIAMOND, { x: gx + gw * b / 20 - 0.09, y: y + 0.08, w: 0.18, h: 0.18, fill: { color: C.navy }, line: { color: 'FFFFFF', width: 1 } });
    T(s, gate, { x: 6.5, y, w: 1.85, h: 0.34, fontSize: 9, color: C.slate, valign: 'middle' });
  });
  R(s, gx + gw * 7 / 20 - 0.01, 1.92, 0.025, 1.9, C.red);
  T(s, 'pilot go / no-go', { x: gx + gw * 7 / 20 - 0.65, y: 3.83, w: 1.3, h: 0.2, fontSize: 9, bold: true, color: C.red });

  // pilot scope: the line itself, then what is in and what waits
  ph(s, 8.6, 1.35, 4.23, 'Pilot scope: in and out');
  photo(s, 'img/photos/mill.jpg', 8.6, 1.66, 4.23, 0.72);
  R(s, 8.6, 2.1, 4.23, 0.28, C.rail, { fill: { color: C.rail, transparency: 20 } });
  T(s, 'One 300-series cold-rolled line', { x: 8.7, y: 2.1, w: 4.0, h: 0.28, fontSize: 10.5, bold: true, color: C.white, valign: 'middle' });
  T(s, 'IN THE PILOT', { x: 8.7, y: 2.45, w: 2.0, h: 0.2, fontSize: 9, bold: true, color: C.accent, charSpacing: 1 });
  T(s, 'LATER', { x: 10.85, y: 2.45, w: 1.9, h: 0.2, fontSize: 9, bold: true, color: C.muted, charSpacing: 1 });
  [['Both faces, 2B + BA', 'Hot end, caster side'], ['6 defect families', 'Edge cracks'], ['Read-only until P3', 'Other plants']].forEach(([a, b], i) => {
    const y = 2.68 + i * 0.3;
    dot(s, 8.72, y + 0.08, 0.1, C.accent);
    T(s, a, { x: 8.88, y, w: 1.9, h: 0.26, fontSize: 9.5, color: C.ink, valign: 'middle' });
    dot(s, 10.87, y + 0.08, 0.1, C.grey);
    T(s, b, { x: 11.03, y, w: 1.8, h: 0.26, fontSize: 9.5, color: C.slate, valign: 'middle' });
  });
  T(s, [{ text: 'Owner: ', options: { color: C.muted } }, { text: 'JSL Quality', options: { bold: true, color: C.ink } },
        { text: '   Integration: ', options: { color: C.muted } }, { text: 'Pragati team', options: { bold: true, color: C.ink } }],
    { x: 8.7, y: 3.64, w: 4.1, h: 0.26, fontSize: 9.5, valign: 'middle' });

  // risks: likelihood x impact map with numbered chips, short legend
  ph(s, 0.5, 4.08, 6.06, 'Top risks: where they sit, and when each is retired');
  const gx0 = 0.95, gy0 = 4.5, cs = 0.56;
  const heat = [[C.h2, C.h3, C.h4], [C.h1, C.h2, C.h3], [C.neutral, C.h1, C.h2]];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R(s, gx0 + c * cs, gy0 + r * cs, cs - 0.03, cs - 0.03, heat[r][c]);
  ['High', 'Med', 'Low'].forEach((l, r) => T(s, l, { x: 0.55, y: gy0 + r * cs + 0.17, w: 0.38, h: 0.2, fontSize: 9, color: C.muted }));
  ['Low', 'Med', 'High'].forEach((l, c) => T(s, l, { x: gx0 + c * cs, y: gy0 + 3 * cs, w: cs, h: 0.18, fontSize: 9, color: C.muted, align: 'center' }));
  T(s, 'likelihood ↑   impact →', { x: 0.55, y: gy0 + 3 * cs + 0.18, w: 2.2, h: 0.2, fontSize: 9, italic: true, color: C.muted });
  const risks = [
    [1, 'Misses on JSL\'s strip at first', 'P1', 0, 2, 0], [2, 'Mirror-like BA/2B hides defects', 'P0-P1', 1, 2, 0],
    [3, 'Rare defects missed', 'P1-P3', 0, 1, 0], [4, 'Operators ignore alarms', 'P2', 1, 1, 0],
    [5, 'Plant network, heat, fumes', 'P0', 2, 1, 0], [6, 'Quality slips after an update', 'P2+', 2, 2, 0],
  ];
  risks.forEach(([n, , , r, c]) => {
    const x = gx0 + c * cs + 0.13, y = gy0 + r * cs + 0.12;
    dot(s, x, y, 0.28, C.rail);
    T(s, String(n), { x, y, w: 0.28, h: 0.28, fontSize: 10, bold: true, color: C.white, align: 'center', valign: 'middle' });
  });
  risks.forEach(([n, name, ph0], i) => {
    const y = 4.47 + i * 0.31;
    dot(s, 2.78, y + 0.03, 0.22, C.rail);
    T(s, String(n), { x: 2.78, y: y + 0.03, w: 0.22, h: 0.22, fontSize: 9, bold: true, color: C.white, align: 'center', valign: 'middle' });
    T(s, name, { x: 3.08, y, w: 2.55, h: 0.28, fontSize: 9.5, color: C.ink, valign: 'middle' });
    tag(s, 5.72, y + 0.04, `retire ${ph0}`, C.accent, 0.78);
  });

  // KPIs
  ph(s, 6.77, 4.08, 6.06, 'Pilot KPIs: how JSL will know it works');
  const k = [
    ['target', '≥ 90%', 'defects flagged', 'now 89.6%'],
    ['shield-check', '≤ 25%', 'clean strip flagged', 'now 4.7%'],
    ['scan-line', '100%', 'coils mapped', 'target'],
    ['clock', '< 0.2 s', 'to the operator', 'target'],
    ['user-check', '≥ 90%', 'operator agreement', 'in P2'],
    ['trending-up', '↓', 'downgrade rate', 'in P3'],
  ];
  k.forEach(([ic, n, l, m], i) => {
    const x = 6.77 + (i % 3) * 2.04, y = 4.44 + Math.floor(i / 3) * 1.0;
    RR(s, x, y, 1.98, 0.93, C.neutral);
    badge(s, ic, x + 1.5, y + 0.1, 0.38, i < 2 ? C.accent : C.navy2);
    T(s, n, { x: x + 0.1, y: y + 0.06, w: 1.4, h: 0.4, fontSize: 18, bold: true, color: C.navy, valign: 'middle' });
    T(s, l, { x: x + 0.1, y: y + 0.48, w: 1.85, h: 0.2, fontSize: 9.5, color: C.ink });
    T(s, m, { x: x + 0.1, y: y + 0.68, w: 1.85, h: 0.2, fontSize: 9, bold: i < 2, color: i === 0 ? C.amber : (i === 1 ? C.green : C.muted) });
  });
  s.addNotes('Implementation. Five phases, 20 months, with a hard go/no-go at month 7 after shadow mode. The system never acts on the mill until P3; before that it only reads data and advises. The pilot scope is deliberately narrow: one 300-series cold-rolled line, both faces, 2B and BA finishes, the six defect families plus unclassified defects. The hot end, edge cracks, automatic process correction and other plants wait until the pilot has proved itself. JSL Quality owns every go/no-go; the Pragati team owns integration, because Pragati Phase 2 already brings Level-2 process data at Jajpur. Mirror-like BA and 2B finishes are handled the way stainless lines already do it: bright and dark-field light, with deflectometry for BA. The plant network is protected by IEC 62443 zones and the system runs offline. Every update must beat the live system on a fixed JSL test set before it ships. Photo: cold-rolling mill, ThyssenKrupp Steel USA (Free Art License), Wikimedia Commons.');
}

// ============ SLIDE 7: IMPACT ============
{
  const s = pres.addSlide();
  chrome(s, 'Impact',
    'At a conservative 1% downgrade rate, each line has ~₹6.7 cr a year at stake; ~₹43 cr across the fleet',
    'Recommendation: approve a 7-month pilot on one 300-series cold-rolled line. JSL\'s own downgrade rate decides fleet rollout.',
    'Inputs: JSL FY26 (INR 167,407/t; 2,565,902 t sold). Base case: a 0.4 Mt line, 1% downgraded at a 10% discount. Value at stake, not a savings forecast: JSL\'s own data sets the real figure.');
  // loss pool heatmap
  ph(s, 0.5, 1.35, 4.35, 'Loss pool per line, ₹ cr a year');
  T(s, '0.4 Mt × downgrade rate × price discount × ₹167,407/t', { x: 0.5, y: 1.7, w: 4.35, h: 0.24, fontSize: 9.5, italic: true, color: C.slate });
  const heat = v => (v < 6 ? [C.h1, C.ink] : v < 12 ? [C.h2, C.ink] : v < 20 ? [C.h3, C.white] : [C.h4, C.white]);
  const grid = [['0.5% of volume', 3.3, 5.0, 6.7], ['1.0% of volume', 6.7, 10.0, 13.4], ['2.0% of volume', 13.4, 20.1, 26.8]];
  const rows = [[hd('Downgraded'), hd('10% off', { align: 'center' }), hd('15% off', { align: 'center' }), hd('20% off', { align: 'center' })]];
  grid.forEach((r, i) => rows.push([cl(r[0], { bold: true })].concat(r.slice(1).map((v, j) => {
    const base = i === 1 && j === 0;
    const [f, c] = heat(v);
    return cl(v.toFixed(1), { align: 'center', bold: true, fontSize: 12, fill: { color: base ? C.navy : f }, color: base ? C.white : c });
  }))));
  table(s, rows, { x: 0.5, y: 2.0, w: 4.35, colW: [1.35, 1.0, 1.0, 1.0], rowH: [0.3, 0.42, 0.42, 0.42] });
  T(s, [{ text: 'Base case (dark): ', options: { bold: true, color: C.navy } }, { text: '₹6.7 cr a year per line, ~₹43 cr across 2.57 Mt. Non-prime coil sells 10-30% below prime.', options: { color: C.slate } }], { x: 0.5, y: 3.62, w: 4.35, h: 0.62, fontSize: 9.5 });

  // value tree: the prize branching into five levers, each with its pilot measure
  ph(s, 5.05, 1.35, 7.78, 'Where the value comes from, and how the pilot measures it');
  RR(s, 5.15, 2.28, 1.55, 1.05, C.accent);
  T(s, [{ text: '~₹6.7 cr', options: { bold: true, fontSize: 18 } }, { text: '\na year at stake\nper line', options: { fontSize: 9.5 } }],
    { x: 5.15, y: 2.28, w: 1.55, h: 1.05, align: 'center', valign: 'middle', color: C.white });
  const lev = [['coins', 'Fewer downgrades', 'downgrade rate by line and grade'], ['users', 'Fewer customer claims', 'claims per 1,000 t shipped'],
    ['refresh-cw', 'Less rework', 'rework tonnes per month'], ['wrench', 'Faster root-cause fixes', 'hours from defect to fix'],
    ['trending-up', 'Premium orders', 'share of exposed-finish orders']];
  const ly0 = 1.72, lstep = 0.48, spineX = 6.95;
  R(s, 6.7, 2.795, spineX - 6.7, 0.02, C.grey);
  R(s, spineX, ly0 + 0.2, 0.02, lstep * 4, C.grey);
  lev.forEach(([ic, n, m], i) => {
    const y = ly0 + i * lstep;
    R(s, spineX, y + 0.2, 0.25, 0.02, C.grey);
    RR(s, 7.2, y, 5.55, 0.42, C.neutral);
    badge(s, ic, 7.27, y + 0.05, 0.32, C.accent);
    T(s, n, { x: 7.68, y, w: 2.2, h: 0.42, fontSize: 10.5, bold: true, color: C.navy, valign: 'middle' });
    T(s, [{ text: 'measure: ', options: { color: C.muted } }, { text: m, options: { color: C.slate } }], { x: 9.9, y, w: 2.8, h: 0.42, fontSize: 9, valign: 'middle' });
  });

  // beyond P&L: photo tiles
  ph(s, 0.5, 4.38, 8.4, 'Impact beyond the P&L');
  [['img/det_scratches.png', 'Coverage', 'every coil, both faces'],
   ['img/photos/vb_coaches.jpg', 'Customers', 'a defect map per coil'],
   ['img/photos/mill.jpg', 'Process', 'defect tied to its heat'],
   ['img/cover_coil.jpg', 'Safety', 'hold before a strip break'],
   ['img/photos/okp_mill.jpg', 'Sustainability', 'less rework and remelt'],
   ['img/photos/worker.jpg', 'People', 'inspectors decide, not search'],
  ].forEach(([img, h, b], i) => {
    const x = 0.5 + i * 1.415;
    RR(s, x, 4.74, 1.35, 1.7, C.neutral);
    photo(s, img, x + 0.06, 4.8, 1.23, 0.92);
    T(s, h, { x: x + 0.04, y: 5.76, w: 1.27, h: 0.26, fontSize: 10.5, bold: true, color: C.navy, align: 'center' });
    T(s, b, { x: x + 0.05, y: 6.02, w: 1.25, h: 0.38, fontSize: 9, color: C.slate, align: 'center' });
  });

  // ask
  RR(s, 9.1, 4.38, 3.73, 2.06, C.navy);
  T(s, 'WHAT WE ASK OF JSL', { x: 9.25, y: 4.46, w: 3.5, h: 0.26, fontSize: 11, bold: true, color: C.onNavy });
  T(s, [
    { text: 'One 300-series cold-rolled line, 7 months', options: { bullet: true } },
    { text: 'Vendor quotes for cameras and lighting in P0', options: { bullet: true } },
    { text: 'Downgrade rate by line and grade', options: { bullet: true } },
    { text: 'A metallurgist at 20% for root causes', options: { bullet: true } },
  ], { x: 9.25, y: 4.76, w: 3.5, h: 1.05, fontSize: 10, color: C.white, paraSpaceAfter: 2 });
  T(s, 'In return: the measured downgrade rate, the value captured, and a go / no-go at month 7.', { x: 9.25, y: 5.84, w: 3.5, h: 0.52, fontSize: 9.5, bold: true, color: C.onNavy });
  s.addNotes('Impact. We size the prize, not the price. At a conservative 1% of volume downgraded at a 10% discount, one 0.4 Mt line puts about INR 6.7 crore a year at stake; across JSL\'s 2.57 Mt it is about INR 43 crore. Non-prime coil typically sells 10-30% below prime, so the discounts are conservative. The value comes through five levers: fewer downgrades, fewer customer claims, less rework, faster root-cause fixes, and premium exposed-finish orders. Each has a measure the pilot tracks, so the fleet decision rests on JSL\'s own numbers rather than our assumptions. Beyond the P&L: coverage, customers, process, safety, sustainability and people. Close on the ask. Proof points for the tiles: JSL supplies Vande Bharat and metro coach steel; defect families map to owners on slide 3; strip-break avoidance is the Ternium business case; JSL emission intensity was 1.76 tCO2e per tonne in FY26; JSL\'s HR AI assistant resolves over 95% of routine queries. Photo credits: coaches, Ravi Dwivedi (CC BY-SA 4.0); mill, ThyssenKrupp Steel USA (FAL); Outokumpu Tornio, Methem (CC BY 3.0); inspector in a pipe, Italsider archive (CC BY-SA 4.0); coil, Methem (public domain).');
}

pres.writeFile({ fileName: '../out/JSW_Round2_Surface_Defect_Detection.pptx' }).then(f => console.log('wrote', f));
