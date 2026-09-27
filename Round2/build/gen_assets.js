// Renders deck icons (Lucide, ISC licence) to PNG in brand colours, and crops the cover photo.
// Run once before build_round2.js:  node gen_assets.js
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const OUT = path.join(__dirname, 'img', 'icons');
fs.mkdirSync(OUT, { recursive: true });
const ICONS = ['flame', 'factory', 'droplets', 'layers', 'sparkles', 'truck', 'camera', 'shield-check',
  'scan-search', 'gauge', 'split', 'refresh-cw', 'coins', 'cpu', 'users', 'wrench', 'leaf', 'user-check',
  'scan-line', 'target', 'hard-hat', 'clock', 'microscope', 'brain-circuit', 'archive', 'eye-off',
  'trending-up', 'ship', 'badge-check', 'lock'];
const COLOURS = { white: '#FFFFFF', navy: '#143A5A', accent: '#0086C3' };

(async () => {
  for (const name of ICONS) {
    const svg = fs.readFileSync(require.resolve(`lucide-static/icons/${name}.svg`), 'utf8');
    for (const [tag, hex] of Object.entries(COLOURS)) {
      const coloured = svg.replace(/currentColor/g, hex).replace(/width="24"/, 'width="256"').replace(/height="24"/, 'height="256"');
      await sharp(Buffer.from(coloured)).png().toFile(path.join(OUT, `${name}-${tag}.png`));
    }
  }
  // Cover photo: public-domain coil (Methem / Mikko J. Putkonen, Wikimedia Commons), cropped portrait on the coil
  await sharp(path.join(__dirname, 'img', 'src_coil_oulu.jpg'))
    .extract({ left: 250, top: 0, width: 560, height: 801 })
    .modulate({ saturation: 0.55 })
    .jpeg({ quality: 88 })
    .toFile(path.join(__dirname, 'img', 'cover_coil.jpg'));
  console.log(`icons: ${ICONS.length} x ${Object.keys(COLOURS).length}, cover photo cropped`);
})();
