/**
 * Check the claim the deck makes about its own QR code and URL.
 *
 * The deck tells a judge to open https://surface-vision.github.io and says three
 * things about what happens: the page loads without an install, it runs the real
 * detector in the tab, and no image is uploaded anywhere. This script drives the
 * live published page in real Chrome, clicks every built-in sample, and records
 * every network request the page makes, so all three claims are checked rather
 * than asserted.
 *
 * No new dependency: puppeteer-core and a pinned Chrome build already exist in
 * site/verify/node_modules for the browser parity harness. They are used
 * read-only; nothing under site/ is written.
 *
 * Run:
 *   node deck/build/verify_live_demo.mjs
 *   node deck/build/verify_live_demo.mjs --url http://127.0.0.1:8790/ --json out.json
 */

import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ROOT = '/Users/prathmeshwalimbe/Downloads/JSW-PS1';
const puppeteer = require(path.join(
  ROOT, 'site/verify/node_modules/puppeteer-core/lib/cjs/puppeteer/puppeteer-core.js'));

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const URL_BASE = arg('--url', 'https://surface-vision.github.io/');
const OUT = arg('--json', path.join(ROOT, 'deck/build/live_demo_check.json'));

function findChrome() {
  const cache = `${process.env.HOME}/.cache/puppeteer/chrome`;
  if (existsSync(cache)) {
    for (const dir of readdirSync(cache).sort().reverse()) {
      for (const rel of [
        'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        'chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        'chrome-linux64/chrome',
      ]) {
        const p = path.join(cache, dir, rel);
        if (existsSync(p)) return p;
      }
    }
  }
  for (const p of [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
  ]) if (existsSync(p)) return p;
  throw new Error('no Chrome found; set CHROME=/path/to/chrome');
}

const executablePath = process.env.CHROME && existsSync(process.env.CHROME)
  ? process.env.CHROME : findChrome();

const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });

const requests = [];
const pageErrors = [];
const consoleErrors = [];
const badRequests = [];
page.on('request', (r) => {
  const body = r.postData();
  requests.push({
    method: r.method(),
    url: r.url().length > 160 ? `${r.url().slice(0, 157)}...` : r.url(),
    type: r.resourceType(),
    post_bytes: body ? Buffer.byteLength(body) : 0,
  });
});
page.on('pageerror', (e) => pageErrors.push(String(e.message)));
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('requestfailed', (r) => badRequests.push(['FAILED', r.url()]));
page.on('requestfinished', (r) => {
  const s = r.response()?.status();
  if (s && s >= 400) badRequests.push([s, r.url()]);
});

await page.evaluateOnNewDocument(() => {
  window.__caught = [];
  document.addEventListener('surface:detections', (e) => window.__caught.push(e.detail));
});

const t0 = Date.now();
const response = await page.goto(URL_BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
const domReadyMs = Date.now() - t0;
const httpStatus = response ? response.status() : 0;
const title = await page.title();

const chips = await page.$$('#chips .chip');
if (!chips.length) throw new Error('no sample chips on the page');
for (let i = 0; i < chips.length; i++) {
  await chips[i].click();
  await page.waitForFunction((n) => window.__caught.length > n,
    { timeout: 180000, polling: 60 }, i);
}
const caught = await page.evaluate(() => window.__caught);
const chrome = await browser.version();
await browser.close();

const detections = caught.reduce((a, c) => a + c.detections.length, 0);
const perImage = caught.map((c) => ({
  image: c.image,
  detections: c.detections.length,
  infer_ms: Number(c.timing.infer.toFixed(1)),
}));
const inferMs = caught.map((c) => c.timing.infer).sort((a, b) => a - b);
const median = inferMs[Math.floor(inferMs.length / 2)];
const origin = new URL(URL_BASE).origin;
const thirdParty = requests.filter((r) => !r.url.startsWith(origin) && !r.url.startsWith('data:'));
const withBody = requests.filter((r) => r.post_bytes > 0);
const nonGet = requests.filter((r) => r.method !== 'GET');

const out = {
  generated_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  url: URL_BASE,
  chrome,
  http_status: httpStatus,
  title,
  dom_ready_ms: domReadyMs,
  samples_clicked: chips.length,
  frames_with_detections: caught.length,
  total_detections: detections,
  browser_infer_ms: { min: inferMs[0], median, max: inferMs[inferMs.length - 1] },
  per_image: perImage,
  requests_total: requests.length,
  requests_non_get: nonGet.length,
  requests_with_body: withBody.length,
  third_party_hosts: [...new Set(thirdParty.map((r) => new URL(r.url).host))],
  requests,
  page_errors: pageErrors,
  console_errors: consoleErrors,
  bad_requests: badRequests,
};
out.verdict = (httpStatus === 200 && caught.length === chips.length && detections > 0
  && nonGet.length === 0 && withBody.length === 0 && pageErrors.length === 0
  && badRequests.length === 0) ? 'PASS' : 'FAIL';

writeFileSync(OUT, `${JSON.stringify(out, null, 2)}\n`);
console.log(`url                 ${out.url}`);
console.log(`http status         ${out.http_status}   dom ready ${out.dom_ready_ms} ms`);
console.log(`chrome              ${out.chrome}`);
console.log(`samples clicked     ${out.samples_clicked}`);
console.log(`frames detected on  ${out.frames_with_detections}`);
console.log(`detections total    ${out.total_detections}`);
console.log(`in-tab session.run  min ${out.browser_infer_ms.min.toFixed(1)} / `
  + `median ${out.browser_infer_ms.median.toFixed(1)} / max ${out.browser_infer_ms.max.toFixed(1)} ms`);
console.log(`requests            ${out.requests_total} total, ${out.requests_non_get} non-GET, `
  + `${out.requests_with_body} carrying a request body`);
console.log(`third-party hosts   ${out.third_party_hosts.join(', ') || 'none'}`);
console.log(`page errors         ${out.page_errors.length}`);
console.log(`failed / 4xx / 5xx  ${out.bad_requests.length}`);
console.log(`verdict             ${out.verdict}`);
console.log(`wrote               ${OUT}`);
process.exit(out.verdict === 'PASS' ? 0 : 1);
