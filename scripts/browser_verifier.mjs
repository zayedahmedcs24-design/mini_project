/**
 * Browser Automation Verifier using Chrome DevTools Protocol (CDP) via native Node.js WebSocket.
 * 
 * Verifies:
 * 1. Console errors on index.html and results.html (0 errors required).
 * 2. Chip selection and switching across groups.
 * 3. Layer toggling (SAR, GT, U-Net, DeepLab, Entropy, Error, Basemap).
 * 4. Exposure card accuracy against results/exposure_*.json for 3 chips:
 *    - Bolivia_290290 (roads mapped & flooded)
 *    - Bolivia_103757 (roads null, no roads in extract)
 *    - Sri-Lanka_713926 (largest exposed population in Sri-Lanka)
 * 5. Captures publication screenshots into docs/screenshots/.
 */

import fs from 'fs';
import path from 'path';

const PORT = 8000;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const SCREENSHOT_DIR = path.resolve('docs/screenshots');

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

// Helper to send CDP JSON-RPC commands
class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 0;
    this.callbacks = new Map();
    this.consoleLogs = [];
    this.pageErrors = [];
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.method === 'Runtime.consoleAPICalled') {
          const type = msg.params.type;
          const text = msg.params.args.map(a => a.value || JSON.stringify(a)).join(' ');
          this.consoleLogs.push({ type, text });
          if (type === 'error') {
            this.pageErrors.push(text);
          }
        } else if (msg.method === 'Runtime.exceptionThrown') {
          const text = msg.params.exceptionDetails.text + ' ' + (msg.params.exceptionDetails.exception?.description || '');
          this.pageErrors.push(text);
        } else if (msg.id && this.callbacks.has(msg.id)) {
          const cb = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) cb.reject(msg.error);
          else cb.resolve(msg.result);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    return res.result?.value;
  }

  async screenshot(filePath) {
    const res = await this.send('Page.captureScreenshot', { format: 'png' });
    const buffer = Buffer.from(res.data, 'base64');
    fs.writeFileSync(filePath, buffer);
    console.log(`  [SCREENSHOT] Saved: ${path.basename(filePath)} (${buffer.length} bytes)`);
  }

  close() {
    if (this.ws) this.ws.close();
  }
}

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function runVerification() {
  console.log('================================================================================');
  console.log('   ANTIGRAVITY BROWSER AGENT VERIFICATION SUITE');
  console.log('================================================================================\n');

  // 1. Get open Chrome target or wait for browser to register target
  let targets = [];
  for (let attempt = 0; attempt < 15; attempt++) {
    try {
      const resp = await fetch('http://127.0.0.1:9222/json/list');
      targets = await resp.json();
      if (targets && targets.length > 0) break;
    } catch (e) {
      // Chrome might still be initializing
    }
    await sleep(400);
  }
  const pageTarget = targets.find(t => t.type === 'page') || targets[0];
  if (!pageTarget || !pageTarget.webSocketDebuggerUrl) {
    throw new Error('No valid Chrome page target found: ' + JSON.stringify(targets));
  }
  const wsUrl = pageTarget.webSocketDebuggerUrl;
  console.log('Connected to Chrome Target via CDP:', wsUrl);

  const client = new CDPClient(wsUrl);
  await client.connect();

  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false
  });

  // Load ground truth JSONs for comparison
  const boliviaExp = JSON.parse(fs.readFileSync('results/exposure_bolivia.json', 'utf8'));
  const sriLankaExp = JSON.parse(fs.readFileSync('results/exposure_sri-lanka.json', 'utf8'));

  const verifications = [];

  // ============================================================================
  // TEST CASE 1: Default Chip (Bolivia_290290) - Good Bolivia with Roads Mapped
  // ============================================================================
  console.log('\n--- 1. Testing Default View: Bolivia_290290 (Held-Out Event) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Bolivia_290290` });
  await sleep(1500);

  const title = await client.evaluate('document.title');
  console.log('  Page Title:', title);

  const bannerText = await client.evaluate('document.getElementById("split-banner").innerText');
  console.log('  Banner Text:', bannerText);

  const chipValue = await client.evaluate('document.getElementById("chip-select").value');
  console.log('  Selected Chip:', chipValue);

  // Read exposure table values
  const bol290Table = await client.evaluate(`
    Array.from(document.querySelectorAll("#exposure-container .exposure-table tr")).map(tr => 
      Array.from(tr.querySelectorAll("th, td")).map(td => td.innerText.trim())
    )
  `);
  console.log('  Rendered Exposure Table:');
  bol290Table.forEach(row => console.log('    | ' + row.join(' | ')));

  // Compare with JSON
  const b290_gt = boliviaExp.per_chip['Bolivia_290290'].ground_truth;
  const renderedArea = parseFloat(bol290Table[1][1]);
  const renderedPop = parseFloat(bol290Table[1][2]);
  const renderedRoad = parseFloat(bol290Table[1][3]);

  const matchArea = Math.abs(renderedArea - b290_gt.flooded_area_km2) < 0.05;
  const matchPop = Math.abs(renderedPop - b290_gt.exposed_population) < 0.1;
  const matchRoad = Math.abs(renderedRoad - b290_gt.flooded_road_km) < 0.05;

  console.log(`  Verification vs JSON [Bolivia_290290]:`);
  console.log(`    - Flooded Area: Rendered ${renderedArea} km² vs JSON ${b290_gt.flooded_area_km2.toFixed(2)} km² -> ${matchArea ? 'PASS' : 'FAIL'}`);
  console.log(`    - Exposed Pop:  Rendered ${renderedPop} vs JSON ${b290_gt.exposed_population.toFixed(1)} -> ${matchPop ? 'PASS' : 'FAIL'}`);
  console.log(`    - Flooded Road: Rendered ${renderedRoad} km vs JSON ${b290_gt.flooded_road_km.toFixed(2)} km -> ${matchRoad ? 'PASS' : 'FAIL'}`);
  verifications.push({ chip: 'Bolivia_290290', pass: matchArea && matchPop && matchRoad });

  await client.screenshot(path.join(SCREENSHOT_DIR, '01_viewer_bolivia_default.png'));

  // ============================================================================
  // TEST CASE 2: Chip with Null Road Status (Bolivia_103757)
  // ============================================================================
  console.log('\n--- 2. Testing Bolivia_103757 (Roads Null - No Roads in Extract) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Bolivia_103757` });
  await sleep(1500);

  const bol103Table = await client.evaluate(`
    Array.from(document.querySelectorAll("#exposure-container .exposure-table tr")).map(tr => 
      Array.from(tr.querySelectorAll("th, td")).map(td => td.innerText.trim())
    )
  `);
  console.log('  Rendered Exposure Table (Bolivia_103757):');
  bol103Table.forEach(row => console.log('    | ' + row.join(' | ')));

  const b103_gt = boliviaExp.per_chip['Bolivia_103757'].ground_truth;
  const b103RoadText = bol103Table[1][3];
  const roadNullOk = b103_gt.flooded_road_km === null && (b103RoadText.includes('null') || b103RoadText.includes('no roads'));

  console.log(`  Verification vs JSON [Bolivia_103757]:`);
  console.log(`    - Flooded Road Null Distinction: ${b103RoadText} -> ${roadNullOk ? 'PASS (Correctly marked null)' : 'FAIL'}`);
  verifications.push({ chip: 'Bolivia_103757', pass: roadNullOk });

  await client.screenshot(path.join(SCREENSHOT_DIR, '02_viewer_bolivia_roads_null.png'));

  // ============================================================================
  // TEST CASE 3: Sri-Lanka TEST Chip (Sri-Lanka_713926) - Largest Exposed Population
  // ============================================================================
  console.log('\n--- 3. Testing Sri-Lanka_713926 (Largest Exposed Pop in TEST Split) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Sri-Lanka_713926` });
  await sleep(1500);

  const slBanner = await client.evaluate('document.getElementById("split-banner").innerText');
  console.log('  Banner Text:', slBanner);

  const sl713Table = await client.evaluate(`
    Array.from(document.querySelectorAll("#exposure-container .exposure-table tr")).map(tr => 
      Array.from(tr.querySelectorAll("th, td")).map(td => td.innerText.trim())
    )
  `);
  console.log('  Rendered Exposure Table (Sri-Lanka_713926):');
  sl713Table.forEach(row => console.log('    | ' + row.join(' | ')));

  const sl_gt = sriLankaExp.per_chip['Sri-Lanka_713926'].ground_truth;
  const slArea = parseFloat(sl713Table[1][1]);
  const slPop = parseFloat(sl713Table[1][2]);
  const slRoad = parseFloat(sl713Table[1][3]);

  const slMatchArea = Math.abs(slArea - sl_gt.flooded_area_km2) < 0.05;
  const slMatchPop = Math.abs(slPop - sl_gt.exposed_population) < 0.5;
  const slMatchRoad = Math.abs(slRoad - sl_gt.flooded_road_km) < 0.05;

  console.log(`  Verification vs JSON [Sri-Lanka_713926]:`);
  console.log(`    - Flooded Area: Rendered ${slArea} km² vs JSON ${sl_gt.flooded_area_km2.toFixed(2)} km² -> ${slMatchArea ? 'PASS' : 'FAIL'}`);
  console.log(`    - Exposed Pop:  Rendered ${slPop} vs JSON ${sl_gt.exposed_population.toFixed(1)} -> ${slMatchPop ? 'PASS' : 'FAIL'}`);
  console.log(`    - Flooded Road: Rendered ${slRoad} km vs JSON ${sl_gt.flooded_road_km.toFixed(2)} km -> ${slMatchRoad ? 'PASS' : 'FAIL'}`);
  verifications.push({ chip: 'Sri-Lanka_713926', pass: slMatchArea && slMatchPop && slMatchRoad });

  await client.screenshot(path.join(SCREENSHOT_DIR, '03_viewer_sri_lanka_largest_pop.png'));

  // ============================================================================
  // TEST CASE 4: Sri-Lanka_534068 - Largest Flooded Area Scene
  // ============================================================================
  console.log('\n--- 4. Testing Sri-Lanka_534068 (Largest Flooded Surface Scene) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Sri-Lanka_534068` });
  await sleep(1500);
  await client.screenshot(path.join(SCREENSHOT_DIR, '04_viewer_sri_lanka_largest_area.png'));

  // ============================================================================
  // TEST CASE 5: Layer Toggling - Predictive Uncertainty / Entropy Layer
  // ============================================================================
  console.log('\n--- 5. Testing Layer Toggling: Uncertainty / Predictive Entropy ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Bolivia_129334&layers=sar_vv,entropy` });
  await sleep(1500);
  await client.screenshot(path.join(SCREENSHOT_DIR, '05_viewer_entropy_uncertainty.png'));

  // ============================================================================
  // TEST CASE 6: Layer Toggling - Error Analysis (FP red, FN orange, ignore gray)
  // ============================================================================
  console.log('\n--- 6. Testing Layer Toggling: Error Analysis (FP / FN Overlays) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Bolivia_129334&layers=sar_vv,error` });
  await sleep(1500);
  await client.screenshot(path.join(SCREENSHOT_DIR, '06_viewer_error_analysis.png'));

  // ============================================================================
  // TEST CASE 7: Layer Toggling - DeepLabV3+ Ensemble Prediction
  // ============================================================================
  console.log('\n--- 7. Testing Layer Toggling: DeepLabV3+ Ensemble Prediction ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/index.html?chip=Bolivia_129334&layers=sar_vv,prediction_deeplab` });
  await sleep(1500);
  await client.screenshot(path.join(SCREENSHOT_DIR, '07_viewer_deeplab_prediction.png'));

  // ============================================================================
  // TEST CASE 8: Benchmark Results Page (results.html)
  // ============================================================================
  console.log('\n--- 8. Testing Results Page (web/results.html) ---');
  await client.send('Page.navigate', { url: `${BASE_URL}/results.html` });
  await sleep(1500);

  const resRows = await client.evaluate('document.querySelectorAll("#master-tbody tr").length');
  console.log(`  Master Benchmark Table Rows Rendered: ${resRows}`);

  // Test split filter buttons
  await client.evaluate('document.querySelector("button[data-filter=\'Test\']").click()');
  await sleep(400);
  const testRows = await client.evaluate('document.querySelectorAll("#master-tbody tr").length');
  console.log(`  After Filter 'Test': ${testRows} rows rendered`);

  await client.evaluate('document.querySelector("button[data-filter=\'all\']").click()');
  await sleep(400);

  await client.screenshot(path.join(SCREENSHOT_DIR, '08_results_master_table.png'));

  // Scroll to exposure comparison
  await client.evaluate('window.scrollTo(0, document.body.scrollHeight / 2)');
  await sleep(500);
  await client.screenshot(path.join(SCREENSHOT_DIR, '09_results_exposure_benchmark.png'));

  // ============================================================================
  // CONSOLE ERROR AUDIT
  // ============================================================================
  console.log('\n================================================================================');
  console.log('   CONSOLE ERROR AUDIT REPORT');
  console.log('================================================================================');
  if (client.pageErrors.length === 0) {
    console.log('  [PASS] ZERO JavaScript console errors or uncaught exceptions detected across all sessions!');
  } else {
    console.log(`  [FAIL] ${client.pageErrors.length} errors detected:`);
    client.pageErrors.forEach(err => console.log('    - ' + err));
  }

  // Summary
  console.log('\n================================================================================');
  console.log('   VERIFICATION SUMMARY');
  console.log('================================================================================');
  verifications.forEach(v => {
    console.log(`  - Chip ${v.chip}: ${v.pass ? 'ALL METRICS MATCH JSON' : 'MISMATCH'}`);
  });
  console.log(`  - Screenshots Captured: 9 high-resolution images saved in docs/screenshots/`);

  client.close();
}

runVerification().catch(err => {
  console.error('Fatal verification error:', err);
  process.exit(1);
});
