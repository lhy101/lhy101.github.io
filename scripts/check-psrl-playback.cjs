// Regression for playback, including an embedded webview with a different rAF time origin.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const modules = process.env.CODEX_NODE_MODULES || path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const { chromium } = require(require.resolve('playwright', { paths: [modules, __dirname] }));
const url = process.env.PSRL_PLAYER_URL || pathToFileURL(path.resolve(__dirname, '../assets/blog/psrl-staleness/animation/index.html')).href;
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PSRL_BROWSER_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  const reports = [];
  try {
    for (const offset of [0, -120000]) {
      const page = await browser.newPage({ viewport: { width: 1024, height: 900 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      if (offset) await page.addInitScript(offset => {
        const original = window.requestAnimationFrame.bind(window);
        window.requestAnimationFrame = callback => original(time => callback(time + offset));
      }, offset);
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.psrlPlayer, null, { timeout: 90000 });
      await page.locator('#restart').click();
      await page.locator('#play').click();
      const before = await page.locator('canvas').screenshot();
      await page.locator('#speed').selectOption('2');
      await page.locator('#play').click();
      await page.waitForTimeout(5500);
      await page.locator('#play').click();
      const advanced = await page.evaluate(() => psrlPlayer.time);
      assert(advanced > 8 && advanced < 20, `Playback failed with rAF offset ${offset}: ${advanced}`);
      assert(!before.equals(await page.locator('canvas').screenshot()), 'Canvas remained at the initial frame');
      await page.waitForTimeout(250);
      assert.equal(await page.evaluate(() => psrlPlayer.time), advanced, 'Pause changed playback time');
      await page.locator('#play').click();
      await page.waitForTimeout(350);
      await page.locator('#play').click();
      const resumed = await page.evaluate(() => psrlPlayer.time);
      assert(resumed > advanced + .4 && resumed < advanced + 2, 'Resume reset or stalled playback');
      await page.locator('#timeline').fill('30');
      assert.equal(await page.evaluate(() => psrlPlayer.time), 30, 'Seeking failed');
      await page.locator('#play').click();
      await page.waitForTimeout(350);
      await page.locator('#play').click();
      assert(await page.evaluate(() => psrlPlayer.time) > 30.4, 'Playback stalled after seeking');
      await page.locator('#restart').click();
      await page.waitForTimeout(350);
      const restarted = await page.evaluate(() => psrlPlayer.time);
      assert(restarted > .3 && restarted < 2, 'Restart did not play from zero');
      assert.deepEqual(errors, [], 'Player script errors');
      reports.push({ rafOffsetMs: offset, advanced, resumed, restarted, checks: 'passed' });
      await page.close();
    }
    console.log(JSON.stringify({ url, reports }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
