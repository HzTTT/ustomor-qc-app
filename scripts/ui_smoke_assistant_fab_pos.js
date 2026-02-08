/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || process.env.DEFAULT_ADMIN_USERNAME || "Sean";
const PASSWORD = process.env.QC_PASSWORD || process.env.DEFAULT_ADMIN_PASSWORD || "0357zaqxswcde";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-fab-pos");

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

async function screenshot(page, name) {
  const file = path.join(OUT_DIR, name);
  await page.screenshot({ path: file, fullPage: true });
  console.log("screenshot:", file);
}

async function login(page) {
  await page.goto(`${BASE_URL}/login`, { waitUntil: "domcontentloaded" });
  await page.locator('input[name="username"]').fill(USERNAME);
  await page.locator('input[name="password"]').fill(PASSWORD);
  await Promise.all([
    page.waitForNavigation({ waitUntil: "domcontentloaded" }),
    page.locator('button[type="submit"], button:has-text("登录")').first().click(),
  ]);
}

function approxEqual(a, b, tol = 3) {
  return Math.abs(Number(a) - Number(b)) <= tol;
}

async function rootPos(page) {
  return page.evaluate(() => {
    const el = document.getElementById("marllenAssistant");
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
}

async function run() {
  ensureDir(OUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    baseURL: BASE_URL,
    viewport: { width: 1280, height: 720 },
  });
  const page = await context.newPage();

  try {
    await login(page);
    await page.waitForSelector("#marllenAssistantFab");

    const intro = await page.locator("#marllenAssistantFab span.leading-tight > span").nth(1).innerText();
    if (String(intro || "").trim() !== "欢迎咨询任何关于客服的问题!") {
      throw new Error(`unexpected FAB intro text: ${JSON.stringify(intro)}`);
    }

    // Drag FAB close to the right edge.
    const fab = page.locator("#marllenAssistantFab");
    const box = await fab.boundingBox();
    if (!box) throw new Error("FAB boundingBox is null");
    const fromX = box.x + box.width / 2;
    const fromY = box.y + box.height / 2;
    const toX = 1272; // near viewport right edge, clamp happens in JS
    const toY = fromY;

    await page.mouse.move(fromX, fromY);
    await page.mouse.down();
    await page.mouse.move(toX, toY, { steps: 16 });
    await page.mouse.up();
    await page.waitForTimeout(120);
    const before = await rootPos(page);
    if (!before) throw new Error("missing #marllenAssistant");
    await screenshot(page, "01-after-drag.png");

    // Open panel (1st click may be suppressed after drag).
    await fab.click();
    await page.waitForTimeout(80);
    if (!(await page.locator("#marllenAssistantPanel").isVisible().catch(() => false))) {
      await fab.click();
    }
    await page.waitForSelector("#marllenAssistantPanel:not(.hidden)", { timeout: 4000 });
    await page.waitForTimeout(120);
    await screenshot(page, "02-open.png");

    // Collapse and verify it returns to the pre-open position.
    await page.locator("#marllenAssistantMin").click();
    await page.waitForSelector("#marllenAssistantFab:not(.hidden)", { timeout: 4000 });
    await page.waitForTimeout(120);
    const after = await rootPos(page);
    if (!after) throw new Error("missing #marllenAssistant after close");
    await screenshot(page, "03-collapsed.png");

    if (!approxEqual(after.left, before.left) || !approxEqual(after.top, before.top)) {
      throw new Error(`FAB pos mismatch: before=${JSON.stringify(before)} after=${JSON.stringify(after)}`);
    }

    console.log("OK: assistant FAB pos restored after collapse");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

