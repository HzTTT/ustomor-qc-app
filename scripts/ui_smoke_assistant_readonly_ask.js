/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium, devices } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || "agent1";
const PASSWORD = process.env.QC_PASSWORD || "agent123";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-readonly-ask");

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

async function tapByCenter(page, selector) {
  const loc = page.locator(selector);
  await loc.waitFor({ state: "visible", timeout: 15000 });
  const box = await loc.boundingBox();
  if (!box) throw new Error(`cannot resolve bounding box for ${selector}`);
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await page.touchscreen.tap(x, y);
}

async function run() {
  ensureDir(OUT_DIR);

  const device = devices["iPhone 15"];
  if (!device) throw new Error("Missing device descriptor: iPhone 15");

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ ...device, baseURL: BASE_URL });
  const page = await context.newPage();

  try {
    await login(page);
    await page.goto(`${BASE_URL}/conversations`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(250);
    await screenshot(page, "01-after-login.png");

    // Open assistant panel
    await tapByCenter(page, "#marllenAssistantFab");
    await page.waitForSelector("#marllenAssistantPanel:not(.hidden)");
    await page.waitForTimeout(250);
    await screenshot(page, "02-open.png");

    // Read-only badge should be visible for non-admin.
    const badge = page.locator("#marllenAssistantMode");
    if (await badge.count()) {
      const cls = (await badge.getAttribute("class")) || "";
      if (cls.includes("hidden")) throw new Error("expected read-only badge to be visible for non-admin");
    } else {
      throw new Error("missing #marllenAssistantMode badge");
    }

    // Send a DB overview question (fast path: no upstream AI required)
    await page.locator("#marllenAssistantInput").fill("查数据库");
    await tapByCenter(page, "#marllenAssistantSend");
    await screenshot(page, "03-after-send.png");

    // Wait for assistant to answer and charts to appear.
    await page.waitForSelector("#marllenAssistantMessages .marllen-chart", { timeout: 60000 });
    await page.waitForTimeout(500);
    await screenshot(page, "04-after-reply.png");

    console.log("OK: assistant readonly ask smoke passed");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
