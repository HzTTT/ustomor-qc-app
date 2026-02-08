/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium, devices } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || process.env.DEFAULT_ADMIN_USERNAME || "Sean";
const PASSWORD = process.env.QC_PASSWORD || process.env.DEFAULT_ADMIN_PASSWORD || "0357zaqxswcde";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-scroll-lock");

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

async function getScrollState(page) {
  return page.evaluate(() => ({
    y: window.scrollY || window.pageYOffset || 0,
    bodyPos: document.body ? (document.body.style.position || "") : "",
    bodyTop: document.body ? (document.body.style.top || "") : "",
    bodyOverflow: document.body ? (document.body.style.overflow || "") : "",
    htmlOverflow: document.documentElement ? (document.documentElement.style.overflow || "") : "",
  }));
}

async function run() {
  ensureDir(OUT_DIR);

  const device = devices["iPhone 15"];
  if (!device) throw new Error("Missing device descriptor: iPhone 15");

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ ...device, baseURL: BASE_URL });
  // Make assistant state deterministic for this smoke.
  await context.addInitScript(() => {
    try { localStorage.setItem("marllenAssistant.open.v1", "0"); } catch (e) {}
  });
  const page = await context.newPage();

  try {
    await login(page);
    await page.goto(`${BASE_URL}/settings/tags`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("#categoryList");

    // Ensure no in-flight assistant job forces auto-open (refresh-wait behavior).
    try {
      const pending = await page.evaluate(async () => {
        const resp = await fetch("/api/marllen-assistant/thread?limit=1", { credentials: "same-origin" });
        const json = await resp.json().catch(() => ({}));
        return json && json.pending_job ? json.pending_job : null;
      });
      if (pending && pending.id && pending.status && pending.status !== "done" && pending.status !== "error") {
        await page.evaluate(async (jobId) => {
          await fetch("/api/marllen-assistant/jobs/" + String(jobId) + "/cancel", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
        }, pending.id);
      }
    } catch (e) {}

    // If assistant is already open, close it first so "beforeOpen" is truly before.
    try {
      const panelVisible = await page.locator("#marllenAssistantPanel").isVisible();
      if (panelVisible) {
        await page.locator("#marllenAssistantMin").tap();
        await page.waitForSelector("#marllenAssistantFab:not(.hidden)", { state: "visible" });
        await page.waitForTimeout(150);
      }
    } catch (e) {}

    // Scroll down a bit so background scrolling is detectable.
    await page.evaluate(() => window.scrollTo(0, Math.min(900, document.body.scrollHeight)));
    const beforeOpen = await getScrollState(page);
    console.log("beforeOpen:", beforeOpen);
    await screenshot(page, "01-before-open.png");

    // Open assistant
    await page.locator("#marllenAssistantFab").tap();
    await page.waitForSelector("#marllenAssistantPanel:not(.hidden)");
    await page.waitForTimeout(250);
    const afterOpen = await getScrollState(page);
    console.log("afterOpen:", afterOpen);
    await screenshot(page, "02-after-open.png");

    // Attempt to scroll page with input (should be locked)
    await page.mouse.wheel(0, 600);
    await page.waitForTimeout(150);
    const afterWheelLocked = await getScrollState(page);
    console.log("afterWheelLocked:", afterWheelLocked);
    await screenshot(page, "03-after-wheel-locked.png");

    if (afterWheelLocked.y !== afterOpen.y) {
      throw new Error(`assistant scroll lock failed: scrollY changed from ${afterOpen.y} to ${afterWheelLocked.y}`);
    }
    if (afterOpen.bodyPos !== "fixed") {
      throw new Error(`assistant scroll lock failed: body.position is "${afterOpen.bodyPos}" (expected "fixed")`);
    }

    // Close assistant
    await page.locator("#marllenAssistantMin").tap();
    await page.waitForSelector("#marllenAssistantPanel.hidden", { state: "attached" });
    await page.waitForSelector("#marllenAssistantFab:not(.hidden)", { state: "visible" });
    await page.waitForTimeout(250);
    const afterClose = await getScrollState(page);
    console.log("afterClose:", afterClose);
    await screenshot(page, "04-after-close.png");

    // Scrolling should work again (best-effort: wheel should change scrollY)
    await page.mouse.wheel(0, 600);
    await page.waitForTimeout(150);
    const afterWheelUnlocked = await getScrollState(page);
    console.log("afterWheelUnlocked:", afterWheelUnlocked);
    await screenshot(page, "05-after-wheel-unlocked.png");

    if (afterWheelUnlocked.y === afterClose.y) {
      throw new Error(`assistant unlock failed: scrollY did not change (still ${afterWheelUnlocked.y})`);
    }

    console.log("OK: assistant scroll lock smoke passed");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
