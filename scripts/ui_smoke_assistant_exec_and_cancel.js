/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || process.env.DEFAULT_ADMIN_USERNAME || "Sean";
const PASSWORD = process.env.QC_PASSWORD || process.env.DEFAULT_ADMIN_PASSWORD || "0357zaqxswcde";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-exec-cancel");

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

async function run() {
  ensureDir(OUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL });
  // Make open state deterministic for this smoke.
  await context.addInitScript(() => {
    try { localStorage.setItem("marllenAssistant.open.v1", "0"); } catch (e) {}
    try { localStorage.setItem("marllenAssistant.execOpen.v1", "1"); } catch (e) {}
  });
  const page = await context.newPage();

  try {
    await login(page);
    await page.goto(`${BASE_URL}/reports/tags`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(250);
    await screenshot(page, "01-after-login.png");

    // Open assistant
    const panelVisible0 = await page.locator("#marllenAssistantPanel").isVisible().catch(() => false);
    if (!panelVisible0) {
      await page.locator("#marllenAssistantFab").click();
      await page.waitForSelector("#marllenAssistantPanel:not(.hidden)");
    }
    await page.waitForTimeout(200);
    await screenshot(page, "02-open.png");

    // Ensure input enabled
    await page.waitForFunction(() => {
      const el = document.getElementById("marllenAssistantInput");
      return !!(el && !el.disabled);
    }, { timeout: 60000 });

    // Send a question then cancel quickly (should still be pending for a moment).
    const q = "先别回答：我想看你在执行什么。请开始分析，然后我会点终止。";
    await page.locator("#marllenAssistantInput").fill(q);
    await page.locator("#marllenAssistantSend").click();

    // Exec panel should appear immediately while busy.
    await page.waitForSelector("#marllenAssistantExec:not(.hidden)", { timeout: 5000 });
    await screenshot(page, "03-exec-visible.png");

    // Cancel (ideally while pending/running).
    await page.locator("#marllenAssistantCancel").click();

    // Wait for cancel message to land (backend inserts assistant message with meta.type=job_canceled).
    await page.waitForFunction(async () => {
      try {
        const resp = await fetch("/api/marllen-assistant/thread?limit=30", { credentials: "same-origin" });
        const json = await resp.json().catch(() => ({}));
        const msgs = Array.isArray(json && json.messages) ? json.messages : [];
        return msgs.some((m) => m && String(m.role || "") === "assistant" && m.meta && String(m.meta.type || "") === "job_canceled");
      } catch (e) {
        return false;
      }
    }, { timeout: 15000 });

    // Input should be re-enabled after cancel.
    await page.waitForFunction(() => {
      const el = document.getElementById("marllenAssistantInput");
      return !!(el && !el.disabled);
    }, { timeout: 15000 });

    await page.waitForTimeout(200);
    await screenshot(page, "04-after-cancel.png");

    console.log("OK: assistant exec+cancel smoke finished");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

