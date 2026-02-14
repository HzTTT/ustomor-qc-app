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
  // Prefer same-origin ECharts (avoid CDN flakiness).
  await context.route("https://unpkg.com/**", (route) => route.abort());
  await context.route("https://cdn.jsdelivr.net/**", (route) => route.abort());
  // Make open state deterministic for this smoke.
  await context.addInitScript(() => {
    try { localStorage.setItem("marllenAssistant.open.v1", "0"); } catch (e) {}
  });
  const page = await context.newPage();

  try {
    await login(page);
    await page.goto(`${BASE_URL}/conversations`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(250);
    await screenshot(page, "01-after-login.png");

    // Open assistant panel
    const panelVisible0 = await page.locator("#marllenAssistantPanel").isVisible().catch(() => false);
    if (!panelVisible0) {
      await tapByCenter(page, "#marllenAssistantFab");
      await page.waitForSelector("#marllenAssistantPanel:not(.hidden)");
    }
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

    // Wait for assistant to answer (charts are optional in read-only mode).
    await page.waitForFunction(() => {
      const err = document.getElementById("marllenAssistantError");
      if (err && !err.classList.contains("hidden") && (err.textContent || "").trim()) return true;
      const dots = document.querySelector(".marllen-typing-dots");
      return !dots;
    }, { timeout: 180000 }).catch(() => {});

    const errText = await page.locator("#marllenAssistantError").textContent().catch(() => "");
    if (errText && String(errText).trim()) {
      throw new Error("assistant error banner: " + String(errText).trim().slice(0, 300));
    }
    const finalState = await page.evaluate(async () => {
      const resp = await fetch("/api/marllen-assistant/thread?limit=30", { credentials: "same-origin" });
      const json = await resp.json().catch(() => ({}));
      return { ok: resp.ok, json };
    });
    const msgs = finalState && finalState.json && Array.isArray(finalState.json.messages) ? finalState.json.messages : [];
    const hasAssistant = msgs.some((m) => m && String(m.role || "") === "assistant" && String(m.content || "").trim());
    if (!hasAssistant) {
      throw new Error("assistant produced no assistant message");
    }

    // If charts exist, ensure ECharts actually rendered a canvas (not just an empty container).
    await page.waitForFunction(() => {
      const nodes = Array.from(document.querySelectorAll("#marllenAssistantMessages .marllen-chart"));
      if (!nodes.length) return true; // charts are optional here
      for (const el of nodes) {
        const txt = (el.textContent || "").trim();
        if (txt.includes("图表组件加载失败") || txt.includes("图表渲染失败")) {
          throw new Error("chart failed: " + txt.slice(0, 80));
        }
        const canvas = el.querySelector("canvas");
        if (!canvas) continue;
        const r = canvas.getBoundingClientRect();
        if (r.width > 10 && r.height > 10) return true;
      }
      return false;
    }, { timeout: 60000 });
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
