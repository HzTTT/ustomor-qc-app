/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || process.env.DEFAULT_ADMIN_USERNAME || "Sean";
const PASSWORD = process.env.QC_PASSWORD || process.env.DEFAULT_ADMIN_PASSWORD || "0357zaqxswcde";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-chart-render");

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

async function waitPendingJobIfAny(page, timeoutMs) {
  const deadline = Date.now() + (timeoutMs || 60000);
  try {
    while (Date.now() < deadline) {
      const data = await page.evaluate(async () => {
        const resp = await fetch("/api/marllen-assistant/thread?limit=1", { credentials: "same-origin" });
        const json = await resp.json().catch(() => ({}));
        return { ok: resp.ok, json };
      });
      const pending = data && data.json ? data.json.pending_job : null;
      const st = pending && pending.status ? String(pending.status) : "";
      if (!pending || !pending.id || !st || st === "done" || st === "error") return;
      await page.waitForTimeout(1500);
    }
  } catch (e) {
    // ignore
  }
}

async function run() {
  ensureDir(OUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL });

  // Force ECharts to use the same-origin vendor bundle (Shanghai/offline friendly).
  await context.route("https://unpkg.com/**", (route) => route.abort());
  await context.route("https://cdn.jsdelivr.net/**", (route) => route.abort());

  // Make open state deterministic for this smoke.
  await context.addInitScript(() => {
    try { localStorage.setItem("marllenAssistant.open.v1", "0"); } catch (e) {}
  });

  const page = await context.newPage();
  try {
    await login(page);
    await page.goto(`${BASE_URL}/reports/tags`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(250);
    await waitPendingJobIfAny(page, 45000);
    await screenshot(page, "01-page.png");

    const panelVisible0 = await page.locator("#marllenAssistantPanel").isVisible().catch(() => false);
    if (!panelVisible0) {
      await page.locator("#marllenAssistantFab").click();
      await page.waitForSelector("#marllenAssistantPanel:not(.hidden)", { timeout: 15000 });
    }
    await page.waitForTimeout(200);
    await screenshot(page, "02-open.png");

    // Ask for a prompt that tends to trigger the built-in DB snapshot context (and thus chart blocks).
    const q = "查数据库";
    await page.locator("#marllenAssistantInput").fill(q);
    await page.locator("#marllenAssistantSend").click();

    // Best-effort: wait for completion or an error banner.
    await page.waitForFunction(() => {
      const err = document.getElementById("marllenAssistantError");
      if (err && !err.classList.contains("hidden") && (err.textContent || "").trim()) return true;
      const dots = document.querySelector(".marllen-typing-dots");
      return !dots;
    }, { timeout: 180000 }).catch(() => {});

    // Confirm the assistant actually returned chart specs.
    const apiState = await page.evaluate(async () => {
      const resp = await fetch("/api/marllen-assistant/thread?limit=30", { credentials: "same-origin" });
      const json = await resp.json().catch(() => ({}));
      return { ok: resp.ok, json };
    });
    const msgs = apiState && apiState.json && Array.isArray(apiState.json.messages) ? apiState.json.messages : [];
    const hasCharts = msgs.some((m) => m && m.role === "assistant" && m.meta && Array.isArray(m.meta.charts) && m.meta.charts.length);
    if (!hasCharts) {
      throw new Error("assistant produced no charts meta; cannot validate chart rendering");
    }

    // Chart container appears once meta.charts exists.
    await page.waitForSelector("#marllenAssistantMessages .marllen-chart", { timeout: 60000 });
    // Real assertion: ECharts actually rendered a canvas with non-trivial size.
    await page.waitForFunction(() => {
      const nodes = Array.from(document.querySelectorAll("#marllenAssistantMessages .marllen-chart"));
      if (!nodes.length) return false;
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
    }, { timeout: 90000 });

    await page.waitForTimeout(300);
    await screenshot(page, "03-chart.png");

    console.log("OK: assistant chart render smoke passed");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
