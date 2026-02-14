/* eslint-disable no-console */

const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";
const USERNAME = process.env.QC_USERNAME || process.env.DEFAULT_ADMIN_USERNAME || "Sean";
const PASSWORD = process.env.QC_PASSWORD || process.env.DEFAULT_ADMIN_PASSWORD || "0357zaqxswcde";
const OUT_DIR = process.env.PW_OUT_DIR || path.join(process.cwd(), "output", "playwright", "assistant-refresh-wait");

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

async function getPendingJob(page) {
  try {
    const data = await page.evaluate(async () => {
      const resp = await fetch("/api/marllen-assistant/thread?limit=10", { credentials: "same-origin" });
      const json = await resp.json().catch(() => ({}));
      return { ok: resp.ok, json };
    });
    const pending = data && data.json ? data.json.pending_job : null;
    return pending && pending.id ? pending : null;
  } catch (e) {
    return null;
  }
}

async function waitUntilPendingOrFinished(page, timeoutMs) {
  const deadline = Date.now() + (timeoutMs || 10000);
  while (Date.now() < deadline) {
    const pending = await getPendingJob(page);
    const st = pending && pending.status ? String(pending.status) : "";
    if (pending && pending.id && st && st !== "done" && st !== "error") return { pending: true, job: pending };
    // Not pending: either finished quickly or transient fetch. Allow a short retry window.
    await page.waitForTimeout(300);
  }
  return { pending: false, job: null };
}

async function run() {
  ensureDir(OUT_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL });
  // Make open state deterministic for this smoke.
  await context.addInitScript(() => {
    try { localStorage.setItem("marllenAssistant.open.v1", "0"); } catch (e) {}
  });
  const page = await context.newPage();

  try {
    await login(page);
    await page.goto(`${BASE_URL}/reports/tags`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(250);
    // Keep it tolerant: assistant jobs can legitimately take > 45s for large datasets.
    await waitPendingJobIfAny(page, 180000);
    await screenshot(page, "01-initial.png");

    // Open assistant
    const panelVisible0 = await page.locator("#marllenAssistantPanel").isVisible().catch(() => false);
    if (!panelVisible0) {
      await page.locator("#marllenAssistantFab").click();
      await page.waitForSelector("#marllenAssistantPanel:not(.hidden)");
    }
    await page.waitForTimeout(250);
    await screenshot(page, "02-open.png");

    // If there is still an in-flight job, the input is disabled. Wait briefly before proceeding.
    await page.waitForFunction(() => {
      const el = document.getElementById("marllenAssistantInput");
      return !!(el && !el.disabled);
    }, { timeout: 60000 }).catch(() => {});

    // Send a question that should take some time so we can refresh mid-flight.
    const q = "给我一份数据快照（概览），并解释一下最近7天的变化趋势。";
    await page.locator("#marllenAssistantInput").fill(q);
    await page.locator("#marllenAssistantSend").click();

    // Prefer testing refresh-while-pending. If the answer comes back very fast, skip that path.
    const st0 = await waitUntilPendingOrFinished(page, 10000);
    await screenshot(page, st0.pending ? "03-pending.png" : "03-after-send.png");

    if (st0.pending) {
      // Close panel so localStorage open=0, then reload and expect auto-open while job is still pending.
      await page.locator("#marllenAssistantMin").click();
      await page.waitForSelector("#marllenAssistantPanel.hidden", { state: "attached" });
      await page.waitForSelector("#marllenAssistantFab:not(.hidden)", { state: "visible" });
      await screenshot(page, "04-collapsed.png");

      await page.reload({ waitUntil: "domcontentloaded" });
      await page.waitForSelector("#marllenAssistantPanel:not(.hidden)", { timeout: 15000 });
      await screenshot(page, "05-auto-open-after-reload.png");
    }

    // Best-effort: wait for completion or an error.
    const doneOrError = page.waitForFunction(() => {
      const err = document.getElementById("marllenAssistantError");
      if (err && !err.classList.contains("hidden") && (err.textContent || "").trim()) return true;
      const dots = document.querySelector(".marllen-typing-dots");
      return !dots;
    }, { timeout: 180000 });
    await doneOrError.catch(() => {});
    await page.waitForTimeout(250);
    await screenshot(page, "06-final.png");

    // Assert: no error banner, and at least one assistant reply exists.
    const finalState = await page.evaluate(async () => {
      const resp = await fetch("/api/marllen-assistant/thread?limit=20", { credentials: "same-origin" });
      const json = await resp.json().catch(() => ({}));
      return { ok: resp.ok, json };
    });
    const errText = await page.locator("#marllenAssistantError").textContent().catch(() => "");
    if (errText && String(errText).trim()) {
      throw new Error("assistant error banner: " + String(errText).trim().slice(0, 300));
    }
    const msgs = finalState && finalState.json && Array.isArray(finalState.json.messages) ? finalState.json.messages : [];
    const hasAssistant = msgs.some((m) => m && String(m.role || "") === "assistant" && String(m.content || "").trim());
    if (!hasAssistant) {
      throw new Error("assistant produced no assistant message");
    }

    console.log("OK: assistant refresh-wait smoke finished");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
