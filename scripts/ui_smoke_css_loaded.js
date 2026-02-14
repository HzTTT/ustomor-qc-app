/* eslint-disable no-console */

const { chromium } = require("playwright");

const BASE_URL = process.env.QC_BASE_URL || "http://localhost:8000";

async function run() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ baseURL: BASE_URL });
  const page = await context.newPage();

  try {
    const cssRespP = page
      .waitForResponse((r) => r.url().includes("/static/tailwind.css"), { timeout: 15000 })
      .catch(() => null);

    await page.goto(`${BASE_URL}/login`, { waitUntil: "domcontentloaded" });

    const cssResp = await cssRespP;
    if (!cssResp || !cssResp.ok()) {
      throw new Error("tailwind.css not loaded (missing or non-200). This usually means static path/proxy/CDN issue.");
    }

    const state = await page.evaluate(() => {
      const link = document.querySelector('link[rel="stylesheet"][href*="tailwind.css"]');
      const bg = getComputedStyle(document.body).backgroundColor;
      return { hasLink: !!link, bg };
    });

    // base.html sets: <body class="bg-slate-50 ...">, so background should be slate-50.
    if (!state.hasLink) {
      throw new Error("tailwind link tag missing in HTML.");
    }
    if (state.bg !== "rgb(248, 250, 252)") {
      throw new Error(`tailwind styles not applied (body bg=${state.bg}). Page likely rendered without Tailwind CSS.`);
    }

    console.log("OK: Tailwind CSS loaded + applied");
  } finally {
    await context.close();
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

