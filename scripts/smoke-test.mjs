/**
 * Smoke test: página carrega, KB fetch, UI e início do WebLLM (timeout curto).
 * Uso: node scripts/smoke-test.mjs
 */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "http://localhost:8080";
const MODEL_TIMEOUT_MS = 120_000;

const results = {
  baseUrl: BASE,
  indexOk: false,
  kbOk: false,
  webgpuMessage: "",
  loadingOverlaySeen: false,
  modelReady: false,
  chatEnabled: false,
  inspectorVisible: false,
  errors: [],
};

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  page.on("pageerror", (err) => results.errors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") results.errors.push(`console: ${msg.text()}`);
  });

  try {
    await page.goto(BASE, { waitUntil: "networkidle", timeout: 30000 });
    results.indexOk = true;

    const kbRes = await page.request.get(`${BASE}/data/links-kb.json`);
    results.kbOk = kbRes.ok() && (await kbRes.json()).links?.length === 46;

    results.inspectorVisible = await page.locator("#inspector-panel").isVisible();

    const webgpu = await page.locator("#webgpu-status").textContent({ timeout: 5000 }).catch(() => "");
    results.webgpuMessage = webgpu?.trim() || "";

    const overlay = page.locator("#loading-overlay");
    results.loadingOverlaySeen = !(await overlay.getAttribute("hidden"));

    await page.waitForFunction(
      () => {
        const s = document.getElementById("status-pill")?.dataset.state;
        return s === "ready" || s === "error";
      },
      { timeout: MODEL_TIMEOUT_MS }
    );
    const state = await page.locator("#status-pill").getAttribute("data-state");
    results.finalState = state;
    results.modelReady = state === "ready";
    if (state === "error") {
      throw new Error("Modelo não carregou (estado error — provável falta de WebGPU/GPU)");
    }

    const inputDisabled = await page.locator("#user-input").isDisabled();
    results.chatEnabled = !inputDisabled;

    if (results.chatEnabled) {
      await page.locator('.chip[data-prompt*="LLM local"]').click();
      await page.waitForFunction(
        () => document.querySelectorAll(".msg.assistant").length >= 1,
        { timeout: 60000 }
      );
      await page.waitForFunction(
        () => {
          const pill = document.getElementById("status-pill");
          return pill?.dataset.state === "ready";
        },
        { timeout: 120000 }
      );
      const assistantText = await page.locator(".msg.assistant .body").last().textContent();
      results.sampleReplyLength = (assistantText || "").length;
      results.sampleReplyPreview = (assistantText || "").slice(0, 120);
    }
  } catch (err) {
    results.errors.push(err.message);
    const state = await page
      .locator("#status-pill")
      .getAttribute("data-state")
      .catch(() => "unknown");
    results.finalState = state;
  } finally {
    await browser.close();
  }

  console.log(JSON.stringify(results, null, 2));
  const ok =
    results.indexOk &&
    results.kbOk &&
    results.modelReady &&
    results.chatEnabled &&
    results.errors.length === 0;
  process.exit(ok ? 0 : 1);
}

main();
