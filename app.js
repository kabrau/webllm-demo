import { CreateWebWorkerMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
import {
  retrieveLinks,
  buildSystemPrompt,
  buildRetrievalQuery,
} from "./retrieve.js";
import {
  sampleRam,
  sampleGpu,
  formatRamDisplay,
  formatVramDisplay,
  formatTpsDisplay,
  estimateTokens,
  usageToTokenStats,
} from "./metrics.js";
import { getModelCatalog, getDefaultModelId, describeModel } from "./models.js";

let currentModelId = getDefaultModelId();
let modelCatalog = [];

const $ = (id) => document.getElementById(id);

const els = {
  modelSelect: $("model-select"),
  modelDesc: $("model-desc"),
  filterQ4f32Only: $("filter-q4f32-only"),
  statusPill: $("status-pill"),
  toggleInspector: $("toggle-inspector"),
  loadingOverlay: $("loading-overlay"),
  loadingModelName: $("loading-model-name"),
  progressBar: $("progress-bar"),
  progressText: $("progress-text"),
  loadingWait: $("loading-wait"),
  loadingLog: $("loading-log"),
  gpuRecoveryPanel: $("gpu-recovery-panel"),
  gpuRecoveryText: $("gpu-recovery-text"),
  btnReloadPage: $("btn-reload-page"),
  webgpuStatus: $("webgpu-status"),
  mainLayout: $("main-layout"),
  inspectorPanel: $("inspector-panel"),
  messages: $("messages"),
  chatForm: $("chat-form"),
  userInput: $("user-input"),
  sendBtn: $("send-btn"),
  btnStop: $("btn-stop"),
  btnResetContext: $("btn-reset-context"),
  suggestions: $("suggestions"),
  turnSelect: $("turn-select"),
  inspectorContent: $("inspector-content"),
  btnCopy: $("btn-copy"),
  metricRam: $("metric-ram"),
  metricRamHint: $("metric-ram-hint"),
  metricVram: $("metric-vram"),
  metricVramHint: $("metric-vram-hint"),
  metricTps: $("metric-tps"),
  metricTpsHint: $("metric-tps-hint"),
};

let metricsTimer = null;
let gpuInfo = { vendor: null, maxBufferBytes: null };
let liveTps = { instant: 0, decode: 0, prefill: 0, tokenStats: null };
let streamTpsState = null;

/** Histórico enviado ao WebLLM (evita estourar contexto após várias listas). */
const MAX_LLM_HISTORY_MESSAGES = 4;
const MAX_ASSISTANT_CHARS_IN_HISTORY = 600;

function formatEngineError(err) {
  if (err == null) return "erro desconhecido";
  if (typeof err === "string") return err;
  if (err.message) return err.message;
  if (err.name) return err.name;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

function buildMessagesForEngine(systemPrompt, history) {
  const recent = history.slice(-MAX_LLM_HISTORY_MESSAGES).map((m) => {
    if (m.role === "assistant" && m.content.length > MAX_ASSISTANT_CHARS_IN_HISTORY) {
      return {
        role: m.role,
        content: `${m.content.slice(0, MAX_ASSISTANT_CHARS_IN_HISTORY)}…`,
      };
    }
    return { role: m.role, content: m.content };
  });
  return [{ role: "system", content: systemPrompt }, ...recent];
}

function completionExtraBody(modelId) {
  if (/^Qwen3/i.test(modelId)) {
    return { enable_thinking: false };
  }
  return undefined;
}

function isGpuDeviceLostError(err) {
  const s = formatEngineError(err);
  return /DXGI_ERROR_DEVICE_REMOVED|DEVICE_REMOVED|887A0005|requestDevice.*GPUAdapter|Device was lost|d3d12|D3D12/i.test(
    s
  );
}

function isGpuBufferError(err) {
  if (isGpuDeviceLostError(err)) return true;
  const s = formatEngineError(err);
  return /GPUBuffer|mapAsync|unmapped|AbortError/i.test(s);
}

function gpuDeviceLostHelpText() {
  return `A GPU foi resetada pelo Windows (DXGI_ERROR_DEVICE_REMOVED).

O WebLLM não consegue iniciar até a placa voltar ao normal.

O que fazer (nessa ordem):
1. Feche jogos, Discord stream, outras abas com IA e vídeos 4K
2. Espere 10–30 segundos
3. Recarregue esta página (F5) — use o botão abaixo
4. Escolha Qwen2.5-0.5B ou 1.5B (evite Qwen3/Gemma logo após erro)
5. Se repetir: reinicie o PC ou atualize o driver da placa`;
}

function showGpuFatalOverlay(detail = "") {
  if (els.loadingOverlay) {
    els.loadingOverlay.hidden = false;
    els.loadingOverlay.classList.add("is-active", "gpu-fatal");
  }
  if (els.loadingModelName) {
    els.loadingModelName.textContent = "GPU indisponível no momento";
  }
  if (els.progressBar) els.progressBar.style.width = "0%";
  if (els.progressText) els.progressText.textContent = "DXGI_ERROR_DEVICE_REMOVED";
  if (els.loadingWait) els.loadingWait.hidden = true;
  if (els.gpuRecoveryText) {
    els.gpuRecoveryText.textContent = gpuDeviceLostHelpText();
  }
  if (els.gpuRecoveryPanel) els.gpuRecoveryPanel.hidden = false;
  if (els.webgpuStatus) {
    els.webgpuStatus.textContent =
      detail.slice(0, 200) || "GPU resetada — recarregue a página";
    els.webgpuStatus.className = "webgpu-status warn";
  }
  setChatEnabled(false);
}

async function probeWebGpuAdapter() {
  if (!navigator.gpu) {
    return { ok: false, reason: "WebGPU não disponível neste navegador." };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "Nenhum adaptador WebGPU encontrado." };
    const device = await adapter.requestDevice();
    device.destroy();
    return { ok: true };
  } catch (err) {
    return {
      ok: false,
      reason: formatEngineError(err),
      deviceLost: isGpuDeviceLostError(err),
    };
  }
}

async function markGpuDeviceLost(err) {
  gpuDeviceLost = true;
  modelReady = false;
  await interruptGenerationQuiet();
  if (engine?.unload) {
    try {
      await engine.unload();
    } catch {
      /* ignore */
    }
  }
  engine = null;
  const detail = formatEngineError(err);
  appendLog(`GPU indisponível: ${detail}`);
  showGpuFatalOverlay(detail);
}

function isModelNotLoadedError(err) {
  const s = formatEngineError(err);
  return /ModelNotLoadedError|Model not loaded/i.test(s);
}

/** Garante pesos carregados (resetChat/GPU às vezes descarrega o modelo). */
async function ensureModelReady() {
  if (gpuDeviceLost) {
    throw new Error(gpuDeviceLostHelpText());
  }
  if (!engine) {
    throw new Error("Motor WebLLM não inicializado. Recarregue a página (F5).");
  }
  if (modelReady) return;
  await initEngine(currentModelId);
  if (!modelReady) {
    throw new Error("Modelo não pôde ser recarregado.");
  }
}

async function interruptGenerationQuiet() {
  if (!engine || typeof engine.interruptGenerate !== "function") return;
  try {
    await engine.interruptGenerate();
  } catch {
    /* ignore */
  }
  await new Promise((r) => setTimeout(r, 150));
}

/** Após erro leve de GPU — não recarrega pesos (evita piorar DEVICE_REMOVED). */
async function recoverEngineAfterGpuError(err) {
  if (gpuDeviceLost || isGpuDeviceLostError(err)) {
    await markGpuDeviceLost(err);
    return;
  }
  if (!engine) return;
  await interruptGenerationQuiet();
  modelReady = false;
  await softResetEngineChat();
}

/** Limpa KV cache sem descarregar pesos (se falhar, marca para reload). */
async function softResetEngineChat() {
  if (!engine || typeof engine.resetChat !== "function") return;
  try {
    await engine.resetChat();
  } catch (err) {
    console.warn("resetChat:", err);
    modelReady = false;
  }
}

let generationInFlight = false;
let stopRequested = false;

function isInterruptedError(err) {
  if (stopRequested) return true;
  const s = formatEngineError(err);
  return /interrupt|aborted|cancel|stopped/i.test(s);
}

function setGeneratingUI(generating) {
  if (generating) {
    els.userInput.disabled = true;
    els.sendBtn.disabled = true;
    els.sendBtn.hidden = true;
    if (els.btnStop) {
      els.btnStop.hidden = false;
      els.btnStop.disabled = false;
    }
    if (els.modelSelect) els.modelSelect.disabled = true;
    els.suggestions.querySelectorAll(".chip").forEach((b) => {
      b.disabled = true;
    });
  } else {
    els.sendBtn.hidden = false;
    if (els.btnStop) {
      els.btnStop.hidden = true;
      els.btnStop.disabled = true;
    }
  }
}

async function stopActiveGeneration() {
  if (!generationInFlight || !engine) return;
  stopRequested = true;
  if (els.btnStop) els.btnStop.disabled = true;
  await interruptGenerationQuiet();
}

/** Detecta loop de repetição no stream (modelo pequeno). */
function detectStreamLoop(text) {
  const trimmed = text.trim();
  if (trimmed.length < 100) return false;

  const lines = trimmed
    .split(/\n+/)
    .map((l) => l.trim())
    .filter((l) => l.length > 25);
  if (lines.length >= 2) {
    const freq = new Map();
    for (const line of lines) {
      const key = line.toLowerCase().replace(/\s+/g, " ");
      freq.set(key, (freq.get(key) + 1) || 1);
      if (freq.get(key) >= 2) return true;
    }
  }

  const lower = trimmed.toLowerCase();
  const chunk = 50;
  if (lower.length > chunk * 3) {
    const tail = lower.slice(-chunk * 2);
    const a = tail.slice(0, chunk);
    const b = tail.slice(chunk, chunk * 2);
    if (a.length >= 30 && a === b) return true;
  }

  const sentences = trimmed.split(/(?<=[.!?])\s+/).filter((s) => s.length > 30);
  if (sentences.length >= 3) {
    const last = sentences[sentences.length - 1].toLowerCase();
    let repeats = 0;
    for (const s of sentences) {
      if (s.toLowerCase() === last) repeats++;
    }
    if (repeats >= 2) return true;
  }

  return false;
}

function trimReplyAtLoop(text) {
  const lines = text.split(/\n+/);
  const seen = new Set();
  const kept = [];
  for (const line of lines) {
    const key = line.trim().toLowerCase();
    if (key.length > 20 && seen.has(key)) break;
    if (key.length > 20) seen.add(key);
    kept.push(line);
  }
  return kept.join("\n").trim() || text.slice(0, 500).trim();
}

function maxTokensForMode(retrievalMode) {
  if (retrievalMode === "greeting") return 160;
  if (retrievalMode === "catalog-help") return 280;
  if (retrievalMode === "utility-text") return 1024;
  if (retrievalMode?.startsWith("category-list")) return 384;
  return 512;
}

function finalizeAssistantTurn(assistantEl, turn, t0, reply, { stopped = false } = {}) {
  let text = stripThinkingFromReply(reply);
  if (turn.meta?.loopDetected) {
    text = trimReplyAtLoop(text);
    if (!text.includes("repetição")) {
      text += "\n\n— Resposta cortada (repetição detectada).";
    }
  }
  if (stopped) {
    if (!text.trim()) text = "[Geração interrompida]";
    else if (!text.includes("interrompida") && !text.includes("repetição")) {
      text += "\n\n— Parado pelo usuário.";
    }
    turn.meta.stopped = true;
  }
  turn.response = text;
  updateStreamingMessage(assistantEl, text);
  delete assistantEl.dataset.streaming;
  if (text && !stopped) {
    chatHistory.push({ role: "assistant", content: text });
  } else if (text && stopped) {
    chatHistory.push({ role: "assistant", content: text });
  }
  turn.meta.durationMs = Math.round(performance.now() - t0);
  turn.meta.finishedAt = new Date().toISOString();
  streamTpsState = null;
  refreshTurnSelect();
}

/** @type {import('@mlc-ai/web-llm').MLCEngineInterface | null} */
let engine = null;
let modelReady = false;
let gpuDeviceLost = false;
let engineLoading = false;
/** @type {Array<{role: string, content: string}>} */
let chatHistory = [];
/** @type {Array<object>} */
let kbLinks = [];
/** @type {Array<object>} */
let turnLog = [];
let activeTab = "rag";
let selectedTurnIndex = -1;

function setState(state) {
  els.statusPill.dataset.state = state;
  const labels = {
    idle: "Aguardando",
    loading: "Carregando modelo…",
    ready: "Pronto",
    generating: "Gerando…",
    error: "Erro",
  };
  els.statusPill.textContent = labels[state] || state;
  updateResetContextButton();
}

function updateResetContextButton() {
  if (!els.btnResetContext) return;
  const state = els.statusPill?.dataset.state;
  els.btnResetContext.disabled = !engine || state === "loading";
}

function appendLog(line) {
  const ts = new Date().toLocaleTimeString("pt-BR");
  els.loadingLog.textContent += `[${ts}] ${line}\n`;
  els.loadingLog.scrollTop = els.loadingLog.scrollHeight;
}

function checkWebGPU() {
  if (navigator.gpu) {
    els.webgpuStatus.textContent =
      "WebGPU: disponível. Modelo q4f32 (compatível sem extensão shader-f16).";
    els.webgpuStatus.className = "webgpu-status ok";
    return true;
  }
  els.webgpuStatus.textContent =
    "WebGPU: não detectado. Use Chrome ou Edge recente com WebGPU habilitado.";
  els.webgpuStatus.className = "webgpu-status warn";
  return false;
}

function parseProgressFromReport(report) {
  let p = report?.progress ?? 0;
  const text = report?.text || "";
  const pctMatch = text.match(/(\d+)%\s*completed/i);
  if (pctMatch) {
    p = Math.max(p, parseInt(pctMatch[1], 10) / 100);
  }
  const chunkMatch = text.match(/\[(\d+)\/(\d+)\]/);
  if (chunkMatch) {
    const cur = parseInt(chunkMatch[1], 10);
    const total = parseInt(chunkMatch[2], 10);
    if (total > 0) p = Math.max(p, cur / total);
  }
  return Math.min(1, Math.max(0, p));
}

function setProgress(progress, text, report) {
  const p =
    typeof report === "object" && report !== null
      ? parseProgressFromReport(report)
      : Math.min(1, Math.max(0, progress || 0));

  const pct = Math.round(p * 100);
  els.progressBar.style.width = `${pct}%`;

  let label = `${pct}%`;
  const chunkMatch = text?.match(/\[(\d+)\/(\d+)\]/);
  if (chunkMatch) {
    label = `${pct}% — download ${chunkMatch[1]}/${chunkMatch[2]}`;
  } else if (pct === 0 && text) {
    label = "Aguarde…";
  }
  els.progressText.textContent = label;

  if (els.loadingWait) {
    els.loadingWait.hidden = pct >= 100;
  }

  if (text) appendLog(text);
}

function addMessage(role, content, streaming = false) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<span class="role">${role === "user" ? "Você" : "Assistente"}</span><span class="body">${escapeHtml(content)}</span>`;
  if (streaming) div.dataset.streaming = "1";
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Remove raciocínio interno (Qwen/reasoning) antes de exibir no chat. */
function stripThinkingFromReply(text) {
  if (!text) return "";
  let out = text;
  out = out.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, "");
  out = out.replace(/<think>[\s\S]*?<\/redacted_thinking>/gi, "");
  out = out.replace(/^<think[^>]*>[\s\S]*/i, "");
  out = out.replace(/^<think>[\s\S]*/i, "");
  return out.trim();
}

function updateStreamingMessage(el, text) {
  const body = el.querySelector(".body");
  if (body) body.textContent = stripThinkingFromReply(text);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setMetric(elValue, elSub, { value, sub }) {
  elValue.textContent = value;
  elSub.textContent = sub;
}

function updateMetricsDisplay() {
  setMetric(els.metricTps, els.metricTpsHint, formatTpsDisplay(liveTps));
}

function applyTokenStatsToMetrics(stats) {
  if (!stats) return;
  liveTps.tokenStats = stats;
  if (stats.decodeTps > 0) liveTps.decode = stats.decodeTps;
  if (stats.prefillTps > 0) liveTps.prefill = stats.prefillTps;
  updateMetricsDisplay();
}

async function refreshMemoryMetrics() {
  const ram = await sampleRam();
  setMetric(els.metricRam, els.metricRamHint, formatRamDisplay(ram));
  setMetric(els.metricVram, els.metricVramHint, formatVramDisplay(gpuInfo));
}

function startMetricsPolling() {
  if (metricsTimer) return;
  refreshMemoryMetrics();
  metricsTimer = setInterval(refreshMemoryMetrics, 2000);
}

function applyUsageMetrics(usage, replyText, durationMs) {
  const stats = usageToTokenStats(usage, replyText, durationMs);
  applyTokenStatsToMetrics(stats);
  return stats;
}

function beginStreamTpsTracking() {
  streamTpsState = {
    lastTokens: 0,
    lastAt: performance.now(),
  };
  liveTps.instant = 0;
  updateMetricsDisplay();
}

function tickStreamTps(reply) {
  if (!streamTpsState) return;
  const tokens = estimateTokens(reply);
  const now = performance.now();
  const dt = (now - streamTpsState.lastAt) / 1000;
  if (dt >= 0.15) {
    const delta = tokens - streamTpsState.lastTokens;
    if (delta > 0) {
      liveTps.instant = delta / dt;
      updateMetricsDisplay();
    }
    streamTpsState.lastTokens = tokens;
    streamTpsState.lastAt = now;
  }
}

function setChatEnabled(enabled) {
  if (generationInFlight) return;
  els.userInput.disabled = !enabled;
  els.sendBtn.disabled = !enabled;
  els.sendBtn.hidden = false;
  if (els.btnStop) {
    els.btnStop.hidden = true;
    els.btnStop.disabled = true;
  }
  if (els.modelSelect) els.modelSelect.disabled = !enabled;
  els.suggestions.querySelectorAll(".chip").forEach((b) => {
    b.disabled = !enabled;
  });
}

async function resetChatContext() {
  if (!engine) return;

  if (generationInFlight) {
    stopRequested = true;
    await interruptGenerationQuiet();
  }
  generationInFlight = false;
  stopRequested = false;

  await softResetEngineChat();
  if (!gpuDeviceLost && engine && !modelReady) {
    try {
      await ensureModelReady();
    } catch (err) {
      if (isGpuDeviceLostError(err)) await markGpuDeviceLost(err);
    }
  }

  chatHistory.length = 0;
  turnLog.length = 0;
  selectedTurnIndex = -1;
  els.messages.innerHTML = "";
  streamTpsState = null;
  liveTps = { instant: 0, decode: 0, prefill: 0, tokenStats: null };
  updateMetricsDisplay();
  refreshTurnSelect();
  updateInspectorForTurn(null);

  const ready = els.statusPill?.dataset.state === "ready";
  setState(ready ? "ready" : "idle");
  setGeneratingUI(false);
  setChatEnabled(ready);
  els.userInput?.focus();
}

function updateInspectorForTurn(turn) {
  if (!turn) {
    els.inspectorContent.textContent = "Envie uma pergunta para inspecionar o prompt.";
    els.btnCopy.disabled = true;
    return;
  }
  els.btnCopy.disabled = false;
  renderInspectorTab(turn);
}

function renderInspectorTab(turn) {
  switch (activeTab) {
    case "rag": {
      const lines = turn.ragResults.map(
        (r, i) =>
          `#${i + 1} score=${r.score} | ${r.link.title}\n   ${r.link.url}\n   [${r.link.category}]`
      );
      const mode = turn.meta?.retrievalMode
        ? `\nModo: ${turn.meta.retrievalMode}${turn.meta.category ? ` · ${turn.meta.category}` : ""}${turn.meta.filter ? ` · filtro ${turn.meta.filter}` : ""}${turn.meta.totalInCategory != null ? ` (${turn.meta.totalInCategory} itens)` : ""}`
        : "";
      const rq = turn.retrievalQuery
        ? `\nBusca expandida: ${turn.retrievalQuery}`
        : "";
      els.inspectorContent.textContent =
        `Query: ${turn.userQuery}${rq}\nTokens: ${turn.queryTokens.join(", ") || "(nenhum)"}${mode}\n\n` +
        lines.join("\n\n");
      break;
    }
    case "system":
      els.inspectorContent.textContent = turn.systemPrompt;
      break;
    case "messages":
      els.inspectorContent.textContent = JSON.stringify(turn.messages, null, 2);
      break;
    case "response":
      els.inspectorContent.textContent =
        turn.response +
        (turn.streamChunks?.length
          ? `\n\n--- chunks (${turn.streamChunks.length}) ---\n` +
            turn.streamChunks.map((c, i) => `[${i}] ${JSON.stringify(c)}`).join("\n")
          : "");
      break;
    case "meta":
      els.inspectorContent.textContent = JSON.stringify(turn.meta, null, 2);
      break;
    default:
      break;
  }
}

function refreshTurnSelect() {
  els.turnSelect.innerHTML = "";
  if (turnLog.length === 0) {
    els.turnSelect.disabled = true;
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "—";
    els.turnSelect.appendChild(opt);
    return;
  }
  els.turnSelect.disabled = false;
  turnLog.forEach((t, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    const preview = t.userQuery.slice(0, 40);
    opt.textContent = `Turno ${i + 1}: ${preview}${t.userQuery.length > 40 ? "…" : ""}`;
    els.turnSelect.appendChild(opt);
  });
  els.turnSelect.value = String(selectedTurnIndex >= 0 ? selectedTurnIndex : turnLog.length - 1);
  selectedTurnIndex = Number(els.turnSelect.value);
  updateInspectorForTurn(turnLog[selectedTurnIndex]);
}

async function loadKnowledgeBase() {
  const res = await fetch("./data/links-kb.json");
  if (!res.ok) throw new Error(`KB não carregada: ${res.status}`);
  const data = await res.json();
  kbLinks = data.links || [];
  appendLog(`Base de conhecimento: ${kbLinks.length} links.`);
}

function getFilteredCatalog() {
  if (els.filterQ4f32Only?.checked) {
    return modelCatalog.filter((m) => m.quant === "q4f32");
  }
  return modelCatalog;
}

function populateModelSelect(preserveId = currentModelId) {
  const filtered = getFilteredCatalog();
  const select = els.modelSelect;
  if (!select) return;

  const byFamily = new Map();
  for (const m of filtered) {
    if (!byFamily.has(m.family)) byFamily.set(m.family, []);
    byFamily.get(m.family).push(m);
  }

  select.innerHTML = "";
  for (const [family, models] of [...byFamily.entries()].sort((a, b) =>
    a[0].localeCompare(b[0], "pt")
  )) {
    const group = document.createElement("optgroup");
    group.label = family;
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.optionLabel;
      opt.title = m.description.replace(/\n/g, " · ");
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  if (filtered.some((m) => m.id === preserveId)) {
    select.value = preserveId;
  } else if (filtered.length > 0) {
    select.value = filtered[0].id;
    currentModelId = filtered[0].id;
  }

  updateModelDescription();
}

function updateModelDescription() {
  const id = els.modelSelect?.value || currentModelId;
  if (els.modelDesc) {
    els.modelDesc.textContent = describeModel(id);
  }
}

async function loadModelCatalog() {
  modelCatalog = getModelCatalog();
  populateModelSelect(currentModelId);
}

async function initEngine(modelId = currentModelId) {
  if (gpuDeviceLost) {
    setState("error");
    throw new Error(gpuDeviceLostHelpText());
  }
  if (engineLoading) return;
  engineLoading = true;
  currentModelId = modelId;
  setState("loading");
  els.loadingModelName.textContent = modelId;
  if (els.modelSelect) els.modelSelect.value = modelId;
  updateModelDescription();
  els.loadingOverlay.hidden = false;
  els.loadingOverlay.classList.add("is-active");
  els.loadingLog.textContent = "";
  setProgress(0, null);
  if (els.loadingWait) els.loadingWait.hidden = false;
  setChatEnabled(false);
  checkWebGPU();

  const initProgressCallback = (report) => {
    const t = report.text || report.type || "Carregando…";
    setProgress(null, t, report);
  };

  try {
    const probe = await probeWebGpuAdapter();
    if (!probe.ok) {
      appendLog(`Teste WebGPU: ${probe.reason}`);
      if (probe.deviceLost) {
        await markGpuDeviceLost(new Error(probe.reason));
        setState("error");
        engineLoading = false;
        return;
      }
      throw new Error(probe.reason);
    }

    appendLog("Iniciando Web Worker + WebLLM…");
    if (!engine) {
      engine = await CreateWebWorkerMLCEngine(
        new Worker(new URL("./worker.js", import.meta.url), { type: "module" }),
        modelId,
        { initProgressCallback }
      );
    } else {
      appendLog(`Recarregando modelo: ${modelId}…`);
      appendLog("Aguarde — baixando pesos do modelo (1ª vez pode demorar vários minutos).");
      if (typeof engine.setInitProgressCallback === "function") {
        engine.setInitProgressCallback(initProgressCallback);
      }
      await engine.reload(modelId);
    }
    appendLog("Modelo pronto.");
    modelReady = true;
    gpuInfo = await sampleGpu(engine);
    setMetric(els.metricVram, els.metricVramHint, formatVramDisplay(gpuInfo));
    setProgress(1, "Carregamento concluído.");
    if (els.gpuRecoveryPanel) els.gpuRecoveryPanel.hidden = true;
    els.loadingOverlay?.classList.remove("gpu-fatal");
    els.loadingOverlay.hidden = true;
    els.loadingOverlay.classList.remove("is-active");
    setState("ready");
    setChatEnabled(true);
    startMetricsPolling();
  } catch (err) {
    modelReady = false;
    console.error(err);
    const msg = formatEngineError(err);
    if (isGpuDeviceLostError(err)) {
      await markGpuDeviceLost(err);
      setState("error");
      addMessage("assistant", `Falha ao carregar o modelo:\n\n${gpuDeviceLostHelpText()}`);
      engineLoading = false;
      return;
    }
    appendLog(`ERRO: ${msg}`);
    setState("error");
    const gpuHint =
      msg.includes("f16") || msg.includes("shader-f16")
        ? "\n\nDica: seu GPU/browser não suporta f16. Este demo usa q4f32; limpe o cache do site (Application → Clear storage) e recarregue."
        : msg.includes("GPU") || msg.includes("WebGPU")
          ? "\n\nDica: use Chrome ou Edge atualizado. Headless/automação sem GPU não funciona."
          : "";
    addMessage(
      "assistant",
      `Falha ao carregar o modelo:\n${msg}${gpuHint}\n\nConfira também sua conexão (download do modelo na 1ª vez).`
    );
    els.loadingOverlay.hidden = false;
    els.progressText.textContent = "Falhou";
  } finally {
    engineLoading = false;
  }
}

async function runChatCompletion({ messages, assistantEl, turn, t0 }) {
  await ensureModelReady();
  stopRequested = false;
  const mode = turn.meta?.retrievalMode;
  const request = {
    messages,
    temperature:
      mode === "greeting" || mode?.startsWith("category-list") || mode === "utility-text"
        ? 0.15
        : 0.2,
    max_tokens: maxTokensForMode(mode),
    stream: true,
    stream_options: { include_usage: true },
  };
  const extraBody = completionExtraBody(currentModelId);
  if (extraBody) request.extra_body = extraBody;

  const chunks = await engine.chat.completions.create(request);

  let reply = "";
  let lastUsage = null;
  try {
    for await (const chunk of chunks) {
      if (stopRequested) break;
      turn.streamChunks.push(chunk);
      const delta = chunk.choices?.[0]?.delta?.content || "";
      reply += delta;
      turn.response = reply;
      updateStreamingMessage(assistantEl, reply);
      tickStreamTps(reply);
      if (detectStreamLoop(reply)) {
        turn.meta.loopDetected = true;
        reply = trimReplyAtLoop(reply);
        turn.response = reply;
        updateStreamingMessage(assistantEl, reply);
        await stopActiveGeneration();
        break;
      }
      if (selectedTurnIndex === turnLog.length - 1 && activeTab === "response") {
        renderInspectorTab(turn);
      }
      if (chunk.usage) lastUsage = chunk.usage;
    }
  } catch (err) {
    if (!isInterruptedError(err)) throw err;
  }

  if (!stopRequested) {
    try {
      const final = await engine.getMessage();
      if (final && final.length > reply.length) {
        reply = final;
      }
    } catch (err) {
      if (!isInterruptedError(err)) throw err;
    }
  }

  finalizeAssistantTurn(assistantEl, turn, t0, reply, { stopped: stopRequested });

  if (lastUsage) turn.meta.usage = lastUsage;
  const stats = applyUsageMetrics(lastUsage, turn.response, turn.meta.durationMs);
  turn.meta.tokenStats = stats;
  turn.meta.tokensPerSecond = {
    instant: liveTps.instant,
    decode: stats.decodeTps,
    prefill: stats.prefillTps,
  };
  refreshMemoryMetrics();
}

async function handleUserMessage(text) {
  const query = text.trim();
  if (!query || !engine) return;
  if (generationInFlight) return;

  generationInFlight = true;
  addMessage("user", query);
  chatHistory.push({ role: "user", content: query });
  els.userInput.value = "";
  setState("generating");
  setGeneratingUI(true);

  const priorMessages = chatHistory.slice(0, -1);
  const retrievalQuery = buildRetrievalQuery(query, priorMessages);
  let retrieval;
  try {
    retrieval = retrieveLinks(retrievalQuery, kbLinks, 12);
  } catch (err) {
    console.error("retrieveLinks:", err);
    generationInFlight = false;
    setState("ready");
    setGeneratingUI(false);
    setChatEnabled(true);
    addMessage(
      "assistant",
      `Erro na busca na curadoria: ${formatEngineError(err)}\n\nTente reformular (ex.: "liste crawlers").`
    );
    return;
  }
  const {
    results,
    queryTokens,
    retrievalMode,
    category,
    filter,
    totalInCategory,
    categories,
  } = retrieval;
  const ragResults = results.map((r) => ({ link: r.link, score: r.score }));
  const systemPrompt = buildSystemPrompt(results, {
    retrievalMode,
    category,
    filter,
    categories,
  });

  const messages = buildMessagesForEngine(systemPrompt, chatHistory);

  const turn = {
    userQuery: query,
    retrievalQuery: retrievalQuery !== query ? retrievalQuery : undefined,
    queryTokens,
    ragResults,
    systemPrompt,
    messages: structuredClone(messages),
    response: "",
    streamChunks: [],
    meta: {
      model: currentModelId,
      startedAt: new Date().toISOString(),
      retrievalMode,
      category: category || null,
      filter: filter || null,
      totalInCategory: totalInCategory ?? null,
    },
  };

  turnLog.push(turn);
  selectedTurnIndex = turnLog.length - 1;
  refreshTurnSelect();
  updateInspectorForTurn(turn);

  const assistantEl = addMessage("assistant", "…", true);
  const t0 = performance.now();

  beginStreamTpsTracking();

  try {
    try {
      await runChatCompletion({ messages, assistantEl, turn, t0 });
    } catch (firstErr) {
      if (isModelNotLoadedError(firstErr)) {
        console.warn("Modelo descarregado, recarregando…", firstErr);
        modelReady = false;
        await ensureModelReady();
        await runChatCompletion({ messages, assistantEl, turn, t0 });
      } else if (isGpuBufferError(firstErr) && !isGpuDeviceLostError(firstErr)) {
        console.warn("GPU buffer error, soft recovery and retry once…", firstErr);
        await recoverEngineAfterGpuError(firstErr);
        if (!gpuDeviceLost) {
          await runChatCompletion({ messages, assistantEl, turn, t0 });
        }
      } else {
        throw firstErr;
      }
    }
  } catch (err) {
    if (isInterruptedError(err) || stopRequested) {
      finalizeAssistantTurn(assistantEl, turn, t0, turn.response || "", {
        stopped: true,
      });
    } else {
      console.error(err);
      const detail = formatEngineError(err);
      let errText = `Erro na geração: ${detail}`;
      if (isModelNotLoadedError(err)) {
        errText += "\n\nO modelo foi descarregado. Tente enviar de novo ou recarregue a página (F5).";
        modelReady = false;
      } else if (isGpuDeviceLostError(err)) {
        await markGpuDeviceLost(err);
        errText = gpuDeviceLostHelpText();
      } else if (isGpuBufferError(err)) {
        errText += "\n\nDica: recarregue a página (F5) se o WebGPU travar após muitas perguntas.";
        await recoverEngineAfterGpuError(err);
      }
      turn.response = errText;
      turn.meta.error = detail;
      updateStreamingMessage(assistantEl, errText);
      delete assistantEl.dataset.streaming;
    }
  } finally {
    generationInFlight = false;
    stopRequested = false;
    setState("ready");
    setGeneratingUI(false);
    setChatEnabled(true);
    els.userInput.focus();
  }
}

async function onModelChange() {
  const nextId = els.modelSelect?.value;
  if (!nextId || nextId === currentModelId) return;
  if (els.modelSelect.disabled) return;
  if (gpuDeviceLost) {
    addMessage("assistant", gpuDeviceLostHelpText());
    return;
  }

  els.modelSelect.disabled = true;
  chatHistory.length = 0;
  els.messages.innerHTML = "";
  try {
    await initEngine(nextId);
  } finally {
    els.modelSelect.disabled = false;
  }
}

function setupUI() {
  setMetric(els.metricRam, els.metricRamHint, { value: "…", sub: "aguardando" });
  setMetric(els.metricVram, els.metricVramHint, { value: "…", sub: "aguardando modelo" });
  setMetric(els.metricTps, els.metricTpsHint, { value: "—", sub: "durante a geração" });
  els.toggleInspector.addEventListener("change", () => {
    els.mainLayout.classList.toggle("inspector-hidden", !els.toggleInspector.checked);
  });

  document.querySelectorAll(".inspector-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".inspector-tabs .tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      const turn = turnLog[selectedTurnIndex];
      if (turn) renderInspectorTab(turn);
    });
  });

  els.turnSelect.addEventListener("change", () => {
    selectedTurnIndex = Number(els.turnSelect.value);
    updateInspectorForTurn(turnLog[selectedTurnIndex]);
  });

  els.btnCopy.addEventListener("click", async () => {
    const text = els.inspectorContent.textContent;
    try {
      await navigator.clipboard.writeText(text);
      els.btnCopy.textContent = "Copiado!";
      setTimeout(() => {
        els.btnCopy.textContent = "Copiar aba";
      }, 1500);
    } catch {
      els.btnCopy.textContent = "Falha ao copiar";
    }
  });

  els.btnResetContext?.addEventListener("click", () => {
    resetChatContext();
  });

  els.btnReloadPage?.addEventListener("click", () => {
    location.reload();
  });

  els.btnStop?.addEventListener("click", () => {
    stopActiveGeneration();
  });

  els.suggestions.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.dataset.prompt;
      if (prompt && engine) {
        els.userInput.value = prompt;
        handleUserMessage(prompt);
      }
    });
  });

  els.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleUserMessage(els.userInput.value);
  });

  els.modelSelect?.addEventListener("change", () => {
    updateModelDescription();
    onModelChange();
  });

  els.filterQ4f32Only?.addEventListener("change", () => {
    populateModelSelect(els.modelSelect?.value || currentModelId);
  });
}

async function main() {
  setupUI();
  setState("idle");
  try {
    await loadModelCatalog();
    await loadKnowledgeBase();
    await initEngine(currentModelId);
  } catch (err) {
    console.error(err);
    setState("error");
    addMessage("assistant", `Erro na inicialização: ${err.message}`);
  }
}

main();
