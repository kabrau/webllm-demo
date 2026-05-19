import { CreateWebWorkerMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
import {
  retrieveLinks,
  buildSystemPrompt,
  buildRetrievalQuery,
  formatDeterministicList,
  shouldUseDeterministicList,
} from "./retrieve.js";
import {
  sampleRam,
  sampleGpu,
  formatRamDisplay,
  formatVramDisplay,
  formatTpsDisplay,
  estimateTokens,
} from "./metrics.js";

// Qwen2.5 0.5B q4f32: leve e sem extensão shader-f16 (Qwen2-0.5B não tem q4f32 no catálogo)
const MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f32_1-MLC";

const $ = (id) => document.getElementById(id);

const els = {
  modelBadge: $("model-badge"),
  statusPill: $("status-pill"),
  toggleInspector: $("toggle-inspector"),
  loadingOverlay: $("loading-overlay"),
  loadingModelName: $("loading-model-name"),
  progressBar: $("progress-bar"),
  progressText: $("progress-text"),
  loadingLog: $("loading-log"),
  webgpuStatus: $("webgpu-status"),
  mainLayout: $("main-layout"),
  inspectorPanel: $("inspector-panel"),
  messages: $("messages"),
  chatForm: $("chat-form"),
  userInput: $("user-input"),
  sendBtn: $("send-btn"),
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
let liveTps = { instant: 0, decode: 0, prefill: 0 };
let streamTpsState = null;

/** @type {import('@mlc-ai/web-llm').MLCEngineInterface | null} */
let engine = null;
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

function setProgress(progress, text) {
  const pct = Math.round((progress || 0) * 100);
  els.progressBar.style.width = `${pct}%`;
  els.progressText.textContent = `${pct}%`;
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

function updateStreamingMessage(el, text) {
  const body = el.querySelector(".body");
  if (body) body.textContent = text;
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setMetric(elValue, elSub, { value, sub }) {
  elValue.textContent = value;
  elSub.textContent = sub;
}

function updateMetricsDisplay() {
  setMetric(els.metricTps, els.metricTpsHint, formatTpsDisplay(liveTps));
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

function applyUsageMetrics(usage) {
  if (!usage) return;
  const extra = usage.extra || {};
  if (extra.decode_tokens_per_s) liveTps.decode = extra.decode_tokens_per_s;
  if (extra.prefill_tokens_per_s) liveTps.prefill = extra.prefill_tokens_per_s;
  if (usage.completion_tokens && usage.extra?.e2e_latency_s) {
    const avg = usage.completion_tokens / usage.extra.e2e_latency_s;
    if (!liveTps.decode) liveTps.decode = avg;
  }
  updateMetricsDisplay();
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
  els.userInput.disabled = !enabled;
  els.sendBtn.disabled = !enabled;
  els.suggestions.querySelectorAll(".chip").forEach((b) => {
    b.disabled = !enabled;
  });
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
      const det = turn.meta?.deterministic ? "\nResposta: gerada da KB (lista exata)" : "";
      els.inspectorContent.textContent =
        `Query: ${turn.userQuery}${rq}\nTokens: ${turn.queryTokens.join(", ") || "(nenhum)"}${mode}${det}\n\n` +
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

async function initEngine() {
  setState("loading");
  els.modelBadge.textContent = `Modelo: ${MODEL_ID}`;
  els.loadingModelName.textContent = MODEL_ID;
  els.loadingOverlay.hidden = false;
  els.loadingLog.textContent = "";
  setChatEnabled(false);
  checkWebGPU();

  const initProgressCallback = (report) => {
    const p = report.progress ?? 0;
    const t = report.text || report.type || "Carregando…";
    setProgress(p, t);
  };

  try {
    appendLog("Iniciando Web Worker + WebLLM…");
    engine = await CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.js", import.meta.url), { type: "module" }),
      MODEL_ID,
      { initProgressCallback }
    );
    appendLog("Modelo pronto.");
    gpuInfo = await sampleGpu(engine);
    setMetric(els.metricVram, els.metricVramHint, formatVramDisplay(gpuInfo));
    setProgress(1, "Finish loading.");
    els.loadingOverlay.hidden = true;
    setState("ready");
    setChatEnabled(true);
    startMetricsPolling();
  } catch (err) {
    console.error(err);
    const msg = err.message || String(err);
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
  }
}

async function handleUserMessage(text) {
  const query = text.trim();
  if (!query || !engine) return;

  addMessage("user", query);
  chatHistory.push({ role: "user", content: query });
  els.userInput.value = "";
  setState("generating");
  setChatEnabled(false);

  const priorMessages = chatHistory.slice(0, -1);
  const retrievalQuery = buildRetrievalQuery(query, priorMessages);
  const retrieval = retrieveLinks(retrievalQuery, kbLinks, 12);
  const { results, queryTokens, retrievalMode, category, filter, totalInCategory } = retrieval;
  const ragResults = results.map((r) => ({ link: r.link, score: r.score }));
  const systemPrompt = buildSystemPrompt(results, {
    retrievalMode,
    category,
    filter,
  });

  const messages = [
    { role: "system", content: systemPrompt },
    ...chatHistory,
  ];

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
      model: MODEL_ID,
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

  if (shouldUseDeterministicList(retrievalMode, results.length)) {
    const reply = formatDeterministicList(results, { category, filter });
    turn.response = reply;
    turn.meta.deterministic = true;
    turn.meta.durationMs = Math.round(performance.now() - t0);
    turn.meta.finishedAt = new Date().toISOString();
    updateStreamingMessage(assistantEl, reply);
    delete assistantEl.dataset.streaming;
    chatHistory.push({ role: "assistant", content: reply });
    refreshTurnSelect();
    setState("ready");
    setChatEnabled(true);
    els.userInput.focus();
    return;
  }

  beginStreamTpsTracking();

  try {
    const chunks = await engine.chat.completions.create({
      messages,
      temperature: 0.3,
      max_tokens: 768,
      stream: true,
      stream_options: { include_usage: true },
    });

    let reply = "";
    for await (const chunk of chunks) {
      turn.streamChunks.push(chunk);
      const delta = chunk.choices?.[0]?.delta?.content || "";
      reply += delta;
      turn.response = reply;
      updateStreamingMessage(assistantEl, reply);
      tickStreamTps(reply);
      if (selectedTurnIndex === turnLog.length - 1 && activeTab === "response") {
        renderInspectorTab(turn);
      }
      if (chunk.usage) {
        turn.meta.usage = chunk.usage;
        applyUsageMetrics(chunk.usage);
        turn.meta.tokensPerSecond = {
          instant: liveTps.instant,
          decode: liveTps.decode,
          prefill: liveTps.prefill,
        };
      }
    }

    const final = await engine.getMessage();
    if (final && final.length > reply.length) {
      reply = final;
      turn.response = reply;
    }

    updateStreamingMessage(assistantEl, reply);
    delete assistantEl.dataset.streaming;
    chatHistory.push({ role: "assistant", content: reply });

    turn.meta.durationMs = Math.round(performance.now() - t0);
    turn.meta.finishedAt = new Date().toISOString();
    if (!turn.meta.tokensPerSecond) {
      const secs = turn.meta.durationMs / 1000;
      turn.meta.tokensPerSecond = {
        decode: secs > 0 ? estimateTokens(reply) / secs : 0,
      };
      if (turn.meta.tokensPerSecond.decode) {
        liveTps.decode = turn.meta.tokensPerSecond.decode;
        updateMetricsDisplay();
      }
    }
    streamTpsState = null;
    refreshMemoryMetrics();
    refreshTurnSelect();
  } catch (err) {
    console.error(err);
    const errText = `Erro na geração: ${err.message}`;
    turn.response = errText;
    turn.meta.error = err.message;
    updateStreamingMessage(assistantEl, errText);
  }

  setState("ready");
  setChatEnabled(true);
  els.userInput.focus();
}

function setupUI() {
  els.modelBadge.textContent = `Modelo: ${MODEL_ID}`;
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
}

async function main() {
  setupUI();
  setState("idle");
  try {
    await loadKnowledgeBase();
    await initEngine();
  } catch (err) {
    console.error(err);
    setState("error");
    addMessage("assistant", `Erro na inicialização: ${err.message}`);
  }
}

main();
