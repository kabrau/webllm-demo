/**
 * Métricas visíveis na demo: RAM (estimativas do browser), GPU/VRAM (limites) e tok/s.
 */

export function formatMB(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return "—";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatTps(n) {
  if (n == null || Number.isNaN(n) || n <= 0) return "—";
  return `${n.toFixed(1)} tok/s`;
}

/** Estimativa grosseira (~4 chars/token em PT). */
export function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * RAM: heap JS (Chrome) e, se disponível, memória do agente da aba.
 * @returns {Promise<{ heapUsed: number|null, heapTotal: number|null, tabBytes: number|null, deviceRamGb: number|null }>}
 */
export async function sampleRam() {
  const out = {
    heapUsed: null,
    heapTotal: null,
    tabBytes: null,
    deviceRamGb: navigator.deviceMemory ?? null,
  };

  if (performance.memory) {
    out.heapUsed = performance.memory.usedJSHeapSize;
    out.heapTotal = performance.memory.totalJSHeapSize;
  }

  if (typeof performance.measureUserAgentSpecificMemory === "function") {
    try {
      const r = await performance.measureUserAgentSpecificMemory();
      out.tabBytes = r.bytes;
    } catch {
      /* requer contexto seguro / sem iframes cross-origin */
    }
  }

  return out;
}

/**
 * @param {import('@mlc-ai/web-llm').MLCEngineInterface | null} engine
 */
export async function sampleGpu(engine) {
  if (!engine) {
    return { vendor: null, maxBufferBytes: null };
  }
  try {
    const [vendor, maxBufferBytes] = await Promise.all([
      engine.getGPUVendor?.() ?? Promise.resolve(null),
      engine.getMaxStorageBufferBindingSize?.() ?? Promise.resolve(null),
    ]);
    return { vendor, maxBufferBytes };
  } catch {
    return { vendor: null, maxBufferBytes: null };
  }
}

/** @returns {{ value: string, sub: string }} */
export function formatRamDisplay(ram) {
  const pc = ram.deviceRamGb != null ? `PC ~${ram.deviceRamGb} GB` : null;
  if (ram.tabBytes != null) {
    return {
      value: formatMB(ram.tabBytes),
      sub: pc ? `memória da aba · ${pc}` : "memória da aba (Chrome)",
    };
  }
  if (ram.heapUsed != null) {
    const sub = ram.heapTotal
      ? `heap JS · total ${formatMB(ram.heapTotal)}`
      : "heap JavaScript";
    return { value: formatMB(ram.heapUsed), sub: pc ? `${sub} · ${pc}` : sub };
  }
  return { value: "—", sub: "métrica indisponível neste browser" };
}

/** @returns {{ value: string, sub: string }} */
export function formatVramDisplay(gpu) {
  if (!gpu.vendor && gpu.maxBufferBytes == null) {
    return { value: "—", sub: "carregue o modelo" };
  }
  const value = gpu.vendor || "WebGPU";
  const parts = ["uso VRAM não exposto"];
  if (gpu.maxBufferBytes != null) parts.push(`buf ${formatMB(gpu.maxBufferBytes)}`);
  return { value, sub: parts.join(" · ") };
}

/**
 * Extrai contagem de tokens e tok/s do usage do WebLLM (com fallbacks).
 * @param {object|null} usage
 * @param {string} replyText
 * @param {number} durationMs
 */
export function usageToTokenStats(usage, replyText, durationMs) {
  const secs = Math.max(durationMs / 1000, 0.001);
  const estimatedCompletion = estimateTokens(replyText);

  if (!usage) {
    return {
      promptTokens: null,
      completionTokens: estimatedCompletion,
      totalTokens: estimatedCompletion,
      decodeTps: estimatedCompletion / secs,
      prefillTps: 0,
      source: "estimate",
    };
  }

  const extra = usage.extra || {};
  const promptTokens = usage.prompt_tokens ?? null;
  const completionTokens = usage.completion_tokens ?? estimatedCompletion;
  const totalTokens =
    usage.total_tokens ?? (promptTokens ?? 0) + completionTokens;

  let decodeTps = extra.decode_tokens_per_s;
  if (!decodeTps && completionTokens && extra.e2e_latency_s > 0) {
    decodeTps = completionTokens / extra.e2e_latency_s;
  }
  if (!decodeTps && completionTokens) {
    decodeTps = completionTokens / secs;
  }

  const prefillTps = extra.prefill_tokens_per_s || 0;

  return {
    promptTokens,
    completionTokens,
    totalTokens,
    decodeTps: decodeTps || 0,
    prefillTps,
    source: usage.completion_tokens != null ? "webllm" : "estimate",
  };
}

/** @returns {{ value: string, sub: string }} */
export function formatTpsDisplay({ instant, decode, prefill, tokenStats }) {
  const tokHint =
    tokenStats?.completionTokens != null
      ? `${tokenStats.completionTokens} tok saída` +
        (tokenStats.promptTokens != null ? ` · ${tokenStats.promptTokens} prompt` : "")
      : null;

  if (decode > 0) {
    const speedParts = [];
    if (instant > 0) speedParts.push(`pico ${instant.toFixed(1)}`);
    if (prefill > 0) speedParts.push(`prefill ${prefill.toFixed(1)}`);
    let sub = tokHint || "";
    if (speedParts.length) {
      const speeds = `${speedParts.join(" · ")} tok/s`;
      sub = sub ? `${sub} · ${speeds}` : speeds;
    } else if (!sub) {
      sub = "decode (WebLLM)";
    }
    return { value: decode.toFixed(1), sub };
  }
  if (instant > 0) {
    return {
      value: instant.toFixed(1),
      sub: tokHint ? `${tokHint} · estimativa` : "agora (estimativa)",
    };
  }
  if (tokenStats?.source === "kb-deterministic") {
    return { value: "—", sub: "resposta da KB (sem inferência LLM)" };
  }
  if (tokHint) {
    return { value: "—", sub: `${tokHint} · sem tok/s` };
  }
  return { value: "—", sub: "durante a geração" };
}
