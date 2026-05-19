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

/** @returns {{ value: string, sub: string }} */
export function formatTpsDisplay({ instant, decode, prefill }) {
  if (decode > 0) {
    const sub = [];
    if (instant > 0) sub.push(`pico ${instant.toFixed(1)}`);
    if (prefill > 0) sub.push(`prefill ${prefill.toFixed(1)}`);
    return {
      value: decode.toFixed(1),
      sub: sub.length ? sub.join(" · ") + " tok/s" : "decode (WebLLM)",
    };
  }
  if (instant > 0) {
    return { value: instant.toFixed(1), sub: "agora (estimativa)" };
  }
  return { value: "—", sub: "durante a geração" };
}
