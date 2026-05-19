/**
 * Catálogo e metadados dos modelos WebLLM (prebuiltAppConfig).
 */
import { prebuiltAppConfig } from "https://esm.run/@mlc-ai/web-llm";

const DEFAULT_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f32_1-MLC";

const NOTES = {
  "Qwen2.5-0.5B-Instruct-q4f32_1-MLC":
    "Muito leve, mas costuma ignorar o contexto RAG em listas. Prefira 1.5B.",
  "Qwen2.5-1.5B-Instruct-q4f32_1-MLC":
    "Melhor equilíbrio RAG/qualidade. Use se 0.5B alucinar e a GPU aguentar.",
  "Qwen3.5-0.8B-q4f32_1-MLC":
    "Qwen 3.5 — no Windows pode exigir mais VRAM; após erro de GPU prefira Qwen2.5.",
  "Llama-3.2-1B-Instruct-q4f32_1-MLC":
    "Bom upgrade leve se respostas do 0.5B forem fracas.",
  "Qwen2.5-1.5B-Instruct-q4f32_1-MLC":
    "Melhor equilíbrio qualidade/velocidade para chat aberto na aula.",
  "gemma-2-2b-it-q4f32_1-MLC":
    "Google Gemma 2 instruct. Download grande na 1ª vez (~42 partes).",
};

function inferFamily(modelId) {
  if (/^Qwen3\.5/i.test(modelId)) return "Qwen 3.5";
  if (/^Qwen3/i.test(modelId)) return "Qwen 3";
  if (/^Qwen2\.5/i.test(modelId)) return "Qwen 2.5";
  if (/^Qwen2/i.test(modelId)) return "Qwen 2";
  if (/Llama-3\.2/i.test(modelId)) return "Llama 3.2";
  if (/Llama-3\.1/i.test(modelId)) return "Llama 3.1";
  if (/Llama-3/i.test(modelId)) return "Llama 3";
  if (/Llama-2/i.test(modelId)) return "Llama 2";
  if (/Phi-4/i.test(modelId)) return "Phi 4";
  if (/Phi-3/i.test(modelId)) return "Phi 3";
  if (/gemma-2/i.test(modelId)) return "Gemma 2";
  if (/Mistral|Ministral/i.test(modelId)) return "Mistral / Ministral";
  if (/DeepSeek/i.test(modelId)) return "DeepSeek";
  if (/Hermes/i.test(modelId)) return "Hermes";
  if (/SmolLM/i.test(modelId)) return "SmolLM";
  if (/TinyLlama/i.test(modelId)) return "TinyLlama";
  if (/stablelm/i.test(modelId)) return "StableLM";
  if (/OLMo/i.test(modelId)) return "OLMo";
  if (/RedPajama/i.test(modelId)) return "RedPajama";
  if (/vision/i.test(modelId)) return "Multimodal";
  return "Outros";
}

function inferQuant(modelId) {
  if (modelId.includes("q4f32"))
    return { label: "q4f32", f16: false, compat: "Windows OK", bits: "4-bit pesos, calc f32" };
  if (modelId.includes("q4f16"))
    return { label: "q4f16", f16: true, compat: "Precisa shader-f16", bits: "4-bit + f16" };
  if (modelId.includes("q0f32"))
    return { label: "q0f32", f16: false, compat: "Windows OK", bits: "FP32 (mais pesado)" };
  if (modelId.includes("q0f16"))
    return { label: "q0f16", f16: true, compat: "Precisa shader-f16", bits: "FP16" };
  return { label: "?", f16: false, compat: "?", bits: "?" };
}

/** @returns {{ paramsB: number, paramsLabel: string, isMoE: boolean, moeDetail: string|null }} */
function parseParameters(modelId) {
  const id = modelId;

  if (/Mixtral|mixtral|8x7B|8x7b/i.test(id)) {
    return {
      paramsB: 7,
      paramsLabel: "8×7B MoE (~47B total, ~13B ativos)",
      isMoE: true,
      moeDetail: "Mixture of Experts: vários especialistas, ativa subset por token",
    };
  }

  const moeActive = id.match(/A(\d+(?:\.\d+)?)B/i);
  if (moeActive) {
    const active = moeActive[1] + "B";
    const totalMatch = id.match(/(\d+(?:\.\d+)?)B/i);
    const total = totalMatch ? totalMatch[1] + "B" : "?";
    return {
      paramsB: parseFloat(moeActive[1]),
      paramsLabel: `MoE ${active} ativos / ${total} total`,
      isMoE: true,
      moeDetail: "Arquitetura MoE (especialistas)",
    };
  }

  if (/MoE|moe/i.test(id)) {
    return {
      paramsB: 7,
      paramsLabel: "MoE",
      isMoE: true,
      moeDetail: "Mixture of Experts",
    };
  }

  if (/phi-3\.5-mini|Phi-3\.5-mini/i.test(id)) {
    return {
      paramsB: 3.8,
      paramsLabel: "3.8B parâmetros (Phi mini)",
      isMoE: false,
      moeDetail: null,
    };
  }

  const allBm = [...id.matchAll(/(\d+(?:\.\d+)?)([BM])/gi)];
  if (allBm.length) {
    const bm = allBm[allBm.length - 1];
    const val = parseFloat(bm[1]);
    const unit = bm[2].toUpperCase();
    const paramsB = unit === "M" ? val / 1000 : val;
    return {
      paramsB,
      paramsLabel: `${bm[1]}${unit} parâmetros`,
      isMoE: false,
      moeDetail: null,
    };
  }

  return { paramsB: 3, paramsLabel: "?", isMoE: false, moeDetail: null };
}

function inferWeightClass(paramsB, downloadGB) {
  if (paramsB < 1 || downloadGB < 0.8) return "Leve";
  if (paramsB < 3.5 || downloadGB < 2.5) return "Médio";
  if (paramsB < 9 || downloadGB < 6) return "Pesado";
  return "Muito pesado";
}

/** Estimativa de download na 1ª visita (ordem de grandeza). */
function estimateDownloadGB(paramsB, quantLabel, isMoE, modelId) {
  const mult =
    quantLabel === "q0f32" ? 2.0 : quantLabel === "q0f16" ? 1.5 : quantLabel === "q4f16" ? 0.55 : 0.85;
  let gb = paramsB * mult;
  if (isMoE) gb *= 1.15;
  if (modelId.includes("-1k")) gb *= 0.75;
  return Math.max(0.25, Math.round(gb * 10) / 10);
}

function inferContext(modelId) {
  if (modelId.includes("-1k") || modelId.includes("_1k")) {
    return { label: "1k tokens", short: "ctx 1k", ram: "menos RAM" };
  }
  if (modelId.includes("4k")) return { label: "4k tokens", short: "ctx 4k", ram: "" };
  return { label: "contexto padrão (~4k)", short: "", ram: "" };
}

function inferRole(modelId) {
  if (/Coder|code/i.test(modelId)) return "Código";
  if (/Math/i.test(modelId)) return "Matemática";
  if (/vision/i.test(modelId)) return "Multimodal (visão)";
  if (/Reasoning|R1|DeepSeek-R1/i.test(modelId)) return "Raciocínio";
  if (/Base/i.test(modelId) && !/Instruct|it|chat|zephyr/i.test(modelId)) return "Base (não chat)";
  if (/Instruct|it|chat|zephyr/i.test(modelId)) return "Chat / instruct";
  return "Uso geral";
}

function shortDisplayName(modelId) {
  const id = modelId.replace(/-q4f32_1-MLC.*|-q4f16_1-MLC.*|-q0f32-MLC.*|-q0f16-MLC.*|-MLC.*$/i, "");
  if (id.length <= 42) return id;
  return id.slice(0, 40) + "…";
}

/**
 * Metadados completos de um modelo.
 */
export function getModelMeta(modelId) {
  const quant = inferQuant(modelId);
  const params = parseParameters(modelId);
  const downloadGB = estimateDownloadGB(params.paramsB, quant.label, params.isMoE, modelId);
  const weightClass = inferWeightClass(params.paramsB, downloadGB);
  const context = inferContext(modelId);
  const role = inferRole(modelId);
  const family = inferFamily(modelId);

  const tags = [weightClass, params.paramsLabel, quant.label, `~${downloadGB} GB`];
  if (params.isMoE) tags.push("MoE");
  if (context.short) tags.push(context.short);
  if (quant.f16) tags.push("⚠ f16");

  return {
    id: modelId,
    family,
    shortName: shortDisplayName(modelId),
    weightClass,
    paramsLabel: params.paramsLabel,
    paramsB: params.paramsB,
    isMoE: params.isMoE,
    moeDetail: params.moeDetail,
    quant: quant.label,
    quantBits: quant.bits,
    quantCompat: quant.compat,
    needsF16: quant.f16,
    downloadGB,
    downloadLabel: `~${downloadGB} GB download (1ª vez)`,
    context: context.label,
    contextShort: context.short,
    role,
    optionLabel: `${shortDisplayName(modelId)} — ${tags.join(" · ")}`,
  };
}

/** Texto longo abaixo do select. */
export function describeModel(modelId) {
  const m = getModelMeta(modelId);
  const lines = [
    NOTES[modelId] || null,
    `Classe: ${m.weightClass} · ${m.paramsLabel} · ${m.downloadLabel}`,
    `Quantização: ${m.quant} (${m.quantBits}) — compatibilidade: ${m.quantCompat}`,
    `Função: ${m.role} · Contexto: ${m.context}`,
    m.isMoE ? `Arquitetura: MoE — ${m.moeDetail}` : null,
    m.needsF16
      ? "⚠ No Windows sem shader-f16 este modelo pode falhar. Prefira q4f32 ou marque o filtro."
      : "Recomendado para Windows: quant q4f32.",
  ].filter(Boolean);
  return lines.join("\n");
}

export function formatModelOptionLabel(modelId) {
  return getModelMeta(modelId).optionLabel;
}

export function getModelCatalog() {
  const seen = new Set();
  const list = [];

  for (const rec of prebuiltAppConfig.model_list) {
    const id = rec.model_id;
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const meta = getModelMeta(id);
    list.push({
      id,
      family: meta.family,
      description: describeModel(id),
      optionLabel: meta.optionLabel,
      quant: meta.quant,
      needsF16: meta.needsF16,
      weightClass: meta.weightClass,
      paramsB: meta.paramsB,
      downloadGB: meta.downloadGB,
    });
  }

  list.sort((a, b) => {
    const fc = a.family.localeCompare(b.family, "pt");
    if (fc !== 0) return fc;
    if (a.needsF16 !== b.needsF16) return a.needsF16 ? 1 : -1;
    return a.paramsB - b.paramsB || a.id.localeCompare(b.id, "pt");
  });

  return list;
}

export function getDefaultModelId() {
  const catalog = getModelCatalog();
  if (catalog.some((m) => m.id === DEFAULT_MODEL_ID)) return DEFAULT_MODEL_ID;
  return catalog.find((m) => !m.needsF16)?.id ?? catalog[0]?.id;
}

export { DEFAULT_MODEL_ID };
