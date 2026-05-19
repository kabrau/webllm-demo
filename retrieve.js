/**
 * RAG leve: busca por palavras-chave + listagem por categoria.
 */

const STOP_WORDS = new Set([
  "liste",
  "listar",
  "listagem",
  "lista",
  "quais",
  "qual",
  "mostre",
  "mostrar",
  "enumere",
  "enumerar",
  "todos",
  "todas",
  "todo",
  "toda",
  "me",
  "os",
  "as",
  "de",
  "da",
  "do",
  "das",
  "dos",
  "na",
  "no",
  "em",
  "um",
  "uma",
  "por",
  "para",
  "com",
  "sobre",
  "o",
  "a",
  "e",
]);

const LIST_VERBS = new Set([
  "liste",
  "listar",
  "listagem",
  "lista",
  "quais",
  "mostre",
  "mostrar",
  "enumere",
  "enumerar",
]);

/** Token do usuário → categoria exata na KB */
const CATEGORY_ALIASES = {
  cursos: "Cursos",
  curso: "Cursos",
  "llms locais": "LLMs Locais",
  local: "LLMs Locais",
  locais: "LLMs Locais",
  ollama: "LLMs Locais",
  scraping: "Crawlers & Scrapers",
  scraper: "Crawlers & Scrapers",
  crawlers: "Crawlers & Scrapers",
  crawler: "Crawlers & Scrapers",
  deploy: "Deploy & Hosting",
  hosting: "Deploy & Hosting",
  hospedagem: "Deploy & Hosting",
  observabilidade: "Observabilidade & Monitoramento",
  monitoramento: "Observabilidade & Monitoramento",
  agent: "Agent Codes",
  agents: "Agent Codes",
  agentes: "Agent Codes",
  apis: "LLM APIs",
  api: "LLM APIs",
  llm: "LLM APIs",
  gratuitas: "LLM APIs",
  gratuito: "LLM APIs",
  "free tier": "LLM APIs",
};

function normalize(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s#-]/g, " ");
}

function tokenize(text) {
  return normalize(text)
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function meaningfulTokens(queryTokens) {
  return queryTokens.filter((t) => !STOP_WORDS.has(t));
}

function isListIntent(queryTokens) {
  return queryTokens.some((t) => LIST_VERBS.has(t));
}

/**
 * @param {string[]} queryTokens
 * @param {Array} links
 * @returns {string|null}
 */
function resolveCategory(queryTokens, links) {
  const meaningful = meaningfulTokens(queryTokens);

  for (const token of meaningful) {
    if (CATEGORY_ALIASES[token]) return CATEGORY_ALIASES[token];
  }

  const categories = [...new Set(links.map((l) => l.category))];
  for (const cat of categories) {
    const catNorm = normalize(cat);
    for (const token of meaningful) {
      if (catNorm === token || catNorm.includes(token)) return cat;
    }
  }

  return null;
}

function scoreLink(link, queryTokens) {
  const titleNorm = normalize(link.title);
  const catNorm = normalize(link.category);
  const descNorm = normalize(link.description);
  const tagsNorm = (link.tags || []).map(normalize).join(" ");

  let score = 0;
  for (const token of queryTokens) {
    if (STOP_WORDS.has(token)) continue;
    if (titleNorm.includes(token)) score += 4;
    if (catNorm.includes(token)) score += 5;
    if (tagsNorm.includes(token)) score += 3;
    if (descNorm.includes(token) && !catNorm.includes(token)) score += 1;
  }
  return score;
}

/** Cursos/recursos com tier gratuito (exclui só freemium sem free/gratuito). */
function linkIsFree(link) {
  const tags = (link.tags || []).map((t) => t.toLowerCase());
  const hasFree = tags.some((t) =>
    ["free", "gratuito", "free-tier", "free-misto"].includes(t)
  );
  const freemiumOnly =
    tags.some((t) => t === "freemium" || t === "fremium") && !hasFree;
  if (freemiumOnly) return false;
  if (hasFree) return true;
  const desc = normalize(link.description);
  return desc.includes("gratuito") && !desc.startsWith("freemium");
}

function linkIsFreemium(link) {
  const tags = (link.tags || []).map((t) => t.toLowerCase());
  return tags.some((t) => t === "freemium" || t === "fremium");
}

/**
 * @param {string[]} meaningful
 * @returns {"free"|"freemium"|null}
 */
function resolveFilter(meaningful) {
  for (const token of meaningful) {
    if (["free", "gratis", "gratuito", "gratuita", "gratuitos"].includes(token)) {
      return "free";
    }
    if (["freemium", "fremium"].includes(token)) return "freemium";
  }
  return null;
}

function applyFilter(links, filter) {
  if (filter === "free") return links.filter(linkIsFree);
  if (filter === "freemium") return links.filter(linkIsFreemium);
  return links;
}

function resultsFromCategory(links, category, filter = null) {
  let pool = links.filter((l) => l.category === category);
  if (filter) pool = applyFilter(pool, filter);
  return pool
    .sort((a, b) => a.title.localeCompare(b.title, "pt"))
    .map((link) => ({ link, score: 100 }));
}

/**
 * Enriquece busca com mensagens anteriores (ex.: "liste" após "me liste cursos").
 * @param {string} query
 * @param {Array<{role: string, content: string}>} priorMessages
 */
export function buildRetrievalQuery(query, priorMessages = []) {
  const queryTokens = tokenize(query);
  const meaningful = meaningfulTokens(queryTokens);
  const needsContext =
    meaningful.length === 0 ||
    (isListIntent(queryTokens) && meaningful.every((t) => LIST_VERBS.has(t) || STOP_WORDS.has(t)));

  if (!needsContext) return query;

  const priorText = priorMessages
    .filter((m) => m.role === "user")
    .slice(-2)
    .map((m) => m.content)
    .join(" ");

  if (!priorText) return query;
  return `${priorText} ${query}`.trim();
}

/**
 * @param {string} query
 * @param {Array} links
 * @param {number} limit
 */
export function retrieveLinks(query, links, limit = 8) {
  const queryTokens = tokenize(query);
  const listIntent = isListIntent(queryTokens);
  const category = resolveCategory(queryTokens, links);
  const meaningful = meaningfulTokens(queryTokens);
  const filter = resolveFilter(meaningful);
  const categoryTokens = meaningful.filter(
    (t) => !["free", "gratis", "gratuito", "gratuita", "gratuitos", "freemium", "fremium"].includes(t)
  );

  const useCategoryList =
    category &&
    (listIntent || categoryTokens.length <= 1) &&
    (listIntent || categoryTokens.length === 0 || categoryTokens.every((t) => normalize(category).includes(t)));

  if (useCategoryList) {
    const results = resultsFromCategory(links, category, filter);
    const mode = filter ? `category-list-${filter}` : "category-list";
    return {
      results,
      queryTokens,
      retrievalMode: mode,
      category,
      filter,
      totalInCategory: results.length,
    };
  }

  if (queryTokens.length === 0) {
    const featured = links.filter((l) => l.featured).slice(0, limit);
    return {
      results: featured.map((link) => ({ link, score: 0 })),
      queryTokens: [],
      retrievalMode: "featured",
    };
  }

  const scored = links
    .map((link) => ({ link, score: scoreLink(link, queryTokens) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score || a.link.title.localeCompare(b.link.title, "pt"));

  let results = scored.slice(0, limit);

  if (results.length === 0) {
    const featured = links.filter((l) => l.featured).slice(0, limit);
    results = featured.map((link) => ({ link, score: 0 }));
    return { results, queryTokens, retrievalMode: "fallback-featured" };
  }

  return { results, queryTokens, retrievalMode: "keyword", category: category || undefined };
}

export function formatContextBlock(results) {
  return results
    .map(({ link }, i) => {
      const tags = (link.tags || []).join(", ");
      return `[${i + 1}] ${link.title} (${link.category})
URL: ${link.url}
Descrição: ${link.description}
Tags: ${tags}`;
    })
    .join("\n\n");
}

const SYSTEM_RULES = `Você é o assistente dos Links Úteis de Marcelo Cabral (marcelocabral.com.br/links).
Responda em português do Brasil.

REGRAS:
- Use APENAS os links da seção LINKS RECUPERADOS abaixo.
- Copie títulos e URLs exatamente como estão (não invente, não troque palavras).
- Para listas: uma linha por item no formato "- Título — URL"
- Nunca diga que não tem acesso à lista.
- Se não houver dados no contexto: "Não encontrei na curadoria de links."`;

/** Resposta exata para listagens — evita alucinação do modelo pequeno. */
export function formatDeterministicList(results, meta = {}) {
  const n = results.length;
  let intro = `Encontrei ${n} item(ns) na curadoria`;
  if (meta.category) intro += ` (${meta.category})`;
  if (meta.filter === "free") intro += " com tier gratuito";
  intro += ":\n\n";

  const lines = results.map(
    ({ link }) => `- ${link.title} — ${link.url}`
  );
  return intro + lines.join("\n");
}

export function shouldUseDeterministicList(retrievalMode, resultsLength) {
  return retrievalMode?.startsWith("category-list") && resultsLength > 0;
}

export function buildSystemPrompt(results, meta = {}) {
  const context = formatContextBlock(results);
  const isCategoryList = meta.retrievalMode?.startsWith("category-list");
  const listHint =
    isCategoryList && meta.category
      ? `\nO usuário pediu uma listagem da categoria "${meta.category}"${meta.filter ? ` (filtro: ${meta.filter})` : ""}. Há ${results.length} itens no contexto — inclua TODOS na resposta, apenas estes.\n`
      : "";

  return `${SYSTEM_RULES}${listHint}

--- LINKS RECUPERADOS (contexto RAG) ---
${context}
--- FIM DO CONTEXTO ---`;
}

export { SYSTEM_RULES };
