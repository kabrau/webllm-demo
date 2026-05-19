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
  "qual",
  "mostre",
  "mostrar",
  "enumere",
  "enumerar",
  "indique",
  "indica",
  "indicar",
  "sugira",
  "sugerir",
  "recomende",
  "recomendar",
  "apresente",
  "apresentar",
]);

/** Tokens de filtro/preço — não contam como tema da busca. */
const FILTER_TOKENS = new Set([
  "free",
  "gratis",
  "gratuito",
  "gratuita",
  "gratuitas",
  "gratuitos",
  "freemium",
  "fremium",
  "tier",
  "tiers",
]);

/** Tokens curtos só contam se forem palavra inteira (evita "oi" em "endpoints"). */
const MIN_SUBSTRING_TOKEN_LEN = 3;

const GREETING_PHRASES = new Set([
  "oi",
  "ola",
  "hey",
  "hi",
  "hello",
  "salve",
  "fala",
  "opa",
  "e ai",
  "eai",
  "bom dia",
  "boa tarde",
  "boa noite",
  "tudo bem",
  "td bem",
  "blz",
  "beleza",
  "ok",
  "okay",
  "obrigado",
  "obrigada",
  "valeu",
]);

const GREETING_WORDS = new Set([
  "oi",
  "ola",
  "hey",
  "hi",
  "hello",
  "salve",
  "fala",
  "opa",
  "bom",
  "boa",
  "dia",
  "tarde",
  "noite",
  "tudo",
  "bem",
  "td",
  "blz",
  "beleza",
  "ok",
  "okay",
  "obrigado",
  "obrigada",
  "valeu",
  "eai",
  "ai",
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
  if (text == null || text === "") return "";
  return String(text)
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

/** "O que mais você sabe responder?", capacidades do assistente. */
function isCatalogHelpIntent(query) {
  const norm = normalize(query).replace(/\s+/g, " ").trim();
  return (
    /\b(o que mais|o que voce sabe|o que vc sabe|quais temas|como posso perguntar|o que consegue|o que sabe responder)\b/.test(
      norm
    ) || /\bme ajuda|ajuda com\b/.test(norm)
  );
}

function isExploreIntent(query) {
  const norm = normalize(query).replace(/\s+/g, " ").trim();
  return (
    /\b(o que|oque)\s+(tem|ha)\b/.test(norm) ||
    /\b(quais|que)\s+.+\s+(tem|ha|existem)\b/.test(norm) ||
    /\b(tem algo|ha algo|existem)\b/.test(norm)
  );
}

function isVagueFollowUp(query) {
  const norm = normalize(query).replace(/\s+/g, " ").trim();
  if (!norm) return true;
  return (
    /^(links?|isso|esses?|aquilo|os mesmos|sim|nao|não)\b/.test(norm) ||
    /\b(fazem isso|faz isso|sobre isso|disso|pra isso|para isso)\b/.test(norm) ||
    /^(quero|preciso de)\s+links?\b/.test(norm)
  );
}

function categoryMatchesTokens(category, tokens) {
  if (!category) return false;
  if (tokens.length === 0) return true;
  const catNorm = normalize(category);
  return tokens.every((t) => catNorm.includes(t));
}

/** Saudação curta — sem RAG de links; o LLM responde com prompt de boas-vindas. */
function isGreetingQuery(query) {
  const norm = normalize(query).replace(/\s+/g, " ").trim();
  if (!norm) return true;
  if (GREETING_PHRASES.has(norm)) return true;
  const tokens = tokenize(query);
  const meaningful = meaningfulTokens(tokens);
  if (meaningful.length === 0) return true;
  return meaningful.every((t) => GREETING_WORDS.has(t));
}

/**
 * Token aparece como palavra inteira (não substring dentro de "endpoints").
 */
function tokenMatchesInText(token, text) {
  const norm = normalize(text);
  if (!norm || !token) return false;

  const parts = norm.split(/[\s#_,-]+/).filter(Boolean);
  if (parts.includes(token)) return true;

  if (token.length >= MIN_SUBSTRING_TOKEN_LEN && norm.includes(token)) {
    const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`(?:^|[\\s#_,-])${escaped}(?:$|[\\s#_,-])`).test(norm);
  }

  return false;
}

function tokenMatchesTag(token, tag) {
  const tagNorm = normalize(tag);
  return tagNorm === token || tokenMatchesInText(token, tagNorm);
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
  const tags = link.tags || [];

  let score = 0;
  for (const token of queryTokens) {
    if (STOP_WORDS.has(token)) continue;
    if (token.length < MIN_SUBSTRING_TOKEN_LEN) continue;
    if (tokenMatchesInText(token, titleNorm)) score += 4;
    if (tokenMatchesInText(token, catNorm)) score += 5;
    if (tags.some((tag) => tokenMatchesTag(token, tag))) score += 3;
    if (tokenMatchesInText(token, descNorm) && !tokenMatchesInText(token, catNorm)) score += 1;
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
function resolveFilter(meaningful, rawQuery = "") {
  const norm = normalize(rawQuery);
  if (/tier\s+gratuit|free\s+tier|plano\s+gratuit|sem\s+custo/.test(norm)) {
    return "free";
  }
  for (const token of meaningful) {
    if (["free", "gratis", "gratuito", "gratuita", "gratuitas", "gratuitos"].includes(token)) {
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
    isVagueFollowUp(query) ||
    (isListIntent(queryTokens) &&
      meaningful.every((t) => LIST_VERBS.has(t) || STOP_WORDS.has(t)));

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

  if (isUtilityTextTask(query)) {
    return {
      results: [],
      queryTokens,
      retrievalMode: "utility-text",
    };
  }

  if (isGreetingQuery(query)) {
    const categories = [...new Set(links.map((l) => l.category))].sort((a, b) =>
      a.localeCompare(b, "pt")
    );
    return {
      results: [],
      queryTokens,
      retrievalMode: "greeting",
      categories,
    };
  }

  if (isCatalogHelpIntent(query)) {
    const featured = links.filter((l) => l.featured).slice(0, limit);
    const categories = [...new Set(links.map((l) => l.category))].sort((a, b) =>
      a.localeCompare(b, "pt")
    );
    return {
      results: featured.map((link) => ({ link, score: 0 })),
      queryTokens,
      retrievalMode: "catalog-help",
      categories,
    };
  }
  const listIntent = isListIntent(queryTokens);
  const category = resolveCategory(queryTokens, links);
  const meaningful = meaningfulTokens(queryTokens);
  const filter = resolveFilter(meaningful, query);
  const categoryTokens = meaningful.filter((t) => !FILTER_TOKENS.has(t));
  const exploreIntent = isExploreIntent(query);
  const categoryFocus = categoryMatchesTokens(category, categoryTokens);

  const useCategoryList =
    category &&
    (listIntent || filter != null || exploreIntent || categoryFocus);

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

const LINKS_RULES = `Você também ajuda com a curadoria de Links Úteis (marcelocabral.com.br/links).

Quando a pergunta for sobre LINKS DA CURADORIA:
- Use APENAS a seção LINKS RECUPERADOS abaixo.
- Copie título e URL exatamente; formato de lista: "- Título — URL"
- Se não houver links no contexto: "Não encontrei na curadoria de links."
- PROIBIDO inventar sites (TensorFlow, Udemy, Medium, etc.).
- PROIBIDO dizer que não tem acesso à lista.`;

const UTILITY_TEXT_RULES = `Quando o usuário colar TEXTO OU LISTA na mensagem (nomes, itens, bullets com *, etc.) e pedir tarefa sobre ESSE texto:
- Trabalhe SOMENTE com o que ele enviou (mensagem atual e histórico recente se precisar).
- NÃO invente links, sites ou itens que não estão no texto do usuário.
- NÃO redirecione para a curadoria de links, a menos que ele peça links explicitamente.
- Execute a tarefa pedida (ordenar, formatar, limpar, padronizar, remover duplicatas, etc.).
- Para ORDENAR NOMES: ordem alfabética pt-BR (ignorar maiúsculas/minúsculas); inclua TODOS os nomes da mensagem do usuário, sem omitir nem fundir itens.
- Formato da resposta: SOMENTE a lista final, um nome por linha, sem vírgulas juntando tudo, sem introdução, sem explicação, sem marcadores *.
- Preserve acentos, underscores e grafias do usuário (ex.: joao_pedro); normalize só espaços extras.
- Se o pedido for ambíguo, faça a interpretação mais útil e mostre o resultado direto (sem pedir desculpas genéricas).`;

const GENERAL_RULES = `REGRAS GERAIS:
- PROIBIDO repetir a mesma frase ou parágrafo.
- Respostas diretas; evite "estou pesquisando" ou "não tenho acesso" quando os dados estão na mensagem do usuário.`;

const ASSISTANT_INTRO =
  "Você é um assistente da demo WebLLM, em português do Brasil.";

function systemRulesForMode(retrievalMode) {
  if (retrievalMode === "utility-text") {
    return `${ASSISTANT_INTRO}\n\n${UTILITY_TEXT_RULES}\n\n${GENERAL_RULES}`;
  }
  return `${ASSISTANT_INTRO}\n\n${LINKS_RULES}\n\n${GENERAL_RULES}`;
}

/** Texto colado pelo usuário (ordenar nomes, etc.) — não misturar com busca de links. */
function isUtilityTextTask(query) {
  const norm = normalize(query).replace(/\s+/g, " ");
  if (!norm) return false;

  const linkIntent =
    /\b(links?|curadoria|marcelocabral|apis?\s+gratuit|tier\s+gratuit|liste\s+(apis?|cursos?|llm|scraping|crawlers?|hospedagem))\b/.test(
      norm
    );
  const utilityVerb =
    /\b(ordene|ordenar|organize|organizar|classifique|classificar|formate|formatar|padronize|padronizar|separe|separar|limpe|limpar|corrija|corrigir|remova|remover|ordem\s+alfab|alfabeticamente|sort)\b/.test(
      norm
    );
  const aboutUserList =
    /\b(esta|essa|a)\s+lista\b/.test(norm) &&
    /\b(nomes?|itens?|linhas?|texto|lista|entradas?)\b/.test(norm);
  const pastedListMarkers =
    (query.match(/\*/g) || []).length >= 2 ||
    (query.match(/,/g) || []).length >= 4 ||
    /^\s*[\*\-•]/m.test(query);

  if (utilityVerb || (aboutUserList && pastedListMarkers)) return true;
  if (linkIntent) return false;
  return false;
}

export function buildSystemPrompt(results, meta = {}) {
  const rules = systemRulesForMode(meta.retrievalMode);

  if (meta.retrievalMode === "utility-text") {
    return `${rules}

MODO ATUAL: TAREFA SOBRE TEXTO DO USUÁRIO (não é busca de links).
Leia a última mensagem do usuário no histórico (itens separados por * ou quebras de linha).
Responda APENAS com o resultado da tarefa — sem prefácio, sem "aqui está", sem links da curadoria.
Exemplo de formato para ordenar nomes (não copie estes nomes; use os do usuário):
AMANDA COSTA
felipe nunes
Maria das Graças`;
  }
  const isCategoryList = meta.retrievalMode?.startsWith("category-list");
  const isGreeting = meta.retrievalMode === "greeting";
  const categories = meta.categories || [];
  const listLines = results.map(
    ({ link }) => `- ${link.title} — ${link.url}`
  );

  if (isCategoryList && meta.category && results.length > 0) {
    const topicNote =
      meta.category === "Deploy & Hosting"
        ? '\n"Hospedagem" = categoria Deploy & Hosting.\n'
        : "";
    return `${rules}${topicNote}
TAREFA: listar ${results.length} link(s) — ${meta.category}${meta.filter === "free" ? " (tier gratuito)" : ""}.
Uma frase curta de introdução, depois copie TODAS as linhas (título e URL exatos):

${listLines.join("\n")}`;
  }

  if (isGreeting && categories.length > 0) {
    return `${rules}
TAREFA: o usuário disse olá. Responda em NO MÁXIMO 4 frases curtas.
Você é o assistente da curadoria marcelocabral.com.br/links.
NÃO cite URLs nesta saudação. NÃO invente plataformas.
Sugira 3 exemplos de pergunta que o usuário pode fazer:
- "liste APIs gratuitas"
- "liste cursos"
- "o que tem para scraping"
Temas disponíveis: ${categories.join(", ")}.`;
  }

  const context = formatContextBlock(results);
  const listHint =
    meta.retrievalMode === "catalog-help" && categories.length > 0
      ? `\nO usuário quer saber o que você pode responder. Temas: ${categories.join(", ")}.
Responda em até 6 frases. Dê 4 exemplos de pergunta. Não invente URLs.\n`
      : results.length > 0
        ? `\nHá ${results.length} link(s) no contexto. Use-os na resposta; não invente sites.\n`
        : "";

  return `${rules}${listHint}

--- LINKS RECUPERADOS (contexto RAG) ---
${context}
--- FIM DO CONTEXTO ---`;
}
