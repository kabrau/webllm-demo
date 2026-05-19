# WebLLM Demo — Links Úteis

Site de demonstração didática: inferência de LLM **no navegador** com [WebLLM](https://github.com/mlc-ai/web-llm), RAG simples sobre a curadoria [marcelocabral.com.br/links](https://marcelocabral.com.br/links) e painel inspetor para ver prompts e respostas.

## Requisitos

- **Chrome** ou **Edge** recente com **WebGPU** habilitado
- Conexão de internet na **primeira visita** (download do modelo ~300–500 MB)
- Servidor HTTP local (`file://` não funciona por causa dos Workers)

## Como rodar

```bash
cd c:\acertpix\llm-navegador
npx --yes serve . -p 8080
```

Abra: **http://localhost:8080**

Aguarde o overlay de carregamento terminar antes de usar o chat.

## Roteiro da aula (5–8 min)

1. Abrir o site → mostrar overlay de loading, barra de progresso e log de etapas.
2. Quando o status ficar **Pronto**, explicar as 3 colunas: Como funciona | Chat | Inspetor.
3. Clicar em um chip de sugestão (ex.: “LLM local”).
4. No **Inspetor → RAG**: mostrar quais links foram recuperados e o score.
5. Aba **System**: mostrar o prompt com contexto injetado.
6. Aba **Messages**: mostrar o JSON enviado à API WebLLM.
7. Durante a geração: streaming no chat; depois aba **Resposta** e **Meta** (tempo, tokens).
8. Recarregar a página → loading mais rápido (cache do browser).

## Estrutura

| Arquivo | Função |
|---------|--------|
| `index.html` | Layout demo (aula + chat + inspetor + overlay) |
| `styles.css` | Tema escuro legível em projetor |
| `app.js` | Orquestração, chat, inspetor, histórico de turnos |
| `worker.js` | Web Worker com handler WebLLM |
| `retrieve.js` | Busca RAG por palavras-chave |
| `data/links-kb.json` | 46 links da curadoria |

## Modelo

Padrão: `Qwen2.5-0.5B-Instruct-q4f32_1-MLC` (leve; **q4f32** sem extensão `shader-f16`).

Alternativas no catálogo WebLLM: `Llama-3.2-1B-Instruct-q4f32_1-MLC`.  
**Não use** `Qwen2-0.5B-Instruct-q4f32_1-MLC` — esse ID não existe (só q4f16 ou q0f32).

Altere em `app.js` → constante `MODEL_ID`.

### Erro `extension 'f16' is not allowed`

Modelo ou cache antigo usa **f16**. Faça:

1. DevTools → Application → **Clear site data**
2. Recarregue com Ctrl+Shift+R
3. Use um `MODEL_ID` com sufixo **q4f32_1-MLC**

### Erro `ModelNotFoundError`

O `model_id` não está no `prebuiltAppConfig` do WebLLM. Confira IDs em [web-llm config](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts) ou use o padrão acima.

## Resultado dos testes

Testado em **2026-05-19** (ambiente de build do agente + servidor local).

| Item | Resultado |
|------|-----------|
| Servidor `npx serve` :8080 | **OK** — index, app.js, worker.js, KB retornam HTTP 200 |
| KB (`links-kb.json`) | **OK** — 46 links |
| RAG (`retrieve.js`) | **OK** — ex.: query "LLM local gratuito" retorna Ollama, LM Studio, etc. |
| UI / layout / inspetor | **OK** — página carrega, overlay de loading, 3 colunas |
| WebLLM + inferência | **Não testado aqui** — Playwright headless sem GPU (`Unable to find a compatible GPU`) |
| Teste completo no seu PC | **Pendente no seu browser** — veja abaixo |

### Validar em 1 minuto (no seu computador)

1. `npm start` ou `npx serve . -p 8080`
2. Abra **http://localhost:8080** no **Chrome** ou **Edge**
3. Aguarde o overlay de carregamento (1ª vez pode demorar vários minutos)
4. Clique em **LLM local** e confira o inspetor (abas RAG / System / Resposta)

### Teste automatizado (opcional)

```bash
npm run test:smoke
```

Requer servidor rodando e **GPU/WebGPU** — falha em CI/headless sem placa compatível.

## Métricas na tela (RAM, VRAM, tok/s)

Barra abaixo do cabeçalho, atualizada a cada ~2 s e durante o streaming:

| Métrica | Fonte |
|---------|--------|
| **RAM** | Heap JS (`performance.memory`) ou memória da aba (`measureUserAgentSpecificMemory` no Chrome) |
| **VRAM** | WebLLM informa **fabricante GPU** e **limite de buffer**; o browser **não expõe VRAM em uso** |
| **Tokens/s** | Instantâneo (estimativa no stream) + **decode/prefill** do `usage.extra` do WebLLM no fim da geração |

Valores exatos de VRAM/RAM do processo exigiriam ferramentas do SO (Gerenciador de Tarefas, `nvidia-smi`).

## Limitações

- Modelos pequenos podem errar detalhes; o prompt restringe respostas à KB.
- KB estática: não reflete alterações automáticas no site ao vivo.
- Firefox: suporte WebGPU variável.
- **q4f32** usa mais memória que q4f16, mas é mais compatível no Windows.
- Aviso `powerPreference` no console é inofensivo ([crbug.com/369219127](https://crbug.com/369219127)).

## Licença dos dados

Links curados por Marcelo Cabral. WebLLM: Apache-2.0.
