# Modelos na demo WebLLM

Este documento explica **de onde vêm os modelos** usados no [llm-navegador](README.md), **qual o formato**, e **o que é possível (ou não) usar** além do catálogo padrão do WebLLM.

---

## Resumo

| Pergunta | Resposta |
|----------|----------|
| Todos os modelos vêm só do WebLLM? | **Na demo atual, sim** — a lista vem do `prebuiltAppConfig` do pacote `@mlc-ai/web-llm`. |
| Dá para buscar de outro lugar? | **Sim, com ressalvas** — o modelo precisa estar **compilado para MLC** (runtime no navegador), não basta um `.gguf` do Ollama. |
| Qual o formato? | **Pacote MLC** (pesos quantizados + metadados + kernels WebGPU/WASM), não PyTorch/GGUF “cru”. |

---

## Como a demo carrega os modelos

1. O arquivo [`models.js`](models.js) importa o catálogo oficial:

   ```javascript
   import { prebuiltAppConfig } from "https://esm.run/@mlc-ai/web-llm";
   ```

2. Cada entrada em `prebuiltAppConfig.model_list` define um `model_id` (ex.: `Qwen2.5-0.5B-Instruct-q4f32_1-MLC`).

3. O [`worker.js`](worker.js) usa `WebWorkerMLCEngineHandler` do WebLLM para baixar os pesos (na 1ª vez) e rodar inferência via **WebGPU** no navegador.

4. O seletor na interface agrupa os modelos por família (Qwen, Llama, Gemma, etc.) e mostra metadados estimados (tamanho, quantização, classe Leve/Médio/Pesado).

5. A demo **filtra** o catálogo oficial: entram só modelos de **chat/instruct**. Ficam de fora, por exemplo, **Snowflake Arctic Embed** (`snowflake-arctic-embed-*`) — são modelos de **embedding** (vetores para busca), não LLM de conversa.

**Não há servidor de inferência:** tudo roda no cliente (browser + Worker).

---

## O que significa o `model_id`

Exemplo: `Qwen2.5-0.5B-Instruct-q4f32_1-MLC`

| Parte | Significado |
|-------|-------------|
| `Qwen2.5-0.5B-Instruct` | Modelo base e variante (instrução/chat) |
| `q4f32_1` | Quantização **4-bit**, cálculo em **float32** (mais compatível no Windows) |
| `MLC` | Pacote compilado para o runtime **MLC** no browser |

Outras quantizações comuns no catálogo:

| Sufixo | Compatibilidade | Observação |
|--------|-----------------|------------|
| `q4f32` | **Alta** (recomendado no Windows) | Demo filtra só `q4f32` por padrão |
| `q4f16` | Requer **shader-f16** no WebGPU | Pode falhar em alguns PCs Windows |
| `q0f32` | Alta | Mais pesado (menos compressão) |

---

## Formato técnico (pacote MLC)

Não é um único arquivo como no Ollama (`.gguf`). O WebLLM baixa um **pacote** preparado para WebGPU/WASM, em geral hospedado no Hugging Face (contas/projeto MLC), contendo:

- **Pesos quantizados** em vários arquivos (shards) — por isso o progresso tipo `[12/42]` na 1ª carga
- **Tokenizer e config** do modelo (template de chat, vocabulário)
- **Artefatos de compilação MLC** (kernels para aquela quantização e dispositivo WebGPU)

Esse pacote é produzido **fora** do browser, com a ferramenta [MLC LLM](https://mlc.ai/mlc-llm), e só então consumido pelo WebLLM.

---

## Dá para usar modelos de “outro lugar”?

### Funciona

| Origem | Como |
|--------|------|
| Catálogo `prebuiltAppConfig` | Padrão — zero config extra |
| Modelo **compilado por você** com MLC LLM | Publicar artefatos (ex.: Hugging Face) e registrar em `appConfig` / `model_list` |
| CDN ou HF **seu** | Desde que o layout seja o esperado pelo MLC e o `model_id` aponte para lá |

### Não funciona direto

| Origem | Por quê |
|--------|---------|
| Arquivo `.gguf` do Ollama | Formato e runtime diferentes |
| `safetensors` / PyTorch do Hugging Face “cru” | Precisa compilação MLC antes |
| Checkpoint sem quantização para browser | Muito grande e sem kernels WebGPU |

### Arquitetura diferente (não é “só trocar o arquivo”)

| Abordagem | O que muda |
|-----------|------------|
| **API remota** (OpenAI, Groq, OpenRouter, etc.) | `fetch` + JSON OpenAI-compatible; **sem** WebGPU local |
| **Ollama / LM Studio** no PC | Servidor local HTTP; a demo atual **não** usa isso |

Misturar **WebLLM (local)** + **API (nuvem)** é possível no código, mas são dois pipelines distintos.

---

## Fluxo para adicionar um modelo customizado (visão geral)

1. Escolher o modelo base (ex.: Llama 3.2 1B Instruct).
2. Compilar com **MLC LLM** para WebGPU, com quantização desejada (`q4f32` recomendado no Windows).
3. Publicar os artefatos (Hugging Face ou outro storage estático).
4. Estender a config do WebLLM (`appConfig.model_list`) com o novo `model_id`, URLs e parâmetros.
5. (Opcional) Atualizar [`models.js`](models.js) com descrições e aliases para o seletor da demo.

Documentação oficial: [WebLLM — Bring Your Own Model](https://webllm.mlc.ai/docs/user/bring_your_own_model) e [MLC LLM docs](https://llm.mlc.ai/docs/).

---

## Modelo padrão desta demo

| Item | Valor |
|------|--------|
| ID padrão | `Qwen2.5-0.5B-Instruct-q4f32_1-MLC` |
| Motivo | Leve, rápido download, melhor compatibilidade WebGPU no Windows |

Modelos maiores (1.5B, 2B, 7B) melhoram a qualidade do texto, mas exigem mais **RAM/VRAM** e aumentam o risco de erro `DXGI_ERROR_DEVICE_REMOVED` em GPUs modestas.

---

## Modelos maiores e erro ao trocar no seletor

A [documentação do WebLLM](https://webllm.mlc.ai/docs/user/basic_usage.html) e issues no GitHub (ex.: [#517](https://github.com/mlc-ai/web-llm/issues/517), [#647](https://github.com/mlc-ai/web-llm/issues/647)) explicam o padrão:

| Mensagem típica | Causa provável |
|-----------------|----------------|
| `Device was lost during reload` | **VRAM insuficiente** ao carregar o novo modelo (o anterior ainda ocupa a GPU) |
| `DXGI_ERROR_DEVICE_REMOVED` (Windows) | Driver/GPU resetou após sobrecarga |

**O que o WebLLM recomenda na prática:**

1. **Menos VRAM:** modelos menores, sufixo **`-1k`** (contexto/KV menor), ou quantização **q4f16** se o browser tiver shader-f16 ([webgpureport.org](https://webgpureport.org/)).
2. **Ordem de grandeza (q4f32, contexto padrão):** ~1–2 GB para 1B, ~3–5 GB para 3B, **~5–8 GB para 7–8B** — varia com GPU e abas abertas.
3. **Trocar modelo:** `reload()` já chama `unload()` internamente, mas no Windows a VRAM pode não liberar a tempo; a demo **encerra o worker**, chama `unload()` e espera ~600 ms antes de criar um engine novo.
4. **Se falhar:** F5 na página, feche Discord/jogos/outras abas com IA e volte por **Qwen2.5-0.5B** ou **1.5B**.

Avisos por modelo aparecem no log de carregamento (`getModelLoadWarnings` em [`models.js`](models.js)).

---

## Limitações importantes

1. **Catálogo ≠ todos os modelos do mundo** — só o que a equipe MLC/WebLLM pré-compilou e listou.
2. **1ª visita** — download grande; depois o cache do navegador acelera.
3. **GPU** — WebGPU obrigatório (Chrome/Edge recentes); falhas de driver não são corrigidas só no JavaScript.
4. **Qualidade vs tamanho** — modelos &lt; 1B tendem a ignorar o RAG ou repetir texto; a demo usa prompt forte, limite de tokens e detecção de loop.

---

## Referências

- [WebLLM (GitHub)](https://github.com/mlc-ai/web-llm)
- [Config `prebuiltAppConfig`](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts)
- [MLC LLM](https://mlc.ai/mlc-llm)
- [Documentação WebLLM](https://webllm.mlc.ai/docs/)
