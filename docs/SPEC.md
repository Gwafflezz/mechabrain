# Spec: Mecha-Brain — memória agentica drop-in para vaults Markdown

> [!abstract] Natureza deste documento
> Este é o **contrato agnóstico a vault** do Mecha-Brain, na sua casa canônica: o
> repositório do kernel. Nada aqui referencia uma vault, agente, plugin ou máquina
> específicos — tudo que é particular de um deployment pertence ao manifest
> (`_meta/config.yaml`) e ao doc de deploy correspondente da vault hospedeira.
> Base conceitual: **CoALA** (Sumers et al., [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)).
>
> A spec nasceu como doc de projeto (v0.1, 2026-07-16) e foi importada para cá em
> 2026-07-22. Emendas posteriores à v0.1 estão marcadas no texto com a versão do
> kernel que as introduziu — p.ex. **(v0.2.0)**, **(v0.2.6)**. Ver também o
> [README](../README.md) (visão geral), o [guia de setup](SETUP.md) (passo a passo)
> e o [CLAUDE.md](../CLAUDE.md) (governança de agentes neste repositório).

---

## 1. O que é

O **Mecha-Brain** é um sistema de memória agentica *drop-in* para qualquer vault Markdown (Obsidian ou não), composto de duas partes:

1. **Uma pasta contratual** (`mecha-brain/`) instalada na raiz da vault hospedeira — a área agent-writable, versionada junto com a vault.
2. **Um kernel** (serviço de memória + servidor MCP, CLI `mechabrain`) instalado fora da vault, que serve busca híbrida (RAG), escrita governada e manutenção sobre essa pasta.

Ele dá a um conjunto arbitrário de agentes LLM os quatro tipos de memória do CoALA (working, episódica, semântica, procedural), recall semântico compartilhado, e um ciclo de decisão explícito para escrita — sem descaracterizar o PKM humano da vault hospedeira.

**Teste de aceitação da spec:** `mechabrain init <qualquer-vault>` + editar um `config.yaml` deve ser *tudo* que é preciso para portar o sistema. Se um passo de instalação exigir editar código, a spec falhou.

---

## 2. Princípios de design

| # | Princípio | Racional |
|---|---|---|
| P1 | **Markdown versionado é a fonte-da-verdade**; todo índice é derivado e rebuildável | auditabilidade e git de graça; busca semântica sem perder a verdade legível |
| P2 | **Três camadas estritamente separadas**: kernel (código) / deployment (dados+política) / runtime (por máquina) | código, dados e estado de máquina têm ciclos de vida e mecanismos de distribuição diferentes |
| P3 | **Zero symlinks, zero caminhos absolutos** — descoberta por convenção, compartilhamento via config + caminhos relativos | symlinks e paths absolutos não sobrevivem a sync de nuvem, multi-SO ou realocação da vault |
| P4 | **Sandbox por default**: escrita livre só em `mecha-brain/`; o resto da vault é read-only + propostas | CoALA §6 — "define read/write access per memory module" |
| P5 | **Escrita passa por gate**: procedimento de decisão explícito (propor → avaliar → selecionar → executar) | CoALA §4.6 — decision cycle; memória sem curadoria vira ruído |
| P6 | **Tudo específico da vault é dado, não código** — vive no manifest | portabilidade drop-in: portar = editar config, nunca código |
| P7 | **Proveniência obrigatória**: todo resultado de retrieval carrega caminho/wikilink da fonte; toda memória carrega autor e origem | memória citável é auditável; multi-agente exige autoria clara |
| P8 | **Consolidação nunca destrói**: dedup preserva detalhe; decay arquiva, não deleta | super-resumo derruba acurácia; decay Ebbinghaus com reversibilidade |

---

## 3. Contrato fixo: estrutura da pasta

Esta árvore é o invariante portável. Os nomes abaixo **não são configuráveis** (são o contrato que o kernel conhece); o *conteúdo* é governado pelo manifest.

```
mecha-brain/
├── AGENTS.md            # contrato para agentes — GERADO do template + config (bloco gerenciado)
├── hot.md               # blackboard compartilhado (ver §8.4) — escrito só pelo consolidador
├── index.md             # MOC mestre — magro; sharda por escopo em indices/ quando cresce (§9)
├── indices/             # (gerado sob demanda) índices por escopo: <scope>.md
├── Semantic/            # fatos/insights consolidados, atômicos, curados
│   └── <nota INS>
├── Episodic/            # eventos/sessões — IMUTÁVEL, append-only, uma subpasta por agente
│   └── <agente-id>/<nota MEM>
├── Procedural/          # playbooks/how-tos destilados, com deprecação
│   └── <nota PROC>
├── Research/            # (opcional, habilitável no manifest) relatórios de pesquisa longos
│   └── <nota RES>
├── _inbox/              # propostas de mudança em notas humanas (default; redirecionável no manifest)
└── _meta/
    ├── config.yaml      # ★ O MANIFEST — única casa de tudo que é específico do deployment
    ├── links.jsonl      # (v0.2.x) arestas autoradas de memory.link — VERSIONADO (P1)
    ├── schema.md        # spec de frontmatter renderizada (gerada do manifest, legível por humanos)
    └── index/           # vetores/BM25/derivados — GITIGNORED, por máquina, rebuildável
```

Mapeamento CoALA → estrutura: working memory = contexto de cada agente (não é pasta; ver §8.4 sobre `hot.md`); episódica = `Episodic/`; semântica = `Semantic/`; procedural *como conhecimento* = `Procedural/`; procedural *como código* = o kernel + skills dos agentes (fora da vault, ver §4).

---

## 4. As três camadas

Código, dados e estado por máquina nunca se misturam — código em pasta sincronizada por nuvem, ou caminhos de máquina dentro do código, são proibidos por construção. A separação é normativa:

| Camada | O quê | Onde vive | Sincroniza? |
|---|---|---|---|
| **Kernel** | CLI `mechabrain`, serviço de memória, servidor MCP, templates | Repositório próprio; instalado via `pipx`/`uv tool install` | Não (é software versionado, com releases) |
| **Deployment** | `mecha-brain/` inteiro: memórias, manifest, AGENTS.md | Dentro da vault hospedeira | Sim, com o git da vault |
| **Runtime** | Índice vetorial (`_meta/index/`), portas, env, caches | Por máquina, gitignored | **Nunca** |

Regras normativas:

- **R4.1** O kernel nunca contém um caminho, nome de vault, nome de agente ou chave de frontmatter de deployment. Se precisar de um valor desses, ele vem do manifest.
- **R4.2** O deployment nunca contém caminho absoluto. Todos os paths do manifest são relativos à raiz da vault (definida como o diretório pai de `mecha-brain/`).
- **R4.3** Descoberta por convenção: o kernel localiza a vault por (nesta ordem) argumento explícito `--vault`, env `MECHABRAIN_VAULT`, ou subindo a árvore a partir do CWD até encontrar `mecha-brain/_meta/config.yaml` (como o git encontra `.git`).
- **R4.4** Symlinks não são mecanismo de nada. Compartilhar = configurar.
- **R4.5** O manifest declara `kernel_min_version`; o kernel recusa servir um deployment mais novo do que entende.

---

## 5. O manifest: `_meta/config.yaml`

Única fonte de tudo que é específico do deployment. Spec comentada com os defaults:

```yaml
mecha_brain:
  spec_version: "0.1"          # versão deste contrato
  kernel_min_version: "0.1.0"

# ── Agentes ─────────────────────────────────────────────────────
# O kernel NÃO conhece agentes por nome. Este registry gera:
# subpastas de Episodic/, valores válidos de `agent:` e tags agent/<id>.
agents:
  - id: exemplo                # slug curto, minúsculo — o RUNTIME (quem executa)
    display_name: "Exemplo"
    profiles: []               # personas deste runtime (ex.: [tutor, orquestrador]) — R6.6
    private_store: none        # descrição de memória privada externa, se houver
                               # (informativo — o kernel não a gerencia; ver §8.3;
                               #  com perfis, pode ser declarado por perfil)

# ── Escopos (projetos) ──────────────────────────────────────────
# Toda memória carrega `scope:` — slug de projeto ou "global" (R6.5).
scopes:
  known: []                    # slugs válidos (vazio = qualquer slug aceito)
  default: global              # se `known` for não-vazio, `default` DEVE constar nele (R5.1)

# ── Nomenclatura ────────────────────────────────────────────────
naming:
  note_name: "{date}_{prefix}_{slug}.md"   # template; {date} = YYYY-MM-DD
  dated_types: [episodic, semantic, research]  # procedural é atemporal
  prefixes:
    episodic: MEM
    semantic: INS
    procedural: PROC
    research: RES
  proposal_name: "{date}_AI-PROPOSAL_{slug}.md"

# ── Zonas e fronteiras (paths relativos à raiz da vault) ────────
zones:
  proposals_dir: "mecha-brain/_inbox/"   # deployment pode apontar p/ inbox nativo da vault
  read_only_index: []                    # pastas humanas indexadas como contexto read-only
    # - "PastaHumana/"
  research_enabled: true                 # habilita Research/

# ── Frontmatter e tags ──────────────────────────────────────────
frontmatter:
  denylist_keys: []            # chaves PROIBIDAS em notas de agente
                               # (ex.: chaves de plugin de publicação da vault hospedeira)
  denylist_tags: []            # tags proibidas (ex.: tags que disparam automações humanas)
  tag_namespaces:
    memory: "mem"              # gera mem/{episodic,semantic,procedural,research}
    agent: "agent"             # gera agent/<id>
  required_extra_tags: []      # tags que o deployment exige em toda nota de agente

# ── Gate (§8.2) — v0.2.0 ────────────────────────────────────────
gate:
  reject_on: []                # warnings do gate elevados a rejeição neste deployment.
                               # Único valor elevável: confidence_unverified (a condição
                               # — high sem meta.evidence — é fato mecânico). Itens de
                               # julgamento/heurística (reusable, atomic) NÃO são eleváveis.

# ── Retrieval ───────────────────────────────────────────────────
retrieval:
  embedding:
    provider: sentence-transformers   # sentence-transformers | http | hash (v0.2.x)
                                      # http = endpoint compatível OpenAI; hash =
                                      # determinístico, SÓ teste/CI — sem semântica.
    model: "BAAI/bge-m3"              # default multilíngue
  hybrid:
    vector_weight: 0.6
    bm25_weight: 0.4
  contextual_retrieval: true   # prepend de contexto do documento ao chunk antes de indexar
  rerank: false                # reservado. Enquanto o kernel não traz um reranker,
                               # `true` é RECUSADO com erro claro (v0.2.7) — aceitar a
                               # flag sem honrá-la seria default silencioso (R5.1).
  link_expansion:              # expansão "graph-lite" pelo grafo AUTORADO (§7.1)
    default_hops: 1            # 0 desliga por default
    max_hops: 2
  store: numpy                 # numpy | lancedb | sqlite-vec — sempre em _meta/index/
                               # numpy é o DEFAULT: brute-force cosine, zero dep pesada,
                               # <10ms em vault pessoal (~1e4 chunks); ANN não compra nada
                               # nessa escala. lancedb/sqlite-vec exigem o extra do kernel.

# ── Manutenção ──────────────────────────────────────────────────
maintenance:
  decay_days: 90               # sem acesso há N dias → status: arquivado (nunca deletar)
  dedup_similarity: 0.92       # acima disso, candidato a fusão/supersedes
  commit_prefix: "chore(ai-memory):"
  proc_stale_days: 180         # v0.2.0 — PROC ativo sem teste há N dias entra no
                               # RELATÓRIO da consolidação (§9.4); 0 desliga.
  hot_days: 21                 # v0.2.5 — janela de atenção do hot.md (R8.2): só entra
                               # no blackboard memória tocada nos últimos N dias.
                               # 0 desliga a janela.
```

**R5.1** O kernel valida o manifest no boot e falha alto (mensagem clara) em chave desconhecida ou valor inválido — nada de defaults silenciosos para chave com typo.

---

## 6. Schema de frontmatter (genérico)

Toda nota escrita pelo kernel carrega:

```yaml
---
title: "..."
tags: [<namespaces do manifest>, <required_extra_tags>]
  # ex. gerado: [mem/semantic, agent/<id>, ...extra]
created: <timestamp>
modified: <timestamp>
agent: <id do registry>        # quem escreveu (runtime)
profile: <persona>             # opcional — perfil do runtime autor (R6.6)
scope: <slug | global>         # projeto ao qual a memória pertence (R6.5)
source: "<sessão/execução/ponte de origem>"  # rastreabilidade (P7)
confidence: high|medium|low
last_accessed: <timestamp>     # atualizado pelo consolidador (ver §7.3)
last_tested: <timestamp>       # v0.2.0, só procedural — carimbado pelo kernel na escrita
                               # (a evidência do §8.2 item 6 atesta uma execução naquela
                               # data); atualizado por quem retesta. Alimenta §9.4.
supersedes: "[[...]]"          # opcional — memória que esta substitui
status: ativo|arquivado|deprecado   # deprecado só para procedural
---
```

> **(v0.2.6) Timestamps carregam a hora.** Na v0.1, `created`/`modified`/
> `last_accessed`/`last_tested` eram `YYYY-MM-DD`. Desde o kernel v0.2.6 são
> datetimes com timezone (`YYYY-MM-DD HH:MM:SS±HH:MM`) — a data pura segue aceita
> na leitura, e o `{date}` do nome de arquivo continua só o dia. O `schema.md`
> gerado documenta o formato vigente do deployment.

Regras:

- **R6.1** `memory.write` rejeita nota contendo qualquer chave de `denylist_keys` ou tag de `denylist_tags`. O erro cita a regra do manifest violada.
- **R6.2** `agent:` deve existir no registry; o kernel recusa autor desconhecido.
- **R6.3** Notas em `Episodic/` nunca são editadas após criadas (append-only). Correção = nova nota com `supersedes`.
- **R6.4** `_meta/schema.md` é *gerado* do manifest (não editado à mão) — é a versão legível por humanos e agentes das regras acima, sempre consistente com o config.
- **R6.5** `scope:` é obrigatório (`global` na ausência de projeto). Se `scopes.known` for não-vazio, o valor deve constar nele. Escopo existe para impedir **contaminação cruzada**: um fato verdadeiro no projeto A não pode ser recuperado como verdade no projeto B sem sinalização.
- **R6.6** `profile:`, quando presente, deve constar nos `profiles` do agente autor. **Identidade em dois níveis:** `agent` = runtime — quem executa, zona de escrita, accountability (a única fronteira que o kernel consegue *impor*, já que perfis do mesmo runtime compartilham processo e credenciais); `profile` = persona — metadado de proveniência e filtro. Por isso `Episodic/` é por agente, nunca por perfil.
- **R6.7 (v0.2.7)** `confidence:`, quando presente, deve pertencer ao enum `high|medium|low` — os filtros de retrieval e o item 4b do gate comparam contra esse conjunto fechado, então um valor fora dele seria memória que a busca não consegue ranquear. Ausente, o kernel grava `medium`.

---

## 7. Contrato MCP

O kernel expõe um servidor MCP (`mechabrain serve`). Ferramentas (na wire os nomes usam underscore — `memory_search` — porque o charset de nome de tool do MCP não aceita ponto):

### 7.1 Leitura

- **`memory.search(query, k=8, filters?, expand_links?)`** → busca híbrida (vetorial + BM25, pesos do manifest) com Contextual Retrieval. `filters`: `{type, agent, profile, scope, tags, status, min_confidence}`. Recomendação normativa aos agentes (codificada no `AGENTS.md`): filtrar por `scope` do trabalho corrente — um hit de outro escopo é contexto, não verdade local.
  **Expansão por links ("graph-lite")**: com `expand_links: N` (default e teto no manifest, `retrieval.link_expansion`), os top-k seeds são expandidos N saltos pelo **grafo autorado** — wikilinks do corpo, `supersedes` e relações de `memory.link` — e o conjunto expandido é rerankeado antes de retornar. Hits alcançados por expansão carregam a cadeia de proveniência (`via: [[seed]] → [[intermediária]]`). O grafo consultado é o que humanos e agentes escreveram nas notas — **nunca** um grafo extraído por LLM na ingestão: em corpus autorado, extração automática só adiciona ruído e custo.
  Retorno (por hit): `{id, path, wikilink, title, type, agent, confidence, score, excerpt, created}`.
  **R7.1** Proveniência é obrigatória: `path` e `wikilink` sempre presentes, para o agente citar a fonte na resposta (P7).
- **`memory.get(id | wikilink)`** → nota completa (frontmatter + corpo).
- **`memory.status()`** → saúde do índice, contagens por tipo, data da última consolidação.

### 7.2 Escrita

- **`memory.write(type, content, meta)`** → executa o **gate de escrita** (§8.2); se aprovado, resolve o nome via template do manifest, grava o `.md` no path correto e reindexa incrementalmente. Retorna `{path, id}` ou `{rejected, reason, near_duplicates[]}`.
- **`memory.propose(target_path, proposed_change, rationale)`** → formaliza o fluxo de proposta: grava nota em `zones.proposals_dir` usando `proposal_name`, com diff/sugestão + justificativa. É a **única** via para um agente afetar nota fora de `mecha-brain/`.
- **`memory.link(a, b, relation?)`** → registra relação entre notas. Alimenta diretamente a expansão por links do `memory.search` (§7.1) — quanto melhor os agentes linkarem, melhor o recall multi-hop: o grafo melhora por curadoria, não por extração. **(v0.2.x)** As arestas persistem em `_meta/links.jsonl`, versionado com a vault: aresta autorada é fonte-da-verdade (P1), não estado derivado.

### 7.3 Rastreio de acesso sem ruído no git

`last_accessed` no frontmatter alimentaria o decay, mas atualizá-lo a cada busca geraria commits infinitos (vaults com auto-commit). Solução normativa:

- **R7.2** Acessos são registrados em `_meta/index/access.jsonl` (**gitignored**, por máquina) no momento do `search`/`get`.
- **R7.3** O job de consolidação (§9) agrega os logs de acesso e só então atualiza `last_accessed` no frontmatter — um commit por ciclo de manutenção, não por consulta.

### 7.4 Concorrência e modelo de consistência

- **R7.4** **Um único escritor por máquina**: `mechabrain serve` roda como **daemon local** (HTTP/SSE em porta local) e os clientes MCP dos agentes apontam para ele. Modo stdio-por-sessão (um processo de kernel por cliente) é proibido como default — várias sessões simultâneas seriam múltiplos escritores no mesmo índice LanceDB/SQLite, corrompendo-o. Fallback mínimo sem daemon: lock de arquivo no índice.
- **R7.5** Toda escrita de `.md` é atômica (tempfile + rename no mesmo filesystem).
- **R7.6** Entre máquinas, o modelo é **consistência eventual via git da vault**: o gate de dedup consulta o índice *local*, então quase-duplicatas podem nascer em máquinas diferentes entre syncs — o job de consolidação (§9) as reconcilia depois. Trade-off aceito para escala pessoal; `Episodic/` append-only por agente mantém conflitos de merge raros.

---

## 8. Política de escrita e ciclo de decisão (CoALA §4.6)

Memórias e retrieval definem o que os agentes *podem* acessar; esta seção define **quando e o que escrever** — o estágio de decisão do CoALA (§4.6) aplicado ao learning: roteamento explícito, gate de avaliação antes da escrita e dono claro para cada superfície.

### 8.1 Roteamento — "para onde vai isto?"

Árvore de decisão (pare na primeira regra que casar):

1. É config/segredo/estado de máquina? → **camada runtime/L1** (nunca entra na vault).
2. É comportamento ou modelo-do-usuário de um agente específico? → **store privado do agente** (`agents[].private_store`), se existir. Não é candidato ao Mecha-Brain.
3. É evento/registro do que aconteceu numa sessão? → **`Episodic/<agente>/`** (escrita direta, sem gate — é diário, não verdade).
4. É procedimento/how-to testado e reutilizável? → **`Procedural/`** (gate completo, §8.2).
5. É relatório longo de pesquisa? → **`Research/`** (se habilitado).
6. É fato/insight reutilizável e citável? → **`Semantic/`** (gate completo).
7. Requer mudar nota humana/fora do sandbox? → **`memory.propose`** (nunca escrita direta).
8. Nada acima? → não escreve. Fica no contexto da sessão.

Em qualquer escrita (passos 3–6), o agente declara **`scope:`** — o slug do projeto a que a memória pertence, ou `global`. Na dúvida entre projeto e global, prefira o projeto: **promover a global é decisão de consolidação, não de escrita**.

### 8.2 Gate de escrita (para `Semantic/` e `Procedural/`)

> **Enforced × instruído (fidelidade do kernel):** o kernel não chama LLM, então só policia o **mecanicamente verificável** — itens **2, 4, 5, 6, 7** são *rejeição*. Os itens de **julgamento — 1 (reutilizável) e 3 (atômico)** — não podem ser policiados por código sem fingir; voltam como **warning** e são instruídos no `AGENTS.md`, responsabilidade do agente. Fingir enforcement com um booleano que o agente sempre marca `true` seria teatro.
>
> **Elevação opt-in (v0.2.0):** `gate.reject_on` no manifest eleva a *rejeição* warnings cuja condição é fato mecânico. Hoje só `confidence_unverified` qualifica — "`confidence: high` sem `meta.evidence` ao lado" é checável, mesmo que "a fonte é primária?" não seja. `reusable` e `atomic` seguem ineleváveis: elevar julgamento/heurística seria exatamente o teatro que este bloco proíbe.

Checklist que o `memory.write` aplica (e o `AGENTS.md` codifica em linguagem imperativa) — na terminologia CoALA, o estágio de *evaluation* antes do *learning action*:

1. **Reutilizável?** Vale além da sessão atual? Se não → rejeitar.
2. **Já existe?** `memory.search` interno obrigatório; similaridade acima de `dedup_similarity` **no mesmo escopo** → retornar candidatos e exigir decisão explícita: `supersedes`, fusão, ou desistir. Escrita cega de quase-duplicata é rejeitada.
3. **Atômico?** Um insight por nota (Zettelkasten). Vários → dividir.
4. **Fonte declarada?** `source:` preenchido; `confidence: high` só com verificação ou fonte primária.
5. **Escopado?** `scope:` declarado e correto — fato de projeto gravado como `global` é contaminação cruzada em potencial (R6.5).
6. **Procedural: testado?** `PROC` exige que o procedimento tenha sido executado com sucesso ao menos uma vez, com a evidência citada no corpo. (CoALA: escrever em memória procedural é a forma mais arriscada de learning — um playbook ruim propaga erro para todos os agentes.)
7. **Limpo?** Sem chaves/tags das denylists (R6.1).

### 8.3 Fronteira com stores privados

O kernel gerencia **apenas** o store compartilhado. Stores privados de agentes (memória associativa interna, perfis) ficam fora, por design — e podem existir **por perfil** (uma memória comportamental por persona do mesmo runtime); para o kernel seguem externos e meramente declarados no registry. A ponte é unidirecional e curada: conhecimento *geral* que nasce num store privado é **promovido** a `Semantic/` via `memory.write` normal (passa pelo gate como qualquer escrita). O comportamental nunca é promovido.

### 8.4 `hot.md` é blackboard, não working memory

Working memory, no CoALA, é por agente e por ciclo de decisão — é o próprio contexto do LLM, e não é papel do kernel. O `hot.md` é outra coisa: um **blackboard compartilhado** sempre-carregado (foco atual, ponteiros para o que importa agora).

- **R8.1** Só o **consolidador** (job do §9) escreve `hot.md`, `index.md` e `indices/`. Agentes leem.
- **R8.2** `hot.md` é organizado em **seções por escopo ativo** (escopo com escrita/acesso recente), renderizadas pelo consolidador, com teto por seção (sugestão: ~15 linhas) — "o foco atual" não é um só quando há múltiplos projetos. É cache de atenção, não arquivo de memória: tudo nele aponta para notas reais. **(v0.2.5)** A janela de atenção é configurável (`maintenance.hot_days`): só entra no board memória tocada dentro dela.

---

## 9. Manutenção: consolidação, decay, deprecação

Job periódico (`mechabrain consolidate`), agendável por qualquer mecanismo externo (cron, skill de schedule de um agente — a *agenda* é deployment; o *pipeline* é kernel):

1. **Flush de acessos** — agrega `access.jsonl` → atualiza `last_accessed` (R7.3).
2. **Dedup semântico** — pares acima de `dedup_similarity` **dentro do mesmo `scope`**. A *fusão* preserva detalhe (nunca super-resumir — clustering+resumo derruba acurácia), mas **fundir prosa é julgamento e o kernel não chama LLM**: ele **detecta e reporta** os candidatos no relatório da consolidação, e a fusão é executada por um **agente** via `memory_write` com `supersedes` (que arquiva a substituída, P8). É a divisão CoALA §6 — código para o determinístico, LLM para o julgamento. **Nunca fundir através de escopos** — semelhança textual entre projetos é justamente a distinção que importa; pares cross-scope similares vão numa lista separada do relatório, nunca fundidos.
3. **Decay Ebbinghaus** — sem acesso há `decay_days` → `status: arquivado`. Nunca deletar (P8). Arquivadas saem do `index.md` e perdem peso no retrieval, mas continuam buscáveis com filtro explícito.
4. **Deprecação procedural** — `PROC` substituído (`supersedes` de um sucessor) → `status: deprecado`, com link para o sucessor. **Relatório de staleness (v0.2.0):** `PROC` ativo sem teste há mais de `proc_stale_days` (medido por `last_tested:`, senão `created:` — leitura não é teste) é **listado** no relatório para um agente retestar. Detect-and-report como o dedup: o kernel nunca decide se um playbook envelheceu — só mede o calendário. *Nota de fidelidade:* "contradito por execução mais recente" é julgamento — detectar contradição exige entender os dois textos —, então essa metade é dos agentes (via reteste do relatório), nunca do kernel.
5. **Rebuild** — reindexação **incremental** (diff por mtime+hash: só as notas que o ciclo tocou ou que um humano editou são re-embeddadas; upgrade automático para rebuild completo se o fingerprint do deployment mudou — modelo de embedding, store, corpus) + regeneração de `index.md`, `indices/` e `hot.md`. O rebuild é **atômico**: interrompido no meio (OOM, kill), o índice anterior permanece íntegro — o lado lexical roda clear+upserts numa única transação SQLite e o vector store só persiste ao final. Se `index.md` exceder ~200 linhas, shardar por escopo (`indices/<scope>.md`) e reduzir `index.md` a índice-mestre magro: uma linha por escopo + as memórias `global`.
6. **Commit** — um único commit com `maintenance.commit_prefix`.

---

## 10. Instalação e ciclo de vida

CLI do kernel (o passo a passo completo está no [guia de setup](SETUP.md)):

| Comando | Efeito |
|---|---|
| `mechabrain init <vault>` | Cria o skeleton (§3), escreve manifest default, adiciona `mecha-brain/_meta/index/` ao `.gitignore` da vault, gera `AGENTS.md` e `schema.md`, imprime o snippet de integração para o `CLAUDE.md`/instruções da vault. **Idempotente** — rodar de novo não destrói nada (nem ressuscita `Research/` desabilitada no manifest, v0.2.7). |
| `mechabrain sync` | Regenera artefatos derivados do manifest (`AGENTS.md`, `schema.md`, subpastas de `Episodic/` para agentes novos) após edição do config. |
| `mechabrain serve` | Sobe o servidor MCP (descoberta da vault via R4.3). |
| `mechabrain reindex [--full]` | Reconstrói o índice derivado. |
| `mechabrain consolidate` | Roda o pipeline do §9. |
| `mechabrain check` | Lint do deployment: manifest válido, denylists respeitadas em todas as notas, ausência de caminhos absolutos, `.gitignore` correto. |

**AGENTS.md gerado, não escrito à mão:** o arquivo tem um **bloco gerenciado** (delimitado por marcadores `<!-- mechabrain:begin -->` / `<!-- mechabrain:end -->`) que o kernel regenera a partir de template + manifest — fronteiras, paths, denylists, registry de agentes, checklist do gate. Fora do bloco, o humano pode adicionar seções livres que sobrevivem ao `sync`. Isso elimina o drift doc-vs-config: documentação de fronteiras mantida à mão diverge do config com o tempo.

**Integração de agentes** é sempre via MCP — nenhum agente precisa conhecer paths internos do `mecha-brain/`. Um agente que hoje grava arquivo direto (ex.: um script de pesquisa) é adaptado para chamar `memory.write(type="research", ...)`.

---

## 11. Subsistemas Classe B (plugins de escrita viva)

Subsistemas que **escrevem em notas humanas vivas** (tutores que gravam scores em notas de estudo, trackers que atualizam frontmatter de projetos) são, em termos CoALA, *grounding actions* — não learning — e são intrinsecamente específicos da vault hospedeira. Portanto:

- **R11.1** Classe B fica **fora do kernel**. Cada subsistema é um plugin com manifest próprio de zonas: `{ zones: [{path, fields_owned}], policy: never-overwrite-user-fields }`.
- **R11.2** Um plugin Classe B pode *ler* o Mecha-Brain via MCP como qualquer agente, e *escrever conhecimento* nele via `memory.write` — mas sua escrita em notas vivas é governada pelo seu próprio manifest, não por esta spec.
- A spec do formato de manifest de zonas Classe B é trabalho futuro (v0.2); a v0.1 apenas reserva o conceito.
- **(v0.2.x)** O kernel *distribui* um artefato Classe B como conveniência de deploy: a skill **Mecha-Scribe** (`.claude/skills/mecha-scribe/SKILL.md`, instalada por `init`/`sync` a partir de um template agnóstico). A skill não ganha zona de escrita própria — toda escrita dela passa por `memory_write`/`memory_propose` e pelo template padrão da vault —, então P4 permanece intacto; o que o kernel carrega é o *texto da rotina*, não um privilégio.

---

## 12. Teste de portabilidade (teste de mesa)

Antes de qualquer release da spec ou do kernel:

1. `grep -rE "/home/|/Users/|[A-Z]:\\\\" <kernel>` → vazio (R4.1, R4.2).
2. Nenhum nome de agente, vault, plugin de Obsidian ou pasta humana no kernel ou nesta spec → se aparecer, pertence ao manifest/deploy.
3. `mechabrain init` numa vault descartável vazia → `check` passa, `serve` sobe, `memory.write` + `memory.search` funcionam sem editar nada além do manifest.
4. Mover a vault de lugar (ou de máquina/SO) → tudo continua funcionando após `reindex` (nenhum estado depende de path absoluto ou symlink).

---

## 13. Fora de escopo da v0.1

- GraphRAG completo (extração de entidades por LLM + community summaries; custo ~1.4x ingest / ~1.8x storage). O multi-hop já é coberto pela expansão por links (§7.1). **Critério para reavaliar na v0.2** — o formato das consultas reais, não a tecnologia: perguntas globais/sensemaking frequentes ("sintetize tudo que aprendemos sobre X") que `index.md`/`indices/` + consolidação não respondam bem. Mesmo então, avaliar só a metade "global" (resumos hierárquicos por escopo); extração de entidades segue desnecessária num corpus autorado.
- Spec do manifest de zonas Classe B (§11, v0.2).
- Consistência forte multi-máquina — o modelo adotado é consistência eventual via git da vault (R7.6).
- Gerenciamento de stores privados de agentes (§8.3 — fronteira, por design).
