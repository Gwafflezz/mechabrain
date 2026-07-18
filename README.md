# Mecha-Brain

**Memória agentica *drop-in* para qualquer vault Markdown.**

O Mecha-Brain dá a um conjunto arbitrário de agentes LLM os quatro tipos de memória do
[CoALA](https://arxiv.org/abs/2309.02427) (working, episódica, semântica, procedural), recall
semântico compartilhado e um ciclo de decisão explícito para escrita — sem descaracterizar o PKM
humano da vault hospedeira.

Ele é composto de duas partes:

1. **Uma pasta contratual** (`mecha-brain/`) instalada na raiz da vault — a área onde os agentes
   podem escrever, versionada junto com a vault.
2. **Um kernel** (este repositório): CLI `mechabrain` + servidor MCP, instalado **fora** da vault,
   que serve busca híbrida, escrita governada e manutenção sobre essa pasta.

**Teste de aceitação:** `mechabrain init <qualquer-vault>` + editar um `config.yaml` deve ser *tudo*
que é preciso para portar o sistema. Se um passo de instalação exigir editar código, o projeto
falhou.

---

## Instalação

```bash
uv tool install mechabrain            # núcleo (pyyaml, numpy, mcp)
uv tool install "mechabrain[embed]"   # + sentence-transformers (embeddings reais)
uv tool install "mechabrain[all]"     # + lancedb + sqlite-vec
```

O kernel exige Python >= 3.11. Ele mora fora da vault por design: código é software versionado com
releases, não conteúdo sincronizado por nuvem.

```bash
cd /caminho/da/sua/vault
mechabrain init .
```

O `init` cria o esqueleto, escreve um `config.yaml` default, adiciona `mecha-brain/_meta/index/` ao
`.gitignore` da vault, gera o `AGENTS.md` e o `schema.md`, e imprime o snippet de integração para as
instruções da sua vault. É **idempotente**: rodar de novo não destrói nada.

---

## As três camadas

Código, dados e estado de máquina têm ciclos de vida diferentes e nunca se misturam:

| Camada         | O quê                                                 | Onde vive               | Sincroniza?             |
| -------------- | ----------------------------------------------------- | ----------------------- | ----------------------- |
| **Kernel**     | CLI, serviço de memória, servidor MCP, templates       | instalado via `uv tool` | não (tem releases)      |
| **Deployment** | `mecha-brain/` inteiro: memórias, manifest, AGENTS.md  | dentro da vault         | sim, com o git da vault |
| **Runtime**    | índice vetorial (`_meta/index/`), portas, env, caches  | por máquina, gitignored | **nunca**               |

Consequências práticas, todas normativas:

- O kernel **não contém** nenhum caminho, nome de vault, nome de agente ou chave de frontmatter do
  seu deployment. Se ele precisa de um valor desses, o valor vem do manifest.
- O deployment **não contém** nenhum caminho absoluto — todo path do `config.yaml` é relativo à raiz
  da vault. Assim a vault sobrevive a mudar de pasta, de máquina e de sistema operacional.
- **Zero symlinks.** A vault é encontrada por convenção: argumento `--vault`, depois a env
  `MECHABRAIN_VAULT`, depois subindo a árvore a partir do diretório atual até achar
  `mecha-brain/_meta/config.yaml` — do jeito que o git acha o `.git`.

---

## A estrutura instalada

```
mecha-brain/
├── AGENTS.md            # contrato para agentes — GERADO do template + config
├── hot.md               # blackboard compartilhado — escrito só pelo consolidador
├── index.md             # MOC mestre, magro; sharda por escopo quando cresce
├── indices/             # índices por escopo: <scope>.md
├── Semantic/            # fatos/insights consolidados, atômicos, curados
├── Episodic/            # eventos/sessões — IMUTÁVEL, append-only, uma subpasta por agente
├── Procedural/          # playbooks/how-tos destilados, com deprecação
├── Research/            # (opcional) relatórios de pesquisa longos
├── _inbox/              # propostas de mudança em notas humanas
└── _meta/
    ├── config.yaml      # ★ O MANIFEST — a única casa de tudo específico do deployment
    ├── links.jsonl      # arestas autoradas (memory_link) — versionado
    ├── schema.md        # spec de frontmatter, gerada do manifest
    └── index/           # vetores/BM25/derivados — GITIGNORED, por máquina, rebuildável
```

Os **nomes** dessa árvore são o contrato e não são configuráveis. O **conteúdo** é todo governado
pelo `config.yaml`.

---

## Os comandos

| Comando                       | Efeito                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `mechabrain init <vault>`     | Cria o esqueleto, o manifest default, o `.gitignore`, o `AGENTS.md` e o `schema.md`. Idempotente.                    |
| `mechabrain sync`             | Regenera os artefatos derivados do manifest (`AGENTS.md`, `schema.md`, subpastas de `Episodic/` para agentes novos). |
| `mechabrain serve`            | Sobe o servidor MCP.                                                                                                |
| `mechabrain reindex [--full]` | Reconstrói o índice derivado. Sempre seguro: o índice deriva do Markdown, que é a fonte-da-verdade.                  |
| `mechabrain consolidate`      | Roda o pipeline de manutenção: flush de acessos, decay, deprecação, rebuild, commit — e o relatório de duplicatas.   |
| `mechabrain check`            | Lint do deployment: manifest válido, denylists respeitadas, ausência de caminhos absolutos, `.gitignore` correto.    |

Depois de editar o `config.yaml`, rode `mechabrain sync`. O `AGENTS.md` tem um **bloco gerenciado**
(entre `<!-- mechabrain:begin -->` e `<!-- mechabrain:end -->`) que o kernel regenera a partir do
manifest; o que você escrever fora do bloco sobrevive ao `sync`. Isso existe para eliminar o drift
entre documentação e config — fronteiras mantidas à mão divergem do `config.yaml` com o tempo.

---

## Ferramentas MCP

Os agentes falam com o Mecha-Brain só por MCP: nenhum agente precisa conhecer os paths internos da
pasta.

| Ferramenta       | O que faz                                                                          |
| ---------------- | ---------------------------------------------------------------------------------- |
| `memory_search`  | Busca híbrida (vetorial + BM25, pesos do manifest) com expansão opcional por links. |
| `memory_get`     | Nota completa por id ou wikilink.                                                   |
| `memory_status`  | Saúde do índice, contagens por tipo, data da última consolidação.                   |
| `memory_write`   | Escreve uma memória — passando pelo gate de escrita.                                |
| `memory_propose` | Propõe mudança em nota **fora** do sandbox. É a única via para isso.                |
| `memory_link`    | Registra uma relação entre duas notas; alimenta a expansão por links da busca.      |

> **Nomenclatura:** a spec descreve as ferramentas como `memory.search`, `memory.get` etc. O charset
> de nome de tool do MCP não aceita ponto, então o nome real na wire usa underscore
> (`memory_search`). A notação com ponto é o contrato conceitual; o underscore é o nome que você
> configura no cliente.

Todo resultado de busca carrega `path` e `wikilink` da fonte, para o agente citar de onde tirou a
informação. Memória citável é memória auditável.

---

## Limites por design

Esta seção é a parte honesta do README. **O kernel nunca chama um LLM.** Ele implementa o que é
mecanicamente verificável e *reporta* o resto — julgamento é dos agentes. Isso é o CoALA §6 levado a
sério (código para o determinístico, LLM para o julgamento), e tem consequências que você deve
conhecer antes de confiar no sistema:

**O gate de escrita só impõe metade do checklist.** Dos sete itens do gate, o kernel *impõe* cinco:
duplicata no mesmo escopo, `source:` preenchido, escopo válido, procedural com evidência, denylists.
Os outros dois — *"isto é reutilizável?"* e *"isto é atômico?"* — são julgamento, e código não
policia julgamento. Eles estão instruídos no `AGENTS.md` e voltam como **warnings**, nunca como
rejeição. Não fingimos enforcement com um booleano que o agente sempre marca `true`: um gate que
mente é pior que um gate ausente. A única exceção é opt-in e mecânica: `gate.reject_on:
[confidence_unverified]` no manifest eleva a rejeição o caso "`confidence: high` sem
`meta.evidence` ao lado" — a *condição* é um fato checável, mesmo que "a fonte é primária?" não
seja. `reusable` e `atomic` não são eleváveis, por design.

**A fusão de duplicatas não é automática.** O `consolidate` executa os passos mecânicos (flush de
acessos, decay, deprecação de procedural com sucessor, rebuild, commit). Mas *fundir* duas memórias
preservando detalhe exige entender as duas — então o kernel **detecta e reporta** os candidatos
(mesmo escopo, acima de `dedup_similarity`) num relatório, e a fusão é feita por um agente via
`memory_write` com `supersedes`. Pares **cross-scope** similares vão para uma lista separada do
relatório e **nunca** são fundidos: semelhança textual entre dois projetos é justamente a distinção
que importa.

**O contexto do Contextual Retrieval é determinístico, não gerado.** O prefixo prependido a cada
chunk antes de indexar é `scope + título + tags + caminho de headings` — não um resumo escrito por
LLM. O corpus é autorado e atômico; extração por LLM na ingestão adicionaria custo e ruído. Pela
mesma razão, a expansão multi-hop consulta o **grafo autorado** (wikilinks do corpo, `supersedes`,
arestas de `memory_link`) e nunca um grafo extraído automaticamente. O grafo melhora por curadoria.

**Consolidação nunca destrói.** Decay arquiva (`status: arquivado`), não deleta — notas arquivadas
saem do `index.md` e perdem peso no retrieval, mas continuam buscáveis com filtro explícito. Dedup
preserva detalhe e registra `supersedes`.

**Um escritor por máquina.** O `serve` roda como daemon local e os clientes MCP apontam para ele.
Uma sessão-por-processo seria vários escritores no mesmo índice, corrompendo-o. Sem daemon, o
fallback é lock de arquivo.

**Entre máquinas, a consistência é eventual, via o git da vault.** O gate de dedup consulta o índice
*local*, então quase-duplicatas podem nascer em máquinas diferentes entre syncs; o `consolidate` as
reconcilia depois. `Episodic/` ser append-only por agente mantém conflitos de merge raros.

**Escopo é uma fronteira, não uma sugestão.** Toda memória carrega `scope:`. Um fato verdadeiro no
projeto A não pode ser recuperado como verdade no projeto B sem sinalização. Na dúvida entre projeto
e global, prefira o projeto: promover a global é decisão de consolidação, não de escrita.

**Fora do escopo da v0.1:** GraphRAG completo; o manifest de zonas para subsistemas que escrevem em
notas humanas vivas; consistência forte multi-máquina; gerenciamento de stores privados de agentes
(fronteira deliberada — o kernel gerencia só o store compartilhado).

---

## Desenvolvimento

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
uv run pytest -q
```

Código, docstrings e mensagens de erro em inglês (o kernel é agnóstico e OSS); README e docs de
usuário em PT-BR.

## Licença

MIT — Davi Bezerra Barros.
