# Guia de setup — do zero ao Mecha-Brain rodando

Passo a passo prático para instalar, configurar e operar o Mecha-Brain numa vault
Markdown qualquer. A visão geral do sistema está no [README](../README.md); o
contrato normativo completo é a [Spec](SPEC.md); a governança de agentes que
trabalham **neste repositório** (não na sua vault) é o [CLAUDE.md](../CLAUDE.md).

O critério de sucesso deste guia é o teste de aceitação da
[Spec §1](SPEC.md#1-o-que-é): `mechabrain init <vault>` + editar um `config.yaml`
deve ser **tudo**. Se algum passo abaixo pedir para editar código, é bug — reporte.

---

## 0. Pré-requisitos

- **Python >= 3.11** na máquina (o kernel testa até 3.13).
- **[uv](https://docs.astral.sh/uv/)** ou `pipx` para instalar a CLI isolada.
- Uma **vault Markdown** (Obsidian ou não), de preferência versionada com git —
  o modelo de consistência entre máquinas é o git da vault
  ([Spec R7.6](SPEC.md#74-concorrência-e-modelo-de-consistência)).

## 1. Instale o kernel (fora da vault)

```bash
uv tool install mechabrain            # núcleo (pyyaml, numpy, mcp)
uv tool install "mechabrain[embed]"   # + sentence-transformers (embeddings reais)
uv tool install "mechabrain[all]"     # + lancedb + sqlite-vec (stores opcionais)
```

O kernel mora **fora** da vault por design: código é software com releases, não
conteúdo sincronizado por nuvem ([Spec §4](SPEC.md#4-as-três-camadas)). Confirme:

```bash
mechabrain --version
```

> Sem embeddings reais (`[embed]` ou um endpoint `http`), a busca vetorial não
> tem semântica — o provider `hash` existe só para teste/CI. Para uso de verdade,
> instale `[embed]` ou aponte `retrieval.embedding.provider: http` para um
> endpoint compatível.

## 2. Inicialize a vault

```bash
cd /caminho/da/sua/vault
mechabrain init .
```

O `init` cria a árvore contratual da [Spec §3](SPEC.md#3-contrato-fixo-estrutura-da-pasta)
e é **idempotente** — rodar de novo nunca destrói nada:

| Artefato | O que é |
|---|---|
| `mecha-brain/` + subpastas | o sandbox agent-writable (Semantic, Episodic, Procedural, Research, `_inbox`, `_meta`) |
| `mecha-brain/_meta/config.yaml` | **o manifest** — única casa do que é específico do seu deployment |
| `mecha-brain/AGENTS.md` | contrato para os agentes, gerado do [template](../src/mechabrain/templates/agents.md.tmpl) + manifest |
| `mecha-brain/_meta/schema.md` | schema de frontmatter legível, gerado do manifest (nunca edite à mão) |
| `mecha-brain/index.md`, `hot.md` | MOC mestre e blackboard — só o consolidador escreve neles depois |
| `.gitignore` da vault | ganha `mecha-brain/_meta/index/` (índice derivado, por máquina) |
| `.claude/skills/mecha-scribe/SKILL.md` | a skill do escriba, para documentação longa ir para a vault pelo gate certo |

Ao final, o `init` imprime um **snippet de integração** — guarde-o para o passo 6.

## 3. Configure o manifest

Abra `mecha-brain/_meta/config.yaml` e ajuste ao seu deployment. O arquivo é
autocomentado (gerado deste [template](../src/mechabrain/templates/config.yaml.tmpl));
a referência completa de cada chave é a [Spec §5](SPEC.md#5-o-manifest-_metaconfigyaml).
O mínimo que você vai querer tocar:

1. **`agents:`** — registre cada runtime que vai escrever memória (o default traz
   um agente `exemplo`; troque). O registry gera as subpastas de `Episodic/` e os
   valores válidos de `agent:`.
2. **`scopes:`** — liste os slugs dos seus projetos em `known:` (ou deixe `[]`
   para aceitar qualquer slug). Escopo é a fronteira contra contaminação cruzada
   entre projetos ([Spec R6.5](SPEC.md#6-schema-de-frontmatter-genérico)).
3. **`frontmatter.denylist_keys` / `denylist_tags`** — chaves e tags que agente
   nenhum pode usar (ex.: chaves do seu plugin de publicação, tags que disparam
   automações humanas).
4. **`retrieval.embedding`** — o modelo/provider de embedding da sua máquina.

Depois de **qualquer** edição no manifest:

```bash
mechabrain sync    # regenera AGENTS.md, schema.md e Episodic/<novos agentes>
```

## 4. Valide o deployment

```bash
mechabrain check
```

O `check` falha alto em manifest inválido, denylist violada, caminho absoluto,
`.gitignore` errado ou artefato gerado defasado ([Spec §10](SPEC.md#10-instalação-e-ciclo-de-vida)).
Rode-o sempre que mexer no manifest ou suspeitar de drift.

## 5. Suba o daemon MCP

```bash
mechabrain serve --vault /caminho/da/sua/vault
```

Por default o servidor sobe em `http://127.0.0.1:8765/sse` (porta via `--port` ou
`$MECHABRAIN_PORT` — porta é runtime, nunca vai no manifest). O modelo é **um
único escritor por máquina**: um daemon, vários clientes MCP apontando para ele
([Spec R7.4](SPEC.md#74-concorrência-e-modelo-de-consistência)). Existe um modo
`--stdio` para um único cliente, mas ele imprime um aviso explícito — várias
sessões stdio simultâneas seriam múltiplos escritores no mesmo índice.

A vault também pode ser descoberta sem `--vault`: via `$MECHABRAIN_VAULT`, ou
subindo a árvore a partir do diretório atual ([Spec R4.3](SPEC.md#4-as-três-camadas)).

## 6. Registre nos clientes MCP e nas instruções da vault

Com o daemon no ar, aponte cada cliente para o endpoint SSE. No Claude Code:

```bash
claude mcp add mechabrain --transport sse http://127.0.0.1:8765/sse
```

(Qualquer cliente MCP com suporte a SSE/HTTP serve — o nome das ferramentas na
wire é `memory_search`, `memory_get`, `memory_status`, `memory_write`,
`memory_propose`, `memory_link`; ver a
[tabela no README](../README.md#ferramentas-mcp).)

Por fim, cole o **snippet que o `init` imprimiu** no `CLAUDE.md`/instruções de
agente da sua vault. Ele aponta os agentes para o contrato real —
`mecha-brain/AGENTS.md` — e resume as quatro obrigações: buscar antes de agir,
citar a fonte (`path`/`wikilink`), filtrar por escopo e escrever só pelo gate.

## 7. Teste de fumaça

Com um cliente conectado, faça um ciclo completo:

1. `memory_status` → índice de pé, contagens zeradas.
2. `memory_write` de um episódico (`type: episodic`, com `agent:` do seu registry
   e `scope:`) → deve gravar em `mecha-brain/Episodic/<agente>/`.
3. `memory_search` pelo que acabou de escrever → o hit volta com `path` e
   `wikilink` (proveniência obrigatória, [Spec R7.1](SPEC.md#71-leitura)).

Se mexeu em notas por fora (edição humana, git pull), reindexe:

```bash
mechabrain reindex          # incremental
mechabrain reindex --full   # do zero (sempre seguro: o índice deriva do Markdown)
```

## 8. Agende a manutenção

O pipeline de consolidação ([Spec §9](SPEC.md#9-manutenção-consolidação-decay-deprecação))
— flush de acessos, relatório de duplicatas, decay, deprecação, rebuild, um
commit — roda sob demanda:

```bash
mechabrain consolidate            # o ciclo completo
mechabrain consolidate --dry-run  # só o relatório, sem tocar nada
```

Agende por qualquer mecanismo externo (a *agenda* é sua; o *pipeline* é do
kernel). Ex.: cron diário:

```cron
0 6 * * * MECHABRAIN_VAULT=/caminho/da/vault mechabrain consolidate
```

O relatório lista o que exige julgamento de um agente: candidatos a fusão
(mesmo escopo), pares similares cross-scope (nunca fundidos) e procedurais
velhos para retestar. O kernel detecta e reporta; quem decide é agente ou humano.

## 9. Segunda máquina / mudança de máquina

Nada de copiar índice, nada de symlink:

```bash
# na máquina nova
uv tool install "mechabrain[embed]"
git clone <sua-vault>              # ou o sync que você já usa
cd sua-vault && mechabrain reindex --full
mechabrain serve
```

O índice em `_meta/index/` é por máquina e rebuildável; a verdade viaja no git
da vault ([Spec §12](SPEC.md#12-teste-de-portabilidade-teste-de-mesa), item 4).

## 10. Atualizando o kernel

```bash
uv tool upgrade mechabrain
mechabrain sync && mechabrain check
```

O manifest declara `kernel_min_version`; um kernel velho demais para o seu
deployment recusa servir com erro claro ([Spec R4.5](SPEC.md#4-as-três-camadas)).

---

## Problemas comuns

| Sintoma | Causa provável | Saída |
|---|---|---|
| `vault not found` | nenhum `mecha-brain/_meta/config.yaml` no caminho de descoberta | passe `--vault`, exporte `MECHABRAIN_VAULT`, ou rode de dentro da vault |
| erro de manifest com "unknown key" | typo no `config.yaml` — o kernel nunca defaulta chave errada ([R5.1](SPEC.md#5-o-manifest-_metaconfigyaml)) | a mensagem sugere a grafia mais próxima; corrija e rode `mechabrain check` |
| `rerank: true` recusado | o kernel ainda não traz reranker; aceitar a flag sem honrá-la seria default silencioso | mantenha `rerank: false` |
| escrita rejeitada pelo gate | o retorno cita a regra (dedup, source, escopo, evidência, denylist) | é o gate funcionando — decida: `supersedes`, corrigir meta, ou desistir |
| `AGENTS.md`/`schema.md` "stale" no `check` | manifest editado sem regenerar | `mechabrain sync` |
| busca sem resultados após git pull | índice da máquina defasado | `mechabrain reindex` |
