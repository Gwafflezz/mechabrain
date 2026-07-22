# CLAUDE.md — governança do agente no repositório mechabrain

Guia para agentes (Claude e afins) que trabalham **neste repositório** — o *kernel* do
Mecha-Brain (CLI `mechabrain` + servidor MCP + templates). O contrato que vai para dentro
das vaults hospedeiras é gerado de `src/mechabrain/templates/agents.md.tmpl`; **este**
arquivo governa o trabalho sobre o código do kernel.

## Orientação rápida

- **O que é este repo:** o kernel, instalado **fora** da vault (via `uv tool`). Ele não
  contém nenhum caminho, nome de vault ou nome de agente de deployment — esses valores vêm
  sempre do manifest (`_meta/config.yaml`). Não codifique nada específico de deployment aqui.
- **Fontes de verdade:** o contrato normativo é [docs/SPEC.md](docs/SPEC.md) (casa canônica
  da spec); o passo a passo de operação é [docs/SETUP.md](docs/SETUP.md). Mudança de
  comportamento do kernel que contradiga a spec exige emenda marcada nela, no mesmo commit.
- **Idioma:** código, docstrings e mensagens de erro em inglês; README e docs de usuário em
  PT-BR.
- **Dev:**
  ```bash
  uv venv --python 3.13
  uv pip install -e ".[dev]"
  uv run pytest -q      # testes
  uv run ruff check .   # lint
  ```

---

## Pilar 1 — Restrição de ferramentas e isolamento de escrita (sandboxing)

O agente de desenvolvimento opera com **privilégio mínimo de ferramentas**.

- **Habilitado:**
  - modificação de **código-fonte** do kernel (ferramentas de arquivo dentro deste repo:
    `src/`, `tests/`, `dashboard/`, configs do repo);
  - ferramentas MCP do Mecha-Brain — principalmente `memory_search` (recall) e
    `memory_write` (registro governado).
- **Bloqueio estrito:** é **proibido** usar ferramentas genéricas de arquivo
  (`Edit`/`Write`/`Bash` com redirecionamento etc.) para editar a árvore da vault do
  Mecha-Brain — em especial `mecha-brain/` e qualquer `02_Docs/`. Essas pastas não são
  editáveis à mão a partir daqui.
- **Única via de escrita na memória:** `memory_write`. Nada de escrever arquivo `.md` direto
  no vault. Para mudar algo **fora** do sandbox (nota humana, doc, MOC, dashboard) a via é
  `memory_propose` — a proposta vai para o inbox e um humano decide. Nunca escrita direta.

Racional: a fronteira de escrita é a única que o kernel impõe de fato. Escrever memória por
fora do gate contamina o índice e some da auditoria (memória sem proveniência não é
auditável).

---

## Pilar 2 — Protocolo de handoff automático (roteamento sistêmico)

Documentação longa é **conteúdo, não memória** — não vai para dentro de `mecha-brain/`, e o
agente de desenvolvimento **não a redige à mão**.

**Matriz de roteamento de intenção:**

| Intenção do usuário | Classe | Ação do agente |
|---|---|---|
| Alterar/ler código, corrigir bug, refatorar | Trabalho no kernel | executa aqui, com as ferramentas do Pilar 1 |
| Fato/insight/procedimento reutilizável e citável | Memória | `memory_write` (passa pelo gate) |
| Doc longo: **spec, plano, relatório, nota técnica, explicação** | **Classe B** | **delega — NÃO redige manualmente** |

Para Classe B, **delegue ao Mecha-Scribe** (a rotina executável é a skill `mecha-scribe`):
repasse o contexto e deixe o escriba fazer o registro **passando pelo gate de escrita
correto** — ele roteia o doc para o `02_Docs/` do projeto certo, cria a nota do template
padrão da vault (nunca sobrescreve) e fecha com um episódico que cita o doc por wikilink.

> Observação de ecossistema: num setup multiagente o handoff de Classe B é assumido pelo
> agente escriba (p.ex. "Feynman") via delegação; neste repositório o mecanismo concreto e
> disponível é a skill `mecha-scribe`. Delegue por ela. O agente de desenvolvimento não
> escreve spec/plano/relatório na mão.

---

## Pilar 3 — Ciclo de vida nativo com pós-hooks de execução

Todo turno substantivo segue o ciclo:

```
[Início do Turno] -> [Implementação] -> [Validação por Teste] -> [Escrita de Memória] -> [Entrega]
```

**Regra de pós-hook (encerramento de turno):** antes de finalizar a resposta e entregar o
controle, **havendo decisão técnica ou alteração de estado**, rode `memory_write` para
registrar o aprendizado/episódio no Mecha-Brain.

- Decisão com razão (por que A e não B), insight não-óbvio reutilizável, ou procedimento
  testado → registre (`type=semantic`/`procedural`, com evidência quando exigido pelo gate).
- Registro de uma sessão que outro agente — ou você amanhã — precisaria para retomar →
  `type=episodic`, ligado por `memory_link` aos insights que gerou.
- **Não** registre rotina, dump de sessão "por via das dúvidas", ou o que já se recupera do
  código, do git ou de doc existente. Registro é decisão de julgamento, sob demanda — nunca
  despejo automático por sessão.

Sempre declare `scope:` na escrita; na dúvida entre projeto e global, prefira o projeto.
