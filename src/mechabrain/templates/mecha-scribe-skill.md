---
name: mecha-scribe
description: Use when producing project documentation — a spec, plan, report, technical note, or any long-form explanation/doc — that belongs in the Obsidian vault rather than inside a code repository. Routes the doc to the right project's 02_Docs/ (or the project root for a high-level DOC_*), creates it from the vault's standard note template, and closes by recording an episodic memory in the mecha-brain that cites the doc. Long docs are content, not memory. Triggers: "document this in the vault", "write a spec/plan/report for <project>", "add docs for X", any request to write durable project documentation.
---

# Mecha-Scribe — documentar projetos NA vault

Subsistema Classe B de documentação (spec §11). Você (agente) produz documentação
longa **na vault** — nos diretórios de projeto, que já têm infra de docs — e **não**
nos repositórios de código, e **não** como memória no mecha-brain. O brain guarda só
o rastro (episódico) e o conhecimento destilado (INS).

Princípio: **documento longo é conteúdo, não memória.** Não escreva spec/plano/
relatório dentro de `mecha-brain/`.

## Fluxo (siga na ordem)

1. **ROTEAR** — o projeto do trabalho corrente define a zona:
   - `<Área>/<Projeto>/02_Docs/` para docs técnicos, specs, planos, relatórios;
   - a raiz do projeto só para `DOC_*` de alto nível;
   - subprojeto (padrão fractal) → `.../<Projeto>/<Subprojeto>/02_Docs/`.

2. **CRIAR a partir do template** — de duas formas, nesta ordem de preferência:
   - **Obsidian CLI + Templater (preferido):** com a CLI do Obsidian habilitada
     (Settings → General → Advanced), crie a nota pelo Templater a partir do template
     geral da vault (`.../Templates/1_Template_Nota_Geral`, ramo não-task). O Templater
     preenche frontmatter, seções e navegação. Como o template é interativo
     (prompts/suggester), responda os prompts com os valores da rota (título, tipo
     DOC/REL/TEC, projeto).
   - **Replicar o resultado (fallback headless):** sem a CLI (outra máquina, sessão
     sem GUI), replique a **saída** do template — mesmo frontmatter, mesmas seções,
     mesma navegação. Frontmatter mínimo:
     ```yaml
     ---
     title: "<título descritivo>"
     tags: [<projeto/*>, tipo/documentacao, <area/*>, status/ativo]   # só tags oficiais
     created: YYYY-MM-DD
     modified: YYYY-MM-DD
     project: "[[DASHBOARD_<Projeto>|<Projeto>]]"
     code: "DOC"       # ou REL / TEC
     status: ativo
     dg-publish: <conforme a política da área>
     ---
     ```

3. **NOMEAR** — `YYYY-MM-DD_DOC_Nome.md` (ou `REL`/`TEC` conforme o tipo).

4. **GRAVAR** — **criar apenas; NUNCA sobrescrever** conteúdo existente. Editar um doc
   que já existe, um `MOC_*` ou um `DASHBOARD_*` NÃO é escrita direta → use
   `memory_propose` (a proposta vai para o inbox para um humano decidir). MOC/DASHBOARD
   só por pedido humano explícito.

5. **FECHAR** — registre um episódico no mecha-brain citando o doc por wikilink:
   `memory_write(type=episodic, scope=<projeto>)`, corpo curto com o que foi
   documentado e `[[<doc criado>]]`.

6. **DESTILAR** — se saiu insight reutilizável, vire `memory_write(type=semantic)` (INS)
   com `meta.evidence` (o gate strict exige evidência para `confidence: high`).

## Regras transversais

- Nunca inventar tag fora do Sistema de Tags da vault.
- **never-overwrite**: conteúdo existente nunca é alterado direto; mudança → `memory_propose`.
- O contrato compartilhado vive no brain como PROC (`memory_search "como documentar
  projetos"`); esta skill é a rotina executável dele.

## Verificação

O doc aparece na query de docs do `DASHBOARD_<Projeto>` (por path + `tipo/documentacao`);
os wikilinks resolvem; o episódico aparece no `memory_search` e no painel de atividade
da dashboard do mecha-brain. A consolidação (Fase 2) reporta se um doc passar a citar
memória morta ou tiver link quebrado — corrija via `memory_propose`.
