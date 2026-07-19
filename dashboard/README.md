# Dashboard de observabilidade do mecha-brain

Painel para **visualizar a memória agentica** de um deployment do `mechabrain` dentro
do Obsidian: as memórias como cards, o grafo de relações (2D/3D), o blackboard
(`hot.md`), o inbox de propostas e o **log de ações do kernel** (`actions.jsonl`,
v0.2.1) — aceitas, rejeitadas, promoções, propostas e links.

Os widgets são [Datacore](https://github.com/blacksmithgu/datacore) JSX puros (sem
build, sem CDN). Este diretório é um **snapshot autocontido** dos widgets em uso na
vault de referência (`megabrain`): traz a biblioteca de componentes inteira (Glo* +
theme provider) para rodar sem depender de mais nada.

## Requisitos

- Obsidian com o plugin **Datacore** habilitado.
- Um deployment do `mechabrain` na vault (diretório `mecha-brain/` — ver
  [`../README.md`](../README.md) e o comando `mechabrain init`).

## Instalação

Copie a árvore `System/Scripts/` deste diretório para a raiz da sua vault (se já
existir uma pasta `System/Scripts/`, mescle — os nomes de arquivo não colidem fora
da própria biblioteca):

```bash
cp -r dashboard/System/Scripts/* /caminho/da/vault/System/Scripts/
```

Crie uma nota (ex.: `DASHBOARD_Mechabrain.md`) com apenas este corpo:

````markdown
```datacorejsx
const script = await dc.require(dc.fileLink("System/Scripts/Widgets/dc-mechabrainDashboard.jsx"));
return function View() { return script.Func(); }
```
````

Os widgets leem os caminhos padrão do deployment (`mecha-brain/`, `System/Inbox/`).
Se o seu manifest usa outro `zones.proposals_dir`, ajuste as constantes no topo de
`dc-mechabrainDashboard.jsx`.

## Conteúdo

| Caminho | Papel |
|---|---|
| `System/Scripts/Widgets/dc-mechabrainDashboard.jsx` | Widget master: stats, grafo, blackboard, inbox, atividade do kernel, manutenção |
| `System/Scripts/Widgets/dc-memoryBrowser.jsx` | Browser de memórias em cards (com trecho do texto) + painel de manutenção |
| `System/Scripts/Componentes/dc-glo*.jsx` | Biblioteca de componentes (GloCard/Badge/Button/Stat/Graph/Graph3D/…) |
| `System/Scripts/Core/dc-themeProvider.jsx` | Provedor de tema — os widgets seguem 100% as cores do tema da vault |

O grafo 3D porta o funcionamento do mapa de memória (núcleo→hubs→nós, halos,
flutuação, opacidade por confiança, anel de frescor) e se auto-enquadra no primeiro
render. Toda a animação é imperativa (rAF escrevendo no DOM via refs, auto-pausa
quando oculto) para não pesar.

## Notas

- Nada aqui entra em decisão do kernel — a dashboard **só lê**. O `actions.jsonl` é
  camada de runtime (por máquina, gitignored); um deployment novo começa com o painel
  de atividade vazio até a primeira escrita/proposta/link via MCP.
- Snapshot mantido em sincronia manual com a vault de referência; a fonte viva dos
  widgets é a vault.
