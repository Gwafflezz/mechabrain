/* ==================================================================
   MECHABRAIN DASHBOARD (master widget)
   Painel completo da memória agentica, tudo visível — sem abas nem
   seções colapsadas:
     1. Linha de stats (GloStat)
     2. Grafo de relações (GloGraph): wikilinks + supersedes + memory_link
     3. Blackboard (hot.md) renderizado por escopo
     4. Inbox de propostas com aceitar/rejeitar (decisão HUMANA —
        aceitar copia a mudança p/ o clipboard e abre a nota alvo; o
        arquivo alvo nunca é alterado automaticamente)
     5. Painel da última consolidação (§9)
     6. Browser de memórias com filtros
   - Usage:
       const s = await dc.require(dc.fileLink("System/Scripts/Widgets/dc-mechabrainDashboard.jsx"));
       return function View() { return s.Func(); }
   ================================================================== */

const { useTheme, deriveChartColors } = await dc.require(dc.fileLink("System/Scripts/Core/dc-themeProvider.jsx"));
const { GloCard } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloCard.jsx"));
const { GloBadge, GloBadgeGroup } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloBadge.jsx"));
const { GloButton } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloButton.jsx"));
const { GloStat, GloStatGroup } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloStat.jsx"));
const { GloGraph } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloGraph.jsx"));
const { GloGraph3D } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloGraph3d.jsx"));
const { MemoryBrowser, MaintenancePanel } = await dc.require(dc.fileLink("System/Scripts/Widgets/dc-memoryBrowser.jsx"));

const ROOT = "mecha-brain";
const HOT_PATH = `${ROOT}/hot.md`;
const LINKS_PATH = `${ROOT}/_meta/links.jsonl`;
const REPORT_PATH = `${ROOT}/_meta/index/consolidation-report.json`;
const ACTIONS_PATH = `${ROOT}/_meta/index/actions.jsonl`;
const INBOX_PATH = "System/Inbox";

const TYPE_META = {
    Semantic: { id: "semantic", label: "Semânticas", chart: "chart-color-5" },
    Episodic: { id: "episodic", label: "Episódicas", chart: "chart-color-2" },
    Procedural: { id: "procedural", label: "Procedurais", chart: "chart-color-3" },
    Research: { id: "research", label: "Research", chart: "chart-color-1" },
};

// Cores 100% do tema (como a Home): chart-colors p/ os tipos; accent p/ links.
// deriveChartColors muta o objeto — clonamos p/ não tocar o theme do provider.
function themePalette(theme) {
    const t = deriveChartColors({ ...(theme || {}) });
    const colors = {};
    for (const meta of Object.values(TYPE_META)) colors[meta.id] = t[meta.chart];
    return {
        type: colors,
        accent: t["color-accent"] || "var(--interactive-accent)",
        supersedes: t["chart-color-4"],
        memoryLink: t["chart-color-6"],
        faint: t["color-text-faint"] || "var(--text-faint)",
    };
}

// ── helpers (mesmo padrão defensivo dos outros widgets) ─────────────
const getValue = (p, key) => {
    if (typeof p.value === "function") {
        const val = p.value(key);
        if (val !== undefined && val !== null) return val;
    }
    if (p[key] !== undefined) return p[key];
    if (p.$frontmatter && p.$frontmatter[key] !== undefined) {
        const fm = p.$frontmatter[key];
        return fm && fm.value !== undefined ? fm.value : fm;
    }
    if (p.frontmatter && p.frontmatter[key] !== undefined) return p.frontmatter[key];
    return null;
};
const pathOf = (p) => p.$path || p.file?.path || p.path || "";
const openNote = (ref) => app.workspace.openLinkText(String(ref), "", false);
const typeOf = (path) => {
    const m = path.match(/^mecha-brain\/(Semantic|Episodic|Procedural|Research)\//);
    return m ? TYPE_META[m[1]] : null;
};
const idsFrom = (raw) => {
    // extrai ids de um valor supersedes: "[[id]]", lista, ou id cru
    const list = Array.isArray(raw) ? raw : raw ? [raw] : [];
    return list.flatMap((v) => {
        const s = String(v);
        const m = [...s.matchAll(/\[\[([^\]|#]+)/g)].map((x) => x[1].trim());
        return m.length ? m : s.trim() ? [s.trim()] : [];
    });
};
const stripFm = (text) => text.replace(/^---\n[\s\S]*?\n---\n?/, "");
// "YYYY-MM-DD" ou "YYYY-MM-DD HH:MM" quando há hora (≠ meia-noite):
// desde v0.2.6 created/modified carregam a hora de alteração.
const fmtStamp = (s) => {
    const m = String(s).match(/^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?/);
    if (!m) return String(s).slice(0, 10);
    return m[2] && m[2] !== "00:00" ? `${m[1]} ${m[2]}` : m[1];
};
const toDateStr = (val) => {
    if (!val) return "";
    if (typeof val === "string") return fmtStamp(val);
    if (typeof val.toFormat === "function") return fmtStamp(val.toFormat("yyyy-MM-dd HH:mm")); // luxon
    if (typeof val.toISOString === "function") return fmtStamp(val.toISOString());
    return fmtStamp(String(val));
};
// Frescor temporal [0..1] à la Hina: decaimento exponencial pela idade do
// último acesso (fallback: criação); meia-vida ~14 dias.
const freshnessOf = (lastAccessed, created) => {
    const ref = Date.parse(lastAccessed || created || "");
    if (Number.isNaN(ref)) return 0;
    const days = Math.max(0, (Date.now() - ref) / 86400000);
    return Math.exp(-days / 14);
};

// ═══════════════════════════════════════════════════════════════════
// Observabilidade: o que o kernel fez (actions.jsonl + consolidação)
// ═══════════════════════════════════════════════════════════════════
const ACTION_META = {
    write_accepted: { status: "success", label: "escrita aceita" },
    write_rejected: { status: "error", label: "escrita rejeitada" },
    proposal: { status: "info", label: "proposta" },
    link: { status: "neutral", label: "link autorado" },
    decayed: { status: "warning", label: "arquivada (decay)" },
    deprecated: { status: "warning", label: "deprecada" },
};
const fmtTs = (ts) => {
    const d = new Date(ts);
    return isNaN(d) ? String(ts).slice(0, 16) : `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};

function ActivityFeed() {
    const [events, setEvents] = dc.useState(null);
    const [stats, setStats] = dc.useState({ accepted: 0, rejected: 0 });

    dc.useEffect(() => {
        (async () => {
            const out = [];
            let accepted = 0, rejected = 0;
            // ações do serviço MCP (v0.2.1)
            try {
                if (await app.vault.adapter.exists(ACTIONS_PATH)) {
                    for (const line of (await app.vault.adapter.read(ACTIONS_PATH)).split("\n")) {
                        if (!line.trim()) continue;
                        try {
                            const e = JSON.parse(line);
                            if (e.action === "write_accepted") accepted++;
                            if (e.action === "write_rejected") rejected++;
                            out.push(e);
                        } catch (err) { /* linha rasgada */ }
                    }
                }
            } catch (e) { /* sem log ainda */ }
            // ações do consolidador (relatório §9)
            try {
                if (await app.vault.adapter.exists(REPORT_PATH)) {
                    const r = JSON.parse(await app.vault.adapter.read(REPORT_PATH));
                    for (const d of r.decayed || []) out.push({ ts: r.generated, action: "decayed", id: String(d.note || "").replace(/^\[\[|\]\]$/g, "") });
                    for (const d of r.deprecated || []) out.push({ ts: r.generated, action: "deprecated", id: String(d.note || "").replace(/^\[\[|\]\]$/g, "") });
                }
            } catch (e) { /* sem relatório */ }
            out.sort((a, b) => String(b.ts).localeCompare(String(a.ts)));
            setEvents(out.slice(0, 25));
            setStats({ accepted, rejected });
        })();
    }, []);

    if (events === null) return <p style={{ opacity: 0.5 }}>Lendo o log de ações…</p>;
    if (!events.length) return (
        <p style={{ opacity: 0.5 }}>
            Nenhuma ação registrada ainda — o log (<code>actions.jsonl</code>) começa a
            preencher na próxima escrita/proposta/link via MCP.
        </p>
    );

    const describe = (e) => {
        switch (e.action) {
            case "write_accepted": {
                const promo = e.type === "semantic" ? "promovida a Semantic/" : e.type ? `gravada em ${e.type}` : "gravada";
                const sup = (e.superseded || []).length ? ` · substituiu ${(e.superseded).length}` : "";
                return <span>
                    <a style={{ cursor: "pointer" }} onClick={() => openNote(e.id)}>{e.id}</a>
                    <span style={{ opacity: 0.6 }}> — {promo}{sup}</span>
                </span>;
            }
            case "write_rejected":
                return <span><strong>{e.title || "(sem título)"}</strong><span style={{ opacity: 0.6 }}> — {e.reason || "rejeitada pelo gate"}</span></span>;
            case "proposal":
                return <span>proposta <a style={{ cursor: "pointer" }} onClick={() => openNote(e.id)}>{e.id}</a><span style={{ opacity: 0.6 }}> → alvo {e.target}</span></span>;
            case "link":
                return <span><a style={{ cursor: "pointer" }} onClick={() => openNote(e.a)}>{e.a}</a><span style={{ opacity: 0.6 }}> ⟷ </span><a style={{ cursor: "pointer" }} onClick={() => openNote(e.b)}>{e.b}</a><span style={{ opacity: 0.6 }}> ({e.relation || "related"})</span></span>;
            case "decayed":
            case "deprecated":
                return <span><a style={{ cursor: "pointer" }} onClick={() => openNote(e.id)}>{e.id}</a><span style={{ opacity: 0.6 }}> — pelo consolidador</span></span>;
            default:
                return <span>{e.action}</span>;
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <GloBadgeGroup>
                <GloBadge variant="soft" status="success">{stats.accepted} aceitas</GloBadge>
                <GloBadge variant="soft" status={stats.rejected ? "error" : "neutral"}>{stats.rejected} rejeitadas</GloBadge>
            </GloBadgeGroup>
            <div>
                {events.map((e, i) => {
                    const meta = ACTION_META[e.action] || { status: "neutral", label: e.action };
                    return (
                        <div key={i} style={{ display: "flex", alignItems: "baseline", gap: "8px", padding: "4px 0", borderBottom: "1px solid var(--background-modifier-border)", fontSize: "0.85em" }}>
                            <GloBadge size="small" variant="soft" status={meta.status}>{meta.label}</GloBadge>
                            <span style={{ flex: 1, minWidth: 0 }}>{describe(e)}</span>
                            {e.agent && <GloBadge size="small" variant="outlined" status="neutral">{e.agent}</GloBadge>}
                            <span style={{ fontSize: "0.85em", opacity: 0.45, whiteSpace: "nowrap" }}>{fmtTs(e.ts)}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function MechabrainDashboard() {
    // Queries estreitas: cada uma só invalida quando a SUA pasta muda —
    // edições no resto da vault não re-renderizam a dashboard.
    const pages = dc.useQuery('@page and path("mecha-brain")');
    const inboxPages = dc.useQuery('@page and path("System/Inbox")');
    const { theme } = useTheme();
    const palette = dc.useMemo(() => themePalette(theme), [theme]);

    // ── memórias (sync, via índice do datacore) ─────────────────────
    const memories = dc.useMemo(() => {
        const out = [];
        for (const p of pages) {
            const path = pathOf(p);
            const type = typeOf(path);
            if (!type) continue;
            out.push({
                path,
                id: path.split("/").pop().replace(/\.md$/, ""),
                title: String(getValue(p, "title") || path.split("/").pop().replace(/\.md$/, "")),
                type,
                scope: String(getValue(p, "scope") || "global").trim(),
                status: String(getValue(p, "status") || "ativo").trim(),
                supersedes: idsFrom(getValue(p, "supersedes")),
                supersededBy: idsFrom(getValue(p, "superseded_by")),
                confidence: { high: 1, medium: 0.65, low: 0.35 }[String(getValue(p, "confidence") || "medium").trim().toLowerCase()] ?? 0.65,
                created: toDateStr(getValue(p, "created")),
                lastAccessed: toDateStr(getValue(p, "last_accessed")),
            });
        }
        return out;
    }, [pages]);

    // ── propostas no inbox ──────────────────────────────────────────
    const proposals = dc.useMemo(() => {
        const out = [];
        for (const p of inboxPages) {
            const path = pathOf(p);
            if (!path.startsWith(INBOX_PATH + "/")) continue;
            const target = getValue(p, "target");
            if (!target) continue; // só notas de proposta do mecha-brain
            out.push({
                path,
                id: path.split("/").pop().replace(/\.md$/, ""),
                title: String(getValue(p, "title") || path.split("/").pop()),
                target: String(target),
                agent: String(getValue(p, "agent") || ""),
                status: String(getValue(p, "status") || "ativo").trim(),
                created: String(getValue(p, "created") || "").slice(0, 10),
            });
        }
        out.sort((a, b) => b.created.localeCompare(a.created));
        return out;
    }, [inboxPages]);
    const pending = proposals.filter((p) => p.status === "ativo");
    const decided = proposals.length - pending.length;

    // ── grafo: wikilinks dos corpos + supersedes + links.jsonl ──────
    const [graph, setGraph] = dc.useState(null);
    const memKey = memories.map((m) => m.id).join("|");
    dc.useEffect(() => {
        (async () => {
            const known = new Map(memories.map((m) => [m.id, m]));
            const edges = [];
            const seen = new Set();
            const addEdge = (from, to, style, opts = {}) => {
                const k = `${from}→${to}#${style}`;
                if (from === to || seen.has(k)) return;
                seen.add(k);
                edges.push({ from, to, style, ...opts });
            };
            const ghostRefs = new Map(); // id → contagem
            for (const m of memories) {
                // wikilinks do corpo (grafo autorado, §7.1)
                try {
                    const body = stripFm(await app.vault.adapter.read(m.path));
                    for (const match of body.matchAll(/\[\[([^\]|#]+)/g)) {
                        const t = match[1].trim();
                        if (!t) continue;
                        if (!known.has(t)) ghostRefs.set(t, (ghostRefs.get(t) || 0) + 1);
                        addEdge(m.id, t, "solid");
                    }
                } catch (e) { /* nota sumiu entre query e leitura */ }
                // supersedes (novo → antigo, direcionado)
                for (const t of m.supersedes) {
                    if (!known.has(t)) ghostRefs.set(t, (ghostRefs.get(t) || 0) + 1);
                    addEdge(m.id, t, "dashed", { directed: true, label: "supersedes", color: palette.supersedes });
                }
            }
            // arestas autoradas via memory_link
            try {
                if (await app.vault.adapter.exists(LINKS_PATH)) {
                    const lines = (await app.vault.adapter.read(LINKS_PATH)).split("\n").filter(Boolean);
                    for (const line of lines) {
                        try {
                            const e = JSON.parse(line);
                            const a = String(e.a || e.source || ""), b = String(e.b || e.target || "");
                            if (!a || !b) continue;
                            if (!known.has(a)) ghostRefs.set(a, (ghostRefs.get(a) || 0) + 1);
                            if (!known.has(b)) ghostRefs.set(b, (ghostRefs.get(b) || 0) + 1);
                            addEdge(a, b, "dotted", { label: e.relation || "related", color: palette.memoryLink });
                        } catch (err) { /* linha malformada */ }
                    }
                }
            } catch (e) { /* sem links.jsonl */ }

            // ── topologia núcleo→hubs→memórias (funcionamento do mapa da Hina):
            // core = a vault de memória; um hub por tipo; cada memória pendura
            // no hub do seu tipo. As arestas REAIS (wikilinks/supersedes/
            // memory_link) atravessam a estrutura por cima.
            const CORE = "__core__";
            const structural = [];
            const hubs = [];
            for (const [folder, meta] of Object.entries(TYPE_META)) {
                const members = memories.filter((m) => m.type.id === meta.id);
                if (!members.length) continue;
                const hubId = `__hub_${meta.id}__`;
                hubs.push({
                    id: hubId, label: `${meta.label} (${members.length})`,
                    color: palette.type[meta.id], size: 10.5, pin: "hub",
                });
                structural.push({ from: CORE, to: hubId, rest: 60 });
                for (const m of members) structural.push({ from: hubId, to: m.id, rest: 55 });
            }

            // ghosts referenciados (cap p/ layout legível)
            const ghosts = [...ghostRefs.entries()]
                .sort((a, b) => b[1] - a[1])
                .slice(0, Math.max(0, 70 - memories.length))
                .map(([gid]) => ({ id: gid, label: gid, ghost: true, size: 5 }));
            const ghostSet = new Set(ghosts.map((g) => g.id));

            const nodes = [
                { id: CORE, label: "mecha-brain", color: palette.accent, size: 15, pin: "core" },
                ...hubs,
                ...memories.map((m) => ({
                    id: m.id, label: m.title, color: palette.type[m.type.id],
                    // canais visuais à la Hina: raio+opacidade = confiança;
                    // arquivada/deprecada esmaece; anel = frescor de acesso.
                    size: 5.5 + m.confidence * 3.5,
                    opacity: (0.35 + m.confidence * 0.6) * (m.status === "ativo" ? 1 : 0.45),
                    ring: freshnessOf(m.lastAccessed, m.created),
                })),
                ...ghosts,
            ];
            const nodeSet = new Set(nodes.map((n) => n.id));
            setGraph({
                nodes,
                edges: [
                    ...structural,
                    ...edges.filter((e) => nodeSet.has(e.from) && nodeSet.has(e.to)),
                ],
            });
        })();
    }, [memKey, palette]);

    // ── blackboard (hot.md) ─────────────────────────────────────────
    const [hot, setHot] = dc.useState(null);
    dc.useEffect(() => {
        (async () => {
            try {
                const text = await app.vault.adapter.read(HOT_PATH);
                const sections = [];
                let current = null;
                for (const line of text.split("\n")) {
                    const h = line.match(/^##\s+(.+)$/);
                    if (h) { current = { scope: h[1].trim(), items: [] }; sections.push(current); continue; }
                    const item = line.match(/^-\s+\[\[([^\]|#]+)\]\]\s*—?\s*(.*)$/);
                    if (item && current) current.items.push({ id: item[1].trim(), rest: item[2] });
                    else if (line.match(/^-\s+_…/) && current) current.items.push({ id: null, rest: line.replace(/^-\s+/, "") });
                }
                setHot(sections.filter((s) => s.items.length));
            } catch (e) { setHot([]); }
        })();
    }, []);

    // ── consolidação (p/ o stat de idade) ───────────────────────────
    const [reportAge, setReportAge] = dc.useState(null);
    dc.useEffect(() => {
        (async () => {
            try {
                if (await app.vault.adapter.exists(REPORT_PATH)) {
                    const r = JSON.parse(await app.vault.adapter.read(REPORT_PATH));
                    const days = Math.floor((Date.now() - new Date(r.generated).getTime()) / 86400000);
                    setReportAge(days);
                }
            } catch (e) { /* sem relatório */ }
        })();
    }, []);

    // ── ações do inbox (decisão humana — P4) ────────────────────────
    const decide = async (p, verdict) => {
        const file = app.vault.getAbstractFileByPath(p.path);
        if (!file) { new Notice("Proposta não encontrada no disco."); return; }
        if (verdict === "aceita") {
            try {
                const text = await app.vault.read(file);
                const change = text.split(/^## Mudança proposta\s*$/m)[1];
                if (change && navigator.clipboard) await navigator.clipboard.writeText(change.trim());
            } catch (e) { /* clipboard é conveniência, não requisito */ }
        }
        await app.fileManager.processFrontMatter(file, (fm) => {
            fm.status = verdict;
            fm.decidida = new Date().toISOString().slice(0, 10);
        });
        if (verdict === "aceita") {
            new Notice("Proposta aceita — mudança copiada p/ o clipboard. Aplique na nota alvo.");
            openNote(p.target);
        } else {
            new Notice("Proposta rejeitada.");
        }
    };

    // ── stats ───────────────────────────────────────────────────────
    const ativas = memories.filter((m) => m.status === "ativo");
    const countType = (id) => ativas.filter((m) => m.type.id === id).length;
    const [graphMode, setGraphMode] = dc.useState("3d");

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {/* 1 — stats */}
            <GloStatGroup min="120px">
                <GloStat value={ativas.length} label="Memórias ativas" />
                {Object.values(TYPE_META).map((t) => (
                    <GloStat key={t.id} value={countType(t.id)} label={t.label} color={palette.type[t.id]} />
                ))}
                <GloStat
                    value={pending.length} label="Propostas pendentes"
                    status={pending.length ? "warning" : "success"}
                    sub={decided ? `${decided} decididas` : null}
                />
                <GloStat
                    value={reportAge === null ? "—" : reportAge === 0 ? "hoje" : `${reportAge}d`}
                    label="Última consolidação"
                    status={reportAge === null ? "neutral" : reportAge > 2 ? "warning" : "success"}
                />
            </GloStatGroup>

            {/* 2+3+4 — grafo | blackboard + inbox */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "14px", alignItems: "start" }}>
                <GloCard
                    variant="glass"
                    size="small" headerStyle={{ padding: "8px 14px 2px" }}
                    title="Grafo de relações" subtitle="wikilinks · supersedes · memory_link"
                    actions={
                        <div style={{ display: "flex", gap: "4px" }}>
                            <GloButton size="small" variant="liquid-glass" active={graphMode === "3d"} glow={false} lift={false} press={true} label="3D" onClick={() => setGraphMode("3d")} />
                            <GloButton size="small" variant="liquid-glass" active={graphMode === "2d"} glow={false} lift={false} press={true} label="2D" onClick={() => setGraphMode("2d")} />
                        </div>
                    }
                    style={{ gridColumn: "span 1" }}
                >
                    {graph === null ? (
                        <p style={{ opacity: 0.5 }}>Montando o grafo…</p>
                    ) : (() => {
                        const graphProps = {
                            nodes: graph.nodes,
                            edges: graph.edges,
                            height: 420,
                            onNodeClick: (id) => {
                                const m = memories.find((x) => x.id === id);
                                openNote(m ? m.path : id);
                            },
                            legend: [
                                ...Object.values(TYPE_META).map((t) => ({ color: palette.type[t.id], label: t.label })),
                                { color: palette.faint, label: "fora do mecha-brain" },
                                { style: "solid", color: palette.accent, label: "wikilink" },
                                { style: "dashed", color: palette.supersedes, label: "supersedes" },
                                { style: "dotted", color: palette.memoryLink, label: "memory_link" },
                            ],
                            emptyText: "Sem memórias para grafar ainda.",
                        };
                        return graphMode === "3d" ? <GloGraph3D {...graphProps} /> : <GloGraph {...graphProps} />;
                    })()}
                </GloCard>

                <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                    <GloCard variant="glass" size="small" headerStyle={{ padding: "8px 14px 2px" }} title="Blackboard (hot.md)" subtitle="foco atual por escopo — escrito pelo consolidador">
                        {hot === null ? (
                            <p style={{ opacity: 0.5 }}>Lendo…</p>
                        ) : hot.length === 0 ? (
                            <p style={{ opacity: 0.5 }}>Nenhum escopo ativo ainda — a primeira consolidação depois da primeira escrita preenche o blackboard.</p>
                        ) : hot.map((sec) => (
                            <div key={sec.scope} style={{ marginBottom: "10px" }}>
                                <GloBadge variant="soft" status="primary" size="small">{sec.scope}</GloBadge>
                                <div style={{ marginTop: "4px" }}>
                                    {sec.items.map((it, i) => it.id ? (
                                        <div key={i} style={{ fontSize: "0.85em", padding: "2px 0" }}>
                                            <a style={{ cursor: "pointer" }} onClick={() => openNote(it.id)}>{it.id}</a>
                                            {it.rest && <span style={{ opacity: 0.55 }}> — {it.rest}</span>}
                                        </div>
                                    ) : (
                                        <div key={i} style={{ fontSize: "0.8em", opacity: 0.5 }}>{it.rest}</div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </GloCard>

                    <GloCard variant="glass" size="small" headerStyle={{ padding: "8px 14px 2px" }} title="Inbox de propostas" subtitle="mudanças fora do sandbox — a decisão é sua">
                        {pending.length === 0 ? (
                            <p style={{ opacity: 0.5 }}>Nada pendente{decided ? ` (${decided} já decididas)` : ""}.</p>
                        ) : pending.map((p) => (
                            <div key={p.path} style={{ borderBottom: "1px solid var(--background-modifier-border)", padding: "8px 0" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                                    <a style={{ cursor: "pointer", fontWeight: 500, flex: "1 1 auto" }} onClick={() => openNote(p.path)}>{p.title}</a>
                                    <GloBadgeGroup gap="4px">
                                        {p.agent && <GloBadge size="small" variant="outlined" status="neutral">{p.agent}</GloBadge>}
                                        <GloBadge size="small" variant="soft" status="info">{p.created}</GloBadge>
                                    </GloBadgeGroup>
                                </div>
                                <div style={{ fontSize: "0.8em", opacity: 0.6, margin: "3px 0 6px" }}>
                                    alvo: <a style={{ cursor: "pointer" }} onClick={() => openNote(p.target)}>{p.target}</a>
                                </div>
                                <div style={{ display: "flex", gap: "8px" }}>
                                    <GloButton size="small" label="Aceitar" onClick={() => decide(p, "aceita")} />
                                    <GloButton size="small" variant="ghost" label="Rejeitar" onClick={() => decide(p, "rejeitada")} />
                                </div>
                            </div>
                        ))}
                    </GloCard>
                </div>
            </div>

            {/* 5 — observabilidade + manutenção, sempre visíveis */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "14px", alignItems: "start" }}>
                <GloCard size="small" headerStyle={{ padding: "8px 14px 2px" }} title="Atividade do kernel" subtitle="aceitas · rejeitadas · promoções · propostas · links" variant="glass">
                    <ActivityFeed />
                </GloCard>
                <GloCard size="small" headerStyle={{ padding: "8px 14px 2px" }} title="Última consolidação" subtitle="o que o kernel detectou e reportou para você decidir" variant="glass">
                    <MaintenancePanel />
                </GloCard>
            </div>

            {/* 6 — cards de memória */}
            <MemoryBrowser showMaintenance={false} />
        </div>
    );
}

return { Func: MechabrainDashboard, MechabrainDashboard };
