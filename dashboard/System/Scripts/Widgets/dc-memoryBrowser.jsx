/* ==================================================================
   MEMORY BROWSER (Mecha-Brain)
   - Navega as memórias dos agentes em `mecha-brain/` (Semantic/,
     Episodic/, Procedural/, Research/) com filtros por tipo, escopo,
     status e busca por texto.
   - Lê o relatório da última consolidação (§9) de
     `mecha-brain/_meta/index/consolidation-report.json` (gitignored,
     por máquina) — merge candidates, PROCs stale, decay.
   - Componentes: GloCard, GloBadge, GloInput, GloButton (liquid-glass p/ abas, como a Home).
   - Usage (dashboard):
       const s = await dc.require(dc.fileLink("System/Scripts/Widgets/dc-memoryBrowser.jsx"));
       return function View() { return s.Func(); }
   ================================================================== */

const { useTheme, deriveChartColors } = await dc.require(dc.fileLink("System/Scripts/Core/dc-themeProvider.jsx"));
const { GloCard } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloCard.jsx"));
const { GloBadge, GloBadgeGroup } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloBadge.jsx"));
const { GloInput } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloInput.jsx"));
const { GloButton } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloButton.jsx"));

// ── Contrato mecha-brain (espelha a spec §3/§6; nomes fixos) ─────────
const ROOT = "mecha-brain";
const TYPE_BY_FOLDER = {
    Semantic: { id: "semantic", label: "Semânticas", chart: "chart-color-5" },
    Episodic: { id: "episodic", label: "Episódicas", chart: "chart-color-2" },
    Procedural: { id: "procedural", label: "Procedurais", chart: "chart-color-3" },
    Research: { id: "research", label: "Research", chart: "chart-color-1" },
};

// Cores dos tipos derivadas do TEMA (chart-colors), como na Home.
const typeColorsFrom = (theme) => {
    const t = deriveChartColors({ ...(theme || {}) });
    const out = {};
    for (const meta of Object.values(TYPE_BY_FOLDER)) out[meta.id] = t[meta.chart];
    return out;
};

// ── Excerto do corpo p/ os cards (estilo Mnemosyne) ─────────────────
const EXCERPT_CHARS = 220;
const excerptOf = (raw) => {
    let text = raw.replace(/^---\n[\s\S]*?\n---\n?/, "");   // frontmatter
    text = text
        .replace(/```[\s\S]*?```/g, " ")                     // code fences
        .replace(/^#+\s+.*$/gm, " ")                         // headings
        .replace(/^>\s?\[![^\]]*\][+-]?\s?/gm, "")           // callout markers
        .replace(/^>\s?/gm, "")                              // blockquote bars
        .replace(/!\[\[[^\]]*\]\]/g, " ")                    // embeds
        .replace(/\[\[([^\]|#]+)(?:[^\]]*)?\]\]/g, "$1")     // wikilinks → texto
        .replace(/[*_`]/g, "")
        .replace(/\s+/g, " ")
        .trim();
    return text.length > EXCERPT_CHARS ? text.slice(0, EXCERPT_CHARS - 1).trimEnd() + "…" : text;
};
const CONFIDENCE_STATUS = { high: "success", medium: "info", low: "neutral" };
const NOTE_STATUS = {
    ativo: null, // sem badge — é o normal
    arquivado: { status: "warning", label: "arquivado" },
    deprecado: { status: "error", label: "deprecado" },
};
const REPORT_PATH = `${ROOT}/_meta/index/consolidation-report.json`;
const PAGE_SIZE = 30;

// ── Helpers defensivos (mesmo padrão do dc-academicDashboard) ───────
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

// Formata "YYYY-MM-DD" ou "YYYY-MM-DD HH:MM" quando há hora (≠ meia-noite):
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
    if (val instanceof Date) return fmtStamp(val.toISOString());
    if (typeof val.toISOString === "function") return fmtStamp(val.toISOString());
    return fmtStamp(String(val));
};

const memoryTypeOf = (path) => {
    const m = path.match(/^mecha-brain\/(Semantic|Episodic|Procedural|Research)\//);
    return m ? TYPE_BY_FOLDER[m[1]] : null;
};

// ── Painel da última consolidação (auto-contido; também exportado) ──
function MaintenancePanel() {
    const [report, setReport] = dc.useState(undefined); // undefined=carregando, null=ausente
    dc.useEffect(() => {
        (async () => {
            try {
                if (await app.vault.adapter.exists(REPORT_PATH)) {
                    setReport(JSON.parse(await app.vault.adapter.read(REPORT_PATH)));
                } else setReport(null);
            } catch (e) { setReport(null); }
        })();
    }, []);

    const dimStyle = { fontSize: "0.78em", opacity: 0.55 };
    const open = (path) => app.workspace.openLinkText(path, "", false);
    const wl = (raw) => String(raw || "").replace(/^\[\[|\]\]$/g, "");
    const noteLink = (raw) => (
        <a style={{ cursor: "pointer" }} onClick={() => open(`${ROOT}/${wl(raw)}`)}>{wl(raw)}</a>
    );

    if (report === undefined) return <span style={dimStyle}>Lendo relatório…</span>;
    if (report === null) return (
        <span style={dimStyle}>
            Sem relatório de consolidação nesta máquina — rode <code>mechabrain consolidate</code> (o timer diário também o gera).
        </span>
    );
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <GloBadgeGroup>
                <GloBadge variant="soft" status="info">{toDateStr(report.generated)}</GloBadge>
                <GloBadge variant="soft" status="neutral">{report.counts?.notes_scanned ?? 0} notas · {report.counts?.readonly_scanned ?? 0} read-only</GloBadge>
                <GloBadge variant="soft" status={report.counts?.decayed ? "warning" : "neutral"}>{report.counts?.decayed ?? 0} arquivadas</GloBadge>
                <GloBadge variant="soft" status={report.counts?.deprecated ? "warning" : "neutral"}>{report.counts?.deprecated ?? 0} deprecadas</GloBadge>
            </GloBadgeGroup>
            {(report.merge_candidates || []).length > 0 && (
                <div>
                    <strong style={{ fontSize: "0.85em" }}>Candidatas a fusão (mesmo escopo):</strong>
                    {report.merge_candidates.map((c, i) => (
                        <div key={i} style={{ fontSize: "0.85em", marginLeft: "12px" }}>
                            {noteLink(c.a)} ~ {noteLink(c.b)} <span style={dimStyle}>({c.similarity}, {c.scope_a})</span>
                        </div>
                    ))}
                </div>
            )}
            {(report.stale_procedurals || []).length > 0 && (
                <div>
                    <strong style={{ fontSize: "0.85em" }}>Procedurais sem reteste:</strong>
                    {report.stale_procedurals.map((s, i) => (
                        <div key={i} style={{ fontSize: "0.85em", marginLeft: "12px" }}>
                            {noteLink(s.note)} <span style={dimStyle}>({s.days_stale} dias, {s.scope})</span>
                        </div>
                    ))}
                </div>
            )}
            {(report.cross_scope_similar || []).length > 0 && (
                <div>
                    <strong style={{ fontSize: "0.85em" }}>Similares cross-scope (nunca fundidas):</strong>
                    {report.cross_scope_similar.map((c, i) => (
                        <div key={i} style={{ fontSize: "0.85em", marginLeft: "12px" }}>
                            {noteLink(c.a)} ~ {noteLink(c.b)} <span style={dimStyle}>({c.scope_a} × {c.scope_b})</span>
                        </div>
                    ))}
                </div>
            )}
            {(report.docs_citing_dead || []).length > 0 && (
                <div>
                    <strong style={{ fontSize: "0.85em" }}>Docs citando memórias mortas (propor edição):</strong>
                    {report.docs_citing_dead.map((d, i) => (
                        <div key={i} style={{ fontSize: "0.85em", marginLeft: "12px" }}>
                            {noteLink(d.doc)} cita {noteLink(d.cites)} <span style={dimStyle}>({d.status}{d.successor ? " → " : ""})</span>{d.successor ? noteLink(d.successor) : null}
                        </div>
                    ))}
                </div>
            )}
            {!(report.merge_candidates || []).length && !(report.stale_procedurals || []).length && !(report.cross_scope_similar || []).length && !(report.docs_citing_dead || []).length && (
                <span style={dimStyle}>Nada pendente de decisão — sem duplicatas, sem PROC stale.</span>
            )}
        </div>
    );
}

function MemoryBrowser({ showMaintenance = true } = {}) {
    // Query estreita: só páginas de mecha-brain/ — o índice do datacore não
    // invalida (nem re-renderiza) este widget quando qualquer OUTRA nota da
    // vault muda. O filtro fino por pasta de tipo continua no useMemo abaixo.
    const pages = dc.useQuery('@page and path("mecha-brain")');
    const { theme } = useTheme();
    const typeColor = dc.useMemo(() => typeColorsFrom(theme), [theme]);

    const [typeFilter, setTypeFilter] = dc.useState("all");
    const [statusFilter, setStatusFilter] = dc.useState("ativo");
    const [scopeFilter, setScopeFilter] = dc.useState("all");
    const [search, setSearch] = dc.useState("");
    const [limit, setLimit] = dc.useState(PAGE_SIZE);
    // Cache de excertos em REF (sobrevive a re-renders sem virar dependência de
    // efeito); o contador só dispara o re-render quando algo novo foi lido.
    const excerptCache = dc.useRef({});
    const [, bumpExcerpts] = dc.useState(0);
    const excerpts = excerptCache.current;

    // ── Coleta e normalização das memórias ──────────────────────────
    const memories = dc.useMemo(() => {
        const out = [];
        for (const p of pages) {
            const path = pathOf(p);
            const type = memoryTypeOf(path);
            if (!type) continue; // fora das 4 pastas de memória
            const status = String(getValue(p, "status") || "ativo").trim();
            out.push({
                path,
                id: path.split("/").pop().replace(/\.md$/, ""),
                title: String(getValue(p, "title") || path.split("/").pop().replace(/\.md$/, "")),
                type,
                scope: String(getValue(p, "scope") || "global").trim(),
                agent: String(getValue(p, "agent") || "").trim(),
                confidence: String(getValue(p, "confidence") || "").trim().toLowerCase(),
                status,
                created: toDateStr(getValue(p, "created")),
                lastAccessed: toDateStr(getValue(p, "last_accessed")),
                lastTested: toDateStr(getValue(p, "last_tested")),
            });
        }
        out.sort((a, b) => (b.created || "").localeCompare(a.created || "") || a.id.localeCompare(b.id));
        return out;
    }, [pages]);

    const scopes = dc.useMemo(
        () => [...new Set(memories.map((m) => m.scope))].sort(),
        [memories]
    );

    // ── Filtros ─────────────────────────────────────────────────────
    const filtered = dc.useMemo(() => {
        const q = search.trim().toLowerCase();
        return memories.filter((m) => {
            if (typeFilter !== "all" && m.type.id !== typeFilter) return false;
            if (statusFilter === "ativo" && m.status !== "ativo") return false;
            if (statusFilter === "arquivado" && m.status !== "arquivado") return false;
            if (statusFilter === "deprecado" && m.status !== "deprecado") return false;
            if (scopeFilter !== "all" && m.scope !== scopeFilter) return false;
            if (q && !`${m.title} ${m.id} ${m.agent} ${m.scope}`.toLowerCase().includes(q)) return false;
            return true;
        });
    }, [memories, typeFilter, statusFilter, scopeFilter, search]);

    const countBy = (id) => memories.filter((m) => m.type.id === id && m.status === "ativo").length;

    const open = (path) => app.workspace.openLinkText(path, "", false);

    // Excertos dos cards visíveis (lê o corpo uma vez por nota, sob demanda)
    const visible = filtered.slice(0, limit);
    dc.useEffect(() => {
        (async () => {
            const cache = excerptCache.current;
            const missing = visible.filter((m) => cache[m.path] === undefined);
            if (!missing.length) return;
            for (const m of missing) {
                try { cache[m.path] = excerptOf(await app.vault.adapter.read(m.path)); }
                catch (e) { cache[m.path] = ""; }
            }
            bumpExcerpts((v) => v + 1);
        })();
    }, [visible.map((m) => m.path).join("|")]);

    // ── Estilos ─────────────────────────────────────────────────────
    const dimStyle = { fontSize: "0.78em", opacity: 0.55, whiteSpace: "nowrap" };

    // ── Render ──────────────────────────────────────────────────────
    if (memories.length === 0) {
        return (
            <GloCard variant="glass" title="Memórias do Mecha-Brain">
                <p style={{ opacity: 0.6 }}>
                    Nenhuma memória em <code>{ROOT}/</code> ainda. Os agentes escrevem via
                    <code> memory_write</code>.
                </p>
            </GloCard>
        );
    }

    return (
        <GloCard
            variant="glass"
            size="small"
            headerStyle={{ padding: "8px 14px 2px" }}
            title="Memórias do Mecha-Brain"
            subtitle={`${memories.length} notas · ${filtered.length} no filtro`}
        >
            {/* filtro por tipo — botões liquid-glass, estilo da Home */}
            <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {[{ id: "all", label: `Todas (${memories.filter((m) => m.status === "ativo").length})` },
                  ...Object.values(TYPE_BY_FOLDER).map((t) => ({ id: t.id, label: `${t.label} (${countBy(t.id)})` }))].map((tab) => (
                    <GloButton key={tab.id} size="small" variant="liquid-glass" active={typeFilter === tab.id} glow={false} lift={false} press={true} label={tab.label} onClick={() => { setTypeFilter(tab.id); setLimit(PAGE_SIZE); }} />
                ))}
            </div>

            {/* escopo + status + busca */}
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center", margin: "8px 0 12px" }}>
                <GloBadgeGroup>
                    <GloBadge
                        variant={scopeFilter === "all" ? "filled" : "soft"} status="primary" clickable
                        onClick={() => setScopeFilter("all")}
                    >todos escopos</GloBadge>
                    {scopes.map((s) => (
                        <GloBadge
                            key={s} clickable status="primary"
                            variant={scopeFilter === s ? "filled" : "soft"}
                            onClick={() => setScopeFilter(scopeFilter === s ? "all" : s)}
                        >{s}</GloBadge>
                    ))}
                </GloBadgeGroup>
                <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                    {[{ id: "ativo", label: "ativas" }, { id: "arquivado", label: "arquivadas" }, { id: "deprecado", label: "deprecadas" }, { id: "todas", label: "todas" }].map((tab) => (
                        <GloButton key={tab.id} size="small" variant="liquid-glass" active={statusFilter === tab.id} glow={false} lift={false} press={true} label={tab.label} onClick={() => { setStatusFilter(tab.id); setLimit(PAGE_SIZE); }} />
                    ))}
                </div>
                <GloInput
                    placeholder="Buscar título, id, agente…" clearable
                    size="small" width="240px" debounce={200}
                    onChange={(v) => { setSearch(v ?? ""); setLimit(PAGE_SIZE); }}
                    onClear={() => setSearch("")}
                />
            </div>

            {/* cards de memória (GloCard glass, densos — sem espaço morto) */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: "8px" }}>
                {visible.map((m) => {
                    const st = NOTE_STATUS[m.status];
                    const excerpt = excerpts[m.path];
                    return (
                        <GloCard
                            key={m.path}
                            variant="glass"
                            size="small"
                            clickable
                            glow
                            onClick={() => open(m.path)}
                            title={m.title}
                            style={{ opacity: m.status === "ativo" ? 1 : 0.65, minWidth: 0 }}
                            headerStyle={{ padding: "8px 12px 2px" }}
                            bodyStyle={{ padding: "2px 12px 4px" }}
                            footerStyle={{ padding: "4px 12px 8px" }}
                            footer={
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "4px", flexWrap: "wrap" }}>
                                    <GloBadgeGroup gap="3px">
                                        <GloBadge size="small" variant="soft" color={typeColor[m.type.id]}>{m.type.label}</GloBadge>
                                        <GloBadge size="small" variant="soft" status="primary">{m.scope}</GloBadge>
                                        {m.confidence && (
                                            <GloBadge size="small" variant="soft" status={CONFIDENCE_STATUS[m.confidence] || "neutral"}>{m.confidence}</GloBadge>
                                        )}
                                        {m.agent && <GloBadge size="small" variant="outlined" status="neutral">{m.agent}</GloBadge>}
                                        {st && <GloBadge size="small" variant="dot" status={st.status}>{st.label}</GloBadge>}
                                    </GloBadgeGroup>
                                    <span style={dimStyle} title={m.lastAccessed ? `último acesso ${m.lastAccessed}` : ""}>{m.created}</span>
                                </div>
                            }
                        >
                            <div style={{ fontSize: "0.8em", opacity: 0.7, lineHeight: 1.4 }} title={m.id}>
                                {excerpt === undefined ? "…" : excerpt || <em style={{ opacity: 0.6 }}>(sem corpo)</em>}
                            </div>
                        </GloCard>
                    );
                })}
            </div>
            {filtered.length === 0 && (
                <p style={{ opacity: 0.5, padding: "8px" }}>Nada casa com o filtro.</p>
            )}
            {filtered.length > limit && (
                <div style={{ textAlign: "center", marginTop: "10px" }}>
                    <GloButton size="small" variant="ghost" label={`mostrar mais (${filtered.length - limit})`} onClick={() => setLimit(limit + PAGE_SIZE)} />
                </div>
            )}

            {/* manutenção — sempre visível (nada escondido atrás de colapso) */}
            {showMaintenance && (
                <div style={{ marginTop: "14px" }}>
                    <GloCard title="Última consolidação" size="small" variant="glass" headerStyle={{ padding: "8px 14px 2px" }}>
                        <MaintenancePanel />
                    </GloCard>
                </div>
            )}
        </GloCard>
    );
}

return { Func: MemoryBrowser, MemoryBrowser, MaintenancePanel };
