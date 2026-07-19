// ═══════════════════════════════════════════════════════════════════════════════
// DC-GLO-GRAPH - Global Themed Relationship Graph (2D)
// Grafo de forças em SVG puro (zero dependências): nós coloridos, arestas com
// estilos (sólida/tracejada/pontilhada, direção), zoom (roda), pan (arrastar o
// fundo), drag de nós, highlight de vizinhança, legenda.
// Família Glo*: lê o tema do dc-themeProvider.
//
// PERFORMANCE: não há loop de animação. Pan/zoom escrevem o transform do grupo
// raiz DIRETO no DOM (1 atributo por evento); arrastar um nó escreve o
// transform do nó + endpoints das arestas adjacentes via refs. O VDOM só
// re-renderiza em mudança de dataset/tema/hover — nunca por pointermove.
// ═══════════════════════════════════════════════════════════════════════════════

const { useTheme } = await dc.require(dc.fileLink("System/Scripts/Core/dc-themeProvider.jsx"));

const VW = 800;

// ── Simulação de forças (uma vez por dataset; determinística) ──────────────────
function simulate(nodes, edges, height, iterations) {
    const pos = {};
    nodes.forEach((node, i) => {
        const r = 30 + 14 * Math.sqrt(i);
        const a = i * 2.39996; // golden angle
        pos[node.id] = { x: VW / 2 + r * Math.cos(a), y: height / 2 + r * Math.sin(a) };
    });
    const ids = nodes.map((nd) => nd.id);
    const springs = edges
        .filter((e) => pos[e.from] && pos[e.to])
        .map((e) => [e.from, e.to]);

    const REPULSION = 2600, SPRING = 0.035, REST = 95, GRAVITY = 0.012;
    for (let it = 0; it < iterations; it++) {
        const cool = 1 - it / iterations;
        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) {
                const a = pos[ids[i]], b = pos[ids[j]];
                let dx = a.x - b.x, dy = a.y - b.y;
                let d2 = dx * dx + dy * dy;
                if (d2 < 1) { dx = (i - j) * 0.1 || 0.1; dy = 0.1; d2 = 0.02; }
                const f = (REPULSION / d2) * cool;
                const d = Math.sqrt(d2);
                a.x += (dx / d) * f; a.y += (dy / d) * f;
                b.x -= (dx / d) * f; b.y -= (dy / d) * f;
            }
        }
        for (const [fa, fb] of springs) {
            const a = pos[fa], b = pos[fb];
            const dx = b.x - a.x, dy = b.y - a.y;
            const d = Math.sqrt(dx * dx + dy * dy) || 1;
            const f = SPRING * (d - REST) * cool;
            a.x += (dx / d) * f; a.y += (dy / d) * f;
            b.x -= (dx / d) * f; b.y -= (dy / d) * f;
        }
        for (const id of ids) {
            const p = pos[id];
            p.x += (VW / 2 - p.x) * GRAVITY;
            p.y += (height / 2 - p.y) * GRAVITY;
        }
    }
    for (const id of ids) {
        pos[id].x = Math.max(24, Math.min(VW - 24, pos[id].x));
        pos[id].y = Math.max(20, Math.min(height - 24, pos[id].y));
    }
    return pos;
}

const DASH = { solid: "none", dashed: "7 4", dotted: "2 4" };

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT: GloGraph
// ═══════════════════════════════════════════════════════════════════════════════
function GloGraph({
    nodes = [],                   // [{ id, label?, color?, size?, ghost? }]
    edges = [],                   // [{ from, to, style?, color?, directed?, label? }]
    height = 420,                 // Altura do espaço de coordenadas (px)
    iterations = 260,             // Iterações da simulação de forças
    legend = null,                // [{ color, label, style? }]
    emptyText = "Sem nós para exibir.",
    onNodeClick = null,           // (id) => void
    style = {},
}) {
    const { theme } = useTheme();
    const accent = theme?.["color-accent"] || "#0099ff";

    const key = dc.useMemo(
        () => nodes.map((n) => n.id).join("|") + "#" + edges.length,
        [nodes, edges]
    );
    const basePos = dc.useMemo(() => simulate(nodes, edges, height, iterations), [key]);

    // Posições movidas à mão: em REF durante o drag (escrita imperativa);
    // commit p/ estado só no pointerup — 1 re-render por drag, não por move.
    const dragPos = dc.useRef({});
    const [, commitDrag] = dc.useState(0);
    const viewRef = dc.useRef({ x: 0, y: 0, k: 1 });
    const gRef = dc.useRef(null);
    const svgRef = dc.useRef(null);
    const nodeEls = dc.useRef({});
    const edgeEls = dc.useRef({});
    const dragging = dc.useRef(null);
    const moved = dc.useRef(false);
    const [hover, setHover] = dc.useState(null);
    dc.useEffect(() => {
        dragPos.current = {};
        viewRef.current = { x: 0, y: 0, k: 1 };
        if (gRef.current) gRef.current.setAttribute("transform", "translate(0,0) scale(1)");
    }, [key]);

    const P = (id) => dragPos.current[id] || basePos[id];

    // Adjacência: id → índices de arestas (p/ drag imperativo de nó).
    const edgesOf = dc.useMemo(() => {
        const m = {};
        edges.forEach((e, i) => {
            (m[e.from] = m[e.from] || []).push(i);
            (m[e.to] = m[e.to] || []).push(i);
        });
        return m;
    }, [edges]);

    const neighbors = dc.useMemo(() => {
        const m = {};
        for (const e of edges) {
            (m[e.from] = m[e.from] || new Set()).add(e.to);
            (m[e.to] = m[e.to] || new Set()).add(e.from);
        }
        return m;
    }, [edges]);

    const applyView = () => {
        const v = viewRef.current;
        if (gRef.current) gRef.current.setAttribute("transform", `translate(${v.x},${v.y}) scale(${v.k})`);
    };

    // pointer(cliente) → coordenadas do grafo
    const toGraph = (ev) => {
        const svg = svgRef.current;
        const r = svg.getBoundingClientRect();
        const v = viewRef.current;
        const sx = (ev.clientX - r.left) * (VW / r.width);
        const sy = (ev.clientY - r.top) * (VW / r.width);
        return { x: (sx - v.x) / v.k, y: (sy - v.y) / v.k };
    };

    // Escreve, direto no DOM, a posição de um nó e das arestas ligadas a ele.
    const writeNode = (id) => {
        const p = P(id);
        const el = nodeEls.current[id];
        if (el && p) el.setAttribute("transform", `translate(${p.x} ${p.y})`);
        for (const i of edgesOf[id] || []) {
            const e = edges[i];
            const le = edgeEls.current[i];
            const a = P(e.from), b = P(e.to);
            if (!le || !a || !b) continue;
            le.setAttribute("x1", a.x); le.setAttribute("y1", a.y);
            le.setAttribute("x2", b.x); le.setAttribute("y2", b.y);
        }
    };

    const onPointerMove = (ev) => {
        const d = dragging.current;
        if (!d) return;
        moved.current = true;
        if (d.id) {
            const g = toGraph(ev);
            dragPos.current = { ...dragPos.current, [d.id]: { x: g.x, y: g.y } };
            writeNode(d.id);
        } else if (d.pan) {
            const r = svgRef.current.getBoundingClientRect();
            const f = VW / r.width;
            viewRef.current.x = d.ox + (ev.clientX - d.startX) * f;
            viewRef.current.y = d.oy + (ev.clientY - d.startY) * f;
            applyView();
        }
    };
    const stopDrag = () => {
        if (dragging.current?.id) commitDrag((v) => v + 1); // consolida no VDOM
        dragging.current = null;
    };
    const onWheel = (ev) => {
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
        viewRef.current.k = Math.max(0.4, Math.min(4, viewRef.current.k * factor));
        applyView();
    };

    if (!nodes.length) return <p style={{ opacity: 0.5 }}>{emptyText}</p>;

    const dim = (id) => hover && id !== hover && !(neighbors[hover]?.has(id));
    const edgeDim = (e) => hover && e.from !== hover && e.to !== hover;
    const v = viewRef.current;

    return (
        <div style={{ width: "100%", ...style }}>
            <svg
                ref={svgRef}
                viewBox={`0 0 ${VW} ${height}`}
                style={{ width: "100%", display: "block", background: "var(--background-primary-alt)", borderRadius: "8px", border: "1px solid var(--background-modifier-border)", touchAction: "none", cursor: "grab" }}
                onPointerMove={onPointerMove}
                onPointerUp={stopDrag}
                onPointerLeave={stopDrag}
                onWheel={onWheel}
                onPointerDown={(ev) => {
                    if (ev.target.tagName === "svg" || ev.target.dataset?.bg) {
                        dragging.current = { pan: true, startX: ev.clientX, startY: ev.clientY, ox: viewRef.current.x, oy: viewRef.current.y };
                        moved.current = false;
                    }
                }}
            >
                <defs>
                    <marker id="glo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                        <path d="M 0 1 L 9 5 L 0 9 z" fill="var(--text-faint)" />
                    </marker>
                </defs>
                <rect data-bg="1" x="0" y="0" width={VW} height={height} fill="transparent" />
                <g ref={gRef} transform={`translate(${v.x},${v.y}) scale(${v.k})`}>
                    {edges.map((e, i) => {
                        const a = P(e.from), b = P(e.to);
                        if (!a || !b) return null;
                        return (
                            <g key={i} opacity={edgeDim(e) ? 0.12 : 0.75}>
                                <line
                                    ref={(el) => { if (el) edgeEls.current[i] = el; }}
                                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                                    stroke={e.color || "var(--text-faint)"}
                                    strokeWidth={hover && !edgeDim(e) ? 2 : 1.1}
                                    strokeDasharray={DASH[e.style || "solid"]}
                                    markerEnd={e.directed ? "url(#glo-arrow)" : undefined}
                                />
                                {e.label && !edgeDim(e) && hover && (
                                    <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 3} fontSize="8" fill="var(--text-muted)" textAnchor="middle">{e.label}</text>
                                )}
                            </g>
                        );
                    })}
                    {nodes.map((n) => {
                        const p = P(n.id);
                        if (!p) return null;
                        const r = n.size || 8;
                        const c = n.color || accent;
                        return (
                            <g
                                key={n.id}
                                ref={(el) => { if (el) nodeEls.current[n.id] = el; }}
                                transform={`translate(${p.x} ${p.y})`}
                                opacity={dim(n.id) ? 0.18 : 1}
                                style={{ cursor: "pointer" }}
                                onPointerDown={(ev) => { ev.stopPropagation(); dragging.current = { id: n.id }; moved.current = false; }}
                                onPointerUp={() => {
                                    const wasDrag = moved.current;
                                    stopDrag();
                                    if (!wasDrag && onNodeClick) onNodeClick(n.id);
                                }}
                                onMouseEnter={() => setHover(n.id)}
                                onMouseLeave={() => setHover(null)}
                            >
                                <title>{n.label || n.id}</title>
                                <circle
                                    r={r}
                                    fill={n.ghost ? "transparent" : c}
                                    stroke={n.ghost ? "var(--text-faint)" : c}
                                    strokeWidth={n.ghost ? 1.4 : 0}
                                    strokeDasharray={n.ghost ? "3 2" : "none"}
                                />
                                <text y={r + 10} fontSize="9" textAnchor="middle" fill={n.ghost ? "var(--text-faint)" : "var(--text-normal)"}>
                                    {(n.label || n.id).length > 22 ? (n.label || n.id).slice(0, 21) + "…" : (n.label || n.id)}
                                </text>
                            </g>
                        );
                    })}
                </g>
            </svg>
            {legend && (
                <div style={{ display: "flex", gap: "14px", flexWrap: "wrap", marginTop: "6px", fontSize: "0.75em", opacity: 0.75 }}>
                    {legend.map((l, i) => (
                        <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: "5px" }}>
                            {l.style ? (
                                <svg width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" stroke={l.color || "var(--text-faint)"} strokeWidth="2" strokeDasharray={DASH[l.style] || "none"} /></svg>
                            ) : (
                                <span style={{ width: "9px", height: "9px", borderRadius: "50%", background: l.color, display: "inline-block" }} />
                            )}
                            {l.label}
                        </span>
                    ))}
                    <span style={{ opacity: 0.6 }}>· roda = zoom · arraste o fundo = pan · arraste um nó = reposicionar</span>
                </div>
            )}
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEMO VIEW
// ═══════════════════════════════════════════════════════════════════════════════
function renderedView() {
    const nodes = [
        { id: "a", label: "Nota A", color: "#4da3ff" },
        { id: "b", label: "Nota B", color: "#ff7a45" },
        { id: "c", label: "Nota C", color: "#a06bff" },
        { id: "d", label: "Externa", ghost: true },
    ];
    const edges = [
        { from: "a", to: "b" },
        { from: "b", to: "c", style: "dashed", directed: true, label: "supersedes" },
        { from: "a", to: "d", style: "dotted", label: "related" },
    ];
    return <GloGraph nodes={nodes} edges={edges} height={300} legend={[{ color: "#4da3ff", label: "tipo A" }, { style: "dashed", label: "supersedes" }]} />;
}

return { GloGraph, renderedView };
