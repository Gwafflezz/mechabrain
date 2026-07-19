// ═══════════════════════════════════════════════════════════════════════════════
// DC-GLO-GRAPH-3D - Global Themed 3D Relationship Graph
// Grafo de forças em três eixos com projeção perspectiva, em SVG puro (sem
// three.js/CDN). Funcionamento portado do mapa de memória da Hina:
//   • topologia livre ou núcleo→hubs→nós (props pin: "core" | "hub")
//   • halo + glow radiantes por nó, flutuação orgânica (bob + pulso por fase)
//   • opacidade contínua por nó (ex.: confiança), anel de frescor temporal
//   • labels com LOD, hover acende a vizinhança e esmaece o resto
// CORES: 100% do tema (dc-themeProvider).
//
// PERFORMANCE (por que este arquivo é imperativo):
//   A estrutura SVG é renderizada UMA vez por dataset/tema. A animação
//   (rotação, bob, pulso, profundidade, hover) roda num requestAnimationFrame
//   que escreve `transform`/`opacity` DIRETO nos elementos via refs — zero
//   setState, zero diff de VDOM por frame. O loop se auto-pausa quando a aba
//   está oculta (document.hidden / offsetParent null) e é throttled a ~30fps.
//   Interação (arraste/zoom/hover) também vive em refs, não em estado.
// ═══════════════════════════════════════════════════════════════════════════════

const { useTheme } = await dc.require(dc.fileLink("System/Scripts/Core/dc-themeProvider.jsx"));

const VW = 800;

function isDarkHex(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ""));
    if (!m) return true;
    const v = parseInt(m[1], 16);
    const r = (v >> 16) & 255, g = (v >> 8) & 255, b = v & 255;
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5;
}

// ── Simulação de forças em 3D (uma vez por dataset; determinística) ────────────
function simulate3D(nodes, edges, iterations) {
    const pos = {};
    const n = Math.max(nodes.length, 1);
    nodes.forEach((node, i) => {
        const t = (i + 0.5) / n;
        const phi = Math.acos(1 - 2 * t);
        const theta = i * 2.39996;
        const r = node.pin === "core" ? 0 : node.pin === "hub" ? 55 : 70 + 80 * Math.cbrt(t);
        pos[node.id] = {
            x: r * Math.sin(phi) * Math.cos(theta),
            y: r * Math.sin(phi) * Math.sin(theta),
            z: r * Math.cos(phi),
        };
    });
    const ids = nodes.map((nd) => nd.id);
    const byId = Object.fromEntries(nodes.map((nd) => [nd.id, nd]));
    const springs = edges
        .filter((e) => pos[e.from] && pos[e.to])
        .map((e) => [e.from, e.to, e.rest || 95]);

    const REPULSION = 52000, SPRING = 0.03, GRAVITY = 0.015;
    for (let it = 0; it < iterations; it++) {
        const cool = 1 - it / iterations;
        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) {
                const a = pos[ids[i]], b = pos[ids[j]];
                let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
                let d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 1) { dx = 0.5; dy = 0.3; dz = 0.2; d2 = 0.4; }
                const d = Math.sqrt(d2);
                const f = (REPULSION / (d2 * d)) * cool;
                a.x += dx * f; a.y += dy * f; a.z += dz * f;
                b.x -= dx * f; b.y -= dy * f; b.z -= dz * f;
            }
        }
        for (const [fa, fb, rest] of springs) {
            const a = pos[fa], b = pos[fb];
            const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
            const f = (SPRING * (d - rest) * cool) / d;
            a.x += dx * f; a.y += dy * f; a.z += dz * f;
            b.x -= dx * f; b.y -= dy * f; b.z -= dz * f;
        }
        for (const id of ids) {
            const p = pos[id];
            const g = byId[id]?.pin === "core" ? 0.25 : GRAVITY;
            p.x -= p.x * g; p.y -= p.y * g; p.z -= p.z * g;
        }
    }
    return pos;
}

const DASH = { solid: "none", dashed: "7 4", dotted: "2 4" };

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT: GloGraph3D
// ═══════════════════════════════════════════════════════════════════════════════
function GloGraph3D({
    nodes = [],                   // [{ id, label?, color?, size?, ghost?, opacity?, ring?, pin? }]
    edges = [],                   // [{ from, to, style?, color?, label?, rest? }]
    height = 460,                 // Altura do espaço de coordenadas (px)
    iterations = 220,             // Iterações da simulação
    autoRotate = true,            // Gira sozinho até a primeira interação
    fov = 520,                    // Distância focal da perspectiva
    background = null,            // Fundo CSS; null = radial derivado do tema
    legend = null,                // [{ color, label, style? }]
    emptyText = "Sem nós para exibir.",
    onNodeClick = null,           // (id) => void
    style = {},
}) {
    const { theme } = useTheme();
    const accent = theme?.["color-accent"] || "var(--interactive-accent)";
    const themeBg = theme?.["color-background"] || null;
    const themeSurface = theme?.["color-surface"] || null;
    const textColor = theme?.["color-text"] || "var(--text-normal)";
    const faintColor = theme?.["color-text-faint"] || "var(--text-faint)";
    const dark = isDarkHex(themeBg);
    const bg = background || (themeBg
        ? `radial-gradient(ellipse at center, ${themeSurface || themeBg} 0%, ${themeBg} 85%)`
        : "radial-gradient(ellipse at center, var(--background-primary-alt), var(--background-primary))");
    const blend = dark ? "screen" : "multiply";

    const key = dc.useMemo(() => nodes.map((n) => n.id).join("|") + "#" + edges.length, [nodes, edges]);
    const world = dc.useMemo(() => simulate3D(nodes, edges, iterations), [key]);

    // Vizinhança p/ o hover-dim (uma vez por dataset).
    const neighbors = dc.useMemo(() => {
        const m = {};
        for (const e of edges) {
            (m[e.from] = m[e.from] || new Set()).add(e.to);
            (m[e.to] = m[e.to] || new Set()).add(e.from);
        }
        return m;
    }, [edges]);

    // Zoom inicial que ENQUADRA o grafo no viewport (auto-fit), seja qual for a
    // contagem/dispersão de nós — o zoom fixo abria fechado demais em datasets
    // pequenos e cortava os de fora em datasets grandes. Projeta na rotação
    // inicial e escolhe o zoom que põe o nó mais externo a ~80% da meia-largura.
    const fitZoom = dc.useMemo(() => {
        const rx = -0.3, ry = 0.5;
        const cy = Math.cos(ry), sy = Math.sin(ry), cx = Math.cos(rx), sx = Math.sin(rx);
        let maxX = 1, maxY = 1;
        for (const n of nodes) {
            const w = world[n.id];
            if (!w) continue;
            const x1 = w.x * cy + w.z * sy;
            const z1 = -w.x * sy + w.z * cy;
            const y2 = w.y * cx - z1 * sx;
            const z2 = w.y * sx + z1 * cx;
            const f = fov / (fov + z2);
            maxX = Math.max(maxX, Math.abs(x1 * f));
            maxY = Math.max(maxY, Math.abs(y2 * f));
        }
        const z = Math.min((VW * 0.40) / maxX, (height * 0.40) / maxY);
        return Math.max(0.4, Math.min(3.5, z));
    }, [world, height, fov]);

    // Estado interativo em REFS (o loop lê; nada disso re-renderiza o VDOM).
    const view = dc.useRef(null);
    if (!view.current) view.current = { rot: { rx: -0.3, ry: 0.5 }, zoom: fitZoom, spinning: autoRotate, hover: null, drag: null, moved: false };
    const svgRef = dc.useRef(null);
    const nodeEls = dc.useRef({});   // id → <g>
    const edgeEls = dc.useRef({});   // índice → <line>
    // Único estado React: o rótulo pausar/girar do rodapé.
    const [spinLabel, setSpinLabel] = dc.useState(autoRotate);

    // ── Loop de animação: escreve transform/opacity direto no DOM ──────
    dc.useEffect(() => {
        let raf = 0, frame = 0, lastOrder = -1;
        const st = view.current;
        const labelShown = {};   // id → bool (só escreve display quando muda)

        const loop = () => {
            raf = requestAnimationFrame(loop);
            frame++;
            if (frame % 2) return;                       // ~30fps é suficiente
            const svg = svgRef.current;
            if (!svg || !svg.isConnected) return;
            if (document.hidden || svg.offsetParent === null) return;  // oculto: nada
            const t = performance.now() / 1000;
            if (st.spinning && !st.drag && !st.hover) st.rot.ry += 0.012;

            const cy = Math.cos(st.rot.ry), sy = Math.sin(st.rot.ry);
            const cx = Math.cos(st.rot.rx), sx = Math.sin(st.rot.rx);
            const proj = {};
            for (let i = 0; i < nodes.length; i++) {
                const n = nodes[i];
                const w = world[n.id];
                if (!w) continue;
                const bob = n.pin ? 0 : Math.sin(t * 1.3 + i * 0.7) * 3.2;
                const yy = w.y + bob;
                const x1 = w.x * cy + w.z * sy;
                const z1 = -w.x * sy + w.z * cy;
                const y2 = yy * cx - z1 * sx;
                const z2 = yy * sx + z1 * cx;
                const f = (fov / (fov + z2)) * st.zoom;
                proj[n.id] = { x: VW / 2 + x1 * f, y: height / 2 + y2 * f, f, depth: z2 };
            }
            const depthFade = (d) => Math.max(0.22, Math.min(1, 1 - d / (fov * 0.9)));
            const hover = st.hover;
            const hoodSet = hover ? neighbors[hover] : null;

            for (let i = 0; i < nodes.length; i++) {
                const n = nodes[i];
                const el = nodeEls.current[n.id];
                const p = proj[n.id];
                if (!el || !p) continue;
                const hovered = hover === n.id;
                const dim = hover && !hovered && !(hoodSet && hoodSet.has(n.id));
                const pulse = n.pin ? 1 : 1 + Math.sin(t * 2 + i * 0.7) * 0.06;
                const s = p.f * pulse * (hovered ? 1.4 : 1);
                const own = n.opacity != null ? n.opacity : 1;
                const o = (dim ? 0.1 : depthFade(p.depth)) * own;
                el.setAttribute("transform", `translate(${p.x} ${p.y}) scale(${s})`);
                el.setAttribute("opacity", o);
                // Label com LOD: só escreve display quando o estado muda.
                const show = !!(n.pin || p.f > 0.8 || hovered);
                if (labelShown[n.id] !== show) {
                    labelShown[n.id] = show;
                    const txt = el.querySelector("text");
                    if (txt) txt.style.display = show ? "" : "none";
                }
            }
            for (let i = 0; i < edges.length; i++) {
                const e = edges[i];
                const el = edgeEls.current[i];
                const a = proj[e.from], b = proj[e.to];
                if (!el || !a || !b) continue;
                const dim = hover && e.from !== hover && e.to !== hover;
                el.setAttribute("x1", a.x); el.setAttribute("y1", a.y);
                el.setAttribute("x2", b.x); el.setAttribute("y2", b.y);
                el.setAttribute("opacity", dim ? 0.05 : 0.5 * Math.min(depthFade(a.depth), depthFade(b.depth)));
            }
            // Ordem de profundidade: re-anexa os <g> de trás p/ frente, só de
            // tempos em tempos e só quando algo gira (appendChild move, é barato
            // em SVG, mas não precisa rodar a cada frame).
            const orderStamp = Math.floor(st.rot.ry * 8) + Math.floor(st.rot.rx * 8) * 1000;
            if (orderStamp !== lastOrder) {
                lastOrder = orderStamp;
                const layer = svg.querySelector("[data-nodes]");
                if (layer) {
                    [...nodes]
                        .filter((n) => proj[n.id])
                        .sort((a, b) => proj[b.id].depth - proj[a.id].depth)
                        .forEach((n) => { const el = nodeEls.current[n.id]; if (el) layer.appendChild(el); });
                }
            }
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [key, world, height, fov, neighbors]);

    if (!nodes.length) return <p style={{ opacity: 0.5 }}>{emptyText}</p>;

    // ── Estrutura estática (re-renderiza só quando dataset/tema mudam) ──
    // Cada nó é um <g> com geometria LOCAL (raios fixos); posição/escala/
    // opacidade vêm do loop. Opacidades relativas de halo/glow/anel são fixas
    // por elemento, então a opacidade do grupo carrega confiança+profundidade.
    return (
        <div style={{ width: "100%", ...style }}>
            <svg
                ref={svgRef}
                viewBox={`0 0 ${VW} ${height}`}
                style={{ width: "100%", display: "block", background: bg, borderRadius: "8px", border: "1px solid var(--background-modifier-border)", touchAction: "none", cursor: "grab" }}
                onPointerDown={(ev) => {
                    const st = view.current;
                    st.drag = { startX: ev.clientX, startY: ev.clientY, rx: st.rot.rx, ry: st.rot.ry };
                    st.moved = false;
                    if (st.spinning) { st.spinning = false; setSpinLabel(false); }
                }}
                onPointerMove={(ev) => {
                    const st = view.current;
                    if (!st.drag) return;
                    st.moved = true;
                    st.rot.ry = st.drag.ry + (ev.clientX - st.drag.startX) * 0.009;
                    st.rot.rx = Math.max(-1.4, Math.min(1.4, st.drag.rx + (ev.clientY - st.drag.startY) * 0.009));
                }}
                onPointerUp={() => { view.current.drag = null; }}
                onPointerLeave={() => { view.current.drag = null; }}
                onWheel={(ev) => {
                    ev.preventDefault();
                    const st = view.current;
                    st.zoom = Math.max(0.4, Math.min(3.5, st.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12)));
                }}
            >
                <g data-edges>
                    {edges.map((e, i) => (
                        <line
                            key={i}
                            ref={(el) => { if (el) edgeEls.current[i] = el; }}
                            stroke={e.color || accent}
                            strokeWidth={1}
                            strokeDasharray={DASH[e.style || "solid"]}
                            opacity={0}
                            style={{ mixBlendMode: blend }}
                        />
                    ))}
                </g>
                <g data-nodes>
                    {nodes.map((n) => {
                        const r = n.size || 8;
                        const c = n.color || accent;
                        const ring = n.ring != null ? n.ring : 0;
                        return (
                            <g
                                key={n.id}
                                ref={(el) => { if (el) nodeEls.current[n.id] = el; }}
                                opacity={0}
                                style={{ cursor: "pointer" }}
                                onPointerUp={() => { if (!view.current.moved && onNodeClick) onNodeClick(n.id); }}
                                onMouseEnter={() => { view.current.hover = n.id; }}
                                onMouseLeave={() => { view.current.hover = null; }}
                            >
                                <title>{n.label || n.id}</title>
                                {!n.ghost && <circle r={r * 3.2} fill={c} opacity={0.05} style={{ mixBlendMode: blend }} />}
                                {!n.ghost && <circle r={r * 1.9} fill={c} opacity={0.13} style={{ mixBlendMode: blend }} />}
                                <circle r={r}
                                    fill={n.ghost ? "transparent" : c}
                                    stroke={n.ghost ? faintColor : c}
                                    strokeWidth={n.ghost ? 1.2 : 0}
                                    strokeDasharray={n.ghost ? "3 2" : "none"}
                                />
                                {ring > 0.02 && !n.ghost && (
                                    <circle r={r * 1.55} fill="none" stroke={c}
                                        strokeWidth={1 + ring} opacity={0.1 + 0.5 * ring}
                                        style={{ mixBlendMode: blend }} />
                                )}
                                <text y={r + 11} fontSize={n.pin ? 10.5 : 9} textAnchor="middle"
                                    fill={n.ghost ? faintColor : textColor} style={{ display: "none" }}>
                                    {(n.label || n.id).length > 20 ? (n.label || n.id).slice(0, 19) + "…" : (n.label || n.id)}
                                </text>
                            </g>
                        );
                    })}
                </g>
            </svg>
            <div style={{ display: "flex", gap: "14px", flexWrap: "wrap", marginTop: "6px", fontSize: "0.75em", opacity: 0.75, alignItems: "center" }}>
                {(legend || []).map((l, i) => (
                    <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: "5px" }}>
                        {l.style ? (
                            <svg width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" stroke={l.color || accent} strokeWidth="2" strokeDasharray={DASH[l.style] || "none"} /></svg>
                        ) : (
                            <span style={{ width: "9px", height: "9px", borderRadius: "50%", background: l.color, display: "inline-block" }} />
                        )}
                        {l.label}
                    </span>
                ))}
                <span style={{ opacity: 0.6 }}>· arraste = orbitar · roda = zoom</span>
                <a style={{ cursor: "pointer", opacity: 0.7 }} onClick={() => {
                    const st = view.current;
                    st.spinning = !st.spinning;
                    setSpinLabel(st.spinning);
                }}>{spinLabel ? "pausar" : "girar"}</a>
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEMO VIEW
// ═══════════════════════════════════════════════════════════════════════════════
function renderedView() {
    const nodes = [
        { id: "core", label: "núcleo", size: 15, pin: "core" },
        { id: "h1", label: "hub A", size: 11, pin: "hub" },
        { id: "h2", label: "hub B", size: 11, pin: "hub" },
        ...Array.from({ length: 10 }, (_, i) => ({
            id: `n${i}`, label: `Nó ${i}`,
            size: 6 + (i % 3), opacity: 0.5 + (i % 5) * 0.12, ring: (i % 4) / 4,
        })),
    ];
    const edges = [
        { from: "core", to: "h1", rest: 60 }, { from: "core", to: "h2", rest: 60 },
        ...nodes.slice(3).map((n, i) => ({ from: i % 2 ? "h1" : "h2", to: n.id, rest: 55 })),
    ];
    return <GloGraph3D nodes={nodes} edges={edges} height={340} />;
}

return { GloGraph3D, renderedView };
