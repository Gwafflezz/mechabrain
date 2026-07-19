// ═══════════════════════════════════════════════════════════════════════════════
// DC-GLO-STAT - Global Themed Stat Card
// Número grande + rótulo, centralizados, para linhas de métricas em dashboards.
// Compõe o GloCard (variant="glass") — nenhum estilo de card próprio: o chrome
// é o da família; este componente só define o CONTEÚDO de uma métrica.
// ═══════════════════════════════════════════════════════════════════════════════

const { useTheme } = await dc.require(dc.fileLink("System/Scripts/Core/dc-themeProvider.jsx"));
const { GloCard } = await dc.require(dc.fileLink("System/Scripts/Componentes/dc-gloCard.jsx"));
const { useComponentCSS } = await dc.require(
    dc.fileLink("System/Scripts/Componentes/dc-gloButton.jsx")
);

const STATUS_COLORS = {
    success: "#33ff00",
    warning: "#ff9900",
    error: "#ff0000",
    info: "#0099ff",
    neutral: "#888888",
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT: GloStat
// ═══════════════════════════════════════════════════════════════════════════════
function GloStat({
    value,                        // Número/texto principal (obrigatório)
    label = null,                 // Rótulo pequeno abaixo do valor
    sub = null,                   // Linha extra (ex.: variação, contexto)
    icon = null,                  // Emoji no canto
    status = null,                // "success" | "warning" | "error" | "info" | "neutral"
    color = null,                 // Cor custom do VALOR (sobrepõe status; default: accent)
    size = "medium",              // "small" | "medium" | "large"
    onClick = null,               // Torna o cartão clicável
    pulse = false,                // Anima o valor (dc-pulse-anim)
    style = {},
}) {
    useComponentCSS();
    const { theme } = useTheme();
    const accent =
        color || (status && STATUS_COLORS[status]) ||
        theme?.["color-accent"] || "var(--interactive-accent)";

    const valueSize = { small: "1.3em", medium: "1.8em", large: "2.4em" }[size] || "1.8em";

    return (
        <GloCard
            variant="glass"
            size="small"
            clickable={!!onClick}
            onClick={onClick || undefined}
            glow={!!onClick}
            glowColor={accent}
            style={{ minWidth: 0, ...style }}
            bodyStyle={{ padding: "8px 12px" }}
        >
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
                <span className={pulse ? "dc-pulse-anim" : ""} style={{ fontSize: valueSize, fontWeight: 700, color: accent, lineHeight: 1.1 }}>
                    {value}{icon && <span style={{ fontSize: "0.55em", opacity: 0.8, marginLeft: "4px" }}>{icon}</span>}
                </span>
                {label && (
                    <div style={{ fontSize: "0.72em", textTransform: "uppercase", letterSpacing: "0.06em", opacity: 0.6, marginTop: "2px" }}>
                        {label}
                    </div>
                )}
                {sub && <div style={{ fontSize: "0.75em", opacity: 0.5, marginTop: "2px" }}>{sub}</div>}
            </div>
        </GloCard>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP: grade responsiva de stats
// ═══════════════════════════════════════════════════════════════════════════════
function GloStatGroup({ children, min = "110px", gap = "8px", style = {} }) {
    return (
        <div style={{ display: "grid", gridTemplateColumns: `repeat(auto-fit, minmax(${min}, 1fr))`, gap, ...style }}>
            {children}
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEMO VIEW
// ═══════════════════════════════════════════════════════════════════════════════
function renderedView() {
    return (
        <GloStatGroup>
            <GloStat value={42} label="Ativas" />
            <GloStat value={7} label="Pendentes" status="warning" />
            <GloStat value="99%" label="Saúde" status="success" pulse />
            <GloStat value={3} label="Erros" status="error" onClick={() => new Notice("clicado")} />
        </GloStatGroup>
    );
}

return { GloStat, GloStatGroup, renderedView };
