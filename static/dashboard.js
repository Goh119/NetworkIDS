// NetworkIDS Dashboard — display analysis results and export PDF.
function getElement(id) {
    return document.getElementById(id);
}

// Check for missing or invalid values.
function missing(v) {
    return v === null || v === undefined || v === ""
        || (typeof v === "number" && !Number.isFinite(v))
        || (typeof v === "string" && ["nan", "none", "null", "infinity", "-infinity", "inf", "-inf"].includes(v.trim().toLowerCase()));
}

// Format numbers, percentages, and timestamps.
function num(v) {
    return missing(v) || !Number.isFinite(Number(v)) ? "-" : Number(v).toLocaleString();
}

function percent(v) {
    return missing(v) || !Number.isFinite(Number(v)) ? "-" : `${Number(v).toFixed(2)}%`;
}

function time(v) {
    if (missing(v) || !Number.isFinite(Number(v))) return "-";
    const d = new Date(Number(v) * 1000);
    if (Number.isNaN(d.getTime())) return "-";
    const pad = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// Apply the selected light or dark theme.
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    getElement("themeToggle").textContent = theme === "light" ? "☀" : "☾";
    localStorage.setItem("networkids-theme", theme);
}

applyTheme(
    ["light", "dark"].includes(localStorage.getItem("networkids-theme"))
        ? localStorage.getItem("networkids-theme")
        : "dark"
);

getElement("themeToggle").addEventListener("click", () => {
    applyTheme(
        document.documentElement.getAttribute("data-theme") === "dark"
            ? "light"
            : "dark"
    );
});

// Fetch and validate JSON responses from Flask.
async function getJSON(path, options = {}) {
    const response = await fetch(path, { cache: "no-store", ...options });
    const data = await response.json();

    if (!response.ok || !data.success) {
        throw new Error(data.error || `Request failed: ${response.status}`);
    }

    return data;
}

// Shared chart colours.
const palette = {
    attack: "var(--attack)",
    benign: "var(--benign)",
    warning: "var(--warning)",
    tampered: "var(--tampered)",
    primary: "var(--primary)"
};

// Display a message when chart data is unavailable.
function empty(container, msg = "No available data for this chart.") {
    container.replaceChildren();

    const n = document.createElement("div");
    n.className = "chart-empty";
    n.textContent = msg;

    container.appendChild(n);
}

// Render a donut chart with its legend.
function donut(id, items, headline, subtext) {
    const container = getElement(id);
    container.replaceChildren();

    const positive = items.filter(item => Number(item.value) > 0);
    const total = positive.reduce((s, item) => s + Number(item.value), 0);

    if (!total) {
        empty(container);
        return;
    }

    const layout = document.createElement("div");
    layout.className = "donut-layout";

    const chart = document.createElement("div");
    chart.className = "donut-chart";

    let angle = 0;
    const stops = [];

    positive.forEach(item => {
        const end = angle + Number(item.value) / total * 100;
        stops.push(`${item.color} ${angle}% ${end}%`);
        angle = end;
    });

    chart.style.background = `conic-gradient(${stops.join(",")})`;

    const inner = document.createElement("div");
    inner.className = "donut-inner";

    const n = document.createElement("strong");
    n.textContent = headline;

    const label = document.createElement("span");
    label.textContent = subtext;

    inner.append(n, label);
    chart.appendChild(inner);

    const legend = document.createElement("div");
    legend.className = "chart-legend";

    items.forEach(item => {
        const line = document.createElement("div");
        line.className = "legend-item";

        const name = document.createElement("div");
        name.className = "legend-name";

        const dot = document.createElement("span");
        dot.className = "legend-swatch";
        dot.style.background = item.color;

        const text = document.createElement("span");
        text.textContent = item.name;

        name.append(dot, text);

        const count = document.createElement("strong");
        count.className = "legend-count";
        count.textContent = num(item.value);

        line.append(name, count);
        legend.appendChild(line);
    });

    layout.append(chart, legend);
    container.append(layout);
}

// Render horizontal bar charts for IPs and SHAP features.
function bars(id, rows, color, valueFormatter = num) {
    const container = getElement(id);
    container.replaceChildren();

    const available = Array.isArray(rows)
        ? rows.filter(r => r && !missing(r.name) && Number(r.value) > 0)
        : [];

    if (!available.length) {
        empty(container);
        return;
    }

    const max = Math.max(...available.map(r => Number(r.value)));
    const list = document.createElement("div");
    list.className = "hbar-list";

    available.forEach(item => {
        const row = document.createElement("div");

        const labels = document.createElement("div");
        labels.className = "hbar-labels";

        const name = document.createElement("span");
        name.className = "hbar-name";
        name.textContent = String(item.name);

        const count = document.createElement("span");
        count.className = "hbar-count";
        count.textContent = valueFormatter(item.value);

        labels.append(name, count);

        const track = document.createElement("div");
        track.className = "hbar-track";

        const fill = document.createElement("div");
        fill.className = "hbar-fill";
        fill.style.width = `${Number(item.value) / max * 100}%`;
        fill.style.background = color;

        track.append(fill);
        row.append(labels, track);
        list.append(row);
    });

    container.append(list);
}

// Convert integrity rule IDs into readable descriptions.
const RULE_DESCRIPTIONS = {
    S1: "Unusually low TTL",
    S2: "MAC-IP inconsistency",
    S3: "TCP SYN + FIN",
    S4: "TCP SYN + RST",
    S5: "TCP FIN + RST",
    S6: "TCP NULL flags",
    S7: "Invalid L4 checksum (indicator only)",
    T1: "Invalid IPv4 header checksum",
    T2: "Contradictory IPv4 fragmentation",
    T3: "Invalid IPv4 header length"
};

// Calculate whole-number ticks for vertical chart axes.
function countAxis(maxValue) {
    const roughStep = Math.max(1, maxValue / 4);
    const magnitude = 10 ** Math.floor(Math.log10(roughStep));
    const step = [1, 2, 5, 10]
        .map(n => n * magnitude)
        .find(n => n >= roughStep) || magnitude * 10;

    const max = Math.max(step, Math.ceil(maxValue / step) * step);

    return {
        max,
        ticks: Array.from(
            { length: Math.round(max / step) + 1 },
            (_, i) => i * step
        )
    };
}

// Render vertical charts for destination ports and integrity rules.
function countColumns(id, rows, options = {}) {
    const container = getElement(id);
    container.replaceChildren();

    const available = (Array.isArray(rows) ? rows : [])
        .filter(row => row && !missing(row.name) && Number(row.value) > 0)
        .slice(0, options.limit || 6);

    if (!available.length) {
        empty(container);
        return;
    }

    const axis = countAxis(Math.max(...available.map(item => Number(item.value))));

    const wrap = document.createElement("div");
    wrap.className = `count-chart ${options.rules ? "count-chart-rules" : "count-chart-ports"}`;

    const layout = document.createElement("div");
    layout.className = "count-chart-layout";

    const yAxis = document.createElement("div");
    yAxis.className = "count-chart-y-axis";

    const plot = document.createElement("div");
    plot.className = "count-chart-plot";

    // Add Y-axis labels and horizontal gridlines.
    axis.ticks.forEach(tick => {
        const label = document.createElement("span");
        label.className = "count-chart-tick";
        label.style.bottom = `${tick / axis.max * 100}%`;
        label.textContent = num(tick);
        yAxis.append(label);

        const gridline = document.createElement("div");
        gridline.className = "count-chart-gridline";
        gridline.style.bottom = `${tick / axis.max * 100}%`;
        plot.append(gridline);
    });

    const bars = document.createElement("div");
    bars.className = "count-chart-bars";
    bars.style.setProperty("--column-count", available.length);

    const xLabels = document.createElement("div");
    xLabels.className = "count-chart-x-labels";
    xLabels.style.setProperty("--column-count", available.length);

    // Draw each bar with its label and hover details.
    available.forEach(item => {
        const value = Number(item.value);
        const shortName = options.rules
            ? item.name.split("_", 1)[0]
            : String(item.name);

        const fullName = options.rules
            ? (RULE_DESCRIPTIONS[shortName] || String(item.name))
            : `Port ${item.name}`;

        const color = options.rules
            ? shortName === "S7"
                ? "var(--primary)"
                : shortName.startsWith("T")
                    ? "var(--tampered)"
                    : "var(--warning)"
            : "var(--primary)";

        const detail = options.rules
            ? `${shortName} — ${fullName} · ${num(value)} triggered packets`
            : `Port ${item.name} · ${num(value)} attack flows`;

        const slot = document.createElement("div");
        slot.className = "count-chart-bar-slot";

        const bar = document.createElement("div");
        bar.className = "count-chart-bar";
        bar.style.height = `${value / axis.max * 100}%`;
        bar.style.background = color;
        bar.title = detail;
        bar.tabIndex = 0;
        bar.setAttribute("role", "img");
        bar.setAttribute("aria-label", detail);

        slot.append(bar);
        bars.append(slot);

        const label = document.createElement("span");
        label.textContent = shortName;
        label.title = detail;
        xLabels.append(label);
    });

    plot.append(bars);
    layout.append(yAxis, plot);
    wrap.append(layout, xLabels);

    // Add axis titles.
    const axisTitles = document.createElement("div");
    axisTitles.className = "count-chart-axis-titles";

    const yTitle = document.createElement("span");
    yTitle.textContent = options.rules ? "Triggered packets" : "Attack flows";

    const xTitle = document.createElement("span");
    xTitle.textContent = options.rules ? "Integrity rule" : "Destination port";

    axisTitles.append(yTitle, xTitle);
    wrap.append(axisTitles);

    // Show full rule descriptions below the chart.
    if (options.rules) {
        const legend = document.createElement("div");
        legend.className = "count-rule-legend";

        available.forEach(item => {
            const code = item.name.split("_", 1)[0];

            const line = document.createElement("div");
            line.className = "count-rule-legend-item";

            const dot = document.createElement("span");
            dot.className = "count-rule-legend-dot";
            dot.style.background = code === "S7"
                ? "var(--primary)"
                : code.startsWith("T")
                    ? "var(--tampered)"
                    : "var(--warning)";

            const text = document.createElement("span");
            text.textContent = `${code} · ${RULE_DESCRIPTIONS[code] || item.name}`;

            const value = document.createElement("strong");
            value.textContent = num(item.value);

            line.append(dot, text, value);
            legend.append(line);
        });

        wrap.append(legend);
    }

    container.append(wrap);
}

// Display KPI values, classification charts, and integrity results.
function renderSummary(summary) {
    getElement("sourceName").textContent = summary.source_name || "-";
    getElement("totalFlows").textContent = num(summary.total_flows);
    getElement("attackFlows").textContent = num(summary.attack_flows);
    getElement("attackRateKpi").textContent = percent(summary.attack_rate);
    getElement("suspiciousPackets").textContent = num(summary.suspicious_packets);
    getElement("tamperedPackets").textContent = num(summary.tampered_packets);

    donut("classificationChart", [
        { name: "Attack", value: summary.attack_flows, color: palette.attack },
        { name: "Benign", value: summary.benign_flows, color: palette.benign }
    ], percent(summary.attack_rate), "Attack flows");

    donut("integrityChart", [
        { name: "Intact", value: summary.intact_packets, color: palette.benign },
        { name: "Suspicious", value: summary.suspicious_packets, color: palette.warning },
        { name: "Tampered", value: summary.tampered_packets, color: palette.tampered }
    ], num(summary.total_packets), "Packets");

    const rules = Object.entries(summary.integrity_rule_counts || {})
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 5);

    countColumns("rulesChart", rules, { rules: true, limit: 5 });
}

// Display protocol, IP, port, and SHAP charts.
function renderOverview(data) {
    getElement("captureRange").textContent =
        missing(data.capture_start) || missing(data.capture_end)
            ? "-"
            : `${time(data.capture_start)}  –  ${time(data.capture_end)}`;

    const protocols = (data.protocol_distribution || [])
        .slice(0, 6)
        .map((d, i) => ({
            name: d.name,
            value: d.value,
            color: [
                "var(--primary)",
                "var(--attack)",
                "var(--warning)",
                "var(--benign)",
                "var(--tampered)",
                "var(--muted)"
            ][i]
        }));

    donut(
        "protocolChart",
        protocols,
        num(protocols.reduce((n, x) => n + Number(x.value || 0), 0)),
        "Flows"
    );

    bars("sourceChart", data.top_attack_sources, palette.attack);
    bars("destinationChart", data.top_attack_destinations, palette.primary);
    countColumns("portsChart", data.top_attack_ports, { limit: 6 });
    bars("shapChart", data.shap_importance, palette.attack, n => Number(n).toFixed(5));

    if (data.shap_error) {
        empty(getElement("shapChart"), `SHAP importance unavailable: ${data.shap_error}`);
    }
}

// Load analysis data and render the dashboard.
async function showOverview() {
    window.scrollTo(0, 0);

    const area = getElement("overviewError");
    area.classList.add("hidden");

    try {
        const summary = await getJSON("/api/summary");
        renderSummary(summary.summary);

        try {
            const overview = await getJSON("/api/overview");
            renderOverview(overview.overview);
        } catch (err) {
            area.textContent = `Some charts could not be loaded: ${err.message}`;
            area.classList.remove("hidden");
        }
    } catch (err) {
        area.textContent = err.message;
        area.classList.remove("hidden");
    }
}

// Load the dashboard when the page opens.
showOverview();

// Export the current analysis as a backend-generated PDF.
const exportPdfButton = getElement("exportPdfButton");

if (exportPdfButton) {
    exportPdfButton.addEventListener("click", async () => {
        const errorBox = getElement("exportPdfError");

        errorBox.classList.add("hidden");
        errorBox.textContent = "";

        exportPdfButton.disabled = true;
        exportPdfButton.textContent = "Generating PDF...";

        try {
            const response = await fetch("/api/report/pdf", {
                cache: "no-store"
            });

            if (!response.ok) {
                let message = `PDF export failed (${response.status}).`;

                try {
                    const result = await response.json();
                    message = result.error || message;
                } catch (_) {
                    // Use the HTTP error if the response is not JSON.
                }

                throw new Error(message);
            }

            // Download the PDF returned by Flask.
            const blob = await response.blob();
            const disposition = response.headers.get("Content-Disposition") || "";
            const nameMatch = disposition.match(/filename="?([^";]+)"?/i);
            const filename = nameMatch
                ? nameMatch[1]
                : "NetworkIDS_Security_Analysis_Report.pdf";

            const downloadUrl = URL.createObjectURL(blob);
            const link = document.createElement("a");

            link.href = downloadUrl;
            link.download = filename;

            document.body.appendChild(link);
            link.click();
            link.remove();

            setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);
        } catch (error) {
            errorBox.textContent = error.message;
            errorBox.classList.remove("hidden");
            errorBox.scrollIntoView({
                behavior: "smooth",
                block: "nearest"
            });
        } finally {
            exportPdfButton.disabled = false;
            exportPdfButton.innerHTML =
                '<span aria-hidden="true">&#8595;</span> Export PDF';
        }
    });
}