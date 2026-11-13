// NetworkIDS Events — display flows, timeline, and flow details.
function getElement(id) {
    return document.getElementById(id);
}

let currentPage = 1, totalPages = 1, activeFlowId = null;
let activeRequestId = 0, searchTimer = null;
let selectedTime = null, timelineBuckets = [], shapLoadedForFlow = null;
const integrityCache = new Map();

// Check whether a value is missing or invalid.
function missing(value) {
    if (value === null || value === undefined || value === "") {
        return true;
    }

    if (typeof value === "number" && !Number.isFinite(value)) {
        return true;
    }

    if (typeof value === "string") {
        const invalidValues = ["nan", "none", "null", "infinity", "-infinity", "inf", "-inf"];

        if (invalidValues.includes(value.trim().toLowerCase())) {
            return true;
        }
    }
    return false;
}

// Convert values into readable text.
function display(value) {
    if (missing(value)) {
        return "-";
    }
    return String(value);
}

// Format numbers with thousand separators.
function number(value) {
    if (missing(value) || !Number.isFinite(Number(value))) {
        return "-";
    }
    return Number(value).toLocaleString();
}

// Format decimal numbers.
function float(value, digits = 2) {
    if (missing(value) || !Number.isFinite(Number(value))) {
        return "-";
    }
    return Number(value).toFixed(digits);
}

// Convert probabilities into percentages.
function probability(value) {
    if (missing(value) || !Number.isFinite(Number(value))) {
        return "-";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
}

// Format flow duration in seconds.
function duration(value) {
    if (missing(value) || !Number.isFinite(Number(value))) {
        return "-";
    }
    return `${Number(value).toFixed(2)}s`;
}

// Convert Unix timestamps into readable date and time.
function timestamp(v) {
    if (missing(v) || !Number.isFinite(Number(v))) return "-";
    const d = new Date(Number(v) * 1000);
    if (!Number.isFinite(d.getTime())) return "-";

    function pad(n, length = 2) {
        return String(n).padStart(length, "0");
    }

    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(),3)}`;
}

// Show or clear error messages.
function showError(msg) {
    getElement("eventsError").textContent = msg;
    getElement("eventsError").classList.remove("hidden");
}
function clearError() {
    getElement("eventsError").textContent = "";
    getElement("eventsError").classList.add("hidden");
}

// Fetch and validate JSON responses from Flask.
async function api(path) {
    const response = await fetch(path, { cache: "no-store" });
    const data = await response.json();

    if (!response.ok || !data.success) {
        throw new Error(data.error || `Request failed: ${response.status}`);
    }

    return data;
}

// Apply the selected light or dark theme.
function applyTheme(mode) {
    document.documentElement.setAttribute("data-theme", mode);
    getElement("themeToggle").textContent = mode === "light" ? "☀" : "☾";
    localStorage.setItem("networkids-theme", mode);
}

applyTheme(
    ["dark", "light"].includes(localStorage.getItem("networkids-theme"))
        ? localStorage.getItem("networkids-theme")
        : "dark"
);

getElement("themeToggle").addEventListener("click", function() {
    applyTheme(
        document.documentElement.getAttribute("data-theme") === "dark"
            ? "light"
            : "dark"
    );
});

// Create a table cell with the given value.
function cell(row, value, className = "") {
    const td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = value;
    row.appendChild(td);
    return td;
}

// Create a coloured status label.
function label(text, cssClass, baseClass = "prediction-badge") {
    const el = document.createElement("span");
    el.className = `${baseClass} ${cssClass}`;
    el.textContent = text;
    return el;
}

// Get the integrity status of a flow.
function integrityText(flow) {
    return integrityCache.get(String(flow.flow_id)) ||
        (missing(flow.integrity_state) ? "-" : flow.integrity_state);
}

// Display flow records in the Events table.
function renderRows(rows) {
    const body = getElement("flowTableBody");
    body.replaceChildren();

    if (!rows.length) {
        const tr = document.createElement("tr");
        const td = cell(tr, "No matching flows.");
        td.colSpan = 13;
        body.appendChild(tr);
        return;
    }

    for (const flow of rows) {
        const tr = document.createElement("tr");

        cell(tr, timestamp(flow.timestamp), "timestamp-cell");
        cell(tr, display(flow.src_ip));
        cell(tr, display(flow.src_port));
        cell(tr, display(flow.dst_ip));
        cell(tr, display(flow.dst_port));
        cell(tr, display(flow.protocol));

        const predictionCell = document.createElement("td");
        const predicted = display(flow.prediction);

        predictionCell.appendChild(
            label(
                predicted,
                predicted === "Attack"
                    ? "prediction-attack"
                    : predicted === "Benign"
                        ? "prediction-benign"
                        : ""
            )
        );

        tr.appendChild(predictionCell);

        cell(tr, probability(flow.attack_probability));
        cell(tr, number(flow.packets));
        cell(tr, number(flow.bytes));
        cell(tr, duration(flow.duration));

        const integrityCell = document.createElement("td");
        const integrity = integrityText(flow);
        const status = label(integrity, integrity.toLowerCase(), "integrity-inline");

        integrityCell.appendChild(status);
        tr.appendChild(integrityCell);

        const action = document.createElement("td");
        const view = document.createElement("button");

        view.className = "button button-small button-secondary view-flow";
        view.type = "button";
        view.textContent = "View";

        view.addEventListener("click", function() {
            openFlowModal(flow.flow_id, status);
        });

        action.appendChild(view);
        tr.appendChild(action);
        body.appendChild(tr);
    }
}

// Load flows using the current filters and page number.
async function loadFlows() {
    // Ignore older API responses if the user changes filters quickly.
    const requestId = ++activeRequestId;

    const params = new URLSearchParams({
        page: currentPage,
        page_size: 25,
        prediction: getElement("predictionFilter").value,
        protocol: getElement("protocolFilter").value,
        q: getElement("searchInput").value.trim()
    });

    if (selectedTime) {
        params.set("from_ts", selectedTime.start);
        params.set("to_ts", selectedTime.end);
    }

    try {
        const data = await api(`/api/flows?${params}`);

        if (requestId !== activeRequestId) return;

        currentPage = data.pagination.page;
        totalPages = data.pagination.total_pages;

        renderRows(data.rows);

        getElement("pageInformation").textContent =
            `Page ${currentPage} of ${totalPages}`;

        getElement("flowCountInformation").textContent =
            `${number(data.pagination.total_rows)} ${data.pagination.total_rows === 1 ? "flow" : "flows"}`;

        getElement("previousPage").disabled = currentPage <= 1;
        getElement("nextPage").disabled = currentPage >= totalPages;

        clearError();
    } catch (err) {
        if (requestId === activeRequestId) {
            showError(err.message);
        }
    }
}

// Reset to the first page when filters change.
function reload() {
    currentPage = 1;
    loadFlows();
}

// Handle search, dropdown filters, and pagination.
getElement("searchInput").addEventListener("input", function() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(reload, 350);
});

getElement("predictionFilter").addEventListener("change", reload);
getElement("protocolFilter").addEventListener("change", reload);

getElement("previousPage").addEventListener("click", function() {
    if (currentPage > 1) {
        currentPage--;
        loadFlows();
    }
});

getElement("nextPage").addEventListener("click", function() {
    if (currentPage < totalPages) {
        currentPage++;
        loadFlows();
    }
});

// Render the flow timeline using actual backend counts.
function renderTimeline(buckets) {
    timelineBuckets = buckets;

    const host = getElement("timelineChart");
    host.replaceChildren();

    if (!buckets.length) {
        host.textContent = "No capture timestamps are available.";
        return;
    }

    // SVG is used to draw the timeline directly in the browser.
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");

    svg.setAttribute("viewBox", "0 0 1000 240");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Attack and benign network flows over time");

    // Create an SVG element and apply its attributes.
    function make(tag, attributes) {
        const e = document.createElementNS(ns, tag);

        for (const [k, v] of Object.entries(attributes)) {
            e.setAttribute(k, String(v));
        }

        svg.appendChild(e);
        return e;
    }

    const left = 52;
    const right = 986;
    const top = 20;
    const bottom = 205;
    const plotH = bottom - top;

    const max = Math.max(
        1,
        ...buckets.map(function(b) {
            return Number(b.attack || 0) + Number(b.benign || 0);
        })
    );

    // Draw the timeline gridlines and Y-axis labels.
    for (let i = 0; i <= 4; i++) {
        const y = bottom - (plotH * i / 4);

        make("line", {
            x1: left,
            y1: y,
            x2: right,
            y2: y,
            class: "timeline-grid"
        });

        const t = make("text", {
            x: 42,
            y: y + 4,
            "text-anchor": "end",
            class: "timeline-axis"
        });

        t.textContent = number(Math.round(max * i / 4));
    }

    const step = (right - left) / buckets.length;
    const barW = Math.max(4, Math.min(42, step * .74));

    // Draw stacked benign and attack bars for each time bucket.
    buckets.forEach(function(b, i) {
        const x = left + i * step + (step - barW) / 2;
        const benignH = Number(b.benign || 0) / max * plotH;
        const attackH = Number(b.attack || 0) / max * plotH;

        const group = document.createElementNS(ns, "g");
        group.classList.add("timeline-bar");

        if (selectedTime && Number(selectedTime.start) === Number(b.start)) {
            group.classList.add("selected");
        }

        for (const [y, h, color] of [
            [bottom - benignH, benignH, "var(--benign)"],
            [bottom - benignH - attackH, attackH, "var(--attack)"]
        ]) {
            const rect = document.createElementNS(ns, "rect");

            rect.setAttribute("x", String(x));
            rect.setAttribute("y", String(y));
            rect.setAttribute("width", String(barW));
            rect.setAttribute("height", String(Math.max(0, h)));
            rect.setAttribute("fill", color);

            group.appendChild(rect);
        }

        const title = document.createElementNS(ns, "title");
        title.textContent =
            `${timestamp(b.start)}\nAttack: ${b.attack}\nBenign: ${b.benign}`;

        group.appendChild(title);

        // Transparent area makes each time bar easier to click.
        const hit = document.createElementNS(ns, "rect");

        for (const [k, v] of Object.entries({
            x,
            y: top,
            width: Math.max(step, barW),
            height: plotH,
            fill: "transparent"
        })) {
            hit.setAttribute(k, String(v));
        }

        group.appendChild(hit);

        // Clicking a bar filters the flow table by its time range.
        group.addEventListener("click", function() {
            if (selectedTime && Number(selectedTime.start) === Number(b.start)) {
                selectedTime = null;
            } else {
                selectedTime = {
                    start: b.start,
                    end: b.end
                };
            }

            getElement("clearTimeFilter").classList.toggle("hidden", !selectedTime);

            getElement("timeFilterInfo").textContent = selectedTime
                ? `Selected: ${timestamp(b.start)} – ${timestamp(b.end)}`
                : "All capture times";

            renderTimeline(timelineBuckets);
            reload();
        });

        svg.appendChild(group);
    });

    // Display up to five timestamp labels on the X-axis.
    const tickCount = Math.min(5, buckets.length);

    for (let i = 0; i < tickCount; i++) {
        const idx = tickCount === 1
            ? 0
            : Math.round((buckets.length - 1) * i / (tickCount - 1));

        const t = make("text", {
            x: left + idx * step + step / 2,
            y: 228,
            "text-anchor": "middle",
            class: "timeline-axis"
        });

        t.textContent = timestamp(buckets[idx].start).slice(11, 16);
    }

    host.appendChild(svg);
}

// Clear the selected timeline filter.
getElement("clearTimeFilter").addEventListener("click", function() {
    selectedTime = null;

    getElement("clearTimeFilter").classList.add("hidden");
    getElement("timeFilterInfo").textContent = "All capture times";

    renderTimeline(timelineBuckets);
    reload();
});

// Display flow information inside the modal.
function appendDetails(id, pairs) {
    const container = getElement(id);
    container.replaceChildren();

    for (const [name, value] of pairs) {
        const box = document.createElement("div");
        box.className = "detail-item";

        const caption = document.createElement("span");
        caption.textContent = name;

        const strong = document.createElement("strong");
        strong.textContent = value;

        box.append(caption, strong);
        container.appendChild(box);
    }
}

// Reset modal content before viewing another flow.
function resetModal() {
    for (const name of ["shap", "packet"]) {
        getElement(`${name}Content`).classList.add("hidden");
        getElement(`${name}Arrow`).classList.remove("open");
        getElement(`${name}Toggle`).setAttribute("aria-expanded", "false");
    }

    getElement("shapResults").replaceChildren();
    getElement("shapResults").classList.add("hidden");

    getElement("shapLoading").classList.remove("hidden");
    getElement("shapLoading").textContent =
        "SHAP explanation will be generated when opened.";

    getElement("packetTableBody").replaceChildren();

    getElement("integrityLabels").replaceChildren();
    getElement("integrityLabels").textContent = "Loading integrity findings...";

    shapLoadedForFlow = null;
    relatedPacketsLoadedForFlow = null;
    relatedPacketsLoadingForFlow = null;
}

// Open the modal and load the selected flow details.
async function openFlowModal(flowId, tableBadge) {
    const id = String(flowId);
    activeFlowId = id;

    resetModal();

    getElement("flowModal").classList.remove("hidden");
    document.body.style.overflow = "hidden";

    getElement("modalFlowTime").textContent = "Loading flow...";
    getElement("flowInformation").textContent = "Loading...";
    getElement("modalPrediction").textContent = "-";
    getElement("modalProbability").textContent = "-";

    try {
        const data = await api(`/api/flow/${encodeURIComponent(id)}`);

        // Prevent an old response from replacing another flow's details.
        if (activeFlowId !== id) return;

        const f = data.flow;

        getElement("modalFlowTime").textContent = timestamp(f.timestamp);
        getElement("modalPrediction").textContent = display(f.prediction);
        getElement("modalProbability").textContent = probability(f.attack_probability);

        appendDetails("flowInformation", [
            ["Timestamp", timestamp(f.timestamp)],
            ["Source IP", display(f.src_ip)],
            ["Source Port", display(f.src_port)],
            ["Destination IP", display(f.dst_ip)],
            ["Destination Port", display(f.dst_port)],
            ["Protocol", display(f.protocol)],
            ["Average TTL", float(f.avg_ttl)],
            ["Flow Packets", number(f.packets)],
            ["Flow Bytes", number(f.bytes)],
            ["Flow Duration", duration(f.duration)],
            ["Forward Packets", number(f.forward_packets)],
            ["Reverse Packets", number(f.reverse_packets)],
            ["SYN Count", number(f.syn_count)],
            ["ACK Count", number(f.ack_count)],
            ["RST Count", number(f.rst_count)]
        ]);

        renderIntegrityLabels(f.integrity_explanation || {
            integrity_state: f.integrity_state,
            triggered_rules: [],
            additional_indicators: []
        });

        // Load related packets without delaying the integrity labels.
        void fetchRelatedPackets(id);

    } catch (err) {
        if (activeFlowId === id) {
            getElement("integrityLoading").textContent = err.message;

            if (getElement("flowInformation").textContent === "Loading...") {
                getElement("flowInformation").textContent = err.message;
            }
        }
    }
}

// Display the triggered integrity rules and indicators.
function renderIntegrityLabels(result) {
    const root = getElement("integrityLabels");
    root.replaceChildren();

    const state = display(result.integrity_state);
    const rules = Array.isArray(result.triggered_rules)
        ? result.triggered_rules
        : [];
    const indicators = Array.isArray(result.additional_indicators)
        ? result.additional_indicators
        : [];

    // Create a colour-coded integrity badge.
    function addLabel(category, text) {
        const badge = document.createElement("span");
        badge.className = `integrity-reason-label ${category.toLowerCase()}`;
        badge.textContent = text;
        root.appendChild(badge);
    }

    if (state === "-" && rules.length === 0 && indicators.length === 0) {
        addLabel("unknown", "-");
        return;
    }

    if (rules.length === 0) {
        // Indicator-only results do not change the integrity state.
        addLabel(state, state);
    } else {
        for (const rule of rules) {
            const category = display(rule.category);

            addLabel(
                category,
                `${category}: ${display(rule.rule_id)} - ${display(rule.description)}`
            );
        }
    }

    for (const rule of indicators) {
        addLabel(
            "indicator",
            `Indicator: ${display(rule.rule_id)} - ${display(rule.description)}`
        );
    }
}

let relatedPacketsLoadedForFlow = null;
let relatedPacketsLoadingForFlow = null;

// Fetch packets associated with the selected flow.
async function fetchRelatedPackets(flowId) {
    // Avoid duplicate requests for the same flow.
    if (
        relatedPacketsLoadedForFlow === flowId ||
        relatedPacketsLoadingForFlow === flowId
    ) {
        return;
    }

    relatedPacketsLoadingForFlow = flowId;

    getElement("packetTableBody").replaceChildren();

    const tr = document.createElement("tr");
    const td = cell(tr, "Loading related packets...");
    td.colSpan = 9;

    getElement("packetTableBody").appendChild(tr);

    try {
        const related = await api(
            `/api/flow/${encodeURIComponent(flowId)}/packets`
        );

        if (activeFlowId !== flowId) return;

        renderPackets(related.result.packets || []);
        relatedPacketsLoadedForFlow = flowId;

    } catch (err) {
        if (activeFlowId === flowId) {
            getElement("packetTableBody").replaceChildren();

            const row = document.createElement("tr");
            const error = cell(row, `Unable to load packets: ${err.message}`);
            error.colSpan = 9;

            getElement("packetTableBody").appendChild(row);
        }
    } finally {
        relatedPacketsLoadingForFlow = (
            relatedPacketsLoadingForFlow === flowId
        ) ? null : relatedPacketsLoadingForFlow;
    }
}

// Convert checksum values into readable status.
function checksum(v) {
    if (missing(v)) return "-";
    return Number(v) === 1 ? "Valid" : Number(v) === 0 ? "Invalid" : "-";
}

// Display related packets inside the modal table.
function renderPackets(packets) {
    const body = getElement("packetTableBody");
    body.replaceChildren();

    if (!packets.length) {
        const tr = document.createElement("tr");
        const td = cell(tr, "No related packet records found.");
        td.colSpan = 9;
        body.appendChild(tr);
        return;
    }

    for (const p of packets) {
        const tr = document.createElement("tr");

        for (const value of [
            display(p.packet_id),
            display(p.source),
            display(p.destination),
            display(p.ttl),
            display(p.flags),
            checksum(p.ip_checksum),
            checksum(p.l4_checksum),
            display(p.integrity_state),
            [
                ...(p.triggered_rules || []),
                ...(p.indicator_rules || [])
            ].join(", ") || "-"
        ]) {
            cell(tr, value);
        }

        body.appendChild(tr);
    }
}

// Display SHAP feature contributions for a flow.
function renderShap(explanation) {
    getElement("shapLoading").classList.add("hidden");

    const root = getElement("shapResults");
    root.replaceChildren();
    root.classList.remove("hidden");

    const description = document.createElement("div");
    description.className = "shap-explanation-box";

    const title = document.createElement("span");
    title.className = "shap-explanation-title";
    title.textContent = "Simple explanation";

    const p = document.createElement("p");
    p.textContent = display(explanation.plain_explanation);

    description.append(title, p);
    root.appendChild(description);

    const note = document.createElement("p");
    note.className = "shap-description";
    note.textContent =
        "Negative contributions push toward Benign; positive contributions push toward Attack.";

    root.appendChild(note);

    const contributions = explanation.top_contributions || [];

    // Scale each SHAP bar against the largest contribution.
    const max = Math.max(
        .000001,
        ...contributions.map(function(c) {
            return Number(c.absolute_shap) || 0;
        })
    );

    for (const c of contributions) {
        const row = document.createElement("div");
        row.className = "shap-row";

        const feature = document.createElement("div");
        feature.className = "shap-feature";
        feature.textContent = display(c.feature_label || c.feature);

        const value = document.createElement("div");
        value.className = "shap-value";
        value.textContent = display(c.feature_value);

        const track = document.createElement("div");
        track.className = "shap-track";

        const fill = document.createElement("div");
        fill.className = `shap-fill ${Number(c.shap_value) >= 0 ? "shap-fill-positive" : "shap-fill-negative"}`;
        fill.style.width =
            `${Math.min(50, Math.abs(Number(c.shap_value) || 0) / max * 50)}%`;

        track.appendChild(fill);

        const direction = document.createElement("div");
        direction.className = "shap-direction";
        direction.textContent = float(c.shap_value, 4);

        row.append(feature, value, track, direction);
        root.appendChild(row);
    }
}

// Generate SHAP explanations when the accordion opens.
getElement("shapToggle").addEventListener("click", async function() {
    const id = activeFlowId;
    if (!id) return;

    const content = getElement("shapContent");
    const opening = content.classList.contains("hidden");

    content.classList.toggle("hidden");
    getElement("shapArrow").classList.toggle("open", opening);
    getElement("shapToggle").setAttribute("aria-expanded", String(opening));

    // Do not generate SHAP again if it has already been loaded.
    if (!opening || shapLoadedForFlow === id) return;

    getElement("shapLoading").textContent = "Generating SHAP explanation...";

    try {
        const data = await api(
            `/api/flow/${encodeURIComponent(id)}/explain`
        );

        if (activeFlowId !== id) return;

        renderShap(data.explanation);
        shapLoadedForFlow = id;

    } catch (err) {
        if (activeFlowId === id) {
            getElement("shapLoading").textContent =
                `Unable to generate SHAP explanation: ${err.message}`;
        }
    }
});

// Expand or collapse the related packets section.
getElement("packetToggle").addEventListener("click", function() {
    const content = getElement("packetContent");
    const opening = content.classList.contains("hidden");

    content.classList.toggle("hidden");
    getElement("packetArrow").classList.toggle("open", opening);
    getElement("packetToggle").setAttribute("aria-expanded", String(opening));
});

// Close and reset the flow details modal.
function closeModal() {
    getElement("flowModal").classList.add("hidden");
    document.body.style.overflow = "";
    activeFlowId = null;
    resetModal();
}

getElement("closeModal").addEventListener("click", closeModal);
document.querySelector(".modal-backdrop").addEventListener("click", closeModal);

document.addEventListener("keydown", function(e) {
    if (
        e.key === "Escape" &&
        !getElement("flowModal").classList.contains("hidden")
    ) {
        closeModal();
    }
});

// Load the source name, timeline, and flows when the page opens.
(async function() {
    try {
        const summary = await api("/api/summary");
        getElement("sourceName").textContent =
            display(summary.summary.source_name);

        const timeline = await api("/api/events/timeline");
        renderTimeline(timeline.buckets || []);

        await loadFlows();

    } catch (err) {
        showError(err.message);
        getElement("timelineChart").textContent = "Timeline unavailable.";
    }
})();

// Create custom dropdowns that remain visible in screen recordings.
(function setupRecordedDropdowns() {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "/static/events-dropdown.css";
    document.head.appendChild(css);

    const all = [];

    for (const id of ["predictionFilter", "protocolFilter"]) {
        const select = getElement(id);
        if (!select) continue;

        // Hide the native select while keeping its filter value.
        const wrapper = document.createElement("div");
        wrapper.className = "recordable-select";

        select.parentNode.insertBefore(wrapper, select);
        wrapper.appendChild(select);

        select.classList.add("recordable-native-select");
        select.tabIndex = -1;
        select.setAttribute("aria-hidden", "true");

        const button = document.createElement("button");
        button.type = "button";
        button.className = "recordable-select-trigger";
        button.setAttribute("aria-haspopup", "listbox");
        button.setAttribute("aria-expanded", "false");
        button.setAttribute(
            "aria-label",
            id === "predictionFilter"
                ? "Filter by prediction"
                : "Filter by protocol"
        );

        const selected = document.createElement("span");
        selected.className = "recordable-select-label";

        const chevron = document.createElement("span");
        chevron.className = "recordable-select-chevron";
        chevron.setAttribute("aria-hidden", "true");

        // Draw the dropdown arrow using SVG.
        const arrowSvg = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "svg"
        );

        arrowSvg.setAttribute("viewBox", "0 0 16 16");
        arrowSvg.setAttribute("width", "14");
        arrowSvg.setAttribute("height", "14");
        arrowSvg.setAttribute("fill", "none");
        arrowSvg.setAttribute("stroke", "currentColor");
        arrowSvg.setAttribute("stroke-width", "1.8");
        arrowSvg.setAttribute("stroke-linecap", "round");
        arrowSvg.setAttribute("stroke-linejoin", "round");

        const arrowPath = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path"
        );

        arrowPath.setAttribute("d", "M3.5 6l4.5 4 4.5-4");
        arrowSvg.appendChild(arrowPath);
        chevron.appendChild(arrowSvg);

        button.append(selected, chevron);

        const menu = document.createElement("div");
        menu.className = "recordable-select-menu hidden";
        menu.setAttribute("role", "listbox");
        menu.setAttribute(
            "aria-label",
            id === "predictionFilter"
                ? "Prediction options"
                : "Protocol options"
        );

        const options = [];

        // Create options based on the original HTML select.
        for (const option of select.options) {
            const item = document.createElement("button");
            item.type = "button";
            item.className = "recordable-select-option";
            item.setAttribute("role", "option");
            item.dataset.value = option.value;
            item.textContent = option.textContent;

            item.addEventListener("click", function() {
                select.value = option.value;

                // Trigger the existing flow filter event.
                select.dispatchEvent(
                    new Event("change", { bubbles: true })
                );

                update();
                close();
                button.focus();
            });

            menu.appendChild(item);
            options.push(item);
        }

        wrapper.append(button, menu);

        // Update the displayed selected option.
        function update() {
            selected.textContent =
                select.selectedOptions[0]?.textContent || "Select";

            for (const item of options) {
                const isSelected = item.dataset.value === select.value;

                item.classList.toggle("is-selected", isSelected);
                item.setAttribute("aria-selected", String(isSelected));
            }
        }

        // Close the dropdown menu.
        function close() {
            menu.classList.add("hidden");
            button.setAttribute("aria-expanded", "false");
            wrapper.classList.remove("is-open");
        }

        // Open the dropdown and close other menus.
        function open() {
            for (const other of all) {
                other.close();
            }

            menu.classList.remove("hidden");
            button.setAttribute("aria-expanded", "true");
            wrapper.classList.add("is-open");
        }

        // Open or close the menu when the button is clicked.
        button.addEventListener("click", function() {
            if (menu.classList.contains("hidden")) {
                open();
            } else {
                close();
            }
        });

        // Use arrow keys to navigate dropdown options.
        button.addEventListener("keydown", function(event) {
            if (
                event.key === "ArrowDown" ||
                event.key === "ArrowUp"
            ) {
                event.preventDefault();
                open();

                const index = Math.max(
                    0,
                    options.findIndex(function(item) {
                        return item.dataset.value === select.value;
                    })
                );

                options[
                    event.key === "ArrowDown"
                        ? index
                        : Math.max(0, index - 1)
                ].focus();
            }
        });

        menu.addEventListener("keydown", function(event) {
            const index = options.indexOf(document.activeElement);

            if (event.key === "Escape") {
                event.preventDefault();
                close();
                button.focus();

            } else if (
                event.key === "ArrowDown" ||
                event.key === "ArrowUp"
            ) {
                event.preventDefault();

                // Wrap around when reaching the first or last option.
                const next = (
                    index +
                    (event.key === "ArrowDown" ? 1 : -1) +
                    options.length
                ) % options.length;

                options[next].focus();

            } else if (
                event.key === "Home" ||
                event.key === "End"
            ) {
                event.preventDefault();

                options[
                    event.key === "Home"
                        ? 0
                        : options.length - 1
                ].focus();

            } else if (event.key === "Tab") {
                close();
            }
        });

        select.addEventListener("change", update);

        all.push({ wrapper, close });
        update();
    }

    // Close dropdowns when clicking outside.
    document.addEventListener("pointerdown", function(event) {
        for (const entry of all) {
            if (!entry.wrapper.contains(event.target)) {
                entry.close();
            }
        }
    });

    // Close dropdowns when Escape is pressed.
    document.addEventListener("keydown", function(event) {
        if (event.key === "Escape") {
            for (const entry of all) {
                entry.close();
            }
        }
    });
})();