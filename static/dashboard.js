// NetworkIDS Overview — upload, progress, KPI and chart rendering.
// Chart data comes from the backend, never from placeholder values.
const MAX_FILE_SIZE_BYTES = 512 * 1024 * 1024;
const $ = id => document.getElementById(id);
const stages = ["extract", "build", "headers", "classify"];
let statusTimer = null;
let isAnalyzing = false;
let uploadStart = 0;

function missing(v) {
    return v === null || v === undefined || v === ""
        || (typeof v === "number" && !Number.isFinite(v))
        || (typeof v === "string" && ["nan", "none", "null", "infinity", "-infinity", "inf", "-inf"].includes(v.trim().toLowerCase()));
}
function num(v) { return missing(v) || !Number.isFinite(Number(v)) ? "-" : Number(v).toLocaleString(); }
function percent(v) { return missing(v) || !Number.isFinite(Number(v)) ? "-" : `${Number(v).toFixed(2)}%`; }
function seconds(v) { const n = Math.max(0, Math.floor(Number(v) || 0)); return `${String(Math.floor(n / 60)).padStart(2,"0")}:${String(n % 60).padStart(2,"0")}`; }
function time(v) {
    if (missing(v) || !Number.isFinite(Number(v))) return "-";
    const d = new Date(Number(v) * 1000);
    if (Number.isNaN(d.getTime())) return "-";
    const pad = n => String(n).padStart(2,"0");
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
function showError(message) { $("errorBox").textContent = message; $("errorBox").classList.remove("hidden"); }
function hideError() { $("errorBox").textContent = ""; $("errorBox").classList.add("hidden"); }
function formatSize(n) { return n >= 1048576 ? `${(n/1048576).toFixed(1)} MB` : `${(n/1024).toFixed(1)} KB`; }
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("themeToggle").textContent = theme === "light" ? "☀" : "☾";
    localStorage.setItem("networkids-theme", theme);
}
applyTheme(["light","dark"].includes(localStorage.getItem("networkids-theme")) ? localStorage.getItem("networkids-theme") : "dark");
$("themeToggle").addEventListener("click", () => applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

function resetFile() {
    $("pcapFile").value = "";
    $("selectedFilePanel").classList.add("hidden");
    $("analyzeUploadButton").disabled = true;
    hideError();
}
function setFile(file) {
    hideError();
    if (!file) { resetFile(); return; }
    if (!/\.(pcap|pcapng)$/i.test(file.name)) { resetFile(); showError("Unsupported file type. Please select a .pcap or .pcapng file."); return; }
    if (file.size > MAX_FILE_SIZE_BYTES) { resetFile(); showError("The selected PCAP exceeds the 512 MB upload limit."); return; }
    $("selectedFile").textContent = file.name;
    $("selectedFileSize").textContent = formatSize(file.size);
    $("selectedFilePanel").classList.remove("hidden");
    $("analyzeUploadButton").disabled = false;
}
$("pcapFile").addEventListener("change", e => setFile(e.target.files[0]));
$("clearSelectedFile").addEventListener("click", resetFile);
for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    $("uploadDropZone").addEventListener(eventName, e => {
        e.preventDefault(); e.stopPropagation();
        $("uploadDropZone").classList.toggle("drag-over", eventName === "dragenter" || eventName === "dragover");
        if (eventName === "drop" && e.dataTransfer.files.length) {
            const dt = new DataTransfer(); dt.items.add(e.dataTransfer.files[0]);
            $("pcapFile").files = dt.files; setFile(dt.files[0]);
        }
    });
}
function setBusy(busy) {
    isAnalyzing = busy;
    $("pcapFile").disabled = busy;
    $("clearSelectedFile").disabled = busy;
    $("analyzeUploadButton").disabled = busy || !$("pcapFile").files.length;
    $("analyzeUploadButton").textContent = busy ? "Analyzing..." : "Analyze Network Traffic";
}
function progress(data) {
    $("analysisProgress").classList.remove("hidden");
    const stageIndex = stages.indexOf(data.stage);
    stages.forEach((name, i) => {
        const el = $(`step-${name}`);
        el.classList.remove("is-pending","is-active","is-complete","is-error");
        let state = "Waiting", cls = "is-pending";
        if (data.ready || (data.stage === "prepare") || (stageIndex >= 0 && i < stageIndex)) { state = "✓ Complete"; cls = "is-complete"; }
        else if (i === stageIndex && data.error) { state = "✕ Failed"; cls = "is-error"; }
        else if (i === stageIndex && data.analyzing) { state = "● Running"; cls = "is-active"; }
        el.classList.add(cls); el.querySelector(".stage-state").textContent = state;
    });
    $("analysisStatusTitle").textContent = data.error ? "Analysis failed" : data.ready ? "Analysis completed" : "Analyzing network traffic";
    $("analysisStatusText").textContent = data.error || (data.ready ? "The traffic analysis is complete." : data.stage_message || "Preparing analysis...");
    $("analysisElapsed").textContent = `Elapsed: ${seconds(data.elapsed_seconds ?? (Date.now()-uploadStart)/1000)}`;
}
async function getJSON(path, options) {
    const res = await fetch(path, {cache:"no-store", ...options});
    const json = await res.json();
    if (!res.ok || !json.success) throw new Error(json.error || `Request failed: ${res.status}`);
    return json;
}
function stopPolling() { if (statusTimer !== null) clearTimeout(statusTimer); statusTimer = null; }
async function pollStatus() {
    try {
        const state = await getJSON("/api/status");
        progress(state);
        if (state.error) { stopPolling(); setBusy(false); showError(state.error); return; }
        if (state.ready) { stopPolling(); setBusy(false); await showOverview(); return; }
        if (state.analyzing) statusTimer = setTimeout(pollStatus, 900);
    } catch (err) { stopPolling(); setBusy(false); showError(err.message); }
}
$("analyzeUploadButton").addEventListener("click", async () => {
    const file = $("pcapFile").files[0];
    if (!file || isAnalyzing) return;
    hideError(); setBusy(true); uploadStart = Date.now();
    progress({analyzing:true,stage:"extract",stage_message:"Uploading capture and preparing analysis...",elapsed_seconds:0});
    const form = new FormData(); form.append("file",file);
    try {
        const res = await getJSON("/api/analyze/upload", {method:"POST",body:form});
        // Supports either an asynchronous worker or legacy synchronous response.
        if (res.summary) { progress({ready:true}); setBusy(false); await showOverview(); }
        else pollStatus();
    } catch (err) {
        setBusy(false); progress({stage:"extract",error:err.message,elapsed_seconds:(Date.now()-uploadStart)/1000}); showError(err.message);
    }
});
$("newAnalysisButton").addEventListener("click", () => {
    stopPolling(); $("resultsSection").classList.add("hidden"); $("landingSection").classList.remove("hidden");
    $("analysisProgress").classList.add("hidden"); resetFile(); window.scrollTo(0,0);
});

const palette = {attack:"var(--attack)", benign:"var(--benign)", warning:"var(--warning)", tampered:"var(--tampered)", primary:"var(--primary)"};
function empty(container, msg="No available data for this chart.") { container.replaceChildren(); const n=document.createElement("div"); n.className="chart-empty"; n.textContent=msg; container.appendChild(n); }
function donut(id, items, headline, subtext) {
    const container=$(id); container.replaceChildren();
    const positive=items.filter(item => Number(item.value)>0);
    const total=positive.reduce((s,item)=>s+Number(item.value),0);
    if (!total) { empty(container); return; }
    const layout=document.createElement("div"); layout.className="donut-layout";
    const chart=document.createElement("div"); chart.className="donut-chart";
    let angle=0; const stops=[];
    positive.forEach(item=>{const end=angle+Number(item.value)/total*100; stops.push(`${item.color} ${angle}% ${end}%`); angle=end;});
    chart.style.background=`conic-gradient(${stops.join(",")})`;
    const inner=document.createElement("div"); inner.className="donut-inner";
    const n=document.createElement("strong"); n.textContent=headline;
    const label=document.createElement("span");label.textContent=subtext;
    inner.append(n,label); chart.appendChild(inner);
    const legend=document.createElement("div"); legend.className="chart-legend";
    items.forEach(item=>{
        const line=document.createElement("div");line.className="legend-item";
        const name=document.createElement("div");name.className="legend-name";
        const dot=document.createElement("span");dot.className="legend-swatch";dot.style.background=item.color;
        const text=document.createElement("span");text.textContent=item.name;
        name.append(dot,text);const count=document.createElement("strong");count.className="legend-count";count.textContent=num(item.value);
        line.append(name,count);legend.appendChild(line);
    });
    layout.append(chart,legend);container.append(layout);
}
function bars(id, rows, color, valueFormatter=num) {
    const container=$(id);container.replaceChildren();
    const available=Array.isArray(rows)?rows.filter(r=>r && !missing(r.name) && Number(r.value)>0):[];
    if (!available.length) {empty(container);return;}
    const max=Math.max(...available.map(r=>Number(r.value)));
    const list=document.createElement("div");list.className="hbar-list";
    available.forEach(item=>{
        const row=document.createElement("div");
        const labels=document.createElement("div");labels.className="hbar-labels";
        const name=document.createElement("span");name.className="hbar-name";name.textContent=String(item.name);
        const count=document.createElement("span");count.className="hbar-count";count.textContent=valueFormatter(item.value);
        labels.append(name,count);
        const track=document.createElement("div");track.className="hbar-track";
        const fill=document.createElement("div");fill.className="hbar-fill";fill.style.width=`${Number(item.value)/max*100}%`;fill.style.background=color;
        track.append(fill);row.append(labels,track);list.append(row);
    });
    container.append(list);
}
function renderSummary(summary) {
    $("sourceName").textContent=summary.source_name || "-";
    $("totalFlows").textContent=num(summary.total_flows);
    $("attackFlows").textContent=num(summary.attack_flows);
    $("attackRateKpi").textContent=percent(summary.attack_rate);
    $("suspiciousPackets").textContent=num(summary.suspicious_packets);
    $("tamperedPackets").textContent=num(summary.tampered_packets);
    donut("classificationChart",[
        {name:"Attack",value:summary.attack_flows,color:palette.attack},
        {name:"Benign",value:summary.benign_flows,color:palette.benign}
    ], percent(summary.attack_rate),"Attack flows");
    donut("integrityChart",[
        {name:"Intact",value:summary.intact_packets,color:palette.benign},
        {name:"Suspicious",value:summary.suspicious_packets,color:palette.warning},
        {name:"Tampered",value:summary.tampered_packets,color:palette.tampered}
    ],num(summary.total_packets),"Packets");
    const rules=Object.entries(summary.integrity_rule_counts || {}).map(([name,value])=>({name,value})).sort((a,b)=>b.value-a.value).slice(0,8);
    bars("rulesChart",rules,palette.warning);
}
function renderOverview(data) {
    $("captureRange").textContent=missing(data.capture_start)||missing(data.capture_end) ? "-" : `${time(data.capture_start)}  –  ${time(data.capture_end)}`;
    const protocols=(data.protocol_distribution||[]).slice(0,6).map((d,i)=>({name:d.name,value:d.value,color:["var(--primary)","var(--attack)","var(--warning)","var(--benign)","var(--tampered)","var(--muted)"][i]}));
    donut("protocolChart",protocols,num(protocols.reduce((n,x)=>n+Number(x.value||0),0)),"Flows");
    bars("sourceChart",data.top_attack_sources,palette.attack);
    bars("destinationChart",data.top_attack_destinations,palette.primary);
    bars("portsChart",data.top_attack_ports,palette.primary);
    bars("shapChart",data.shap_importance,palette.attack, n=>Number(n).toFixed(5));
    if (data.shap_error) empty($("shapChart"),`SHAP importance unavailable: ${data.shap_error}`);
}
async function showOverview() {
    $("landingSection").classList.add("hidden");$("resultsSection").classList.remove("hidden");
    window.scrollTo(0,0);
    const area=$("overviewError");area.classList.add("hidden");
    try {
        const summary=await getJSON("/api/summary");renderSummary(summary.summary);
        try {const overview=await getJSON("/api/overview");renderOverview(overview.overview);}
        catch(err) {area.textContent=`Some charts could not be loaded: ${err.message}`;area.classList.remove("hidden");}
    } catch(err) {area.textContent=err.message;area.classList.remove("hidden");}
}
// Do not reset an existing analysis on browser refresh.
if (!new URLSearchParams(location.search).has("new")) {
    getJSON("/api/summary").then(showOverview).catch(()=>{});
}
