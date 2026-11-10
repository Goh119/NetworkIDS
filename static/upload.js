// NetworkIDS Upload — PCAP selection, upload, and analysis progress.
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
function showError(message) { const el=$("errorBox"); if(el){el.textContent=message;el.classList.remove("hidden");} }
function hideError() { const el=$("errorBox"); if(el){el.textContent="";el.classList.add("hidden");} }
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
        if (state.ready) { stopPolling(); setBusy(false); window.location.assign("/dashboard"); return; }
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
        if (res.summary) { progress({ready:true}); setBusy(false); window.location.assign("/dashboard"); }
        else pollStatus();
    } catch (err) {
        setBusy(false); progress({stage:"extract",error:err.message,elapsed_seconds:(Date.now()-uploadStart)/1000}); showError(err.message);
    }
});

// Upload page only; a completed analysis redirects to the dashboard.
