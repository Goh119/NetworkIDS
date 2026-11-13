// NetworkIDS Upload — existing PCAP upload and timed live capture.
const MAX_FILE_BYTES = 512 * 1024 * 1024;
const LARGE_FILE_BYTES = 5 * 1024 * 1024;
const ANALYSIS_STAGES = ["extract", "build", "headers", "classify"];

function getElement(id) {
    return document.getElementById(id);
}

let currentMode = "upload";
let busy = false;
let statusTimer = null;
let startedAt = 0;

// Format the time and file size shown to users.
function formatSeconds(value) {
    const total = Math.max(0, Math.floor(Number(value) || 0));
    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}
function formatSize(bytes) {
    return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${(bytes / 1024).toFixed(1)} KB`;
}

// Show errors without replacing the current page.
function showError(message) {
    getElement("errorBox").textContent = message;
    getElement("errorBox").classList.remove("hidden");
}
function hideError() {
    getElement("errorBox").textContent = "";
    getElement("errorBox").classList.add("hidden");
}

// Keep the same light/dark theme as Dashboard and Events.
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    getElement("themeToggle").textContent = theme === "light" ? "☀" : "☾";
    localStorage.setItem("networkids-theme", theme);
}
const savedTheme = localStorage.getItem("networkids-theme");
applyTheme(["light", "dark"].includes(savedTheme) ? savedTheme : "dark");
getElement("themeToggle").addEventListener("click", () => {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
});

// Switch the visible input mode; do not interrupt an active task.
function switchMode(mode) {
    if (busy) return;
    currentMode = mode;
    const live = mode === "live";
    getElement("uploadModePanel").classList.toggle("hidden", live);
    getElement("liveModePanel").classList.toggle("hidden", !live);
    getElement("uploadModeButton").classList.toggle("is-selected", !live);
    getElement("liveModeButton").classList.toggle("is-selected", live);
    getElement("uploadModeButton").setAttribute("aria-pressed", String(!live));
    getElement("liveModeButton").setAttribute("aria-pressed", String(live));
    getElement("analysisProgress").classList.add("hidden");
    hideError();
}
getElement("uploadModeButton").addEventListener("click", () => switchMode("upload"));
getElement("liveModeButton").addEventListener("click", () => switchMode("live"));

// Validate files and show a non-blocking warning for PCAPs >= 5 MB.
function resetFile() {
    if (busy) return;
    getElement("pcapFile").value = "";
    getElement("selectedFilePanel").classList.add("hidden");
    getElement("largeFileWarning").classList.add("hidden");
    getElement("analyzeUploadButton").disabled = true;
    hideError();
}
function setFile(file) {
    hideError();
    if (!file) { resetFile(); return; }
    if (!/\.(pcap|pcapng)$/i.test(file.name)) {
        resetFile();
        showError("Unsupported file type. Please select a .pcap or .pcapng file.");
        return;
    }
    if (file.size > MAX_FILE_BYTES) {
        resetFile();
        showError("The selected PCAP exceeds the 512 MB upload limit.");
        return;
    }
    getElement("selectedFile").textContent = file.name;
    getElement("selectedFileSize").textContent = formatSize(file.size);
    getElement("selectedFilePanel").classList.remove("hidden");
    getElement("largeFileWarning").classList.toggle("hidden", file.size < LARGE_FILE_BYTES);
    getElement("analyzeUploadButton").disabled = busy;
}
getElement("pcapFile").addEventListener("change", event => setFile(event.target.files[0]));
getElement("clearSelectedFile").addEventListener("click", resetFile);

// Support drag-and-drop using the same file validation.
for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
    getElement("uploadDropZone").addEventListener(eventName, event => {
        event.preventDefault();
        event.stopPropagation();
        if (busy) return;
        getElement("uploadDropZone").classList.toggle("drag-over", ["dragenter", "dragover"].includes(eventName));
        if (eventName === "drop" && event.dataTransfer.files.length) {
            const transfer = new DataTransfer();
            transfer.items.add(event.dataTransfer.files[0]);
            getElement("pcapFile").files = transfer.files;
            setFile(transfer.files[0]);
        }
    });
}

// Disable both modes while capturing or processing.
function setBusy(value) {
    busy = value;
    getElement("pcapFile").disabled = value;
    getElement("clearSelectedFile").disabled = value;
    getElement("captureSeconds").disabled = value;
    getElement("uploadModeButton").disabled = value;
    getElement("liveModeButton").disabled = value;
    getElement("analyzeUploadButton").disabled = value || !getElement("pcapFile").files.length;
    getElement("startCaptureButton").disabled = value;
    getElement("analyzeUploadButton").textContent = value ? "Analyzing..." : "Analyze Network Traffic";
    getElement("startCaptureButton").textContent = value ? "Capturing / Analyzing..." : "Start Capture";
}

// Render the capture stage (live only) and four existing analysis stages.
function renderProgress(data) {
    getElement("analysisProgress").classList.remove("hidden");
    const live = currentMode === "live";
    const stages = live ? ["capture", ...ANALYSIS_STAGES] : ANALYSIS_STAGES;
    getElement("step-capture").classList.toggle("hidden", !live);
    getElement("analysisStageGrid").classList.toggle("has-capture", live);
    ANALYSIS_STAGES.forEach((name, index) => {
        getElement(`step-${name}`).querySelector(".stage-title").textContent = `${index + (live ? 2 : 1)}. ${["Extract", "Build flows", "Check headers", "Classify"][index]}`;
    });
    const stageIndex = stages.indexOf(data.stage);
    stages.forEach((name, index) => {
        const element = getElement(`step-${name}`);
        element.classList.remove("is-pending", "is-active", "is-complete", "is-error");
        let state = "Waiting", style = "is-pending";
        if (data.ready || data.stage === "prepare" || (stageIndex >= 0 && index < stageIndex)) {
            state = "✓ Complete"; style = "is-complete";
        } else if (data.error && index === stageIndex) {
            state = "✕ Failed"; style = "is-error";
        } else if (data.analyzing && index === stageIndex) {
            state = "● Running"; style = "is-active";
        }
        element.classList.add(style);
        element.querySelector(".stage-state").textContent = state;
    });
    const capturing = data.stage === "capture" && data.analyzing;
    getElement("analysisStatusTitle").textContent = data.error ? "Analysis failed"
        : data.ready ? "Analysis completed" : capturing ? "Capturing network traffic" : "Analyzing network traffic";
    const remaining = data.capture_remaining_seconds;
    const countDown = capturing && Number.isFinite(Number(remaining)) && remaining !== null && remaining !== undefined
        ? ` (${formatSeconds(remaining)} remaining)` : "";
    getElement("analysisStatusText").textContent = data.error || (data.ready
        ? "The traffic analysis is complete." : (data.stage_message || "Preparing analysis...") + countDown);
    getElement("analysisElapsed").textContent = `Elapsed: ${formatSeconds(data.elapsed_seconds ?? (Date.now() - startedAt) / 1000)}`;
}

// Share one API helper for upload, live capture, and status polling.
async function getJSON(path, options = {}) {
    const response = await fetch(path, { cache: "no-store", ...options });
    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || `Request failed: ${response.status}`);
    return result;
}

// Poll backend progress until Dashboard is ready or an error occurs.
function stopPolling() {
    if (statusTimer !== null) clearTimeout(statusTimer);
    statusTimer = null;
}
async function pollStatus() {
    try {
        const status = await getJSON("/api/status");
        renderProgress(status);
        if (status.error) {
            stopPolling(); setBusy(false); showError(status.error); return;
        }
        if (status.ready) {
            stopPolling(); setBusy(false);
            window.location.assign("/dashboard");
            return;
        }
        if (status.analyzing) {
            statusTimer = setTimeout(pollStatus, 900);
        } else {
            stopPolling(); setBusy(false);
            showError("The operation stopped before results became available.");
        }
    } catch (error) {
        stopPolling(); setBusy(false); showError(error.message);
    }
}

// Begin the selected operation; both share the existing progress API.
async function startAnalysis(path, options, initialStage, initialMessage) {
    if (busy) return;
    hideError();
    setBusy(true);
    startedAt = Date.now();
    renderProgress({ analyzing: true, stage: initialStage, stage_message: initialMessage, elapsed_seconds: 0 });
    try {
        const response = await getJSON(path, options);
        if (response.summary) {
            window.location.assign("/dashboard");
        } else {
            await pollStatus();
        }
    } catch (error) {
        stopPolling(); setBusy(false);
        renderProgress({ stage: initialStage, error: error.message, elapsed_seconds: (Date.now() - startedAt) / 1000 });
        showError(error.message);
    }
}

// Upload an existing PCAP/PCAPNG.
getElement("analyzeUploadButton").addEventListener("click", () => {
    const file = getElement("pcapFile").files[0];
    if (!file || busy) return;
    const body = new FormData();
    body.append("file", file);
    startAnalysis("/api/analyze/upload", { method: "POST", body }, "extract", "Uploading capture and preparing analysis...");
});

// Capture for 10-300 seconds, then automatically analyse the saved PCAP.
getElement("startCaptureButton").addEventListener("click", () => {
    if (busy) return;
    const raw = getElement("captureSeconds").value.trim();
    const duration = Number(raw);
    if (!raw || !Number.isInteger(duration) || duration < 10 || duration > 300) {
        showError("Capture duration must be a whole number between 10 and 300 seconds.");
        return;
    }
    startAnalysis("/api/analyze/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds: duration })
    }, "capture", "Preparing live capture...");
});

// Resume progress when a user refreshes the page during analysis.
(async function resumeAnalysis() {
    try {
        const status = await getJSON("/api/status");
        if (!status.analyzing) return;
        if (status.capture_duration != null) switchMode("live");
        setBusy(true);
        startedAt = Date.now() - Math.max(0, Number(status.elapsed_seconds) || 0) * 1000;
        renderProgress(status);
        stopPolling();
        statusTimer = setTimeout(pollStatus, 900);
    } catch (_) {
        // The page remains usable if the status API is temporarily unavailable.
    }
})();