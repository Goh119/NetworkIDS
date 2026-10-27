// ============================================================
// NetworkIDS Dashboard
// ============================================================

let currentPage = 1;
let totalPages = 1;
let currentFlowId = null;

let currentPacketResult = null;
let shapLoadedForFlow = null;

let statusPollTimer = null;

const MAX_FILE_SIZE_BYTES =
    512 * 1024 * 1024;


// ============================================================
// Elements
// ============================================================

const landingSection =
    document.getElementById(
        "landingSection"
    );

const resultsSection =
    document.getElementById(
        "resultsSection"
    );

const pcapFile =
    document.getElementById(
        "pcapFile"
    );

const uploadDropZone =
    document.getElementById(
        "uploadDropZone"
    );

const selectedFilePanel =
    document.getElementById(
        "selectedFilePanel"
    );

const selectedFile =
    document.getElementById(
        "selectedFile"
    );

const selectedFileSize =
    document.getElementById(
        "selectedFileSize"
    );

const clearSelectedFile =
    document.getElementById(
        "clearSelectedFile"
    );

const analyzeUploadButton =
    document.getElementById(
        "analyzeUploadButton"
    );

const newAnalysisButton =
    document.getElementById(
        "newAnalysisButton"
    );

const analysisProgress =
    document.getElementById(
        "analysisProgress"
    );

const analysisStatusTitle =
    document.getElementById(
        "analysisStatusTitle"
    );

const analysisStatusText =
    document.getElementById(
        "analysisStatusText"
    );

const analysisElapsed =
    document.getElementById(
        "analysisElapsed"
    );

const errorBox =
    document.getElementById(
        "errorBox"
    );

const searchInput =
    document.getElementById(
        "searchInput"
    );

const predictionFilter =
    document.getElementById(
        "predictionFilter"
    );

const protocolFilter =
    document.getElementById(
        "protocolFilter"
    );

const themeToggle =
    document.getElementById(
        "themeToggle"
    );


const stageSteps = {

    extract:
        document.getElementById(
            "step-extract"
        ),

    build:
        document.getElementById(
            "step-build"
        ),

    headers:
        document.getElementById(
            "step-headers"
        ),

    classify:
        document.getElementById(
            "step-classify"
        ),
};


const stageOrder = [

    "extract",
    "build",
    "headers",
    "classify",
];


// ============================================================
// Formatting
// ============================================================

function formatNumber(value) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "-";
    }


    return Number(
        value
    ).toLocaleString();
}


function formatFloat(
    value,
    decimals = 2
) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "-";
    }


    const number =
        Number(
            value
        );


    if (
        !Number.isFinite(
            number
        )
    ) {

        return "-";
    }


    return number.toFixed(
        decimals
    );
}


function formatProbability(value) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "-";
    }


    return (
        Number(
            value
        )
        *
        100
    ).toFixed(
        2
    )
    +
    "%";
}


function formatBytes(bytes) {

    if (
        bytes === null
        ||
        bytes === undefined
    ) {

        return "-";
    }


    const number =
        Number(
            bytes
        );


    if (
        number < 1024
    ) {

        return (
            number
            +
            " B"
        );
    }


    if (
        number
        <
        1024 * 1024
    ) {

        return (
            (
                number
                /
                1024
            ).toFixed(
                1
            )
            +
            " KB"
        );
    }


    return (
        (
            number
            /
            (
                1024
                *
                1024
            )
        ).toFixed(
            1
        )
        +
        " MB"
    );
}


function formatDuration(seconds) {

    const totalSeconds =
        Math.max(
            0,
            Math.floor(
                Number(
                    seconds
                )
                ||
                0
            )
        );


    const minutes =
        Math.floor(
            totalSeconds
            /
            60
        );


    const remainingSeconds =
        totalSeconds
        %
        60;


    return (
        String(
            minutes
        ).padStart(
            2,
            "0"
        )
        +
        ":"
        +
        String(
            remainingSeconds
        ).padStart(
            2,
            "0"
        )
    );
}


function formatTimestamp(value) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "-";
    }


    const timestamp =
        Number(
            value
        );


    if (
        !Number.isFinite(
            timestamp
        )
    ) {

        return "-";
    }


    const date =
        new Date(
            timestamp
            *
            1000
        );


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return "-";
    }


    const year =
        date.getFullYear();


    const month =
        String(
            date.getMonth()
            +
            1
        ).padStart(
            2,
            "0"
        );


    const day =
        String(
            date.getDate()
        ).padStart(
            2,
            "0"
        );


    const hours =
        String(
            date.getHours()
        ).padStart(
            2,
            "0"
        );


    const minutes =
        String(
            date.getMinutes()
        ).padStart(
            2,
            "0"
        );


    const seconds =
        String(
            date.getSeconds()
        ).padStart(
            2,
            "0"
        );


    const milliseconds =
        String(
            date.getMilliseconds()
        ).padStart(
            3,
            "0"
        );


    return (
        `${year}-${month}-${day} `
        +
        `${hours}:${minutes}:${seconds}.${milliseconds}`
    );
}


function formatFeatureValue(value) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "N/A";
    }


    const number =
        Number(
            value
        );


    if (
        !Number.isFinite(
            number
        )
    ) {

        return String(
            value
        );
    }


    if (
        Number.isInteger(
            number
        )
    ) {

        return number.toLocaleString();
    }


    const absolute =
        Math.abs(
            number
        );


    if (
        absolute > 0
        &&
        absolute < 0.0001
    ) {

        return number
            .toFixed(
                8
            )
            .replace(
                /0+$/,
                ""
            )
            .replace(
                /\.$/,
                ""
            );
    }


    return number
        .toFixed(
            2
        )
        .replace(
            /0+$/,
            ""
        )
        .replace(
            /\.$/,
            ""
        );
}


function checksumText(value) {

    if (
        value === null
        ||
        value === undefined
    ) {

        return "N/A";
    }


    return (
        Number(
            value
        )
        ===
        1
    )
        ? "Valid"
        : "Invalid";
}


// ============================================================
// Theme
// ============================================================

function applyTheme(theme) {

    document.documentElement.setAttribute(
        "data-theme",
        theme
    );


    themeToggle.textContent =
        (
            theme
            ===
            "light"
        )
            ? "☀"
            : "☾";


    localStorage.setItem(
        "networkids-theme",
        theme
    );
}


function initializeTheme() {

    const savedTheme =
        localStorage.getItem(
            "networkids-theme"
        );


    if (
        savedTheme === "light"
        ||
        savedTheme === "dark"
    ) {

        applyTheme(
            savedTheme
        );

        return;
    }


    applyTheme(
        "dark"
    );
}


themeToggle.addEventListener(
    "click",
    () => {

        const currentTheme =
            document.documentElement.getAttribute(
                "data-theme"
            )
            ||
            "dark";


        applyTheme(
            (
                currentTheme
                ===
                "dark"
            )
                ? "light"
                : "dark"
        );
    }
);


initializeTheme();


// ============================================================
// UI state
// ============================================================

function showError(message) {

    errorBox.textContent =
        message;


    errorBox.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorBox.classList.add(
        "hidden"
    );


    errorBox.textContent =
        "";
}


function setAnalyzing(
    isAnalyzing
) {

    clearSelectedFile.disabled =
        isAnalyzing;


    pcapFile.disabled =
        isAnalyzing;


    if (
        isAnalyzing
    ) {

        analyzeUploadButton.disabled =
            true;

        analyzeUploadButton.textContent =
            "Analyzing...";

    } else {

        analyzeUploadButton.textContent =
            "Analyze Network Traffic";

        analyzeUploadButton.disabled =
            !pcapFile.files.length;
    }
}


// ============================================================
// Progress state
// ============================================================

function resetStageUI() {

    stageOrder.forEach(
        stage => {

            const element =
                stageSteps[
                    stage
                ];


            element.classList.remove(
                "is-complete",
                "is-active",
                "is-pending",
                "is-error"
            );


            element.classList.add(
                "is-pending"
            );


            element.querySelector(
                ".stage-state"
            ).textContent =
                "Waiting";
        }
    );
}


function resetProgressUI() {

    resetStageUI();


    analysisProgress.classList.add(
        "hidden"
    );


    analysisStatusTitle.textContent =
        "Analyzing network traffic";


    analysisStatusText.textContent =
        "Preparing analysis...";


    analysisElapsed.textContent =
        "Elapsed: 00:00";
}


function updateStageUI(
    currentStage,
    analyzing,
    ready,
    hasError
) {

    resetStageUI();


    if (
        !currentStage
    ) {

        return;
    }


    const currentIndex =
        stageOrder.indexOf(
            currentStage
        );


    stageOrder.forEach(
        (
            stage,
            index
        ) => {

            const element =
                stageSteps[
                    stage
                ];


            const stateText =
                element.querySelector(
                    ".stage-state"
                );


            if (
                ready
            ) {

                element.classList.remove(
                    "is-pending"
                );

                element.classList.add(
                    "is-complete"
                );

                stateText.textContent =
                    "✓ Complete";

                return;
            }


            if (
                index
                <
                currentIndex
            ) {

                element.classList.remove(
                    "is-pending"
                );

                element.classList.add(
                    "is-complete"
                );

                stateText.textContent =
                    "✓ Complete";

                return;
            }


            if (
                index
                ===
                currentIndex
            ) {

                element.classList.remove(
                    "is-pending"
                );


                if (
                    hasError
                ) {

                    element.classList.add(
                        "is-error"
                    );

                    stateText.textContent =
                        "✕ Failed";

                } else if (
                    analyzing
                ) {

                    element.classList.add(
                        "is-active"
                    );

                    stateText.textContent =
                        "● Running";

                } else {

                    stateText.textContent =
                        "Waiting";
                }
            }
        }
    );
}


function updateAnalysisStatus(
    data
) {

    analysisProgress.classList.remove(
        "hidden"
    );


    const analyzing =
        Boolean(
            data.analyzing
        );


    const ready =
        Boolean(
            data.ready
        );


    const error =
        data.error
        ||
        null;


    updateStageUI(
        data.stage,
        analyzing,
        ready,
        Boolean(
            error
        )
    );


    analysisElapsed.textContent =
        (
            "Elapsed: "
            +
            formatDuration(
                data.elapsed_seconds
            )
        );


    if (
        error
        &&
        !analyzing
    ) {

        analysisStatusTitle.textContent =
            "Analysis failed";


        analysisStatusText.textContent =
            error;

        return;
    }


    if (
        ready
    ) {

        analysisStatusTitle.textContent =
            "Analysis completed";


        analysisStatusText.textContent =
            "The traffic analysis is complete.";

        return;
    }


    if (
        analyzing
    ) {

        analysisStatusTitle.textContent =
            "Analyzing network traffic";


        analysisStatusText.textContent =
            (
                data.stage_message
                ||
                "Processing..."
            );

        return;
    }


    analysisStatusTitle.textContent =
        "Waiting for analysis";


    analysisStatusText.textContent =
        "Select a PCAP file and start the analysis.";
}


// ============================================================
// Main view switching
// ============================================================

function showResultsView() {

    landingSection.classList.add(
        "hidden"
    );


    resultsSection.classList.remove(
        "hidden"
    );


    window.scrollTo({
        top: 0,
        behavior: "smooth",
    });
}


function showLandingView() {

    stopStatusPolling();


    resultsSection.classList.add(
        "hidden"
    );


    landingSection.classList.remove(
        "hidden"
    );


    resetSelectedFile();

    resetProgressUI();


    searchInput.value =
        "";


    predictionFilter.value =
        "All";


    protocolFilter.value =
        "All";


    currentPage =
        1;


    window.scrollTo({
        top: 0,
        behavior: "smooth",
    });
}


// ============================================================
// File selection
// ============================================================

function resetSelectedFile() {

    pcapFile.value =
        "";


    selectedFile.textContent =
        "-";


    selectedFileSize.textContent =
        "-";


    selectedFilePanel.classList.add(
        "hidden"
    );


    analyzeUploadButton.disabled =
        true;


    hideError();
}


function validateSelectedFile(
    file
) {

    if (
        !file
    ) {

        return {
            valid: false,
            message: "No file was selected.",
        };
    }


    const lowerName =
        file.name.toLowerCase();


    const allowed =
        (
            lowerName.endsWith(
                ".pcap"
            )
            ||
            lowerName.endsWith(
                ".pcapng"
            )
        );


    if (
        !allowed
    ) {

        return {

            valid:
                false,

            message:
                (
                    "Unsupported file type. "
                    +
                    "Please select a .pcap or .pcapng file."
                ),
        };
    }


    if (
        file.size
        >
        MAX_FILE_SIZE_BYTES
    ) {

        return {

            valid:
                false,

            message:
                (
                    "The selected PCAP exceeds the "
                    +
                    "512 MB upload limit."
                ),
        };
    }


    return {

        valid:
            true,

        message:
            "",
    };
}


function applySelectedFile(
    file
) {

    hideError();

    resetProgressUI();


    const validation =
        validateSelectedFile(
            file
        );


    if (
        !validation.valid
    ) {

        resetSelectedFile();


        showError(
            validation.message
        );


        return;
    }


    selectedFile.textContent =
        file.name;


    selectedFileSize.textContent =
        formatBytes(
            file.size
        );


    selectedFilePanel.classList.remove(
        "hidden"
    );


    analyzeUploadButton.disabled =
        false;
}


pcapFile.addEventListener(
    "change",
    () => {

        if (
            !pcapFile.files.length
        ) {

            resetSelectedFile();

            resetProgressUI();

            return;
        }


        applySelectedFile(
            pcapFile.files[
                0
            ]
        );
    }
);


clearSelectedFile.addEventListener(
    "click",
    event => {

        event.preventDefault();


        resetSelectedFile();

        resetProgressUI();
    }
);


// ============================================================
// Drag and drop
// ============================================================

[
    "dragenter",
    "dragover",
].forEach(
    eventName => {

        uploadDropZone.addEventListener(
            eventName,
            event => {

                event.preventDefault();

                event.stopPropagation();


                uploadDropZone.classList.add(
                    "drag-over"
                );
            }
        );
    }
);


[
    "dragleave",
    "drop",
].forEach(
    eventName => {

        uploadDropZone.addEventListener(
            eventName,
            event => {

                event.preventDefault();

                event.stopPropagation();


                uploadDropZone.classList.remove(
                    "drag-over"
                );
            }
        );
    }
);


uploadDropZone.addEventListener(
    "drop",
    event => {

        const files =
            event.dataTransfer.files;


        if (
            !files.length
        ) {

            return;
        }


        const file =
            files[
                0
            ];


        const dataTransfer =
            new DataTransfer();


        dataTransfer.items.add(
            file
        );


        pcapFile.files =
            dataTransfer.files;


        applySelectedFile(
            file
        );
    }
);


// ============================================================
// Status polling
// ============================================================

function stopStatusPolling() {

    if (
        statusPollTimer
        !==
        null
    ) {

        clearTimeout(
            statusPollTimer
        );


        statusPollTimer =
            null;
    }
}


async function fetchStatus() {

    const response =
        await fetch(
            "/api/status",
            {
                cache:
                    "no-store",
            }
        );


    const data =
        await response.json();


    if (
        !response.ok
        ||
        !data.success
    ) {

        throw new Error(
            data.error
            ||
            "Unable to load analysis status."
        );
    }


    return data;
}


async function loadSummary() {

    const response =
        await fetch(
            "/api/summary",
            {
                cache:
                    "no-store",
            }
        );


    const data =
        await response.json();


    if (
        !response.ok
        ||
        !data.success
    ) {

        throw new Error(
            data.error
            ||
            "Unable to load analysis summary."
        );
    }


    renderSummary(
        data.summary
    );
}


async function handleStatusPoll() {

    try {

        const data =
            await fetchStatus();


        updateAnalysisStatus(
            data
        );


        if (
            data.analyzing
        ) {

            statusPollTimer =
                setTimeout(
                    handleStatusPoll,
                    1000
                );


            return;
        }


        stopStatusPolling();


        if (
            data.error
        ) {

            setAnalyzing(
                false
            );


            showError(
                data.error
            );


            return;
        }


        if (
            data.ready
        ) {

            await loadSummary();


            currentPage =
                1;


            searchInput.value =
                "";


            predictionFilter.value =
                "All";


            protocolFilter.value =
                "All";


            await loadFlows();


            setAnalyzing(
                false
            );


            showResultsView();
        }


    } catch (
        error
    ) {

        stopStatusPolling();


        setAnalyzing(
            false
        );


        showError(
            error.message
        );
    }
}


function startStatusPolling() {

    stopStatusPolling();


    handleStatusPoll();
}


// ============================================================
// Analyze upload
// ============================================================

analyzeUploadButton.addEventListener(
    "click",
    async () => {

        if (
            !pcapFile.files.length
        ) {

            return;
        }


        const file =
            pcapFile.files[
                0
            ];


        const validation =
            validateSelectedFile(
                file
            );


        if (
            !validation.valid
        ) {

            showError(
                validation.message
            );


            return;
        }


        hideError();

        resetStageUI();


        updateAnalysisStatus({

            analyzing:
                true,

            ready:
                false,

            error:
                null,

            stage:
                "extract",

            stage_message:
                "Preparing packet extraction...",

            elapsed_seconds:
                0,
        });


        setAnalyzing(
            true
        );


        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    "/api/analyze/upload",
                    {

                        method:
                            "POST",

                        body:
                            formData,
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    "Unable to start analysis."
                );
            }


            startStatusPolling();


        } catch (
            error
        ) {

            setAnalyzing(
                false
            );


            showError(
                error.message
            );


            updateAnalysisStatus({

                analyzing:
                    false,

                ready:
                    false,

                error:
                    error.message,

                stage:
                    "extract",

                stage_message:
                    "Analysis could not be started.",

                elapsed_seconds:
                    0,
            });
        }
    }
);


// ============================================================
// New analysis
// ============================================================

newAnalysisButton.addEventListener(
    "click",
    () => {

        showLandingView();
    }
);


// ============================================================
// Summary
// ============================================================

function renderSummary(
    summary
) {

    document.getElementById(
        "sourceName"
    ).textContent =
        summary.source_name;


    document.getElementById(
        "totalPackets"
    ).textContent =
        formatNumber(
            summary.total_packets
        );


    document.getElementById(
        "totalFlows"
    ).textContent =
        formatNumber(
            summary.total_flows
        );


    document.getElementById(
        "attackFlows"
    ).textContent =
        formatNumber(
            summary.attack_flows
        );


    document.getElementById(
        "benignFlows"
    ).textContent =
        formatNumber(
            summary.benign_flows
        );


    document.getElementById(
        "suspiciousPackets"
    ).textContent =
        formatNumber(
            summary.suspicious_packets
        );


    document.getElementById(
        "tamperedPackets"
    ).textContent =
        formatNumber(
            summary.tampered_packets
        );


    const attackRate =
        Number(
            summary.attack_rate
            ||
            0
        );


    const benignRate =
        Math.max(
            0,
            100
            -
            attackRate
        );


    document.getElementById(
        "attackRate"
    ).textContent =
        (
            attackRate.toFixed(
                2
            )
            +
            "% Attack"
        );


    document.getElementById(
        "attackBar"
    ).style.width =
        (
            attackRate
            +
            "%"
        );


    document.getElementById(
        "benignBar"
    ).style.width =
        (
            benignRate
            +
            "%"
        );
}


// ============================================================
// Flow table
// ============================================================

async function loadFlows() {

    const params =
        new URLSearchParams({

            page:
                currentPage,

            page_size:
                25,

            prediction:
                predictionFilter.value,

            protocol:
                protocolFilter.value,

            q:
                searchInput.value.trim(),
        });


    try {

        const response =
            await fetch(
                "/api/flows?"
                +
                params.toString()
            );


        const data =
            await response.json();


        if (
            !response.ok
            ||
            !data.success
        ) {

            throw new Error(
                data.error
                ||
                "Unable to load flows."
            );
        }


        renderFlowTable(
            data.rows
        );


        currentPage =
            data.pagination.page;


        totalPages =
            data.pagination.total_pages;


        document.getElementById(
            "pageInformation"
        ).textContent =
            `Page ${currentPage} of ${totalPages}`;


        document.getElementById(
            "flowCountInformation"
        ).textContent =
            (
                `${formatNumber(data.pagination.total_rows)} `
                +
                (
                    data.pagination.total_rows
                    ===
                    1
                        ? "flow"
                        : "flows"
                )
            );


        document.getElementById(
            "previousPage"
        ).disabled =
            currentPage <= 1;


        document.getElementById(
            "nextPage"
        ).disabled =
            currentPage >= totalPages;


    } catch (
        error
    ) {

        showError(
            error.message
        );
    }
}


function renderFlowTable(
    rows
) {

    const body =
        document.getElementById(
            "flowTableBody"
        );


    body.innerHTML =
        "";


    if (
        !rows.length
    ) {

        body.innerHTML = `
            <tr>
                <td colspan="10">
                    No matching flows.
                </td>
            </tr>
        `;


        return;
    }


    rows.forEach(
        flow => {

            const tr =
                document.createElement(
                    "tr"
                );


            const predictionClass =
                (
                    flow.prediction
                    ===
                    "Attack"
                )
                    ? "prediction-attack"
                    : "prediction-benign";


            tr.innerHTML = `

                <td class="timestamp-cell">
                    ${formatTimestamp(flow.timestamp)}
                </td>

                <td>
                    ${flow.source}
                </td>

                <td>
                    ${flow.destination}
                </td>

                <td>
                    ${flow.protocol}
                </td>

                <td>
                    ${formatFloat(flow.avg_ttl, 1)}
                </td>

                <td>
                    ${formatNumber(flow.packets)}
                </td>

                <td>
                    ${formatNumber(flow.bytes)}
                </td>

                <td>

                    <span
                        class="prediction-badge ${predictionClass}"
                    >
                        ${flow.prediction}
                    </span>

                </td>

                <td>
                    ${formatProbability(flow.attack_probability)}
                </td>

                <td>

                    <button
                        class="button button-small button-secondary view-flow"
                        data-flow-id="${flow.flow_id}"
                        type="button"
                    >
                        View
                    </button>

                </td>
            `;


            body.appendChild(
                tr
            );
        }
    );


    document.querySelectorAll(
        ".view-flow"
    ).forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    openFlowModal(
                        button.dataset.flowId
                    );
                }
            );
        }
    );
}


// ============================================================
// Filters
// ============================================================

let searchTimer =
    null;


searchInput.addEventListener(
    "input",
    () => {

        clearTimeout(
            searchTimer
        );


        searchTimer =
            setTimeout(
                () => {

                    currentPage =
                        1;


                    loadFlows();

                },
                350
            );
    }
);


predictionFilter.addEventListener(
    "change",
    () => {

        currentPage =
            1;


        loadFlows();
    }
);


protocolFilter.addEventListener(
    "change",
    () => {

        currentPage =
            1;


        loadFlows();
    }
);


document.getElementById(
    "previousPage"
).addEventListener(
    "click",
    () => {

        if (
            currentPage > 1
        ) {

            currentPage -=
                1;


            loadFlows();
        }
    }
);


document.getElementById(
    "nextPage"
).addEventListener(
    "click",
    () => {

        if (
            currentPage < totalPages
        ) {

            currentPage +=
                1;


            loadFlows();
        }
    }
);


// ============================================================
// Modal reset
// ============================================================

function resetModalSections() {

    document.getElementById(
        "shapContent"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "packetContent"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "shapArrow"
    ).classList.remove(
        "open"
    );


    document.getElementById(
        "packetArrow"
    ).classList.remove(
        "open"
    );


    document.getElementById(
        "shapResults"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "shapResults"
    ).innerHTML =
        "";


    document.getElementById(
        "shapLoading"
    ).classList.remove(
        "hidden"
    );


    document.getElementById(
        "shapLoading"
    ).textContent =
        "SHAP explanation will be generated when opened.";


    document.getElementById(
        "packetTableBody"
    ).innerHTML =
        "";
}


// ============================================================
// Modal
// ============================================================

async function openFlowModal(
    flowId
) {

    currentFlowId =
        flowId;


    currentPacketResult =
        null;


    shapLoadedForFlow =
        null;


    resetModalSections();


    const modal =
        document.getElementById(
            "flowModal"
        );


    modal.classList.remove(
        "hidden"
    );


    document.body.style.overflow =
        "hidden";


    document.getElementById(
        "modalFlowTime"
    ).textContent =
        "Loading flow...";


    document.getElementById(
        "flowInformation"
    ).innerHTML =
        "Loading...";


    document.getElementById(
        "modalPrediction"
    ).textContent =
        "-";


    document.getElementById(
        "modalProbability"
    ).textContent =
        "-";


    document.getElementById(
        "integrityLoading"
    ).classList.remove(
        "hidden"
    );


    document.getElementById(
        "integrityLoading"
    ).textContent =
        "Checking packets associated with this flow...";


    document.getElementById(
        "integritySummary"
    ).classList.add(
        "hidden"
    );


    try {

        const response =
            await fetch(
                `/api/flow/${flowId}`
            );


        const data =
            await response.json();


        if (
            !response.ok
            ||
            !data.success
        ) {

            throw new Error(
                data.error
                ||
                "Unable to load flow."
            );
        }


        renderFlowDetails(
            data.flow
        );


        await loadRelatedPackets(
            flowId
        );


    } catch (
        error
    ) {

        document.getElementById(
            "flowInformation"
        ).textContent =
            error.message;
    }
}


function renderFlowDetails(
    flow
) {

    document.getElementById(
        "modalFlowTime"
    ).textContent =
        formatTimestamp(
            flow.timestamp
        );


    document.getElementById(
        "modalPrediction"
    ).textContent =
        flow.prediction;


    document.getElementById(
        "modalProbability"
    ).textContent =
        formatProbability(
            flow.attack_probability
        );


    const details = [

        [
            "Timestamp",
            formatTimestamp(
                flow.timestamp
            ),
        ],

        [
            "Source",
            flow.source,
        ],

        [
            "Destination",
            flow.destination,
        ],

        [
            "Protocol",
            flow.protocol,
        ],

        [
            "Average TTL",
            formatFloat(
                flow.avg_ttl,
                2
            ),
        ],

        [
            "Packets",
            formatNumber(
                flow.packets
            ),
        ],

        [
            "Bytes",
            formatNumber(
                flow.bytes
            ),
        ],

        [
            "Duration",
            (
                formatFloat(
                    flow.duration,
                    6
                )
                +
                " s"
            ),
        ],

        [
            "Forward Packets",
            formatNumber(
                flow.forward_packets
            ),
        ],

        [
            "Reverse Packets",
            formatNumber(
                flow.reverse_packets
            ),
        ],

        [
            "SYN Count",
            formatNumber(
                flow.syn_count
            ),
        ],

        [
            "ACK Count",
            formatNumber(
                flow.ack_count
            ),
        ],

        [
            "RST Count",
            formatNumber(
                flow.rst_count
            ),
        ],
    ];


    const container =
        document.getElementById(
            "flowInformation"
        );


    container.innerHTML =
        details.map(
            item => `

                <div class="detail-item">

                    <span>
                        ${item[0]}
                    </span>

                    <strong>
                        ${item[1]}
                    </strong>

                </div>
            `
        )
        .join(
            ""
        );
}


// ============================================================
// Related packets
// ============================================================

async function loadRelatedPackets(
    flowId
) {

    try {

        const response =
            await fetch(
                `/api/flow/${flowId}/packets`
            );


        const data =
            await response.json();


        if (
            !response.ok
            ||
            !data.success
        ) {

            throw new Error(
                data.error
                ||
                "Unable to load packet integrity."
            );
        }


        currentPacketResult =
            data.result;


        renderIntegritySummary(
            data.result
        );


        renderPacketTable(
            data.result.packets
        );


    } catch (
        error
    ) {

        document.getElementById(
            "integrityLoading"
        ).textContent =
            (
                "Unable to load packet integrity: "
                +
                error.message
            );
    }
}


function renderIntegritySummary(
    result
) {

    document.getElementById(
        "integrityLoading"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "integritySummary"
    ).classList.remove(
        "hidden"
    );


    const badge =
        document.getElementById(
            "integrityBadge"
        );


    badge.textContent =
        result.integrity_state;


    badge.className =
        "integrity-badge";


    if (
        result.integrity_state
        ===
        "Tampered"
    ) {

        badge.classList.add(
            "integrity-tampered"
        );

    } else if (
        result.integrity_state
        ===
        "Suspicious"
    ) {

        badge.classList.add(
            "integrity-suspicious"
        );

    } else {

        badge.classList.add(
            "integrity-intact"
        );
    }


    const counts =
        document.getElementById(
            "integrityCounts"
        );


    counts.innerHTML = `

        <div class="mini-kpi">

            <span>
                Related Packets
            </span>

            <strong>
                ${formatNumber(result.total_packets)}
            </strong>

        </div>


        <div class="mini-kpi">

            <span>
                Intact
            </span>

            <strong>
                ${formatNumber(result.intact_packets)}
            </strong>

        </div>


        <div class="mini-kpi">

            <span>
                Suspicious
            </span>

            <strong>
                ${formatNumber(result.suspicious_packets)}
            </strong>

        </div>


        <div class="mini-kpi">

            <span>
                Tampered
            </span>

            <strong>
                ${formatNumber(result.tampered_packets)}
            </strong>

        </div>
    `;
}


function renderPacketTable(
    packets
) {

    const body =
        document.getElementById(
            "packetTableBody"
        );


    body.innerHTML =
        "";


    if (
        !packets.length
    ) {

        body.innerHTML = `
            <tr>
                <td colspan="8">
                    No related packet records found.
                </td>
            </tr>
        `;


        return;
    }


    packets.forEach(
        packet => {

            const tr =
                document.createElement(
                    "tr"
                );


            tr.innerHTML = `

                <td>
                    ${packet.packet_id ?? "-"}
                </td>

                <td>
                    ${packet.source}
                </td>

                <td>
                    ${packet.destination}
                </td>

                <td>
                    ${formatFeatureValue(packet.ttl)}
                </td>

                <td>
                    ${packet.flags}
                </td>

                <td>
                    ${checksumText(packet.ip_checksum)}
                </td>

                <td>
                    ${checksumText(packet.l4_checksum)}
                </td>

                <td>
                    ${packet.integrity_state}
                </td>
            `;


            body.appendChild(
                tr
            );
        }
    );
}


// ============================================================
// SHAP accordion
// ============================================================

document.getElementById(
    "shapToggle"
).addEventListener(
    "click",
    async () => {

        const content =
            document.getElementById(
                "shapContent"
            );


        const arrow =
            document.getElementById(
                "shapArrow"
            );


        const opening =
            content.classList.contains(
                "hidden"
            );


        content.classList.toggle(
            "hidden"
        );


        arrow.classList.toggle(
            "open",
            opening
        );


        if (
            !opening
            ||
            !currentFlowId
        ) {

            return;
        }


        if (
            shapLoadedForFlow
            ===
            currentFlowId
        ) {

            return;
        }


        document.getElementById(
            "shapLoading"
        ).classList.remove(
            "hidden"
        );


        document.getElementById(
            "shapLoading"
        ).textContent =
            "Generating SHAP explanation...";


        try {

            const response =
                await fetch(
                    `/api/flow/${currentFlowId}/explain`
                );


            const data =
                await response.json();


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    "Unable to generate SHAP explanation."
                );
            }


            renderShap(
                data.explanation
            );


            shapLoadedForFlow =
                currentFlowId;


        } catch (
            error
        ) {

            document.getElementById(
                "shapLoading"
            ).textContent =
                (
                    "Unable to generate SHAP explanation: "
                    +
                    error.message
                );
        }
    }
);


// ============================================================
// SHAP rendering
// ============================================================

function renderShap(
    explanation
) {

    const loading =
        document.getElementById(
            "shapLoading"
        );


    loading.classList.add(
        "hidden"
    );


    const results =
        document.getElementById(
            "shapResults"
        );


    results.classList.remove(
        "hidden"
    );


    const contributions =
        explanation.top_contributions
        ||
        [];


    const maxValue =
        Math.max(
            ...contributions.map(
                item =>
                    Number(
                        item.absolute_shap
                    )
            ),
            0.000001
        );


    results.innerHTML = `

        <div class="shap-explanation-box">

            <span class="shap-explanation-title">
                Simple explanation
            </span>

            <p>
                ${explanation.plain_explanation}
            </p>

        </div>


        <p class="shap-description">
            The chart below shows the strongest feature influences
            for this flow. Values on the left push the model toward
            <strong>Benign</strong>, while values on the right push
            it toward <strong>Attack</strong>.
        </p>


        <div class="shap-axis">

            <div>
                Feature
            </div>

            <div>
                Value
            </div>

            <div class="shap-axis-track">

                <span>
                    Benign
                </span>

                <span>
                    Attack
                </span>

            </div>

            <div>
                SHAP
            </div>

        </div>


        ${
            contributions
            .map(
                item => {

                    const relativeWidth =
                        Math.min(
                            50,
                            (
                                Number(
                                    item.absolute_shap
                                )
                                /
                                maxValue
                            )
                            *
                            50
                        );


                    let fillClass =
                        "shap-fill-neutral";


                    if (
                        item.shap_value > 0
                    ) {

                        fillClass =
                            "shap-fill-positive";

                    } else if (
                        item.shap_value < 0
                    ) {

                        fillClass =
                            "shap-fill-negative";
                    }


                    const sign =
                        item.shap_value > 0
                            ? "+"
                            : "";


                    return `

                        <div class="shap-row">

                            <div
                                class="shap-feature"
                                title="${item.feature}"
                            >

                                <span class="shap-feature-name">
                                    ${item.feature_label || item.feature}
                                </span>

                                <span class="shap-feature-raw">
                                    ${item.feature}
                                </span>

                            </div>


                            <div class="shap-value">
                                ${formatFeatureValue(item.feature_value)}
                            </div>


                            <div class="shap-track">

                                <div
                                    class="shap-fill ${fillClass}"
                                    style="width:${relativeWidth}%"
                                ></div>

                            </div>


                            <div class="shap-direction">
                                ${sign}${Number(item.shap_value).toFixed(4)}
                            </div>

                        </div>
                    `;
                }
            )
            .join(
                ""
            )
        }
    `;
}


// ============================================================
// Packet accordion
// ============================================================

document.getElementById(
    "packetToggle"
).addEventListener(
    "click",
    () => {

        const content =
            document.getElementById(
                "packetContent"
            );


        const arrow =
            document.getElementById(
                "packetArrow"
            );


        const opening =
            content.classList.contains(
                "hidden"
            );


        content.classList.toggle(
            "hidden"
        );


        arrow.classList.toggle(
            "open",
            opening
        );
    }
);


// ============================================================
// Close modal
// ============================================================

function closeModal() {

    document.getElementById(
        "flowModal"
    ).classList.add(
        "hidden"
    );


    document.body.style.overflow =
        "";


    currentFlowId =
        null;


    currentPacketResult =
        null;


    shapLoadedForFlow =
        null;


    resetModalSections();
}


document.getElementById(
    "closeModal"
).addEventListener(
    "click",
    closeModal
);


document.querySelector(
    ".modal-backdrop"
).addEventListener(
    "click",
    closeModal
);


document.addEventListener(
    "keydown",
    event => {

        if (
            event.key
            ===
            "Escape"
        ) {

            closeModal();
        }
    }
);


// ============================================================
// Initial UI
// ============================================================

resetProgressUI();