// ============================================================
// NetworkIDS Dashboard
// ============================================================

let currentPage = 1;
let totalPages = 1;
let currentFlowId = null;

let currentPacketResult = null;
let shapLoadedForFlow = null;


// ============================================================
// Elements
// ============================================================

const pcapFile =
    document.getElementById("pcapFile");

const selectedFile =
    document.getElementById("selectedFile");

const analyzeUploadButton =
    document.getElementById("analyzeUploadButton");

const demoButton =
    document.getElementById("demoButton");

const analysisStatus =
    document.getElementById("analysisStatus");

const errorBox =
    document.getElementById("errorBox");

const resultsSection =
    document.getElementById("resultsSection");

const searchInput =
    document.getElementById("searchInput");

const predictionFilter =
    document.getElementById("predictionFilter");

const protocolFilter =
    document.getElementById("protocolFilter");


// ============================================================
// Formatting
// ============================================================

function formatNumber(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "-";
    }

    return Number(value)
        .toLocaleString();
}


function formatFloat(
    value,
    decimals = 2
) {

    if (
        value === null ||
        value === undefined
    ) {
        return "-";
    }

    return Number(value)
        .toFixed(decimals);
}


function formatProbability(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "-";
    }

    return (
        Number(value) * 100
    ).toFixed(2) + "%";
}


function checksumText(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "N/A";
    }

    return Number(value) === 1
        ? "Valid"
        : "Invalid";
}


// ============================================================
// UI states
// ============================================================

function showError(message) {

    errorBox.textContent = message;

    errorBox.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorBox.classList.add(
        "hidden"
    );

    errorBox.textContent = "";
}


function setAnalyzing(isAnalyzing) {

    if (isAnalyzing) {

        analysisStatus.classList.remove(
            "hidden"
        );

        analyzeUploadButton.disabled = true;
        demoButton.disabled = true;

    } else {

        analysisStatus.classList.add(
            "hidden"
        );

        demoButton.disabled = false;

        analyzeUploadButton.disabled =
            !pcapFile.files.length;
    }
}


// ============================================================
// File selection
// ============================================================

pcapFile.addEventListener(
    "change",
    () => {

        hideError();

        if (!pcapFile.files.length) {

            selectedFile.textContent =
                "No file selected";

            analyzeUploadButton.disabled =
                true;

            return;
        }


        selectedFile.textContent =
            pcapFile.files[0].name;

        analyzeUploadButton.disabled =
            false;
    }
);


// ============================================================
// Analyze upload
// ============================================================

analyzeUploadButton.addEventListener(
    "click",
    async () => {

        if (!pcapFile.files.length) {
            return;
        }


        hideError();

        setAnalyzing(true);


        const formData =
            new FormData();


        formData.append(
            "file",
            pcapFile.files[0]
        );


        try {

            const response =
                await fetch(
                    "/api/analyze/upload",
                    {
                        method: "POST",
                        body: formData,
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    "Analysis failed."
                );
            }


            renderSummary(
                data.summary
            );


            currentPage = 1;

            await loadFlows();


            resultsSection.classList.remove(
                "hidden"
            );


            resultsSection.scrollIntoView({
                behavior: "smooth",
            });


        } catch (error) {

            showError(
                error.message
            );

        } finally {

            setAnalyzing(false);
        }
    }
);


// ============================================================
// Demo
// ============================================================

demoButton.addEventListener(
    "click",
    async () => {

        hideError();

        setAnalyzing(true);


        try {

            const response =
                await fetch(
                    "/api/analyze/demo",
                    {
                        method: "POST",
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    "Demo analysis failed."
                );
            }


            renderSummary(
                data.summary
            );


            currentPage = 1;

            await loadFlows();


            resultsSection.classList.remove(
                "hidden"
            );


            resultsSection.scrollIntoView({
                behavior: "smooth",
            });


        } catch (error) {

            showError(
                error.message
            );

        } finally {

            setAnalyzing(false);
        }
    }
);


// ============================================================
// Summary
// ============================================================

function renderSummary(summary) {

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
            summary.attack_rate || 0
        );


    const benignRate =
        Math.max(
            0,
            100 - attackRate
        );


    document.getElementById(
        "attackRate"
    ).textContent =
        attackRate.toFixed(2)
        + "% Attack";


    document.getElementById(
        "attackBar"
    ).style.width =
        attackRate + "%";


    document.getElementById(
        "benignBar"
    ).style.width =
        benignRate + "%";
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


    const response =
        await fetch(
            "/api/flows?"
            + params.toString()
        );


    const data =
        await response.json();


    if (
        !response.ok ||
        !data.success
    ) {

        showError(
            data.error ||
            "Unable to load flows."
        );

        return;
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
        "previousPage"
    ).disabled =
        currentPage <= 1;


    document.getElementById(
        "nextPage"
    ).disabled =
        currentPage >= totalPages;
}


function renderFlowTable(rows) {

    const body =
        document.getElementById(
            "flowTableBody"
        );


    body.innerHTML = "";


    if (!rows.length) {

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
                flow.prediction === "Attack"
                    ? "prediction-attack"
                    : "prediction-benign";


            tr.innerHTML = `

                <td>
                    ${flow.flow_id}
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
                    ${formatProbability(
                        flow.attack_probability
                    )}
                </td>

                <td>
                    <button
                        class="button button-small button-secondary view-flow"
                        data-flow-id="${flow.flow_id}"
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

let searchTimer = null;


searchInput.addEventListener(
    "input",
    () => {

        clearTimeout(
            searchTimer
        );


        searchTimer = setTimeout(
            () => {

                currentPage = 1;

                loadFlows();

            },
            350
        );
    }
);


predictionFilter.addEventListener(
    "change",
    () => {

        currentPage = 1;

        loadFlows();
    }
);


protocolFilter.addEventListener(
    "change",
    () => {

        currentPage = 1;

        loadFlows();
    }
);


document.getElementById(
    "previousPage"
).addEventListener(
    "click",
    () => {

        if (currentPage > 1) {

            currentPage -= 1;

            loadFlows();
        }
    }
);


document.getElementById(
    "nextPage"
).addEventListener(
    "click",
    () => {

        if (currentPage < totalPages) {

            currentPage += 1;

            loadFlows();
        }
    }
);


// ============================================================
// Modal
// ============================================================

async function openFlowModal(flowId) {

    currentFlowId = flowId;

    currentPacketResult = null;

    shapLoadedForFlow = null;


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
        "modalFlowId"
    ).textContent =
        flowId;


    document.getElementById(
        "flowInformation"
    ).innerHTML =
        "Loading...";


    document.getElementById(
        "integrityLoading"
    ).classList.remove(
        "hidden"
    );


    document.getElementById(
        "integritySummary"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "packetTableBody"
    ).innerHTML =
        "";


    document.getElementById(
        "shapResults"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "shapLoading"
    ).classList.remove(
        "hidden"
    );


    document.getElementById(
        "shapLoading"
    ).textContent =
        "Open this section to generate the SHAP explanation.";


    try {

        const response =
            await fetch(
                `/api/flow/${flowId}`
            );


        const data =
            await response.json();


        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error
            );
        }


        renderFlowDetails(
            data.flow
        );


        // Packet integrity is loaded immediately for the popup
        // so the user sees integrity status without opening SHAP.

        loadRelatedPackets(
            flowId
        );


    } catch (error) {

        document.getElementById(
            "flowInformation"
        ).textContent =
            error.message;
    }
}


function renderFlowDetails(flow) {

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
                + " s"
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
        .join("");
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
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error
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


    } catch (error) {

        document.getElementById(
            "integrityLoading"
        ).textContent =
            "Unable to load packet integrity: "
            + error.message;
    }
}


function renderIntegritySummary(result) {

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


function renderPacketTable(packets) {

    const body =
        document.getElementById(
            "packetTableBody"
        );


    body.innerHTML = "";


    if (!packets.length) {

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
                    ${packet.ttl ?? "-"}
                </td>

                <td>
                    ${packet.flags}
                </td>

                <td>
                    ${checksumText(
                        packet.ip_checksum
                    )}
                </td>

                <td>
                    ${checksumText(
                        packet.l4_checksum
                    )}
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


        const opening =
            content.classList.contains(
                "hidden"
            );


        content.classList.toggle(
            "hidden"
        );


        if (
            !opening ||
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
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error
                );
            }


            renderShap(
                data.explanation
            );


            shapLoadedForFlow =
                currentFlowId;


        } catch (error) {

            document.getElementById(
                "shapLoading"
            ).textContent =
                "Unable to generate SHAP explanation: "
                + error.message;
        }
    }
);


function renderShap(explanation) {

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
        explanation.top_contributions;


    const maxValue =
        Math.max(
            ...contributions.map(
                item =>
                    item.absolute_shap
            ),
            0.000001
        );


    results.innerHTML = `

        <p class="loading-text">
            Positive SHAP values push the prediction toward
            <strong>Attack</strong>. Negative values push it toward
            <strong>Benign</strong>.
        </p>

        ${
            contributions
            .map(
                item => {

                    const width =
                        (
                            item.absolute_shap
                            /
                            maxValue
                            *
                            100
                        );


                    const barClass =
                        item.shap_value >= 0
                            ? "shap-attack"
                            : "shap-benign";


                    return `

                        <div class="shap-row">

                            <div class="shap-feature">
                                ${item.feature}
                            </div>


                            <div class="shap-value">
                                ${item.feature_value ?? "NaN"}
                            </div>


                            <div class="shap-track">

                                <div
                                    class="shap-fill ${barClass}"
                                    style="width:${width}%"
                                >
                                </div>

                            </div>


                            <div class="shap-direction">
                                ${
                                    item.shap_value >= 0
                                        ? "+"
                                        : ""
                                }
                                ${item.shap_value.toFixed(4)}
                            </div>

                        </div>
                    `;
                }
            )
            .join("")
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

        document.getElementById(
            "packetContent"
        ).classList.toggle(
            "hidden"
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


    currentFlowId = null;
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