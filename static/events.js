// NetworkIDS Events — compatible with the existing /api/flows and Flow Details APIs.
const $ = id => document.getElementById(id);
let currentPage = 1, totalPages = 1, activeFlowId = null;
let activeRequestId = 0, searchTimer = null;
let selectedTime = null, timelineBuckets = [], shapLoadedForFlow = null;
const integrityCache = new Map();

const missing = v => v === null || v === undefined || v === "" || (typeof v === "number" && !Number.isFinite(v)) || (typeof v === "string" && ["nan","none","null","infinity","-infinity","inf","-inf"].includes(v.trim().toLowerCase()));
const display = v => missing(v) ? "-" : String(v);
const number = v => missing(v) || !Number.isFinite(Number(v)) ? "-" : Number(v).toLocaleString();
const float = (v, digits = 2) => missing(v) || !Number.isFinite(Number(v)) ? "-" : Number(v).toFixed(digits);
const probability = v => missing(v) || !Number.isFinite(Number(v)) ? "-" : `${(Number(v) * 100).toFixed(2)}%`;
const duration = v => missing(v) || !Number.isFinite(Number(v)) ? "-" : `${Number(v).toFixed(2)}s`;
function timestamp(v) {
    if (missing(v) || !Number.isFinite(Number(v))) return "-";
    const d = new Date(Number(v) * 1000);
    if (!Number.isFinite(d.getTime())) return "-";
    const pad = (n, length = 2) => String(n).padStart(length, "0");
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(),3)}`;
}
function showError(msg) { $("eventsError").textContent = msg; $("eventsError").classList.remove("hidden"); }
function clearError() { $("eventsError").textContent = ""; $("eventsError").classList.add("hidden"); }
async function api(path) {
    const response = await fetch(path, { cache:"no-store" });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || `Request failed: ${response.status}`);
    return data;
}
function applyTheme(mode) {
    document.documentElement.setAttribute("data-theme", mode);
    $("themeToggle").textContent = mode === "light" ? "☀" : "☾";
    localStorage.setItem("networkids-theme", mode);
}
applyTheme(["dark","light"].includes(localStorage.getItem("networkids-theme")) ? localStorage.getItem("networkids-theme") : "dark");
$("themeToggle").addEventListener("click", () => applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

function cell(row, value, className = "") {
    const td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = value;
    row.appendChild(td);
    return td;
}
function label(text, cssClass, baseClass = "prediction-badge") {
    const el = document.createElement("span");
    el.className = `${baseClass} ${cssClass}`;
    el.textContent = text;
    return el;
}
function integrityText(flow) {
    return integrityCache.get(String(flow.flow_id)) || (missing(flow.integrity_state) ? "-" : flow.integrity_state);
}
function renderRows(rows) {
    const body = $("flowTableBody");
    body.replaceChildren();
    if (!rows.length) {
        const tr = document.createElement("tr");
        const td = cell(tr, "No matching flows."); td.colSpan = 13;
        body.appendChild(tr); return;
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
        predictionCell.appendChild(label(predicted, predicted === "Attack" ? "prediction-attack" : predicted === "Benign" ? "prediction-benign" : ""));
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
        view.type = "button"; view.textContent = "View";
        view.addEventListener("click", () => openFlowModal(flow.flow_id, status));
        action.appendChild(view); tr.appendChild(action);
        body.appendChild(tr);
    }
}
async function loadFlows() {
    const requestId = ++activeRequestId;
    const params = new URLSearchParams({
        page:currentPage, page_size:25, prediction:$("predictionFilter").value,
        protocol:$("protocolFilter").value, q:$("searchInput").value.trim()
    });
    if (selectedTime) {
        params.set("from_ts",selectedTime.start);
        params.set("to_ts",selectedTime.end);
    }
    try {
        const data = await api(`/api/flows?${params}`);
        if (requestId !== activeRequestId) return;
        currentPage = data.pagination.page; totalPages = data.pagination.total_pages;
        renderRows(data.rows);
        $("pageInformation").textContent = `Page ${currentPage} of ${totalPages}`;
        $("flowCountInformation").textContent = `${number(data.pagination.total_rows)} ${data.pagination.total_rows === 1 ? "flow" : "flows"}`;
        $("previousPage").disabled = currentPage <= 1;
        $("nextPage").disabled = currentPage >= totalPages;
        clearError();
    } catch(err) { if(requestId === activeRequestId) showError(err.message); }
}
function reload() { currentPage = 1; loadFlows(); }
$("searchInput").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(reload,350); });
$("predictionFilter").addEventListener("change", reload);
$("protocolFilter").addEventListener("change", reload);
$("previousPage").addEventListener("click", () => { if(currentPage > 1) {currentPage--;loadFlows();} });
$("nextPage").addEventListener("click", () => { if(currentPage < totalPages) {currentPage++;loadFlows();} });

// Timeline is drawn from real bucket counts returned by /api/events/timeline.
function renderTimeline(buckets) {
    timelineBuckets = buckets;
    const host = $("timelineChart"); host.replaceChildren();
    if (!buckets.length) { host.textContent = "No capture timestamps are available."; return; }
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns,"svg"); svg.setAttribute("viewBox","0 0 1000 240");
    svg.setAttribute("role","img"); svg.setAttribute("aria-label","Attack and benign network flows over time");
    const make = (tag, attributes) => { const e = document.createElementNS(ns,tag); for(const [k,v] of Object.entries(attributes)) e.setAttribute(k,String(v)); svg.appendChild(e); return e; };
    const left=52, right=986, top=20, bottom=205, plotH=bottom-top;
    const max = Math.max(1,...buckets.map(b=>Number(b.attack||0)+Number(b.benign||0)));
    for(let i=0;i<=4;i++) {
        const y=bottom-(plotH*i/4);
        make("line",{x1:left,y1:y,x2:right,y2:y,class:"timeline-grid"});
        const t=make("text",{x:42,y:y+4,"text-anchor":"end",class:"timeline-axis"});t.textContent=number(Math.round(max*i/4));
    }
    const step=(right-left)/buckets.length, barW=Math.max(4,Math.min(42,step*.74));
    buckets.forEach((b,i)=>{
        const x=left+i*step+(step-barW)/2;
        const benignH=Number(b.benign||0)/max*plotH;
        const attackH=Number(b.attack||0)/max*plotH;
        const group = document.createElementNS(ns,"g"); group.classList.add("timeline-bar");
        if (selectedTime && Number(selectedTime.start) === Number(b.start)) group.classList.add("selected");
        for (const [y,h,color] of [[bottom-benignH,benignH,"var(--benign)"],[bottom-benignH-attackH,attackH,"var(--attack)"]]) {
            const rect=document.createElementNS(ns,"rect");
            rect.setAttribute("x",String(x));rect.setAttribute("y",String(y));rect.setAttribute("width",String(barW));rect.setAttribute("height",String(Math.max(0,h)));rect.setAttribute("fill",color);group.appendChild(rect);
        }
        const title=document.createElementNS(ns,"title");title.textContent=`${timestamp(b.start)}\nAttack: ${b.attack}\nBenign: ${b.benign}`;group.appendChild(title);
        const hit=document.createElementNS(ns,"rect");
        for (const [k,v] of Object.entries({x,y:top,width:Math.max(step,barW),height:plotH,fill:"transparent"}))hit.setAttribute(k,String(v));
        group.appendChild(hit);
        group.addEventListener("click",()=>{
            if (selectedTime && Number(selectedTime.start)===Number(b.start)) selectedTime=null;
            else selectedTime={start:b.start,end:b.end};
            $("clearTimeFilter").classList.toggle("hidden",!selectedTime);
            $("timeFilterInfo").textContent=selectedTime?`Selected: ${timestamp(b.start)} – ${timestamp(b.end)}`:"All capture times";
            renderTimeline(timelineBuckets);reload();
        });
        svg.appendChild(group);
    });
    const tickCount=Math.min(5,buckets.length);
    for(let i=0;i<tickCount;i++) {
        const idx=tickCount === 1 ? 0 : Math.round((buckets.length-1)*i/(tickCount-1));
        const t=make("text",{x:left+idx*step+step/2,y:228,"text-anchor":"middle",class:"timeline-axis"});
        t.textContent=timestamp(buckets[idx].start).slice(11,16);
    }
    host.appendChild(svg);
}
$("clearTimeFilter").addEventListener("click",()=>{selectedTime=null;$("clearTimeFilter").classList.add("hidden");$("timeFilterInfo").textContent="All capture times";renderTimeline(timelineBuckets);reload();});

function appendDetails(id, pairs) {
    const container=$(id);container.replaceChildren();
    for(const [name,value] of pairs){
        const box=document.createElement("div");box.className="detail-item";
        const caption=document.createElement("span");caption.textContent=name;
        const strong=document.createElement("strong");strong.textContent=value;
        box.append(caption,strong);container.appendChild(box);
    }
}
function resetModal() {
    for(const name of ["shap","packet"]) {
        $(`${name}Content`).classList.add("hidden");
        $(`${name}Arrow`).classList.remove("open");
        $(`${name}Toggle`).setAttribute("aria-expanded","false");
    }
    $("shapResults").replaceChildren();$("shapResults").classList.add("hidden");
    $("shapLoading").classList.remove("hidden");
    $("shapLoading").textContent="SHAP explanation will be generated when opened.";
    $("packetTableBody").replaceChildren();
    $("integrityLabels").replaceChildren();
    $("integrityLabels").textContent = "Loading integrity findings...";
    shapLoadedForFlow=null;
    relatedPacketsLoadedForFlow=null;
    relatedPacketsLoadingForFlow=null;
}
async function openFlowModal(flowId, tableBadge) {
    const id=String(flowId); activeFlowId=id; resetModal();
    $("flowModal").classList.remove("hidden");document.body.style.overflow="hidden";
    $("modalFlowTime").textContent="Loading flow...";
    $("flowInformation").textContent="Loading...";
    $("modalPrediction").textContent="-";$("modalProbability").textContent="-";
    try {
        const data=await api(`/api/flow/${encodeURIComponent(id)}`);
        if(activeFlowId!==id)return;
        const f=data.flow;
        $("modalFlowTime").textContent=timestamp(f.timestamp);
        $("modalPrediction").textContent=display(f.prediction);
        $("modalProbability").textContent=probability(f.attack_probability);
        appendDetails("flowInformation",[
            ["Timestamp",timestamp(f.timestamp)],["Source IP",display(f.src_ip)],["Source Port",display(f.src_port)],
            ["Destination IP",display(f.dst_ip)],["Destination Port",display(f.dst_port)],
            ["Protocol",display(f.protocol)],["Average TTL",float(f.avg_ttl)],
            ["Flow Packets",number(f.packets)],["Flow Bytes",number(f.bytes)],
            ["Flow Duration",duration(f.duration)],["Forward Packets",number(f.forward_packets)],
            ["Reverse Packets",number(f.reverse_packets)],["SYN Count",number(f.syn_count)],
            ["ACK Count",number(f.ack_count)],["RST Count",number(f.rst_count)]
        ]);
        renderIntegrityLabels(f.integrity_explanation || {
            integrity_state: f.integrity_state,
            triggered_rules: [],
            additional_indicators: []
        });
        // Restore the original behavior: load related packet rows when View opens.
        // The request runs independently and does not delay the integrity labels.
        void fetchRelatedPackets(id);
    }catch(err){if(activeFlowId===id){$("integrityLoading").textContent=err.message; if($("flowInformation").textContent==="Loading...") $("flowInformation").textContent=err.message;}}
}
// ============================================================
// Compact, color-coded integrity labels (precomputed evidence)
// ============================================================
function renderIntegrityLabels(result) {
    const root = $("integrityLabels");
    root.replaceChildren();
    const state = display(result.integrity_state);
    const rules = Array.isArray(result.triggered_rules) ? result.triggered_rules : [];
    const indicators = Array.isArray(result.additional_indicators) ? result.additional_indicators : [];

    const addLabel = (category, text) => {
        const badge = document.createElement("span");
        badge.className = `integrity-reason-label ${category.toLowerCase()}`;
        badge.textContent = text;
        root.appendChild(badge);
    };

    if (state === "-" && rules.length === 0 && indicators.length === 0) {
        addLabel("unknown", "-");
        return;
    }

    if (rules.length === 0) {
        // An indicator-only result remains Intact under the existing rule definitions.
        addLabel(state, state);
    } else {
        for (const rule of rules) {
            const category = display(rule.category);
            addLabel(category, `${category}: ${display(rule.rule_id)} - ${display(rule.description)}`);
        }
    }

    for (const rule of indicators) {
        addLabel("indicator", `Indicator: ${display(rule.rule_id)} - ${display(rule.description)}`);
    }
}

let relatedPacketsLoadedForFlow = null;
let relatedPacketsLoadingForFlow = null;
async function fetchRelatedPackets(flowId) {
    if (relatedPacketsLoadedForFlow === flowId || relatedPacketsLoadingForFlow === flowId) return;
    relatedPacketsLoadingForFlow = flowId;
    $("packetTableBody").replaceChildren();
    const tr = document.createElement("tr");
    const td = cell(tr,"Loading related packets..."); td.colSpan=9;
    $("packetTableBody").appendChild(tr);
    try {
        const related = await api(`/api/flow/${encodeURIComponent(flowId)}/packets`);
        if (activeFlowId !== flowId) return;
        renderPackets(related.result.packets || []);
        relatedPacketsLoadedForFlow = flowId;
    } catch (err) {
        if (activeFlowId === flowId) {
            $("packetTableBody").replaceChildren();
            const row = document.createElement("tr");
            const error = cell(row, `Unable to load packets: ${err.message}`);error.colSpan=9;
            $("packetTableBody").appendChild(row);
        }
    } finally {
        if (relatedPacketsLoadingForFlow === flowId) relatedPacketsLoadingForFlow = null;
    }
}

function checksum(v) { if(missing(v))return "-"; return Number(v)===1?"Valid":Number(v)===0?"Invalid":"-"; }
function renderPackets(packets) {
    const body=$("packetTableBody");body.replaceChildren();
    if(!packets.length){const tr=document.createElement("tr"),td=cell(tr,"No related packet records found.");td.colSpan=9;body.appendChild(tr);return;}
    for(const p of packets){const tr=document.createElement("tr");
        for(const value of [display(p.packet_id),display(p.source),display(p.destination),display(p.ttl),display(p.flags),checksum(p.ip_checksum),checksum(p.l4_checksum),display(p.integrity_state),
            [...(p.triggered_rules || []), ...(p.indicator_rules || [])].join(", ") || "-"
        ])cell(tr,value);
        body.appendChild(tr);
    }
}
function renderShap(explanation) {
    $("shapLoading").classList.add("hidden");
    const root=$("shapResults");root.replaceChildren();root.classList.remove("hidden");
    const description=document.createElement("div");description.className="shap-explanation-box";
    const title=document.createElement("span");title.className="shap-explanation-title";title.textContent="Simple explanation";
    const p=document.createElement("p");p.textContent=display(explanation.plain_explanation);
    description.append(title,p);root.appendChild(description);
    const note=document.createElement("p");note.className="shap-description";
    note.textContent="Negative contributions push toward Benign; positive contributions push toward Attack.";root.appendChild(note);
    const contributions=explanation.top_contributions||[];
    const max=Math.max(.000001,...contributions.map(c=>Number(c.absolute_shap)||0));
    for(const c of contributions){
        const row=document.createElement("div");row.className="shap-row";
        const feature=document.createElement("div");feature.className="shap-feature";feature.textContent=display(c.feature_label||c.feature);
        const value=document.createElement("div");value.className="shap-value";value.textContent=display(c.feature_value);
        const track=document.createElement("div");track.className="shap-track";
        const fill=document.createElement("div");fill.className=`shap-fill ${Number(c.shap_value)>=0?"shap-fill-positive":"shap-fill-negative"}`;
        fill.style.width=`${Math.min(50,Math.abs(Number(c.shap_value)||0)/max*50)}%`;track.appendChild(fill);
        const direction=document.createElement("div");direction.className="shap-direction";direction.textContent=float(c.shap_value,4);
        row.append(feature,value,track,direction);root.appendChild(row);
    }
}
$("shapToggle").addEventListener("click",async()=>{
    const id=activeFlowId;if(!id)return;
    const content=$("shapContent"),opening=content.classList.contains("hidden");
    content.classList.toggle("hidden");$("shapArrow").classList.toggle("open",opening);
    $("shapToggle").setAttribute("aria-expanded",String(opening));
    if(!opening||shapLoadedForFlow===id)return;
    $("shapLoading").textContent="Generating SHAP explanation...";
    try{const data=await api(`/api/flow/${encodeURIComponent(id)}/explain`);if(activeFlowId!==id)return;renderShap(data.explanation);shapLoadedForFlow=id;}
    catch(err){if(activeFlowId===id)$("shapLoading").textContent=`Unable to generate SHAP explanation: ${err.message}`;}
});
$("packetToggle").addEventListener("click",()=>{
    const content=$("packetContent"),opening=content.classList.contains("hidden");
    content.classList.toggle("hidden");$("packetArrow").classList.toggle("open",opening);
    $("packetToggle").setAttribute("aria-expanded",String(opening));
});
function closeModal(){ $("flowModal").classList.add("hidden");document.body.style.overflow="";activeFlowId=null;resetModal(); }
$("closeModal").addEventListener("click",closeModal);
document.querySelector(".modal-backdrop").addEventListener("click",closeModal);
document.addEventListener("keydown",e=>{if(e.key==="Escape" && !$("flowModal").classList.contains("hidden"))closeModal();});

(async()=>{
    try {
        const summary=await api("/api/summary");
        $("sourceName").textContent=display(summary.summary.source_name);
        const timeline=await api("/api/events/timeline");
        renderTimeline(timeline.buckets||[]);
        await loadFlows();
    }catch(err){showError(err.message);$("timelineChart").textContent="Timeline unavailable.";}
})();
