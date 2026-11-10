"""NetworkIDS: export a current-analysis security report (no PCAP processing).

Only existing summary/overview/flow results are presented. Recommendation
mapping is a deterministic, conservative investigation aid, NOT remediation.
"""
from datetime import datetime, timezone
from xml.sax.saxutils import escape
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, HRFlowable,
)

NAVY = colors.HexColor('#122238')
BLUE = colors.HexColor('#316ac9')
GREY = colors.HexColor('#53657a')
PALE = colors.HexColor('#f2f5fa')
BORDER = colors.HexColor('#dbe3ed')
RED = colors.HexColor('#d84d5c')
GREEN = colors.HexColor('#23845c')
GOLD = colors.HexColor('#bc811d')
PURPLE = colors.HexColor('#9546b0')

RULE_DESCRIPTIONS = {
    'T1': 'Invalid IPv4 header checksum',
    'T2': 'Contradictory IPv4 fragmentation fields',
    'T3': 'Invalid IPv4 header length',
    'S1': 'Unusually low TTL',
    'S2': 'Capture-local MAC-IP inconsistency',
    'S3': 'TCP SYN and FIN flags set together',
    'S4': 'TCP SYN and RST flags set together',
    'S5': 'TCP FIN and RST flags set together',
    'S6': 'TCP NULL flags',
    'S7': 'Invalid Layer 4 checksum (indicator only)',
}

INVESTIGATION_ACTIONS = {
    'T1': 'Check the affected IPv4 header fields and validate the capture conditions before treating the checksum as evidence of corruption.',
    'T2': 'Inspect fragmentation flags, fragment offsets and packet lengths; verify whether packet construction or parsing explains the inconsistency.',
    'T3': 'Inspect the IPv4 header length and the original packet representation for malformed or truncated headers.',
    'S1': 'Review the observed TTL against expected routing paths; determine whether the low value is legitimate.',
    'S2': 'Compare the observed source MAC-IP associations with trusted ARP, DHCP or asset records; consider multi-interface and capture-location effects.',
    'S3': 'Review SYN+FIN packets alongside the wider connection sequence and related source activity.',
    'S4': 'Review SYN+RST packets for abnormal flag behaviour and correlate with neighbouring TCP packets.',
    'S5': 'Check FIN+RST combinations against connection termination sequences and packet timing.',
    'S6': 'Investigate NULL-flag TCP packets and correlate with possible scanning behaviour without assuming a confirmed scan.',
    'S7': 'Recheck transport checksums and exclude NIC checksum offloading or partial capture effects before drawing conclusions.',
}


def _text(value, default='-'):
    if value is None: return default
    try:
        if isinstance(value, float) and not math.isfinite(value): return default
    except (TypeError, ValueError): pass
    string = str(value).strip()
    return string if string else default


def _n(value):
    try: return f'{int(float(value)):,}'
    except (ValueError, TypeError, OverflowError): return '-'


def _pct(value):
    try: return f'{float(value):.2f}%'
    except (ValueError, TypeError, OverflowError): return '-'


def _time(timestamp):
    try:
        if timestamp is None: return 'Not available'
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(float(timestamp), tz=ZoneInfo('Asia/Kuala_Lumpur'))
        return dt.strftime('%Y-%m-%d %H:%M:%S MYT')
    except (ValueError, OverflowError, TypeError): return 'Not available'


def _code(rule): return str(rule).split('_', 1)[0]


def _para(s, style): return Paragraph(escape(_text(s)).replace('\n', '<br/>'), style)


def _footer(canvas, doc):
    canvas.saveState()
    width, _ = doc.pagesize
    canvas.setStrokeColor(BORDER)
    canvas.line(19*mm, 17*mm, width-19*mm, 17*mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawString(19*mm, 12*mm, 'NetworkIDS  |  Offline network traffic analysis')
    canvas.drawRightString(width-19*mm, 12*mm, f'Page {doc.page}')
    canvas.restoreState()


def _table(rows, widths, *, header=True, font=8.1):
    t = Table(rows, colWidths=widths, hAlign='LEFT', repeatRows=1 if header else 0)
    commands = [
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1 if header else 0), (-1,-1), [colors.white, PALE]),
        ('FONTSIZE', (0,0), (-1,-1), font),
        ('TEXTCOLOR', (0,0), (-1,-1), NAVY),
    ]
    if header:
        commands.extend([
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ])
    t.setStyle(TableStyle(commands))
    return t


def build_security_report(output, summary, overview, top_attacks):
    """Write PDF into a file-like object. Values are taken from existing analysis."""
    from reportlab.lib.pagesizes import A4
    doc = SimpleDocTemplate(
        output, pagesize=A4, rightMargin=19*mm, leftMargin=19*mm,
        topMargin=19*mm, bottomMargin=24*mm,
        title='Network Traffic Security Analysis Report',
        author='NetworkIDS',
    )
    content_width = A4[0] - 38*mm
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleN', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=21, leading=27, alignment=TA_LEFT, textColor=NAVY, spaceAfter=12))
    styles.add(ParagraphStyle(name='SubN', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12.5, leading=17, textColor=NAVY, spaceBefore=15, spaceAfter=8))
    styles.add(ParagraphStyle(name='BodyN', parent=styles['BodyText'], fontName='Helvetica', fontSize=9, leading=13.6, textColor=NAVY, spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallN', parent=styles['BodyText'], fontName='Helvetica', fontSize=8, leading=11, textColor=GREY, spaceAfter=6))
    styles.add(ParagraphStyle(name='CellN', parent=styles['BodyText'], fontName='Helvetica', fontSize=7.7, leading=11, textColor=NAVY))
    styles.add(ParagraphStyle(name='KpiN', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=BLUE))
    body = styles['BodyN']; small=styles['SmallN']; cell=styles['CellN'];
    story=[]

    def heading(title): story.append(_para(title, styles['SubN']))
    def write(text): story.append(_para(text, body))
    def note(text): story.append(_para(text, small))
    def page(): story.append(PageBreak())
    def label_value_table(items):
        rows=[[_para(str(k), cell), _para(str(v), cell)] for k,v in items]
        story.append(_table(rows,[45*mm,content_width-45*mm],header=False))
    def ranked(title, rows, limit=6):
        heading(title)
        valid=[]
        for item in (rows or [])[:limit]:
            if isinstance(item,dict):
                name=item.get('name','-'); value=item.get('value',0)
                valid.append([_para(name,cell),_para(_n(value),cell)])
        if valid:
            story.append(KeepTogether([story.pop(), _table([['Item','Flow count'],*valid],[content_width-31*mm,31*mm])]))
        else: note('No data available for this breakdown.')

    # Page 1: Summary
    story.append(_para('Network Traffic Security Analysis Report',styles['TitleN']))
    note('OFFLINE PCAP ANALYSIS  |  Automated summary from current NetworkIDS analysis')
    story.append(HRFlowable(width='100%',thickness=1,color=BORDER,spaceAfter=13))
    label_value_table([
        ('Capture file',summary.get('source_name','-')),
        ('Capture start',_time(overview.get('capture_start'))),
        ('Capture end',_time(overview.get('capture_end'))),
        ('Report generated',datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')),
        ('ML model','Random Forest (binary Attack / Benign)'),
    ])
    heading('Executive Summary')
    write('The following counts represent model classifications and independent packet-header checks. An Attack prediction is a signal for analyst investigation, not a confirmed incident or identified attack family.')
    kpis=[
        ('Total Flows',_n(summary.get('total_flows'))),
        ('Predicted Attack Flows',_n(summary.get('attack_flows'))),
        ('Attack Rate',_pct(summary.get('attack_rate'))),
        ('Suspicious Packets',_n(summary.get('suspicious_packets'))),
        ('Tampered Packets',_n(summary.get('tampered_packets'))),
    ]
    cards=[]
    for label, value in kpis:
        cards.append([_para(label,small),_para(value,styles['KpiN'])])
    story.append(_table(cards,[content_width*0.59,content_width*0.41],header=False))
    note(f"Packet integrity categories: Intact {_n(summary.get('intact_packets'))}, Suspicious {_n(summary.get('suspicious_packets'))}, Tampered {_n(summary.get('tampered_packets'))}. S7 is an additional indicator and does not independently change a packet's integrity state.")
    if int(summary.get('attack_flows') or 0):
        write('Analyst priority: examine the highest-probability predicted Attack flows in the Events view and correlate with network context before making incident-response decisions.')
    else:
        write('No flows were classified as Attack in this capture. A Benign prediction does not guarantee the absence of malicious activity.')

    # Page 2: traffic and explainability
    page()
    story.append(_para('Traffic Classification and Explainability',styles['TitleN']))
    heading('Classification Overview')
    label_value_table([
        ('Predicted Attack',_n(summary.get('attack_flows'))),
        ('Predicted Benign',_n(summary.get('benign_flows'))),
        ('Total Flows',_n(summary.get('total_flows'))),
    ])
    ranked('Protocol Distribution',overview.get('protocol_distribution'),6)
    ranked('Top Attack Source IPs',overview.get('top_attack_sources'),5)
    ranked('Top Attack Destination IPs',overview.get('top_attack_destinations'),5)
    ranked('Top Targeted Destination Ports',overview.get('top_attack_ports'),5)
    heading('Global SHAP: Top Contributing Features')
    shap=overview.get('shap_importance') or []
    if shap:
        rows=[['Feature','Mean absolute SHAP']]
        for r in shap[:6]:
            try: val=f"{float(r.get('value')):.5f}"
            except (TypeError,ValueError): val='-'
            rows.append([_para(r.get('name','-'),cell),val])
        story.append(_table(rows,[content_width*0.72,content_width*0.28]))
        note('Feature importance describes model contribution magnitude on sampled flows, not attack causality.')
    else:
        note('Global SHAP importance is not available for this analysis.')
        if overview.get('shap_error'): note(f"Reason: {overview['shap_error']}")

    # Page 3: packet integrity, full rule list if triggered
    page()
    story.append(_para('Packet Header Integrity Findings',styles['TitleN']))
    write('Packet-level integrity checks operate independently of Random Forest flow classification. Suspicious and Tampered are rule-based states; S7 is recorded as an indicator only.')
    label_value_table([
        ('Intact Packets',_n(summary.get('intact_packets'))),
        ('Suspicious Packets',_n(summary.get('suspicious_packets'))),
        ('Tampered Packets',_n(summary.get('tampered_packets'))),
        ('S7 Indicators',_n(summary.get('l4_checksum_indicators'))),
    ])
    heading('Triggered Integrity Rules')
    raw_counts=summary.get('integrity_rule_counts') or {}
    totals={}
    for name,v in raw_counts.items():
        key=_code(name)
        if key not in RULE_DESCRIPTIONS: continue
        try: count=int(v)
        except (ValueError,TypeError): continue
        totals[key]=totals.get(key,0)+count
    triggered=[(rule,totals[rule]) for rule in RULE_DESCRIPTIONS if totals.get(rule,0)>0]
    if triggered:
        rule_rows=[['Rule','Description','Packets']]
        for rule,count in triggered:
            rule_rows.append([rule,_para(RULE_DESCRIPTIONS[rule],cell),_n(count)])
        story.append(_table(rule_rows,[17*mm,content_width-48*mm,31*mm]))
        note('Rule counts can overlap: one packet may trigger more than one rule. They must not be summed to calculate unique packet totals.')
    else:
        note('No packet integrity rules were triggered in the current analysis.')
    heading('Recommended Investigation Actions')
    if triggered:
        for rule,count in triggered:
            story.append(KeepTogether([
                _para(f'{rule} - {RULE_DESCRIPTIONS[rule]} ({_n(count)} packets)',cell),
                _para(INVESTIGATION_ACTIONS[rule],body),
                Spacer(1, 3*mm),
            ]))
    else:
        write('No rule-specific investigation action is available because no integrity rule was triggered. Continue routine review of the traffic classification and capture quality.')

    # Page 4: selected flows + limitations
    page()
    story.append(_para('Selected Flows and Report Limitations',styles['TitleN']))
    heading('Top Predicted Attack Flows')
    note('Selected by highest Random Forest Attack probability, up to 10 flows; probability is a model output, not a verified incident likelihood.')
    if top_attacks:
        rows=[['Flow','Source','Destination','P(Attack)','Integrity']]
        for flow in top_attacks[:10]:
            src=f"{_text(flow.get('src_ip'))}:{_text(flow.get('src_port'))}"
            dst=f"{_text(flow.get('dst_ip'))}:{_text(flow.get('dst_port'))}"
            try: prob=f"{float(flow.get('attack_probability'))*100:.2f}%"
            except (TypeError,ValueError): prob='-'
            rows.append([
                _para(flow.get('flow_id'),cell),
                _para(src,cell),_para(dst,cell),prob,
                _para(flow.get('integrity_state'),cell),
            ])
        story.append(_table(rows,[22*mm,51*mm,51*mm,23*mm,content_width-147*mm],font=7.3))
    else: note('No predicted Attack flows to list.')
    heading('Methodology')
    write('The uploaded packet capture is parsed into L2-L4 packet features and aggregated into flows. A trained Random Forest assigns a binary flow classification. SHAP values explain model feature contributions. Independent rule-based checks assess packet-header integrity.')
    heading('Scope and Limitations')
    for line in [
        'Results describe the provided offline capture only; this is not a continuous real-time monitor.',
        'Attack/Benign predictions do not identify an attack technique or establish that an incident occurred.',
        'SHAP explains model behaviour; it is not proof of attacker intention or a vulnerability.',
        'Header anomalies may result from capture conditions, offloading, malformed traffic, or non-malicious network behaviour.',
        'Investigation actions are predefined guidance, not automatic mitigation, blocking, or remediation decisions.',
        'The report lists selected flows and summaries, not every packet or a complete forensic record.',
    ]:
        write('- ' + line)
    note('For packet-level evidence and individual SHAP explanations, consult the corresponding NetworkIDS Events page and original capture.')
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
