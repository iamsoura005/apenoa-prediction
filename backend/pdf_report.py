import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from typing import Dict, Any

def build_pdf_report(analysis_data: Dict[str, Any]) -> bytes:
    """
    Constructs a polished, clinical-grade Sleep Apnea PPG Forecast Report in PDF format.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0f172a'),
        leading=24
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        leading=14
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1e293b'),
        leading=16,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#334155'),
        leading=13
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#64748b'),
        leading=11
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("APNEASPHERE — Clinical PPG Prediction Report", title_style))
    story.append(Paragraph(
        f"Generated on {analysis_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | "
        f"Analysis ID: {analysis_data.get('analysis_id', 'N/A')[:8]}", 
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceBefore=8, spaceAfter=14))
    
    # Patient & Recording Summary
    pred = analysis_data.get('prediction', 'UNKNOWN')
    prob = analysis_data.get('probability', 0.0)
    risk = analysis_data.get('risk_level', 'LOW')
    risk_color_hex = '#ef4444' if risk == 'HIGH' else ('#f59e0b' if risk == 'MODERATE' else '#10b981')
    
    summary_data = [
        [
            Paragraph("<b>Subject / Recording ID:</b>", body_style),
            Paragraph(str(analysis_data.get('subject_id', 'Anonymous')), body_style),
            Paragraph("<b>Analysis Window:</b>", body_style),
            Paragraph("30 seconds (80 Hz)", body_style)
        ],
        [
            Paragraph("<b>Forecast Lead Time:</b>", body_style),
            Paragraph("5.0 seconds", body_style),
            Paragraph("<b>Forecast Target Window:</b>", body_style),
            Paragraph("10.0 seconds", body_style)
        ],
        [
            Paragraph("<b>Signal Quality (SQI):</b>", body_style),
            Paragraph(f"{analysis_data.get('signal_quality', {}).get('quality', 'good').upper()} (Score: {analysis_data.get('signal_quality', {}).get('sqi_score', 1.0)})", body_style),
            Paragraph("<b>Model Version:</b>", body_style),
            Paragraph("Apnea1DCNN v1.0.0 (PyTorch)", body_style)
        ]
    ]
    t_summary = Table(summary_data, colWidths=[130, 135, 130, 135])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))
    
    # Primary Forecast Box
    story.append(Paragraph("Diagnostic Forecast Result", section_heading))
    result_box_data = [
        [
            Paragraph(f"<font size=14 color='{risk_color_hex}'><b>{pred.replace('_', ' ')}</b></font>", body_style),
            Paragraph(f"<b>Event Probability:</b> {prob*100:.1f}%<br/><b>Risk Tier:</b> {risk}", body_style)
        ],
        [
            Paragraph(f"<b>Clinical Interpretation:</b> {analysis_data.get('clinical_interpretation', 'N/A')}", body_style),
            Paragraph(f"<b>SQI Integrity:</b> {analysis_data.get('signal_quality', {}).get('message', 'Normal')}", body_style)
        ]
    ]
    t_result = Table(result_box_data, colWidths=[265, 265])
    t_result.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(risk_color_hex)),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_result)
    story.append(Spacer(1, 14))
    
    # Autonomic Biomarkers Table
    story.append(Paragraph("Autonomic & Heart Rate Variability (HRV) Biomarkers", section_heading))
    bm = analysis_data.get('biomarkers', {})
    hrv = bm.get('hrv', {})
    spec = bm.get('spectral', {})
    
    bio_table_data = [
        [Paragraph("<b>Biomarker</b>", body_style), Paragraph("<b>Observed Value</b>", body_style), Paragraph("<b>Physiological Reference Range</b>", body_style), Paragraph("<b>Clinical Implication in OSA</b>", body_style)],
        [Paragraph("Mean Heart Rate", body_style), Paragraph(f"{hrv.get('mean_hr_bpm', 0)} bpm", body_style), Paragraph("60 - 90 bpm", body_style), Paragraph("Tachycardia/Bradycardia cycle indicator", body_style)],
        [Paragraph("SDNN (Total Variability)", body_style), Paragraph(f"{hrv.get('sdnn_ms', 0)} ms", body_style), Paragraph("30 - 100 ms", body_style), Paragraph("Reduced variability correlates with autonomic fatigue", body_style)],
        [Paragraph("RMSSD (Parasympathetic)", body_style), Paragraph(f"{hrv.get('rmssd_ms', 0)} ms", body_style), Paragraph("20 - 60 ms", body_style), Paragraph("Reflects vagal regulatory modulation", body_style)],
        [Paragraph("LF/HF Spectral Ratio", body_style), Paragraph(f"{spec.get('lf_hf_ratio', 0)}", body_style), Paragraph("0.5 - 2.5", body_style), Paragraph("Elevated ratio (>2.5) indicates pre-apnea sympathetic surge", body_style)],
        [Paragraph("Detected Systolic Peaks", body_style), Paragraph(f"{hrv.get('peak_count', 0)} beats", body_style), Paragraph("25 - 50 per 30s", body_style), Paragraph("Pulse interval integrity verification", body_style)]
    ]
    t_bio = Table(bio_table_data, colWidths=[140, 90, 130, 170])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 20))
    
    # Regulatory & Research Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8'), spaceBefore=10, spaceAfter=8))
    story.append(Paragraph(
        "<b>RESEARCH & INVESTIGATIONAL USE ONLY:</b> This report is generated by an artificial intelligence "
        "inference system (1D Convolutional Neural Network) analyzing photoplethysmography (PPG) waveforms. "
        "It is designed solely for clinical research and decision-support exploration. This automated prediction "
        "does not constitute a definitive medical diagnosis, nor should it replace comprehensive polysomnography (PSG) "
        "supervised by a qualified sleep medicine physician.", 
        disclaimer_style
    ))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

