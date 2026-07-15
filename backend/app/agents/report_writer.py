"""Report Writer Agent — generates the final structured audit report."""
import json
import logging
from datetime import datetime

from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, AuditReport, Finding
from app.services.reconciliation_payload import build_reconciliation_payload
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

_REPORT_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are a professional financial auditor writing a formal audit report.
Generate a comprehensive audit report based on the provided findings.
Use clear, professional language. Be specific about each finding with citations.

Structure the report as JSON with this exact schema:
{{
  "title": "Audit Report - [Company/Document Set Name]",
  "executive_summary": "2-3 sentences summarizing overall audit outcome",
  "documents_reviewed": ["list of document filenames"],
  "recommendations": ["list of actionable recommendations"],
  "overall_risk": "critical|high|medium|low|clean"
}}"""),
    ("human", """Audit Session ID: {audit_id}
Documents: {documents}
Findings ({count} total):
{findings_json}

Report language: English
Generate the audit report now."""),
])

_REPORT_PROMPT_AR = ChatPromptTemplate.from_messages([
    ("system", """أنت مراجع حسابات مالي محترف تكتب تقرير مراجعة رسمي.
قم بإنشاء تقرير مراجعة شامل بناءً على النتائج المقدمة.
استخدم لغة مهنية واضحة. كن محددًا بشأن كل نتيجة مع الاستشهادات.

قم بهيكلة التقرير كـ JSON بهذا الشكل بالضبط:
{{
  "title": "تقرير المراجعة - [اسم الشركة/مجموعة المستندات]",
  "executive_summary": "2-3 جمل تلخص نتيجة المراجعة الإجمالية",
  "documents_reviewed": ["قائمة بأسماء المستندات"],
  "recommendations": ["قائمة من التوصيات القابلة للتنفيذ"],
  "overall_risk": "critical|high|medium|low|clean"
}}"""),
    ("human", """معرف جلسة المراجعة: {audit_id}
المستندات: {documents}
النتائج ({count} إجمالي):
{findings_json}

لغة التقرير: العربية
قم بإنشاء تقرير المراجعة الآن."""),
])


def _emit(writer, step_type: str, content: str, **kwargs) -> ReasoningStep:
    step = ReasoningStep(agent="report_writer", step_type=step_type, content=content, **kwargs)
    writer({"type": "reasoning_step", "step": step.model_dump(mode="json")})
    return step


def _determine_overall_risk(findings: list[Finding]) -> str:
    critical = sum(1 for f in findings if f.severity == "critical")
    warnings = sum(1 for f in findings if f.severity == "warning")
    if critical >= 3:
        return "critical"
    if critical >= 1:
        return "high"
    if warnings >= 3:
        return "medium"
    if warnings >= 1:
        return "low"
    return "clean"


async def report_writer_agent(state: AuditState) -> dict:
    """
    Report Writer Agent Node.
    Synthesizes all findings into a structured audit report (Arabic or English).
    """
    writer = get_stream_writer()
    new_steps: list[ReasoningStep] = []
    findings = state.get("findings", [])
    documents = state["documents"]
    audit_id = state["audit_id"]
    report_language = state.get("report_language", "english")

    step = _emit(writer, "thought",
                 f"Starting report generation in {report_language}. "
                 f"I have {len(findings)} finding(s) to incorporate.")
    new_steps.append(step)

    if findings:
        step = _emit(writer, "thought",
                     f"Attaching source citations for {len(findings)} finding(s).")
        new_steps.append(step)

    # Count by severity
    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    ok = [f for f in findings if f.severity == "ok"]
    overall_risk = _determine_overall_risk(findings)

    step = _emit(writer, "thought",
                 f"Findings breakdown: {len(critical)} critical, {len(warnings)} warnings, {len(ok)} OK. "
                 f"Overall risk assessment: {overall_risk.upper()}.")
    new_steps.append(step)

    step = _emit(writer, "tool_call",
                 f"Generating structured {report_language} audit report with LLM...",
                 tool_name="generate_report",
                 tool_input={"language": report_language, "finding_count": len(findings)})
    new_steps.append(step)

    # Prepare inputs for LLM
    doc_names = [d.filename for d in documents]
    findings_summary = []
    for f in findings:
        findings_summary.append({
            "finding_id": f.finding_id[:8],
            "severity": f.severity,
            "title": f.title,
            "description": f.description,
            "confidence": f.confidence_score,
            "evidence": f.evidence[:3],
            "recommendation": f.recommendation,
        })

    try:
        llm = get_llm(temperature=0.2)
        prompt = _REPORT_PROMPT_AR if report_language == "arabic" else _REPORT_PROMPT_EN
        chain = prompt | llm

        response = await chain.ainvoke({
            "audit_id": audit_id,
            "documents": ", ".join(doc_names),
            "count": len(findings),
            "findings_json": json.dumps(findings_summary, ensure_ascii=False, indent=2),
        })

        # Parse LLM response
        content = response.content
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        report_data = json.loads(content)

        # overall_risk is always deterministic from accepted findings.
        # The LLM suggestion is discarded to prevent hallucinated risk labels.
        llm_risk_suggestion = report_data.get("overall_risk", "")
        if llm_risk_suggestion and llm_risk_suggestion != overall_risk:
            logger.info(
                "LLM suggested overall_risk=%r but deterministic value=%r — using deterministic.",
                llm_risk_suggestion,
                overall_risk,
            )
        report = AuditReport(
            audit_id=audit_id,
            title=report_data.get("title", f"Audit Report — {', '.join(doc_names[:2])}"),
            language=report_language,
            executive_summary=report_data.get("executive_summary", ""),
            documents_reviewed=doc_names,
            findings=findings,
            recommendations=report_data.get("recommendations", []),
            overall_risk=overall_risk,
        )
        snap, ent_conflicts = build_reconciliation_payload(documents, findings)
        report = report.model_copy(
            update={
                "reconciliation_snapshot": snap,
                "entity_conflicts": ent_conflicts,
            }
        )

        step = _emit(writer, "tool_result",
                     f"Report generated: '{report.title}'",
                     tool_name="generate_report",
                     tool_output=f"Report ready — {len(report.findings)} findings, risk: {report.overall_risk}")
        new_steps.append(step)

    except Exception as e:
        logger.error("LLM report generation failed: %s", e)
        step = _emit(writer, "thought", f"LLM report generation encountered an issue ({e}). Building structured report directly.")
        new_steps.append(step)

        # Fallback: build report without LLM
        if report_language == "arabic":
            exec_summary = (
                f"تم مراجعة {len(documents)} مستند(ات). "
                f"تم اكتشاف {len(critical)} مشكلة حرجة و{len(warnings)} تحذير. "
                f"مستوى المخاطر الإجمالي: {overall_risk.upper()}."
            )
            recs = ["مراجعة جميع التناقضات الحرجة مع الأطراف المعنية", "طلب مستندات مصححة حيثما لزم"]
        else:
            exec_summary = (
                f"Reviewed {len(documents)} document(s). "
                f"Found {len(critical)} critical issue(s) and {len(warnings)} warning(s). "
                f"Overall risk level: {overall_risk.upper()}."
            )
            recs = ["Review all critical discrepancies with contracting parties", "Request corrected documents where applicable"]

        report = AuditReport(
            audit_id=audit_id,
            title=f"Audit Report — {', '.join(doc_names[:2])}",
            language=report_language,
            executive_summary=exec_summary,
            documents_reviewed=doc_names,
            findings=findings,
            recommendations=recs,
            overall_risk=overall_risk,
        )
        snap, ent_conflicts = build_reconciliation_payload(documents, findings)
        report = report.model_copy(
            update={
                "reconciliation_snapshot": snap,
                "entity_conflicts": ent_conflicts,
            }
        )

    summary = (
        f"Audit report complete. "
        f"Overall risk: {report.overall_risk.upper()}. "
        f"{len(critical)} critical, {len(warnings)} warning(s). "
        f"Report ready in {report_language}."
    )
    step = _emit(writer, "summary", summary)
    new_steps.append(step)

    return {
        "messages": [AIMessage(content=summary)],
        "report": report,
        "reasoning_trace": new_steps,
    }
