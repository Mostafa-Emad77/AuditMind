"""Tools for the Audit Planner Agent."""
import json
import logging
from langchain_core.tools import tool
from app.models.schemas import ChecklistItem
from app.utils.llm_factory import get_llm
from app.utils.llm_json import extract_json_obj as _extract_json_obj, normalize_llm_content

logger = logging.getLogger(__name__)
_VALID_CHECK_TYPES = {
    "amount_match",
    "date_consistency",
    "party_match",
    "clause_completeness",
    "signature_check",
    "cross_doc_consistency",
    "other",
}
_VALID_PRIORITIES = {"high", "medium", "low"}


def _generate_checklist_deterministic(doc_types: list[str], doc_ids: list[str]) -> list[dict]:
    """Existing rule-based checklist generation (fallback)."""
    doc_type_set = set(doc_types)
    checklist_items = []

    # Invoice + Contract pair
    if "invoice" in doc_type_set and "contract" in doc_type_set:
        checklist_items.extend([
            ChecklistItem(
                description="Verify invoice amounts match contracted payment terms",
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Confirm party names (client, vendor) are consistent across invoice and contract",
                check_type="party_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Check invoice date falls within contract validity period",
                check_type="date_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Verify VAT/tax amounts are correctly calculated per contract terms",
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="medium",
            ).model_dump(),
            ChecklistItem(
                description="Verify contract payment schedule milestones (including monthly retainers and final deliverables) are fully invoiced; flag missing months or milestones",
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Reconcile invoice subtotal against explicitly listed line items/components and flag unexplained subtotal gaps",
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
        ])

    # Bank statement cross-check
    if "bank_statement" in doc_type_set:
        checklist_items.extend([
            ChecklistItem(
                description="Cross-reference invoice/payment amounts with bank statement transactions",
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Verify payment dates in bank statement match invoice due dates",
                check_type="date_consistency",
                doc_ids_involved=doc_ids,
                priority="medium",
            ).model_dump(),
        ])

    # Deterministic bank vs contract / invoice reconciliation
    if "bank_statement" in doc_type_set and "contract" in doc_type_set:
        checklist_items.extend([
            ChecklistItem(
                description=(
                    "Reconcile total bank payments against contract total value; "
                    "flag material overpayment or underpayment beyond tolerance"
                ),
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description=(
                    "Reconcile contract milestone payment schedule against actual bank payments "
                    "(strict per-installment and cumulative running totals)"
                ),
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
        ])

    if "bank_statement" in doc_type_set and "invoice" in doc_type_set:
        checklist_items.append(
            ChecklistItem(
                description=(
                    "Reconcile sum of bank payments against invoice total; "
                    "flag paid-status inconsistency beyond tolerance"
                ),
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
        )

    if {"bank_statement", "invoice", "contract"} <= doc_type_set:
        checklist_items.append(
            ChecklistItem(
                description=(
                    "Verify invoice covers all contract periods that have bank payment activity; "
                    "flag any months or milestones with bank payments but no corresponding invoice line"
                ),
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
        )

    if "balance_sheet" in doc_type_set:
        checklist_items.extend([
            ChecklistItem(
                description="Verify total assets equal total liabilities plus equity",
                check_type="amount_match",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Check for unusual or unexplained large entries",
                check_type="other",
                doc_ids_involved=doc_ids,
                priority="medium",
            ).model_dump(),
        ])

    if "audit_report" in doc_type_set:
        checklist_items.extend([
            ChecklistItem(
                description="Verify figures cited in audit report match source documents",
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ).model_dump(),
            ChecklistItem(
                description="Check that all required audit report clauses are present",
                check_type="clause_completeness",
                doc_ids_involved=doc_ids,
                priority="medium",
            ).model_dump(),
        ])

    checklist_items.extend([
        ChecklistItem(
            description="Detect any cross-document amount contradictions using knowledge graph",
            check_type="cross_doc_consistency",
            doc_ids_involved=doc_ids,
            priority="high",
        ).model_dump(),
        ChecklistItem(
            description="Check for duplicate invoice or contract IDs",
            check_type="other",
            doc_ids_involved=doc_ids,
            priority="medium",
        ).model_dump(),
        ChecklistItem(
            description="Verify all dates are logically consistent (no future-dated payments, etc.)",
            check_type="date_consistency",
            doc_ids_involved=doc_ids,
            priority="medium",
        ).model_dump(),
    ])
    return checklist_items


def _generate_checklist_llm(doc_types: list[str], doc_ids: list[str]) -> list[dict]:
    """LLM-driven checklist generation with schema validation."""
    llm = get_llm(temperature=0.0)
    prompt = (
        "Generate an audit checklist for these document types.\n"
        "Output JSON only:\n"
        "{\n"
        '  "checklist": [\n'
        "    {\n"
        '      "description": "string",\n'
        '      "check_type": "amount_match|date_consistency|party_match|clause_completeness|signature_check|cross_doc_consistency|other",\n'
        '      "priority": "high|medium|low"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Requirements:\n"
        "- Focus on cross-document reconciliation and contradiction detection.\n"
        "- If contract+invoice exist, include payment schedule and subtotal reconciliation checks.\n"
        "- If bank_statement+contract exist, include total paid vs contract total and milestone-vs-payment checks.\n"
        "- If bank_statement+invoice exist, include total paid vs invoice total check.\n"
        "- If all three exist, include missing invoice coverage for paid periods/milestones.\n"
        "- Include 6-12 checks total.\n\n"
        f"doc_types: {json.dumps(doc_types)}\n"
        f"doc_ids: {json.dumps(doc_ids)}\n"
    )
    resp = llm.invoke(prompt)
    content = normalize_llm_content(resp)
    data = _extract_json_obj(content) or {}
    raw_items = data.get("checklist", [])
    if not isinstance(raw_items, list):
        return []
    out: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        ctype = str(item.get("check_type", "other")).strip()
        prio = str(item.get("priority", "medium")).strip()
        if not desc:
            continue
        if ctype not in _VALID_CHECK_TYPES:
            ctype = "other"
        if prio not in _VALID_PRIORITIES:
            prio = "medium"
        try:
            out.append(
                ChecklistItem(
                    description=desc,
                    check_type=ctype,
                    doc_ids_involved=doc_ids,
                    priority=prio,
                ).model_dump()
            )
        except Exception:
            continue
    return out


@tool
def generate_checklist(doc_types_json: str, doc_ids_json: str) -> str:
    """
    Generate a custom audit checklist based on the types of documents provided.
    This tool analyzes what document combinations are present and creates
    targeted checks to perform.

    Args:
        doc_types_json: JSON array of document types (e.g. '["invoice", "contract"]')
        doc_ids_json: JSON array of corresponding document IDs

    Returns:
        JSON string with a list of audit checklist items to verify
    """
    try:
        doc_types = json.loads(doc_types_json)
        doc_ids = json.loads(doc_ids_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})
    checklist_items = []
    try:
        checklist_items = _generate_checklist_llm(doc_types, doc_ids)
    except Exception as e:
        logger.warning("LLM checklist generation failed: %s", e)
    if not checklist_items:
        checklist_items = _generate_checklist_deterministic(doc_types, doc_ids)

    return json.dumps({
        "checklist": checklist_items,
        "total_items": len(checklist_items),
        "doc_types_analyzed": list(set(doc_types)),
    })
