"""Pure helper functions for evidence analysis — no heavy service imports."""
from typing import Any


from app.models.schemas import ReasoningStep
from app.utils.arabic_normalizer import extract_amounts
from app.utils.money_parse import parse_monetary_amount, amounts_within_tiny_tolerance


def _emit(writer, step_type: str, content: str, **kwargs) -> ReasoningStep:
    step = ReasoningStep(agent="cross_checker", step_type=step_type, content=content, **kwargs)
    writer({"type": "reasoning_step", "step": step.model_dump(mode="json")})
    return step


def _amounts_in_snippet(text: str) -> list[float]:
    out: list[float] = []
    for am in extract_amounts(text):
        v = am.get("value")
        if v is not None:
            out.append(float(v))
            continue
        raw = (am.get("raw") or "").strip()
        p = parse_monetary_amount(raw)
        if p is not None:
            out.append(float(p))
    return out


def _evidence_line_backs_value(evidence_line: str, monetary_raw: str) -> bool:
    """True if the evidence text contains a parsed amount matching monetary_raw."""
    target = parse_monetary_amount(str(monetary_raw).strip())
    if target is None:
        return False
    tail = evidence_line.split(":", 1)[-1].strip() if ":" in evidence_line else evidence_line
    direct = parse_monetary_amount(tail)
    if direct is not None and amounts_within_tiny_tolerance(direct, target):
        return True
    for am in extract_amounts(tail):
        v = am.get("value")
        if v is not None and amounts_within_tiny_tolerance(float(v), target):
            return True
    whole = parse_monetary_amount(evidence_line)
    return whole is not None and amounts_within_tiny_tolerance(whole, target)


def _evidence_snippets_back_values(evidence: list[str], *values: float) -> bool:
    """True if each value appears (within tiny tolerance) in at least one evidence snippet."""
    if not values:
        return True
    collected: list[float] = []
    for ev in evidence:
        tail = ev.split(":", 1)[-1].strip() if ":" in ev else ev
        collected.extend(_amounts_in_snippet(tail))
    for val in values:
        if not any(amounts_within_tiny_tolerance(c, val) for c in collected):
            return False
    return True


def _graph_context_for_compare(c: dict, doc1: str, doc2: str) -> str:
    parts = [
        "Detected via knowledge graph: amounts linked through a shared anchor entity.",
        f"Documents: {doc1} (page {c.get('page1', 0)}) and {doc2} (page {c.get('page2', 0)}).",
    ]
    if c.get("anchor_type") or c.get("anchor_value"):
        parts.append(
            f"Anchor: {c.get('anchor_type', 'entity')} = {c.get('anchor_value', '')!r} "
            f"(id={c.get('anchor_entity_id', '')})."
        )
    return " ".join(parts)


def _is_bank_doc(doc: Any) -> bool:
    fn = (getattr(doc, "filename", "") or "").lower()
    dt = (getattr(doc, "doc_type", "") or "").lower()
    return dt == "bank_statement" or any(k in fn for k in ("bank", "statement", "stmt"))


# Rule-based intent classification keyword sets (no LLM).
_BANK_RECON_KEYWORDS = (
    "bank", "statement", "reconcil", "payment", "paid", "transfer",
    "disbursement", "remittance", "deposit", "debit", "credit", "transaction",
    "بنك", "تحويل", "دفع", "سداد", "كشف",
)
_INVOICE_KEYWORDS = ("invoice", "billed", "billing", "فاتورة")
_MILESTONE_KEYWORDS = ("milestone", "installment", "دفعة", "مرحلة")
_SCHEDULE_KEYWORDS = ("schedule", "due date", "payment date", "مواعيد", "جدول")


def _classify_check_intents(description: str, check_type: str) -> set[str]:
    """
    Rule-based checklist intent classification (no LLM call).
    Returns intents among: schedule, subtotal, bank_contract_recon, invoice_bank_recon, milestone_bank_recon, generic.
    """
    blob = f"{description} {check_type}".lower()
    intents: set[str] = set()

    is_amountish = check_type in ("amount_match", "cross_doc_consistency")
    if is_amountish and any(k in blob for k in _BANK_RECON_KEYWORDS):
        intents.add("bank_contract_recon")
        if any(k in blob for k in _INVOICE_KEYWORDS):
            intents.add("invoice_bank_recon")
        if any(k in blob for k in _MILESTONE_KEYWORDS):
            intents.add("milestone_bank_recon")

    if check_type == "date_consistency" or any(k in blob for k in _SCHEDULE_KEYWORDS):
        intents.add("schedule")

    if not intents:
        intents.add("generic")
    return intents
