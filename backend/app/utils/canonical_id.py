"""Deterministic UUID5 ids so Neo4j MERGE collapses the same entity/relationship
seen in many chunks or documents into one node/edge."""
import re
import uuid

# Fixed namespace — must not change; rotating it would orphan all existing nodes.
_ENTITY_NAMESPACE = uuid.UUID("6f3a8f5a-2c1b-4d3e-8a9f-1b2c3d4e5f6a")

# Global identity (same value in any doc = same entity); other types are doc-scoped.
_GLOBAL_DEDUPE_TYPES = frozenset({"contract_id", "invoice_id", "company", "person"})

# Identifier types where formatting is noise, not meaning (CTR-2024-044 == CTR2024044).
_IDENTIFIER_DEDUPE_TYPES = frozenset({"contract_id", "invoice_id"})

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_identifier(value: str) -> str:
    """Alphanumeric core of an identifier: "CTR-2024-044" → "ctr2024044"."""
    return _NON_ALNUM_RE.sub("", (value or "").strip().lower())


def canonical_entity_id(
    entity_type: str,
    normalized_value: str,
    doc_id: str,
    amount_value: float | None = None,
    amount_currency: str | None = None,
    txn_ref: str | None = None,
    txn_date: str | None = None,
) -> str:
    """Deterministic UUID5 for an entity.

    Amounts key on parsed value + currency (+ txn_ref/txn_date for bank rows, so
    same-amount transactions stay distinct). Global types collapse across docs.
    """
    etype = (entity_type or "other").strip().lower()

    if etype == "amount" and amount_value is not None:
        cur = (amount_currency or "").strip().upper() or "NONE"
        ref = normalize_identifier(txn_ref or "")
        tdate = (txn_date or "").strip().lower()
        txn_part = f"::{ref}::{tdate}" if (ref or tdate) else ""
        # round to cents to suppress float noise
        key = f"amount::{doc_id or ''}::{cur}::{round(float(amount_value), 2)}{txn_part}"
        return str(uuid.uuid5(_ENTITY_NAMESPACE, key))

    norm = (normalized_value or "").strip().lower()
    if etype in _IDENTIFIER_DEDUPE_TYPES:
        norm = normalize_identifier(norm) or norm  # fall back if stripping leaves nothing
    if etype in _GLOBAL_DEDUPE_TYPES:
        key = f"{etype}::{norm}"
    else:
        key = f"{etype}::{doc_id or ''}::{norm}"
    return str(uuid.uuid5(_ENTITY_NAMESPACE, key))


def canonical_chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id from (doc_id, chunk_index) so re-uploads upsert."""
    return str(uuid.uuid5(_ENTITY_NAMESPACE, f"chunk::{doc_id}::{int(chunk_index)}"))


def canonical_rel_id(source_entity_id: str, target_entity_id: str, relationship_type: str) -> str:
    """Deterministic UUID5 for a relationship (same edge from N chunks → one edge)."""
    rel = (relationship_type or "other").strip().lower()
    key = f"{source_entity_id}->{target_entity_id}::{rel}"
    return str(uuid.uuid5(_ENTITY_NAMESPACE, key))
