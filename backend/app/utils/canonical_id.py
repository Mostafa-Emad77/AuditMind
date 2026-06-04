"""Deterministic ID helpers for entities and relationships.

By deriving entity_id from (entity_type, normalized_value [, doc_id]) we make
Neo4j's MERGE collapse the same real-world thing into one node even when it
appears across many chunks or documents. This is what enables cross-doc graph
traversal (e.g. find_contradictions) to actually work.

Likewise, rel_id is derived from (source_id, target_id, relationship_type) so
the same relationship extracted from multiple chunks does not create duplicate
edges.

We intentionally keep `entity_id` / `rel_id` as the storage key (no schema
migration required) — only the *values* become deterministic.
"""
import uuid

# Fixed namespace — must not change; rotating it would orphan all existing nodes.
_ENTITY_NAMESPACE = uuid.UUID("6f3a8f5a-2c1b-4d3e-8a9f-1b2c3d4e5f6a")

# Entity types whose identity is *global* (same value in any doc = same entity).
# Other types (amount, date, clause, other) are scoped to their source document
# because the same surface form can mean different things in different contexts.
_GLOBAL_DEDUPE_TYPES = frozenset({"contract_id", "invoice_id", "company", "person"})


def canonical_entity_id(
    entity_type: str,
    normalized_value: str,
    doc_id: str,
    amount_value: float | None = None,
    amount_currency: str | None = None,
) -> str:
    """Return a deterministic UUID5 for an entity.

    Global types (companies, parties, contract/invoice IDs) collapse across docs.
    Other types are doc-scoped.

    Amount entities: when `amount_value` is provided, the key uses the parsed
    numeric value + currency instead of the surface `normalized_value`. This
    ensures "50,000 EGP", "50000 EGP" and "EGP 50000.00" within a single doc
    collapse into one node.
    """
    etype = (entity_type or "other").strip().lower()

    if etype == "amount" and amount_value is not None:
        cur = (amount_currency or "").strip().upper() or "NONE"
        # round to cents to suppress float noise
        key = f"amount::{doc_id or ''}::{cur}::{round(float(amount_value), 2)}"
        return str(uuid.uuid5(_ENTITY_NAMESPACE, key))

    norm = (normalized_value or "").strip().lower()
    if etype in _GLOBAL_DEDUPE_TYPES:
        key = f"{etype}::{norm}"
    else:
        key = f"{etype}::{doc_id or ''}::{norm}"
    return str(uuid.uuid5(_ENTITY_NAMESPACE, key))


def canonical_chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point ID derived from (doc_id, chunk_index).

    Re-uploading the same document upserts onto the same points instead of
    creating duplicates.
    """
    return str(uuid.uuid5(_ENTITY_NAMESPACE, f"chunk::{doc_id}::{int(chunk_index)}"))


def canonical_rel_id(source_entity_id: str, target_entity_id: str, relationship_type: str) -> str:
    """Return a deterministic UUID5 for a relationship.

    Two extractions of "Invoice X total_value_of Contract Y" become a single
    edge instead of N parallel edges.
    """
    rel = (relationship_type or "other").strip().lower()
    key = f"{source_entity_id}->{target_entity_id}::{rel}"
    return str(uuid.uuid5(_ENTITY_NAMESPACE, key))
