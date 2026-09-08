"""Build and query the Neo4j knowledge graph from extracted entities and relationships."""
import logging
import re
from typing import Optional

from neo4j import GraphDatabase, Driver

from app.config import get_settings
from app.models.schemas import Entity, Relationship, DocumentMeta
from app.utils.canonical_id import normalize_identifier

logger = logging.getLogger(__name__)

_driver: Optional[Driver] = None


def _create_driver() -> Driver:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_lifetime=300,      # close connections after 5 min idle
        keep_alive=True,
    )
    driver.verify_connectivity()
    logger.info("Neo4j connectivity verified")
    return driver


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = _create_driver()
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def _run_with_reconnect(fn):
    """Run fn(driver). On routing/connection failure, reset driver and retry once."""
    global _driver
    try:
        return fn(get_driver())
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ("routing", "connection", "closed", "reset", "timeout")):
            logger.warning("Neo4j connection stale (%s) — reconnecting...", e)
            close_driver()
            return fn(get_driver())
        raise


def init_graph_schema() -> None:
    """Create constraints and indexes on first use."""
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
            session.run("CREATE INDEX entity_doc IF NOT EXISTS FOR (e:Entity) ON (e.source_doc_id)")
        logger.info("Neo4j schema initialized")

    _run_with_reconnect(_run)


def delete_audit_documents(doc_ids: list[str]) -> None:
    """Delete the audit's Document nodes; entities go only once no FOUND_IN edge
    remains (global-dedupe entities are shared across audits)."""
    if not doc_ids:
        return
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                UNWIND $doc_ids AS did
                MATCH (d:Document {doc_id: did})
                OPTIONAL MATCH (e:Entity)-[r:FOUND_IN]->(d)
                DELETE r
                WITH collect(DISTINCT e) AS candidates, collect(DISTINCT d) AS docs
                FOREACH (d IN docs | DETACH DELETE d)
                WITH candidates
                UNWIND candidates AS e
                WITH DISTINCT e
                WHERE e IS NOT NULL AND NOT (e)-[:FOUND_IN]->()
                DETACH DELETE e
                """,
                doc_ids=doc_ids,
            )
        logger.info("Deleted Neo4j Document nodes and orphaned entities for %d doc(s)", len(doc_ids))

    _run_with_reconnect(_run)


def store_document_node(meta: DocumentMeta) -> None:
    """Create or update a Document node in Neo4j."""
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.filename = $filename,
                    d.doc_type = $doc_type,
                    d.language = $language,
                    d.page_count = $page_count
                """,
                doc_id=meta.doc_id,
                filename=meta.filename,
                doc_type=meta.doc_type,
                language=meta.language,
                page_count=meta.page_count,
            )

    _run_with_reconnect(_run)


def store_entities(entities: list[Entity]) -> None:
    """Batch-store entities as nodes in Neo4j."""
    if not entities:
        return
    settings = get_settings()
    records = [
        {
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "value": e.value,
            "normalized_value": e.normalized_value,
            "source_doc_id": e.source_doc_id,
            "source_page": e.source_page,
            "confidence": e.confidence,
            "amount_role": getattr(e, "amount_role", None) or "unknown",
            "source_language": getattr(e, "source_language", None) or "unknown",
            "coreference_note": getattr(e, "coreference_note", None),
            "western_numeral_form": getattr(e, "western_numeral_form", None),
            "arabic_indic_numeral_form": getattr(e, "arabic_indic_numeral_form", None),
            "numeral_mismatch": getattr(e, "numeral_mismatch", None),
            "amount_value": getattr(e, "amount_value", None),
            "amount_currency": getattr(e, "amount_currency", None),
            "txn_ref": getattr(e, "txn_ref", None),
            "txn_date": getattr(e, "txn_date", None),
            "role_title": getattr(e, "role_title", None),
            "signing_authority_level": getattr(e, "signing_authority_level", None),
            "document_signed": getattr(e, "document_signed", None),
        }
        for e in entities
    ]

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                UNWIND $records AS rec
                MERGE (e:Entity {entity_id: rec.entity_id})
                ON CREATE SET
                    e.entity_type = rec.entity_type,
                    e.value = rec.value,
                    e.normalized_value = rec.normalized_value,
                    e.source_doc_id = rec.source_doc_id,
                    e.source_page = rec.source_page,
                    e.confidence = rec.confidence,
                    e.amount_role = rec.amount_role,
                    e.source_language = rec.source_language,
                    e.coreference_note = rec.coreference_note,
                    e.western_numeral_form = rec.western_numeral_form,
                    e.arabic_indic_numeral_form = rec.arabic_indic_numeral_form,
                    e.numeral_mismatch = rec.numeral_mismatch,
                    e.amount_value = rec.amount_value,
                    e.amount_currency = rec.amount_currency,
                    e.txn_ref = rec.txn_ref,
                    e.txn_date = rec.txn_date,
                    e.role_title = rec.role_title,
                    e.signing_authority_level = rec.signing_authority_level,
                    e.documents_signed = CASE
                        WHEN rec.document_signed IS NULL THEN []
                        ELSE [rec.document_signed]
                    END,
                    e.source_doc_ids = [rec.source_doc_id],
                    e.source_pages = [rec.source_page],
                    e.mention_count = 1
                ON MATCH SET
                    e.source_doc_ids = CASE
                        WHEN rec.source_doc_id IN coalesce(e.source_doc_ids, [])
                            THEN e.source_doc_ids
                        ELSE coalesce(e.source_doc_ids, []) + rec.source_doc_id
                    END,
                    e.source_pages = CASE
                        WHEN rec.source_page IN coalesce(e.source_pages, [])
                            THEN e.source_pages
                        ELSE coalesce(e.source_pages, []) + rec.source_page
                    END,
                    e.mention_count = coalesce(e.mention_count, 1) + 1,
                    e.numeral_mismatch = CASE
                        WHEN rec.numeral_mismatch = true THEN true
                        ELSE e.numeral_mismatch
                    END,
                    e.coreference_note = coalesce(e.coreference_note, rec.coreference_note),
                    e.western_numeral_form = coalesce(e.western_numeral_form, rec.western_numeral_form),
                    e.arabic_indic_numeral_form = coalesce(e.arabic_indic_numeral_form, rec.arabic_indic_numeral_form),
                    e.amount_value = coalesce(e.amount_value, rec.amount_value),
                    e.amount_currency = coalesce(e.amount_currency, rec.amount_currency),
                    e.txn_ref = coalesce(e.txn_ref, rec.txn_ref),
                    e.txn_date = coalesce(e.txn_date, rec.txn_date),
                    e.role_title = coalesce(e.role_title, rec.role_title),
                    e.signing_authority_level = coalesce(
                        e.signing_authority_level, rec.signing_authority_level
                    ),
                    // Person nodes dedupe globally; accumulate everything they signed.
                    e.documents_signed = CASE
                        WHEN rec.document_signed IS NULL
                            THEN coalesce(e.documents_signed, [])
                        WHEN rec.document_signed IN coalesce(e.documents_signed, [])
                            THEN e.documents_signed
                        ELSE coalesce(e.documents_signed, []) + rec.document_signed
                    END
                WITH e, rec
                MATCH (d:Document {doc_id: rec.source_doc_id})
                MERGE (e)-[:FOUND_IN]->(d)
                """,
                records=records,
            )
        logger.info("Stored %d entities in Neo4j (deduped by canonical_id)", len(entities))

    _run_with_reconnect(_run)


def store_relationships(relationships: list[Relationship]) -> None:
    """Batch-store relationships as edges in Neo4j."""
    if not relationships:
        return
    settings = get_settings()
    records = [
        {
            "rel_id": r.rel_id,
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "relationship_type": r.relationship_type,
            "source_doc_id": r.source_doc_id,
            "source_page": r.source_page,
            "confidence": r.confidence,
        }
        for r in relationships
    ]

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                UNWIND $records AS rec
                MATCH (src:Entity {entity_id: rec.source_entity_id})
                MATCH (tgt:Entity {entity_id: rec.target_entity_id})
                MERGE (src)-[r:RELATES {rel_id: rec.rel_id}]->(tgt)
                ON CREATE SET
                    r.relationship_type = rec.relationship_type,
                    r.source_doc_id = rec.source_doc_id,
                    r.source_page = rec.source_page,
                    r.confidence = rec.confidence,
                    r.mention_count = 1
                ON MATCH SET
                    r.mention_count = coalesce(r.mention_count, 1) + 1
                """,
                records=records,
            )
        logger.info("Stored %d relationships in Neo4j", len(relationships))

    _run_with_reconnect(_run)


# Allow-list of comparable amount-role pairs with the reason shown in the report.
_COMPARABLE_ROLE_REASONS: dict[frozenset[str], str] = {
    frozenset({"total_contract_value", "total_invoice"}):
        "contract total vs invoice total",
    frozenset({"total_contract_value", "invoice_referenced_contract_value"}):
        "contract total vs invoice's stated contract reference",
    frozenset({"total_contract_value"}):
        "contract total vs contract total across documents",
    frozenset({"total_invoice"}):
        "invoice total vs invoice total across documents",
    frozenset({"invoice_referenced_contract_value"}):
        "invoice's stated contract reference vs another invoice's",
    frozenset({"total_invoice", "invoice_subtotal"}):
        "invoice total vs invoice subtotal (VAT reconciliation)",
    frozenset({"milestone_scheduled"}):
        "scheduled milestone vs scheduled milestone",
    frozenset({"retainer"}):
        "retainer rate vs retainer rate",
    frozenset({"milestone_scheduled", "transaction_debit"}):
        "scheduled milestone vs bank payment",
    frozenset({"retainer", "transaction_debit"}):
        "retainer rate vs bank payment",
    frozenset({"transaction_debit"}):
        "bank payment vs bank payment",
    # invoice_total vs single payment excluded: reconcile against the SUM instead.
}

# Cumulative account state / pre-aggregated totals — never comparable to a line item.
_INCOMPATIBLE_ROLES: frozenset[str] = frozenset({
    "opening_balance",
    "closing_balance",
    "running_balance",
    "statement_total_debits",
    "statement_total_credits",
})


# A recurring rate is comparable without a shared anchor: a retainer billed at a
# different rate than the contract sets is a finding on its own. Milestones are NOT —
# every contract milestone would pair with every bank payment, N x M rows of noise.
_ANCHORLESS_SAME_ROLE_PAIRS: frozenset[str] = frozenset({"retainer"})


def _role_pair_key(role1: str, role2: str) -> str:
    """Order-independent "a|b" key for a role pair, matching _COMPARABLE_PAIR_KEYS."""
    a, b = sorted((role1, role2))
    return f"{a}|{b}"


# Allow-list as sortable "a|b" keys so Cypher applies it before LIMIT.
_COMPARABLE_PAIR_KEYS: list[str] = sorted({
    _role_pair_key(*(tuple(pair) * 2 if len(pair) == 1 else tuple(pair)))
    for pair in _COMPARABLE_ROLE_REASONS
})


def _role_pair_reason(role1: str, role2: str) -> str | None:
    """Return the human-readable reason two roles are comparable, or None if they aren't."""
    r1 = (role1 or "unknown").strip().lower()
    r2 = (role2 or "unknown").strip().lower()
    if r1 in _INCOMPATIBLE_ROLES or r2 in _INCOMPATIBLE_ROLES:
        return None
    # An unclassified amount is not grounds for comparison.
    if r1 == "unknown" or r2 == "unknown":
        return None
    return _COMPARABLE_ROLE_REASONS.get(frozenset({r1, r2}))


def _roles_are_comparable(role1: str, role2: str) -> bool:
    """Return True if two amount_roles should be compared for contradictions."""
    return _role_pair_reason(role1, role2) is not None


def find_contradictions(doc_ids: list[str]) -> list[dict]:
    """Cross-document amount pairs sharing an anchor, role-filtered, by relative difference."""
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[:FOUND_IN]->(d1:Document)
                MATCH (e2:Entity)-[:FOUND_IN]->(d2:Document)
                WHERE e1.entity_type = 'amount'
                  AND e2.entity_type = 'amount'
                  // Symmetric match: keep one direction only.
                  AND e1.entity_id < e2.entity_id
                  AND d1.doc_id <> d2.doc_id
                  AND d1.doc_id IN $doc_ids
                  AND d2.doc_id IN $doc_ids
                  AND NOT coalesce(e1.amount_role, 'unknown') IN $excluded_roles
                  AND NOT coalesce(e2.amount_role, 'unknown') IN $excluded_roles
                  // Role allow-list applied here, before ranking.
                  AND (CASE
                        WHEN coalesce(e1.amount_role, 'unknown') < coalesce(e2.amount_role, 'unknown')
                          THEN coalesce(e1.amount_role, 'unknown') + '|' + coalesce(e2.amount_role, 'unknown')
                          ELSE coalesce(e2.amount_role, 'unknown') + '|' + coalesce(e1.amount_role, 'unknown')
                      END) IN $comparable_pairs
                  AND (
                       e1.amount_currency IS NULL
                    OR e2.amount_currency IS NULL
                    OR e1.amount_currency = e2.amount_currency
                  )
                  AND (
                    (e1.amount_value IS NOT NULL AND e2.amount_value IS NOT NULL
                       AND abs(e1.amount_value - e2.amount_value) >
                           CASE
                             WHEN abs(e1.amount_value) > abs(e2.amount_value) THEN abs(e1.amount_value) * 0.0001 + 0.01
                             ELSE abs(e2.amount_value) * 0.0001 + 0.01
                           END)
                    OR
                    ((e1.amount_value IS NULL OR e2.amount_value IS NULL)
                       AND e1.normalized_value <> e2.normalized_value)
                  )
                // Anchor optional for rate-like roles (see _ANCHORLESS_SAME_ROLE_PAIRS).
                OPTIONAL MATCH (e1)-[:RELATES]-(anchor:Entity)-[:RELATES]-(e2)
                WHERE anchor.entity_type <> 'amount'
                WITH e1, e2, d1, d2, anchor
                WHERE anchor IS NOT NULL
                   OR (
                        coalesce(e1.amount_role, 'unknown') = coalesce(e2.amount_role, 'unknown')
                        AND coalesce(e1.amount_role, 'unknown') IN $anchorless_roles
                      )
                RETURN e1.entity_id AS entity1_id,
                       e2.entity_id AS entity2_id,
                       e1.value AS value1,
                       e2.value AS value2,
                       e1.normalized_value AS norm1,
                       e2.normalized_value AS norm2,
                       e1.amount_value AS amount_value1,
                       e2.amount_value AS amount_value2,
                       e1.amount_currency AS currency1,
                       e2.amount_currency AS currency2,
                       coalesce(e1.amount_role, 'unknown') AS role1,
                       coalesce(e2.amount_role, 'unknown') AS role2,
                       anchor.entity_id AS anchor_entity_id,
                       anchor.value AS anchor_value,
                       anchor.entity_type AS anchor_type,
                       d1.doc_id AS doc1_id,
                       d2.doc_id AS doc2_id,
                       d1.filename AS doc1_name,
                       d2.filename AS doc2_name,
                       e1.source_page AS page1,
                       e2.source_page AS page2
                ORDER BY CASE
                           WHEN e1.amount_value IS NULL OR e2.amount_value IS NULL THEN 0.0
                           WHEN abs(e1.amount_value) + abs(e2.amount_value) = 0 THEN 0.0
                           ELSE abs(e1.amount_value - e2.amount_value) /
                                CASE
                                  WHEN abs(e1.amount_value) > abs(e2.amount_value)
                                    THEN abs(e1.amount_value)
                                  ELSE abs(e2.amount_value)
                                END
                         END DESC
                LIMIT 50
                """,
                doc_ids=doc_ids,
                excluded_roles=sorted(_INCOMPATIBLE_ROLES),
                comparable_pairs=_COMPARABLE_PAIR_KEYS,
                anchorless_roles=sorted(_ANCHORLESS_SAME_ROLE_PAIRS),
            )
            return [dict(r) for r in result]

    rows = _run_with_reconnect(_run)

    # Re-apply allow-list and attach the reason.
    kept: list[dict] = []
    for r in rows:
        reason = _role_pair_reason(r.get("role1"), r.get("role2"))
        if reason is None:
            continue
        r["comparison_reason"] = reason
        kept.append(r)
    if len(kept) < len(rows):
        logger.info(
            "find_contradictions: filtered %d/%d pair(s) with incomparable amount roles",
            len(rows) - len(kept), len(rows),
        )
    return kept


# Self-reference validation: claim role → (authoritative role, doc_type, label).
_REFERENCE_CLAIMS: dict[str, tuple[str, str, str]] = {
    # claim role → (authoritative role, authoritative doc_type, human label)
    "invoice_referenced_contract_value": ("total_contract_value", "contract", "contract value"),
}


def find_reference_mismatches(doc_ids: list[str]) -> list[dict]:
    """Values a document asserts about another vs. that document's own figure,
    via their shared identifier. Deterministic; one row per mismatch."""
    if not doc_ids:
        return []
    settings = get_settings()
    claim_roles = sorted(_REFERENCE_CLAIMS)

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (claim:Entity)-[:FOUND_IN]->(cd:Document)
                WHERE cd.doc_id IN $doc_ids
                  AND claim.entity_type = 'amount'
                  AND claim.amount_role IN $claim_roles
                  AND claim.amount_value IS NOT NULL
                MATCH (claim)-[:RELATES]-(anchor:Entity)
                WHERE anchor.entity_type IN ['contract_id', 'invoice_id']
                MATCH (anchor)-[:RELATES]-(truth:Entity)-[:FOUND_IN]->(td:Document)
                WHERE td.doc_id IN $doc_ids
                  AND td.doc_id <> cd.doc_id
                  AND truth.entity_type = 'amount'
                  AND truth.amount_value IS NOT NULL
                RETURN claim.amount_role AS claim_role,
                       claim.amount_value AS claim_value,
                       claim.value AS claim_raw,
                       claim.amount_currency AS claim_currency,
                       coalesce(claim.source_page, 0) AS claim_page,
                       cd.doc_id AS claim_doc_id,
                       cd.filename AS claim_doc_name,
                       cd.doc_type AS claim_doc_type,
                       anchor.value AS anchor_value,
                       anchor.entity_type AS anchor_type,
                       truth.amount_role AS truth_role,
                       truth.amount_value AS truth_value,
                       truth.value AS truth_raw,
                       truth.amount_currency AS truth_currency,
                       coalesce(truth.source_page, 0) AS truth_page,
                       td.doc_id AS truth_doc_id,
                       td.filename AS truth_doc_name,
                       td.doc_type AS truth_doc_type
                """,
                doc_ids=doc_ids,
                claim_roles=claim_roles,
            )
            return [dict(r) for r in result]

    try:
        rows = _run_with_reconnect(_run)
    except Exception as e:
        logger.warning("Reference mismatch query failed: %s", e)
        return []

    out: list[dict] = []
    seen: set[tuple] = set()
    for r in rows:
        truth_role, truth_doc_type, label = _REFERENCE_CLAIMS[r["claim_role"]]
        # Only the referenced document's own figure is the truth (not echoes elsewhere).
        if r.get("truth_role") != truth_role or r.get("truth_doc_type") != truth_doc_type:
            continue
        cur_c = (r.get("claim_currency") or "").upper()
        cur_t = (r.get("truth_currency") or "").upper()
        if cur_c and cur_t and cur_c != cur_t:
            continue
        cv, tv = float(r["claim_value"]), float(r["truth_value"])
        if abs(cv - tv) <= max(1.0, abs(tv) * 0.0001):
            continue  # restatement matches — nothing to report
        key = (r["claim_doc_id"], round(cv, 2), r["truth_doc_id"], round(tv, 2))
        if key in seen:
            continue
        seen.add(key)
        r["label"] = label
        r["difference"] = round(cv - tv, 2)
        r["relative_difference"] = abs(cv - tv) / max(abs(tv), 1.0)
        out.append(r)

    if out:
        logger.info("Reference validation: %d restated value(s) disagree with source", len(out))
    return out


def audit_identifier_values(doc_ids: list[str]) -> set[str]:
    """Alphanumeric cores of the contract/invoice numbers this audit is about.

    A bank row naming one of them ("سداد نهائي عقد CTR-2024-044") is a payment under
    this audit's contract, whether or not extraction happened to emit a RELATES edge
    for it — which it did for only one of four such payments.
    """
    if not doc_ids:
        return set()
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:FOUND_IN]->(d:Document)
                WHERE d.doc_id IN $doc_ids
                  AND e.entity_type IN ['contract_id', 'invoice_id']
                RETURN collect(DISTINCT e.normalized_value) AS values
                """,
                doc_ids=doc_ids,
            )
            rec = result.single()
            return list(rec["values"]) if rec and rec["values"] else []

    try:
        values = _run_with_reconnect(_run)
    except Exception as e:
        logger.warning("Audit identifier fetch failed: %s", e)
        return set()
    # Short cores would match by accident inside unrelated numbers.
    return {core for core in (normalize_identifier(v or "") for v in values) if len(core) >= 6}


def query_graph_for_entities(query_entities: list[str], doc_ids: list[str], depth: int = 2) -> list[dict]:
    """Traverse graph to find all entities related to given entity values."""
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (start:Entity)
                WHERE start.source_doc_id IN $doc_ids
                  AND (start.value IN $entities OR start.normalized_value IN $entities)
                CALL apoc.path.subgraphNodes(start, {maxLevel: $depth}) YIELD node
                WHERE node:Entity AND node.source_doc_id IN $doc_ids
                RETURN node.value AS value, node.normalized_value AS normalized_value,
                       node.entity_type AS entity_type, node.source_doc_id AS doc_id,
                       node.source_page AS page
                """,
                entities=query_entities,
                doc_ids=doc_ids,
                depth=depth,
            )
            rows = [dict(r) for r in result]
            if not rows:
                result2 = session.run(
                    """
                    MATCH (start:Entity)-[r:RELATES]-(neighbor:Entity)
                    WHERE start.source_doc_id IN $doc_ids
                      AND (start.value IN $entities OR start.normalized_value IN $entities)
                      AND neighbor.source_doc_id IN $doc_ids
                    RETURN neighbor.value AS value,
                           neighbor.normalized_value AS normalized_value,
                           neighbor.entity_type AS entity_type,
                           neighbor.source_doc_id AS doc_id,
                           neighbor.source_page AS page,
                           r.relationship_type AS via_relationship
                    LIMIT 50
                    """,
                    entities=query_entities,
                    doc_ids=doc_ids,
                )
                rows = [dict(r) for r in result2]
            return rows

    return _run_with_reconnect(_run)


def trace_document_linkage(doc_ids: list[str]) -> list[dict]:
    """Per-document entity/edge counts; entities > 0 with 0 cross-doc edges = isolated."""
    if not doc_ids:
        return []
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (d:Document) WHERE d.doc_id IN $doc_ids
                OPTIONAL MATCH (e:Entity)-[:FOUND_IN]->(d)
                WITH d, collect(DISTINCT e) AS ents
                UNWIND (CASE WHEN size(ents) = 0 THEN [null] ELSE ents END) AS e
                OPTIONAL MATCH (e)-[r:RELATES]-(other:Entity)
                OPTIONAL MATCH (other)-[:FOUND_IN]->(d2:Document)
                WITH d,
                     count(DISTINCT e) AS entity_count,
                     count(DISTINCT r) AS edge_count,
                     count(DISTINCT CASE
                         WHEN d2 IS NOT NULL AND d2.doc_id <> d.doc_id THEN r
                     END) AS cross_doc_edges,
                     count(DISTINCT CASE
                         WHEN e IS NOT NULL AND e.entity_type IN
                             ['contract_id', 'invoice_id', 'company', 'person']
                         THEN e
                     END) AS shared_key_entities
                RETURN d.doc_id AS doc_id,
                       d.filename AS filename,
                       entity_count,
                       edge_count,
                       cross_doc_edges,
                       shared_key_entities
                ORDER BY filename
                """,
                doc_ids=doc_ids,
            )
            return [dict(r) for r in result]

    try:
        rows = _run_with_reconnect(_run)
    except Exception as e:
        logger.warning("Document linkage trace failed: %s", e)
        return []

    for r in rows:
        msg = (
            "Linkage trace — %s: %d entities, %d edges (%d cross-doc), "
            "%d globally-shared key entities"
        )
        args = (
            r.get("filename"), r.get("entity_count", 0), r.get("edge_count", 0),
            r.get("cross_doc_edges", 0), r.get("shared_key_entities", 0),
        )
        if r.get("entity_count", 0) > 0 and r.get("cross_doc_edges", 0) == 0:
            logger.warning(
                msg + " — ISOLATED: this document cannot produce cross-document findings",
                *args,
            )
        else:
            logger.info(msg, *args)
    return rows


# Describes a gap ("CFO vs Chairman"); not a signing policy.
_AUTHORITY_RANK: dict[str, int] = {
    "chairman": 5,
    "ceo": 4,
    "cfo": 3,
    "director": 2,
    "manager": 1,
    "other": 0,
    "unknown": 0,
}


# Marks a signed instrument as an amendment/successor.
_AMENDMENT_KEYWORDS: tuple[str, ...] = (
    "amendment", "amend", "amended", "addendum", "variation order", "variation",
    "supplement", "supplementary", "successor", "side letter", "change order",
    "revised agreement", "novation",
    "ملحق", "تعديل", "معدل", "اتفاقية معدلة",
)

# Document/contract reference tokens inside a `documents_signed` string, e.g.
# "amendment 2 to BLD-2024-019" → {"bld2024019"}.
_SIGNED_REF_RE = re.compile(r"\b[A-Za-z]{2,6}[-\s/]?\d{2,4}(?:[-\s/]?\d{1,6})+\b")


def _signed_ref_tokens(text: str) -> set[str]:
    return {
        normalize_identifier(m.group(0))
        for m in _SIGNED_REF_RE.finditer(text or "")
        if normalize_identifier(m.group(0))
    }


def _is_amendment_of(candidate: str, base: str) -> bool:
    """True when `candidate` names itself an amendment of `base` AND shares its reference."""
    c = (candidate or "").strip().lower()
    b = (base or "").strip().lower()
    if not c or not b or c == b:
        return False
    if not any(kw in c for kw in _AMENDMENT_KEYWORDS):
        return False
    c_refs = _signed_ref_tokens(candidate)
    b_refs = _signed_ref_tokens(base)
    if b_refs and (c_refs & b_refs):
        return True
    # No ref on the base: fall back to its name being quoted in the amendment.
    if not b_refs and len(b) >= 6 and b in c:
        return True
    return False


def find_signatory_mismatches(doc_ids: list[str]) -> list[dict]:
    """Amendment/successor signed BELOW the original signer's authority, with an
    explicit shared reference. A normal multi-role approval chain never matches."""
    if not doc_ids:
        return []
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (p:Entity)-[:FOUND_IN]->(d:Document)
                WHERE p.entity_type = 'person'
                  AND d.doc_id IN $doc_ids
                  AND p.signing_authority_level IS NOT NULL
                  AND size(coalesce(p.documents_signed, [])) > 0
                RETURN p.normalized_value AS name,
                       p.role_title AS title,
                       p.signing_authority_level AS authority,
                       coalesce(p.documents_signed, []) AS signed,
                       d.doc_id AS doc_id,
                       d.filename AS doc_name,
                       coalesce(p.source_page, 0) AS page
                LIMIT 200
                """,
                doc_ids=doc_ids,
            )
            return [dict(r) for r in result]

    try:
        people = _run_with_reconnect(_run)
    except Exception as e:
        logger.warning("Signatory mismatch query failed: %s", e)
        return []

    seen: set[frozenset] = set()
    out: list[dict] = []
    for base in people:
        base_name = str(base.get("name") or "").strip()
        base_rank = _AUTHORITY_RANK.get(str(base.get("authority") or "unknown").lower(), 0)
        base_signed = [str(s) for s in (base.get("signed") or []) if s]
        for amend in people:
            amend_name = str(amend.get("name") or "").strip()
            if not base_name or not amend_name or base_name.lower() == amend_name.lower():
                continue
            amend_rank = _AUTHORITY_RANK.get(str(amend.get("authority") or "unknown").lower(), 0)
            # Only when the amendment is signed below the original's level.
            if amend_rank >= base_rank:
                continue
            amend_signed = [str(s) for s in (amend.get("signed") or []) if s]
            link = next(
                (
                    (bs, cs)
                    for bs in base_signed
                    for cs in amend_signed
                    if _is_amendment_of(cs, bs)
                ),
                None,
            )
            if link is None:
                continue
            pair_key = frozenset({base_name.lower(), amend_name.lower()})
            if pair_key in seen:
                continue
            seen.add(pair_key)
            base_instrument, amend_instrument = link
            out.append({
                "name1": base_name,
                "title1": base.get("title") or base.get("authority"),
                "authority1": base.get("authority"),
                "signed1": [base_instrument],
                "doc1_id": base.get("doc_id"),
                "doc1_name": base.get("doc_name"),
                "page1": base.get("page", 0),
                "name2": amend_name,
                "title2": amend.get("title") or amend.get("authority"),
                "authority2": amend.get("authority"),
                "signed2": [amend_instrument],
                "doc2_id": amend.get("doc_id"),
                "doc2_name": amend.get("doc_name"),
                "page2": amend.get("page", 0),
                "authority_gap": abs(base_rank - amend_rank),
            })

    if out:
        logger.info("Signatory mismatch: %d amendment/authority pair(s) found", len(out))
    return out
