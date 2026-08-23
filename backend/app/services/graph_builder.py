"""Build and query the Neo4j knowledge graph from extracted entities and relationships."""
import logging
from collections import defaultdict, deque
from typing import Any, Optional

from neo4j import GraphDatabase, Driver

from app.config import get_settings
from app.models.schemas import Entity, Relationship, DocumentMeta, GraphData, GraphNode, GraphEdge

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
    """
    Remove Document nodes for an audit and prune entities that no longer belong
    to any remaining document.

    Only the FOUND_IN edges for the given docs are deleted (not the entities
    themselves), because global-dedupe entity types (company, contract_id, ...)
    can be shared across multiple audits' documents via canonical IDs. An entity
    is only DETACH DELETEd once it has no remaining FOUND_IN edge to any document.
    """
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
                    e.role_title = coalesce(e.role_title, rec.role_title),
                    e.signing_authority_level = coalesce(
                        e.signing_authority_level, rec.signing_authority_level
                    ),
                    // Person nodes dedupe globally, so one signatory seen across several
                    // documents accumulates everything they signed — that list is what
                    // makes signatory/authority mismatches detectable.
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


# Explicit allow-list of amount-role pairs that are meaningful to compare, each with a
# human-readable reason explaining WHY the two are comparable. The reason is surfaced as
# the conflict_type on exported rows, so a reviewer sees "contract total vs invoice's
# stated contract value" instead of an opaque "Doc A vs Doc B".
#
# Pairs are declared once, unordered; the symmetric lookup is built below. Anything not
# listed here is NOT compared — co-occurring in a chunk or sharing a document ID is
# explicitly not sufficient grounds to compare two amounts.
_COMPARABLE_ROLE_REASONS: dict[frozenset[str], str] = {
    frozenset({"total_contract_value", "total_invoice"}):
        "contract total vs invoice total",
    # The restatement check: an invoice asserting a different contract value than the
    # contract itself states is a real finding, which is why these carry distinct roles.
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
    # Legacy `single_payment` (pre debit/credit split) kept comparable so historical
    # graph data keeps producing findings.
    frozenset({"single_payment"}):
        "bank payment vs bank payment",
    frozenset({"milestone_scheduled", "single_payment"}):
        "scheduled milestone vs bank payment",
    frozenset({"retainer", "single_payment"}):
        "retainer rate vs bank payment",
    # NOTE: invoice_total vs an individual payment is deliberately NOT here. An invoice
    # is legitimately settled by several installments, so it must be reconciled against
    # the SUM of payments (see reconciliation_payload), never pairwise against one debit.
}

# Roles that must never participate in a contradiction comparison: cumulative account
# state and pre-aggregated statement totals are not transaction values, so comparing
# them against line items is meaningless by construction.
_INCOMPATIBLE_ROLES: frozenset[str] = frozenset({
    "opening_balance",
    "closing_balance",
    "running_balance",
    "statement_total_debits",
    "statement_total_credits",
})


def _role_pair_key(role1: str, role2: str) -> str:
    """Order-independent "a|b" key for a role pair, matching _COMPARABLE_PAIR_KEYS."""
    a, b = sorted((role1, role2))
    return f"{a}|{b}"


# The allow-list flattened into sortable string keys so the SAME rule can be applied
# inside Cypher. This matters because LIMIT runs in the database: filtering roles only
# in Python let high-materiality but incomparable pairs consume the row budget and
# push genuine findings out of the result set entirely.
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
    # `unknown` used to pass through as comparable, which — combined with untagged
    # running balances — is what produced conflicts like "running balance 750,000 vs
    # line item 25,000". An unclassified amount is no longer grounds for comparison.
    if r1 == "unknown" or r2 == "unknown":
        return None
    return _COMPARABLE_ROLE_REASONS.get(frozenset({r1, r2}))


def _roles_are_comparable(role1: str, role2: str) -> bool:
    """Return True if two amount_roles should be compared for contradictions."""
    return _role_pair_reason(role1, role2) is not None


def find_contradictions(doc_ids: list[str]) -> list[dict]:
    """
    Find amount contradictions across documents.

    Only pairs of amount entities that share a non-amount anchor (contract, invoice,
    party, etc.) via RELATES are considered — not arbitrary shortest-path neighbors.

    Each unordered pair is returned once (the match is symmetric, so without the
    entity_id ordering constraint every pair came back twice), ordered by relative
    amount difference so the callers' top-N are the most material discrepancies.

    Results include amount_role; incomparable role pairs are filtered out with
    `_roles_are_comparable` before returning.
    Pairs where one entity is an opening/closing balance are excluded at query time.
    """
    settings = get_settings()

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[:FOUND_IN]->(d1:Document)
                MATCH (e2:Entity)-[:FOUND_IN]->(d2:Document)
                WHERE e1.entity_type = 'amount'
                  AND e2.entity_type = 'amount'
                  // Unordered pairs: the match is symmetric, so keep one direction only.
                  AND e1.entity_id < e2.entity_id
                  AND d1.doc_id <> d2.doc_id
                  AND d1.doc_id IN $doc_ids
                  AND d2.doc_id IN $doc_ids
                  // Cumulative balances and pre-aggregated statement totals are not
                  // transaction values — excluded here so they never reach the LLM.
                  AND NOT coalesce(e1.amount_role, 'unknown') IN $excluded_roles
                  AND NOT coalesce(e2.amount_role, 'unknown') IN $excluded_roles
                  // Apply the role allow-list HERE, before ORDER BY/LIMIT. Filtering it
                  // only in Python let incomparable pairs occupy the LIMIT budget and
                  // silently drop real findings that ranked below them.
                  AND (CASE
                        WHEN coalesce(e1.amount_role, 'unknown') < coalesce(e2.amount_role, 'unknown')
                          THEN coalesce(e1.amount_role, 'unknown') + '|' + coalesce(e2.amount_role, 'unknown')
                          ELSE coalesce(e2.amount_role, 'unknown') + '|' + coalesce(e1.amount_role, 'unknown')
                      END) IN $comparable_pairs
                  // Currency must match when both are known (USD vs EGP is not a contradiction).
                  AND (
                       e1.amount_currency IS NULL
                    OR e2.amount_currency IS NULL
                    OR e1.amount_currency = e2.amount_currency
                  )
                  // Distinct amounts: prefer numeric compare with tolerance; fall back to string.
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
                MATCH (e1)-[:RELATES]-(anchor:Entity)-[:RELATES]-(e2)
                WHERE anchor.entity_type <> 'amount'
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
            )
            return [dict(r) for r in result]

    rows = _run_with_reconnect(_run)

    # Apply the full role-compatibility allow-list. The Cypher only drops the
    # never-comparable roles; this rejects pairs like retainer vs total_contract_value
    # that would otherwise reach the LLM and burn an adjudication call each. The
    # matched reason rides along so the report can say why the pair was compared.
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


_GLOBAL_DEDUPE_TYPES = {"contract_id", "invoice_id", "company", "person"}


def _canonical_entity_key(row: dict[str, Any]) -> str:
    e_type = (row.get("entity_type") or row.get("node_type") or "other").strip().lower()
    norm = (
        row.get("normalized_value")
        or row.get("properties", {}).get("normalized_value")
        or row.get("value")
        or row.get("label")
        or row.get("entity_id")
        or row.get("id")
        or ""
    ).strip().lower()
    if e_type in _GLOBAL_DEDUPE_TYPES:
        return f"{e_type}:{norm}"
    return f"{e_type}:{row.get('doc_id', '')}:{norm}"


def _pick_center_node(nodes: list[GraphNode], edges: list[GraphEdge], requested: Optional[str]) -> Optional[str]:
    by_id = {n.id: n for n in nodes}
    if requested and requested in by_id:
        return requested
    if not nodes:
        return None

    def _degree(n_id: str) -> int:
        return sum(1 for e in edges if e.source == n_id or e.target == n_id)

    for n in nodes:
        n.degree = _degree(n.id)

    typed = [n for n in nodes if n.node_type in {"contract_id", "invoice_id"}]
    if typed:
        return max(typed, key=lambda n: n.degree).id

    docs = [n for n in nodes if n.node_type == "document"]
    if docs:
        return max(docs, key=lambda n: n.degree).id

    return max(nodes, key=lambda n: n.degree).id


def _apply_center_depth(graph: GraphData, center_node: Optional[str], depth: int) -> GraphData:
    if not center_node or depth < 1:
        return graph

    node_ids = {n.id for n in graph.nodes}
    if center_node not in node_ids:
        return graph

    adj: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)

    keep: set[str] = {center_node}
    q: deque[tuple[str, int]] = deque([(center_node, 0)])
    while q:
        nid, d = q.popleft()
        if d >= depth:
            continue
        for nxt in adj.get(nid, set()):
            if nxt not in keep:
                keep.add(nxt)
                q.append((nxt, d + 1))

    nodes = [n for n in graph.nodes if n.id in keep]
    edges = [e for e in graph.edges if e.source in keep and e.target in keep]
    for n in nodes:
        n.is_center = n.id == center_node
    return GraphData(nodes=nodes, edges=edges, center_node=center_node, view=graph.view)


def _canonicalize_graph(raw_nodes: list[dict[str, Any]], raw_edges: list[dict[str, Any]]) -> GraphData:
    documents: list[GraphNode] = []
    entity_rows = [n for n in raw_nodes if n.get("node_type") != "document"]

    for d in [n for n in raw_nodes if n.get("node_type") == "document"]:
        documents.append(GraphNode(
            id=d["id"],
            label=d.get("label", d["id"]),
            node_type="document",
            doc_id=d["id"],
            raw_ids=[d["id"]],
            properties=d.get("properties", {}),
        ))

    grouped: dict[str, dict[str, Any]] = {}
    raw_to_canon: dict[str, str] = {}

    for row in entity_rows:
        key = _canonical_entity_key(row)
        canon_id = f"ent::{key}"
        raw_id = row["id"]
        raw_to_canon[raw_id] = canon_id

        g = grouped.setdefault(canon_id, {
            "id": canon_id,
            "label": row.get("label") or row.get("properties", {}).get("normalized_value", raw_id),
            "node_type": row.get("node_type", "other"),
            "doc_ids": set(),
            "pages": set(),
            "normalized_values": set(),
            "raw_ids": [],
            "mention_count": 0,
        })
        g["mention_count"] += 1
        g["raw_ids"].append(raw_id)
        if row.get("doc_id"):
            g["doc_ids"].add(row["doc_id"])
        page = row.get("properties", {}).get("source_page")
        if isinstance(page, int) and page > 0:
            g["pages"].add(page)
        nval = row.get("properties", {}).get("normalized_value")
        if nval:
            g["normalized_values"].add(str(nval))
        # Prefer shortest non-empty label to avoid noisy long OCR values.
        curr = row.get("label", "")
        if curr and len(curr) < len(g["label"]):
            g["label"] = curr

    nodes: list[GraphNode] = documents[:]
    for g in grouped.values():
        doc_ids = sorted(g["doc_ids"])
        node_doc_id = doc_ids[0] if len(doc_ids) == 1 else None
        nodes.append(GraphNode(
            id=g["id"],
            label=g["label"],
            node_type=g["node_type"],
            doc_id=node_doc_id,
            mention_count=g["mention_count"],
            raw_ids=g["raw_ids"],
            properties={
                "normalized_values": sorted(g["normalized_values"]),
                "source_docs": doc_ids,
                "pages": sorted(g["pages"]),
            },
        ))

    agg: dict[tuple[str, str, str], GraphEdge] = {}

    for e in raw_edges:
        src = e.get("source")
        tgt = e.get("target")
        rel = e.get("relationship", "other")
        src_id = raw_to_canon.get(src, src)
        tgt_id = raw_to_canon.get(tgt, tgt)
        if src_id == tgt_id:
            continue
        key = (src_id, tgt_id, rel)
        if key not in agg:
            agg[key] = GraphEdge(
                source=src_id,
                target=tgt_id,
                relationship=rel,
                count=1,
                is_contradiction=bool(e.get("is_contradiction", False)),
                properties=e.get("properties", {}),
            )
        else:
            agg[key].count += 1
            agg[key].is_contradiction = agg[key].is_contradiction or bool(e.get("is_contradiction", False))

    # Build canonical FOUND_IN edges from grouped metadata (stable and deduped).
    for g in grouped.values():
        for d_id in g["doc_ids"]:
            key = (g["id"], d_id, "found_in")
            if key not in agg:
                agg[key] = GraphEdge(source=g["id"], target=d_id, relationship="found_in", count=1)
            else:
                agg[key].count += 1

    return GraphData(nodes=nodes, edges=list(agg.values()), view="canonical")


def get_entity_graph(
    doc_ids: list[str],
    center_node: Optional[str] = None,
    depth: int = 1,
    view: str = "canonical",
) -> GraphData:
    """Retrieve nodes + edges for visualization, optionally canonicalized and center-filtered."""
    settings = get_settings()
    raw_nodes: list[dict[str, Any]] = []
    raw_edges: list[dict[str, Any]] = []

    def _run(driver: Driver):
        with driver.session(database=settings.neo4j_database) as session:
            for record in session.run(
                "MATCH (d:Document) WHERE d.doc_id IN $doc_ids RETURN d",
                doc_ids=doc_ids,
            ):
                d = record["d"]
                raw_nodes.append({
                    "id": d["doc_id"],
                    "label": d.get("filename", d["doc_id"]),
                    "node_type": "document",
                    "doc_id": d["doc_id"],
                    "properties": {"doc_type": d.get("doc_type", ""), "language": d.get("language", "")},
                })

            for record in session.run(
                """
                MATCH (e:Entity)-[:FOUND_IN]->(d:Document)
                WHERE d.doc_id IN $doc_ids
                RETURN e.entity_id AS entity_id,
                       e.value AS value,
                       e.entity_type AS entity_type,
                       e.normalized_value AS normalized_value,
                       e.source_page AS source_page,
                       d.doc_id AS doc_id
                LIMIT 500
                """,
                doc_ids=doc_ids,
            ):
                row = dict(record)
                raw_nodes.append({
                    "id": row["entity_id"],
                    "label": row.get("value", ""),
                    "node_type": row.get("entity_type", "other"),
                    "doc_id": row.get("doc_id"),
                    "properties": {
                        "normalized_value": row.get("normalized_value", ""),
                        "source_page": row.get("source_page", 0),
                    },
                })

            for record in session.run(
                """
                MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
                WHERE e1.source_doc_id IN $doc_ids AND e2.source_doc_id IN $doc_ids
                RETURN e1.entity_id AS src, e2.entity_id AS tgt,
                       r.relationship_type AS rel_type
                LIMIT 800
                """,
                doc_ids=doc_ids,
            ):
                raw_edges.append({
                    "source": record["src"],
                    "target": record["tgt"],
                    "relationship": record["rel_type"],
                })

            for record in session.run(
                """
                MATCH (e:Entity)-[:FOUND_IN]->(d:Document)
                WHERE d.doc_id IN $doc_ids
                RETURN e.entity_id AS src, d.doc_id AS tgt
                LIMIT 500
                """,
                doc_ids=doc_ids,
            ):
                raw_edges.append({
                    "source": record["src"],
                    "target": record["tgt"],
                    "relationship": "found_in",
                })

    _run_with_reconnect(_run)

    if view == "raw":
        graph = GraphData(
            nodes=[
                GraphNode(
                    id=n["id"],
                    label=n["label"],
                    node_type=n["node_type"],
                    doc_id=n.get("doc_id"),
                    raw_ids=[n["id"]],
                    properties=n.get("properties", {}),
                )
                for n in raw_nodes
            ],
            edges=[
                GraphEdge(
                    source=e["source"],
                    target=e["target"],
                    relationship=e.get("relationship", "other"),
                    is_contradiction=bool(e.get("is_contradiction", False)),
                    properties=e.get("properties", {}),
                )
                for e in raw_edges
            ],
            view="raw",
        )
    else:
        graph = _canonicalize_graph(raw_nodes, raw_edges)

    resolved_center = _pick_center_node(graph.nodes, graph.edges, center_node)
    for n in graph.nodes:
        n.is_center = n.id == resolved_center
    graph.center_node = resolved_center
    return _apply_center_depth(graph, resolved_center, max(1, depth))


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
    """
    Per-document diagnostic: how many entities each document contributed, and how many
    graph edges actually touch them.

    A document showing entities > 0 but cross_doc_edges == 0 is the signature of an
    entity-resolution failure: its entities exist but never linked to the shared nodes
    (same contract number, same vendor) that the cross-checker traverses — so it can
    never contribute a cross-document finding.
    """
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
