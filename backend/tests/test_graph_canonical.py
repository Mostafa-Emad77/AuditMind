from app.services.graph_builder import _apply_center_depth, _canonicalize_graph, _pick_center_node


def test_canonical_grouping_contract_ids():
    raw_nodes = [
        {"id": "doc-a", "label": "A.pdf", "node_type": "document", "doc_id": "doc-a", "properties": {}},
        {"id": "doc-b", "label": "B.pdf", "node_type": "document", "doc_id": "doc-b", "properties": {}},
        {
            "id": "e1",
            "label": "CTR-2024-044",
            "node_type": "contract_id",
            "doc_id": "doc-a",
            "properties": {"normalized_value": "ctr-2024-044", "source_page": 1},
        },
        {
            "id": "e2",
            "label": "CTR-2024-044",
            "node_type": "contract_id",
            "doc_id": "doc-b",
            "properties": {"normalized_value": "ctr-2024-044", "source_page": 2},
        },
    ]
    raw_edges = [
        {"source": "e1", "target": "doc-a", "relationship": "found_in"},
        {"source": "e2", "target": "doc-b", "relationship": "found_in"},
    ]

    graph = _canonicalize_graph(raw_nodes, raw_edges)
    contract_nodes = [n for n in graph.nodes if n.node_type == "contract_id"]
    assert len(contract_nodes) == 1
    assert contract_nodes[0].mention_count == 2
    assert set(contract_nodes[0].properties["source_docs"]) == {"doc-a", "doc-b"}


def test_edge_aggregation_counts():
    raw_nodes = [
        {"id": "doc-a", "label": "A.pdf", "node_type": "document", "doc_id": "doc-a", "properties": {}},
        {
            "id": "e1",
            "label": "CTR-1",
            "node_type": "contract_id",
            "doc_id": "doc-a",
            "properties": {"normalized_value": "ctr-1", "source_page": 1},
        },
        {
            "id": "e2",
            "label": "ACME",
            "node_type": "company",
            "doc_id": "doc-a",
            "properties": {"normalized_value": "acme", "source_page": 1},
        },
    ]
    raw_edges = [
        {"source": "e1", "target": "e2", "relationship": "party_to"},
        {"source": "e1", "target": "e2", "relationship": "party_to"},
    ]

    graph = _canonicalize_graph(raw_nodes, raw_edges)
    rel_edges = [e for e in graph.edges if e.relationship == "party_to"]
    assert len(rel_edges) == 1
    assert rel_edges[0].count == 2


def test_center_pick_and_depth_filter():
    raw_nodes = [
        {"id": "doc-a", "label": "A.pdf", "node_type": "document", "doc_id": "doc-a", "properties": {}},
        {"id": "doc-b", "label": "B.pdf", "node_type": "document", "doc_id": "doc-b", "properties": {}},
        {"id": "e1", "label": "CTR-1", "node_type": "contract_id", "doc_id": "doc-a", "properties": {"normalized_value": "ctr-1", "source_page": 1}},
        {"id": "e2", "label": "ACME", "node_type": "company", "doc_id": "doc-a", "properties": {"normalized_value": "acme", "source_page": 1}},
        {"id": "e3", "label": "INV-1", "node_type": "invoice_id", "doc_id": "doc-b", "properties": {"normalized_value": "inv-1", "source_page": 2}},
    ]
    raw_edges = [
        {"source": "e1", "target": "e2", "relationship": "party_to"},
        {"source": "e1", "target": "e3", "relationship": "references"},
        {"source": "e2", "target": "doc-a", "relationship": "found_in"},
        {"source": "e3", "target": "doc-b", "relationship": "found_in"},
    ]
    graph = _canonicalize_graph(raw_nodes, raw_edges)
    center = _pick_center_node(graph.nodes, graph.edges, requested=None)
    assert center is not None

    filtered = _apply_center_depth(graph, center, depth=1)
    assert filtered.center_node == center
    assert all(n.id == center or any(e.source == n.id or e.target == n.id for e in filtered.edges) for n in filtered.nodes)
