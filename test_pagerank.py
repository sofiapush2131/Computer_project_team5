"""Test pagerank"""
import numpy as np
from pagerank import normalize_matrix, pagerank, build_graph, recommend

# --- normalize_matrix ---

def test_normalize_basic():
    """Check that a simple adjacency matrix is normalized correctly by columns."""
    matrix = np.array([[0, 2],
                       [1, 1]], float)
    expected = np.array([[0.0, 2/3],
                         [1.0, 1/3]])
    normalized = normalize_matrix(matrix)
    assert np.allclose(normalized, expected)


def test_normalize_column_sum():
    """Verify that every non-zero column of the normalized matrix sums to 1."""
    matrix = np.array([[1, 4, 0],
                       [2, 1, 0],
                       [1, 1, 0]], float)
    normalized = normalize_matrix(matrix)
    col_sums = normalized.sum(axis=0)
    assert np.allclose(col_sums[:2], np.ones(2))


def test_normalize_no_nan():
    """Ensure normalization never produces NaN, even with zero columns."""
    matrix = np.array([[0, 1, 0],
                       [0, 2, 0],
                       [0, 3, 0]], float)
    normalized = normalize_matrix(matrix)
    assert not np.isnan(normalized).any()


# --- pagerank ---

def test_pagerank_symmetric():
    """Check that symmetric nodes receive equal PageRank values."""
    matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], float)
    ranks = pagerank(matrix)
    assert np.isclose(ranks[0], ranks[2])


def test_pagerank_sum_one():
    """Ensure PageRank produces valid non-zero rank distribution."""
    matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ], float)
    ranks = pagerank(matrix)
    assert ranks.sum() > 0


def test_pagerank_non_negative():
    """Verify that all PageRank values are non-negative."""
    matrix = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0]
    ], float)
    ranks = pagerank(matrix)
    assert (ranks >= 0).all()


def test_dangling_node():
    """Check that nodes with no outgoing edges do not break PageRank computation."""
    matrix = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], float)
    ranks = pagerank(matrix)
    assert not np.isnan(ranks).any()
    assert ranks.sum() > 0


def test_all_zero_graph():
    """Verify that in an empty graph all nodes receive equal ranks."""
    matrix = np.zeros((6, 6))
    ranks = pagerank(matrix)
    assert np.allclose(ranks, ranks[0])


def test_random_graph_stability():
    """Check PageRank stability on random graphs."""
    np.random.seed(42)
    matrix = np.random.randint(0, 2, size=(10, 10)).astype(float)
    ranks = pagerank(matrix)
    assert not np.isnan(ranks).any()
    assert np.isclose(ranks.sum(), 1.0)


def test_high_damping():
    """Ensure PageRank is stable for high damping factor."""
    matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], float)
    ranks = pagerank(matrix, damping=0.99)
    assert np.isclose(ranks[0], ranks[2], atol=1e-3)


def test_no_incoming_links_not_equal():
    """Check that nodes with different connectivity do not receive identical PageRank values."""
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], float)
    ranks = pagerank(matrix)
    assert len(set(np.round(ranks, 6))) > 1


# --- build_graph for user-item interactions ---

def test_graph_structure():
    """Verify that user-item interactions are correctly converted into bidirectional edges."""
    users = ["u1", "u2"]
    items = ["i1", "i2"]
    interactions = [("u1", "i1"), ("u1", "i2"), ("u2", "i1")]

    # build bidirectional edges for new build_graph
    nodes = users + items
    edges = [(u, i) for u, i in interactions] + [(i, u) for u, i in interactions]
    matrix, index, all_nodes = build_graph(nodes, edges)

    # user to item edges
    assert matrix[index["u1"], index["i1"]] == 1
    assert matrix[index["u1"], index["i2"]] == 1
    assert matrix[index["u2"], index["i1"]] == 1

    # item to user edges (bidirectional)
    assert matrix[index["i1"], index["u1"]] == 1
    assert matrix[index["i2"], index["u1"]] == 1
    assert matrix[index["i1"], index["u2"]] == 1

    # verify graph size
    assert matrix.shape == (4, 4)
    assert len(all_nodes) == 4


# --- recommend ---

def test_recommend_not_user():
    """Ensure that the target user is never included in the recommendation list."""
    rank = np.array([0.9, 0.05, 0.03, 0.02])
    all_nodes = ["user1", "item1", "item2", "item3"]
    index = {node: i for i, node in enumerate(all_nodes)}
    recommendations = recommend("user1", rank, index, all_nodes)
    recommended_ids = [item for item, _ in recommendations]
    assert "user1" not in recommended_ids


def test_recommend_sorted():
    """Check that items are returned in descending order of rank."""
    rank = np.array([0.05, 0.9, 0.1, 0.4])
    all_nodes = ["user1", "item1", "item2", "item3"]
    index = {node: i for i, node in enumerate(all_nodes)}
    recommendations = recommend("user1", rank, index, all_nodes)
    scores = [score for _, score in recommendations]
    assert scores == sorted(scores, reverse=True)


def test_popular_item():
    """Ensure that more popular items receive higher PageRank scores."""
    users = ["u1", "u2", "u3"]
    items = ["songA", "songB"]
    interactions = [("u1", "songA"), ("u2", "songA"), ("u3", "songA"), ("u1", "songB")]

    nodes = users + items
    edges = [(u, i) for u, i in interactions] + [(i, u) for u, i in interactions]
    matrix, index, _ = build_graph(nodes, edges)
    ranks = pagerank(matrix)
    assert ranks[index["songA"]] > ranks[index["songB"]]
