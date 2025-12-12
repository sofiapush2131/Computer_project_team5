"""
Module: Universal PageRank algorithm
Provides functions to build any directed graph, compute PageRank scores,
and recommend top nodes based on PageRank.
Includes doctests for automatic testing.
"""

import numpy as np

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize adjacency matrix columns to sum to 1.

    >>> import numpy as np
    >>> M = np.array([[0, 1],
    ...               [1, 0]])
    >>> normalize_matrix(M)
    array([[0., 1.],
           [1., 0.]])
    >>> M = np.array([
    ...     [0, 2, 0],
    ...     [3, 3, 0],
    ...     [0, 1, 0]
    ... ], float)
    >>> normalize_matrix(M)
    array([[0.        , 0.33333333, 0.        ],
           [1.        , 0.5       , 0.        ],
           [0.        , 0.16666667, 0.        ]])
    """
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return matrix / col_sums


def pagerank(matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for any directed graph using power iteration.

    >>> import numpy as np
    >>> M = np.array([[0, 1],
    ...               [1, 0]], float)
    >>> pagerank(M)
    array([0.5, 0.5])
    """
    n = matrix.shape[0]
    matrix_norm = normalize_matrix(matrix)
    rank = np.ones(n) / n
    teleport = np.ones(n) / n

    for _ in range(max_iter):
        new_rank = damping * matrix_norm.dot(rank) + (1 - damping) * teleport
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank

    return rank


def build_graph(nodes: list, edges: list) -> tuple:
    """
    Build adjacency matrix for any directed graph.

    >>> nodes_list = ["A", "B"]
    >>> edges_list = [("A", "B")]
    >>> build_graph(nodes_list, edges_list)
    (array([[0., 0.],
           [1., 0.]]), {'A': 0, 'B': 1}, ['A', 'B'])
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((n, n))

    for src, tgt in edges:
        matrix[index[tgt], index[src]] = 1

    return matrix, index, nodes


def recommend(user_id: str, rank: np.ndarray, index: dict, candidates: list, top_k: int = 5) -> list:
    """
    Recommend top-k nodes based on PageRank scores from the candidate list.

    >>> rank = np.array([0.1, 0.4, 0.3, 0.2])
    >>> all_nodes = ["user1", "user2", "itemA", "itemB"]
    >>> index = {"user1": 0, "user2": 1, "itemA": 2, "itemB": 3}
    >>> recommend("user1", rank, index, ["itemA", "itemB"], top_k=2)
    [('itemA', 0.3), ('itemB', 0.2)]
    """
    scores = [(node, float(rank[index[node]])) for node in candidates if node != user_id]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_interaction_graph(items: list, users: list, interactions: list) -> tuple:
    """
    Build bipartite graph from user-item interactions.

    >>> items = ["i1", "i2"]
    >>> users = ["u1", "u2"]
    >>> interactions = [("u1", "i1"), ("u2", "i2")]
    >>> build_interaction_graph(items, users, interactions)
    (array([[0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.]]), {'u1': 0, 'u2': 1, 'i1': 2, 'i2': 3}, ['u1', 'u2', 'i1', 'i2'])
    """
    all_nodes = users + items
    n = len(all_nodes)
    index = {node: i for i, node in enumerate(all_nodes)}
    matrix = np.zeros((n, n))

    for user, item in interactions:
        ui = index[user]
        ii = index[item]
        matrix[ii][ui] = 1
        matrix[ui][ii] = 1

    return matrix, index, all_nodes


# Example usage
if __name__ == "__main__":
    users = ["user1", "user2", "user3"]
    items = ["songA", "songB", "songC", "songD"]
    interactions = [("user1", "songA"), ("user1", "songB"), ("user2", "songA"), ("user3", "songC")]

    matrix, index, all_nodes = build_interaction_graph(items, users, interactions)
    ranks = pagerank(matrix)
    recs = recommend("user1", ranks, index, all_nodes, top_k=3)

    print("Recommendations for user1:")
    for item, score in recs:
        print(f"{item}: {score:.4f}")

    import doctest
    doctest.testmod()
