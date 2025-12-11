"""Module PageRank algorithm"""
import numpy as np


def normalize_matrix(matrix):
    """
    Normalize the adjacency matrix so that each column sums to 1.
    This is necessary for the PageRank algorithm to work correctly.

    Parameters
    ----------
    matrix : numpy.ndarray
        Raw adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Column-normalized matrix.
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
    col_sums[col_sums == 0] = 1  # Prevent division by zero for isolated nodes
    return matrix / col_sums


def pagerank(matrix, d=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank scores for all nodes in the graph.

    Parameters
    ----------
    matrix : numpy.ndarray
        Normalized adjacency matrix.
    d : float, optional
        Damping factor (default is 0.85).
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Convergence tolerance (default is 1e-6).

    Returns
    -------
    numpy.ndarray
        Vector of PageRank scores for each node.
    >>> import numpy as np
    >>> M = np.array([
    ...     [0, 1],
    ...     [1, 0]
    ... ], float)
    >>> pagerank(M)
    array([0.5, 0.5])
    """
    n = matrix.shape[0]
    matrix_norm = normalize_matrix(matrix)

    rank = np.ones(n) / n  # Initial PageRank values
    teleport = np.ones(n) / n  # Teleportation vector

    for _ in range(max_iter):
        new_rank = d * matrix_norm.dot(rank) + (1 - d) * teleport

        # Check convergence
        if np.linalg.norm(new_rank - rank) < tol:
            break

        rank = new_rank

    return rank


def build_interaction_graph(items, users, interactions):
    """
    Construct a graph representation (adjacency matrix) from user-item interactions.

    The graph is bipartite:
        user <-> item
    Meaning that edges exist between a user and an item if the user interacted with it.

    Parameters
    ----------
    items : list
        List of item IDs (products, songs, movies, etc.).
    users : list
        List of user IDs.
    interactions : list of tuples
        Each tuple is (user_id, item_id) meaning that user interacted with this item.

    Returns
    -------
    marix : numpy.ndarray
        Adjacency matrix of the graph.
    index : dict
        Mapping from node ID to numeric index.
    all_nodes : list
        List of all nodes (users + items) in index order.
    >>> import numpy as np
    >>> items = ["i1", "i2"]
    >>> users = ["u1", "u2"]
    >>> interactions = [
    ...     ("u1", "i1"),
    ...     ("u2", "i2")
    ... ]
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

    # Create bi-directional edges between user and item
    for user, item in interactions:
        ui = index[user]
        ii = index[item]

        matrix[ii][ui] = 1  # user → item
        matrix[ui][ii] = 1  # item → user

    return matrix, index, all_nodes


def recommend(user_id, rank, index, all_nodes, top_k=5):
    """
    Generate a list of top-k recommended items for a given user.

    Parameters
    ----------
    user_id : str
        ID of the user for whom recommendations are generated.
    rank : numpy.ndarray
        PageRank score for each node.
    index : dict
        Mapping from node to matrix index.
    all_nodes : list
        List of all nodes in index order.
    top_k : int, optional
        Number of recommendations to return (default is 5).

    Returns
    -------
    list of tuples
        Each tuple contains (item_id, score) sorted by score in descending order.
    >>> import numpy as np
    >>> rank = np.array([0.1, 0.4, 0.3, 0.2])
    >>> all_nodes = ["user1", "user2", "itemA", "itemB"]
    >>> index = {"user1": 0, "user2": 1, "itemA": 2, "itemB": 3}
    >>> recommend("user1", rank, index, all_nodes, top_k=2)
    [('itemA', np.float64(0.3)), ('itemB', np.float64(0.2))]
    """
    scores = []
    for node in all_nodes:
        # Recommend only items, skip users
        if node != user_id and not str(node).startswith("user"):
            idx = index[node]
            scores.append((node, rank[idx]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# Example of usage
if __name__ == "__main__":
    users = ["user1", "user2", "user3"]
    items = ["songA", "songB", "songC", "songD"]

    interactions = [
        ("user1", "songA"),
        ("user1", "songB"),
        ("user2", "songA"),
        ("user3", "songC"),
    ]

    matrix, index, all_nodes = build_interaction_graph(
        items, users, interactions)

    ranks = pagerank(matrix)

    recs = recommend("user1", ranks, index, all_nodes, top_k=3)

    print("Recommendations for user1:")
    for item, score in recs:
        print(f"{item}: {score:.4f}")

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
