"""
Module: Universal PageRank algorithm
Provides functions to build any directed graph, compute PageRank scores,
and recommend top nodes based on PageRank.
"""

import numpy as np


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize adjacency matrix columns to sum to 1."""
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return matrix / col_sums


def pagerank(matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for any directed graph using power iteration.

    Args:
        matrix (np.ndarray): adjacency matrix
        damping (float): damping factor
        max_iter (int): maximum iterations
        tol (float): tolerance for convergence

    Returns:
        np.ndarray: PageRank scores
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

    Args:
        nodes (list): list of node IDs
        edges (list): list of tuples (source, target)

    Returns:
        tuple: adjacency matrix, index mapping, list of nodes
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((n, n))

    for src, tgt in edges:
        matrix[index[tgt], index[src]] = 1  # column = source, row = target

    return matrix, index, nodes


def recommend(node_id: str, rank: np.ndarray, index: dict, candidates: list, top_k: int = 5) -> list:
    """
    Recommend top-k nodes from any set of candidates based on PageRank scores.

    Args:
        node_id (str): node to exclude from recommendations
        rank (np.ndarray): PageRank scores
        index (dict): mapping node -> matrix index
        candidates (list): nodes to consider for recommendation
        top_k (int): number of recommendations

    Returns:
        list: top-k recommendations as tuples (node, score)
    """
    scores = [(node, rank[index[node]]) for node in candidates if node != node_id]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# Example usage
if __name__ == "__main__":
    # Example: a graph where any node can be connected to any node
    nodes_list = ["A", "B", "C", "D", "E"]
    edges_list = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
        ("D", "E"),
        ("E", "A"),  # creating a cycle
    ]

    adjacency_matrix, node_index, all_nodes = build_graph(nodes_list, edges_list)
    page_rank_scores = pagerank(adjacency_matrix)

    recommendations = recommend("A", page_rank_scores, node_index, all_nodes, top_k=3)

    print("Recommendations for A:")
    for node, score in recommendations:
        print(f"{node}: {score:.4f}")
