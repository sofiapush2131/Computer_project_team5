import argparse
from pagerank import (
    build_interaction_graph,
    pagerank,
    recommend,
)

users = ['user1', 'user2', 'user3']
items = ['songA', 'songB', 'songC', 'songD']

interactions = [
    ('user1', 'songA'),
    ('user1', 'songB'),
    ('user2', 'songA'),
    ('user3', 'songC'),
]


def main():
    parser = argparse.ArgumentParser(description='Recommendation System CLI')

    parser.add_argument('-u', '--user', type=str,
                        help='User ID to recommend for')
    parser.add_argument('--top', type=int, default=5,
                        help='Top-K recommendations')

    parser.add_argument('--pagerank', action='store_true',
                        help='Print PageRank scores')
    parser.add_argument('--damping', type=float,
                        default=0.85, help='Damping factor')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max PageRank iterations')

    parser.add_argument('--graph', action='store_true',
                        help='Print graph connections')

    args = parser.parse_args()

    matrix, index, all_nodes = build_interaction_graph(
        items, users, interactions)
# 1
    if args.user:
        if args.user not in users:
            print(f'User {args.user} not found.')
            return

        rank = pagerank(matrix, d=args.damping, max_iter=args.max_iter)
        recs = recommend(args.user, rank, index, all_nodes, top_k=args.top)

        print(f'Recommendations for {args.user}:')
        for item, score in recs:
            print(f'{item}: {score:.4f}')
        return
# 2
    if args.pagerank:
        rank = pagerank(matrix, d=args.damping, max_iter=args.max_iter)

        print('PageRank values:')
        for node, idx in index.items():
            print(f'{node}: {rank[idx]:.4f}')
        return
# 3
    if args.graph:
        print('Graph:')
        for u, i in interactions:
            print(f'{u} -> {i}')
        return

    parser.print_help()


if __name__ == '__main__':
    main()


# python CLI.py --user user1 --top 3
# python CLI.py --pagerank --damping 0.9
# python CLI.py --graph
