import json
import networkx as nx
import matplotlib.pyplot as plt
import torch 


def get_graph_from_json(json_path: str, alpha: float = 0.6, topk: int | None = None):
    """
    Build graph edges for j -> i (observed -> canonical), weight = P(i | j)
    alpha: temperature smoothing (w^alpha then renorm per source node j)
    topk: keep only top-k outgoing edges per source node j (optional)
    """
    data = json.load(open(json_path, "r", encoding="utf8"))

    # grouped_sum over observed node j
    grouped_sum = {}
    raw_edges_by_j = {}

    for key, value in data.items():
        i, j = map(int, key.split('_'))  # i=canonical, j=observed
        grouped_sum[j] = grouped_sum.get(j, 0) + value
        raw_edges_by_j.setdefault(j, []).append((i, value))

    edges = []
    weights = []

    for j, items in raw_edges_by_j.items():
        denom = grouped_sum.get(j, 0)
        if denom == 0:
            continue

        # base weights: P(i|j)
        tmp = []
        for (i, cnt) in items:
            w = cnt / denom
            tmp.append((i, w))

        # optional smoothing
        if alpha != 1.0:
            s = sum((w ** alpha) for (_, w) in tmp)
            if s > 0:
                tmp = [(i, (w ** alpha) / s) for (i, w) in tmp]

        # optional top-k pruning
        if topk is not None and topk > 0:
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)[:topk]
            s = sum(w for (_, w) in tmp)
            if s > 0:
                tmp = [(i, w / s) for (i, w) in tmp]

        for (i, w) in tmp:
            edges.append((j, i))     # j -> i
            weights.append(float(w))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_weight = torch.tensor(weights, dtype=torch.float)              # (E,)
    return edge_index, edge_weight


# all_edges, all_weights = get_graph_from_json("data_all.json", alpha=0.6, topk=None)
# print(all_edges)
# print(all_weights)




# def visualize_multi_nodes(edges, weights, target_nodes):
#     G = nx.DiGraph()

#     for (u, v), w in zip(edges, weights):
#         if u in target_nodes:
#             G.add_edge(u, v, weight=round(w, 3))

#     pos = nx.spring_layout(G, seed=42)

#     plt.figure(figsize=(8, 8))
#     nx.draw(
#         G, pos,
#         with_labels=True,
#         node_size=1000,
#         arrows=True
#     )

#     nx.draw_networkx_edge_labels(
#         G, pos,
#         edge_labels=nx.get_edge_attributes(G, 'weight')
#     )

#     plt.title(f"Ego graph of nodes {target_nodes}")
#     plt.show()


