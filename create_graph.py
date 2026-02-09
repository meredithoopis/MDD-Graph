import json
import networkx as nx
import matplotlib.pyplot as plt
import torch 


def get_graph_from_json( json_path: str, alpha: float = 1.0, topk: int | None = None, 
    min_prob: float = 0.0,
    renorm_after_filter: bool = True,
):
    """
    Build graph edges for j -> i (observed -> canonical), weight = P(i | j)
    alpha: temperature smoothing (w^alpha then renorm per source node j)
    topk: keep only top-k outgoing edges per source node j (optional)
    min_prob: drop edges with prob < min_prob (after normalization/smoothing)
    renorm_after_filter: renormalize probs after applying filters
    """
    data = json.load(open(json_path, "r", encoding="utf8"))

    raw_edges_by_j = {}
    for key, value in data.items():
        i, j = map(int, key.split('_'))  # i=canonical, j=observed
        cnt = float(value)
        raw_edges_by_j.setdefault(j, []).append((i, cnt))

    edges = []
    weights = []

    for j, items in raw_edges_by_j.items():
        denom = sum(cnt for _, cnt in items)
        if denom <= 0:
            continue

        # base probs P(i|j)
        tmp = [(i, cnt / denom) for (i, cnt) in items]

        # smoothing
        if alpha != 1.0:
            s = sum((w ** alpha) for (_, w) in tmp)
            if s > 0:
                tmp = [(i, (w ** alpha) / s) for (i, w) in tmp]

        if min_prob > 0.0:
            tmp = [(i, w) for (i, w) in tmp if w >= min_prob]
            if len(tmp) == 0:
                continue
            if renorm_after_filter:
                s = sum(w for _, w in tmp)
                tmp = [(i, w / s) for (i, w) in tmp]

        if topk is not None and topk > 0:
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)[:topk]
            if renorm_after_filter:
                s = sum(w for _, w in tmp)
                if s > 0:
                    tmp = [(i, w / s) for (i, w) in tmp]

        for (i, w) in tmp:
            edges.append((j, i))   # j -> i
            weights.append(float(w))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_weight = torch.tensor(weights, dtype=torch.float)              # (E,)
    return edge_index, edge_weight


# all_edges, all_weights = get_graph_from_json("data_all.json", alpha=0.7, topk=None, min_prob=0.0, renorm_after_filter=True)
# print(all_edges)
# print(all_weights)
