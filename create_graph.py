import json
import networkx as nx
import matplotlib.pyplot as plt
import torch 


def get_graph_from_json(
    json_path: str,
    alpha: float = 1.0,
    topk: int | None = None,
    min_prob: float = 0.0,
    renorm_after_filter: bool = True,
):

    data = json.load(open(json_path, "r", encoding="utf8"))

    # group by canonical i
    raw_by_i = {}
    for key, value in data.items():
        i, j = map(int, key.split("_"))  # i=canonical, j=observed
        cnt = float(value)
        if cnt <= 0:
            continue
        # if i == j:
        #     continue  
        raw_by_i.setdefault(i, []).append((j, cnt))

    edges = []
    weights = []

    for i, items in raw_by_i.items():
        denom = sum(cnt for _, cnt in items)
        if denom <= 0:
            continue

        tmp = [(j, cnt / denom) for (j, cnt) in items]

        # smoothing (per i)
        if alpha != 1.0:
            s = sum((w ** alpha) for (_, w) in tmp)
            if s > 0:
                tmp = [(j, (w ** alpha) / s) for (j, w) in tmp]

        # filter by min_prob
        if min_prob > 0.0:
            tmp = [(j, w) for (j, w) in tmp if w >= min_prob]
            if not tmp:
                continue
            if renorm_after_filter:
                s = sum(w for _, w in tmp)
                if s > 0:
                    tmp = [(j, w / s) for (j, w) in tmp]

        # top-k per i
        if topk is not None and topk > 0:
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)[:topk]
            if renorm_after_filter:
                s = sum(w for _, w in tmp)
                if s > 0:
                    tmp = [(j, w / s) for (j, w) in tmp]
        for (j, w) in tmp:
            edges.append((j, i))
            weights.append(float(w))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


# all_edges, all_weights = get_graph_from_json("data_all.json", alpha=0.6, topk=None, min_prob=0.0, renorm_after_filter=True)
# print(all_edges)
# print(all_weights)


