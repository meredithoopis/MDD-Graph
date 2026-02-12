import json
import networkx as nx
import matplotlib.pyplot as plt


l1 = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def get_graph(
    languages,
    threshold: float = 0.0,
    alpha: float = 1.0,
    topk: int | None = None,
    renorm_after_topk: bool = True,
):
    edges = {}
    weights = {}

    for lang in languages:
        data = json.load(open(f"data_{lang}.json", "r", encoding="utf8"))
        
        by_i = {}  # i -> list[(j, cnt)]
        for key, value in data.items():
            i, j = map(int, key.split("_"))
            cnt = float(value)

            if cnt <= 0:
                continue
            # if i == j:
            #     continue  

            by_i.setdefault(i, []).append((j, cnt))

        lang_edges = []
        lang_weights = []

        for i, items in by_i.items():
            denom = sum(cnt for (_, cnt) in items)
            if denom <= 0:
                continue
            dist = [(j, cnt / denom) for (j, cnt) in items]

            if alpha is not None and alpha != 1.0:
                s = sum((w ** alpha) for (_, w) in dist)
                if s > 0:
                    dist = [(j, (w ** alpha) / s) for (j, w) in dist]

            if topk is not None and topk > 0:
                dist = sorted(dist, key=lambda x: x[1], reverse=True)[:topk]
                if renorm_after_topk:
                    s = sum(w for (_, w) in dist)
                    if s > 0:
                        dist = [(j, w / s) for (j, w) in dist]

            if threshold is not None and threshold > 0.0:
                dist = [(j, w) for (j, w) in dist if w >= threshold]
                
            for j, w in dist:
                lang_edges.append((j, i))
                lang_weights.append(float(w))

        edges[lang] = lang_edges
        weights[lang] = lang_weights

    return edges, weights


def visualize_multi_nodes(edges, weights, target_nodes):
    G = nx.DiGraph()
    for (u, v), w in zip(edges, weights):
        if u in target_nodes:
            G.add_edge(u, v, weight=round(w, 3))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=1000, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"))
    plt.title(f"Ego graph of nodes {target_nodes}")
    plt.show()



# all_edges, all_weights = get_graph(l1, threshold=0.01, alpha=0.7, topk=10)
