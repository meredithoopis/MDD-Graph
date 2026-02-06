import json
import networkx as nx
import matplotlib.pyplot as plt


# i mispronounced as j: j → i
l1 = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

def get_graph(languages):
    edges = {}
    weights = {}

    for lang in languages:
        data = json.load(open(f"data_{lang}.json", "r", encoding="utf8"))

        grouped_sum = {}
        for key, value in data.items():
            i, j = map(int, key.split('_'))   # i = canonical, j = observed
            grouped_sum[j] = grouped_sum.get(j, 0) + value

        lang_edges = []
        lang_weights = []

        # j -> i
        for key, value in data.items():
            i, j = map(int, key.split('_'))
            weight = value / grouped_sum[j]
            lang_edges.append((j, i))  
            lang_weights.append(weight)
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
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1000,
        arrows=True
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=nx.get_edge_attributes(G, 'weight')
    )

    plt.title(f"Ego graph of nodes {target_nodes}")
    plt.show()


all_edges, all_weights = get_graph(l1)
# print(all_edges)
# print(all_weights)




# visualize_multi_nodes(all_edges["arabic"], all_weights["arabic"], target_nodes=[0, 2])

# batching theo language trước xong forward => tiết kiệm thời gian không cần phải forward cả 6 graph
# norm lại weights theo từng node do one caveat
# embedding graph xong rồi mới lấy indices
# dùng mixture of expert? fusion graph statistic vs graph convolutional neural network


# print(all_edges["arabic"])
# print(all_weights["arabic"])


# 6️⃣ One caveat ⚠️ (very important)
# If weights are: extremely peaked (e.g. 0.99 / 0.01)
# or noisy / data-sparse
# GCN can:
# overfit
# oversmooth
# ignore rare but important errors
# Common fixes
# Temperature smoothing:

# w^alpha / (sum (w^alpha)), α ∈ (0.5, 1)
# Top-K pruning
# Entropy regularization