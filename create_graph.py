import json
import torch
from itertools import combinations
from torch_geometric.utils import to_undirected

def build_category_graph(
    dict_vocab: dict,
    category_json_path: str = "category.json",
    pad_token: str = "<eps>",
    blank_token: str = "<blank>",
    device: str | torch.device = "cpu",
):
    with open(category_json_path, "r") as f:
        cat_map = json.load(f)  

    pad_id = dict_vocab.get(pad_token, None)
    blank_id = dict_vocab.get(blank_token, None)
    vocab_size = len(dict_vocab)

    # group node ids by category
    by_cat = {}
    for tok, idx in dict_vocab.items():
        if tok in (pad_token, blank_token):
            continue
        if tok not in cat_map:
            continue

        c = int(cat_map[tok])
        by_cat.setdefault(c, []).append(int(idx))

    # connect nodes within each category
    edges = []
    for nodes in by_cat.values():
        if len(nodes) < 2:
            continue
        for i, j in combinations(nodes, 2):
            edges.append([i, j])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index, pad_id, blank_id, vocab_size

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_index = to_undirected(edge_index).to(device)

    return edge_index, pad_id, blank_id, vocab_size
