import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class LookUpGCN(nn.Module):
    """
    Linguistic encoder:
      - Nodes = vocab items (phonemes + special tokens)
      - Edges = undirected connections based on shared category
    """
    def __init__(self, num_phonemes, embed_dim, hidden_channels, out_channels, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, embed_dim, padding_idx=pad_id)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.5)

        if pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[pad_id].zero_()

        # GCN
        self.conv1 = GCNConv(embed_dim, hidden_channels, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True, normalize=True)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_ids, edge_index):
        x = self.embedding(node_ids)

        h1 = self.conv1(x, edge_index)
        x = self.norm1(x + self.dropout(h1))

        h2 = self.conv2(x, edge_index)
        x = self.norm2(x + self.dropout(h2))

        return x


class GCN_MDD(Wav2Vec2PreTrainedModel):
    def __init__(self, config, vocab_size: int, pad_id: int):
        super().__init__(config)

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # Acoustic encoder
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        hidden = self.wav2vec2.config.hidden_size  # 1024

        # Linguistic encoder (graph)
        self.look_up_model = LookUpGCN(
            num_phonemes=vocab_size,
            embed_dim=hidden,
            hidden_channels=hidden,
            out_channels=hidden,
            pad_id=pad_id,
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=16,
            dropout=0.2,
            batch_first=True,
        )

        self.linear_q = nn.Linear(hidden, hidden)
        self.linear_k = nn.Linear(hidden, hidden)
        self.linear_v = nn.Linear(hidden, hidden)

        # Final classifier
        self.classifier = nn.Linear(hidden * 2, vocab_size)

        # Graph buffers
        self.register_buffer(
            "indices",
            torch.arange(self.vocab_size, dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False
        )

        self.post_init()

    def set_graph(self, edge_index: torch.Tensor):
        self.edge_index = edge_index.contiguous()

    def look_up_table(self, canonical_ids):
        all_nodes = self.look_up_model(self.indices, self.edge_index)
        return all_nodes[canonical_ids]  # (B, T_can, H)

    def forward(self, audio_input, canonical, audio_mask=None):
        # Acoustic encoder
        if audio_mask is None:
            acoustic = self.wav2vec2(audio_input).last_hidden_state
        else:
            acoustic = self.wav2vec2(audio_input, attention_mask=audio_mask).last_hidden_state

        # Linguistic graph lookup
        linguistic = self.look_up_table(canonical)  # (B, T_can, H)

        # Cross-attention (audio queries linguistic)
        Q = self.linear_q(acoustic)
        K = self.linear_k(linguistic)
        V = self.linear_v(linguistic)

        context, _ = self.cross_attn(
            Q, K, V,
            key_padding_mask=canonical.eq(self.pad_id)
        )

        fused = torch.cat([acoustic, context], dim=-1)
        logits = self.classifier(fused)  # (B, T', V)

        return logits
