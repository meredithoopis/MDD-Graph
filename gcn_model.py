import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class LookUpGCN(nn.Module):
    def __init__(self, num_phonemes, embed_dim, hidden_channels, out_channels, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, embed_dim, padding_idx=pad_id)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.5)  # Reduce gain
        
        if pad_id is not None and pad_id < num_phonemes:
            with torch.no_grad():
                self.embedding.weight[pad_id].zero_()

        self.conv1 = GATv2Conv(embed_dim, hidden_channels, heads=1, concat=False, add_self_loops=False, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False, add_self_loops=False, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, phoneme_indices, edge_index, edge_weight):
        if phoneme_indices.max() >= self.embedding.num_embeddings:
            raise ValueError(f"Index out of range: max={phoneme_indices.max()}")
        x = self.embedding(phoneme_indices)
        if edge_index.numel() == 0:
            return x
        edge_attr = edge_weight.unsqueeze(-1)  # (E, 1)
        
        x1 = self.conv1(x, edge_index, edge_attr=edge_attr) 
        x = self.norm1(x + self.dropout(x1))
        x2 = self.conv2(x, edge_index,edge_attr=edge_attr)
        x = self.norm2(x + self.dropout(x2))
        
        return x



class GCN_MDD(Wav2Vec2PreTrainedModel):
    def __init__(self, config, vocab_size=41, pad_id=40):
        super().__init__(config)

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.wav2vec2 = Wav2Vec2Model(config)

        self.look_up_model = LookUpGCN(
            num_phonemes=vocab_size,
            embed_dim=config.hidden_size,
            hidden_channels=config.hidden_size,
            out_channels=config.hidden_size,
            pad_id=pad_id,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=16,
            dropout=0.2,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(config.hidden_size)
        self.norm_k = nn.LayerNorm(config.hidden_size)

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.diag_head = nn.Linear(config.hidden_size, vocab_size)
        self.err_head = nn.Linear(config.hidden_size, 1)

        self._initialize_custom_weights()
        # graph buffers
        self.register_buffer("indices", torch.arange(self.vocab_size, dtype=torch.long), persistent=False)
        self.register_buffer("edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("edge_weight", torch.empty((0,), dtype=torch.float), persistent=False)

        self.post_init()  

    def _initialize_custom_weights(self):
        """Initialize weights for custom layers"""
        for module in [self.linear_q, self.linear_k, self.linear_v]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Increase slightly
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in [self.diag_head, self.err_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def set_graph(self, edge_index, edge_weight=None):
        edge_index = edge_index.to(device=self.indices.device, dtype=torch.long).contiguous()
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=self.indices.device, dtype=torch.float)
        else:
            edge_weight = edge_weight.to(device=self.indices.device, dtype=torch.float).contiguous()

        if edge_index.numel() > 0:
            if edge_index.min().item() < 0 or edge_index.max().item() >= self.vocab_size:
                raise ValueError(f"edge_index out of range: min={edge_index.min().item()} "
                                f"max={edge_index.max().item()} vocab_size={self.vocab_size}")
            if edge_weight.numel() != edge_index.size(1):
                raise ValueError("edge_weight length must match num_edges")

        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    
    def look_up_table(self, canonical_ids):
        if self.edge_index.numel() == 0:
            raise RuntimeError("Graph not set. Call set_graph(...) first.")

        if torch.any(canonical_ids < 0) or torch.any(canonical_ids >= self.vocab_size):
            bad = canonical_ids[(canonical_ids < 0) | (canonical_ids >= self.vocab_size)]
            raise ValueError(f"canonical has out-of-range ids. min={canonical_ids.min().item()} "
                            f"max={canonical_ids.max().item()} bad_sample={bad[:10].tolist()}")

        all_nodes = self.look_up_model(self.indices, self.edge_index, self.edge_weight)  # (V, H)
        canonical_ids = canonical_ids.to(dtype=torch.long, device=all_nodes.device)
        return all_nodes[canonical_ids]  # (B, N, H)

# 
    def forward(self, audio_input, audio_mask, canonical):
        # Extract acoustic features
        acoustic = self.wav2vec2(audio_input, attention_mask=audio_mask).last_hidden_state
        acoustic = torch.nan_to_num(acoustic, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get prompt embeddings
        prompt_embed = self.look_up_table(canonical)    
        prompt_embed = torch.nan_to_num(prompt_embed, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply linear transformations 
        Q = self.linear_q(prompt_embed)
        Q = self.norm_q(Q)
        K = self.linear_k(acoustic)
        K = self.norm_k(K)
        V = self.linear_v(acoustic)

        # Create masks
        pad_mask = canonical.eq(self.pad_id)
        
        B, T_wav = audio_mask.shape
        T_feat = acoustic.size(1)
        
        audio_mask_feat = torch.nn.functional.interpolate(
            audio_mask.float().unsqueeze(1),
            size=T_feat,
            mode="nearest"
        ).squeeze(1).long()
        key_padding_mask = ~audio_mask_feat.bool()
        
        if not torch.isfinite(Q).all():
            Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(K).all():
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(V).all():
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        
        aligned, _ = self.cross_attn(
            Q, K, V,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        aligned = torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        aligned = aligned.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        
        # Output heads
        diag_logits = self.diag_head(aligned)
        err_logits = self.err_head(aligned).squeeze(-1)

        return diag_logits, err_logits