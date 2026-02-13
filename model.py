import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class L1_BASELINE_MDD(Wav2Vec2PreTrainedModel):
    def __init__(self, config, vocab_size: int, pad_id: int, num_l1: int):
        super().__init__(config)

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        hidden = self.wav2vec2.config.hidden_size  # 1024

        self.ph_embed = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        nn.init.xavier_uniform_(self.ph_embed.weight, gain=0.5)
        with torch.no_grad():
            self.ph_embed.weight[pad_id].zero_()

        self.l1_embed = nn.Embedding(num_l1, hidden)
        nn.init.normal_(self.l1_embed.weight, mean=0.0, std=0.02)

        self.l1_gate = nn.Linear(hidden, hidden)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=16,
            dropout=0.2,
            batch_first=True,
        )
        self.linear_q = nn.Linear(hidden, hidden)
        self.linear_k = nn.Linear(hidden, hidden)
        self.linear_v = nn.Linear(hidden, hidden)

        self.classifier = nn.Linear(hidden * 2, vocab_size)
        self.post_init()

    def forward(self, audio_input, canonical, l1_ids, audio_mask=None):
        if audio_mask is None:
            acoustic = self.wav2vec2(audio_input).last_hidden_state  # (B,T,H)
        else:
            acoustic = self.wav2vec2(audio_input, attention_mask=audio_mask).last_hidden_state

        l1_vec = self.l1_embed(l1_ids).unsqueeze(1)  # (B,1,H)
        g = torch.sigmoid(self.l1_gate(l1_vec))      # (B,1,H)
        acoustic = acoustic + g * l1_vec

        linguistic = self.ph_embed(canonical)        # (B,T_can,H)

        Q = self.linear_q(acoustic)
        K = self.linear_k(linguistic)
        V = self.linear_v(linguistic)

        context, _ = self.cross_attn(
            Q, K, V,
            key_padding_mask=canonical.eq(self.pad_id)
        )

        fused = torch.cat([acoustic, context], dim=-1)
        logits = self.classifier(fused)
        return logits
