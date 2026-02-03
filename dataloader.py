import torch
from torch.utils.data import Dataset
import numpy as np
import json
import librosa
import ast
from transformers import Wav2Vec2FeatureExtractor
import pandas as pd


with open('vocab.json') as f:
    dict_vocab = json.load(f)

PAD_ID = dict_vocab["<eps>"]
VOCAB_SIZE = len(dict_vocab)


def text_to_tensor(string_text):
    text = string_text.split(" ")
    return [dict_vocab[t] for t in text]


class MDD_Dataset(Dataset):
    def __init__(self, data):
        self.len_data   = len(data)
        self.path       = list(data['Path'])
        self.canonical  = list(data['Canonical'])
        self.transcript = list(data['Transcript'])
        self.error      = list(data['Error'])

    def __getitem__(self, index):
        waveform, _ = librosa.load("EN_MDD/WAV/" + self.path[index] + ".wav", sr=16000)
        canonical_ids  = text_to_tensor(self.canonical[index])
        transcript_ids = text_to_tensor(self.transcript[index])
        error_str      = self.error[index]
        return waveform, canonical_ids, transcript_ids, error_str

    def __len__(self):
        return self.len_data


feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    padding_side='right',
    do_normalize=True,
    return_attention_mask=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, pad_id=PAD_ID, err_pad_id=2):
    """
    Returns:
      input_values:      (B, wav_len)
      attention_mask:    (B, wav_len)
      canonical:         (B, Nmax)
      transcript:        (B, Nmax)
      error_gt:          (B, Nmax)
      aligned_lengths:   (B,)   
    """

    with torch.no_grad():
        max_wav = 0
        max_len = 0  
        for wav, can, trn, err_str in batch:
            max_wav = max(max_wav, wav.shape[0])

            err = ast.literal_eval(err_str)
            min_len = min(len(can), len(trn), len(err))
            max_len = max(max_len, min_len)

        cols = {
            "waveform": [],
            "canonical": [],
            "transcript": [],
            "error": [],
            "aligned_lengths": [],
        }

        wav_lengths = []

        for wav, can, trn, err_str in batch:
            # Clean waveform
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            wav = np.clip(wav, -1.0, 1.0)

    
            err = ast.literal_eval(err_str)
            # Align lengths
            min_len = min(len(can), len(trn), len(err))
            cols["aligned_lengths"].append(min_len)
            can = can[:min_len]
            trn = trn[:min_len]
            err = err[:min_len]

            # Validate IDs
            if len(can) > 0:
                if max(can) >= VOCAB_SIZE or min(can) < 0:
                    raise ValueError(f"canonical IDs out of range: min={min(can)} max={max(can)} vocab={VOCAB_SIZE}")
            if len(trn) > 0:
                if max(trn) >= VOCAB_SIZE or min(trn) < 0:
                    raise ValueError(f"transcript IDs out of range: min={min(trn)} max={max(trn)} vocab={VOCAB_SIZE}")

            # Pad waveform
            wav_lengths.append(len(wav))
            pad_wav = np.concatenate([wav, np.zeros(max_wav - len(wav))])
            cols["waveform"].append(pad_wav)

            # Pad canonical/transcript/error to max_len
            can_pad = list(can) + [pad_id] * (max_len - len(can))
            trn_pad = list(trn) + [pad_id] * (max_len - len(trn))
            err_pad = list(err) + [err_pad_id] * (max_len - len(err))

            cols["canonical"].append(can_pad)
            cols["transcript"].append(trn_pad)
            cols["error"].append(err_pad)

        # Build attention mask for audio padding
        attention_mask = []
        for L in wav_lengths:
            mask = [1] * L + [0] * (max_wav - L)
            attention_mask.append(mask)

        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        # Wav2vec2 input values
        inputs = feature_extractor(cols["waveform"], sampling_rate=16000)
        input_values = torch.from_numpy(np.array(inputs.input_values)).to(device).float()
        input_values = torch.nan_to_num(input_values, nan=0.0, posinf=0.0, neginf=0.0)
        #input_values = torch.clamp(input_values, -5.0, 5.0)

        canonical = torch.tensor(cols["canonical"], dtype=torch.long, device=device)
        transcript = torch.tensor(cols["transcript"], dtype=torch.long, device=device)
        error_gt = torch.tensor(cols["error"], dtype=torch.long, device=device)
        aligned_lengths = torch.tensor(cols["aligned_lengths"], dtype=torch.long, device=device)

    return input_values, attention_mask, canonical, transcript, error_gt, aligned_lengths






# Graph-stats builder
def build_stats_graph_from_df(df, vocab_size: int = VOCAB_SIZE, pad_id: int = PAD_ID,
    top_k: int | None = None,
    add_self_loops: bool = False,
    smoothing: float = 0.0,
):
    """
    Build a directed weighted graph from train statistics:
    edge i->j exists if canonical i was realized as transcript j in train.
    weight w(i->j) = count(i->j) / sum_k count(i->k)

    Should be: Canonical and Transcript are position-aligned sequences (checked).
    """

    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    for _, row in df.iterrows():
        can_ids = text_to_tensor(row["Canonical"])
        trn_ids = text_to_tensor(row["Transcript"])

        L = min(len(can_ids), len(trn_ids))
        for n in range(L):
            i = can_ids[n]
            j = trn_ids[n]
            if i == pad_id or j == pad_id:
                continue
            counts[i, j] += 1.0

    if add_self_loops: #set true if want to increase self-loop probability and reduce confusion prob (dh->d smaller)
        for i in range(vocab_size):
            counts[i, i] += 1.0   

    if smoothing > 0:
        counts += smoothing

    row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
    probs = counts / row_sums

    edge_src, edge_dst, edge_w = [], [], []

    for i in range(vocab_size):
        if i == pad_id:
            continue

        p = probs[i].copy()

        if top_k is not None and top_k > 0:
            idx = np.argsort(-p)[:top_k]
            mask = np.zeros_like(p, dtype=bool)
            mask[idx] = True
        else:
            mask = p > 0

        for j in range(vocab_size):
            if j == pad_id:
                continue
            if mask[j] and p[j] > 0:
                edge_src.append(i)
                edge_dst.append(j)
                edge_w.append(float(p[j]))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_w, dtype=torch.float)

    # safety clamp
    valid = (
        (edge_index[0] >= 0) & (edge_index[0] < vocab_size) &
        (edge_index[1] >= 0) & (edge_index[1] < vocab_size)
    )

    edge_index = edge_index[:, valid]
    edge_weight = edge_weight[valid]
    edge_index = edge_index.contiguous()
    edge_weight = edge_weight.contiguous()
        
    
    return edge_index, edge_weight
