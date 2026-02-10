import os
import gc
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Config

from dataloader import (
    text_to_tensor,
    dict_vocab,
    PAD_ID,
    BLANK_ID,
    VOCAB_SIZE,
    feature_extractor,
)
from gcn_model import GCN_MDD
from create_graph import get_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoint"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")

L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

gc.collect()

id_to_token = {v: k for k, v in dict_vocab.items()}

df_test = pd.read_csv("test.csv")

def ctc_greedy_decode(log_probs, blank_id):
    pred = log_probs.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for p in pred:
        if p != blank_id and p != prev:
            decoded.append(p)
        prev = p
    return decoded

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-100h")
model = GCN_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_results = []

all_edges, all_weights = get_graph(L1_LIST, threshold=0.01, alpha=0.7, topk=10)

with torch.no_grad():
    for L1_LANG in L1_LIST:
        df_l1 = df_test[df_test["L1"].str.lower() == L1_LANG].reset_index(drop=True)
        if len(df_l1) == 0:
            print(f"No test samples for {L1_LANG}, skipping.")
            continue

        edges = all_edges[L1_LANG]
        weights = all_weights[L1_LANG]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(weights, dtype=torch.float).to(device)
        model.set_graph(edge_index, edge_weight)

        for i in tqdm(range(len(df_l1)), desc=L1_LANG):
            wav_path = "EN_MDD/WAV/" + df_l1.loc[i, "Path"] + ".wav"
            wav, _ = librosa.load(wav_path, sr=16000)

            inputs = feature_extractor(wav, sampling_rate=16000)
            audio = torch.tensor(inputs.input_values, device=device).float()
            audio_mask = torch.ones_like(audio, dtype=torch.long, device=device)

            canonical_str = df_l1.loc[i, "Canonical"]
            canonical_ids = text_to_tensor(canonical_str)
            canonical = torch.tensor(canonical_ids, dtype=torch.long, device=device).unsqueeze(0)

            logits = model(audio, audio_mask, canonical)
            log_probs = logits.log_softmax(dim=-1)

            decoded_ids = ctc_greedy_decode(log_probs.squeeze(0), blank_id=BLANK_ID)

            decoded_tokens = []
            for pid in decoded_ids:
                if pid in (BLANK_ID, PAD_ID):
                    continue
                decoded_tokens.append(id_to_token.get(pid, "UNK"))

            all_results.append({
                "Path": df_l1.loc[i, "Path"],
                "L1": df_l1.loc[i, "L1"],
                "Canonical": canonical_str,
                "Transcript": df_l1.loc[i, "Transcript"],
                "Predicted": " ".join(decoded_tokens),
            })

out_df = pd.DataFrame(all_results)
out_path = "predictions.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
