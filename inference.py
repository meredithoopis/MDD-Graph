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
from create_graph import get_graph_from_json  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoint"
gc.collect()


id_to_token = {v: k for k, v in dict_vocab.items()}

df_test = pd.read_csv("test.csv")

def ctc_greedy_decode(log_probs, blank_id):
    """
    log_probs: (T, V)
    returns: list[int]
    """
    pred = log_probs.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for p in pred:
        if p != blank_id and p != prev:
            decoded.append(p)
        prev = p
    return decoded


edge_index, edge_weight = get_graph_from_json(
    "data_all.json",
    alpha=0.7, 
    topk=None
)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)


ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pth")  
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-100h")
model = GCN_MDD(
    config,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.set_graph(edge_index, edge_weight)
model.eval()

all_results = []

with torch.no_grad():
    for i in tqdm(range(len(df_test)), desc="Inference"):
        wav_path = "EN_MDD/WAV/" + df_test.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)

        # wav2vec2 input
        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1, T)
        audio_mask = torch.ones_like(audio, dtype=torch.long, device=device)

        # canonical prompt
        canonical_str = df_test.loc[i, "Canonical"]
        canonical_ids = text_to_tensor(canonical_str)
        canonical = torch.tensor(canonical_ids, dtype=torch.long, device=device).unsqueeze(0)

        # forward
        logits = model(audio, audio_mask, canonical)
        log_probs = logits.log_softmax(dim=-1)
    
        # CTC decode
        decoded_ids = ctc_greedy_decode(log_probs.squeeze(0), blank_id=BLANK_ID)
        decoded_tokens = [id_to_token.get(pid, "UNK") for pid in decoded_ids if pid != PAD_ID]

        all_results.append({
            "Path": df_test.loc[i, "Path"],
            "L1": df_test.loc[i, "L1"],  
            "Canonical": canonical_str,
            "Transcript": df_test.loc[i, "Transcript"],
            "Predicted": " ".join(decoded_tokens),
        })


out_df = pd.DataFrame(all_results)
out_path = "predictions.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
