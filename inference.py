import os, gc, re
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Config

from dataloader import text_to_tensor, dict_vocab, PAD_ID, BLANK_ID, VOCAB_SIZE, feature_extractor
from gcn_model import GCN_MDD
from create_graph import get_graph_from_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()

CHECKPOINT_DIR = "checkpoint"
ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pth")

df_test = pd.read_csv("test.csv")

# graph
edge_index, edge_weight = get_graph_from_json("data_all.json", alpha=0.7, topk=None, min_prob=0.0, renorm_after_filter=True)
edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

# model
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-100h")
model = GCN_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.set_graph(edge_index, edge_weight)
model.eval()

# decoder labels
id_to_token = {v: k for k, v in dict_vocab.items()}
EPS_SYM = "¤"
labels = []
for i in range(VOCAB_SIZE):
    tok = id_to_token[i]
    if i == BLANK_ID:
        labels.append("")
    elif i == PAD_ID:
        labels.append(EPS_SYM)
    else:
        labels.append(tok + " ")
decoder = build_ctcdecoder(labels=labels)

def clean_hyp(s: str) -> str:
    s = s.replace(EPS_SYM, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

all_results = []
with torch.no_grad():
    for i in tqdm(range(len(df_test)), desc="Inference"):
        wav_path = "EN_MDD/WAV/" + df_test.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)

        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1,T)

        canonical_ids = text_to_tensor(df_test.loc[i, "Canonical"])
        canonical = torch.tensor(canonical_ids, device=device).long().unsqueeze(0)

        logits = model(audio, canonical)
        log_probs = logits.log_softmax(dim=-1).squeeze(0).cpu().numpy()  # (T',V)

        hyp = clean_hyp(decoder.decode(log_probs))

        all_results.append({
            "Path": df_test.loc[i, "Path"],
            "L1": df_test.loc[i, "L1"],
            "Canonical": df_test.loc[i, "Canonical"],
            "Transcript": df_test.loc[i, "Transcript"],
            "Predicted": hyp,
        })

out_df = pd.DataFrame(all_results)
out_df.to_csv("predictions.csv", index=False)
print("Saved: predictions.csv")
