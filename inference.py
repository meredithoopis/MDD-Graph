import os
import gc
import re
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Config
from pyctcdecode import build_ctcdecoder
from dataloader import (
    text_to_tensor,
    dict_vocab,
    PAD_ID,
    BLANK_ID,
    VOCAB_SIZE,
    feature_extractor,
)
from gcn_model import GCN_MDD
from create_graph import build_category_graph  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoint"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")

L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
gc.collect()

id_to_token = {v: k for k, v in dict_vocab.items()}
EPS_SYM = "¤"

labels_for_decoder = []
for idx in range(VOCAB_SIZE):
    tok = id_to_token[idx]
    if idx == BLANK_ID:
        labels_for_decoder.append("")
    elif idx == PAD_ID:
        labels_for_decoder.append(EPS_SYM)
    else:
        labels_for_decoder.append(tok + " ")

decoder_ctc = build_ctcdecoder(labels=labels_for_decoder)

def clean_hyp(h: str) -> str:
    h = h.replace(EPS_SYM, " ")
    h = re.sub(r"\s+", " ", h).strip()
    return h

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = GCN_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

edge_index, _, _, _ = build_category_graph(
    dict_vocab=dict_vocab,
    category_json_path="category.json",
    pad_token="<eps>",
    blank_token="<blank>",
    device=device,
)

model.set_graph(edge_index)

df_test = pd.read_csv("test.csv")
all_results = []

with torch.no_grad():
    for L1_LANG in L1_LIST:
        df_l1 = df_test[df_test["L1"].str.lower() == L1_LANG].reset_index(drop=True)
        if len(df_l1) == 0:
            print(f"No test samples for {L1_LANG}, skipping.")
            continue

        for i in tqdm(range(len(df_l1)), desc=L1_LANG):
            wav_path = "EN_MDD/WAV/" + df_l1.loc[i, "Path"] + ".wav"
            wav, _ = librosa.load(wav_path, sr=16000)

            inputs = feature_extractor(wav, sampling_rate=16000)
            audio = torch.tensor(inputs.input_values, device=device).float()

            canonical_str = df_l1.loc[i, "Canonical"]
            canonical_ids = text_to_tensor(canonical_str)
            canonical = torch.tensor(canonical_ids, dtype=torch.long, device=device).unsqueeze(0)

            logits = model(audio, canonical)
            log_probs = logits.log_softmax(dim=-1).squeeze(0).detach().cpu().numpy()

            hyp = clean_hyp(decoder_ctc.decode(log_probs))

            all_results.append({
                "Path": df_l1.loc[i, "Path"],
                "L1": df_l1.loc[i, "L1"],
                "Canonical": canonical_str,
                "Transcript": df_l1.loc[i, "Transcript"],
                "Predicted": hyp,
            })

out_df = pd.DataFrame(all_results)
out_path = "predictions.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
