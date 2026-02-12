import os, gc, re
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Config
from pyctcdecode import build_ctcdecoder

from dataloader import (
    text_to_tensor, dict_vocab,
    PAD_ID, BLANK_ID, VOCAB_SIZE, feature_extractor,
    L1_TO_ID, NUM_L1
)
from model import L1_BASELINE_MDD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()

CHECKPOINT_DIR = "checkpoint"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")

df_test = pd.read_csv("test.csv")

# decoder labels
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
model = L1_BASELINE_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID, num_l1=NUM_L1).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_results = []
with torch.no_grad():
    for i in tqdm(range(len(df_test)), desc="Inference"):
        wav_path = "EN_MDD/WAV/" + df_test.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1,T)

        canonical_str = df_test.loc[i, "Canonical"]
        canonical_ids = text_to_tensor(canonical_str)
        canonical = torch.tensor(canonical_ids, device=device).long().unsqueeze(0)

        l1 = str(df_test.loc[i, "L1"]).lower()
        l1_id = torch.tensor([L1_TO_ID.get(l1, 0)], device=device).long()

        logits = model(audio, canonical, l1_id)
        log_probs = logits.log_softmax(dim=-1).squeeze(0).detach().cpu().numpy()

        hyp = clean_hyp(decoder_ctc.decode(log_probs))

        all_results.append({
            "Path": df_test.loc[i, "Path"],
            "L1": df_test.loc[i, "L1"],
            "Canonical": canonical_str,
            "Transcript": df_test.loc[i, "Transcript"],
            "Predicted": hyp,
        })

out_df = pd.DataFrame(all_results)
out_df.to_csv("predictions.csv", index=False)
print("Saved: predictions.csv")
