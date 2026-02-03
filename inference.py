import gc
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from dataloader import text_to_tensor, dict_vocab, PAD_ID, VOCAB_SIZE, feature_extractor
from gcn_model import GCN_MDD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ERR_PAD_ID = 2

gc.collect()

# Reverse vocab for decoding ids -> phoneme tokens
id_to_token = {v: k for k, v in dict_vocab.items()}

df_test = pd.read_csv("EN_MDD/test.csv")

# Load model checkpoint
model = GCN_MDD.from_pretrained(
    "facebook/wav2vec2-base-100h",
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID
).to(device)

ckp = torch.load("checkpoint/gcn_mdd_stats_best.pth", map_location=device)
model.load_state_dict(ckp["model_state_dict"])

# Set graph from checkpoint
model.set_graph(ckp["edge_index"].to(device), ckp["edge_weight"].to(device))
model.indices = torch.arange(model.vocab_size, dtype=torch.long, device=device)

model.eval()

PATH = []
CANONICAL = []
TRANSCRIPT = []
PRED_DIAG = []
PRED_ERR = []

with torch.no_grad():
    for i in tqdm(range(len(df_test))):
        wav_path = "EN_MDD/WAV/" + df_test["Path"][i] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)

        # Extract wav2vec2 input
        audio = feature_extractor(wav, sampling_rate=16000).input_values
        audio = torch.tensor(audio, device=device).float()  # (1, wav_len)

        # Build attention mask (1 for real audio, no padding here)
        audio_mask = torch.ones_like(audio, dtype=torch.long, device=device)  # (1, wav_len)

        # Canonical ids
        canonical_str = df_test["Canonical"][i]
        canonical_ids = text_to_tensor(canonical_str)
        canonical_ids = torch.tensor(canonical_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, N)

        # Forward
        diag_logits, err_logits = model(audio, audio_mask, canonical_ids)
        # diag_logits: (1, N, VOCAB_SIZE)
        # err_logits:  (1, N)

        # Diagnosis prediction
        diag_pred = diag_logits.argmax(dim=-1).squeeze(0)  # (N,)

        # Error prediction: sigmoid(logit) -> probability
        err_prob = torch.sigmoid(err_logits).squeeze(0)    # (N,)
        err_pred = (err_prob > 0.5).long()                 # (N,)

        # Decode diagnosis tokens
        diag_tokens = []
        for pid in diag_pred.tolist():
            if pid == PAD_ID:
                continue
            diag_tokens.append(id_to_token.get(pid, "UNK"))

        PATH.append(df_test["Path"][i])
        CANONICAL.append(canonical_str)
        TRANSCRIPT.append(df_test["Transcript"][i])
        PRED_DIAG.append(" ".join(diag_tokens))
        PRED_ERR.append(str(err_pred.tolist()))

out_df = pd.DataFrame({
    "Path": PATH,
    "Canonical": CANONICAL,
    "Transcript": TRANSCRIPT,
    "Predict_Diagnosis": PRED_DIAG,
    "Predict_Error": PRED_ERR,
})

out_df.to_csv("gmdd_predictions.csv", index=False)
print("Saved: gmdd_predictions.csv")
