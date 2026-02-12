import os, gc, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Config
from pyctcdecode import build_ctcdecoder
from jiwer import wer
from dataloader import (
    MDD_Dataset, collate_fn,
    text_to_tensor, feature_extractor,
    PAD_ID, BLANK_ID, VOCAB_SIZE, dict_vocab,
    NUM_L1, L1_TO_ID
)
from model import L1_BASELINE_MDD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()

NUM_EPOCH = 15
BATCH_SIZE = 8
LR = 1e-5
CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df_train = pd.read_csv("train.csv")
df_dev   = pd.read_csv("dev.csv")

train_loader = torch.utils.data.DataLoader(
    MDD_Dataset(df_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

# model
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = L1_BASELINE_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID, num_l1=NUM_L1).to(device)
model.wav2vec2.feature_extractor._freeze_parameters()

ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

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

@torch.no_grad()
def eval_dev_wer(model, df_dev: pd.DataFrame) -> float:
    model.eval()
    wers = []
    for i in tqdm(range(len(df_dev)), desc="Dev", leave=False):
        wav_path = "EN_MDD/WAV/" + df_dev.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1,T)

        canonical_ids = text_to_tensor(df_dev.loc[i, "Canonical"])
        canonical = torch.tensor(canonical_ids, device=device).long().unsqueeze(0)

        l1 = str(df_dev.loc[i, "L1"]).lower()
        l1_id = torch.tensor([L1_TO_ID.get(l1, 0)], device=device).long()

        logits = model(audio, canonical, l1_id)  # (1, T', V)
        log_probs = F.log_softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()

        hyp = clean_hyp(decoder_ctc.decode(log_probs))
        ref = str(df_dev.loc[i, "Transcript"]).strip()
        wers.append(wer(ref, hyp))

    return float(sum(wers) / max(1, len(wers)))

best_dev_wer = 1e9

for epoch in range(NUM_EPOCH):
    model.train()
    epoch_losses = []

    print(f"\nEPOCH {epoch}\n" + "=" * 40)
    for batch in tqdm(train_loader, desc="Train", leave=False):
        audio, canonical, transcript_flat, transcript_lengths, l1_ids = batch

        with torch.cuda.amp.autocast():
            logits = model(audio, canonical, l1_ids)  # (B, T', V)

            input_lengths = torch.full(
                (logits.size(0),),
                logits.size(1),
                dtype=torch.long,
                device=device
            )

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T',B,V)
            loss = ctc_loss(log_probs, transcript_flat, input_lengths, transcript_lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(loss.item())

    dev_wer = eval_dev_wer(model, df_dev)
    print(f"Dev WER: {dev_wer:.4f}")

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "dev_wer": dev_wer},
        os.path.join(CHECKPOINT_DIR, "last.pth")
    )

    if dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "dev_wer": best_dev_wer},
            os.path.join(CHECKPOINT_DIR, "best.pth")
        )
        print("Saved best.pth")
