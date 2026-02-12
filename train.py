import os, gc, re
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import librosa

from dataloader import (
    MDD_Dataset,
    collate_fn,
    text_to_tensor,
    dict_vocab,
    PAD_ID,
    BLANK_ID,
    VOCAB_SIZE,
    feature_extractor,
)
from jiwer import wer
from pyctcdecode import build_ctcdecoder
from gcn_model import GCN_MDD
from create_graph import get_graph_from_json
from transformers import Wav2Vec2Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()

num_epoch = 15
batch_size = 32
lr = 5e-5

CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

df_train = pd.read_csv("train.csv")
df_dev = pd.read_csv("dev.csv")

train_loader = torch.utils.data.DataLoader(
    MDD_Dataset(df_train),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

edge_index, edge_weight = get_graph_from_json(
    "data_all.json",
    alpha=0.7,
    topk=None,
    min_prob=0.0,
    renorm_after_filter=True,
)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = GCN_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).to(device)
model.wav2vec2.feature_extractor._freeze_parameters()
model.set_graph(edge_index, edge_weight)

ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()


id_to_token = {v: k for k, v in dict_vocab.items()}
EPS_SYM = "¤"  

labels = []
for i in range(VOCAB_SIZE):
    tok = id_to_token[i]
    if i == BLANK_ID:
        labels.append("")          # blank
    elif i == PAD_ID:
        labels.append(EPS_SYM)     # eps
    else:
        labels.append(tok + " ")   # add space

decoder = build_ctcdecoder(labels=labels)

def clean_hyp(s: str) -> str:
    s = s.replace(EPS_SYM, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

        logits = model(audio, canonical)  # (1, T', V)
        log_probs = logits.log_softmax(dim=-1).squeeze(0).cpu().numpy()  # (T',V)

        hyp = clean_hyp(decoder.decode(log_probs))
        ref = df_dev.loc[i, "Transcript"].strip()
        wers.append(wer(ref, hyp))

    return float(sum(wers) / max(1, len(wers)))

best_dev_wer = 1e9

for epoch in range(num_epoch):
    model.train()
    losses = []

    print(f"\nEPOCH {epoch}")
    for batch in tqdm(train_loader, desc="Train", leave=False):
        audio, canonical, transcript_flat, transcript_lengths = batch  # audio: (B,T)

        with torch.cuda.amp.autocast():
            logits = model(audio, canonical)  # (B, T', V)

            input_lengths = torch.full(
                (logits.size(0),),
                logits.size(1),
                dtype=torch.long,
                device=device
            )

            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T',B,V)
            loss = ctc_loss(log_probs, transcript_flat, input_lengths, transcript_lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

    train_loss = sum(losses) / max(1, len(losses))
    #print(f"Train loss: {train_loss:.4f}")

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
