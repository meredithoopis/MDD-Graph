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
    PAD_ID, BLANK_ID, VOCAB_SIZE, dict_vocab
)
from gcn_model import GCN_MDD
from create_graph import get_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_EPOCH = 15
BATCH_SIZE = 8
LR = 5e-5
CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

df_train = pd.read_csv("train.csv")
df_dev   = pd.read_csv("dev.csv")

train_loaders = {}
for L1 in L1_LIST:
    df_l1 = df_train[df_train["L1"].str.lower() == L1].reset_index(drop=True)
    if len(df_l1) == 0:
        print(f"No train samples for {L1}, skip.")
        continue
    train_loaders[L1] = torch.utils.data.DataLoader(
        MDD_Dataset(df_l1),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )


all_edges, all_weights = get_graph(
    L1_LIST,
    threshold=0.01,   
    alpha=0.7,
    topk=10
)


config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = GCN_MDD(config, vocab_size=VOCAB_SIZE, pad_id=PAD_ID).to(device)
model.wav2vec2.feature_extractor._freeze_parameters()

ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()


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
def eval_dev_wer(model: GCN_MDD, df_dev: pd.DataFrame) -> float:
    model.eval()
    wers = []

    for i in tqdm(range(len(df_dev)), desc="Dev", leave=False):
        l1 = str(df_dev.loc[i, "L1"]).lower()
        if l1 not in all_edges:
            continue

        edge_index = torch.tensor(all_edges[l1], dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(all_weights[l1], dtype=torch.float).to(device)
        model.set_graph(edge_index, edge_weight)

        wav_path = "EN_MDD/WAV/" + df_dev.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1,T)

        canonical_ids = text_to_tensor(df_dev.loc[i, "Canonical"])
        canonical = torch.tensor(canonical_ids, device=device).long().unsqueeze(0)

        logits = model(audio, canonical)  # (1, T', V)

        log_probs = F.log_softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()  # (T',V)

        hyp = decoder_ctc.decode(log_probs)
        hyp = clean_hyp(hyp)

        ref = str(df_dev.loc[i, "Transcript"]).strip()
        wers.append(wer(ref, hyp))

    return float(sum(wers) / max(1, len(wers)))

best_dev_wer = 1e9

for epoch in range(NUM_EPOCH):
    model.train()
    epoch_losses = []

    print(f"\nEPOCH {epoch}\n" + "=" * 40)

    for L1, loader in train_loaders.items():
        edge_index = torch.tensor(all_edges[L1], dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(all_weights[L1], dtype=torch.float).to(device)
        model.set_graph(edge_index, edge_weight)

        for batch in tqdm(loader, desc=f"Train[{L1}]", leave=False):
            audio, canonical, transcript_flat, transcript_lengths = batch  # audio: (B,T)

            with torch.cuda.amp.autocast():
                logits = model(audio, canonical)  # (B, T', V)

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

    train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
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

