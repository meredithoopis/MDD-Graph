import os
import gc
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from dataloader import (
    MDD_Dataset,
    collate_fn,
    PAD_ID,
    BLANK_ID,  
    VOCAB_SIZE,
)
from gcn_model import GCN_MDD
from create_graph import get_graph
from transformers import Wav2Vec2Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 5
batch_size = 32
lr = 5e-5

CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

gc.collect()


df_train = pd.read_csv("train.csv")
L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


all_edges, all_weights = get_graph(L1_LIST, threshold=0.01, alpha=0.7, topk=10)

loaders = {}

for L1 in L1_LIST:
    df_l1 = df_train[df_train["L1"].str.lower() == L1]
    if len(df_l1) == 0:
        print(f"No data for {L1}, skipping.")
        continue
    loaders[L1] = torch.utils.data.DataLoader(
        MDD_Dataset(df_l1),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-100h")
model = GCN_MDD(
    config,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
).to(device)


model.wav2vec2.feature_extractor._freeze_parameters()
ctc_loss = nn.CTCLoss(
    blank=BLANK_ID,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
best_loss = float("inf")


for epoch in range(num_epoch):
    model.train()
    epoch_losses = []

    print(f"EPOCH {epoch}")
    print(f"==============================")
    for L1, loader in loaders.items():
        edge_index = torch.tensor(
            all_edges[L1], dtype=torch.long
        ).t().contiguous().to(device)
        edge_weight = torch.tensor(
            all_weights[L1], dtype=torch.float
        ).to(device)
        model.set_graph(edge_index, edge_weight)

        for step, batch in enumerate(tqdm(loader, leave=False)):
            try:
                audio, canonical, transcript_flat, transcript_lengths = batch

                with torch.no_grad():
                    audio_lengths = torch.full(
                        (audio.size(0),),
                        audio.size(1),
                        dtype=torch.long,
                        device=device
                    )
                    input_lengths = model.wav2vec2._get_feat_extract_output_lengths(
                        audio_lengths
                    )

                with torch.cuda.amp.autocast():
                    audio_mask = torch.ones_like(audio, dtype=torch.long)
                    logits = model(audio, audio_mask, canonical)
                    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

                    loss = ctc_loss(
                        log_probs,
                        transcript_flat,
                        input_lengths,
                        transcript_lengths
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())

            except Exception as e:
                print(f"[ERROR][{L1}] step={step}: {e}")
                torch.cuda.empty_cache()
                continue

    if len(epoch_losses) == 0:
        print("No valid batches this epoch.")
        continue

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"\nEpoch {epoch} mean CTC loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            },
            ckpt_path
        )
        print(f"Saved checkpoint: {ckpt_path}")
