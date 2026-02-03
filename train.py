import os
import gc
import ast
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from dataloader import MDD_Dataset, collate_fn, build_stats_graph_from_df, PAD_ID, VOCAB_SIZE
from gcn_model import GCN_MDD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 5 
batch_size = 16

ERR_PAD_ID = 2     # error padding id
BETA = 0.3         # weight for diagnosis loss (paper uses beta for diagnosis)

gc.collect()

# Load data
df_train = pd.read_csv("train_canonical_error.csv")
df_dev = pd.read_csv("EN_MDD/dev.csv")

train_dataset = MDD_Dataset(df_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_id=PAD_ID, err_pad_id=ERR_PAD_ID),
)

# Build graph stats from train
edge_index, edge_weight = build_stats_graph_from_df(
    df_train,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
    top_k=5,              # 3,5,10
    add_self_loops=True,
    smoothing=0.01,
)


# Load model + set graph
model = GCN_MDD.from_pretrained("facebook/wav2vec2-base-100h", vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
model.set_graph(edge_index, edge_weight)
model = model.to(device)
model.indices = torch.arange(model.vocab_size, dtype=torch.long, device=device)

model.wav2vec2.feature_extractor._freeze_parameters()


# Losses
diag_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
pos_weight = torch.tensor([4.0], device=device)
bce_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.cuda.amp.GradScaler()
os.makedirs("checkpoint", exist_ok=True)
best_total_loss = 1e9

# Train
for epoch in range(num_epoch):
    model.train()
    running_total, running_diag, running_err = [], [], []

    print(f"\nEPOCH {epoch}:")
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        try:
            acoustic, audio_mask, canonical, transcript, error_gt, aligned_lengths = batch
            canonical = canonical.long().contiguous()
            transcript = transcript.long().contiguous()
            error_gt = error_gt.long().contiguous()

            with torch.cuda.amp.autocast():
                diag_logits, err_logits = model(acoustic, audio_mask, canonical)
                
                B, N, V = diag_logits.shape
                diag_loss_per_pos = torch.nn.functional.cross_entropy(
                    diag_logits.view(-1, V),
                    transcript.view(-1),
                    ignore_index=PAD_ID,
                    reduction="none"
                ).view(B, N)
                pos_ids = torch.arange(N, device=transcript.device)[None, :]
                diag_mask = (pos_ids < aligned_lengths[:, None]).float()
                diag_loss = (diag_loss_per_pos * diag_mask).sum() / (diag_mask.sum() + 1e-8)

                err_target = (error_gt == 0).float()
                pos_ids = torch.arange(error_gt.size(1), device=error_gt.device)[None, :]
                len_mask = (pos_ids < aligned_lengths[:, None]).float()

                err_mask = (error_gt != ERR_PAD_ID).float() * len_mask

                err_loss_per_pos = bce_criterion(err_logits, err_target)
                err_loss = (err_loss_per_pos * err_mask).sum() / (err_mask.sum() + 1e-8)
                
                # Total loss
                loss = err_loss + BETA * diag_loss
                
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_total.append(loss.item())
            running_diag.append(diag_loss.item())
            running_err.append(err_loss.item())
            
        except Exception as e:
            print(f"ERROR in batch {step}: {e}")
            torch.cuda.empty_cache()
            continue
        
        if step % 10 == 0:
            torch.cuda.empty_cache()

    if len(running_total) == 0:
        print("ERROR: No valid batches in this epoch!")
        continue

    avg_total = sum(running_total) / len(running_total)
    avg_diag = sum(running_diag) / len(running_diag)
    avg_err = sum(running_err) / len(running_err)

    print(f"Train total loss: {avg_total:.4f} | diag loss: {avg_diag:.4f} | err loss: {avg_err:.4f}")


    if avg_total < best_total_loss:
        best_total_loss = avg_total
        print("Saving checkpoint...")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "epoch": epoch,
                "best_total_loss": best_total_loss,
            },
            "checkpoint/gcn_mdd_stats_best.pth"
        )
        print(f"Saved: checkpoint/gcn_mdd_stats_best.pth (best_total_loss={best_total_loss:.4f})")

