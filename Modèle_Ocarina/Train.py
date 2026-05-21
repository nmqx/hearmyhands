"""
Training script.

CRITICAL FIX vs original:
  -- The split is now built ONCE on a sorted file list, with a fixed
     random seed. The previous version created two separate Dataset
     instances and indexed them with the same permutation -- but each
     instance ran its own os.listdir(), so the indices did NOT line up
     reliably. This is the single biggest reason your val loss was 10x
     your train loss: leaky / scrambled split.

Other changes:
  -- Uses (frames, mask, label) batches.
  -- AdamW + cosine LR schedule + label smoothing.
  -- Class weights (in case some letters have fewer recordings).
  -- Saves ocarina_classes.json alongside the weights, so the inference
     script knows the class order.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Dataset import SignLanguageDataset, list_samples
from Ocarina_GRU import SignLanguageGRU


SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    BATCH_SIZE = 32
    MAX_FRAMES = 45
    EPOCHS = 200
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 40

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ---------------- BUILD SPLIT (once, deterministically) ----------------
    all_samples = list_samples(DATA_DIR)
    labels_present = sorted({l for _, l in all_samples})
    class_to_idx = {c: i for i, c in enumerate(labels_present)}
    print(f"Total samples: {len(all_samples)}")
    print(f"Classes ({len(labels_present)}): {labels_present}")

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(all_samples))
    cut = int(0.8 * len(indices))
    train_samples = [all_samples[i] for i in indices[:cut]]
    val_samples = [all_samples[i] for i in indices[cut:]]

    # Sanity check -- the train and val sample paths must not overlap.
    assert not (set(p for p, _ in train_samples) & set(p for p, _ in val_samples)), \
        "Train/val leakage!"

    train_ds = SignLanguageDataset(
        train_samples, class_to_idx, max_frames=MAX_FRAMES, augment=True
    )
    val_ds = SignLanguageDataset(
        val_samples, class_to_idx, max_frames=MAX_FRAMES, augment=False
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------- CLASS WEIGHTS (handle imbalance) ----------------
    counts = np.zeros(len(class_to_idx))
    for _, lbl in train_samples:
        counts[class_to_idx[lbl]] += 1
    counts = np.maximum(counts, 1)
    weights = torch.tensor(counts.sum() / (len(counts) * counts),
                           dtype=torch.float32, device=device)

    # ---------------- MODEL ----------------
    model = SignLanguageGRU(
        input_size=42, hidden_size=96, num_layers=2,
        num_classes=len(class_to_idx), dropout=0.3, bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------- TRAIN ----------------
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    save_path = os.path.join(BASE_DIR, "ocarina_gru_v2.pth")
    classes_path = os.path.join(BASE_DIR, "ocarina_classes.json")

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        total_loss = 0.0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # --- val ---
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                logits = model(x, m)
                v_loss += criterion(logits, y).item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = v_loss / max(len(val_loader), 1)
        val_acc = 100.0 * correct / max(total, 1)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            with open(classes_path, "w") as f:
                json.dump(list(class_to_idx.keys()), f)
            no_improve = 0
        else:
            no_improve += 1

        flag = " ★" if improved else ""
        print(f"E{epoch+1:03d} | tr {train_loss:.3f} | vl {val_loss:.3f} | "
              f"acc {val_acc:5.2f}% | lr {scheduler.get_last_lr()[0]:.5f}{flag}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop @ epoch {epoch+1}. Best acc = {best_acc:.2f}% "
                  f"(epoch {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best: {best_acc:.2f}% (epoch {best_epoch})")

    # Save history + split so diagnostics.py can reproduce everything later
    # without re-training. Useful when you want to redo plots, or compare runs.
    with open(os.path.join(BASE_DIR, "training_history.json"), "w") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "train_files": [p for p, _ in train_samples],
            "val_files":   [p for p, _ in val_samples],
            "class_to_idx": class_to_idx,
            "seed": SEED,
        }, f, indent=2)
    print("Saved training_history.json")

    # ---------------- DIAGNOSTICS ----------------
    plot_curves(history)
    plot_confusion(model, val_loader, device,
                   list(class_to_idx.keys()))


def plot_curves(h):
    e = range(1, len(h["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(e, h["train_loss"], label="Train")
    plt.plot(e, h["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(e, h["val_acc"]); plt.xlabel("Epoch"); plt.ylabel("Val acc (%)")
    plt.grid(); plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)


def plot_confusion(model, loader, device, classes):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            pred = model(x, m).argmax(1)
            P.extend(pred.cpu().numpy()); Y.extend(y.cpu().numpy())
    cm = confusion_matrix(Y, P, labels=list(range(len(classes))), normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    plt.figure(figsize=(10, 10))
    disp.plot(cmap="Blues", xticks_rotation=90, values_format=".2f")
    plt.title("Confusion matrix (val, normalized)")
    plt.savefig("confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()