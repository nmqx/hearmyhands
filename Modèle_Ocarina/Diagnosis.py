"""
Diagnostics standalone pour le modèle Ocarina.

Lance ça APRÈS un Train.py (qui produit ocarina_gru_v2.pth,
ocarina_classes.json et training_history.json).

Produit dans le dossier ./diagnostics/ :
  1. training_curves.png      -- loss train/val + accuracy + LR effectif
  2. confusion_matrix.png     -- matrice normalisée (val set)
  3. confusion_counts.png     -- matrice en nombres bruts (utile pour voir
                                  quelles classes ont peu d'échantillons)
  4. per_class_metrics.png    -- précision / rappel / F1 par lettre
  5. top_confusions.txt       -- les 15 paires (vrai -> prédit) les plus
                                  confondues, triées par fréquence
  6. class_distribution.png   -- nb d'échantillons par lettre dans train/val
  7. sequence_lengths.png     -- distribution des longueurs (avant padding)
                                  + position du seuil max_frames
  8. confidence_histogram.png -- distribution des confidences max sur
                                  les prédictions correctes vs incorrectes
                                  (utile pour calibrer un seuil "je sais pas")
  9. report.txt               -- résumé chiffré : accuracy globale, top-3
                                  accuracy, macro/weighted F1, par-classe,
                                  classes les plus problématiques

Utilisation :
    python diagnostics.py
    python diagnostics.py --weights path/to.pth --history path/to.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dataset import SignLanguageDataset, list_samples  # noqa: E402
from Ocarina_GRU import SignLanguageGRU                # noqa: E402


# ============================================================
def load_everything(args):
    """Recharge dataset, modèle, et historique."""
    with open(args.history) as f:
        hist = json.load(f)

    class_to_idx = hist["class_to_idx"]
    classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])

    # Reconstitue les splits depuis l'historique (chemins sauvegardés).
    # On parse le label depuis le nom de fichier comme dans list_samples.
    from Dataset import _extract_label
    def to_pairs(paths):
        return [(p, _extract_label(os.path.basename(p))) for p in paths]

    train_pairs = to_pairs(hist["train_files"])
    val_pairs   = to_pairs(hist["val_files"])

    train_ds = SignLanguageDataset(train_pairs, class_to_idx, augment=False)
    val_ds   = SignLanguageDataset(val_pairs,   class_to_idx, augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = SignLanguageGRU(
        input_size=42, hidden_size=96, num_layers=2,
        num_classes=len(classes), bidirectional=True,
    ).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return {
        "history": hist["history"],
        "best_epoch": hist["best_epoch"],
        "best_acc": hist["best_acc"],
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "val_loader": val_loader,
        "model": model,
        "device": device,
    }


# ============================================================
def run_inference(model, loader, device):
    """Retourne (y_true, y_pred, probs) pour tout le val set."""
    y_true, y_pred, all_probs = [], [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m = x.to(device), m.to(device)
            logits = model(x, m)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred.extend(probs.argmax(1).tolist())
            y_true.extend(y.tolist())
            all_probs.append(probs)
    return (np.array(y_true), np.array(y_pred), np.concatenate(all_probs))


# ============================================================
def plot_training_curves(history, best_epoch, out_dir):
    """Loss + accuracy + indication du best epoch."""
    n = len(history["train_loss"])
    e = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(e, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(e, history["val_loss"],   label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss train vs val")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(e, history["val_acc"], color="C2", linewidth=1.5)
    axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.6)
    best_acc = history["val_acc"][best_epoch - 1]
    axes[1].axhline(best_acc, color="green", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val accuracy (%)")
    axes[1].set_title(f"Val accuracy (best = {best_acc:.2f}% @ E{best_epoch})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


# ============================================================
def plot_confusion_matrices(y_true, y_pred, classes, out_dir):
    """Deux versions : normalisée (taux), et brute (nb d'échantillons)."""
    labels = list(range(len(classes)))

    # Normalized
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Matrice de confusion (normalisée par ligne)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # annotations seulement pour les cases significatives
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Raw counts (log scale, mieux pour voir les rares)
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 10))
    # +1 pour pouvoir log les zéros
    im = ax.imshow(cm_raw + 0.1, cmap="Blues", norm=LogNorm(vmin=0.1))
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Matrice de confusion (nombres bruts, échelle log)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_raw[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > cm_raw.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_counts.png"), dpi=150)
    plt.close()


# ============================================================
def plot_per_class_metrics(y_true, y_pred, classes, out_dir):
    """Précision / rappel / F1 par classe en barres groupées."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, p,  w, label="Précision", color="#4C72B0")
    ax.bar(x,     r,  w, label="Rappel",    color="#55A868")
    ax.bar(x + w, f1, w, label="F1",        color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Métriques par classe (val set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    # support en haut de chaque colonne
    for xi, s in zip(x, support):
        ax.text(xi, 1.02, f"n={s}", ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150)
    plt.close()


# ============================================================
def write_top_confusions(y_true, y_pred, classes, out_dir, k=15):
    """Top-k paires (vrai -> prédit) les plus fréquentes (hors diagonale)."""
    pairs = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(classes[t], classes[p])] += 1

    path = os.path.join(out_dir, "top_confusions.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Top {k} confusions (true -> predicted) sur le val set\n")
        f.write("=" * 50 + "\n")
        for (t, p), n in pairs.most_common(k):
            f.write(f"  {t} -> {p} : {n}\n")
        if not pairs:
            f.write("  Aucune erreur sur le val set !\n")


# ============================================================
def plot_class_distribution(train_pairs, val_pairs, classes, out_dir):
    """Combien d'échantillons par classe dans train/val ?"""
    tr = Counter(l for _, l in train_pairs)
    va = Counter(l for _, l in val_pairs)
    x = np.arange(len(classes))
    w = 0.4

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - w / 2, [tr.get(c, 0) for c in classes], w,
           label="Train", color="#4C72B0")
    ax.bar(x + w / 2, [va.get(c, 0) for c in classes], w,
           label="Val", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Nb d'échantillons")
    ax.set_title("Distribution des classes (déséquilibres = à surveiller)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()


# ============================================================
def plot_sequence_lengths(train_ds, val_ds, max_frames, out_dir):
    """Distribution des longueurs réelles AVANT padding/truncation.

    Si beaucoup de séquences dépassent max_frames, on coupe de l'info utile.
    Si presque toutes sont très courtes, on pad pour rien.
    """
    lengths = []
    for pairs, _ in [(train_ds.samples, "train"), (val_ds.samples, "val")]:
        for path, _ in pairs:
            with open(path) as f:
                data = json.load(f)
            ann_ids = {a["image_id"] for a in data.get("annotations", [])}
            n = sum(1 for img in data.get("images", [])
                    if img["id"] in ann_ids)
            lengths.append(n)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(max_frames, color="red", linestyle="--",
               label=f"max_frames = {max_frames}")
    ax.set_xlabel("Nb de frames avec annotation")
    ax.set_ylabel("Nb d'échantillons")
    pct_truncated = 100 * sum(1 for L in lengths if L > max_frames) / len(lengths)
    ax.set_title(f"Distribution des longueurs ({pct_truncated:.1f}% tronqués)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sequence_lengths.png"), dpi=150)
    plt.close()


# ============================================================
def plot_confidence_histogram(y_true, y_pred, probs, out_dir):
    """Distribution des confidences max séparée correct / incorrect.

    Lecture : si les distributions se chevauchent beaucoup, un seuil
    "rejet si confidence < X" ne marchera pas. Si elles sont bien
    séparées, on peut filtrer les prédictions douteuses en prod.
    """
    confs = probs.max(axis=1)
    correct = (y_true == y_pred)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(confs[correct],  bins=25, alpha=0.7, label="Correct",
            color="#55A868", edgecolor="white")
    ax.hist(confs[~correct], bins=25, alpha=0.7, label="Incorrect",
            color="#C44E52", edgecolor="white")
    ax.set_xlabel("Confidence max (softmax)")
    ax.set_ylabel("Nb de prédictions")
    ax.set_title("Calibration : confiance vs justesse")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_histogram.png"), dpi=150)
    plt.close()


# ============================================================
def write_report(y_true, y_pred, probs, classes, history, best_epoch, out_dir):
    """Résumé texte chiffré."""
    acc = (y_true == y_pred).mean()
    # top-3 acc : la vraie classe est-elle dans le top-3 ?
    top3 = np.argsort(-probs, axis=1)[:, :3]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

    report_txt = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )

    # par-classe accuracy pour repérer les classes pourries
    per_class_acc = {}
    for c_idx, c_name in enumerate(classes):
        mask = (y_true == c_idx)
        if mask.sum() > 0:
            per_class_acc[c_name] = (y_pred[mask] == c_idx).mean()
    sorted_worst = sorted(per_class_acc.items(), key=lambda kv: kv[1])

    path = os.path.join(out_dir, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAPPORT DE DIAGNOSTIC -- Ocarina GRU\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best epoch : {best_epoch}\n")
        f.write(f"Val accuracy (top-1) : {acc * 100:.2f}%\n")
        f.write(f"Val accuracy (top-3) : {top3_acc * 100:.2f}%\n")
        f.write(f"Mean max-confidence  : {probs.max(axis=1).mean():.3f}\n")
        f.write(f"Final train loss     : {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final val loss       : {history['val_loss'][-1]:.4f}\n")
        f.write(f"Min val loss         : {min(history['val_loss']):.4f}\n\n")

        f.write("Top 5 classes les plus faibles (rappel) :\n")
        for c, a in sorted_worst[:5]:
            f.write(f"  {c} : {a * 100:5.1f}%\n")
        f.write("\nTop 5 classes les plus fortes :\n")
        for c, a in sorted_worst[-5:][::-1]:
            f.write(f"  {c} : {a * 100:5.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Détails sklearn classification_report\n")
        f.write("=" * 60 + "\n")
        f.write(report_txt)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--weights", default=os.path.join(here, "ocarina_gru_v2.pth"))
    ap.add_argument("--history", default=os.path.join(here, "training_history.json"))
    ap.add_argument("--out",     default=os.path.join(here, "diagnostics"))
    ap.add_argument("--max-frames", type=int, default=45)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for path, name in [(args.weights, "weights"), (args.history, "history")]:
        if not os.path.exists(path):
            print(f"[error] {name} introuvable : {path}")
            print("  -> lance d'abord Train.py (avec la version qui sauve "
                  "training_history.json).")
            sys.exit(1)

    print("Chargement du modèle et reconstruction du split...")
    ctx = load_everything(args)

    print(f"Val set : {len(ctx['val_ds'])} échantillons")
    print("Inférence sur le val set...")
    y_true, y_pred, probs = run_inference(
        ctx["model"], ctx["val_loader"], ctx["device"]
    )

    print("Génération des graphiques...")
    plot_training_curves(ctx["history"], ctx["best_epoch"], args.out)
    plot_confusion_matrices(y_true, y_pred, ctx["classes"], args.out)
    plot_per_class_metrics(y_true, y_pred, ctx["classes"], args.out)
    write_top_confusions(y_true, y_pred, ctx["classes"], args.out)
    plot_class_distribution(ctx["train_pairs"], ctx["val_pairs"],
                            ctx["classes"], args.out)
    plot_sequence_lengths(ctx["train_ds"], ctx["val_ds"],
                          args.max_frames, args.out)
    plot_confidence_histogram(y_true, y_pred, probs, args.out)
    write_report(y_true, y_pred, probs, ctx["classes"], ctx["history"],
                 ctx["best_epoch"], args.out)

    print(f"\nTout est dans {args.out}/")
    print("  training_curves.png      -- loss + acc")
    print("  confusion_matrix.png     -- normalisée")
    print("  confusion_counts.png     -- comptes bruts")
    print("  per_class_metrics.png    -- P/R/F1 par lettre")
    print("  top_confusions.txt       -- erreurs les plus fréquentes")
    print("  class_distribution.png   -- équilibre des classes")
    print("  sequence_lengths.png     -- longueurs avant padding")
    print("  confidence_histogram.png -- calibration confiance")
    print("  report.txt               -- résumé chiffré")


if __name__ == "__main__":
    main()