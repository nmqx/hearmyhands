import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Ocarina_GRU import SignLanguageGRU
from Dataset import SignLanguageDataset


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    BATCH_SIZE = 32
    MAX_FRAMES = 50
    NUM_CLASSES = 26
    EPOCHS = 500
    LEARNING_RATE = 0.0015


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device utilisé : {device}")

    print("Chargement des données...")

    base_dataset = SignLanguageDataset(
        data_dir=DATA_DIR,
        max_frames=MAX_FRAMES,
        augment=False
    )

    print(f"{len(base_dataset)} samples trouvés")
    print("Classes :", base_dataset.classes)

    indices = np.random.permutation(len(base_dataset))
    train_size = int(0.8 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Train = augmentation
    train_dataset = SignLanguageDataset(
        data_dir=DATA_DIR,
        max_frames=MAX_FRAMES,
        augment=True
    )

    # Val = pas d'augmentation
    val_dataset = SignLanguageDataset(
        data_dir=DATA_DIR,
        max_frames=MAX_FRAMES,
        augment=False
    )

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train : {len(train_dataset)} | Val : {len(val_dataset)}")

    # =========================
    # MODEL
    # =========================
    model = SignLanguageGRU(
        input_size=42,
        hidden_size=64,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================
    # TRAIN LOOP
    # =========================
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    print("Début entraînement...\n")

    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    # =========================
    # MATRICE DE CONFUSION
    # =========================
    print("\nCalcul matrice de confusion...")
    compute_confusion_matrix(model, val_loader, device, base_dataset.classes)

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "ocarina_gru_v1.pth")
    print("Modèle sauvegardé")

    # =========================
    # COURBES
    # =========================
    plot_training_curves(
        history_train_loss,
        history_val_loss,
        history_val_acc
    )


# =========================
# CONFUSION MATRIX
# =========================
def compute_confusion_matrix(model, dataloader, device, classes):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    plt.figure(figsize=(10, 10))
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title("Matrice de confusion (normalisée)")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


def plot_training_curves(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # LOSS
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid()

    # ACCURACY
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()