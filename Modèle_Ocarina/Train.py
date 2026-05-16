import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Ocarina_GRU import SignLanguageGRU 
from Dataset import SignLanguageDataset
import matplotlib.pyplot as plt

def main():
    #hyperparamètres
    DATA_DIR = "./dataset/"
    BATCH_SIZE = 16
    MAX_FRAMES = 60
    NUM_CLASSES = 26
    EPOCHS = 300
    LEARNING_RATE = 0.001

    #gpu si possible
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Apprentissage sur : {device}")

    #prep et separation des données
    print("Chargement des données...")
    full_dataset = SignLanguageDataset(data_dir=DATA_DIR, max_frames=MAX_FRAMES)
    print(f"{len(full_dataset)} json trouvés")
    print("Classes trouvées :", full_dataset.classes)

    #on divise en 80% train 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Vidéos Train : {train_size} | Vidéos Val : {val_size}")

    #GRU init
    model = SignLanguageGRU(input_size=42, hidden_size=128, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    print("Début de l'entraînement...")
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []


    for epoch in range(EPOCHS):
        
        #train
        model.train() #active le dropout
        train_loss = 0.0
        
        for batch_inputs, batch_labels in train_loader:
            #envoie les datas sur le device
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        #validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad(): #pas de grad, gpu gain
            for batch_inputs, batch_labels in val_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                #accuracy calcul
                _, predicted_classes = torch.max(predictions, 1)
                correct_predictions += (predicted_classes == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_accuracy)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.3f}%")
    
    torch.save(model.state_dict(), "ocarina_gru_v1.pth")
    print("Modèle sauvegardé avec succès.")
    plot_training_curves(history_train_loss, history_val_loss, history_val_acc)
    
def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Génère et sauvegarde un graphique des courbes d'apprentissage."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # --- 1er Graphique : L'erreur (Loss) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss ')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss ')
    plt.title('Évolution de l\'Erreur ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- 2ème Graphique : La précision (Accuracy) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300) # Sauvegarde en haute qualité pour le PDF LaTeX
    print("\n Graphique sauvegardé sous 'training_curves.png'")
    plt.show() # Affiche la fenêtre à la fin



if __name__ == "__main__":
    main()