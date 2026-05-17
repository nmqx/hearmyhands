import torch
import torch.nn as nn

class SignLanguageGRU(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, num_layers=1, num_classes=26):
        """
        input_size: 21 points * 2 (x, y) = 42
        hidden_size: Taille de la mémoire interne du GRU (128 ou 256 c'est bien)
        num_layers: Nombre d'étages du GRU (2 permet de comprendre des mouvements complexes)
        num_classes: Le nombre de mots/lettres que tu veux prédire (ex: 26 lettres + 10 mots)
        """
        super(SignLanguageGRU, self).__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_size),
            nn.ReLU()
        )
        

        self.gru = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        x = self.embedding(x)
        out, hn = self.gru(x)
        last_frame_memory = out[:, -1, :]
        predictions = self.classifier(last_frame_memory)
        
        #renvoie la prédiction
        return predictions
    


