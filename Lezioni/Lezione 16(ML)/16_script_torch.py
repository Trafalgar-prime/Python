import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Importa il dataset Iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim


# Caricare un dataset (Esempio: Iris dataset)
dati = load_iris()  # Carica i dati di Iris
#Questa funzione carica in memoria il dataset Iris e lo restituisce come un oggetto “contenitore” di scikit-learn.

# Creiamo un DataFrame Pandas con i dati e aggiungiamo la colonna "target"
df = pd.DataFrame(dati.data, columns=dati.feature_names)
#~pd.set_option('display.max_rows', None) #serve per vedere tutti i dati e non solo i parziali in output, perche altrimenti pandas mi stampa solo i parziali per non intasare l'output
#pd.set_option('display.max_columns', None) #stesso del precedente ma per le colonne [in questo caso è inutile perche le colonne sono solo 4]
print(df)
print("\nProva\n")
print(df.shape)
print(df.columns)
print("\n")


df.index.name = "id" # la colonna degli indici la stampa pandas in automatico, in questo modo gli do un nome
df['target'] = dati.target


#Mostriamo le prime 5 righe del dataset
print(df.head())
print("\n",df,"\n")
print("\n")


# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
#.iloc permette di definire le posizioni di quello di cui ho bisogno in base alle colonne

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class ReteNeurale(nn.Module):
    def __init__(self):
        super(ReteNeurale, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

modello = ReteNeurale()
# tutto quello che ti serve sapere si trova su: LEZIONE16_pytorch_classe_spiegazione.md

# 1) Convertiamo in tensori PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test_t  = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 2) Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modello.parameters(), lr=0.01)

# 3) Training loop
epochs = 200
for epoch in range(epochs):
    modello.train()

    logits = modello(X_train_t)               # forward
    loss = criterion(logits, y_train_t)       # calcolo loss

    optimizer.zero_grad()                     # azzera gradienti
    loss.backward()                           # backprop
    optimizer.step()                          # aggiorna pesi

    # stampa ogni tot epoche
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y_train_t).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} - loss={loss.item():.4f} - train_acc={acc:.4f}")

# 4) Valutazione su test
modello.eval()
with torch.no_grad():
    logits_test = modello(X_test_t)
    y_pred_t = logits_test.argmax(dim=1)

test_acc = (y_pred_t == y_test_t).float().mean().item()
print("\nTest accuracy:", test_acc)

# 5) Confusion matrix (sklearn vuole numpy)
y_pred = y_pred_t.cpu().numpy()
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)
