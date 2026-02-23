# Lezione 17 — PyTorch “vero”: Dataset → DataLoader → Modello → Training loop → Valutazione → Salvataggio

## Obiettivo
Alla fine di questa lezione devi saper fare **da solo** (senza magia):
1) preparare dati in tensori  
2) creare un `Dataset` e un `DataLoader`  
3) scrivere un modello `nn.Module` (con `forward`)  
4) allenarlo con un training loop corretto  
5) valutarlo bene (`eval`, `no_grad`)  
6) salvare/caricare pesi  
7) capire **shape** e **dtype** (gli errori più comuni)

> Questo è lo “scheletro” che poi riusi identico per MLP, CNN, Transformers, VAE, GAN (cambia l’architettura e la loss, ma il loop è quello).

---

# 1) Le 6 righe mentali di PyTorch
In PyTorch, allenare un modello è sempre questa sequenza:

```python
model.train()
for Xb, yb in loader:
    optimizer.zero_grad()
    logits = model(Xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()
```

- `model.train()` = abilita modalità training (Dropout/BatchNorm si comportano “da training”)
- `zero_grad()` = azzera i gradienti (altrimenti si sommano)
- `model(Xb)` = forward
- `criterion(...)` = calcola la loss
- `backward()` = calcola gradienti (autograd)
- `step()` = aggiorna i pesi

---

# 2) Dati: dtypes e shapes (fondamentale)
### Per classificazione multiclasse (es. 10 classi)
- `X` deve essere **float32**
- `y` deve essere **interi** (class indices) di tipo **long** (`int64`)
- output del modello: **logits** shape `(batch, num_classi)`  
- loss: `nn.CrossEntropyLoss()` (vuole logits + target interi)

Esempio:
```python
Xb.shape == (batch, 64)      # se usi Digits (64 feature)
yb.shape == (batch,)         # etichette 0..9
logits.shape == (batch, 10)
```

---

# 3) Codice completo: Digits (sklearn) + PyTorch MLP (multiclasse 10 classi)
Questo esempio è perfetto perché **non scarichi niente**: il dataset viene da `sklearn.datasets`.

> Se non hai sklearn installato: `pip install scikit-learn`

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ========= 1) Dataset PyTorch =========
class DigitsDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array (N, 64)
        # y: numpy array (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # quanti esempi totali
        return len(self.X)

    def __getitem__(self, idx):
        # restituisce un singolo esempio (X_i, y_i)
        return self.X[idx], self.y[idx]


# ========= 2) Modello =========
class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden1=128, hidden2=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)  # logits, NO softmax qui
        )

    def forward(self, x):
        return self.net(x)


# ========= 3) Utility metriche =========
def accuracy_from_logits(logits, y_true):
    # logits: (B, C)
    # y_true: (B,)
    pred = logits.argmax(dim=1)          # classe con logit massimo
    acc = (pred == y_true).float().mean()
    return acc.item()


# ========= 4) Main =========
def main():
    # ---- Carico dataset ----
    digits = load_digits()
    X = digits.data           # (N, 64)
    y = digits.target         # (N,)

    # ---- Split train/test ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---- Scaling (fit SOLO su train, poi transform su test) ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ---- Dataset e DataLoader ----
    train_ds = DigitsDataset(X_train, y_train)
    test_ds  = DigitsDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    # ---- Device (CPU/GPU) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Modello, loss, ottimizzatore ----
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Training ----
    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, yb)
            n_batches += 1

        train_loss = total_loss / n_batches
        train_acc = total_acc / n_batches

        # ---- Evaluation ----
        model.eval()
        test_acc = 0.0
        test_batches = 0

        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                test_acc += accuracy_from_logits(logits, yb)
                test_batches += 1

        test_acc /= test_batches

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    # ---- Salvataggio pesi ----
    torch.save(model.state_dict(), "mlp_digits_state_dict.pth")
    print("Salvato: mlp_digits_state_dict.pth")

    # ---- Caricamento pesi (esempio) ----
    model2 = MLP().to(device)
    model2.load_state_dict(torch.load("mlp_digits_state_dict.pth", map_location=device))
    model2.eval()
    print("Caricato in model2 e messo in eval().")


if __name__ == "__main__":
    main()
```

---

# 4) Spiegazione “precisa ma semplice” dei punti che ti fregano sempre

### Perché **NON** metto `softmax` nel modello?
Perché `nn.CrossEntropyLoss()` vuole i **logits** (numeri grezzi) e internamente fa:
- `log_softmax` + `nll_loss`  
Quindi se ci metti softmax prima, peggiori numericamente e spesso sbagli.

✅ Giusto:
```python
nn.Linear(..., 10)   # logits
criterion = nn.CrossEntropyLoss()
```

---

### Perché `y` deve essere `torch.long`?
Perché CrossEntropyLoss vuole le classi come **indici** (0..9), non one-hot.

✅ Giusto:
```python
y = torch.tensor(y, dtype=torch.long)
```

---

### Perché `model.train()` e `model.eval()`?
Perché alcuni layer cambiano comportamento:
- Dropout: in train “spegne neuroni”, in eval no
- BatchNorm: in train aggiorna statistiche, in eval usa quelle salvate

✅ Pattern corretto:
```python
model.train()
...
model.eval()
with torch.no_grad():
    ...
```
---

# 5) Esercizi (li devi saper fare per dire “so PyTorch”)

### E1 — Cambia architettura
Aggiungi `Dropout(0.2)` tra i layer e guarda se test_acc cambia.
```python
nn.Linear(hidden1, hidden2),
nn.ReLU(),
nn.Dropout(0.2),
nn.Linear(hidden2, num_classes)
```

### E2 — Cambia ottimizzatore
Prova SGD con momentum:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```

### E3 — Stampa le shape (debug)
Dentro il loop, per un batch, stampa:
```python
print(Xb.shape, yb.shape, logits.shape)
```
Deve uscire (circa):
- `(64, 64) (64,) (64, 10)`

### E4 — Accuracy “vera” sul test (non media batch)
Invece di fare la media delle accuracy dei batch, conta corretti/totale:
```python
correct = 0
total = 0
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb.to(device))
        pred = logits.argmax(dim=1)
        correct += (pred == yb.to(device)).sum().item()
        total += yb.size(0)
print(correct/total)
```

---

## Nota finale (piano)
Dopo che provi gli esercizi, il passo successivo sarà **Lezione 17.5**:  
**Binary classification in PyTorch + BCEWithLogitsLoss + thresholding**, così integri perfettamente “soglia” e training loop DL.
