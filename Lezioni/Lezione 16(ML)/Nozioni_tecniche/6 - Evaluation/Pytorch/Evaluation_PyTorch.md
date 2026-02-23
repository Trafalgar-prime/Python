# BLOCCO 4 — Valutazione dei Modelli con PyTorch (Ultra completo + Codice)

In PyTorch la valutazione non è “automatica”: devi costruire tu il metodo, con loop di training/validation, metriche e logging.
Questo documento contiene una guida completa **da progetto serio**, con codice riutilizzabile.

---

## 1) Regole d’oro (se le rompi, la valutazione è falsa)

1. **Train vs Eval mode**
   - training: `model.train()`
   - evaluation: `model.eval()`
   (Dropout e BatchNorm cambiano comportamento)

2. **Niente gradienti in valutazione**
   - usa `torch.no_grad()` in validation/test
   - più veloce e meno memoria

3. **Monitora loss + metriche**
   - loss = ottimizzazione
   - metriche = obiettivo reale

4. **Split e DataLoader corretti**
   - train loader con `shuffle=True`
   - val/test loader con `shuffle=False`
   - attenzione a leakage, duplicati, split per gruppi/tempo

---

## 2) Skeleton standard: train loop + eval loop separati

### 2.1 Utility: batch su device
```python
import torch

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device)
```

### 2.2 Training di 1 epoca
```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / n
```

### 2.3 Valutazione loss (validation/test)
```python
@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        bs = X.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / n
```

---

## 3) Classificazione binaria: predizioni + metriche + confusion

### 3.1 Raccolta predizioni (logits → sigmoid)
```python
import numpy as np
import torch

@torch.no_grad()
def collect_binary_preds(model, loader, device):
    model.eval()
    all_y = []
    all_proba = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X).squeeze(1)   # [B]
        proba = torch.sigmoid(logits)  # [0,1]

        all_y.append(y.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_proba = np.concatenate(all_proba)
    return y_true, y_proba
```

### 3.2 Metriche e confusion matrix (sklearn)
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

y_true, y_proba = collect_binary_preds(model, val_loader, device)

threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1       :", f1_score(y_true, y_pred))

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
```

---

## 4) ROC-AUC e PR-AUC (fondamentali se sbilanciato)

```python
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_true, y_proba)
pr_auc  = average_precision_score(y_true, y_proba)

print("ROC-AUC:", roc_auc)
print("PR-AUC :", pr_auc)
```

---

## 5) Soglia ottimale (es. massimizzare F1)

```python
import numpy as np
from sklearn.metrics import f1_score

thresholds = np.linspace(0.05, 0.95, 19)
best_t, best_f1 = None, -1

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_true, y_pred_t)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best threshold:", best_t, "Best F1:", best_f1)
```

---

## 6) Multiclasse: logits → softmax/argmax + metriche

### 6.1 Raccolta predizioni
```python
@torch.no_grad()
def collect_multiclass_preds(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    all_proba = []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)                     # [B, C]
        proba = torch.softmax(logits, dim=1)  # [B, C]
        pred = proba.argmax(dim=1)            # [B]

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )
```

### 6.2 Metriche
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_true, y_pred, y_proba = collect_multiclass_preds(model, val_loader, device)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
```

---

## 7) Regressione: MAE, RMSE, R² + residui

### 7.1 Raccolta predizioni
```python
@torch.no_grad()
def collect_regression_preds(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []

    for X, y in loader:
        X = X.to(device)
        pred = model(X).squeeze(1)
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred
```

### 7.2 Metriche
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true, y_pred = collect_regression_preds(model, val_loader, device)

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2   = r2_score(y_true, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

residuals = y_true - y_pred
print("Residual mean:", residuals.mean())
print("Residual std :", residuals.std())
```

---

## 8) Loss functions corrette (PyTorch)

### Binaria
- logits (senza sigmoid nel modello) + `BCEWithLogitsLoss`
```python
criterion = torch.nn.BCEWithLogitsLoss()
```

### Multiclasse
- logits [B,C] + target int [B] + `CrossEntropyLoss`
```python
criterion = torch.nn.CrossEntropyLoss()
```

### Regressione
```python
criterion = torch.nn.MSELoss()       # oppure L1Loss / SmoothL1Loss
```

---

## 9) Logging con TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp1")

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate_loss(model, val_loader, criterion, device)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

writer.close()
```

---

## 10) Early stopping + checkpoint (manuale)

```python
import torch

best_val = float("inf")
patience = 5
counter = 0

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate_loss(model, val_loader, criterion, device)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

model.load_state_dict(torch.load("best.pt"))
```

---

## 11) Calibration: Brier score (binaria)

```python
brier = np.mean((y_proba - y_true) ** 2)
print("Brier:", brier)
```

Per calibrazione vera: temperature scaling (post-hoc) o metodi sklearn.

---

## 12) Cross-validation in PyTorch

Non è built-in: si fa manualmente con KFold sugli indici.
In DL spesso è più pratico:
- hold-out + early stopping
- oppure ripetere training con più seed e fare media/std

---

## 13) Robustezza

Valuta anche:
- rumore, missing
- data shift (periodo diverso)
- sottogruppi
- out-of-distribution

---

## 14) Criteri pratici: “buon lavoro” in PyTorch

- train_loss e val_loss coerenti (no divergenze)
- best checkpoint scelto su validation
- metriche adeguate (F1/PR-AUC se sbilanciato)
- soglia ottimizzata su validation
- confusion matrix coerente con costi FP/FN
- logging, riproducibilità, robustezza

---

## Conclusione

PyTorch ti dà massima libertà:  
**se scrivi bene la valutazione, hai controllo totale e risultati affidabili.**
