# Pre-elaborazione dei Dati in Ingresso — BLOCCO 4 (PyTorch)

Questo documento descrive **in modo completo e pratico** come gestire la pre-elaborazione dei dati in PyTorch.
In PyTorch il preprocessing è esplicito: sei tu a costruire Dataset, transforms e DataLoader.
Questo è un vantaggio enorme, ma richiede metodo.

---

## 1) Dove vive il preprocessing in PyTorch

Il preprocessing può vivere in tre punti:

1. **Nel Dataset** (`__getitem__`)
2. **Nei transforms** (torchvision o custom)
3. **Dentro il modello** (Embedding, BatchNorm, ecc.)

Regola pratica:
- pulizia e trasformazioni deterministiche → Dataset / transforms
- data augmentation → SOLO train
- statistiche (mean/std, vocab) → calcolate SOLO su train
- preprocessing “learned” → nel modello

---

## 2) Struttura base: Dataset e DataLoader

### Dataset custom
```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y
```

### DataLoader
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)

val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)
```

---

## 3) Scaling numeriche (StandardScaler manuale)

### Calcolo statistiche SOLO su train
```python
import numpy as np
import torch

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

mean_t = torch.tensor(mean, dtype=torch.float32)
std_t  = torch.tensor(std, dtype=torch.float32)
```

### Transform di normalizzazione
```python
class NormalizeTabular:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std
```

---

## 4) Gestione dei missing values

```python
class FillNaN:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, x):
        x = x.clone()
        x[torch.isnan(x)] = self.value
        return x
```

Composizione transforms:
```python
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
```

---

## 5) Feature categoriche

### One-hot (bassa cardinalità)
```python
import torch.nn.functional as F

def one_hot(idx, num_classes):
    return F.one_hot(idx, num_classes=num_classes).float()
```

### Embedding (alta cardinalità, consigliato)
- costruisci vocabolario SOLO su train
- riserva un token OOV

```python
def build_vocab(categories):
    uniq = sorted(set(categories))
    return {c:i+1 for i,c in enumerate(uniq)}  # 0 = OOV
```

Nel modello:
```python
import torch.nn as nn
emb = nn.Embedding(num_embeddings=len(vocab)+1, embedding_dim=16)
```

---

## 6) Testo (NLP) in PyTorch

### Tokenizzazione semplice
```python
def tokenize(text):
    return text.lower().split()
```

### Vocabolario (solo train)
```python
from collections import Counter

def build_text_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab
```

### Numericalizzazione
```python
def numericalize(text, vocab):
    return [vocab.get(w, vocab["<unk>"]) for w in tokenize(text)]
```

### collate_fn con padding
```python
def collate_text_batch(batch, pad_id=0):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])

    max_len = max(lengths)
    x_pad = torch.full((len(xs), max_len), pad_id)

    for i, x in enumerate(xs):
        x_pad[i, :len(x)] = torch.tensor(x)

    y = torch.tensor(ys)
    return x_pad, lengths, y
```

---

## 7) Immagini con torchvision

### Train (con augmentation)
```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Val/Test (no augmentation)
```python
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 8) Time series: windowing

```python
class WindowedSeries(Dataset):
    def __init__(self, series, window=30, horizon=1):
        self.series = series
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.window - self.horizon + 1

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.window]
        y = self.series[idx + self.window + self.horizon - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

---

## 9) Anti-leakage e riproducibilità

- statistiche calcolate SOLO su train
- vocabolari costruiti SOLO su train
- salva mean/std e vocab per inference
- seed fisso

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 10) Checklist finale

- transforms separati per train/val
- augmentation solo train
- mean/std e vocab solo train
- collate_fn per sequenze variabili
- pipeline riproducibile e serializzabile

---

## Conclusione

In PyTorch il preprocessing è **esplicito e controllabile**.
Se progettato bene:
- eviti leakage
- migliori generalizzazione
- hai pipeline solide e professionali
