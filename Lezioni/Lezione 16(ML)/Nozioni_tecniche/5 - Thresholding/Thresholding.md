# Thresholding (Soglia decisionale) — spiegato super semplice + codice

Questa guida spiega **thresholding** (soglia) in modo molto semplice, con **codice ogni volta** che appare un’opzione o un uso.

---

## 1) Cos’è il thresholding (in 1 frase)
Il modello ti dà un **numero** (quanto è sicuro). Tu scegli una **soglia**:

- se `numero >= soglia` → dici **SI (classe 1)**
- se `numero < soglia` → dici **NO (classe 0)**

Esempio:
- numero = `0.73`
- soglia = `0.50`
- `0.73 >= 0.50` → classe `1`

---

## 2) Funziona solo in sklearn?
**No.** È un’idea generale:

- **sklearn**: soglia su `predict_proba()` oppure `decision_function()`  
- **TensorFlow / PyTorch**: soglia su `sigmoid` (probabilità) oppure su logits (score)

**In questo file lavoriamo SOLO su sklearn**, ma sappi che la logica è identica altrove.

---

## 3) Dove si mette nel codice?
**Dopo** l’allenamento, quando hai le **probabilità** o gli **score**.

Schema base:
```python
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]   # numeri tra 0 e 1
pred = (prob >= soglia).astype(int)        # QUI è il thresholding
```

---

## 4) Come si applica (opzioni) — con codice

### Opzione A — soglia fissa a 0.5 (la più semplice)
```python
soglia = 0.5
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= soglia).astype(int)
```

### Opzione B — soglia più severa (es. 0.8)
Dici “SI” solo se sei molto sicuro.
```python
soglia = 0.8
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= soglia).astype(int)
```

### Opzione C — soglia più permissiva (es. 0.2)
Preferisci non perderti i veri “SI” (ma aumentano i falsi positivi).
```python
soglia = 0.2
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= soglia).astype(int)
```

---

## 5) Che numero uso per fare la soglia? (scelte reali)

### 5A) `predict_proba` (probabilità tra 0 e 1) ✅ consigliato
```python
prob = model.predict_proba(X)[:, 1]
pred = (prob >= 0.5).astype(int)
```

### 5B) `decision_function` (score non in 0..1)
```python
score = model.decision_function(X)
soglia = 0.0
pred = (score >= soglia).astype(int)
```

---

## 6) Come scelgo la soglia (modo “intelligente”)?
Esempio: **voglio recall ≥ 0.90**.  
Scegliamo la soglia su **validation**, non sul test.

### 6A) Split train/val/test (60/20/20)
```python
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
```

### 6B) Scegli soglia con `precision_recall_curve`
```python
import numpy as np
from sklearn.metrics import precision_recall_curve

prob_val = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, prob_val)

# precision/recall hanno 1 elemento in più di thresholds
precision = precision[1:]
recall = recall[1:]

mask = recall >= 0.90
best_threshold = thresholds[mask][np.argmax(precision[mask])]

print("Soglia scelta:", best_threshold)
```

### 6C) Applica la soglia scelta al test
```python
prob_test = model.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= best_threshold).astype(int)
```

---

## 7) Esempio COMPLETO (sklearn) — Digits: “9 vs non 9”
Qui fai tutto: dataset, train/val/test, training, scelta soglia, test finale.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

# 1) Dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2) Binario: 1 se è 9, 0 altrimenti
y_bin = (y == 9).astype(int)

# 3) Split train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_bin, test_size=0.2, stratify=y_bin, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# 4) Modello (pipeline = no leakage)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

# 5) Allenamento
model.fit(X_train, y_train)

# 6) Probabilità su validation
prob_val = model.predict_proba(X_val)[:, 1]

# 7) Scegli soglia: voglio recall >= 0.90
precision, recall, thresholds = precision_recall_curve(y_val, prob_val)

precision = precision[1:]
recall = recall[1:]

mask = recall >= 0.90
best_threshold = thresholds[mask][np.argmax(precision[mask])]

print("Soglia scelta:", best_threshold)

# 8) Applico la soglia al test
prob_test = model.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= best_threshold).astype(int)

# 9) Valuto
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred_test))
print("\nReport:\n", classification_report(y_test, pred_test, digits=4))
```

---

## 8) Riassunto “da bambino”
- Il modello ti dà un numero tipo `0.73`.
- Tu scegli una soglia tipo `0.5`.
- Se `0.73` è più grande → dici “SI”.
- Se è più piccolo → dici “NO”.
- La soglia la scegli su **validation**, non su test.

---

## Fine
Se vuoi, come passo successivo posso aggiungere (sempre sklearn):
- soglia per massimizzare F1
- soglia con ROC (Youden J)
- confronto `predict_proba` vs `decision_function`
