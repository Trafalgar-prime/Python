# Tabella completa: Output → Attivazione finale → Loss (Keras/TensorFlow)

Questa guida collega in modo coerente:
- **tipo di problema**
- **shape dell’output**
- **attivazione finale**
- **loss corrette** (incluse le varianti `from_logits=True`)

> Regola generale: l’ultimo layer deve produrre output coerenti con la loss.  
> Se **non** metti l’attivazione finale (quindi output = **logits**), allora la loss deve essere configurata con `from_logits=True`.

---

## 1) Tabella principale (classificazione + regressione + RL)

| Tipo problema | Output | Attivazione finale | Loss principali |
|---|---:|---|---|
| **Binaria** | 1 neurone | **Sigmoid** | `BinaryCrossentropy` |
| **Binaria (logits)** | 1 neurone | **Nessuna** | `BinaryCrossentropy(from_logits=True)` |
| **Multiclasse (label intere)** | K neuroni | **Softmax** | `SparseCategoricalCrossentropy` |
| **Multiclasse (label intere, logits)** | K neuroni | **Nessuna** | `SparseCategoricalCrossentropy(from_logits=True)` |
| **Multiclasse (one-hot)** | K neuroni | **Softmax** | `CategoricalCrossentropy` |
| **Multiclasse (one-hot, logits)** | K neuroni | **Nessuna** | `CategoricalCrossentropy(from_logits=True)` |
| **Multi-label** | K neuroni | **Sigmoid** | `BinaryCrossentropy` |
| **Multi-label (logits)** | K neuroni | **Nessuna** | `BinaryCrossentropy(from_logits=True)` |
| **Regressione libera (ℝ)** | 1 neurone | **Lineare (nessuna)** | `MeanSquaredError (MSE)`, `MeanAbsoluteError (MAE)`, `Huber` |
| **Regressione positiva (>0)** | 1 neurone | **Softplus** | `MSE`, `MAE`, `Huber` *(su target > 0)* |
| **Output limitato [-1,1]** | 1 neurone | **Tanh** | `MSE`, `MAE` |
| **Poisson regression (conteggi)** | 1 neurone | **Softplus / exp** | `Poisson` |
| **Quantile regression** | 1 neurone | **Lineare** | `Quantile loss` *(custom/pinball)* |
| **Policy RL (azioni discrete)** | K neuroni | **Softmax** | `CategoricalCrossentropy` *(in policy gradient si usa come -logprob pesata)* |
| **Policy RL (azioni continue)** | N neuroni | **Lineare / Tanh** | Loss custom (es. -logprob Gaussian) |

---

## 2) Tre regole mentali (che evitano il 90% degli errori)

### Regola 1 — Se l’output rappresenta **probabilità**
Usi una **cross-entropy** (o BCE):
- Binaria → `BinaryCrossentropy`
- Multiclasse → `CategoricalCrossentropy` / `SparseCategoricalCrossentropy`
- Multi-label → `BinaryCrossentropy` (una BCE per classe)

### Regola 2 — Se l’output rappresenta **quantità continue**
Usi una loss di **distanza**:
- `MSE` (penalizza molto gli errori grandi)
- `MAE` (più robusta agli outlier)
- `Huber` (compromesso tra MSE e MAE)

### Regola 3 — Se l’output deve rispettare un **vincolo**
Usi un’attivazione che impone il dominio:
- `Softplus` → output > 0
- `Sigmoid` → output in (0, 1)
- `Tanh` → output in (-1, 1)

---

## 3) Logits: cosa significa e perché esiste `from_logits=True`

**Logits** = output lineare prima dell’attivazione:
\[
z = Wx + b
\]

- Se metti `Dense(K, activation="softmax")`, il modello produce probabilità.
- Se metti `Dense(K)` (nessuna attivazione), il modello produce logits.

Molte loss in Keras possono ricevere logits e applicare internamente la trasformazione (softmax/sigmoid) **in modo numericamente stabile**:

- `CategoricalCrossentropy(from_logits=True)`
- `SparseCategoricalCrossentropy(from_logits=True)`
- `BinaryCrossentropy(from_logits=True)`

> Regola d’oro: **se l’ultimo layer NON ha attivazione**, allora `from_logits=True`.

---

## 4) Differenze chiave che spesso confondono

### Multiclasse vs Multi-label
- **Multiclasse**: una sola classe vera → **Softmax** + (Sparse)CategoricalCrossentropy  
- **Multi-label**: più classi vere contemporaneamente → **Sigmoid** (K neuroni) + BinaryCrossentropy

### One-hot vs label intere
- **One-hot** → `CategoricalCrossentropy`
- **Label intere (0..K-1)** → `SparseCategoricalCrossentropy`

---

## 5) Esempi di `compile()` corretti (Keras)

### A) Multiclasse one-hot (logits consigliato)
```python
import tensorflow as tf

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

### B) Multiclasse label intere (logits consigliato)
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

### C) Binaria (logits consigliato)
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

### D) Regressione
```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

## 6) “Che metriche uso?” (valutazione minima consigliata)

### Classificazione
- `accuracy`
- `precision`, `recall`, `f1` (spesso via sklearn)
- `confusion_matrix` (via sklearn)
- `AUC` (binaria e multilabel; attenzione a come la configuri)

### Regressione
- `MAE`, `MSE/RMSE`, `R2` (R2 spesso via sklearn)

> Nota: in Keras `model.evaluate(X, y_true)` restituisce loss + metriche definite in `compile`.  
> Se vuoi confusion matrix o f1, spesso conviene calcolarle a parte con sklearn usando `y_pred`.

---

## 7) Errori tipici (da evitare)

- **BCE con target 0..9** → concettualmente sbagliato (BCE assume target in {0,1})
- `Dense(K)` (logits) ma `from_logits=False` → la loss interpreta logits come probabilità → instabilità
- `Dense(K, softmax)` ma `from_logits=True` → doppia softmax interna → risultati incoerenti
- `Dense(1, sigmoid)` per problema a 10 classi → collasso (stai risolvendo un binario, non un multiclasse)

---

## Conclusione

La coppia **(output layer, attivazione)** definisce il tipo di output del modello.  
La **loss** deve essere scelta in modo coerente con:
- tipo di problema (binario/multiclasse/multilabel/regressione)
- formato delle label (intere vs one-hot)
- presenza o assenza di attivazione finale (probabilità vs logits)

Se vuoi, il prossimo step naturale è costruire un “decision tree” (albero di scelta) automatico:
dato `y` (shape e tipo), scegliere sempre `Dense(...)`, attivazione e loss corretti.
