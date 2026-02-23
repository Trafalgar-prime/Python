# Esercizi strutturati — Lezione 16 (ML/DL)  
**Target:** consolidare ML/DL su **scikit-learn / TensorFlow-Keras / PyTorch** con esercizi progressivi, fino ai mattoni per **AE → VAE**, **GAN**, **Transformer**.  
**Dataset consigliato (offline):** `sklearn.datasets.load_digits()` (8x8 immagini, 10 classi). Niente download.

> Regola: qui trovi **solo esercizi** (con tracce e vincoli).  
> Quando vuoi le **soluzioni complete**, scrivi: **“Dammi le soluzioni dell’esercizio X (e Y…)”**.

---

## Setup comune (per tutti gli esercizi)
### A. Carica dataset Digits (sempre uguale)
```python
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = digits.data          # shape: (n_samples, 64)  -> pixel flatten 8x8
y = digits.target        # shape: (n_samples,)
```

### B. Split (con stratificazione)
- usa `train_test_split(..., stratify=y, random_state=42, test_size=0.2)`

### C. Scaling
- **scikit-learn**: `StandardScaler()` o `MinMaxScaler()`
- **TF/PyTorch**: usa lo stesso scaling per confronto equo

---

# BLOCCO 1 — Esercizi “fondamentali” (output/loss/shape/metriche)

### E1 — Output + loss corretti (scelta *obbligata*)
**Obiettivo:** associare correttamente **output layer** e **loss** al task.  
Per ognuno dei casi sotto, indica:
1) shape output, 2) attivazione finale (se serve), 3) loss corretta, 4) formato `y`.

**Casi:**
- (a) regressione scalare (prezzo casa)  
- (b) classificazione binaria (spam vs ham)  
- (c) classificazione multiclasse con K=10 (digits) usando label intere  
- (d) multiclasse con K=10 usando one-hot

**Vincoli tecnici (devi rispettarli):**
- PyTorch: `CrossEntropyLoss` vuole **logits** e `y` come `LongTensor` (class indices).  
- Keras: `SparseCategoricalCrossentropy(from_logits=...)` dipende se metti softmax o no.

---

### E2 — Debug shape (errore tipico)
**Obiettivo:** individuare e correggere mismatch di shape.  
Ti do tre righe “sospette”; per ognuna:
- dimmi quale shape produce
- se è corretta o no per il caso multiclasse Digits (10 classi)
- come correggerla

**Snippet A (Keras)**
```python
Dense(1, activation="softmax")
```

**Snippet B (PyTorch)**
```python
nn.Linear(64, 1)  # poi CrossEntropyLoss
```

**Snippet C (sklearn)**
```python
MLPClassifier(hidden_layer_sizes=(32,16), activation="relu")
# e poi y_onehot (matrice Nx10)
```

---

### E3 — Metriche “complete” (non solo accuracy)
**Obiettivo:** misurare in modo serio.  
Per Digits multiclasse:
- calcola `accuracy`
- calcola `confusion_matrix`
- calcola `classification_report`
- calcola **macro-F1**
- calcola AUC **macro** con One-vs-Rest (se riesci)

**Vincoli:**
- devi produrre probabilità o score per AUC (ovr)  
- in scikit-learn: usa `predict_proba` se disponibile, altrimenti `decision_function`

---

# BLOCCO 2 — scikit-learn (pipeline, tuning, warning, threshold)

### E4 — Pipeline “anti-leakage”
**Obiettivo:** costruire una pipeline corretta (no leakage).  
Crea:
```python
Pipeline([
  ("scaler", StandardScaler()),
  ("clf", LogisticRegression(...))
])
```
e confrontala con la versione **senza pipeline** (scaler fit su tutto).  
**Domanda:** qual è l’errore concettuale nel caso senza pipeline?

**Opzioni da provare in LogisticRegression:**
- `solver` ∈ {`lbfgs`, `saga`}
- `C` ∈ {0.1, 1, 10}
- `penalty` coerente col solver (se sbagli deve fallire → voglio che vedi l’errore)

---

### E5 — GridSearchCV “serio”
**Obiettivo:** tuning riproducibile.  
Fai `GridSearchCV` su:
- `LogisticRegression`: `C`, `solver`, `penalty`
- scoring: `f1_macro`
- cv: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

**Output richiesto:**
- best params
- best score
- valutazione su test (con report + confusion matrix)

---

### E6 — ConvergenceWarning su MLPClassifier (riproduci e risolvi)
**Obiettivo:** capire la convergenza *davvero*.  
1) Allena un `MLPClassifier` con `max_iter=50` e osserva warning.  
2) Risolvi usando **almeno 3 strategie diverse** tra:
- scaling migliore
- aumentare `max_iter`
- cambiare `learning_rate_init`
- cambiare `solver` (`adam` vs `sgd` vs `lbfgs`)
- `early_stopping=True` + `validation_fraction`

**Output richiesto:**
- tabellina (anche stampa) con: strategia → accuracy test → note (warning sì/no)

---

### E7 — Thresholding (anche se Digits è multiclasse)
**Obiettivo:** capire threshold in pratica.  
Trasforma Digits in binario: “digit == 9” vs “non 9”.  
Allena LogisticRegression e:
- calcola ROC curve + AUC
- scegli threshold per ottenere **recall ≥ 0.90**
- stampa confusion matrix con quel threshold

---

# BLOCCO 3 — TensorFlow / Keras (compile/fit/callbacks/custom training)

> Usa Digits anche qui: X è (N,64).  
> Per CNN: reshape a (N,8,8,1).

### E8 — MLP Keras baseline (from_logits vs softmax)
**Obiettivo:** costruire MLP e scegliere correttamente loss.  
Crea due versioni equivalenti:
- (A) output `Dense(10, activation="softmax")` + loss `SparseCategoricalCrossentropy(from_logits=False)`
- (B) output `Dense(10)` (no softmax) + loss `SparseCategoricalCrossentropy(from_logits=True)`

**Output richiesto:**
- accuracy finale test per A e B
- dimmi perché B è numericamente “più pulita” (in generale)

---

### E9 — Callbacks (EarlyStopping + ReduceLROnPlateau)
**Obiettivo:** controllo training.  
Allena per `epochs=200`, ma con:
- `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)`

**Output richiesto:**
- numero epoche effettive
- miglior val_loss e epoca migliore
- accuracy test finale

---

### E10 — CNN “mini” su Digits (reshape + conv)
**Obiettivo:** capire conv + shape.  
- Reshape: `X.reshape(-1, 8, 8, 1)`
- CNN minima:
  - `Conv2D(32, 3, activation="relu", padding="same")`
  - `MaxPool2D()`
  - `Flatten()`
  - `Dense(10)` (logits)

**Vincoli:**
- usa `from_logits=True`
- prova almeno 2 batch_size (32 e 128)

---

### E11 — Custom training step (GradientTape)
**Obiettivo:** capire cosa fa Keras sotto al cofano.  
Implementa un loop manuale per 1 epoca:
- `with tf.GradientTape() as tape: ...`
- `grads = tape.gradient(loss, model.trainable_variables)`
- `optimizer.apply_gradients(zip(grads, model.trainable_variables))`

**Output richiesto:**
- loss media epoca manuale
- confronta con `model.fit` per 1 epoca (devono essere compatibili)

---

# BLOCCO 4 — PyTorch (nn.Module, DataLoader, training loop, device)

### E12 — Dataset + DataLoader (corretto dtype)
**Obiettivo:** costruire pipeline torch.  
Crea un `TensorDataset` o un Dataset custom che restituisce:
- `x`: `float32`
- `y`: `long` (class indices)

**Vincoli:**
- `CrossEntropyLoss` deve funzionare senza cast al volo nel loop.

---

### E13 — MLP PyTorch baseline (loop completo)
**Obiettivo:** training loop corretto.  
Architettura:
- `Linear(64, 128)` + `ReLU`
- `Linear(128, 10)` (logits)

Loop:
- `model.train()` / `model.eval()`
- `optimizer.zero_grad(set_to_none=True)`
- `loss.backward()`
- `optimizer.step()`
- eval con `torch.no_grad()`

**Output richiesto:**
- stampa loss ogni N batch
- accuracy test finale

---

### E14 — Perché CrossEntropyLoss NON vuole softmax?
**Obiettivo:** comprensione non negoziabile.  
Fai due versioni:
- (A) modello che restituisce logits (NO softmax) + `CrossEntropyLoss`
- (B) modello che applica softmax + `CrossEntropyLoss` (sbagliata)

**Output richiesto:**
- confronta training stability e accuracy
- spiega cosa succede matematicamente (a livello di log-softmax interno)

---

### E15 — Scheduler (StepLR o Cosine)
**Obiettivo:** imparare LR schedule.  
Usa `AdamW` e uno scheduler:
- `StepLR(step_size=10, gamma=0.5)` **oppure**
- `CosineAnnealingLR(T_max=epochs)`

**Output richiesto:**
- stampa LR a ogni epoca
- confronta accuracy con e senza scheduler (stesso seed)

---

### E16 — Checkpoint + resume
**Obiettivo:** salvare e riprendere.  
- salva `model.state_dict()` + `optimizer.state_dict()` + epoca
- riprendi da checkpoint e continua 5 epoche

**Output richiesto:**
- dimostra che l’epoca riparte correttamente
- dimostra che i pesi cambiano (loss continua a scendere o metriche migliorano)

---

# BLOCCO 5 — Ponte verso Generative Models (AE → VAE, GAN, Transformer)

> Qui non serve dataset enorme. Digits va bene per capire **meccanica**.

### E17 — Autoencoder (PyTorch o Keras)
**Obiettivo:** encoder/decoder e reconstruction.  
- encoder: 64 → 32 → latent_dim
- decoder: latent_dim → 32 → 64
- loss: MSE

**Output richiesto:**
- reconstruction loss su test
- visualizza (anche numericamente) un input e la ricostruzione (reshape 8x8)

---

### E18 — VAE “minimo” (reparam trick)
**Obiettivo:** implementare `mu`, `logvar`, `z = mu + eps*std`.  
- encoder produce `mu` e `logvar`
- `std = exp(0.5*logvar)`
- `eps ~ N(0,1)`
- decoder ricostruisce

**Loss:**
- recon (MSE o BCE)
- + KL: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`

**Output richiesto:**
- stampa separatamente `recon_loss` e `kl_loss`
- prova `beta` ∈ {0.5, 1, 4} e commenta trade-off

---

### E19 — GAN “toy” (DCGAN-style semplificato)
**Obiettivo:** 2 reti + training alternato.  
- Discriminatore: input 64 → ... → 1 logit
- Generatore: noise_dim → ... → 64

**Vincoli:**
- usa `BCEWithLogitsLoss`
- usa Adam con betas “GAN friendly” (es. (0.5, 0.999))

**Output richiesto:**
- loss D e loss G per epoca
- genera 5 campioni e reshape 8x8 (anche come numeri)

---

### E20 — Transformer block “shape-only” (senza dataset)
**Obiettivo:** padroneggiare le shape di attention.  
Costruisci un blocco che prende `x` shape `(B, T, d_model)` e restituisce stessa shape, con:
- self-attention (`MultiHeadAttention` in Keras o `nn.MultiheadAttention` in torch)
- residual + norm
- FFN (Dense/Linear) + residual + norm

**Output richiesto:**
- dimostra che input/output shape sono uguali
- aggiungi (facoltativo) una causal mask e mostra che non crasha

---

## Come chiedermi le soluzioni
Scrivi una di queste:
- **“Dammi le soluzioni di E4, E5”**
- **“Soluzione completa E13 in PyTorch (con codice)”**
- **“Correggimi il mio codice per E11”** (incolla il codice)

Ti rispondo con soluzione **completa**, super specifica e (se vuoi) file GitHub aggiornato.
