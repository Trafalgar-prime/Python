# PyTorch: addestramento e valutazione di una rete sul dataset Iris (spiegazione riga per riga)

Questa guida spiega **riga per riga** il codice PyTorch necessario per:

1. Convertire i dati (NumPy/Pandas) in **tensori** PyTorch.
2. Definire **loss** e **ottimizzatore**.
3. Eseguire il **training loop** (forward → loss → backward → update).
4. Valutare il modello in modalità **eval** (accuracy + confusion matrix).

> Contesto: hai già creato `X_train, X_test, y_train, y_test` con `train_test_split` e hai normalizzato `X_train/X_test` con `StandardScaler`.

---

## Codice completo (da incollare dopo `modello = ReteNeurale()`)

```python
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
```

---

## Spiegazione riga per riga

### 1) Conversione in tensori

```python
X_train_t = torch.tensor(X_train, dtype=torch.float32)
```
- `X_train` qui è un **numpy.ndarray** (uscito da `StandardScaler`).
- `torch.tensor(...)` crea un **Tensor** PyTorch con gli stessi valori.
- `dtype=torch.float32` è fondamentale:
  - le reti neurali lavorano tipicamente in **float32**;
  - se lasci `float64` (default NumPy) puoi avere più uso di memoria e, in certi casi, incompatibilità / lentezza.

```python
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
```
- Stessa cosa per il set di test.

```python
y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)
```
- `y_train` è una **Pandas Series** (perché viene da `df['target']`).
- `to_numpy()` la converte in un array NumPy “pulito”.
- `dtype=torch.long` è obbligatorio con `nn.CrossEntropyLoss()`:
  - i target devono essere **interi** che indicano la classe (0, 1, 2),
  - e PyTorch li vuole in formato `int64` (che in PyTorch è `long`).

```python
y_test_t  = torch.tensor(y_test.to_numpy(), dtype=torch.long)
```
- Stessa conversione per i target del test.

**Tip**: se vuoi verificare forme e tipi:
```python
print(X_train_t.shape, X_train_t.dtype)
print(y_train_t.shape, y_train_t.dtype)
```

---

### 2) Loss e ottimizzatore

```python
criterion = nn.CrossEntropyLoss()
```
- `CrossEntropyLoss` è la loss standard per **classificazione multiclasse**.
- Importantissimo:
  - vuole in input i **logits** (uscita lineare, senza softmax),
  - e applica internamente `log_softmax + negative log likelihood`.
- Il tuo modello restituisce `self.fc3(x)` senza softmax → perfetto.

```python
optimizer = optim.Adam(modello.parameters(), lr=0.01)
```
- `modello.parameters()` restituisce tutti i tensori “allenabili” (pesi e bias dei layer).
- `Adam` è un ottimizzatore robusto e spesso “funziona subito”.
- `lr=0.01` è il learning rate: controlla quanto grandi sono gli aggiornamenti dei pesi.
  - Se vedi instabilità, prova `lr=0.001`.
  - Se impara troppo lentamente, aumenta un po’.

---

### 3) Training loop (il cuore)

```python
epochs = 200
```
- Numero di passaggi completi sul training set.
- Iris è piccolo: 200 epoche vanno bene.

```python
for epoch in range(epochs):
```
- Ciclo che ripete il training per ogni epoca.

```python
modello.train()
```
- Imposta il modello in modalità **training**.
- Serve soprattutto se hai layer come `Dropout` o `BatchNorm`.
- Anche se qui non li hai, è buona pratica sempre farlo.

```python
logits = modello(X_train_t)  # forward
```
- **Forward pass**: passi gli input nel modello.
- `logits` ha forma `(n_campioni, 3)`.
  - Ogni riga contiene 3 numeri (uno per classe).
  - Non sono probabilità: sono **punteggi** (logits).

```python
loss = criterion(logits, y_train_t)
```
- Calcola quanto il modello “sbaglia” rispetto alle etichette vere.
- Output: un singolo numero (tensor scalare).

```python
optimizer.zero_grad()
```
- PyTorch **accumula** i gradienti per default.
- Quindi prima di fare backprop devi azzerarli, altrimenti sommi i gradienti di epoche precedenti.

```python
loss.backward()
```
- **Backpropagation**: calcola i gradienti (derivate) della loss rispetto a tutti i parametri del modello.
- Dopo questa riga, ogni parametro `p` avrà `p.grad` popolato.

```python
optimizer.step()
```
- Aggiorna i pesi usando i gradienti calcolati.
- In pratica: `parametro = parametro - lr * gradiente` (semplificando; Adam è più sofisticato).

#### Stampa ogni 20 epoche

```python
if (epoch + 1) % 20 == 0:
```
- Ogni 20 epoche stampa metriche per monitorare l’andamento.

```python
with torch.no_grad():
```
- Disattiva il calcolo dei gradienti:
  - più veloce,
  - meno memoria,
  - evita di “sporcare” il grafo dei gradienti.

```python
preds = logits.argmax(dim=1)
```
- `argmax(dim=1)` prende, per ogni riga, l’indice della classe con logit più alto.
- Output: tensor di forma `(n_campioni,)` con valori 0/1/2.

```python
acc = (preds == y_train_t).float().mean().item()
```
- `(preds == y_train_t)` produce un tensor booleano (`True/False`) riga per riga.
- `.float()` trasforma `True/False` in `1.0/0.0`.
- `.mean()` fa la media: percentuale corretta (accuracy).
- `.item()` estrae il valore Python (float) da un tensor scalare.

```python
print(f"...")
```
- Stampa loss e accuracy di training.

---

### 4) Valutazione sul test

```python
modello.eval()
```
- Modalità **evaluation**: disattiva comportamenti specifici del training (dropout, batchnorm, ecc.).
- Best practice: sempre prima di validazione/test.

```python
with torch.no_grad():
    logits_test = modello(X_test_t)
    y_pred_t = logits_test.argmax(dim=1)
```
- Forward sul test senza gradienti.
- `y_pred_t` sono le classi predette (0/1/2).

```python
test_acc = (y_pred_t == y_test_t).float().mean().item()
print("\nTest accuracy:", test_acc)
```
- Stesso schema dell’accuracy di training, ma sul test set.

---

### 5) Confusion matrix (sklearn)

```python
y_pred = y_pred_t.cpu().numpy()
```
- `sklearn` lavora bene con NumPy.
- `.cpu()` serve se un giorno usi GPU: porta i dati su CPU prima di convertirli in NumPy.
- `.numpy()` converte tensor → array.

```python
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)
```
- `confusion_matrix` confronta etichette vere `y_test` con predette `y_pred`.
- `cm[i, j]` = quanti esempi della classe vera `i` sono stati predetti come classe `j`.

---

## Perché `CrossEntropyLoss` e niente softmax nel modello?

- Il layer finale del tuo modello è `nn.Linear(8, 3)` e restituisce logits.
- `CrossEntropyLoss` vuole logits perché:
  - è numericamente più stabile (evita underflow/overflow),
  - fa internamente `log_softmax`.

Se tu mettessi `softmax` nel `forward`, spesso:
- perdi stabilità numerica,
- e fai lavoro duplicato.

---

## Miglioramenti consigliati (facoltativi)

1. **Mini-batch** con DataLoader (più “PyTorch style”).
2. **Metriche per epoca** anche sul test (non solo train).
3. **Seed** per riproducibilità (`torch.manual_seed(42)`).
4. **Early stopping** per fermarti quando non migliora.

Se vuoi, posso fornirti la versione con `DataLoader` e mini-batch, sempre con spiegazione riga per riga.
