# Lezione 16 (continuazione) — Keras `Sequential` per Iris: analisi estremamente approfondita
*(Questo si collega direttamente al codice precedente con `train_test_split`, scaling (`StandardScaler`/`MinMaxScaler`) e variabili `X_train, X_test, y_train, y_test`.)*

---

## 0) Contesto: perché questo codice “si lega” al precedente

Nel codice precedente hai fatto 3 cose chiave:

1. **Split**:
   - `X_train, X_test, y_train, y_test = train_test_split(...)`

2. **Scaling**:
   - `X_train = scaler.fit_transform(X_train)`
   - `X_test = scaler.transform(X_test)`

3. **Addestramento con scikit-learn**:
   - `MLPClassifier(...).fit(X_train, y_train)`

Qui stai facendo la *stessa idea* (rete neurale + classificazione) ma usando **Keras** (di solito tramite TensorFlow) invece di `MLPClassifier`.

Quindi:
- `MLPClassifier` = “rete neurale pronta” dentro scikit-learn
- `keras.Sequential` = rete neurale definita esplicitamente layer per layer (più controllo, più “deep learning style”)

---

## 1) Il codice

```python
modello = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # 16 neuroni
    keras.layers.Dense(8, activation='relu'),  # 8 neuroni
    keras.layers.Dense(3, activation='softmax')  # 3 classi in output
])

modello.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modello.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
```

Lo analizziamo riga per riga, includendo:
- forme (shape) dei tensori
- cosa significa ogni parametro
- cosa succede “sotto”
- collegamento diretto a Iris (4 feature, 3 classi)

---

## 2) `keras.Sequential([...])`: cosa significa davvero

### 2.1) Che cos’è `Sequential`
`keras.Sequential` crea un **modello a pipeline lineare**:
- l’output del layer 1 va nel layer 2
- l’output del layer 2 va nel layer 3
- ecc.

Non puoi fare “ramificazioni” o “skip connections” con `Sequential` (per quelle serve l’API funzionale), ma per Iris è perfetto.

### 2.2) Perché una lista di layer
Passi una lista:

```python
[
  Dense(...),
  Dense(...),
  Dense(...)
]
```

Keras costruisce l’architettura nell’ordine dato.

---

## 3) Primo layer: `Dense(16, relu, input_shape=(4,))`

```python
keras.layers.Dense(16, activation='relu', input_shape=(4,))
```

### 3.1) `Dense` = layer completamente connesso
Un layer Dense implementa:

\[
\mathbf{y} = f(\mathbf{x} \mathbf{W} + \mathbf{b})
\]

- **x**: input (feature)
- **W**: matrice dei pesi
- **b**: bias
- **f**: funzione di attivazione (qui ReLU)

### 3.2) `16` = numero di neuroni
Vuol dire:
- questo layer produce un vettore di dimensione 16 per ogni campione

Se il batch ha `N` esempi:
- input shape: `(N, 4)`
- output shape: `(N, 16)`

### 3.3) `input_shape=(4,)` e collegamento con Iris
Iris ha 4 feature:
- sepal length
- sepal width
- petal length
- petal width

Quindi ogni campione è un vettore di 4 numeri.

`input_shape=(4,)` dice a Keras:
- “mi aspetto vettori lunghi 4 come input”
- non include la dimensione batch (quella è implicita)

### 3.4) Pesi e bias: dimensioni ESATTE
- W1 ha shape `(4, 16)`
- b1 ha shape `(16,)`

Numero parametri:
- pesi: `4 * 16 = 64`
- bias: `16`
- totale: `80`

Keras infatti (se stampi `model.summary()`) ti mostrerà 80 params per questo layer.

### 3.5) `activation='relu'`: che cosa fa e perché si usa
ReLU = Rectified Linear Unit:

\[
\text{ReLU}(z) = \max(0, z)
\]

Effetti pratici:
- introduce **non linearità** (senza attivazioni la rete sarebbe solo una trasformazione lineare totale)
- aiuta con gradienti più stabili rispetto a sigmoid/tanh in molti casi
- “spegne” neuroni con output negativo (0)

Nota: ReLU può avere neuroni “morti” se restano sempre negativi, ma su Iris di solito non è un problema.

---

## 4) Secondo layer: `Dense(8, relu)`

```python
keras.layers.Dense(8, activation='relu')
```

### 4.1) Input implicito
Non metti `input_shape` perché Keras lo deduce:
- l’input di questo layer è l’output del precedente: `(N, 16)`

### 4.2) Dimensioni parametri
- W2 shape: `(16, 8)`
- b2 shape: `(8,)`

Numero parametri:
- pesi: `16 * 8 = 128`
- bias: `8`
- totale: `136`

### 4.3) Significato architetturale
Stai costruendo una rete “a imbuto”:
- 4 → 16 → 8 → 3

In pratica:
- il primo layer crea 16 combinazioni non lineari delle feature
- il secondo “riassume”/comprime in 8 feature interne più astratte

Su un dataset piccolo come Iris, è una rete già abbastanza potente.

---

## 5) Output layer: `Dense(3, softmax)`

```python
keras.layers.Dense(3, activation='softmax')
```

### 5.1) Perché 3 neuroni
Perché hai 3 classi:
- 0, 1, 2

Quindi l’output per ogni campione sarà un vettore lungo 3.

Shape:
- input: `(N, 8)`
- output: `(N, 3)`

### 5.2) Dimensioni parametri
- W3 shape: `(8, 3)`
- b3 shape: `(3,)`

Parametri:
- pesi: `8 * 3 = 24`
- bias: `3`
- totale: `27`

### 5.3) Softmax: cosa fa matematicamente
Softmax trasforma 3 “logit” (numeri reali) in una distribuzione di probabilità:

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{3} e^{z_j}}
\]

Proprietà:
- ogni valore è tra 0 e 1
- la somma dei 3 valori è 1
- il più grande corrisponde alla classe più probabile

Esempio output:
- `[0.98, 0.01, 0.01]` → quasi sicuramente classe 0

### 5.4) Predizione finale
Quando userai `modello.predict(X_test)` otterrai probabilità.
Per ottenere classi (0/1/2) di solito fai `argmax`:

```python
y_pred = modello.predict(X_test).argmax(axis=1)
```

---

## 6) Quanti parametri totali ha la rete?

Sommiamo:
- Layer1: 80
- Layer2: 136
- Layer3: 27

Totale: **243 parametri**

Per Iris (solo 150 esempi) è una rete piccola, quindi ok.

---

## 7) `compile`: optimizer, loss, metriche

```python
modello.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 7.1) Perché serve `compile`
In Keras, `compile` “prepara” il modello all’allenamento:

- sceglie **come aggiornare i pesi** → optimizer
- definisce **cosa minimizzare** → loss
- definisce **cosa monitorare** → metrics

Senza `compile`, non puoi fare `fit`.

---

### 7.2) `optimizer='adam'`
Adam = Adaptive Moment Estimation, uno degli optimizer più usati.

Idea (intuitiva ma concreta):
- è un gradiente discendente “evoluto”
- mantiene:
  - una media mobile dei gradienti (momento 1)
  - una media mobile dei gradienti al quadrato (momento 2)
- così adatta il learning rate per ogni parametro.

Per dataset piccoli e reti piccole, Adam spesso converge bene senza troppi tuning.

---

### 7.3) `loss='sparse_categorical_crossentropy'`
Questa è una scelta **fondamentale** e si collega direttamente alla forma di `y_train`.

- **categorical_crossentropy** si usa quando `y` è one-hot:
  - esempio: classe 2 → `[0, 0, 1]`
- **sparse_categorical_crossentropy** si usa quando `y` è un intero:
  - classe 2 → `2`

Nel tuo caso (Iris con target 0/1/2) è perfetto usare `sparse_categorical_crossentropy`.

Intuizione:
- misura quanto le probabilità predette (softmax) sono “allineate” con la classe vera.
- penalizza molto quando il modello assegna probabilità bassa alla classe corretta.

---

### 7.4) `metrics=['accuracy']`
Accuracy in Keras qui significa:
- percentuale di campioni in cui `argmax(pred)` coincide con `y`.

Attenzione concettuale:
- la loss è quella che **ottimizzi** davvero
- la metrica è ciò che **monitori** (puoi ottimizzare bene la loss anche se accuracy oscilla nelle prime epoche)

---

## 8) `fit`: addestramento vero e proprio

```python
modello.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
```

### 8.1) Che cosa passa in input
- `X_train`: array (idealmente float32/float64) shape `(n_train, 4)`
- `y_train`: array/Series shape `(n_train,)` con valori 0,1,2

Il fatto che tu abbia già scalato i dati prima (con StandardScaler/MinMaxScaler) è ottimo:
- le reti neurali lavorano meglio con input “ben scalati”
- riduci problemi di ottimizzazione

### 8.2) `epochs=100`: cos’è un’epoca
Un’**epoca** = un passaggio completo su tutto il training set.

Quindi con 100 epoche:
- il modello vede l’intero training set 100 volte.

Internamente, Keras usa anche i **batch** (dimensione predefinita spesso 32):
- se hai 120 esempi e batch=32, avrai circa 4 step per epoca.

### 8.3) `validation_data=(X_test, y_test)`
Qui stai usando il tuo **test set come validation**.

È importante capire la differenza:

- **Validation set**: usato durante il training per monitorare generalizzazione e tuning.
- **Test set**: dovrebbe essere usato solo alla fine per la valutazione finale.

Quello che fai qui è comune nei tutorial, ma in un progetto “pulito” faresti:
- train
- validation
- test (tenuto da parte)

Per un esercizio su Iris va bene, ma devi sapere la regola.

### 8.4) Cosa succede a ogni epoca (meccanica interna)
Per ogni batch:
1. forward pass (calcolo output)
2. calcolo loss (sparse crossentropy)
3. backprop (gradienti)
4. update pesi (Adam)

Alla fine di ogni epoca:
- calcola loss/accuracy su training
- calcola val_loss/val_accuracy su validation_data

Output tipico:
```
Epoch 1/100
... loss: ... accuracy: ... val_loss: ... val_accuracy: ...
```

### 8.5) Cosa aspettarti su Iris
Molto spesso:
- `accuracy` cresce rapidamente
- `val_accuracy` può arrivare molto alta (anche 1.0) con split fortunato
- 100 epoche possono essere “tante” per Iris: rischio di overfitting, ma con rete piccola spesso non è drammatico

---

## 9) Collegamento diretto con `MLPClassifier` di scikit-learn (confronto utile)

| Aspetto | `MLPClassifier` (sklearn) | Keras `Sequential` |
|---|---|---|
| Definizione rete | “implicita” tramite parametri | “esplicita” layer per layer |
| Training | `.fit()` senza compile | `compile()` + `fit()` |
| Output | `.predict()` dà classi | `predict()` dà probabilità (con softmax) |
| Loss | gestita internamente | scelta esplicita (`sparse_categorical_crossentropy`) |
| Controllo | più semplice, meno flessibile | più flessibile e standard DL |

Due cose che stai facendo bene in entrambi:
- split train/test
- scaling fit sul train e transform sul test

---

## 10) Miglioramenti pratici (coerenti con questo esercizio)

### 10.1) Stampare il riepilogo del modello (super utile)
```python
modello.summary()
```
Ti mostra:
- layer
- output shape di ogni layer
- parametri per layer
- totale parametri

### 10.2) Predire classi e stampare confusion matrix (come prima)
```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

proba = modello.predict(X_test)
y_pred = np.argmax(proba, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
```

Questo ti riallaccia esattamente alla parte precedente (accuracy + confusion matrix) ma in stile Keras.

---

Fine.
