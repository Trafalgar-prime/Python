# Lezione 16 — Appunti completi (pandas + MLPClassifier + valutazione)
*(Raccolta in ordine cronologico delle spiegazioni fornite in questa conversazione, pronta da mettere su GitHub.)*

---

## 1) Perché pandas stampa solo “parzialmente” il DataFrame (con `...`)

Quando fai:

```python
print(df)
```

e il DataFrame è “grande” (molte righe e/o colonne), **pandas applica un comportamento predefinito di visualizzazione**:

- mostra **le prime righe**
- mostra **le ultime righe**
- sostituisce la parte centrale con `...`
- stampa un riepilogo tipo: `[150 rows x 4 columns]`

Questo non è un bug: è fatto per evitare di “intasare” il terminale.

### 1.1) Come stampare *tutte* le righe e colonne (opzioni globali)

```python
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)
```

- `display.max_rows = None` → non limitare le righe visualizzate
- `display.max_columns = None` → non limitare le colonne visualizzate

⚠️ Attenzione: se il dataset è molto grande, l’output diventa enorme e scomodo.

### 1.2) Come vedere solo una parte (approccio consigliato)

```python
print(df.head(20))   # prime 20 righe
print(df.tail(20))   # ultime 20 righe
```

È l’approccio più comune perché in pratica raramente serve stampare “tutto” in ML.

### 1.3) Forzare la stampa completa senza cambiare opzioni globali

```python
print(df.to_string())
```

- Converte il DataFrame in stringa completa e la stampa.
- Utile per casi piccoli/medi, ma può essere pesante su dataset grandi.

---

## 2) Analisi riga-per-riga del blocco: split + scaling + MLP

### Codice (come nel tuo messaggio)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creiamo una rete neurale con un solo livello nascosto di 10 neuroni
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)  # Addestriamo la rete
```

---

### 2.1) Import: `train_test_split`

```python
from sklearn.model_selection import train_test_split
```

- Importa la funzione che divide i dati in **train** e **test**.
- Serve per valutare il modello su dati **mai visti** (test) invece che sugli stessi dati su cui è stato addestrato (train).
- Evita valutazioni “falsate” da overfitting.

---

### 2.2) Import: `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler
```

- `StandardScaler` standardizza ogni feature con:

  \[
  x' = \frac{x - \mu}{\sigma}
  \]

  dove:
  - \(\mu\) è la media della feature (calcolata sul train)
  - \(\sigma\) è la deviazione standard della feature (calcolata sul train)

- Obiettivo: portare le feature su scale confrontabili (media ~0, std ~1).
- Per reti neurali spesso migliora stabilità e velocità di addestramento.

---

### 2.3) Import: `MLPClassifier`

```python
from sklearn.neural_network import MLPClassifier
```

- Classificatore basato su rete neurale **Multi-Layer Perceptron** (feed-forward).
- Può avere uno o più layer nascosti, ottimizzato tramite backpropagation.

---

### 2.4) Import: `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler
```

- Alternativa a `StandardScaler`.
- Normalizza in un intervallo (default `[0, 1]`):

  \[
  x' = \frac{x - x_{min}}{x_{max} - x_{min}}
  \]

- Nel tuo codice è importato ma **non usato** perché commentato.

---

### 2.5) Import: `confusion_matrix`

```python
from sklearn.metrics import confusion_matrix
```

- Serve a costruire la **matrice di confusione** (veri vs predetti).
- Utile perché mostra **quali classi** vengono confuse.

---

## 3) Split train/test

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1],
    df['target'],
    test_size=0.2,
    random_state=42
)
```

### 3.1) Parte X: `df.iloc[:, :-1]`
- `.iloc` usa indicizzazione **posizionale** (per posizione di righe/colonne, non per nome).
- `:` = tutte le righe
- `:-1` = tutte le colonne **tranne l’ultima**
- Quindi questa espressione prende **le feature** (X), assumendo che l’ultima colonna sia (o non sia) il target; nel tuo caso, dall’output, sembra che l’ultima colonna sia proprio `target`, quindi è coerente.

### 3.2) Parte y: `df['target']`
- Seleziona la colonna etichetta (classi) come `Series`.
- Nel dataset Iris tipicamente `target` è in `{0,1,2}`.

### 3.3) Parametri chiave
- `test_size=0.2`: 20% dei campioni va in test
  - su 150 righe → ~30 test, ~120 train
- `random_state=42`: rende lo split **riproducibile** (stesso split a ogni esecuzione)

### 3.4) Output
Ritorna 4 oggetti:
- `X_train`, `X_test` (feature)
- `y_train`, `y_test` (etichette)

---

## 4) Scaling: StandardScaler vs MinMaxScaler

### 4.1) Creazione dello scaler

```python
scaler = StandardScaler()
# scaler = MinMaxScaler()
```

- Qui crei un oggetto scaler “vuoto”: non ha ancora calcolato statistiche.
- Se usi `StandardScaler`, lo scaler dovrà stimare:
  - media per feature (`mean_`)
  - deviazione standard per feature (`scale_`)
- La riga con `MinMaxScaler` è commentata: non viene eseguita.

---

### 4.2) Fit + transform sul training

```python
X_train = scaler.fit_transform(X_train)
```

Questa riga combina due operazioni:

1) **fit**: calcola statistiche sul training set (media/std oppure min/max)
2) **transform**: applica la trasformazione ai dati di training

Importante:
- dopo `fit_transform`, `X_train` diventa tipicamente un `numpy.ndarray` (non più DataFrame pandas).

---

### 4.3) Transform sul test (senza fit)

```python
X_test = scaler.transform(X_test)
```

- Applica al test **le stesse statistiche del training**.
- Questo evita il problema chiamato **data leakage**:
  - Se facessi `fit_transform` anche sul test, useresti informazioni del test per normalizzare → valutazione non “pulita”.

---

## 5) Modello: MLPClassifier

```python
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
```

### 5.1) Architettura: `hidden_layer_sizes=(10,)`
- È una tupla: ogni elemento rappresenta un layer nascosto.
- `(10,)` significa:
  - **1 layer nascosto**
  - **10 neuroni** in quel layer
- Esempi:
  - `(10,)` → 1 layer da 10
  - `(10, 10)` → 2 layer: 10 e 10
  - `(50, 20, 10)` → 3 layer: 50, 20, 10

Per Iris (4 feature, 3 classi), la struttura concettuale è:
- input: 4
- hidden: 10
- output: 3

### 5.2) Iterazioni: `max_iter=2000`
- Numero massimo di iterazioni dell’ottimizzatore.
- Se troppo basso, il modello può non convergere e scikit-learn spesso avvisa.
- 2000 è alto per essere sicuri su dataset piccoli.

### 5.3) Riproducibilità: `random_state=42`
- Controlla casualità (es. inizializzazione pesi).
- Utile per avere risultati replicabili.

---

## 6) Addestramento del modello

```python
mlp.fit(X_train, y_train)
```

- Addestra la rete neurale sui dati di training.
- Input:
  - `X_train`: shape `(n_train, n_features)` → Iris tipicamente `(120, 4)`
  - `y_train`: shape `(n_train,)` → tipicamente `(120,)`

### 6.1) Cosa succede internamente (alto livello ma concreto)
1) inizializzazione pesi e bias (numeri piccoli casuali)
2) forward pass: calcolo output
3) calcolo loss (per multiclasse tipicamente cross-entropy)
4) backpropagation: gradienti della loss rispetto ai parametri
5) aggiornamento pesi (optimizer; spesso Adam di default)
6) ripeti fino a convergenza o fino a `max_iter`

Dopo `fit`, l’oggetto `mlp` contiene parametri appresi e attributi (es. `coefs_`, `intercepts_`, `loss_`, `n_iter_`).

---

## 7) Analisi riga-per-riga del blocco: valutazione + predizioni + confusion matrix

### Codice (come nel tuo messaggio)

```python
# Valutiamo il modello
accuracy = mlp.score(X_test, y_test)
print(f"Accuratezza: {accuracy:.2f}")

# Predizioni sul test set
y_pred = mlp.predict(X_test)

# Creazione della confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

---

### 7.1) Accuracy con `.score`

```python
accuracy = mlp.score(X_test, y_test)
```

Per i classificatori scikit-learn, `.score(X, y)` restituisce l’**accuracy**:

\[
accuracy = \frac{\#corrette}{\#totali}
\]

Concettualmente equivale a:
1. `y_pred = mlp.predict(X_test)`
2. `accuracy = accuracy_score(y_test, y_pred)`

Output: un `float` tra 0 e 1.

---

### 7.2) Stampa con f-string

```python
print(f"Accuratezza: {accuracy:.2f}")
```

- `f"..."` consente interpolazione variabili.
- `{accuracy:.2f}`:
  - formatta `accuracy` come float con **2 decimali**
  - es. 1.0 → `1.00`, 0.9333 → `0.93`

---

### 7.3) Predizione delle classi

```python
y_pred = mlp.predict(X_test)
```

- Calcola la classe predetta per ogni campione del test.
- Output:
  - `y_pred` è un `numpy.ndarray` 1D
  - shape: `(n_test,)` → Iris tipicamente `(30,)`
  - contiene etichette (es. 0,1,2)

Nota: se vuoi le probabilità invece delle classi:

```python
proba = mlp.predict_proba(X_test)  # shape (n_test, n_classi)
```

---

### 7.4) Confusion matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

- Crea una matrice `cm` dove:
  - **righe = classi vere**
  - **colonne = classi predette**
  - `cm[i, j]` = quante volte la classe vera è `i` ma la predizione è `j`

Con Iris (3 classi) → matrice 3×3.

Esempio perfetto:

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

Interpretazione:
- 10 esempi veri della classe 0 predetti come 0
- 9 esempi veri della classe 1 predetti come 1
- 11 esempi veri della classe 2 predetti come 2
- tutti gli off-diagonal sono 0 → nessun errore

Esempio con errori:

```text
[[10  0  0]
 [ 0  8  1]
 [ 0  2  9]]
```

- 1 campione della classe 1 scambiato per 2
- 2 campioni della classe 2 scambiati per 1

---

### 7.5) Stampa della confusion matrix

```python
print("Confusion Matrix:")
print(cm)
```

- Prima riga: stampa una “etichetta” per rendere l’output leggibile.
- Seconda riga: stampa la matrice (array numpy).

---

## 8) Note operative importanti (riassunto pratico)

1) **Il truncation di pandas** (`...`) è normale: puoi usare `set_option`, `head/tail`, o `to_string()`.
2) **Fit dello scaler SOLO sul training**, poi `transform` sul test (per evitare data leakage).
3) `MLPClassifier(hidden_layer_sizes=(10,))` = 1 layer nascosto da 10 neuroni.
4) `.score` per classificazione = accuracy.
5) `confusion_matrix` ti dice esattamente **quali classi** si confondono, non solo “quanto”.

---

Fine file.
