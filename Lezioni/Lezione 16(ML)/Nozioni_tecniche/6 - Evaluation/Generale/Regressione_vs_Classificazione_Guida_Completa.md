# Regressione vs Classificazione -- Guida Completa

## 1. Tipi di Problema

### 🔵 Regressione

Obiettivo: prevedere un numero continuo.

Esempi: - Prezzo casa - Temperatura - Altezza

Target (y): Numeri reali → \[12.3, 8.7, 15.2\]

------------------------------------------------------------------------

### 🔴 Classificazione

Obiettivo: prevedere una classe.

Esempi: - Gatto / Cane - Cifre 0--9 - Spam / Non spam

Target (y): Classi intere → \[0, 3, 7, 1, 9\]

------------------------------------------------------------------------

## 2. Struttura dei Dati

### Regressione

X.shape = (N, features)\
y.shape = (N,)

### Classificazione multiclasse (sparse)

X.shape = (N, features)\
y.shape = (N,)

### Classificazione multiclasse (one-hot)

y.shape = (N, K)

------------------------------------------------------------------------

## 3. Struttura del Modello

### 🔵 Regressione

Output layer: Dense(1, activation='linear')

Output shape: (N, 1)

------------------------------------------------------------------------

### 🔴 Classificazione multiclasse (K classi)

Opzione 1: Dense(K, activation='softmax')

Opzione 2: Dense(K, activation='linear')
loss=SparseCategoricalCrossentropy(from_logits=True)

Output shape: (N, K)

------------------------------------------------------------------------

## 4. Loss Function

### Regressione

Mean Squared Error (MSE) Minimizza distanza numerica tra valore vero e
predetto.

MSE = (y - y_hat)\^2

------------------------------------------------------------------------

### Classificazione

CrossEntropy Massimizza probabilità della classe corretta.

Non misura distanza numerica. Misura qualità probabilistica.

------------------------------------------------------------------------

## 5. Output del Modello

### Regressione

Predizione: \[15.3, 9.2, 8.7\]

Un numero per campione.

------------------------------------------------------------------------

### Classificazione

Predizione: \[\[2.3, 0.1, -1.2, 4.5, ...\], \[0.2, 3.8, 1.1, -0.3,
...\]\]

Una matrice (N, K).

------------------------------------------------------------------------

## 6. Evaluation

### 🔵 Regressione

Metriche corrette: - MAE - MSE - RMSE - R²

Perché? Si confrontano numeri continui con numeri continui.

------------------------------------------------------------------------

### 🔴 Classificazione

Prima conversione: y_pred = np.argmax(logits, axis=1)

Poi metriche: - Accuracy - Precision - Recall - F1-score - Confusion
Matrix

Perché? Si confrontano classi con classi.

------------------------------------------------------------------------

## 7. Errori Comuni

❌ Usare MAE su output multiclasse (N, K)\
❌ Valutare su dati di training\
❌ Fare scaling prima dello split (data leakage)

------------------------------------------------------------------------

## 8. Regola Definitiva

  Tipo Problema   Output Layer         Loss                 Metriche
  --------------- -------------------- -------------------- ---------------
  Regressione     Dense(1)             MSE                  MAE, RMSE, R²
  Binaria         Dense(1) + sigmoid   BinaryCrossentropy   Accuracy, F1
  Multiclasse     Dense(K)             CrossEntropy         Accuracy, F1

------------------------------------------------------------------------

## 9. Concetto Fondamentale

Le metriche devono essere coerenti con: - struttura del target -
dimensione dell'output - tipo di loss - natura matematica del problema

Se non sono coerenti → risultati senza senso.
