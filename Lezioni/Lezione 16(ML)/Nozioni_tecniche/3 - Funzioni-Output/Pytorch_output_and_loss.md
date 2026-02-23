# PyTorch: Output Layer e Loss per Tipo di Task

## Tabella di riferimento rapido

  -----------------------------------------------------------------------
  Tipo di Task            Descrizione             Esempio fisico
  ----------------------- ----------------------- -----------------------
  **Regressione 📈**      Prevedere un valore     Calcolare la
                          numerico continuo.      temperatura di una
                                                  stella.

  **Classificazione       Scegliere tra due       Decidere se un segnale
  Binaria ⚖️**            opzioni mutualmente     è "Rumore" o "Evento".
                          esclusive.              

  **Classificazione       Scegliere **una sola**  Identificare un tipo di
  Multiclasse 🏷️**        opzione tra 3 o più     particella tra 5
                          categorie.              possibili.

  **Classificazione       Assegnare **più         Un materiale può essere
  Multilabel 📑**         etichette               "Magnetico" **e**
                          contemporaneamente**.   "Conduttore".
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 1) Regressione (valori continui)

Qui non classifichiamo: stimiamo una grandezza numerica.

-   **Ultimo layer**: `nn.Linear(n, 1)` (un solo neurone in uscita)
-   **Attivazione**: nessuna (identità) oppure **ReLU** se il valore
    deve essere solo positivo
-   **Loss**:
    -   `nn.MSELoss()` (Mean Squared Error)
    -   `nn.L1Loss()` (Mean Absolute Error)

------------------------------------------------------------------------

# 2) Classificazione binaria

-   **Ultimo layer**: `nn.Linear(n, 1)`
-   **Attivazione**: `torch.sigmoid()` (schiaccia l'output tra 0 e 1 →
    interpretazione probabilistica)
-   **Loss**: `nn.BCELoss()` (Binary Cross Entropy)

------------------------------------------------------------------------

# 3) Classificazione multiclasse (single-label)

Il modello deve scegliere **una sola classe** tra **num_classi**
possibili.

Qui entra in gioco il concetto di: - **One-Hot Encoding** (target come
vettore con un solo "1" e il resto "0") oppure - **indici interi**
(target come intero 0..num_classi-1)

-   **Ultimo layer**: `nn.Linear(n, num_classi)`
-   **Attivazione**: `torch.softmax()` per ottenere una distribuzione di
    probabilità (somma = 1)
-   **Loss**: `nn.CrossEntropyLoss()`

### Nota tecnica (importantissima)

In PyTorch, `nn.CrossEntropyLoss()` **include già** internamente
`LogSoftmax` + `NLLLoss`.\
Quindi l'ultimo layer del modello dovrebbe essere **lineare** (senza
softmax nel `forward`).

------------------------------------------------------------------------

# 4) Classificazione multilabel

Ogni etichetta è trattata come un problema binario indipendente (Sì/No
per ciascuna etichetta).

-   **Ultimo layer**: `nn.Linear(n, num_labels)`
-   **Attivazione**: `torch.sigmoid()` applicata a ogni neurone in
    uscita
-   **Loss**: `nn.BCEWithLogitsLoss()`

### Nota tecnica

`BCEWithLogitsLoss()` integra internamente la sigmoid in modo
numericamente stabile.\
In questo caso, l'ultimo layer del modello dovrebbe rimanere **lineare**
(logits).

------------------------------------------------------------------------

# Un dubbio sulla Multiclasse 🧠

Nella classificazione multiclasse (es. distinguere tra 3 tipi di
galassie), il modello deve fornire **un'unica risposta**:\
cioè **una sola classe** scelta tra tutte quelle possibili.

In PyTorch, la scelta dell'ultimo layer e della funzione di loss dipende
direttamente dal tipo di problema che stiamo affrontando. Di seguito una
guida strutturata per i quattro casi principali.

------------------------------------------------------------------------

## 1️⃣ Modello per Regressione (Valori Continui) 📈

Nel caso della regressione vogliamo prevedere un valore numerico
continuo (es. la massa di un pianeta, la temperatura di una stella, una
coordinata spaziale).

### 🔹 Output Layer

nn.Linear(hidden_size, 1)

Un solo neurone in uscita perché dobbiamo prevedere un singolo valore
scalare.

### 🔹 Attivazione Finale

Nessuna attivazione (identità) → caso standard.

nn.ReLU() → solo se sappiamo che il valore previsto deve essere
necessariamente positivo.

### 🔹 Loss

nn.MSELoss()

oppure

nn.L1Loss()

MSELoss penalizza maggiormente errori grandi.

L1Loss (MAE) è più robusta agli outlier.

------------------------------------------------------------------------

## 2️⃣ Modello per Classificazione Binaria ⚖️

Qui dobbiamo scegliere tra due opzioni mutualmente esclusive (es.
"Segnale" vs "Rumore").

### 🔹 Output Layer

nn.Linear(hidden_size, 1)

Un solo neurone che rappresenta la probabilità della classe positiva.

### 🔹 Attivazione Finale

nn.Sigmoid()

La Sigmoid comprime l'output tra 0 e 1, permettendo un'interpretazione
probabilistica.

### 🔹 Loss

nn.BCELoss()

⚠ Nota importante: In alternativa (e più stabile numericamente) si può
usare:

nn.BCEWithLogitsLoss()

In questo caso non bisogna applicare la Sigmoid nel modello, perché la
loss la integra internamente.

------------------------------------------------------------------------

## 3️⃣ Modello per Classificazione Multiclasse 🏷️

Dobbiamo scegliere una sola categoria tra $N$ possibili (es. "Protone",
"Neutrone", "Elettrone").

### 🔹 Output Layer

nn.Linear(hidden_size, num_classi)

Il numero di neuroni in uscita è pari al numero di classi.

### 🔹 Attivazione Finale

Nessuna attivazione nel modello (se si usa la loss standard di PyTorch).

### 🔹 Loss

nn.CrossEntropyLoss()

⚠ Nota Tecnica Fondamentale

nn.CrossEntropyLoss() combina internamente:

LogSoftmax

NLLLoss

Per questo motivo:

Il modello deve restituire logits grezzi (senza Softmax).

Non bisogna applicare Softmax nel forward().

La loss si occupa automaticamente della normalizzazione probabilistica.

------------------------------------------------------------------------

## 4️⃣ Modello per Classificazione Multilabel 📑

In questo scenario un campione può appartenere a più etichette
contemporaneamente (es. una stella può contenere più elementi chimici).

Ogni etichetta è trattata come un problema binario indipendente.

### 🔹 Output Layer

nn.Linear(hidden_size, num_labels)

Un neurone per ogni etichetta.

### 🔹 Attivazione Finale

nn.Sigmoid()

Applicata separatamente a ogni neurone in uscita.

### 🔹 Loss

Opzione consigliata:

nn.BCEWithLogitsLoss()

Oppure:

nn.BCELoss()

(se la Sigmoid è già stata applicata nel modello)

⚠ Nota tecnica

BCEWithLogitsLoss() è preferibile perché:

-   integra internamente la Sigmoid
-   è più stabile numericamente
-   evita problemi di saturazione

------------------------------------------------------------------------

## 📌 Riassunto Concettuale

  Task          Output      Attivazione Finale   Loss
  ------------- ----------- -------------------- ---------------------
  Regressione   1 neurone   Nessuna              MSE / L1
  Binaria       1 neurone   Sigmoid              BCE / BCEWithLogits
  Multiclasse   N neuroni   Nessuna              CrossEntropy
  Multilabel    N neuroni   Sigmoid              BCEWithLogits