# Pre-elaborazione dei Dati in Ingresso — BLOCCO 1 (Universale)

Questo blocco raccoglie **tutte le tecniche di pre-elaborazione** che valgono a prescindere dal framework (scikit-learn, TensorFlow/Keras, PyTorch).
L’obiettivo è capire *cosa* fare e *perché*, prima di vedere *come* farlo in ogni pacchetto.

---

## Obiettivo della pre-elaborazione

La pre-elaborazione serve a:

1) **Pulire** i dati (missing, outlier, duplicati, errori)  
2) **Rendere le feature utilizzabili** (numeriche/categoriche/testo/immagini/tempo)  
3) **Stabilizzare il training** (scaling/normalizzazione, trasformazioni)  
4) **Evitare leakage** (fit solo su train)  
5) **Migliorare generalizzazione** (feature engineering + augmentation)

---

## 1) Tipi di dato in input (e cosa implica)

- **Tabellare** (numeriche + categoriche) → scaling, encoding, imputazione  
- **Testo** → tokenizzazione, vocab, embeddings  
- **Immagini** → resize, normalization, augmentation  
- **Serie temporali** → split temporale, lag features, rolling stats, scaling “per tempo”  
- **Audio** → resampling, spectrogram/MFCC, normalization  
- **Dati misti** → pipeline multi-ramo (numeriche+categoriche+testo)

---

## 2) Pulizia dati (Data Cleaning)

### 2.1 Missing values (valori mancanti)

**Strategie principali:**
- **Drop** righe/colonne (solo se pochi missing e non informativi)
- **Imputazione semplice**:
  - numeriche: media/mediana
  - categoriche: moda / “Unknown”
- **Imputazione avanzata**:
  - KNN imputer
  - modelli (iterative/mice)
- **Missing come informazione**:
  - aggiungi una feature “is_missing”

📌 Nota: se imputi **prima** dello split → leakage.

---

### 2.2 Duplicati

- rimuovere duplicati identici
- attenzione ai duplicati “quasi uguali” (stesso utente, stessa transazione)

---

### 2.3 Outlier (valori anomali)

Cosa puoi fare:
- **detect** (IQR, z-score, MAD, isolation forest)
- **clip/winsorize** (tagli i valori estremi)
- **robust scaling** (scaler resistente)
- **trasformazioni** (log, sqrt, Box-Cox, Yeo-Johnson)

📌 Outlier non sempre sono “errori”: a volte sono casi reali importanti.

---

### 2.4 Noise / errori di misura

- smoothing (serie temporali)
- filtri (moving average)
- validazione range (es. età non può essere -5)

---

## 3) Trasformazioni numeriche (Scaling / Normalization)

### 3.1 Standardizzazione (StandardScaler)
Porta a media 0 e deviazione standard 1.
- ottimo per modelli basati su distanza o gradiente (SVM, KNN, NN)

### 3.2 Min-Max scaling
Porta nel range [0,1] o [-1,1].
- utile se vuoi range controllato

### 3.3 Robust scaling
Usa mediana e IQR.
- utile con outlier

### 3.4 Normalizzazione “per norma”
Rende i vettori lunghezza 1 (L2 norm).
- utile per cosine similarity, embedding, testo

📌 Regola pratica:
- Alberi (RF, XGBoost) spesso non richiedono scaling
- KNN/SVM/NN quasi sempre sì

---

## 4) Encoding categoriche (trasformare stringhe in numeri)

### 4.1 One-Hot Encoding
- crea una colonna per categoria
- ottimo con modelli lineari e molti modelli classici
- problema: dimensione esplode con alta cardinalità

### 4.2 Ordinal Encoding
- assegna un numero per categoria
- SOLO se c’è un ordine reale (es. “basso/medio/alto”)

### 4.3 Target Encoding / Mean Encoding
- sostituisce categoria con media del target
- potente, ma rischio leakage altissimo (va fatto con CV interna)

### 4.4 Hashing trick
- mappa categorie in uno spazio fisso
- utile per cardinalità enorme

---

## 5) Feature Engineering (creare feature migliori)

Dipende dal dominio, ma esempi tipici:
- interazioni: `x1 * x2`, `x1 / x2`
- log-transform su variabili skewed
- binning (discretizzazione)
- aggregazioni per gruppo (per utente, per giorno, ecc.)
- PCA / riduzione dimensionale

📌 Attenzione: feature engineering fatta usando info del futuro → leakage.

---

## 6) Gestione sbilanciamento classi (classification)

Non è “preprocessing” puro, ma è **prima del training**.

- **class weights**
- **oversampling** (RandomOverSampler, SMOTE)
- **undersampling**
- metriche adeguate (F1, PR-AUC)

---

## 7) Split corretto (fondamentale per la pre-elaborazione)

Tipi:
- random split
- stratified split
- group split (stesso soggetto nello stesso split)
- time-based split (serie temporali)

📌 Regola d’acciaio:
> tutto ciò che “impara dai dati” si fitta su train e si applica a val/test.

---

## 8) Preprocessing per testo (NLP)

- cleaning (lowercase, punctuation) *dipende dal modello*
- tokenizzazione
- stopwords (solo per modelli classici)
- stemming/lemmatization (classici)
- vectorization:
  - Bag-of-Words
  - TF-IDF
  - embeddings (Word2Vec, FastText)
  - token IDs per Transformers

---

## 9) Preprocessing per immagini (Computer Vision)

- resize/crop
- normalization (mean/std)
- augmentation:
  - flip, rotate, color jitter
  - random crop
  - cutout/mixup (più avanzate)

---

## 10) Preprocessing per time series

- split temporale (mai shuffle casuale se c’è dipendenza)
- lag features (t-1, t-7…)
- rolling mean/std
- differencing (stazionarietà)
- scaling “fittato” sul passato

---

## 11) Pipeline e riproducibilità

Una buona pre-elaborazione deve essere:
- **ripetibile**
- **serializzabile**
- identica tra training e inference

Quindi: pipeline salvabile (sklearn Pipeline / TF preprocessing / PyTorch transforms).

---

## Conclusione

Il preprocessing è parte del modello: se è sbagliato, il modello può sembrare “forte” ma fallire in produzione.
La regola più importante è sempre la stessa:

> **fit su train, applica su val/test** (no leakage).

