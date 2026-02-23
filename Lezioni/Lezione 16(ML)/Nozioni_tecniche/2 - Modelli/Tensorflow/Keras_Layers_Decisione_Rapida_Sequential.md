# Keras Layers — Riorganizzazione “Decisione Rapida” (Sequential)

Questa guida serve a non perdersi tra decine di layer Keras.  
L’idea è semplice:

> Non devi conoscere tutti i layer. Devi conoscere quelli giusti per il tuo tipo di dato.

La scelta dei layer dipende **dalla struttura dei dati** (tabellare, immagini, testo/sequenze, time series).

---

## 1) TABELLARE (numeri, colonne tipo Excel)

✅ 90% dei casi industriali “classici”.

### Layer fondamentali (6–8)
| Layer | Perché serve |
|---|---|
| `Dense` | Costruisce una rete MLP (feed-forward) |
| `Dropout` | Riduce overfitting spegnendo neuroni in training |
| `BatchNormalization` | Stabilizza training e accelera convergenza |
| `Activation` | Separa attivazione dal Dense (se vuoi modularità) |
| `Embedding` | Per categoriche ad alta cardinalità (ID → vettori) |
| `Normalization` | Preprocessing integrato nel modello (`adapt()` su train) |
| `LayerNormalization` | Alternativa a BN (utile in certi contesti) |
| `Reshape` | Ristruttura l’input quando serve |

### Architettura tipica tabellare
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(10)  # output
])
```

---

## 2) IMMAGINI (CNN)

Se hai immagini/pixel/feature map, serve estrarre pattern locali.

### Layer fondamentali (≈10)
| Layer | Perché |
|---|---|
| `Conv2D` | Estrae pattern locali (bordi, texture, forme) |
| `MaxPooling2D` | Downsample e robustezza a piccole traslazioni |
| `BatchNormalization` | Stabilizza attivazioni |
| `ReLU` / `Activation` | Non-linearità |
| `Flatten` | (Vecchio stile) passa da 2D a 1D |
| `GlobalAveragePooling2D` | Alternativa moderna a Flatten (più robusta) |
| `Dense` | Classificatore finale |
| `Dropout` | Regolarizzazione |
| `Conv2DTranspose` | Decoder / upsampling learnable |
| `SeparableConv2D` | CNN leggere (efficienza) |

### CNN minimale
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

model = Sequential([
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    GlobalAveragePooling2D(),
    Dense(10)
])
```

---

## 3) TESTO / SEQUENZE (NLP)

Qui la difficoltà è gestire **ordine** e **lunghezze**.

### Layer fondamentali (8–10)
| Layer | Perché |
|---|---|
| `Embedding` | Parole/ID → vettori continui |
| `LSTM` | Memoria temporale (sequenze) |
| `GRU` | Versione più leggera di LSTM |
| `Bidirectional` | Usa contesto avanti+indietro (NLP) |
| `Masking` | Ignora padding |
| `GlobalMaxPooling1D` | Riassume una sequenza in un vettore |
| `Dense` | Classificatore finale |
| `Dropout` | Regolarizza |
| `MultiHeadAttention` | Base dei Transformer |
| `LayerNormalization` | Stabilizza attenzione e reti moderne |

### Modello NLP semplice (baseline)
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(64),
    Dense(1, activation="sigmoid")
])
```

---

## 4) TIME SERIES (numeriche nel tempo)

Due approcci validi: Conv1D (spesso stabile) o RNN (LSTM/GRU).

### A) Conv1D (moderno e stabile)
Layer tipici:
- `Conv1D`
- `MaxPooling1D` (opzionale)
- `GlobalAveragePooling1D`
- `Dense`

### B) LSTM / GRU (sequenze “classiche”)
Layer tipici:
- `LSTM` / `GRU`
- `Bidirectional` (se ha senso)
- `Dropout`
- `Dense`

### Esempio Conv1D
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense

model = Sequential([
    Conv1D(64, 3, activation="relu"),
    GlobalAveragePooling1D(),
    Dense(1)
])
```

---

## 5) Quando NON usare `Sequential`

Non usare `Sequential` se:
- hai più input
- hai più output
- hai skip connections (tipo ResNet)
- hai architetture multi-ramo
- hai attenzione/Transformer non banale con più flussi

In questi casi usa la Functional API:
```python
tf.keras.Model(...)
```

---

## Riassunto ultra-pratico (cosa studiare davvero)

- **Tabellare** → `Dense`, `Dropout`, `BatchNormalization`
- **Immagini** → `Conv2D`, `Pooling`, `GlobalAveragePooling2D`
- **Testo** → `Embedding`, `LSTM/GRU` (o `MultiHeadAttention`)
- **Time series** → `Conv1D` oppure `LSTM`

---

## Mentalità corretta

Non scegliere layer perché “esistono”.  
Sceglili perché la **struttura dei dati** lo richiede.
