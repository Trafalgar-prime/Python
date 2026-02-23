# Pandas DataFrame — Guida completa e spiegazione concettuale

Questa guida spiega **in modo approfondito** come funziona il **DataFrame di Pandas**:
non solo *come usarlo*, ma *come pensarlo* e *perché funziona così*.

È pensata per studio universitario, Machine Learning e Data Science.

---

## 1. Cos’è Pandas

**Pandas** è una libreria Python per la manipolazione e l’analisi di dati strutturati.

Serve per:
- lavorare con **tabelle**
- caricare dati da file (CSV, Excel, SQL, JSON…)
- preparare dati per **Machine Learning**

È costruita sopra **NumPy**.

---

## 2. Oggetti fondamentali di Pandas

Pandas ha due strutture principali:

- **Series** → colonna monodimensionale
- **DataFrame** → tabella bidimensionale

Questa guida è focalizzata sul **DataFrame**.

---

## 3. Cos’è un DataFrame (concetto chiave)

Un **DataFrame** è una **tabella bidimensionale etichettata** composta da:

- righe (identificate da un **Index**)
- colonne (ognuna è una **Series**)
- celle con valori (numeri, stringhe, booleani…)

Concettualmente:
```
DataFrame = insieme di Series allineate sullo stesso Index
```

---

## 4. Struttura interna di un DataFrame

Un DataFrame è composto da tre elementi fondamentali:

### 4.1 Index
- etichetta delle righe
- NON è una colonna
- può avere un nome
- usato per allineamenti automatici

```python
df.index
df.index.name = "id"
```

### 4.2 Columns
- nomi delle colonne
- ogni colonna è una Series

```python
df.columns
```

### 4.3 Data
- valori veri e propri
- spesso memorizzati come array NumPy

---

## 5. Creare un DataFrame

### 5.1 Da array NumPy
```python
import pandas as pd
import numpy as np

data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(data, columns=["A", "B"])
```

### 5.2 Da dizionario
```python
df = pd.DataFrame({
    "nome": ["Anna", "Luca", "Marco"],
    "eta": [22, 25, 30]
})
```

### 5.3 Da file
```python
pd.read_csv("file.csv")
pd.read_excel("file.xlsx")
```

---

## 6. Dimensioni e struttura

```python
df.shape      # (righe, colonne)
df.ndim       # 2
df.size       # righe * colonne
```

---

## 7. Colonne e Series

Ogni colonna è una **Series**:

```python
type(df["eta"])  # pandas.Series
```

Accesso:
```python
df["eta"]
```

---

## 8. Selezione dei dati (fondamentale)

### 8.1 .loc — per etichetta
```python
df.loc[0, "eta"]
df.loc[0:2, ["nome", "eta"]]
```

- usa index e nomi colonna
- slicing inclusivo

### 8.2 .iloc — per posizione
```python
df.iloc[0, 1]
df.iloc[:, :-1]
```

- usa posizioni numeriche
- slicing esclusivo

---

## 9. Aggiungere colonne

```python
df["nuova"] = 10
df["somma"] = df["A"] + df["B"]
```

Regola:
- lunghezza colonna = numero righe

---

## 10. Eliminare righe o colonne

```python
df.drop(columns=["eta"])
df.drop(index=[0, 1])
```

Serve riassegnare:
```python
df = df.drop(...)
```

---

## 11. Operazioni vettoriali

```python
df["eta"] = df["eta"] + 1
```

- niente loop
- molto veloce
- sfrutta NumPy

---

## 12. Tipi di dato

```python
df.dtypes
```

Tipi comuni:
- int64
- float64
- object (stringhe)
- bool
- category

---

## 13. Metodi fondamentali

```python
df.head()
df.tail()
df.info()
df.describe()
```

---

## 14. Allineamento automatico (concetto avanzato)

Pandas allinea per **Index**, non per posizione:

```python
a = pd.Series([1, 2], index=[0, 1])
b = pd.Series([10, 20], index=[1, 2])

a + b
```

Risultato:
```
0     NaN
1    12.0
2     NaN
```

---

## 15. DataFrame vs NumPy

| Aspetto | NumPy | Pandas |
|------|------|------|
| Etichette | ❌ | ✅ |
| Tipi misti | ❌ | ✅ |
| Allineamento | ❌ | ✅ |
| Tabelle | ❌ | ✅ |

---

## 16. Concetti chiave da ricordare

- un DataFrame è una tabella
- è composto da Series
- l’Index NON è una colonna
- Pandas lavora in modo vettoriale
- perfetto per analisi dati e ML

---

## 17. Perché Pandas è fondamentale nel ML

- pulizia dati
- selezione feature
- gestione target
- integrazione con scikit-learn

---

## Fine

Questo file è pensato come **README GitHub / dispensa di studio**.
