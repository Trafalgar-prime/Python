# Pre-elaborazione dei Dati in Ingresso — BLOCCO 2 (scikit-learn)

In scikit-learn la pre-elaborazione fatta bene significa:
- **Pipeline** per evitare leakage e rendere il flusso ripetibile
- **ColumnTransformer** per gestire colonne numeriche e categoriche in modo separato
- preprocess + modello in un unico oggetto serializzabile

Questo documento contiene teoria + esempi di codice riutilizzabili.

---

## 1) Strumenti chiave in sklearn

- `Pipeline` → sequenza di step (imputer → scaler → modello)
- `ColumnTransformer` → preprocessing diverso per colonne diverse
- `SimpleImputer` → missing values
- `StandardScaler / MinMaxScaler / RobustScaler` → scaling
- `OneHotEncoder / OrdinalEncoder` → categoriche
- `PolynomialFeatures` → feature engineering
- `PCA` → riduzione dimensionale
- `FunctionTransformer` → trasformazioni custom
- split: `train_test_split`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`

---

## 2) Caso reale tabellare: numeriche + categoriche (best practice)

### 2.1 Definizione colonne
```python
num_cols = ["age", "income", "balance"]
cat_cols = ["city", "job", "gender"]
```

### 2.2 ColumnTransformer: preprocessing separato
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)
```

**Perché è corretto:**
- imputazione/scaling fittati solo su train (dentro `fit`)
- categorie nuove nel test non rompono tutto (`handle_unknown="ignore"`)

---

## 3) Pipeline completa: preprocess + modello (anti leakage totale)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

clf.fit(X_train, y_train)
```

Ora `clf` contiene preprocessing + modello in un unico oggetto.

---

## 4) Gestione outlier

### 4.1 RobustScaler (resistente agli outlier)
```python
from sklearn.preprocessing import RobustScaler

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])
```

### 4.2 Clip / Winsorize con FunctionTransformer
```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def clip_extremes(X):
    return np.clip(X, -10, 10)

clipper = FunctionTransformer(clip_extremes)
```
Poi inserisci `clipper` nella pipeline numerica prima dello scaler.

---

## 5) Feature engineering dentro pipeline

### 5.1 Log transform (variabili skewed)
```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(
    lambda X: np.log1p(X),
    feature_names_out="one-to-one"
)
```

### 5.2 Polynomial features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
```

---

## 6) PCA (riduzione dimensionale)

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("pca", PCA(n_components=20)),
    ("model", LogisticRegression(max_iter=2000))
])
```

---

## 7) Split corretti (fondamentali)

### 7.1 Classificazione sbilanciata: stratify
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 7.2 Gruppi (stesso utente non deve dividersi)
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
```

### 7.3 Serie temporali
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

---

## 8) Testo in sklearn (approccio classico TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
    ("model", LogisticRegression(max_iter=2000))
])

text_clf.fit(text_train, y_train)
```

---

## 9) Errori comuni (da evitare)

- `scaler.fit_transform(X)` prima dello split → **data leakage**
- dimenticare `handle_unknown="ignore"` → crash su categorie nuove
- usare `OrdinalEncoder` senza ordine reale → numeri “finti” e bias
- split random su time series → predici passato col futuro (errore grave)

---

## Conclusione

In sklearn un buon preprocessing è:
- ripetibile (`Pipeline`)
- multi-colonna (`ColumnTransformer`)
- anti-leakage (fit solo su train)
- pronto per deploy (preprocess + modello insieme)

