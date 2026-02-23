# GridSearchCV e ricerca iperparametri (scikit-learn)

## Vale solo per sklearn?
**GridSearchCV** è uno strumento **specifico di scikit-learn** (modulo `sklearn.model_selection`).  
Quindi: **sì, GridSearchCV “nativo” è sklearn**.

Detto questo, **l’idea** di “cercare i migliori iperparametri con cross-validation” esiste anche in PyTorch e TensorFlow, ma **si implementa con altri strumenti** (wrapper o librerie dedicate). Qui però lavoriamo **solo su sklearn**.

---

## Cos’è GridSearchCV (definizione operativa)
`GridSearchCV` esegue una ricerca “a griglia” sugli **iperparametri**:
1. tu definisci una **griglia** di valori (tutte le combinazioni)
2. per ogni combinazione fa **cross-validation** sul training set
3. calcola uno **score** (`scoring`) per ogni combinazione
4. sceglie la combinazione con score migliore e ti fornisce:
   - `best_params_` → i migliori iperparametri trovati
   - `best_score_` → score medio CV migliore
   - `best_estimator_` → modello migliore **già rifittato** (se `refit=True`)

**Idea chiave:** non usa il test set durante la ricerca. Il test serve alla valutazione finale “onesta”.

---

## Dove si usa nel codice (posizione corretta)
Sequenza corretta (anti-leakage):

1) Split **train/test**  
2) Definisci una **Pipeline** (preprocessing + modello)  
3) Definisci `param_grid`  
4) `GridSearchCV(...).fit(X_train, y_train)`  
5) Valuta `best_estimator_` su `X_test, y_test`

> Se fai preprocessing (scaling) “a mano” prima della CV, rischi leakage.  
> Pipeline risolve: lo scaler viene fittato **solo sui fold di train**.

---

## Come si usa nel codice (meccanica)
Tu fornisci:
- `estimator`: il modello o una `Pipeline`
- `param_grid`: dizionario (o lista di dizionari) con i valori da provare
- `cv`: strategia di cross-validation
- `scoring`: metrica da ottimizzare
- opzionali: `n_jobs`, `verbose`, `refit`, ecc.

Poi:
```python
grid = GridSearchCV(estimator, param_grid=..., cv=..., scoring=...)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

---

## Implicazioni pratiche (quelle che contano davvero)

### 1) Costo computazionale
Se hai:
- 50 combinazioni nella griglia
- `cv=5`

alleni **50 × 5 = 250** modelli.

### 2) Overfitting al processo di tuning
Se provi tantissime griglie e fai molti tentativi, rischi di “adattarti” ai fold.  
Il test set finale è ciò che ti salva da una valutazione ottimistica.

### 3) Leakage (errore concettuale)
Fare scaling su tutto il dataset prima della CV è leakage perché le statistiche del test/fold entrano nel preprocessing.  
Soluzione: **Pipeline**.

### 4) Scelta della metrica
Accuracy non sempre è corretta (class imbalance, costi diversi degli errori, ecc.).  
Per multiclasse spesso `f1_macro` è più robusta.

### 5) Refit
Con `refit=True` (default), dopo la ricerca sklearn:
- prende la migliore combinazione
- rifitta il modello su **tutto `X_train`**

Quindi `best_estimator_` è pronto da usare sul test.

---

## Parametri principali di GridSearchCV (quelli che “interferiscono” davvero)

### Parametri core
- **`estimator`**  
  Modello o Pipeline su cui eseguire la ricerca.

- **`param_grid`**  
  Dizionario o lista di dizionari:
  - chiavi = nomi iperparametri
  - valori = lista dei valori da provare

  Se usi `Pipeline`, i nomi sono:  
  `nome_step__nome_parametro`  
  Esempio: `clf__C`, `scaler__with_mean`.

- **`scoring`**  
  Metrica da ottimizzare:
  - stringa (es. `"accuracy"`, `"f1_macro"`, `"roc_auc_ovr"`, `"neg_mean_squared_error"`)
  - oppure una funzione callable custom.

- **`cv`**  
  Strategia di cross-validation:
  - int (es. `5`)
  - oppure oggetto (consigliato) come `StratifiedKFold(...)` per classificazione.

- **`refit`**  
  - `True` → rifitta il best su tutto il train
  - stringa se usi multi-metric (`refit="f1_macro"`)

- **`n_jobs`**  
  Parallelizzazione:
  - `-1` = usa tutti i core disponibili.

- **`verbose`**  
  Quanta “log” stampare:
  - `0` silenzioso
  - `1`/`2` … più dettagli.

### Diagnostica / robustezza
- **`return_train_score`**  
  Se `True`, salva anche lo score sui fold di train: utile per osservare overfitting.

- **`error_score`**  
  Cosa fare se una combinazione fallisce:
  - `"raise"` (consigliato per vedere l’errore subito)
  - oppure un numero (es. `np.nan`).

- **`pre_dispatch`**  
  Controlla quante job vengono lanciate contemporaneamente (utile se RAM limitata).

### Nota su `fit_params`
`grid.fit(X, y, **fit_params)` permette di passare parametri al `.fit()` del modello (es. `sample_weight`).  
Ma questi **non sono iperparametri ricercati**: vengono solo passati al fit.

---

## Si usa anche con PyTorch o TensorFlow?
Non “nativamente” come oggetto ufficiale, perché GridSearchCV è sklearn.  
Esistono però wrapper e librerie di hyperparameter tuning.  
**Qui ci fermiamo a sklearn**, come richiesto.

---

## Esempio completo: Digits + Pipeline + GridSearchCV (no leakage)

Esempio “pulito”, con metrica **f1_macro** e gestione solver/penalty coerenti.

```python
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1) Dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2) Split train/test (il test resta fuori dalla CV)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Pipeline (anti-leakage): scaler viene fittato SOLO sui fold di train
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

# 4) Griglia iperparametri (nota la sintassi: step__param)
param_grid = [
    {
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 1, 10]
    },
    {
        "clf__solver": ["saga"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.1, 1, 10]
    }
]

# 5) CV stratificata (coerente con classificazione)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6) GridSearchCV
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True,
    error_score="raise"
)

# 7) Fit SOLO sul train
grid.fit(X_train, y_train)

print("\nBEST PARAMS:", grid.best_params_)
print("BEST CV SCORE (f1_macro):", grid.best_score_)

# 8) Best model già rifittato su tutto il train
best_model = grid.best_estimator_

# 9) Valutazione finale su test (mai visto durante la CV)
y_pred = best_model.predict(X_test)

print("\nTEST accuracy:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))
```

---

## Bonus (opzionale)
- `grid.cv_results_` per vedere tutte le combinazioni e i punteggi
- multi-metric scoring (accuracy + f1_macro) con `refit="f1_macro"`
- `RandomizedSearchCV` quando la griglia è troppo grande
