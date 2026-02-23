# BLOCCO 2 --- Valutazione dei Modelli con scikit-learn (Teoria + Codice)

Questo documento mostra **come valutare correttamente un modello in
scikit-learn**, unendo teoria e **codice pratico** pronto all'uso.

------------------------------------------------------------------------

## 1) Split corretto e Pipeline (anti data leakage)

``` python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

------------------------------------------------------------------------

## 2) Metriche di classificazione

### Accuracy e report completo

``` python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

``` python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

### Precision, Recall, F1

``` python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

------------------------------------------------------------------------

## 3) ROC-AUC e Precision--Recall AUC

``` python
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc_score(y_test, y_proba)
average_precision_score(y_test, y_proba)
```

------------------------------------------------------------------------

## 4) Scelta della soglia ottimale

``` python
import numpy as np
from sklearn.metrics import f1_score

thresholds = np.linspace(0.05, 0.95, 19)
best_f1, best_t = -1, None

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1, best_t = f1, t

best_t, best_f1
```

------------------------------------------------------------------------

## 5) Cross-Validation

``` python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

scores.mean(), scores.std()
```

------------------------------------------------------------------------

## 6) Regressione: metriche principali

``` python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

------------------------------------------------------------------------

## 7) Learning Curves

``` python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=cv, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 5)
)
```

------------------------------------------------------------------------

## 8) Calibration delle probabilità

``` python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

cal = CalibratedClassifierCV(model, method="isotonic", cv=5)
cal.fit(X_train, y_train)
proba_cal = cal.predict_proba(X_test)[:, 1]

CalibrationDisplay.from_predictions(y_test, proba_cal)
```

------------------------------------------------------------------------

## 9) Dataset sbilanciati

-   usare `class_weight="balanced"`
-   preferire F1, PR-AUC, confusion matrix

``` python
LogisticRegression(class_weight="balanced")
```

------------------------------------------------------------------------

## 10) Quando puoi dire "ho fatto un buon lavoro"

-   Cross-validation stabile
-   Metriche adeguate al problema
-   Confusion matrix coerente con il costo degli errori
-   Soglia scelta su validation
-   Nessun data leakage (Pipeline)
