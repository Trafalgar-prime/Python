# BLOCCO 3 — Valutazione dei Modelli con TensorFlow / Keras (Approfondito + Codice)

Questo documento spiega **come valutare correttamente un modello in TensorFlow/Keras**, includendo:
- valutazione integrata in `fit()` (History)
- valutazione finale con `evaluate()`
- analisi avanzata (confusion matrix, report, soglie, AUC ROC/PR, calibration)
- callback e TensorBoard
- metriche custom
- note su cross-validation in TF

---

## 1) Workflow corretto in Keras

1. Preparazione dataset (`tf.data`)
2. `model.compile(loss=..., metrics=[...])`
3. `model.fit(..., validation_data=...)`
4. `model.evaluate(test_ds)`  Test_ds = X_test
5. Analisi avanzata su `model.predict()` (confusion matrix, soglie, calibration, ecc.)

**3 livelli di valutazione:**
- Durante training: History + TensorBoard
- Finale: `model.evaluate()`
- Custom: calcoli offline su `predict()`

---

## 2) Loss vs Metrics

- **Loss**: ottimizzata dal gradiente.
- **Metrics**: monitoraggio e valutazione pratica.

Pattern tipici:
- loss train ↓ e val ↓ → ok
- loss train ↓ e val ↑ → overfitting
- entrambe piatte → underfitting
- oscillazioni → LR alto / instabilità

---

## 3) Compile: loss e metriche nel modo giusto

### 3.1 Binaria (sigmoid)
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]
)
```

### 3.1b Binaria (logits, spesso consigliato)
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)  # logits
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.AUC(name="roc_auc")]
)
```

### 3.2 Multiclasse
One-hot:
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
)
```
| Target          | Loss                          |
| --------------- | ----------------------------- |
| One-hot         | CategoricalCrossentropy       |
| Interi (0..K-1) | SparseCategoricalCrossentropy |


Sparse labels (0..K-1):
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
```

### 3.3 Regressione
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.Huber(),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ]
)
```

---

## 4) Valutazione durante training: History

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

print(history.history.keys())
```

Interpretazione:
- `val_loss` è la metrica principale per diagnosi overfitting/underfitting.
- le metriche possono migliorare anche se la loss non migliora allo stesso modo.

---

## 5) Callback fondamentali

### EarlyStopping
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]
history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

### ModelCheckpoint
```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]
```

### ReduceLROnPlateau
```python
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3
    )
]
```

---

## 6) Valutazione finale: model.evaluate()

```python
results = model.evaluate(X_test, Y_test, return_dict=True)
print(results)
```

Consiglio: `return_dict=True` per non dipendere dall’ordine.

---

## 7) Confusion matrix e classification report (offline)

### 7.1 Binaria
```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_proba = model.predict(test_ds).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
```

### 7.2 Multiclasse
```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_proba = model.predict(test_ds)
y_pred = np.argmax(y_proba, axis=1)

# Se y_true è one-hot:
# y_true = np.argmax(y_true, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
```

---

## 8) Scelta della soglia ottimale (es. massimizzare F1)

```python
import numpy as np
from sklearn.metrics import f1_score

y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_proba = model.predict(val_ds).ravel()

thresholds = np.linspace(0.05, 0.95, 19)
best_t, best_f1 = None, -1

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("Best threshold:", best_t, "Best F1:", best_f1)
```

---

## 9) TensorBoard

```python
log_dir = "logs/run1"
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[tb]
)
```

(Avvio tipico: `tensorboard --logdir logs`)

---

## 10) Metriche custom (esempio F1)

> Nota: F1 “batch-wise” può differire dalla F1 globale.  
> Per valutazione finale, calcola F1 offline su `predict()`.

```python
import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)
```

---

## 11) Calibration (Brier score)

```python
import numpy as np

y_true = np.concatenate([y for x, y in test_ds], axis=0).astype(float)
y_proba = model.predict(test_ds).ravel()

brier = np.mean((y_proba - y_true) ** 2)
print("Brier score:", brier)
```

---

## 12) Cross-Validation in TensorFlow: come si fa davvero

TensorFlow non ha un `cross_val_score` nativo.

Approcci:
- **Manuale con KFold**: ricrei modello da zero per ogni fold, alleni e valuti, poi fai media/std.
- Wrapper stile sklearn (opzionale): es. `scikeras` (non necessario, dipende dal progetto).

---

## 13) Robustezza e valutazione realistica

Valuta anche su:
- dataset con rumore / missing
- dataset shiftato (periodo diverso)
- sottogruppi (analisi per condizioni)

In TF lo fai creando dataset alternativi e confrontando `model.evaluate()`.

---

## 14) Criteri pratici: “modello addestrato bene” in TF/Keras

- `val_loss` scende senza divergere
- callback selezionano best stabile (EarlyStopping + Checkpoint)
- `evaluate(test)` conferma
- metriche adeguate (F1/PR-AUC se sbilanciato)
- soglia scelta su validation
- (se serve) probabilità calibrate e robuste a shift

---

## 15) Conclusione

In TF/Keras la valutazione è:
- integrata nel training (history/metrics)
- confermata su test (`evaluate`)
- completata con analisi offline (confusion, thresholding, calibration)

Allenare è facile. **Valutare bene richiede metodo.**
