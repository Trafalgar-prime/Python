# Pre-elaborazione dei Dati in Ingresso — BLOCCO 3 (TensorFlow / Keras)

Questo documento descrive **in modo completo e approfondito** come gestire la pre-elaborazione dei dati in TensorFlow / Keras.
Il preprocessing può vivere:
1) fuori dal modello (Python/pandas)
2) nella pipeline `tf.data`
3) **dentro il modello** con i Keras preprocessing layers (scelta migliore per deploy)

---

## 1) Filosofia del preprocessing in TensorFlow

In TensorFlow il preprocessing è parte del **grafo computazionale**.
Se fatto bene:
- evita leakage
- è identico tra training e inference
- viene salvato nel `SavedModel`

Regola chiave:
> ogni operazione che “impara dai dati” usa `adapt()` **solo sul training set**.

---

## 2) Pipeline `tf.data` (base solida e performante)

### Template standard
```python
import tensorflow as tf

BATCH = 64
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10_000).batch(BATCH).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH).prefetch(AUTOTUNE)
```

---

## 3) Scaling / Normalizzazione (equivalente di StandardScaler)

### Normalization layer
```python
norm = tf.keras.layers.Normalization()
norm.adapt(train_ds.map(lambda x, y: x))  # SOLO train
```

Uso nel modello:
```python
inputs = tf.keras.Input(shape=(num_features,))
x = norm(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
```

Vantaggi:
- no leakage
- preprocessing salvato nel modello
- deploy immediato

---

## 4) Gestione dei missing values

### 4.1 Sostituzione in `tf.data`
```python
def fill_nan(x, y):
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    return x, y

train_ds = train_ds.map(fill_nan, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(fill_nan, num_parallel_calls=AUTOTUNE)
```

### 4.2 Missing come informazione
- aggiungi feature binaria `is_missing`
- concateni al vettore di input

---

## 5) Feature categoriche

### 5.1 StringLookup + One-Hot
```python
lookup = tf.keras.layers.StringLookup(output_mode="one_hot")
lookup.adapt(train_strings)  # SOLO train
```

Uso:
```python
inp = tf.keras.Input(shape=(1,), dtype=tf.string)
x = lookup(inp)
x = tf.keras.layers.Dense(32, activation="relu")(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, out)
```

---

### 5.2 Categoriche ad alta cardinalità → Embedding
```python
lookup = tf.keras.layers.StringLookup()
lookup.adapt(train_strings)

vocab_size = lookup.vocabulary_size()
embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16)

inp = tf.keras.Input(shape=(1,), dtype=tf.string)
ids = lookup(inp)
x = embed(ids)
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, out)
```

---

## 6) Testo (NLP) in TensorFlow

### TextVectorization
```python
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=50000,
    output_mode="tf_idf"  # oppure "int"
)
vectorizer.adapt(train_texts)  # SOLO train
```

Uso nel modello:
```python
inp = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inp)
x = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, out)
```

Con `output_mode="int"` → Embedding + RNN/Transformer.

---

## 7) Immagini: resize, normalization, augmentation

### Preprocessing layers
```python
data_aug = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
])

rescale = tf.keras.layers.Rescaling(1./255)
```

Uso:
```python
inp = tf.keras.Input(shape=(None, None, 3))
x = data_aug(inp)
x = rescale(x)
# backbone CNN
```

Augmentation:
- applicata solo in training
- automaticamente disattivata in inference

---

## 8) Time series in TensorFlow

Regole:
- split temporale (mai shuffle globale)
- windowing con `tf.data`
- scaling adattato solo sul passato

### Windowing esempio
```python
series_ds = tf.data.Dataset.from_tensor_slices(series)
series_ds = series_ds.window(30, shift=1, drop_remainder=True)
series_ds = series_ds.flat_map(lambda w: w.batch(30))
```

---

## 9) Anti-leakage: errori più comuni

❌ `adapt()` su train+val+test  
❌ vocabolari costruiti con dati futuri  
❌ normalizzazione con statistiche globali  

Regola:
> ogni `adapt()` SOLO su training set.

---

## 10) Checklist preprocessing fatto bene (TensorFlow)

- preprocessing ripetibile (`tf.data` o layers)
- `adapt()` solo su train
- pipeline identica tra training e inference
- gestione OOV automatica (`StringLookup`)
- augmentation solo su train
- preprocessing incluso nel modello per il deploy

---

## Conclusione

In TensorFlow il preprocessing **fa parte del modello**.
Se lo progetti bene:
- eviti leakage
- semplifichi il deploy
- rendi il sistema robusto e riproducibile
