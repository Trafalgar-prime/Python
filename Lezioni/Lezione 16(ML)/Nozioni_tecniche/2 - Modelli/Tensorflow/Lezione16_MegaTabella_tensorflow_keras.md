# Lezione 16 — Mega tabella (Parte 2/3): TensorFlow / Keras

> Obiettivo: una “mappa” pratica dei **10 modelli/archetipi Keras più importanti** + una mega tabella delle **opzioni** (compile/fit/optimizer/loss/layer) con **argomenti e valori ammessi**.

> Nota importante: in Keras **non esistono “modelli” separati come in scikit-learn** (tipo `LogisticRegression`, `SVC`, ecc.). In Keras il “modello” è una **composizione di layer**, e quello che cambia task-per-task è soprattutto: **struttura (layer)** + **loss** + **optimizer** + **metriche**.

---

## I 10 modelli/archetipi più importanti in Keras (quelli che ti portano a VAE/GAN/Transformer)

1. **Linear/Logistic model** (1 layer `Dense`)
2. **MLP** (più `Dense`)
3. **CNN** (vision: `Conv2D` + pooling)
4. **RNN** (sequenze: `SimpleRNN`)
5. **LSTM**
6. **GRU**
7. **Autoencoder**
8. **VAE** (Variational Autoencoder)
9. **GAN** (Generatore + Discriminatore)
10. **Transformer block** (Self-attention con `MultiHeadAttention`)

---

# Mega tabella A — “Modelli” (archetipi) a confronto

| # | Archetip o | Task tipico | “Modello” Keras (come lo costruisci) | Output & loss tipica | Iperparametri chiave (dove stanno) |
|---:|---|---|---|---|---|
| 1 | Linear/Logistic (1 Dense) | regressione / classificazione | `Sequential([Dense(...)])` oppure Functional API | Regr: `Dense(1)` + MSE • Classif binaria: `Dense(1, sigmoid)` + BCE • Multi: `Dense(K, softmax)` + (Sparse) CCE | `units`, `activation` (Dense) • `optimizer`, `learning_rate` (compile) • `batch_size`, `epochs` (fit) |
| 2 | MLP (Dense stack) | tabulare, baseline | più `Dense` + eventuale `Dropout`, `BatchNorm` | come sopra | `hidden_units`, `activation`, `dropout_rate`, `kernel_regularizer` |
| 3 | CNN | immagini | `Conv2D` + pooling + `Flatten`/`GlobalAvgPool` + Dense | classif: softmax + CCE • regr: Dense(1) + MSE | `filters`, `kernel_size`, `strides`, `padding` (Conv2D) |
| 4 | SimpleRNN | sequenze semplici | `SimpleRNN(...)` + Dense output | dipende da task | `units`, `return_sequences`, `dropout`, `recurrent_dropout` |
| 5 | LSTM | testo, serie temporali | `LSTM(...)` + Dense output | dipende da task | come RNN + gating (interno) • attenzione a `return_sequences` |
| 6 | GRU | sequenze | `GRU(...)` + Dense output | dipende da task | simile a LSTM ma più “leggera” |
| 7 | Autoencoder | compressione, anomaly detection | Encoder( Dense/Conv ) + Decoder( Dense/ConvTranspose ) | spesso MSE (ricostruzione) | `latent_dim`, regolarizzazione, architettura simmetrica |
| 8 | VAE | generazione (latente probabilistico) | Encoder produce `mu, logvar` + reparam trick + Decoder | loss = ricostruzione + KL | `latent_dim`, `beta` (beta-VAE), scelta likelihood (MSE/BCE) |
| 9 | GAN | generazione | 2 modelli: `Generator`, `Discriminator`, training alternato | loss avversaria (BCE / hinge / WGAN) | `lr`, `beta_1/beta_2` (Adam), ratio update D/G, stabilizzazione |
| 10 | Transformer block | NLP, sequenze | `MultiHeadAttention` + FFN (Dense) + residual + LayerNorm | classificazione: softmax • LM: CCE | `num_heads`, `key_dim`, `dropout`, `ff_dim`, maschere attenzione |

---

# Mega tabella B — `Model.compile(...)` (opzioni, argomenti e valori ammessi)

> `compile()` “dice al modello” **come allenarsi**: ottimizzatore, loss, metriche, ecc.

| Argomento | Tipo accettato | Valori/forme tipiche | Cosa controlla |
|---|---|---|---|
| `optimizer` | `str` **oppure** `tf.keras.optimizers.Optimizer` | string: `"adam"`, `"sgd"`, … **oppure** oggetto `Adam(...)` ecc. | algoritmo di aggiornamento pesi |
| `loss` | `str` **oppure** callable **oppure** `Loss` instance | `"mse"`, `"binary_crossentropy"`, `"sparse_categorical_crossentropy"`, … oppure `tf.keras.losses.*` | funzione obiettivo |
| `metrics` | list di `str` / callable / `Metric` | es: `["accuracy"]`, `["mae"]`, `[AUC()]`, custom | cosa misuri durante training/eval |
| `loss_weights` | `float`/list/dict o `None` | per modelli multi-output | pesa le loss dei vari output |
| `weighted_metrics` | come metrics | | metriche pesate da sample_weight |
| `run_eagerly` | `bool` | `True/False` | se True: esecuzione eager (più debug, più lenta) |
| `steps_per_execution` | `int` | `>=1` | esegue più batch per call, più veloce su TPU/GPU |
| `jit_compile` | `bool` o `"auto"` | | abilita compilazione XLA (se supportata) |

---

# Mega tabella C — `Model.fit(...)` (opzioni, argomenti e valori ammessi)

> `fit()` “fa partire” l’allenamento. Firma base (semplificata): `fit(x=None, y=None, batch_size=None, epochs=1, ... )`

| Argomento | Tipo accettato | Valori/forme tipiche | Effetto |
|---|---|---|---|
| `x` | array/tensor/dataset/generator | `np.ndarray`, `tf.Tensor`, `tf.data.Dataset`, generator | input |
| `y` | array/tensor o `None` | `np.ndarray`, `tf.Tensor` | target (se non incluso in dataset) |
| `batch_size` | `int` o `None` | es. `32`, `64` | dimensione batch (se x/y sono array) |
| `epochs` | `int >= 1` | es. `10`, `100` | epoche |
| `verbose` | `"auto"` / `0` / `1` / `2` | | logging |
| `callbacks` | list di Callback | es: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard` | hook durante training |
| `validation_split` | `float` in `[0,1)` | es. `0.2` | split automatico dei dati (solo se array) |
| `validation_data` | tuple / dataset | `(X_val, y_val)` oppure `Dataset` | validazione esplicita |
| `shuffle` | `bool` o `"batch"` | | shuffle (solo array) |
| `class_weight` | dict o `None` | `{classe: peso}` | pesa classi sbilanciate |
| `sample_weight` | array o `None` | pesi per esempio | pesa esempi |
| `initial_epoch` | int | | riparte da epoca N |
| `steps_per_epoch` | int o `None` | | numero di batch per epoca (dataset infinito/generator) |
| `validation_steps` | int o `None` | | batch di validazione |
| `validation_batch_size` | int o `None` | | batch size per validazione |
| `validation_freq` | int o list | | ogni quante epoche valida |

---

# Mega tabella D — Ottimizzatori Keras (quelli più importanti) + TUTTI gli argomenti principali

> In Keras, **l’optimizer** può essere una stringa oppure un’istanza `Optimizer(...)`.
> Qui ti elenco gli optimizer più usati e i parametri che tipicamente controlli.

## 1) `Adam(...)` (il default più comune)
Argomenti tipici:
- `learning_rate`: `float` **oppure** schedule
- `beta_1`: `float` in `(0,1)`
- `beta_2`: `float` in `(0,1)`
- `epsilon`: `float`
- `amsgrad`: `bool`

## 2) `SGD(...)`
- `learning_rate`: `float` o schedule
- `momentum`: `float >= 0`
- `nesterov`: `bool`

## 3) `RMSprop(...)`
- `learning_rate`: `float` o schedule
- `rho`: `float`
- `momentum`: `float`
- `epsilon`: `float`
- `centered`: `bool`

## 4) `AdamW(...)` (Adam + weight decay)
- come `Adam` + `weight_decay` (di solito `float`)

## 5) `Nadam(...)`
- come `Adam` + variante Nesterov (parametri simili, include `learning_rate`, `beta_1`, `beta_2`, `epsilon`)

## 6) `Adagrad(...)`
- `learning_rate`, `initial_accumulator_value`, `epsilon`

## 7) `Adadelta(...)`
- `learning_rate`, `rho`, `epsilon`

## 8) `Ftrl(...)` (più “classico” per sparse/linear models)
- `learning_rate`, `learning_rate_power`, `initial_accumulator_value`, `l1_regularization_strength`, `l2_regularization_strength`, ecc.

> Keras espone anche optimizer moderni (es. **Lion**, **Lamb**, …). Vanno bene, ma per partire “da zero” verso VAE/GAN/Transformer: **Adam/AdamW/SGD** coprono il 90% dei casi.

---

# Mega tabella E — Loss principali (top) + argomenti tipici

| Loss (classe / stringa) | Task | Argomenti tipici (quando usi la classe) |
|---|---|---|
| `MeanSquaredError` / `"mse"` | regressione, ricostruzione (AE) | `reduction`, `name` |
| `MeanAbsoluteError` / `"mae"` | regressione robusta | `reduction`, `name` |
| `BinaryCrossentropy` / `"binary_crossentropy"` | classificazione binaria, GAN | `from_logits: bool`, `label_smoothing: float`, `axis`, `reduction`, `name` |
| `CategoricalCrossentropy` / `"categorical_crossentropy"` | multiclasse one-hot | `from_logits`, `label_smoothing`, `axis`, … |
| `SparseCategoricalCrossentropy` / `"sparse_categorical_crossentropy"` | multiclasse label intere | `from_logits`, `axis`, … |
| `Huber` | regressione con outlier | `delta`, `reduction`, `name` |
| `KLDivergence` | VAE (parte KL) | `reduction`, `name` |

---

# Mega tabella F — Layer chiave per i tuoi obiettivi (Dense / Conv2D / LSTM / GRU / MultiHeadAttention)

## 1) `Dense(units, ...)`
Argomenti (principali e comuni):
- `units: int` (numero neuroni)
- `activation: str|callable|None` (es. `'relu'`, `'softmax'`, `None`)
- `use_bias: bool`
- `kernel_initializer`, `bias_initializer`
- `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`
- `kernel_constraint`, `bias_constraint`

## 2) `Conv2D(filters, kernel_size, ...)`
Argomenti principali:
- `filters: int`
- `kernel_size: int|tuple`
- `strides: int|tuple`
- `padding: 'valid'|'same'`
- `dilation_rate: int|tuple`
- `activation: str|callable|None`
- `use_bias: bool`
- `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, …

## 3) `LSTM(units, ...)`
Argomenti principali:
- `units: int`
- `activation`, `recurrent_activation`
- `return_sequences: bool`
- `return_state: bool`
- `dropout: float in [0,1]`
- `recurrent_dropout: float in [0,1]`
- `kernel_initializer`, `recurrent_initializer`, `bias_initializer`
- `go_backwards: bool`
- `stateful: bool`
- `unroll: bool`

## 4) `GRU(units, ...)`
Argomenti simili a LSTM:
- `units`, `activation`, `recurrent_activation`
- `return_sequences`, `return_state`
- `dropout`, `recurrent_dropout`
- `reset_after: bool` (importante per compatibilità/cuda)

## 5) `MultiHeadAttention(num_heads, key_dim, ...)`
Argomenti principali:
- `num_heads: int`
- `key_dim: int`
- `value_dim: int|None`
- `dropout: float`
- `use_bias: bool`
- `output_shape: int|tuple|None`
- `attention_axes: int|tuple|None`
- `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, …

---

# Esempi mini (per collegare i concetti)

## Regressione “lineare” (Keras)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1, input_shape=(d,))  # d = numero feature
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse",
              metrics=["mae"])
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2)
```

## Classificazione multiclasse
```python
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(d,)),
    layers.Dense(K, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

---

## Perché questa tabella ti serve per arrivare a VAE/GAN/Transformer
- VAE: devi dominare **loss multiple** (ricostruzione + KL), `Model` custom e training loop.
- GAN: devi dominare **2 modelli**, training alternato, stabilità optimizer (Adam/AdamW) e loss.
- Transformer: devi dominare `MultiHeadAttention`, maschere, residual, LayerNorm, FFN.

Nella Parte 3/3 (PyTorch) farò lo stesso: **modelli top 10 + opzioni + argomenti** ma nel paradigma `nn.Module`, `DataLoader`, `optim`, `loss`, `training loop`.
