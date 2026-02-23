# Catalogo layer Keras (Keras 3 / tf.keras) + opzioni principali

> Nota: Keras è enorme. Qui trovi (1) l'elenco dei layer per categoria **con le opzioni più importanti** e (2) uno **script** che ti genera automaticamente un file con **tutte** le firme (tutti gli argomenti) per **la tua versione installata**.

---

## Parametri “base” comuni a quasi tutti i layer
Questi compaiono come `**kwargs` o parametri “standard” del costruttore:
- `name` (nome del layer)
- `dtype` (tipo numerico)
- `trainable` (se i pesi si aggiornano)
- spesso anche altri flag interni/di backend

---

## Core layers
- **Input / Input object**: `shape`, `batch_size`, `dtype`, `name`
- **InputSpec**: vincoli su forma/dtype/rank di un input
- **Dense**: `units`, `activation`, `use_bias`, `kernel_initializer`, `bias_initializer`, `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`, `kernel_constraint`, `bias_constraint`
- **EinsumDense**: `equation`, `output_shape`, `activation`, (initializer/regularizer/constraint)
- **Activation**: `activation`
- **Embedding**: `input_dim`, `output_dim`, `mask_zero`, `embeddings_initializer`, `embeddings_regularizer`, `embeddings_constraint`
- **Masking**: `mask_value`
- **Lambda**: `function`, `output_shape`, `mask`, `arguments`
- **Identity**: nessuna opzione “di rete”, solo parametri base

---

## Convolution layers
Layer: `Conv1D`, `Conv2D`, `Conv3D`, `SeparableConv1D`, `SeparableConv2D`, `DepthwiseConv1D`, `DepthwiseConv2D`, `Conv1DTranspose`, `Conv2DTranspose`, `Conv3DTranspose`

Opzioni principali (quasi sempre):
- `filters` (tranne depthwise), `kernel_size`, `strides`, `padding`, `dilation_rate`, `activation`, `use_bias`
- `data_format` (`channels_last` / `channels_first`)
- initializer/regularizer/constraint

---

## Pooling layers
Layer: `MaxPooling1D/2D/3D`, `AveragePooling1D/2D/3D`,
`GlobalMaxPooling1D/2D/3D`, `GlobalAveragePooling1D/2D/3D`,
`AdaptiveAveragePooling1D/2D/3D`, `AdaptiveMaxPooling1D/2D/3D`

Opzioni principali:
- pooling “classico”: `pool_size`, `strides`, `padding`, `data_format`
- global pooling: spesso `data_format`, `keepdims`
- adaptive pooling: target/output size (varia per layer)

---

## Recurrent layers
Layer: `LSTM`, `LSTMCell`, `GRU`, `GRUCell`, `SimpleRNN`, `SimpleRNNCell`,
`BaseRNN`, `StackedRNNCells`, `TimeDistributed`, `Bidirectional`,
`ConvLSTM1D/2D/3D`

Opzioni principali:
- `units`, `activation`, `recurrent_activation`, `dropout`, `recurrent_dropout`
- `return_sequences`, `return_state`, `stateful`, `go_backwards`, `unroll`
- initializer/regularizer/constraint

---

## Attention layers
Layer: `GroupQueryAttention`, `MultiHeadAttention`, `Attention`, `AdditiveAttention`

Opzioni principali (soprattutto in Transformer):
- `num_heads`, `key_dim`, `value_dim`, `dropout`
- `attention_axes`, (eventuale) `flash_attention`
- initializer/regularizer/constraint

---

## Normalization layers
Layer: `BatchNormalization`, `LayerNormalization`, `UnitNormalization`, `GroupNormalization`, `RMSNormalization`

Opzioni principali:
- assi e numerica: `axis`, `epsilon`
- `momentum` (batch norm), `center`, `scale`
- initializer/regularizer/constraint per `beta/gamma` (se presenti)

---

## Regularization layers
Layer: `Dropout`, `SpatialDropout1D/2D/3D`, `GaussianDropout`, `AlphaDropout`, `GaussianNoise`, `ActivityRegularization`

Opzioni principali:
- dropout: `rate`, `noise_shape`, `seed`
- gaussian noise: `stddev`, `seed`
- activity regularization: `l1`, `l2` (o parametri equivalenti)

---

## Reshaping layers
Layer: `Reshape`, `Flatten`, `RepeatVector`, `Permute`,
`Cropping1D/2D/3D`, `UpSampling1D/2D/3D`, `ZeroPadding1D/2D/3D`

Opzioni principali:
- `target_shape` (Reshape), `data_format`
- upsampling: `size`, `interpolation` (dove previsto)
- cropping/padding: entità del taglio/padding

---

## Merging layers
Layer: `Concatenate`, `Average`, `Maximum`, `Minimum`, `Add`, `Subtract`, `Multiply`, `Dot`

Opzioni principali:
- `axis` (concatenate)
- `axes`/`normalize` (dot)
- il resto è tipicamente “strutturale” (lista di tensori in input)

---

## Activation layers
Layer: `ReLU`, `Softmax`, `LeakyReLU`, `PReLU`, `ELU`

Opzioni principali:
- ReLU: `max_value`, `negative_slope`, `threshold`
- Softmax: `axis`
- LeakyReLU: `negative_slope`
- PReLU: initializer/constraint su `alpha`
- ELU: `alpha`

---

## Backend-specific layers
Layer: `TorchModuleWrapper`, `TensorflowSavedModelLayer`, `JaxLayer`, `FlaxLayer`
Opzioni principali: dipendono dal backend (wrappano moduli esterni).

---

## Preprocessing layers
### Testo
- `TextVectorization`: tokenizzazione e vettorizzazione (vocabolario, output_mode, sequence_length, standardize, split, ngrams, ecc.)

### Numerico
- `Normalization`, `SpectralNormalization`, `Discretization`

### Categoriale
- `CategoryEncoding`, `Hashing`, `HashedCrossing`, `StringLookup`, `IntegerLookup`

### Immagini (preprocess)
- `Resizing`, `Rescaling`, `CenterCrop`, `AutoContrast`

### Immagini (augmentation)
- `AugMix`, `CutMix`, `Equalization`, `MaxNumBoundingBoxes`, `MixUp`, `Pipeline`, `RandAugment`,
  `RandomBrightness`, `RandomColorDegeneration`, `RandomColorJitter`, `RandomContrast`, `RandomCrop`,
  `RandomElasticTransform`, `RandomErasing`, `RandomFlip`, `RandomGaussianBlur`, `RandomGrayscale`,
  `RandomHue`, `RandomInvert`, `RandomPerspective`, `RandomPosterization`, `RandomRotation`,
  `RandomSaturation`, `RandomSharpness`, `RandomShear`, `RandomTranslation`, `RandomZoom`, `Solarization`

### Audio
- `MelSpectrogram`, `STFTSpectrogram`

---

