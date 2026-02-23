# Lezione 16 — Mega tabella (Parte 3/3): PyTorch

> Obiettivo: una “mappa” pratica di **PyTorch** per arrivare a scrivere **VAE / GAN / Transformer** da solo:  
> **modelli/archetipi**, **moduli (layer)**, **loss**, **optimizer**, **scheduler**, **dataloader**, e soprattutto **training loop** (che in PyTorch è *il punto*).  
>
> Nota: PyTorch è molto flessibile. Non esistono “modelli pronti” come scikit-learn: costruisci tu un `nn.Module` combinando layer.  
> Per la **lista completa** (tutti gli argomenti reali nella tua versione) ti ho messo anche qui uno **script** che stampa le firme di: layer, loss, optimizer, scheduler.

---

## -1)IMPORTANTISSIMO DA RICORDARE

- Senza questa conversione quindi : 

```python
# 1) Convertiamo in tensori PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32) #la X sempre in float32
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

Y_train_t = torch.tensor(Y_train, dtype=torch.long) #il target sempre in long int
Y_test_t  = torch.tensor(Y_test, dtype=torch.long)

```
 - Il codice non gira se tutti i dati non vengono passati come tensori di pytorch quindi bisogna assolutamente controllarlo;

```python
X train tipo : <class 'numpy.ndarray'>
X test tipo : <class 'numpy.ndarray'>
Y train tipo : <class 'numpy.ndarray'>
Y test tipo : <class 'numpy.ndarray'>
X train tipo : <class 'torch.Tensor'>
X test tipo : <class 'torch.Tensor'>
Y train tipo : <class 'torch.Tensor'>
Y test tipo : <class 'torch.Tensor'>
```
## 0) I 10 modelli/archetipi PyTorch più importanti (per non andare all’infinito)

1. **Linear/Logistic**: `nn.Linear` (+ loss adatta)
2. **MLP**: stack di `nn.Linear` + attivazioni + dropout/norm
3. **CNN**: `nn.Conv2d` + pooling + norm + head
4. **RNN**: `nn.RNN`
5. **LSTM**: `nn.LSTM`
6. **GRU**: `nn.GRU`
7. **Autoencoder**: encoder + decoder
8. **VAE**: encoder → `mu, logvar` + reparameterization + decoder
9. **GAN**: generator + discriminator + training alternato
10. **Transformer**: `nn.Transformer` / `nn.MultiheadAttention` + embedding + pos encoding

---

# A) Mega tabella — “Blocchi” PyTorch: cosa sono e dove si configurano

| Blocco | Oggetto PyTorch | Dove lo “setti” | A cosa serve davvero |
|---|---|---|---|
| **Modello** | `torch.nn.Module` | definisci `__init__` (layer) e `forward` (flusso) | definisce la rete |
| **Parametri** | `model.parameters()` | automatico (tutti i layer registrati) | ciò che l’optimizer aggiorna |
| **Loss** | `torch.nn.*Loss` o funzioni (`F.*`) | scelto in base a task/output | misura l’errore |
| **Optimizer** | `torch.optim.*` | scelto in base a stabilità/velocità | aggiorna i pesi (grad step) |
| **Scheduler** | `torch.optim.lr_scheduler.*` | optional, spesso utile | cambia learning rate nel tempo |
| **Dataset** | `torch.utils.data.Dataset` | classe custom o dataset già pronto | come leggi un esempio |
| **DataLoader** | `torch.utils.data.DataLoader` | batch, shuffle, num_workers | crea batch e parallelizza I/O |
| **Device** | `torch.device('cpu'/'cuda')` | `.to(device)` | sposta tensori/modello su GPU |
| **AMP** | `torch.cuda.amp` | scaler + autocast | mixed precision (veloce su GPU) |
| **Checkpoint** | `torch.save(...)` / `torch.load(...)` | salva `state_dict` | riprendi training/inferenza |

---

# B) Mega tabella — Layer (nn modules) più usati + opzioni principali

> Non posso scrivere “tutti i layer possibili” in un messaggio senza diventare infinito, ma questa è la lista **pratica** che copre quasi tutto (e ti metto lo script per l’elenco completo + firme).

## 1) Core (MLP / FFN)
| Layer | Firma/concetto | Opzioni principali |
|---|---|---|
| `nn.Linear(in_features, out_features, bias=True)` | affine: `xW^T + b` | `bias` |
| `nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, ...)` | lookup embedding | `padding_idx`, `max_norm`, `norm_type`, `scale_grad_by_freq`, `sparse` |
| `nn.Dropout(p=0.5, inplace=False)` | dropout | `p`, `inplace` |
| `nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)` | norm per feature | `eps`, `elementwise_affine` |
| `nn.BatchNorm1d/2d/3d(num_features, eps=..., momentum=..., affine=True, track_running_stats=True)` | norm su batch | `eps`, `momentum`, `affine`, `track_running_stats` |
| `nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)` | norm per gruppi | `num_groups`, `eps`, `affine` |

## 2) Attivazioni
| Layer | Opzioni principali |
|---|---|
| `nn.ReLU(inplace=False)` | `inplace` |
| `nn.LeakyReLU(negative_slope=0.01, inplace=False)` | `negative_slope`, `inplace` |
| `nn.GELU(approximate='none')` | `approximate` |
| `nn.SiLU(inplace=False)` | `inplace` |
| `nn.Tanh()` | (nessuna opzione critica) |
| `nn.Sigmoid()` | (nessuna opzione critica) |
| `nn.Softmax(dim)` / `nn.LogSoftmax(dim)` | `dim` **obbligatorio** |

## 3) Convoluzioni (Vision / GAN)
| Layer | Firma base | Opzioni principali |
|---|---|---|
| `nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')` | conv 2D | `stride`, `padding`, `dilation`, `groups`, `bias`, `padding_mode` |
| `nn.ConvTranspose2d(...)` | “deconvolution”/upsampling learnable | stessi concetti + `output_padding` |
| `nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` | max pool | `kernel_size`, `stride`, `padding`, `ceil_mode` |
| `nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)` | avg pool | `count_include_pad`, `divisor_override` |
| `nn.AdaptiveAvgPool2d(output_size)` | output shape fisso | `output_size` |
| `nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)` | upsample non learnable | `mode`, `align_corners` |
| `nn.Flatten(start_dim=1, end_dim=-1)` | flatten | `start_dim`, `end_dim` |

## 4) Ricorrenti (sequenze)
| Layer | Opzioni principali |
|---|---|---|
| `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False)` | `num_layers`, `nonlinearity`, `batch_first`, `dropout`, `bidirectional` |
| `nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, proj_size=0)` | come RNN + `proj_size` |
| `nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)` | come RNN |

## 5) Attention / Transformer
| Layer | Opzioni principali |
|---|---|---|
| `nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False)` | `embed_dim`, `num_heads`, `dropout`, `batch_first`, `kdim/vdim` |
| `nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5, batch_first=False, norm_first=False)` | `d_model`, `nhead`, `num_*_layers`, `dim_feedforward`, `dropout`, `batch_first`, `norm_first` |
| `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, norm_first=False)` | blocco encoder |
| `nn.TransformerDecoderLayer(...)` | blocco decoder |

---

# C) Mega tabella — Loss (criteri) più importanti + argomenti principali

| Loss | Quando usarla | Argomenti principali / note |
|---|---|---|
| `nn.MSELoss(reduction='mean')` | regressione, ricostruzione AE | `reduction ∈ {'none','mean','sum'}` |
| `nn.L1Loss(reduction='mean')` | regressione robusta | `reduction` |
| `nn.SmoothL1Loss(beta=1.0, reduction='mean')` | Huber-like | `beta`, `reduction` |
| `nn.BCELoss(weight=None, reduction='mean')` | binaria **con output già sigmoid** | `weight`, `reduction` |
| `nn.BCEWithLogitsLoss(pos_weight=None, weight=None, reduction='mean')` | binaria **consigliata** (include sigmoid numericamente stabile) | `pos_weight` (class imbalance), `weight`, `reduction` |
| `nn.CrossEntropyLoss(weight=None, ignore_index=-100, label_smoothing=0.0, reduction='mean')` | multiclasse **consigliata** (include log-softmax) | `weight`, `ignore_index`, `label_smoothing`, `reduction` |
| `nn.NLLLoss(...)` | se tu fai già `LogSoftmax` | simile a CrossEntropy ma separata |
| `nn.KLDivLoss(reduction='batchmean', log_target=False)` | KL divergence (VAE / distill) | `reduction` importante (`batchmean` spesso usata), `log_target` |
| `nn.CosineEmbeddingLoss(...)` | metric learning | `margin`, `reduction` |

> Nota critica:  
> - Multiclasse: usa quasi sempre `CrossEntropyLoss` e **NON** applicare softmax nel modello (lo fa la loss).  
> - Binaria: preferisci `BCEWithLogitsLoss` e **NON** applicare sigmoid nel modello (lo fa la loss).

---

# D) Mega tabella — Optimizer (i più importanti) + argomenti principali

| Optimizer | Quando usarlo | Argomenti principali |
|---|---|---|
| `optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)` | baseline, CNN classiche, quando vuoi controllo | `lr` (obbl.), `momentum`, `weight_decay`, `nesterov` |
| `optim.Adam(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False)` | default robusto | `lr`, `betas`, `eps`, `weight_decay`, `amsgrad` |
| `optim.AdamW(params, lr=..., betas=..., eps=..., weight_decay=...)` | Transformers, weight decay “giusto” | come Adam + `weight_decay` usato bene |
| `optim.RMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)` | a volte su RNN/CNN | `alpha`, `centered`, `momentum` |
| `optim.Adagrad(...)` / `optim.Adadelta(...)` | casi particolari / sparse | parametri standard |

**Regola pratica per i tuoi obiettivi**
- **Transformer**: `AdamW` + scheduler warmup/decay
- **GAN** (DCGAN style): spesso `Adam(lr=2e-4, betas=(0.5, 0.999))`
- **VAE**: `Adam` / `AdamW` con LR moderato

---

# E) Mega tabella — LR Scheduler (utilissimi)

| Scheduler | Idea | Argomenti principali |
|---|---|---|
| `StepLR(optimizer, step_size, gamma=0.1)` | scende ogni N epoche | `step_size`, `gamma` |
| `MultiStepLR(optimizer, milestones, gamma=0.1)` | scende su epoche “milestone” | `milestones`, `gamma` |
| `ExponentialLR(optimizer, gamma)` | decrescita esponenziale | `gamma` |
| `CosineAnnealingLR(optimizer, T_max, eta_min=0)` | cosine decay | `T_max`, `eta_min` |
| `ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=..., cooldown=0, min_lr=0)` | scende se metric non migliora | tanti parametri “monitor” |
| `OneCycleLR(optimizer, max_lr, total_steps=..., epochs=..., steps_per_epoch=..., pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)` | ciclo LR (spesso efficace) | `max_lr`, `pct_start`, ecc. |

> Per Transformer spesso si usa warmup + cosine/linear (spesso implementato custom o con librerie).

---

# F) Data pipeline: Dataset e DataLoader (dove nascono batch/shuffle)

## Dataset (interfaccia)
Un dataset custom implementa:
- `__len__(self)` → numero esempi
- `__getitem__(self, idx)` → restituisce (x, y) o dict

## DataLoader
`DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, persistent_workers=False, prefetch_factor=2)`

Parametri chiave:
- `batch_size`: dimensione batch
- `shuffle`: mescola indici ogni epoca
- `num_workers`: parallelizza lettura/transform (su Windows spesso 0/2 per iniziare)
- `collate_fn`: come unisci esempi in batch (fondamentale per sequenze di lunghezza variabile)
- `pin_memory`: utile con GPU (trasferimenti più rapidi)
- `drop_last`: scarta ultimo batch incompleto

---

# G) Training loop “standard” (pattern che userai SEMPRE)

## 1) Setup
```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel(...).to(device)
criterion = nn.CrossEntropyLoss()          # es. classificazione multiclasse
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
```

## 2) Train step per batch
```python
model.train()                # abilita dropout, batchnorm in modalità train
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad(set_to_none=True)  # azzera gradienti (meglio per performance)
    logits = model(x)                      # forward
    loss = criterion(logits, y)            # calcolo loss
    loss.backward()                        # backprop: calcola grad
    optimizer.step()                       # aggiorna pesi
```

## 3) Eval loop
```python
model.eval()                 # disabilita dropout, usa running stats in BN
with torch.no_grad():        # evita tracking dei grad (più veloce e meno memoria)
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
```

### Perché questi 3 comandi sono “sacri”
- `model.train()` / `model.eval()` cambiano comportamento di **Dropout** e **BatchNorm**
- `torch.no_grad()` evita di salvare il grafo dei gradienti (memoria/velocità)

---

# H) Mixed precision (AMP) — utile su GPU moderne

```python
scaler = torch.cuda.amp.GradScaler()

model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = criterion(logits, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

# I) Checkpoint: salva e carica correttamente

## Salvataggio
```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}, "ckpt.pt")
```

## Caricamento
```python
ckpt = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1
```

---


# K) Perché PyTorch è cruciale per VAE/GAN/Transformer

- **VAE**: devi implementare *tu* la loss: `reconstruction_loss + KL(mu, logvar)` e spesso fai forward “multiplo” (mu/logvar/z).
- **GAN**: devi gestire *tu* due optimizer e alternare gli update, e spesso usare trucchi di stabilità.
- **Transformer**: devi gestire maschere, padding, causal mask, e training efficiente (AdamW + schedulers).

Quindi questa Parte 3/3 ti dà i mattoni e soprattutto il **pattern mentale**: *loop → grad → step → eval*.

