# 🧠 PyTorch Deep Dive: Manuale Completo per Fisici

Questo documento analizza **riga per riga** la costruzione di una rete neurale in PyTorch, spiegando la logica computazionale e i **parallelismi con i sistemi fisici**.

---

## 1) Il Cuore del Sistema: i Tensori 🧱

Il **tensore** è l’estensione del concetto di matrice a **n dimensioni**.

```python
import torch

# Creazione di un input sintetico
input_dati = torch.randn(1, 10)
```

### Analisi riga per riga

- `torch.randn(1, 10)`:
  - Genera un tensore di forma **(1 riga, 10 colonne)**.
  - I valori sono campionati da una distribuzione normale standard: \(\mathcal{N}(0, 1)\).

### Significato fisico (intuizione)
Puoi immaginare `input_dati` come **lo stato iniziale** di un sistema con **10 gradi di libertà** (10 variabili misurate/coordinate/feature).

---

## 2) Definizione del Modello (`nn.Module`) 🏗️

In PyTorch, un modello è una **classe** che eredita da `nn.Module`. Questo permette al framework di:
- registrare automaticamente i **parametri** (pesi e bias),
- gestire la **backpropagation**,
- abilitare metodi utili come `.parameters()`, `.train()`, `.eval()`, ecc.

```python
import torch.nn as nn
import torch.nn.functional as F

class MioModello(nn.Module):
    def __init__(self):
        super().__init__()  # Inizializza la classe base per registrare i layer

        # LAYER 1: Trasformazione Affine (10 -> 20)
        self.layer1 = nn.Linear(10, 20)

        # LAYER 2: Trasformazione Affine (20 -> 5)
        self.layer2 = nn.Linear(20, 5)
```

### Analisi dettagliata

- `class MioModello(nn.Module):`
  - definisce un modello come **oggetto** PyTorch “trainable”.
- `def __init__(self):`
  - costruttore: qui definisci i componenti (layer) che costituiscono il modello.
- `super().__init__()`
  - inizializza la parte `nn.Module`:
    - fondamentale per far sì che PyTorch **registri** automaticamente i layer assegnati come `self.qualcosa = ...`.

### Cosa fa `nn.Linear(in, out)`
`nn.Linear(in_features, out_features)` implementa:

\[
y = xW^T + b
\]

- \(W\) (**weights**): matrice dei pesi inizializzata casualmente.
  - interpreta l’**intensità delle connessioni** tra neuroni.
- \(b\) (**bias**): vettore di scostamento.
  - intuizione fisica: una **traslazione** (offset) del riferimento per ciascun neurone.

### Dimensioni (vincolo a cascata)

- `layer1 = nn.Linear(10, 20)` produce un output con **20 componenti**.
- Queste 20 componenti diventano l’input obbligatorio del layer successivo:
  - `layer2 = nn.Linear(20, 5)`.

È un sistema “a cascata” dove lo spazio delle variabili viene trasformato step-by-step.

---

## 3) Il Flusso dei Dati (`forward`) 🔄

Il metodo `forward` descrive come l’input **evolve** attraverso il sistema.

```python
    def forward(self, x):
        # Passo 1: Trasformazione lineare + Attivazione
        x = self.layer1(x)
        x = F.relu(x)

        # Passo 2: Trasformazione finale
        x = self.layer2(x)
        return x
```

### Analisi dettagliata

- `x = self.layer1(x)`
  - applica la trasformazione affine 10→20.
- `x = F.relu(x)`
  - applica la non-linearità:

\[
\mathrm{ReLU}(x) = \max(0, x)
\]

#### Perché è fondamentale?
Se componi solo trasformazioni lineari, ottieni ancora una trasformazione lineare:

\[
L_2(L_1(x)) = L_{comp}(x)
\]

Quindi senza ReLU (o altra attivazione) la rete **non può modellare** fenomeni non lineari complessi.

**Intuizione fisica:** la ReLU introduce una “rottura di simmetria” / non-linearità che permette al sistema di rappresentare dinamiche più ricche.

- `x = self.layer2(x)`
  - trasformazione finale 20→5.
- `return x`
  - restituisce l’output (spesso interpretato come predizione o “osservabile” stimata dal modello).

### Nessuna ReLU finale
Spesso l’ultimo layer viene lasciato lineare per permettere al modello di produrre **qualsiasi valore reale** (tipico della regressione).
Per classificazione, invece, potresti applicare softmax/sigmoid (o usare loss che include internamente la trasformazione).

---

## 4) Ottimizzazione e Loss Function 📉

Dobbiamo definire:
- una **funzione di costo** (quanto sbagliamo),
- un **ottimizzatore** (come correggiamo i parametri).

```python
modello = MioModello()
criterio = nn.MSELoss()
ottimizzatore = torch.optim.SGD(modello.parameters(), lr=0.01)
```

### Spiegazione

- `modello = MioModello()`
  - istanzia la rete (crea pesi e bias).
- `criterio = nn.MSELoss()`
  - Mean Squared Error:

\[
\mathrm{MSE} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
\]

È l’analogo del **principio dei minimi quadrati**.

- `modello.parameters()`
  - restituisce un iteratore su **tutti** i parametri trainabili (tutti i \(W\) e \(b\)).
- `torch.optim.SGD(..., lr=0.01)`
  - discesa del gradiente stocastica (SGD).
- `lr=0.01`
  - learning rate \(\eta\): passo con cui ti muovi lungo il gradiente.

---

## 5) Il Ciclo di Training (Analisi Passo-Passo) ⚙️

Ecco cosa succede in **un singolo step** di addestramento.

> Nota: qui `target` deve essere definito e deve avere shape compatibile con l’output del modello.

### 5.1 Reset dei gradienti

```python
optimizer.zero_grad()
```

**Perché?**  
In PyTorch i gradienti vengono **accumulati**:
\[
\text{grad} \leftarrow \text{grad} + \text{nuovo\_grad}
\]
Quindi prima di una nuova backward bisogna “pulire la lavagna”.

---

### 5.2 Forward Pass

```python
output = modello(input_dati)
loss = criterio(output, target)
```

- l’input fluisce nella rete → ottieni `output`;
- confronti `output` con `target` tramite la loss → ottieni un numero scalare `loss`.

---

### 5.3 Backward Pass (Backpropagation)

```python
loss.backward()
```

Qui avviene il cuore del calcolo differenziale:
- PyTorch attraversa il grafo computazionale al contrario.
- Usa la **regola della catena** (chain rule) per calcolare, per ogni parametro:

\[
\frac{\partial L}{\partial W}
\quad\text{e}\quad
\frac{\partial L}{\partial b}
\]

---

### 5.4 Aggiornamento dei parametri

```python
optimizer.step()
```

Applica la regola di aggiornamento (SGD):

\[
W \leftarrow W - \eta \cdot \nabla L
\]

I pesi e i bias vengono modificati per “scendere” verso il minimo della funzione di costo.
---

# Sezione: Il Training Loop con monitoraggio
```python
print(f"Inizio addestramento...")
for epoca in range(num_epoche):
    # 1. Forward Pass: Il modello genera una previsione
    predizione = modello(input_dati)
    
    # 2. Calcolo della Loss: Misuriamo l'errore
    loss = criterio(predizione, target)
    
    # 3. Pulizia dei gradienti: Reset per il nuovo calcolo
    optimizer.zero_grad()
    
    # 4. Backward Pass: Calcolo della pendenza (gradiente)
    loss.backward()
    
    # 5. Step dell'ottimizzatore: Aggiornamento dei pesi
    optimizer.step()
    
    # Stampa dei progressi
    if (epoca + 1) % 10 == 0:
        print(f'Epoca [{epoca+1}/{num_epoche}], Errore: {loss.item():.4f}')
```
---

## (Extra) Mini-esempio completo e coerente

Questo blocco rende il documento “eseguibile” al 100% (con `target` definito correttamente).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MioModello(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

# Dati
input_dati = torch.randn(1, 10)
target = torch.randn(1, 5)  # shape compatibile con output (1, 5)

# Modello + loss + ottimizzatore
modello = MioModello()
criterio = nn.MSELoss()
optimizer = torch.optim.SGD(modello.parameters(), lr=0.01)

# 1) reset grad
optimizer.zero_grad()

# 2) forward
output = modello(input_dati)
loss = criterio(output, target)

# 3) backward
loss.backward()

# 4) step
optimizer.step()

print("loss:", loss.item())
```

---

## Struttura consigliata per una repo GitHub

Se vuoi trasformarlo in una repo pulita:

```
pytorch-deep-dive-fisici/
├─ README.md
└─ examples/
   └─ training_step.py
```

---

_Fine documento._
