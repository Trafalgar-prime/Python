# PyTorch -- Training vs Evaluation Mode

## Introduzione

Passare dalla fase di addestramento (training) a quella di valutazione
(evaluation) è un momento critico: è qui che verifichiamo se il modello
ha davvero imparato le regole del problema oppure se ha semplicemente
memorizzato i dati (overfitting).

------------------------------------------------------------------------

# 1. model.eval()

Il comando:

    model.eval()

mette il modello in modalità valutazione.

Questo è fondamentale perché alcuni layer si comportano in modo diverso
tra training ed evaluation.

## Layer coinvolti

### Dropout

Durante il training spegne neuroni casualmente per migliorare la
generalizzazione. In evaluation deve essere completamente disattivato.

### Batch Normalization

Durante il training aggiorna le statistiche (media e varianza). In
evaluation utilizza le statistiche apprese durante l'addestramento.

------------------------------------------------------------------------

# 2. with torch.no_grad()

Il blocco:

    with torch.no_grad():

disattiva il motore di calcolo dei gradienti (Autograd).

## Perché è fondamentale?

### 1. Efficienza computazionale

Senza no_grad, PyTorch costruisce il grafo di calcolo anche durante il
test, occupando più memoria e tempo di calcolo.

### 2. Integrità scientifica

Evita che il modello "impari" dai dati di test. Se accidentalmente si
chiama backward() in evaluation senza no_grad, i gradienti si
sommerebbero a quelli del training.

------------------------------------------------------------------------

# Struttura tipica del blocco di evaluation

``` python
model.eval()

with torch.no_grad():
    predizioni = model(dati_test)
    val_loss = criterio(predizioni, target_test)

print(f"Loss di Validazione: {val_loss.item():.4f}")
```

------------------------------------------------------------------------

# Differenza tra model.train() e model.eval()

model.train(): - Attiva comportamento stocastico (Dropout attivo) -
BatchNorm aggiorna statistiche

model.eval(): - Disattiva Dropout - Congela BatchNorm

Importante: questi comandi non eseguono l'addestramento, ma modificano
il comportamento interno dei layer.

------------------------------------------------------------------------

# Calcolo dell'Accuracy

Nei problemi di classificazione, oltre alla loss, si calcola l'accuracy.

## 1. Ottenere la classe predetta

``` python
previsioni_classi = torch.argmax(output, dim=1)
```

Oppure:

``` python
_, predizioni = torch.max(output, 1)
```

## 2. Confronto con i target

``` python
corrette = (previsioni_classi == target).sum().item()
accuracy = corrette / len(target)
```

------------------------------------------------------------------------

# Blocco completo di evaluation con DataLoader

``` python
model.eval()

totale_corrette = 0
totale_campioni = 0

with torch.no_grad():
    for dati, target in loader_test:
        output = model(dati)
        
        _, predizioni = torch.max(output, 1)
        
        totale_campioni += target.size(0)
        totale_corrette += (predizioni == target).sum().item()

accuracy_finale = 100 * totale_corrette / totale_campioni
print(f'Accuracy sul set di test: {accuracy_finale:.2f}%')
```

------------------------------------------------------------------------

# Intuizione concettuale

Training = fase in cui modifichi il sistema (aggiorni i pesi).
Evaluation = fase in cui osservi senza alterare lo stato.

Durante evaluation: - niente aggiornamento pesi - niente costruzione del
grafo - comportamento deterministico dei layer

------------------------------------------------------------------------

# Conclusione

Una corretta fase di evaluation richiede SEMPRE:

1)  model.eval()
2)  with torch.no_grad()

Se manca uno dei due, i risultati possono essere: - instabili -
inefficienti - scientificamente scorretti
