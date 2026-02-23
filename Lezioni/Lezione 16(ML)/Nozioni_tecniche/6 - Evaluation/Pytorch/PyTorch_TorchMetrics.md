# TorchMetrics in PyTorch -- Valutazione Senza DataLoader

## Introduzione

Possiamo immaginare torchmetrics come un registro contabile: non importa
se i dati arrivano tutti insieme o a blocchi, la metrica accumula le
informazioni e calcola il risultato finale quando viene chiamato
compute().

Questo approccio è particolarmente utile quando i dati di test sono già
disponibili in un unico blocco (x_test, y_test).

------------------------------------------------------------------------

# 1. Inizializzazione delle Metriche

Prima della fase di evaluation, prepariamo gli strumenti. È importante
specificare il tipo di task e il numero di classi.

``` python
import torchmetrics

metriche = torchmetrics.MetricCollection([
    torchmetrics.Accuracy(task="multiclass", num_classes=5),
    torchmetrics.F1Score(task="multiclass", num_classes=5)
])
```

Qui stiamo creando una collezione di metriche che verranno aggiornate
insieme.

------------------------------------------------------------------------

# 2. Fase di Valutazione (Senza DataLoader)

Se abbiamo già x_test e y_test pronti:

``` python
modello.eval()

with torch.no_grad():
    output = modello(x_test)
    metriche.update(output, y_test)
```

Spiegazione:

-   model.eval() → disattiva Dropout e BatchNorm dinamico
-   torch.no_grad() → disattiva Autograd
-   metriche.update() → aggiorna lo stato interno della metrica

Le metriche accumulano i risultati internamente.

------------------------------------------------------------------------

# 3. Calcolo Finale

Quando vogliamo ottenere i risultati:

``` python
risultati = metriche.compute()

print(f"Accuracy: {risultati['MulticlassAccuracy']:.4f}")
print(f"F1-Score: {risultati['MulticlassF1Score']:.4f}")

metriche.reset()
```

compute() restituisce un dizionario con i risultati finali. reset()
pulisce lo stato interno per riutilizzare le metriche.

------------------------------------------------------------------------

# Struttura Completa Integrata

``` python
# --- BLOCCO 1: MODELLO ---
modello = MioModello().to(device)
criterio = nn.CrossEntropyLoss()
ottimizzatore = torch.optim.Adam(modello.parameters())

# --- BLOCCO 2: METRICHE ---
metriche = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=5
).to(device)

# --- BLOCCO 3: EVALUATION ---
modello.eval()

with torch.no_grad():
    output = modello(x_test)
    metriche.update(output, y_test)
```

------------------------------------------------------------------------

# Perché usare torchmetrics?

1.  Riduce errori matematici
2.  Codice più pulito e modulare
3.  Facile estensione (aggiungere Precision, Recall, ecc.)
4.  Compatibile con GPU (.to(device))

------------------------------------------------------------------------

# Differenza rispetto al calcolo manuale

Senza torchmetrics: - Devi gestire contatori manualmente - Devi
implementare formule matematiche - Rischio di errori nei dataset
sbilanciati

Con torchmetrics: - Stato interno gestito automaticamente - Calcolo
corretto per task multiclass, multilabel, binary - API coerente e
riutilizzabile

------------------------------------------------------------------------

# Conclusione

TorchMetrics è uno strumento modulare e professionale per la valutazione
dei modelli PyTorch.

Anche senza DataLoader, è perfettamente utilizzabile quando si dispone
dell'intero test set in memoria.

Struttura corretta:

1)  model.eval()
2)  with torch.no_grad()
3)  metriche.update()
4)  metriche.compute()
5)  metriche.reset()
