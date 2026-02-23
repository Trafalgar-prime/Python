In PyTorch, il metodo più comune e raccomandato non è salvare l'intero oggetto Python del modello, ma salvare il suo state_dict.

## Cos'è lo state_dict? 
- Lo state_dict è un semplice dizionario Python che mappa ogni livello (layer) del modello ai suoi parametri addestrabili (pesi e bias). Immaginalo come una "scheda tecnica" che contiene tutti i numeri magici che la tua rete ha imparato.

Ecco i tre pilastri del salvataggio:

- **Salvataggio dei pesi**: Si usa torch.save().

- **Caricamento dei pesi**: Si usa load_state_dict().

- **Checkpoint**: Salvare non solo il modello, ma anche lo stato dell'ottimizzatore e il numero dell'epoca corrente. ⏱️

Procediamo per gradi 🪜
Supponiamo che tu abbia appena finito di addestrare il tuo modello. Per salvare solo i parametri, useresti questo comando:


```python
torch.save(modello.state_dict(), 'modello_pesi.pth')

```


# PyTorch: Training Completo + Salvataggio e Caricamento dei Pesi

Di seguito trovi un esempio completo che integra:

-   Transformations e Augmentation (`torchvision.transforms`)
-   Dataset custom (`torch.utils.data.Dataset`)
-   DataLoader
-   Modello (`nn.Module`)
-   Training + Evaluation (`model.train()` / `model.eval()` +
    `torch.no_grad()`)
-   Metriche con `torchmetrics`
-   Salvataggio del modello (`torch.save(state_dict)`)
-   Procedura corretta di caricamento pesi (`load_state_dict`)

------------------------------------------------------------------------

## Codice Completo

``` python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchmetrics

# --- 1. CONFIGURAZIONE E TRASFORMAZIONI ---
# Definiamo le trasformazioni per il training (Augmentation) e per il test
train_transforms = transforms.Compose([
    transforms.ToPILImage(),        # Necessario per alcune trasformazioni visive
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation: ribalta a caso
    transforms.ToTensor(),          # Converte in Tensore e scala 0-1
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizza
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- 2. IL DATASET CUSTOM ---
class MioDataset(Dataset):
    def __init__(self, num_samples, transform=None):
        # Simuliamo delle immagini (3 canali, 32x32) e delle etichette (0, 1 o 2)
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 3, (num_samples,))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Applichiamo le trasformazioni se definite
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# Creiamo le istanze del dataset
train_data = MioDataset(num_samples=100, transform=train_transforms)
test_data = MioDataset(num_samples=20, transform=test_transforms)

# Creiamo i DataLoader (i corrieri)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# --- 3. IL MODELLO ---
class MioModello(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 canali (RGB) * 32 * 32. Output: 3 classi
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3) # 3 neuroni in uscita per 3 classi

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x # Ritorniamo i Logits (senza Softmax)

# Inizializziamo Modello, Loss, Ottimizzatore e Metriche
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modello = MioModello().to(device)
criterion = nn.CrossEntropyLoss() # Include la LogSoftmax
optimizer = optim.Adam(modello.parameters(), lr=0.001)

# Metrica di accuratezza
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)

# --- 4. CICLO DI TRAINING E VALIDAZIONE ---
num_epochs = 2

for epoch in range(num_epochs):
    print(f"\n--- Epoca {epoch+1} ---")
    
    # --- FASE DI TRAIN ---
    modello.train() # Attiva dropout/batchnorm
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # 1. Forward
        preds = modello(x_batch)
        loss = criterion(preds, y_batch)
        
        # 2. Backward
        optimizer.zero_grad() # Pulisce i gradienti vecchi
        loss.backward()       # Calcola i nuovi gradienti
        optimizer.step()      # Aggiorna i pesi
        
    print(f"Loss di Training finale: {loss.item():.4f}")

    # --- FASE DI VALUTAZIONE ---
    modello.eval() # Disattiva dropout/batchnorm
    with torch.no_grad(): # Spegne il calcolo dei gradienti (risparmia memoria)
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            val_preds = modello(x_val)
            
            # Aggiorniamo la metrica (accumulo dati)
            accuracy_metric.update(val_preds, y_val)
            
    # Calcoliamo l'accuratezza totale sull'epoca
    val_acc = accuracy_metric.compute()
    print(f"Accuratezza di Validazione: {val_acc:.4f}")
    accuracy_metric.reset() # Reset per la prossima epoca

# --- 5. SALVATAGGIO ---
print("\nSalvataggio del modello...")
torch.save(modello.state_dict(), 'mio_modello_completo.pth')
print("Fatto!")
```

------------------------------------------------------------------------

## Salvataggio vs Caricamento in PyTorch

Per ricaricare i dati (i pesi) in PyTorch, dobbiamo prima avere
un'istanza del modello pronta ad accoglierli.

PyTorch non salva l'intera "scatola" (il codice Python della classe), ma
solo il "contenuto" (i numeri dei parametri).\
Per questo motivo, il processo segue solitamente questi passaggi:

1.  **Definizione dell'architettura**: Creiamo un'istanza della classe
    del modello (es. `modello = MiaRete()`).
2.  **Caricamento del dizionario**: Usiamo `torch.load()` per leggere il
    file dal disco.
3.  **Iniezione dei pesi**: Usiamo `load_state_dict()` per copiare quei
    parametri dentro l'oggetto modello.
4.  **Modalità evaluation**: `model.eval()` prima di test/inferenza.

------------------------------------------------------------------------

## Codice di Caricamento dei Pesi

``` python
# 1. Ricreiamo la struttura (deve essere identica a quella usata per il salvataggio)
modello = MiaRete()  # 🏗️

# 2. Carichiamo lo stato salvato
stato_salvato = torch.load('modello_pesi.pth')  # 📂

# 3. Carichiamo i pesi nel modello
modello.load_state_dict(stato_salvato)  # 💉

# 4. Fondamentale per la valutazione!
modello.eval()  # 🧊
```

------------------------------------------------------------------------

## Perché dobbiamo ricreare la struttura? 🧠

Immagina lo `state_dict` come una lista di istruzioni che dice:

-   "Il layer chiamato `fc1` deve avere questi valori"
-   "Il layer chiamato `fc2` deve avere questi altri"

Se provi a caricare questi valori in un modello diverso (ad esempio uno
che ha solo un layer invece di due), PyTorch non saprà dove mettere i
parametri mancanti o avanzati e restituirà un errore di mismatch.


# Strutturare un Progetto PyTorch in Più File

Hai ragione: per costruire l'istanza

modello = MioModello()

Python deve assolutamente sapere cosa sia `MioModello`.\
Se non trova la "planimetria", restituirà un errore:

NameError: name 'MioModello' is not defined

------------------------------------------------------------------------

Nel mondo reale, raramente teniamo tutto in un unico file gigante
(diventerebbe illegibile!).\
La soluzione standard è dividere il codice in più file e collegarli.

------------------------------------------------------------------------

## Struttura tipica di un progetto reale

### 1️⃣ Il file della "Planimetria" (model.py)

Qui metti solo la definizione della classe.\
È il tuo archivio dell'architettura.

``` python
# File: model.py
import torch.nn as nn

class MioModello(nn.Module):
    def __init__(self):
        super().__init__()
        # ... definizione layer ...
    
    def forward(self, x):
        # ... forward ...
        return x
```

Questo file contiene solo la struttura della rete.

------------------------------------------------------------------------

### 2️⃣ Il file dell'esecuzione (main.py o inference.py)

Qui è dove carichi e usi il modello.\
Per rendere visibile la classe che sta nell'altro file, usiamo un
"ponte": l'import.

``` python
# File: main.py
import torch

# IL PONTE: Importiamo la classe dal file 'model'
from model import MioModello 

# Ora Python "conosce" la classe, anche se è scritta altrove!
modello = MioModello() 
modello.load_state_dict(torch.load('pesi.pth'))
modello.eval()
```

In questo modo:

-   `model.py` contiene solo l'architettura
-   `main.py` contiene esecuzione o inferenza
-   eventuali altri file possono contenere training, utilities, dataset,
    ecc.

------------------------------------------------------------------------

## Perché questa struttura è importante?

Separare i file permette di:

-   mantenere il codice leggibile
-   evitare duplicazioni
-   riutilizzare facilmente l'architettura
-   lavorare in team in modo ordinato

------------------------------------------------------------------------

## Caso Notebook (Jupyter / Colab)

Se stai lavorando in un Notebook:

-   Tutto è nello stesso documento
-   Le celle vengono eseguite in sequenza

Finché esegui prima la cella con la definizione della classe e poi
quella che istanzia il modello, non avrai problemi.

Ma in un progetto professionale, la divisione in file separati è la
pratica standard.
