# PyTorch Transformations & Augmentation

Iniziare dalle Transformations e dall'Augmentation è fondamentale
perché, come si dice in informatica, "Garbage In, Garbage Out" (se i
dati in ingresso sono scadenti, il modello sarà scadente). 🗑️➡️📉

In fisica, questo concetto è molto potente: se addestriamo una rete a
riconoscere un segnale, vogliamo che sia in grado di riconoscerlo anche
se c'è del rumore o se il sensore è leggermente inclinato.
L'Augmentation consiste nel creare artificialmente nuove varianti dei
dati esistenti per rendere il modello più robusto. 🛡️

In PyTorch, usiamo principalmente la libreria torchvision.transforms.

------------------------------------------------------------------------

## Il concetto di "Pipeline" 🛠️

Le trasformazioni vengono solitamente raggruppate in una "pipeline"
usando transforms.Compose. Questa catena di operazioni viene applicata a
ogni dato che passa per il modello.

Le trasformazioni si dividono in due categorie:

### Obbligatorie

Come ToTensor(), che trasforma un'immagine o un array in un Tensore di
PyTorch e ne scala i valori (solitamente da \[0, 255\] a \[0, 1\]). 📏

### Di Augmentation

Come rotazioni, tagli (crop), o variazioni di colore, usate per
"disturbare" il modello e costringerlo a imparare le caratteristiche
davvero importanti. 🌀

------------------------------------------------------------------------

## Esempio pratico: Classificazione di immagini

``` python
from torchvision import transforms

transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),      
    transforms.RandomRotation(20),      
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),              
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

------------------------------------------------------------------------

## La Normalizzazione: un tocco di Fisica ⚛️

La funzione Normalize sottrae la media (μ) e divide per la deviazione
standard (σ):

z = (x - μ) / σ

Questo serve a centrare i dati attorno allo zero, facilitando
enormemente il lavoro dell'ottimizzatore durante la discesa del
gradiente.

------------------------------------------------------------------------

Le trasformazioni si applicano solitamente all'interno della classe
Dataset.

------------------------------------------------------------------------

## 1. Trasformazioni di Colore e Contrasto 🎨

ColorJitter: Cambia casualmente luminosità, contrasto e saturazione.

Grayscale: Converte in bianco e nero (utile se il colore non è
un'informazione rilevante per il task).

------------------------------------------------------------------------

## 2. Trasformazioni Geometriche Avanzate 📐

RandomResizedCrop: Ritaglia una parte casuale dell'immagine e la
ridimensiona.

RandomAffine: Permette di traslare, scalare e shear dell'immagine.

------------------------------------------------------------------------

## 3. Trasformazioni Specifiche per Tensori 🔢

RandomErasing: Cancella un rettangolo casuale dell'immagine.

------------------------------------------------------------------------

## Esempio di Pipeline Professionale 🏗️

``` python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

Nota: Nel val_transforms non usiamo trasformazioni casuali.

------------------------------------------------------------------------

## CPU Bottleneck e DataLoader

In molti casi reali, la CPU diventa il collo di bottiglia perché deve
leggere i dati dal disco, decodificarli e applicare le trasformazioni,
mentre la GPU rimane in attesa.

PyTorch introduce il DataLoader per caricare e trasformare i dati in
parallelo usando più worker.

------------------------------------------------------------------------

## Trasformazioni Deterministiche vs Casuali 🎲

Deterministiche: Resize, ToTensor.

Casuali: RandomRotation, RandomHorizontalFlip.

------------------------------------------------------------------------

## Normalizzazione Specifica per Dominio 🧪

Media: \[0.485, 0.456, 0.406\]

Deviazione Standard: \[0.229, 0.224, 0.225\]

------------------------------------------------------------------------

## Trasformazioni Custom 🛠️

È possibile creare classi Python personalizzate per manipolare i dati in
modo specifico.

------------------------------------------------------------------------

## Categorie principali in torchvision.transforms

### 1. Trasformazioni Geometriche 📐

Resize\
RandomRotation\
RandomHorizontalFlip / VerticalFlip\
CenterCrop / RandomCrop

### 2. Trasformazioni di Colore 🎨

ColorJitter\
Grayscale\
GaussianBlur

### 3. Trasformazioni di Conversione 🧪

ToTensor\
Normalize\
RandomErasing

ToTensor converte da PIL/numpy a torch.Tensor e scala da \[0, 255\] a
\[0, 1\].

Normalize applica la formula:

z = (x - μ) / σ
