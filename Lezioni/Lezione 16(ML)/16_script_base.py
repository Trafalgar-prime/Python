import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Importa il dataset Iris

# Caricare un dataset (Esempio: Iris dataset)
dati = load_iris()  # Carica i dati di Iris
#Questa funzione carica in memoria il dataset Iris e lo restituisce come un oggetto “contenitore” di scikit-learn.
"""
Che tipo di oggetto è dati?
È un oggetto chiamato spesso Bunch (simile a un dizionario, ma con accesso anche “a punti”).
Contiene varie cose utili, ad esempio:
dati.data → matrice con le feature (numeri)
dati.feature_names → nomi delle colonne
dati.target → etichette/classi (0,1,2)
(spesso anche dati.target_names, dati.DESCR, ecc.)
"""

# Creiamo un DataFrame Pandas con i dati e aggiungiamo la colonna "target"
df = pd.DataFrame(dati.data, columns=dati.feature_names)
pd.set_option('display.max_rows', None) #serve per vedere tutti i dati e non solo i parziali in output, perche altrimenti pandas mi stampa solo i parziali per non intasare l'output
#pd.set_option('display.max_columns', None) #stesso del precedente ma per le colonne [in questo caso è inutile perche le colonne sono solo 4]
print(df)
print("\nProva\n")
print(df.shape)
print(df.columns)
print("\n")

"""
df = pd.DataFrame(dati.data, columns=dati.feature_names)

Questa è una riga chiave: stai costruendo un DataFrame.
pd.DataFrame(...): crea una tabella pandas.

Primo argomento: dati.data
È una struttura tipo array (di solito un numpy.ndarray) con forma:
(numero_campioni, numero_feature)
Nel dataset Iris tipicamente: 150 righe × 4 colonne.
Contiene i valori numerici delle misure dei fiori (lunghezze/larghezze).

Secondo argomento: columns=dati.feature_names
columns= imposta i nomi delle colonne del DataFrame.
dati.feature_names è una lista di stringhe (es. “sepal length (cm)”, ecc.)

Assegnazione: il DataFrame creato viene salvato nella variabile df.
Risultato di questa riga
df ora è una tabella con:
righe = esempi (fiori)
colonne = caratteristiche numeriche (feature)
"""
df.index.name = "id" # la colonna degli indici la stampa pandas in automatico, in questo modo gli do un nome
df['target'] = dati.target

"""
Qui aggiungi una nuova colonna al DataFrame.

df['target']: seleziona (o crea) la colonna chiamata "target".
Se non esiste, pandas la crea.
= dati.target: assegna a quella colonna l’array delle etichette.
dati.target contiene, per ogni riga del dataset, la classe del fiore (ad es. 0, 1, 2).

Regola importante: pandas allinea gli elementi per indice di riga.
Se df ha 150 righe, anche dati.target deve avere 150 valori (uno per riga), altrimenti errore.

Perché farlo?
Così hai feature + etichetta nella stessa tabella:
feature → input
target → output/label per classificazione
"""

#Mostriamo le prime 5 righe del dataset
print(df.head())
print("\n",df,"\n")
print("\n")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
#.iloc permette di definire le posizioni di quello di cui ho bisogno in base alle colonne

"""
Questa è una riga densissima. La spacchiamo.

✅ df.iloc[:, :-1]

df è un DataFrame pandas.
.iloc indica indicizzazione posizionale (per indice numerico, non per nome).
[:, :-1] significa:
: = tutte le righe
:-1 = tutte le colonne tranne l’ultima
Quindi stai prendendo le feature (X), assumendo che:
l’ultima colonna non sia una feature
e che la colonna target sia separata (infatti usi df['target']) quindi non la considero.
⚠️ Nota importante: tu poi prendi anche df['target'], quindi stai implicitamente assumendo che:
la colonna target è presente
e che non sia l’ultima colonna (oppure anche se lo fosse, la stai comunque escludendo con :-1, ma poi la riprendi correttamente con df['target'])
Se il tuo DataFrame è come nell’output:
4 feature + target come 5ª colonna
allora df.iloc[:, :-1] prende le 4 feature, perfetto.

✅ df['target']
Seleziona la colonna target come Series pandas.
y deve essere un vettore di etichette (0,1,2 nell’Iris).

✅ train_test_split(X, y, test_size=0.2, random_state=42)
Divide X e y in modo coerente: la stessa permutazione casuale viene applicata a entrambi.

Parametri:
test_size=0.2 → 20% dei dati nel test

Se hai 150 righe, test ≈ 30 righe
train ≈ 120 righe
random_state=42 → “seme” del generatore casuale
significa: ogni volta che esegui, la divisione sarà identica
utile per replicabilità (fondamentale nello studio)
Output della funzione

Ritorna 4 oggetti nello stesso ordine:
X_train (features training)
X_test (features test)
y_train (label training)
y_test (label test)
"""
# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
"""
scaler = StandardScaler()
Crei un oggetto scaler vuoto, non ha ancora “imparato” nulla.
Al suo interno ci saranno parametri da stimare:
mean_ (media per ogni feature)
scale_ (deviazione standard per ogni feature)
Questi parametri devono essere calcolati SOLO sul training set.
"""
#scaler = MinMaxScaler()
"""
#scaler = MinMaxScaler()
È commentata → non viene eseguita.
Se la usassi al posto di StandardScaler, cambierebbe la distribuzione dei dati:
StandardScaler: centra e scala ma lascia range libero
MinMax: schiaccia tutto in [0,1]
Per MLP spesso vanno bene entrambe, ma StandardScaler è la scelta più comune.
"""
X_train = scaler.fit_transform(X_train)
"""
Questa riga fa due cose in una:

1) fit(X_train)
Lo scaler “impara” dai dati:
calcola media e deviazione standard di ogni colonna (feature) usando SOLO X_train
Formalmente:
se X_train ha shape (120, 4) nell’Iris
calcola 4 medie e 4 deviazioni standard

2) transform(X_train)
Applica la trasformazione:
ad ogni valore, sottrae la media e divide per la deviazione standard della feature.

Risultato finale
X_train diventa un array numpy (non più DataFrame pandas).
Questo è importante:
prima era tipo: pandas.DataFrame
dopo: numpy.ndarray
"""
X_test = scaler.transform(X_test)
"""
Qui non fai fit, fai solo transform.

Perché è cruciale:
se facessi fit_transform anche su test, useresti informazioni del test per normalizzare → data leakage(si manifesta quando informazioni presenti nel set di addestramento passano nel set di valutazione (che sia di validazione o di test)
il test deve restare “mai visto”, anche nelle statistiche.

Quindi:
X_test viene scalato usando la media e deviazione standard del training.
"""
# Creiamo una rete neurale con un solo livello nascosto di 10 neuroni
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42) #sotto le 1000 iterazioni potrebbe non convergere
"""
Questa riga costruisce un classificatore MLP (rete neurale).

✅ MLPClassifier(...)
È un modello di classificazione supervisionata.
Parametri che hai impostato
hidden_layer_sizes=(10,)
Definisce l’architettura dei layer nascosti.
È una tupla: ogni elemento è un layer.
(10,) significa:
1 layer nascosto
con 10 neuroni
Per esempio:
(10,) → 1 layer da 10
(10, 10) → 2 layer da 10 e 10
(50, 20, 10) → 3 layer (50 → 20 → 10)
Internamente:
input layer: 4 feature (Iris)
hidden layer: 10 neuroni
output layer: 3 neuroni (classi Iris 0,1,2)

max_iter=2000
Numero massimo di iterazioni dell’algoritmo di ottimizzazione.
MLPClassifier usa di default un solver (di solito adam), che aggiorna i pesi iterativamente.
Se metti troppo basso, potresti avere:
warning “Stochastic Optimizer: Maximum iterations reached”
modello non convergente
2000 è “alto” per garantire convergenza su dataset piccoli.

random_state=42
Controlla le componenti casuali:
inizializzazione dei pesi della rete
eventuali shuffle interni
Serve per replicabilità: stesso risultato a ogni run (a parità di ambiente).
"""
mlp.fit(X_train, y_train)  # Addestriamo la rete
"""
Che cosa fa fit
Addestra la rete, cioè trova pesi e bias che minimizzano una loss.

Input:
X_train: array shape (n_samples_train, n_features)
Iris: circa (120, 4)
y_train: array/Series shape (n_samples_train,)
Iris: circa (120,)
Cosa succede dentro (in modo concreto)
Inizializza i pesi
numeri piccoli casuali
dimensioni:
W1: (4,10) (da input a hidden)
b1: (10,)
W2: (10,3) (da hidden a output)
b2: (3,)
Forward pass
calcola output layer dopo layer usando funzioni di attivazione.

Default dell’MLPClassifier:
activation: 'relu' (di solito)
output: softmax/logistic a seconda del caso (multiclasse usa softmax internamente)
Calcolo della loss
Per classificazione multiclasse: tipicamente cross-entropy.
Backpropagation
calcola gradienti della loss rispetto ai pesi e bias (derivate).
Aggiornamento pesi
con il solver (di default Adam):
usa momenti e learning rate adattivo
Ripete fino a:
convergenza (stopping)
oppure fino a max_iter=2000

Output
fit ritorna self (quindi mlp), ma tu non lo assegni perché non serve.
Dopo fit, l’oggetto mlp contiene:
coefs_ (lista di matrici dei pesi)
intercepts_ (bias)
n_iter_ (iterazioni usate)
loss_ (loss finale)
"""

# Valutiamo il modello
accuracy = mlp.score(X_test, y_test)
print(f"Accuratezza: {accuracy:.2f}")

"""
Cosa fa score per un classificatore
In sklearn, per i classificatori, score(X, y) restituisce l’accuracy:

Quindi mlp.score(X_test, y_test) equivale concettualmente a:
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

Cosa passa dentro
X_test: array numpy (perché l’hai trasformato con lo scaler)
shape tipica Iris: (30, 4) se test_size=0.2 su 150 righe
y_test: una Series pandas o array 1D
shape: (30,)

Output
accuracy è un float tra 0 e 1 (es. 1.0, 0.93, 0.80…).

Nota importante
score non cambia il modello: è solo valutazione.
"""

# Predizioni sul test set
y_pred = mlp.predict(X_test) #la predizione va fatta sempre sul X_test, sulla X_train è solo per vedere se il modello mi ha tirato fuori qualcosa
#proba = mlp.predict_proba(X_test) #se volessi le probabilità invece che le classi
print(y_pred)
#print(proba)
"""
Cosa fa predict
Riceve in input X_test (feature scalate).
Esegue un forward pass sulla rete neurale:
input → hidden layer → output layer
Per ogni esempio restituisce la classe più probabile (argmax delle probabilità).

Output
y_pred è un array numpy 1D (dtype spesso int64):
shape: (n_test_samples,) → es. (30,)
"""

# Creazione della confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

"""
Cosa fa confusion_matrix
Costruisce una matrice C dove:
righe = classe vera (y_test)
colonne = classe predetta (y_pred)
C[i,j] = quante volte la classe vera è i ma il modello ha predetto j
Per Iris hai 3 classi (0,1,2), quindi cm sarà 3×3.
"""