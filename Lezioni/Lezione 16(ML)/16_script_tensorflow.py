import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Importa il dataset Iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras


# Caricare un dataset (Esempio: Iris dataset)
dati = load_iris()  # Carica i dati di Iris
#Questa funzione carica in memoria il dataset Iris e lo restituisce come un oggetto “contenitore” di scikit-learn.

# Creiamo un DataFrame Pandas con i dati e aggiungiamo la colonna "target"
df = pd.DataFrame(dati.data, columns=dati.feature_names)
#pd.set_option('display.max_rows', None) #serve per vedere tutti i dati e non solo i parziali in output, perche altrimenti pandas mi stampa solo i parziali per non intasare l'output
#pd.set_option('display.max_columns', None) #stesso del precedente ma per le colonne [in questo caso è inutile perche le colonne sono solo 4]
print(df)
print("\nProva\n")
print(df.shape)
print(df.columns)
print("\n")


df.index.name = "id" # la colonna degli indici la stampa pandas in automatico, in questo modo gli do un nome
df['target'] = dati.target


#Mostriamo le prime 5 righe del dataset
print(df.head())
print("\n",df,"\n")
print("\n")


# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
#.iloc permette di definire le posizioni di quello di cui ho bisogno in base alle colonne

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



modello = keras.Sequential([ #per tutta la spiegazione cercare sul GitHub :LEZIONE16_keras_sequential_analisi.md
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # 16 neuroni
    keras.layers.Dense(8, activation='relu'),  # 8 neuroni
    keras.layers.Dense(3, activation='softmax')  # 3 classi in output
])

"""
Cos’è Sequential
Sequential significa che la rete è una pipeline lineare:
output layer 1 → input layer 2
output layer 2 → input layer 3
ecc.
Quindi nessuna ramificazione, nessun “salto” (skip connection). Per Iris va benissimo.
"""

modello.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""
perché serve compile
Keras deve sapere:
come aggiornare i pesi (optimizer)
cosa minimizzare (loss)
cosa monitorare (metrics)
Senza compile non puoi fare training con fit.

7.2 optimizer = Adam
Adam è un gradiente discendente adattivo:
tiene traccia di una media dei gradienti (momento 1)
e una media dei gradienti al quadrato (momento 2)
adatta il passo di aggiornamento parametro per parametro
È una scelta standard e robusta.

7.3 loss = sparse_categorical_crossentropy
Questa è la scelta giusta quando le label sono interi:
se y_train = [0, 2, 1, 0, ...] → sparse_categorical_crossentropy ✅
se y_train fosse one-hot ([1,0,0], [0,0,1]...) → categorical_crossentropy
Perché:
con softmax in output vuoi una cross-entropy multiclasse
e “sparse” indica che le classi sono codificate come interi, non one-hot

7.4 metric = accuracy
Accuracy = percentuale di volte in cui:
argmax(pred) == y_true

Nota concettuale importante:
tu ottimizzi la loss
l’accuracy è solo una metrica di monitoraggio
"""

modello.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

"""
input e shape
X_train: (n_train, 4) (es. 120×4)
y_train: (n_train,) valori 0/1/2
Lo scaling che hai fatto prima è un grande vantaggio:
input su scale simili → training più stabile

8.2 epochs=100
Un’epoca = un passaggio completo su tutti i dati di training.
Quindi:
100 epoche = il modello rivede tutto il training 100 volte
Sotto al cofano:
Keras usa batch (default spesso 32)
se train ha 120 esempi → circa 4 batch per epoca

8.3 validation_data=(X_test, y_test)
Qui usi il test come validation.
È un punto importante:
in un progetto “pulito” dovresti avere:
train
validation
test (tenuto da parte fino alla fine)
qui invece il test viene “guardato” ogni epoca (anche se non fa fit su test, stai comunque facendo tuning mentale/visivo sul test)
Per esercizio su Iris va bene, ma devi sapere che:
tecnicamente “sporchi” il test come valutazione finale

8.4 cosa succede a ogni epoca
Per ogni batch:
forward pass
loss
backprop (gradienti)
update pesi (Adam)

Alla fine epoca:
training loss/accuracy
validation loss/accuracy

8.5 cosa aspettarti su Iris
Spesso:
accuracy sale molto velocemente
val_accuracy può arrivare anche a 1.0
100 epoche possono essere molte (possibile overfitting), ma rete piccola e dataset semplice spesso reggono.
"""

modello.summary()

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = modello.predict(X_test).argmax(axis=1)
print("\nY predetta : \n",y_pred)
"""
predict() restituisce probabilità.
Per avere la classe (0/1/2):
argmax(axis=1) prende l’indice del valore più grande tra le 3 probabilità.
"""

proba = modello.predict(X_test)
y_pred = np.argmax(proba, axis=1)  # l'utilizzo di argmax è legato alla funzione di attivazione dell'ultimo strato
#questo lo fa in 2 righe mentre l'altro con una sola riga
print("\nY predetta : \n",y_pred)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
