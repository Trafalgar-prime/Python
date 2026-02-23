import pandas as pd
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


dati = load_digits()
df = pd.DataFrame(dati.data, columns=dati.feature_names) #in questo modo ho trasformato in un dataframe, e poi ridivido per la X e la Y
df.index.name = "id"
X = dati.data
Y = dati.target
print("\n")
print("Dimensioni della X:\n",X.shape)
print("\n")
print("Dimensioni della Y:\n",Y.shape)
print("\n")
print("Valori nella Y:\n",Y,"\n")
print("Valori nella X:\n",X,"\n")
print(df)
print(df.shape)

X = load_digits().data  #sono i il numero di oggetti, e la dimensione, in questo caso 64 cioè 8x8 ma già flattened
Y = load_digits().target  #sono le etichette: quindi so già che è un problema multiclasse con classi da 0 a 9
print("\n")
print("Dimensioni della X:\n",X.shape)
print("\n")
print("Dimensioni della Y:\n",Y.shape)
print("\n")
print("Valori nella Y:\n",Y,"\n")
print("Valori nella X:\n",X,"\n")


print("\n")
print("-------PICCOLO TEST ---------")
print("\n")

print(X.shape[0])  #numero di oggetti
print(X.shape[1])  #dimensioni degli oggetti
print(Y.shape[0])  #numero di etichette totali
print(len(np.unique(Y)))  #numero di etichette unicheù
print(np.unique(Y))   #quali etichette sono

print("\n")
print("-------FINE TEST ---------")
print("\n")

print("\n")
print("-------INIZIO SPLIT ---------")
print("\n")
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=42)
#help(train_test_split)

print("\n")
print("Dimensioni della X_train:\n",X_train.shape)
print("\n")
print("Dimensioni della Y_train:\n",Y_train.shape)
print("\n")
print("Dimensioni della X_test:\n",X_test.shape)
print("\n")
print("Dimensioni della Y_test:\n",Y_test.shape)
print("\n")
print("Valori nella Y_train:\n",Y_train,"\n")
print("Valori nella X_train:\n",X_train,"\n")
print("Valori nella Y_test:\n",Y_test,"\n")
print("Valori nella X_test:\n",X_test,"\n")


print("\n")
print("-------FINE SPLIT  ---------")
print("\n")

print("\n")
print("-------INIZIO NORMALIZZAZIONE ---------")
print("\n")

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n")
print("Dimensioni della X_train:\n",X_train.shape)
print("\n")
print("Dimensioni della Y_train:\n",Y_train.shape)
print("\n")
print("Dimensioni della X_test:\n",X_test.shape)
print("\n")
print("Dimensioni della Y_test:\n",Y_test.shape)
print("\n")
print("Valori nella Y_train:\n",Y_train,"\n")
print("Valori nella X_train:\n",X_train,"\n")
print("Valori nella Y_test:\n",Y_test,"\n")
print("Valori nella X_test:\n",X_test,"\n")


print("\n")
print("-------FINE NORMALIZZAZIONE ---------")
print("\n")

print("\n")
print("----------------------------------")
print("-------INIZIO SKLEARN ------------")
print("--------REGRESSIONE---------------")
print("\n")

print("\n")
print("-------INIZIO ADDESTRAMENTO ---------")
print("\n")

from sklearn.linear_model import LinearRegression
#help(LinearRegression)
model = LinearRegression()
#help(model.fit)
LR = model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("\n")
print("-------FINE ADDESTRAMENTO ---------")
print("\n")

print("\n")
print("-------INIZIO EVALUATION ---------")
print("\n")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

mae = mean_absolute_error(Y_test, Y_pred)
rmse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Absolute Errore : ",mae,"\n")
print("Mean Squared Errore : ",rmse,"\n")
print("R2 score : ",r2,"\n")

print("\n")
print("-------FINE EVALUATION ---------")
print("\n")

print("\n")
print("--------REGRESSIONE---------------")
print("-------FINE SKLEARN --------------")
print("----------------------------------")
print("\n")

print("\n")
print("----------------------------------")
print("-------INIZIO TENSORFLOW ---------")
print("--------CLASSIFICAZIONE-----------")
print("\n")

print("\n")
print("-------INIZIO ADDESTRAMENTO ---------")
print("\n")

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation = 'relu', input_shape = (64,)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(8, activation = 'relu'),
    keras.layers.Dense(10, activation = 'linear') #10 è il numero di classi del modello, per la regressione è =1 mentre per la classificazione metto il numero
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #from_logits=True mi permette di definire più classi e non un andamento probabilista tra 0 e 1 come di default
    metrics=['accuracy']
    )

model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
print(model.summary())

logits = model.predict(X_test)
Y_pred = np.argmax(logits, axis=1)   #in questo ottengo il valore più probabile per la classe riducendo la matrice in un'unico array
#non uso questo tipo di codice solo nel caso in cui come funzione di attivazione ho la 'sigmoid'
print("Y predetti : \n",Y_pred,"\n")
print(Y_pred.shape,"\n")
print(Y_test.shape,"\n")

print("\n")
print("-------FINE ADDESTRAMENTO ---------")
print("\n")

print("\n")
print("-------INIZIO EVALUATION ---------")
print("\n")

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
#help(accuracy_score)
#help(model.evaluate)

#loss , accuracy = model.evaluate(X_test, Y_test, verbose = 0) #con questo codice faccio anche le predizioni in automatico
#print(loss)
#print(accuracy)
#help(f1_score)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy : ",accuracy,"\n")
f1 = f1_score(Y_test, Y_pred, average='macro')
print("F1 : ",f1,"\n")
precision = precision_score(Y_test, Y_pred, average='macro')
print("Precision : ",precision,"\n")
recall = recall_score(Y_test, Y_pred, average='macro')
print("Recall : ",recall,"\n")
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix : \n",cm,"\n")
mae = mean_absolute_error(Y_test, Y_pred)
rmse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Absolute Errore : ",mae,"\n")
print("Mean Squared Errore : ",rmse,"\n")
print("R2 score : ",r2,"\n")


print("\n")
print("-------FINE EVALUATION ---------")
print("\n")

print("\n")
print("---------CLASSIFICAZIONE----------")
print("-------- FINE TENSORFLOW ---------")
print("----------------------------------")
print("\n")

print("\n")
print("----------------------------------")
print("-------INIZIO TENSORFLOW ---------")
print("--------CLASSIFICAZIONE BINARIA-----------")
print("\n")

print("\n")
print("-------INIZIO ADDESTRAMENTO ---------")
print("\n")

model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(64,)),
    #keras.layers.BatchNormalization(),
    #keras.layers.Dropout(0.3),
    #keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    #keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),

    keras.layers.Dense(1, activation= 'sigmoid')  # output
])
model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.BinaryCrossentropy(), #(from_logits=True)
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
print(model.summary())

Y_pred = model.predict(X_test)
print("Y predetti : \n",Y_pred,"\n")
print(Y_pred.shape,"\n")
print(Y_test.shape,"\n")

print("\n")
print("-------FINE ADDESTRAMENTO ---------")
print("\n")

print("\n")
print("-------INIZIO EVALUATION ---------")
print("\n")

loss , metric = model.evaluate(X_test, Y_test)
print("Loss : ",loss,"\n")
print("Accuracy : ",metric,"\n")

f1 = f1_score(Y_test, Y_pred, average='macro')
print("F1 : ",f1,"\n")
precision = precision_score(Y_test, Y_pred, average='macro')
print("Precision : ",precision,"\n")
recall = recall_score(Y_test, Y_pred, average='macro')
print("Recall : ",recall,"\n")
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix : \n",cm,"\n")
mae = mean_absolute_error(Y_test, Y_pred)
rmse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Absolute Errore : ",mae,"\n")
print("Mean Squared Errore : ",rmse,"\n")
print("R2 score : ",r2,"\n")

print("\n")
print("-------FINE EVALUATION ---------")
print("\n")

print("\n")
print("---------CLASSIFICAZIONE BINARIA----------")
print("-------- FINE TENSORFLOW ---------")
print("----------------------------------")
print("\n")


print("\n")
print("----------------------------------")
print("-------INIZIO TENSORFLOW ---------")
print("--------CLASSIFICAZIONE ONE-HOT-----------")
print("\n")

print("\n")
print("-------INIZIO ADDESTRAMENTO ---------")
print("\n")

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)  #queste righe mi servono per passare al one-hot delle classi, altrimenti avrei un semplice multiclasse
Y_test  = tf.keras.utils.to_categorical(Y_test, num_classes=10)

model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(64,)),
    #keras.layers.BatchNormalization(),
    #keras.layers.Dropout(0.3),
    #keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    #keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),

    keras.layers.Dense(10, activation= 'softmax')  # output
])
model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.CategoricalCrossentropy(), #(from_logits=True) non necessario
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))
print(model.summary())

logits = model.predict(X_test)
Y_pred = np.argmax(logits, axis=1)
print("Y predetti : \n",Y_pred,"\n")
print("Y_pred.shape : ",Y_pred.shape,"\n")
print("Y_test.shape : ",Y_test.shape,"\n")
Y_true = np.argmax(Y_test, axis = 1) #riporto il one-hot in multiclasse per fare l'evaluation
print("Y_test.shape : ",Y_test.shape,"\n")

print("\n")
print("-------FINE ADDESTRAMENTO ---------")
print("\n")

loss , metric = model.evaluate(X_test, Y_test, verbose = 0)
print("Loss : ",loss,"\n")
print("Accuracy : ",metric,"\n")

f1 = f1_score(Y_true, Y_pred, average='macro')
print("F1 : ",f1,"\n")
precision = precision_score(Y_true, Y_pred, average='macro')
print("Precision : ",precision,"\n")
recall = recall_score(Y_true, Y_pred, average='macro')
print("Recall : ",recall,"\n")
cm = confusion_matrix(Y_true, Y_pred)
print("Confusion Matrix : \n",cm,"\n")
mae = mean_absolute_error(Y_true, Y_pred)
rmse = mean_squared_error(Y_true, Y_pred)
r2 = r2_score(Y_true, Y_pred)

print("Mean Absolute Errore : ",mae,"\n")
print("Mean Squared Errore : ",rmse,"\n")
print("R2 score : ",r2,"\n")

print("\n")
print("-------FINE EVALUATION ---------")
print("\n")

print("\n")
print("---------CLASSIFICAZIONE ONE-HOT--------")
print("-------- FINE TENSORFLOW ---------")
print("----------------------------------")
print("\n")