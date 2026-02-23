# Python - Introduzione al Machine Learning e Deep Learning (Super Approfondito)

Questa lezione introduce i concetti di **Machine Learning (ML)** e **Deep Learning (DL)** con spiegazioni dettagliate per ogni riga di codice.

---

## **📌 1️⃣ Differenza tra Machine Learning e Deep Learning**
| **Concetto**       | **Machine Learning (ML)** | **Deep Learning (DL)** |
|--------------------|----------------------|----------------------|
| Dati in input     | Feature ingegnerizzate manualmente | Dati grezzi (immagini, testo, audio) |
| Apprendimento     | Modelli statistici supervisionati | Reti neurali artificiali |
| Algoritmi comuni  | Regressione, Alberi decisionali, SVM | CNN, RNN, Transformer |
| Necessità di dati | Medio-bassa | Molto alta (milioni di esempi) |
| Calcolo richiesto | CPU sufficiente | GPU o TPU necessarie |

---

## **📌 2️⃣ Installare le Librerie Necessarie**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch torchvision transformers
```

---

## **📌 3️⃣ Importare e Manipolare Dati**
```python
import pandas as pd
from sklearn.datasets import load_iris

dati = load_iris()  
df = pd.DataFrame(dati.data, columns=dati.feature_names)
df['target'] = dati.target

print(df.head())
```

---

## **📌 4️⃣ Creare un Modello ML Base con Scikit-learn**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

accuracy = mlp.score(X_test, y_test)
print(f"Accuratezza: {accuracy:.2f}")
```

---

## **📌 5️⃣ Creare una Rete Neurale con TensorFlow/Keras**
```python
import tensorflow as tf
from tensorflow import keras

modello = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

modello.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modello.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
```

---

## **📌 6️⃣ Creare una Rete Neurale con PyTorch**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ReteNeurale(nn.Module):
    def __init__(self):
        super(ReteNeurale, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

modello = ReteNeurale()
```

---

## **📌 7️⃣ Prossimi Passi verso GANs, VAE e Transformers**
1. **Architetture avanzate**: CNN, RNN, MLP avanzati  
2. **Backpropagation e ottimizzazione**  
3. **Tecniche di regularizzazione e normalizzazione**  
4. **Autoencoder e VAE**  
5. **GANs e modelli generativi**  
6. **Transformers e NLP avanzato**  

---

## **✅ Obiettivo raggiunto**
✅ **Hai compreso le basi del ML e DL.**  
✅ **Hai creato reti neurali con TensorFlow e PyTorch.**  
✅ **Hai le basi per modelli più avanzati.**  
✅ **Sei pronto per approfondire le architetture neurali!** 🚀

