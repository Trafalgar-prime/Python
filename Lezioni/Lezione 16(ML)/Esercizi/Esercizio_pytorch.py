import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix , classification_report
# il train split e la normalizzazione dei dati avviene sempre sklearn

dati = load_digits()  #estratti già in numpy.ndarray
X = dati.data
Y = dati.target

print("X : ",X,"\n")
print("Y : ",Y,"\n")
print("Forma di X : ",X.shape,"\n")
print("Forma di Y : ",Y.shape,"\n")
print(len(np.unique(Y)),"\n") 
print(np.unique(Y),"\n")   

X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

print("\n")
print("Dimensioni della X_train:\n",X_train.shape)
print("\n")
print("Dimensioni della Y_train:\n",Y_train.shape)
print("\n")
print("Dimensioni della X_test:\n",X_test.shape)
print("\n")
print("Dimensioni della Y_test:\n",Y_test.shape)
print("\n")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X train tipo :", type(X_train))
print("X test tipo :", type(X_test))
print("Y train tipo :", type(Y_train))
print("Y test tipo :", type(Y_test))

# 1) Convertiamo in tensori PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32) #la X sempre in float32
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

#Y_train_t = torch.tensor(Y_train.to_numpy(), dtype=torch.long) #il target sempre in long int
#Y_test_t  = torch.tensor(Y_test.to_numpy(), dtype=torch.long)

Y_train_t = torch.tensor(Y_train, dtype=torch.long)
Y_test_t  = torch.tensor(Y_test, dtype=torch.long)

print("X train tipo :", type(X_train_t))
print("X test tipo :", type(X_test_t))
print("Y train tipo :", type(Y_train_t))
print("Y test tipo :", type(Y_test_t))

print("---------------------------------------------")
print("--------- INIZIO PARTE DELICATA -------------")
print("---------------------------------------------")


class ReteNeurale(nn.Module):
    def __init__(self):
        super(ReteNeurale,self).__init__()
        self.func1 = nn.Linear(64,32)
        self.func2 = nn.Linear(32,16)
        self.output = nn.Linear(16,10)

        self.relu = nn.ReLU()  #meglio usare questo tipo di scrittura e poi successivamente di richiamo nel forward
        self.leaky = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self,x):
        x = F.relu(self.func1(x))  #non uso self perchè sto importando direttamente da F. altrimenti self.relu ma perchè l'ho scritta in init
        x = F.leaky_relu(self.func2(x), negative_slope=0.01, inplace=False)
        x = self.output(x)
        return x
    
modello = ReteNeurale()


# 2) Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modello.parameters(), lr=1e-3)

for epoch in range(200):
    modello.train()

    logits = modello(X_train_t)
    loss = criterion(logits, Y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss:", loss.item())

    # 3. Monitoraggio (ogni 10 epoche per non intasare il terminale)
    if (epoch + 1) % 10 == 0:
        # Usiamo .item() per estrarre il valore numerico puro dal tensore della loss
        print(f'Epoca [{epoch+1}/100], Loss: {loss.item():.4f}')


# 4) Valutazione su test
modello.eval()

with torch.no_grad():
    logits_test = modello(X_test_t)
    y_pred_t = logits_test.argmax(dim=1)

test_acc = (y_pred_t == Y_test_t).float().mean().item()
print("\nTest accuracy:", test_acc)

# 5) Confusion matrix (sklearn vuole numpy)
y_pred = y_pred_t.cpu().numpy()
cm = confusion_matrix(Y_test, y_pred)
print("\nConfusion matrix:\n", cm)

report = classification_report(y_pred_t, Y_test_t)
print("\nClassifiaction report:\n ",report)