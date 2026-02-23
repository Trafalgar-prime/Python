# Python - Lezione 4: Le Liste in Python

Le **liste** sono una delle strutture dati più importanti in Python perché permettono di **memorizzare più valori** in un’unica variabile e modificarli facilmente.

---

## 1) Creare una Lista in Python

Una lista si definisce usando **le parentesi quadre `[]`**, separando gli elementi con una virgola:

```python
numeri = [1, 2, 3, 4, 5]  # Lista di numeri interi
frutti = ["mela", "banana", "ciliegia"]  # Lista di stringhe
mista = [10, "ciao", 3.14, True]  # Lista con tipi diversi
```

📌 **Cose importanti da ricordare:**  
- Gli **elementi sono ordinati** (seguono l’ordine di inserimento).  
- Possiamo **avere tipi di dati diversi** nella stessa lista.  
- Le **liste possono essere vuote** (`[]`).  

---

## 2) Accedere agli Elementi di una Lista

Gli **elementi di una lista** sono indicizzati **da 0 in poi**:  

```python
frutti = ["mela", "banana", "ciliegia"]

print(frutti[0])  # "mela"
print(frutti[1])  # "banana"
print(frutti[2])  # "ciliegia"
```

📌 **Indice negativo per contare dalla fine**:

```python
print(frutti[-1])  # "ciliegia"
print(frutti[-2])  # "banana"
print(frutti[-3])  # "mela"
```

---

## 3) Modificare una Lista

### Cambiare un elemento
```python
numeri = [1, 2, 3, 4, 5]
numeri[2] = 10  # Cambia il valore in posizione 2
print(numeri)  # [1, 2, 10, 4, 5]
```

---

## 4) Aggiungere e Rimuovere Elementi

### Aggiungere un elemento alla lista

📌 **`append()` → Aggiunge in fondo alla lista**
```python
frutti.append("arancia")
print(frutti)  # ['mela', 'banana', 'ciliegia', 'arancia']
```

📌 **`insert()` → Aggiunge in una posizione specifica**
```python
frutti.insert(1, "kiwi")  # Inserisce "kiwi" in posizione 1
print(frutti)  # ['mela', 'kiwi', 'banana', 'ciliegia', 'arancia']
```

### Rimuovere elementi dalla lista

📌 **`remove()` → Rimuove un valore specifico**
```python
frutti.remove("banana")
print(frutti)
```

📌 **`pop()` → Rimuove un elemento specifico o l’ultimo**
```python
ultimo = frutti.pop()  # Rimuove l’ultimo elemento
print(frutti)
```

---

## 5) Scorrere una Lista con un Loop

📌 **Usare `for`**
```python
frutti = ["mela", "banana", "ciliegia"]
for frutto in frutti:
    print(frutto)
```

📌 **Usare `enumerate()` per ottenere l’indice**
```python
for i, frutto in enumerate(frutti):
    print(f"Indice {i}: {frutto}")
```

---

## 6) Chiarimento: Perché usare `f` nel `print()`?

L’`f` davanti alla stringa crea una **f-string**, che permette di inserire variabili direttamente nel testo:

```python
nome = "Luca"
eta = 25
print(f"Mi chiamo {nome} e ho {eta} anni.")
```
🔹 Output:
```
Mi chiamo Luca e ho 25 anni.
```

📌 **Senza `f`, Python stampa il testo letteralmente:**

```python
print("Mi chiamo {nome} e ho {eta} anni.")
```
🔹 Output errato:
```
Mi chiamo {nome} e ho {eta} anni.
```

✅ **Le f-string funzionano anche nei loop con `enumerate()`**:
```python
for i, frutto in enumerate(frutti):
    print(f"Indice {i}: {frutto}")
```
🔹 Output corretto:
```
Indice 0: mela
Indice 1: banana
Indice 2: ciliegia
```

---

## 7) Controllare se un Elemento è nella Lista

📌 **Usiamo l'operatore `in`**  
```python
if "banana" in frutti:
    print("Banana è presente nella lista!")
```

---

## 8) Ordinare e Invertire una Lista

📌 **Ordinare con `sort()`**
```python
numeri = [5, 2, 8, 1, 3]
numeri.sort()  # Ordina in modo crescente
print(numeri)  # [1, 2, 3, 5, 8]
```

📌 **Invertire con `reverse()`**
```python
numeri.reverse()
print(numeri)  # [8, 5, 3, 2, 1]
```

---

## 9) Liste Annidate (Matrici)

```python
matrice = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(matrice[1][2])  # Accede all'elemento nella riga 1, colonna 2 → 6
```

---

## 10) ✍ Esercizi da svolgere

1️⃣ **Lista di numeri**  
   - Chiedi all’utente **5 numeri** e inseriscili in una lista.  
   - Stampa la lista.  

2️⃣ **Lista con somma e media**  
   - Chiedi all’utente di **inserire numeri finché non scrive `"stop"`**.  
   - Salvali in una lista.  
   - **Calcola e stampa** la somma e la media dei numeri inseriti.  

3️⃣ **Lista ordinata**  
   - Chiedi all’utente **5 parole**.  
   - Salvale in una lista.  
   - **Ordina la lista e stampala**.  

---

## Obiettivo raggiunto
✅ Ora sai:  
- **Creare e modificare liste**  
- **Aggiungere, rimuovere e cercare elementi**  
- **Scorrere una lista con `for` e `enumerate()`**  
- **Usare le f-string (`f""`) per formattare le stringhe**  
- **Ordinare e invertire liste**  
- **Usare liste annidate (matrici)**  

📌 **Ora prova gli esercizi e fammi sapere se hai domande!** 🚀
