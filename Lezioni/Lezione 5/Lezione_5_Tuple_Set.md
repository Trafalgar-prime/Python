# Python - Lezione 5: Tuple e Set in Python

In questa lezione approfondiamo due strutture dati fondamentali in Python:

- **Tuple (`tuple`)** → simili alle liste, ma **immutabili**.
- **Set (`set`)** → collezioni **non ordinate** di elementi **unici**.

---

## 1️⃣ Le Tuple in Python

Le **tuple** sono simili alle liste, ma **NON possono essere modificate** dopo la creazione (**immutabili**).

### 📌 Quando usare una tuple?
✅ Quando i dati **non devono cambiare** (es. coordinate, giorni della settimana).  
✅ Quando serve maggiore **efficienza** (le tuple sono più veloci delle liste).  
✅ Quando servono dati **come chiavi nei dizionari**.

---

## 2️⃣ Creazione di una Tuple

```python
tupla_vuota = ()  # Tuple vuota
tupla_un_solo_elemento = (5,)  # Serve la virgola! Senza, sarebbe un numero normale
tupla_numeri = (1, 2, 3, 4, 5)
tupla_mista = ("ciao", 3.14, True, 42)
```

### 📌 Attenzione: Tuple con un solo elemento

```python
x = (5)    # Non è una tuple, è un numero intero!
y = (5,)   # Questa è una tuple
print(type(x))  # <class 'int'>
print(type(y))  # <class 'tuple'>
```

---

## 3️⃣ Accedere agli Elementi di una Tuple

```python
numeri = (10, 20, 30, 40)
print(numeri[0])  # 10
print(numeri[-1])  # 40 (ultimo elemento)
print(numeri[1:3])  # (20, 30)
```

---

## 4️⃣ Le Tuple Sono Immutabili

```python
numeri = (10, 20, 30)
numeri[1] = 50  # ERRORE! Le tuple non possono essere modificate
```

### 📌 Per modificare una tuple, convertirla in lista:

```python
numeri_lista = list(numeri)
numeri_lista[1] = 50
numeri = tuple(numeri_lista)
print(numeri)  # (10, 50, 30)
```

---

## 5️⃣ I Set in Python

Un **set** è una **collezione non ordinata** di **elementi unici**.

### 📌 Quando usare i set?
✅ Quando vogliamo **evitare duplicati**.  
✅ Quando servono operazioni come **unioni e intersezioni**.  
✅ Quando l’**ordine degli elementi non è importante**.

---

## 6️⃣ Creazione di un Set

```python
set_vuoto = set()  # Set vuoto (NON `{}` perché quello è un dizionario!)
set_numeri = {1, 2, 3, 4, 5}
set_misto = {"ciao", 3.14, True}
```

📌 **I set NON accettano duplicati:**

```python
numeri = {1, 2, 3, 3, 4, 4, 5}
print(numeri)  # {1, 2, 3, 4, 5} → Duplicati eliminati automaticamente
```

---

## 7️⃣ Aggiungere e Rimuovere Elementi nei Set

```python
numeri.add(4)  # Aggiunge un elemento
numeri.remove(2)  # Rimuove un elemento (errore se non esiste)
numeri.discard(10)  # Non dà errore se l'elemento non esiste
valore = numeri.pop()  # Rimuove un elemento casuale
```

---

## 8️⃣ Operazioni tra Set

| Operazione | Metodo | Esempio |
|------------|--------|---------|
| **Unione** (tutti gli elementi) | `set1.union(set2)` | `{1, 2, 3}.union({3, 4, 5}) → {1, 2, 3, 4, 5}` |
| **Intersezione** (elementi comuni) | `set1.intersection(set2)` | `{1, 2, 3}.intersection({3, 4, 5}) → {3}` |
| **Differenza** (solo elementi del primo set) | `set1.difference(set2)` | `{1, 2, 3}.difference({3, 4, 5}) → {1, 2}` |

---

## 9️⃣ ✍ Esercizi da svolgere

1️⃣ **Converti una lista con duplicati in un set per rimuovere i doppioni.**  
2️⃣ **Usa un set per verificare se una parola è già stata inserita.**  
3️⃣ **Crea due set e trova gli elementi in comune tra loro (intersezione).**  

---

## ✅ Obiettivo raggiunto

✅ **Hai imparato le tuple (immutabili) e i set (collezioni uniche).**  
✅ **Sai quando usarli, come modificarli e come fare operazioni tra set.**  
✅ **Ora puoi provare gli esercizi per rafforzare la teoria!** 🚀
