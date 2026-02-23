# Python - Approfondimento: Come dare una lista in input?

In Python, la funzione `input()` **accetta solo una stringa alla volta**, quindi **non possiamo inserire direttamente una lista**. Tuttavia, esistono modi per permettere all'utente di inserire più valori contemporaneamente e convertirli in una lista.

---

## 1️⃣ `input()` accetta solo stringhe

Quando chiediamo un input all’utente con `input()`, Python lo considera sempre una **stringa**:

```python
dati = input("Inserisci qualcosa: ")  # L'utente digita: ciao
print(type(dati))  # <class 'str'>
```

---

## 2️⃣ Come inserire più elementi in una lista?

Se vogliamo **inserire più valori contemporaneamente**, possiamo chiedere all'utente di digitare gli elementi separati da uno spazio o da una virgola, e poi **convertirli in una lista**.

### 🔹 Metodo 1: Usare `split()` per dividere gli input

```python
numeri = input("Inserisci numeri separati da uno spazio: ").split()
print(numeri)  # Lista di stringhe
```

**Esempio di input dell'utente:**  
```
10 20 30 40 50
```

**Output:**  
```python
['10', '20', '30', '40', '50']
```

📌 **Nota:** Tutti gli elementi saranno stringhe. Se vogliamo convertirli in numeri, usiamo `map(int, ...)`:

```python
numeri = list(map(int, input("Inserisci numeri separati da uno spazio: ").split()))
print(numeri)  # Ora la lista contiene interi
```

**Output:**  
```python
[10, 20, 30, 40, 50]
```

---

### 🔹 Metodo 2: Inserire valori separati da virgola

```python
frutti = input("Inserisci frutti separati da virgola: ").split(", ")
print(frutti)
```

**Esempio di input dell'utente:**  
```
mela, banana, ciliegia, arancia
```

**Output:**  
```python
['mela', 'banana', 'ciliegia', 'arancia']
```

---

## 3️⃣ Come chiedere un numero fisso di valori?

Se vogliamo **chiedere un numero specifico di elementi**, possiamo usare un ciclo:

```python
numeri = []
for i in range(5):  # Chiediamo 5 numeri
    num = int(input(f"Inserisci il numero {i+1}: "))
    numeri.append(num)

print("Lista di numeri:", numeri)
```

🔹 **Output se l'utente inserisce i numeri uno per volta:**  
```
Inserisci il numero 1: 3
Inserisci il numero 2: 7
Inserisci il numero 3: 2
Inserisci il numero 4: 8
Inserisci il numero 5: 5
Lista di numeri: [3, 7, 2, 8, 5]
```

---

## 💡 Riassunto

✅ **`input()` può accettare solo una stringa alla volta.**  
✅ **Per inserire più valori contemporaneamente, si usa `.split()`.**  
✅ **Per trasformare gli input in numeri, si usa `map(int, ...)`.**  
✅ **Se vogliamo un numero fisso di input, usiamo un ciclo `for`.**  

📌 **Ora puoi provare a implementarlo nel tuo codice! 🚀**