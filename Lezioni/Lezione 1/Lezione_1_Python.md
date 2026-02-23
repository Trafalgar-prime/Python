# Python - Lezione 1: Fondamenti di Python

---

## 1. Eseguire uno script Python

Per eseguire un programma Python:

1. Apri **IDLE** e crea un nuovo file (`File > New File`).
2. Scrivi il codice, ad esempio:

```python
print("Ciao, mondo!")
```

3. Salva il file con estensione **.py** (es. `primo_script.py`).
4. Esegui (`Run > Run Module` oppure premi **F5**).

---

## 2. Stampare output con `print()`

La funzione `print()` mostra testo o valori in output:

```python
print("Benvenuto in Python!")
print(5 + 3)  # Stampa 8
print("Il risultato è:", 10 * 2)
```

---

## 3. Variabili e tipi di dati

In Python non serve dichiarare il tipo:

```python
nome = "Alice"    # Stringa
eta = 25          # Intero
altezza = 1.75    # Float
is_online = True  # Booleano

print(nome, eta, altezza, is_online)
```

Per controllare il tipo di una variabile:

```python
print(type(nome))   # <class 'str'>
print(type(eta))    # <class 'int'>
print(type(altezza)) # <class 'float'>
print(type(is_online)) # <class 'bool'>
```

---

## 4. Input dell’utente

Usiamo `input()` per ricevere dati dall’utente:

```python
nome = input("Come ti chiami? ")
print("Ciao,", nome)
```

L'`input()` restituisce una **stringa**, quindi per numeri bisogna convertirli:

```python
eta = int(input("Quanti anni hai? "))
print("Tra 5 anni avrai", eta + 5)
```

---

## ✍ Esercizi

1️⃣ Scrivi un programma che chiede **nome** ed **età**, poi stampa un messaggio di benvenuto.  
2️⃣ Chiedi all’utente due numeri e stampa la loro somma.  
3️⃣ Stampa il tipo di variabile per tre valori diversi inseriti dall’utente.

---

✅ **Obiettivo raggiunto**: Hai imparato a eseguire script Python, usare `print()`, variabili, tipi di dati e `input()`! 🚀
