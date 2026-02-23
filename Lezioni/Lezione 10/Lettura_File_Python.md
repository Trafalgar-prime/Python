# Python - Lettura dei File: `read()`, `readline()`, `readlines()`

Python offre diversi metodi per leggere i file di testo.

---

## **1️⃣ Leggere un File con `read()`**

```python
with open("testo.txt", "r") as file:
    print(file.read())  # Legge tutto il file e lo stampa
```
🔹 **Problema:** Se chiami `read()` due volte, la seconda volta restituirà una stringa vuota `''` perché il cursore sarà alla fine.

```python
with open("testo.txt", "r") as file:
    print(file.read())  # Legge tutto
    print(file.read())  # NON stamperà nulla
```

✅ **Soluzione:** Usa `file.seek(0)` per riportare il cursore all'inizio.

```python
with open("testo.txt", "r") as file:
    print(file.read())  # Legge tutto
    file.seek(0)  # Riporta il cursore all'inizio
    print(file.read())  # Ora legge di nuovo
```

---

## **2️⃣ Leggere una Riga alla Volta con `readline()`**
```python
with open("testo.txt", "r") as file:
    print(file.readline())  # Stampa la prima riga
    print(file.readline())  # Stampa la seconda riga
```

---

## **3️⃣ Leggere Tutte le Righe in una Lista con `readlines()`**
```python
with open("testo.txt", "r") as file:
    righe = file.readlines()
    print(righe)  # ['Prima riga\n', 'Seconda riga\n']
```

---

## **📌 Confronto tra i Metodi di Lettura**
| Metodo | Descrizione |
|--------|------------|
| `read()` | Legge tutto il file come una stringa |
| `readline()` | Legge solo **una riga** alla volta |
| `readlines()` | Legge **tutte le righe** e le restituisce come lista |

---

## **🔟 Esercizi**
1️⃣ **Leggi un file di testo e stampa solo la prima riga.**  
2️⃣ **Stampa tutte le righe di un file una alla volta usando un ciclo `for`.**  
3️⃣ **Apri un file, leggi tutto il contenuto e poi rileggilo usando `seek(0)`.**  
4️⃣ **Conta quante righe ci sono in un file di testo.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a usare `read()`, `readline()` e `readlines()` per leggere file in Python.**  
✅ **Sai come evitare problemi con il cursore del file usando `seek(0)`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

