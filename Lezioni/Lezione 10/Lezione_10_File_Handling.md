# Python - Lezione 10: Gestione dei File (File Handling)

La gestione dei file (**File Handling**) in Python permette di leggere, scrivere e manipolare file su disco.

---

## **1️⃣ Aprire un File con `open()`**
```python
file = open("testo.txt", "r")  # Modalità di apertura in lettura
contenuto = file.read()
file.close()
```

| Modalità | Descrizione |
|----------|------------|
| `"r"`  | **Read** (solo lettura) |
| `"w"`  | **Write** (cancella il file se esiste) |
| `"a"`  | **Append** (aggiunge dati alla fine) |
| `"x"`  | **Create** (crea un file, errore se esiste) |
| `"r+"` | **Lettura e scrittura** |

---

## **2️⃣ Leggere un File (`read()`, `readline()`, `readlines()`)**
```python
with open("testo.txt", "r") as file:
    contenuto = file.read()
    print(contenuto)  
```

```python
with open("testo.txt", "r") as file:
    print(file.readline())  # Legge una riga
```

```python
with open("testo.txt", "r") as file:
    righe = file.readlines()  # Lista di righe
    print(righe)
```

---

## **3️⃣ Scrivere su un File (`write()`)**
```python
with open("output.txt", "w") as file:
    file.write("Questa è una nuova riga di testo.
")
```

```python
with open("output.txt", "a") as file:
    file.write("Questa è un'altra riga aggiunta.
")
```

---

## **4️⃣ Usare `with open()` (Best Practice)**
```python
with open("testo.txt", "r") as file:
    contenuto = file.read()
    print(contenuto)  
```

---

## **5️⃣ Leggere e Scrivere un File Contemporaneamente (`r+`)**
```python
with open("testo.txt", "r+") as file:
    contenuto = file.read()
    file.write("
Nuova riga aggiunta.")
```

---

## **6️⃣ Lavorare con File CSV**
```python
import csv

with open("dati.csv", "r") as file:
    lettore = csv.reader(file)
    for riga in lettore:
        print(riga)  
```

```python
import csv

with open("output.csv", "w", newline="") as file:
    scrittore = csv.writer(file)
    scrittore.writerow(["Nome", "Età", "Città"])
    scrittore.writerow(["Anna", 25, "Roma"])
```

---

## **7️⃣ Lavorare con File JSON**
```python
import json

dati = {"nome": "Luca", "età": 30, "città": "Roma"}

with open("dati.json", "w") as file:
    json.dump(dati, file)  
```

```python
import json

with open("dati.json", "r") as file:
    dati = json.load(file)  

print(dati)  
```

---

## **8️⃣ Eliminare un File**
```python
import os

if os.path.exists("output.txt"):
    os.remove("output.txt")
else:
    print("Il file non esiste")
```

---

## **🔟 Esercizi**
1️⃣ **Leggi un file di testo e conta quante righe contiene.**  
2️⃣ **Scrivi un programma che chiede all'utente di inserire testo e lo salva in un file.**  
3️⃣ **Apri un file CSV ed estrai solo i nomi dalla prima colonna.**  
4️⃣ **Crea un programma che legge un file JSON e stampa il valore di una chiave specifica.**  
5️⃣ **Scrivi un programma che cancella un file se esiste.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato ad aprire, leggere e scrivere file in Python.**  
✅ **Sai usare `with open()` per gestire i file in modo sicuro.**  
✅ **Hai visto come lavorare con file CSV e JSON.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

