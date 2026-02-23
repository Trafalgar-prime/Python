# Python - Lettura dei File JSON

Python offre il modulo integrato **`json`** per lavorare con file JSON.  
Per leggere un file JSON, usiamo **`json.load()`**.

---

## **1️⃣ Leggere un File JSON**
```python
import json

with open("dati.json", "r") as file:
    dati = json.load(file)  # Carica il contenuto JSON in una variabile

print(dati)  # Mostra il contenuto come un dizionario Python
```

📌 **Esempio di file `dati.json`:**
```json
{
    "nome": "Luca",
    "età": 30,
    "città": "Roma"
}
```

🔹 **Output del programma:**
```
{'nome': 'Luca', 'età': 30, 'città': 'Roma'}
```

---

## **2️⃣ Accedere ai Valori del JSON**
Una volta caricato il JSON come dizionario, possiamo accedere ai valori con le chiavi.

```python
print(dati["nome"])  # Luca
print(dati["età"])   # 30
print(dati["città"]) # Roma
```

---

## **3️⃣ Leggere un JSON con Liste**
Se il file JSON contiene una lista di oggetti:
```json
[
    {"nome": "Luca", "età": 30},
    {"nome": "Anna", "età": 25}
]
```
🔹 **Per leggerlo in Python:**
```python
with open("dati.json", "r") as file:
    dati = json.load(file)

for persona in dati:
    print(persona["nome"], "-", persona["età"])
```
🔹 **Output:**
```
Luca - 30
Anna - 25
```

---

## **4️⃣ Gestire Errori con `try-except`**
Se il file JSON è malformato, il programma potrebbe generare un errore.  
Usiamo `try-except` per gestire il problema.

```python
import json

try:
    with open("dati.json", "r") as file:
        dati = json.load(file)
    print(dati)
except json.JSONDecodeError:
    print("Errore: Il file JSON non è valido!")
except FileNotFoundError:
    print("Errore: Il file JSON non esiste!")
```

✅ **Ora il programma non si blocca se il JSON è errato o il file manca.**  

---

## **🔟 Esercizi**
1️⃣ **Leggi un file JSON e stampa il valore di una chiave specifica.**  
2️⃣ **Stampa tutti i nomi da una lista di oggetti JSON.**  
3️⃣ **Gestisci gli errori quando il file JSON è corrotto o inesistente.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a leggere file JSON con `json.load()`.**  
✅ **Sai accedere ai dati e gestire JSON con liste.**  
✅ **Hai visto come gestire errori con `try-except`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

