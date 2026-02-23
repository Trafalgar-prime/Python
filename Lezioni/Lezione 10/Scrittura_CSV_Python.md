# Python - Scrivere nei File CSV: `csv.writer()` e `writerows()`

Quando si lavora con i file CSV in Python, bisogna utilizzare `csv.writer()` per poter scrivere i dati.  
Non è possibile chiamare direttamente `csv.writerows()` senza prima creare un oggetto scrittore.

---

## **1️⃣ Posso Scrivere più Righe con `writerows()`?**
Sì, ma devi prima creare un **oggetto scrittore (`csv.writer`)**.

### **🔹 Esempio Corretto di `writerows()`**
```python
import csv

dati = [
    ["Nome", "Età", "Città"],
    ["Luca", 30, "Roma"],
    ["Anna", 25, "Milano"]
]

with open("dati.csv", "w", newline="") as file:
    scrittore = csv.writer(file)  # Creo lo scrittore
    scrittore.writerows(dati)  # Scrivo più righe nel CSV
```
✅ **Funziona perché abbiamo creato `scrittore = csv.writer(file)`.**  

---

## **2️⃣ Devo per forza generare uno scrittore (`csv.writer`)?**  
Sì, **senza `csv.writer()` non puoi scrivere nel file CSV**.

🔹 **Esempio ERRATO:**
```python
import csv

with open("dati.csv", "w", newline="") as file:
    csv.writerows([["Nome", "Età", "Città"], ["Luca", 30, "Roma"]])  # ERRORE!
```
🚨 **Errore:**
```
AttributeError: module 'csv' has no attribute 'writerows'
```
✅ **Soluzione:** Creare sempre uno scrittore prima di usare `writerows()`.  

---

## **3️⃣ Aggiungere Dati Senza Sovrascrivere (`a` mode)**
Se vuoi **aggiungere righe senza cancellare i dati esistenti**, usa la modalità **`a` (append)**:

```python
import csv

with open("dati.csv", "a", newline="") as file:
    scrittore = csv.writer(file)
    scrittore.writerow(["Giulia", 22, "Firenze"])  # Aggiunge una nuova riga
```

🔹 **Ora il file conterrà:**
```
Nome,Età,Città
Luca,30,Roma
Anna,25,Milano
Giulia,22,Firenze
```
✅ **Perfetto per aggiornare file CSV senza perdere i dati!**  

---

## **🔟 Esercizi**
1️⃣ **Crea un file CSV e scrivi una riga con `writerow()`.**  
2️⃣ **Scrivi più righe in un CSV con `writerows()`.**  
3️⃣ **Prova a scrivere senza `csv.writer()` e osserva l'errore.**  
4️⃣ **Apri un CSV esistente e aggiungi nuove righe senza sovrascrivere.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato come usare `csv.writer()` per scrivere nei file CSV.**  
✅ **Sai perché `csv.writerows()` richiede prima la creazione di un oggetto scrittore.**  
✅ **Hai visto come aggiungere dati senza sovrascrivere il file.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

