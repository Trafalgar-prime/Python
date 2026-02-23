# Python - Differenza tra `r+` e `a+` nella Gestione dei File

In Python, la modalità di apertura di un file influisce sul comportamento della lettura e scrittura.  
Vediamo la differenza tra `r+` (read & write) e `a+` (append & read).

---

## **1️⃣ `r+` Sovrascrive il Testo**

Quando apriamo un file in **modalità `r+`**, il cursore inizia all'inizio del file.  
Se scriviamo, il testo **sovrascrive** i caratteri esistenti.

### **🔹 Esempio di Sovrascrittura con `r+`**

Se il file `testo.txt` contiene:
```
Questo è un esempio di testo.
```
e eseguiamo:
```python
with open("testo.txt", "r+") as file:
    file.write("CIAO")
```
🔹 Il file ora conterrà:
```
CIAOo è un esempio di testo.
```
✅ **I primi 4 caratteri sono stati sovrascritti!**  

---

## **2️⃣ `a+` Aggiunge Senza Sovrascrivere**

La modalità **`a+`** apre il file per lettura e scrittura, ma il cursore parte dalla fine, quindi il nuovo testo viene **aggiunto senza sovrascrivere**.

```python
with open("testo.txt", "a+") as file:
    file.write("
Nuova riga aggiunta.")  # Aggiunge in fondo senza sovrascrivere
```

---

## **3️⃣ Come Evitare la Sovrascrittura con `r+`?**

Se vuoi usare `r+` ma senza sovrascrivere, sposta il cursore alla fine prima di scrivere:

```python
with open("testo.txt", "r+") as file:
    file.seek(0, 2)  # Sposta il cursore alla fine del file
    file.write("
Nuova riga aggiunta.")
```

✅ **Ora il nuovo testo viene aggiunto alla fine.**  

---

## **4️⃣ Differenza tra `r+` e `a+`**

| Modalità | Lettura | Scrittura | Cancella il contenuto? | Posizione iniziale del cursore |
|----------|---------|-----------|------------------------|--------------------------------|
| `r+` | ✅ Sì | ✅ Sì (sovrascrive) | ❌ No | All'inizio del file |
| `a+` | ✅ Sì | ✅ Sì (aggiunge) | ❌ No | Alla fine del file |

---

## **🔟 Esercizi**
1️⃣ **Crea un file e scrivi un testo iniziale.**  
2️⃣ **Usa `r+` per sovrascrivere solo una parte del testo.**  
3️⃣ **Usa `a+` per aggiungere una nuova riga senza modificare il resto.**  
4️⃣ **Leggi il file dopo aver scritto per verificare il risultato.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato la differenza tra `r+` e `a+` nella gestione dei file.**  
✅ **Sai come evitare la sovrascrittura usando `seek(0, 2)`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

