# Python - Approfondimenti sui Moduli, Pacchetti e Creazione di File

Dopo la lezione 11, abbiamo approfondito diversi argomenti legati ai **moduli, pacchetti** e alla **creazione di file** in Python.

---

## **1️⃣ Creare un Modulo in Python**

Un **modulo** è un file Python `.py` che contiene funzioni, classi o variabili riutilizzabili.

### **📌 Esempio di Modulo**
Creiamo un modulo chiamato **`matematica.py`** con alcune funzioni matematiche:

```python
# matematica.py

def somma(a, b):
    return a + b

def sottrazione(a, b):
    return a - b

def moltiplicazione(a, b):
    return a * b

def divisione(a, b):
    if b == 0:
        return "Errore: Divisione per zero!"
    return a / b
```

✅ **Ora possiamo importarlo e usarlo in un altro script!**

---

## **2️⃣ Usare il Modulo in un Altro Script**
Ora possiamo **importare** il modulo in un altro file Python e usare le funzioni.

```python
import matematica  # Importiamo il modulo

print(matematica.somma(10, 5))  # Output: 15
print(matematica.sottrazione(10, 5))  # Output: 5
```

✅ **Il nostro script può ora usare le funzioni di `matematica.py`!**  

---

## **3️⃣ Creare un Pacchetto in Python**

Un **pacchetto** è una **cartella** che contiene più **moduli** (`.py`) e un file speciale chiamato `__init__.py`.

Esempio di pacchetto **`mio_pacchetto`**:
```
mio_pacchetto/       # Cartella del pacchetto
│── __init__.py      # Indica che è un pacchetto
│── matematica.py    # Modulo con funzioni matematiche
│── geometria.py     # Modulo con funzioni geometriche
```

### **📌 Contenuto di `__init__.py`**
```python
# __init__.py
from .matematica import somma, sottrazione
from .geometria import area_quadrato, perimetro_quadrato
```

✅ **Ora il pacchetto è pronto per essere usato!**

### **📌 Importare il Pacchetto in uno Script**
```python
import mio_pacchetto.matematica as mat
import mio_pacchetto.geometria as geo

print(mat.somma(5, 3))  # Output: 8
print(geo.area_quadrato(4))  # Output: 16
```

✅ **Il pacchetto è stato importato e usato con successo!**

---

## **4️⃣ Creare un File Vuoto in Python, Windows e Linux/Mac**

Puoi creare un file vuoto in diversi modi.

### **📌 1. Creare un File Vuoto con Python**
```python
open("miofile.txt", "w").close()
```

### **📌 2. Creare un File Vuoto in Windows (Prompt dei Comandi)**
```powershell
type nul > miofile.txt
New-Item miofile.txt  # Con PowerShell
```

### **📌 3. Creare un File Vuoto in Linux/Mac (Terminale)**
```bash
touch miofile.txt
```

### **📌 4. Creare un File Vuoto Manualmente**
- **Windows**: **Tasto destro → Nuovo → Documento di testo**  
- **Mac/Linux**: **Clic destro → Nuovo Documento**  

✅ **Ora sai come creare file vuoti in diversi modi!**

---

## **🔟 Esercizi**
1️⃣ **Crea un modulo con tre funzioni e usale in un altro script.**  
2️⃣ **Crea un pacchetto chiamato `calcoli` con due moduli e importali in un file principale.**  
3️⃣ **Prova a creare un file vuoto con Python e verifica che esista.**  
4️⃣ **Prova a importare un pacchetto da una cartella diversa con `sys.path.append()`.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare moduli e pacchetti in Python.**  
✅ **Sai importare funzioni e usarle in altri script.**  
✅ **Hai capito come creare file vuoti in Python, Windows e Linux.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

