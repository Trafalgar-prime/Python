# Python - Moduli e Pacchetti

Python permette di suddividere il codice in **moduli** e organizzarli in **pacchetti**.  
Questa lezione spiega come creare, importare e usare moduli e pacchetti.

---

## **1️⃣ Cos'è un Modulo in Python?**  
Un **modulo** è un file Python (`.py`) che contiene **funzioni, classi o variabili** riutilizzabili.

### **🔹 Creare un Modulo**
```python
# matematica.py
def somma(a, b):
    return a + b
```

✅ **Ora possiamo importarlo e usarlo in un altro script!**  

---

## **2️⃣ Importare un Modulo**  

### **🔹 Importare un modulo intero**
```python
import matematica
print(matematica.somma(3, 4))  # Output: 7
```

### **🔹 Importare solo una funzione**
```python
from matematica import somma
print(somma(5, 6))  # Output: 11
```

### **🔹 Importare con un alias**
```python
import matematica as m
print(m.somma(2, 3))  # Output: 5
```

---

## **3️⃣ Differenza tra `import` e `from ... import`**
| Metodo | Esempio | Accesso |
|--------|--------|--------|
| `import` | `import matematica` | `matematica.somma(3,4)` |
| `from ... import` | `from matematica import somma` | `somma(3,4)` |
| `import ... as` | `import matematica as m` | `m.somma(3,4)` |

✅ **Se vuoi solo una funzione, usa `from ... import`.**  
✅ **Se vuoi tutto il modulo, usa `import`.**  

---

## **4️⃣ Moduli Python Integrati**

### **🔹 `math` - Funzioni matematiche avanzate**
```python
import math
print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793
```

### **🔹 `random` - Generazione di numeri casuali**
```python
import random
print(random.randint(1, 10))  # Numero casuale tra 1 e 10
```

### **🔹 `datetime` - Lavorare con date e orari**
```python
import datetime
oggi = datetime.date.today()
print(oggi)  # YYYY-MM-DD
```

---

## **5️⃣ Creare un Pacchetto**
Un **pacchetto** è una cartella che contiene **moduli** e un file speciale `__init__.py`.  
Esempio di struttura:

```
mio_pacchetto/
│── __init__.py  # Indica che è un pacchetto
│── matematica.py
│── geometria.py
```

📌 **Importare dal pacchetto**:
```python
from mio_pacchetto import matematica
print(matematica.somma(4, 5))  # Output: 9
```

---

## **6️⃣ Installare e Usare Moduli Esterni**
Puoi installare moduli extra usando **`pip`**.

### **🔹 Installare un modulo**
```bash
pip install requests
```

### **🔹 Usare un modulo installato**
```python
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # Output: 200
```

---

## **🔟 Esercizi**
1️⃣ **Crea un modulo con una funzione e importalo in un altro script.**  
2️⃣ **Usa `math` per calcolare la radice quadrata di un numero.**  
3️⃣ **Installa e usa un modulo esterno (`requests`).**  
4️⃣ **Crea un pacchetto con due moduli e importali in un file principale.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare e importare moduli.**  
✅ **Sai usare pacchetti per organizzare il codice.**  
✅ **Sai installare librerie con `pip`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

