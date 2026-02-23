# Python - Creare un Modulo Personalizzato

Un **modulo** in Python è semplicemente un file `.py` che contiene funzioni, classi o variabili riutilizzabili.  
Puoi creare un tuo modulo e importarlo in altri script per organizzare meglio il codice.

---

## **1️⃣ Creare un Modulo**
Un modulo è solo un **file Python con estensione `.py`**.  

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

✅ **Ora abbiamo un modulo Python chiamato `matematica.py`!**  

---

## **2️⃣ Usare il Modulo in un Altro Script**
Ora possiamo **importare** il modulo in un altro file Python e usare le funzioni.  
Creiamo un file chiamato **`main.py`** nello stesso percorso e scriviamo:

```python
import matematica  # Importiamo il modulo

print(matematica.somma(10, 5))  # Output: 15
print(matematica.sottrazione(10, 5))  # Output: 5
```

✅ **Il nostro script può ora usare le funzioni di `matematica.py`!**  

---

## **3️⃣ Differenti Modi di Importare il Modulo**
| Metodo | Sintassi | Accesso alle Funzioni |
|--------|---------|----------------------|
| **Importare tutto il modulo** | `import matematica` | `matematica.somma(3,4)` |
| **Importare solo una funzione** | `from matematica import somma` | `somma(3,4)` |
| **Importare più funzioni** | `from matematica import somma, sottrazione` | `somma(3,4)` |
| **Importare con un alias** | `import matematica as m` | `m.somma(3,4)` |

---

## **4️⃣ Dove Salvare il Modulo?**
Per poter importare il modulo senza errori:
- Deve essere **nella stessa cartella** dello script che lo importa.  
- Oppure deve essere in una **cartella inclusa nel `sys.path`** di Python.

Se il modulo si trova in una cartella diversa, puoi **aggiungere il percorso manualmente**:

```python
import sys
sys.path.append("/percorso/della/cartella")
import matematica
```

---

## **🔟 Esercizi**
1️⃣ **Crea un modulo con tre funzioni e usale in un altro script.**  
2️⃣ **Importa solo una funzione dal modulo e usala.**  
3️⃣ **Prova a usare `sys.path.append()` per importare un modulo da un’altra cartella.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare un modulo personalizzato.**  
✅ **Sai importarlo e usarlo in altri script.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

