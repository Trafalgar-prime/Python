# Python - Lezione 9: Gestione delle Eccezioni

In Python, le **eccezioni** vengono usate per gestire errori ed evitare che il programma si blocchi.

---

## **1️⃣ Cosa Sono le Eccezioni?**
Un'**eccezione** è un errore che **interrompe l'esecuzione** del programma.  
Se non gestita, il programma termina con un **messaggio di errore**.

Esempio di errore senza gestione:
```python
x = 10 / 0  # ERRORE! Divisione per zero
```
🔹 **Output:**  
```
ZeroDivisionError: division by zero
```

---

## **2️⃣ Usare `try` e `except` per Catturare le Eccezioni**
```python
try:
    x = 10 / 0  # Operazione rischiosa
except ZeroDivisionError:
    print("Errore: Non puoi dividere per zero!")
```

---

## **3️⃣ Gestire Più Eccezioni**
```python
try:
    numero = int(input("Inserisci un numero: "))  # ERRORE se l'input non è un numero
    risultato = 10 / numero  # ERRORE se numero = 0
except ZeroDivisionError:
    print("Errore: Non puoi dividere per zero!")
except ValueError:
    print("Errore: Devi inserire un numero valido!")
```

---

## **4️⃣ Usare `except Exception` per Catturare Tutti gli Errori**
```python
try:
    x = int("abc")  # ERRORE
except Exception as e:
    print("Errore generico:", e)
```

---

## **5️⃣ Usare `else` con `try`**
```python
try:
    numero = int(input("Inserisci un numero: "))
    risultato = 10 / numero
except ZeroDivisionError:
    print("Errore: Non puoi dividere per zero!")
except ValueError:
    print("Errore: Devi inserire un numero valido!")
else:
    print("Risultato:", risultato)  # Stampato solo se non ci sono errori
```

---

## **6️⃣ Usare `finally` per il Codice Sempre Eseguito**
```python
try:
    f = open("file.txt", "r")  # ERRORE se il file non esiste
    contenuto = f.read()
except FileNotFoundError:
    print("Errore: Il file non esiste!")
finally:
    print("Operazione completata.")  # Stampato sempre
```

---

## **7️⃣ Sollevare un'eccezione con `raise`**
```python
def verifica_età(età):
    if età < 18:
        raise ValueError("Errore: Devi avere almeno 18 anni!")  # Errore personalizzato
    return "Accesso consentito"

try:
    print(verifica_età(16))  # ERRORE
except ValueError as e:
    print(e)
```

---

## **8️⃣ Creare Eccezioni Personalizzate**
```python
class ErrorePersonalizzato(Exception):
    pass  # Creiamo una classe che eredita da Exception

def controlla_numero(n):
    if n < 0:
        raise ErrorePersonalizzato("Errore: Il numero non può essere negativo!")

try:
    controlla_numero(-5)
except ErrorePersonalizzato as e:
    print(e)
```

---

## **🔟 Esercizi**
1️⃣ **Scrivi un programma che gestisce un errore di input (`ValueError`).**  
2️⃣ **Gestisci un errore di divisione per zero (`ZeroDivisionError`).**  
3️⃣ **Apri un file inesistente e cattura l'errore (`FileNotFoundError`).**  
4️⃣ **Crea un'eccezione personalizzata se un numero è negativo.**  
5️⃣ **Scrivi una funzione che verifica l'età e solleva un'eccezione se è minore di 18.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a gestire errori con `try`, `except`, `else` e `finally`.**  
✅ **Sai catturare errori specifici e generali.**  
✅ **Hai visto come sollevare errori con `raise` e creare eccezioni personalizzate.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

