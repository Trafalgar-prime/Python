# Python - Decoratori in Python

I **decoratori** permettono di modificare il comportamento di una funzione senza modificarne il codice direttamente.

---

## **📌 1️⃣ Creare un Decoratore Base**
```python
def decoratore(f):
    def wrapper():
        print("Prima dell'esecuzione")
        f()
        print("Dopo l'esecuzione")
    return wrapper

@decoratore
def saluta():
    print("Ciao!")

saluta()
```

---

## **📌 2️⃣ Decoratore Generico con Argomenti**
```python
def decoratore(f):
    def wrapper(*args, **kwargs):
        print(f"Chiamata a {f.__name__} con argomenti: {args}, {kwargs}")
        risultato = f(*args, **kwargs)
        print(f"Risultato: {risultato}")
        return risultato
    return wrapper

@decoratore
def somma(a, b):
    return a + b

somma(3, 5)
```

---

## **📌 3️⃣ Applicare Più Decoratori**
```python
def decoratore1(f):
    def wrapper(*args, **kwargs):
        print("Decoratore 1 - Prima")
        risultato = f(*args, **kwargs)
        print("Decoratore 1 - Dopo")
        return risultato
    return wrapper

def decoratore2(f):
    def wrapper(*args, **kwargs):
        print("Decoratore 2 - Prima")
        risultato = f(*args, **kwargs)
        print("Decoratore 2 - Dopo")
        return risultato
    return wrapper

@decoratore1
@decoratore2
def saluta():
    print("Ciao!")

saluta()
```

---

## **📌 4️⃣ Usare `functools.wraps` per Mantenere le Informazioni della Funzione**
```python
import functools

def decoratore(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Eseguendo {f.__name__}")
        return f(*args, **kwargs)
    return wrapper

@decoratore
def esempio():
    """Questa è una funzione di esempio"""
    print("Funzione originale")

print(esempio.__name__)  # ✅ Output: esempio
print(esempio.__doc__)   # ✅ Output: Questa è una funzione di esempio
```

---

## **📌 5️⃣ Misurare il Tempo di Esecuzione**
```python
import time

def tempo_di_esecuzione(f):
    def wrapper(*args, **kwargs):
        inizio = time.time()
        risultato = f(*args, **kwargs)
        fine = time.time()
        print(f"Tempo di esecuzione: {fine - inizio:.4f} secondi")
        return risultato
    return wrapper

@tempo_di_esecuzione
def operazione_lenta():
    time.sleep(2)
    print("Operazione completata!")

operazione_lenta()
```

---

## **📌 6️⃣ Decoratore per Autenticazione**
```python
def richiede_autenticazione(f):
    def wrapper(*args, **kwargs):
        utente_autenticato = False  # Cambia in True per testare
        if not utente_autenticato:
            print("Accesso negato!")
            return
        return f(*args, **kwargs)
    return wrapper

@richiede_autenticazione
def area_riservata():
    print("Benvenuto nell'area riservata!")

area_riservata()
```

---

## **📌 7️⃣ Decoratore per Caching**
```python
import functools

def cache(f):
    memoria = {}

    @functools.wraps(f)
    def wrapper(*args):
        if args in memoria:
            print("Restituisco valore dalla cache")
            return memoria[args]
        risultato = f(*args)
        memoria[args] = risultato
        return risultato
    return wrapper

@cache
def quadrato(n):
    print(f"Calcolando quadrato di {n}")
    return n * n

print(quadrato(4))  # Calcolo
print(quadrato(4))  # Cache
```

---

## **📌 8️⃣ Funzioni Utili con i Decoratori**
| Funzione | Descrizione |
|----------|------------|
| `@decoratore` | Applica un decoratore a una funzione |
| `functools.wraps(f)` | Mantiene il nome e la docstring della funzione originale |
| `*args, **kwargs` | Permette al decoratore di accettare qualsiasi argomento |
| `time.time()` | Misura il tempo di esecuzione |
| `functools.lru_cache()` | Crea una cache automatica |

---

## **🔟 Esercizi**
1️⃣ **Crea un decoratore che stampa "Inizio" e "Fine" prima e dopo l’esecuzione di una funzione.**  
2️⃣ **Crea un decoratore che misura il tempo di esecuzione di una funzione.**  
3️⃣ **Crea un decoratore che impedisce l’esecuzione di una funzione se l’utente non è autenticato.**  
4️⃣ **Crea un decoratore che salva i risultati di una funzione in un dizionario cache.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare e usare i decoratori in Python.**  
✅ **Sai applicare più decoratori e usare `functools.wraps`.**  
✅ **Hai visto esempi pratici per logging, caching e autenticazione.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

