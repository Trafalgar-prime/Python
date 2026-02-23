# Python - Programmazione Multi-threading (Dettagliata)

Il **multi-threading** consente di eseguire più attività contemporaneamente all'interno dello stesso processo.

---

# **📌 1️⃣ Creare e Avviare un Thread (`Thread()`)**
```python
import threading

def stampa_messaggio():
    print("Ciao dal thread!")

mio_thread = threading.Thread(target=stampa_messaggio)
mio_thread.start()

print("Questo è il thread principale")
```

---

# **📌 2️⃣ Creare un Thread con Argomenti (`args`)**
```python
import threading

def stampa_nome(nome):
    print(f"Thread eseguito da: {nome}")

thread1 = threading.Thread(target=stampa_nome, args=("Alice",))
thread2 = threading.Thread(target=stampa_nome, args=("Bob",))

thread1.start()
thread2.start()
```

---

# **📌 3️⃣ Usare una Classe per Creare un Thread**
```python
import threading

class MioThread(threading.Thread):
    def __init__(self, nome):
        super().__init__()
        self.nome = nome

    def run(self):
        print(f"Thread {self.nome} in esecuzione")

mio_thread = MioThread("Alice")
mio_thread.start()
```

---

# **📌 4️⃣ Aspettare la Fine di un Thread (`join()`)**
```python
import threading
import time

def lavora():
    time.sleep(2)
    print("Lavoro completato!")

thread = threading.Thread(target=lavora)
thread.start()

print("Aspettando che il thread finisca...")
thread.join()
print("Il thread è terminato")
```

---

# **📌 5️⃣ Sincronizzare i Thread con `Lock`**
```python
import threading

saldo = 100
lock = threading.Lock()

def preleva(quantità):
    global saldo
    lock.acquire()
    if saldo >= quantità:
        saldo -= quantità
        print(f"Prelevati {quantità}, saldo rimanente: {saldo}")
    else:
        print("Saldo insufficiente")
    lock.release()

thread1 = threading.Thread(target=preleva, args=(50,))
thread2 = threading.Thread(target=preleva, args=(80,))

thread1.start()
thread2.start()
thread1.join()
thread2.join()
```

---

# **📌 6️⃣ Usare `ThreadPoolExecutor` per Creare Thread in Modo Semplice**
```python
from concurrent.futures import ThreadPoolExecutor

def saluta(nome):
    print(f"Ciao {nome}!")

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(saluta, "Alice")
    executor.submit(saluta, "Bob")
    executor.submit(saluta, "Charlie")
```

---

# **📌 7️⃣ Tutte le Funzioni del Modulo `threading`**
| Funzione | Descrizione |
|----------|------------|
| `threading.Thread(target=funzione)` | Crea un thread |
| `thread.start()` | Avvia il thread |
| `thread.join()` | Attende la fine del thread |
| `thread.is_alive()` | Controlla se un thread è ancora in esecuzione |
| `thread.daemon = True` | Imposta il thread come "daemon" (si chiude con il programma) |
| `threading.Lock()` | Crea un Lock per evitare race conditions |
| `lock.acquire()` | Blocca l'accesso agli altri thread |
| `lock.release()` | Sblocca l'accesso agli altri thread |
| `threading.Semaphore(n)` | Permette a massimo `n` thread di eseguire un'operazione |
| `threading.current_thread()` | Restituisce il thread attuale |
| `threading.enumerate()` | Restituisce tutti i thread attivi |
| `threading.active_count()` | Conta i thread attivi |

---

# **🔟 Esercizi**
1️⃣ **Crea un programma che esegue due thread contemporaneamente.**  
2️⃣ **Usa `join()` per far sì che il programma principale aspetti la fine dei thread.**  
3️⃣ **Proteggi una variabile condivisa con `threading.Lock()`.**  
4️⃣ **Usa `ThreadPoolExecutor` per eseguire più funzioni in parallelo.**  

---

# ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare e gestire thread in Python.**  
✅ **Sai sincronizzare i thread con `Lock`.**  
✅ **Hai visto tutte le funzioni del modulo `threading`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

