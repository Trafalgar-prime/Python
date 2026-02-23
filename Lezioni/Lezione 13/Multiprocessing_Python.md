# Python - Multiprocessing (Eseguire Codice su Più Core della CPU)

Il **multiprocessing** permette di eseguire più processi separati, evitando il **Global Interpreter Lock (GIL)** di Python.

---

## **📌 1️⃣ Differenza tra Multi-threading e Multiprocessing**
| **Caratteristica**      | **Multi-threading** | **Multiprocessing** |
|------------------------|----------------|----------------|
| Usa più core della CPU? | ❌ No (bloccato dal GIL) | ✅ Sì |
| Condivisione della memoria | ✅ Sì | ❌ No (processi separati) |
| Utile per... | I/O (file, rete, database) | Calcoli pesanti |
| Modulo Python | `threading` | `multiprocessing` |

---

## **📌 2️⃣ Creare un Processo con `Process()`**
```python
import multiprocessing

def stampa_messaggio():
    print("Ciao dal processo!")

mio_processo = multiprocessing.Process(target=stampa_messaggio)
mio_processo.start()
mio_processo.join()
```

---

## **📌 3️⃣ Creare più Processi**
```python
import multiprocessing

def stampa_nome(nome):
    print(f"Processo eseguito da: {nome}")

processo1 = multiprocessing.Process(target=stampa_nome, args=("Alice",))
processo2 = multiprocessing.Process(target=stampa_nome, args=("Bob",))

processo1.start()
processo2.start()

processo1.join()
processo2.join()
```

---

## **📌 4️⃣ Creare una Classe per un Processo**
```python
import multiprocessing

class MioProcesso(multiprocessing.Process):
    def __init__(self, nome):
        super().__init__()
        self.nome = nome

    def run(self):
        print(f"Processo {self.nome} in esecuzione")

mio_processo = MioProcesso("Alice")
mio_processo.start()
mio_processo.join()
```

---

## **📌 5️⃣ Usare una `Queue` per Scambiare Dati tra Processi**
```python
import multiprocessing

def scrivi_nella_coda(coda):
    coda.put("Messaggio dal processo figlio")

coda = multiprocessing.Queue()

processo = multiprocessing.Process(target=scrivi_nella_coda, args=(coda,))
processo.start()
processo.join()

messaggio = coda.get()
print(f"Messaggio ricevuto: {messaggio}")
```

---

## **📌 6️⃣ Sincronizzare i Processi con `Lock`**
```python
import multiprocessing

def stampa_con_lock(lock, messaggio):
    lock.acquire()
    print(messaggio)
    lock.release()

lock = multiprocessing.Lock()

processo1 = multiprocessing.Process(target=stampa_con_lock, args=(lock, "Processo 1"))
processo2 = multiprocessing.Process(target=stampa_con_lock, args=(lock, "Processo 2"))

processo1.start()
processo2.start()

processo1.join()
processo2.join()
```

---

## **📌 7️⃣ Creare più Processi con `Pool`**
```python
import multiprocessing

def quadrato(n):
    return n * n

with multiprocessing.Pool(processes=4) as pool:
    risultati = pool.map(quadrato, [1, 2, 3, 4, 5])
    print(risultati)  # Output: [1, 4, 9, 16, 25]
```

---

## **📌 8️⃣ Tutte le Funzioni del Modulo `multiprocessing`**
| Funzione | Descrizione |
|----------|------------|
| `multiprocessing.Process(target=funzione)` | Crea un processo |
| `process.start()` | Avvia il processo |
| `process.join()` | Attende la fine del processo |
| `process.is_alive()` | Controlla se un processo è ancora attivo |
| `multiprocessing.Queue()` | Crea una coda per scambiare dati tra processi |
| `multiprocessing.Lock()` | Blocca l'accesso a una risorsa condivisa |
| `lock.acquire()` | Acquisisce il lock |
| `lock.release()` | Rilascia il lock |
| `multiprocessing.Pool(n)` | Crea un pool di `n` processi |

---

## **🔟 Esercizi**
1️⃣ **Crea due processi che eseguono funzioni diverse in parallelo.**  
2️⃣ **Usa una `Queue` per scambiare dati tra processi.**  
3️⃣ **Proteggi una risorsa condivisa con `multiprocessing.Lock()`.**  
4️⃣ **Usa `Pool` per eseguire una funzione su una lista di numeri.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare e gestire processi in Python.**  
✅ **Sai sincronizzare i processi con `Lock`.**  
✅ **Hai visto tutte le funzioni del modulo `multiprocessing`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

