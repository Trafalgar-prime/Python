# Python - Programmazione Multi-threading

Il **multi-threading** permette a Python di eseguire più attività contemporaneamente all'interno dello stesso processo.

---

## **1️⃣ Cosa Sono i Thread?**  
Un **thread** è un'unità di esecuzione che può funzionare in parallelo con altri thread nello stesso programma.

🔹 **Vantaggi del Multi-threading:**  
✅ Migliora la reattività di programmi con operazioni lente (I/O, rete).  
✅ Permette di eseguire più operazioni contemporaneamente.  

🔹 **Limiti del Multi-threading in Python:**  
❌ Python ha il **Global Interpreter Lock (GIL)** che limita il multi-threading per il calcolo pesante.  
❌ Meglio usare **multiprocessing** se vuoi sfruttare più core della CPU.  

---

## **2️⃣ Creare e Avviare un Thread**
```python
import threading

def saluta():
    print("Ciao dal thread!")

mio_thread = threading.Thread(target=saluta)
mio_thread.start()

print("Questo è il thread principale")
```

---

## **3️⃣ Creare più Thread**
```python
import threading

def stampa_messaggio(messaggio):
    print(messaggio)

thread1 = threading.Thread(target=stampa_messaggio, args=("Thread 1 in esecuzione",))
thread2 = threading.Thread(target=stampa_messaggio, args=("Thread 2 in esecuzione",))

thread1.start()
thread2.start()

print("Thread principale continua...")
```

---

## **4️⃣ Usare una Classe per Creare un Thread**
```python
import threading

class MioThread(threading.Thread):
    def run(self):
        print(f"Thread {self.name} in esecuzione")

mio_thread = MioThread()
mio_thread.start()
```

---

## **5️⃣ Aspettare la Fine di un Thread (`join()`)**
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

## **6️⃣ Sincronizzare i Thread con Lock**
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

## **7️⃣ Usare `ThreadPoolExecutor` per Creare Thread in Modo Semplice**
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

## **🔟 Esercizi**
1️⃣ **Crea un programma che esegue due thread contemporaneamente.**  
2️⃣ **Usa `join()` per far sì che il programma principale aspetti la fine dei thread.**  
3️⃣ **Proteggi una variabile condivisa con `threading.Lock()`.**  
4️⃣ **Usa `ThreadPoolExecutor` per eseguire più funzioni in parallelo.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a creare e gestire thread in Python.**  
✅ **Sai sincronizzare i thread con `Lock`.**  
✅ **Hai visto come usare `ThreadPoolExecutor` per la gestione dei thread.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

