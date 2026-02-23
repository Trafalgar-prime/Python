# Python - Programmazione Asincrona con `asyncio`

La **programmazione asincrona** consente di eseguire più operazioni contemporaneamente senza bloccare il programma.

---

## **📌 1️⃣ Creare una Funzione Asincrona (`async def`)**
```python
import asyncio

async def stampa_messaggio():
    print("Inizio operazione...")
    await asyncio.sleep(2)  # Simula un'operazione lunga
    print("Operazione completata!")

asyncio.run(stampa_messaggio())
```

---

## **📌 2️⃣ Avviare Più Task Asincroni con `asyncio.gather()`**
```python
import asyncio

async def operazione(nome, tempo):
    print(f"{nome} inizia...")
    await asyncio.sleep(tempo)
    print(f"{nome} completata!")

async def main():
    await asyncio.gather(
        operazione("Task 1", 2),
        operazione("Task 2", 3),
        operazione("Task 3", 1)
    )

asyncio.run(main())
```

---

## **📌 3️⃣ Usare un Loop Asincrono (`async for`)**
```python
import asyncio

async def conta():
    for i in range(5):
        print(i)
        await asyncio.sleep(1)

asyncio.run(conta())
```

---

## **📌 4️⃣ Creare e Lanciare Task con `asyncio.create_task()`**
```python
import asyncio

async def lavoro(nome):
    print(f"{nome} iniziato")
    await asyncio.sleep(2)
    print(f"{nome} completato")

async def main():
    task1 = asyncio.create_task(lavoro("Task A"))
    task2 = asyncio.create_task(lavoro("Task B"))

    print("Entrambi i task sono in esecuzione...")

    await task1
    await task2

asyncio.run(main())
```

---

## **📌 5️⃣ Eseguire un'Operazione con Timeout (`asyncio.wait_for()`)**
```python
import asyncio

async def operazione():
    await asyncio.sleep(5)
    return "Dati ricevuti"

async def main():
    try:
        risultato = await asyncio.wait_for(operazione(), timeout=3)
        print(risultato)
    except asyncio.TimeoutError:
        print("Tempo scaduto!")

asyncio.run(main())
```

---

## **📌 6️⃣ Sincronizzare Task con `asyncio.Lock()`**
```python
import asyncio

lock = asyncio.Lock()

async def operazione(nome):
    async with lock:
        print(f"{nome} sta usando la risorsa...")
        await asyncio.sleep(2)
        print(f"{nome} ha finito!")

async def main():
    await asyncio.gather(
        operazione("Task 1"),
        operazione("Task 2")
    )

asyncio.run(main())
```

---

## **📌 7️⃣ Tutte le Funzioni Principali di `asyncio`**
| Funzione | Descrizione |
|----------|------------|
| `asyncio.run(coroutine)` | Esegue una funzione asincrona |
| `async def nome_funzione()` | Definisce una funzione asincrona |
| `await coroutine()` | Attende il risultato di una funzione asincrona |
| `asyncio.sleep(n)` | Aspetta `n` secondi senza bloccare |
| `asyncio.gather(task1, task2, ...)` | Esegue più task in parallelo |
| `asyncio.create_task(coroutine())` | Crea e avvia un task |
| `asyncio.wait_for(task, timeout=n)` | Imposta un timeout su un task |
| `asyncio.Lock()` | Crea un lock per sincronizzare i task |

---

## **🔟 Esercizi**
1️⃣ **Crea due funzioni asincrone e avviale contemporaneamente con `asyncio.gather()`.**  
2️⃣ **Usa `asyncio.create_task()` per avviare tre funzioni in parallelo.**  
3️⃣ **Proteggi una risorsa condivisa con `asyncio.Lock()`.**  
4️⃣ **Imposta un timeout su una funzione con `asyncio.wait_for()`.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato a usare `asyncio` per la programmazione asincrona in Python.**  
✅ **Sai avviare e gestire più task senza bloccare il programma.**  
✅ **Hai visto tutte le funzioni principali del modulo `asyncio`.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

