# Asyncio Python Guide

## Introduzione

`asyncio` è un modulo di Python che permette di scrivere codice **asincrono**, consentendo di gestire operazioni concorrenti senza bloccare l'esecuzione del programma. È utile per operazioni di I/O come richieste di rete, accesso a file o database.

---

## 🚀 Perché usare `asyncio`?

Python esegue il codice in modo **sincrono**, ovvero ogni istruzione viene eseguita una dopo l'altra. Se una funzione impiega molto tempo (ad esempio, una richiesta web), il programma rimane bloccato.

Con `asyncio`, possiamo dire a Python:  
*"Esegui questa operazione e nel frattempo continua a fare altre cose finché non ho bisogno del risultato."*

### 📌 Quando è utile `asyncio`?
- **Operazioni di rete** (es. richieste HTTP a un'API)
- **Lettura/scrittura su file o database** senza bloccare l'esecuzione
- **Gestione di molteplici task contemporaneamente** (es. server web, bot)
- **Esecuzione parallela di operazioni lente** senza dover attendere ogni singolo task

---

## 🛠️ Funzionamento di `asyncio`

### 1️⃣ Concetti base

- `async`: dichiara una funzione asincrona.
- `await`: indica a Python di attendere il risultato senza bloccare il resto del codice.

### 2️⃣ Esempio di base

#### 🐢 **Versione Sincrona (lenta)**

```python
import time

def operazione_lenta(nome):
    print(f"Inizio {nome}...")
    time.sleep(3)
    print(f"Fine {nome}.")

operazione_lenta("Task 1")
operazione_lenta("Task 2")
```
⏳ **Tempo totale ≈ 6 secondi** (le operazioni vengono eseguite una alla volta).

---

#### ⚡ **Versione Asincrona (più veloce!)**

```python
import asyncio

async def operazione_lenta(nome):
    print(f"Inizio {nome}...")
    await asyncio.sleep(3)
    print(f"Fine {nome}.")

async def main():
    await asyncio.gather(
        operazione_lenta("Task 1"),
        operazione_lenta("Task 2")
    )

asyncio.run(main())
```
⚡ **Tempo totale ≈ 3 secondi!**  
Le due operazioni vengono eseguite in parallelo invece che in sequenza.

---

## 🎯 `asyncio.gather()` vs `asyncio.create_task()`

### **1️⃣ `asyncio.gather()`**
- Raccoglie più task e aspetta il loro completamento prima di proseguire.

Esempio:
```python
async def task1():
    await asyncio.sleep(2)
    return "Task 1 completato"

async def task2():
    await asyncio.sleep(1)
    return "Task 2 completato"

async def main():
    risultati = await asyncio.gather(task1(), task2())
    print(risultati)  # ['Task 1 completato', 'Task 2 completato']

asyncio.run(main())
```

---

### **2️⃣ `asyncio.create_task()`**
- Avvia un task in background **senza aspettarlo immediatamente**.

Esempio:
```python
async def lunga_operazione():
    await asyncio.sleep(3)
    print("Operazione completata!")

async def main():
    task = asyncio.create_task(lunga_operazione())

    print("Nel frattempo, il codice continua...")
    await asyncio.sleep(1)
    print("Ancora in esecuzione...")
    
    await task

asyncio.run(main())
```
🔹 **Perfetto per eseguire attività in background senza bloccare il resto del codice!**

---

## 🌍 Esempio pratico: Scaricare più pagine web contemporaneamente

```python
import asyncio
import aiohttp

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = ["https://www.python.org", "https://www.wikipedia.org", "https://www.github.com"]
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print("Pagine scaricate:", len(results))

asyncio.run(main())
```
⚡ Senza `asyncio`, queste pagine sarebbero scaricate **una alla volta**, rallentando il processo.

---

## 📝 Conclusione

✅ `asyncio` **non velocizza il codice**, ma evita blocchi inutili.  
✅ Perfetto per **operazioni I/O asincrone** (API, database, file, bot).  
✅ Usa `await` per **non bloccare il codice principale**.  
✅ `asyncio.gather()` = raccoglie risultati di più task.  
✅ `asyncio.create_task()` = avvia un task in background.

🚀 **Se hai molte operazioni che possono essere eseguite in parallelo, `asyncio` è la soluzione perfetta!**  
