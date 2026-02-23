# Guida pratica ad `asyncio` (Python) — appunti completi

Questi appunti raccolgono in modo ordinato ciò che abbiamo visto su `asyncio`: **coroutine**, **task**, **event loop**, `await`, `create_task`, `gather`, **lock**, misurazione dei tempi e gli errori più comuni.

> Nota: il pacchetto si chiama **`asyncio`** (non “asynco”).

---

## 1) Concetti base

### `async def` e coroutine
- Una funzione definita con `async def` **non viene eseguita subito** quando la chiami.
- Quando fai `coro = funzione_async()`, ottieni un **oggetto coroutine**, non il risultato.

Esempio:
```python
async def lavoro():
    return 42

coro = lavoro()  # NON è 42, è una coroutine
```

Per ottenere il risultato devi usare `await` **dentro** un contesto asincrono:
```python
async def main():
    risultato = await lavoro()
    print(risultato)  # 42
```

---

## 2) `await`: cosa significa davvero

`await` significa: “**sospendi questa coroutine finché non ottieni il risultato**”.

- `await` **non blocca tutto il programma**: lascia lavorare l’event loop e permette ad altri task di avanzare.
- Senza `await`, **non puoi usare** il valore di ritorno “più avanti” come se fosse già pronto.

Regola d’oro:
> **Se un valore ti serve, devi `await`-arlo prima di usarlo.**

---

## 3) `asyncio.run()`, event loop e perché senza `await` i task non finiscono

`asyncio.run(main())`:
1. crea un **event loop**
2. esegue `main()`
3. quando `main()` finisce, **chiude l’event loop** (e cancella eventuali task pendenti)

Esempio chiave:

```python
import asyncio

async def lavoro(nome):
    print(f"{nome} iniziato")
    await asyncio.sleep(2)
    print(f"{nome} completato")

async def main():
    task1 = asyncio.create_task(lavoro("Task A"))
    task2 = asyncio.create_task(lavoro("Task B"))
    print("Task creati!")
    # se NON fai await su niente, main finisce subito e il loop si chiude

asyncio.run(main())
```

Se commenti **entrambi** gli `await`, spesso non vedrai mai “completato” perché:
- `main()` finisce
- `asyncio.run()` chiude il loop
- i task pendenti vengono **cancellati**

Se invece `await`-i almeno un task, il loop resta vivo abbastanza e spesso completano entrambi.

---

## 4) `asyncio.create_task()` (concorrenza) e `.result()`

### Creare un task
`create_task` serve per avviare una coroutine in modo concorrente (sullo stesso loop):

```python
task = asyncio.create_task(lavoro("A"))
```

### Ottenere il risultato di un task
- `await task` aspetta la fine e restituisce il valore.
- `task.result()` funziona **solo se il task è già finito**, altrimenti dà errore.

Esempio:
```python
task = asyncio.create_task(lavoro())
task.result()  # ❌ InvalidStateError se non è pronto
```

---

## 5) `asyncio.gather()`: aspettare più cose insieme

`gather` è il modo “pulito” per aspettare più coroutine/task:

```python
await asyncio.gather(
    lavoro("Task 1"),
    lavoro("Task 2"),
)
```

Oppure con task già creati:
```python
t1 = asyncio.create_task(lavoro("Task 1"))
t2 = asyncio.create_task(lavoro("Task 2"))
ris1, ris2 = await asyncio.gather(t1, t2)
```

---

## 6) `asyncio.Lock()`: quando serve e cosa cambia

### A cosa serve
Un `Lock` serve per proteggere una **risorsa condivisa** (dati, file, stampa ordinata, ecc.) e impedire che più task la usino contemporaneamente.

### Effetto sul tempo totale
Se metti **tutto** dentro il lock, i task diventano **seriali**:

- due task con `sleep(3)` → ~6 secondi totali (3 + 3), non ~3.

Esempio:
```python
async def operazione(nome, lock):
    async with lock:
        print(f"Inizio {nome}")
        await asyncio.sleep(3)
        print(f"Fine {nome}")
```

Questo forza l’esecuzione uno alla volta.

### Lock solo sulla sezione critica
Se vuoi concorrenza e lock solo dove serve:
```python
async def operazione(nome, lock):
    async with lock:
        print(f"Inizio {nome}")

    await asyncio.sleep(3)   # fuori dal lock

    async with lock:
        print(f"Fine {nome}")
```

---

## 7) Errore classico: “Lock bound to a different event loop”

Errore:
```
RuntimeError: <asyncio.locks.Lock ...> is bound to a different event loop
```

### Perché succede
Tipicamente succede quando:
- crei `lock = asyncio.Lock()` **globale**
- poi chiami `asyncio.run()` più volte nello stesso file
- il lock resta “legato” al primo loop, e nel secondo loop esplode

### Soluzione consigliata
Crea il lock **dentro** la coroutine che gira nel loop e passalo come argomento:

```python
async def main():
    lock = asyncio.Lock()
    await asyncio.gather(
        operazione("Task 1", lock),
        operazione("Task 2", lock),
    )
```

Oppure, ancora meglio a livello di struttura:
> in uno script, **di solito fai un solo `asyncio.run()`** per tutta l’esecuzione.

---

## 8) Misurare i tempi: usa `time.perf_counter()`

Per misurare durate reali (benchmark), evita `time.time()` e usa:

```python
import time
t0 = time.perf_counter()
...
dt = time.perf_counter() - t0
```

Perché:
- `perf_counter()` è **monotonic** e ad alta risoluzione
- `time.time()` è wall-clock e può avere più jitter

---

## 9) Perché a volte i tempi cambiano anche “senza modificare il codice”

È normale vedere variazioni perché:
- `asyncio.sleep(n)` significa “**non prima di n secondi**”, non “esattamente n”
- lo scheduling del sistema può introdurre jitter (CPU occupata, WSL/VM, terminale lento, I/O di `print`)
- quindi puoi vedere 3.00, 3.20, 3.60… o anche peggio se il sistema è sotto carico

Per rendere i risultati più stabili:
- usa `perf_counter()`
- ripeti più volte e fai media
- riduci rumore (stampe, altre parti dello script)

---

## 10) Esempi completi pronti da copiare

### A) Concorrenza vera con `gather`
```python
import asyncio
import time

async def lavoro(nome):
    print(f"Inizio {nome}")
    await asyncio.sleep(3)
    print(f"Fine {nome}")
    return nome

async def main():
    t0 = time.perf_counter()
    r1, r2 = await asyncio.gather(lavoro("Task 1"), lavoro("Task 2"))
    print("Risultati:", r1, r2)
    print(f"Tempo totale: {time.perf_counter()-t0:.2f}s")

asyncio.run(main())
```

### B) Lock su tutta l’operazione (seriale)
```python
import asyncio
import time

async def lavoro(nome, lock):
    async with lock:
        print(f"Inizio {nome}")
        await asyncio.sleep(3)
        print(f"Fine {nome}")

async def main():
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    await asyncio.gather(
        lavoro("Task 1", lock),
        lavoro("Task 2", lock),
    )
    print(f"Tempo totale: {time.perf_counter()-t0:.2f}s")

asyncio.run(main())
```

### C) Lock solo sulla stampa (concorrenza + ordine)
```python
import asyncio
import time

async def lavoro(nome, lock):
    async with lock:
        print(f"Inizio {nome}")

    await asyncio.sleep(3)

    async with lock:
        print(f"Fine {nome}")

async def main():
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    await asyncio.gather(
        lavoro("Task 1", lock),
        lavoro("Task 2", lock),
    )
    print(f"Tempo totale: {time.perf_counter()-t0:.2f}s")

asyncio.run(main())
```

---

## Mini-checklist mentale (da usare sempre)

- Mi serve un valore? → **`await`**
- Voglio concorrenza? → **`create_task`** o **`gather`**
- Voglio aspettare più operazioni? → **`asyncio.gather(...)`**
- Uso un lock? → Solo sulla **sezione critica**
- Ho più `asyncio.run()` nello stesso file? → rischio loop diversi, evita globali async
- Misuro tempi? → **`time.perf_counter()`**

---

### Fine
Se vuoi, posso aggiungere una sezione “errori comuni” in stile Q/A (con le tue trace reali) oppure un esempio con `asyncio.wait_for()` e gestione `TimeoutError`.
