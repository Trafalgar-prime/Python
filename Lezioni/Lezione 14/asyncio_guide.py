# **`asyncio` in Python: Cos'è e perché dovresti usarlo?**

import asyncio
import time
import aiohttp  # type: ignore # Libreria per richieste HTTP asincrone, funziona anche se non pare

# Funzione sincrona che simula un'operazione lenta
def operazione_lenta(nome):
    print(f"Inizio {nome}...")
    time.sleep(3)
    print(f"Fine {nome}.")

# Esecuzione sincrona
def esecuzione_sincrona():
    t0 = time.perf_counter()
    operazione_lenta("Task 1")
    operazione_lenta("Task 2")
    print(f"Tempo totale: {time.perf_counter() - t0:.2f} secondi")

# Funzione asincrona che simula un'operazione lenta
async def operazione_lenta_async(nome):  #async def dichiara una funzione asincrona
    print(f"Inizio {nome}...")
    await asyncio.sleep(3)  #await permette di attendere un'operazione senza bloccare l'intero programma.
    #asyncio.sleep(3) simula un’operazione lunga senza bloccare il resto del codice.
    print(f"Fine {nome}.")

# Esecuzione asincrona con asyncio.gather()
async def esecuzione_asincrona():
    t0 = time.perf_counter()
    await asyncio.gather( #Possiamo eseguire più funzioni asincrone in parallelo con asyncio.gather(), sempre ovviamente richiamando le funzioni già create.
        operazione_lenta_async("Task 1"),
        operazione_lenta_async("Task 2")
    )
    print(f"Tempo totale: {time.perf_counter() - t0:.2f} secondi")

# Funzione per testare asyncio.create_task()
async def test_create_task():
    task = asyncio.create_task(operazione_lenta_async("Operazione in background"))
    print("Nel frattempo, il codice continua...")
    await asyncio.sleep(1)
    print("Ancora in esecuzione...")
    await task

# Funzione per scaricare più pagine web contemporaneamente
async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def scarica_pagine():
    urls = ["https://www.python.org", "https://www.wikipedia.org", "https://www.github.com"]
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print("Pagine scaricate:", len(results))

# Esecuzione dei test
#if __name__ == "__main__":
print("--- Esecuzione sincrona ---")
esecuzione_sincrona()
    
print("\n--- Esecuzione asincrona ---")
asyncio.run(esecuzione_asincrona())  #Con asyncio.run() la funzione viene eseguita in un ciclo di eventi asincrono.
    
print("\n--- Test asyncio.create_task() ---")
asyncio.run(test_create_task())
    
print("\n--- Scaricamento pagine web ---")
asyncio.run(scarica_pagine())

async def conta():
    for i in range(5):
        print(i)
        print("Sto attendendo...")
        await asyncio.sleep(1)  # Attende senza bloccare il resto del codice

print("\n--- Utlizzo del for ---")
asyncio.run(conta())

async def lavoro(nome):
    print(f"{nome} iniziato")
    await asyncio.sleep(2)
    print(f"{nome} completato")

async def main():
    task1 = asyncio.create_task(lavoro("Task A"))
    task2 = asyncio.create_task(lavoro("Task B"))
    #await asyncio.gather(lavoro("Task A"), lavoro("Task B")) # questa sostituisce le due righe sopra e anche i due await sotto

    print("Entrambi i task sono in esecuzione...")

    await task1  # Aspetta la fine del task, e poi ti dice task completato
    await task2
    #await asyncio.gather(task1, task2) #modo diverso di scrivere ma il risultato è lo stesso
    

    #Se commento entrambi gli await il codice va avanti senza completare i task, passando dopo asyncio.run() si chiude il loop e i task non sono completati
    #Se avessi un valore che devo ottenere dai task e i task non si completano il codice va avanti ma sbagliando perchè non conosce il valore 

print("\n--- Utlizzo del task per vedere ---")
asyncio.run(main())

async def operazione():
    await asyncio.sleep(3)  # Simula un'operazione lunga
    return "Dati ricevuti"

async def main():
    try:
        risultato = await asyncio.wait_for(operazione(), timeout=7) #Impostando un timeout maggiore del valore di sleep ottengo i dati, altrimenti se è minore o uguale ottengo tempo scaduto!
        print(risultato)
    except asyncio.TimeoutError:
        print("Tempo scaduto!")

print("\n--- Operazione con timeout nel caso di troppo tempo impiegato ---")
asyncio.run(main())

#lock = asyncio.Lock() #Se più task devono accedere a una risorsa condivisa, usiamo asyncio.Lock().
#se la uso in più di un processo la devo mettere direttamente nelle funzioni

async def operazione(nome,lock):
    async with lock:  #In questo modo i task devono attendere il loro turno e non funzionare tutti insieme
        print(f"{nome} sta usando la risorsa...")
        await asyncio.sleep(2)
        print(f"{nome} ha finito!")

async def main():
    lock = asyncio.Lock()
    await asyncio.gather(
        operazione("Task 1",lock), #usando lo stesso nome per entrambe i processi di questa e della prossima funzioni la variabile lock devo legarla all'interno della funzione 
        operazione("Task 2",lock)
    )
print("\n--- Operazione con lock ---")
asyncio.run(main())

# Funzione asincrona che simula un'operazione lenta
async def operazione_lenta_asyncrona2(nome,lock):  #async def dichiara una funzione asincrona
    async with lock:
        print(f"Inizio {nome}...")
        await asyncio.sleep(3)  #await permette di attendere un'operazione senza bloccare l'intero programma.
        #asyncio.sleep(3) simula un’operazione lunga senza bloccare il resto del codice.
        print(f"Fine {nome}.")

# Esecuzione asincrona con asyncio.gather()
async def esecuzione_asincrona2():
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    await asyncio.gather( #Possiamo eseguire più funzioni asincrone in parallelo con asyncio.gather(), sempre ovviamente richiamando le funzioni già create.
        operazione_lenta_asyncrona2("Task 1",lock),
        operazione_lenta_asyncrona2("Task 2",lock)
    )
    print(f"Tempo totale: {time.perf_counter() - t0:.2f} secondi")

print("\n--- Operazione con lock2, per asincrona ---")
asyncio.run(esecuzione_asincrona2())
print("Questa ci impiega di più perchè con lock ottengo due task separati, mentre senza il lock tutti i task vengono generati insieme")


