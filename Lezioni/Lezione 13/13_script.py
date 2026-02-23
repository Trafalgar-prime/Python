import multiprocessing
import time

def stampa_messaggio():
    time.sleep(1)
    print("\nCiao dal processo!", flush = True)
#Su alcuni sistemi, l'output viene bufferizzato(cioè gestito in un'altra area della memoria) e potrebbe non apparire immediatamente. Puoi forzare la stampa immediata aggiungendo flush=True.

# Creiamo un nuovo processo
mio_processo = multiprocessing.Process(target=stampa_messaggio)

# Avviamo il processo
mio_processo.start()

# Il processo principale continua l'esecuzione
print("Questo è il processo principale")

# Aspettiamo che il processo termini
mio_processo.join()

def stampa_nome(nome):
    print(f"Processo eseguito da: {nome}")

processo1 = multiprocessing.Process(target=stampa_nome, args=("Alice",))
processo2 = multiprocessing.Process(target=stampa_nome, args=("Bob",))

processo1.start()
processo2.start()

processo1.join()
processo2.join()

class MioProcesso(multiprocessing.Process):
    def __init__(self, nome):
        super().__init__()
        self.nome = nome

    def run(self):
        print(f"Processo {self.nome} in esecuzione")

# Creiamo e avviamo il processo
mio_processo = MioProcesso("Alice")
mio_processo.start()
mio_processo.join()

def scrivi_nella_coda(coda):
    coda.put("Messaggio dal processo figlio")

coda = multiprocessing.Queue() #I processi non condividono memoria, quindi devi usare una coda (Queue) per scambiare dati.
#Le Queue permettono di inviare dati tra processi.

processo = multiprocessing.Process(target=scrivi_nella_coda, args=(coda,))
processo.start()
processo.join()

# Leggiamo il messaggio dalla coda
messaggio = coda.get()
print(f"Messaggio ricevuto: {messaggio}")
#senza il Queue() non potrei stampare il messaggio figlio attraverso il processo principale, perchè messaggio deriva dal principale e non dal figlio

def stampa_con_lock(lock, messaggio):
    lock.acquire()
    print(messaggio)
    lock.release()

lock = multiprocessing.Lock()
#Il Lock impedisce che più processi scrivano contemporaneamente su una risorsa.

processo1 = multiprocessing.Process(target=stampa_con_lock, args=(lock, "Processo 1"))
processo2 = multiprocessing.Process(target=stampa_con_lock, args=(lock, "Processo 2"))

processo1.start()
processo2.start()

processo1.join()
processo2.join()

def quadrato(n):
    return n * n

with multiprocessing.Pool(processes=4) as pool:
    risultati = pool.map(quadrato, [1, 2, 3, 4, 5, 6])
    print(risultati)  # Output: [1, 4, 9, 16, 25]





if __name__ == "__main__": #Su Windows, i processi devono essere avviati all'interno di un blocco if __name__ == "__main__" per evitare che il modulo multiprocessing venga rieseguito all'infinito.

    #creiamo un nuovo processo
    mio_processo = multiprocessing.Process(target = stampa_messaggio)

    #avviamo il processo
    mio_processo.start()


    #il processo principale continua l'esecuzione
    print("\nQuesto è il processo principale")

    # Aspettiamo che il processo termini
    mio_processo.join() #non leggerà nulla perchè il multiprocessing su idle su windows da problemi, se lo runno dal terminale cmd ottengo il risultato giusto

import multiprocessing
if __name__ == "__main__": #Su Windows, i processi devono essere avviati all'interno di un blocco if __name__ == "__main__" per evitare che il modulo multiprocessing venga rieseguito all'infinito.
    def stampa_nome(nome):
        print(f"Processo eseguito da: {nome}")

    processo1 = multiprocessing.Process(target=stampa_nome, args=("Alice",))
    processo2 = multiprocessing.Process(target=stampa_nome, args=("Bob",))

    processo1.start()
    processo2.start()

    processo1.join()
    processo2.join()


