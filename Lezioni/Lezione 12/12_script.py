import threading

def stampa_messaggio():
    print("Ciao dal thread!\n")

#CREIAMO UN THREAD
mio_thread = threading.Thread(target = stampa_messaggio)
#tutto il codice è obbligatorio, solo stampa_messaggio varia in base al nome della funzione, ma è importante notare che non sono passati argomenti
#target serve per definire la funzione che poi deve avviare il threrad

#AVVIAMO IL THREAD
mio_thread.start()

print("Questo è il thread principale\n") #l'output è causale tra i due print perche i due processi avvengono contemporaneamente

def saluta_nome(nome):
    print(f"Ciao {nome} dal thread!")

mio_thread = threading.Thread(target=saluta_nome, args=("Marco",))# in questo caso uso per passare gli argomenti della funzione come una tupla *args
mio_thread = threading.Thread(target=saluta_nome, kwargs={"nome": "Marco"})  #ancora passando gli argomenti della funzione come un dizionario **kwargs


def stampa_nome(nome):
    print(f"{nome}")

thread1 = threading.Thread(target = stampa_nome, args = ("Alice",)) #in questo modo è sbagliato, perche passa prima questo di tutti gli altri con args=
thread2 = threading.Thread(target = stampa_nome("Bob"))
thread3 = threading.Thread(target = stampa_nome , args = ("Carlotta",)) #la virgola in args è obbligatoria

thread1.start()
thread2.start()
thread3.start()


class MioThread(threading.Thread):
    def __init__(self,nome):
        super().__init__()# è obbligatorio se non passo come argomento solo il self, e lo devo generare con un'ereditarietà
        self.nome = nome

    def run(self): #questa funzione deve esistere per forza all'interno di ogni classe thread e deve necessariamente chiamarsi run e avere un unico argomento ovvero self , altrimenti non si avvia nulla quando avvio lo start del thread
        print(f"\nThread {self.nome} in esecuzione")

#Creiamo e avviamo il thread

mio_thread = MioThread("Alice") #ho definito un oggetto della classe, che ha un nome
mio_thread.start() #quando avvio il thread praticamente sto avviando la funzione run, l'unico modo per avviarla è che la funzione si chiami run e abbia un solo argomento cioè self 
print("\nDopo la classe\n")

class TuoThread(threading.Thread):
    def run(self):
        self.lavoro()
#richiamo altri metodi ma utilizzando il metodo di run

    def lavoro(self): #posso mettere piu metodi ma uno deve necessariamente essere run
        print(f"Ciao da {self.nome}")

    def __init__(self, nome):
        super().__init__()
        self.nome = nome

tuo_thread = TuoThread("Bob")
tuo_thread.start()


import time

def lavora():
    time.sleep(1) #anche con sleep(0) il programma potrebbe terminare prima che il thread finisca
    print("Lavoro completato")

thread = threading.Thread(target = lavora)
thread.start()

print("\nAspettando che il thread finisca\n")
thread.join()   #In questo modo il programma principale aspetta che il thread finisca, e poi riprende con il codice successivo
#se non facessi ogni volta join(), i thread potrebbero chiudersi in qualsiasi parte del codice e il codice andrebbe avanti lo stesso nel frattempo
print("\nQui il thread è finito e mi dice lavoro completato e poi esegue il prossimo codice del blocco principale dicendo che il thread è teriminato\n")
print("Il thread è terminato\n")


#Se più thread modificano la stessa variabile contemporaneamente, possono verificarsi race conditions.
#Possiamo usare Lock per evitare problemi di concorrenza.


saldo = 100
lock = threading.Lock()

def preleva(quantità):
    global saldo
    lock.acquire()   # with lock:      è un altro metodo invece di acquire() e relaease()
    if saldo >= quantità:
        time.sleep(0.001)  # amplifica la race condition
        saldo -= quantità
        print(f"Prelevati {quantità}, saldo rimanente: {saldo}")

    else:
        print("Saldo insufficiente")
    lock.release()
#lock.acquire() e lock.release() impediscono che i thread modifichino saldo simultaneamente. altrimenti scalerebbe 2 volte al saldo prima 50 con saldo 100 e poi 80 con saldo 100, 
#invece in questo modo chiude il saldo di 100-50 e poi fa il restande -80

thread1 = threading.Thread(target = preleva , args = (50,))
#thread2 = threading.Thread(target = preleva(80))  # ❌ ERRORE perche manca la virgola, e in questo modo passa il target come se fosse una funzione del codice principale e la passa subito senza il thread
thread2 = threading.Thread(target = preleva, args = (80,))

thread1.start()
thread2.start()
thread1.join()
thread2.join()

from concurrent.futures import ThreadPoolExecutor

#Importi ThreadPoolExecutor, una classe che: crea, gestisce, riutilizza un pool di thread.
#Tu non crei né distruggi thread manualmente: lo fa lui.

def saluta(nome):
    print(f"saluta {nome}")

with ThreadPoolExecutor(max_workers=3) as executor:
    #Crei un pool con al massimo 3 thread executor cioè è l’oggetto che: accetta task li assegna ai thread liberi. 
    #Il with è fondamentale: quando esci dal blocco aspetta che tutti i task finiscano chiude correttamente i thread (shutdown(wait=True))
    #Senza with dovresti chiamare manualmente: executor.shutdown() o meglio executor.shutdown(wait=True) sempre se i thread sono stati eseguiti tutti nel modo corretto

    executor.submit(saluta, "alice")
    executor.submit(saluta, "bob")
    executor.submit(saluta, "charlie")

#Internamente succede questo: Il task (saluta("Alice")) viene messo in coda;
#Se c’è un thread libero: lo prende e esegue la funzione, se non c’è: resta in attesa 
#submit() non blocca: ritorna subito.

#Ogni submit() ritorna un Future: future = executor.submit(saluta, "Alice")
#Un Future rappresenta: un task in corso; o completato; o fallito

#future.result()   # aspetta e prende il valore di ritorno
#future.done()     # True se finito
#future.exception()  # eventuale errore

