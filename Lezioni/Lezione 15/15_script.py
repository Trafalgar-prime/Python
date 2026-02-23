def decoratore(f):
    def wrapper():  #wrapper non è un nome obbligatorio ma si usa per consuetudine
        print("Prima dell'esecuzione")
        f()
        print("Dopo l'esecuzione")
    return wrapper
#praticamente wrapper avvolge la funzione facendo qualcosa prima o dopo o in entrambi i casi

@decoratore #applica il decoratore alla funzione, altrimenti non vale il decoratore
#@decoratore perche la funzione si chiama decoratore: se avesse def pluto... -> @pluto
def saluta():
    print("ciao")

saluta() #attivo la funzione saluta, ma anche il decoratore sulla funzione

def decoratore(f):
    def wrapper(*args , **kwargs): #in questo modo creo una funzione generica che può accettare ogni parametro, quindi utile per ogni caso
        print(f"Chiamata a {f.__name__} con argomenti : {args} , {kwargs}")  #f.__name__ chiama il nome della funzione, args sta per le tuple, kwargs per i dizionari
        risultato = f(*args , **kwargs) #è il risultato della funzione
        print(f"Risultato = {risultato}")
        return risultato #mi ritorna il risultato della funzione che ho dato in wrapper
    return wrapper # mi ritorna dal decoratore, in modo da richiamare la funzione wrapper, con @decoratore richiamo la funzione decoratore, ma questa senza il return wrapper che mi attiva la funzione wrapper non farebbe nulla

@decoratore
def somma(a,b,**kwargs):# quando definisco la funzione in f ho gia dato tutti i possibili argomenti, quindi cosi mi chiamo solo quelli che mi servono realmente
    return a + b

somma(3,5)
somma(6,7,nome = 'Luca',età=17)

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
#in questo modo mi si incapsulano e si attiva dec1 che ha come funzione dec2 che attiva saluta, poi chiudo dec2 e poi dec1
#non vengono lette come 2 decoratori ma come uno che attiva il secondo che attiva la funzione
def saluta():
    print("Ciao!")

saluta()

import functools

def decoratore(f):
    @functools.wraps(f)  # Mantiene il nome e la docstring della funzione originale
    def wrapper(*args, **kwargs):
        print(f"Eseguendo {f.__name__}")
        return f(*args, **kwargs)
    return wrapper

@decoratore
def esempio():
    """Questa è una funzione di esempio""" #questa è una docstring ovvero una linea che spiega cosa fa la funzione, nel 99% dei casi è superflua
    print("Funzione originale")

print(esempio.__name__)  # ✅ Output corretto: esempio, che è il nome della funzione
print(esempio.__doc__)   # ✅ Output corretto: Questa è una funzione di esempio, mi stampa la docstring della funzione
esempio()    # così attivo la funzione che attiva il decoratore

import time

def tempo_di_esecuzione(f):
    def wrapper(*args, **kwargs):
        inizio = time.time()
        print("Inizio tempo")
        risultato = f(*args, **kwargs) #attivo la funzione quindi il time.sleep(2)
        fine = time.time()
        print("Fine tempo")
        print(f"Tempo di esecuzione: {fine - inizio:.4f} secondi")
        return risultato
    return wrapper

@tempo_di_esecuzione
def operazione_lenta():
    time.sleep(2)  # Simula un'operazione lunga
    print("Operazione completata!")

operazione_lenta()


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

import functools

def cache(f):
    memoria = {}

    @functools.wraps(f)
    def wrapper(*args):
        if args in memoria:
            print("Restituisco valore dalla cache")
            return memoria[args]
        risultato = f(*args)
        memoria[args] = risultato  #non ci va l'asterisco perchè altrimenti chiamerei l'intera tupla, invece sto generando un dizionare in cui l'args è la chiave e risultato è il valore associato
        print("Stampo la memoria: ",memoria[args])
        return risultato
    return wrapper

@cache
def quadrato(n):
    print(f"Calcolando quadrato di {n}")
    return n * n

print(quadrato(4))  # Calcolo
print(quadrato(2))  # Calcolo
print(quadrato(2))  # Cache


