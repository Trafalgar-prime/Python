try:
    x = 10/0
except ZeroDivisionError: #ovviamente devo gia conoscere l'errore
    print("Non puoi dividere per zero")


try:
    numero = int(input("Inserisci un numero: "))
    risultato = 10 / numero
except ZeroDivisionError:
    print("Non si può dividere per zero!")
except ValueError:
    print("Dammi un valore valido")
else:
    print("Stampa se non ci sono errori: ",risultato)        


try:
    x = int("abc")
except Exception as e:
    print("Errore generico:", e)

try:
    f = open("file.txt","r")
    contenuto = f.read()
except FileNotFoundError:
    print("il file non esiste")
finally: #questo viene stampato sempre
    print("Operazione completata.")
print("\n")

def verifica_età(età):  #ho definito la funzione per la verifica
    if età < 18:
        raise ValueError("Errore: devi avere almeno 18 anni")  #la parentesi tonda è la variabile e che poi richiamo subito dopo
    return "accesso consentito"


try:
    print(verifica_età(16))
except ValueError as e: #value error deve essere scritto per forza perche lo chiamo nella funzione
    print(e)

class ErrorePersonalizzato(Exception): #in questo modo introduco errore età negli errore di exception, altrimenti è solo una variabile qualunque
    pass

def controlla_numero(n):
    if n < 0:
        raise ErrorePersonalizzato("Errore: Il numero non può essere negativo!")
print("\n")

try:
    controlla_numero(-5)
except ErrorePersonalizzato as e:
    print(e)

################### ESERCIZI #########################


try:
    numero = int(input("Inserisci un valore: "))
    if numero < 0:
        raise ErrorePersonalizzato("Non puoi dare valori negativi")
    rapporto = 10 / numero
except ValueError:
    print("Il numero deve essere intero")
except ZeroDivisionError:
    print("Il numero non può essere zero")
except ErrorePersonalizzato as e:
    print(e)

class Errore_Età(Exception):
    pass

def controllo_età(età):
    if età < 18:
        raise Errore_Età("Sei minorenne")
try:
    controllo_età(16)
except Errore_Età as a:
    print(a)


                 
