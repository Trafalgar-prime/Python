def saluta():
    print("ciao! Questa è una funzione.")

saluta()


def saluta(nome):
    print(f"Ciao {nome}!")

saluta('marco')

def somma(a,b):
    return a + b

risultato = somma(5,4)
print(risultato)


def saluta(nome = 'Ospite'):
    print(f"Ciao {nome}!")

saluta()
saluta('marco')

def somma_numeri(*numeri): #uso una tupla quando non conosco il numero di argomenti
    return sum(numeri)

print(somma_numeri(1,2,3,4))
print(somma_numeri(10,20))

def mostra_dati(**dati):   #una funzione il cui argomento è un dizionario
    for chiave, valore in dati.items():
        print(f" {chiave} : {valore}")

mostra_dati(nome='Giulia',età=24,città='Roma')

x = 10  # Variabile globale

def funzione():
    x = 5  # Variabile locale
    print("Dentro la funzione:", x)

funzione()
print("Fuori dalla funzione:", x)

def cambia_x():
    global x
    x = 20

cambia_x()
print("La x global é : ",x)  # 20

def fattoriale(n):
    if n == 1:
        return 1
    elif n == 0:
        return 1
    return n*fattoriale(n-1)

print(fattoriale(0))
print(fattoriale(5))

quadrato = lambda x : x**2  #lamba è un tipo di funzione breve in cui definisco una o più variabili e poi la funzione con le variabili definite
print(quadrato(4))

numeri = [1,2,3,4,5,6]
quadrati = list(map(lambda x :x**3, numeri))
print(quadrati)

################## ESERCIZI ###############

def media(*lista):
    somma = 0    
    for i in lista:
        print(i)
        somma += i
    return somma/len(lista)

print(media(1,2,3,4,56,76,35))

def saluto(nome,cognome):
    print(f"Ciao {nome} {cognome}")

saluto('Lorenzo','Tabolacci')


def piu_alto(val1,val2,val3):
    lista = [val1,val2,val3]
    lista.sort()
    print(lista[-1])

piu_alto(2,5,4)

def prodotto(*numeri):
    prod = 1
    if len(numeri) < 2:
        print("errore, dai almeno 2 numeri")
        return 
    else:
        for i in numeri:
            prod *= i
        return prod

print(prodotto(2,3,4,5,6))
print(prodotto(2)) #il print mi da None subito dopo avermi detto che servono 2 numeri
prodotto(2)


def ricorsiva(n):
    if n == 0:
        return 0
    return n + ricorsiva(n-1)

print(ricorsiva(7))
    



