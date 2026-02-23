tupla_vuota = ()
tupla_un_singolo_elemento = (5,) #senza la virogla non è una tupla ma solo un valore assegnato ad una variabile
tupla_di_numeri = (1,2,3,4,5)
tupla_mista = ('ciao', 3.14, True, 10)

x = (5) #non è una tupla
y = (5,) #è una tupla
print(type(x))
print(type(y))

numeri = (10,20,30,40,50)
print(numeri[1])
print(numeri[-2])
print(numeri[1:4])

#numeri[1] = 60 #cosi mi da errore perche le tuple non possono essere modificate

numeri_lista = list(numeri)
print(numeri_lista)
numeri_lista[1] = 60
numeri = tuple(numeri_lista)
print(numeri)

colori = ('blu','verde','giallo','rosso')

for colore in colori:
    print(colore)

print('nero' in colori) #false
print('giallo' in colori) #true

coordinate = { #uso le tuple nei dizionari
    (41.9028, 12.4964): "Roma",
    (48.8566, 2.3522): "Parigi"
}
print(coordinate[(41.9028, 12.4964)])  # "Roma"

############ SET ##########

set_vuoto = ()
set_numeri = {1,2,3,4,5}
set_misto = {"ciao", 3.14, True}

numeri = {1,2,2,3,3,3,4,4,4,4,5,5} 
print(numeri) # leggerà solo valori unici, e non i duplicati

numeri.add(6)
print(numeri)

#numeri.remove(2) #rimuove il valore 2 se c'è
print(numeri)

numeri.discard(10) #rimuove il numero 10 se c'è

print(numeri)
valore = numeri.pop() #elimina sempre il primo valore, per eliminare un valore a scelta dobbiamo convertirlo in una lista
print(valore)
print(numeri)

numeri = {10, 20, 30, 40, 50}
print("Set originale:", numeri)

for _ in range(5):  # Chiamiamo pop() 5 volte
    elemento_rimosso = numeri.pop() #praticamente rimuove un valore completamenrte casuale nel set, altrimenti di solito rimuove un valore per noi casuale ma è definito tramite un'idea di python per cui non sarà mai casuale
    print(f"Elemento rimosso: {elemento_rimosso}")
    print("Set aggiornato:", numeri)

print("\n")
print("\n")
print("\n")

############# ESERCIZI ########
lista = [1,1,2,2,3,3,4,4,4,4,4,5,5,5,5]
lista_unica = set(lista)
print(lista_unica)

colori = {'blu','verde','giallo','rosso'}

print('blu' in colori)


lista_2 = {1,1,1,4,4,4,7,7,7,8,8,8,8,9,9}
print(lista_unica.intersection(lista_2))

import random

numeri = {10, 20, 30, 40, 50}

# Scegliamo un elemento casuale
elemento_rimosso = random.choice(list(numeri))
numeri.remove(elemento_rimosso)

print("Elemento rimosso:", elemento_rimosso)
print("Set aggiornato:", numeri)
