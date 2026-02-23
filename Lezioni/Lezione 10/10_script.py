file = open("output.txt", "w") #creo un file output.txt vuoto e ci scrivo sopra, se esiste già e c'è gia scritto qualcosa viene cancellato
file.write("Questa è una nuova riga di testo\n") 
file.close()

file = open("output.txt","a")
file.write("Questa è un'altra riga\n")
file.close()


file = open("output.txt","r")
contenuto = file.read() #legge gia una volta file.read() quindi se lo richiamo anche nel print non mi leggerà nulla se non stampando la variabile contenuto
print(contenuto)
print("Questo è il primo contenuto\n")
file.seek(0) #serve per riportare il cursore all'inizio
print(file.read())
print("Questo è il file.read()\n")
file.seek(0)
file.close()

#print(file.read()) non stampa nulla perche il file è chiuso
print(contenuto) #stampa anche se il file è chiuso perche l'ho copiato su una variabile
print("Questo è il secondo contenuto\n")

file = open("output.txt","r")
print(file.readline()) #mi stampa solo la prima riga
print(file.readline()) #cosi mi stampa anche la seconda
file.close()

file = open("output.txt","r")
righe = file.readlines() #creo una lista con tutte le righe
print(righe)
file.seek(0)
print("\n")
print(file.readlines()) #stampo la lista con tutte le righe del file
file.close()

print(righe, "\n Inizio del with open \n")




#per evitare di scrivere ogni volta file.close() possiamo usare un altro medoto
with open("output.txt","r+") as file: # con r+ possiamo leggere e scrivere contemporaneamente, ma se non do file.read() allora sovrascrive il contenuto gia esistente
    contenuto = file.read() #praticamente con file.read() do importanza al testo già presente nel file
    file.write("Nuova riga aggiunta\n") #in questo modo non cancello il contenuto gia esistente
    print("Primo contenuto in questa sezione\n")
    print(contenuto) #legge nel contenuto solo le prime 2 righe
    file.seek(0)
    print("Primo file.read() in questa sezione\n")
    print(file.read())
    

print("Fine del with open")


##################### CSV ##########################
print("\nPartiamo con i csv\n")

import csv #per lavorare con i csv devo importare la libreria

with open("output.csv", "w", newline="") as file:  #newline="" evita righe nuove nei file csv su windows
    scrittore = csv.writer(file)
    scrittore.writerow(["Nome", "Età", "Città"])
    scrittore.writerow(["Anna", 25, "Roma"])
    scrittore.writerow(["Luca", 30, "Milano"])
    scrittore.writerow(["posso scrivere quello che voglio"]) #senza le parentesi quadre scrive ogni lettera staccata

with open("output.csv", "r") as file:
    lettore = csv.reader(file) #creiamo un lettore csv
    for riga in lettore:
        print(riga)

print("\nAndiamo con i JSON\n")
############################### JSON ##########################

import json

dati = {"nome": "Luca", "età": 30, "città": "Roma"}  #è una classe
#print(dati)

with open("dati.json", "w") as file:
    json.dump(dati, file)  # Scrive i dati nel file JSON

with open("dati.json", "r") as file:
    dato = json.load(file)  # Carica i dati dal file JSON su una variabile dato
    

print(dato)  # {'nome': 'Luca', 'età': 30, 'città': 'Roma'}


################### ELIMINARE ###########

import os

if os.path.exists("dati.json"): # Controlliamo se il file esiste
    print("esiste e lo elimino\n")
    os.remove("dati.json")
else:
    print("il file non esiste\n")


############### ESERCIZI ##################

file = open("output.txt", "a")
file.write("Questa è l'utlima riga che inserisco\n")
file.close()
file = open("output.txt","r")
print(file.read())
file.seek(0)
print(len(file.readlines()),"\n")
file.close()


es_2 = open("es_2.txt","w")
es_2.write(input("Inserisci una frase: "))
es_2.close()
print("\n")
es_2 = open("es_2.txt","r")
print(es_2.read())
es_2.close()


import csv

with open("es_3.csv", "w",newline = "") as es_3:
    scrittore = csv.writer(es_3)
    scrittore.writerow(["Nome", "Età", "Città"])
    scrittore.writerow(["Anna", 25, "Roma"])
    scrittore.writerow(["Luca", 30, "Milano"])
    scrittore.writerow(["posso scrivere quello che voglio"])

with open("es_3.csv", "r") as es_3:
    lettore = csv.reader(es_3) #creiamo un lettore csv
    for riga in lettore:
        print(riga)
        print(riga[0])
