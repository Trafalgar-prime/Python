dizionario_vuoto = {}
dizionario = {
    'nome' : 'Lorenzo',
    'età' : 27,
    'città' : 'Olevano Romano'
    }

print(dizionario)

dati = {'nome' : 'Lorenzo', 'età' : 27, 'età' : 35}
print(dati)

persona = {'nome' : 'Giulia', 'età' : 28, 'città' : 'Milano'}
print(persona['nome'])
print(persona.get("età"))

print(persona.get("professione"))  # None  get non da alcun errore, e al massimo restituisce None
#print(persona["professione"])  # ERRORE: KeyError mentre senza get se non esiste la chiave ottengo un errore

persona['Professione'] = 'ingegnere'
persona['età'] = 29
print(persona)

del persona['città']
print(persona)

professione = persona.pop('Professione', "non specificata") #se in Professione metto la P minuscola, non mi legge piu il lavoro, ma solo 'non specificata'
print(professione)
print(persona)

ultima_chiave, ultimo_valore = persona.popitem() # se gli do solo una variabile, mi stampa (chiave,valore)
print(ultima_chiave,"\n" ,ultimo_valore)

persona = {'nome' : 'Luca', 'età' : 30, 'città' : 'Torino'}

for i in persona:
    print(i, " -> ", persona[i])
    print(f"{i} -> {persona[i]}") #la stessa cosa

for i, valore in persona.items(): #con items vado a prendere solo il valore, ma se gli do 2 variabili divise dalla virgola, allora legge anche la chiave
    print(f"{i}: {valore}")

for valore in persona.items():#in questo caso legge entrambi cosi : (chiave,valore)
    print(f"{valore}")

print(persona.keys())  # dict_keys(['nome', 'età', 'città'])
print(persona.values()) # dict_values(['Luca', 30, 'Torino'])

if 'età' in persona:
    print("c'è l'età in persona")

copia_persona = persona.copy()
print(persona)
print("\n")
print(copia_persona)
copia_alternativa = dict(persona)
print(copia_alternativa)
print("\n")
print("\n")


copia_persona['età'] = 40
print(copia_alternativa)
print(persona)
print(copia_persona)



studenti = {
    "studente1": {"nome": "Alice", "età": 22},
    "studente2": {"nome": "Marco", "età": 24}
}

print(studenti["studente1"]["nome"])  # "Alice"

############## ESERCIZI #################

persona = {
    'nome' : 'Syria',
    'età' : 20,
    'città' : 'Roma'
    }

persona['professione'] = 'studente'
print(persona)

parole = {
    'parola_1' : 'gioco',
    'parola_2' : 'compiti',
    'parola_3' : 'bambini'
    }

frase = "Il gioco dei bambini è sempre più divertente quando il gioco si alterna ai compiti , ma i bambini preferirebbero giocare piuttosto che fare i compiti , perché il gioco rende i bambini felici , mentre i compiti sembrano sempre un ostacolo al gioco ."
lista_parole = frase.split()

for i in lista_parole:
    #print(i)
    for chiave, valore in parole.items():
        #print(valore)
        #print("\n")
        if valore == i:
            print("parola trovata")


parole = {
    'parola_1': 'gioco',
    'parola_2': 'compiti',
    'parola_3': 'bambini',
    'parola_4': 'il'
}

# Dizionario per contare le occorrenze
conteggio = {chiave: 0 for chiave in parole}  # genero un nuovo dizionario con una sola variabile che si ripete su tutte le variabili di parole, quindi 1*(varibili di parola)
print(conteggio)

frase = """Il gioco dei bambini è sempre più divertente quando il gioco si alterna ai compiti , 
ma i bambini preferirebbero giocare piuttosto che fare i compiti , perché il gioco rende i bambini felici , 
mentre i compiti sembrano sempre un ostacolo al gioco ."""

lista_parole = frase.lower().replace(",", "").replace(".", "").split()  # Rimuove punteggiatura e converte in minuscolo

# Scansioniamo le parole della frase
for parola in lista_parole:
    for chiave, valore in parole.items():
        if valore == parola:
            conteggio[chiave] += 1  # Incrementiamo il conteggio

print("Conteggio delle parole:", conteggio)
print("\n")

import random
alunni = {
    'studente 1' : {'nome' : 'lorenzo', 'età' : 22, 'voto' : random.randint(60,110)},
    'studente 2' : {'nome' : 'syria', 'età' : 26, 'voto' : random.randint(60,110)},
    'studente 3' : {'nome' : 'carlotta', 'età' : 23, 'voto' : random.randint(60,110)},
    'studente 4' : {'nome' : 'giovanni', 'età' : 21, 'voto' : random.randint(60,110)}
    }

print(alunni)
lista_voti = []

for chiave_principale, sotto_dizionario in alunni.items():
    print(f"{sotto_dizionario['nome']} : {sotto_dizionario['voto']}")
    lista_voti.append(sotto_dizionario['voto'])
print("\n")
print(lista_voti)

lista_voti.sort()
print(lista_voti)
lista_voti.reverse()
print(lista_voti)
    

        
