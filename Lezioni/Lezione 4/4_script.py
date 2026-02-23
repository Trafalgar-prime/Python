numeri = [1,2,3,4,5]
frutti = ['mela','banana','ciliegia']
mista = [1,'mela',3.14,True]

print(frutti[0])
print(frutti[1])
print(frutti[2])

print(mista[3])
print(frutti[-1])
print(frutti[-2])
print(frutti[-3])

numeri[2] = 10
print(numeri)

frutti.append('arancia')
frutti.insert(2,'susina')
print(frutti)

#frutti.remove('banana')
print(frutti)
#frutti.pop(1) #se non inserisco un numero elimina l'ultimo valore
print(frutti)

#del frutti[1] #Cancella un elemento o l’intera lista
print(frutti)

#del frutti #cancello tutto l'array
print(frutti)

for frutto in frutti:
    print(frutto)

for i, frutto in enumerate(frutti):
    print(f"indice {i}: {frutto}")
print("\n")
for i, frutto in enumerate(frutti, start=2):  # Inizia da 1 invece che da 0
    print(f"Indice {i}: {frutto}")
print("\n")

if 'banana' in frutti:
    print("banana è in frutti")

numeri = [2,8,7,3,5]
numeri.sort() #ordina in modo crescente
print(numeri)
numeri.reverse()#ordina in modo decrescente
print(numeri)

#frutti_copia = frutti.copy()
frutti_copia = frutti[:]
frutti_copia.append('pera')
print(frutti)
print(frutti_copia)




matrice = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ]

print(matrice)
print(matrice[1][1]) #ricordati che partono da zero, il centro non è 2,2 ma 1,1 perche si parte da 0,0


######ESERCIZI########
lista = []
for i in range(5):
    numeri = int(input("Dammi un valore:"))
    lista.append(numeri)

print(lista)


#numeri = list(map(int, input("Inserisci 5 numeri separati da una virgola: ").split(",")))
#print(numeri)

numeri = []
print("\n")

while True:
    valore = input("Inserisci un numero (o scrivi 'stop' per terminare): ")
    if valore == 'stop':
        print("Hai digitato stop")
        break
    try:
        numero = int(valore)
        numeri.append(numero)
    except ValueError:
        print("Per favore, inserisci un numero valido o 'stop' per terminare.")

print(numeri)

numeri = [34, 65, 87, 23, 59, 10]
somma = 0
for i in numeri:
    somma += i
    print(i)
media = somma/(len(numeri))
print(f"la somma è : {somma} e la media è : {media}")

somma = 0 #inizializzo le variabili
media = 0

for i in range(len(numeri)):
    somma += numeri[i]
    print(numeri[i])
    print("ciao")
media = somma/(len(numeri))
print(f"la somma è : {somma} e la media è : {media}")

lista = input("dammi 5 parole (divise dalla virgola): ").split(",")
print(lista)
lista.sort()
print(lista)
