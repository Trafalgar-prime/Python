import random

for i in range(5):
    print("Stampa tutti i vari cicli del for: ",i)
print("\n")
for i in range (2,13,2):
    print("Stampa tutti i vari cicli del for: ",i)

frutti = ['mela', 'banana', 'pera']

for frutto in frutti: #frutto funzone come l'oggetto i-esimo
    print("STAMPA TUTTI I FRUTTI:", frutto)

parola = 'python' #posso usare sia le virgolette doppie che singole per le parole

for j in parola:
    print("stampa ogni lettera: ",j)

x = 0

while x < 5:
    print("stampo il ciclo:",x)
    x +=2

print("\n")
password = ''


while password != 'segreta':
    #break
    password = input("Inserisci la password: ")
    print("l'unica password accettata è 'segreta'")
    if (password == 'segreta'):
        print("Accesso consentito")


print("\n")
print("Ho effettuato l'accesso.\n")


for numero in range(10):
    if numero == 5:
        print("è uscito dal ciclo")
        break
    print(numero)

print("\n")

for numero in range(10):
    if numero == 2:
        continue
    print(numero)

print("\n")

for i in range(8):
    print(i)
    if (i == 4):
        continue # con break non vado oltre, mentre con continue arrivo alla stampa dell'else
else:   #si può usare anche nel ciclo while, purche non ci sia break
    print("il ciclo non è stato interrotto")

print("\n")

N = 120
somma = 0 #devo inizializzarla per forza
for i in range(N):
    somma += i
    print(i, ": ", somma)
print("\n")

for i in range(11):
    print(i*5)

print("\n")

tabellina = 0
while tabellina < 51:
    print(tabellina)
    tabellina += 5

numero = 0
print("\n")

while numero != 3:
    numero = random.randint(1,10)
    print(numero)
print("Finito")
    


    
    
    
