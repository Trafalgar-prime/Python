a = 10
b = 3

print(a + b)  # Somma: 13
print(a - b)  # Sottrazione: 7
print(a * b)  # Moltiplicazione: 30
print(a / b)  # Divisione: 3.333...
print(a // b) # Divisione intera: 3
print(a % b)  # Modulo (resto): 1
print(a ** b) # Potenza: 10³ = 1000

x = 5
y = 10

print(x == y)  # False (x è uguale a y?)
print(x != y)  # True (x è diverso da y?)
print(x > y)   # False (x è maggiore di y?)
print(x < y)   # True (x è minore di y?)
print(x >= y)  # False (x è maggiore o uguale a y?)
print(x <= y)  # True (x è minore o uguale a y?)

a = True
b = False

print(a and b)  # False (Vero AND Falso → Falso) con and entrambi vero allora vero, altrimenti sempre falso
print(a or b)   # True (Vero OR Falso → Vero) con or enrtrambi falsi allora falso, altrimenti sempre vero
print(not a)    # False (NOT Vero → Falso), in questo modo inverto la sentenza

# Controllare se un numero è divisibile per 2 e per 3
num = int(input("Inserisci un numero: "))

if (num % 2 == 0) and (num % 3 == 0):
    print("Il numero è divisibile per 2 e 3.")
else:
    print("Il numero NON è divisibile per 2 e 3 contemporaneamente.")

if (num % 2 == 0) or (num % 3 == 0):
    print("Il numero è divisibile o per 2 o per 3 o entrambi i casi.")
else:
    print("Il numero NON è divisibile per 2 e 3 contemporaneamente.")

num_test = int (input("Inserisci un altro valore: "))

if (num_test % 2 == 0):
    print("Il numero è pari.")
else: 
    print("Il numero è dispari.")


voto = int(input("Con quanto sei uscito? "))

if voto >= 80:
    print("Hai preso una B!")
elif  voto >= 70:
    print("Hai preso una C!")
elif voto >= 90:
    print("Hai preso una A!")
else:
    print("Devi migliorare") #se non sono in ordine decrescente non lo leggerà mai nel modo giusto, infatti in questo modo con piu di 90 ti darà una B


numero_1 = int(input("Dammi un valore: "))
numero_2 = int(input("Dammi un altro valore: "))

if numero_1 > numero_2:
    print(numero_1, " è maggiore del ",numero_2)
elif numero_2 > numero_1:
    print(numero_2, " è maggiore del ",numero_1)
elif numero_1 == numero_2:
    print(numero_1, " è uguale al ",numero_2)

età = int(input("Dimmi la tua età: "))

if età >= 20: #scritto invertito dirà sempre che sei un bambino
    print("Sei un adulto")
elif età >= 13:
    print("Sei un adolescente")
elif età >= 0:
    print("Sei un bambino")


