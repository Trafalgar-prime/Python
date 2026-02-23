# Python - Lezione 2: Operatori e Strutture di Controllo

## 1) Operatori in Python

Gli **operatori** sono simboli (o parole chiave) che permettono di eseguire determinate operazioni sui dati (variabili o valori). In Python ne esistono diversi tipi:

### 1.1 Operatori Aritmetici

Questi operatori permettono di eseguire operazioni matematiche di base:

- `+` (addizione)  
- `-` (sottrazione)  
- `*` (moltiplicazione)  
- `/` (divisione float, con decimali)  
- `//` (divisione intera, tronca la parte decimale)  
- `%` (modulo, resto della divisione)  
- `**` (potenza)

Esempio:
```python
a = 10
b = 3

print(a + b)   # 13 (somma)
print(a - b)   # 7  (sottrazione)
print(a * b)   # 30 (moltiplicazione)
print(a / b)   # 3.333... (divisione float)
print(a // b)  # 3 (divisione intera)
print(a % b)   # 1 (resto)
print(a ** b)  # 10^3 = 1000
```

### 1.2 Operatori di Confronto (Relazionali)

Servono a confrontare due valori/variabili, restituendo `True` o `False`.  
- `==` (uguaglianza)  
- `!=` (diversità)  
- `>`  (maggiore)  
- `<`  (minore)  
- `>=` (maggiore o uguale)  
- `<=` (minore o uguale)

Esempio:
```python
x = 5
y = 10

print(x == y)  # False
print(x != y)  # True
print(x > y)   # False
print(x < y)   # True
print(x >= y)  # False
print(x <= y)  # True
```

### 1.3 Operatori Logici

Usati per unire o invertire condizioni booleane:

- `and` restituisce `True` se entrambe le condizioni sono vere  
- `or`  restituisce `True` se almeno una condizione è vera  
- `not` inverte `True` in `False` e viceversa

Esempio:
```python
a = True
b = False

print(a and b)  # False (Vero AND Falso)
print(a or b)   # True  (Vero OR Falso)
print(not a)    # False (NOT Vero)
```

---

## 2) Strutture di Controllo: if, elif, else

Le strutture di controllo permettono di eseguire blocchi di codice solo se certe condizioni sono vere. In Python, l’**indentazione** (gli spazi a inizio riga) definisce il blocco di istruzioni.

### 2.1 if-else

```python
if condizione:
    # codice eseguito se condizione è vera
else:
    # codice eseguito se condizione è falsa
```

Esempio:
```python
eta = int(input("Quanti anni hai? "))

if eta >= 18:
    print("Puoi guidare!")
else:
    print("Non puoi guidare ancora.")
```

### 2.2 if-elif-else

Se ci sono più possibili condizioni:

```python
if prima_condizione:
    # blocco se prima_condizione è vera
elif seconda_condizione:
    # blocco se seconda_condizione è vera
else:
    # blocco se nessuna condizione è vera
```

Esempio:
```python
voto = int(input("Inserisci il tuo voto (0-100): "))

if voto >= 90:
    print("Hai preso un A!")
elif voto >= 80:
    print("Hai preso un B!")
elif voto >= 70:
    print("Hai preso un C!")
else:
    print("Devi migliorare!")
```

---

## 3) Esempi pratici

### Esempio: Numero positivo, negativo o zero
```python
num = int(input("Inserisci un numero intero: "))

if num > 0:
    print("Il numero è positivo!")
elif num < 0:
    print("Il numero è negativo!")
else:
    print("Il numero è zero!")
```

### Esempio: Divisibilità
```python
num = int(input("Inserisci un numero: "))

if (num % 2 == 0) and (num % 3 == 0):
    print("Il numero è divisibile per 2 e 3.")
else:
    print("Il numero NON è divisibile per 2 e 3 insieme.")
```

---

## 4) Esercizi

1. **Numero maggiore**  
   - Chiedi all’utente due numeri.  
   - Stampa quale è maggiore (oppure se sono uguali).  

2. **Pari o dispari**  
   - Chiedi un numero.  
   - Se `num % 2 == 0`, stampa "Pari", altrimenti stampa "Dispari".  

3. **Fasce di età**  
   - Chiedi l’età all’utente.  
   - 0-12: “Bambino”  
   - 13-19: “Adolescente”  
   - 20+: “Adulto”  

---

## Obiettivo raggiunto

Ora sai usare:

- Gli operatori aritmetici, di confronto e logici in Python  
- Le strutture di controllo `if`, `elif`, `else` per gestire il flusso del programma  

Buono studio! 🚀
