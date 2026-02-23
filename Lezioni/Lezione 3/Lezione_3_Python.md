# Python - Lezione 3: Strutture Iterative (Loop: for e while)

## 1) Perché usare i loop?
I **loop** servono per **ripetere** un'operazione più volte senza dover scrivere il codice manualmente ogni volta.

Python ha due tipi principali di loop:
- **`for` loop** → usato quando sappiamo quante volte ripetere il codice.
- **`while` loop** → usato quando vogliamo ripetere un'azione finché una condizione è vera.

---

## 2) Il loop `for`
### 2.1 Uso di `for` con `range()`
Il ciclo `for` è utile per **iterare** su una sequenza di valori (numeri, liste, stringhe, ecc.).

```python
for i in range(5):  # range(5) genera i numeri 0,1,2,3,4
    print("Iterazione numero:", i)
```
📌 **Cosa succede qui?**
- `range(5)` genera i numeri **0, 1, 2, 3, 4**.
- Il ciclo stampa il valore di `i` ad ogni iterazione.

---

### 2.2 Il `for` con intervallo personalizzato
Possiamo personalizzare **inizio, fine e passo** in `range()`:

```python
for i in range(2, 10, 2):  # Parte da 2, arriva a 10 (escluso), con passi di 2
    print(i)
```
🔹 Stampa: `2, 4, 6, 8`

---

### 2.3 Iterare su una lista
Possiamo iterare direttamente sugli elementi di una lista:

```python
frutti = ["mela", "banana", "ciliegia"]
for frutto in frutti:
    print("Frutto:", frutto)
```
🔹 Stampa: `mela, banana, ciliegia`

---

### 2.4 Iterare su una stringa (lettera per lettera)

```python
parola = "Python"
for lettera in parola:
    print(lettera)
```
🔹 Stampa ogni lettera su una riga separata.

---

## 3) Il loop `while`
Il `while` esegue il blocco **finché una condizione è vera**.

```python
x = 0
while x < 5:  # Continua finché x è minore di 5
    print("Valore di x:", x)
    x += 1  # Incremento di x
```
🔹 Stampa `0, 1, 2, 3, 4` e poi si ferma.

---

### 3.1 Loop `while` con input utente
Possiamo chiedere dati all’utente fino a quando inserisce un valore valido:

```python
password = ""
while password != "segreta":
    password = input("Inserisci la password: ")
print("Accesso consentito!")
```
🔹 Il ciclo **continua a chiedere la password** finché non viene digitato `"segreta"`.

---

## 4) Controllo dei loop: `break` e `continue`

### 4.1 Uscire dal ciclo con `break`
`break` interrompe immediatamente il ciclo:

```python
for numero in range(10):
    if numero == 5:
        print("Interruzione del ciclo!")
        break  # Esce dal ciclo
    print(numero)
```
🔹 Stampa i numeri fino a `4`, poi si ferma.

---

### 4.2 Saltare un'iterazione con `continue`
`continue` **salta** un'iterazione e passa alla successiva:

```python
for numero in range(5):
    if numero == 2:
        continue  # Salta il numero 2
    print(numero)
```
🔹 Stampa `0, 1, 3, 4` (salta il `2`).

---

## 5) Il ciclo `for` con `else`
Possiamo usare `else` con `for` o `while`, che viene eseguito solo se il ciclo **non** viene interrotto da un `break`:

```python
for i in range(3):
    print(i)
else:
    print("Il ciclo è terminato senza interruzioni.")
```
🔹 Se `break` non viene usato, **else viene eseguito**.

---

## 6) ✍ Esercizi da svolgere
1️⃣ **Somma di numeri da 1 a N**  
   - Chiedi all’utente un numero `N`.
   - Usa un `for` per calcolare la somma da `1` a `N`.
   - Stampa il risultato.

2️⃣ **Tabellina del 5 fino a 50**  
   - Usa un `for` per stampare i multipli di 5 da `5` a `50`.

3️⃣ **Indovina il numero!**  
   - Genera un numero casuale tra 1 e 10.
   - Chiedi all'utente di indovinare finché non inserisce il numero giusto.
   - Se il numero è corretto, stampa `"Bravo! Hai indovinato!"`.

---

## Obiettivo raggiunto
✅ Ora sai usare:
- **Il ciclo `for`** per iterare su sequenze e `range()`.
- **Il ciclo `while`** per ripetere istruzioni finché una condizione è vera.
- **`break` e `continue`** per controllare il flusso del ciclo.
- **`for` con `else`** per eseguire codice extra solo se il ciclo termina normalmente.

📌 **Ora prova gli esercizi e fammi sapere se hai domande!** 🚀
