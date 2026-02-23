# Python - Uso della Funzione `map()`

La funzione **`map()`** viene usata per **applicare una funzione a ogni elemento di un iterabile** (come una lista o una tupla) e restituire un nuovo iterabile con i risultati.

---

## **1️⃣ Sintassi di `map()`**
```python
map(funzione, iterabile)
```
- **`funzione`** → Funzione che viene applicata a ogni elemento dell'iterabile.
- **`iterabile`** → Può essere una lista, una tupla, un insieme, ecc.

📌 **Nota:** `map()` restituisce un oggetto di tipo `map`, che deve essere convertito in una lista o un'altra struttura dati per visualizzare i risultati.

---

## **2️⃣ Esempio Base: Convertire una Lista di Stringhe in Numeri**
```python
numeri_stringa = ["1", "2", "3", "4"]
numeri_interi = list(map(int, numeri_stringa))

print(numeri_interi)  # [1, 2, 3, 4]
```
✅ **Qui `map(int, ...)` converte ogni elemento della lista da stringa a intero.**

---

## **3️⃣ Usare `map()` con una Funzione Definita**
```python
def quadrato(x):
    return x ** 2

numeri = [1, 2, 3, 4, 5]
risultato = list(map(quadrato, numeri))

print(risultato)  # [1, 4, 9, 16, 25]
```
✅ **Qui `map()` applica la funzione `quadrato()` a ogni elemento della lista.**

---

## **4️⃣ Usare `map()` con una Funzione Lambda**
Possiamo usare **una funzione anonima (`lambda`)** direttamente dentro `map()`.

```python
numeri = [1, 2, 3, 4, 5]
risultato = list(map(lambda x: x ** 2, numeri))

print(risultato)  # [1, 4, 9, 16, 25]
```
✅ **Stesso risultato, ma senza definire una funzione separata.**

---

## **5️⃣ Usare `map()` con Più Liste**
Se passiamo più iterabili a `map()`, la funzione deve accettare lo stesso numero di parametri.

```python
numeri1 = [1, 2, 3]
numeri2 = [4, 5, 6]

somma = list(map(lambda x, y: x + y, numeri1, numeri2))
print(somma)  # [5, 7, 9]
```
✅ **Qui `map()` somma gli elementi corrispondenti delle due liste.**

---

## **6️⃣ Esempio Avanzato: Capitalizzare Nomi**
```python
nomi = ["mario", "luigi", "peach"]
nomi_maiuscoli = list(map(str.capitalize, nomi))

print(nomi_maiuscoli)  # ['Mario', 'Luigi', 'Peach']
```
✅ **Qui `map()` usa `str.capitalize` per rendere maiuscola la prima lettera di ogni nome.**

---

## **7️⃣ Differenza tra `map()` e `for`**
### 🔹 **Con `map()`**
```python
numeri = [1, 2, 3, 4]
quadrati = list(map(lambda x: x ** 2, numeri))
print(quadrati)  # [1, 4, 9, 16]
```
### 🔹 **Con `for`**
```python
numeri = [1, 2, 3, 4]
quadrati = []
for num in numeri:
    quadrati.append(num ** 2)

print(quadrati)  # [1, 4, 9, 16]
```
✅ **`map()` è più conciso, ma il `for` è più leggibile per chi non conosce `map()`.**

---

## **📌 Riassunto**
| Funzione | Descrizione |
|----------|------------|
| `map(f, lista)` | Applica `f()` a ogni elemento di `lista`. |
| `map(f, lista1, lista2)` | Applica `f()` agli elementi di `lista1` e `lista2` in parallelo. |
| `map(int, lista)` | Converte ogni elemento in intero. |
| `map(str.capitalize, lista)` | Converte ogni stringa in formato capitalizzato. |

✅ **Ora prova questi esempi e implementa `map()` nei tuoi progetti!** 🚀

