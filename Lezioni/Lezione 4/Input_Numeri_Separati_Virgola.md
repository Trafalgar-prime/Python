# Python - Come inserire numeri separati da virgola con `input()`

Quando l’utente deve inserire **più numeri separati da virgola**, Python legge tutto come una **stringa**.  
Per convertirli in una **lista di numeri interi o decimali**, possiamo usare `split(",")` e `map()`.  

---

## 1️⃣ Inserire numeri interi separati da virgola

Se l'utente inserisce **"10,20,30,40,50"**, possiamo convertirli in una lista di interi:

```python
numeri = list(map(int, input("Inserisci numeri separati da una virgola: ").split(",")))
print(numeri)
```

🔹 **Esempio di input:**  
```
10,20,30,40,50
```
🔹 **Output corretto:**  
```python
[10, 20, 30, 40, 50]
```

---

## 2️⃣ Inserire numeri decimali separati da virgola

Se i numeri possono contenere **decimali**, usiamo `map(float, ...)`:

```python
numeri = list(map(float, input("Inserisci numeri decimali separati da una virgola: ").split(",")))
print(numeri)
```

🔹 **Esempio di input:**  
```
10.5,20.3,30.0,40.7,50.2
```
🔹 **Output corretto:**  
```python
[10.5, 20.3, 30.0, 40.7, 50.2]
```

---

## 3️⃣ Gestire errori se l'utente inserisce dati sbagliati

Se l'utente scrive un valore non numerico (es. `"10, abc, 30"`), il programma si blocca con un errore.  
Per evitare crash, possiamo usare `try-except`:

```python
try:
    numeri = list(map(int, input("Inserisci numeri separati da una virgola: ").split(",")))
    print("Lista di numeri:", numeri)
except ValueError:
    print("Errore! Assicurati di inserire solo numeri separati da virgola.")
```

🔹 **Se l’utente inserisce dati corretti:**  
```
10,20,30
```
✅ Output:
```
Lista di numeri: [10, 20, 30]
```

🔹 **Se l’utente inserisce dati errati:**  
```
10, abc, 30
```
❌ Output:
```
Errore! Assicurati di inserire solo numeri separati da virgola.
```

---

## 💡 Riassunto

✅ **Usa `.split(",")`** per separare gli input.  
✅ **Usa `map(int, ...)` o `map(float, ...)`** per convertire in numeri.  
✅ **Gestisci gli errori con `try-except`** per evitare crash.  

📌 **Ora puoi provare a implementarlo nel tuo codice! 🚀**