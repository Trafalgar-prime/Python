# Python - Lezione 7: Le Funzioni in Python

Le **funzioni** in Python permettono di organizzare il codice in blocchi riutilizzabili.  
Scrivere codice con funzioni rende il programma **più leggibile, modulare ed efficiente**.  

---

## **1️⃣ Cos'è una Funzione e Perché Usarla?**

✅ Evita ripetizioni di codice (riutilizzabile).  
✅ Migliora la leggibilità e la manutenibilità.  
✅ Può ricevere **parametri** e restituire **risultati**.  

---

## **2️⃣ Definire e Chiamare una Funzione**

```python
def saluta():
    print("Ciao! Questa è una funzione.")

saluta()
```

🔹 **Output:**  
```
Ciao! Questa è una funzione.
```

---

## **3️⃣ Funzioni con Argomenti**

```python
def saluta(nome):
    print(f"Ciao {nome}!")

saluta("Marco")
saluta("Anna")
```

🔹 **Output:**  
```
Ciao Marco!
Ciao Anna!
```

---

## **4️⃣ Valori di Default nei Parametri**

```python
def saluta(nome="Ospite"):
    print(f"Ciao {nome}!")

saluta()  
saluta("Luca")
```

🔹 **Output:**  
```
Ciao Ospite!
Ciao Luca!
```

---

## **5️⃣ Funzioni con un Numero Variabile di Argomenti (`*args`)**

```python
def somma_numeri(*numeri):
    return sum(numeri)

print(somma_numeri(1, 2, 3, 4))  # 10
print(somma_numeri(10, 20))  # 30
```

---

## **6️⃣ Funzioni con Argomenti Chiave-Valore (`**kwargs`)**

```python
def mostra_dati(**dati):
    for chiave, valore in dati.items():
        print(f"{chiave}: {valore}")

mostra_dati(nome="Giulia", età=25, città="Roma")
```

---

## **7️⃣ Scope delle Variabili (Locale vs Globale)**

```python
x = 10  # Variabile globale

def funzione():
    x = 5  # Variabile locale
    print("Dentro la funzione:", x)

funzione()
print("Fuori dalla funzione:", x)
```

---

## **8️⃣ Funzioni Ricorsive**

```python
def fattoriale(n):
    if n == 1:
        return 1
    return n * fattoriale(n - 1)

print(fattoriale(5))  # 120
```

---

## **9️⃣ Funzioni Lambda (Anonime)**

```python
quadrato = lambda x: x ** 2
print(quadrato(4))  # 16
```

---

## **🔟 Esercizi**

1️⃣ **Scrivi una funzione che calcola la media di una lista di numeri.**  
2️⃣ **Scrivi una funzione che accetta un nome e un cognome e restituisce un saluto personalizzato.**  
3️⃣ **Crea una funzione che restituisce il valore più grande tra tre numeri.**  
4️⃣ **Crea una funzione che usa `*args` per moltiplicare un numero qualsiasi di valori.**  
5️⃣ **Scrivi una funzione ricorsiva che calcola la somma dei primi `n` numeri naturali.**  

---

## ✅ **Obiettivo raggiunto**

✅ **Hai imparato a creare e usare funzioni in Python.**  
✅ **Hai visto `args`, `kwargs`, scope, ricorsione e lambda.**  
✅ **Ora prova gli esercizi per rafforzare la teoria!** 🚀

