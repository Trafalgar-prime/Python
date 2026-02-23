# Python - Lezione 6: Dizionari in Python

I **dizionari** (`dict`) sono una delle strutture dati più potenti di Python.  
Consentono di **mappare una chiave a un valore**, permettendo di organizzare e recuperare i dati in modo efficiente.

## **1️⃣ Creare un Dizionario**

Un dizionario si definisce con **parentesi graffe `{}`**, usando **chiavi e valori separati da `:`**.

```python
dizionario_vuoto = {}  # Dizionario vuoto
dizionario_con_dati = {
    "nome": "Alice",
    "età": 25,
    "città": "Roma"
}
```

🔹 **Le chiavi devono essere uniche!**  
Se una chiave si ripete, Python **sovrascrive il valore precedente**.

---

## **2️⃣ Accedere ai Valori con le Chiavi**

```python
persona = {"nome": "Giulia", "età": 28, "città": "Milano"}

print(persona["nome"])  # "Giulia"
print(persona.get("età"))  # 28
```

📌 **Se la chiave non esiste, `[]` dà errore, mentre `.get()` restituisce `None`**.

```python
print(persona.get("professione"))  # None
# print(persona["professione"])  # ERRORE: KeyError
```

---

## **3️⃣ Aggiungere e Modificare Valori**

```python
persona["professione"] = "Ingegnere"  # Aggiunge un nuovo valore
persona["età"] = 29  # Modifica il valore esistente
print(persona)
```

---

## **4️⃣ Rimuovere Elementi**

📌 **`del` → Rimuove una chiave (errore se non esiste)**  
```python
del persona["città"]
```

📌 **`.pop()` → Rimuove una chiave e restituisce il valore**  
```python
professione = persona.pop("professione", "Non specificata")
```

📌 **`.popitem()` → Rimuove l'ultimo elemento aggiunto**  
```python
ultima_chiave, ultimo_valore = persona.popitem()
```

---

## **5️⃣ Scorrere un Dizionario con i Loop**

```python
persona = {"nome": "Luca", "età": 30, "città": "Torino"}

for chiave, valore in persona.items():
    print(f"{chiave}: {valore}")
```

📌 **Ottenere solo le chiavi:**  
```python
print(persona.keys())
```

📌 **Ottenere solo i valori:**  
```python
print(persona.values())
```

---

## **6️⃣ Controllare se una Chiave Esiste**

```python
if "età" in persona:
    print("La chiave 'età' esiste nel dizionario")
```

---

## **7️⃣ Copiare un Dizionario**

```python
copia_persona = persona.copy()  # Metodo sicuro per copiare il dizionario
```

---

## **8️⃣ Dizionari Annidati**

```python
studenti = {
    "studente1": {"nome": "Alice", "età": 22},
    "studente2": {"nome": "Marco", "età": 24}
}
print(studenti["studente1"]["nome"])  # "Alice"
```

---

## **9️⃣ Esercizi**

1️⃣ **Crea un dizionario con informazioni su una persona e aggiungi una chiave "professione".**  
2️⃣ **Scrivi un programma che conta quante volte appare ogni parola in una frase usando un dizionario.**  
3️⃣ **Crea un dizionario con studenti e voti. Trova lo studente con il voto più alto.**  

---

## ✅ **Obiettivo raggiunto**

✅ **Hai imparato a creare, modificare e iterare sui dizionari.**  
✅ **Sai come rimuovere elementi e lavorare con dizionari annidati.**  
✅ **Ora prova gli esercizi per consolidare la teoria!** 🚀

