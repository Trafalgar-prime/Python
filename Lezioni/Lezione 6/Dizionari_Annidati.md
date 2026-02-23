# Python - Dizionari Annidati in Python

I **dizionari annidati** sono dizionari che contengono altri dizionari al loro interno.  
Permettono di organizzare e gestire dati strutturati in modo efficiente.

---

## **1️⃣ Creare un Dizionario Annidato**

```python
studenti = {
    "studente1": {"nome": "Alice", "età": 22, "voto": 8},
    "studente2": {"nome": "Marco", "età": 24, "voto": 9},
    "studente3": {"nome": "Elena", "età": 21, "voto": 7}
}
```

Qui abbiamo un **dizionario `studenti`**, dove ogni chiave (`studente1`, `studente2`, etc.) contiene **un altro dizionario** con i dettagli dello studente.

---

## **2️⃣ Accedere a Tutti i Valori di una Chiave Specifica del Secondo Dizionario**

Se vogliamo **ottenere il valore di una chiave specifica** (es. `"voto"`) per **tutti gli studenti**, possiamo iterare così:

```python
for chiave_principale, sotto_dizionario in studenti.items():
    print(f"{chiave_principale}: {sotto_dizionario['voto']}")
```

🔹 **Output:**
```
studente1: 8
studente2: 9
studente3: 7
```

---

## **3️⃣ Stampare Tutte le Chiavi del Secondo Dizionario per Ogni Chiave del Primo**

```python
for chiave_principale, sotto_dizionario in studenti.items():
    print(f"
Dati di {chiave_principale}:")
    for chiave_secondaria, valore in sotto_dizionario.items():
        print(f"  {chiave_secondaria}: {valore}")
```

🔹 **Output:**
```
Dati di studente1:
  nome: Alice
  età: 22
  voto: 8

Dati di studente2:
  nome: Marco
  età: 24
  voto: 9

Dati di studente3:
  nome: Elena
  età: 21
  voto: 7
```

---

## **4️⃣ Controllare Se una Chiave Esiste nel Secondo Dizionario**

Se non siamo sicuri che una chiave esista nel secondo dizionario (es. `"media_voti"`), possiamo usare `.get()`:

```python
for chiave_principale, sotto_dizionario in studenti.items():
    voto = sotto_dizionario.get("media_voti", "Nessun dato")  # Se non esiste, restituisce "Nessun dato"
    print(f"{chiave_principale}: {voto}")
```

🔹 **Output:**
```
studente1: Nessun dato
studente2: Nessun dato
studente3: Nessun dato
```

---

## **5️⃣ Modificare un Valore in un Dizionario Annidato**

Possiamo aggiornare un valore specifico:

```python
studenti["studente1"]["voto"] = 9
print(studenti["studente1"]["voto"])  # Output: 9
```

---

## **6️⃣ Aggiungere un Nuovo Sottodizionario al Dizionario Principale**

```python
studenti["studente4"] = {"nome": "Giorgio", "età": 23, "voto": 10}
print(studenti["studente4"])
```

🔹 **Output:**
```
{'nome': 'Giorgio', 'età': 23, 'voto': 10}
```

---

## **7️⃣ Esercizi**

1️⃣ **Crea un dizionario annidato con informazioni su più prodotti (nome, prezzo, quantità). Stampa il prezzo di ogni prodotto.**  
2️⃣ **Scrivi un programma che aggiorna il voto più alto in un dizionario di studenti.**  
3️⃣ **Aggiungi un nuovo studente al dizionario esistente e stampa i dati aggiornati.**  

---

## ✅ **Obiettivo raggiunto**

✅ **Hai imparato a lavorare con dizionari annidati in Python.**  
✅ **Sai come accedere, modificare e iterare su di essi.**  
✅ **Ora prova gli esercizi per consolidare la teoria!** 🚀

