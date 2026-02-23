# 📦 Il file `__init__.py` in Python

Questo documento spiega **a cosa serve il file `__init__.py`**, quando deve essere vuoto e quando invece è utile scriverci del codice.
È pensato per studenti universitari e per chi vuole lavorare con **progetti Python strutturati**.

---

## 📁 Struttura tipica di un package Python

Un package Python è una cartella che contiene almeno un file `__init__.py`.

Esempio:

```
progetto/
│
├── main.py
│
└── mio_pacchetto/
    ├── __init__.py
    └── funzioni.py
```

La presenza di `__init__.py` dice a Python che `mio_pacchetto` è un **package importabile**.

---

## ✅ Caso 1 — `__init__.py` vuoto (caso base)

Il file può essere **completamente vuoto**:

```python
# __init__.py
```

✔️ Python riconosce la cartella come package  
✔️ Gli import funzionano correttamente  
✔️ È il caso più comune all’inizio  

Esempio di import valido:

```python
from mio_pacchetto import funzioni
```

Uso:

```python
funzioni.nome_funzione()
```

---

## ✅ Caso 2 — Esporre funzioni direttamente dal package

Se vuoi importare una funzione **direttamente dal package**, ad esempio:

```python
from mio_pacchetto import somma
```

devi **dichiararlo esplicitamente** in `__init__.py`.

### `mio_pacchetto/__init__.py`
```python
from .funzioni import somma, media
```

Ora puoi usare:

```python
somma(3, 4)
media([1, 2, 3])
```

🎯 Questo migliora l’**interfaccia pubblica** del package  
📌 È una **best practice** nei progetti professionali

---

## ✅ Caso 3 — Organizzazione interna del package

Puoi anche usare `__init__.py` per rendere esplicita la struttura:

```python
from . import funzioni
```

Uso:

```python
mio_pacchetto.funzioni.somma()
```

---

## ❌ Cosa NON fare in `__init__.py`

❌ Non usare `__init__.py` come fosse un `main.py`  
❌ Non scrivere codice che ha **effetti collaterali**

Esempi da evitare:

```python
print("ciao")        # ❌
calcolo_costoso()    # ❌
scrivi_file()        # ❌
```

📌 `__init__.py` viene eseguito **ogni volta che il package viene importato**

---

## 🧠 Regola fondamentale

> `__init__.py` serve a **definire cosa è pubblico del package**,  
> non a contenere logica applicativa.

---

## 🎯 Consiglio pratico

- All’inizio: **lascialo vuoto**
- Nei progetti più grandi:
  - usalo per esporre funzioni e moduli
  - controllare l’API del package
  - rendere il codice più pulito e leggibile

Questo è esattamente l’approccio usato nei **veri progetti Python e Machine Learning**.

---

## ✅ In sintesi

| Situazione | Scrivere codice in `__init__.py` |
|----------|----------------------------------|
| Package semplice | ❌ No |
| Import diretti comodi | ✅ Sì |
| Progetto strutturato | ✅ Sì |
| Logica applicativa | ❌ Mai |

---
