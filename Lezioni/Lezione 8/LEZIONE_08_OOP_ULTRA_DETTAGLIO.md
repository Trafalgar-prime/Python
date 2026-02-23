# Lezione 8 — Programmazione ad Oggetti (OOP) in Python  
## Versione GitHub **ULTRA‑DETTAGLIATA** (riga‑per‑riga, parola‑per‑parola, simbolo‑per‑simbolo)

> **Regola di questo documento**: ogni volta che vedi un blocco di codice, sotto trovi:
> 1) spiegazione **riga per riga**  
> 2) spiegazione **token per token** (parole, simboli, virgole, parentesi, punti, due punti, ecc.)  
> 3) perché quelle scelte sono sensate (quando è “best practice” e quando no)  
>
> L’obiettivo è che tu possa guardare un file OOP e **capire ogni singola cosa** senza buchi.

---

## Indice super‑chiaro
- 0. Prima di partire: cosa significa “classe” in Python (davvero)
- 1. OOP: concetto, utilità, e perché ti serve per ML/DL
- 2. Sintassi dei blocchi: indentazione, `:` e perché sono fondamentali
- 3. Classe, oggetto, istanza: definizioni operative (non teoriche)
- 4. Il primo esempio: `Persona` (spiegato fino alle virgole)
- 5. `self`: binding dei metodi e “magia” spiegata
- 6. Attributi: lookup order, istanza vs classe, shadowing
- 7. Metodi di istanza: parametri, default, `*args`, `**kwargs` dentro OOP
- 8. `@classmethod`: factory methods, alternative constructors, e uso reale
- 9. `@staticmethod`: utility, validazioni, e cosa NON fare
- 10. `@property`: getter/setter/deleter, validazione, e API stabile
- 11. Ereditarietà e `super()`: cosa succede davvero
- 12. Polimorfismo e duck typing: OOP “pythonica”
- 13. Composizione: modello reale per robotica/AI
- 14. Metodi speciali (dunder): `__repr__`, `__str__`, `__len__`, `__eq__`, ecc.
- 15. Errori tipici e debug: TypeError/AttributeError/NameError/IndentationError
- 16. Mini‑progetto completo: `Account` (con spiegazione token‑level)
- 17. Esercizi strutturati (con output atteso)

---

# 0) Prima di partire: cos’è una “classe” in Python (davvero)

In Python, una classe non è solo un “concetto”: è un **oggetto** anche lei.  
Quindi:
- le classi esistono a runtime
- puoi passarle come argomento
- puoi assegnarle a variabili
- hanno attributi e metodi (come gli oggetti che creano)

Questo è importante perché ti fa capire che Python è molto “dinamico”.  
Ma per imparare bene, noi useremo la versione “disciplinata”: classi chiare, attributi chiari, metodi chiari.

---

# 1) OOP: cos’è e perché ti serve (motivo pratico)

**OOP = Object Oriented Programming**.

In OOP organizzi il codice in **unità** che contengono:
- **stato** (dati) → attributi
- **comportamento** (azioni) → metodi

### Perché ti serve concretamente (non filosofia)
- **Pulizia**: eviti file giganteschi con funzioni scollegate.
- **Riutilizzo**: ereditarietà o composizione.
- **Modellazione**: “Robot ha un sensore”, “Utente ha un profilo”, ecc.
- **ML/DL reale**:  
  - in PyTorch un modello è una classe: `class MyNet(nn.Module): ...`
  - un dataset spesso è una classe: `class MyDataset(Dataset): ...`
  - molte pipeline moderne usano oggetti configurabili

---

# 2) Sintassi dei blocchi: indentazione e `:` (non è un dettaglio)

Python usa:
- `:` (due punti) per dire “qui inizia un blocco”
- indentazione (spazi o tab) per dire “queste righe appartengono al blocco”

Esempio:
```python
if True:
    print("dentro")
print("fuori")
```

### Token essenziali qui
- `if` è keyword
- `True` è un boolean
- `:` apre il blocco
- l’indentazione (di solito 4 spazi) è **sintassi**, non estetica

**Errore tipico:** `IndentationError` se mischi spazi e tab o sbagli livello.

---

# 3) Classe, oggetto, istanza (definizioni operative)

- **Classe**: un oggetto “stampino” che descrive come creare oggetti.
- **Istanza**: l’oggetto creato dalla classe.
- **Oggetto** (in Python): quasi tutto è un oggetto (liste, funzioni, classi, ecc.)

Esempio:
- classe: `Persona`
- istanza: `p = Persona("Lorenzo", 24)`
- attributo: `p.nome`
- metodo: `p.saluta()`

---

# 4) Il primo esempio: `Persona` (spiegazione fino alle virgole)

## 4.1 Codice
```python
class Persona:
    def __init__(self, nome, eta):
        self.nome = nome
        self.eta = eta

    def saluta(self):
        print(f"Ciao, mi chiamo {self.nome} e ho {self.eta} anni.")

p = Persona("Lorenzo", 24)
p.saluta()
```

## 4.2 Spiegazione riga‑per‑riga + token‑per‑token

### Riga 1: `class Persona:`
Token:
- `class` → keyword: dichiara una classe.
- (spazio) → separatore (non “token logico”, ma necessario per non unire parole).
- `Persona` → identificatore (nome classe). Convenzione: CamelCase.
- `:` → **fondamentale**: apre il blocco della classe.

### Riga 2: `def __init__(self, nome, eta):`
Token:
- `def` → keyword: definisce una funzione/metodo.
- spazio
- `__init__` → nome speciale: metodo chiamato alla creazione dell’istanza.
- `(` → apre lista parametri.
- `self` → primo parametro: riceve l’istanza creata (binding).
- `,` → virgola: separa parametri. **Essenziale** se ce ne sono più di uno.
- spazio (opzionale ma comune)
- `nome` → parametro 2.
- `,` → separatore.
- spazio
- `eta` → parametro 3.
- `)` → chiude lista parametri.
- `:` → apre blocco del metodo.

### Riga 3: `self.nome = nome`
Token:
- `self` → istanza.
- `.` → accesso attributo.
- `nome` → nome attributo sull’istanza.
- spazio
- `=` → assegnazione.
- spazio
- `nome` → parametro ricevuto.

### Riga 4: `self.eta = eta`
Stesso schema.

### Riga 6: `def saluta(self):`
Token:
- `def` keyword
- spazio
- `saluta` nome metodo
- `(` apre parametri
- `self` parametro istanza
- `)` chiude parametri
- `:` apre blocco

### Riga 7: `print(f"Ciao, mi chiamo {self.nome} e ho {self.eta} anni.")`
Token (dettaglio completo):
- `print` → funzione built‑in.
- `(` → apre argomenti.
- `f` → prefisso f‑string: abilita sostituzione `{...}`.
- `" ... "` → stringa delimitata da doppi apici.
- `{` e `}` → delimitano un’espressione Python dentro stringa.
- `self.nome` → espressione: attributo dell’istanza.
- `.` → in `self.nome` è operatore di accesso attributo (essenziale).
- `)` → chiude la chiamata a `print`.

### Riga 9: `p = Persona("Lorenzo", 24)`
Token:
- `p` nome variabile
- spazio
- `=` assegnazione
- spazio
- `Persona` chiamata alla classe (costruzione istanza)
- `(` apre argomenti
- `"Lorenzo"` stringa (argomento 1)
- `,` separatore argomenti
- spazio
- `24` intero (argomento 2)
- `)` chiude argomenti

### Riga 10: `p.saluta()`
Token:
- `p` istanza
- `.` accesso metodo
- `saluta` nome metodo
- `(` e `)` chiamata senza argomenti espliciti (oltre a `self` implicito)

---

# 5) `self` e il binding dei metodi (la “magia” spiegata)

Quando scrivi:
```python
p.saluta()
```
Python in pratica fa una cosa equivalente a:
```python
Persona.saluta(p)
```

## 5.1 Dimostrazione concreta (stesso risultato)
```python
class X:
    def f(self, y):
        return (self, y)

x = X()
print(x.f(10))
print(X.f(x, 10))
```

### Spiegazione token‑level
- `x.f(10)`:
  - `x` istanza
  - `.` accesso attributo → qui “f” viene trasformato in **bound method**
  - `(10)` argomento per `y`
  - `self` viene inserito automaticamente (è `x`)
- `X.f(x, 10)`:
  - chiami la funzione “grezza” sulla classe
  - passi `x` manualmente come primo argomento (self)
  - poi `10`

> Quindi: `self` è il modo con cui un metodo sa **su quale oggetto** sta lavorando.

### “self è obbligatorio?”
- Non è keyword, quindi il nome può cambiare.
- Ma il **primo parametro** deve esserci, perché Python passa l’istanza automaticamente quando chiami da istanza.

---

# 6) Attributi: lookup order, istanza vs classe, shadowing

## 6.1 Codice
```python
class Cane:
    specie = "Canis lupus familiaris"

    def __init__(self, nome):
        self.nome = nome

c = Cane("Luna")
print(c.nome)
print(c.specie)
print(Cane.specie)
```

### Spiegazione: come Python trova `c.specie`
Ordine semplificato:
1) Cerca `specie` in `c.__dict__` (attributi dell’istanza)
2) Se non lo trova, cerca in `Cane.__dict__` (attributi della classe)
3) Se non lo trova, risale nelle classi base

### Shadowing (ombra): quando l’istanza “sovrascrive” la classe
```python
c.specie = "SPECIE SOLO DI QUESTA ISTANZA"
print(c.specie)       # vede l'attributo dell'istanza
print(Cane.specie)    # resta quello di classe
```

Token‑level chiave:
- `c.specie = ...` crea `specie` dentro `c.__dict__`
- da quel momento `c.specie` non vede più quello di classe (finché quello di istanza esiste)

---

# 7) Metodi di istanza: parametri, default, `*args`, `**kwargs` in OOP

## 7.1 Default parameter
```python
class Contatore:
    def __init__(self):
        self.valore = 0

    def incrementa(self, passo=1):
        self.valore += passo
```

Token importanti:
- `passo=1`:
  - `passo` nome parametro
  - `=` assegna valore di default
  - `1` default
- `+=`:
  - operatore: `a += b` equivale a `a = a + b` (concettualmente)

## 7.2 `*args` e `**kwargs` in un metodo
```python
class Logger:
    def log(self, *args, **kwargs):
        print("args:", args)
        print("kwargs:", kwargs)

l = Logger()
l.log(1, 2, 3, livello="INFO", modulo="auth")
```

Spiegazione:
- `*args` raccoglie argomenti posizionali extra in una **tupla**
- `**kwargs` raccoglie argomenti keyword extra in un **dizionario**
- `livello="INFO"`:
  - `livello` chiave
  - `=` associa
  - `"INFO"` valore

---

# 8) `@classmethod`: factory methods e alternative constructors

## 8.1 Codice base
```python
class Rettangolo:
    def __init__(self, base, altezza):
        self.base = base
        self.altezza = altezza

    @classmethod
    def quadrato(cls, lato):
        return cls(lato, lato)
```

### Token‑level sul decoratore
- `@classmethod`:
  - `@` indica “decoratore applicato alla funzione sotto”
  - `classmethod` è un oggetto chiamabile che modifica il modo in cui la funzione viene legata

### Perché `cls` è importante
Se erediti:
```python
class Quadrato(Rettangolo):
    pass

q = Quadrato.quadrato(5)
```
`cls` sarà `Quadrato`, quindi `cls(lato, lato)` crea un Quadrato (non forza Rettangolo).

---

# 9) `@staticmethod`: utility senza self/cls

## 9.1 Codice
```python
class Matematica:
    @staticmethod
    def somma(a, b):
        return a + b
```

Token:
- `@staticmethod` → decoratore
- `somma(a, b)` → nessun `self/cls`
- `return` → restituisce valore al chiamante
- `a + b` → operatore di somma

### Regola d’oro
Se dentro lo staticmethod ti viene voglia di usare `self`… allora non è staticmethod.

---

# 10) `@property`: getter/setter/deleter

## 10.1 Codice completo
```python
class Prodotto:
    def __init__(self, nome, prezzo):
        self.nome = nome
        self.prezzo = prezzo

    @property
    def prezzo(self):
        return self._prezzo

    @prezzo.setter
    def prezzo(self, valore):
        if valore < 0:
            raise ValueError("Il prezzo non può essere negativo")
        self._prezzo = valore
```

### Token‑level: `@prezzo.setter`
- `@prezzo.setter`:
  - `@` decoratore
  - `prezzo` = nome della property
  - `.` accesso attributo `setter`
  - `setter` è un metodo speciale dell’oggetto `property` che registra la funzione come setter

### Perché `_prezzo` e non `prezzo`?
Se dentro il setter facessi `self.prezzo = valore`, richiameresti il setter all’infinito (ricorsione).  
Per evitare ciò usi un attributo “interno”: `_prezzo`.

---

# 11) Ereditarietà e `super()`

## 11.1 Codice
```python
class Persona:
    def __init__(self, nome):
        self.nome = nome

    def saluta(self):
        return f"Ciao, sono {self.nome}"

class Studente(Persona):
    def __init__(self, nome, matricola):
        super().__init__(nome)
        self.matricola = matricola

    def saluta(self):
        base = super().saluta()
        return base + f" e la mia matricola è {self.matricola}"
```

### Token‑level su `super().__init__(nome)`
- `super()`:
  - chiama l’oggetto “super” che consente accesso alla classe base seguendo MRO
- `.` accesso metodo
- `__init__` costruttore della base
- `(nome)` argomento passato

**Perché serve?**
Per riusare il codice del costruttore base e non duplicarlo.

---

# 12) Polimorfismo e duck typing

Python spesso usa il concetto: “se fa quack, è un’anatra”.  
Se un oggetto ha il metodo che ti serve, lo puoi usare.

```python
def presenta(x):
    print(x.saluta())
```

Non ti interessa la classe specifica, ti interessa l’interfaccia (il metodo).

---

# 13) Composizione (has‑a): la cosa più usata nei progetti reali

```python
class Sensore:
    def __init__(self, tipo):
        self.tipo = tipo

    def leggi(self):
        return f"Lettura da sensore {self.tipo}"

class Robot:
    def __init__(self, nome, sensore):
        self.nome = nome
        self.sensore = sensore

    def stato(self):
        return f"{self.nome}: " + self.sensore.leggi()
```

Qui Robot **non è** un Sensore, Robot **ha** un Sensore.

---

# 14) Dunder methods: come rendere gli oggetti “naturali” in Python

## 14.1 `__repr__`
```python
class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Punto(x={self.x}, y={self.y})"
```

- `__repr__` controlla cosa stampi in debug e console.
- f‑string dentro `return` costruisce la rappresentazione.

---

# 15) Errori tipici e debug

1) `TypeError: ... takes 1 positional argument but 2 were given`  
   → di solito hai dimenticato `self`.

2) `AttributeError: 'X' object has no attribute 'y'`  
   → stai leggendo un attributo mai creato in `__init__` o altrove.

3) `IndentationError`  
   → blocchi non allineati, tab/spazi mescolati.

---

# 16) Mini‑progetto: `Account` (spiegazione estesa)

## 16.1 Codice
```python
class Account:
    banca = "Banca Python"

    def __init__(self, intestatario, saldo_iniziale=0.0):
        self.intestatario = intestatario
        self.saldo = float(saldo_iniziale)

    def deposita(self, importo):
        if importo <= 0:
            raise ValueError("Importo non valido")
        self.saldo += importo

    def preleva(self, importo):
        if importo <= 0:
            raise ValueError("Importo non valido")
        if importo > self.saldo:
            raise ValueError("Saldo insufficiente")
        self.saldo -= importo

    @classmethod
    def da_bonus(cls, intestatario, bonus):
        acc = cls(intestatario, bonus)
        return acc

    @staticmethod
    def euro(valore):
        return f"€{valore:.2f}"

    def __repr__(self):
        return f"Account(intestatario={self.intestatario!r}, saldo={self.saldo})"
```

## 16.2 Token‑level: le parti “critiche”

### `saldo_iniziale=0.0`
- `saldo_iniziale` parametro
- `=` assegna default
- `0.0` float: è float e non int (scelta intenzionale per soldi in questo esempio didattico)

### `self.saldo = float(saldo_iniziale)`
- `float(...)` converte qualunque numero in float (es. `100` → `100.0`)
- le parentesi `()` sono essenziali per chiamare `float`

### `if importo <= 0:`
- `if` keyword
- `importo` variabile
- `<=` operatore confronto
- `0` letterale
- `:` apre blocco

### `raise ValueError("...")`
- `raise` keyword: lancia eccezione
- `ValueError` classe dell’eccezione
- `("...")` stringa messaggio

### `@classmethod` + `cls(...)`
- ti permette di creare oggetti con un “costruttore alternativo”

### `{self.intestatario!r}`
- `!r` forza `repr(...)` (utile per vedere virgolette e caratteri speciali)

---

# 17) Esercizi (strutturati, con output atteso)

### ES1 — Libro (base)
Crea classe `Libro` con:
- `titolo`, `autore`, `pagine`
- metodo `descrizione()` che ritorna una stringa
- implementa `__repr__`

**Output atteso (simile):**
`Libro(titolo='...', autore='...', pagine=...)`

### ES2 — Factory con `@classmethod`
Aggiungi `da_stringa("titolo;autore;pagine")`.

### ES3 — `property` su pagine
`pagine` non può essere `<= 0`.

### ES4 — Ereditarietà
Crea `Ebook(Libro)` con `formato` e override di `descrizione()`.

### ES5 — Composizione (Libreria)
Crea `Libreria` che contiene lista di libri e ha:
- `aggiungi(libro)`
- `totale_pagine()`
- `cerca(titolo)`

---

## Checklist finale (se hai capito)
- So spiegare perché `p.saluta()` equivale a `Persona.saluta(p)`
- So distinguere attributi di istanza e di classe
- So quando usare `classmethod`, `staticmethod`, `property`
- So leggere `super()` senza panico
- So scegliere tra ereditarietà e composizione

---

## Fine Lezione 8
Se vuoi, il passo successivo è: **OOP applicata a PyTorch** (perché i modelli sono classi).  


---

# APPENDICE A — Mega‑Esempio OOP (Biblioteca) con spiegazione riga‑per‑riga **token‑level**
Questa appendice è volutamente lunga: qui trovi un esempio realistico (più classi) e sotto **ogni riga** trovi i token e la spiegazione.
## A.1 Codice completo (da copiare ed eseguire)
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


class Libro:
    """
    Rappresenta un libro "fisico" o concettuale.
    Dimostra: attributi di istanza, property, metodi, __repr__.
    """

    # Attributo di classe: condiviso da tutte le istanze
    formato_default: str = "cartaceo"

    def __init__(self, titolo: str, autore: str, pagine: int) -> None:
        self.titolo = titolo
        self.autore = autore
        self.pagine = pagine  # passa dal setter della property

    @property
    def pagine(self) -> int:
        return self._pagine

    @pagine.setter
    def pagine(self, valore: int) -> None:
        if valore <= 0:
            raise ValueError("Le pagine devono essere un intero positivo")
        self._pagine = int(valore)

    def descrizione(self) -> str:
        return f"{self.titolo} — {self.autore} ({self.pagine} pagine)"

    def __repr__(self) -> str:
        return f"Libro(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine})"


class Ebook(Libro):
    """
    Sottoclasse di Libro.
    Dimostra: ereditarietà, super(), override, attributi aggiuntivi.
    """

    formato_default: str = "digitale"

    def __init__(self, titolo: str, autore: str, pagine: int, formato: str) -> None:
        super().__init__(titolo, autore, pagine)
        self.formato = formato

    def descrizione(self) -> str:
        base = super().descrizione()
        return base + f" [{self.formato}]"

    def __repr__(self) -> str:
        return (
            f"Ebook(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine}, formato={self.formato!r})"
        )


@dataclass(frozen=True)
class Utente:
    """
    Dataclass immutabile (frozen=True).
    Dimostra: oggetti-value, __init__ generato automaticamente, confronto.
    """
    nome: str
    id_utente: str


@dataclass
class Prestito:
    """
    Prestito = composizione: collega Utente e Libro.
    Dimostra: composizione, stato, Optional.
    """
    utente: Utente
    libro: Libro
    giorni: int = 14
    restituito: bool = False

    def chiudi(self) -> None:
        self.restituito = True


class Libreria:
    """
    Libreria che contiene libri e prestiti.
    Dimostra: composizione, classmethod factory, staticmethod utility, __len__.
    """

    # attributo di classe per generare ID semplici
    _seed: int = 1000

    def __init__(self, nome: str) -> None:
        self.nome = nome
        self.catalogo: Dict[str, Libro] = {}
        self.prestiti: List[Prestito] = []

    @staticmethod
    def _normalizza_chiave(titolo: str, autore: str) -> str:
        # Utility pura: non usa né self né cls.
        return f"{titolo.strip().lower()}::{autore.strip().lower()}"

    @classmethod
    def con_id(cls, nome: str) -> Libreria:
        # Factory method: crea una libreria con nome + id.
        cls._seed += 1
        return cls(f"{nome} #{cls._seed}")

    def aggiungi(self, libro: Libro) -> None:
        chiave = self._normalizza_chiave(libro.titolo, libro.autore)
        self.catalogo[chiave] = libro

    def cerca(self, titolo: str, autore: str) -> Optional[Libro]:
        chiave = self._normalizza_chiave(titolo, autore)
        return self.catalogo.get(chiave)

    def presta(self, utente: Utente, titolo: str, autore: str) -> Prestito:
        libro = self.cerca(titolo, autore)
        if libro is None:
            raise KeyError("Libro non trovato in catalogo")
        prestito = Prestito(utente=utente, libro=libro)
        self.prestiti.append(prestito)
        return prestito

    def prestiti_attivi(self) -> List[Prestito]:
        return [p for p in self.prestiti if not p.restituito]

    def chiudi_prestito(self, prestito: Prestito) -> None:
        prestito.chiudi()

    def __len__(self) -> int:
        return len(self.catalogo)

    def __repr__(self) -> str:
        return f"Libreria(nome={self.nome!r}, libri={len(self)}, prestiti={len(self.prestiti)})"


def demo() -> None:
    lib = Libreria.con_id("Biblioteca Centrale")

    l1 = Libro("Clean Code", "Robert C. Martin", 464)
    l2 = Ebook("Deep Learning", "Ian Goodfellow", 800, formato="pdf")

    lib.aggiungi(l1)
    lib.aggiungi(l2)

    u = Utente(nome="Lorenzo", id_utente="U001")

    p = lib.presta(u, "Clean Code", "Robert C. Martin")
    print("Prestito creato:", p)
    print("Prestiti attivi:", lib.prestiti_attivi())

    lib.chiudi_prestito(p)
    print("Prestiti attivi dopo chiusura:", lib.prestiti_attivi())

    print("Catalogo size:", len(lib))
    print("Rappresentazione:", lib)


if __name__ == "__main__":
    demo()
```
## A.2 Spiegazione riga‑per‑riga (ogni parola e simbolo)
> Nota: gli spazi tra token migliorano la leggibilità; l’**indentazione** viene rappresentata come token `INDENT/DEDENT` perché in Python è sintassi.
### Riga 1
```python
from __future__ import annotations
```
**Token presenti (in ordine):**
- `from` → KEYWORD: `from` è una parola chiave del linguaggio.
- `__future__` → NAME: identificatore `__future__` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `import` → KEYWORD: `import` è una parola chiave del linguaggio.
- `annotations` → NAME: identificatore `annotations` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 2
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 3
```python
from dataclasses import dataclass
```
**Token presenti (in ordine):**
- `from` → KEYWORD: `from` è una parola chiave del linguaggio.
- `dataclasses` → NAME: identificatore `dataclasses` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `import` → KEYWORD: `import` è una parola chiave del linguaggio.
- `dataclass` → NAME: identificatore `dataclass` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 4
```python
from typing import Dict, List, Optional
```
**Token presenti (in ordine):**
- `from` → KEYWORD: `from` è una parola chiave del linguaggio.
- `typing` → NAME: identificatore `typing` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `import` → KEYWORD: `import` è una parola chiave del linguaggio.
- `Dict` → NAME: identificatore `Dict` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `List` → NAME: identificatore `List` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `Optional` → NAME: identificatore `Optional` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 5
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 6
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 7
```python
class Libro:
```
**Token presenti (in ordine):**
- `class` → KEYWORD: `class` è una parola chiave del linguaggio.
- `Libro` → NAME: nome della classe dichiarata: `Libro`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 8
```python
    """
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `"""
    Rappresenta un libro "fisico" o concettuale.
    Dimostra: attributi di istanza, property, metodi, __repr__.
    """` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.

### Riga 9
```python
    Rappresenta un libro "fisico" o concettuale.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 10
```python
    Dimostra: attributi di istanza, property, metodi, __repr__.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 11
```python
    """
```
**Token presenti (in ordine):**
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 12
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 13
```python
    # Attributo di classe: condiviso da tutte le istanze
```
**Token presenti (in ordine):**
- `# Attributo di classe: condiviso da tutte le istanze` → COMMENT: commento, ignorato dall’esecuzione (ma utile per umani).
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 14
```python
    formato_default: str = "cartaceo"
```
**Token presenti (in ordine):**
- `formato_default` → NAME: identificatore `formato_default` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `"cartaceo"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 15
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 16
```python
    def __init__(self, titolo: str, autore: str, pagine: int) -> None:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__init__` → NAME: nome della funzione/metodo dichiarato: `__init__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `pagine` → NAME: identificatore `pagine` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 17
```python
        self.titolo = titolo
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `titolo` → NAME: attributo/metodo chiamato dopo un punto: `titolo`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 18
```python
        self.autore = autore
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `autore` → NAME: attributo/metodo chiamato dopo un punto: `autore`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 19
```python
        self.pagine = pagine  # passa dal setter della property
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `pagine` → NAME: attributo/metodo chiamato dopo un punto: `pagine`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `pagine` → NAME: identificatore `pagine` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `# passa dal setter della property` → COMMENT: commento, ignorato dall’esecuzione (ma utile per umani).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 20
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 21
```python
    @property
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `property` → NAME: nome del decoratore `property` (modifica il comportamento della funzione sottostante).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 22
```python
    def pagine(self) -> int:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `pagine` → NAME: nome della funzione/metodo dichiarato: `pagine`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 23
```python
        return self._pagine
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `_pagine` → NAME: attributo/metodo chiamato dopo un punto: `_pagine`.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 24
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 25
```python
    @pagine.setter
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `pagine` → NAME: nome del decoratore `pagine` (modifica il comportamento della funzione sottostante).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `setter` → NAME: attributo/metodo chiamato dopo un punto: `setter`.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 26
```python
    def pagine(self, valore: int) -> None:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `pagine` → NAME: nome della funzione/metodo dichiarato: `pagine`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `valore` → NAME: identificatore `valore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 27
```python
        if valore <= 0:
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `if` → KEYWORD: `if` è una parola chiave del linguaggio.
- `valore` → NAME: identificatore `valore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `<=` → OP: simbolo `<=`. Confronto: minore o uguale.
- `0` → NUMBER: numero letterale (int/float).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 28
```python
            raise ValueError("Le pagine devono essere un intero positivo")
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `raise` → KEYWORD: `raise` è una parola chiave del linguaggio.
- `ValueError` → NAME: identificatore `ValueError` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Le pagine devono essere un intero positivo"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 29
```python
        self._pagine = int(valore)
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `_pagine` → NAME: attributo/metodo chiamato dopo un punto: `_pagine`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `valore` → NAME: identificatore `valore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 30
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 31
```python
    def descrizione(self) -> str:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `descrizione` → NAME: nome della funzione/metodo dichiarato: `descrizione`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 32
```python
        return f"{self.titolo} — {self.autore} ({self.pagine} pagine)"
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `f"{self.titolo} — {self.autore} ({self.pagine} pagine)"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 33
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 34
```python
    def __repr__(self) -> str:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__repr__` → NAME: nome della funzione/metodo dichiarato: `__repr__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 35
```python
        return f"Libro(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine})"
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `f"Libro(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine})"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 36
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 37
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 38
```python
class Ebook(Libro):
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `class` → KEYWORD: `class` è una parola chiave del linguaggio.
- `Ebook` → NAME: nome della classe dichiarata: `Ebook`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 39
```python
    """
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `"""
    Sottoclasse di Libro.
    Dimostra: ereditarietà, super(), override, attributi aggiuntivi.
    """` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.

### Riga 40
```python
    Sottoclasse di Libro.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 41
```python
    Dimostra: ereditarietà, super(), override, attributi aggiuntivi.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 42
```python
    """
```
**Token presenti (in ordine):**
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 43
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 44
```python
    formato_default: str = "digitale"
```
**Token presenti (in ordine):**
- `formato_default` → NAME: identificatore `formato_default` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `"digitale"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 45
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 46
```python
    def __init__(self, titolo: str, autore: str, pagine: int, formato: str) -> None:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__init__` → NAME: nome della funzione/metodo dichiarato: `__init__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `pagine` → NAME: identificatore `pagine` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `formato` → NAME: identificatore `formato` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 47
```python
        super().__init__(titolo, autore, pagine)
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `super` → NAME: identificatore `super` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `__init__` → NAME: attributo/metodo chiamato dopo un punto: `__init__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `pagine` → NAME: identificatore `pagine` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 48
```python
        self.formato = formato
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `formato` → NAME: attributo/metodo chiamato dopo un punto: `formato`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `formato` → NAME: identificatore `formato` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 49
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 50
```python
    def descrizione(self) -> str:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `descrizione` → NAME: nome della funzione/metodo dichiarato: `descrizione`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 51
```python
        base = super().descrizione()
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `base` → NAME: identificatore `base` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `super` → NAME: identificatore `super` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `descrizione` → NAME: attributo/metodo chiamato dopo un punto: `descrizione`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 52
```python
        return base + f" [{self.formato}]"
```
**Token presenti (in ordine):**
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `base` → NAME: identificatore `base` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `+` → OP: simbolo `+`. Somma/concatenazione.
- `f" [{self.formato}]"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 53
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 54
```python
    def __repr__(self) -> str:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__repr__` → NAME: nome della funzione/metodo dichiarato: `__repr__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 55
```python
        return (
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 56
```python
            f"Ebook(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine}, formato={self.formato!r})"
```
**Token presenti (in ordine):**
- `f"Ebook(titolo={self.titolo!r}, autore={self.autore!r}, pagine={self.pagine}, formato={self.formato!r})"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 57
```python
        )
```
**Token presenti (in ordine):**
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 58
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 59
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 60
```python
@dataclass(frozen=True)
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `dataclass` → NAME: nome del decoratore `dataclass` (modifica il comportamento della funzione sottostante).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `frozen` → NAME: identificatore `frozen` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `True` → KEYWORD: `True` è una parola chiave del linguaggio.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 61
```python
class Utente:
```
**Token presenti (in ordine):**
- `class` → KEYWORD: `class` è una parola chiave del linguaggio.
- `Utente` → NAME: nome della classe dichiarata: `Utente`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 62
```python
    """
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `"""
    Dataclass immutabile (frozen=True).
    Dimostra: oggetti-value, __init__ generato automaticamente, confronto.
    """` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.

### Riga 63
```python
    Dataclass immutabile (frozen=True).
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 64
```python
    Dimostra: oggetti-value, __init__ generato automaticamente, confronto.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 65
```python
    """
```
**Token presenti (in ordine):**
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 66
```python
    nome: str
```
**Token presenti (in ordine):**
- `nome` → NAME: identificatore `nome` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 67
```python
    id_utente: str
```
**Token presenti (in ordine):**
- `id_utente` → NAME: identificatore `id_utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 68
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 69
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 70
```python
@dataclass
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `dataclass` → NAME: nome del decoratore `dataclass` (modifica il comportamento della funzione sottostante).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 71
```python
class Prestito:
```
**Token presenti (in ordine):**
- `class` → KEYWORD: `class` è una parola chiave del linguaggio.
- `Prestito` → NAME: nome della classe dichiarata: `Prestito`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 72
```python
    """
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `"""
    Prestito = composizione: collega Utente e Libro.
    Dimostra: composizione, stato, Optional.
    """` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.

### Riga 73
```python
    Prestito = composizione: collega Utente e Libro.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 74
```python
    Dimostra: composizione, stato, Optional.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 75
```python
    """
```
**Token presenti (in ordine):**
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 76
```python
    utente: Utente
```
**Token presenti (in ordine):**
- `utente` → NAME: identificatore `utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Utente` → NAME: identificatore `Utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 77
```python
    libro: Libro
```
**Token presenti (in ordine):**
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 78
```python
    giorni: int = 14
```
**Token presenti (in ordine):**
- `giorni` → NAME: identificatore `giorni` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `14` → NUMBER: numero letterale (int/float).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 79
```python
    restituito: bool = False
```
**Token presenti (in ordine):**
- `restituito` → NAME: identificatore `restituito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `bool` → NAME: identificatore `bool` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `False` → KEYWORD: `False` è una parola chiave del linguaggio.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 80
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 81
```python
    def chiudi(self) -> None:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `chiudi` → NAME: nome della funzione/metodo dichiarato: `chiudi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 82
```python
        self.restituito = True
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `restituito` → NAME: attributo/metodo chiamato dopo un punto: `restituito`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `True` → KEYWORD: `True` è una parola chiave del linguaggio.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 83
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 84
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 85
```python
class Libreria:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `class` → KEYWORD: `class` è una parola chiave del linguaggio.
- `Libreria` → NAME: nome della classe dichiarata: `Libreria`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 86
```python
    """
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `"""
    Libreria che contiene libri e prestiti.
    Dimostra: composizione, classmethod factory, staticmethod utility, __len__.
    """` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.

### Riga 87
```python
    Libreria che contiene libri e prestiti.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 88
```python
    Dimostra: composizione, classmethod factory, staticmethod utility, __len__.
```
**Token presenti (in ordine):**
- (nessun token: riga vuota)

### Riga 89
```python
    """
```
**Token presenti (in ordine):**
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 90
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 91
```python
    # attributo di classe per generare ID semplici
```
**Token presenti (in ordine):**
- `# attributo di classe per generare ID semplici` → COMMENT: commento, ignorato dall’esecuzione (ma utile per umani).
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 92
```python
    _seed: int = 1000
```
**Token presenti (in ordine):**
- `_seed` → NAME: identificatore `_seed` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `1000` → NUMBER: numero letterale (int/float).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 93
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 94
```python
    def __init__(self, nome: str) -> None:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__init__` → NAME: nome della funzione/metodo dichiarato: `__init__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `nome` → NAME: identificatore `nome` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 95
```python
        self.nome = nome
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `nome` → NAME: attributo/metodo chiamato dopo un punto: `nome`.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `nome` → NAME: identificatore `nome` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 96
```python
        self.catalogo: Dict[str, Libro] = {}
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `catalogo` → NAME: attributo/metodo chiamato dopo un punto: `catalogo`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Dict` → NAME: identificatore `Dict` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `{` → OP: simbolo `{`. Graffa aperta: inizia dizionario/set o una sezione (nelle f-string).
- `}` → OP: simbolo `}`. Graffa chiusa.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 97
```python
        self.prestiti: List[Prestito] = []
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `prestiti` → NAME: attributo/metodo chiamato dopo un punto: `prestiti`.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `List` → NAME: identificatore `List` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `Prestito` → NAME: identificatore `Prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 98
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 99
```python
    @staticmethod
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `staticmethod` → NAME: nome del decoratore `staticmethod` (modifica il comportamento della funzione sottostante).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 100
```python
    def _normalizza_chiave(titolo: str, autore: str) -> str:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `_normalizza_chiave` → NAME: nome della funzione/metodo dichiarato: `_normalizza_chiave`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 101
```python
        # Utility pura: non usa né self né cls.
```
**Token presenti (in ordine):**
- `# Utility pura: non usa né self né cls.` → COMMENT: commento, ignorato dall’esecuzione (ma utile per umani).
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 102
```python
        return f"{titolo.strip().lower()}::{autore.strip().lower()}"
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `f"{titolo.strip().lower()}::{autore.strip().lower()}"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 103
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 104
```python
    @classmethod
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `@` → OP: simbolo `@`. Operatore/simbolo di Python: significato dipende dal contesto della riga.
- `classmethod` → NAME: nome del decoratore `classmethod` (modifica il comportamento della funzione sottostante).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 105
```python
    def con_id(cls, nome: str) -> Libreria:
```
**Token presenti (in ordine):**
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `con_id` → NAME: nome della funzione/metodo dichiarato: `con_id`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `cls` → NAME: `cls` è la convenzione per il primo parametro dei classmethod (riferimento alla classe).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `nome` → NAME: identificatore `nome` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `Libreria` → NAME: identificatore `Libreria` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 106
```python
        # Factory method: crea una libreria con nome + id.
```
**Token presenti (in ordine):**
- `# Factory method: crea una libreria con nome + id.` → COMMENT: commento, ignorato dall’esecuzione (ma utile per umani).
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 107
```python
        cls._seed += 1
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `cls` → NAME: `cls` è la convenzione per il primo parametro dei classmethod (riferimento alla classe).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `_seed` → NAME: attributo/metodo chiamato dopo un punto: `_seed`.
- `+=` → OP: simbolo `+=`. Aggiornamento: a += b equivale concettualmente a a = a + b.
- `1` → NUMBER: numero letterale (int/float).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 108
```python
        return cls(f"{nome} #{cls._seed}")
```
**Token presenti (in ordine):**
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `cls` → NAME: `cls` è la convenzione per il primo parametro dei classmethod (riferimento alla classe).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `f"{nome} #{cls._seed}"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 109
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 110
```python
    def aggiungi(self, libro: Libro) -> None:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `aggiungi` → NAME: nome della funzione/metodo dichiarato: `aggiungi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 111
```python
        chiave = self._normalizza_chiave(libro.titolo, libro.autore)
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `chiave` → NAME: identificatore `chiave` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `_normalizza_chiave` → NAME: attributo/metodo chiamato dopo un punto: `_normalizza_chiave`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `titolo` → NAME: attributo/metodo chiamato dopo un punto: `titolo`.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `autore` → NAME: attributo/metodo chiamato dopo un punto: `autore`.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 112
```python
        self.catalogo[chiave] = libro
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `catalogo` → NAME: attributo/metodo chiamato dopo un punto: `catalogo`.
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `chiave` → NAME: identificatore `chiave` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 113
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 114
```python
    def cerca(self, titolo: str, autore: str) -> Optional[Libro]:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `cerca` → NAME: nome della funzione/metodo dichiarato: `cerca`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `Optional` → NAME: identificatore `Optional` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 115
```python
        chiave = self._normalizza_chiave(titolo, autore)
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `chiave` → NAME: identificatore `chiave` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `_normalizza_chiave` → NAME: attributo/metodo chiamato dopo un punto: `_normalizza_chiave`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 116
```python
        return self.catalogo.get(chiave)
```
**Token presenti (in ordine):**
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `catalogo` → NAME: attributo/metodo chiamato dopo un punto: `catalogo`.
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `get` → NAME: attributo/metodo chiamato dopo un punto: `get`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `chiave` → NAME: identificatore `chiave` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 117
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 118
```python
    def presta(self, utente: Utente, titolo: str, autore: str) -> Prestito:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `presta` → NAME: nome della funzione/metodo dichiarato: `presta`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `utente` → NAME: identificatore `utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Utente` → NAME: identificatore `Utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `Prestito` → NAME: identificatore `Prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 119
```python
        libro = self.cerca(titolo, autore)
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `cerca` → NAME: attributo/metodo chiamato dopo un punto: `cerca`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `titolo` → NAME: identificatore `titolo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `autore` → NAME: identificatore `autore` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 120
```python
        if libro is None:
```
**Token presenti (in ordine):**
- `if` → KEYWORD: `if` è una parola chiave del linguaggio.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `is` → KEYWORD: `is` è una parola chiave del linguaggio.
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 121
```python
            raise KeyError("Libro non trovato in catalogo")
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `raise` → KEYWORD: `raise` è una parola chiave del linguaggio.
- `KeyError` → NAME: identificatore `KeyError` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Libro non trovato in catalogo"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 122
```python
        prestito = Prestito(utente=utente, libro=libro)
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `prestito` → NAME: identificatore `prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `Prestito` → NAME: identificatore `Prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `utente` → NAME: identificatore `utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `utente` → NAME: identificatore `utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `libro` → NAME: identificatore `libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 123
```python
        self.prestiti.append(prestito)
```
**Token presenti (in ordine):**
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `prestiti` → NAME: attributo/metodo chiamato dopo un punto: `prestiti`.
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `append` → NAME: attributo/metodo chiamato dopo un punto: `append`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `prestito` → NAME: identificatore `prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 124
```python
        return prestito
```
**Token presenti (in ordine):**
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `prestito` → NAME: identificatore `prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 125
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 126
```python
    def prestiti_attivi(self) -> List[Prestito]:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `prestiti_attivi` → NAME: nome della funzione/metodo dichiarato: `prestiti_attivi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `List` → NAME: identificatore `List` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `Prestito` → NAME: identificatore `Prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 127
```python
        return [p for p in self.prestiti if not p.restituito]
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `[` → OP: simbolo `[`. Parentesi quadra aperta: inizia lista/indice/annotation di tipo.
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `for` → KEYWORD: `for` è una parola chiave del linguaggio.
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `in` → KEYWORD: `in` è una parola chiave del linguaggio.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `prestiti` → NAME: attributo/metodo chiamato dopo un punto: `prestiti`.
- `if` → KEYWORD: `if` è una parola chiave del linguaggio.
- `not` → KEYWORD: `not` è una parola chiave del linguaggio.
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `restituito` → NAME: attributo/metodo chiamato dopo un punto: `restituito`.
- `]` → OP: simbolo `]`. Parentesi quadra chiusa.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 128
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 129
```python
    def chiudi_prestito(self, prestito: Prestito) -> None:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `chiudi_prestito` → NAME: nome della funzione/metodo dichiarato: `chiudi_prestito`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `prestito` → NAME: identificatore `prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- `Prestito` → NAME: identificatore `Prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 130
```python
        prestito.chiudi()
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `prestito` → NAME: identificatore `prestito` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `chiudi` → NAME: attributo/metodo chiamato dopo un punto: `chiudi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 131
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 132
```python
    def __len__(self) -> int:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__len__` → NAME: nome della funzione/metodo dichiarato: `__len__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `int` → NAME: identificatore `int` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 133
```python
        return len(self.catalogo)
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `len` → NAME: identificatore `len` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `catalogo` → NAME: attributo/metodo chiamato dopo un punto: `catalogo`.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 134
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 135
```python
    def __repr__(self) -> str:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `__repr__` → NAME: nome della funzione/metodo dichiarato: `__repr__`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `self` → NAME: `self` è la convenzione per il primo parametro dei metodi di istanza (riferimento all’oggetto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `str` → NAME: identificatore `str` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 136
```python
        return f"Libreria(nome={self.nome!r}, libri={len(self)}, prestiti={len(self.prestiti)})"
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `return` → KEYWORD: `return` è una parola chiave del linguaggio.
- `f"Libreria(nome={self.nome!r}, libri={len(self)}, prestiti={len(self.prestiti)})"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 137
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 138
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 139
```python
def demo() -> None:
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `def` → KEYWORD: `def` è una parola chiave del linguaggio.
- `demo` → NAME: nome della funzione/metodo dichiarato: `demo`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `->` → OP: simbolo `->`. Annotazione del tipo di ritorno (type hint).
- `None` → KEYWORD: `None` è una parola chiave del linguaggio.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 140
```python
    lib = Libreria.con_id("Biblioteca Centrale")
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `Libreria` → NAME: identificatore `Libreria` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `con_id` → NAME: attributo/metodo chiamato dopo un punto: `con_id`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Biblioteca Centrale"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 141
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 142
```python
    l1 = Libro("Clean Code", "Robert C. Martin", 464)
```
**Token presenti (in ordine):**
- `l1` → NAME: identificatore `l1` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `Libro` → NAME: identificatore `Libro` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Clean Code"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `"Robert C. Martin"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `464` → NUMBER: numero letterale (int/float).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 143
```python
    l2 = Ebook("Deep Learning", "Ian Goodfellow", 800, formato="pdf")
```
**Token presenti (in ordine):**
- `l2` → NAME: identificatore `l2` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `Ebook` → NAME: identificatore `Ebook` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Deep Learning"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `"Ian Goodfellow"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `800` → NUMBER: numero letterale (int/float).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `formato` → NAME: identificatore `formato` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `"pdf"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 144
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 145
```python
    lib.aggiungi(l1)
```
**Token presenti (in ordine):**
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `aggiungi` → NAME: attributo/metodo chiamato dopo un punto: `aggiungi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `l1` → NAME: identificatore `l1` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 146
```python
    lib.aggiungi(l2)
```
**Token presenti (in ordine):**
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `aggiungi` → NAME: attributo/metodo chiamato dopo un punto: `aggiungi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `l2` → NAME: identificatore `l2` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 147
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 148
```python
    u = Utente(nome="Lorenzo", id_utente="U001")
```
**Token presenti (in ordine):**
- `u` → NAME: identificatore `u` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `Utente` → NAME: identificatore `Utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `nome` → NAME: identificatore `nome` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `"Lorenzo"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `id_utente` → NAME: identificatore `id_utente` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `"U001"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 149
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 150
```python
    p = lib.presta(u, "Clean Code", "Robert C. Martin")
```
**Token presenti (in ordine):**
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `=` → OP: simbolo `=`. Assegnazione: imposta il valore a destra nel nome/attributo a sinistra.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `presta` → NAME: attributo/metodo chiamato dopo un punto: `presta`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `u` → NAME: identificatore `u` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `"Clean Code"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `"Robert C. Martin"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 151
```python
    print("Prestito creato:", p)
```
**Token presenti (in ordine):**
- `print` → NAME: identificatore `print` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Prestito creato:"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 152
```python
    print("Prestiti attivi:", lib.prestiti_attivi())
```
**Token presenti (in ordine):**
- `print` → NAME: identificatore `print` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Prestiti attivi:"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `prestiti_attivi` → NAME: attributo/metodo chiamato dopo un punto: `prestiti_attivi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 153
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 154
```python
    lib.chiudi_prestito(p)
```
**Token presenti (in ordine):**
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `chiudi_prestito` → NAME: attributo/metodo chiamato dopo un punto: `chiudi_prestito`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `p` → NAME: identificatore `p` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 155
```python
    print("Prestiti attivi dopo chiusura:", lib.prestiti_attivi())
```
**Token presenti (in ordine):**
- `print` → NAME: identificatore `print` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Prestiti attivi dopo chiusura:"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `.` → OP: simbolo `.`. Punto: accesso a attributo/metodo o qualificazione di un nome.
- `prestiti_attivi` → NAME: attributo/metodo chiamato dopo un punto: `prestiti_attivi`.
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 156
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 157
```python
    print("Catalogo size:", len(lib))
```
**Token presenti (in ordine):**
- `print` → NAME: identificatore `print` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Catalogo size:"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `len` → NAME: identificatore `len` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 158
```python
    print("Rappresentazione:", lib)
```
**Token presenti (in ordine):**
- `print` → NAME: identificatore `print` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `"Rappresentazione:"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `,` → OP: simbolo `,`. Virgola: separa elementi/argomenti/parametri. Essenziale quando richiesto dalla sintassi.
- `lib` → NAME: identificatore `lib` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 159
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 160
```python

```
**Token presenti (in ordine):**
- <NL> → NL: newline “non logica” (es. linee vuote o continuazioni).

### Riga 161
```python
if __name__ == "__main__":
```
**Token presenti (in ordine):**
- <DEDENT> → DEDENT: fine di un blocco indentato.
- `if` → KEYWORD: `if` è una parola chiave del linguaggio.
- `__name__` → NAME: identificatore `__name__` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `==` → OP: simbolo `==`. Confronto di uguaglianza.
- `"__main__"` → STRING: letterale stringa (testo). Può essere docstring se in testa a classe/funzione.
- `:` → OP: simbolo `:`. Due punti: apre un blocco indentato (sintassi fondamentale).
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

### Riga 162
```python
    demo()
```
**Token presenti (in ordine):**
- <INDENT> → INDENT: indentazione (spazi). In Python fa parte della sintassi dei blocchi.
- `demo` → NAME: identificatore `demo` (nome di variabile, classe, funzione o attributo a seconda del contesto).
- `(` → OP: simbolo `(`. Parentesi tonda aperta: inizia lista di argomenti/parametri o raggruppa espressioni.
- `)` → OP: simbolo `)`. Parentesi tonda chiusa: termina lista di argomenti/parametri o raggruppamento.
- <NEWLINE> → NEWLINE: fine riga logica (termina uno statement).

