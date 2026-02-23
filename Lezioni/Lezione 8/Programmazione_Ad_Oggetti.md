# Python - Programmazione ad Oggetti (OOP) e Metodi Speciali

La **Programmazione ad Oggetti (OOP)** organizza il codice in **classi** e **oggetti**, rendendolo più modulare, riutilizzabile e scalabile.

---

## **1️⃣ Concetti Base della OOP**
✅ **Oggetto** → Un'istanza di una classe.  
✅ **Classe** → Un modello che definisce attributi e metodi.  
✅ **Attributi** → Variabili di un oggetto.  
✅ **Metodi** → Funzioni definite dentro una classe.  
✅ **Incapsulamento** → Proteggere i dati dall'accesso esterno.  
✅ **Ereditarietà** → Una classe può ereditare attributi e metodi da un'altra.  
✅ **Polimorfismo** → Un metodo può avere diverse implementazioni in classi diverse.  

---

## **2️⃣ Creare una Classe e un Oggetto**
```python
class Auto:
    def __init__(self, marca, modello, anno):
        self.marca = marca
        self.modello = modello
        self.anno = anno

    def descrizione(self):
        return f"{self.marca} {self.modello} ({self.anno})"

mia_auto = Auto("Toyota", "Corolla", 2022)
print(mia_auto.descrizione())  # Toyota Corolla (2022)
```

---

## **3️⃣ Attributi di Classe e di Istanza**
```python
class Persona:
    specie = "Homo Sapiens"  # Attributo di classe

    def __init__(self, nome, età):
        self.nome = nome
        self.età = età

p1 = Persona("Luca", 30)
p2 = Persona("Anna", 25)
print(p1.specie, p1.nome)  # Homo Sapiens Luca
```

---

## **4️⃣ Metodi di Classe e Statici**

| Tipo di Metodo | Parametro | Quando usarlo? |
|---------------|------------|----------------------|
| **Metodo di istanza** | `self` | Quando serve modificare attributi dell'oggetto |
| **Metodo di classe (`@classmethod`)** | `cls` | Quando serve modificare attributi della classe |
| **Metodo statico (`@staticmethod`)** | Nessuno | Quando il metodo non usa né `self` né `cls` |

### **Esempio `@classmethod` per Modificare un Attributo di Classe**
```python
class Studente:
    scuola = "Liceo Galileo"

    def __init__(self, nome, voto):
        self.nome = nome
        self.voto = voto

    @classmethod
    def cambia_scuola(cls, nuova_scuola):
        cls.scuola = nuova_scuola

Studente.cambia_scuola("Istituto Tecnico")
print(Studente.scuola)  # Istituto Tecnico
```

### **Esempio `@staticmethod` per Funzioni Indipendenti**
```python
class Matematica:
    @staticmethod
    def somma(a, b):
        return a + b

print(Matematica.somma(3, 4))  # 7
```

---

## **5️⃣ Incapsulamento: Proteggere i Dati**
```python
class ContoBancario:
    def __init__(self, saldo):
        self.__saldo = saldo  # Variabile privata

    def deposito(self, importo):
        self.__saldo += importo

    def get_saldo(self):
        return self.__saldo

conto = ContoBancario(1000)
conto.deposito(500)
print(conto.get_saldo())  # 1500
```

---

## **6️⃣ Ereditarietà: Creare Sottoclassi**
```python
class Persona:
    def __init__(self, nome, età):
        self.nome = nome
        self.età = età

    def info(self):
        return f"{self.nome}, {self.età} anni"

class Studente(Persona):
    def __init__(self, nome, età, scuola):
        super().__init__(nome, età)
        self.scuola = scuola

s1 = Studente("Anna", 20, "Università di Roma")
print(s1.info())  # Anna, 20 anni
```

---

## **7️⃣ Polimorfismo: Stesso Metodo, Comportamenti Diversi**
```python
class Animale:
    def verso(self):
        return "Suono generico"

class Cane(Animale):
    def verso(self):
        return "Bau!"

class Gatto(Animale):
    def verso(self):
        return "Miao!"

animali = [Cane(), Gatto(), Animale()]
for animale in animali:
    print(animale.verso())
```

---

## **🔟 Riassunto**
| Concetto | Descrizione |
|----------|------------|
| **Classe** | Modello per creare oggetti |
| **Oggetto** | Istanza di una classe |
| **Incapsulamento** | Nasconde i dettagli interni |
| **Ereditarietà** | Una classe può ereditare da un'altra |
| **Polimorfismo** | Stesso metodo, comportamenti diversi |

✅ **Ora prova questi concetti nei tuoi progetti Python!** 🚀

