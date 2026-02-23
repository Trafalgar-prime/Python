# Python - Lezione 8: Programmazione ad Oggetti (OOP)

La **Programmazione ad Oggetti (OOP)** è un paradigma che organizza il codice in **classi** e **oggetti**, rendendolo più modulare, riutilizzabile e scalabile.

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
```python
class Studente:
    scuola = "Liceo Galileo"

    def __init__(self, nome, voto):
        self.nome = nome
        self.voto = voto

    @classmethod
    def cambia_scuola(cls, nuova_scuola):
        cls.scuola = nuova_scuola

    @staticmethod
    def benvenuto():
        return "Benvenuto nella scuola!"

s1 = Studente("Marco", 8)
print(Studente.benvenuto())  # Benvenuto nella scuola!
```

---

## **5️⃣ Incapsulamento: Attributi Privati**
```python
class ContoBancario:
    def __init__(self, saldo):
        self.__saldo = saldo

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

## **🔟 Esercizi**
1️⃣ **Crea una classe `Libro` con titolo, autore e anno.**  
2️⃣ **Crea una classe `Rettangolo` con metodi per calcolare l'area.**  
3️⃣ **Crea una classe `Veicolo` e una sottoclasse `Auto`.**  
4️⃣ **Crea una classe `Banca` con metodi per depositare e prelevare.**  
5️⃣ **Crea una classe `Animale` e due sottoclassi `Cane` e `Gatto` con suoni diversi.**  

---

## ✅ **Obiettivo raggiunto**
✅ **Hai imparato i concetti base della OOP: classi, oggetti, metodi e attributi.**  
✅ **Hai visto ereditarietà, incapsulamento e polimorfismo.**  
✅ **Ora prova gli esercizi per mettere in pratica la teoria!** 🚀

