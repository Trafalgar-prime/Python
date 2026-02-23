class Auto:
    def __init__(self, marca, modello, anno): #__init__ mi serve per forza per inizializzare l'oggetto
    
        self.marca = marca #attributo dell'istanza
        self.modello = modello #self.[variabile1] (può essere totalmente inventata e casuale) = [variabile2] (questa variabile deve essere uguale alla variabile chiamata nella funzione)
        self.anno = anno

    def descrizione(self):
        return f"{self.marca} {self.modello} ({self.anno})"

#creiamo un oggetto della classe auto
mia_auto = Auto("Toyota", "Corolla", 2022)
print(mia_auto.descrizione())


class Persona:
    specie = "Homo Sapiens" #attributo della classe

    def __init__(self,nome,età):
        self.nome = nome
        self.età = età

#creiamo 2 oggetti
p1 = Persona("Luca", 30)
p2 = Persona("Anna", 25)

print(p1.specie,p1.nome)
print(p2.specie,p2.nome)


class Studente:
    scuola = "Liceo Galileo" #attributo della classe

    def __init__(self,nome,voto):
        self.nome = nome
        self.voto = voto

    def info(self):
        return f"{self.nome} ha preso {self.voto}"
    #quando pongo self sto cambiando attributo di un oggetto, ma solo all'oggetto specificato

    @classmethod
    def cambio_scuola(cls, nome_scuola):
        cls.scuola = nome_scuola
    #classmethod fa cambiare scuola a tutti gli studenti, e non solo ad uno studente in particolare, non posso infatti cambiare la scuola solo a luca e non a marco, e viceversa. 
    #anche se ponessi s1.cambio_scuola, cambierei scuola comunque a tutta la classe

    @staticmethod
    def benvenuto():
        return "Benvenuto nella nuova scuola"
    #non cambio nessun attributo ma definisco solo qualcosa

s1 = Studente("Marco", 8)
print(s1.info())
print(Studente.benvenuto())


class ContoBancario:
    def __init__(self,saldo):
        self.__saldo = saldo #variabile privata, grazie a quei due underscore, ma non è effettivamente privata, ma solo per evitare errori accidentali
#però va poi comunque richiamata attraverso una funzione per vedere cosa c'è nel saldo, non basta conto.__saldo perche otterrei un errore;
#invece con conto.get_saldo ottengo il saldo

    def deposito(self,importo):
        self.__saldo += importo

    def get_saldo(self):
        return self.__saldo


#creiamo un conto
conto = ContoBancario(1000)
conto.deposito(500)
print(conto.get_saldo())
conto.deposito(2000)
print(conto.get_saldo())
# print(conto.__saldo)  # ERRORE! L'attributo è privato


class Persona:
    def __init__(self,nome,età):
        self.nome = nome
        self.età = età

    def info(self):
        return f"{self.nome} e ha {self.età} anni"

class Studente(Persona):
    def __init__(self,nome,età,scuola):
        super().__init__(nome,età) #super(). è obbligatorio perchè mi si riferisce alla classe genitore, per definire la classe figlio
        self.scuola = scuola

    def info(self):
        return f"{self.nome}, {self.età} anni , studia a {self.scuola}"

#creiamo un oggetto studente
s1 = Studente("Giovanni", 35, "Liceo Galileo")
print(s1.info())

class Animali:
    def verso(self):
        return "Suono generico"

class Cane(Animali):
    def verso(self):
        return "Bau!"

class Gatto(Animali):
    def verso(self):
        return "Miao!"

animali = [Cane(),Gatto(),Animali()]
for animale in animali:
    print(animale.verso()) #qui riprendo animale perchè è la variabile principale del for

################ ESERCIZI #####################

class Libro:
    def __init__(self,nome,autore,anno):
        self.nome = nome
        self.autore = autore
        self.anno = anno

    def descrizione(self):
        return f"Il titolo del libro è {self.nome}, scritto da {self.autore} nell'anno {self.anno}"

libro = Libro("Il nome della rosa","Umberto Eco",1890)
print(libro.descrizione())

class Rettangolo:
    def __init__(self,base,altezza):
        self.base = base
        self.altezza = altezza
        
    @classmethod  #modifica l'intera classe
    def da_stringa(cls,stringa):
        base , altezza = stringa.split(",")
        return cls(int(base), int(altezza))
    
    @staticmethod #non modifica in alcun modo l'oggetto della classe
    def Area(base,altezza):
        return base*altezza
    
    #@staticmethod SE METTO SELF NON PUò ESSERE UNO STATIC METHOD
    def Area1(self):
        return self.base*self.altezza

r1 = Rettangolo(10,20)
r2 = Rettangolo.da_stringa("15,25")
print(r1.base)
print(r1.altezza)
print(r2.base)
print(r2.altezza)
print("\n")
print(Rettangolo.Area(20,30))
print(Rettangolo.Area(25,25))
print("\n")
print(Rettangolo.da_stringa("15,12").Area1())
print(Rettangolo.Area1(r1))
print(r1.Area1())

class Veicolo:
    def __init__(self,marca,modello,anno):
        self.marca = marca
        self.modello = modello
        self.anno = anno

    def info(self):
        return f"{self.marca}, {self.modello} del {self.anno}"

class Auto(Veicolo):
    def __init__(self,marca,modello,anno,num_porte):
        super().__init__(marca,modello,anno)
        self.num_porte = num_porte

    def info(self):
        return f"{self.marca}, {self.modello} del {self.anno}, con {self.num_porte} porte"

    @classmethod
    def da_stringa(cls,stringa):
        marca,modello,anno,num_porte =  stringa.split(",")
        return cls(marca,modello,int(anno),int(num_porte))

v1 = Veicolo("Subaru","Impreza",1995)
print(v1.info())
print(Auto("Fiat","Panda",1999,4).info())
a1 = Auto("Ford","Fiesta",2022,5)
print(a1.info())

v2 = Auto.da_stringa("Ford,Focus,2001,5")
print(v2.info())

class ContoBancario:
    def __init__(self,saldo):
        self.__saldo = saldo

    def get_saldo(self):
        return self.__saldo

    def prelievo(self,importo):
        self.__saldo -= importo

    def deposito(self,importo):
        self.__saldo += importo

conto = ContoBancario(2500)
print(conto.get_saldo())
prel1 = conto.prelievo(500)
print(conto.get_saldo())
prel2 = conto.prelievo(350)
print(conto.get_saldo())
dep1 = conto.deposito(400)
print(conto.get_saldo())

class Animali:
    def verso(self):
        return f"Suono generico!"

class Cane(Animali):
    def verso(self):
        return f"Bau!!"

class Gatto(Animali):
    def verso(self):
        return f"Miao!!"

animali = [Animali(),Cane(),Gatto()]
for animale in animali:
    print(animale.verso())
        



    
