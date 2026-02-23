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
