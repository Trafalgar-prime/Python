from mio_pacchetto.matematica import somma

print(somma(3,4))


import mio_pacchetto.matematica as m

print(m.somma(3,4))

from mio_pacchetto import matematica

print(matematica.somma(3,9))
print("\n")


from mio_pacchetto import funzioni as f #cosi ho tirato fuori dalla cartella solo il sotto pacchetto di cui ho bisogno
from mio_pacchetto import * #in questo modo riesco a tirare fuori anche le funzioni ma perhcè l'ho specificate nell'__init__.py, e tiro fuori tutti i sotto pacchetti e tutte le funzioni specificate
print(quadrato(5)) #grazie all'asterisco della riga precedente tiro fuori le funzioni senza richiamare il sotto pacchetto


print(f.quadrato(3))
print(f.quadrato(4))

import requests

response = requests.get("https://api.github.com")
print(response.status_code)  # Output: 200
