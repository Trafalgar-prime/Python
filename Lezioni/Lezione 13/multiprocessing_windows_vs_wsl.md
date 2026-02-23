# Multiprocessing in Python: Windows vs WSL/Linux

Questo documento spiega perché il modulo `multiprocessing` in Python si comporta in modo diverso su **Windows** rispetto a **WSL/Linux** e perché l'uso di `if __name__ == "__main__"` è fondamentale.

---

## 🔴 Windows (nativo)

### Metodo di avvio: `spawn`
- Ogni processo figlio **ri-esegue il file Python dall'inizio**
- Se il codice non è protetto da:
```python
if __name__ == "__main__":
```
si verificano problemi come:
- loop infinito
- mancata esecuzione del processo figlio
- stampa solo del processo principale

👉 **Su Windows `if __name__ == "__main__"` è OBBLIGATORIO**

---

## 🟢 Linux / WSL

### Metodo di avvio: `fork`
- Il processo figlio è una **copia del processo padre**
- Il file **non viene rieseguito**
- Il codice funziona anche senza `if __name__ == "__main__"`

👉 Su **WSL** il codice funziona correttamente senza problemi

---

## 🟡 macOS
- Può usare `fork` o `spawn` (dipende dalla versione e dal contesto)
- **Consigliato** usare sempre `if __name__ == "__main__"`

---

## 📊 Confronto rapido

| Ambiente | Metodo | Serve `__main__` |
|--------|--------|------------------|
| Windows | spawn | ✅ SÌ |
| WSL | fork | ❌ No |
| Linux | fork | ❌ No |
| macOS | fork/spawn | ⚠️ Consigliato |

---

## ✅ Best Practice (sempre valida)

Scrivere **sempre**:

```python
if __name__ == "__main__":
```
perché:
- rende il codice portabile
- evita bug nascosti
- è richiesto in produzione, università ed esami
- è una buona pratica professionale

---

## 🧪 Esempio corretto

```python
import multiprocessing

def stampa_messaggio():
    print("Ciao dal processo!")

if __name__ == "__main__":
    p = multiprocessing.Process(target=stampa_messaggio)
    p.start()
    print("Questo è il processo principale")
    p.join()
```

---

📌 **Conclusione**
- Su WSL/Linux il codice funziona senza problemi
- Su Windows nativo **fallisce senza `__main__`**
- Usarlo sempre è la scelta corretta
