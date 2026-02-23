# Python multiprocessing: cosa significa `processes=4`

Questo documento spiega in modo chiaro e rigoroso cosa indica il parametro `processes=4` nel modulo `multiprocessing.Pool` e come usarlo correttamente.

---

## 🔢 Significato di `processes=4`

```python
multiprocessing.Pool(processes=4)
```

👉 **Crea un pool di 4 processi figli indipendenti** che possono lavorare **in parallelo**.

⚠️ Non sono thread:
- ogni processo ha un **PID diverso**
- memoria separata
- un interprete Python distinto

---

## 🧠 Cosa succede internamente

Con:

```python
risultati = pool.map(quadrato, [1, 2, 3, 4, 5, 6])
```

1. Vengono creati **4 processi worker**
2. La lista di input viene suddivisa in **task**
3. I task vengono assegnati ai processi **quando sono disponibili**
4. `map` **garantisce l’ordine dei risultati**

Esempio possibile:

| Processo | Task eseguiti |
|--------|---------------|
| P1 | quadrato(1), quadrato(5) |
| P2 | quadrato(2), quadrato(6) |
| P3 | quadrato(3) |
| P4 | quadrato(4) |

⚠️ L’ordine di esecuzione non è prevedibile  
✅ L’ordine dei risultati è preservato

---

## ⚙️ Relazione con la CPU

- Se hai **≥ 4 core** → vero parallelismo
- Se hai **< 4 core** → time-slicing (overhead)

Metodo consigliato:

```python
import multiprocessing
multiprocessing.cpu_count()
```

Uso tipico:

```python
Pool(processes=multiprocessing.cpu_count())
```

---

## ❌ Cosa NON significa `processes=4`

- ❌ non sono 4 thread
- ❌ non è un processo per ogni elemento
- ❌ non garantisce che tutti lavorino sempre
- ❌ non migliora sempre le prestazioni

---

## 🧪 Valori estremi

### `processes=1`
- nessun parallelismo
- overhead inutile
- più lento di un `for`

### `processes` troppo alto
- saturazione CPU
- overhead di creazione processi
- peggioramento prestazioni

---

## 📌 Regole pratiche

| Scenario | Scelta |
|--------|--------|
| CPU-bound | `cpu_count()` |
| I/O-bound | `threading` |
| Test | 2–4 |
| Produzione | benchmark |

---

## ✅ Best practice (Windows, WSL, Linux)

Usare sempre:

```python
if __name__ == "__main__":
```
per evitare problemi su Windows (metodo `spawn`).

---

## 🧠 Frase chiave da ricordare

> `processes=4` indica **il numero massimo di processi worker attivi in parallelo**

---

## 🧪 Esempio completo corretto

```python
import multiprocessing

def quadrato(n):
    return n * n

if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        risultati = pool.map(quadrato, [1, 2, 3, 4, 5, 6])
        print(risultati)
```

Output:
```text
[1, 4, 9, 16, 25, 36]
```
