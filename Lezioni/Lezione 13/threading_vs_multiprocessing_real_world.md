# Threading e Multiprocessing: a cosa servono davvero nella programmazione quotidiana

Questo documento spiega **quando e perché** usare `threading` e `multiprocessing` nella programmazione reale, andando oltre la teoria.

---

## 🎯 Il problema reale

Un programma sequenziale fa **una cosa alla volta**.

Nella pratica, però, i programmi:
- aspettano reti
- leggono/scrivono file
- eseguono calcoli pesanti
- devono restare reattivi per l’utente

👉 **Thread e processi servono a non restare fermi.**

---

## 🧵 THREADING — uso reale

### Idea chiave
> Usa i thread quando il programma **aspetta qualcosa** (I/O).

### Esempi concreti
- Download multipli
- Web scraping
- Server web (più utenti insieme)
- Bot Telegram / Discord
- Applicazioni con interfaccia grafica
- Logging in background

### Perché funzionano
- mentre un thread aspetta I/O
- un altro può lavorare
- la CPU non resta inutilizzata

### Limite in Python
❌ **GIL (Global Interpreter Lock)**  
- un solo thread esegue bytecode Python alla volta
- pessimo per calcoli pesanti

### Frase chiave
> **Thread = I/O bound**

---

## 🧨 MULTIPROCESSING — uso reale

### Idea chiave
> Usa i processi quando devi sfruttare **tutta la CPU**.

### Esempi concreti
- elaborazione immagini / video
- machine learning e deep learning
- simulazioni fisiche
- analisi dati pesanti
- compressione e hashing
- giochi (AI, pathfinding)

### Perché servono
- ogni processo ha un core CPU
- niente GIL
- vero parallelismo

### Controindicazioni
- overhead elevato
- memoria duplicata
- comunicazione più lenta

### Frase chiave
> **Multiprocessing = CPU bound**

---

## ⚔️ Thread vs Multiprocessing (pratico)

| Caso reale | Tecnologia |
|----------|------------|
| Scaricare 100 URL | Thread |
| Leggere molti file | Thread |
| API / Web server | Thread |
| Elaborare immagini | Process |
| Machine Learning | Process |
| Calcoli matematici | Process |

---

## 🤯 Perché non usare sempre multiprocessing?
- costa più risorse
- codice più complesso
- debugging difficile
- spreco di RAM

👉 Usalo **solo quando serve davvero**.

---

## 🧠 Regola d’oro

1. Il programma **aspetta** → Thread  
2. Il programma **calcola** → Process  
3. Entrambi → combinazione (avanzato)

---

## 🔥 Caso reale molto comune

Un server moderno:
- usa **thread** per gestire le richieste
- usa **processi** per elaborazioni pesanti

Esempi reali:
- Django
- FastAPI
- Gunicorn

---

## 📌 Conclusione

> Thread = mantenere il programma reattivo  
> Processi = sfruttare tutta la CPU

Questa distinzione è fondamentale nella programmazione professionale.
