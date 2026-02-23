# Python - Errore nel Passaggio di Argomenti ai Thread

Quando creiamo un **thread in Python**, dobbiamo passare correttamente la funzione e i suoi argomenti.  
Un errore comune è **eseguire la funzione invece di passarla come riferimento**.

---

## **📌 Errore Comune: Eseguire la Funzione Prima di Creare il Thread**
```python
import threading

def preleva(quantità):
    global saldo
    if saldo >= quantità:
        saldo -= quantità
        print(f"Prelevati {quantità}, saldo rimanente: {saldo}")
    else:
        print("Saldo insufficiente")

saldo = 100

thread1 = threading.Thread(target=preleva, args=(50,))
thread2 = threading.Thread(target=preleva(80))  # ❌ ERRORE!
```

🔴 **Errore:** Qui `preleva(80)` viene eseguita subito e il risultato (`None`) viene passato come `target`.  
✅ **Correzione:** Deve essere `args=(80,)` come in `thread1`:
```python
thread2 = threading.Thread(target=preleva, args=(80,))
```

---

## **📌 Cosa Succede nel Codice Errato?**
```python
thread1 = threading.Thread(target=preleva, args=(50,))
thread2 = threading.Thread(target=preleva(80))  # ❌ ERRORE
```

🔹 **Ecco cosa succede passo dopo passo:**  
1. `preleva(80)` viene eseguita immediatamente nel thread principale **prima di avviare `thread1`**.  
2. Il saldo scende da **100 → 20**.  
3. Ora `thread1` viene creato e prova a prelevare **50**, ma il saldo è già **20**, quindi ottieni *"Saldo insufficiente"*.  

🔹 **Output risultante:**
```
Prelevati 80, saldo rimanente: 20
Saldo insufficiente
```

✅ **Correzione: Passare correttamente la funzione al thread**
```python
thread1 = threading.Thread(target=preleva, args=(50,))
thread2 = threading.Thread(target=preleva, args=(80,))  # ✅ Ora il thread funziona correttamente

thread1.start()
thread2.start()

thread1.join()
thread2.join()
```

🔹 **Ora il prelievo avverrà in ordine casuale a seconda della velocità di esecuzione dei thread.**  

---

## **📌 Conclusione**
- Se scrivi **`target=preleva(80)`**, Python esegue subito `preleva(80)` e passa `None` al thread.  
- Per passare un argomento a un thread, usa **`args=(80,)`** correttamente.  
- Con i thread, **l'ordine di esecuzione è imprevedibile**, dipende dal sistema operativo e dal gestore dei thread.  

✅ **Ora hai capito l'errore e come evitarlo!** 🚀

