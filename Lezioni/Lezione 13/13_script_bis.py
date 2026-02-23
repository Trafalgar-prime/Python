import multiprocessing
import time

def stampa_messaggio():
    time.sleep(1)  # Simula un processo più lungo
    print("\nCiao dal processo!", flush=True)

if __name__ == "__main__":
    mio_processo = multiprocessing.Process(target=stampa_messaggio)
    mio_processo.start()

    print("\nQuesto è il processo principale", flush=True)
    print(mio_processo.is_alive())  # Dovrebbe restituire True

    mio_processo.join()
