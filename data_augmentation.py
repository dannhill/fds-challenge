import json
import itertools
import copy
import os
import multiprocessing
from tqdm import tqdm

# --- FUNZIONE WORKER ---
# Ora prende una coda come argomento per inviare i risultati.
def worker_process(input_queue, output_queue):
    """
    Estrae righe dalla coda di input, le elabora e mette i risultati
    (blocchi di testo) nella coda di output.
    """
    for line in iter(input_queue.get, None): # Loop finché non riceve il segnale 'None'
        try:
            original_data = json.loads(line)
            p1_team = original_data.get('p1_team_details')

            if p1_team and len(p1_team) == 6:
                # Genera le 720 permutazioni e le unisce in un unico blocco di testo
                # L'uso di memoria è limitato a questo blocco temporaneo.
                augmented_lines = []
                base_data = copy.deepcopy(original_data)
                for permuted_team_tuple in itertools.permutations(p1_team):
                    base_data['p1_team_details'] = list(permuted_team_tuple)
                    augmented_lines.append(json.dumps(base_data))
                
                output_chunk = '\n'.join(augmented_lines) + '\n'
                output_queue.put(output_chunk)
            else:
                # Se non aumentata, invia la riga originale
                output_queue.put(line)
        except (json.JSONDecodeError, AttributeError):
            # In caso di errore, invia comunque la riga originale
            output_queue.put(line)

# --- FUNZIONE SCRITTORE ---
def writer_process(output_queue, output_file_path, total_lines, num_workers):
    """
    Estrae blocchi di testo dalla coda di output e li scrive su file.
    Gestisce anche la barra di avanzamento.
    """
    workers_done = 0
    with open(output_file_path, 'w', encoding='utf-8') as f, \
         tqdm(total=total_lines, desc="Augmenting Data") as pbar:
        while workers_done < num_workers:
            item = output_queue.get()
            if item is None: # Segnale di fine da un worker
                workers_done += 1
            else:
                f.write(item)
                # L'aggiornamento della pbar è un'approssimazione,
                # ma dà un'idea del progresso.
                if len(item) > 1000: # Heuristica: è un blocco aumentato
                    pbar.update(1)
                else: # È una singola riga non aumentata
                    pbar.update(1)

def main():
    # --- Configuration ---
    dataset_dir = './fds-challenge-dataset/'
    input_file_name = 'train.jsonl'
    output_file_name = 'train_augmented_final.jsonl'

    input_file_path = os.path.join(dataset_dir, input_file_name)
    output_file_path = os.path.join(dataset_dir, output_file_name)

    # Lasciamo un core libero per il processo scrittore e il sistema operativo.
    # Questo evita contese tra i worker (CPU-bound) e lo scrittore (I/O-bound).
    NUM_WORKERS = max(1, os.cpu_count() - 1)
    
    print(f"Starting data augmentation with {NUM_WORKERS} workers and 1 writer process...")
    print(f"Input file:  '{input_file_path}'")
    print(f"Output file: '{output_file_path}'")

    try:
        # Contiamo le righe per la progress bar
        with open(input_file_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"Found {num_lines} lines to process.")

        # Creazione delle code di comunicazione
        # Impostiamo una dimensione massima per evitare che la coda di output
        # cresca indefinitamente se lo scrittore è più lento dei worker.
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue(maxsize=NUM_WORKERS * 4)

        # Avvio del processo scrittore
        writer = multiprocessing.Process(
            target=writer_process,
            args=(output_queue, output_file_path, num_lines, NUM_WORKERS)
        )
        writer.start()

        # Avvio dei processi worker
        workers = []
        for _ in range(NUM_WORKERS):
            worker = multiprocessing.Process(
                target=worker_process,
                args=(input_queue, output_queue)
            )
            worker.start()
            workers.append(worker)

        # Il processo principale ora riempie la coda di input
        print("Distributing work to workers...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                input_queue.put(line)

        # Invia il segnale di fine ai worker
        print("All work distributed. Sending termination signals...")
        for _ in range(NUM_WORKERS):
            input_queue.put(None)

        # Attendi la terminazione di tutti i processi
        for worker in workers:
            worker.join()
        
        # Invia il segnale di fine allo scrittore (dopo che i worker sono terminati)
        output_queue.put(None) # Questo segnale è per il caso in cui l'ultimo worker finisca
                               # ma lo scrittore sia ancora in attesa.
                               # La logica interna dello scrittore gestisce N segnali.
        writer.join()

        print("\n---------------------------------")
        print("Data augmentation complete!")
        print(f"Augmented data has been saved to '{output_file_path}'.")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"The input file was not found at '{input_file_path}'.")

if __name__ == '__main__':
    main()