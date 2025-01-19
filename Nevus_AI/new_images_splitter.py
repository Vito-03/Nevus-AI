import os
import sys
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import logging


def initialize_directories(base_path, directories, directory_to_skip):
    """
    Crea le directory di output se non esistono e le svuota.
    La directory 'bcn_images' non viene creata né svuotata.

    Args:
        base_path (str): Percorso base dove si trovano le directory.
        directories (dict): Dizionario con i nomi delle directory.

    Returns:
        dict: Dizionario aggiornato con i percorsi completi delle directory.
    """
    try:
        for key, path in directories.items():
            # Controlla se il nome della directory è 'bcn_images'
            if os.path.basename(path) == directory_to_skip:
                logging.info(f"Directory '{path}' è 'bcn_images'. Skippata.")
                continue  # Salta la creazione e lo svuotamento di 'bcn_images'

            # Crea la directory se non esiste
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory '{path}' creata o già esistente.")

        # Svuota le directory di output, escludendo 'bcn_images'
        for key, path in directories.items():
            if os.path.basename(path) == "bcn_images":
                logging.info(
                    f"Directory '{path}' è 'bcn_images'. Skippata dallo svuotamento."
                )
                continue  # Salta lo svuotamento di 'bcn_images'

            try:
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        logging.debug(f"File '{file_path}' rimosso.")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logging.debug(f"Directory '{file_path}' rimossa.")
                logging.info(f"Directory '{path}' svuotata.")
            except Exception as e:
                logging.error(
                    f"Errore nello svuotare la directory '{path}'. Dettagli: {e}"
                )
                sys.exit(1)
    except Exception as e:
        logging.error(
            f"Errore durante l'inizializzazione delle directory. Dettagli: {e}"
        )
        sys.exit(1)

    return directories


def process_metadata(metadata_path, images_directory, directories):
    """
    Legge il CSV dei metadata, gestisce i valori NaN, copia le immagini nelle directory appropriate
    e conta le categorie.

    Args:
        metadata_path (str): Percorso del file CSV dei metadata.
        images_directory (str): Percorso della directory contenente le immagini originali.
        directories (dict): Dizionario con i percorsi delle directory di destinazione.

    Returns:
        tuple: (conteggi delle categorie, numero totale di immagini)
    """
    try:
        data = pd.read_csv(metadata_path)
        print("Metadata letti correttamente.")
    except Exception as e:
        print(f"Errore nella lettura del file CSV. Dettagli: {e}")
        sys.exit(1)

    # Sostituisci i NaN nella colonna 'diagnosis_1' con 'indeterminate' e converti in minuscolo
    data["diagnosis_1"] = data["diagnosis_1"].fillna("indeterminate").str.lower()

    # Inizializza i contatori
    total_images = len(data)
    counts = {"benign": 0, "malignant": 0, "indeterminate": 0}

    # Itera sulle righe dei metadata
    for index, row in data.iterrows():
        image_id = row.get("isic_id")
        diagnosis_1 = row.get("diagnosis_1")

        # Determina il percorso del file immagine
        image_file = os.path.join(images_directory, f"{image_id}.jpg")

        # Determina la directory di destinazione basandosi su diagnosis_1
        if diagnosis_1 == "benign":
            target_directory = directories["benign"]
            counts["benign"] += 1
        elif diagnosis_1 == "malignant":
            target_directory = directories["malignant"]
            counts["malignant"] += 1
        elif diagnosis_1 == "indeterminate":
            target_directory = directories["indeterminate"]
            counts["indeterminate"] += 1
        else:
            # Gestione di diagnosi non previste
            target_directory = directories["indeterminate"]
            counts["indeterminate"] += 1
            print(
                f"Diagnosi non riconosciuta '{diagnosis_1}' per l'immagine {image_id} - impostata a 'indeterminate'."
            )

        # Verifica se il file immagine esiste
        if os.path.exists(image_file):
            try:
                # Copia il file immagine nella directory di destinazione
                shutil.copy(image_file, target_directory)
            except Exception as e:
                print(f"Errore nella copia del file '{image_file}'. Dettagli: {e}")
        else:
            print(f"Errore: file immagine non trovato '{image_file}'.")

    return counts, total_images


def create_bar_chart(counts, total_images, output_path):
    """
    Crea e salva un grafico a barre che mostra il numero di immagini per categoria.

    Args:
        counts (dict): Conteggi delle categorie di diagnosi.
        total_images (int): Numero totale di immagini.
        output_path (str): Percorso dove salvare il grafico a barre.
    """
    categorie = ["Benigni", "Maligni", "Indeterminati"]
    conteggi = [counts["benign"], counts["malignant"], counts["indeterminate"]]
    colori = ["#4CAF50", "#F44336", "#FFC107"]  # Verde, Rosso, Giallo

    plt.figure(figsize=(8, 6))
    plt.bar(categorie, conteggi, color=colori)
    plt.title("Distribuzione delle Diagnosi delle Immagini")
    plt.xlabel("Categoria")
    plt.ylabel("Numero di Immagini")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Aggiungi le etichette sopra le barre
    for i, conteggio in enumerate(conteggi):
        plt.text(
            i,
            conteggio + total_images * 0.005,
            str(conteggio),
            ha="center",
            va="bottom",
        )

    try:
        plt.savefig(output_path)
        print(f"Grafico a barre salvato in '{output_path}'.")
    except Exception as e:
        print(f"Errore nel salvataggio del grafico a barre. Dettagli: {e}")

    plt.close()


def create_pie_chart(counts, total_images, output_path):
    """
    Crea e salva un grafico a torta che mostra la percentuale delle categorie di diagnosi.

    Args:
        counts (dict): Conteggi delle categorie di diagnosi.
        total_images (int): Numero totale di immagini.
        output_path (str): Percorso dove salvare il grafico a torta.
    """
    percentuali = [
        (counts["benign"] / total_images) * 100,
        (counts["malignant"] / total_images) * 100,
        # (counts["indeterminate"] / total_images) * 100,
    ]

    # etichette = ["Benigni", "Maligni", "Indeterminati"]
    etichette = ["Benigni", "Maligni"]
    # colori = ["#4CAF50", "#F44336", "#FFC107"]  # Verde, Rosso, Giallo
    colori = ["#4CAF50", "#F44336"]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        percentuali,
        labels=etichette,
        colors=colori,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 12},
    )

    plt.title("Distribuzione delle Diagnosi delle Immagini (%)")

    # Migliora la leggibilità delle percentuali
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")

    try:
        plt.savefig(output_path)
        print(f"Grafico a torta salvato in '{output_path}'.")
    except Exception as e:
        print(f"Errore nel salvataggio del grafico a torta. Dettagli: {e}")

    plt.close()


def main():
    # Ottieni la directory dello script
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Percorso base
    my_path = "/Users/vito/Desktop/Nevus_AI/isic_bcn20000/"

    # Percorsi delle directory di output
    directories = {
        "benign": os.path.join(my_path, "images/bcn_benigni"),
        "malignant": os.path.join(my_path, "images/bcn_maligni"),
        "indeterminate": os.path.join(my_path, "images/bcn_indeterminati"),
    }

    # Percorso della directory delle immagini originali (non modificabile)
    images_directory = os.path.join(my_path, "images/bcn_images")

    # Percorso del file CSV dei metadata
    metadata_path = os.path.join(my_path, "metadata/bcn20000_metadata_2024-12-27.csv")

    # Inizializza e svuota le directory di output
    directories = initialize_directories(
        base_path=my_path, directories=directories, directory_to_skip="bcn_images"
    )

    # Processa i metadata e copia le immagini nelle directory di output
    counts, total_images = process_metadata(
        metadata_path, images_directory, directories
    )
    print(f"Totale immagini: {total_images}")
    print(f"Benigni: {counts['benign']}")
    print(f"Maligni: {counts['malignant']}")
    # print(f"Indeterminati: {counts['indeterminate']}")

    # Percorsi per i grafici
    grafico_bar_path = os.path.join(script_directory, "isic_bcn_diagram_bar.png")
    grafico_pie_path = os.path.join(script_directory, "isic_bcn_diagram_pie.png")

    # Crea i grafici
    create_bar_chart(counts, total_images, grafico_bar_path)
    create_pie_chart(counts, total_images, grafico_pie_path)


if __name__ == "__main__":
    main()
