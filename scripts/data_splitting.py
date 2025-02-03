import os, sys, shutil
import pandas as pd
import matplotlib.pyplot as plt


def build_structure(base_path, datasets, subfolders):
    """
    Crea una struttura di cartelle per il dataset.

    Args:
        base_path (str): Percorso base dove creare le cartelle.
        datasets (list): Lista di nomi di dataset.
        subfolders (list): Lista di sottocartelle da creare.
    """

    structure = {}

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        structure[dataset] = {}

        for folder in subfolders:
            folder_path = os.path.join(dataset_path, folder)
            print(
                f"Checking if folder exists: {folder_path}"
            )  # Debug: stampa il percorso

            if not os.path.exists(folder_path):
                try:
                    os.makedirs(folder_path)
                    print(f"Folder created: {folder_path}")  # Debug: cartella creata
                except Exception as e:
                    print(
                        f"Errore nella creazione della cartella '{folder_path}'. Dettagli: {e}"
                    )

            structure[dataset][folder] = folder_path
            print(f"Folder path saved: {folder_path}")  # Debug: percorso salvato

    return structure


def process_metadata(directories, datasets):
    """
    Legge il CSV dei metadata, gestisce i valori NaN, copia le immagini nelle directory appropriate
    e conta le categorie.

    Args:
        directories (dict): Dizionario con i percorsi delle directory per metadata, immagini e destinazioni.

    Returns:
        tuple: (conteggi delle categorie, numero totale di immagini)
    """
    counts = {"benign": 0, "malignant": 0, "indeterminate": 0}
    total_images = 0

    for dataset in datasets:
        metadata_path = directories[dataset]["metadata"]
        images_directory = directories[dataset]["images"]

        # Verifica se la cartella images_directory non è vuota
        if (
            os.path.exists(directories[dataset]["nei_benigni"])
            and len(os.listdir(directories[dataset]["nei_benigni"])) > 0
        ):
            continue  # Skip if no images in the folder

        # Percorso del file CSV
        metadata_file = (
            os.path.join(metadata_path, dataset + ".csv") if metadata_path else None
        )

        # Verifica se il file di metadata esiste
        if not metadata_file or not os.path.exists(metadata_file):
            print(f"Errore: file metadata non trovato '{metadata_file}'.")
            sys.exit(1)

        try:
            data = pd.read_csv(metadata_file)
            print(f"Metadata letti correttamente per il dataset {dataset}.")
        except Exception as e:
            print(f"Errore nella lettura del file CSV. Dettagli: {e}")
            sys.exit(1)

        # Sostituisci i NaN nella colonna 'diagnosis_1' con 'indeterminate' e converti in minuscolo
        data["diagnosis_1"] = data["diagnosis_1"].fillna("indeterminate").str.lower()

        # Inizializza i contatori
        total_images += len(data)

        # Itera sulle righe dei metadata
        for index, row in data.iterrows():
            image_id = row.get("isic_id")
            diagnosis_1 = row.get("diagnosis_1")

            # Determina il percorso del file immagine
            image_file = (
                os.path.join(images_directory, f"{image_id}.jpg")
                if images_directory
                else None
            )

            # Determina la directory di destinazione basandosi su diagnosis_1
            if diagnosis_1 == "benign":
                target_directory = directories[dataset]["nei_benigni"]
                counts["benign"] += 1
                print(f"{dataset} - Diagnosi benigna per l'immagine {image_id}")
            elif diagnosis_1 == "malignant":
                target_directory = directories[dataset]["nei_maligni"]
                counts["malignant"] += 1
                print(f"{dataset} - Diagnosi maligna per l'immagine {image_id}")
            elif diagnosis_1 == "indeterminate":
                target_directory = directories[dataset]["nei_indeterminati"]
                counts["indeterminate"] += 1
                print(f"{dataset} - Diagnosi indeterminata per l'immagine {image_id}")
            else:
                target_directory = directories[dataset]["nei_indeterminati"]
                counts["indeterminate"] += 1
                print(
                    f"{dataset} - Diagnosi non riconosciuta '{diagnosis_1}' per l'immagine {image_id} - impostata a 'indeterminate'."
                )

            # Verifica se la directory di destinazione esiste e non è None
            if target_directory and os.path.exists(target_directory):
                # Verifica se il file immagine esiste
                if image_file and os.path.exists(image_file):
                    try:
                        # Copia il file immagine nella directory di destinazione
                        shutil.copy(image_file, target_directory)
                    except Exception as e:
                        print(
                            f"Errore nella copia del file '{image_file}'. Dettagli: {e}"
                        )
                else:
                    print(f"Errore: file immagine non trovato '{image_file}'.")
            else:
                print(
                    f"Errore: directory di destinazione '{target_directory}' non valida per l'immagine {image_id}."
                )

    return counts, total_images


def print_info(datasets, directories, subfolders):
    """Stampa informazioni sul dataset e sulle sottocartelle."""
    for dataset in datasets:
        print(
            f"Totali in {dataset}: ",
            len([f for f in os.listdir(directories[dataset]["images"])]),
        )
        for subfolder in subfolders[:3]:
            print(
                f"Totali in {subfolder} per {dataset}: ",
                len([f for f in os.listdir(directories[dataset][subfolder])]),
            )


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


def create_pie_chart(counts, total_images, grafico_pie_path, exclude_indeterminate):
    """
    Crea un grafico a torta delle percentuali per le categorie 'benign', 'malignant' e 'indeterminate'.

    Args:
        counts (dict): Un dizionario con il conteggio delle immagini per ciascuna categoria.
        total_images (int): Il numero totale di immagini.
        grafico_pie_path (str): Il percorso per salvare il grafico.
        exclude_indeterminate (bool): Se True, esclude la categoria 'indeterminate' nel grafico.
    """
    # Calcola le percentuali
    percentuali = [
        (counts["benign"] / total_images) * 100,
        (counts["malignant"] / total_images) * 100,
    ]

    labels = ["Benign", "Malignant"]  # Etichette per le prime due categorie

    # Aggiungi la percentuale per 'indeterminate' se necessario
    if not exclude_indeterminate:
        percentuali.append((counts["indeterminate"] / total_images) * 100)
        labels.append("Indeterminate")

    # Crea il grafico a torta - verde, rosso, (giallo)
    plt.figure(figsize=(7, 7))  # Imposta la dimensione del grafico
    wedges, texts, autotexts = plt.pie(
        percentuali,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=(
            ["#4CAF50", "#F44336", "#FFC107"]
            if not exclude_indeterminate
            else ["#4CAF50", "#F44336"]
        ),
        wedgeprops={"edgecolor": "white"},
    )

    # Migliora l'aspetto delle etichette
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Salva il grafico
    plt.axis("equal")  # Mantieni il grafico a torta circolare
    plt.title("Distribuzione delle Diagnosi", fontsize=16)
    plt.savefig(grafico_pie_path, bbox_inches="tight")
    print(f"Grafico a barre salvato in '{grafico_pie_path}'.")
    plt.close()


def create_charts(directories, datasets):
    for dataset in datasets:
        grafico_bar_path = os.path.join(
            directories[dataset]["diagrams"], dataset + "_diagram_bar.png"
        )
        grafico_pie_path = os.path.join(
            directories[dataset]["diagrams"], dataset + "_diagram_pie.png"
        )
        grafico_pie_path_with_indeterminate = os.path.join(
            directories[dataset]["diagrams"],
            dataset + "_diagram_pie_with_indeterminate.png",
        )

        counts = {
            "benign": len([f for f in os.listdir(directories[dataset]["nei_benigni"])]),
            "malignant": len(
                [f for f in os.listdir(directories[dataset]["nei_maligni"])]
            ),
            "indeterminate": len(
                [f for f in os.listdir(directories[dataset]["nei_indeterminati"])]
            ),
        }

        total_images = len([f for f in os.listdir(directories[dataset]["images"])])

        create_bar_chart(counts, total_images, grafico_bar_path)
        create_pie_chart(counts, total_images, grafico_pie_path, True)
        create_pie_chart(
            counts, total_images, grafico_pie_path_with_indeterminate, False
        )


def main():

    # percorso base
    my_path = os.path.join(os.getcwd(), "dataset")

    # dataset
    datasets = ["BCN20000"]

    # sottocartelle
    subfolders = [
        "nei_benigni",
        "nei_maligni",
        "nei_indeterminati",
        "images",
        "metadata",
        "diagrams",
    ]

    directories = build_structure(my_path, datasets, subfolders)
    process_metadata(directories=directories, datasets=datasets)

    print_info(datasets, directories, subfolders[:3])
    create_charts(directories, datasets)


if __name__ == "__main__":
    main()
