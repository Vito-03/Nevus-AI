# Nevus-AI

## Introduzione
Nevus AI è un progetto di **Machine Learning** applicato alla **medicina di precisione** per la **diagnosi precoce del melanoma**. L'obiettivo è sviluppare un modello di **Deep Learning** capace di classificare immagini di nei in **benigni** o **maligni**, fornendo supporto ai professionisti del settore sanitario.

## Autori
- **Michele Antonio Annunziata** - [micheleantonioannunziata](https://github.com/micheleantonioannunziata)
- **Vito Vernellati** - [Vito-03](https://github.com/Vito-03)

## Struttura della Repository
La repository è organizzata come segue:
```
Nevus-AI/
|-- dataset/                  # contiene i dataset usati nel progetto
|   |-- HAM10000/
|   |-- BCN20000/
|   |   |-- images/           # cartella con solo immagini (da creare quando si utilizza il dataset per il training)
|   |   |-- metadata/         # cartella con file metadata
|   |   |   |-- nomedataset.csv # metadata rinominato
|
|-- docs/                     # cocumentazione (pdf) e presentazione (pptx) del progetto
|
|-- scripts/                   # script di supporto
|   |-- data_splitting.py      # script per la suddivisione dei dataset
|   |-- training.py            # script per il training del modello
|   |-- ui.py                  # script per l'interfaccia utente
|
|-- README.md                  # questo file
|-- requirements.txt           # dipendenze del progetto
```

## Dataset
Il dataset utilizzato è stato costruito partendo dalle collezioni **HAM10000** e **BCN20000** disponibili nell'archivio ISIC. Per migliorare le prestazioni del modello, sono state applicate tecniche di **data augmentation** e bilanciamento delle classi.

### Download del Dataset
Il dataset può essere scaricato dal seguente link:
[ISIC Archive - BCN20000](https://api.isic-archive.com/collections/249/)

### Utilizzare altri dataset
Dopo aver scaricato il dataset:
1. **Eliminare i file inutili**, lasciando solo le immagini e il csv relativo a *metadata*.
2. **Creare una cartella** con il nome del dataset specifico nella cartella `dataset`.
3. **Inserire** le immagini nella cartella `dataset/nomeDataset/images/`.
4. **Spostare il file metadata.csv** dalla cartella scaricata a `dataset/nomeDataset/metadata/`, rinominandolo come `nomedataset.csv`.

## Modello di Machine Learning
Abbiamo adottato una **CNN (Convolutional Neural Network)** per la classificazione delle immagini. In particolare:
- Abbiamo utilizzato il **transfer learning** basato su **NASNetMobile**.
- Il modello è stato addestrato su **20 epoche** con **early stopping** per evitare overfitting.
- Abbiamo valutato le performance con metriche come **accuracy, precision, recall e F1-score**.

## Installazione e Uso
### 1. Requisiti
Assicurati di avere **Python 3.8+** installato. Poi installa le dipendenze:
```bash
pip install -r requirements.txt
```

### 2. Addestramento del modello
Per addestrare il modello eseguire:
```bash
python scripts/training.py
```

### 3. Avvio dell'interfaccia utente
L'interfaccia grafica è stata sviluppata con **Streamlit**. Per avviarla:
```bash
streamlit run scripts/ui.py
```

## Risultati
Dopo l'addestramento, il modello ha ottenuto:
- **Precisione finale:** 75%
- **Recall:** 71%
- **F1-Score:** 73%
- **Accuracy:** 74%

## Disclaimer
I risultati forniti dal modello **NON sostituiscono una diagnosi medica**. Il modello deve essere usato solo come supporto alla diagnosi.

## Possibili Miglioramenti
- **Espansione del dataset** con immagini più diversificate.
- **Ottimizzazione degli iperparametri** per migliorare la precisione.
- **Integrazione di dati testuali** per aumentare la spiegabilità del modello.

## Contributi
Se vuoi contribuire al progetto, sentiti libero di **fare una pull request** o di aprire una **issue**!

## License
Questo progetto è distribuito sotto licenza MIT.

---
**Repository GitHub**: [Nevus AI](https://github.com/Vito-03/Nevus-AI)