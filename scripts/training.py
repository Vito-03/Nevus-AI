# TRAINING CON LA GPU DI GOOGLE COLAB
# https://colab.research.google.com/drive/1Ru5fmUcEQXs09vkkfYFBwTmVpVQiMzoG


import numpy as np  # Libreria per operazioni numeriche e matriciali
import matplotlib.pyplot as plt  # Per la visualizzazione di grafici
import seaborn as sns  # Per grafici statistici avanzati come la confusion matrix
from sklearn.metrics import confusion_matrix  # Per calcolare la confusion matrix
import tensorflow as tf  # Framework principale per il deep learning
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # Per il data augmentation delle immagini
from tensorflow.keras.applications import (
    NASNetMobile,
)  # Modello pre-addestrato NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout  # Layer per la rete neurale
from tensorflow.keras.models import Model  # Per definire modelli personalizzati
from tensorflow.keras.optimizers import (
    Adam,
)  # Ottimizzatore Adam per la discesa del gradiente
from tensorflow.keras.callbacks import (
    EarlyStopping,
)  # Callback per fermare l'addestramento in anticipo se necessario
from datetime import (
    datetime,
)  # Per gestire date e orari (utile per i salvataggi dei file)
import os  # Per la gestione dei file e directory sul filesystem

# Configurazione delle cartelle
data_dir = "/content/BCN20000"
folders_to_move = ["nei_benigni", "nei_maligni"]

batch_size = 64  # Numero di immagini processate contemporaneamente in ogni batch
img_height = 224  # Altezza delle immagini (input per il modello)
img_width = 224  # Larghezza delle immagini (input per il modello)
epochs = 20  # Numero massimo di epoche di addestramento


datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizza i pixel tra 0 e 1 (prima erano tra 0 e 255)
    rotation_range=30,  # Ruota casualmente le immagini fino a 30 gradi
    width_shift_range=0.2,  # Sposta casualmente l'immagine in orizzontale fino al 20%
    height_shift_range=0.2,  # Sposta casualmente l'immagine in verticale fino al 20%
    shear_range=0.2,  # Applica trasformazioni di taglio (shearing) fino al 20%
    zoom_range=0.2,  # Applica uno zoom casuale fino al 20%
    horizontal_flip=True,  # Ribalta casualmente le immagini orizzontalmente
    validation_split=0.2,  # Suddivide il dataset: 80% per il training e 20% per la validazione
)


train_generator = datagen.flow_from_directory(
    data_dir,  # Directory principale del dataset
    target_size=(
        img_height,
        img_width,
    ),  # Ridimensiona le immagini alla dimensione specificata
    batch_size=batch_size,  # Batch size definito sopra
    class_mode="categorical",  # Output in formato categorico (one-hot encoding)
    subset="training",  # Utilizza l'80% dei dati per il training
    classes=folders_to_move,  # Classi specifiche da includere
    shuffle=True,  # Mescola i dati ad ogni epoca per migliorare la generalizzazione
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size // 2,  # Batch più piccolo per la validazione (più efficiente)
    class_mode="categorical",  # Anche qui categoriale
    subset="validation",  # Utilizza il 20% dei dati per la validazione
    classes=folders_to_move,
    shuffle=False,  # Non mescola per la validazione (importante per la Confusion Matrix)
)


# Carica NASNetMobile pre-addestrato su ImageNet senza la parte finale (include_top=False)
base_model = NASNetMobile(
    weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # Congela i pesi del modello pre-addestrato per non modificarli durante l'addestramento


x = Flatten()(base_model.output)  # Appiattisce l'output del NASNetMobile
x = Dense(512, activation="relu")(
    x
)  # Layer fully-connected con 512 neuroni e funzione di attivazione ReLU
x = Dropout(0.5)(x)  # Dropout del 50% per ridurre l'overfitting
output = Dense(2, activation="softmax")(
    x
)  # Layer di output per classificazione binaria (2 classi) con softmax


model = Model(inputs=base_model.input, outputs=output)  # Crea il modello finale
model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),  # Ottimizzatore Adam con learning rate di 0.0001
    loss="categorical_crossentropy",  # Funzione di perdita per classificazione multi-classe
    metrics=["accuracy"],  # Monitora l'accuratezza durante l'addestramento
)
model.summary()  # Stampa un riepilogo della struttura del modello


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, val_generator, output_dir="confusion_matrices", class_labels=None
    ):
        super().__init__()
        self.val_generator = val_generator
        self.output_dir = output_dir
        self.class_labels = class_labels or ["Benigno", "Maligno"]
        os.makedirs(
            output_dir, exist_ok=True
        )  # Crea la cartella per salvare le confusion matrix se non esiste

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(
            self.val_generator, verbose=0
        )  # Predizioni sul set di validazione
        y_pred = np.argmax(preds, axis=1)  # Classi predette (con maggiore probabilità)
        y_true = self.val_generator.classes  # Classi reali

        cm = confusion_matrix(y_true, y_pred)  # Calcola la confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",  # Heatmap con annotazioni
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
        )
        plt.title(f"Confusion Matrix - Epoca {epoch}")
        plt.xlabel("Predetto")
        plt.ylabel("Reale")
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Aggiunge timestamp per ogni file salvato
        plt.savefig(
            os.path.join(self.output_dir, f"epoch_{epoch}_{timestamp}.png")
        )  # Salva la confusion matrix
        plt.close()
        print(
            f"[INFO] Confusion matrix salvata in: confusion_matrices/epoch_{epoch}_{timestamp}.png"
        )


early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)
# Ferma l'addestramento se la validazione non migliora dopo 5 epoche consecutive

cm_callback = ConfusionMatrixCallback(
    validation_generator
)  # Callback per la confusion matrix


history = model.fit(
    train_generator,  # Dati di training
    epochs=epochs,  # Numero massimo di epoche
    validation_data=validation_generator,  # Dati di validazione
    callbacks=[
        early_stopping,
        cm_callback,
    ],  # Callback per early stopping e confusion matrix
)


model.save("/content/neo_classifier.h5")  # Salva il modello in formato HDF5
print("✅ Modello salvato in '/content/neo_classifier.h5'")  # Messaggio di conferma


plt.figure(figsize=(12, 5))

# Grafico della Loss (perdita)
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")  # Loss durante il training
plt.plot(
    history.history["val_loss"], label="Validation Loss"
)  # Loss durante la validazione
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# Grafico dell'Accuracy
plt.subplot(1, 2, 2)
plt.plot(
    history.history["accuracy"], label="Training Accuracy"
)  # Accuratezza sul training
plt.plot(
    history.history["val_accuracy"], label="Validation Accuracy"
)  # Accuratezza sulla validazione
plt.xlabel("Epoca")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()  # Mostra entrambi i grafici
