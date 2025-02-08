# TRAINING CON LA GPU DI GOOGLE COLAB
# https://colab.research.google.com/drive/1Ru5fmUcEQXs09vkkfYFBwTmVpVQiMzoG

# Training del Modello NASNetMobile con classificazione binaria
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import os
import zipfile

# Configurazioni
batch_size = 64
img_height = 224
img_width = 224
epochs = 20

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# Generatori di Dati
data_dir = "/content/BCN20000"
folders_to_move = ["nei_benigni", "nei_maligni"]

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    classes=folders_to_move,
    shuffle=True,
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size // 2,
    class_mode="binary",
    subset="validation",
    classes=folders_to_move,
    shuffle=False,
)

# Modello NASNetMobile
base_model = NASNetMobile(
    weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # Freezing dei pesi pre-addestrati

# Aggiunta di layer personalizzati
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)  # ⚠️ Sigmoid per output binario

# Costruzione del modello finale
model = Model(inputs=base_model.input, outputs=output)

# Compilazione del modello con binary_crossentropy
model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
)

model.summary()


# Callback per la Confusion Matrix
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, val_generator, output_dir="confusion_matrices", class_labels=None
    ):
        super().__init__()
        self.val_generator = val_generator
        self.output_dir = output_dir
        self.class_labels = class_labels or ["Benigno", "Maligno"]
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_generator, verbose=0)
        y_pred = (preds > 0.5).astype(int).flatten()
        y_true = self.val_generator.classes

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
        )
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"epoch_{epoch}_{timestamp}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Confusion matrix saved in: {file_path}")


# Early Stopping
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)
cm_callback = ConfusionMatrixCallback(validation_generator)

# Addestramento del modello
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, cm_callback],
)

# Salvataggio del modello
output_dir = "./output"  # 📂 Directory di output
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "neo_binary_classifier.h5")
model.save(model_path)
print(f"Modello salvato in '{model_path}'")

# Grafici delle Performance
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Salvataggio del grafico finale
performance_plot_path = os.path.join(output_dir, "performance_metrics.png")
plt.savefig(performance_plot_path)
plt.close()
print(f"Grafico delle performance salvato in: {performance_plot_path}")

# Creazione dello ZIP con grafici e modello
zip_filename = os.path.join(output_dir, "training_results.zip")
conf_matrix_dir = "confusion_matrices"

with zipfile.ZipFile(zip_filename, "w") as zipf:
    # Aggiungi il modello
    zipf.write(model_path, arcname="neo_binary_classifier.h5")

    # Aggiungi il grafico delle performance
    zipf.write(performance_plot_path, arcname="performance_metrics.png")

    # Aggiungi tutte le confusion matrix
    for root, _, files_list in os.walk(conf_matrix_dir):
        for file in files_list:
            file_path = os.path.join(root, file)
            arcname = os.path.join("confusion_matrices", file)
            zipf.write(file_path, arcname=arcname)

print(f"ZIP creato: {zip_filename}")
