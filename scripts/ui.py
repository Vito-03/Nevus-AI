import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# streamlit run ui.py

# Caricamento del modello pre-addestrato
MODEL = ""
MODEL_PATH = os.path.join(os.getcwd(), MODEL)
print(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# Classi del modello
CLASS_NAMES = ["Benigno", "Maligno"]


# Funzione per la previsione
def predict_image(image):
    img = image.resize((224, 224))  # Dimensioni compatibili con NASNetMobile
    img_array = np.array(img) / 255.0  # Normalizzazione dei pixel
    img_array = np.expand_dims(img_array, axis=0)  # Aggiunta della dimensione batch

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    class_index = np.argmax(prediction)
    label = CLASS_NAMES[class_index]

    # Probabilità per entrambe le classi
    probabilities = {
        CLASS_NAMES[i]: float(prediction[0][i]) * 100 for i in range(len(CLASS_NAMES))
    }

    return label, confidence, probabilities


# Interfaccia Streamlit
st.set_page_config(page_title="Classificatore di Melanomi", page_icon="🩺")

st.title("🩺 Classificatore di Melanomi (Benigno o Maligno)")
st.write(
    "Carica un'immagine di un neo per ottenere una previsione basata sul modello addestrato."
)

# Caricamento dell'immagine
uploaded_file = st.file_uploader(
    "📤 Carica un'immagine (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Immagine caricata", use_column_width=True)

    if st.button("Classifica"):
        with st.spinner("🔍 Analisi in corso..."):
            label, confidence, probabilities = predict_image(image)

        # ✅ Esito Finale
        st.markdown(f"## 📝 **Esito dell'Analisi:**")
        if label == "Benigno":
            st.success(
                f"✅ Il neo analizzato è **{label.upper()}** con una precisione del **{confidence:.2f}%**."
            )
        else:
            st.error(
                f"⚠️ Il neo analizzato è **{label.upper()}** con una precisione del **{confidence:.2f}%**."
            )

        # 📊 Probabilità dettagliate per entrambe le classi
        st.markdown("### 📊 **Probabilità per ciascuna classe:**")
        st.write(f"- **Benigno:** {probabilities['Benigno']:.2f}%")
        st.write(f"- **Maligno:** {probabilities['Maligno']:.2f}%")

        # Barra di progresso visiva per la classe predetta
        st.markdown("### 🚦 **Confidenza del Modello:**")
        st.progress(int(confidence))

        # 📝 Consigli finali
        st.markdown(
            """
        ---
        **ℹ️ Nota Bene:**  
        Questo strumento non sostituisce una diagnosi medica.  
        Per qualsiasi dubbio, consulta un dermatologo specializzato. 🩺
        """
        )
