###########################################################################################################
#                                        DEMO CROWD COUNTING (STREAMLIT)                                  #
###########################################################################################################
#
# DESCRIZIONE:
# Questo script implementa una semplice interfaccia grafica basata su Streamlit per il conteggio di persone 
# in immagini, sfruttando un modello di regressione basato su ResNet50 pre-addestrata e addestrata sul 
# dataset di interesse. 
#
# FUNZIONAMENTO:
# - Carica il modello migliore (best model salvato durante il training).
# - Permette di caricare un’immagine tramite interfaccia web (jpg, jpeg, png).
# - Preprocessa l’immagine (resize, tensor, normalizzazione come in fase di training).
# - Esegue la predizione del numero di persone (output scalare).
# - Mostra il risultato sia in forma continua (float) che arrotondata all’intero più vicino.
# - Se il nome del file rispetta il formato "test_####_%%%%.jpg" (con #### id incrementale e %%%% etichetta),
#   mostra anche il numero reale di persone come etichetta di riferimento.
# - In caso contrario, avvisa l’utente che il nome file non è nel formato corretto.
#
#
# USO:
# Avviare da terminale con (trovandoti nella cartella dove è presente il file demo.py):
#     streamlit run demo.py
# e caricare un’immagine per ottenere la predizione del numero di persone.
#
# AUTORE:
#  Elia Antonio Malve'
# 
# DATA ULTIMA MODIFICA:
#  09/09/2025
# 
###########################################################################################################

import streamlit as st                                     
import torch                                               
import torch.nn as nn                                      
from torchvision import transforms                         
from torchvision.models import resnet50, ResNet50_Weights  
from PIL import Image                                      
import os                                                  

# ============================== IPERPARAMETRI ==============================
DROPOUT_RATE = 0.55   # tasso di dropout usato nel training

# ============================== MODELLO ==============================
class ResNet50Regressor(nn.Module):
    """ResNet50 per regressione (conteggio di persone)."""
    def __init__(self, dropout_p=DROPOUT_RATE):
        super().__init__()
        # carica ResNet50 con pesi pre-addestrati ImageNet
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features  # numero di feature in input all'ultimo layer
        # sostituiamo l'ultimo layer con Dropout + Linear(1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),        # regolarizzazione
            nn.Linear(in_features, 1)       # output scalare (conteggio persone)
        )

    def forward(self, x):
        x = self.backbone(x)                # forward standard di ResNet50
        return x.squeeze(1)                 # rimuove dimensione inutile (da [B,1] a [B])

# ============================== CARICAMENTO MODELLO ==============================
@st.cache_resource
def load_model(model_path, dropout_p=DROPOUT_RATE):
    """Carica il modello salvato (best model) in modalità eval."""
    model = ResNet50Regressor(dropout_p=dropout_p)                     # istanzia modello
    model.load_state_dict(torch.load(model_path, map_location="cpu"))  # carica pesi
    model.eval()                                                       # setta in modalità valutazione
    return model

# Percorso del modello salvato (best model dal training)
MODEL_PATH = r"C:\Users\eliam\Desktop\progetto_ML\codice\best_ResNet50_crowd.pth"
model = load_model(MODEL_PATH, dropout_p=DROPOUT_RATE)

# ============================== PREPROCESSING ==============================
def preprocess_image(image):
    """Preprocessa immagine come nel training (resize, tensor, normalize)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                       # ridimensiona a 224x224
        transforms.ToTensor(),                               # PIL -> Tensor [0,1]
        transforms.Normalize((0.485, 0.456, 0.406),          # normalizzazione (media ImageNet)
                             (0.229, 0.224, 0.225))          # normalizzazione (deviazione standard ImageNet)
    ])
    # se immagine non è RGB, la converte
    if image.mode != "RGB":
        image = image.convert("RGB")
    # applica trasformazioni e aggiunge dimensione batch
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# ============================== INTERFACCIA STREAMLIT ==============================
st.title("Demo counting di persone (ResNet50)")   # titolo app

# widget per upload immagine
uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostra immagine caricata
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine caricata")   

    # Preprocessa immagine per il modello
    img_tensor = preprocess_image(image)

    # Predizione con il modello
    with torch.no_grad():                          # disattiva gradienti 
        output = model(img_tensor)                 # forward 
        predicted_count = output.item()            # previsione

    # Mostra risultato (conteggio stimato, sia float che arrotondato)
    st.success(f"Persone stimate presenti nell'immagine: **{predicted_count:.2f}** (≈ {round(predicted_count)})")

    # ============================== ETICHETTA REALE (se disponibile) ==============================
    # Estrae il nome file originale 
    fname = uploaded_file.name

    # Controlla formato "test_####_%%%%.jpg"
    parts = os.path.splitext(fname)[0].split("_")  # separa nome base e divide con "_"
    if len(parts) >= 3 and parts[0] == "test":
        try:
            # ultima parte = etichetta, convertibile in numero
            true_label = float(parts[-1])
            st.info(f"Valore reale presente nel nome file: **{true_label:.0f}** persone")
        except ValueError:
            # se non è convertibile in numero
            st.warning("Formato etichetta non valido! Deve essere test_####_%%%% con etichetta numerica finale.")
    else:
        # messaggio se non rispetta il formato corretto
        st.warning("Nome file non nel formato corretto! Deve essere test_####_%%%% con #### id incrementale e %%%% etichetta.")
