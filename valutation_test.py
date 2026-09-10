###########################################################################################################
#                                EVALUATION SCRIPT FOR CROWD COUNTING (RESNET50)                          #
###########################################################################################################
#
# DESCRIZIONE:
# Questo script valuta un modello di crowd counting basato su ResNet50, precedentemente addestrato 
# e salvato durante la fase di training. Il modello è stato usato per regressione al fine di 
# prevedere il numero di persone presenti in un'immagine.
#
# FUNZIONAMENTO:
# - Carica il modello migliore salvato (best model).
# - Legge le immagini del test set in formato "test_####_%%%%.jpg|jpeg|png",
#   dove #### è un id incrementale che identifica univocamente l'immagine e %%%% rappresenta l'etichetta (numero di persone reali).
# - Applica le trasformazioni di preprocessing (resize, normalizzazione).
# - Esegue predizioni con il modello e calcola le metriche di regressione:
#      Mean Squared Error (MSE)
#      Mean Absolute Error (MAE)
#      Root Mean Squared Error (RMSE)
# - Stima un intervallo di confidenza al 95% per il MAE utilizzando Bootstrap, una tecnica statistica di ricampionamento con reinserimento.
#   In pratica, invece di raccogliere nuovi dati, si creano molti dataset "fittizi" a partire dal dataset originale, estraendo campioni casuali 
#   e potendo reinserire la stessa immagine più volte. Su ciascun dataset si calcola poi la statistica, in questo caso la MAE, e si trovano 
#   gli intervalli di confidenza. In questo caso, è stato usato per via delle dimensioni ridotte del dataset di test.
# - Genera e salva grafici:
#      Scatter plot (predizioni vs valori reali)
#      Istogramma degli errori (visualizza di quanto il modello tende a sbagliare).
#
# AUTORE:
#   Elia Antonio Malve'
#
# DATA ULTIMA MODIFICA:
#   09/09/2025
#
# USO:
# 1. Assicurarsi che il modello allenato sia disponibile nel percorso definito in MODEL_PATH.
# 2. Posizionare le immagini di test nella cartella definita in TEST_DIR, rispettando il formato del nome.
# 3. Eseguire lo script per ottenere metriche, intervallo di confidenza e grafici di valutazione.
#
# COMANDO PER L'USO DA TERMINALE:
#   python valutazione_test.py
#
###########################################################################################################

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, UnidentifiedImageError                       # gestione immagini + fix errori se l'immagine non rispetta il formato richiesto
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========================= IPERPARAMETRI =========================
INPUT_SIZE   = 224                                                     # dimensione di input del modello
BATCH_SIZE   = 32                                                      # dimensione batch 
DROPOUT_RATE = 0.55                                                    # rate di dropout
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa GPU se disponibile
N_BOOTSTRAP  = 1000                                                    # numero di campioni bootstrap

# Percorso del modello salvato (best model dal training)
MODEL_PATH = r"C:\Users\eliam\Desktop\progetto_ML\codice\best_ResNet50_crowd.pth" 

# Percorso cartella di test contenente i file immagine
TEST_DIR = r"C:\Users\eliam\Desktop\progetto_ML\test_dataset"

# ========================= DATASET =========================
class CrowdTestDataset(Dataset):
    """
    Dataset custom: legge immagini .jpg/.jpeg/.png e ricava l'etichetta dal nome file.
    Formato nome atteso: "test_####_%%%%.jpg"
        - #### : id incrementale (univoco per l'immagine)
        - %%%% : numero persone (label)
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # Seleziona solo i file con estensioni valide
        self.files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.files.sort()  # ordina i file per consistenza
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]                        # nome del file 
        fpath = os.path.join(self.img_dir, fname)      # percorso completo

        # ---- carica immagine ----
        # Tenta di aprire l’immagine con PIL (Python Imaging Library).  
        # Se il file è corrotto o non leggibile, intercetta l’errore e stampa un warning,  
        # sostituendo l’immagine con un placeholder nero della stessa dimensione.  
        try:
            img = Image.open(fpath).convert("RGB")     # tenta apertura con PIL
        except UnidentifiedImageError:
            print(f"[WARN] Immagine corrotta o non leggibile: {fname}. Uso immagine nera placeholder.")
            # se non si riesce ad aprire, usa un'immagine nera "dummy"
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0,0,0))

        # ---- estrai label dal nome file ----
        # esempio "test_0001_0023.jpg" -> split("_") = ["test", "0001", "0023"]
        parts = os.path.splitext(fname)[0].split("_")  # togli estensione e dividi con "_"
        label_str = parts[-1]                          # ultima parte = etichetta ("0023")
        label = float(label_str)                       # converti in float per regressione

        # ---- trasformazioni ----
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

# ========================= TRASFORMAZIONI =========================
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),      # ridimensiona l'immagine
    transforms.ToTensor(),                            # converte in tensore [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],       # normalizzazione standard (ImageNet)
                         [0.229, 0.224, 0.225])
])

# ========================= MODELLO =========================
class ResNet50Regressor(nn.Module):
    """ResNet50 per regressione (conteggio di persone)."""
    def __init__(self, dropout_p=DROPOUT_RATE):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)  # carica pesi pre-addestrati
        in_features = self.backbone.fc.in_features                  # numero input all'ultimo layer
        # sostituisci l'ultimo layer con Dropout + Linear (1 output)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        x = self.backbone(x)  # forward classico
        return x.squeeze(1)   # rimuove dimensione inutile (da [B,1] a [B])

# ========================= FUNZIONE BOOTSTRAP =========================
def bootstrap_mae(y_true, y_pred, n_bootstrap=1000):
    """
    Stima intervallo di confidenza del MAE tramite bootstrap.
    - y_true, y_pred: array numpy dei valori reali e predetti
    - n_bootstrap: numero di campioni bootstrap
    """
    # calcola errori assoluti per ciascuna immagine
    errors = np.abs(y_pred - y_true)
    n = len(errors)

    # lista per memorizzare i MAE ricampionati
    mae_samples = []

    # ciclo bootstrap
    for _ in range(n_bootstrap):
        # scegli n indici casuali con reinserimento
        idxs = np.random.choice(n, n, replace=True)
        # calcola MAE su questo campione
        mae_samples.append(np.mean(errors[idxs]))

    # calcola intervallo di confidenza al 95%
    lower = np.percentile(mae_samples, 2.5)
    upper = np.percentile(mae_samples, 97.5)

    return lower, upper

# ========================= FUNZIONE DI VALUTAZIONE =========================
def evaluate_model(model, loader, device):
    """Valuta il modello sul test set e calcola metriche di regressione."""
    model.eval()
    y_true, y_pred = [], []

    # disattiva i gradienti per velocizzare
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)                    # sposta batch su device
            outputs = model(imgs).cpu().numpy()       # predizioni -> numpy
            y_pred.extend(outputs)                    # aggiungi predizioni
            y_true.extend(labels.numpy())             # aggiungi etichette reali

    # converte in array numpy
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calcolo metriche regressione
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Bootstrap per intervallo di confidenza del MAE
    mae_lower, mae_upper = bootstrap_mae(y_true, y_pred, n_bootstrap=N_BOOTSTRAP)

    # stampa risultati
    print("\n===== RISULTATI SUL TEST SET =====")
    print(f"MSE:   {mse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"Intervallo di confidenza MAE (95%): [{mae_lower:.4f}, {mae_upper:.4f}]")

    return y_true, y_pred, {"MSE": mse, "MAE": mae, "RMSE": rmse,
                            "MAE_CI95": (mae_lower, mae_upper)}

# ========================= GRAFICI =========================
def plot_results(y_true, y_pred, output_dir="evaluation_results"):
    """Genera scatter plot e istogramma degli errori."""
    os.makedirs(output_dir, exist_ok=True)  # crea cartella se non esiste

    # Scatter plot: valori reali vs predetti
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--', lw=2)  # linea identità
    plt.xlabel("Valori Reali")
    plt.ylabel("Predizioni")
    plt.title("Scatter Plot - Predizioni vs Reali")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_ytrue_ypred.png"), dpi=150)
    plt.show()

    # Istogramma errori (residui)
    errors = y_pred - y_true
    plt.figure(figsize=(6,6))
    plt.hist(errors, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
    plt.xlabel("Errore (Pred - Reale)")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione degli errori (Residui)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_residuals.png"), dpi=150)
    plt.show()

# ========================= MAIN =========================
if __name__ == "__main__":
    # Carica dataset di test
    test_dataset = CrowdTestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Carica modello
    model = ResNet50Regressor(dropout_p=DROPOUT_RATE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Valutazione (con bootstrap per CI)
    y_true, y_pred, metrics = evaluate_model(model, test_loader, DEVICE)

    # Grafici
    plot_results(y_true, y_pred, output_dir="evaluation_results")
