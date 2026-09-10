###########################################################################################################
#                                     CROWD COUNTING WITH RESNET50                                        #
###########################################################################################################    
#
# DESCRIZIONE:
# Script per il train di un modello di machine learning basato su ResNet50,
# usato per la regressione al fine di stimare il numero di persone presenti in un’immagine.
# Il modello viene addestrato su immagini e label pre-salvati in file .npy (formato di file binario di NumPy).
#
# LINK AL DATASET:
# https://www.kaggle.com/datasets/fmena14/crowd-counting/data
#
# FUNZIONAMENTO:
# - Caricamento del dataset da file numpy (images.npy, labels.npy).
# - Preprocessing con trasformazioni e data augmentation (resize, flip, piccole rotazioni, jitter).
# - Training loop con:
#       - Loss: MAE (Mean Absolute Error) per maggiore robustezza agli outlier e miglior significato fisico.
#       - Ottimizzatore: SGD con momentum e fattore di regolarizzazione (weight decay).
#       - Scheduler: ReduceLROnPlateau per ridurre il learning rate in caso di stagnazione.
#       - Early stopping e salvataggio del miglior modello in base alla loss di validazione.
#       - Logging su TensorBoard (loss e metriche), per verificare il processo in tempo reale.
# - Valutazione finale con varie metriche normalmente usate per la regressione (MSE, MAE, RMSE).
# - Generazione e salvataggio di grafici (loss curve ed andamento delle metriche in funzione delle epoche).
#
# CARATTERISTICHE TECNICHE:
# - Framework: PyTorch
# - Architettura: ResNet50 pre-addestrata su ImageNet, adattata alla regressione (output scalare).
# - Dataset: immagini in formato NumPy array, con labels numeriche nello stesso formato.
# - Metriche: MSE (Mean Square Error), MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
# - Real-Time Logging: TensorBoard
#
# COMANDO PER L'USO DA TERMINALE:
#   python crowd_counting_resnet50.py
#
#   
# AUTORE:
#   Elia Antonio Malve'
#   
#
# DATA ULTIMA MODIFICA (ED ULTIMA VERIFICA DEL DATASET):
#   09/09/2025
#
###########################################################################################################

# ======================================= IMPORT DELLE LIBRERIE ===========================================

import numpy as np                                                   # operazioni numeriche
import torch                                                         # framework PyTorch
import torch.nn as nn                                                # moduli per NN
from torch.utils.data import Dataset, DataLoader, random_split       # dataset/dataloader e split
from torchvision import transforms                                   # trasformazioni per data augmentation
from torchvision.models import resnet50, ResNet50_Weights            # ResNet50 pre-addestrata
import os                                                            # gestione percorsi e salvataggi
import torch.optim as optim                                          # ottimizzatori (SGD)
from torch.optim.lr_scheduler import ReduceLROnPlateau               # scheduler LR su plateau
import matplotlib.pyplot as plt                                      # grafici
from sklearn.metrics import mean_squared_error, mean_absolute_error  # metriche per la regressione
from torch.utils.tensorboard import SummaryWriter                    # logger per TensorBoard

# ========================================= IPERPARAMETRI =================================================
BATCH_SIZE     = 32               # dimensione batch
EPOCHS         = 50               # epoche di training
LEARNING_RATE  = 1e-3             # learning rate per SGD
MOMENTUM       = 0.92             # momentum per SGD
WEIGHT_DECAY   = 1e-4             # fattore di regolarizzazione
PATIENCE       = 4                # parametro per early stopping (epoche senza migliorare)
INPUT_SIZE     = 224              # dimensione immagini per ResNet
VAL_SPLIT      = 0.2              # grandezza del validation set 
SEED           = 42               # seed per riproducibilità
DROPOUT_RATE   = 0.55             # rate di dropout
LR_REDUCTION   = 0.5              # fattore di riduzione del LR 

# cartella di salvataggio 
SAVE_DIR = r"C:\Users\eliam\Desktop\progetto_ML\codice"  # per best model e grafici, definita ad inizio codice

# fissaggio dei seed 
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================ DATASET =====================================================
class CrowdDataset(Dataset):
    """Dataset custom: carica immagini (X.npy) e conteggi (y.npy) da file .npy."""
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path)          # carica tutte le immagini
        self.y = np.load(y_path)          # carica tutti i label
        self.transform = transform        # pipeline per le trasformazioni necessarie alla data augmentation

    def __len__(self):
        return len(self.y)                # numero di campioni

    def __getitem__(self, idx):
        # ---- legge immagine e label dal dataset ----
        img = self.X[idx]                 # immagini
        label = self.y[idx]               # label

        # Le immagini caricate dal file .npy possono avere formati diversi:
        #   - float con valori normalizzati in [0,1]
        #   - float con valori in [0,255]
        #   - interi ma non necessariamente uint8
        #
        # La trasformazione ToPILImage() di PyTorch accetta solo tensori di tipo uint8
        # con valori in [0..255], oppure float in [0..1]. Per evitare errori o distorsioni:
        #
        # 1. Se il tipo non è già uint8, lo convertiamo in modo sicuro.
        # 2. Se i valori massimi > 1.0 assumiamo che l'immagine sia in [0..255] float
        #    e facciamo il cast a uint8 (dopo clipping a [0,255]).
        # 3. Altrimenti, se i valori sono in [0..1], li riportiamo a [0..255]
        #    moltiplicando per 255 e convertendo in uint8.
        #
        # In questo modo ci assicuriamo che tutte le immagini siano nello standard
        # atteso da ToPILImage(), evitando warning o errori di tipo.
        # Questa correzione si è resa necessaria a seguito di un errore riportato dal terminale di Windows.
        if img.dtype != np.uint8:
            if img.max() > 1.0:           # se sembra 0..255 float
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:                          # se è in 0..1 float
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

        # converte a Tensor e cambia ordine canali 
        img = torch.from_numpy(img).permute(2, 0, 1)    # torch.uint8

        # Converte la label in scalare float32:
        # - se è un array NumPy (es. shape (1,)), lo riduce a scalare con squeeze()
        # - la trasforma poi in tensore PyTorch float32, adatto alla regressione
        if isinstance(label, np.ndarray):
            label = np.squeeze(label)
        label = torch.tensor(float(label), dtype=torch.float32)

        # applica trasformazioni se presenti 
        if self.transform:
            img = self.transform(img)

        return img, label

# ========================================= TRASFORMAZIONI (AUGMENTATION) =================================
# media/deviazione standard ImageNet per normalizzazione (parametri noti)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# pipeline di training con data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),                                               # Tensor(uint8)->PIL
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),                           # resize a 224x224
    transforms.RandomHorizontalFlip(p=0.5),                                # flip orizzontale
    transforms.RandomRotation(degrees=10),                                 # rotazione +/-10°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # jitter colori
    transforms.ToTensor(),                                                 # PIL->Tensor float [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)                      # normalizza come ImageNet
])

# pipeline di validazione (senza data augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ==================================================== MODELLO ====================================================
class ResNet50Regressor(nn.Module):
    """ResNet50 pre-addestrata su ImageNet, adattata alla regressione (quindi con uscita scalare)."""
    def __init__(self, dropout_p=DROPOUT_RATE):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)  # carica pesi ImageNet
        in_features = self.backbone.fc.in_features                  # dimensione in ingresso all'ultimo layer
        # sostituiamo l'ultimo layer con Dropout + Linear(1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),                                # dropout
            nn.Linear(in_features, 1)                               # output scalare
        )

    def forward(self, x):
        x = self.backbone(x)        # (B,1)
        x = x.squeeze(1)            # (B, ) -> vettore di conteggi
        return x

# ============================================== TRAINING LOOP + TENSORBOARD =========================================
def train_model(model, train_loader, val_loader, device, exp_name="ResNet50_crowd"):
    """Allena il modello, logga su TensorBoard e salva il best model su Val Loss."""

    # Si usa come criterio di loss la MAE (Mean Absolute Error),
    # più robusta agli outlier rispetto alla MSE, nonche' piu' adeguata ad un conteggio di persone.
    criterion = nn.L1Loss()  

    # ottimizzatore: SGD con momentum e parametro di regolarizzazione
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # scheduler: riduce il LR se la val_loss non migliora
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE, factor=LR_REDUCTION)

    # SummaryWriter per TensorBoard (cartella "runs/exp_name")
    writer = SummaryWriter(log_dir=os.path.join("runs", exp_name))

    # Tenta di loggare l'architettura del modello su TensorBoard usando un input fittizio.
    # Se fallisce (es. incompatibilità versioni), mostra solo un warning senza interrompere il training.
    try:
        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)  # batch fittizio
        writer.add_graph(model, dummy)                                    # grafo rete
    except Exception as e:
        print(f"[WARN] add_graph non riuscito: {e}")

    # Inizializza i parametri per early stopping:
    # best_val_loss parte da infinito, epochs_no_improve conta epoche senza miglioramenti.
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # history per grafici
    train_losses, val_losses = [], []
    val_mse_list, val_mae_list, val_rmse_list = [], [], []

    # percorso salvataggio best model
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"best_{exp_name}.pth")

    # ciclo epoche
    for epoch in range(EPOCHS):
        # -------------------- TRAIN --------------------
        model.train()                                   # modalità training
        running_loss = 0.0                              # accumulatore loss

        for imgs, labels in train_loader:               # iterazione batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()                       # azzera gradienti
            outputs = model(imgs)                       # forward
            loss = criterion(outputs, labels)           # MAE
            loss.backward()                             # backward
            optimizer.step()                            # update pesi
            running_loss += loss.item() * imgs.size(0)  # accumula loss pesata per batch size

        # loss media sul training set
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -------------------- VALIDATION --------------------
        model.eval()                               # modalità validation
        val_running_loss = 0.0
        y_true_epoch, y_pred_epoch = [], []        # per metriche (MSE/MAE/RMSE)

        with torch.no_grad():                      # no grad in validation
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)  # MAE anche in validation
                val_running_loss += loss.item() * imgs.size(0)

                # accumula per calcolo metriche
                y_pred_epoch.extend(outputs.detach().cpu().numpy())
                y_true_epoch.extend(labels.detach().cpu().numpy())

        # loss media sul validation set (MAE media)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # calcolo metriche di regressione su validation (per curva metriche)
        y_true_epoch = np.array(y_true_epoch, dtype=np.float32)
        y_pred_epoch = np.array(y_pred_epoch, dtype=np.float32)
        val_mse = mean_squared_error(y_true_epoch, y_pred_epoch)   # MSE
        val_mae = mean_absolute_error(y_true_epoch, y_pred_epoch)  # MAE
        val_rmse = float(np.sqrt(val_mse))                         # RMSE

        val_mse_list.append(val_mse)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)

        # stampa a schermo i valori per epoca
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
              f"Train Loss (MAE): {epoch_train_loss:.4f} | Val Loss (MAE): {epoch_val_loss:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # ---- LOG SU TENSORBOARD  ----
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)   # loss train (MAE)
        writer.add_scalar("Loss/Val",   epoch_val_loss,   epoch)   # loss val (MAE)
        writer.add_scalar("Metrics/Val_MSE",  val_mse,  epoch)     # MSE val
        writer.add_scalar("Metrics/Val_MAE",  val_mae,  epoch)     # MAE val
        writer.add_scalar("Metrics/Val_RMSE", val_rmse, epoch)     # RMSE val
        writer.add_scalar("LR", current_lr, epoch)                 # learning rate corrente

        # aggiorna scheduler con la val_loss 
        prev_lr = current_lr
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"  --> ReduceLROnPlateau: LR ridotto a {new_lr:.6f}")

        # early stopping + salvataggio best model (ora basato su MAE val)
        if epoch_val_loss < best_val_loss - 1e-4:   # margine minimo per considerare miglioramento
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  --> Miglior modello salvato in: {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping!")
                break

    # carica best model, se esiste
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("[ATTENZIONE] Nessun best model salvato; rimane l'ultimo stato del modello.")

    # chiudi writer TensorBoard
    writer.close()

    # ritorna storico per i grafici
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_mse": val_mse_list,
        "val_mae": val_mae_list,
        "val_rmse": val_rmse_list
    }
    return model, history

# ============================================ VALUTAZIONE FINALE ======================================================
def evaluate_model(model, loader, device):
    """Valuta il modello (MSE, MAE, RMSE) sul loader passato."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)                          # sposta su device
            outputs = model(imgs).cpu().numpy()             # predizioni -> CPU numpy
            y_pred.extend(outputs)                          # accumula predizioni
            y_true.extend(labels.numpy())                   # accumula ground truth

    # converte in array numpy
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # metriche per la regressione
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # stampa risultati
    print("\n===== VALUTAZIONE =====")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return y_true, y_pred, {"mse": mse, "mae": mae, "rmse": rmse}

# ================================= GRAFICI TRAINING (salvati in SAVE_DIR) ===================================
def plot_training_curves(history, save_dir=SAVE_DIR, show=True):
    """Crea e salva i grafici: (1) loss curve; (2) curve metriche (MSE/MAE/RMSE)."""
    os.makedirs(save_dir, exist_ok=True)                # assicura cartella
    epochs = range(1, len(history["train_loss"]) + 1)   # asse x

    # -------- Figura 1: Loss train/val --------
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss (MAE)") # ora loss=MAE
    plt.plot(epochs, history["val_loss"],   label="Val Loss (MAE)")   # idem per val
    plt.xlabel("Epoche")
    plt.ylabel("MAE Loss")
    plt.title("Learning Curve - Loss (MAE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out_loss, dpi=150)
    print(f"[SALVATO] {out_loss}")
    if show:
        plt.show()
    else:
        plt.close()

    # -------- Figura 2: Metriche val (MSE/MAE/RMSE) --------
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["val_mse"],  label="Val MSE")
    plt.plot(epochs, history["val_mae"],  label="Val MAE")
    plt.plot(epochs, history["val_rmse"], label="Val RMSE")
    plt.xlabel("Epoche")
    plt.ylabel("Valore")
    plt.title("Learning Curves - Metriche (Val)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_metrics = os.path.join(save_dir, "metrics_curves.png")
    plt.savefig(out_metrics, dpi=150)
    print(f"[SALVATO] {out_metrics}")
    if show:
        plt.show()
    else:
        plt.close()

# ================================================ MAIN ============================================================
if __name__ == "__main__":
    # ---- percorsi ai file .npy ----
    X_path = r"C:\Users\eliam\Desktop\progetto_ML\dataset\train\file_npy\images.npy"
    y_path = r"C:\Users\eliam\Desktop\progetto_ML\dataset\train\file_npy\labels.npy"

    # ---- carica dataset completo ----
    full_dataset = CrowdDataset(X_path, y_path, transform=None)

    # ---- split del dataset  ----
    val_size = int(VAL_SPLIT * len(full_dataset))                    # dimensione dataset validation
    train_size = len(full_dataset) - val_size                        # dimensione dataset training
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)                # riproducibilita'
    )

    # ---- assegna trasformazioni (augmentation solo su train) ----
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform   = val_transform

    # ---- DataLoader  ----
    use_cuda   = torch.cuda.is_available()
    num_workers = 0                              # evita problemi di pickling su Windows, riga introdotta in seguito ad errore su terminale
    pin_memory  = bool(use_cuda)                 # pin_memory solo se CUDA disponibile

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    # ---- device ----
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # ---- modello ----
    model = ResNet50Regressor(dropout_p=DROPOUT_RATE).to(device)

    # ---- training (con TensorBoard) ----
    model, history = train_model(model, train_loader, val_loader, device, exp_name="ResNet50_crowd")

    # ---- valutazione finale su validation ----
    y_true, y_pred, metrics = evaluate_model(model, val_loader, device)

    # ---- grafici training salvati in SAVE_DIR ----
    plot_training_curves(history, save_dir=SAVE_DIR, show=True)

