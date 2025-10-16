import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ============ HYPERPARAMÈTRES ============
IMG_SIZE = 64  # Taille des images (ajustez selon vos besoins)
BATCH_SIZE = 8
NUM_WORKERS = 2
HIDDEN_DIM = 64  # Dimension des features cachées
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Utilisation du device: {DEVICE}")

# ============ DATASET ============
class FrameSequenceDataset(Dataset):
    """Dataset pour charger des séquences d'images frame_XXXX.png"""
    def __init__(self, img_dir, img_size):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, 'frame_*.png')))
        self.img_size = img_size
        assert len(self.img_paths) > 1, 'Il faut au moins deux frames dans le dossier.'
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()  # Convertit en [0, 1]
        ])
        
        print(f"Dataset initialisé avec {len(self.img_paths)} frames")
    
    def __len__(self):
        return len(self.img_paths) - 1
    
    def __getitem__(self, idx):
        # Charge frame_t et frame_t+1
        img_input = Image.open(self.img_paths[idx]).convert('RGB')
        img_target = Image.open(self.img_paths[idx + 1]).convert('RGB')
        
        img_input = self.transform(img_input)
        img_target = self.transform(img_target)
        
        return img_input, img_target

# ============ CONVLSTM CELL ============
class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM qui utilise des convolutions au lieu de multiplications.
    Cette approche capture les relations spatiales dans les images.
    """
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Une seule convolution pour les 4 portes (input, forget, output, cell)
        self.conv = nn.Conv2d(
            input_channels + hidden_dim, 
            4 * hidden_dim, 
            kernel_size, 
            padding=padding
        )
    
    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        
        # Concatène input et état caché précédent
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        
        # Sépare les 4 portes
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        
        # Applique les fonctions d'activation (portes LSTM)
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Nouvel état de cellule et état caché
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        H, W = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=DEVICE),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=DEVICE)
        )

# ============ MODÈLE PRINCIPAL ============
class FramePredictor(nn.Module):
    """
    Modèle Encoder-Decoder avec ConvLSTM pour prédire la prochaine frame.
    Architecture: 2 couches encoder + 2 couches decoder + couche de sortie
    """
    def __init__(self, channels=3, hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        # ENCODER: Encode l'image d'entrée en représentation latente
        self.encoder1 = ConvLSTMCell(channels, hidden_dim, kernel_size=3)
        self.encoder2 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        
        # DECODER: Décode la représentation pour générer l'image suivante
        self.decoder1 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        self.decoder2 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        
        # Couche finale: transforme les features en image RGB
        self.conv_out = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: Image d'entrée (batch, channels, H, W)
        Returns:
            Prédiction de la prochaine image (batch, channels, H, W)
        """
        b, c, H, W = x.shape
        
        # Initialise les états cachés
        h1, c1 = self.encoder1.init_hidden(b, (H, W))
        h2, c2 = self.encoder2.init_hidden(b, (H, W))
        
        # PHASE ENCODER
        h1, c1 = self.encoder1(x, (h1, c1))
        h2, c2 = self.encoder2(h1, (h2, c2))
        
        # Vecteur encodé (représentation latente)
        encoder_vector = h2
        
        # Initialise les états du decoder
        h3, c3 = self.decoder1.init_hidden(b, (H, W))
        h4, c4 = self.decoder2.init_hidden(b, (H, W))
        
        # PHASE DECODER
        h3, c3 = self.decoder1(encoder_vector, (h3, c3))
        h4, c4 = self.decoder2(h3, (h4, c4))
        
        # Génère l'image de sortie
        out = self.conv_out(h4)
        out = torch.sigmoid(out)  # Valeurs entre [0, 1]
        
        return out

# ============ BOUCLE D'ENTRAÎNEMENT ============
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prédiction de frames avec ConvLSTM')
    parser.add_argument('--frames', type=str, default='frames', 
                        help='Dossier contenant frame_*.png')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                        help='Taille des images')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Taille du batch')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    args = parser.parse_args()
    
    # Charge le dataset
    dataset = FrameSequenceDataset(args.frames, args.img_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    
    # Initialise le modèle
    model = FramePredictor(channels=3, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # Loss pour comparer images prédites et réelles
    
    print(f"\nDébut de l'entraînement sur {DEVICE}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters())}")
    
    # ENTRAÎNEMENT
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            pred = model(x)
            loss = criterion(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.6f}')
        
        # Sauvegarde périodique
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → Modèle sauvegardé: {checkpoint_path}")
            
            # Sauvegarde des exemples de prédictions
            model.eval()
            with torch.no_grad():
                pred = model(x)
                pred_img = T.ToPILImage()(pred.cpu())
                gt_img = T.ToPILImage()(y.cpu())
                pred_img.save(f'prediction_epoch{epoch+1}.png')
                gt_img.save(f'groundtruth_epoch{epoch+1}.png')
            model.train()
    
    # Sauvegarde finale
    torch.save(model.state_dict(), 'frame_predictor_final.pth')
    print("\n✓ Entraînement terminé! Modèle sauvegardé: frame_predictor_final.pth")
    
    # INFÉRENCE - Prédire la prochaine frame
    print("\nMode inférence:")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = dataset
        sample_x = sample_x.unsqueeze(0).to(DEVICE)
        prediction = model(sample_x)
        pred_img = T.ToPILImage()(prediction.cpu())
        pred_img.save('next_frame_prediction.png')
        print("✓ Prédiction sauvegardée: next_frame_prediction.png")