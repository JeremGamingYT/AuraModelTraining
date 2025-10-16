import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============ HYPERPARAMÈTRES ============
IMG_SIZE = 128  # Augmenté pour meilleure qualité
BATCH_SIZE = 4   # Réduit car images plus grandes
NUM_WORKERS = 2
HIDDEN_DIMS = [64, 128, 256]  # Multi-scale features
NUM_EPOCHS_PRETRAIN = 50  # Phase 1: Autoencoder
NUM_EPOCHS_FINETUNE = 100  # Phase 2: Prédiction
LEARNING_RATE = 2e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Utilisation du device: {DEVICE}")

# ============ SSIM IMPLEMENTATION ============
class SSIM(nn.Module):
    """Implémentation de SSIM (Structural Similarity Index)"""
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        window = self.window.to(img1.device).type_as(img1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

# ============ DATASET AMÉLIORÉ ============
class EnhancedFrameDataset(Dataset):
    """Dataset amélioré avec augmentation et modes multiples"""
    def __init__(self, img_dir, img_size, mode='predict', augment=True):
        """
        Args:
            mode: 'reconstruct' pour autoencoder, 'predict' pour prédiction
            augment: Activer l'augmentation de données
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, 'frame_*.png')))
        self.img_size = img_size
        self.mode = mode
        self.augment = augment
        
        assert len(self.img_paths) > 1, 'Il faut au moins deux frames dans le dossier.'
        
        # Transformations de base
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        
        # Augmentations (uniquement en entraînement)
        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ])
        
        print(f"Dataset initialisé: {len(self.img_paths)} frames, mode={mode}")
    
    def __len__(self):
        if self.mode == 'reconstruct':
            return len(self.img_paths)  # Toutes les images
        else:
            return len(self.img_paths) - 1  # Paires consécutives
    
    def __getitem__(self, idx):
        if self.mode == 'reconstruct':
            # Mode autoencoder: entrée = sortie (reconstruction)
            img = Image.open(self.img_paths[idx]).convert('RGB')
            img_tensor = self.base_transform(img)
            
            if self.augment and np.random.rand() > 0.5:
                img_tensor = self.augment_transform(img_tensor)
            
            # Ajoute du bruit pour robustesse
            if self.augment:
                noise = torch.randn_like(img_tensor) * 0.02
                img_input = torch.clamp(img_tensor + noise, 0, 1)
            else:
                img_input = img_tensor
                
            return img_input, img_tensor  # (noisy, clean)
        
        else:  # mode == 'predict'
            # Mode prédiction: frame_t -> frame_t+1
            img_input = Image.open(self.img_paths[idx]).convert('RGB')
            img_target = Image.open(self.img_paths[idx + 1]).convert('RGB')
            
            img_input = self.base_transform(img_input)
            img_target = self.base_transform(img_target)
            
            if self.augment:
                # Applique les mêmes augmentations aux deux images
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                img_input = self.augment_transform(img_input)
                torch.manual_seed(seed)
                img_target = self.augment_transform(img_target)
            
            return img_input, img_target

# ============ CONVLSTM AMÉLIORÉ ============
class ImprovedConvLSTMCell(nn.Module):
    """ConvLSTM avec batch normalization et dropout"""
    def __init__(self, input_channels, hidden_dim, kernel_size, dropout=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_dim, 
            4 * hidden_dim, 
            kernel_size, 
            padding=padding
        )
        
        # Normalisation pour stabilité
        self.bn = nn.BatchNorm2d(4 * hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        conv_out = self.bn(conv_out)
        conv_out = self.dropout(conv_out)
        
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        H, W = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=DEVICE),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=DEVICE)
        )

# ============ BLOC RÉSIDUEL ============
class ResidualBlock(nn.Module):
    """Bloc résiduel pour connexions skip"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# ============ MODÈLE AMÉLIORÉ ============
class EnhancedFramePredictor(nn.Module):
    """
    Modèle amélioré avec:
    - Architecture multi-échelle
    - Skip connections
    - Attention spatiale
    - Normalisation
    """
    def __init__(self, channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        # ENCODER avec downsampling progressif
        self.encoder_conv1 = nn.Conv2d(channels, hidden_dims[0], 3, padding=1)
        self.encoder_lstm1 = ImprovedConvLSTMCell(hidden_dims[0], hidden_dims[0], 3)
        self.encoder_res1 = ResidualBlock(hidden_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder_conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1)
        self.encoder_lstm2 = ImprovedConvLSTMCell(hidden_dims[1], hidden_dims[1], 3)
        self.encoder_res2 = ResidualBlock(hidden_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder_conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, padding=1)
        self.encoder_lstm3 = ImprovedConvLSTMCell(hidden_dims[2], hidden_dims[2], 3)
        
        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            ResidualBlock(hidden_dims[2]),
            ResidualBlock(hidden_dims[2])
        )
        
        # DECODER avec upsampling progressif
        self.decoder_lstm3 = ImprovedConvLSTMCell(hidden_dims[2], hidden_dims[2], 3)
        self.upconv3 = nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], 2, stride=2)
        
        self.decoder_lstm2 = ImprovedConvLSTMCell(hidden_dims[1]*2, hidden_dims[1], 3)  # *2 pour skip
        self.decoder_res2 = ResidualBlock(hidden_dims[1])
        self.upconv2 = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 2, stride=2)
        
        self.decoder_lstm1 = ImprovedConvLSTMCell(hidden_dims[0]*2, hidden_dims[0], 3)  # *2 pour skip
        self.decoder_res1 = ResidualBlock(hidden_dims[0])
        
        # ATTENTION MODULE
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[0]//2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0]//2, 1, 1),
            nn.Sigmoid()
        )
        
        # OUTPUT
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, H, W = x.shape
        
        # === ENCODER ===
        # Level 1
        enc1 = F.relu(self.encoder_conv1(x))
        h1, c1 = self.encoder_lstm1.init_hidden(b, (H, W))
        h1, c1 = self.encoder_lstm1(enc1, (h1, c1))
        enc1_res = self.encoder_res1(h1)
        skip1 = enc1_res  # Sauvegarde pour skip connection
        pool1 = self.pool1(enc1_res)
        
        # Level 2
        enc2 = F.relu(self.encoder_conv2(pool1))
        h2, c2 = self.encoder_lstm2.init_hidden(b, (H//2, W//2))
        h2, c2 = self.encoder_lstm2(enc2, (h2, c2))
        enc2_res = self.encoder_res2(h2)
        skip2 = enc2_res  # Sauvegarde pour skip connection
        pool2 = self.pool2(enc2_res)
        
        # Level 3
        enc3 = F.relu(self.encoder_conv3(pool2))
        h3, c3 = self.encoder_lstm3.init_hidden(b, (H//4, W//4))
        h3, c3 = self.encoder_lstm3(enc3, (h3, c3))
        
        # === BOTTLENECK ===
        bottleneck = self.bottleneck(h3)
        
        # === DECODER ===
        # Level 3
        h3d, c3d = self.decoder_lstm3.init_hidden(b, (H//4, W//4))
        h3d, c3d = self.decoder_lstm3(bottleneck, (h3d, c3d))
        up3 = self.upconv3(h3d)
        
        # Level 2 avec skip connection
        dec2_input = torch.cat([up3, skip2], dim=1)
        h2d, c2d = self.decoder_lstm2.init_hidden(b, (H//2, W//2))
        h2d, c2d = self.decoder_lstm2(dec2_input, (h2d, c2d))
        dec2_res = self.decoder_res2(h2d)
        up2 = self.upconv2(dec2_res)
        
        # Level 1 avec skip connection
        dec1_input = torch.cat([up2, skip1], dim=1)
        h1d, c1d = self.decoder_lstm1.init_hidden(b, (H, W))
        h1d, c1d = self.decoder_lstm1(dec1_input, (h1d, c1d))
        dec1_res = self.decoder_res1(h1d)
        
        # Attention spatiale
        attention_map = self.attention(dec1_res)
        dec1_attended = dec1_res * attention_map
        
        # Output
        output = self.output_conv(dec1_attended)
        
        return output

# ============ LOSS PERCEPTUELLE ============
class PerceptualLoss(nn.Module):
    """Combine plusieurs losses pour meilleure qualité"""
    def __init__(self, use_ssim=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.use_ssim = use_ssim
        if use_ssim:
            self.ssim_loss = SSIM()
    
    def forward(self, pred, target):
        # Pixel-wise losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        if self.use_ssim:
            # Structural similarity (inversé car SSIM est une métrique de similarité)
            ssim = 1 - self.ssim_loss(pred, target)
            total_loss = l1 + 0.5 * mse + 0.5 * ssim
            return total_loss, {'l1': l1.item(), 'mse': mse.item(), 'ssim': ssim.item()}
        else:
            total_loss = l1 + 0.5 * mse
            return total_loss, {'l1': l1.item(), 'mse': mse.item()}

# ============ CHECKPOINT LOADER COMPAT ============
def load_checkpoint(path, map_location=DEVICE):
    """Compat loader for PyTorch 2.6+ (weights_only default) and older versions.

    Uses weights_only=False to allow loading older checkpoints that may contain
    non-tensor objects (e.g., numpy scalars) saved in the dict.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument
        return torch.load(path, map_location=map_location)

# ============ ENTRAÎNEMENT EN DEUX PHASES ============
def train_model(args):
    # === PHASE 1: PRÉ-ENTRAÎNEMENT AUTOENCODER ===
    print("\n" + "="*50)
    print("PHASE 1: PRÉ-ENTRAÎNEMENT AUTOENCODER")
    print("="*50)
    
    # Dataset pour reconstruction
    pretrain_dataset = EnhancedFrameDataset(
        args.frames, args.img_size, mode='reconstruct', augment=True
    )
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Initialise le modèle
    model = EnhancedFramePredictor(channels=3, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PRETRAIN)
    criterion = PerceptualLoss(use_ssim=True)
    
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    
    # Entraînement phase 1
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        model.train()
        epoch_losses = []
        
        for batch_idx, (noisy, clean) in enumerate(pretrain_loader):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            # Forward
            reconstructed = model(noisy)
            loss, loss_dict = criterion(reconstructed, clean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Log périodique
            if batch_idx % 10 == 0:
                if 'ssim' in loss_dict:
                    print(f"  Batch {batch_idx}/{len(pretrain_loader)}: "
                          f"L1={loss_dict['l1']:.4f}, "
                          f"MSE={loss_dict['mse']:.4f}, "
                          f"SSIM={loss_dict['ssim']:.4f}")
                else:
                    print(f"  Batch {batch_idx}/{len(pretrain_loader)}: "
                          f"L1={loss_dict['l1']:.4f}, "
                          f"MSE={loss_dict['mse']:.4f}")
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_PRETRAIN}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Sauvegarde du meilleur modèle
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(avg_loss),
            }, 'best_autoencoder.pth')
            print(f"  ✓ Meilleur modèle sauvegardé (loss: {best_loss:.6f})")
        
        # Visualisation périodique
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Limiter à 4 images pour la visualisation
                n_vis = min(4, noisy.size(0))
                save_image(
                    torch.cat([noisy[:n_vis], reconstructed[:n_vis], clean[:n_vis]], dim=0),
                    f'pretrain_epoch{epoch+1}.png',
                    nrow=n_vis,
                    normalize=True
                )
            print(f"  → Images sauvegardées: pretrain_epoch{epoch+1}.png")
    
    # === PHASE 2: FINE-TUNING POUR PRÉDICTION ===
    print("\n" + "="*50)
    print("PHASE 2: FINE-TUNING POUR PRÉDICTION")
    print("="*50)
    
    # Charge le meilleur modèle pré-entraîné
    if os.path.exists('best_autoencoder.pth'):
        checkpoint = load_checkpoint('best_autoencoder.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modèle pré-entraîné chargé (epoch {checkpoint['epoch']+1})")
    
    # Dataset pour prédiction
    finetune_dataset = EnhancedFrameDataset(
        args.frames, args.img_size, mode='predict', augment=True
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Nouvel optimizer avec learning rate plus faible
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr/10, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FINETUNE)
    
    best_pred_loss = float('inf')
    
    # Entraînement phase 2
    for epoch in range(NUM_EPOCHS_FINETUNE):
        model.train()
        epoch_losses = []
        
        for batch_idx, (current_frame, next_frame) in enumerate(finetune_loader):
            current_frame = current_frame.to(DEVICE)
            next_frame = next_frame.to(DEVICE)
            
            # Forward
            predicted_frame = model(current_frame)
            loss, loss_dict = criterion(predicted_frame, next_frame)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Sauvegarde du meilleur modèle
        if avg_loss < best_pred_loss:
            best_pred_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(avg_loss),
            }, 'best_predictor.pth')
            print(f"  ✓ Meilleur prédicteur sauvegardé (loss: {best_pred_loss:.6f})")
        
        # Visualisation périodique
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                n_vis = min(4, current_frame.size(0))
                save_image(
                    torch.cat([current_frame[:n_vis], predicted_frame[:n_vis], next_frame[:n_vis]], dim=0),
                    f'prediction_epoch{epoch+1}.png',
                    nrow=n_vis,
                    normalize=True
                )
            print(f"  → Prédictions sauvegardées: prediction_epoch{epoch+1}.png")
    
    print("\n" + "="*50)
    print("✓ ENTRAÎNEMENT TERMINÉ!")
    print(f"  Meilleure loss autoencoder: {best_loss:.6f}")
    print(f"  Meilleure loss prédiction: {best_pred_loss:.6f}")
    print("="*50)
    
    return model

# ============ INFÉRENCE ============
def generate_predictions(model, dataset, num_predictions=5):
    """Génère plusieurs prédictions en chaîne"""
    model.eval()
    
    # Prend la dernière frame comme point de départ
    last_frame, _ = dataset[-1]
    last_frame = last_frame.unsqueeze(0).to(DEVICE)
    
    predictions = []
    current_frame = last_frame
    
    with torch.no_grad():
        for i in range(num_predictions):
            # Prédit la prochaine frame
            next_frame = model(current_frame)
            predictions.append(next_frame)
            
            # Utilise la prédiction comme nouvelle entrée
            current_frame = next_frame
            
            # Sauvegarde
            save_image(next_frame, f'future_frame_{i+1}.png', normalize=True)
            print(f"  Généré: future_frame_{i+1}.png")
    
    # Crée une grille avec toutes les prédictions
    all_preds = torch.cat(predictions, dim=0)
    save_image(all_preds, 'future_sequence.png', nrow=num_predictions, normalize=True)
    print(f"✓ Séquence complète: future_sequence.png")

# ============ MAIN ============
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prédiction avancée de frames avec ConvLSTM')
    parser.add_argument('--frames', type=str, default='frames', 
                        help='Dossier contenant frame_*.png')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                        help='Taille des images')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Taille du batch')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--inference_only', action='store_true',
                        help='Charger un modèle et faire seulement l\'inférence')
    args = parser.parse_args()
    
    if args.inference_only:
        # Mode inférence uniquement
        print("\nMODE INFÉRENCE")
        model = EnhancedFramePredictor(channels=3, hidden_dims=HIDDEN_DIMS).to(DEVICE)
        
        if os.path.exists('best_predictor.pth'):
            checkpoint = torch.load('best_predictor.pth', map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Modèle chargé")
        else:
            print("⚠ Aucun modèle trouvé, utilisation du modèle non entraîné")
        
        test_dataset = EnhancedFrameDataset(
            args.frames, args.img_size, mode='predict', augment=False
        )
        
        generate_predictions(model, test_dataset, num_predictions=10)
    else:
        # Entraînement complet
        model = train_model(args)
        
        # Génération de prédictions finales
        print("\nGÉNÉRATION DE PRÉDICTIONS FUTURES")
        test_dataset = EnhancedFrameDataset(
            args.frames, args.img_size, mode='predict', augment=False
        )
        generate_predictions(model, test_dataset, num_predictions=10)