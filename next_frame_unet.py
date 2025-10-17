import os
import glob
import math
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid


# ==================================================
#  CONFIG GÉNÉRALE
# ==================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True


# ==================================================
#  DATASET: Chargement de paires (frame_t -> frame_{t+1})
# ==================================================
class NextFrameDataset(Dataset):
    def __init__(
        self,
        frames_dir: str,
        img_size: int = 128,
        augment: bool = True,
        mode: str = 'predict',  # 'predict' ou 'reconstruct'
        add_noise: bool = True,
        noise_std: float = 0.02,
    ) -> None:
        self.img_paths = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
        assert len(self.img_paths) >= 2, 'Il faut au moins 2 images frame_*.png'
        self.img_size = img_size
        self.augment = augment
        self.mode = mode
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self) -> int:
        if self.mode == 'predict':
            return len(self.img_paths) - 1
        return len(self.img_paths)

    def _paired_augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return x, y

        # Flip horizontal
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        # Léger jitter de couleur (mêmes facteurs pour x et y)
        if torch.rand(1).item() < 0.5:
            b = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # ±10%
            c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            s = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            h = (torch.rand(1).item() - 0.5) * 0.1        # ±0.05
            x = TF.adjust_brightness(x, b)
            y = TF.adjust_brightness(y, b)
            x = TF.adjust_contrast(x, c)
            y = TF.adjust_contrast(y, c)
            x = TF.adjust_saturation(x, s)
            y = TF.adjust_saturation(y, s)
            x = TF.adjust_hue(x, h)
            y = TF.adjust_hue(y, h)

        # Affine partagé (petites déformations)
        if torch.rand(1).item() < 0.5:
            angle = (torch.rand(1).item() - 0.5) * 10.0  # ±5°
            translate = (int((torch.rand(1).item()-0.5) * 0.06 * x.shape[2]),
                         int((torch.rand(1).item()-0.5) * 0.06 * x.shape[1]))
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1  # 0.95-1.05
            shear = (torch.rand(1).item() - 0.5) * 6.0        # ±3°
            x = TF.affine(x, angle=angle, translate=translate, scale=scale, shear=shear)
            y = TF.affine(y, angle=angle, translate=translate, scale=scale, shear=shear)

        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'predict':
            img_t = Image.open(self.img_paths[idx]).convert('RGB')
            img_tp1 = Image.open(self.img_paths[idx + 1]).convert('RGB')

            img_t = self.resize(img_t)
            img_tp1 = self.resize(img_tp1)

            t = TF.to_tensor(img_t)
            tp1 = TF.to_tensor(img_tp1)

            t, tp1 = self._paired_augment(t, tp1)
            return t, tp1
        else:
            # Reconstruction autoencodeur: entrée bruitée -> cible propre
            img = Image.open(self.img_paths[idx]).convert('RGB')
            img = self.resize(img)
            clean = TF.to_tensor(img)
            inp, target = self._paired_augment(clean, clean)
            if self.add_noise:
                noise = torch.randn_like(inp) * self.noise_std
                inp = torch.clamp(inp + noise, 0.0, 1.0)
            return inp, target


# ==================================================
#  BLOCS RÉSEAUX (UNet + attention canal simple)
# ==================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        def _gn(ch: int) -> nn.Module:
            # GroupNorm robuste aux petits batchs
            for g in (32, 16, 8, 4, 2, 1):
                if ch % g == 0:
                    return nn.GroupNorm(g, ch)
            return nn.GroupNorm(1, ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Sequential(
            nn.Conv2d(base, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.Sigmoid(),  # Sortie [0,1]
        )

        # Initialisation stable
        def _init(m: nn.Module) -> None:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        # clamp pour stabilité des sorties
        out = torch.clamp(out, 0.0, 1.0)
        return out


# ==================================================
#  LOSSES non « par pixel » (robustes et perceptuelles locales)
# ==================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = epsilon * epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff2 = (pred - target) ** 2
        return torch.mean(torch.sqrt(diff2 + self.eps2))


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(w: int, sigma: float) -> torch.Tensor:
        gauss = torch.tensor([math.exp(-(x - w // 2) ** 2 / (2 * sigma ** 2)) for x in range(w)])
        return gauss / gauss.sum()

    def _create_window(self, w: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(w, 1.5).unsqueeze(1)
        _2d = (_1d @ _1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channel, 1, w, w).contiguous()
        return window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=x.dtype, device=x.device)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return ssim_map.mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def _gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = img[:, :, 1:, :] - img[:, :, :-1, :]
        dy = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dx, dy

    dx_p, dy_p = _gradients(pred)
    dx_t, dy_t = _gradients(target)
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


class CompositeLoss(nn.Module):
    def __init__(self, w_char: float = 1.0, w_grad: float = 0.2, w_ssim: float = 0.5) -> None:
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim = SSIM()
        self.w_char = w_char
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Empêche saturation totale en blanc/noir
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        l_char = self.charb(pred, target)
        l_grad = gradient_loss(pred, target)
        l_ssim = 1.0 - self.ssim(pred, target)
        # Pénalité légère pour s'éloigner du centre (évite images constantes)
        center_penalty = 0.01 * torch.mean((pred - 0.5) ** 2)
        total = self.w_char * l_char + self.w_grad * l_grad + self.w_ssim * l_ssim + center_penalty
        return total, {
            'char': float(l_char.item()),
            'grad': float(l_grad.item()),
            'ssim': float((1.0 - l_ssim).item()),  # SSIM itself
        }


# ==================================================
#  ENTRAÎNEMENT + VALIDATION + SAUVEGARDE
# ==================================================
def make_loaders(
    frames_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    augment: bool = True,
    mode: str = 'predict',
    add_noise: bool = True,
    noise_std: float = 0.02,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset = NextFrameDataset(
        frames_dir,
        img_size=img_size,
        augment=augment,
        mode=mode,
        add_noise=add_noise,
        noise_std=noise_std,
    )
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == 'cuda'),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(DEVICE == 'cuda'),
        )
    return train_loader, val_loader


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str, best_metric: float) -> None:
    torch.save({
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': float(best_metric),
    }, path)


def load_checkpoint(path: str, map_location: str = DEVICE) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train(
    frames_dir: str,
    out_dir: str = 'outputs',
    img_size: int = 128,
    batch_size: int = 8,
    pre_epochs: int = 20,
    pred_epochs: int = 30,
    pre_lr: float = 2e-4,
    pred_lr: float = 2e-5,
    num_workers: int = 2,
    val_split: float = 0.1,
    amp: bool = True,
    resume: Optional[str] = None,
    pre_noise_std: float = 0.02,
) -> nn.Module:
    os.makedirs(out_dir, exist_ok=True)

    model = UNet(in_ch=3, out_ch=3, base=64).to(DEVICE)
    # Phase 1: reconstruction (autoencoder)
    pre_optimizer = torch.optim.AdamW(model.parameters(), lr=pre_lr, weight_decay=1e-5)
    pre_scheduler = CosineAnnealingLR(pre_optimizer, T_max=max(1, pre_epochs))
    recon_criterion = CompositeLoss(w_char=1.0, w_grad=0.1, w_ssim=0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and DEVICE == 'cuda'))

    # Reprise si demandé (sur phase 2 généralement)
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ Reprise depuis {resume}")

    print(f"Device: {DEVICE}")
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # ============================
    # Phase 1 — Pré-entraînement AE
    # ============================
    pre_train_loader, pre_val_loader = make_loaders(
        frames_dir, img_size, batch_size, num_workers, val_split,
        augment=True, mode='reconstruct', add_noise=True, noise_std=pre_noise_std,
    )
    print(f"Phase 1 (AE): train={len(pre_train_loader.dataset)}" + (f", val={len(pre_val_loader.dataset)}" if pre_val_loader else ""))

    best_pre = float('inf')
    for epoch in range(pre_epochs):
        model.train()
        losses: List[float] = []
        for i, (noisy, clean) in enumerate(pre_train_loader):
            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            pre_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                recon = model(noisy)
                loss, loss_dict = recon_criterion(recon, clean)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(pre_optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if i % 25 == 0:
                print(f"  [AE {epoch+1}/{pre_epochs}] Batch {i}/{len(pre_train_loader)} "
                      f"Loss={loss.item():.5f} (Char={loss_dict['char']:.4f}, Grad={loss_dict['grad']:.4f}, SSIM={loss_dict['ssim']:.4f})")

        pre_scheduler.step()
        avg_train = float(np.mean(losses)) if losses else 0.0
        # Logs de saturation
        if len(losses) > 0:
            with torch.no_grad():
                mn, mx = float(recon.min().item()), float(recon.max().item())
            print(f"    Recon stats[min/max]: {mn:.4f}/{mx:.4f}")
        avg_val = avg_train
        if pre_val_loader is not None and len(pre_val_loader) > 0:
            model.eval()
            vloss: List[float] = []
            with torch.no_grad():
                for noisy, clean in pre_val_loader:
                    noisy = noisy.to(DEVICE, non_blocking=True)
                    clean = clean.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                        recon = model(noisy)
                        l, _ = recon_criterion(recon, clean)
                    vloss.append(float(l.item()))
            avg_val = float(np.mean(vloss)) if vloss else avg_train

        print(f"AE Epoch {epoch+1}/{pre_epochs}: train={avg_train:.6f} val={avg_val:.6f} lr={pre_scheduler.get_last_lr()[0]:.6e}")

        # Sauvegarde meilleur AE
        if avg_val < best_pre:
            best_pre = avg_val
            pre_ckpt = os.path.join(out_dir, 'best_unet_autoencoder.pth')
            save_checkpoint(model, pre_optimizer, epoch, pre_ckpt, best_pre)
            print(f"  ✓ Meilleur AE sauvegardé ({best_pre:.6f}) → {pre_ckpt}")

        # Visualisation AE
        try:
            model.eval()
            with torch.no_grad():
                noisy_v, clean_v = next(iter(pre_train_loader))
                noisy_v = noisy_v.to(DEVICE)[:4]
                clean_v = clean_v.to(DEVICE)[:4]
                with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                    recon_v = model(noisy_v)
                grid = make_grid(torch.cat([noisy_v, recon_v, clean_v], dim=0), nrow=4)
                save_image(grid, os.path.join(out_dir, f'ae_vis_epoch_{epoch+1}.png'))
        except Exception as e:
            print(f"(AE vis) Ignoré: {e}")

    # ============================
    # Phase 2 — Fine-tuning prédiction
    # ============================
    pred_optimizer = torch.optim.AdamW(model.parameters(), lr=pred_lr, weight_decay=1e-5)
    pred_scheduler = CosineAnnealingLR(pred_optimizer, T_max=max(1, pred_epochs))
    pred_criterion = CompositeLoss(w_char=1.0, w_grad=0.2, w_ssim=0.5)

    pred_train_loader, pred_val_loader = make_loaders(
        frames_dir, img_size, batch_size, num_workers, val_split,
        augment=True, mode='predict', add_noise=False,
    )
    print(f"Phase 2 (Pred): train={len(pred_train_loader.dataset)}" + (f", val={len(pred_val_loader.dataset)}" if pred_val_loader else ""))

    best_pred = float('inf')
    for epoch in range(pred_epochs):
        model.train()
        losses: List[float] = []
        for i, (t, tp1) in enumerate(pred_train_loader):
            t = t.to(DEVICE, non_blocking=True)
            tp1 = tp1.to(DEVICE, non_blocking=True)

            pred_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                y = model(t)
                loss, loss_dict = pred_criterion(y, tp1)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(pred_optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if i % 25 == 0:
                print(f"  [Pred {epoch+1}/{pred_epochs}] Batch {i}/{len(pred_train_loader)} "
                      f"Loss={loss.item():.5f} (Char={loss_dict['char']:.4f}, Grad={loss_dict['grad']:.4f}, SSIM={loss_dict['ssim']:.4f})")

        pred_scheduler.step()
        avg_train = float(np.mean(losses)) if losses else 0.0
        if len(losses) > 0:
            with torch.no_grad():
                mn, mx = float(y.min().item()), float(y.max().item())
            print(f"    Pred stats[min/max]: {mn:.4f}/{mx:.4f}")
        avg_val = avg_train
        if pred_val_loader is not None and len(pred_val_loader) > 0:
            model.eval()
            vloss: List[float] = []
            with torch.no_grad():
                for t, tp1 in pred_val_loader:
                    t = t.to(DEVICE, non_blocking=True)
                    tp1 = tp1.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                        y = model(t)
                        l, _ = pred_criterion(y, tp1)
                    vloss.append(float(l.item()))
            avg_val = float(np.mean(vloss)) if vloss else avg_train

        print(f"Pred Epoch {epoch+1}/{pred_epochs}: train={avg_train:.6f} val={avg_val:.6f} lr={pred_scheduler.get_last_lr()[0]:.6e}")

        # Sauvegarde meilleur prédicteur
        if avg_val < best_pred:
            best_pred = avg_val
            pred_ckpt = os.path.join(out_dir, 'best_unet_predictor.pth')
            save_checkpoint(model, pred_optimizer, epoch, pred_ckpt, best_pred)
            print(f"  ✓ Meilleur prédicteur sauvegardé ({best_pred:.6f}) → {pred_ckpt}")

        # Visualisation prédiction
        try:
            model.eval()
            with torch.no_grad():
                t_v, y_v = next(iter(pred_train_loader))
                t_v = t_v.to(DEVICE)[:4]
                y_v = y_v.to(DEVICE)[:4]
                with torch.cuda.amp.autocast(enabled=(amp and DEVICE == 'cuda')):
                    p_v = model(t_v)
                grid = make_grid(torch.cat([t_v, p_v, y_v], dim=0), nrow=4)
                save_image(grid, os.path.join(out_dir, f'pred_vis_epoch_{epoch+1}.png'))
        except Exception as e:
            print(f"(Pred vis) Ignoré: {e}")

    # Dernier checkpoint prédicteur
    last_path = os.path.join(out_dir, 'last_unet_predictor.pth')
    save_checkpoint(model, pred_optimizer, pred_epochs - 1, last_path, best_pred)
    print(f"✓ Entraînement terminé. BestPred={best_pred:.6f}. Dernier checkpoint: {last_path}")

    return model


# ==================================================
#  INFÉRENCE: génération récursive de N frames futures
# ==================================================
@torch.no_grad()
def generate_future(
    model: nn.Module,
    frames_dir: str,
    out_dir: str = 'outputs',
    img_size: int = 128,
    num_future: int = 10,
    start_index: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Dataset sans augmentation pour préparer l'entrée
    ds = NextFrameDataset(frames_dir, img_size=img_size, augment=False, mode='predict', add_noise=False)

    # Détermine l'image de départ (frame_t uniquement, sans wrap vers frame_1)
    if start_index is None:
        img_path = ds.img_paths[-1]
    else:
        idx = max(0, min(start_index - 1, len(ds.img_paths) - 1))
        img_path = ds.img_paths[idx]

    start_img = Image.open(img_path).convert('RGB')
    start_img = ds.resize(start_img)
    x0 = TF.to_tensor(start_img)
    x = x0.unsqueeze(0).to(DEVICE)

    preds = []
    for i in range(num_future):
        y = model(x)
        preds.append(y.cpu())
        x = y  # réinjecte la prédiction comme entrée
        save_image(y, os.path.join(out_dir, f'future_{i+1:03d}.png'))
        y_min, y_mean, y_max = float(y.min().item()), float(y.mean().item()), float(y.max().item())
        print(f"  Généré: future_{i+1:03d}.png  stats[min/mean/max]={y_min:.4f}/{y_mean:.4f}/{y_max:.4f}")

    grid = make_grid(torch.cat(preds, dim=0), nrow=num_future)
    save_image(grid, os.path.join(out_dir, 'future_sequence.png'))
    print("✓ Séquence sauvegardée: future_sequence.png")


# ==================================================
#  MAIN (CLI)
# ==================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='UNet Next-Frame Predictor (non-GAN, pertes perceptuelles locales)')
    parser.add_argument('--frames', type=str, default='frames', help='Dossier contenant frame_*.png')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Dossier de sortie')
    parser.add_argument('--img_size', type=int, default=128, help='Taille des images (carrée)')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille de batch')
    parser.add_argument('--pre_epochs', type=int, default=20, help="Epochs pré-entrainement (autoencodeur)")
    parser.add_argument('--pred_epochs', type=int, default=30, help='Epochs fine-tuning prédiction')
    parser.add_argument('--pre_lr', type=float, default=2e-4, help='LR phase pré-entrainement')
    parser.add_argument('--pred_lr', type=float, default=2e-5, help='LR phase prédiction')
    parser.add_argument('--pre_noise_std', type=float, default=0.02, help='Bruit gaussien std pour AE')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers DataLoader')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction validation [0,1)')
    parser.add_argument('--amp', action='store_true', help='Activer AMP (FP16) pour GPU (P100 OK)')
    parser.add_argument('--resume', type=str, default='', help='Chemin checkpoint pour reprise')
    parser.add_argument('--inference_only', action='store_true', help='Mode inférence uniquement (ne pas entraîner)')
    parser.add_argument('--num_future', type=int, default=10, help='Nombre de frames à prédire en chaîne')
    parser.add_argument('--start_index', type=int, default=None, help='Indice de départ (frame_t) 1-based')

    args = parser.parse_args()

    if args.inference_only:
        print('MODE INFÉRENCE')
        model = UNet(in_ch=3, out_ch=3, base=64).to(DEVICE)
        ckpt_best = os.path.join(args.out_dir, 'best_unet_predictor.pth')
        ckpt_last = os.path.join(args.out_dir, 'last_unet_predictor.pth')
        ckpt_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_last
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"✓ Checkpoint chargé: {ckpt_path}")
        else:
            print('⚠ Aucun checkpoint trouvé, utilisation du modèle non entraîné')

        model.eval()
        generate_future(model, args.frames, out_dir=args.out_dir, img_size=args.img_size, num_future=args.num_future, start_index=args.start_index)
        return

    # Entraînement
    model = train(
        frames_dir=args.frames,
        out_dir=args.out_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        pre_epochs=args.pre_epochs,
        pred_epochs=args.pred_epochs,
        pre_lr=args.pre_lr,
        pred_lr=args.pred_lr,
        num_workers=args.num_workers,
        val_split=args.val_split,
        amp=args.amp,
        resume=args.resume if args.resume else None,
        pre_noise_std=args.pre_noise_std,
    )

    # Génération finale
    model.eval()
    print('\nGÉNÉRATION DE PRÉDICTIONS FUTURES')
    generate_future(model, args.frames, out_dir=args.out_dir, img_size=args.img_size, num_future=args.num_future, start_index=args.start_index)


if __name__ == '__main__':
    main()


