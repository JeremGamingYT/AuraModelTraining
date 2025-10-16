from __future__ import annotations
import os
import json
import math
import random
import warnings
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import plotly.graph_objects as go
import plotly.express as px

# Optimisations
from functools import lru_cache
from tqdm import tqdm
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION AVANCÉE
# ==========================================
@dataclass
class ModelConfig:
    """Configuration optimisée pour GPU 6Go avec architecture Transformer complète"""
    
    # Architecture Transformer
    d_model: int = 256          # Dimension du modèle (augmentée mais optimisée)
    n_heads: int = 8            # Têtes d'attention
    n_encoder_layers: int = 6   # Couches d'encodeur
    n_decoder_layers: int = 4   # Couches de décodeur
    d_ff: int = 1024           # Dimension feedforward
    max_seq_length: int = 512  # Longueur maximale de séquence
    vocab_size: int = 50000    # Taille du vocabulaire pour embeddings
    
    # Optimisations mémoire
    gradient_checkpointing: bool = True  # Pour économiser la VRAM
    mixed_precision: bool = True         # FP16 pour économiser la mémoire
    accumulation_steps: int = 4          # Gradient accumulation
    
    # Régularisation
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    
    # Entraînement
    batch_size: int = 16       # Réduit pour 6Go VRAM
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 100
    patience: int = 10
    
    # Multi-tâches
    n_planet_predictions: int = 10  # Top-K planètes à prédire
    n_star_classes: int = 7         # Classes spectrales
    
    # Chemins
    root_dir: Path = Path("./astroformer_pro")
    model_dir: Path = root_dir / "models"
    data_dir: Path = root_dir / "data"
    viz_dir: Path = root_dir / "visualizations"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers: int = 4
    
    def __post_init__(self):
        """Créer les répertoires nécessaires"""
        for dir_path in [self.root_dir, self.model_dir, self.data_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CHARGEUR DE DONNÉES UNIVERSEL
# ==========================================
class UniversalAstroDataLoader:
    """Chargeur universel pour tous types de données astronomiques JSON"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_cache = {}
        self.feature_extractors = {
            'binary_systems': self._process_binary_systems,
            'exoplanets': self._process_exoplanets,
            'stars': self._process_stars
        }
        
    def load_json_data(self, file_path: Path) -> pd.DataFrame:
        """Charge et parse intelligemment n'importe quel JSON astronomique"""
        logger.info(f"Chargement de {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Détection automatique du type de données
        if 'binary_systems' in data:
            return self._process_binary_systems(data['binary_systems'])
        elif 'exoplanets' in data:
            return self._process_exoplanets(data['exoplanets'])
        elif 'stars' in data:
            return self._process_stars(data['stars'])
        else:
            # Tentative de traitement générique
            return self._process_generic(data)
    
    def _flatten_nested_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Aplatit récursivement les dictionnaires imbriqués"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_nested_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    for i, item in enumerate(v):
                        items.extend(self._flatten_nested_dict(item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _process_binary_systems(self, systems: List[Dict]) -> pd.DataFrame:
        """Traite les systèmes binaires avec extraction de features avancées"""
        processed_data = []
        
        for system in systems:
            flat_system = self._flatten_nested_dict(system)
            
            # Extraction de features physiques clés
            features = {
                'system_id': flat_system.get('system_id', ''),
                'n_components': flat_system.get('system_properties_number_of_components', 2),
                'total_mass': flat_system.get('system_properties_total_mass', np.nan),
                'age': flat_system.get('system_properties_age', np.nan),
                'metallicity': flat_system.get('system_properties_metallicity', np.nan),
                'separation': flat_system.get('orbital_configuration_inner_binary_separation', np.nan),
                'period': flat_system.get('orbital_configuration_inner_binary_period', np.nan),
                'eccentricity': flat_system.get('orbital_configuration_inner_binary_eccentricity', np.nan),
                'mass_a': flat_system.get('component_a_mass', np.nan),
                'mass_b': flat_system.get('component_b_mass', np.nan),
                'radius_a': flat_system.get('component_a_radius', np.nan),
                'radius_b': flat_system.get('component_b_radius', np.nan),
                'teff_a': flat_system.get('component_a_teff', np.nan),
                'teff_b': flat_system.get('component_b_teff', np.nan),
                'has_planets': flat_system.get('planetary_companions_has_planets', False),
                'stability': flat_system.get('stability_stable_configuration', True),
            }
            
            # Calcul de features dérivées
            if not np.isnan(features['mass_a']) and not np.isnan(features['mass_b']):
                features['mass_ratio'] = features['mass_b'] / features['mass_a']
                features['total_calculated_mass'] = features['mass_a'] + features['mass_b']
            
            processed_data.append(features)
        
        return pd.DataFrame(processed_data)
    
    def _process_exoplanets(self, planets: List[Dict]) -> pd.DataFrame:
        """Traite les exoplanètes avec calculs d'habitabilité"""
        df = pd.DataFrame(planets)
        
        # Calculs d'habitabilité avancés
        if 'pl_eqt' in df.columns and 'pl_rade' in df.columns:
            df['habitability_temp_score'] = np.exp(-((df['pl_eqt'] - 288) / 50)**2)
            df['habitability_size_score'] = np.exp(-((df['pl_rade'] - 1) / 0.5)**2)
            df['habitability_total'] = (df['habitability_temp_score'] + df['habitability_size_score']) / 2
        
        # Classification planétaire avancée
        if 'pl_bmasse' in df.columns and 'pl_rade' in df.columns:
            conditions = [
                (df['pl_rade'] < 1.25) & (df['pl_bmasse'] < 2),
                (df['pl_rade'] < 2) & (df['pl_bmasse'] < 10),
                (df['pl_rade'] < 4) & (df['pl_bmasse'] < 50),
                (df['pl_rade'] < 10) & (df['pl_bmasse'] < 500),
            ]
            choices = ['Earth-like', 'Super-Earth', 'Neptune-like', 'Jupiter-like']
            df['planet_class'] = np.select(conditions, choices, default='Giant')
        
        return df
    
    def _process_stars(self, stars: List[Dict]) -> pd.DataFrame:
        """Traite les étoiles avec classification spectrale"""
        df = pd.DataFrame(stars)
        
        # Classification spectrale basée sur la température
        if 'stellar_parameters' in df.columns:
            stellar_params = pd.json_normalize(df['stellar_parameters'])
            df = pd.concat([df, stellar_params], axis=1)
        
        if 'teff' in df.columns:
            conditions = [
                df['teff'] > 30000,
                df['teff'] > 10000,
                df['teff'] > 7500,
                df['teff'] > 6000,
                df['teff'] > 5200,
                df['teff'] > 3700,
            ]
            choices = ['O', 'B', 'A', 'F', 'G', 'K']
            df['spectral_class'] = np.select(conditions, choices, default='M')
        
        return df
    
    def _process_generic(self, data: Any) -> pd.DataFrame:
        """Traitement générique pour données non reconnues"""
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Format de données non supporté: {type(data)}")

# ==========================================
# 3. ARCHITECTURE TRANSFORMER COMPLÈTE
# ==========================================
class MultiHeadAttentionWithCache(nn.Module):
    """Attention multi-têtes optimisée avec cache pour l'inférence"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Cache pour l'inférence
        self.cache_k = None
        self.cache_v = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Projections linéaires
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Utilisation du cache si demandé
        if use_cache and self.cache_k is not None:
            K = torch.cat([self.cache_k, K], dim=2)
            V = torch.cat([self.cache_v, V], dim=2)
            self.cache_k = K
            self.cache_v = V
        
        # Calcul de l'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    """Bloc Transformer avec normalisation pré-activation et connexions résiduelles"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttentionWithCache(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture pour une meilleure stabilité
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal avec support de sequences longues"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AstroFormerPro(nn.Module):
    """Architecture Transformer complète encoder-decoder pour l'astrophysique"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Embedding des features d'entrée
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_length, config.dropout
        )
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])
        
        # Decoder (pour génération)
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_decoder_layers)
        ])
        
        # Têtes de prédiction multi-tâches
        self.planet_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.n_planet_predictions * 8)  # 8 features par planète
        )
        
        self.habitability_scorer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.n_planet_predictions),
            nn.Sigmoid()
        )
        
        self.star_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.n_star_classes)
        )
        
        self.system_stability_predictor = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour une meilleure convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodeur avec gradient checkpointing optionnel"""
        x = self.input_projection(x)
        x = x.unsqueeze(1) if x.dim() == 2 else x  # Ajouter dimension séquence si nécessaire
        x = self.positional_encoding(x)
        
        for layer in self.encoder_layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(lambda inp: layer(inp, mask), x)
            else:
                x = layer(x, mask)
        
        return x
    
    def decode(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Décodeur pour génération"""
        x = encoded
        
        for layer in self.decoder_layers:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(lambda inp: layer(inp, mask), x)
            else:
                x = layer(x, mask)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass complet avec toutes les prédictions"""
        # Encodage
        encoded = self.encode(x)
        
        # Décodage
        decoded = self.decode(encoded)
        
        # Agrégation (mean pooling sur la dimension séquence)
        if decoded.dim() == 3:
            aggregated = decoded.mean(dim=1)
        else:
            aggregated = decoded.squeeze(1) if decoded.dim() == 2 else decoded
        
        # Prédictions multi-tâches
        planets = self.planet_generator(aggregated)
        planets = planets.view(-1, self.config.n_planet_predictions, 8)  # Reshape
        
        habitability = self.habitability_scorer(aggregated)
        star_class = self.star_classifier(aggregated)
        stability = self.system_stability_predictor(aggregated)
        
        return {
            'planets': planets,
            'habitability_scores': habitability,
            'star_classification': star_class,
            'system_stability': stability,
            'encoded_features': encoded
        }

# ==========================================
# 4. SYSTÈME D'ENTRAÎNEMENT OPTIMISÉ
# ==========================================
class OptimizedTrainer:
    """Entraîneur optimisé avec mixed precision et gradient accumulation"""
    
    def __init__(self, model: AstroFormerPro, config: ModelConfig):
        self.model = model.to(config.device)
        self.config = config
        
        # Optimiseur avec warmup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler avec warmup linéaire
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_scheduler(self):
        """Crée un scheduler avec warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Entraînement d'une époque avec optimisations"""
        self.model.train()
        total_loss = 0
        accumulation_counter = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.config.device)
            targets = {k: v.to(self.config.device) for k, v in targets.items()}
            
            # Mixed precision training
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(features)
                    loss = self._compute_loss(outputs, targets)
                    loss = loss / self.config.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets)
                loss = loss / self.config.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item() * self.config.accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calcul de la loss multi-tâches"""
        # Loss pour les planètes générées
        planet_loss = self.mse_loss(outputs['planets'], targets['planets'])
        
        # Loss pour l'habitabilité
        hab_loss = self.bce_loss(outputs['habitability_scores'], targets['habitability'])
        
        # Loss pour la classification stellaire
        star_loss = self.ce_loss(outputs['star_classification'], targets['star_class'])
        
        # Loss pour la stabilité du système
        stability_loss = self.bce_loss(outputs['system_stability'], targets['stability'])
        
        # Combinaison pondérée
        total_loss = (planet_loss * 2.0 + hab_loss * 1.5 + 
                     star_loss * 1.0 + stability_loss * 1.0)
        
        return total_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation avec calcul de métriques"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.config.device)
                targets = {k: v.to(self.config.device) for k, v in targets.items()}
                
                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Boucle d'entraînement complète"""
        logger.info("Début de l'entraînement...")
        
        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping à l'epoch {epoch}")
                    break
            
            # Libération mémoire
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    def save_checkpoint(self, epoch: int):
        """Sauvegarde du checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.config.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint sauvegardé: {path}")

# ==========================================
# 5. SYSTÈME DE VISUALISATION AVANCÉ
# ==========================================
class AstroVisualizer:
    """Système de visualisation scientifique pour les prédictions"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        sns.set_style("darkgrid")
        plt.style.use('seaborn-darkgrid')
    
    def plot_habitable_zone(self, star_data: Dict, predicted_planets: np.ndarray, 
                           save_path: Optional[Path] = None):
        """Visualise la zone habitable avec les planètes prédites"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calcul de la zone habitable
        star_lum = star_data.get('luminosity', 1.0)
        hz_inner = 0.95 * np.sqrt(star_lum)
        hz_outer = 1.37 * np.sqrt(star_lum)
        
        # Zone habitable
        hz_rect = Rectangle((hz_inner, -2), hz_outer - hz_inner, 4, 
                           alpha=0.3, facecolor='green', label='Zone Habitable')
        ax.add_patch(hz_rect)
        
        # Étoile
        star_circle = Circle((0, 0), 0.1, color='yellow', label='Étoile')
        ax.add_patch(star_circle)
        
        # Planètes prédites
        for i, planet in enumerate(predicted_planets):
            distance = planet[0]  # Distance orbitale
            radius = planet[1] * 0.05  # Rayon (mise à l'échelle)
            habitability = planet[7]  # Score d'habitabilité
            
            color = plt.cm.RdYlGn(habitability)
            planet_circle = Circle((distance, 0), radius, color=color, 
                                  label=f'Planète {i+1}')
            ax.add_patch(planet_circle)
            
            # Orbite
            orbit = Circle((0, 0), distance, fill=False, linestyle='--', 
                         alpha=0.3, color='gray')
            ax.add_patch(orbit)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (UA)')
        ax.set_ylabel('Distance (UA)')
        ax.set_title(f'Système Planétaire Prédit - Étoile: {star_data.get("name", "Unknown")}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_3d_system(self, star_data: Dict, planets: np.ndarray, 
                      save_path: Optional[Path] = None):
        """Visualisation 3D interactive du système planétaire"""
        fig = go.Figure()
        
        # Étoile centrale
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=20, color='yellow'),
            name='Étoile'
        ))
        
        # Planètes avec orbites
        for i, planet in enumerate(planets):
            # Paramètres orbitaux
            a = planet[0]  # Semi-major axis
            e = planet[2]  # Eccentricity
            inc = planet[3]  # Inclination
            
            # Génération de l'orbite
            theta = np.linspace(0, 2*np.pi, 100)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta) * np.cos(np.radians(inc))
            z = r * np.sin(theta) * np.sin(np.radians(inc))
            
            # Orbite
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=2, color=f'rgba(100,100,255,0.3)'),
                name=f'Orbite {i+1}',
                showlegend=False
            ))
            
            # Planète
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(
                    size=5 + planet[1] * 2,  # Taille basée sur le rayon
                    color=planet[7],  # Couleur basée sur l'habitabilité
                    colorscale='RdYlGn',
                    showscale=True
                ),
                name=f'Planète {i+1}'
            ))
        
        fig.update_layout(
            title=f'Système Planétaire 3D - {star_data.get("name", "Unknown")}',
            scene=dict(
                xaxis_title='X (UA)',
                yaxis_title='Y (UA)',
                zaxis_title='Z (UA)',
                aspectmode='cube'
            ),
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
        return fig
    
    def plot_planet_properties(self, planets: np.ndarray, save_path: Optional[Path] = None):
        """Graphique des propriétés planétaires prédites"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Masse vs Rayon
        axes[0, 0].scatter(planets[:, 4], planets[:, 1], 
                          c=planets[:, 7], cmap='RdYlGn', s=100)
        axes[0, 0].set_xlabel('Masse (M⊕)')
        axes[0, 0].set_ylabel('Rayon (R⊕)')
        axes[0, 0].set_title('Relation Masse-Rayon')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        
        # Distribution des périodes
        axes[0, 1].hist(planets[:, 5], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Période Orbitale (jours)')
        axes[0, 1].set_ylabel('Nombre de Planètes')
        axes[0, 1].set_title('Distribution des Périodes')
        
        # Habitabilité vs Distance
        axes[1, 0].scatter(planets[:, 0], planets[:, 7], s=100, alpha=0.6)
        axes[1, 0].set_xlabel('Distance Orbitale (UA)')
        axes[1, 0].set_ylabel('Score d\'Habitabilité')
        axes[1, 0].set_title('Habitabilité en Fonction de la Distance')
        axes[1, 0].axhspan(0.7, 1.0, alpha=0.2, color='green', label='Zone Optimale')
        
        # Température d'équilibre
        axes[1, 1].bar(range(len(planets)), planets[:, 6], color='orange')
        axes[1, 1].axhline(y=288, color='blue', linestyle='--', label='Temp. Terre')
        axes[1, 1].set_xlabel('Index Planète')
        axes[1, 1].set_ylabel('Température (K)')
        axes[1, 1].set_title('Températures d\'Équilibre')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_report(self, star_data: Dict, predictions: Dict, 
                                   save_dir: Optional[Path] = None):
        """Génère un rapport complet avec toutes les visualisations"""
        if save_dir is None:
            save_dir = self.config.viz_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extraction des planètes prédites
        planets = predictions['planets'].cpu().numpy()
        
        # 1. Zone habitable
        self.plot_habitable_zone(star_data, planets, save_dir / 'habitable_zone.png')
        
        # 2. Système 3D
        self.plot_3d_system(star_data, planets, save_dir / 'system_3d.html')
        
        # 3. Propriétés planétaires
        self.plot_planet_properties(planets, save_dir / 'planet_properties.png')
        
        # 4. Rapport textuel
        report_text = self._generate_text_report(star_data, predictions)
        with open(save_dir / 'report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info(f"Rapport complet généré dans: {save_dir}")
        return save_dir
    
    def _generate_text_report(self, star_data: Dict, predictions: Dict) -> str:
        """Génère un rapport textuel détaillé"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RAPPORT D'ANALYSE ASTROPHYSIQUE                   ║
╚══════════════════════════════════════════════════════════════════════╝

    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ▓▓▓ DONNÉES STELLAIRES ▓▓▓
    Nom: {star_data.get('name', 'Unknown')}
    Type Spectral: {star_data.get('spectral_type', 'G2V')}
    Masse: {star_data.get('mass', 1.0):.2f} M☉
    Rayon: {star_data.get('radius', 1.0):.2f} R☉
    Temperature: {star_data.get('teff', 5778):.0f} K
    Luminosité: {star_data.get('luminosity', 1.0):.2f} L☉
    Âge: {star_data.get('age', 4.6):.1f} Gyr
    Métallicité: {star_data.get('metallicity', 0.0):.2f}

    ▓▓▓ SYSTÈME PLANÉTAIRE PRÉDIT ▓▓▓
    Nombre de planètes détectées: {len(predictions['planets'])}
    Stabilité du système: {predictions['system_stability'].item():.1%}

"""
        
        planets = predictions['planets'].cpu().numpy()
        hab_scores = predictions['habitability_scores'].cpu().numpy()
        
        for i, (planet, hab_score) in enumerate(zip(planets, hab_scores)):
            report += f"""
┌─────────────────────────────────────────────────────────────────────┐
│ PLANÈTE {i+1:02d}                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Distance Orbitale: {planet[0]:.3f} UA                                │
│ Rayon: {planet[1]:.2f} R⊕                                           │
│ Masse: {planet[4]:.2f} M⊕                                           │
│ Période: {planet[5]:.1f} jours                                      │
│ Température d'équilibre: {planet[6]:.0f} K                         │
│ Score d'habitabilité: {hab_score:.1%}                              │
│ Classification: {self._classify_planet(planet)}                    │
└─────────────────────────────────────────────────────────────────────┘
"""
        
        # Top 3 des planètes les plus habitables
        top_hab_indices = np.argsort(hab_scores)[-3:][::-1]
        report += "\n▓▓▓ TOP 3 PLANÈTES HABITABLES ▓▓▓\n"
        for rank, idx in enumerate(top_hab_indices, 1):
            report += f"{rank}. Planète {idx+1} - Score: {hab_scores[idx]:.1%}\n"
        
        return report
    
    def _classify_planet(self, planet: np.ndarray) -> str:
        """Classifie une planète selon ses propriétés"""
        mass = planet[4]
        radius = planet[1]
        
        if mass < 2 and radius < 1.5:
            return "Terrestre (Type Terre)"
        elif mass < 10 and radius < 2.5:
            return "Super-Terre"
        elif mass < 50 and radius < 6:
            return "Mini-Neptune"
        elif mass < 500 and radius < 15:
            return "Géante Gazeuse (Type Jupiter)"
        else:
            return "Super-Jupiter"

# ==========================================
# 6. SYSTÈME DE PRÉDICTION
# ==========================================
class AstroPredictor:
    """Système de prédiction avec support des top-K planètes"""
    
    def __init__(self, model: AstroFormerPro, config: ModelConfig):
        self.model = model
        self.config = config
        self.visualizer = AstroVisualizer(config)
        
    def predict_system(self, star_features: torch.Tensor, 
                      generate_report: bool = True) -> Dict[str, Any]:
        """Prédit un système planétaire complet avec top-K planètes"""
        self.model.eval()
        
        with torch.no_grad():
            if star_features.dim() == 1:
                star_features = star_features.unsqueeze(0)
            
            star_features = star_features.to(self.config.device)
            predictions = self.model(star_features)
        
        # Post-traitement des prédictions
        planets = predictions['planets'][0].cpu().numpy()
        
        # Tri par habitabilité
        hab_scores = predictions['habitability_scores'][0].cpu().numpy()
        sorted_indices = np.argsort(hab_scores)[::-1]
        
        # Sélection des top-K planètes
        top_planets = planets[sorted_indices[:self.config.n_planet_predictions]]
        top_hab_scores = hab_scores[sorted_indices[:self.config.n_planet_predictions]]
        
        # Classification stellaire
        star_class_logits = predictions['star_classification'][0].cpu()
        star_class = torch.argmax(star_class_logits).item()
        star_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        
        result = {
            'planets': top_planets,
            'habitability_scores': top_hab_scores,
            'star_class': star_types[star_class],
            'system_stability': predictions['system_stability'][0].item(),
            'raw_predictions': predictions
        }
        
        # Génération du rapport si demandé
        if generate_report:
            star_data = {
                'name': 'System_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                'spectral_type': star_types[star_class],
                'mass': 1.0,  # À extraire des features
                'radius': 1.0,
                'teff': 5778,
                'luminosity': 1.0,
                'age': 4.6,
                'metallicity': 0.0
            }
            
            report_dir = self.visualizer.create_comprehensive_report(
                star_data, predictions
            )
            result['report_directory'] = str(report_dir)
        
        return result
    
    def find_earth_like_planets(self, predictions: Dict[str, np.ndarray]) -> List[int]:
        """Identifie les planètes similaires à la Terre"""
        planets = predictions['planets']
        earth_like = []
        
        for i, planet in enumerate(planets):
            # Critères pour une planète de type Terre
            if (0.8 < planet[1] < 1.5 and  # Rayon
                0.5 < planet[4] < 2.0 and  # Masse
                250 < planet[6] < 320 and  # Température
                predictions['habitability_scores'][i] > 0.7):  # Habitabilité
                earth_like.append(i)
        
        return earth_like

# ==========================================
# 7. DATASET PERSONNALISÉ
# ==========================================
class AstroDataset(Dataset):
    """Dataset PyTorch pour données astronomiques avec augmentation"""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                augment: bool = False):
        self.features = torch.FloatTensor(features)
        self.targets = {
            'planets': torch.FloatTensor(targets['planets']),
            'habitability': torch.FloatTensor(targets['habitability']),
            'star_class': torch.LongTensor(targets['star_class']),
            'stability': torch.FloatTensor(targets['stability'])
        }
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        
        # Augmentation des données si activée
        if self.augment and random.random() > 0.5:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        targets = {k: v[idx] for k, v in self.targets.items()}
        return features, targets

# ==========================================
# 8. PIPELINE PRINCIPAL
# ==========================================
class AstroFormerPipeline:
    """Pipeline complet pour l'entraînement et l'inférence"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_loader = UniversalAstroDataLoader(config)
        self.model = None
        self.trainer = None
        self.predictor = None
        self.preprocessor = None
        
    def prepare_data(self, json_files: List[Path]) -> Tuple[np.ndarray, Dict]:
        """Prépare les données pour l'entraînement"""
        logger.info("Chargement et préparation des données...")
        
        # Chargement des données
        dfs = []
        for file in json_files:
            df = self.data_loader.load_json_data(file)
            dfs.append(df)
        
        # Fusion des données
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Données combinées: {combined_df.shape}")
        
        # Prétraitement
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        features = combined_df[numeric_cols].values
        
        # Imputation et normalisation
        imputer = KNNImputer(n_neighbors=5)
        features = imputer.fit_transform(features)
        
        scaler = RobustScaler()
        features = scaler.fit_transform(features)
        
        self.preprocessor = {
            'imputer': imputer,
            'scaler': scaler,
            'columns': numeric_cols.tolist()
        }
        
        # Génération des cibles synthétiques pour l'entraînement
        n_samples = len(features)
        targets = self._generate_synthetic_targets(n_samples)
        
        return features, targets
    
    def _generate_synthetic_targets(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Génère des cibles synthétiques réalistes"""
        np.random.seed(42)
        
        # Génération de planètes (10 par système, 8 features chacune)
        planets = []
        for _ in range(n_samples):
            system_planets = []
            for p in range(self.config.n_planet_predictions):
                # [distance, rayon, eccentricité, inclinaison, masse, période, temp, densité]
                planet = [
                    np.random.lognormal(0, 1),  # Distance
                    np.random.lognormal(0, 0.5),  # Rayon
                    np.random.beta(2, 5),  # Eccentricité
                    np.random.normal(90, 30),  # Inclinaison
                    np.random.lognormal(0, 1),  # Masse
                    np.random.lognormal(2, 1),  # Période
                    np.random.normal(300, 100),  # Température
                    np.random.lognormal(1, 0.5)  # Densité
                ]
                system_planets.append(planet)
            planets.append(system_planets)
        
        planets = np.array(planets)
        
        # Scores d'habitabilité
        habitability = np.random.beta(2, 5, (n_samples, self.config.n_planet_predictions))
        
        # Classes stellaires
        star_class = np.random.randint(0, self.config.n_star_classes, n_samples)
        
        # Stabilité du système
        stability = np.random.beta(5, 2, (n_samples, 1))
        
        return {
            'planets': planets,
            'habitability': habitability,
            'star_class': star_class,
            'stability': stability
        }
    
    def train(self, json_files: List[Path]):
        """Entraîne le modèle"""
        # Préparation des données
        features, targets = self.prepare_data(json_files)
        
        # Division train/val/test
        indices = np.arange(len(features))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        
        # Création des datasets
        train_dataset = AstroDataset(
            features[train_idx], 
            {k: v[train_idx] for k, v in targets.items()},
            augment=True
        )
        val_dataset = AstroDataset(
            features[val_idx],
            {k: v[val_idx] for k, v in targets.items()},
            augment=False
        )
        test_dataset = AstroDataset(
            features[test_idx],
            {k: v[test_idx] for k, v in targets.items()},
            augment=False
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.n_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            pin_memory=True
        )
        
        # Création du modèle
        input_dim = features.shape[1]
        self.model = AstroFormerPro(self.config, input_dim)
        
        # Entraînement
        self.trainer = OptimizedTrainer(self.model, self.config)
        self.trainer.train(train_loader, val_loader)
        
        # Évaluation finale
        test_loss = self.trainer.validate(test_loader)
        logger.info(f"Loss finale sur le test set: {test_loss:.4f}")
        
        # Sauvegarde
        self.save_model()
        
    def save_model(self):
        """Sauvegarde le modèle et les preprocessors"""
        model_path = self.config.model_dir / 'astroformer_pro.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'preprocessor': self.preprocessor
        }, model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")
    
    def load_model(self, model_path: Path):
        """Charge un modèle pré-entraîné"""
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        self.preprocessor = checkpoint['preprocessor']
        input_dim = len(self.preprocessor['columns'])
        
        self.model = AstroFormerPro(self.config, input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        self.model.eval()
        
        self.predictor = AstroPredictor(self.model, self.config)
        logger.info("Modèle chargé avec succès")
    
    def predict(self, star_data: Dict[str, float]) -> Dict[str, Any]:
        """Fait une prédiction pour une étoile donnée"""
        if self.predictor is None:
            raise ValueError("Modèle non chargé. Utilisez load_model() d'abord.")
        
        # Préparation des features
        df = pd.DataFrame([star_data])
        df = df.reindex(columns=self.preprocessor['columns'], fill_value=0)
        
        features = self.preprocessor['imputer'].transform(df.values)
        features = self.preprocessor['scaler'].transform(features)
        features = torch.FloatTensor(features)
        
        # Prédiction
        result = self.predictor.predict_system(features, generate_report=True)
        
        return result

# ==========================================
# 9. EXEMPLE D'UTILISATION
# ==========================================
def main():
    """Fonction principale de démonstration"""
    # Configuration
    config = ModelConfig()
    
    # Pipeline
    pipeline = AstroFormerPipeline(config)
    
    # Création de données d'exemple
    logger.info("Création de données d'exemple...")
    
    # Simule des fichiers JSON
    sample_data = {
        'stars': [
            {
                'source_id': f'star_{i}',
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'parallax': np.random.uniform(1, 100),
                'stellar_parameters': {
                    'teff': np.random.uniform(3000, 8000),
                    'logg': np.random.uniform(3, 5),
                    'metallicity_feh': np.random.uniform(-1, 0.5),
                    'mass': np.random.uniform(0.5, 2),
                    'radius': np.random.uniform(0.5, 2),
                    'luminosity': np.random.uniform(0.1, 10)
                }
            }
            for i in range(1000)
        ]
    }
    
    # Sauvegarde temporaire
    temp_file = config.data_dir / 'sample_stars.json'
    with open(temp_file, 'w') as f:
        json.dump(sample_data, f)
    
    # Entraînement
    logger.info("Début de l'entraînement...")
    pipeline.train([temp_file])
    
    # Prédiction sur une nouvelle étoile
    logger.info("Test de prédiction...")
    
    test_star = {
        'teff': 5778,  # Comme le Soleil
        'logg': 4.44,
        'metallicity_feh': 0.0,
        'mass': 1.0,
        'radius': 1.0,
        'luminosity': 1.0,
        'ra': 0.0,
        'dec': 0.0,
        'parallax': 10.0
    }
    
    # Chargement du modèle
    model_path = config.model_dir / 'astroformer_pro.pt'
    if model_path.exists():
        pipeline.load_model(model_path)
        
        # Prédiction
        result = pipeline.predict(test_star)
        
        logger.info("Prédiction terminée!")
        logger.info(f"Nombre de planètes prédites: {len(result['planets'])}")
        logger.info(f"Meilleur score d'habitabilité: {result['habitability_scores'].max():.2%}")
        logger.info(f"Classe stellaire prédite: {result['star_class']}")
        logger.info(f"Stabilité du système: {result['system_stability']:.2%}")
        
        # Identification des planètes de type Terre
        earth_like = pipeline.predictor.find_earth_like_planets(result)
        if earth_like:
            logger.info(f"Planètes similaires à la Terre trouvées: {earth_like}")
        
        logger.info(f"Rapport complet généré dans: {result.get('report_directory', 'N/A')}")

if __name__ == "__main__":
    main()