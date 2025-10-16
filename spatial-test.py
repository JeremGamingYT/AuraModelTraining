from __future__ import annotations
import os, json, math, random, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

import joblib, yaml
warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
@dataclass
class Config:
    # ML
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 300
    BATCH_SIZE: int = 128
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128,64,32])
    DROPOUT_RATE: float = 0.35
    WEIGHT_DECAY: float = 3e-4
    GRAD_CLIP_NORM: float = 1.0
    PATIENCE: int = 15
    USE_AMP: bool = True
    # Regularisation avancée
    MIXUP_ALPHA: float = 0.40
    FEATURE_NOISE_STD: float = 0.01
    LABEL_SMOOTHING: float = 0.05

    # Physique / BPI
    ALBEDO_DEFAULT: float = 0.30
    TEMP_TARGET_K: float = 288.0            # cible Terre
    TEMP_HABITABLE_MIN: float = 273.0
    TEMP_HABITABLE_MAX: float = 373.0
    GREENHOUSE_K_BASE: float = 0.15         # valeur de départ
    GREENHOUSE_AUTO_CALIBRATE: bool = True  # calibrer k pour que T_surf(1 AU, A=0.3)=288 K
    SOFT_HZ_WIDTH: float = 0.03             # largeur logistique en AU (transition HZ)
    FORCE_PHYSICS_MR: bool = False          # fallback relation M-R si pas de modèle

    # Chemins
    ROOT: Path = Path(".")
    OUT_DIR: Path = ROOT / "outputs"
    MODEL_PATH: Path = ROOT / "models/predictor.pth"
    SCALER_PATH: Path = ROOT / "models/scaler.gz"
    DATA_PATH: Path = ROOT / "data/comprehensive_exoplanet_database.json"  # CSV ou JSON (auto-détection)
    CONFIG_PATH: Path = ROOT / "config.yaml"
    CURVES_PATH: Path = OUT_DIR / "training_curves.png"
    ANALYSIS_JSON: Path = OUT_DIR / "oracle_analysis_results.json"
    FIG_FINAL: Path = OUT_DIR / "final_fig.png"
    FIG_SCENARIO_HEATMAP: Path = OUT_DIR / "earth_scenarios_heatmap.png"
    FIG_SCENARIO_HIST: Path = OUT_DIR / "earth_scenarios_hist.png"
    FIG_EARTH_TIMELINE: Path = OUT_DIR / "earth_future_timeline.png"
    EARTH_TIMELINE_JSON: Path = OUT_DIR / "earth_future_timeline.json"

    RANDOM_SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    VERBOSE: bool = True

    @classmethod
    def load_from_file(cls, path: Optional[Path] = None) -> "Config":
        inst = cls()
        cfg_path = path or inst.CONFIG_PATH
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)
        inst.OUT_DIR.mkdir(parents=True, exist_ok=True)
        inst.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        inst.SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        inst.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        return inst


# ==========================================
# CONSTANTES
# ==========================================
@dataclass(frozen=True)
class PhysicsConstants:
    sigma: float = 5.670374419e-8
    solar_mass: float = 1.98847e30
    solar_radius: float = 6.957e8
    solar_luminosity: float = 3.828e26
    earth_mass_kg: float = 5.972e24
    earth_radius_m: float = 6.371e6
    au_m: float = 1.495978707e11


# ==========================================
# UTILS
# ==========================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2)))


# ==========================================
# PHYSICS
# ==========================================
class PhysicsCalculator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.c = PhysicsConstants()

    def stellar_radius_from_L_T(self, luminosity_log: float, T_star: float) -> float:
        L_star = (10.0 ** luminosity_log) * self.c.solar_luminosity
        return math.sqrt(L_star / (4 * math.pi * self.c.sigma * (T_star ** 4)))

    def calculate_equilibrium_temperature(self, star: Dict, distance_au: float, albedo: float) -> float:
        T_star = float(star["temperature_k"])
        R_star = self.stellar_radius_from_L_T(star["luminosity_log"], T_star)
        d_m = distance_au * self.c.au_m
        return float(T_star * ((1.0 - albedo) ** 0.25) * math.sqrt(R_star / (2.0 * d_m)))

    def greenhouse_surface_temperature(self, T_eq: float, k: float) -> float:
        k = max(0.0, k)
        return float(T_eq * ((1.0 + k) ** 0.25))

    def greenhouse_k_for_target(self, T_eq: float, T_target: float) -> float:
        # Trouver k pour obtenir T_target à partir de T_eq dans le modèle T_surf = T_eq * (1+k)^(1/4)
        ratio = max(1e-6, T_target / max(1e-6, T_eq))
        return max(0.0, ratio**4 - 1.0)

    def habitable_zone_bounds(self, luminosity_log: float) -> Tuple[float, float]:
        L_rel = 10.0 ** luminosity_log
        return (math.sqrt(L_rel / 1.1), math.sqrt(L_rel / 0.53))

    def soft_in_hz(self, d: float, hz_in: float, hz_out: float, w: float) -> float:
        # 1 à l'intérieur, ~0 à l'extérieur, transition douce
        left  = 1.0 / (1.0 + math.exp((hz_in - d) / max(1e-6, w)))
        right = 1.0 / (1.0 + math.exp((d - hz_out) / max(1e-6, w)))
        return float(left * right)

    def orbital_period_years(self, distance_au: float, mass_star_solar: float) -> float:
        return float(math.sqrt((distance_au ** 3) / max(1e-8, mass_star_solar)))


# ==========================================
# RENDERING (cartes thermiques + rendu sphère)
# ==========================================
class PlanetSurfaceRenderer:
    def __init__(self, cfg: Config, physics: PhysicsCalculator):
        self.cfg = cfg
        self.physics = physics
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)

    # --- 1) Modèle thermique simple mais physique ---
    def latitude_ebm(self, T_eq: float, k_greenhouse: float = 0.2,
                      albedo_surface: float = 0.3,
                      diffusion: float = 0.35,
                      obliquity_deg: float = 23.5,
                      n_lat: int = 256) -> np.ndarray:
        """
        EBM 1D en latitude (diffusion linéaire). T_eq est l'équilibre global.
        On résout itératif: (1-A)*S(phi) - sigma*T^4 + D*∇²T = 0,
        simplifié/normalisé autour de T_eq et d'un paramètre de redistribution.
        """
        sigma = PhysicsConstants().sigma
        phi = np.linspace(-np.pi/2, np.pi/2, n_lat)
        # Insolation moyenne journalière simplifiée en fonction de la latitude et obliquité:
        eps = np.deg2rad(obliquity_deg)
        Insol = np.clip(np.cos(phi)**0.5 * (0.9 + 0.1*np.cos(2*(phi-eps))), 0.05, None)
        Insol /= Insol.mean()  # normaliser

        # Cible radiative avec serre
        T_target = T_eq * (1.0 + max(0.0, k_greenhouse))**0.25

        T = np.full(n_lat, T_target, dtype=np.float64)
        for _ in range(300):
            # forcing radiatif local
            F = (1.0 - albedo_surface) * Insol
            # linearisons autour de T
            rad = sigma * T**4
            # diffusion latitudinale
            lap = np.zeros_like(T)
            lap[1:-1] = T[:-2] - 2*T[1:-1] + T[2:]
            lap[0] = T[1] - T[0]
            lap[-1] = T[-2] - T[-1]
            # mise à jour (relaxation)
            T_new = T + 0.1 * (F - rad/rad.mean() + diffusion * lap)
            # clamp physique
            T = np.clip(T_new, 100.0, 800.0)
        return T  # en Kelvin

    # --- 2) Carte equirectangulaire (lon/lat) ---
    def make_temperature_map(self, star: dict, distance_au: float,
                             albedo: float = 0.3, greenhouse_k: float = 0.2,
                             obliquity_deg: float = 23.5,
                             w: int = 1024, h: int = 512) -> np.ndarray:
        T_eq = self.physics.calculate_equilibrium_temperature(star, distance_au, albedo)
        T_lat = self.latitude_ebm(T_eq, k_greenhouse=greenhouse_k,
                                  albedo_surface=albedo,
                                  obliquity_deg=obliquity_deg, n_lat=h)
        # day/night léger: add une modulation longitudinale douce (redistribution)
        lon = np.linspace(0, 2*np.pi, w, endpoint=False)
        day_boost = (0.65 + 0.35*np.cos(lon - np.pi/6.0))  # max ~ jour subi
        temp_map = np.outer(T_lat, day_boost)
        return temp_map  # [h, w] en Kelvin

    # --- 3) Procedural "realistic" surface (relief + biomes) ---
    def _fbm(self, x, y, octaves=6):
        val = np.zeros_like(x)
        amp, freq = 1.0, 1.0
        for _ in range(octaves):
            # bruit pseudo: sin/cos mix pour éviter dépendances externes
            val += amp * (np.sin(x*freq) * np.cos(y*freq))
            amp *= 0.5; freq *= 2.0
        return (val - val.min()) / (val.max() - val.min() + 1e-9)

    def make_surface_albedo_map(self, w=1024, h=512, seed_shift=0.0):
        y, x = np.mgrid[0:h, 0:w]
        x = (x / w) * 2*np.pi + seed_shift
        y = (y / h) * np.pi - np.pi/2
        height = self._fbm(x, y, octaves=7)
        # continents / océans
        sea_level = 0.50
        land = (height > sea_level).astype(np.float32)
        # biomes grossiers via latitude et "humid"
        humid = self._fbm(x+1.37, y+2.41, octaves=5)
        lat_fac = np.cos(y)**0.5
        albedo = 0.06 + 0.04*(1-land)  # océans ~0.06-0.1
        # terres: glace hautes latitudes, désert zones sèches, forêts zones humides
        ice = ((np.abs(y) > 1.10) * land).astype(np.float32)
        desert = ((humid < 0.35) * land * lat_fac).astype(np.float32)
        forest = ((humid >= 0.35) * land).astype(np.float32)
        albedo += 0.35*ice + 0.18*desert + 0.12*forest
        return albedo.clip(0.03, 0.85), dict(land=land, ice=ice, desert=desert, forest=forest)

    # --- 4) Colorisation et rendu sphérique ---
    def colorize_temperature(self, temp_map: np.ndarray) -> np.ndarray:
        # palette "physique": glace -> tempéré -> désert/brûlant
        T = temp_map
        tmin, tmax = 180.0, 420.0
        z = ((T - tmin) / (tmax - tmin)).clip(0, 1)
        # simple gradient (bleu -> vert -> ocre -> rouge)
        c = np.zeros((T.shape[0], T.shape[1], 3), dtype=np.float32)
        # segments
        c[z <= 0.33, 2] = 1.0  # bleu
        mid = (z > 0.33) & (z <= 0.66)
        c[mid, 1] = 1.0        # vert
        hot = z > 0.66
        c[hot, 0] = 1.0        # rouge
        # lissage
        c[..., 1] += np.clip((z-0.0)*1.2, 0, 1)*0.2
        c[..., 2] += np.clip((0.4-z)*1.2, 0, 1)*0.2
        return c.clip(0,1)

    def classify_climate(self, temp_map: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, Tuple[float,float], Tuple[float,float,float]]]]:
        """Discrétise la carte thermique en classes climatiques simples.
        Retourne (map_classes[h,w], legend[(name, (tmin,tmax), rgb0..1)]).
        """
        # Classes (K): <220 glaciaire, 220-260 polaire, 260-300 tempéré, 300-340 aride, >340 extrême
        classes = [
            ("Glaciaire", (-np.inf, 220.0), (0.80, 0.92, 1.00)),
            ("Polaire", (220.0, 260.0), (0.65, 0.80, 0.97)),
            ("Tempéré", (260.0, 300.0), (0.55, 0.80, 0.45)),
            ("Aride", (300.0, 340.0), (0.86, 0.66, 0.39)),
            ("Extrême", (340.0, np.inf), (0.80, 0.15, 0.10)),
        ]
        lab = np.zeros_like(temp_map, dtype=np.int32)
        for idx, (_, (lo, hi), _) in enumerate(classes):
            mask = (temp_map >= lo) & (temp_map < hi)
            lab[mask] = idx
        return lab, classes

    def colorize_climate(self, class_map: np.ndarray, classes: List[Tuple[str, Tuple[float,float], Tuple[float,float,float]]]) -> np.ndarray:
        h, w = class_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for idx, (_, _, col) in enumerate(classes):
            rgb[class_map == idx] = col
        return rgb

    def save_temperature_with_colorbar(self, path: Path, temp_map: np.ndarray):
        plt.figure(figsize=(10,4))
        im = plt.imshow(temp_map, origin="lower", cmap="inferno")
        cbar = plt.colorbar(im)
        cbar.set_label("Température (K)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def shade_sphere(self, tex: np.ndarray, w=1024, h=1024, light_dir=(1, -0.2, 0.8)):
        # map equirect -> sphère + ombrage Lambert
        H, W = tex.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx / w) * 2*np.pi - np.pi
        ny = (yy / h) * np.pi - np.pi/2
        # coords sphère
        X = np.cos(ny)*np.cos(nx)
        Y = np.sin(ny)
        Z = np.cos(ny)*np.sin(nx)
        # indices tex
        u = ((nx + np.pi) / (2*np.pi) * W).astype(int) % W
        v = ((ny + np.pi/2) / np.pi * H).astype(int).clip(0, H-1)
        img = tex[v, u]
        # éclairage
        L = np.array(light_dir, dtype=np.float32)
        L = L / (np.linalg.norm(L) + 1e-9)
        NdotL = (X*L[0] + Y*L[1] + Z*L[2]).clip(0,1)[..., None]
        # term ambiant + diffuse
        shaded = img * (0.25 + 0.75*NdotL)
        # vignette/terminateur doux
        mask = (X**2+Y**2+Z**2) <= 1.0
        out = np.ones((h, w, 3), dtype=np.float32)
        out[:] = 0.0
        out[mask] = shaded[mask]
        return out.clip(0,1)

    def export_planet_images(self, out_dir: Path, planet_id: str,
                             star: dict, distance_au: float,
                             base_albedo: float = 0.30, greenhouse_k: float = 0.20,
                             obliquity_deg: float = 23.5):
        out_dir.mkdir(parents=True, exist_ok=True)

        # carte albédo procédurale (pour la "surface réaliste")
        alb_map, layers = self.make_surface_albedo_map(w=2048, h=1024)
        # carte de températures avec EBM + modulation jour/nuit
        temp_map = self.make_temperature_map(star, distance_au,
                                             albedo=float(base_albedo),
                                             greenhouse_k=float(greenhouse_k),
                                             obliquity_deg=float(obliquity_deg),
                                             w=alb_map.shape[1], h=alb_map.shape[0])

        # colorisation thermique et rendu sphère
        temp_rgb = self.colorize_temperature(temp_map)
        sphere_temp = self.shade_sphere(temp_rgb, w=1600, h=1600)

        # rendu "surface réaliste": on teinte par biomes, glace, désert…
        # on part d'un vert/ocre/bleu basé sur l'albédo + latitude
        H, W = alb_map.shape
        y = (np.arange(H)/H)[:, None]
        biome_rgb = np.dstack([
            0.8*alb_map + 0.2*(1-y),     # R
            0.7*(1-alb_map) + 0.3*y,     # G
            0.6*(1-alb_map)              # B
        ]).clip(0,1)
        # + neige où T < 260K
        snow = (temp_map < 260).astype(np.float32)
        biome_rgb = (biome_rgb*(1 - 0.7*snow[...,None]) + 0.7*snow[...,None]).clip(0,1)

        sphere_surface = self.shade_sphere(biome_rgb, w=1600, h=1600)

        # Sauvegardes
        plt.imsave(out_dir / f"{planet_id}_thermal_map.png", temp_rgb)
        plt.imsave(out_dir / f"{planet_id}_surface_map.png", biome_rgb)
        plt.imsave(out_dir / f"{planet_id}_thermal_sphere.png", sphere_temp)
        plt.imsave(out_dir / f"{planet_id}_surface_sphere.png", sphere_surface)

        # Carte thermique avec barre de couleur (lisible)
        self.save_temperature_with_colorbar(out_dir / f"{planet_id}_thermal_map_legend.png", temp_map)

        # Carte thématique (climats discrets)
        class_map, classes = self.classify_climate(temp_map)
        climate_rgb = self.colorize_climate(class_map, classes)
        plt.imsave(out_dir / f"{planet_id}_climate_map.png", climate_rgb)

        return {
            "thermal_map": str(out_dir / f"{planet_id}_thermal_map.png"),
            "surface_map": str(out_dir / f"{planet_id}_surface_map.png"),
            "thermal_sphere": str(out_dir / f"{planet_id}_thermal_sphere.png"),
            "surface_sphere": str(out_dir / f"{planet_id}_surface_sphere.png"),
            "thermal_map_legend": str(out_dir / f"{planet_id}_thermal_map_legend.png"),
            "climate_map": str(out_dir / f"{planet_id}_climate_map.png"),
        }

    def export_planet_poster(self, out_dir: Path, planet: Dict, star: Dict,
                              base_albedo: float, greenhouse_k: float, obliquity_deg: float = 23.5):
        """Crée une affiche complète pour la planète (globe, carte, métriques)."""
        pid = planet.get("id", "planet")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Recompute maps for consistency
        temp_map = self.make_temperature_map(star, planet["distance_au"],
                                             albedo=float(base_albedo),
                                             greenhouse_k=float(greenhouse_k),
                                             obliquity_deg=float(obliquity_deg),
                                             w=2048, h=1024)
        temp_rgb = self.colorize_temperature(temp_map)
        sphere_temp = self.shade_sphere(temp_rgb, w=1600, h=1600)

        alb_map, _ = self.make_surface_albedo_map(w=2048, h=1024)
        H, W = alb_map.shape
        y = (np.arange(H)/H)[:, None]
        biome_rgb = np.dstack([
            0.8*alb_map + 0.2*(1-y),
            0.7*(1-alb_map) + 0.3*y,
            0.6*(1-alb_map)
        ]).clip(0,1)
        snow = (temp_map < 260).astype(np.float32)
        biome_rgb = (biome_rgb*(1 - 0.7*snow[...,None]) + 0.7*snow[...,None]).clip(0,1)
        sphere_surface = self.shade_sphere(biome_rgb, w=1600, h=1600)

        # Compose poster
        fig = plt.figure(figsize=(16,10))
        gs = fig.add_gridspec(2,3, height_ratios=[2,1], width_ratios=[1,1,1], hspace=0.35, wspace=0.35)
        ax0 = fig.add_subplot(gs[0,0]); ax0.imshow(sphere_surface); ax0.axis("off"); ax0.set_title(f"{pid} — Surface", fontsize=12)
        ax1 = fig.add_subplot(gs[0,1]); ax1.imshow(sphere_temp); ax1.axis("off"); ax1.set_title("Thermal Globe", fontsize=12)
        ax2 = fig.add_subplot(gs[0,2])
        im2 = ax2.imshow(temp_map, origin="lower", cmap="inferno")
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04); cbar.set_label("Temp (K)")
        ax2.set_title("Thermal Map", fontsize=12); ax2.axis("off")

        ax3 = fig.add_subplot(gs[1,:])
        details = planet.get("biosignature_details", {})
        txt = [
            f"Star: {star.get('spectral_class','?')} | T={star.get('temperature_k','?')} K | L(log)={star.get('luminosity_log','?')} | Age={star.get('age_gyr','?')} Ga",
            f"a = {planet.get('distance_au','?'):.3f} AU | P = {planet.get('orbital_period_years','?')} a",
            f"Mass = {planet.get('mass_earth','?')} M⊕ | Radius = {planet.get('radius_earth','?')} R⊕",
            f"Surface: {planet.get('geology',{}).get('surface_type','?')} | T_eq = {details.get('equilibrium_temp_k','?')} K | T_surf = {details.get('surface_temp_k','?')} K",
            f"BPI = {planet.get('biosignature_index','?')} (T:{details.get('temperature_score','?')}, H2O:{details.get('water_score','?')}, HZ:{details.get('hz_soft','?')})",
        ]
        ax3.axis('off'); ax3.text(0.01, 0.95, "\n".join(txt), va='top', ha='left', fontsize=12, family='monospace')
        fig.suptitle(f"Affiche — {pid}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0,0.02,1,0.95])
        poster_path = out_dir / f"{pid}_poster.png"
        plt.savefig(poster_path, dpi=170)
        plt.close()
        return str(poster_path)


# ==========================================
# DATA
# ==========================================
class DataManager:
    """
    Charge CSV **ou** JSON. Colonnes attendues (mêmes noms dans JSON):
    - st_mass, luminosity_log, st_teff, pl_orbsmax, pl_bmasse, pl_rade
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def set_json_paths(self, paths):
        """Enregistre une liste de chemins JSON à ingérer lorsqu'on demande des 'vraies données'."""
        if not paths:
            self.json_paths = []
            return
        self.json_paths = [Path(p) for p in paths if p]

    def _simulate(self, n=2000) -> pd.DataFrame:
        """
        Fallback: génère un dataset synthétique cohérent avec les colonnes attendues
        (st_mass, luminosity_log, st_teff, pl_orbsmax, pl_bmasse, pl_rade)
        + les versions log utilisées par le modèle.
        """
        rng = np.random.default_rng(getattr(self.cfg, "RANDOM_SEED", 42))

        # Étoiles (lois grossières mais plausibles)
        st_mass = rng.uniform(0.1, 1.1, n).astype(np.float32)
        st_teff = np.clip(rng.normal(4800, 800, n), 2600, 6500).astype(np.float32)
        # relation masse-luminosité très grossière ~ M^3.5 (en log10 du L/L☉)
        luminosity_rel = np.clip(st_mass ** 3.5, 1e-4, 100)
        luminosity_log = np.log10(luminosity_rel).astype(np.float32)

        # Orbite (AU) — lognorm pour couvrir proche/loin
        pl_orbsmax = np.clip(rng.lognormal(mean=-0.2, sigma=0.8, size=n), 0.05, 5.0).astype(np.float32)

        # Masse planète (M⊕) — plus grande tendance vers l'intérieur (très grossier)
        base = 1.0 + (1.0 / np.maximum(pl_orbsmax, 0.05)) ** 0.4
        pl_bmasse = np.clip(np.exp(rng.normal(np.log(base), 0.6)) - 1.0, 0.05, 50.0).astype(np.float32)

        # Rayon planète (R⊕) — deux régimes simples
        pl_rade = np.where(pl_bmasse < 3.0,
                           pl_bmasse ** 0.27,
                           np.clip(1.5 + 0.1 * np.sqrt(pl_bmasse), 1.0, 12.0)).astype(np.float32)

        df = pd.DataFrame({
            "st_mass":        st_mass,
            "luminosity_log": luminosity_log,
            "st_teff":        st_teff,
            "pl_orbsmax":     pl_orbsmax,
            "pl_bmasse":      pl_bmasse,
            "pl_rade":        pl_rade,
        })
        df["pl_masse_log"] = np.log1p(df["pl_bmasse"])
        df["pl_rade_log"]  = np.log1p(df["pl_rade"])
        return df.dropna()

    def _luminosity_log_from_rad_teff(self, st_rad_solar: float, st_teff_k: float, T_sun: float = 5772.0) -> float:
        if st_rad_solar is None or st_teff_k is None:
            return None
        L_rel = (float(st_rad_solar)**2) * (float(st_teff_k)/T_sun)**4
        return float(np.log10(max(L_rel, 1e-12)))

    def _json_to_df_exoplanets(self, j: dict) -> pd.DataFrame:
        rows = j.get("exoplanets", j.get("rows", []))
        recs = []
        for r in rows:
            st_teff = r.get("st_teff")
            st_mass = r.get("st_mass")
            st_rad  = r.get("st_rad")
            lum_log = r.get("luminosity_log")
            if lum_log is None and (st_rad is not None and st_teff is not None):
                lum_log = self._luminosity_log_from_rad_teff(st_rad, st_teff)
            recs.append(dict(
                st_mass=st_mass,
                luminosity_log=lum_log,
                st_teff=st_teff,
                pl_orbsmax=r.get("pl_orbsmax"),
                pl_bmasse=r.get("pl_bmasse"),
                pl_rade=r.get("pl_rade"),
            ))
        df = pd.DataFrame(recs).dropna()
        if not df.empty:
            df["pl_masse_log"] = np.log1p(df["pl_bmasse"])
            df["pl_rade_log"]  = np.log1p(df["pl_rade"])
        return df

    def _json_to_df_planetary(self, j: dict) -> pd.DataFrame:
        planets = j.get("planets", [])
        recs = []
        for p in planets:
            orb = p.get("orbital_parameters", {})
            sma = orb.get("semi_major_axis_au") or orb.get("semi_major_axis")
            phys = p.get("physical_characteristics", {})
            mass_kg = phys.get("mass", {}).get("kg")
            mass_earth = phys.get("mass", {}).get("earth_masses")
            rade_er = phys.get("equatorial_radius", {}).get("earth_radii")
            rade_km = phys.get("equatorial_radius", {}).get("km")

            pl_bmasse = mass_earth if mass_earth is not None else (mass_kg / 5.972e24 if mass_kg else None)
            if rade_er is not None:
                pl_rade = float(rade_er)
            elif rade_km is not None:
                pl_rade = float(rade_km) / 6378.137
            else:
                pl_rade = None

            st_teff = 5778.0
            st_mass = 1.0
            luminosity_log = 0.0

            if sma is None or pl_bmasse is None or pl_rade is None:
                continue
            recs.append(dict(
                st_mass=st_mass,
                luminosity_log=luminosity_log,
                st_teff=st_teff,
                pl_orbsmax=float(sma),
                pl_bmasse=float(pl_bmasse),
                pl_rade=float(pl_rade),
            ))
        df = pd.DataFrame(recs).dropna()
        if not df.empty:
            df["pl_masse_log"] = np.log1p(df["pl_bmasse"])
            df["pl_rade_log"]  = np.log1p(df["pl_rade"])
        return df

    def _json_to_df_ultimate(self, j: dict) -> pd.DataFrame:
        # ULTIMATE_COMPLETE_PLANETARY_DATABASE.json structure
        # j["solar_system_objects"] contains groups (e.g., terrestrial_planets, gas_giants, ice_giants, dwarf_planets)
        sso = j.get("solar_system_objects", {})
        groups = []
        if isinstance(sso, dict):
            for key, arr in sso.items():
                if isinstance(arr, list):
                    groups.extend(arr)
        elif isinstance(sso, list):
            groups = sso

        recs = []
        for p in groups:
            try:
                orb = p.get("orbital_characteristics", {})
                sma = None
                if isinstance(orb, dict):
                    sma_obj = orb.get("semi_major_axis", {})
                    if isinstance(sma_obj, dict):
                        sma = sma_obj.get("au") or sma_obj.get("AU") or sma_obj.get("value")
                phys = p.get("physical_characteristics", {})
                mass_earth = None; rade_er = None
                if isinstance(phys, dict):
                    m = phys.get("mass", {})
                    if isinstance(m, dict):
                        mass_earth = m.get("earth_masses") or m.get("earth_mass")
                    r = phys.get("equatorial_radius", {})
                    if isinstance(r, dict):
                        rade_er = r.get("earth_radii") or r.get("earth_radius")

                if sma is None or mass_earth is None or rade_er is None:
                    continue
                recs.append(dict(
                    st_mass=1.0,
                    luminosity_log=0.0,
                    st_teff=5778.0,
                    pl_orbsmax=float(sma),
                    pl_bmasse=float(mass_earth),
                    pl_rade=float(rade_er),
                ))
            except Exception:
                continue
        df = pd.DataFrame(recs).dropna()
        if not df.empty:
            df["pl_masse_log"] = np.log1p(df["pl_bmasse"])
            df["pl_rade_log"]  = np.log1p(df["pl_rade"])
        return df

    def _from_json(self, path: Path) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Routage par schéma
        if isinstance(data, dict) and ("exoplanets" in data or "metadata" in data):
            return self._json_to_df_exoplanets(data)
        if isinstance(data, dict) and ("planets" in data):
            return self._json_to_df_planetary(data)
        if isinstance(data, dict) and ("solar_system_objects" in data):
            return self._json_to_df_ultimate(data)

        # Fallback générique
        rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
        df = pd.DataFrame(rows)
        alias = {
            "st_lum": "luminosity_log",
            "st_teff_k": "st_teff",
            "a_au": "pl_orbsmax",
            "mass_earth": "pl_bmasse",
            "radius_earth": "pl_rade",
        }
        for a, b in alias.items():
            if a in df.columns and b not in df.columns:
                df[b] = df[a]
        needed = ["st_mass", "luminosity_log", "st_teff", "pl_orbsmax", "pl_bmasse", "pl_rade"]
        df = df[[c for c in needed if c in df.columns]].dropna()
        if not df.empty:
            if "pl_masse_log" not in df.columns:
                df["pl_masse_log"] = np.log1p(df["pl_bmasse"])
            if "pl_rade_log" not in df.columns:
                df["pl_rade_log"]  = np.log1p(df["pl_rade"])
        return df

    def load_many_json(self, paths) -> pd.DataFrame:
        frames = []
        for p in paths:
            try:
                frames.append(self._from_json(Path(p)))
            except Exception as e:
                if getattr(self.cfg, "VERBOSE", True):
                    print(f"[DataManager] Skip {p} -> {e}")
        if not frames:
            raise RuntimeError("Aucun JSON valide chargé.")
        df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def guess_all_jsons(self) -> List[str]:
        # Cherche les 5 JSON demandés dans le dossier data/
        base = self.cfg.ROOT / "data"
        names = [
            "comprehensive_binary_systems_database.json",
            "comprehensive_brown_dwarf_database.json",
            "comprehensive_exoplanet_database.json",
            "comprehensive_stellar_database.json",
            "ULTIMATE_COMPLETE_PLANETARY_DATABASE.json",
        ]
        found = []
        for n in names:
            p = base / n
            if p.exists():
                found.append(str(p))
        return found

    def load(self, use_real_data: bool = False) -> pd.DataFrame:
        """
        Compatibilité avec l'appel existant: self.dm.load(self.use_real_data)
        - Si use_real_data et des json_paths ont été fournis -> on charge ces JSONs.
        - Sinon: on retombe sur ton flux historique (DATA_PATH, simulateur, CSV, etc.).
        """
        # Chemins JSON explicitement passés
        if use_real_data and getattr(self, "json_paths", None):
            return self.load_many_json(self.json_paths)

        # DATA_PATH .json (config existante)
        p = getattr(self.cfg, "DATA_PATH", None)
        if use_real_data and p and isinstance(p, Path) and p.suffix.lower() == ".json" and p.exists():
            return self._from_json(p)

        # CSV historique si présent
        if p and isinstance(p, Path) and p.suffix.lower() == ".csv" and p.exists():
            df = pd.read_csv(p)
            return df.dropna()

        # Simu de secours (garde ton comportement d’avant)
        if hasattr(self, "_simulate"):
            return self._simulate(n=getattr(self.cfg, "SIM_N", 2000))

        raise RuntimeError("DataManager.load: aucune source de données valide trouvée (json_paths / DATA_PATH / simulateur).")

# ==========================================
# NN
# ==========================================
class PlanetPredictorNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=None, dropout=0.25):
        super().__init__()
        hidden_dims = hidden_dims or [256,128,64]
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head_mass = nn.Linear(d,1)
        self.head_radius = nn.Linear(d,1)
        self.head_comp = nn.Sequential(nn.Linear(d,3), nn.Softmax(dim=-1))

    def forward(self, x):
        if x.dim()==1: x=x.unsqueeze(0)
        z = self.backbone(x)
        return {"log_mass": self.head_mass(z),
                "log_radius": self.head_radius(z),
                "composition": self.head_comp(z)}


# ==========================================
# GÉO / ÉCO / RENDER
# ==========================================
class GeologicalSimulator:
    def evolve_planet(self, planet: Dict, star: Dict, T_eq: float, T_surf: Optional[float] = None) -> Dict:
        age = star["age_gyr"]; mass = planet["mass_earth"]
        T_ref = T_surf if (T_surf is not None) else T_eq
        if mass > 10:
            surface = "Géante gazeuse chaude" if T_ref>1000 else "Géante gazeuse froide"
            atmosphere = "H/He dominé, métaux" if T_ref>1000 else "H/He (CH4, NH3)"
        elif mass > 2:
            if T_ref>500: surface, atmosphere = "Monde de lave", "Dense (CO2, SO2)"
            elif T_ref>373: surface, atmosphere = "Désert chaud", "Dense (CO2, N2)"
            elif T_ref>273: surface, atmosphere = "Océans + continents possibles", "N2/O2, H2O"
            else: surface, atmosphere = "Monde de glace", "Ténue (N2, CH4 gelé)"
        else:
            if T_ref>600: surface, atmosphere = "Océan de magma", "Vaporisée (silicates)"
            elif T_ref>373: surface, atmosphere = "Désert rocheux", "Ténue (CO2)"
            elif T_ref>273:
                if age>2.0 and mass>0.5: surface, atmosphere = "Continents et océans liquides", "Mature (N2/O2/CO2)"
                else: surface, atmosphere = "Océans primitifs", "Secondaire (CO2/N2/H2O)"
            else: surface, atmosphere = "Glaciation globale", "Très ténue"
        planet["geology"] = {"surface_type":surface,"atmosphere_composition":atmosphere,"temperature_k":round(T_ref,1),"age":f"{age:.2f} Ga"}
        return planet

class EcosystemSimulator:
    def simulate_life(self, planet: Dict) -> Dict:
        T = planet.get("biosignature_details",{}).get("surface_temp_k",0)
        bpi = planet.get("biosignature_index",0)
        if bpi<0.3: eco, c = "Aucun (stérile)", 0.0
        elif bpi<0.5:
            eco = "Microbienne primitive" if 200<=T<=400 else ("Cryophile" if T<200 else "Thermophile")
            c=0.2
        elif bpi<0.7: eco, c = "Biosphère simple", 0.5
        elif bpi<0.85: eco, c = "Écosystème développé", 0.7
        else: eco, c = "Biosphère complexe", 0.9
        planet["ecosystem"]={"type":eco,"complexity":round(c,2),"energy_source":("Photosynthèse" if 273<T<373 else "Chimio-/Thermosynthèse")}
        return planet

class PlanetRenderer:
    def render(self, planet: Dict, ax):
        geology = planet.get("geology",{}); surface = geology.get("surface_type","Inconnu"); temp = geology.get("temperature_k",300)
        size=120; x=np.linspace(-1,1,size); y=np.linspace(-1,1,size); X,Y=np.meshgrid(x,y); mask=X**2+Y**2<=1
        np.random.seed(hash(planet["id"])%2**32); noise=np.random.randn(size,size)
        if "magma" in surface.lower() or temp>600:
            pattern=np.sin(5*X)*np.cos(5*Y)+0.6*noise; cmap=ListedColormap(["#3d0000","#8B0000","#FF0000","#FF4500","#FFA500"])
        elif "glace" in surface.lower() or temp<200:
            pattern=np.sqrt(X**2+Y**2)+0.35*noise; cmap=ListedColormap(["#E0FFFF","#B0E0E6","#87CEEB","#4682B4"])
        elif "océan" in surface.lower() or "continents" in surface.lower():
            pattern=np.sin(3*X+noise)+np.cos(3*Y+noise); cmap=ListedColormap(["#001f4d","#003566","#1f7a1f","#8B4513","#F5DEB3"])
        elif "désert" in surface.lower():
            pattern=np.abs(X)+np.abs(Y)+0.55*noise; cmap=ListedColormap(["#F4A460","#D2691E","#CD853F","#8B4513"])
        elif "gazeuse" in surface.lower():
            pattern=np.sin(Y*10)+0.25*noise; cmap=ListedColormap(["#FFE4B5","#FFDEAD","#F0E68C","#BDB76B"])
        else: pattern=noise; cmap="viridis"
        pattern[~mask]=np.nan
        ax.imshow(pattern,cmap=cmap,interpolation="bicubic",extent=[-1,1,-1,1])
        ax.add_patch(Circle((0,0),1,fill=False,edgecolor="black",linewidth=2))
        ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_aspect("equal"); ax.axis("off")
        title=f"{planet['id']}: {surface[:30]}"; title+=f"\n{temp:.0f} K"
        ax.set_title(title,fontsize=10)


# ==========================================
# AI
# ==========================================
class PlanetaryAI:
    def __init__(self, cfg: Config, use_real_data=False):
        self.cfg = cfg
        self.physics = PhysicsCalculator(cfg)
        self.dm = DataManager(cfg)
        self.geology = GeologicalSimulator()
        self.eco = EcosystemSimulator()
        self.render = PlanetRenderer()
        self.use_real_data = use_real_data

        self.model = PlanetPredictorNetwork(4, cfg.HIDDEN_DIMS, cfg.DROPOUT_RATE).to(cfg.DEVICE)
        self.scaler = StandardScaler()
        self.model_ready = False

        # Calibrage serre pour la Terre si demandé
        self.k_calibrated = cfg.GREENHOUSE_K_BASE
        if cfg.GREENHOUSE_AUTO_CALIBRATE:
            sun = {"temperature_k":5778.0, "luminosity_log":0.0}
            T_eq_earth = self.physics.calculate_equilibrium_temperature(sun, 1.0, cfg.ALBEDO_DEFAULT)
            self.k_calibrated = self.physics.greenhouse_k_for_target(T_eq_earth, cfg.TEMP_TARGET_K)

        self.discovered_systems: List[Dict] = []

    # ------- TRAIN -------
    def _pseudo_comp(self, m_e: np.ndarray, r_e: np.ndarray) -> np.ndarray:
        rocky = np.clip(1.5 - r_e, 0, 1) * np.clip(3.0 - np.sqrt(m_e), 0, 1)
        gaseous = np.clip((r_e - 2.0) / 3.0, 0, 1) + np.clip((m_e - 8.0) / 20.0, 0, 1)
        icy = np.clip((1.8 - r_e) / 1.2, 0, 1) * np.clip((m_e - 0.2) / 2.0, 0, 1)
        raw = np.stack([rocky,gaseous,icy],axis=-1) + 1e-3
        return (raw / raw.sum(axis=1,keepdims=True)).astype(np.float32)

    def train_model(self, epochs: Optional[int] = None):
        set_seed(self.cfg.RANDOM_SEED)
        df = self.dm.load(self.use_real_data)
        X = df[["st_mass","luminosity_log","st_teff","pl_orbsmax"]].values.astype(np.float32)
        yM = df[["pl_masse_log"]].values.astype(np.float32)
        yR = df[["pl_rade_log"]].values.astype(np.float32)
        Xs = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.cfg.SCALER_PATH)
        comp = self._pseudo_comp(np.expm1(yM.squeeze()), np.expm1(yR.squeeze()))
        X_tr, X_te, yM_tr, yM_te, yR_tr, yR_te, cp_tr, cp_te = train_test_split(
            Xs, yM, yR, comp, test_size=0.15, random_state=self.cfg.RANDOM_SEED
        )
        X_tr, X_va, yM_tr, yM_va, yR_tr, yR_va, cp_tr, cp_va = train_test_split(
            X_tr, yM_tr, yR_tr, cp_tr, test_size=0.176, random_state=self.cfg.RANDOM_SEED
        )

        def TT(*arr): return [torch.tensor(a, dtype=torch.float32) for a in arr]
        Xt, yMt, yRt, cpt = TT(X_tr, yM_tr, yR_tr, cp_tr)
        Xv, yMv, yRv, cpv = TT(X_va, yM_va, yR_va, cp_va)
        Xs, yMs, yRs, cps = TT(X_te, yM_te, yR_te, cp_te)

        train_loader = DataLoader(TensorDataset(Xt,yMt,yRt,cpt),batch_size=self.cfg.BATCH_SIZE,shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xv,yMv,yRv,cpv),batch_size=self.cfg.BATCH_SIZE)
        test_loader  = DataLoader(TensorDataset(Xs,yMs,yRs,cps),batch_size=self.cfg.BATCH_SIZE)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY)
        # Scheduler avec redémarrages cosinus
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
        l_huber, l_kl = nn.SmoothL1Loss(), nn.KLDivLoss(reduction="batchmean")
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.USE_AMP and self.cfg.DEVICE.startswith("cuda"))

        best_val, best_state, pat = float("inf"), None, 0
        tr_hist, va_hist = [], []
        print(f"🚀 Entraînement sur {len(Xt)} | Val {len(Xv)} | Test {len(Xs)}")

        for epoch in range(1, (epochs or self.cfg.EPOCHS)+1):
            self.model.train(); s=0.0
            for xb,ymb,yrb,cpb in train_loader:
                xb,ymb,yrb,cpb=xb.to(self.cfg.DEVICE), ymb.to(self.cfg.DEVICE), yrb.to(self.cfg.DEVICE), cpb.to(self.cfg.DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.cfg.USE_AMP and self.cfg.DEVICE.startswith("cuda")):
                    # Label smoothing sur composition
                    if self.cfg.LABEL_SMOOTHING and self.cfg.LABEL_SMOOTHING > 0:
                        eps = float(self.cfg.LABEL_SMOOTHING)
                        num_classes = cpb.shape[-1]
                        cpb_smooth = (1.0 - eps) * cpb + eps / float(num_classes)
                    else:
                        cpb_smooth = cpb

                    # Feature noise
                    if self.cfg.FEATURE_NOISE_STD and self.cfg.FEATURE_NOISE_STD > 0:
                        xb_noisy = xb + float(self.cfg.FEATURE_NOISE_STD) * torch.randn_like(xb)
                    else:
                        xb_noisy = xb

                    # MixUp inputs/labels (mass, radius, composition)
                    use_mixup = self.cfg.MIXUP_ALPHA and self.cfg.MIXUP_ALPHA > 0
                    if use_mixup and xb_noisy.size(0) > 1:
                        alpha = float(self.cfg.MIXUP_ALPHA)
                        lam = np.random.beta(alpha, alpha)
                        perm = torch.randperm(xb_noisy.size(0), device=xb_noisy.device)
                        xb_mix = lam * xb_noisy + (1.0 - lam) * xb_noisy[perm]
                        yM_mix = lam * ymb + (1.0 - lam) * ymb[perm]
                        yR_mix = lam * yrb + (1.0 - lam) * yrb[perm]
                        cp_mix = lam * cpb_smooth + (1.0 - lam) * cpb_smooth[perm]
                        out = self.model(xb_mix)
                        l1 = l_huber(out["log_mass"], yM_mix)
                        l2 = l_huber(out["log_radius"], yR_mix)
                        logp = (out["composition"]+1e-8).log(); l3 = l_kl(logp, cp_mix)
                    else:
                        out=self.model(xb_noisy)
                        l1=l_huber(out["log_mass"],ymb); l2=l_huber(out["log_radius"],yrb)
                        logp=(out["composition"]+1e-8).log(); l3=l_kl(logp,cpb_smooth)

                    loss=l1+l2+0.1*l3
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update()
                s += float(loss.detach().cpu())
            train_loss = s/max(1,len(train_loader)); tr_hist.append(train_loss)

            self.model.eval(); s=0.0
            with torch.no_grad():
                for xb,ymb,yrb,cpb in val_loader:
                    xb,ymb,yrb,cpb=xb.to(self.cfg.DEVICE), ymb.to(self.cfg.DEVICE), yrb.to(self.cfg.DEVICE), cpb.to(self.cfg.DEVICE)
                    out=self.model(xb)
                    l1=l_huber(out["log_mass"],ymb); l2=l_huber(out["log_radius"],yrb)
                    logp=(out["composition"]+1e-8).log(); l3=l_kl(logp,cpb)
                    s += float((l1+l2+0.1*l3).detach().cpu())
            val_loss = s/max(1,len(val_loader)); va_hist.append(val_loss); sched.step(epoch)

            if epoch%10==0 or epoch==1:
                # get_last_lr is available for both schedulers
                try:
                    lr_disp = sched.get_last_lr()[0]
                except Exception:
                    lr_disp = opt.param_groups[0]['lr']
                print(f"[{epoch:03d}] train={train_loss:.3f} val={val_loss:.3f} lr={lr_disp:.2e}")
            if val_loss < best_val - 1e-6:
                best_val, best_state, pat = val_loss, {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= self.cfg.PATIENCE:
                    print(f"⏹️  Early stopping @ epoch {epoch}")
                    break

        if best_state: self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), self.cfg.MODEL_PATH)
        self.model_ready = True

        # Test metrics
        ytm, ypm, ytr, ypr = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for xb,ymb,yrb,_ in test_loader:
                out=self.model(xb.to(self.cfg.DEVICE))
                ytm.append(ymb.numpy()); ytr.append(yrb.numpy())
                ypm.append(out["log_mass"].cpu().numpy()); ypr.append(out["log_radius"].cpu().numpy())
        ytm,ypm,ytr,ypr = np.vstack(ytm),np.vstack(ypm),np.vstack(ytr),np.vstack(ypr)
        m_true, m_pred = np.expm1(ytm), np.expm1(ypm)
        r_true, r_pred = np.expm1(ytr), np.expm1(ypr)
        metrics = {"mass":{"MAE":float(mean_absolute_error(m_true,m_pred)),"RMSE":rmse(m_true,m_pred),"R2":float(r2_score(m_true,m_pred))},
                   "radius":{"MAE":float(mean_absolute_error(r_true,r_pred)),"RMSE":rmse(r_true,r_pred),"R2":float(r2_score(r_true,r_pred))},
                   "best_val_loss":best_val}
        print("✅ Test metrics:", json.dumps(metrics, indent=2))

        # courbes
        plt.figure(figsize=(7,4)); plt.plot(tr_hist,label="train"); plt.plot(va_hist,label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Courbes d'entraînement"); plt.grid(True,alpha=0.3); plt.legend()
        self.cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(self.cfg.CURVES_PATH,dpi=140); plt.close()

    # ------- PREDICTION -------
    def _physical_mr(self, star: Dict, a_au: float) -> Tuple[float,float,Dict[str,float]]:
        # Petit fallback masse-rayon "physique"
        # masses plus grandes vers l'intérieur, puis loi R ~ M^0.27 pour rocheuses
        base = max(0.05, 5.0 * (1 / (a_au ** 0.4)) * (star["mass_solar"] ** 0.2))
        m_e = float(np.clip(np.random.lognormal(np.log(base), 0.3), 0.05, 60))
        if m_e < 3:
            r_e = float((m_e ** 0.27) * np.random.lognormal(0,0.05))
            comp = {"rocky":0.85,"gaseous":0.05,"icy":0.10}
        else:
            r_e = float(np.clip(1.5 + 0.1*m_e**0.5, 1.2, 12.0) * np.random.lognormal(0,0.08))
            comp = {"rocky":0.1,"gaseous":0.85,"icy":0.05}
        return m_e, r_e, {k:round(v,3) for k,v in comp.items()}

    def _predict_mass_radius_comp(self, star: Dict, distance_au: float) -> Tuple[float,float,Dict[str,float]]:
        feats = np.array([star["mass_solar"], star["luminosity_log"], star["temperature_k"], distance_au], dtype=np.float32)
        if not self.model_ready and (not self.cfg.MODEL_PATH.exists() or self.cfg.FORCE_PHYSICS_MR):
            return self._physical_mr(star, distance_au)
        feats = self.scaler.transform(feats.reshape(1,-1))
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(feats, dtype=torch.float32, device=self.cfg.DEVICE))
        m = float(np.expm1(out["log_mass"].cpu().numpy().squeeze()))
        r = float(np.expm1(out["log_radius"].cpu().numpy().squeeze()))
        comp = out["composition"].cpu().numpy().squeeze().tolist()
        return max(0.01,m), max(0.1,r), {"rocky":round(comp[0],3),"gaseous":round(comp[1],3),"icy":round(comp[2],3)}

    def _calc_bpi(self, star: Dict, planet: Dict, albedo: float) -> Dict:
        T_eq = self.physics.calculate_equilibrium_temperature(star, planet["distance_au"], albedo)
        # serre: k calibré (Terre) par défaut
        k = self.k_calibrated
        T_surf = self.physics.greenhouse_surface_temperature(T_eq, k)

        # Score T: centré à 288 K, sigma ~30 K; forte pénalité > 330 K
        score_temp = math.exp(-((T_surf - self.cfg.TEMP_TARGET_K) / 30.0) ** 2)
        if T_surf > 330:
            score_temp *= math.exp(-((T_surf - 330.0) / 25.0) ** 2)

        # Eau liquide
        if self.cfg.TEMP_HABITABLE_MIN <= T_surf <= self.cfg.TEMP_HABITABLE_MAX:
            score_water = 1.0
        elif T_surf < self.cfg.TEMP_HABITABLE_MIN:
            score_water = max(0.0, 1 - (self.cfg.TEMP_HABITABLE_MIN - T_surf) / 120.0)
        else:
            score_water = max(0.0, 1 - (T_surf - self.cfg.TEMP_HABITABLE_MAX) / 120.0)

        # HZ douce
        hz_in, hz_out = self.physics.habitable_zone_bounds(star["luminosity_log"])
        soft_hz = self.physics.soft_in_hz(planet["distance_au"], hz_in, hz_out, self.cfg.SOFT_HZ_WIDTH)

        score_age = min(1.0, star["age_gyr"] / 4.0)
        score_protection = min(1.0, planet["mass_earth"] / 2.0)

        # Poids rééquilibrés (davantage d'importance à T et HZ)
        weights = [2.5, 1.5, 0.8, 0.8, 1.7]
        terms = [score_temp, score_water, score_age, score_protection, soft_hz]
        bpi = sum(w*t for w,t in zip(weights, terms)) / sum(weights)

        planet["biosignature_index"] = round(bpi,3)
        planet["biosignature_details"] = {
            "temperature_score": round(score_temp,3),
            "water_score": round(score_water,3),
            "age_score": round(score_age,3),
            "protection_score": round(score_protection,3),
            "hz_soft": round(soft_hz,3),
            "equilibrium_temp_k": round(T_eq,1),
            "surface_temp_k": round(T_surf,1),
            "hz_inner_au": round(hz_in,3),
            "hz_outer_au": round(hz_out,3),
        }
        return planet

    def generate_star(self, spectral_class: Optional[str] = None) -> Dict:
        if spectral_class is None:
            spectral_class = np.random.choice(["G","K","M"], p=[0.1,0.2,0.7])
        types = {"G":{"mass":(0.8,1.04),"temp":(5200,6000),"lum":(-0.2,0.05)},
                 "K":{"mass":(0.45,0.8),"temp":(3700,5200),"lum":(-0.7,-0.2)},
                 "M":{"mass":(0.08,0.45),"temp":(2400,3700),"lum":(-3.0,-0.7)}}
        p = types[spectral_class]
        return {"spectral_class":spectral_class,
                "mass_solar":round(float(np.random.uniform(*p["mass"])),3),
                "temperature_k":round(float(np.random.uniform(*p["temp"])),0),
                "luminosity_log":round(float(np.random.uniform(*p["lum"])),3),
                "age_gyr":round(float(np.random.uniform(2.0,10.0)),2)}

    def generate_planet(self, star: Dict, dist_au: float, p_id: int, albedo: Optional[float]=None) -> Dict:
        albedo = self.cfg.ALBEDO_DEFAULT if albedo is None else float(albedo)
        m_e, r_e, comp = self._predict_mass_radius_comp(star, dist_au)
        planet = {"id":f"P{p_id}", "distance_au":float(dist_au),
                  "mass_earth":round(m_e,3), "radius_earth":round(r_e,3),
                  "composition":comp, "albedo":float(albedo),
                  "orbital_period_years":round(self.physics.orbital_period_years(dist_au, star["mass_solar"]),3)}
        # Calcul BPI (inclut T_surf via l'effet de serre calibré) puis géologie cohérente sur T_surf
        planet = self._calc_bpi(star, planet, albedo)
        T_eq = planet["biosignature_details"]["equilibrium_temp_k"]
        T_surf = planet["biosignature_details"]["surface_temp_k"]
        planet = self.geology.evolve_planet(planet, star, T_eq, T_surf=T_surf)
        planet = self.eco.simulate_life(planet)
        return planet

    def rank_real_exoplanets(self, json_paths: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
        # Classe les exoplanètes réelles par BPI en utilisant les champs disponibles
        paths = json_paths or getattr(self.dm, "json_paths", []) or [str(self.cfg.DATA_PATH)]
        df = self.dm.load_many_json(paths)
        # Essaye d'utiliser les valeurs observées
        out = []
        for idx, r in df.iterrows():
            try:
                star = {"spectral_class":"?",
                        "mass_solar": float(r["st_mass"]),
                        "temperature_k": float(r["st_teff"]),
                        "luminosity_log": float(r["luminosity_log"]),
                        "age_gyr": 4.6}
                pl = {"id": f"REAL-{idx}",
                      "distance_au": float(r["pl_orbsmax"]),
                      "mass_earth": float(r.get("pl_bmasse", np.expm1(r.get("pl_masse_log", 0.0)))),
                      "radius_earth": float(r.get("pl_rade", np.expm1(r.get("pl_rade_log", 0.0)))),
                      "composition": {"rocky":0.33,"gaseous":0.33,"icy":0.34},
                      "albedo": float(self.cfg.ALBEDO_DEFAULT)}
                pl = self._calc_bpi(star, pl, pl["albedo"])
                out.append({"planet": pl, "star": star})
            except Exception:
                continue
        out.sort(key=lambda x: x["planet"].get("biosignature_index", 0.0), reverse=True)
        return out[:max(1, int(top_k))]

    def _check_system_stability(self, system: Dict) -> bool:
        # même critère de Hill que v6.0
        c = PhysicsConstants()
        planets = sorted(system["planets"], key=lambda p:p["distance_au"])
        if len(planets)<2: return True
        for i in range(len(planets)-1):
            p1,p2=planets[i],planets[i+1]
            m1=p1["mass_earth"]*c.earth_mass_kg; m2=p2["mass_earth"]*c.earth_mass_kg
            M=system["star"]["mass_solar"]*c.solar_mass; a1=p1["distance_au"]*c.au_m; a2=p2["distance_au"]*c.au_m
            r_hill = ((m1+m2)/(3.0*M))**(1/3) * (a1+a2)/2.0
            if (a2-a1) < 3.5*r_hill: return False
        return True

    def discover_and_simulate_systems(self, n_systems=10):
        print(f"\n🔭 Découverte et simulation de {n_systems} systèmes...")
        tries, max_tries = 0, n_systems*6
        self.discovered_systems.clear()
        while len(self.discovered_systems)<n_systems and tries<max_tries:
            tries+=1
            star = self.generate_star()
            n_planets = int(np.random.choice([1,2,3,4,5,6], p=[0.1,0.2,0.3,0.2,0.15,0.05]))
            planets=[]; dist=float(np.clip(np.random.lognormal(-0.4,0.7),0.08,2.5))
            for j in range(n_planets):
                alb=float(np.clip(np.random.normal(self.cfg.ALBEDO_DEFAULT,0.05),0.05,0.8))
                planets.append(self.generate_planet(star, dist, j+1, albedo=alb))
                dist *= float(np.random.uniform(1.35,1.95))
            system={"star":star,"planets":planets,"n_planets":n_planets}
            if not self._check_system_stability(system): continue
            system_id=f"Oracle-{len(self.discovered_systems)+1:03d}"
            max_bpi=max((p.get("biosignature_index",0.0) for p in planets), default=0.0)
            hab=sum(1 for p in planets if p.get("biosignature_index",0.0)>0.6)
            system.update({"id":system_id,"is_stable":True,"max_bpi":max_bpi,"habitable_planets":hab})
            self.discovered_systems.append(system)
            print(f"  ✓ {system_id}: {n_planets} planètes, BPI max={max_bpi:.3f}, {hab} habitable(s)")

        print(f"\n📊 Bilan: {len(self.discovered_systems)} systèmes stables")

    # ------- VISU / RAPPORT -------
    def _create_visualizations(self, all_planets: List[Dict], best_targets: List[Dict]):
        fig = plt.figure(figsize=(17,10)); gs = fig.add_gridspec(3,3,hspace=0.35,wspace=0.35)
        if best_targets:
            ax1=fig.add_subplot(gs[0,0]); self.render.render(best_targets[0]["planet"], ax1)
        ax2=fig.add_subplot(gs[0,1:]); masses=[max(0.01,p["planet"]["mass_earth"]) for p in all_planets]
        radii=[max(0.1,p["planet"]["radius_earth"]) for p in all_planets]; bpis=[p["planet"]["biosignature_index"] for p in all_planets]
        sc=ax2.scatter(masses,radii,c=bpis,cmap="RdYlGn",s=45,alpha=0.85,edgecolors="black",linewidth=0.4)
        ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("Masse (M⊕)"); ax2.set_ylabel("Rayon (R⊕)")
        ax2.set_title("Relation Masse–Rayon (couleur=BPI)"); ax2.grid(True,alpha=0.3); plt.colorbar(sc,ax=ax2,label="BPI")

        ax3=fig.add_subplot(gs[1,:])
        temps=[p["planet"]["biosignature_details"]["surface_temp_k"] for p in all_planets]
        ax3.hist(temps,bins=30,alpha=0.85,edgecolor="black")
        ax3.axvspan(self.cfg.TEMP_HABITABLE_MIN,self.cfg.TEMP_HABITABLE_MAX,alpha=0.25,color="green",label="Zone Habitable (surface)")
        ax3.set_xlabel("Température de surface estimée (K)"); ax3.set_ylabel("Nombre de planètes"); ax3.set_title("Distribution des Températures")
        ax3.legend(); ax3.grid(True,alpha=0.3)

        ax4=fig.add_subplot(gs[2,:])
        for sys in self.discovered_systems:
            d=[p["distance_au"] for p in sys["planets"]]; col=plt.cm.RdYlGn(max(p["biosignature_index"] for p in sys["planets"]))
            ax4.scatter(d,[sys["id"]]*len(d),c=[col]*len(d),s=55,alpha=0.85)
            hz_in,hz_out=self.physics.habitable_zone_bounds(sys["star"]["luminosity_log"])
            ax4.plot([hz_in,hz_out],[sys["id"],sys["id"]],color="green",alpha=0.4,linewidth=6)
        ax4.set_xscale("log"); ax4.set_xlabel("Distance (AU)"); ax4.set_title("Architecture & Zones Habitables (segments verts)")
        ax4.grid(True,alpha=0.25,axis="x")

        plt.suptitle("Analyse des Mondes Extraterrestres — Oracle v6.1",fontsize=16,fontweight="bold")
        plt.tight_layout(rect=[0,0.02,1,0.96]); plt.savefig(self.cfg.FIG_FINAL,dpi=150); plt.close()

        # Poster pour la meilleure cible
        if best_targets:
            try:
                bt = best_targets[0]
                p = bt["planet"]; s = bt["system"]["star"]
                renders_dir = self.cfg.OUT_DIR / "renders"
                # Utiliser le k calibré et l'albédo du planet
                base_alb = float(p.get("albedo", self.cfg.ALBEDO_DEFAULT))
                renderer = PlanetSurfaceRenderer(self.cfg, self.physics)
                renderer.export_planet_poster(renders_dir, p, s, base_alb, getattr(self, "k_calibrated", 0.20))
            except Exception:
                pass

    def generate_final_analysis(self):
        if not self.discovered_systems: print("❌ Aucun système découvert."); return
        all_planets=[{"planet":p,"system":s} for s in self.discovered_systems for p in s["planets"]]
        all_planets.sort(key=lambda x:x["planet"].get("biosignature_index",0), reverse=True)
        best_targets=[p for p in all_planets if p["planet"]["biosignature_index"]==all_planets[0]["planet"]["biosignature_index"]]

        print("\n"+"="*64+"\n🌌 RAPPORT FINAL DE L'ORACLE ASTROPHYSIQUE 🌌\n"+"="*64)
        print(f"\n📊 Statistiques: {len(self.discovered_systems)} systèmes, {len(all_planets)} planètes,",
              f"{sum(1 for p in all_planets if p['planet']['biosignature_index']>0.6)} potentielles (>0.6)")

        if best_targets:
            max_bpi=best_targets[0]["planet"]["biosignature_index"]
            print(f"\n🎯 Meilleures cibles (BPI = {max_bpi:.3f}):")
            for i,t in enumerate(best_targets[:5],1):
                p,s=t["planet"],t["system"]
                print(f"  {i}. {p['id']} — {s['id']} | {s['star']['spectral_class']} {s['star']['age_gyr']:.1f} Ga")
                print(f"     • {p['mass_earth']:.2f} M⊕, {p['radius_earth']:.2f} R⊕, P={p['orbital_period_years']:.2f} a")
                print(f"     • T_surf: {p['biosignature_details']['surface_temp_k']:.0f} K | Surface: {p['geology']['surface_type']}")

        self._create_visualizations(all_planets,best_targets)
        with open(self.cfg.ANALYSIS_JSON,"w",encoding="utf-8") as f:
            json.dump({"systems":self.discovered_systems,
                       "best_targets":[{"system_id":t["system"]["id"],"planet_id":t["planet"]["id"],"bpi":t["planet"]["biosignature_index"]} for t in best_targets[:10]]},
                      f, indent=2)
        print(f"\n💾 Résultats: {self.cfg.ANALYSIS_JSON}")
        print(f"🖼️  Figures: {self.cfg.FIG_FINAL}, {self.cfg.CURVES_PATH}")

    # ------- SCÉNARIOS TERRE -------
    def solar_system_scenarios(self,
        n_monte_carlo: int = 3000,
        grid_albedo: Tuple[float,float,int] = (0.05, 0.70, 50),
        grid_distance: Tuple[float,float,int] = (0.85, 1.15, 50),
        greenhouse_jitter: float = 0.20,           # ± autour de k_terre
        luminosity_scale_range: Tuple[float,float] = (0.98, 1.03),
    ):
        sun = dict(spectral_class="G", mass_solar=1.0, temperature_k=5778.0, luminosity_log=0.0, age_gyr=4.6)
        k0 = self.k_calibrated

        # Monte Carlo
        classes = {"glacé":0,"habitable":0,"brûlant":0}
        bpis=[]
        for _ in range(n_monte_carlo):
            alb = float(np.random.uniform(0.05, 0.70))
            dist = float(np.random.uniform(*grid_distance[:2]))
            k = float(np.clip(np.random.normal(k0, greenhouse_jitter), 0.0, 1.5))
            lum_scale = float(np.random.uniform(*luminosity_scale_range))
            star = sun.copy(); star["luminosity_log"] = math.log10(lum_scale)
            planet = {"id":"Earth*", "distance_au":dist, "mass_earth":1.0, "radius_earth":1.0, "composition":{"rocky":1.0,"gaseous":0.0,"icy":0.0}}
            # BPI (utilise _calc_bpi -> serre k0, donc on recalcule T_surf custom pour classement)
            planet_tmp = self._calc_bpi(star, planet.copy(), alb)
            T_eq = self.physics.calculate_equilibrium_temperature(star, dist, alb)
            T_surf = self.physics.greenhouse_surface_temperature(T_eq, k)
            bpis.append(planet_tmp["biosignature_index"])
            if T_surf < self.cfg.TEMP_HABITABLE_MIN - 10: classes["glacé"] += 1
            elif T_surf > self.cfg.TEMP_HABITABLE_MAX + 10: classes["brûlant"] += 1
            else: classes["habitable"] += 1

        # Heatmap (BPI avec k0)
        a_min,a_max,a_n = grid_albedo; d_min,d_max,d_n = grid_distance
        A = np.linspace(a_min,a_max,int(a_n)); D = np.linspace(d_min,d_max,int(d_n))
        heat = np.zeros((len(A),len(D)),dtype=np.float32)
        for i,a in enumerate(A):
            for j,d in enumerate(D):
                pl = {"id":"Earth_grid","distance_au":float(d),"mass_earth":1.0,"radius_earth":1.0,"composition":{"rocky":1.0,"gaseous":0.0,"icy":0.0}}
                star = sun.copy()
                heat[i,j] = self._calc_bpi(star, pl, float(a))["biosignature_index"]

        # Sauvegardes
        plt.figure(figsize=(7.6,6))
        im = plt.imshow(heat,origin="lower",extent=[D.min(),D.max(),A.min(),A.max()],aspect="auto",cmap="RdYlGn")
        plt.colorbar(im,label="BPI")
        CS = plt.contour(D,A,heat,levels=[0.6,0.75,0.9],colors=["black","blue","purple"],linewidths=1.0)
        plt.clabel(CS, inline=True, fontsize=8)
        plt.xlabel("Distance Terre–Soleil (AU)"); plt.ylabel("Albédo"); plt.title("Scénarios Terre: BPI (Distance, Albédo)")
        plt.tight_layout(); plt.savefig(self.cfg.FIG_SCENARIO_HEATMAP,dpi=150); plt.close()

        plt.figure(figsize=(7.2,4.2))
        names=list(classes.keys()); vals=[classes[k] for k in names]
        plt.bar(names,vals); plt.title("Monte Carlo — états probables Terre (T_surf)")
        plt.ylabel("Comptes"); plt.tight_layout(); plt.savefig(self.cfg.FIG_SCENARIO_HIST,dpi=140); plt.close()

        print(f"🛰️  Scénarios Terre: {self.cfg.FIG_SCENARIO_HEATMAP} ; {self.cfg.FIG_SCENARIO_HIST}")
        return {"class_counts":classes,"bpi_mean":float(np.mean(bpis)),"bpi_std":float(np.std(bpis))}

    def earth_future_projection(self, years: List[int] = None, scenario: str = "baseline",
                                use_solar_system_effects: bool = False, influence: float = 1.0):
        # Projection simple: variation du forçage serre (k) et de l'albédo
        years = years or [2030, 2050, 2100, 2200, 2500]
        sun = dict(spectral_class="G", mass_solar=1.0, temperature_k=5778.0, luminosity_log=0.0, age_gyr=4.6)
        baseline_k = float(self.k_calibrated)
        baseline_alb = float(self.cfg.ALBEDO_DEFAULT)
        results = []
        for y in years:
            # Tendance simplifiée
            if scenario == "high":
                k = baseline_k + 0.30 * (min(2100, y) - 2025) / 75.0
                albedo = max(0.05, baseline_alb - 0.05 * (min(2100, y) - 2025) / 75.0)
            elif scenario == "geoengineering":
                k = max(0.0, baseline_k - 0.08 * (min(2100, y) - 2025) / 75.0)
                albedo = min(0.85, baseline_alb + 0.04 * (min(2100, y) - 2025) / 75.0)
            else:  # baseline
                k = baseline_k + 0.12 * (min(2100, y) - 2025) / 75.0
                albedo = baseline_alb

            # Effets du système solaire (cycles orbitaux et obliquité) optionnels
            if use_solar_system_effects:
                # Excentricité effective (cycles ~100 kyr et 405 kyr) influencée par Jupiter/Saturne
                t100 = 2.0 * math.pi * (y - 2000.0) / 100000.0
                t405 = 2.0 * math.pi * (y - 2000.0) / 405000.0
                e0 = 0.0167
                de = influence * (0.015 * math.sin(t100) + 0.005 * math.sin(t405))
                e = float(np.clip(e0 + de, 0.0, 0.08))
                # Facteur de flux moyen sur orbite elliptique
                flux_factor = float(1.0 / math.sqrt(max(1e-6, 1.0 - e * e)))
                # Obliquité (cycle ~41 kyr) => micro-variation d'albédo
                tob = 2.0 * math.pi * (y - 2000.0) / 41000.0
                obliquity_deg = 23.5 + influence * (1.3 * math.sin(tob))
                albedo = float(np.clip(albedo + 0.01 * math.sin(tob) * influence, 0.03, 0.85))
                # T_eq avec correction de flux moyen lié à e
                T_eq0 = self.physics.calculate_equilibrium_temperature(sun, 1.0, albedo)
                T_eq = float(T_eq0 * (flux_factor ** 0.25))
            else:
                T_eq = self.physics.calculate_equilibrium_temperature(sun, 1.0, albedo)
            T_surf = self.physics.greenhouse_surface_temperature(T_eq, k)
            hz_in, hz_out = self.physics.habitable_zone_bounds(sun["luminosity_log"])
            results.append({"year": int(y), "albedo": round(albedo,3), "k": round(k,3),
                            "T_eq": round(T_eq,1), "T_surf": round(T_surf,1),
                            "hz_inner_au": round(hz_in,3), "hz_outer_au": round(hz_out,3)})

        # Plot
        ys = [r["year"] for r in results]; Ts = [r["T_surf"] for r in results]
        plt.figure(figsize=(7.5,4.2))
        plt.plot(ys, Ts, marker="o", label=f"T_surf ({scenario})")
        plt.axhspan(self.cfg.TEMP_HABITABLE_MIN, self.cfg.TEMP_HABITABLE_MAX, color="green", alpha=0.15, label="Fourchette eau liquide")
        plt.ylabel("Température de surface (K)"); plt.xlabel("Année")
        plt.title("Projection Terre — Température de surface")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(self.cfg.FIG_EARTH_TIMELINE, dpi=150); plt.close()

        with open(self.cfg.EARTH_TIMELINE_JSON, "w", encoding="utf-8") as f:
            json.dump({"scenario": scenario, "results": results}, f, indent=2)
        print(f"🌍 Projection Terre sauvegardée: {self.cfg.FIG_EARTH_TIMELINE} ; {self.cfg.EARTH_TIMELINE_JSON}")
        return results


# ==========================================
# CLI
# ==========================================
def build_argparser():
    import argparse
    ap = argparse.ArgumentParser(description="Oracle Astrophysique v6.1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--systems", type=int, default=10)
    ap.add_argument("--skip-train", action="store_true", help="Ne pas réentraîner si modèle/scaler présents.")
    ap.add_argument("--scenarios", action="store_true", help="Générer les scénarios Terre/Soleil.")
    ap.add_argument("--force-mr", action="store_true", help="Forcer le fallback masse-rayon physique.")

    ap.add_argument("--json", nargs="+", default=[], help="Chemins vers 1..N fichiers JSON (exoplanets / planetary / autres).")
    ap.add_argument("--real", action="store_true", help="Forcer l'utilisation des vraies données (JSON/chemin DATA_PATH).")
    ap.add_argument("--render-images", action="store_true", help="Générer les images (cartes thermiques + globes).")
    ap.add_argument("--render-top", type=int, default=3, help="Nombre de cibles à rendre (défaut: 3).")
    ap.add_argument("--all-json", action="store_true", help="Charger automatiquement tous les JSON du dossier data/ (5 fichiers fournis).")
    ap.add_argument("--rank-real", action="store_true", help="Classer les exoplanètes réelles par BPI et afficher le Top.")
    ap.add_argument("--timeline", action="store_true", help="Générer une projection temporelle Terre (baseline/high/geoengineering).")
    ap.add_argument("--timeline-scenario", type=str, default="baseline", choices=["baseline","high","geoengineering"], help="Scénario pour la projection Terre.")
    ap.add_argument("--timeline-solar", action="store_true", help="Inclure des effets simplifiés du système solaire (excentricité, obliquité).")
    ap.add_argument("--timeline-influence", type=float, default=1.0, help="Amplitude des effets planétaires (par défaut 1.0).")

    return ap

def main():
    print("="*64); print("🚀 ORACLE ASTROPHYSIQUE v6.1 — Réalisme + Scénarios Terre"); print("="*64)
    cfg = Config.load_from_file(); set_seed(cfg.RANDOM_SEED)
    ap = build_argparser(); args = ap.parse_args() if len(os.sys.argv)>1 else None

    if not args:
        use_real = input("\n💾 Utiliser un dataset réel (CSV/JSON local) ? (y/N): ").strip().lower()=="y"
        ai = PlanetaryAI(cfg, use_real_data=use_real)
        need_train=True
        if cfg.MODEL_PATH.exists() and cfg.SCALER_PATH.exists():
            try:
                ai.model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
                ai.scaler = joblib.load(cfg.SCALER_PATH); need_train = (input("🔁 Modèle trouvé. Réentraîner ? (y/N): ").strip().lower()=="y")
            except Exception: 
                need_train=True

        print("\n📚 Entraînement")
        if need_train: 
            ai.train_model()
        else: 
            ai.model_ready=True; print("⏭️  Entraînement sauté (modèle chargé).")

        print("\n🔭 Découverte")
        try: n=int(input("Nombre de systèmes [10]: ") or "10")
        except ValueError: n=10
        ai.discover_and_simulate_systems(n_systems=n)

        print("\n📊 Analyse & Visu")
        ai.generate_final_analysis()

        if input("\n🌎 Lancer les scénarios Terre/Soleil ? (Y/n): ").strip().lower()!="n":
            stats = ai.solar_system_scenarios(); print("Résumé BPI (Monte Carlo) — Terre:", stats)
        print("\n✨ Terminé.")
        return

    # --- Mode CLI --- 
    ai = PlanetaryAI(cfg, use_real_data=bool(args.real))

    # 👉 IMPORTANT: enregistrer les JSON passés en arguments, et activer use_real_data
    if args.all_json:
        all_paths = ai.dm.guess_all_jsons()
        if all_paths:
            ai.dm.set_json_paths(all_paths)
            ai.use_real_data = True
            print(f"📦 --all-json: {len(all_paths)} fichiers détectés: " + ", ".join(all_paths))
    if args.json:
        ai.dm.set_json_paths(args.json)
        ai.use_real_data = True
        print(f"📦 JSON détectés ({len(args.json)}): " + ", ".join(args.json))

    if args.force_mr:
        cfg.FORCE_PHYSICS_MR = True

    # Charger modèle/scaler si demandé, sinon entraîner
    if args.skip_train and cfg.MODEL_PATH.exists() and cfg.SCALER_PATH.exists():
        try:
            ai.model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))
            ai.scaler = joblib.load(cfg.SCALER_PATH); ai.model_ready=True
            print("⏭️  Entraînement sauté (modèle/scaler chargés).")
        except Exception:
            ai.train_model(epochs=args.epochs)
    else:
        ai.train_model(epochs=args.epochs)

    # Simulations + rapport
    ai.discover_and_simulate_systems(n_systems=int(args.systems))
    ai.generate_final_analysis()

    if args.rank_real:
        top = ai.rank_real_exoplanets(json_paths=getattr(ai.dm, "json_paths", None), top_k=10)
        print("\n🏆 Top exoplanètes réelles (BPI):")
        for i, e in enumerate(top, 1):
            p = e["planet"]; print(f"  {i}. {p['id']} — BPI={p.get('biosignature_index',0.0):.3f}")

    # Scénarios optionnels
    if args.scenarios:
        stats = ai.solar_system_scenarios(); print("Résumé scénarios Terre:", stats)

    if args.timeline:
        ai.earth_future_projection(scenario=str(args.timeline_scenario))

    # Rendu d'images optionnel (utilise la physique déjà présente)
    if args and args.render_images:
        renderer = PlanetSurfaceRenderer(cfg, ai.physics)
        out_dir = cfg.OUT_DIR / "renders"

        # Collecte des meilleures cibles
        cand = []
        for s in getattr(ai, "discovered_systems", []):
            for pl in s.get("planets", []):
                st = s.get("star", {})
                if "distance_au" in pl:
                    cand.append(({"planet": pl, "star": st}, pl.get("biosignature_index", 0.0)))
        cand = sorted(cand, key=lambda x: x[1], reverse=True)[:max(1, args.render_top)]

        if not cand:
            # fallback : exemple Terre-Soleil
            sun = dict(spectral_class="G", mass_solar=1.0, temperature_k=5778.0, luminosity_log=0.0, age_gyr=4.6)
            renderer.export_planet_images(out_dir, "EarthLike",
                                          star=sun, distance_au=1.0,
                                          base_albedo=0.30, greenhouse_k=0.20, obliquity_deg=23.5)
        else:
            for (s, _) in cand:
                pl = s["planet"]; st = s.get("star", {"temperature_k":5778.0, "luminosity_log":0.0})
                pid = pl.get("id","candidate")
                dist = float(pl.get("distance_au", 1.0))
                alb  = float(pl.get("albedo", 0.30))
                k    = float(getattr(ai, "k_calibrated", 0.20))
                renderer.export_planet_images(out_dir, pid, star=st, distance_au=dist,
                                              base_albedo=alb, greenhouse_k=k, obliquity_deg=23.5)
            # Générer aussi un poster pour la meilleure planète
            if cand:
                top = cand[0][0]
                renderer.export_planet_poster(out_dir, top["planet"], top["star"],
                                              base_albedo=float(top["planet"].get("albedo", 0.30)),
                                              greenhouse_k=float(getattr(ai, "k_calibrated", 0.20)))
        print(f"🖼  Images rendues dans: {out_dir}")

if __name__=="__main__":
    main()