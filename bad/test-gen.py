import numpy as np
from PIL import Image
import noise
import math
import random

# --- PARAMÈTRES BASÉS SUR LES DONNÉES RÉELLES DE MARS ---

# Dimensions de l'image
LARGEUR = 1280
HAUTEUR = 720

# Topographie (influencée par la faible gravité : grandes structures possibles)
ECHELLE = 300.0  # Plus la valeur est grande, plus les reliefs sont vastes
OCTAVES = 6          # Nombre de niveaux de détail du terrain
PERSISTANCE = 0.5    # Amplitude des détails
LACUNARITE = 2.0     # Fréquence des détails

# Atmosphère (ciel couleur caramel/rose dû à la poussière d'oxyde de fer)
COULEUR_CIEL_HORIZON = (189, 146, 110)
COULEUR_CIEL_ZENITH = (110, 78, 55)

# Palette de couleurs du sol (régolithe riche en oxydes de fer, basalte)
PALETTE_SOL = [
    (40, 30, 20),      # Basalte sombre (basses altitudes, plaines)
    (130, 80, 50),     # Poussière d'oxyde de fer (moyennes altitudes)
    (180, 110, 80),    # Régolithe plus clair
    (210, 150, 120),   # Sols riches en sulfates / poussière très fine
    (220, 225, 230)    # Givre / Glace de CO2 (hautes altitudes)
]

# --- FONCTIONS DE GÉNÉRATION ---

def generer_terrain(largeur, hauteur):
    """Génère une carte de relief (heightmap) en utilisant le bruit de Perlin."""
    print("1. Génération du relief de base...")
    carte_relief = np.zeros((hauteur, largeur))
    for y in range(hauteur):
        for x in range(largeur):
            # Génère une valeur de bruit entre -1 et 1
            valeur_bruit = noise.pnoise2(x / ECHELLE,
                                         y / ECHELLE,
                                         octaves=OCTAVES,
                                         persistence=PERSISTANCE,
                                         lacunarity=LACUNARITE,
                                         repeatx=largeur,
                                         repeaty=hauteur,
                                         base=42) # graine aléatoire
            # Normalise la valeur entre 0 et 1 pour représenter l'altitude
            carte_relief[y][x] = (valeur_bruit + 1) / 2
    return carte_relief

def ajouter_craters(carte_relief):
    """Ajoute des cratères d'impact de tailles et profondeurs variées."""
    print("2. Ajout des cratères d'impact...")
    nb_craters = random.randint(15, 30)
    for _ in range(nb_craters):
        # Rayon du cratère (en % de la plus petite dimension de l'image)
        rayon = random.uniform(0.02, 0.1) * min(LARGEUR, HAUTEUR)
        # Position du centre du cratère
        cx, cy = random.randint(0, LARGEUR-1), random.randint(0, HAUTEUR-1)
        
        profondeur = random.uniform(0.1, 0.3) # Profondeur relative
        bordure = 1.2 # Largeur du rebord

        for y in range(HAUTEUR):
            for x in range(LARGEUR):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < rayon * bordure:
                    # Forme cosinusoïdale pour un cratère réaliste
                    if dist < rayon: # Intérieur du cratère
                        impact = (math.cos(dist / rayon * math.pi) + 1) / 2
                        carte_relief[y][x] -= impact * profondeur
                    else: # Rebord du cratère
                        impact = (math.cos((dist - rayon) / (rayon * (bordure - 1)) * math.pi) - 1) / 2
                        carte_relief[y][x] -= impact * profondeur * 0.3

    # Normalise à nouveau la carte entre 0 et 1 après ajout des cratères
    carte_relief = np.clip(carte_relief, -1, 1) # Limite les valeurs extrêmes
    carte_relief = (carte_relief - np.min(carte_relief)) / (np.max(carte_relief) - np.min(carte_relief))
    return carte_relief

def colorer_surface(carte_relief):
    """Applique la palette de couleurs martiennes en fonction de l'altitude."""
    print("3. Coloration de la surface selon la géologie martienne...")
    image_rgb = np.zeros((HAUTEUR, LARGEUR, 3), dtype=np.uint8)
    niveaux_palette = len(PALETTE_SOL) - 1
    
    for y in range(HAUTEUR):
        for x in range(LARGEUR):
            altitude = carte_relief[y][x]
            # Détermine les deux couleurs de la palette à mélanger
            idx1 = min(niveaux_palette, int(altitude * niveaux_palette))
            idx2 = min(niveaux_palette, idx1 + 1)
            
            # Calcule le facteur de mélange
            melange = (altitude * niveaux_palette) - idx1
            
            # Interpolation linéaire entre les deux couleurs
            r = PALETTE_SOL[idx1][0] * (1 - melange) + PALETTE_SOL[idx2][0] * melange
            g = PALETTE_SOL[idx1][1] * (1 - melange) + PALETTE_SOL[idx2][1] * melange
            b = PALETTE_SOL[idx1][2] * (1 - melange) + PALETTE_SOL[idx2][2] * melange
            
            image_rgb[y][x] = [int(r), int(g), int(b)]
    return image_rgb

def creer_ciel_martien():
    """Crée un dégradé pour le ciel martien."""
    print("4. Création de l'atmosphère et du ciel...")
    ciel = np.zeros((HAUTEUR, LARGEUR, 3), dtype=np.uint8)
    for y in range(HAUTEUR):
        # Facteur de mélange basé sur la position verticale
        melange = y / HAUTEUR
        r = COULEUR_CIEL_ZENITH[0] * (1 - melange) + COULEUR_CIEL_HORIZON[0] * melange
        g = COULEUR_CIEL_ZENITH[1] * (1 - melange) + COULEUR_CIEL_HORIZON[1] * melange
        b = COULEUR_CIEL_ZENITH[2] * (1 - melange) + COULEUR_CIEL_HORIZON[2] * melange
        ciel[y] = [int(r), int(g), int(b)]
    return ciel

def combiner_scene(terrain_color, ciel, carte_relief):
    """Combine le terrain et le ciel et ajoute un effet de brume atmosphérique."""
    # Création d'une "ligne d'horizon" basée sur la carte de relief
    horizon = (1 - carte_relief) * HAUTEUR * 0.7 + HAUTEUR * 0.3 # Ajustez ces valeurs pour la ligne d'horizon
    
    scene_finale = np.copy(ciel)
    for y in range(HAUTEUR):
        for x in range(LARGEUR):
            if y > horizon[y][x]:
                # Effet de brume : plus le terrain est "loin" (haut sur l'image), plus il se mélange avec le ciel
                distance_factor = max(0, (y - horizon[y][x])) / (HAUTEUR - np.min(horizon))
                brume = 1 - math.exp(-distance_factor * 2.5) # Brume exponentielle
                
                couleur_terrain = terrain_color[y][x]
                couleur_ciel = ciel[y][x]
                
                r = couleur_terrain[0] * (1-brume) + couleur_ciel[0] * brume
                g = couleur_terrain[1] * (1-brume) + couleur_ciel[1] * brume
                b = couleur_terrain[2] * (1-brume) + couleur_ciel[2] * brume
                
                scene_finale[y][x] = [int(r), int(g), int(b)]

    return scene_finale

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    carte_relief_base = generer_terrain(LARGEUR, HAUTEUR)
    carte_relief_craters = ajouter_craters(carte_relief_base)
    sol_colore = colorer_surface(carte_relief_craters)
    ciel = creer_ciel_martien()
    
    image_finale = combiner_scene(sol_colore, ciel, carte_relief_craters)
    
    # Conversion du tableau NumPy en image et sauvegarde
    img = Image.fromarray(image_finale, 'RGB')
    nom_fichier = "surface_martienne_procedurale.png"
    img.save(nom_fichier)
    print(f"\n✅ Image '{nom_fichier}' générée avec succès !")
    img.show()