import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import spacy
from scipy.spatial.distance import cosine

# Charger spaCy pour l'analyse syntaxique (guide, pas réponse)
nlp = spacy.load("fr_core_news_sm")

class GuidedTextToGraphNN(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, num_heads=8):
        super(GuidedTextToGraphNN, self).__init__()
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        # Embeddings enrichis
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(1000, embedding_dim)
        self.pos_tag_embedding = nn.Embedding(50, 64)  # Pour les tags grammaticaux
        self.dep_embedding = nn.Embedding(100, 64)  # Pour les dépendances syntaxiques
        
        # Fusion des indices linguistiques
        self.linguistic_fusion = nn.Linear(embedding_dim + 128, embedding_dim)
        
        # Attention multi-têtes avec masquage syntaxique
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.syntax_guided_attention = nn.MultiheadAttention(embedding_dim, num_heads//2, batch_first=True)
        
        # Réseau de découverte de patterns
        self.pattern_discoverer = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
        
        # Analyseur de cohérence sémantique
        self.coherence_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        
        # Détecteur d'entités avec contraintes linguistiques
        self.entity_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Score d'importance basé sur la structure
        self.importance_scorer = nn.Sequential(
            nn.Linear(128 + 64, 128),  # +64 pour les indices syntaxiques
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Extracteur de relations avec logique
        self.relation_reasoner = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 128, hidden_dim * 2),  # Plus d'input pour le contexte
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256)
        )
        
        # Classificateur de type de relation
        self.relation_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 types de relations à découvrir
            nn.Softmax(dim=-1)
        )
        
        # Module d'apprentissage par renforcement pour la cohérence
        self.coherence_reward = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids, positions, pos_tags=None, dep_tags=None, syntax_mask=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Embeddings de base
        word_emb = self.word_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Ajouter les indices linguistiques (sans donner les réponses)
        if pos_tags is not None and dep_tags is not None:
            pos_tag_emb = self.pos_tag_embedding(pos_tags)
            dep_emb = self.dep_embedding(dep_tags)
            linguistic_hints = torch.cat([pos_tag_emb, dep_emb], dim=-1)
            
            # Fusionner les indices avec les embeddings
            combined_emb = torch.cat([word_emb + pos_emb, linguistic_hints], dim=-1)
            embeddings = self.linguistic_fusion(combined_emb)
        else:
            embeddings = word_emb + pos_emb
        
        # Attention standard
        attn_output, attention_weights = self.attention(embeddings, embeddings, embeddings)
        
        # Attention guidée par la syntaxe (optionnel)
        if syntax_mask is not None:
            # Convertir le masque en format booléen si nécessaire
            # Le masque doit être 2D (seq_len, seq_len) ou 3D (batch_size*num_heads, seq_len, seq_len)
            if syntax_mask.dim() == 3 and syntax_mask.size(0) == batch_size:
                # Prendre juste la première dimension du batch
                syntax_mask_2d = syntax_mask[0]  # (seq_len, seq_len)
            else:
                syntax_mask_2d = syntax_mask
            
            # Convertir en masque booléen (True = masqué, False = autorisé)
            # Inverser la logique car dans PyTorch True signifie "masqué"
            bool_mask = syntax_mask_2d < 0.5  # Les valeurs faibles deviennent True (masqué)
            
            try:
                syntax_attn_output, syntax_weights = self.syntax_guided_attention(
                    embeddings, embeddings, embeddings,
                    attn_mask=bool_mask
                )
                # Combiner les deux attentions
                attn_output = 0.7 * attn_output + 0.3 * syntax_attn_output
            except:
                # Si erreur, utiliser seulement l'attention standard
                pass
        
        # Découverte de patterns avec LSTM
        lstm_output, (hidden, cell) = self.pattern_discoverer(attn_output)
        
        # Analyse de cohérence
        coherence_features = self.coherence_analyzer(lstm_output)
        
        # Détection d'entités
        entity_features = self.entity_detector(lstm_output)
        
        return lstm_output, entity_features, coherence_features, attention_weights

class IntelligentKnowledgeExtractor:
    def __init__(self):
        self.model = GuidedTextToGraphNN()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2
        self.pos_tag_map = {}
        self.dep_tag_map = {}
        self.knowledge_graph = nx.DiGraph()
        
        # Mémoire des patterns découverts
        self.discovered_patterns = defaultdict(list)
        self.entity_memory = defaultdict(set)
        self.relation_patterns = defaultdict(lambda: defaultdict(int))
        
    def analyze_linguistic_structure(self, text):
        """Analyser la structure linguistique pour guider l'apprentissage"""
        doc = nlp(text)
        
        # Extraire les indices structurels (pas les réponses!)
        linguistic_hints = {
            'pos_patterns': [],  # Patterns de POS tags
            'dep_patterns': [],  # Patterns de dépendances
            'syntactic_distance': {},  # Distance syntaxique entre mots
            'grammatical_roles': {}  # Rôles grammaticaux
        }
        
        for token in doc:
            # Enregistrer les patterns sans révéler les entités
            if token.pos_ not in self.pos_tag_map:
                self.pos_tag_map[token.pos_] = len(self.pos_tag_map)
            
            if token.dep_ not in self.dep_tag_map:
                self.dep_tag_map[token.dep_] = len(self.dep_tag_map)
            
            # Analyser les relations syntaxiques
            if token.head != token:
                distance = abs(token.i - token.head.i)
                linguistic_hints['syntactic_distance'][(token.i, token.head.i)] = distance
            
            # Identifier les rôles grammaticaux importants
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                linguistic_hints['grammatical_roles'][token.i] = token.dep_
        
        # Créer des patterns de n-grammes syntaxiques
        for sent in doc.sents:
            pos_sequence = [token.pos_ for token in sent]
            linguistic_hints['pos_patterns'].append(pos_sequence)
            
            dep_sequence = [(token.dep_, token.head.i - token.i) for token in sent]
            linguistic_hints['dep_patterns'].append(dep_sequence)
        
        return doc, linguistic_hints
    
    def create_syntax_guided_mask(self, doc, seq_len):
        """Créer un masque d'attention basé sur la syntaxe (guide, pas réponse)"""
        # Créer un masque 2D (seq_len, seq_len)
        mask = torch.ones(seq_len, seq_len) * 0.1  # Connexion faible par défaut
        
        for token in doc:
            if token.i < seq_len:
                # Renforcer l'attention entre mots syntaxiquement liés
                if token.head.i < seq_len:
                    mask[token.i, token.head.i] = 0.8
                    mask[token.head.i, token.i] = 0.8
                
                # Renforcer pour les mots proches
                for j in range(max(0, token.i - 3), min(seq_len, token.i + 4)):
                    distance_weight = 1.0 / (abs(j - token.i) + 1)
                    mask[token.i, j] = max(mask[token.i, j], distance_weight * 0.5)
        
        return mask
    
    def discover_entities_with_reasoning(self, text, doc, encoded_features, linguistic_hints):
        """Découverte d'entités avec raisonnement linguistique"""
        entities = []
        entity_scores = {}
        
        # Analyse des chunks nominaux (indices, pas réponses)
        noun_chunks_hints = [(chunk.start, chunk.end) for chunk in doc.noun_chunks]
        
        # L'IA doit découvrir quels chunks sont importants
        for start_idx, end_idx in noun_chunks_hints:
            chunk_text = doc[start_idx:end_idx].text
            
            # Calculer un score d'importance basé sur plusieurs facteurs
            importance_factors = {
                'length': len(chunk_text.split()),
                'capitalization': sum(1 for c in chunk_text if c.isupper()) / max(1, len(chunk_text)),
                'position': 1.0 / (start_idx + 1),  # Plus important au début
                'grammatical_role': 1.0 if start_idx in linguistic_hints['grammatical_roles'] else 0.5
            }
            
            # Le modèle apprend à pondérer ces facteurs
            if start_idx < len(encoded_features):
                chunk_features = encoded_features[start_idx:min(end_idx, len(encoded_features))].mean(dim=0)
                
                # Ajouter les indices grammaticaux
                grammar_hint = torch.tensor([importance_factors['grammatical_role']])
                combined_features = torch.cat([chunk_features[:128], grammar_hint.repeat(64)])
                
                # Score d'importance appris
                importance_score = self.model.importance_scorer(combined_features.unsqueeze(0))
                
                if importance_score.item() > 0.3:  # Seuil appris
                    entities.append(chunk_text)
                    entity_scores[chunk_text] = importance_score.item()
        
        # Découverte de patterns d'entités
        for token in doc:
            # Si le modèle identifie un pattern intéressant
            if token.ent_type_ and token.text not in [e for e in entities]:
                # L'IA doit apprendre que certains types sont importants
                pattern_key = f"{token.pos_}_{token.dep_}"
                self.discovered_patterns[pattern_key].append(token.text)
                
                # Si un pattern apparaît souvent, c'est peut-être important
                if len(self.discovered_patterns[pattern_key]) > 2:
                    entities.append(token.text)
                    entity_scores[token.text] = 0.5
        
        # Raffinement par clustering sémantique
        if len(entities) > 3:
            entities = self.refine_entities_by_clustering(entities, encoded_features, entity_scores)
        
        return entities, entity_scores
    
    def refine_entities_by_clustering(self, entities, features, scores):
        """Raffiner les entités par clustering sémantique"""
        # Créer des embeddings pour chaque entité
        entity_embeddings = []
        for entity in entities:
            # Utiliser les features du modèle pour créer un embedding
            entity_vec = torch.randn(128)  # Sera remplacé par les vraies features
            entity_embeddings.append(entity_vec.numpy())
        
        if len(entity_embeddings) > 1:
            # Clustering pour regrouper les entités similaires
            clustering = DBSCAN(eps=0.3, min_samples=1)
            clusters = clustering.fit_predict(np.array(entity_embeddings))
            
            # Garder le meilleur représentant de chaque cluster
            refined_entities = []
            for cluster_id in set(clusters):
                cluster_entities = [entities[i] for i, c in enumerate(clusters) if c == cluster_id]
                # Choisir l'entité avec le meilleur score
                best_entity = max(cluster_entities, key=lambda e: scores.get(e, 0))
                refined_entities.append(best_entity)
            
            return refined_entities
        
        return entities
    
    def learn_relation_patterns(self, source, target, doc, linguistic_hints):
        """Apprendre les patterns de relations sans supervision directe"""
        # Trouver le chemin syntaxique entre source et target
        source_tokens = [t for t in doc if source in t.text]
        target_tokens = [t for t in doc if target in t.text]
        
        if source_tokens and target_tokens:
            s_token = source_tokens[0]
            t_token = target_tokens[0]
            
            # Analyser le chemin de dépendance
            path = []
            current = s_token
            visited = set()
            
            while current != t_token and current not in visited:
                visited.add(current)
                path.append(current.dep_)
                current = current.head
            
            # Enregistrer ce pattern
            pattern_key = tuple(path[:3])  # Limiter la longueur
            self.relation_patterns[source][pattern_key] += 1
            
            return pattern_key
        
        return None
    
    def infer_relation_type(self, source, target, features, attention_weights, linguistic_pattern):
        """Inférer le type de relation en combinant neural et linguistique"""
        # Créer un vecteur de caractéristiques pour la relation
        relation_features = []
        
        # Features neurales
        if features is not None:
            relation_features.append(features.mean().item())
        
        # Pattern linguistique
        if linguistic_pattern:
            pattern_hash = hash(linguistic_pattern) % 100
            relation_features.append(pattern_hash / 100.0)
        
        # Attention moyenne
        if attention_weights is not None:
            relation_features.append(attention_weights.mean().item())
        
        # Classification de la relation
        if len(relation_features) > 0:
            features_tensor = torch.tensor(relation_features).unsqueeze(0)
            # Padding pour atteindre la bonne dimension
            padded_features = torch.nn.functional.pad(features_tensor, (0, 256 - features_tensor.size(1)))
            
            relation_probs = self.model.relation_classifier(padded_features)
            relation_type_idx = torch.argmax(relation_probs).item()
            
            # Mapping des types découverts
            relation_types = [
                "est", "appartient", "créé_par", "situé_à", "travaille_pour",
                "parent_de", "associé_à", "produit", "dirige", "contient"
            ]
            
            return relation_types[relation_type_idx]
        
        return "relation"
    
    def train_with_reasoning(self, text, epochs=50):
        """Entraînement avec raisonnement et guidance linguistique"""
        print("Analyse de la structure linguistique...")
        doc, linguistic_hints = self.analyze_linguistic_structure(text)
        
        # Préparer les données
        sentences = [sent.text for sent in doc.sents]
        
        print("Entraînement du modèle avec raisonnement...")
        
        for epoch in range(epochs):
            total_loss = 0
            coherence_scores = []
            
            for sent in doc.sents:
                if len(sent) < 3:
                    continue
                
                # Tokenisation et vocabulaire
                words = [token.text.lower() for token in sent]
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = self.vocab_size
                        self.vocab_size += 1
                
                # Préparer les inputs
                input_ids = torch.tensor([[self.vocab.get(w, 1) for w in words]])
                positions = torch.arange(len(words)).unsqueeze(0)
                
                # Ajouter les indices linguistiques
                pos_tags = torch.tensor([[self.pos_tag_map.get(t.pos_, 0) for t in sent]])
                dep_tags = torch.tensor([[self.dep_tag_map.get(t.dep_, 0) for t in sent]])
                
                # Créer le masque syntaxique
                syntax_mask = self.create_syntax_guided_mask(sent, len(words))
                
                # Forward pass
                encoded, entity_features, coherence_features, attention_weights = self.model(
                    input_ids, positions, pos_tags, dep_tags, syntax_mask.unsqueeze(0)
                )
                
                # Loss multi-objectifs
                
                # 1. Loss de reconstruction (auto-supervisée)
                reconstruction_loss = nn.MSELoss()(
                    encoded, 
                    encoded.detach() + torch.randn_like(encoded) * 0.05
                )
                
                # 2. Loss de cohérence sémantique
                coherence_target = torch.ones(coherence_features.size(0), coherence_features.size(1), 1)
                coherence_loss = nn.BCEWithLogitsLoss()(
                    self.model.coherence_reward(coherence_features),
                    coherence_target
                )
                
                # 3. Loss de diversité (éviter que tout soit associé)
                diversity_loss = -torch.var(entity_features)
                
                # 4. Loss de sparsité (éviter trop de connexions)
                sparsity_loss = torch.mean(torch.abs(attention_weights))
                
                # Combinaison pondérée
                loss = (reconstruction_loss + 
                       0.3 * coherence_loss + 
                       0.2 * diversity_loss + 
                       0.1 * sparsity_loss)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                coherence_scores.append(coherence_features.mean().item())
            
            self.scheduler.step()
            
            if epoch % 10 == 0:
                avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
                print(f"Epoch {epoch}: Loss={total_loss/max(1, len(sentences)):.4f}, "
                      f"Cohérence={avg_coherence:.4f}")
        
        print("Extraction du graphe de connaissances avec raisonnement...")
        self.extract_knowledge_with_logic(text, doc, linguistic_hints)
    
    def extract_knowledge_with_logic(self, text, doc, linguistic_hints):
        """Extraction avec logique et raisonnement"""
        all_entities = set()
        all_relations = []
        
        # Traiter chaque phrase
        for sent in doc.sents:
            if len(sent) < 3:
                continue
            
            # Encoder la phrase
            words = [token.text.lower() for token in sent]
            input_ids = torch.tensor([[self.vocab.get(w, 1) for w in words]])
            positions = torch.arange(len(words)).unsqueeze(0)
            pos_tags = torch.tensor([[self.pos_tag_map.get(t.pos_, 0) for t in sent]])
            dep_tags = torch.tensor([[self.dep_tag_map.get(t.dep_, 0) for t in sent]])
            
            with torch.no_grad():
                encoded, entity_features, coherence_features, attention_weights = self.model(
                    input_ids, positions, pos_tags, dep_tags
                )
            
            # Découvrir les entités avec raisonnement
            entities, entity_scores = self.discover_entities_with_reasoning(
                sent.text, sent, encoded[0], linguistic_hints
            )
            
            all_entities.update(entities)
            
            # Découvrir les relations logiques
            for i, source in enumerate(entities):
                for j, target in enumerate(entities):
                    if i != j:
                        # Apprendre le pattern de relation
                        linguistic_pattern = self.learn_relation_patterns(
                            source, target, sent, linguistic_hints
                        )
                        
                        # Inférer le type de relation
                        relation_type = self.infer_relation_type(
                            source, target, 
                            coherence_features[0] if coherence_features.size(0) > 0 else None,
                            attention_weights,
                            linguistic_pattern
                        )
                        
                        # Calculer le score de confiance
                        confidence = entity_scores.get(source, 0.5) * entity_scores.get(target, 0.5)
                        
                        if confidence > 0.2:  # Seuil de confiance
                            all_relations.append({
                                'source': source,
                                'target': target,
                                'relation': relation_type,
                                'confidence': confidence,
                                'pattern': linguistic_pattern
                            })
        
        # Construire le graphe
        for entity in all_entities:
            self.knowledge_graph.add_node(entity, type='entity')
        
        # Filtrer les relations par cohérence logique
        filtered_relations = self.filter_relations_by_logic(all_relations)
        
        for rel in filtered_relations:
            self.knowledge_graph.add_edge(
                rel['source'],
                rel['target'],
                relation=rel['relation'],
                weight=rel['confidence']
            )
        
        print(f"\nGraphe créé avec {len(all_entities)} entités et {len(filtered_relations)} relations logiques")
        
        # Afficher les relations découvertes
        print("\nRelations découvertes avec raisonnement:")
        for rel in filtered_relations[:10]:
            print(f"  {rel['source']} --[{rel['relation']}]--> {rel['target']} "
                  f"(confiance: {rel['confidence']:.2f})")
    
    def filter_relations_by_logic(self, relations):
        """Filtrer les relations pour ne garder que les plus logiques"""
        # Grouper par paires source-target
        relation_groups = defaultdict(list)
        for rel in relations:
            key = (rel['source'], rel['target'])
            relation_groups[key].append(rel)
        
        # Garder la relation la plus probable pour chaque paire
        filtered = []
        for key, group in relation_groups.items():
            best_rel = max(group, key=lambda r: r['confidence'])
            
            # Vérifier la cohérence logique
            if self.is_logically_coherent(best_rel):
                filtered.append(best_rel)
        
        return filtered
    
    def is_logically_coherent(self, relation):
        """Vérifier la cohérence logique d'une relation"""
        # Règles de cohérence apprises
        source = relation['source'].lower()
        target = relation['target'].lower()
        rel_type = relation['relation']
        
        # Éviter les auto-références
        if source == target:
            return False
        
        # Éviter les relations circulaires immédiates
        reverse_edge = self.knowledge_graph.has_edge(target, source)
        if reverse_edge:
            return relation['confidence'] > 0.7  # Seulement si très confiant
        
        # Autres règles de cohérence peuvent être apprises
        return True
    
    def visualize_graph(self):
        """Visualisation améliorée du graphe"""
        if len(self.knowledge_graph.nodes()) == 0:
            print("Le graphe est vide!")
            return
        
        plt.figure(figsize=(16, 12))
        
        # Layout hiérarchique pour mieux voir la structure
        pos = nx.spring_layout(self.knowledge_graph, k=3, iterations=100, seed=42)
        
        # Couleurs par type de nœud
        node_colors = ['lightblue' for _ in self.knowledge_graph.nodes()]
        
        # Tailles basées sur le degré
        node_sizes = [3000 + 500 * self.knowledge_graph.degree(node) 
                     for node in self.knowledge_graph.nodes()]
        
        # Dessiner le graphe
        nx.draw_networkx_nodes(self.knowledge_graph, pos, 
                              node_size=node_sizes, 
                              node_color=node_colors, 
                              alpha=0.7)
        
        nx.draw_networkx_labels(self.knowledge_graph, pos, 
                                font_size=10, font_weight='bold')
        
        # Dessiner les edges avec épaisseur variable
        edges = self.knowledge_graph.edges()
        weights = [self.knowledge_graph[u][v].get('weight', 0.5) for u, v in edges]
        
        nx.draw_networkx_edges(self.knowledge_graph, pos, 
                              width=[w * 3 for w in weights],
                              edge_color='gray', 
                              arrows=True, 
                              arrowsize=20, 
                              alpha=0.6,
                              connectionstyle="arc3,rad=0.1")
        
        # Labels des relations
        edge_labels = nx.get_edge_attributes(self.knowledge_graph, 'relation')
        nx.draw_networkx_edge_labels(self.knowledge_graph, pos, edge_labels, 
                                     font_size=8, font_color='red')
        
        plt.title("Graphe de Connaissances - Apprentissage avec Raisonnement Logique", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Afficher des statistiques
        self.print_graph_statistics()
    
    def print_graph_statistics(self):
        """Afficher les statistiques du graphe"""
        print("\n=== Statistiques du Graphe ===")
        print(f"Nombre de nœuds: {self.knowledge_graph.number_of_nodes()}")
        print(f"Nombre d'arêtes: {self.knowledge_graph.number_of_edges()}")
        
        if self.knowledge_graph.number_of_nodes() > 0:
            # Centralité
            degree_centrality = nx.degree_centrality(self.knowledge_graph)
            top_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print("\nEntités les plus centrales:")
            for entity, centrality in top_entities:
                print(f"  - {entity}: {centrality:.3f}")
            
            # Types de relations
            relation_counts = Counter()
            for _, _, data in self.knowledge_graph.edges(data=True):
                relation_counts[data.get('relation', 'unknown')] += 1
            
            print("\nTypes de relations découverts:")
            for rel_type, count in relation_counts.most_common():
                print(f"  - {rel_type}: {count}")

# Utilisation
def main():
    # Texte d'exemple enrichi
    text = """
    Albert Einstein était un physicien théoricien d'origine allemande. 
    Il est né le 14 mars 1879 à Ulm dans le Royaume de Wurtemberg.
    Einstein est surtout connu pour sa théorie de la relativité restreinte publiée en 1905.
    Cette théorie révolutionna notre compréhension de l'espace et du temps.
    Il a également développé la théorie de la relativité générale en 1915.
    La relativité générale décrit la gravitation comme une courbure de l'espace-temps.
    Einstein a reçu le prix Nobel de physique en 1921 pour ses travaux sur l'effet photoélectrique.
    Sa célèbre équation E=mc² établit l'équivalence entre masse et énergie.
    Il travailla à l'Institut d'études avancées de Princeton jusqu'à sa mort en 1955.
    Einstein était marié à Mileva Marić, une physicienne serbe.
    Ils eurent trois enfants ensemble : Hans Albert, Eduard et Lieserl.
    Plus tard, Einstein épousa sa cousine Elsa Einstein.
    Ses théories ont profondément influencé la physique moderne et la cosmologie.
    """
    
    # Créer et entraîner le modèle
    extractor = IntelligentKnowledgeExtractor()
    extractor.train_with_reasoning(text, epochs=40)
    
    # Visualiser le graphe
    extractor.visualize_graph()

if __name__ == "__main__":
    main()