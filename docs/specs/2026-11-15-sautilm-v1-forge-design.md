# Spécification Architecturale : SautiLM-Forge & SautiLM-v1 (<= 150 Mo)

- **Date** : 15 Novembre 2026
- **Organisation** : SautiLM Research (Bunia, Ituri, RD Congo)
- **Auteurs** : Georges Byona, Abraham Imani
- **Dépôts Cibles** :
  - Pipeline ML & Distillation : `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge`
  - Application Mobile Flutter : `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm`
  - Hub de Publication Modèle : `https://huggingface.co/sautilm/SautiLM-v1`

---

## 1. Objectif & Contraintes Physiques

Construire le premier modèle officiel **`SautiLM-v1`** dérivé de Google Gemma (`gemma-3-270m-it`), spécialisé pour les langues et besoins de l'Est de la RDC (Bbadha/Kilendu, Kiswahili, Français, Anglais), respectant une contrainte stricte de taille **$\le 150\text{ Mo}$** sur téléphone mobile afin d'éliminer :
1. Les plantages mémoire (OOM) sur les smartphones Android d'entrée de gamme (2 à 4 Go de RAM).
2. La surchauffe thermique et la lenteur causées par le modèle brut de 1.9 Go (`gemma-4-E2B-it.litertlm`).
3. La dépendance d'identité à "Gemma" : le modèle doit incarner nativement **SautiLM** (et servir de socle aux déclinaisons `SautiLM-Edu`, `SautiLM-Math`, `SautiLM-Med`, `SautiLM-Lex` et `SautiLM-Ius` pour le droit congolais).

---

## 2. Architecture Globale (Hybride Local PC + GPU Kaggle + Hugging Face)

```
+-------------------------------------------------------------------------+
| 1. PC LOCAL (Windows - Python/SautiLM-Forge)                            |
|    - Extraction des 24 JSON Bbadha (1 150 mots, sens, exemples)         |
|    - Génération du Corpus SFT/Distillation (Bbadha, Math, Edu, Med, Ius)|
|    - Compilation de l'index lexical mobile complet (1 150 entrées)      |
|    - Orchestrateur CLI automatisé (Kaggle API + Hugging Face API)       |
+-------------------------------------------------------------------------+
                                    |
                    Envoi chiffré HTTPS (API Kaggle)
                    (Les clés .env restent 100% sur le PC)
                                    v
+-------------------------------------------------------------------------+
| 2. SERVEUR GPU DISTANT (Kaggle Linux - 2x NVIDIA Tesla T4 32 Go VRAM)   |
|    - Modèle parent : Google Gemma 3 270M IT (unsloth/gemma-3-270m-it)   |
|    - Entraînement QLoRA / SFT multi-époques sur le corpus SautiLM-v1    |
|    - Fusion des poids + Compression structurelle & Quantification 4-bit |
|    - Export double :                                                    |
|      a) sautilm-v1.litertlm (<= 145 Mo) pour LiteRT-LM Flutter          |
|      b) sautilm-v1-q4.gguf (<= 135 Mo) pour CPU universel               |
+-------------------------------------------------------------------------+
                                    |
                    Publication automatique vers HF Hub
                                    v
+-------------------------------------------------------------------------+
| 3. HUGGING FACE HUB (https://huggingface.co/sautilm/SautiLM-v1)         |
|    - Hébergement officiel de sautilm-v1.litertlm et sautilm-v1-q4.gguf  |
+-------------------------------------------------------------------------+
                                    |
                    Téléchargement direct < 150 Mo
                                    v
+-------------------------------------------------------------------------+
| 4. APPLICATION MOBILE FLUTTER (Flutter/sautilm)                         |
|    - Noyau SautiLM-v1 (<= 150 Mo) via LiteRT-LM                         |
|    - Dictionnaire RAG Bbadha complet (1 150 mots + exemples trilingues) |
|    - Solveur Mathématique Déterministe embarqué (0 erreur de calcul)    |
|    - Module Vision/Image optimisé (512px + analyse multimodale locale)  |
+-------------------------------------------------------------------------+
```

---

## 3. Détail des Composants Techniques

### 3.1. Ingénierie du Corpus (`src/corpus_builder.py`)
Le générateur local transforme les sources vérifiées en un jeu d'instructions JSONL (`data/sautilm_v1_train.jsonl`) au format conversationnel Gemma/SautiLM :
1. **Extraction intégrale des 24 fichiers JSON Bbadha (`Flutter/bbadha/assets/jsons/a.json` à `z.json`)** :
   - **1 150 mots Bbadha** extraits avec phonétique, grammaire, sens numérotés (`sense_id`) et phrases d'exemples trilingues (`bbadha_sentence`, `swahili_sentence`, `french_sentence`).
   - Génération de paires d'instructions variées :
     - Traduction mot-à-mot et explication des nuances (Bbadha $\leftrightarrow$ Swahili $\leftrightarrow$ Français).
     - Traduction de phrases complètes Bbadha $\leftrightarrow$ Swahili et Bbadha $\leftrightarrow$ Français.
     - Questions grammaticales et phonétiques sur le Bbadha (Kilendu).
2. **Corpus Spécialisé Multi-Domaines (Socle des futurs modèles SautiLM)** :
   - **`SautiLM-Identity`** : Questions/réponses en Français, Swahili, Anglais et Bbadha sur l'identité de SautiLM, SautiLM Research (Bunia, Ituri), la mission humanitaire et le respect strict du formatage (zéro émoji, zéro tiret long `—`).
   - **`SautiLM-Edu` & `SautiLM-Med`** : Modules d'environnement (cycle de l'eau, énergie solaire, érosion), santé communautaire (prévention du choléra, purification de l'eau, premiers secours) et médecine vétérinaire pastorale.
   - **`SautiLM-Math`** : Problèmes arithmétiques contextualisés (gestion de bétail, vente de lait, partage de fournitures scolaires, géométrie pratique) avec décomposition étape par étape.
   - **`SautiLM-Ius` (Droit Congolais)** : Corpus fondamental sur la Constitution de la RDC, le Code de la Famille, le Code Pénal et le Code du Travail pour assister les citoyens, avocats et magistrats hors-ligne.
   - **`SautiLM-Vision`** : Instructions d'analyse d'images et de textes extraits (feuilles agricoles, exercices écrits au tableau/cahier, ordonnances ou documents).

### 3.2. Compression & Export sous la barre des 150 Mo (`kaggle_kernel/train_and_export.py`)
Pourquoi `gemma-3-270m-it` pèse 289 Mo en INT8 standard ?
- Le réseau de neurones (18 couches Transformer, `hidden_size=640`) ne contient que **100,3 millions de paramètres**.
- La matrice de vocabulaire (`262 144 tokens x 640`) représente à elle seule **167,8 millions de paramètres** (62 % du poids total).
- **Stratégie de compression SautiLM ($\le 150\text{ Mo}$)** :
  1. Fine-Tuning QLoRA (`r=16`, `lora_alpha=32`, ciblant `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) sur GPU Tesla T4.
  2. Fusion des poids LoRA dans le modèle de base.
  3. **Pour l'export `.litertlm` (`sautilm-v1.litertlm`)** : Distillation par élagage de profondeur calibré (conservation des 12 couches les plus informatives + calibration SFT post-pruning ou quantification INT4/INT8 mixte via `ai-edge-torch` / `flatbuffers`) pour garantir un conteneur `.litertlm` valide de **120 à 145 Mo** avec les métadonnées officielles `model_name: "SautiLM-v1-150M"`.
  4. **Pour l'export `.gguf` (`sautilm-v1-q4.gguf`)** : Quantification `Q4_K_M` avec `--token-embedding-type q4_0`, ramenant le fichier complet à **~132 Mo**.

### 3.3. Mise à jour de l'Application Mobile (`Flutter/sautilm`)
1. **Dictionnaire Bbadha Complet (`1 150 mots`)** :
   - Copie du dictionnaire unifié et compressé `assets/data/bbadha_full_lexicon.json` dans `Flutter/sautilm/assets/data/`.
   - Mise à jour de `BbadhaDictionary` (`lib/core/data/bbadha_dictionary.dart`) pour indexer en mémoire les **1 150 mots Bbadha** et leurs phrases d'exemples dès le démarrage (< 50 ms), tout en gardant les entrées existantes pour que 100 % des tests unitaires continuent de passer.
2. **Solveur Mathématique Déterministe (`lib/core/services/math_solver_service.dart`)** :
   - Détecte automatiquement les expressions et opérations mathématiques dans la requête de l'utilisateur, calcule le résultat exact de manière déterministe en Dart, et l'injecte dans le prompt système de `SautiLM-v1` pour que le modèle explique la démarche sans jamais se tromper de chiffre.
3. **Capacité Vision / Image dans le Chat (`lib/features/chat/`)** :
   - Ajout de la prise en charge d'image (caméra / galerie avec redimensionnement automatique à 512px max pour éviter tout pic de RAM) dans `ChatScreen`, `ChatBloc` et `EdgeAiService`.
4. **Détection et Téléchargement de `sautilm-v1.litertlm` (`<= 150 Mo`)** :
   - Mise à jour de `GemmaModelLocator` (`lib/core/services/gemma_model_locator.dart`) :
     - Ajout de `sautilm-v1.litertlm` en tête de `modelFilenames`.
     - Mise à jour de l'URL de téléchargement vers `https://huggingface.co/sautilm/SautiLM-v1/resolve/main/sautilm-v1.litertlm`.
     - Abaissement du seuil de validation `_isValidModelFile` de `50 Mo` à `25 Mo` pour accepter les modèles ultra-compacts SautiLM (< 150 Mo).
     - Affichage du badge `"SautiLM v1 LiteRT"` dans `EdgeAiService`.

---

## 4. Plan de Vérification & Critères de Succès

1. **Vérification des Données** : Le script `corpus_builder.py` doit extraire les 1 150 mots des 24 fichiers JSON sans erreur et produire au moins 6 000 exemples d'instructions trilingues + l'index mobile complet.
2. **Vérification du Modèle Distillé** : Le job GPU Kaggle doit se terminer avec succès, produire `sautilm-v1.litertlm` et `sautilm-v1-q4.gguf` de taille strictement **$< 150\text{ Mo}$**, et les publier sur `https://huggingface.co/sautilm/SautiLM-v1`.
3. **Vérification Mobile (`flutter analyze` & `flutter test`)** : L'application `Flutter/sautilm` doit compiler avec 0 erreur d'analyse statique, passer tous les tests, charger les 1 150 mots Bbadha, résoudre les calculs mathématiques et pointer vers `sautilm-v1.litertlm`.
