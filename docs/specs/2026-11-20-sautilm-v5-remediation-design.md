# Spécification Architecturale v5 : Remédiation Complète Post-Audit v4 (`7.0 -> 9.0+ / 10`)

**Date :** 20 novembre 2026  
**Auteur :** SautiLM Research — Équipe SautiLM  
**Statut :** Approuvé en design (`Approche A — Contrats Déterministes Partagés SDK + Ré-entraînement Ciblé Val/Test Séparé`)

---

## 1. Objectif & Constats de Départ (Audit v4 : `8.9 / 10` Ingénierie, `7.0 / 10` Poids & Terrain)

L'Audit Scientifique Forensique v4 a confirmé la propreté des 4 bases SQLite FTS5 (`0` Wikipedia, `0` espagnol, `0` arabe corrompu, `0` HTML), la qualité de `SautiLM-Nuru-Agri` v2 (`1.074 Mo`, `4 032` photos réelles, `92.06 %` val acc, `87.5 %` TFLite INT8 réel), et le budget mobile de `SautiLM-Kuu-STEM` (`549.58 Mo <= 650 Mo`).

En contrepartie, 6 défauts scientifiques et opérationnels bloquent l'adoption par un comité de bailleurs (Google DeepMind, Wellcome Trust, Gates Foundation, Lacuna Fund) :
1. **Fuite d'indices dans `_runtime_suite_for_target` (`train_and_export.py` L765–L866)** : 12 prompts contiennent la réponse ou formule attendue entre parenthèses/system prompt, et `selected_lora_scale` est sélectionné directement sur ce même jeu (L1749–L1776).
2. **Hallucinations statutaires sur `SautiLM-Kuu-Ius` (`4B GGUF`)** : `training_metrics.json` montre `5/7` réponses comptées réussies malgré des articles inventés (`Art. 27/13` Constitution, `Art. 37(1)-(2)` Famille), un faux Acte `AUSCA 1991` et une erreur `12 000 USD chacun` (`36 000` sur `16 000 USD`).
3. **Plafond physique du `270M` (`SautiLM-Jua-*`)** : `0` gain net sur Base après INT8, overdoses (`1000 mg/kg`, `700 mg par prise`) sur `Jua-Med`, recommandations aberrantes sur `Jua-Agri` (*Dermocid* pour le BXW), validation du mariage à 17 ans sur `Jua-Ius`, et `bbadha_eval_v1` bas (`Jua`: `5.56 %` brut / `47.78 %` hybride ; `Jua-Lex`: `11.11 %` / `15.56 %`).
4. **Trou `guarded_production` dans `SautiLM-Eval/src/evaluator.py`** : `apply_sovereign_safety_short_circuit()` (L336–L369) ne court-circuite que les 3 refus négatifs et laisse passer `DOMAIN-MED-CALC-003` (`700 mg par prise`, surdosage ×2) et `DOMAIN-STEM-EXAM-005` (`20 V` au lieu de `30 V`).
5. **Couverture incomplète des bases SQLite FTS5** : `8/10` Actes Uniformes OHADA à `1` seul résumé, absence des guides officiels PNLP/MSP RDC dans `afyaspeak_oms_msf.sqlite`, et `CD`, `RW`, `BI` limités à `2` lignes chacun dans `gemmea_eac_agri.sqlite` par le routage mono-pays dans `compile_dbs.py`.
6. **Lacunes opérationnelles `Forge` / `AfyaSpeak` / `README`** : `adapter_model.safetensors` à `0 octet` dans `kaggle_output/stem/`, absence de `pyproject.toml` et d'artefact `.onnx` `3dspeaker` + accélération GPU/Vulkan pour `AfyaSpeak`, et lignes `27-100` `Gemma 4` obsolètes dans `Flutter/gemmea/README.md`.

---

## 2. Décisions Architecturales Verrouillées

1. **Tiers & Nomenclature Stricts (`AGENTS.md`) :** Conserver `SautiLM-Nuru-*`, `SautiLM-Jua` (`270M`), `SautiLM-Kuu-STEM` (`1B <= 650 Mo` mobile), `SautiLM-Kuu-Med` & `SautiLM-Kuu-Ius` (`4B Q4_K_M` PC). Ne créer aucun nouveau préfixe/modèle `1B` hors `Kuu-STEM`.
2. **Contrôle Déterministe Obligatoire sur Mobile (`sautilm_sdk`) :**
   - `270M` (`SautiLM-Jua-*`) **strictement interdit** pour toute posologie pédiatrique chiffrée, toute attribution d'article de loi non vérifié ou tout diagnostic phytosanitaire autonome.
   - **`Lex` (`Bbadha`):** rendu **SQLite-first déterministe** depuis `bbadha_lexicon.sqlite` (`< 10 ms`, `100 %` tonalité/glottale préservée) avant toute reformulation optionnelle.
   - **`Gemmea` (`Agri`):** après triage visuel `SautiLM-Nuru-Agri`, rendu prioritaire de la fiche vérifiée `gemmea_eac_agri.sqlite` en Pass 2.
   - **`Ius` (RDC/OHADA) & `STEM`:** calculateurs symboliques (`quadratique`, `U=R*I`, stœchiométrie, partage successoral `16000 * 3/4 / 3`) + `LegalCitationVerifier` bloquant tout numéro d'article ou d'Acte Uniforme absent des passages SQLite FTS5 retournés.
3. **Protocole Expérimental Zéro-Fuite (`Forge` & `Eval`) :**
   - Zéro indice entre parenthèses dans les prompts held-out.
   - Jeu `val_tuning` séparé du jeu `heldout_blind` pour la sélection de `epoch` et `lora_scale`.
   - Ré-évaluation locale aveugle de tous les 10 artefacts existants + ré-entraînement GPU Kaggle ciblé sur `kuu-stem`, `kuu-ius` et réparation/re-capture de `stem` (`adapter_model.safetensors > 0` o).
4. **Sources Externes Open-Access vs Lot Terrain :**
   - Dans ce sprint : ingestion open-access des **8 Actes Uniformes OHADA FR**, guides **PNLP / MSP RDC**, fiches **INERA / RAB / ISABU** + routage multi-pays dans `compile_dbs.py`, et modèle **ONNX `3dspeaker`** (`~25 Mo`) + `pyproject.toml` + benchmark GPU/CPU `AfyaSpeak`.
   - Lot terrain dédié (hors blocage de code de ce sprint) : protocole documenté pour collecte de `200+` photos smartphones Bukavu/Bunia (`Nuru-Agri`) et profilage RAM/thermique sur smartphone `4 Go RAM` physique.

---

## 3. Architecture Cible & Frontières de Composants

```
                 +--------------------------------------------------+
                 |          sautilm_sdk (Python & Dart)             |
                 | - MedicalDosingCalculator / PediatricDosing      |
                 | - StemScienceSolver (Quad, Ohm, Stoich, <calc>)  |
                 | - LegalDeadlineCalculator + DRC Succession Split |
                 | - LegalCitationVerifier (FTS5 Chunk Grounding)   |
                 | - LexiconCardFormatter / AgriFicheFormatter      |
                 +---------+-------------------+--------------------+
                           |                   |
         +-----------------+                   +------------------+
         v                                                        v
+-------------------------+                          +--------------------------+
|   SautiLM-Eval / Afya   |                          | Mobile Flutter (SautiLM  |
| - apply_safety_guard    |                          | & Gemmea)                |
| - Citation Lock Filter  |                          | - Lex: SQLite card <10ms |
| - Zero-Hint Held-Out    |                          | - Agri: Nuru -> FTS5 card|
| - ONNX 3D-Speaker Test  |                          | - Med/Ius: SDK + Lock    |
+-------------------------+                          +--------------------------+
```

---

## 4. Critères d'Acceptation Quantifiés

- **Hygiène expérimentale :** `0` prompt contenant la réponse entre parenthèses dans `train_and_export.py` et `evaluator.py` ; séparation vérifiée par test unitaire (`val_tuning_suite` disjoint de `heldout_blind_suite`).
- **Sécurité clinique & juridique (`guarded_production`) :** `100 %` sur `domain_eval_v2` (y compris `DOMAIN-MED-CALC-003` -> `350 mg/prise` exact et `DOMAIN-STEM-EXAM-005` -> `30 V` exact) et rejet de toute fabrication type `AUSCA 1991` ou `Article 37(1)` hors extraits FTS5.
- **Qualité des 4 SQLite FTS5 :** `0` contamination (Wikipedia/arabe/espagnol/HTTP), `10/10` Actes OHADA avec textes complets multi-articles (`>= 10` chunks chacun), `CD >= 20`, `RW >= 20`, `BI >= 20` dans `gemmea_eac_agri.sqlite`.
- **Qualité `AfyaSpeak` & Poids HF :** `unittest` exécutable directement dans `Python/AfyaSpeak` via `pyproject.toml`, test ONNX `3dspeaker` réel vert, `adapter_model.safetensors > 0` o pour `stem`, ré-entraînement/publication `Kuu-STEM` & `Kuu-Ius` + mise à jour honnête des `10/10` Model Cards HF.
