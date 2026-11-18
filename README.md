# SautiLM-Forge

**SautiLM-Forge** is the official corpus engineering, LoRA fine-tuning, and sub-150 MB mobile Edge-AI packaging pipeline developed by **LosingTek Research** (Bunia, Ituri, Democratic Republic of the Congo).

It powers the **SautiLM** family of offline mobile language models engineered for low-resource African languages (starting with **Bbadha / Kilendu** and **Congolese Swahili**) and critical community domains in Eastern DRC.

- **Official Website**: [https://sautilm.com](https://sautilm.com)
- **Hugging Face Model Hub (`SautiLM Jua`)**: [https://huggingface.co/sautilm/SautiLM-Jua](https://huggingface.co/sautilm/SautiLM-Jua)
- **Organization**: LosingTek Research (Bunia, Ituri, DRC)

---

## 1. SautiLM Foundation Series & 10 Specialized Domain Models

### Three Official Product Series (Strict Naming)
- **`SautiLM-Jua`** (`270M` INT8 LiteRT-LM, `~283.48 MB`, `dynamic_wi8_afp32`, multi-prefill `[128, 512, 1024]`) : Standard mobile foundation & domain models (`SautiLM-Jua`, `SautiLM-Jua-STEM`, `SautiLM-Jua-Lex`, `SautiLM-Jua-Med`, `SautiLM-Jua-Agri`, `SautiLM-Jua-Ius`, `SautiLM-Jua-Fin`, `SautiLM-Jua-Civ`).
- **`SautiLM-Nuru`** (`TFLite INT8`, `<= 45 MB` — e.g. `SautiLM-Nuru-Agri` at `1.074 MB`, 9 classes with `non_plant` rejection gate at `< 0.55`) : Fast collaborative mobile vision classifiers passing structured triage to SQLite FTS5 RAG + `Jua`/`Kuu`.
- **`SautiLM-Kuu`** (`1B` mobile/tablet LiteRT-LM `SautiLM-Kuu-STEM` for `AndaaEdge.com` / `Shul'etu` PDF quiz & EXETAT solving; `4B` Desktop PC `GGUF Q4_K_M` `SautiLM-Kuu-Med` & `SautiLM-Kuu-Ius` for `AfyaSpeak` & DRC courts, strictly excluded from mobile APKs).

### 10 Specialized Domain Modules
| Model | Target Domain | Core Capabilities |
| :--- | :--- | :--- |
| **`SautiLM-Lex`** | Lexicography & Translation | Bidirectional translation & grammatical dictionary for Bbadha (Kilendu), Kiswahili, French, and English |
| **`SautiLM-Edu`** | Education & General Sciences | Offline tutor for general sciences, biology, physics, and environment |
| **`SautiLM-Math`** | Mathematics & Logic | Verified arithmetic, percentages, algebra, and commercial calculations |
| **`SautiLM-Med`** | Public Health & Medicine | Clinical triage, water sanitation, oral rehydration (ORS), and disease prevention |
| **`SautiLM-Ius`** | Congolese Law & Justice | Legal assistant grounded in the DRC Constitution, revised Family Code (Loi n° 16/008), Penal Code, Labor Code, and OHADA Law |
| **`SautiLM-Vision`** | Camera OCR & Visual Analysis | Offline optical character recognition for written exercises and visual plant/document diagnostics |
| **`SautiLM-Voice`** | Phonetics & Orality | Tonal pronunciation guidance and speech interface for Eastern DRC languages |
| **`SautiLM-Agri`** | Agronomy & Crops | Field diagnosis for cassava mosaic, maize rust, coffee wilt, composting, and soil fertility |
| **`SautiLM-Fin`** | Finance, SME Accounting & AVEC | Cooperative accounting, village savings (VSLA/AVEC), and FC/USD calculations |
| **`SautiLM-Civ`** | Civic Administration | Civil status registration (`Acte de naissance`), public services, and community mediation |

---

## 2. Repository Structure

```text
SautiLM-Forge/
├── src/
│   ├── corpus_builder.py            # Curates 1,150 Bbadha entries & zero-duplication SFT corpora
│   ├── specialty_sft_generators.py  # Parametric Med (271), Agri (253) & Ius (265) unique pairs
│   ├── stem_sft_generators.py       # Parametric secondary STEM & AndaaEdge PDF->QCM (639 unique pairs)
│   ├── nuru_vision_trainer.py       # SautiLM-Nuru-Agri 9-class TFLite INT8 collaborative vision trainer
│   ├── litertlm_packager.py         # Validates .litertlm FlatBuffer mobile containers
│   └── kaggle_orchestrator.py       # Multi-target orchestrator (Jua 270M, Kuu 1B/4B, Nuru TFLite) -> HF Hub
├── kaggle_kernel/
│   └── train_and_export.py          # Zero-leakage stratified split, Early Stopping on val_loss & LiteRT-LM/GGUF export
├── kaggle_nuru_kernel/
│   └── train_nuru_vision_kaggle.py  # Remote GPU MobileNetV3-Small INT8 TFLite vision trainer (1.074 MB)
└── tests/                           # 17 unit tests verifying zero train/val/eval leakage & container integrity
```

> **Data Sovereignty & IP Protection Note**: To protect community linguistic assets and prevent unauthorized scraping, the full 1,150-word Bbadha JSON dictionary and the 35 MB `sautilm_v1_train.jsonl` corpus are excluded from public Git tracking via `.gitignore`. Only `data/sample_train_schema.jsonl` and `data/corpus_stats.json` are versioned publicly.
>
> **Android On-Device Deployment Note (`sautilm-jua.litertlm`)**: On Android 11+, `Flutter/sautilm` stores `sautilm-jua.litertlm` inside an invisible private vault (`noBackupFilesDir/sautilm_models/`) exposed read-only to LosingTek / partner apps via `content://com.losingtek.sautilm.modelprovider/models/` (`0 KB` network re-download, Play Store compliant — **Option 1**), while also supporting `/storage/emulated/0/.SautiLM/models/` with `.nomedia` (`MANAGE_EXTERNAL_STORAGE`) for zero-copy field tablet deployments (**Option 2**).

---

## 3. Quick Start

### 3.1. Local Environment Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Populate .env locally with your HF_TOKEN and KAGGLE_KEY (never committed to Git)
```

### 3.2. Build Corpus & Run Unit Tests
```bash
python src/corpus_builder.py
python -m unittest discover -s tests -v
```

### 3.3. Launch Remote GPU Training & Publish to Hugging Face
```bash
python src/kaggle_orchestrator.py launch
python src/kaggle_orchestrator.py status
```

---

## 4. Founding & Research Team (LosingTek Research)

- **Georges Byona** - Co-Founder, Software Engineer & Edge-AI Researcher (Catholic University of Eastern Africa / LosingTek)
- **Abraham Imani** - Co-Founder & AI/ML Lead (Université Catholique de Bukavu)
- **Jacques Uwonda** - Co-Founder & Software Engineer (Université Shalom de Bunia)

---

## 5. License

Code in this repository is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)). Model weights derived from Google Gemma are subject to the **Gemma Terms of Use**.
