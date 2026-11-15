# SautiLM-Forge

**SautiLM-Forge** is the official corpus engineering, LoRA fine-tuning, and sub-150 MB mobile Edge-AI packaging pipeline developed by **LosingTek Research** (Bunia, Ituri, Democratic Republic of the Congo).

It powers the **SautiLM** family of offline mobile language models engineered for low-resource African languages (starting with **Bbadha / Kilendu** and **Congolese Swahili**) and critical community domains in Eastern DRC.

- **Official Website**: [https://sautilm.com](https://sautilm.com)
- **Hugging Face Model Hub**: [https://huggingface.co/sautilm/SautiLM-v1](https://huggingface.co/sautilm/SautiLM-v1)
- **Organization**: LosingTek Research (Bunia, Ituri, DRC)

---

## 1. Architecture & Product Family

`SautiLM-Forge` builds the foundation corpus and training pipeline for the **SautiLM** ecosystem:

| Model | Target Domain | Core Capabilities |
| :--- | :--- | :--- |
| **`SautiLM-v1`** | Unified Omni Edge ($\le 150\text{ MB}$) | Compact mobile foundation model distilled from Google Gemma 3 (`270M-IT`) for 2 GB - 4 GB RAM Android phones |
| **`SautiLM-Lex`** | East-DRC Indigenous Languages | Bidirectional translation & grammatical dictionary for Bbadha (Kilendu), Kiswahili, French, and English |
| **`SautiLM-Edu`** | Mother-Tongue Education | Offline pocket tutor for environmental science, biology, literacy, and interactive quizzes |
| **`SautiLM-Math`** | Contextual Arithmetic | Step-by-step arithmetic and pastoralist commerce calculations in Congolese Francs (FC) |
| **`SautiLM-Med`** | Community Health & Livestock | Actionable prevention guides for water sanitation, cholera, malaria, nutrition, and veterinary care |
| **`SautiLM-Ius`** | Offline Congolese Law | Legal assistant grounded in the DRC Constitution, revised Family Code (Loi n° 16/008), Penal Code, Labor Code, and OHADA Law |

---

## 2. Repository Structure

```text
SautiLM-Forge/
├── src/
│   ├── corpus_builder.py        # Extracts verified linguistic JSONs & generates 22,834 SFT instructions
│   ├── litertlm_packager.py     # Builds & validates .litertlm v1.5.0 FlatBuffer mobile containers
│   └── kaggle_orchestrator.py   # Hybrid orchestrator: Local PC -> Remote Kaggle T4 GPU -> Hugging Face Hub
├── kaggle_kernel/
│   └── train_and_export.py      # Remote GPU LoRA fine-tuning (r=16, alpha=32) & <= 150 MB compression
├── data/
│   ├── corpus_stats.json        # Verified statistics of the 22,834-instruction corpus
│   └── sample_train_schema.jsonl# Public schema sample (5 example instructions across domains)
├── kaggle_output/
│   └── training_metrics.json    # Empirical Tesla T4 GPU convergence metrics (189 steps, Loss: 5.73 -> 0.48)
├── docs/
│   ├── specs/                   # Architectural specifications
│   └── plans/                   # Implementation plans
└── tests/                       # Unit test suite for corpus generation and .litertlm binary validation
```

> **Data Sovereignty & IP Protection Note**: To protect community linguistic assets and prevent unauthorized scraping, the full 1,150-word Bbadha JSON dictionary and the 69 MB `sautilm_v1_train.jsonl` corpus are excluded from public Git tracking via `.gitignore`. Only `data/sample_train_schema.jsonl` and `data/corpus_stats.json` are versioned publicly.

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

- **Georges Byona** - Co-Founder, Software Engineer & Edge-AI Researcher
- **Abraham Imani** - Co-Founder & AI/ML Lead (Université Catholique de Bukavu)
- **Jacques Uwonda** - Co-Founder & Software Engineer (Université Shalom de Bunia)

---

## 5. License

Code in this repository is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)). Model weights derived from Google Gemma are subject to the **Gemma Terms of Use**.
