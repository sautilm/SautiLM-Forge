# SautiLM-Forge

**SautiLM-Forge** is the official corpus engineering, LoRA fine-tuning, and sub-150 MB mobile Edge-AI packaging pipeline developed by **LosingTek Research** (Bunia, Ituri, Democratic Republic of the Congo).

It powers the **SautiLM** family of offline mobile language models engineered for low-resource African languages (starting with **Bbadha / Kilendu** and **Congolese Swahili**) and critical community domains in Eastern DRC.

- **Official Website**: [https://sautilm.com](https://sautilm.com)
- **Hugging Face Model Hub (`SautiLM Jua`)**: [https://huggingface.co/sautilm/SautiLM-Jua](https://huggingface.co/sautilm/SautiLM-Jua)
- **Organization**: LosingTek Research (Bunia, Ituri, DRC)

---

## 1. SautiLM Foundation Series & 10 Specialized Domain Models

### Foundation Model Series (3-4 Letter Swahili / Kingwana Names)
- **`SautiLM Jua`** (*Sun / Knowledge*) : Sub-100 MB Edge-AI foundation model (`92.48 MB`, Active on Hugging Face)
- **`SautiLM Nuru`** (*Light / Clarity*) : Mid-sized multimodal & reasoning model (In development)
- **`SautiLM Wazo`** (*Thought / Deep Reasoning*) : Analytical model for law, STEM, and research (In development)
- **`SautiLM Moto`** (*Fire / High Speed*) : Ultra-low-latency streaming and voice model (In development)
- **`SautiLM Kuu`** (*Supreme / Flagship*) : Full-scale institutional model for universities, courts, and hospitals (In development)
- **`SautiLM Mali`** (*Heritage / Treasure*) : Sovereign linguistic and cultural corpus foundation (In development)

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
