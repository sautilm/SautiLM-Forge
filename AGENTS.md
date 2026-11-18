# SautiLM-Forge • AI Agent & Developer Guidelines

This file guides developers and AI coding assistants working in `Python/SautiLM-Forge`.

## Private Strategic Docs & Proprietary Datasets (`sautilm-internal-docs`)

* **Expected Local Path:** `../../sautilm-internal-docs` (`https://github.com/sautilm/sautilm-internal-docs.git`, private repository — never commit proprietary datasets or private strategy docs into this public repository).
* **Bootstrap Rule for AI Assistants & Developers:** Before refactoring model architecture, changing datasets, or launching new training pipelines, inspect `../../sautilm-internal-docs/` (specifically `docs/vision-and-architecture/SAUTILM_ARCHITECTURE_ET_VISION.md`) and local `docs/private/` if present. Those files contain the up-to-date architectural decisions, model naming conventions, evaluation baselines, and partnership requirements.

## Architecture & Naming Rules (Strict)

1. **Three Official Product Series (Strict Naming — Ask Founder Before Adding Any New Prefix):**
   * **`SautiLM-Jua`** (`270M` INT8, `~272 MB` — common/standard mobile models, never *SautiLM-Jua-Core*):
     * `SautiLM-Jua-STEM` (Secondary Education & DRC Examen d'État — Maths, Physics, Chemistry, Biology)
     * `SautiLM-Jua-Lex` (Bbadha / Kilendu, Kiswahili, French, English lexicon & translation)
     * `SautiLM-Jua-Med` (Community Health & Mobile Triage)
     * `SautiLM-Jua-Agri` (Agronomy & Crops)
     * `SautiLM-Jua-Ius` (Congolese Law & Justice — always `Ius`)
     * `SautiLM-Jua-Fin` (Finance & Accounting)
     * `SautiLM-Jua-Civ` (Civic Administration & Mediation)
     * `SautiLM-Jua-Voice`, `SautiLM-Jua-Vision`
   * **`SautiLM-Nuru`** (Lightweight, fast collaborative Vision models in `TFLite INT8`, target `~15-45 MB`, passing structured detections/OCR to SQLite RAG + `Jua`/`Kuu` instead of `1.5 GB+` mobile VLM downloads):
     * `SautiLM-Nuru-Agri`, `SautiLM-Nuru-STEM`, `SautiLM-Nuru-Med`, `SautiLM-Nuru-Ius`
   * **`SautiLM-Kuu`** (High-capacity tier: compact `1B` `< 650 MB` for capable mobile/tablets & `4B` Desktop PC/Station):
     * `SautiLM-Kuu-STEM` (`1B` compact for AndaaEdge / PDF quiz & exam solving)
     * `SautiLM-Kuu-Med` (`4B` Desktop PC for AfyaSpeak live offline clinical assistant)
     * `SautiLM-Kuu-Ius` (`4B` Desktop PC for lawyers, magistrates & DRC courts)
2. **Strict Domain Isolation & Evaluation:** Never mix train and held-out evaluation prompts, never leak cross-domain templates (zero Bbadha in STEM/Med/Agri/Ius), and always include negative abstention tests alongside grounded RAG checks.
3. **Export Standard:** Always verify exported `.litertlm` artifacts via real `litert_lm.Engine` execution (`CPU` backend, `dynamic_wi8_afp32` quantization; multi-prefill `[128, 512, 1024]` where applicable) before uploading or shipping. Keep `.env`, `kaggle.json`, `.litertlm`, `.jsonl` corpus, and `docs/private/` strictly excluded from Git (`git status`).
