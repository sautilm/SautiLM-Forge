# SautiLM-Forge • AI Agent & Developer Guidelines

This file guides developers and AI coding assistants working in `Python/SautiLM-Forge`.

## Private Strategic Docs & Proprietary Datasets (`sautilm-internal-docs`)

* **Expected Local Path:** `../../sautilm-internal-docs` (`https://github.com/sautilm/sautilm-internal-docs.git`, private repository — never commit proprietary datasets or private strategy docs into this public repository).
* **Bootstrap Rule for AI Assistants & Developers:** Before refactoring model architecture, changing datasets, or launching new training pipelines, inspect `../../sautilm-internal-docs/` (specifically `docs/vision-and-architecture/SAUTILM_ARCHITECTURE_ET_VISION.md`) and local `docs/private/` if present. Those files contain the up-to-date architectural decisions, model naming conventions, evaluation baselines, and partnership requirements.

## Architecture & Naming Rules (Strict)

1. **Base Foundation Model Naming:** The base core text model is named strictly **`SautiLM-Jua`** (never *SautiLM-Jua-Core*).
2. **Modular Domain Specialists:** Domain-specialized bundles attach cleanly onto the base name with strict isolation (no cross-domain template pollution or unrequested dialect injection):
   * `SautiLM-Jua-STEM` (Secondary Education & National Exams — Maths, Physics, Chemistry, Biology)
   * `SautiLM-Jua-Lex` (Bbadha / Kilendu, Kiswahili, French, English lexicon & translation)
   * `SautiLM-Jua-Med` (Health & Medicine)
   * `SautiLM-Jua-Agri` (Agronomy & Crops)
   * `SautiLM-Jua-Ius` (Congolese Law & Justice — always `Ius`)
   * `SautiLM-Jua-Fin` (Finance & Accounting)
   * `SautiLM-Jua-Civ` (Civic Administration & Mediation)
3. **Export Standard:** Always verify exported `.litertlm` artifacts via real `litert_lm.Engine` execution (`CPU` backend, `dynamic_wi8_afp32` quantization) before uploading or shipping. Keep `.env`, `kaggle.json`, `.litertlm`, `.jsonl` corpus, and `docs/private/` strictly excluded from Git (`git status`).
