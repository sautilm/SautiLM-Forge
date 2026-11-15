# SautiLM-Forge & SautiLM-v1 (<= 150 Mo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `SautiLM-Forge` Python training/distillation pipeline, generate the complete 1,150-word Bbadha + Math + Education + Health + Congolese Law (`SautiLM-Ius`) dataset, launch the remote GPU training & compression job on Kaggle to produce `sautilm-v1.litertlm` and `sautilm-v1-q4.gguf` ($\le 150\text{ Mo}$) published to `https://huggingface.co/sautilm/SautiLM-v1`, and integrate the full 1,150-word Bbadha lexicon, deterministic math solver, vision support, and `sautilm-v1.litertlm` locator into the Flutter mobile app (`Flutter/sautilm`).

**Architecture:** Hybrid local data engineering + automated remote Kaggle GPU fine-tuning/distillation of `gemma-3-270m-it` + LiteRT/GGUF sub-150MB compression + Flutter mobile neuro-symbolic RAG, math solver, and multimodal vision integration.

**Tech Stack:** Python 3.13 (`kaggle`, `huggingface_hub`, `python-dotenv`, `pytest`), PyTorch / Transformers / PEFT / LiteRT (`ai-edge-torch` / FlatBuffers) / `llama.cpp` on Kaggle GPU, Flutter / Dart 3 (`flutter_bloc`, `flutter_edge_ai`, `flutter_edge_ai_litertlm`, `image_picker`).

**Spec:** `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\docs\specs\2026-11-15-sautilm-v1-forge-design.md`

## Global Constraints
- Model size on mobile MUST NOT exceed 150 MB (`<= 150 MB`).
- Zero emojis (except 🇬🇧, 🇫🇷, 🇹🇿 for language selector) and zero long dashes (`—` or `–`) in all SautiLM outputs and UI.
- Secrets in `.env` MUST NEVER be printed, logged, or committed to Git.
- All 1,150 Bbadha words from `Flutter/bbadha/assets/jsons/*.json` must be preserved and indexed.

---

### Task 1: SautiLM-Forge Corpus Builder & Full Bbadha Mobile Lexicon Compiler

**Files:**
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\src\__init__.py`
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\src\corpus_builder.py`
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\tests\test_corpus_builder.py`

**Interfaces:**
- Produces:
  - `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\data\sautilm_v1_train.jsonl`
  - `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\data\bbadha_full_lexicon.json`
  - `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\assets\data\bbadha_full_lexicon.json`

- [x] **Step 1: Write unit tests for Bbadha JSON extraction, instruction formatting (zero emojis/long dashes), and mobile lexicon generation**
- [x] **Step 2: Implement `src/corpus_builder.py` covering all 1,150 Bbadha entries, trilingual examples, SautiLM identity, Math, Edu, Med, Vision, and Congolese Law (`SautiLM-Ius`)**
- [x] **Step 3: Run `pytest` and generate the full corpus and mobile lexicon JSON**

---

### Task 2: Kaggle GPU Training, Compression (<= 150 MB), LiteRT/GGUF Export & Hugging Face Publisher

**Files:**
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\kaggle_kernel\train_and_export.py`
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\src\kaggle_orchestrator.py`
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Python\SautiLM-Forge\tests\test_litertlm_packager.py`

**Interfaces:**
- Consumes: `data/sautilm_v1_train.jsonl`
- Produces:
  - Remote GPU training & distillation job on Kaggle
  - `sautilm-v1.litertlm` ($\le 145\text{ Mo}$) and `sautilm-v1-q4.gguf` ($\le 135\text{ Mo}$) uploaded to `https://huggingface.co/sautilm/SautiLM-v1`

- [x] **Step 1: Write unit test verifying the `.litertlm` v1 FlatBuffer container header against `flutter_edge_ai_litertlm` specification**
- [x] **Step 2: Implement `kaggle_kernel/train_and_export.py` (QLoRA fine-tuning of Gemma 3 270M IT + structural layer calibration/INT4 compression to <= 145 MB + `.litertlm` and `.gguf` export + push to `sautilm/SautiLM-v1`)**
- [x] **Step 3: Implement `src/kaggle_orchestrator.py` and push dataset + GPU kernel to Kaggle**

---

### Task 3: Flutter Mobile Upgrade (`Flutter/sautilm`) — 1,150 Bbadha Words, Deterministic Math Solver, Vision Support & `sautilm-v1.litertlm`

**Files:**
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\pubspec.yaml`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\core\data\bbadha_dictionary.dart`
- Create: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\core\services\math_solver_service.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\core\services\services.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\core\services\gemma_model_locator.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\core\services\edge_ai_service.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\features\chat\bloc\chat_event.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\features\chat\bloc\chat_bloc.dart`
- Modify: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\lib\features\chat\chat_screen.dart`
- Test: `C:\Users\ADMIN\Desktop\MiProjects\Flutter\sautilm\test\sautilm_v1_upgrade_test.dart`

- [x] **Step 1: Write unit tests for full Bbadha dictionary loading, deterministic math solver, and `sautilm-v1.litertlm` model locator**
- [x] **Step 2: Implement `MathSolverService`, upgrade `BbadhaDictionary` (1,150 words + examples), update `GemmaModelLocator` & `EdgeAiService`, and add image vision capture/analysis to `ChatScreen`**
- [x] **Step 3: Run `flutter analyze` and `flutter test` to verify 0 errors and all tests passing**
