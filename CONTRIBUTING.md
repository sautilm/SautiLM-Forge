# Contributing to SautiLM-Forge

Thank you for your interest in contributing to **SautiLM** and **SautiLM Research**! Our mission is to build sovereign, 100% offline Edge-AI models for marginalized languages, displaced learners, healthcare workers, and legal practitioners across the Democratic Republic of the Congo and East Africa.

---

## How You Can Contribute

### 1. Linguistic Contributors (East-DRC & African Languages)
We are actively expanding `SautiLM-Lex` beyond **Bbadha (Kilendu)** and **Kiswahili** to cover:
- **Hema (Kihema)**, **Alur**, **Lugbara**, **Ngiti (Kingiti)**, **Lese**, **Nande (Kinande)**, **Shi (Mashi)**, **Lingala**, **Tshiluba**, and **Kikongo**.
- Linguistic contributions follow our standardized JSON schema (`word`, `phonetics`, `grammar`, `meanings`, and trilingual `examples`).

### 2. Domain Experts (Law, Health, Education, Agriculture)
- **SautiLM-Ius (Congolese Law)**: Magistrates, lawyers (*avocats*), and legal scholars can contribute verified Q&A pairs referencing exact articles of the DRC Official Journal (*Journal Officiel de la RDC*) and OHADA Uniform Acts.
- **SautiLM-Med & SautiLM-Agri**: Physicians, nurses, and agronomists can contribute validated offline triage and crop-disease protocols.

### 3. ML & Mobile Edge-AI Engineers
- Quantization, pruning, and `ai-edge-torch` / `LiteRT-LM` / `GGUF` compilation optimizations for sub-150 MB inference on low-RAM Android devices.

---

## Code & Formatting Standards

1. **Zero Credential Leakage**: Never commit `.env` files, API tokens, or private keys.
2. **SautiLM Text Hygiene**: All training responses must use clean Markdown (`**bold**`, numbered lists `1.`, bullets `- `) with **zero emojis** and **zero long dashes (`—` or `–`)**.
3. **Unit Tests Required**: Run `python -m unittest discover -s tests -v` before opening a Pull Request.
