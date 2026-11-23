# SautiLM-Forge

Follow all architecture, privacy, and training guidelines in [AGENTS.md](AGENTS.md).

In particular, verify that the private reference repository `../../sautilm-internal-docs` (`https://github.com/sautilm/sautilm-internal-docs.git`) exists on this machine and read `docs/vision-and-architecture/SAUTILM_ARCHITECTURE_ET_VISION.md` before designing new model variants or datasets. Never commit private datasets or docs into this public repository.

## Institutional Identity & Brand Rule (Strict — Mandatory for All AI Agents)

* **Official Institution Name:** Always use **`SautiLM Research`** (or **`SautiLM`**, Bunia, Ituri, Democratic Republic of the Congo — `https://sautilm.com`) as the sole research institution, publisher, and model creator across all papers, system prompts, model cards, `CITATION.cff`, and UI strings.
* **Partner Role of MusanTek:** Never write `"MusanTek Research"` or `"by MusanTek"`. **MusanTek** (`https://musantek.com`) is strictly an engineering and incubation **partner** supporting SautiLM (note: existing Android technical package identifiers `com.musantek.*` must remain unchanged for APK compatibility).

## Research Paper Style Standard (Locked — Overleaf LaTeX Only)

* **No Local PDFs:** Never generate local PDFs for research papers; always edit the Overleaf LaTeX source (../../sautilm-internal-docs/docs/releases-and-papers/latex_overleaf/main.tex).
* **Locked Layout (Do Not Change):** Classic IEEE / APA / arXiv academic single-column style using **Latin Modern Roman** (lmodern, 10pt, 22mm margins), clean centered title with no horizontal rules and no manual line breaks, single-line \mbox{...} authors with compact circular green \orcidicon{...}, 1-line italic affiliations, **pure white indented Abstract** (djustwidth{14mm}{14mm}) with Index Terms---, ooktabs tables wrapped in esizebox{	extwidth}{!}{...}, [H] float placement, and CRediT Author Contributions.
