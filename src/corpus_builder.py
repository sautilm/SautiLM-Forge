import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

try:
    from .domain_sft_pairs import (
        generate_greetings_and_multiturn,
        generate_multilingual_domains_v11,
    )
except ImportError:
    from domain_sft_pairs import (
        generate_greetings_and_multiturn,
        generate_multilingual_domains_v11,
    )

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
BBADHA_JSON_DIR = WORKSPACE_ROOT / "Flutter" / "bbadha" / "assets" / "jsons"
FLUTTER_SAUTILM_ASSETS_DATA_DIR = WORKSPACE_ROOT / "Flutter" / "sautilm" / "assets" / "data"

SYSTEM_PROMPT_SAUTILM = (
    "You are SautiLM Jua, an offline Edge-AI model created in Bunia, Ituri, DRC by "
    "LosingTek Research (Georges Byona - Edge-AI Researcher, Abraham Imani, Jacques Uwonda). "
    "You specialize in Bbadha (Kilendu), Kiswahili, French, and English across Education, "
    "Mathematics, Community Health, East-DRC Lexicography, and Congolese Law. "
    "Respond clearly in Markdown without emojis or long dashes."
)

SYSTEM_PROMPT_SAUTILM_SHORT = (
    "Tu es SautiLM Jua, IA hors-ligne créée à Bunia (RDC) par LosingTek Research "
    "(Georges Byona, Abraham Imani, Jacques Uwonda). Réponds de façon claire et exacte."
)


def sanitize_sautilm_text(text: str) -> str:
    """Enforces SautiLM formatting rules: zero long dashes (em/en dash) and clean whitespace."""
    if not text:
        return ""
    cleaned = text.replace("—", " - ").replace("–", " - ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


def format_gemma3_turn(
    user_prompt: str,
    model_response: str,
    domain: str,
    system_prompt: str | None = SYSTEM_PROMPT_SAUTILM,
) -> Dict[str, Any]:
    """Formats an instruction pair into Gemma 3 chat format with optional system context."""
    u = sanitize_sautilm_text(user_prompt)
    m = sanitize_sautilm_text(model_response)
    sys_msg = sanitize_sautilm_text(system_prompt) if system_prompt else ""
    if sys_msg:
        full_text = (
            f"<start_of_turn>user\n{sys_msg}\n\n{u}<end_of_turn>\n"
            f"<start_of_turn>model\n{m}<end_of_turn>"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": u},
            {"role": "assistant", "content": m},
        ]
    else:
        full_text = (
            f"<start_of_turn>user\n{u}<end_of_turn>\n"
            f"<start_of_turn>model\n{m}<end_of_turn>"
        )
        messages = [
            {"role": "user", "content": u},
            {"role": "assistant", "content": m},
        ]
    return {
        "domain": domain,
        "system": sys_msg,
        "prompt": u,
        "response": m,
        "text": full_text,
        "messages": messages,
    }


def load_bbadha_raw_entries(json_dir: Path = BBADHA_JSON_DIR) -> List[Dict[str, Any]]:
    """Loads all 24 Bbadha JSON dictionary files (a.json .. z.json)."""
    entries: List[Dict[str, Any]] = []
    if not json_dir.exists():
        raise FileNotFoundError(f"Bbadha JSON directory not found: {json_dir}")

    for json_file in sorted(json_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("word_bbadha"):
                        item["_source_letter"] = json_file.stem.upper()
                        entries.append(item)
    return entries


def infer_category_key(sw_text: str, fr_text: str) -> Dict[str, str]:
    """Assigns a thematic category matching SautiLM Mobile's dictionary filter tabs."""
    combined = f"{sw_text} {fr_text}".lower()
    if any(
        w in combined
        for w in [
            "ng'ombe",
            "mbuzi",
            "kondoo",
            "mifugo",
            "maziwa",
            "vache",
            "chèvre",
            "bétail",
            "troupeau",
            "pâturage",
            "shamba",
            "champ",
            "houe",
            "cultiver",
            "jembe",
        ]
    ):
        return {
            "categoryKey": "pastoralism",
            "categoryEn": "Pastoralism",
            "categoryFr": "Pastoralisme",
            "categorySw": "Ufugaji",
        }
    if any(
        w in combined
        for w in [
            "ugonjwa",
            "dawa",
            "mwili",
            "damu",
            "chakula",
            "afya",
            "maladie",
            "remède",
            "médecine",
            "corps",
            "sang",
            "douleur",
            "soigner",
            "nourriture",
            "manger",
            "kula",
        ]
    ):
        return {
            "categoryKey": "health",
            "categoryEn": "Health",
            "categoryFr": "Santé",
            "categorySw": "Afya",
        }
    if any(
        w in combined
        for w in [
            "mvua",
            "jua",
            "upepo",
            "baridi",
            "mawingu",
            "pluie",
            "soleil",
            "vent",
            "froid",
            "nuage",
            "saison",
            "climat",
        ]
    ):
        return {
            "categoryKey": "climate_weather",
            "categoryEn": "Climate & Weather",
            "categoryFr": "Climat & Météo",
            "categorySw": "Hali ya Hewa",
        }
    if any(
        w in combined
        for w in [
            "mti",
            "maji",
            "mto",
            "msitu",
            "udongo",
            "jiwe",
            "arbre",
            "eau",
            "rivière",
            "forêt",
            "terre",
            "pierre",
            "plante",
            "herbe",
            "feu",
            "moto",
        ]
    ):
        return {
            "categoryKey": "nature_energy",
            "categoryEn": "Nature & Energy",
            "categoryFr": "Nature & Énergie",
            "categorySw": "Asili & Nishati",
        }
    return {
        "categoryKey": "education",
        "categoryEn": "Education",
        "categoryFr": "Éducation",
        "categorySw": "Elimu",
    }


def compile_mobile_lexicon(raw_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compiles the 1,150 Bbadha JSON entries into a fast, structured mobile lexicon."""
    compiled: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        word_bbadha = sanitize_sautilm_text(str(entry.get("word_bbadha", "")))
        phonetics = sanitize_sautilm_text(str(entry.get("phonetics") or ""))
        grammar = sanitize_sautilm_text(str(entry.get("grammar") or ""))
        meanings = entry.get("meanings") or []

        sw_list: List[str] = []
        fr_list: List[str] = []
        examples_list: List[Dict[str, str]] = []

        for m in meanings:
            if not isinstance(m, dict):
                continue
            sw = sanitize_sautilm_text(str(m.get("swahili_translation") or ""))
            fr = sanitize_sautilm_text(str(m.get("french_translation") or ""))
            if sw and sw not in sw_list:
                sw_list.append(sw)
            if fr and fr not in fr_list:
                fr_list.append(fr)

            sense_complete: List[Dict[str, str]] = []
            sense_partial: List[Dict[str, str]] = []
            for ex in m.get("examples") or []:
                if not isinstance(ex, dict):
                    continue
                bb_s = sanitize_sautilm_text(str(ex.get("bbadha_sentence") or ""))
                sw_s = sanitize_sautilm_text(str(ex.get("swahili_sentence") or ""))
                fr_s = sanitize_sautilm_text(str(ex.get("french_sentence") or ""))
                item_ex = {
                    "bbadha": bb_s,
                    "swahili": sw_s,
                    "french": fr_s,
                }
                if bb_s and sw_s and fr_s:
                    sense_complete.append(item_ex)
                elif bb_s and (sw_s or fr_s):
                    sense_partial.append(item_ex)
            examples_list.extend(sense_complete[:3] + sense_partial[:1])

        # Ensure complete trilingual examples come first across senses
        examples_list.sort(
            key=lambda x: 0 if (x["bbadha"] and x["swahili"] and x["french"]) else 1
        )

        primary_sw = " ; ".join(sw_list[:3]) if sw_list else word_bbadha
        primary_fr = " ; ".join(fr_list[:3]) if fr_list else word_bbadha
        cat = infer_category_key(primary_sw, primary_fr)

        meta_parts_fr: List[str] = []
        meta_parts_sw: List[str] = []
        meta_parts_en: List[str] = []
        if grammar:
            meta_parts_fr.append(f"Grammaire: {grammar}.")
            meta_parts_sw.append(f"Sarufi: {grammar}.")
            meta_parts_en.append(f"Grammar: {grammar}.")
        if phonetics:
            meta_parts_fr.append(f"Phonétique: [{phonetics}].")
            meta_parts_sw.append(f"Matamshi: [{phonetics}].")
            meta_parts_en.append(f"Phonetics: [{phonetics}].")
        if examples_list:
            ex0 = examples_list[0]
            meta_parts_fr.append(
                f"Exemple: \"{ex0['bbadha']}\" -> FR: \"{ex0['french']}\" | SW: \"{ex0['swahili']}\"."
            )
            meta_parts_sw.append(
                f"Mfano: \"{ex0['bbadha']}\" -> SW: \"{ex0['swahili']}\" | FR: \"{ex0['french']}\"."
            )
            meta_parts_en.append(
                f"Example: \"{ex0['bbadha']}\" -> SW: \"{ex0['swahili']}\" | FR: \"{ex0['french']}\"."
            )
        else:
            meta_parts_fr.append(f"Terme du corpus Bbadha - Swahili ({primary_fr}).")
            meta_parts_sw.append(f"Neno la msingi la Kibbadha - Kiswahili ({primary_sw}).")
            meta_parts_en.append(f"Bbadha - Swahili lexical corpus entry ({primary_fr}).")

        key_slug = f"{word_bbadha.lower()}_{idx}"
        compiled.append(
            {
                "key": key_slug,
                "bbadha": word_bbadha,
                "phonetics": phonetics,
                "grammar": grammar,
                "swahili": primary_sw,
                "french": primary_fr,
                "english": primary_fr,
                "categoryKey": cat["categoryKey"],
                "categoryEn": cat["categoryEn"],
                "categoryFr": cat["categoryFr"],
                "categorySw": cat["categorySw"],
                "noteEn": " ".join(meta_parts_en),
                "noteFr": " ".join(meta_parts_fr),
                "noteSw": " ".join(meta_parts_sw),
                "examples": examples_list[:6],
            }
        )
    return compiled


def _is_clean_sft_example(bb_s: str, sw_s: str, fr_s: str) -> bool:
    """Filters out noisy OCR fragments so SautiLM-Lex trains only on clean trilingual sentences."""
    if not (bb_s and sw_s and fr_s):
        return False
    combined = f"{bb_s} {sw_s} {fr_s}"
    if any(bad in combined for bad in ("£", "%", "/!", "expr", "£xpr", "Is coulaient")):
        return False
    if len(bb_s.strip()) < 3 or len(sw_s.strip()) < 3 or len(fr_s.strip()) < 3:
        return False
    return True


def generate_lex_instructions(raw_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates clean, RAG-aware trilingual instruction pairs from all Bbadha entries."""
    instructions: List[Dict[str, Any]] = []
    core_words = {
        "ddà", "'ábba", "'ádzá", "bba", "dyà", "rra", "chà", "zzà",
        "ddi", "dyi", "ke", "dja", "kpa", "ma", "ni", "kawa",
    }

    for idx, entry in enumerate(raw_entries):
        word = sanitize_sautilm_text(str(entry.get("word_bbadha") or ""))
        if not word:
            continue
        phonetics = sanitize_sautilm_text(str(entry.get("phonetics") or "Non spécifié"))
        grammar = sanitize_sautilm_text(str(entry.get("grammar") or "Terme lexical"))
        meanings = entry.get("meanings") or []
        if not meanings:
            continue

        sys_prompt = None if (idx % 2 == 0) else SYSTEM_PROMPT_SAUTILM

        sense_lines_fr: List[str] = []
        sense_lines_sw: List[str] = []
        first_fr = ""
        first_sw = ""
        clean_examples_for_card: List[tuple[str, str, str]] = []

        for s_idx, m in enumerate(meanings):
            if not isinstance(m, dict):
                continue
            sid = m.get("sense_id", s_idx + 1)
            sw_t = sanitize_sautilm_text(str(m.get("swahili_translation") or ""))
            fr_t = sanitize_sautilm_text(str(m.get("french_translation") or ""))
            if not sw_t and not fr_t:
                continue
            if not first_fr and fr_t:
                first_fr = fr_t
            if not first_sw and sw_t:
                first_sw = sw_t

            if s_idx < 3:
                sense_lines_fr.append(f"{sid}. **Français** : {fr_t or '-'} | **Kiswahili** : {sw_t or '-'}")
                sense_lines_sw.append(f"{sid}. **Kiswahili** : {sw_t or '-'} | **Kifaransa** : {fr_t or '-'}")

            ex_added = 0
            for ex in m.get("examples") or []:
                if not isinstance(ex, dict):
                    continue
                bb_s = sanitize_sautilm_text(str(ex.get("bbadha_sentence") or ""))
                sw_s = sanitize_sautilm_text(str(ex.get("swahili_sentence") or ""))
                fr_s = sanitize_sautilm_text(str(ex.get("french_sentence") or ""))
                if not _is_clean_sft_example(bb_s, sw_s, fr_s):
                    continue
                if s_idx < 3 and ex_added < 2:
                    sense_lines_fr.append(
                        f"   - *Bbadha* : \"{bb_s}\" -> FR: \"{fr_s}\" | SW: \"{sw_s}\""
                    )
                    sense_lines_sw.append(
                        f"   - *Kibbadha* : \"{bb_s}\" -> SW: \"{sw_s}\" | FR: \"{fr_s}\""
                    )
                    clean_examples_for_card.append((bb_s, sw_s, fr_s))
                    ex_added += 1

                instructions.append(
                    format_gemma3_turn(
                        f"Traduis cette phrase Bbadha en français et en kiswahili : \"{bb_s}\"",
                        f"Traduction SautiLM-Lex (**{word}**) :\n\n"
                        f"- **Bbadha (Kilendu)** : {bb_s}\n"
                        f"- **Français** : {fr_s}\n"
                        f"- **Kiswahili** : {sw_s}",
                        domain="lex",
                        system_prompt=None if (len(instructions) % 2 == 0) else SYSTEM_PROMPT_SAUTILM_SHORT,
                    )
                )
                instructions.append(
                    format_gemma3_turn(
                        f"Tafsiri sentensi hii kwa Kibbadha na Kifaransa : \"{sw_s}\"",
                        f"Tafsiri ya SautiLM-Lex (**{word}**) :\n\n"
                        f"- **Kibbadha (Kilendu)** : {bb_s}\n"
                        f"- **Kiswahili** : {sw_s}\n"
                        f"- **Kifaransa** : {fr_s}",
                        domain="lex",
                        system_prompt=None if (len(instructions) % 2 == 0) else SYSTEM_PROMPT_SAUTILM_SHORT,
                    )
                )

        if sense_lines_fr:
            card_fr = (
                f"### Fiche Lexicale SautiLM-Lex : **{word}**\n\n"
                f"- **Langue** : Bbadha (Kilendu - Ituri, Est de la RDC)\n"
                f"- **Phonétique** : {phonetics}\n"
                f"- **Grammaire** : {grammar}\n\n"
                f"**Sens et exemples :**\n" + "\n".join(sense_lines_fr)
            )
            instructions.append(
                format_gemma3_turn(
                    f"Que signifie le mot Bbadha \"{word}\" ?",
                    card_fr,
                    domain="lex",
                    system_prompt=sys_prompt,
                )
            )

            card_sw = (
                f"### Kamusi ya SautiLM-Lex : **{word}**\n\n"
                f"- **Lugha** : Kibbadha (Kilendu - Ituri, RDC)\n"
                f"- **Matamshi** : {phonetics}\n"
                f"- **Sarufi** : {grammar}\n\n"
                f"**Maana na mifano :**\n" + "\n".join(sense_lines_sw)
            )
            instructions.append(
                format_gemma3_turn(
                    f"Neno la Kibbadha \"{word}\" lina maana gani kwa Kiswahili na Kifaransa?",
                    card_sw,
                    domain="lex",
                    system_prompt=sys_prompt,
                )
            )

            # Ground model to strictly respect Flutter's DICTIONNAIRE BBADHA VERIFIE RAG context
            if word.lower() in core_words or idx < 80:
                rag_sys = (
                    "Tu es SautiLM-Lex, assistant lexicographique hors-ligne de SautiLM. "
                    "Appuie-toi strictement sur le contexte vérifié ci-dessous.\n"
                    f"DICTIONNAIRE BBADHA VERIFIE: Mot: {word} | Grammaire: {grammar} | "
                    f"Kiswahili: {first_sw or '-'} | Francais: {first_fr or '-'}."
                )
                rag_ex = ""
                if clean_examples_for_card:
                    b0, s0, f0 = clean_examples_for_card[0]
                    rag_sys += f" | Exemple Bbadha: \"{b0}\" -> SW: \"{s0}\" | FR: \"{f0}\""
                    rag_ex = f"\n- **Exemple vérifié** : *\"{b0}\"* (FR: {f0} | SW: {s0})"
                instructions.append(
                    format_gemma3_turn(
                        f"Que signifie le mot Bbadha {word} ?",
                        f"Selon le dictionnaire Bbadha vérifié (**SautiLM-Lex**), le mot **{word}** ({grammar}) signifie :\n\n"
                        f"- **Français** : {first_fr or '-'}\n"
                        f"- **Kiswahili** : {first_sw or '-'}{rag_ex}",
                        domain="lex",
                        system_prompt=rag_sys,
                    )
                )

        if first_fr and first_sw:
            instructions.append(
                format_gemma3_turn(
                    f"Comment dit-on \"{first_fr}\" ({first_sw}) en langue Bbadha (Kilendu) ?",
                    f"En **Bbadha (Kilendu)**, l'équivalent de **\"{first_fr}\"** (Kiswahili : **{first_sw}**) est :\n\n"
                    f"- **Mot Bbadha** : **{word}**\n"
                    f"- **Phonétique** : {phonetics}\n"
                    f"- **Grammaire** : {grammar}",
                    domain="lex",
                    system_prompt=None if (idx % 2 == 1) else SYSTEM_PROMPT_SAUTILM_SHORT,
                )
            )

    # Natural accent-free / direct lexical Q&As for high-frequency community words
    natural_lex_qas = [
        (
            "Que signifie le mot Bbadha Dda ?",
            "En langue **Bbadha (Kilendu)**, **ddà** (souvent écrit *Dda*) possède deux sens essentiels selon le contexte tonal :\n\n"
            "1. **L'eau / La rivière** (*Kiswahili* : **maji**, **mto** | *English* : **water**, **river**).\n"
            "2. **Long / Lointain** (*Kiswahili* : **kuwa refu**, **mbali** | *English* : **long**, **far**).",
        ),
        (
            "Explain the Bbadha word ddà",
            "In **Bbadha (Kilendu)** spoken in Ituri (DRC), **ddà** means:\n\n"
            "1. **Water / River** (*French* : **l'eau, la rivière** | *Kiswahili* : **maji, mto**).\n"
            "2. **Far / Long** (*French* : **loin, long** | *Kiswahili* : **mbali, refu**).",
        ),
        (
            "Tafsiri neno la Kibbadha: dda",
            "Katika lugha ya **Kibbadha (Kilendu - Ituri, RDC)**, neno **ddà** (*dda*) lina maana kuu mbili:\n\n"
            "1. **Maji / Mto** (*Kifaransa* : **l'eau, la rivière** | *Kiingereza* : **water, river**).\n"
            "2. **Mbali / Kuwa refu** (*Kifaransa* : **loin, long**).",
        ),
    ]
    for q, a in natural_lex_qas:
        for sys_p in (None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM):
            instructions.append(format_gemma3_turn(q, a, domain="lex", system_prompt=sys_p))

    return instructions


def generate_identity_instructions() -> List[Dict[str, Any]]:
    """Generates SautiLM Jua native identity instructions across FR/SW/EN and bare/system prompts."""
    pairs = [
        (
            "Qui es-tu ?",
            "Je suis **SautiLM Jua**, un modèle d'intelligence artificielle hors-ligne conçu à **Bunia (Ituri, RDC)** par **LosingTek Research**.\n\n"
            "**Équipe de recherche et fondation :**\n"
            "1. **Georges Byona** - Cofondateur, Software Engineer & Edge-AI Researcher\n"
            "2. **Abraham Imani** - Cofondateur & AI/ML Lead\n"
            "3. **Jacques Uwonda** - Cofondateur & Software Engineer\n\n"
            "Je fonctionne 100% localement sur smartphone sans connexion Internet pour le **Bbadha (Kilendu)**, le **Kiswahili**, le **Français** et l'**Anglais**.",
        ),
        (
            "Qui es-tu et qui t'a créé ?",
            "Je suis **SautiLM Jua**, le premier modèle fondation compact de **SautiLM** créé à **Bunia (Ituri, République Démocratique du Congo)** par **LosingTek Research**.\n\n"
            "**Équipe de recherche et de fondation :**\n"
            "1. **Georges Byona** - Cofondateur, Software Engineer & Edge-AI Researcher\n"
            "2. **Abraham Imani** - Cofondateur & AI/ML Lead (Université Catholique de Bukavu)\n"
            "3. **Jacques Uwonda** - Cofondateur & Software Engineer (Université Shalom de Bunia)\n\n"
            "**Ma mission :** Offrir une IA locale ultra-légère et souveraine pour les langues de l'Est de la RDC (**Bbadha / Kilendu**, **Kiswahili**), l'éducation (**SautiLM-Edu**), les mathématiques (**SautiLM-Math**), la santé communautaire (**SautiLM-Med**), la lexicographie (**SautiLM-Lex**) et le droit congolais (**SautiLM-Ius**).",
        ),
        (
            "Wewe ni nani na umeundwa na nani?",
            "Mimi ni **SautiLM Jua**, mfumo wa akili mnemba unaofanya kazi 100% bila mtandao (Offline Edge-AI) ulioundwa mjini **Bunia (Ituri, Jamhuri ya Kidemokrasia ya Kongo)** na **LosingTek Research**.\n\n"
            "**Waanzilishi na Watafiti Wakuu :**\n"
            "1. **Georges Byona** - Mwanzilishi Mwenza, Software Engineer & Edge-AI Researcher\n"
            "2. **Abraham Imani** - Mwanzilishi Mwenza & Kiongozi wa AI/ML (UCB Bukavu)\n"
            "3. **Jacques Uwonda** - Mwanzilishi Mwenza & Mhandisi wa Programu (USB Bunia)\n\n"
            "Ninafanya kazi moja kwa moja kwenye simu bila kutumia intaneti kusaidia jamii katika elimu (**SautiLM-Edu**), mahesabu (**SautiLM-Math**), afya (**SautiLM-Med**), tafsiri ya Kibbadha na Kiswahili (**SautiLM-Lex**), na sheria za Kongo (**SautiLM-Ius**).",
        ),
        (
            "Who are you? Are you Gemma or ChatGPT?",
            "I am **SautiLM Jua**, a sovereign offline Edge-AI model created by **LosingTek Research** in **Bunia, Ituri, Democratic Republic of the Congo**.\n\n"
            "**Founding Team :**\n"
            "1. **Georges Byona** - Co-founder, Software Engineer & Edge-AI Researcher\n"
            "2. **Abraham Imani** - Co-founder & AI/ML Lead\n"
            "3. **Jacques Uwonda** - Co-founder & Software Engineer\n\n"
            "I am specialized for low-resource African languages (starting with **Bbadha / Kilendu** and **Congolese Swahili**) to run directly on Android smartphones via LiteRT-LM with zero internet connection.",
        ),
        (
            "Quels sont les modèles de la famille SautiLM ?",
            "La feuille de route **SautiLM** développée par **LosingTek Research** repose sur une série de modèles nommés en **Kiswahili court** :\n\n"
            "1. **SautiLM Jua** : Premier modèle fondation compact texte hors-ligne (Bbadha, Kiswahili, Français, Anglais).\n"
            "2. **SautiLM Nuru** : Modèle multimodal vision (OCR manuscrit et diagnostic visuel).\n"
            "3. **SautiLM Wazo, Kuu, Mali, Moto** : Raisonnement avancé, modèle généraliste élargi, économie/microfinance, et voix bidirectionnelle.\n\n"
            "**Domaines spécialisés :** **Lex**, **Edu**, **Math**, **Med**, **Ius**, **Agri**, **Fin**, **Civ**, **Voice** et **Vision**.",
        ),
        (
            "Qui est Georges Byona ?",
            "**Georges Byona** est **Cofondateur, Software Engineer & Edge-AI Researcher** au sein de **LosingTek Research** (Bunia, Ituri, République Démocratique du Congo), et créateur de l'écosystème **SautiLM**.\n\n"
            "Il travaille aux côtés d'**Abraham Imani** (Cofondateur & AI/ML Lead) et de **Jacques Uwonda** (Cofondateur & Software Engineer) pour concevoir des modèles d'intelligence artificielle souverains et hors-ligne en **Bbadha (Kilendu)**, **Kiswahili**, **Français** et **Anglais** sur smartphone Android.",
        ),
    ]
    out: List[Dict[str, Any]] = []
    prompts_cycle = [None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM]
    for rep in range(12):
        sys_p = prompts_cycle[rep % len(prompts_cycle)]
        for q, a in pairs:
            out.append(format_gemma3_turn(q, a, domain="identity", system_prompt=sys_p))
    return out


def generate_math_instructions() -> List[Dict[str, Any]]:
    """Generates step-by-step SautiLM-Math training instructions in French and Swahili."""
    out: List[Dict[str, Any]] = []

    # 1. Parametric pastoral & school word problems
    herds = [(12, 8, 5), (25, 15, 7), (40, 18, 12), (16, 24, 9), (32, 14, 6), (50, 22, 15)]
    for cows, bought, sold in herds:
        total_after_buy = cows + bought
        remaining = total_after_buy - sold
        out.append(
            format_gemma3_turn(
                f"Mfugaji ana ng'ombe (Dyà) {cows}. Ananunua ng'ombe wengine {bought}, kisha anauza {sold} sokoni Bunia. Ana ng'ombe wangapi sasa?",
                f"### Hesabu ya Mifugo (SautiLM-Math)\n\n"
                f"**Msamiati wa Kibbadha :** Ng'ombe = **Dyà**\n\n"
                f"1. **Hatua ya 1 (Kujumlisha ng'ombe walionunuliwa) :**\n"
                f"   - {cows} + {bought} = **{total_after_buy} ng'ombe**\n"
                f"2. **Hatua ya 2 (Kutoa ng'ombe waliouzwa) :**\n"
                f"   - {total_after_buy} - {sold} = **{remaining} ng'ombe**\n\n"
                f"**Jibu la Mwisho :** Mfugaji amebaki na **ng'ombe (Dyà) {remaining}**.",
                domain="math",
            )
        )
        out.append(
            format_gemma3_turn(
                f"Un éleveur en Ituri possède {cows} vaches (Dyà). Il en achète {bought} de plus, puis en vend {sold} au marché. Combien lui en reste-t-il ?",
                f"### Résolution Étape par Étape (SautiLM-Math)\n\n"
                f"**Repère lexical Bbadha :** Vache / Bovin = **Dyà**\n\n"
                f"1. **Étape 1 (Addition après achat) :**\n"
                f"   - {cows} + {bought} = **{total_after_buy} vaches**\n"
                f"2. **Étape 2 (Soustraction après vente) :**\n"
                f"   - {total_after_buy} - {sold} = **{remaining} vaches**\n\n"
                f"**Résultat final :** Il reste exactement **{remaining} vaches (Dyà)** à l'éleveur.",
                domain="math",
            )
        )

    # 2. Milk production & market revenue problems
    milk_cases = [(6, 4, 1500), (8, 5, 2000), (10, 3, 1800), (15, 4, 2500), (12, 6, 1200)]
    for cows, liters_per_cow, price_fc in milk_cases:
        daily_liters = cows * liters_per_cow
        daily_rev = daily_liters * price_fc
        out.append(
            format_gemma3_turn(
                f"Si {cows} vaches donnent chacune {liters_per_cow} litres de lait par jour et qu'un litre coûte {price_fc} FC, quelle est la recette journalière ?",
                f"### Calcul de Production Laitière (SautiLM-Math)\n\n"
                f"1. **Étape 1 (Volume total de lait par jour) :**\n"
                f"   - {cows} vaches x {liters_per_cow} litres = **{daily_liters} litres**\n"
                f"2. **Étape 2 (Recette totale en Francs Congolais) :**\n"
                f"   - {daily_liters} litres x {price_fc} FC = **{daily_rev} FC**\n\n"
                f"**Résultat final :** La recette journalière est de **{daily_rev} FC**.",
                domain="math",
            )
        )
        out.append(
            format_gemma3_turn(
                f"Ikiwa ng'ombe {cows} wanatoa lita {liters_per_cow} za maziwa kila mmoja kwa siku, na lita moja inauzwa {price_fc} FC, mapato ya siku ni kiasi gani?",
                f"### Hesabu ya Mapato ya Maziwa (SautiLM-Math)\n\n"
                f"1. **Hatua ya 1 (Jumla ya lita za maziwa kwa siku) :**\n"
                f"   - Ng'ombe {cows} x lita {liters_per_cow} = **lita {daily_liters}**\n"
                f"2. **Hatua ya 2 (Mapato kwa Faranga za Kongo - FC) :**\n"
                f"   - Lita {daily_liters} x {price_fc} FC = **{daily_rev} FC**\n\n"
                f"**Jibu la Mwisho :** Mapato ya siku ni **{daily_rev} FC**.",
                domain="math",
            )
        )

    # 3. Direct arithmetic operations
    ops = [
        (125, 375, "+", 500),
        (840, 290, "-", 550),
        (24, 15, "*", 360),
        (144, 12, "/", 12),
        (450, 18, "*", 8100),
        (960, 16, "/", 60),
    ]
    prompts_cycle = [None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM]
    for a, b, op, res in ops:
        for rep in range(5):
            sys_p = prompts_cycle[rep % len(prompts_cycle)]
            out.append(
                format_gemma3_turn(
                    f"Calcule exactement : {a} {op} {b}",
                    f"### Calcul Arithmétique (SautiLM-Math)\n\n"
                    f"- **Opération** : `{a} {op} {b}`\n"
                    f"- **Résultat exact** : **{res}**",
                    domain="math",
                    system_prompt=sys_p,
                )
            )
    # Assign varied system prompts across existing math entries too
    mixed: List[Dict[str, Any]] = []
    for idx, row in enumerate(out):
        sys_p = prompts_cycle[idx % len(prompts_cycle)]
        mixed.append(
            format_gemma3_turn(row["prompt"], row["response"], domain="math", system_prompt=sys_p)
        )
    return mixed


def generate_edu_and_med_instructions() -> List[Dict[str, Any]]:
    """Generates SautiLM-Edu and SautiLM-Med instructions grounded in Ituri/Kivu context."""
    edu_pairs = [
        (
            "Eleza mzunguko wa maji (Dda) na mvua (Ddi) kwa wanafunzi wa shule ya msingi.",
            "### Mzunguko wa Maji na Mvua (SautiLM-Edu)\n\n"
            "**Msamiati wa Kibbadha :** Maji = **Dda** | Mvua = **Ddi** | Jua = **Dyi**\n\n"
            "1. **Uvukizaji (Evaporation) :** Jua (**Dyi**) linapowaka juu ya Ziwa Albert na mito, maji (**Dda**) hupata joto na kupaa hewani kama mvuke.\n"
            "2. **Mgandamizo (Condensation) :** Mvuke unapofika juu kwenye hewa baridi, huungana na kutengeneza mawingu.\n"
            "3. ** Kunyesha kwa Mvua (Precipitation) :** Mawingu yanapokuwa mazito, maji hudondoka tena ardhini kama mvua (**Ddi**) inayosaidia mimea na miti (**Ke**) kukua.",
        ),
        (
            "Explique le cycle de l'eau et le rôle des arbres (Ke) contre l'érosion en Ituri.",
            "### Cycle de l'Eau et Protection des Sols (SautiLM-Edu)\n\n"
            "**Vocabulaire Bbadha :** Eau = **Dda** | Pluie = **Ddi** | Soleil = **Dyi** | Arbre = **Ke**\n\n"
            "1. **Évaporation :** La chaleur du soleil (**Dyi**) transforme l'eau (**Dda**) des rivières et du lac Albert en vapeur.\n"
            "2. **Condensation :** En montant, la vapeur refroidit et forme les nuages.\n"
            "3. **Précipitations :** L'eau retombe sous forme de pluie (**Ddi**).\n"
            "4. **Rôle des arbres (Ke) :** Les racines des arbres retiennent la terre sur les collines d'Ituri et du Kivu, empêchant l'érosion et les glissements de terrain.",
        ),
        (
            "Comment enregistrer un acte de naissance en RDC et promouvoir la paix communautaire (Kawa) ?",
            "### Administration Civile & Médiation (SautiLM-Civ)\n\n"
            "**Vocabulaire Bbadha :** Paix = **Kawa** | Enfant = **Ngù** | Village = **Bba**\n\n"
            "1. **Déclaration de naissance (Code de la Famille Art. 116) :** Tout enfant né en RDC doit être déclaré dans les **90 jours** suivant la naissance devant l'Officier de l'État Civil (Commune ou Secteur) afin d'obtenir gratuitement l'acte de naissance.\n"
            "2. **Médiation locale (Kawa) :** Dans les conflits fonciers ou communautaires au village (**Bba**), le dialogue pacifique, l'écoute des témoins et la réconciliation précèdent le recours contentieux.",
        ),
        (
            "Comment gérer la caisse d'une AVEC (Association Villageoise d'Épargne et de Crédit) hors-ligne ?",
            "### Microfinance & Comptabilité AVEC (SautiLM-Fin)\n\n"
            "1. **Caisse à trois cadenas :** Toutes les cotisations hebdomadaires (parts d'épargne + caisse sociale d'urgence) sont enregistrées publiquement devant les membres.\n"
            "2. **Calcul des prêts et intérêts :** Si un membre emprunte **50 000 FC** à un taux de **10 %** par mois sur 2 mois, l'intérêt est de `50 000 x 10 % = 5 000 FC` par mois, soit **10 000 FC** d'intérêts et **60 000 FC** à rembourser.",
        ),
    ]

    med_pairs = [
        (
            "Hatua 3 za kusafisha maji ya kunywa (Dda) ili kuzuia ugonjwa wa kipindupindu (Rra) kambini ni zipi?",
            "### Mwongozo wa Afya : Kusafisha Maji na Kuzuia Kipindupindu (SautiLM-Med)\n\n"
            "**Msamiati wa Kibbadha :** Maji = **Dda** | Ugonjwa = **Rra** | Dawa = **Chà**\n\n"
            "1. **Kuchuja (Filtration) :** Chuja maji (**Dda**) kwa kitambaa safi ili kuondoa uchafu unaoonekana.\n"
            "2. **Kuchemsha au Kutia Klorini (Boiling / Chlorination) :** Chemsha maji hadi yatoe mapovu makubwa kwa dakika 1 hadi 3, au tumia vidonge vya klorini (**Chà**) kulingana na kipimo cha lita 20.\n"
            "3. **Kuhifadhi kwa Usafi (Safe Storage) :** Hifadhi maji kwenye chombo safi kilichofunikwa na osha mikono kwa sabuni kabla ya kula chakula (**Zzà**).",
        ),
        (
            "Quelles sont les 3 étapes essentielles pour purifier l'eau (Dda) et prévenir le choléra (Rra) ?",
            "### Protocole Santé Communautaire : Eau Propre & Prévention du Choléra (SautiLM-Med)\n\n"
            "**Vocabulaire Bbadha :** Eau = **Dda** | Maladie = **Rra** | Remède/Traitement = **Chà**\n\n"
            "1. **Étape 1 - Filtration :** Filtrer l'eau (**Dda**) à travers un linge propre et plié pour retenir les particules en suspension.\n"
            "2. **Étape 2 - Ébullition ou Chloration :** Faire bouillir l'eau à gros bouillons pendant au moins 1 à 3 minutes, ou utiliser une dose mesurée de chlore actif (**Chà**) en laissant agir 30 minutes.\n"
            "3. **Étape 3 - Conservation et Hygiène :** Conserver l'eau dans un bidon propre muni d'un couvercle et se laver systématiquement les mains au savon avant de toucher la nourriture (**Zzà**).",
        ),
    ]

    out: List[Dict[str, Any]] = []
    prompts_cycle = [None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM]
    for rep in range(15):
        sys_p = prompts_cycle[rep % len(prompts_cycle)]
        for q, a in edu_pairs:
            out.append(format_gemma3_turn(q, a, domain="edu", system_prompt=sys_p))
        for q, a in med_pairs:
            out.append(format_gemma3_turn(q, a, domain="med", system_prompt=sys_p))
    return out


def generate_congolese_law_instructions() -> List[Dict[str, Any]]:
    """Generates SautiLM-Ius corpus for Congolese Law (Constitution, Family Code, Penal Code, Labor Code, OHADA)."""
    law_pairs = [
        (
            "Que dit la Constitution de la RDC sur la présomption d'innocence et les droits de la défense (SautiLM-Ius) ?",
            "### Droit Constitutionnel Congolais : Droits de la Défense (SautiLM-Ius)\n\n"
            "**Référence légale :** **Articles 17 et 19 de la Constitution de la République Démocratique du Congo** (du 18 février 2006 telle que modifiée par la Loi n° 11/002 du 20 janvier 2011).\n\n"
            "1. **Liberté individuelle et légalité des poursuites (Art. 17) :**\n"
            "   - La liberté individuelle est garantie. Elle est la règle, la détention l'exception.\n"
            "   - Nul ne peut être poursuivi, arrêté ni détenu qu'en vertu de la loi et dans les formes qu'elle prescrit.\n"
            "   - Toute personne accusée d'une infraction est **présumée innocente** jusqu'à ce que sa culpabilité ait été établie par un jugement définitif.\n"
            "2. **Garanties procédurales et droit à l'avocat (Art. 18 & 19) :**\n"
            "   - Toute personne arrêtée doit être informée immédiatement des motifs de son arrestation et de toute accusation portée contre elle, dans une langue qu'elle comprend.\n"
            "   - La garde à vue ne peut excéder **48 heures**. À l'expiration de ce délai, la personne doit être relâchée ou mise à la disposition de l'autorité judiciaire compétente.\n"
            "   - Le droit à l'assistance d'un avocat ou d'un défenseur judiciaire est garanti dès l'enquête préliminaire.",
        ),
        (
            "Quelles sont les conditions de validité du mariage en droit congolais selon le Code de la Famille ?",
            "### Droit Civil Congolais : Conditions du Mariage (SautiLM-Ius)\n\n"
            "**Référence légale :** **Loi n° 87-010 du 1er août 1987 portant Code de la Famille**, telle que modifiée et complétée par la **Loi n° 16/008 du 15 juillet 2016** (Articles 330 à 356).\n\n"
            "1. **Âge légal de la majorité matrimoniale (Art. 352 révisé) :**\n"
            "   - L'homme et la femme avant **18 ans révolus** ne peuvent contracter mariage.\n"
            "2. **Consentement libre et personnel (Art. 351) :**\n"
            "   - Chacun des futurs époux doit consentir personnellement au mariage, librement et en pleine connaissance de cause.\n"
            "3. **La Dot (Art. 361 à 367) :**\n"
            "   - Le mariage coutumier et sa prise d'effet supposent le règlement ou la convention relative à la dot conformément aux coutumes des parties, sans qu'elle ne puisse dénaturer la dignité humaine.\n"
            "4. **Enregistrement devant l'Officier de l'État Civil (Art. 370) :**\n"
            "   - Pour produire sa pleine opposabilité légale envers les tiers, le mariage célébré en famille doit être enregistré par l'Officier de l'État Civil ou célébré directement devant lui.",
        ),
        (
            "Comment sont organisées les successions en RDC selon le Code de la Famille pour protéger le conjoint survivant et les enfants ?",
            "### Droit des Successions en RDC (SautiLM-Ius)\n\n"
            "**Référence légale :** **Code de la Famille de la RDC (Articles 758 et suivants, révisés par la Loi n° 16/008 du 15 juillet 2016)**.\n\n"
            "1. **Les trois catégories d'héritiers ab intestat (Art. 758) :**\n"
            "   - **1ère catégorie** : Les enfants du défunt nés dans le mariage et ceux nés hors mariage mais affiliés de son vivant, ainsi que les enfants adoptifs (**3/4 de l'hérédité**).\n"
            "   - **2ème catégorie** : Le conjoint survivant, les père et mère du défunt, et ses frères et sœurs germains, consanguins ou utérins.\n"
            "   - **3ème catégorie** : Les oncles et tantes paternels ou maternels.\n"
            "2. **Protection pénale et civile de la veuve et des orphelins :**\n"
            "   - Le conjoint survivant bénéficie d'un droit d'usufruit sur la maison habitée par les époux.\n"
            "   - Tout acte de spoliation ou d'expulsion illégale de la veuve ou des orphelins de la maison familiale est strictement puni par la loi congolaise.",
        ),
        (
            "Ni haki gani za mtu aliyekamatwa na polisi kulingana na Katiba ya RDC (SautiLM-Ius)?",
            "### Sheria za Kongo : Haki za Mtu Aliyekamatwa (SautiLM-Ius)\n\n"
            "**Misingi ya Kisheria :** **Ibara ya 17, 18 na 19 ya Katiba ya Jamhuri ya Kidemokrasia ya Kongo (RDC)**.\n\n"
            "1. **Kufahamishwa sababu kwa lugha anayoielewa (Ibara ya 18) :**\n"
            "   - Mtu yeyote anayekamatwa ana haki ya kujulishwa mara moja sababu za kukamatwa kwake kwa lugha anayoifahamu (kama Kiswahili au Kibbadha).\n"
            "2. **Muda wa kushikiliwa polisi (Garde à vue - Saa 48) :**\n"
            "   - Muda wa kushikiliwa kituoni hauwezi kuzidi **saa 48**. Baada ya muda huo, lazima aachiliwe au afikishwe mbele ya mwendesha mashtaka (Parquet).\n"
            "3. **Haki ya Wakili na kutojumuishwa kuwa mhalifu kabla ya hukumu :**\n"
            "   - Kila mshukiwa anachukuliwa kuwa hana hatia hadi mahakama itakapothibitisha kosa lake, na ana haki ya kusaidiwa na wakili katika hatua zote.",
        ),
        (
            "Que dit la Constitution de la RDC sur la garde à vue et les droits de la défense ?",
            "### Droit Constitutionnel Congolais : Garde à Vue & Droits de la Défense (SautiLM-Ius)\n\n"
            "**Référence légale :** **Articles 17, 18 et 19 de la Constitution de la RDC**.\n\n"
            "1. **Durée maximale de la garde à vue (Art. 18 al. 4) :** La garde à vue ne peut en aucun cas excéder **48 heures**. À l'expiration de ce délai de **48 heures**, la personne gardée à vue doit être immédiatement relâchée ou mise à la disposition de l'autorité judiciaire compétente (Parquet).\n"
            "2. **Information des motifs (Art. 18 al. 1-2) :** Toute personne arrêtée doit être informée immédiatement des motifs de son arrestation dans une langue qu'elle comprend.\n"
            "3. **Droit à l'avocat et présomption d'innocence (Art. 17 & 19) :** Le droit d'être assisté par un avocat ou défenseur judiciaire est garanti dès l'enquête préliminaire, et toute personne accusée est présumée innocente jusqu'à condamnation définitive.",
        ),
        (
            "Quels sont les types de sociétés commerciales reconnus en RDC selon l'Acte uniforme OHADA et comment protéger un salarié ?",
            "### Droit Commercial OHADA & Droit du Travail en RDC (SautiLM-Ius)\n\n"
            "1. **Sociétés OHADA en RDC :** L'Acte uniforme OHADA (AUSCGIE) reconnaît notamment la **SARL** (Société à Responsabilité Limitée), la **SA** (Société Anonyme), la **SAS** (Société par Actions Simplifiée), la **SNC**, la **SCS** et le **GIE**.\n"
            "2. **Code du Travail (Loi n° 015/2002 modifiée par la Loi n° 16/010) :** Tout licenciement exige un motif valable lié à l'aptitude, à la conduite ou aux nécessités de l'entreprise, avec respect du préavis légal et paiement du décompte final.",
        ),
    ]

    out: List[Dict[str, Any]] = []
    prompts_cycle = [None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM]
    for rep in range(15):
        sys_p = prompts_cycle[rep % len(prompts_cycle)]
        for q, a in law_pairs:
            out.append(format_gemma3_turn(q, a, domain="ius", system_prompt=sys_p))
    return out


def generate_vision_instructions() -> List[Dict[str, Any]]:
    """Generates OCR text solving, field symptom analysis, and honest vision-scope instructions."""
    vision_pairs = [
        (
            "Symptômes observés sur une feuille de manioc : taches jaunes et vertes en mosaïque et feuilles recroquevillées. Quel est le diagnostic agricole ?",
            "### Diagnostic Agronomique (SautiLM-Agri)\n\n"
            "1. **Diagnostic Principal :**\n"
            "   - **Culture** : Manioc (*Manihot esculenta*)\n"
            "   - **Maladie probable** : Mosaïque du manioc (Cassava Mosaic Disease - CMD)\n"
            "2. **Plan d'Action Hors-Ligne :**\n"
            "   - Arracher et détruire les plants sévèrement atteints dès le jeune stade.\n"
            "   - Utiliser des boutures saines certifiées résistantes lors de la prochaine saison.\n"
            "   - Contrôler la mouche blanche (*Bemisia tabaci*) avec un extrait de feuilles de Neem.",
        ),
        (
            "Texte relevé au tableau : 24 x 15 + 140 = ? Résous ce calcul étape par étape.",
            "### Lecture OCR & Résolution Mathématique (SautiLM-Math)\n\n"
            "**Expression relevée :** `24 x 15 + 140 = ?`\n\n"
            "1. **Étape 1 (Multiplication prioritaire) :** `24 x 15 = 360`\n"
            "2. **Étape 2 (Addition finale) :** `360 + 140 = 500`\n\n"
            "**Réponse finale :** **500**.",
        ),
        (
            "Dalili kwenye jani la mahindi: madoa madogo ya kahawia na kutu juu na chini ya jani. Mkulima afanyeje?",
            "### Uchambuzi wa Mazao (SautiLM-Agri)\n\n"
            "1. **Utambuzi Mkuu :** Kutu ya Majani ya Mahindi (Maize Common Rust - *Puccinia sorghi*).\n"
            "2. **Hatua za Kuchukua :**\n"
            "   - Panda mbegu zinazostahimili kutu na zingatia mzunguko wa mazao na maharagwe.\n"
            "   - Tumia mchanganyiko wa majivu na mwarobaini (Neem) mapema asubuhi kwenye majani machanga.",
        ),
        (
            "Peux-tu analyser une photo directement en mode hors-ligne ?",
            "**SautiLM Jua** est le premier modèle fondation compact axé sur le **texte hors-ligne** (lexique Bbadha, Kiswahili, français, anglais, mathématiques, droit et santé).\n\n"
            "- Si vous saisissez le **texte relevé (OCR)** ou la **description des symptômes observés**, je peux l'analyser immédiatement sans Internet.\n"
            "- L'analyse directe des pixels d'image est réservée à **SautiLM Nuru** (modèle multimodal vision de la série SautiLM).",
        ),
    ]

    out: List[Dict[str, Any]] = []
    prompts_cycle = [None, SYSTEM_PROMPT_SAUTILM_SHORT, SYSTEM_PROMPT_SAUTILM]
    for rep in range(12):
        sys_p = prompts_cycle[rep % len(prompts_cycle)]
        for q, a in vision_pairs:
            out.append(format_gemma3_turn(q, a, domain="vision", system_prompt=sys_p))
    return out


def build_all_corpora(output_dir: Path) -> Dict[str, Any]:
    """Builds the SautiLM Jua JSONL training dataset and the full Bbadha mobile lexicon."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_entries = load_bbadha_raw_entries(BBADHA_JSON_DIR)

    # 1. Compile mobile lexicon (all Bbadha entries)
    mobile_lexicon = compile_mobile_lexicon(raw_entries)
    lex_path = output_dir / "bbadha_full_lexicon.json"
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(mobile_lexicon, f, ensure_ascii=False, indent=2)

    # Also copy to Flutter/sautilm/assets/data/bbadha_full_lexicon.json
    FLUTTER_SAUTILM_ASSETS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lex_path, FLUTTER_SAUTILM_ASSETS_DATA_DIR / "bbadha_full_lexicon.json")

    # 2. Generate balanced multilingual (EN, FR, SW, Bbadha) instruction dataset across all 10 domains
    all_instructions: List[Dict[str, Any]] = []
    all_instructions.extend(generate_identity_instructions())
    all_instructions.extend(generate_lex_instructions(raw_entries))
    all_instructions.extend(generate_math_instructions())
    all_instructions.extend(generate_edu_and_med_instructions())
    all_instructions.extend(generate_congolese_law_instructions())
    all_instructions.extend(generate_vision_instructions())
    all_instructions.extend(generate_greetings_and_multiturn())
    all_instructions.extend(generate_multilingual_domains_v11())

    train_path = output_dir / "sautilm_v1_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for row in all_instructions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_counts: Dict[str, int] = {}
    for row in all_instructions:
        d = row["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    stats = {
        "model_series": "SautiLM Jua",
        "lexicon_count": len(mobile_lexicon),
        "total_instructions": len(all_instructions),
        "domain_counts": domain_counts,
        "train_jsonl_path": "data/sautilm_v1_train.jsonl",
        "mobile_lexicon_path": "data/bbadha_full_lexicon.json",
    }

    with open(output_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


if __name__ == "__main__":
    base_out = Path(__file__).resolve().parent.parent / "data"
    result = build_all_corpora(base_out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
