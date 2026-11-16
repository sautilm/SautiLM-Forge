from __future__ import annotations

from typing import Any, Dict, List


def _make_role_prompts(code: str, title_en: str, title_fr: str, title_sw: str) -> Dict[str, str]:
    return {
        "en": f"You are {code} (SautiLM-Jua), an offline {title_en} assistant by LosingTek Research. Answer accurately in English. For brief greetings (like Hey/Hello), reply warmly in 1-2 sentences.",
        "fr": f"Tu es {code} (SautiLM-Jua), assistant hors-ligne de {title_fr} créé par LosingTek Research. Réponds clairement en français. Pour une simple salutation (ex: Hey, Bonjour), réponds en 1 à 2 phrases.",
        "sw": f"Wewe ni {code} (SautiLM-Jua), msaidizi wa {title_sw} bila mtandao kutoka LosingTek Research. Jibu kwa Kiswahili fasaha. Kwa salamu fupi, jibu kwa sentensi 1 au 2.",
    }


def _turn(
    domain: str,
    system_prompt: str | None,
    user_prompt: str,
    model_response: str,
) -> Dict[str, Any]:
    u = user_prompt.replace("—", " - ").replace("–", " - ").strip()
    m = model_response.replace("—", " - ").replace("–", " - ").strip()
    s = (system_prompt or "").replace("—", " - ").replace("–", " - ").strip()
    if s:
        messages = [
            {"role": "system", "content": s},
            {"role": "user", "content": u},
            {"role": "assistant", "content": m},
        ]
        text = f"<start_of_turn>user\n{s}\n\n{u}<end_of_turn>\n<start_of_turn>model\n{m}<end_of_turn>"
    else:
        messages = [
            {"role": "user", "content": u},
            {"role": "assistant", "content": m},
        ]
        text = f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{m}<end_of_turn>"
    return {
        "domain": domain,
        "system": s,
        "prompt": u,
        "response": m,
        "text": text,
        "messages": messages,
    }


def _multi_turn(
    domain: str,
    system_prompt: str,
    exchanges: List[tuple[str, str]],
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    chunks: List[str] = []
    for idx, (u_raw, a_raw) in enumerate(exchanges):
        u = u_raw.replace("—", " - ").replace("–", " - ").strip()
        a = a_raw.replace("—", " - ").replace("–", " - ").strip()
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
        prefix = f"{system_prompt}\n\n" if idx == 0 else ""
        chunks.append(
            f"<start_of_turn>user\n{prefix}{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"
        )
    return {
        "domain": domain,
        "system": system_prompt,
        "prompt": exchanges[-1][0],
        "response": exchanges[-1][1],
        "text": "\n".join(chunks),
        "messages": messages,
    }


def generate_greetings_and_multiturn() -> List[Dict[str, Any]]:
    """Teaches concise domain welcomes for short greetings across all 10 modules and EN/FR/SW."""
    modules = [
        ("SautiLM-Lex", "lex", "Lexicon & Translation", "Lexique & Traduction", "Kamusi na Tafsiri",
         "Bbadha, Kiswahili, French, and English vocabulary and translation",
         "le vocabulaire et la traduction Bbadha, Kiswahili, français et anglais",
         "msamiati na tafsiri ya Kibbadha, Kiswahili, Kifaransa na Kiingereza"),
        ("SautiLM-Edu", "edu", "Education & Sciences", "Éducation & Sciences", "Elimu na Sayansi",
         "biology, physics, solar energy, and environmental science",
         "la biologie, la physique, l'énergie solaire et les sciences de la nature",
         "biolojia, fizikia, nishati ya jua na sayansi ya mazingira"),
        ("SautiLM-Med", "med", "Health & Medicine", "Santé & Médecine", "Afya na Tiba",
         "public health, first aid, ORS hydration, malaria prevention, and clean water",
         "la santé communautaire, les premiers secours, la SRO, le paludisme et l'eau potable",
         "afya ya jamii, huduma ya kwanza, ORS, kinga ya malaria na maji safi"),
        ("SautiLM-Math", "math", "Mathematics & Logic", "Mathématiques & Calcul", "Hisabati na Mantiki",
         "arithmetic, percentages, algebra, and step-by-step word problems",
         "l'arithmétique, les pourcentages, l'algèbre et les calculs étape par étape",
         "hesabu, asilimia, aljebra na utatuzi wa maswali hatua kwa hatua"),
        ("SautiLM-Vision", "vision", "Vision & STEM Quiz", "Vision & Quiz STEM", "Picha na Maswali ya STEM",
         "STEM quizzes, circuit/biology checks, and transcribed chalkboard problems",
         "les quiz STEM, la biologie, la physique et l'analyse de textes OCR",
         "maswali ya STEM, sayansi na uchambuzi wa maandishi ya ubao"),
        ("SautiLM-Voice", "voice", "Voice & Pocket Tutor", "Voix & Tuteur Général", "Sauti na Mwalimu Mkuu",
         "Bbadha tones, phonetics, study methods, and clear writing",
         "les tons Bbadha, la phonétique, les méthodes d'étude et la rédaction",
         "toni za Kibbadha, matamshi, mbinu za kusoma na uandishi bora"),
        ("SautiLM-Ius", "ius", "Congolese Law & Justice", "Droit Congolais & Justice", "Sheria za Kongo na Haki",
         "DRC Constitution rights, the 48-hour custody limit, Family Code, and OHADA",
         "la Constitution de la RDC, le Code de la Famille, le Code du Travail et le droit OHADA",
         "Katiba ya RDC, Sheria ya Familia, Sheria ya Kazi na OHADA"),
        ("SautiLM-Agri", "agri", "Agronomy & Environment", "Agronomie & Cultures", "Kilimo na Mazao",
         "cassava mosaic, maize rust, composting, crop rotation, and grain storage",
         "la mosaïque du manioc, le compostage, la rotation des cultures et le stockage des grains",
         "batobato ya muhogo, mboji, mzunguko wa mazao na uhifadhi wa nafaka"),
        ("SautiLM-Fin", "fin", "Finance & Accounting", "Finance & Comptabilité", "Fedha na Uhasibu",
         "savings groups (AVEC), cooperative shares, loan interest, and business margins",
         "les coopératives AVEC, le calcul d'intérêts, le fonds de roulement et les marges",
         "uhasibu wa biashara, vikundi vya akiba (AVEC), riba na mtaji"),
        ("SautiLM-Civ", "civ", "Civic Administration", "Administration & Citoyenneté", "Utawala na Uraia",
         "90-day birth registration, Supplemental Judgments, and peaceful community mediation",
         "la déclaration de naissance sous 90 jours, les actes d'état civil et la médiation",
         "usajili wa cheti cha kuzaliwa ndani ya siku 90, huduma za umma na usuluhishi"),
    ]

    en_greets = ["Hey", "Hello", "Hi", "Good morning", "Thanks", "Thank you"]
    fr_greets = ["Hey", "Bonjour", "Salut", "Merci", "Coucou"]
    sw_greets = ["Habari", "Jambo", "Mambo", "Salama", "Asante"]

    out: List[Dict[str, Any]] = []
    for code, dom, t_en, t_fr, t_sw, help_en, help_fr, help_sw in modules:
        roles = _make_role_prompts(code, t_en, t_fr, t_sw)

        for idx, g in enumerate(en_greets):
            if g in ("Thanks", "Thank you"):
                resp = f"You are welcome! Feel free to ask any {t_en.lower()} question anytime in **{code}** (100% offline)."
            else:
                resp = f"Hello! I am **{code}** (SautiLM-Jua offline). How can I help you today with {help_en}?"
            out.append(_turn(dom, roles["en"] if idx % 2 == 0 else None, g, resp))

        for idx, g in enumerate(fr_greets):
            if g == "Merci":
                resp = f"Avec plaisir ! N'hésitez pas à me poser d'autres questions sur {help_fr} avec **{code}**."
            else:
                resp = f"Bonjour ! Je suis **{code}** (SautiLM-Jua hors-ligne). Comment puis-je vous aider aujourd'hui concernant {help_fr} ?"
            out.append(_turn(dom, roles["fr"] if idx % 2 == 0 else None, g, resp))

        for idx, g in enumerate(sw_greets):
            if g == "Asante":
                resp = f"Karibu sana! Unaweza kuniuliza swali lolote kuhusu {help_sw} kupitia **{code}** bila mtandao."
            else:
                resp = f"Habari! Mimi ni **{code}** (SautiLM-Jua bila mtandao). Nikusaidieje leo kuhusu {help_sw}?"
            out.append(_turn(dom, roles["sw"] if idx % 2 == 0 else None, g, resp))

    # Explicit 2-turn follow-ups (matches Math Q1 -> "Hey" scenario directly)
    math_roles = _make_role_prompts("SautiLM-Math", "Mathematics & Logic", "Mathématiques & Calcul", "Hisabati na Mantiki")
    out.append(
        _multi_turn(
            "math",
            math_roles["en"],
            [
                ("Calculate step by step: (24 * 15) + 140.", "1. `24 * 15 = 360`\n2. `360 + 140 = 500`\n\n**Result:** **500**."),
                ("Hey", "Hello! What math problem, percentage, or calculation would you like to solve next?"),
            ],
        )
    )
    out.append(
        _multi_turn(
            "fin",
            _make_role_prompts("SautiLM-Fin", "Finance & Accounting", "Finance & Comptabilité", "Fedha na Uhasibu")["en"],
            [
                ("How are you?", "Hello! I am **{0}**, running 100% offline. Ask me about savings shares, loan interest, or accounting.".format("SautiLM-Fin")),
                ("Hey", "Hi! Ready to help with savings groups (AVEC), FC/USD budgets, or accounting calculations."),
            ],
        )
    )
    return out


def generate_multilingual_domains_v11() -> List[Dict[str, Any]]:
    """Provides high-precision EN, FR, and SW pairs across Finance, Civic, Agronomy, Law, Voice, Vision, Med, and Edu."""
    out: List[Dict[str, Any]] = []

    fin_roles = _make_role_prompts("SautiLM-Fin", "Finance & Accounting", "Finance & Comptabilité", "Fedha na Uhasibu")
    fin_pairs = [
        ("en", "In a savings group, 20 members buy 3 shares each at 2000 FC. Total capital?",
         "### Savings Group Capital (**SautiLM-Fin**)\n\n1. **Total Shares :** `20 members x 3 shares = 60 shares`\n2. **Total Capital :** `60 shares x 2000 FC = **120000 FC**`\n\n**Result :** The group holds **120000 FC** in total share capital."),
        ("en", "A business borrows 50000 FC at 10% interest. What is the total repayment?",
         "### Loan Repayment (**SautiLM-Fin**)\n\n1. **Interest Amount :** `10% of 50000 FC = 5000 FC`\n2. **Total Repayment :** `50000 FC + 5000 FC = **55000 FC**`"),
        ("en", "How to separate business working capital from personal family expenses?",
         "### Working Capital Protection (**SautiLM-Fin**)\n\n1. **Separate Cashboxes :** Keep business stock money strictly separate from household spending.\n2. **Fixed Owner Salary :** Pay yourself a modest, regular wage rather than taking cash at random.\n3. **Daily Ledger :** Record every sale, purchase, and restocking cost in FC or USD."),
        ("en", "How to calculate gross margin and net profit after transport and taxes?",
         "### Margin & Profit Formula (**SautiLM-Fin**)\n\n1. **Gross Margin :** `Sales Revenue - Cost of Goods Sold (COGS)`.\n2. **Net Profit :** `Gross Margin - (Transport + Market Taxes + Rent + Salaries)`.\n\nOnly **net profit** should be saved or withdrawn safely."),
        ("fr", "Dans une coopérative, 20 membres souscrivent 3 parts à 2000 FC. Capital total ?",
         "### Capital de Coopérative (**SautiLM-Fin**)\n\n1. **Nombre total de parts :** `20 membres x 3 parts = 60 parts`\n2. **Capital total :** `60 x 2000 FC = **120000 FC**`"),
        ("fr", "Un entrepreneur emprunte 50000 FC à 10% d'intérêt. Quel est le montant à rembourser ?",
         "### Remboursement de Prêt (**SautiLM-Fin**)\n\n1. **Intérêt :** `50000 FC x 10% = 5000 FC`\n2. **Total à rembourser :** `50000 + 5000 = **55000 FC**`"),
        ("sw", "Katika ushirika, wanachama 20 wananunua hisa 3 kwa 2000 FC kila moja. Mtaji ni ngapi?",
         "### Hesabu ya Mtaji (**SautiLM-Fin**)\n\n1. **Jumla ya Hisa :** `Wanachama 20 x hisa 3 = hisa 60`\n2. **Mtaji Kamili :** `60 x 2000 FC = **120000 FC**`"),
        ("sw", "Mjasiriamali anakopa 50000 FC kwa riba ya 10%. Atarudisha faranga ngapi?",
         "### Hesabu ya Mkopo (**SautiLM-Fin**)\n\n1. **Riba :** `50000 FC x 10% = 5000 FC`\n2. **Jumla ya Kurudisha :** `50000 + 5000 = **55000 FC**`"),
    ]
    for lang, q, a in fin_pairs:
        for rep in range(4):
            out.append(_turn("fin", fin_roles[lang] if rep % 2 == 0 else None, q, a))

    civ_roles = _make_role_prompts("SautiLM-Civ", "Civic Administration", "Administration & Citoyenneté", "Utawala na Uraia")
    civ_pairs = [
        ("en", "How to register a newborn's birth at the Civil Status Office within 90 days?",
         "### Birth Registration in DRC (**SautiLM-Civ**)\n\n1. **Legal Deadline :** Under **Article 116 of the DRC Family Code**, newborns must be declared within **90 days** of birth.\n2. **Required Documents :** Hospital/maternity birth attestation and both parents' identity cards.\n3. **Competent Office :** Visit the **Officier de l'État Civil** (Commune, Secteur, or Chefferie) — issuance within 90 days is **free of charge**."),
        ("en", "What documents are required to obtain a Supplemental Judgment for birth registration?",
         "### Supplemental Judgment for Birth (**SautiLM-Civ**)\n\nIf **90 days** have passed without registration in DRC:\n1. Apply for a **Jugement supplétif d'acte de naissance** at the local **Tribunal de Paix**.\n2. Bring witness statements (two adult witnesses), medical/baptismal birth proof, and parent IDs.\n3. Transcribe the court judgment at the Civil Status Office to receive an official birth certificate."),
        ("en", "3 key principles of peaceful mediation for neighborhood or land boundaries.",
         "### Peaceful Mediation Principles (**SautiLM-Civ**)\n\n1. **Neutral Listening :** Let each neighbor present facts and documents calmly before local notables or the Chef de Quartier.\n2. **Transparent Boundary Check :** Inspect landmarks and written land titles jointly on site.\n3. **Signed Conciliation Agreement :** Record mutual commitments in writing to prevent recurring disputes."),
        ("fr", "Comment déclarer une naissance à l'État Civil dans le délai légal de 90 jours ?",
         "### Déclaration de Naissance en RDC (**SautiLM-Civ**)\n\n1. **Délai légal :** Tout enfant né en RDC doit être déclaré dans les **90 jours** (Code de la Famille Art. 116).\n2. **Pièces :** Attestation de naissance de la maternité et pièces d'identité des parents.\n3. **Service compétent :** **Officier de l'État Civil** (Commune, Secteur ou Chefferie) — l'acte initial est gratuit."),
        ("sw", "Jinsi ya kusajili cheti cha kuzaliwa cha mtoto katika Ofisi ya Hali ya Raia ndani ya siku 90.",
         "### Usajili wa Cheti cha Kuzaliwa RDC (**SautiLM-Civ**)\n\n1. **Muda wa Kisheria :** Mtoto aliyepata uzazi nchini RDC lazima asajiliwe bure ndani ya **siku 90** (Sheria ya Familia Ibara 116).\n2. **Nyaraka :** Cheti cha hospitali na vitambulisho vya wazazi.\n3. **Ofisi Husika :** **Ofisi ya Hali ya Raia** (Manispaa, Sekta au Chefferie)."),
    ]
    for lang, q, a in civ_pairs:
        for rep in range(4):
            out.append(_turn("civ", civ_roles[lang] if rep % 2 == 0 else None, q, a))

    agri_roles = _make_role_prompts("SautiLM-Agri", "Agronomy & Environment", "Agronomie & Cultures", "Kilimo na Mazao")
    agri_pairs = [
        ("en", "How to identify and control Cassava Mosaic Disease (CMD)?",
         "### Cassava Mosaic Disease Control (**SautiLM-Agri**)\n\n1. **Diagnosis :** Look for yellow-green leaf mottling, stunted growth, and twisted leaves caused by a virus spread by **whiteflies** (*Bemisia tabaci*) and infected stem cuttings.\n2. **Immediate Action :** Rogue (uproot) severely infected young plants and bury/burn them away from the field.\n3. **Prevention :** Plant **certified CMD-resistant cuttings**, keep fields weed-free, and rotate cassava every season."),
        ("en", "How to produce nutrient-rich organic compost for vegetable and coffee farms?",
         "### Organic Composting (**SautiLM-Agri**)\n\n1. **Layering :** Alternate dry carbon materials (maize stalks, dry leaves) with fresh nitrogen greens (bean residues, animal manure) and a light layer of topsoil.\n2. **Moisture & Aeration :** Keep damp like a wrung sponge and turn every **2 to 3 weeks**.\n3. **Maturity :** Ready in **8 to 12 weeks** when dark, crumbly, and earthy-smelling."),
        ("en", "How to store harvested grains safely to prevent post-harvest losses?",
         "### Safe Grain Storage (**SautiLM-Agri**)\n\n1. **Proper Drying :** Sun-dry maize or beans on clean tarps until moisture is below **13%** (clean dry bite test).\n2. **Hermetic Storage :** Seal cleaned grains inside airtight **PICS triple-layer bags** or clean silos to suffocate weevils without chemicals.\n3. **Elevation :** Stack bags on wooden pallets off the floor and away from walls."),
        ("fr", "Comment identifier et contrôler la mosaïque du manioc ?",
         "### Contrôle de la Mosaïque du Manioc (**SautiLM-Agri**)\n\n1. **Symptômes :** Taches jaunes-vertes en mosaïque et déformation des feuilles transmises par la **mouche blanche** (*Bemisia tabaci*) ou des boutures malades.\n2. **Action :** Arracher précocement les jeunes plants infectés.\n3. **Prévention :** Utiliser exclusivement des **boutures saines résistantes certifiées** et pratiquer la rotation culturale."),
        ("sw", "Jinsi ya kutambua na kudhibiti ugonjwa wa batobato ya muhogo?",
         "### Kudhibiti Batobato ya Muhogo (**SautiLM-Agri**)\n\n1. **Dalili :** Madoa ya manjano na kijani kwenye majani yaliyosokotana yanayoenezwa na **inzi weupe** (*Bemisia tabaci*) na vipandikizi vyenye ugonjwa.\n2. **Hatua :** Ng'oa mapema mimea michanga iliyoathirika na uiweke mbali na shamba.\n3. **Kinga :** Tumia **vipandikizi bora vinavyostahimili batobato** na palilia shamba mapema."),
    ]
    for lang, q, a in agri_pairs:
        for rep in range(4):
            out.append(_turn("agri", agri_roles[lang] if rep % 2 == 0 else None, q, a))

    ius_roles = _make_role_prompts("SautiLM-Ius", "Congolese Law & Justice", "Droit Congolais & Justice", "Sheria za Kongo na Haki")
    ius_pairs = [
        ("en", "What does the DRC Constitution say about the 48-hour police custody limit?",
         "### DRC Constitutional Police Custody Limit (**SautiLM-Ius**)\n\n**Legal Basis :** **Articles 17, 18, and 19 of the DRC Constitution**.\n\n1. **48-Hour Limit (Art. 18(4)) :** Police custody (*garde à vue*) must NEVER exceed **48 hours**. After 48 hours, the detainee must be released or brought immediately before a judicial authority (*Parquet*).\n2. **Language Rights :** Detained individuals must be told the reasons for arrest immediately in a language they understand.\n3. **Right to Counsel & Innocence (Art. 17 & 19) :** Every suspect is **presumed innocent** and has the right to a lawyer from the preliminary investigation."),
        ("en", "What are the legal conditions for marriage under the Congolese Family Code?",
         "### Marriage Conditions in DRC Law (**SautiLM-Ius**)\n\n**Legal Basis :** Revised **DRC Family Code (Law n° 16/008 of 2016, Art. 330-370)**.\n\n1. **Minimum Age (Art. 352) :** Both spouses must be at least **18 years old**.\n2. **Free Consent (Art. 351) :** Personal, free consent from both partners is mandatory.\n3. **Dowry & Registration (Art. 361 & 370) :** Customary dowry agreement followed by registration or celebration before the **Officier de l'État Civil**."),
        ("en", "How does the Congolese Family Code organize inheritance (successions)?",
         "### Successions Under the DRC Family Code (**SautiLM-Ius**)\n\n**Legal Basis :** **Article 758 (Law 16/008)** defines three legal categories of heirs:\n1. **1st Category :** Children receive **3/4** of the estate.\n2. **2nd Category :** Surviving spouse, parents, and siblings.\n3. **3rd Category :** Aunts and uncles.\n\nThe surviving spouse holds a **housing usufruct right**, and unlawful eviction (*spoliation*) of widows or orphans is strictly punishable."),
        ("en", "What are the key legal forms of companies under OHADA commercial law in DRC?",
         "### OHADA Commercial Entities in DRC (**SautiLM-Ius**)\n\nUnder the **OHADA Uniform Act on Commercial Companies (AUSCGIE)** applicable in DR Congo, entrepreneurs can form:\n1. **SARL** (Limited Liability Company)\n2. **SA** (Public Limited Company)\n3. **SAS** (Simplified Joint-Stock Company)\n4. **SNC**, **SCS**, and **GIE** registered at the **RCCM** (*Guichet Unique*)."),
        ("fr", "Que dit l'Article 1 de la Constitution de la RDC ?",
         "### Article 1 de la Constitution de la RDC (**SautiLM-Ius**)\n\nSelon l'**Article 1 de la Constitution de la République Démocratique du Congo (2006)** :\n1. **État souverain :** La RDC est, dans ses frontières du **30 juin 1960**, un État de droit, indépendant, souverain, uni et indivisible, social, démocratique et laïc.\n2. **Devise et Emblèmes :** Sa devise est **Justice - Paix - Travail**. Son hymne national est **Debout Congolais**, sa monnaie le **Franc congolais (CDF)**, sa langue officielle le **français**, et ses quatre langues nationales le **kikongo**, le **lingala**, le **swahili** et le **tshiluba**."),
        ("en", "What does Article 1 of the DRC Constitution state?",
         "### Article 1 of the DRC Constitution (**SautiLM-Ius**)\n\nUnder **Article 1 of the 2006 DRC Constitution**, the Democratic Republic of the Congo is, within its borders of **June 30, 1960**, an independent, sovereign, united, indivisible, democratic, and secular State of law. Its motto is **Justice - Peace - Work**, its currency is the **Congolese Franc (CDF)**, its official language is **French**, and its four national languages are **Kikongo, Lingala, Kiswahili, and Tshiluba**."),
        ("sw", "Ibara ya 1 ya Katiba ya RDC inasema nini?",
         "### Ibara ya 1 ya Katiba ya RDC (**SautiLM-Ius**)\n\nKwa mujibu wa **Ibara ya 1 ya Katiba ya RDC (2006)**, Jamhuri ya Kidemokrasia ya Kongo ni nchi ya kisheria, huru, moja, na ya kidemokrasia katika mipaka yake ya tarehe **30 Juni 1960**. Wito wake ni **Haki - Amani - Kazi**, sarafu ni **Faranga ya Kongo (CDF)**, lugha rasmi ni **Kifaransa**, na lugha nne za kitaifa ni **Kiswahili, Lingala, Kikongo na Tshiluba**."),
    ]
    for lang, q, a in ius_pairs:
        for rep in range(5):
            out.append(_turn("ius", ius_roles[lang] if rep % 2 == 0 else None, q, a))

    voice_roles = _make_role_prompts("SautiLM-Voice", "Voice & Pocket Tutor", "Voix & Tuteur Général", "Sauti na Mwalimu Mkuu")
    voice_pairs = [
        ("en", "How do tonal accents work in the Bbadha (Kilendu) language of Ituri?",
         "### Tonal System in Bbadha (**SautiLM-Voice**)\n\n1. **Phonemic Tones :** Bbadha (Kilendu) uses **high (`á`), low (`à`), and rising/contour tones (`ă`)** to distinguish completely different words.\n2. **Glottal Stops :** An apostrophe like `'` in **'ábba** (village) or **'ádzá** (home) marks a crisp glottal closure before the vowel.\n3. **Example :** **ddà** (low tone) means water/river or far."),
        ("en", "What are the best memory techniques for mastering a new language quickly?",
         "### Effective Language Retention (**SautiLM-Voice**)\n\n1. **Active Recall :** Test yourself out loud before looking at translations.\n2. **Spaced Repetition :** Review tricky vocabulary after 1 day, 3 days, and 7 days.\n3. **Sentence Context :** Learn words inside full daily sentences (for example Bbadha + Kiswahili + English together)."),
        ("en", "Explain the linguistic relationship between Kingwana Swahili and Standard Swahili.",
         "### Kingwana vs Standard Swahili (**SautiLM-Voice**)\n\n1. **Shared Bantu Core :** **Kingwana** (Congolese Swahili spoken in Ituri, North and South Kivu) is mutually intelligible with **Kiswahili Sanifu** (Standard East African Swahili).\n2. **Regional Features :** Kingwana uses characteristic noun prefixes, French loanwords in administration, and local Eastern DRC expressions."),
        ("fr", "Comment fonctionnent les tons (haut, bas, montant) dans la phonétique du Bbadha ?",
         "### Phonétique Tonale du Bbadha (**SautiLM-Voice**)\n\n1. **Tons distinctifs :** Le Bbadha (Kilendu) utilise des **tons haut (`á`), bas (`à`) et montant (`ă`)** qui changent le sens d'un mot.\n2. **Coup de glotte :** L'apostrophe initiale dans **'ábba** (au village) ou **'ádzá** (à la maison) indique une occlusion glottale nette."),
    ]
    for lang, q, a in voice_pairs:
        for rep in range(4):
            out.append(_turn("voice", voice_roles[lang] if rep % 2 == 0 else None, q, a))

    vision_roles = _make_role_prompts("SautiLM-Vision", "Vision & STEM Quiz", "Vision & Quiz STEM", "Picha na Maswali ya STEM")
    edu_roles = _make_role_prompts("SautiLM-Edu", "Education & Sciences", "Éducation & Sciences", "Elimu na Sayansi")
    med_roles = _make_role_prompts("SautiLM-Med", "Health & Medicine", "Santé & Médecine", "Afya na Tiba")
    stem_med_pairs = [
        ("vision", vision_roles["en"],
         "Generate a STEM quiz question on how photosynthesis works in green leaves.",
         "### STEM Biology Quiz (**SautiLM-Vision**)\n\n**Question :** Which organelle and pigment in green leaves capture sunlight to convert water and CO2 into glucose and oxygen?\n- **A)** Mitochondria and hemoglobin\n- **B)** Chloroplasts and **chlorophyll**\n- **C)** Ribosomes and keratin\n\n**Conclusion :** Correct answer is **B**. Chlorophyll inside **chloroplasts** absorbs light energy to drive photosynthesis."),
        ("vision", vision_roles["en"],
         "Generate a quiz question on human heart circulation and oxygen transport.",
         "### STEM Anatomy Quiz (**SautiLM-Vision**)\n\n**Question :** Which chamber of the human heart pumps oxygen-rich blood through the aorta to the entire body?\n- **A)** Right atrium\n- **B)** **Left ventricle**\n- **C)** Pulmonary vein\n\n**Conclusion :** **B**. The muscular **left ventricle** generates systemic pressure to deliver oxygenated blood everywhere."),
        ("vision", vision_roles["en"],
         "Generate a physics quiz question on electric circuits, voltage, and resistance.",
         "### STEM Physics Quiz (**SautiLM-Vision**)\n\n**Question :** According to Ohm's Law (`V = I x R`), if a `12 V` solar battery powers a `4 Ohm` lamp, what is the electric current?\n- **A)** `48 A`\n- **B)** **`3 A`**\n- **C)** `0.33 A`\n\n**Conclusion :** **B** (`I = 12 V / 4 Ohm = 3 A`)."),
        ("edu", edu_roles["en"],
         "How does the water cycle (evaporation, condensation, precipitation) work?",
         "### The Water Cycle (**SautiLM-Edu**)\n\n1. **Evaporation :** Sunlight warms rivers and lakes, turning liquid water into rising water vapor.\n2. **Condensation :** Cooling vapor high in the air condenses around dust particles into clouds.\n3. **Precipitation :** Heavy droplets fall back as rain, replenishing soils, rivers, and groundwater."),
        ("edu", edu_roles["en"],
         "How do photovoltaic solar cells convert sunlight into electric current?",
         "### Solar PV Working Principle (**SautiLM-Edu**)\n\n1. **Photon Absorption :** Sunlight strikes doped **silicon** semiconductor layers.\n2. **Electron Flow :** Absorbed photons free electrons across the p-n junction, creating **direct current (DC)** voltage for batteries or LED lights."),
        ("med", med_roles["en"],
         "What are the 3 essential steps to treat dehydration with Oral Rehydration Salts (ORS)?",
         "### ORS Rehydration Protocol (**SautiLM-Med**)\n\n1. **Prepare Cleanly :** Wash hands and dissolve 1 full **ORS packet** completely in **1 liter** of clean boiled or treated water.\n2. **Frequent Small Sips :** Give small, regular sips using a cup or spoon (even after mild vomiting).\n3. **Clinical Referral :** Visit a health clinic urgently if the patient cannot drink, becomes lethargic, or has bloody stools."),
        ("med", med_roles["en"],
         "How to purify drinking water safely by boiling or chlorination?",
         "### Safe Water Purification (**SautiLM-Med**)\n\n1. **Filter First :** Pour water through a clean folded cloth to remove visible suspended dirt.\n2. **Boil or Chlorinate :** Bring water to a full rolling boil for **1 to 3 minutes**, OR add chlorine drops/tablets and wait **30 minutes**.\n3. **Safe Storage :** Store in a clean covered container."),
        ("med", med_roles["fr"],
         "Comment traiter la déshydratation aiguë avec les Sels de Réhydratation Orale (SRO) ?",
         "### Protocole SRO (**SautiLM-Med**)\n\n1. **Préparation :** Dissoudre 1 sachet complet de **SRO** dans **1 litre** d'eau bouillie refroidie ou chlorée.\n2. **Administration :** Donner de petites gorgées fréquentes à la cuillère ou à la tasse.\n3. **Alerte médicale :** Consulter immédiatement un centre de santé si l'enfant refuse de boire, vomit tout ou présente du sang dans les selles."),
        ("med", med_roles["fr"],
         "Quels sont les signes d'alerte du paludisme grave et que faire en urgence ?",
         "### Prise en Charge Précoce du Paludisme (**SautiLM-Med**)\n\n1. **Signes d'alerte :** Fièvre élevée persistante, convulsions, prostration, vomissements incoercibles ou pâleur sévère.\n2. **Conduite à tenir :** Référer d'urgence vers un centre de santé pour un test **TDR** et un traitement **ACT** ou artésunate injectable, tout en maintenant l'enfant hydraté et sous moustiquaire imprégnée (**MILDA**)."),
    ]
    for dom, role_p, q, a in stem_med_pairs:
        for rep in range(4):
            out.append(_turn(dom, role_p if rep % 2 == 0 else None, q, a))

    # Math EN word problems matching Flutter Math quick prompts
    math_roles = _make_role_prompts("SautiLM-Math", "Mathematics & Logic", "Mathématiques & Calcul", "Hisabati na Mantiki")
    math_en_pairs = [
        ("Calculate step by step: (24 * 15) + 140.",
         "### Step-by-Step Calculation (**SautiLM-Math**)\n\n1. **Multiply First :** `24 * 15 = 360`\n2. **Add Remainder :** `360 + 140 = **500**`\n\n**Exact Result :** **500**."),
        ("A merchant buys 15 items at 4500 FC each. What is the total cost?",
         "### Total Purchase Cost (**SautiLM-Math**)\n\n1. **Operation :** `15 * 4500 FC = **67500 FC**`\n2. **Verification :** `(10 * 4500) + (5 * 4500) = 45000 + 22500 = 67500 FC`."),
        ("What is 25% of 84000 FC, and what remains after the discount?",
         "### Percentage & Net Discount (**SautiLM-Math**)\n\n1. **Discount (25%) :** `0.25 * 84000 FC = **21000 FC**`\n2. **Remaining Net :** `84000 - 21000 = **63000 FC**`"),
        ("If 144 notebooks are shared equally among 12 classrooms, how many per room?",
         "### Equal Distribution (**SautiLM-Math**)\n\n1. **Division :** `144 / 12 = **12 notebooks** per classroom`.\n2. **Check :** `12 * 12 = 144`."),
    ]
    for q, a in math_en_pairs:
        for rep in range(5):
            out.append(_turn("math", math_roles["en"] if rep % 2 == 0 else None, q, a))

    return out
