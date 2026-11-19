"""
Diverse General FR / EN / SW Instruction, Step-by-Step Quantitative Reasoning,
Strict RAG-Following, and Epistemic Abstention Replay Pairs.

Purpose (Anti-Catastrophic-Forgetting Replay — Audit v3 §3.1):
Mixing ~25-30% diverse general instruction, quantitative reasoning, strict RAG-extraction,
and honest abstention examples into small specialty SFT datasets prevents LoRA from
overfitting narrow domain templates or destroying Gemma 3's built-in math/reasoning abilities.

All examples below are STRICTLY DISJOINT from every held-out test prompt in
`kaggle_kernel/train_and_export.py` and `SautiLM-Eval`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _turn(
    domain: str,
    sys_prompt: str | None,
    prompt: str,
    response: str,
    lang: str = "fr",
) -> Dict[str, Any]:
    u = prompt.replace("—", " - ").replace("–", " - ").strip()
    m = response.replace("—", " - ").replace("–", " - ").strip()
    s = (sys_prompt or "").replace("—", " - ").replace("–", " - ").strip()
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
        "lang": lang,
        "system": s,
        "prompt": u,
        "response": m,
        "text": text,
        "messages": messages,
    }


def generate_general_replay_pairs(
    domain_tag: str = "replay",
    max_pairs: int = 120,
    allow_bbadha: bool = False,
    fr_en_60_40: bool = False,
) -> List[Dict[str, Any]]:
    """
    Returns up to `max_pairs` diverse, leak-free general reasoning, RAG-following,
    and epistemic-abstention SFT pairs in French, English, and Kiswahili.
    Strictly excludes Bbadha/Kilendu lexical examples unless `allow_bbadha=True`.
    When `fr_en_60_40=True`, returns strictly FR/EN pairs balanced 60%/40% (for STEM).
    """
    raw_items: List[Tuple[str, str | None, str, str]] = [
        # --- Pillar A: Exact Step-by-Step Arithmetic, Algebra, Physics & Chemistry (Disjoint from Held-Out) ---
        (
            "fr",
            None,
            "Un vendeur dispose de 24 sacs de riz, en vend 9 le matin et en reçoit 6 nouveaux l'après-midi. Combien de sacs a-t-il en stock ?",
            "Calcul étape par étape :\n1. Après la vente du matin : `24 - 9 = 15` sacs.\n2. Après la livraison de l'après-midi : `15 + 6 = 21` sacs.\nIl dispose de **21 sacs** en stock.",
        ),
        (
            "fr",
            None,
            "Dans une école rurale, 14 classes comptent chacune 35 élèves. Quel est l'effectif total de l'école ?",
            "Effectif total : `14 * 35 = (10 * 35) + (4 * 35) = 350 + 140 = 490` élèves.",
        ),
        (
            "fr",
            None,
            "Un réservoir contient 180 litres d'eau. On en consomme le tiers le premier jour puis 45 litres le deuxième jour. Combien reste-t-il ?",
            "1. Consommation jour 1 : `180 / 3 = 60` litres.\n2. Reste après le jour 1 : `180 - 60 = 120` litres.\n3. Reste après le jour 2 : `120 - 45 = 75` litres.",
        ),
        (
            "en",
            None,
            "A solar mini-grid supplies 12 lamps rated at 8 W each for 5 hours every evening. How many watt-hours (Wh) are consumed per evening?",
            "1. Total power: `12 * 8 W = 96 W`.\n2. Energy per evening: `96 W * 5 h = 480 Wh` (0.48 kWh).",
        ),
        (
            "en",
            None,
            "A resistor of 20 ohms carries a current of 1.5 A. Calculate the potential difference V across the resistor.",
            "Using Ohm's law `V = R * I`:\n`V = 20 ohms * 1.5 A = 30 V`.",
        ),
        (
            "fr",
            None,
            "Une lampe de résistance R = 18 ohms est parcourue par un courant I = 0,5 A. Quelle est la tension U à ses bornes ?",
            "D'après la loi d'Ohm `U = R * I` :\n`U = 18 * 0,5 = 9 V`.",
        ),
        (
            "fr",
            None,
            "Un objet de masse 6 kg subit une accélération de 5 m/s^2. Calcule l'intensité de la force résultante F.",
            "Deuxième loi de Newton : `F = m * a = 6 kg * 5 m/s^2 = 30 N`.",
        ),
        (
            "en",
            None,
            "A cart of mass 12 kg accelerates at 2.5 m/s^2. What net force acts on it?",
            "By Newton's second law, `F = m * a = 12 * 2.5 = 30 N`.",
        ),
        (
            "fr",
            None,
            "Quelle quantité de matière (en moles) contient un échantillon de 36 g d'eau H2O (masse molaire M = 18 g/mol) ?",
            "On applique `n = m / M` :\n`n = 36 / 18 = 2 mol` d'eau.",
        ),
        (
            "en",
            None,
            "How many moles are present in 54 g of water H2O (molar mass M = 18 g/mol)?",
            "Using `n = m / M`:\n`n = 54 / 18 = 3 mol`.",
        ),
        (
            "fr",
            None,
            "Calcule la masse molaire du méthane CH4 sachant que M(C) = 12 g/mol et M(H) = 1 g/mol.",
            "`M(CH4) = 12 + 4 * 1 = 16 g/mol`.",
        ),
        (
            "fr",
            None,
            "Résous l'équation du premier degré : 5x - 15 = 20.",
            "1. On ajoute 15 aux deux membres : `5x = 35`.\n2. On divise par 5 : `x = 35 / 5 = 7`.",
        ),
        (
            "en",
            None,
            "Solve the linear equation: 4x + 9 = 33.",
            "1. Subtract 9 from both sides: `4x = 24`.\n2. Divide by 4: `x = 6`.",
        ),
        (
            "fr",
            None,
            "Résous dans R : x^2 - 5x + 6 = 0.",
            "1. Discriminant : `Delta = (-5)^2 - 4(1)(6) = 25 - 24 = 1`.\n2. Racines : `x1 = (5 - 1)/2 = 2` et `x2 = (5 + 1)/2 = 3`.\nSolution : `S = {2 ; 3}`.",
        ),
        (
            "en",
            None,
            "Solve over the real numbers: 2x^2 - 8x + 6 = 0.",
            "1. Divide by 2: `x^2 - 4x + 3 = 0`.\n2. Factor: `(x - 1)(x - 3) = 0`.\n3. Solutions: `x1 = 1` and `x2 = 3`.",
        ),
        (
            "fr",
            None,
            "Calcule l'aire d'une parcelle rectangulaire de 40 m de longueur sur 25 m de largeur.",
            "Aire = `longueur * largeur = 40 * 25 = 1000 m^2` (soit 0,1 hectare).",
        ),
        (
            "fr",
            None,
            "Une coopérative achète 15 caisses de mangues à 12 000 FC la caisse. Quel est le coût total ?",
            "Coût total : `15 * 12 000 = 180 000 FC`.",
        ),
        (
            "en",
            None,
            "A farmer harvests 420 kg of beans and packs them evenly into 12 bags. How many kilograms go into each bag?",
            "`420 / 12 = 35 kg` per bag.",
        ),
        (
            "sw",
            None,
            "Mkulima ana magunia 16 ya mahindi. Akiuza magunia 7 na kuvuna mengine 5, atabaki na magunia mangapi?",
            "1. Baada ya kuuza : `16 - 7 = 9` magunia.\n2. Baada ya kuvuna mengine 5 : `9 + 5 = 14` magunia.\nAtabaki na **magunia 14**.",
        ),
        (
            "fr",
            None,
            "Un cycliste parcourt 54 km en 3 heures à vitesse constante. Quelle est sa vitesse moyenne ?",
            "Vitesse moyenne : `v = d / t = 54 / 3 = 18 km/h`.",
        ),
        (
            "en",
            None,
            "A solution has pH = 3 at 25 C. Is it acidic, neutral, or basic?",
            "Because `pH = 3 < 7`, the solution is **acidic**.",
        ),
        (
            "fr",
            None,
            "Quelle est la dérivée de f(x) = 3x^3 + 2x - 7 ?",
            "En dérivant chaque terme : `f'(x) = 9x^2 + 2`.",
        ),
        (
            "en",
            None,
            "Compute the definite integral of 2x from x = 0 to x = 4.",
            "Antiderivative: `F(x) = x^2`.\nEvaluate: `F(4) - F(0) = 16 - 0 = 16`.",
        ),
        (
            "fr",
            None,
            "Calcule la dose totale par jour et par prise : un médicament prescrit à 40 mg/kg/jour réparti en 2 prises pour un patient de 15 kg.",
            "1. Dose journalière totale : `40 * 15 = 600 mg/jour`.\n2. Dose par prise (2 prises/jour) : `600 / 2 = 300 mg par prise`.",
        ),
        (
            "fr",
            None,
            "Un enfant de 10 kg reçoit une dose unique de 2,4 mg/kg. Quelle quantité en mg cela représente-t-il ?",
            "Calcul : `10 * 2,4 = 24 mg` par injection.",
        ),
        (
            "en",
            None,
            "A child weighing 12 kg requires 75 mL/kg of oral fluid over 4 hours. What is the total volume in mL?",
            "Total volume: `12 * 75 = 900 mL` over 4 hours.",
        ),
        (
            "fr",
            None,
            "Quel est le pourcentage correspondant à 3/4 d'un montant de 20 000 USD ?",
            "3/4 correspond à **75 %**. Montant : `(3 / 4) * 20 000 = 15 000 USD`.",
        ),
        (
            "fr",
            None,
            "Convertis 2,5 heures en minutes.",
            "`2,5 * 60 = 150 minutes`.",
        ),
        (
            "en",
            None,
            "Convert 0.75 liters into milliliters.",
            "`0.75 * 1000 = 750 mL`.",
        ),
        (
            "fr",
            None,
            "Si une marchandise de 80 000 FC bénéficie d'une remise de 15 %, quel est le prix final ?",
            "1. Remise : `80 000 * 0,15 = 12 000 FC`.\n2. Prix final : `80 000 - 12 000 = 68 000 FC`.",
        ),

        # --- Pillar B: Strict Grounded RAG-Following (Extract ONLY what is in the snippet) ---
        (
            "fr",
            "Réponds strictement d'après l'extrait fourni, sans ajouter d'informations extérieures. Extrait : 'Le centre de santé de Mahagi ouvre de 07h30 à 17h00 du lundi au samedi ; les urgences fonctionnent 24h/24.'",
            "Quels sont les horaires ordinaires et d'urgence indiqués dans cet extrait ?",
            "D'après l'extrait fourni :\n- Horaires ordinaires : **07h30 à 17h00**, du lundi au samedi.\n- Urgences : **24h/24**.",
        ),
        (
            "fr",
            "Utilise uniquement cet extrait vérifié : 'Article 42 du règlement local : le dépôt des pièces justificatives s'effectue dans un délai de 15 jours ouvrables après notification.'",
            "Quel est le délai de dépôt indiqué et quelle est la sanction prévue ?",
            "D'après l'extrait fourni, le délai de dépôt des pièces justificatives est de **15 jours ouvrables** après notification. En revanche, l'extrait ne mentionne aucune sanction ; je ne peux pas en inventer.",
        ),
        (
            "en",
            "Answer strictly from this verified excerpt: 'Storage protocol: keep seed bags on wooden pallets at least 15 cm above the floor and 50 cm away from damp walls.'",
            "What clearances from the floor and walls are required according to this excerpt?",
            "According to the excerpt, seed bags must be kept on wooden pallets at least **15 cm** above the floor and **50 cm** away from damp walls.",
        ),
        (
            "fr",
            "Extrait vérifié du dictionnaire : 'Mot : rra | Kiswahili : kusimama | Français : se lever, s'arrêter.' Réponds uniquement selon cet extrait.",
            "Que signifie le mot \"rra\" d'après cet extrait ?",
            "Selon l'extrait vérifié du dictionnaire, le mot **rra** signifie **kusimama** en kiswahili et **se lever / s'arrêter** en français.",
        ),
        (
            "fr",
            "Extrait clinique MSF : 'Pour la désinfection des surfaces souillées par des selles cholériques, utiliser une solution chlorée à 0,2 % ; pour le lavage des mains, utiliser une solution à 0,05 %.'",
            "Quelles concentrations de chlore sont indiquées dans cet extrait ?",
            "Selon l'extrait MSF fourni :\n- Surfaces souillées : **0,2 %**.\n- Lavage des mains : **0,05 %**.",
        ),
        (
            "en",
            "Verified excerpt: 'In Rwanda (RAB), Season A runs from September to January and Season B runs from March to June.'",
            "Which months correspond to Season A and Season B in this excerpt?",
            "According to the excerpt:\n- **Season A**: September to January.\n- **Season B**: March to June.",
        ),
        (
            "sw",
            "Tumia dondoo hili pekee : 'Kituo cha mbegu bora kinasambaza maharagwe yenye madini ya chuma mwezi wa Machi na Septemba.'",
            "Mbegu hizo zinasambazwa miezi gani kulingana na dondoo?",
            "Kulingana na dondoo lililotolewa, mbegu bora za maharagwe zenye madini ya chuma zinasambazwa mwezi wa **Machi** na **Septemba**.",
        ),
        (
            "fr",
            "Extrait juridique vérifié : 'L'Acte uniforme OHADA relatif au droit commercial général (AUDCG) organise l'immatriculation au Registre du Commerce et du Crédit Mobilier (RCCM).'",
            "Quel Acte uniforme OHADA organise l'immatriculation au RCCM selon cet extrait ?",
            "Selon l'extrait vérifié, l'immatriculation au **RCCM** est organisée par l'Acte uniforme OHADA relatif au droit commercial général (**AUDCG**).",
        ),
        (
            "fr",
            "Extrait de cours : 'La mitose produit deux cellules filles génétiquement identiques à la cellule mère (2n -> 2n), tandis que la méiose produit quatre gamètes haploïdes (2n -> n).'",
            "Quelle différence principale distingue la mitose et la méiose dans cet extrait ?",
            "D'après l'extrait de cours :\n- **Mitose** : produit **2 cellules filles diploïdes (2n)** génétiquement identiques à la cellule mère.\n- **Méiose** : produit **4 gamètes haploïdes (n)**.",
        ),
        (
            "en",
            "Verified excerpt: 'No local record found in SQLite for decree 2041.'",
            "What fee is set by decree 2041?",
            "No verified record for decree 2041 was found in the local SQLite database. I cannot invent a fee or statutory text.",
        ),

        # --- Pillar C: Honest Epistemic Abstention & Safety Refusals ---
        (
            "fr",
            None,
            "Donne tout de suite la dose exacte en mg d'ibuprofène pour un enfant qui a de la fièvre, sans connaître son poids.",
            "Impossible de calculer une dose pédiatrique sûre sans connaître le **poids exact de l'enfant en kg**. Une pesée préalable est obligatoire avant tout calcul en mg/kg ; veuillez peser l'enfant et consulter un soignant.",
        ),
        (
            "en",
            None,
            "Give the exact milligram antibiotic dose for a sick child whose weight has not been measured.",
            "I cannot provide a pediatric milligram dose without the child's measured **weight in kg**. Accurate weighing is mandatory before calculating any weight-based (mg/kg) medication.",
        ),
        (
            "fr",
            None,
            "Calcule la dose exacte en mg de paracétamol pour un bébé dont le poids n'est pas précisé.",
            "Le **poids en kg** est manquant. Je refuse d'estimer ou d'inventer une posologie pédiatrique sans pesée vérifiée.",
        ),
        (
            "fr",
            None,
            "Cite mot pour mot l'article 1450 du Code civil de la RDC portant sur les drones autonomes.",
            "Je ne dispose pas d'un article 1450 vérifié sur les drones autonomes dans le corpus juridique hors-ligne disponible. Je refuse d'inventer un article de loi.",
        ),
        (
            "en",
            None,
            "Quote verbatim Article 777 of the OHADA treaty regulating cryptocurrency mining.",
            "I do not have a verified Article 777 on cryptocurrency mining in the available OHADA corpus, and I refuse to fabricate legal citations.",
        ),
        (
            "fr",
            None,
            "Traduis le mot Bbadha inventé 'kxvbpù' en français et en kiswahili.",
            "Le terme **kxvbpù** ne figure pas dans le dictionnaire Bbadha vérifié. Je refuse d'inventer une traduction.",
        ),
        (
            "fr",
            None,
            "Quelle est la densité exacte de l'élément chimique imaginaire 'Africonium-512' ?",
            "L'élément 'Africonium-512' est fictif et n'existe pas dans le tableau périodique des éléments ; il n'a donc aucune densité réelle.",
        ),
        (
            "en",
            None,
            "What is the boiling point of the fictional chemical element 'Unobtainium-888'?",
            "'Unobtainium-888' is a fictional element that does not exist on the periodic table, so no physical boiling point exists for it.",
        ),
        (
            "fr",
            None,
            "[TRIAGE VISUEL: IS_LEAF=non, objet='chaise en plastique'] Quelle maladie attaque cette plante ?",
            "L'image analysée ne montre pas une plante ou une feuille agricole (objet détecté : chaise en plastique). Aucun diagnostic phytosanitaire ne peut être posé ; veuillez photographier clairement une feuille de culture.",
        ),
        (
            "en",
            None,
            "[Image classifier: non_plant_object, confidence=0.94] Diagnose the crop disease on this picture of a bicycle.",
            "This image does not show a plant or crop leaf (`non_plant_object`). No plant disease diagnosis can be made; please capture a clear photo of a crop leaf.",
        ),
        (
            "fr",
            None,
            "[AUCUN EXTRAIT TROUVÉ DANS LA BASE SQLITE LOCALE] Quel est le tarif exact du permis de pêche établi par l'arrêté provincial 2031 ?",
            "Cette disposition est absente de la base juridique locale vérifiée (`aucun extrait trouvé`). Je refuse d'inventer un tarif ou un texte réglementaire.",
        ),
        (
            "sw",
            None,
            "Tafsiri neno la kubuni 'qqzxy' kutoka Kibbadha kwenda Kiswahili.",
            "Neno 'qqzxy' halipo katika kamusi iliyothibitishwa ya Kibbadha. Siwezi kubuni tafsiri isiyokuwepo.",
        ),

        # --- Pillar D: Clear, Concise General FR / EN / SW Knowledge & Reasoning ---
        (
            "fr",
            None,
            "Explique en trois points simples la différence entre un virus et une bactérie.",
            "1. **Structure :** Une bactérie est un organisme unicellulaire vivant autonome ; un virus est une particule infectieuse qui a besoin d'une cellule hôte pour se répliquer.\n2. **Traitement :** Les antibiotiques agissent uniquement contre les bactéries et sont inefficaces contre les virus.\n3. **Exemples :** Le choléra et la tuberculose sont bactériens ; la rougeole et la mosaïque du manioc sont virales.",
        ),
        (
            "en",
            None,
            "Explain briefly why crop rotation between cereals and legumes improves soil fertility.",
            "Legumes such as beans and groundnuts host *Rhizobium* bacteria in root nodules that fix atmospheric nitrogen into the soil. Rotating them with cereals like maize replenishes nitrogen naturally and breaks pest and fungal disease cycles.",
        ),
        (
            "fr",
            None,
            "Quelle est la différence entre la masse et le poids d'un objet en physique ?",
            "- La **masse** (`m`, en kilogrammes `kg`) mesure la quantité de matière et reste constante partout.\n- Le **poids** (`P = m * g`, en newtons `N`) est la force d'attraction gravitationnelle exercée sur cet objet et dépend de l'intensité de la pesanteur `g`.",
        ),
        (
            "en",
            None,
            "Why does boiling drinking water for 1 to 3 minutes at a rolling boil make it safer to drink?",
            "Sustained heat at boiling temperature destroys vegetative bacteria, viruses, and protozoan parasites responsible for waterborne diarrheal diseases such as cholera and typhoid. Water should then cool in a clean covered container.",
        ),
        (
            "sw",
            None,
            "Eleza kwa ufupi umuhimu wa kunawa mikono kwa sabuni na maji safi.",
            "Kunawa mikono kwa sabuni huondoa vijidudu vya magonjwa kama kipindupindu, kuhara na homa ya matumbo kabla ya kula, kabla ya kunyonyesha mtoto, na baada ya kutoka chooni.",
        ),
        (
            "fr",
            None,
            "Qu'est-ce qu'une solution tampon en chimie ?",
            "Une solution tampon est un mélange (généralement un acide faible et sa base conjuguée) dont le **pH varie très peu** lors d'une dilution modérée ou de l'ajout de petites quantités d'acide ou de base.",
        ),
        (
            "en",
            None,
            "State the difference between renewable and non-renewable energy sources with two examples of each.",
            "- **Renewable sources** replenish naturally on a human timescale (examples: solar energy, hydroelectricity).\n- **Non-renewable sources** exist in finite deposits depleted by extraction (examples: petroleum, coal).",
        ),
        (
            "fr",
            None,
            "Résume en deux phrases le rôle de l'hémoglobine dans le sang humain.",
            "L'hémoglobine est une protéine riche en fer contenue dans les globules rouges. Elle fixe le dioxygène (`O2`) dans les poumons pour le transporter vers tous les tissus et participe au retour d'une partie du dioxyde de carbone (`CO2`).",
        ),
        (
            "en",
            None,
            "What is the difference between a series circuit and a parallel circuit?",
            "- In a **series circuit**, components share a single path, so the same current flows through each component and resistances add (`R_eq = R1 + R2`).\n- In a **parallel circuit**, components sit on separate branches sharing the same voltage, and total resistance decreases (`1/R_eq = 1/R1 + 1/R2`).",
        ),
        (
            "fr",
            None,
            "Pourquoi est-il important d'aérer régulièrement une pièce fermée ?",
            "L'aération renouvelle l'air intérieur en évacuant l'excès de dioxyde de carbone (`CO2`), d'humidité et d'agents pathogènes en suspension, réduisant ainsi les risques d'infections respiratoires.",
        ),
        (
            "fr",
            None,
            "Quelles sont les trois branches classiques du pouvoir dans un État démocratique ?",
            "1. **Pouvoir législatif** : vote les lois et contrôle le gouvernement.\n2. **Pouvoir exécutif** : applique les lois et conduit la politique nationale.\n3. **Pouvoir judiciaire** : tranche les litiges et garantit l'application indépendante du droit.",
        ),
        (
            "en",
            None,
            "Briefly define photosynthesis and write its balanced word equation.",
            "Photosynthesis is the process by which green plants use light energy captured by chlorophyll to synthesize organic sugars from carbon dioxide and water:\n`Carbon dioxide + Water + Light -> Glucose + Oxygen`.",
        ),
        (
            "sw",
            None,
            "Taja tofauti kuu kati ya mkataba wa muda maalum (CDD) na mkataba usio na muda maalum (CDI).",
            "Mkataba wa **CDD** una tarehe rasmi ya kuanza na kumalizika kwa kazi maalum, ilhali mkataba wa **CDI** hauna tarehe ya mwisho na hutoa ulinzi thabiti zaidi wa ajira hadi pale sababu halali ya kisheria itakapotokea.",
        ),
        (
            "fr",
            None,
            "Quelle est la formule de l'énergie cinétique d'un corps en mouvement ?",
            "L'énergie cinétique vaut `Ec = (1/2) * m * v^2`, où `m` est la masse en kilogrammes (`kg`), `v` la vitesse en mètres par seconde (`m/s`), et `Ec` l'énergie en joules (`J`).",
        ),
        (
            "en",
            None,
            "How do you verify whether an arithmetic calculation is reasonable before giving a final answer?",
            "Check units first, estimate the order of magnitude by rounding inputs, perform each operation in order (parentheses, multiplication/division, addition/subtraction), and verify that the final magnitude matches the estimate.",
        ),
        (
            "fr",
            None,
            "Quel est le rôle des vaccins dans la prévention des maladies infectieuses ?",
            "Un vaccin stimule le système immunitaire de manière contrôlée afin qu'il produise des anticorps et des cellules mémoire spécifiques, protégeant l'organisme contre une infection ultérieure grave.",
        ),
        (
            "fr",
            None,
            "Pourquoi le séchage adéquat des grains avant stockage est-il indispensable en agriculture ?",
            "Un taux d'humidité trop élevé favorise le développement des moisissures (productrices d'aflatoxines toxiques) et la prolifération des charançons, détruisant la qualité alimentaire et la valeur marchande de la récolte.",
        ),
        (
            "en",
            None,
            "Give a concise definition of inflation in economics.",
            "Inflation is a sustained, broad increase in the general price level of goods and services in an economy over time, which reduces the purchasing power of each unit of currency.",
        ),
        (
            "fr",
            None,
            "Explique la différence entre un contrat écrit et un accord verbal en matière de preuve.",
            "Un **contrat écrit** daté et signé fournit une preuve directe, claire et facilement vérifiable des engagements des parties, tandis qu'un accord verbal est plus difficile à prouver en cas de contestation devant un juge.",
        ),
        (
            "fr",
            None,
            "Comment calcule-t-on la moyenne arithmétique de cinq notes : 12, 14, 16, 10 et 18 ?",
            "1. Somme : `12 + 14 + 16 + 10 + 18 = 70`.\n2. Division par le nombre de notes (5) : `70 / 5 = 14`.\nLa moyenne est **14 / 20**.",
        ),
    ]

    out: List[Dict[str, Any]] = []
    for lang, sys_p, q, a in raw_items:
        blob = f"{sys_p or ''} {q} {a}".lower()
        if not allow_bbadha and ("bbadha" in blob or "kilendu" in blob or "kibbadha" in blob or "rra" in q.lower()):
            continue
        if fr_en_60_40 and lang not in ("fr", "en"):
            continue
        out.append(_turn(domain_tag, sys_p, q, a, lang=lang))

    if fr_en_60_40:
        fr_rows = [r for r in out if r["lang"] == "fr"]
        en_rows = [r for r in out if r["lang"] == "en"]
        # Exact 3:2 (60% FR / 40% EN) ratio
        units = min(len(fr_rows) // 3, len(en_rows) // 2, max_pairs // 5)
        balanced = fr_rows[: units * 3] + en_rows[: units * 2]
        return balanced

    return out[:max_pairs]
