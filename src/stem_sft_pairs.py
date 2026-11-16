from __future__ import annotations

from typing import Any, Dict, List


STEM_SYSTEM_FR = (
    "Tu es SautiLM-Jua-STEM, assistant hors-ligne de mathématiques, physique, chimie et biologie "
    "(niveau Secondaire et Examen d'État RDC) créé par LosingTek Research. "
    "Réponds directement, sans bavardage, en suivant un format structuré (QCM explicatif ou résolution étape par étape). "
    "Ne mentionne aucune langue étrangère non demandée ni de tiret long."
)

STEM_SYSTEM_EN = (
    "You are SautiLM-Jua-STEM, an offline secondary school Mathematics, Physics, Chemistry, and Biology "
    "assistant by LosingTek Research. Answer directly and accurately using clear formulas, explanatory MCQ rationale, "
    "or step-by-step revision steps. No emojis, no long dashes."
)


def _stem_turn(lang: str, sys_p: str | None, user_prompt: str, response: str) -> Dict[str, Any]:
    u = user_prompt.replace("—", " - ").replace("–", " - ").strip()
    m = response.replace("—", " - ").replace("–", " - ").strip()
    s = (sys_p or "").replace("—", " - ").replace("–", " - ").strip()
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
        "domain": "stem",
        "lang": lang,
        "system": s,
        "prompt": u,
        "response": m,
        "text": text,
        "messages": messages,
    }


def _french_stem_pairs() -> List[tuple[str, str]]:
    """24 currated French Secondary / Examen d'État pairs (Maths, Physics, Chemistry, Biology — QCM + Step-by-Step)."""
    return [
        # === MATHEMATIQUES (FR) ===
        (
            "Résous dans R l'équation du second degré : 2x^2 - 4x - 6 = 0 (Examen d'État).",
            "### Équation du 2nd Degré (SautiLM-Jua-STEM)\n\n"
            "1. **Formule du discriminant :** `Delta = b^2 - 4ac` avec `a = 2`, `b = -4`, `c = -6`.\n"
            "2. **Calcul :** `Delta = (-4)^2 - 4(2)(-6) = 16 + 48 = 64 = 8^2`.\n"
            "3. **Racines :** `x1 = (4 + 8) / (2 * 2) = 12 / 4 = 3` et `x2 = (4 - 8) / 4 = -4 / 4 = -1`.\n\n"
            "**Réponse finale :** `S = {-1 ; 3}`.",
        ),
        (
            "QCM Maths : Pour l'équation x^2 - 5x + 6 = 0, quelles sont les solutions ? A) {-2; -3}  B) {2; 3}  C) {1; 6}  D) {-1; 6}.",
            "### QCM Explicatif : Mathématiques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) {2 ; 3}**\n\n"
            "1. **Vérification par somme et produit :** `S = -b/a = 5` et `P = c/a = 6`. Les nombres dont la somme vaut `5` et le produit `6` sont `2` et `3`.\n"
            "2. **Pourquoi les autres choix sont faux :**\n"
            "   - `A)` donne une somme négative (`-5`).\n"
            "   - `C)` et `D)` ont un produit ou une somme incorrecte (`1 + 6 = 7`).",
        ),
        (
            "Calcule l'intégrale de sinus x divisé par cosinus x dx étape par étape.",
            "### Calcul Intégral (SautiLM-Jua-STEM)\n\n"
            "**Expression :** Intégrale de `sin(x) / cos(x) dx = tan(x) dx`.\n\n"
            "1. **Changement de variable :** Posons `u = cos(x)`, donc `du = -sin(x) dx`, soit `sin(x) dx = -du`.\n"
            "2. **Intégration :** L'intégrale devient `- int (1 / u) du = -ln|u| + C`.\n"
            "3. **Substitution inverse :** En remplaçant `u`, on obtient `-ln|cos(x)| + C = ln|sec(x)| + C`.\n\n"
            "**Réponse finale :** `-ln|cos(x)| + C` (ou `ln|sec(x)| + C`).",
        ),
        (
            "Calcule la dérivée de f(x) = 3x^3 - 5x^2 + 4x - 7.",
            "### Dérivée Polynomiale (SautiLM-Jua-STEM)\n\n"
            "1. **Règle de base :** `(x^n)' = n * x^(n-1)` et `(constante)' = 0`.\n"
            "2. **Dérivation terme à terme :** `f'(x) = 3(3x^2) - 5(2x) + 4(1) - 0`.\n\n"
            "**Réponse finale :** `f'(x) = 9x^2 - 10x + 4`.",
        ),
        (
            "QCM Maths : Que vaut sin^2(x) + cos^2(x) pour tout réel x ? A) 0  B) 1  C) 2  D) tan(x).",
            "### QCM Explicatif : Trigonométrie (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) 1**\n\n"
            "1. **Justification :** D'après le théorème de Pythagore appliqué au cercle trigonométrique de rayon `1`, les coordonnées `(cos x ; sin x)` vérifient toujours `cos^2(x) + sin^2(x) = 1`.\n"
            "2. **Élimination :** `A`, `C` et `D` ne sont pas des identités constantes sur tout `R`.",
        ),
        (
            "Un triangle rectangle a pour côtés de l'angle droit a = 6 cm et b = 8 cm. Calcule l'hypoténuse c.",
            "### Théorème de Pythagore (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `c^2 = a^2 + b^2`.\n"
            "2. **Application numérique :** `c^2 = 6^2 + 8^2 = 36 + 64 = 100`.\n"
            "3. **Racine carrée :** `c = racine(100) = 10 cm`.\n\n"
            "**Réponse finale :** L'hypoténuse mesure **10 cm**.",
        ),
        (
            "Une suite arithmétique a pour premier terme u1 = 5 et pour raison r = 3. Calcule le 10e terme u10.",
            "### Suite Arithmétique (SautiLM-Jua-STEM)\n\n"
            "1. **Formule générale :** `un = u1 + (n - 1) * r`.\n"
            "2. **Calcul pour n = 10 :** `u10 = 5 + (10 - 1) * 3 = 5 + 9 * 3 = 5 + 27 = 32`.\n\n"
            "**Réponse finale :** `u10 = 32`.",
        ),
        (
            "Résous le système d'équations : x + y = 10 et x - y = 4.",
            "### Système Linéaire (SautiLM-Jua-STEM)\n\n"
            "1. **Méthode d'addition :** Additionnons les deux équations : `(x + y) + (x - y) = 10 + 4`, soit `2x = 14`, donc `x = 7`.\n"
            "2. **Calcul de y :** `7 + y = 10 => y = 10 - 7 = 3`.\n\n"
            "**Réponse finale :** Le couple solution est `(x ; y) = (7 ; 3)`.",
        ),
        # === PHYSIQUE (FR) ===
        (
            "Une lampe de résistance R = 4 Ohms est alimentée sous une tension U = 12 V. Calcule l'intensité I et la puissance P.",
            "### Loi d'Ohm et Puissance Électrique (SautiLM-Jua-STEM)\n\n"
            "1. **Intensité (Loi d'Ohm `U = R * I`) :** `I = U / R = 12 V / 4 Ohms = 3 A`.\n"
            "2. **Puissance (`P = U * I`) :** `P = 12 V * 3 A = 36 W`.\n\n"
            "**Réponse finale :** `I = 3 A` et `P = 36 W`.",
        ),
        (
            "QCM Physique : Selon la 2e loi de Newton, quelle force F faut-il appliquer à une masse de 5 kg pour lui communiquer une accélération de 2 m/s^2 ? A) 2,5 N  B) 7 N  C) 10 N  D) 25 N.",
            "### QCM Explicatif : Dynamique (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) 10 N**\n\n"
            "1. **Relation fondamentale de la dynamique :** `F = m * a`.\n"
            "2. **Calcul :** `F = 5 kg * 2 m/s^2 = 10 N`.\n"
            "3. **Erreurs classiques :** `A` divise `m / a`, `B` additionne `m + a`.",
        ),
        (
            "Calcule l'énergie cinétique d'un solide de masse m = 4 kg se déplaçant à la vitesse v = 5 m/s.",
            "### Énergie Cinétique (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `Ec = (1/2) * m * v^2`.\n"
            "2. **Application numérique :** `Ec = 0,5 * 4 * (5^2) = 2 * 25 = 50 Joules`.\n\n"
            "**Réponse finale :** `Ec = 50 J`.",
        ),
        (
            "QCM Physique : Deux résistances R1 = 6 Ohms et R2 = 3 Ohms sont montées en parallèle. Quelle est la résistance équivalente ? A) 9 Ohms  B) 4,5 Ohms  C) 2 Ohms  D) 18 Ohms.",
            "### QCM Explicatif : Circuits Électriques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) 2 Ohms**\n\n"
            "1. **Formule en parallèle :** `1/Req = 1/R1 + 1/R2 = 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2`.\n"
            "2. **Inversion :** `Req = 2 Ohms`.\n"
            "3. **Piège évité :** `A) 9 Ohms` correspond au montage en série (`R1 + R2`), pas en parallèle.",
        ),
        (
            "Un objet de masse m = 10 kg tombe d'une hauteur h = 5 m (g = 10 m/s^2). Calcule son énergie potentielle de pesanteur.",
            "### Énergie Potentielle de Pesanteur (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `Ep = m * g * h`.\n"
            "2. **Substitution :** `Ep = 10 kg * 10 m/s^2 * 5 m = 500 Joules`.\n\n"
            "**Réponse finale :** `Ep = 500 J`.",
        ),
        (
            "QCM Physique : Que mesure un ampèremètre et comment se branche-t-il ? A) La tension, en parallèle  B) L'intensité du courant, en série  C) La résistance, hors circuit  D) La puissance, en parallèle.",
            "### QCM Explicatif : Mesures Électriques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) L'intensité du courant, en série**\n\n"
            "1. **Justification :** L'ampèremètre mesure le débit de charges (en Ampères `A`) et doit être traversé par le courant total ; il se branche donc **en série**.\n"
            "2. **Rappel :** Le voltmètre mesure la tension et se branche en parallèle.",
        ),
        # === CHIMIE (FR) ===
        (
            "Calcule le nombre de moles n contenu dans m = 18 g d'eau H2O (M(H)=1 g/mol, M(O)=16 g/mol).",
            "### Calcul de Quantité de Matière (SautiLM-Jua-STEM)\n\n"
            "1. **Masse molaire de H2O :** `M(H2O) = 2 * 1 + 16 = 18 g/mol`.\n"
            "2. **Formule :** `n = m / M`.\n"
            "3. **Calcul :** `n = 18 g / 18 g/mol = 1 mol`.\n\n"
            "**Réponse finale :** Il y a exactement **1 mole** d'eau.",
        ),
        (
            "QCM Chimie : Une solution aqueuse a un pH égal à 3 à 25°C. Quelle est sa nature ? A) Basique  B) Neutre  C) Acide  D) Saturée.",
            "### QCM Explicatif : Échelle de pH (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) Acide**\n\n"
            "1. **Principe :** Sur l'échelle de `0` à `14` (à 25°C), une solution est **acide** si `pH < 7`, **neutre** si `pH = 7`, et **basique** si `pH > 7`.\n"
            "2. **Conclusion :** Comme `3 < 7`, la solution est acide (excès d'ions `H3O+`).",
        ),
        (
            "Équilibre l'équation chimique de combustion complète du méthane : CH4 + O2 -> CO2 + H2O.",
            "### Équilibration d'Équation Chimique (SautiLM-Jua-STEM)\n\n"
            "1. **Conservation du Carbone (C) :** 1 C à gauche -> `1 CO2` à droite.\n"
            "2. **Conservation de l'Hydrogène (H) :** 4 H dans `CH4` -> `2 H2O` à droite (`2 * 2 = 4 H`).\n"
            "3. **Conservation de l'Oxygène (O) :** `1 CO2` (2 O) + `2 H2O` (2 O) = 4 O à droite -> il faut `2 O2` à gauche.\n\n"
            "**Équation équilibrée :** `CH4 + 2 O2 -> CO2 + 2 H2O`.",
        ),
        (
            "On dissout n = 0,5 mol de NaCl dans V = 250 mL d'eau. Calcule la concentration molaire C.",
            "### Concentration Molaire (SautiLM-Jua-STEM)\n\n"
            "1. **Conversion du volume en litres :** `V = 250 mL = 0,25 L`.\n"
            "2. **Formule :** `C = n / V`.\n"
            "3. **Calcul :** `C = 0,5 mol / 0,25 L = 2 mol/L`.\n\n"
            "**Réponse finale :** La concentration molaire est de **2 mol/L**.",
        ),
        (
            "QCM Chimie : Qu'est-ce qu'une réaction d'oxydation ? A) Un gain d'électrons  B) Une perte d'électrons  C) Une neutralisation  D) Un changement d'état.",
            "### QCM Explicatif : Oxydoréduction (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Une perte d'électrons**\n\n"
            "1. **Définition :** L'**oxydation** correspond à une **perte d'électrons** par un réducteur (`Fe -> Fe^2+ + 2e-`), tandis que la **réduction** est un **gain d'électrons**.",
        ),
        # === BIOLOGIE / SVT (FR) ===
        (
            "QCM Biologie : Quel organite cellulaire contient la chlorophylle et réalise la photosynthèse ? A) La mitochondrie  B) Le noyau  C) Le chloroplaste  D) Le ribosome.",
            "### QCM Explicatif : Biologie Cellulaire (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) Le chloroplaste**\n\n"
            "1. **Justification :** Le **chloroplaste** renferme la **chlorophylle**, pigment qui capte l'énergie lumineuse pour transformer `CO2 + H2O` en glucose et dioxygène (`O2`).\n"
            "2. **Distinction clé :** La mitochondrie (`A`) assure la respiration cellulaire (production d'ATP), le noyau (`B`) porte l'ADN, et le ribosome (`D`) synthétise les protéines.",
        ),
        (
            "Quelle est la différence fondamentale entre la mitose et la méiose en biologie ?",
            "### Division Cellulaire : Mitose vs Méiose (SautiLM-Jua-STEM)\n\n"
            "1. **La Mitose (division conforme) :** Une cellule mère diploïde (`2n`) donne **2 cellules filles identiques diploïdes (`2n`)**. Elle assure la croissance et le renouvellement des tissus.\n"
            "2. **La Méiose (division réductionnelle) :** Une cellule mère (`2n`) donne **4 gamètes haploïdes (`n`)**. Elle est indispensable à la reproduction sexuée et à la diversité génétique.",
        ),
        (
            "QCM Biologie : Quelle cavité du cœur humain propulse le sang oxygéné dans l'aorte vers tout le corps ? A) Oreillette droite  B) Ventricule droit  C) Oreillette gauche  D) Ventricule gauche.",
            "### QCM Explicatif : Physiologie Cardiaque (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : D) Ventricule gauche**\n\n"
            "1. **Trajet du sang :** Le sang oxygéné venant des poumons arrive dans l'oreillette gauche (`C`), passe dans le **ventricule gauche (`D`)** dont la paroi musculaire épaisse éjecte le sang dans l'**aorte** pour irriguer tout l'organisme.",
        ),
        (
            "En génétique mendélienne, on croise deux plantes hétérozygotes Rr x Rr (R dominant sur r). Quelles sont les proportions phénotypiques en F2 ?",
            "### Génétique Mendélienne : Croisement Monohybride (SautiLM-Jua-STEM)\n\n"
            "1. **Gamètes produits par Rr :** `50% R` et `50% r`.\n"
            "2. **Échiquier de croisement :** `RR` (25%), `Rr` (25%), `Rr` (25%), `rr` (25%).\n"
            "3. **Phénotypes obtenus :** `3/4` (75%) présentent le caractère dominant `[R]` et `1/4` (25%) présentent le caractère récessif `[r]`.\n\n"
            "**Réponse finale :** Proportions phénotypiques **3:1** (75% dominant / 25% récessif).",
        ),
        (
            "QCM Biologie : Quel est le rôle principal de l'ARNm (ARN messager) lors de la synthèse des protéines ? A) Stocker l'ATP  B) Transporter le code génétique du noyau vers les ribosomes  C) Détruire les virus  D) Filtrer le sang.",
            "### QCM Explicatif : Expression Génétique (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Transporter le code génétique du noyau vers les ribosomes**\n\n"
            "1. **Mécanisme :** L'ADN reste protégé dans le noyau. Lors de la **transcription**, l'ARNm copie un gène puis quitte le noyau pour être lu par les **ribosomes** dans le cytoplasme (**traduction**).",
        ),
    ]


def _english_stem_pairs() -> List[tuple[str, str]]:
    """16 curated English Secondary STEM pairs (Maths, Physics, Chemistry, Biology — QCM + Step-by-Step) for 60/40 FR/EN balance."""
    return [
        # === MATHEMATICS (EN) ===
        (
            "Solve step by step the quadratic equation: x^2 - 7x + 12 = 0.",
            "### Quadratic Equation Solution (SautiLM-Jua-STEM)\n\n"
            "1. **Discriminant (`Delta = b^2 - 4ac`):** With `a = 1`, `b = -7`, `c = 12`: `Delta = (-7)^2 - 4(1)(12) = 49 - 48 = 1`.\n"
            "2. **Roots (`x = (-b +- sqrt(Delta)) / 2a`):** `x1 = (7 + 1) / 2 = 4` and `x2 = (7 - 1) / 2 = 3`.\n\n"
            "**Final Answer:** `x = 3` or `x = 4` (`(x - 3)(x - 4) = 0`).",
        ),
        (
            "STEM Math MCQ: What is the derivative of f(x) = sin(x)? A) -cos(x)  B) cos(x)  C) tan(x)  D) -sin(x).",
            "### Explanatory Math MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) cos(x)**\n\n"
            "1. **Rule:** By standard calculus differentiation rules, `d/dx[sin(x)] = cos(x)`.\n"
            "2. **Distinction:** `d/dx[cos(x)] = -sin(x)` (`D`), while `-cos(x)` (`A`) is the integral of `sin(x)`.",
        ),
        (
            "Evaluate step by step: (24 * 15) + 140.",
            "### Step-by-Step Arithmetic (SautiLM-Jua-STEM)\n\n"
            "1. **Multiply First (Order of Operations):** `24 * 15 = 360`.\n"
            "2. **Add Remainder:** `360 + 140 = 500`.\n\n"
            "**Final Answer:** **500**.",
        ),
        (
            "A right triangle has legs a = 5 cm and b = 12 cm. Find the hypotenuse c.",
            "### Pythagoras Theorem (SautiLM-Jua-STEM)\n\n"
            "1. **Formula:** `c^2 = a^2 + b^2`.\n"
            "2. **Compute:** `c^2 = 5^2 + 12^2 = 25 + 144 = 169`.\n"
            "3. **Square Root:** `c = sqrt(169) = 13 cm`.\n\n"
            "**Final Answer:** `c = 13 cm`.",
        ),
        # === PHYSICS (EN) ===
        (
            "STEM Physics MCQ: According to Ohm's Law (V = I * R), if a 12 V battery powers a 4 Ohm resistor, what is the current I? A) 48 A  B) 3 A  C) 0.33 A  D) 8 A.",
            "### Explanatory Physics MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) 3 A**\n\n"
            "1. **Formula:** Rearranging Ohm's Law gives `I = V / R`.\n"
            "2. **Compute:** `I = 12 V / 4 Ohm = 3 A`.\n"
            "3. **Distractor Check:** `A` mistakenly multiplies (`12 * 4`), while `C` inverts the ratio (`4 / 12`).",
        ),
        (
            "Calculate the kinetic energy of a 6 kg cart moving at 4 m/s.",
            "### Kinetic Energy Calculation (SautiLM-Jua-STEM)\n\n"
            "1. **Formula:** `KE = 0.5 * m * v^2`.\n"
            "2. **Substitute Values:** `KE = 0.5 * 6 kg * (4 m/s)^2 = 3 * 16 = 48 J`.\n\n"
            "**Final Answer:** `KE = 48 Joules`.",
        ),
        (
            "How do photovoltaic solar cells convert sunlight into electric current?",
            "### Solar PV Working Principle (SautiLM-Jua-STEM)\n\n"
            "1. **Photon Absorption:** Sunlight strikes doped **silicon** semiconductor layers across a p-n junction.\n"
            "2. **Electron Flow:** Photons release free electrons that flow through an external circuit as **direct current (DC)** electricity.",
        ),
        (
            "STEM Physics MCQ: Two resistors R1 = 4 Ohm and R2 = 6 Ohm are connected in series. What is the total resistance? A) 2.4 Ohm  B) 10 Ohm  C) 24 Ohm  D) 5 Ohm.",
            "### Explanatory Physics MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) 10 Ohm**\n\n"
            "1. **Series Rule:** Total resistance in series is the direct sum: `R_total = R1 + R2 = 4 + 6 = 10 Ohm`.\n"
            "2. **Note:** `A) 2.4 Ohm` corresponds to a parallel circuit, not series.",
        ),
        # === CHEMISTRY (EN) ===
        (
            "Calculate the number of moles in 44 g of carbon dioxide CO2 (C = 12 g/mol, O = 16 g/mol).",
            "### Mole Calculation (SautiLM-Jua-STEM)\n\n"
            "1. **Molar Mass of CO2:** `M(CO2) = 12 + 2 * 16 = 44 g/mol`.\n"
            "2. **Formula:** `n = mass / molar mass = 44 g / 44 g/mol = 1 mol`.\n\n"
            "**Final Answer:** **1 mole** of CO2.",
        ),
        (
            "STEM Chemistry MCQ: A solution has pH = 11 at 25 C. How is it classified? A) Strongly acidic  B) Neutral  C) Basic / Alkaline  D) Salt-free.",
            "### Explanatory Chemistry MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: C) Basic / Alkaline**\n\n"
            "1. **Rule:** Solutions with `pH > 7` have higher hydroxide ion (`OH-`) concentration and are basic (alkaline). Acids have `pH < 7` and pure water is neutral at `pH = 7`.",
        ),
        (
            "Balance the chemical equation for water formation: H2 + O2 -> H2O.",
            "### Balancing Chemical Equations (SautiLM-Jua-STEM)\n\n"
            "1. **Count Atoms:** `O2` supplies 2 oxygen atoms, so we need `2 H2O` on the product side.\n"
            "2. **Balance Hydrogen:** `2 H2O` contains 4 hydrogen atoms, requiring `2 H2` on the reactant side.\n\n"
            "**Balanced Equation:** `2 H2 + O2 -> 2 H2O`.",
        ),
        (
            "How do you calculate molarity C when dissolving 0.4 mol of solute in 500 mL of solution?",
            "### Molarity Step-by-Step (SautiLM-Jua-STEM)\n\n"
            "1. **Convert Volume:** `V = 500 mL = 0.5 L`.\n"
            "2. **Formula (`C = n / V`):** `C = 0.4 mol / 0.5 L = 0.8 mol/L (0.8 M)`.\n\n"
            "**Final Answer:** **0.8 M**.",
        ),
        # === BIOLOGY (EN) ===
        (
            "STEM Biology MCQ: Which organelle and pigment in green leaves capture sunlight for photosynthesis? A) Mitochondria and hemoglobin  B) Chloroplasts and chlorophyll  C) Ribosomes and keratin  D) Lysosomes and melanin.",
            "### Explanatory Biology MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) Chloroplasts and chlorophyll**\n\n"
            "1. **Reason:** **Chlorophyll** inside **chloroplasts** absorbs solar light energy to convert `CO2` and `H2O` into glucose and oxygen (`O2`).\n"
            "2. **Distractors:** Mitochondria (`A`) generate ATP via respiration; ribosomes (`C`) build proteins.",
        ),
        (
            "STEM Biology MCQ: Which human heart chamber pumps oxygen-rich blood into the aorta? A) Right atrium  B) Right ventricle  C) Left atrium  D) Left ventricle.",
            "### Explanatory Biology MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: D) Left ventricle**\n\n"
            "1. **Reason:** Oxygenated blood returns from the lungs to the left atrium (`C`) and enters the thick-walled **left ventricle (`D`)**, which pumps it through the aorta to the whole body.",
        ),
        (
            "Explain the key difference between mitosis and meiosis.",
            "### Cell Division Revision Note (SautiLM-Jua-STEM)\n\n"
            "1. **Mitosis:** Produces **2 genetically identical diploid (`2n`) daughter cells** for growth and tissue repair.\n"
            "2. **Meiosis:** Produces **4 genetically distinct haploid (`n`) gametes** (sperm or egg cells) for sexual reproduction.",
        ),
        (
            "Explain the three stages of the water cycle (evaporation, condensation, precipitation).",
            "### The Water Cycle (SautiLM-Jua-STEM)\n\n"
            "1. **Evaporation:** Solar heat warms surface water from rivers and lakes, turning liquid water into rising water vapor.\n"
            "2. **Condensation:** As warm vapor rises and cools in the atmosphere, it condenses into tiny droplets forming clouds.\n"
            "3. **Precipitation:** Heavy water droplets fall back to the Earth as rain or hail, recharging rivers and groundwater.",
        ),
    ]


def generate_stem_instructions() -> List[Dict[str, Any]]:
    """
    Generates SautiLM-Jua-STEM corpus (Secondary School & Examen d'État RDC).
    Strict balance: 60% French (24 curated base pairs x rep) / 40% English (16 curated base pairs x rep).
    Zero Bbadha tokens, zero non-STEM law/civic chatter.
    """
    out: List[Dict[str, Any]] = []
    fr_pairs = _french_stem_pairs()
    en_pairs = _english_stem_pairs()

    # Exact 60/40 ratio verification: 24 FR vs 16 EN -> 24/(24+16) = 60.0% FR, 40.0% EN
    for idx, (q, a) in enumerate(fr_pairs):
        for rep in range(6):
            sys_p = STEM_SYSTEM_FR if rep % 2 == 0 else None
            out.append(_stem_turn("fr", sys_p, q, a))

    for idx, (q, a) in enumerate(en_pairs):
        for rep in range(6):
            sys_p = STEM_SYSTEM_EN if rep % 2 == 0 else None
            out.append(_stem_turn("en", sys_p, q, a))

    # Dedicated STEM identity & brief polite greetings (also 60/40 FR/EN = 3 FR, 2 EN)
    stem_greetings_identity = [
        ("fr", STEM_SYSTEM_FR, "Qui es-tu ?",
         "Je suis **SautiLM-Jua-STEM**, l'assistant scientifique hors-ligne (Mathématiques, Physique, Chimie, Biologie - niveau Secondaire et Examen d'État RDC) développé par **LosingTek Research**. Posez une question scientifique ou demandez un QCM."),
        ("fr", STEM_SYSTEM_FR, "Bonjour",
         "Bonjour ! Je suis **SautiLM-Jua-STEM**. Quelle notion de mathématiques, physique, chimie ou biologie (Secondaire / Examen d'État) voulez-vous réviser aujourd'hui ?"),
        ("fr", STEM_SYSTEM_FR, "Merci beaucoup",
         "Avec plaisir ! N'hésitez pas à soumettre d'autres exercices ou QCM de sciences dans **SautiLM-Jua-STEM**."),
        ("en", STEM_SYSTEM_EN, "Who are you?",
         "I am **SautiLM-Jua-STEM**, an offline secondary school Science and Mathematics assistant (Math, Physics, Chemistry, Biology) created by **LosingTek Research**."),
        ("en", STEM_SYSTEM_EN, "Hey",
         "Hello! I am **SautiLM-Jua-STEM**. Which secondary Math, Physics, Chemistry, or Biology problem or MCQ would you like to solve today?"),
    ]
    for lang, sys_p, q, a in stem_greetings_identity:
        for rep in range(4):
            out.append(_stem_turn(lang, sys_p if rep % 2 == 0 else None, q, a))

    return out
