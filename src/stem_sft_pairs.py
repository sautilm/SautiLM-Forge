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


def _fmt_num(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.4g}".rstrip("0").rstrip(".")


def _fmt_coeff(c: int, var: str, leading: bool = False) -> str:
    if c == 0:
        return ""
    sign = "" if (leading and c > 0) else (" - " if c < 0 else " + ")
    if leading and c < 0:
        sign = "-"
    abs_c = abs(c)
    coef_str = "" if (abs_c == 1 and var) else str(abs_c)
    return f"{sign}{coef_str}{var}"


def _quadratic_fr(a: int, r1: int, r2: int) -> tuple[str, str]:
    b = -a * (r1 + r2)
    c = a * r1 * r2
    delta = b * b - 4 * a * c
    sqrt_d = int(round(delta ** 0.5))
    eq = f"{_fmt_coeff(a, 'x^2', True)}{_fmt_coeff(b, 'x')}{_fmt_coeff(c, '')} = 0".strip()
    lo, hi = sorted((r1, r2))
    ans = f"`S = {{{lo}}}`" if lo == hi else f"`S = {{{lo} ; {hi}}}`"
    return (
        f"Résous dans R l'équation du second degré : {eq} (Examen d'État).",
        f"### Équation du 2nd Degré (SautiLM-Jua-STEM)\n\n"
        f"1. **Discriminant :** `Delta = b^2 - 4ac` avec `a = {a}`, `b = {b}`, `c = {c}`.\n"
        f"2. **Calcul :** `Delta = ({b})^2 - 4({a})({c}) = {b * b} - ({4 * a * c}) = {delta}` (soit `sqrt(Delta) = {sqrt_d}`).\n"
        f"3. **Racines :** `x1 = ({-b} - {sqrt_d}) / {2 * a} = {lo}` et `x2 = ({-b} + {sqrt_d}) / {2 * a} = {hi}`.\n\n"
        f"**Réponse finale :** {ans}.",
    )


def _quadratic_en(a: int, r1: int, r2: int) -> tuple[str, str]:
    b = -a * (r1 + r2)
    c = a * r1 * r2
    delta = b * b - 4 * a * c
    sqrt_d = int(round(delta ** 0.5))
    eq = f"{_fmt_coeff(a, 'x^2', True)}{_fmt_coeff(b, 'x')}{_fmt_coeff(c, '')} = 0".strip()
    lo, hi = sorted((r1, r2))
    return (
        f"Solve step by step the quadratic equation: {eq}.",
        f"### Quadratic Equation Solution (SautiLM-Jua-STEM)\n\n"
        f"1. **Discriminant (`Delta = b^2 - 4ac`):** With `a = {a}`, `b = {b}`, `c = {c}`: `Delta = ({b})^2 - 4({a})({c}) = {delta}` (`sqrt(Delta) = {sqrt_d}`).\n"
        f"2. **Roots (`x = (-b +- sqrt(Delta)) / 2a`):** `x1 = {lo}` and `x2 = {hi}`.\n\n"
        f"**Final Answer:** `x = {lo}` and `x = {hi}`.",
    )


def _french_stem_pairs() -> List[tuple[str, str]]:
    """Builds 180 diverse French Secondary / Examen d'État STEM problems (Maths, Physics, Chemistry, Biology)."""
    pairs: List[tuple[str, str]] = [
        # --- MATHEMATIQUES (45 paires) ---
        (
            "Résous dans R l'équation du second degré : 2x^2 - 4x - 6 = 0 (Examen d'État).",
            "### Équation du 2nd Degré (SautiLM-Jua-STEM)\n\n"
            "1. **Formule du discriminant :** `Delta = b^2 - 4ac` avec `a = 2`, `b = -4`, `c = -6`.\n"
            "2. **Calcul :** `Delta = (-4)^2 - 4(2)(-6) = 16 + 48 = 64 = 8^2`.\n"
            "3. **Racines :** `x1 = (4 + 8) / 4 = 3` et `x2 = (4 - 8) / 4 = -1`.\n\n"
            "**Réponse finale :** `S = {-1 ; 3}`.",
        ),
        (
            "QCM Maths : Pour l'équation x^2 - 5x + 6 = 0, quelles sont les solutions ? A) {-2; -3}  B) {2; 3}  C) {1; 6}  D) {-1; 6}.",
            "### QCM Explicatif : Mathématiques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) {2 ; 3}**\n\n"
            "1. **Somme et produit :** `S = -b/a = 5` et `P = c/a = 6`, d'où les racines `2` et `3`.\n"
            "2. **Élimination :** `A` a une somme négative (`-5`), `C` et `D` donnent un somme/produit faux.",
        ),
        (
            "Calcule l'intégrale de sinus x divisé par cosinus x dx étape par étape.",
            "### Calcul Intégral (SautiLM-Jua-STEM)\n\n"
            "**Expression :** Intégrale de `sin(x) / cos(x) dx = tan(x) dx`.\n\n"
            "1. **Changement de variable :** Posons `u = cos(x)`, donc `du = -sin(x) dx`, soit `sin(x) dx = -du`.\n"
            "2. **Intégration :** L'intégrale devient `- int (1 / u) du = -ln|u| + C`.\n"
            "3. **Substitution inverse :** `-ln|cos(x)| + C = ln|sec(x)| + C`.\n\n"
            "**Réponse finale :** `-ln|cos(x)| + C`.",
        ),
        (
            "Calcule une primitive de f(x) = cos(x) et vérifie par dérivation.",
            "### Calcul Intégral : Primitives Usuelles (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** La primitive de `cos(x)` est `sin(x) + C` (constante réelle).\n"
            "2. **Vérification :** `(sin(x) + C)' = cos(x)`.\n\n"
            "**Réponse finale :** `F(x) = sin(x) + C`.",
        ),
        (
            "QCM Maths : Que vaut sin^2(x) + cos^2(x) pour tout réel x ? A) 0  B) 1  C) 2  D) tan(x).",
            "### QCM Explicatif : Trigonométrie (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) 1**\n\n"
            "1. **Justification :** D'après le théorème de Pythagore sur le cercle trigonométrique de rayon `1`, `cos^2(x) + sin^2(x) = 1` pour tout réel `x`.",
        ),
        (
            "QCM Maths : Quelle est la dérivée de ln(x) pour x > 0 ? A) 1/x  B) x  C) e^x  D) -1/x^2.",
            "### QCM Explicatif : Dérivation (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : A) 1/x**\n\n"
            "1. **Règle :** Pour tout `x > 0`, `(ln(x))' = 1 / x`. `C) e^x` est la dérivée de `e^x`.",
        ),
        (
            "Un sac contient 3 boules rouges et 5 boules bleues. Quelle est la probabilité de tirer une boule rouge ?",
            "### Probabilités (SautiLM-Jua-STEM)\n\n"
            "1. **Cas possibles :** `3 + 5 = 8` boules.\n"
            "2. **Formule :** `P(Rouge) = cas favorables / cas possibles = 3 / 8 = 0,375`.\n\n"
            "**Réponse finale :** `3/8` (soit `37,5 %`).",
        ),
        (
            "Calcule le nombre de permutations de 4 objets distincts (factorielle 4!).",
            "### Analyse Combinatoire (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `P_n = n!`.\n"
            "2. **Calcul :** `4! = 4 * 3 * 2 * 1 = 24`.\n\n"
            "**Réponse finale :** **24** arrangements possibles.",
        ),
    ]

    # +10 équations quadratiques paramétriques
    for quad_args in [(1, 1, 4), (1, -2, 5), (1, 3, 7), (1, -4, 2), (2, -1, 4), (1, 2, 6), (1, -3, -1), (3, 1, 2), (1, 4, 5), (2, 1, 5)]:
        pairs.append(_quadratic_fr(*quad_args))

    # +8 dérivées polynomiales paramétriques
    for a, b, c, d in [(3, -5, 4, -7), (2, 3, -6, 1), (4, -2, 5, 9), (1, 4, -3, 2), (5, -1, 2, -8), (2, -7, 1, 4), (3, 2, 6, -5), (1, -6, 9, 3)]:
        pairs.append((
            f"Calcule la dérivée de f(x) = {a}x^3 + ({b})x^2 + ({c})x + ({d}).",
            f"### Dérivée Polynomiale (SautiLM-Jua-STEM)\n\n"
            f"1. **Règle :** `(x^n)' = n * x^(n-1)` et `(constante)' = 0`.\n"
            f"2. **Calcul :** `f'(x) = {3 * a}x^2 + ({2 * b})x + ({c})`.\n\n"
            f"**Réponse finale :** `f'(x) = {3 * a}x^2 + ({2 * b})x + ({c})`.",
        ))

    # +7 suites arithmétiques paramétriques
    for u1, r, n in [(5, 3, 10), (2, 4, 8), (10, -2, 6), (3, 5, 11), (7, 2, 15), (1, 6, 9), (12, 3, 7)]:
        un = u1 + (n - 1) * r
        pairs.append((
            f"Une suite arithmétique a pour premier terme u1 = {u1} et pour raison r = {r}. Calcule u{n}.",
            f"### Suite Arithmétique (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `un = u1 + (n - 1) * r`.\n"
            f"2. **Substitution :** `u{n} = {u1} + ({n} - 1) * ({r}) = {u1} + {(n - 1) * r} = {un}`.\n\n"
            f"**Réponse finale :** `u{n} = {un}`.",
        ))

    # +6 triangles rectangles Pythagore paramétriques
    for a, b, c in [(6, 8, 10), (3, 4, 5), (5, 12, 13), (8, 15, 17), (9, 12, 15), (12, 16, 20)]:
        pairs.append((
            f"Un triangle rectangle a pour côtés de l'angle droit a = {a} cm et b = {b} cm. Calcule l'hypoténuse c.",
            f"### Théorème de Pythagore (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `c^2 = a^2 + b^2`.\n"
            f"2. **Calcul :** `c^2 = {a}^2 + {b}^2 = {a * a} + {b * b} = {c * c}`.\n"
            f"3. **Racine :** `c = racine({c * c}) = {c} cm`.\n\n"
            f"**Réponse finale :** **c = {c} cm**.",
        ))

    # +6 systèmes linéaires paramétriques
    for x, y in [(7, 3), (8, 2), (10, 4), (9, 5), (6, 1), (11, 5)]:
        s_val = x + y
        d_val = x - y
        pairs.append((
            f"Résous le système d'équations : x + y = {s_val} et x - y = {d_val}.",
            f"### Système Linéaire (SautiLM-Jua-STEM)\n\n"
            f"1. **Addition :** `2x = {s_val} + {d_val} = {2 * x}`, donc `x = {x}`.\n"
            f"2. **Déduction de y :** `y = {s_val} - {x} = {y}`.\n\n"
            f"**Réponse finale :** `(x ; y) = ({x} ; {y})`.",
        ))

    # --- PHYSIQUE (45 paires) ---
    # 10 Loi d'Ohm U = R * I + puissance P = U * I
    for r_val, i_val in [(4, 3), (5, 2), (6, 4), (10, 2), (8, 3), (15, 2), (20, 3), (3, 5), (9, 2), (12, 4)]:
        u_val = r_val * i_val
        p_val = u_val * i_val
        pairs.append((
            f"Une résistance R = {r_val} Ohms est traversée par un courant I = {i_val} A. Calcule la tension U et la puissance P.",
            f"### Loi d'Ohm et Puissance (SautiLM-Jua-STEM)\n\n"
            f"1. **Tension (`U = R * I`) :** `U = {r_val} Ohms * {i_val} A = {u_val} V`.\n"
            f"2. **Puissance (`P = U * I`) :** `P = {u_val} V * {i_val} A = {p_val} W`.\n\n"
            f"**Réponse finale :** `U = {u_val} V` et `P = {p_val} W`.",
        ))

    # 8 2e loi de Newton F = m * a (held out: m=5, a=3)
    for m_val, a_val in [(4, 2), (5, 2), (6, 4), (10, 3), (8, 2), (12, 3), (7, 4), (15, 2)]:
        f_val = m_val * a_val
        pairs.append((
            f"Un solide de masse m = {m_val} kg subit une accélération a = {a_val} m/s^2. Calcule la force F.",
            f"### Deuxième Loi de Newton (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `F = m * a`.\n"
            f"2. **Calcul :** `F = {m_val} kg * {a_val} m/s^2 = {f_val} N`.\n\n"
            f"**Réponse finale :** `F = {f_val} Newton (N)`.",
        ))

    # 7 Énergie cinétique Ec = 0.5 * m * v^2
    for m_val, v_val in [(4, 5), (2, 6), (6, 4), (8, 3), (10, 4), (12, 5), (4, 10)]:
        ec_val = 0.5 * m_val * (v_val ** 2)
        pairs.append((
            f"Calcule l'énergie cinétique d'un solide de masse m = {m_val} kg animé d'une vitesse v = {v_val} m/s.",
            f"### Énergie Cinétique (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `Ec = (1/2) * m * v^2`.\n"
            f"2. **Calcul :** `Ec = 0,5 * {m_val} * {v_val}^2 = {_fmt_num(ec_val)} Joules`.\n\n"
            f"**Réponse finale :** `Ec = {_fmt_num(ec_val)} J`.",
        ))

    # 6 Énergie potentielle Ep = m * g * h
    for m_val, h_val in [(10, 5), (4, 8), (6, 5), (12, 10), (3, 15), (8, 4)]:
        ep_val = m_val * 10 * h_val
        pairs.append((
            f"Un corps de masse m = {m_val} kg se trouve à une hauteur h = {h_val} m (g = 10 m/s^2). Calcule son énergie potentielle Ep.",
            f"### Énergie Potentielle de Pesanteur (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `Ep = m * g * h`.\n"
            f"2. **Calcul :** `Ep = {m_val} * 10 * {h_val} = {ep_val} Joules`.\n\n"
            f"**Réponse finale :** `Ep = {ep_val} J`.",
        ))

    # 6 Circuits série / parallèle
    for r1, r2 in [(6, 3), (12, 4), (10, 10), (8, 8), (15, 10), (9, 18)]:
        r_par = (r1 * r2) / (r1 + r2)
        pairs.append((
            f"Deux résistances R1 = {r1} Ohms et R2 = {r2} Ohms sont associées en parallèle. Calcule la résistance équivalente Req.",
            f"### Association en Parallèle (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `Req = (R1 * R2) / (R1 + R2)`.\n"
            f"2. **Calcul :** `Req = ({r1} * {r2}) / ({r1} + {r2}) = {r1 * r2} / {r1 + r2} = {_fmt_num(r_par)} Ohms`.\n\n"
            f"**Réponse finale :** `Req = {_fmt_num(r_par)} Ohms`.",
        ))

    # +8 Physique QCM et optique/cinématique
    pairs.extend([
        (
            "QCM Physique : Que mesure un ampèremètre et comment se branche-t-il ? A) La tension, en parallèle  B) L'intensité du courant, en série  C) La résistance, hors circuit  D) La puissance, en parallèle.",
            "### QCM Explicatif : Mesures Électriques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) L'intensité du courant, en série**\n\n"
            "1. **Justification :** Traversé par le courant à mesurer (en Ampères `A`), l'ampèremètre se branche **en série** ; le voltmètre se branche en parallèle.",
        ),
        (
            "QCM Physique : Quelle est l'unité SI du travail mécanique et de l'énergie ? A) Le Watt  B) Le Newton  C) Le Joule  D) Le Pascal.",
            "### QCM Explicatif : Unités SI (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) Le Joule**\n\n"
            "1. **Rappel :** Le **Joule (J)** mesure l'énergie et le travail (`W = F * d`), le Watt mesure la puissance, le Newton mesure la force et le Pascal mesure la pression.",
        ),
        (
            "Un véhicule parcourt d = 150 km à vitesse constante pendant t = 2,5 heures. Calcule sa vitesse moyenne v en km/h.",
            "### Mouvement Rectiligne Uniforme (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `v = d / t`.\n"
            "2. **Calcul :** `v = 150 km / 2,5 h = 60 km/h`.\n\n"
            "**Réponse finale :** `v = 60 km/h`.",
        ),
        (
            "Calcule la fréquence f d'un phénomène périodique dont la période est T = 0,02 s.",
            "### Onde et Périodicité (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `f = 1 / T`.\n"
            "2. **Calcul :** `f = 1 / 0,02 s = 50 Hz`.\n\n"
            "**Réponse finale :** `f = 50 Hertz (Hz)`.",
        ),
        (
            "Une lentille convergente a une distance focale f' = 0,25 m. Calcule sa vergence C en dioptries.",
            "### Optique Géométrique (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `C = 1 / f'`.\n"
            "2. **Calcul :** `C = 1 / 0,25 m = 4 dioptries`.\n\n"
            "**Réponse finale :** `C = +4 delta (dioptries)`.",
        ),
        (
            "QCM Physique : Lorsqu'on chauffe de l'eau pure à pression atmosphérique normale, à quelle température bout-elle ? A) 0°C  B) 50°C  C) 100°C  D) 273°C.",
            "### QCM Explicatif : Changements d'État (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) 100°C**\n\n"
            "1. **Principe :** À 1 atm, l'eau pure fond à `0°C` (`A`) et entre en ébullition à `100°C` (`C`).",
        ),
        (
            "Calcule la pression exercée par une force pressante F = 600 N sur une surface S = 0,2 m^2.",
            "### Pression Mécanique (SautiLM-Jua-STEM)\n\n"
            "1. **Formule :** `P = F / S`.\n"
            "2. **Calcul :** `P = 600 N / 0,2 m^2 = 3000 Pa`.\n\n"
            "**Réponse finale :** `P = 3000 Pascals (Pa)`.",
        ),
        (
            "QCM Physique : Selon la loi de Coulomb ou l'électrostatique, comment interagissent deux charges de même signe ? A) Elles s'attirent  B) Elles se repoussent  C) Elles s'annulent  D) Aucune force.",
            "### QCM Explicatif : Électrostatique (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Elles se repoussent**\n\n"
            "1. **Règle :** Deux charges électriques de même signe (`+`/`+` ou `-`/`-`) se repoussent, tandis que des signes opposés s'attirent.",
        ),
    ])

    # --- CHIMIE (45 paires) ---
    # 15 Calculs de moles n = m / M (held out test probe: 36 g d'eau H2O!)
    mole_cases = [
        ("eau H2O", 18, 18),
        ("eau H2O", 18, 54),
        ("eau H2O", 18, 72),
        ("dioxyde de carbone CO2", 44, 44),
        ("dioxyde de carbone CO2", 44, 88),
        ("dioxyde de carbone CO2", 44, 22),
        ("dioxygène O2", 32, 64),
        ("dioxygène O2", 32, 96),
        ("carbonate de calcium CaCO3", 100, 200),
        ("carbonate de calcium CaCO3", 100, 50),
        ("soude NaOH", 40, 80),
        ("soude NaOH", 40, 120),
        ("acide sulfurique H2SO4", 98, 196),
        ("ammoniac NH3", 17, 34),
        ("méthane CH4", 16, 48),
    ]
    for subst, molar_mass, mass in mole_cases:
        n_moles = mass / molar_mass
        pairs.append((
            f"Calcule le nombre de moles n contenues dans m = {mass} g de {subst} (M = {molar_mass} g/mol).",
            f"### Quantité de Matière (SautiLM-Jua-STEM)\n\n"
            f"1. **Formule :** `n = m / M` (masse en grammes divisée par la masse molaire en `g/mol`).\n"
            f"2. **Application numérique :** `n = {mass} g / {molar_mass} g/mol = {_fmt_num(n_moles)} mol`.\n\n"
            f"**Réponse finale :** **{_fmt_num(n_moles)} mol**.",
        ))

    # 8 Concentration molaire C = n / V
    for n_val, v_ml in [(0.5, 250), (1.0, 500), (0.2, 100), (0.6, 300), (1.5, 500), (0.4, 200), (2.0, 1000), (0.3, 150)]:
        v_l = v_ml / 1000.0
        conc = n_val / v_l
        pairs.append((
            f"On dissout n = {_fmt_num(n_val)} mol de soluté dans V = {v_ml} mL de solution. Calcule la concentration molaire C.",
            f"### Concentration Molaire (SautiLM-Jua-STEM)\n\n"
            f"1. **Conversion en litres :** `V = {v_ml} mL = {_fmt_num(v_l)} L`.\n"
            f"2. **Formule (`C = n / V`) :** `C = {_fmt_num(n_val)} mol / {_fmt_num(v_l)} L = {_fmt_num(conc)} mol/L`.\n\n"
            f"**Réponse finale :** `C = {_fmt_num(conc)} mol/L`.",
        ))

    # 9 pH et solutions acide / neutre / basique
    for ph_val, nature, reason in [
        (2, "C) Acide", "`pH < 7` indique un excès d'ions `H3O+` (solution acide)."),
        (3, "C) Acide", "`3 < 7`, donc la solution est acide (excès d'ions `H3O+`)."),
        (5, "A) Acide", "`5 < 7`, donc la solution est acide."),
        (7, "B) Neutre", "À 25°C, `pH = 7` correspond exactement à une solution neutre (`[H3O+] = [OH-]`)."),
        (9, "D) Basique", "`9 > 7` indique une solution basique / alcaline (excès d'ions `OH-`)."),
        (11, "D) Basique", "`11 > 7` signifie que la solution est fortement basique / alcaline."),
        (13, "C) Basique", "`pH = 13 > 7` correspond à une base forte comme la soude `NaOH`."),
        (1, "B) Fortement acide", "`pH = 1` est très inférieur à `7` (solution fortement acide)."),
        (4, "C) Acide", "`4 < 7` place la solution dans le domaine acide sur l'échelle 0-14."),
    ]:
        pairs.append((
            f"QCM Chimie : Une solution aqueuse présente un pH = {ph_val} à 25°C. Quelle est sa nature ?",
            f"### QCM Explicatif : Échelle de pH (SautiLM-Jua-STEM)\n\n"
            f"**Bonne réponse : {nature}**\n\n"
            f"1. **Justification :** {reason}",
        ))

    # +13 Chimie : équations chimiques, atomistique, redox, organique
    pairs.extend([
        (
            "Équilibre l'équation chimique de combustion complète du méthane : CH4 + O2 -> CO2 + H2O.",
            "### Équilibration d'Équation Chimique (SautiLM-Jua-STEM)\n\n"
            "1. **Carbone :** `1 CH4` -> `1 CO2`. **Hydrogène :** `4 H` -> `2 H2O`.\n"
            "2. **Oxygène :** `2 + 2 = 4 O` à droite -> `2 O2` à gauche.\n\n"
            "**Équation équilibrée :** `CH4 + 2 O2 -> CO2 + 2 H2O`.",
        ),
        (
            "Équilibre la réaction de synthèse de l'ammoniac : N2 + H2 -> NH3.",
            "### Équilibration d'Équation Chimique (SautiLM-Jua-STEM)\n\n"
            "1. **Azote :** `1 N2` donne `2 NH3`.\n"
            "2. **Hydrogène :** `2 NH3` contient `6 H`, soit `3 H2`.\n\n"
            "**Équation équilibrée :** `N2 + 3 H2 -> 2 NH3`.",
        ),
        (
            "Équilibre la combustion complète de l'éthane : C2H6 + O2 -> CO2 + H2O.",
            "### Équilibration d'Équation Chimique (SautiLM-Jua-STEM)\n\n"
            "1. Pour `2 C2H6` : `4 CO2` (4 C) et `6 H2O` (12 H).\n"
            "2. Oxygène à droite : `4*2 + 6 = 14 O`, soit `7 O2`.\n\n"
            "**Équation équilibrée :** `2 C2H6 + 7 O2 -> 4 CO2 + 6 H2O`.",
        ),
        (
            "QCM Chimie : Qu'est-ce qu'une réaction d'oxydation ? A) Un gain d'électrons  B) Une perte d'électrons  C) Une neutralisation  D) Une fusion.",
            "### QCM Explicatif : Oxydoréduction (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Une perte d'électrons**\n\n"
            "1. **Règle :** L'oxydation est une perte d'électrons par un réducteur ; la réduction est un gain d'électrons.",
        ),
        (
            "QCM Chimie : Qu'est-ce qu'une réduction en électrochimie ? A) Gain d'électrons  B) Perte de protons  C) Perte d'électrons  D) Évaporation.",
            "### QCM Explicatif : Oxydoréduction (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : A) Gain d'électrons**\n\n"
            "1. **Exemple :** `Cu^2+ + 2e- -> Cu` : l'ion cuivre capte 2 électrons et subit une **réduction**.",
        ),
        (
            "Quelle est la formule générale des alcanes saturés acycliques et cite les deux premiers ?",
            "### Chimie Organique : Alcanes (SautiLM-Jua-STEM)\n\n"
            "1. **Formule générale :** `C_n H_{2n+2}` (`n >= 1`).\n"
            "2. **Premiers termes :** `n = 1` : **Méthane (`CH4`)** ; `n = 2` : **Éthane (`C2H6`)**.",
        ),
        (
            "Un atome possède le numéro atomique Z = 11 et le nombre de masse A = 23 (Sodium Na). Donne le nombre de protons, d'électrons et de neutrons.",
            "### Structure de l'Atome (SautiLM-Jua-STEM)\n\n"
            "1. **Protons et électrons :** `Z = 11`, donc **11 protons** et **11 électrons** (atome neutre).\n"
            "2. **Neutrons (`N = A - Z`) :** `N = 23 - 11 = 12 neutrons`.\n\n"
            "**Réponse finale :** 11 protons, 11 électrons et 12 neutrons.",
        ),
        (
            "Calcule la masse molaire moléculaire du glucose C6H12O6 (C=12, H=1, O=16 g/mol).",
            "### Masse Molaire Moléculaire (SautiLM-Jua-STEM)\n\n"
            "1. **Calcul :** `M = 6(12) + 12(1) + 6(16) = 72 + 12 + 96 = 180 g/mol`.\n\n"
            "**Réponse finale :** `180 g/mol`.",
        ),
        (
            "QCM Chimie : Quelle liaison unit deux atomes qui mettent en commun une paire d'électrons ? A) Liaison ionique  B) Liaison covalente  C) Liaison métallique  D) Force de gravité.",
            "### QCM Explicatif : Liaisons Chimiques (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Liaison covalente**\n\n"
            "1. **Principe :** Une **liaison covalente** résulte de la mise en commun d'un doublet d'électrons de valence.",
        ),
        (
            "Dans une dilution, on prélève 20 mL d'une solution mère de concentration C1 = 1,5 mol/L et on complète à V2 = 100 mL. Calcule C2.",
            "### Dilution d'une Solution (SautiLM-Jua-STEM)\n\n"
            "1. **Conservation de matière (`C1 * V1 = C2 * V2`) :** `C2 = (1,5 * 20) / 100 = 30 / 100 = 0,3 mol/L`.\n\n"
            "**Réponse finale :** `C2 = 0,3 mol/L`.",
        ),
        (
            "Quelle est la définition chimique d'un acide selon Brönsted-Lowry ?",
            "### Acides et Bases (SautiLM-Jua-STEM)\n\n"
            "1. **Définition :** Selon Brönsted, un **acide** est une espèce chimique capable de **céder au moins un proton `H+`** (tandis qu'une base est capable de capter un proton `H+`).",
        ),
        (
            "QCM Chimie : Quel gaz trouble l'eau de chaux ? A) Dioxygène O2  B) Diazote N2  C) Dioxyde de carbone CO2  D) Dihydrogène H2.",
            "### QCM Explicatif : Identification des Gaz (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) Dioxyde de carbone CO2**\n\n"
            "1. **Test caractéristique :** Le `CO2` réagit avec l'hydroxyde de calcium pour former un précipité blanc de carbonate de calcium qui trouble l'eau de chaux.",
        ),
        (
            "Combien de électrons de valence possède l'atome de Carbone (Z = 6) et combien de liaisons covalentes forme-t-il ?",
            "### Configuration Électronique (SautiLM-Jua-STEM)\n\n"
            "1. **Couches électroniques (`Z = 6`) :** `(K)^2 (L)^4` -> **4 électrons de valence**.\n"
            "2. **Valence :** Pour satisfaire la règle de l'octet, le carbone est **tétravalent** (il forme **4 liaisons covalentes**).",
        ),
    ])

    # --- BIOLOGIE / SVT (45 paires - variées sans sur-répétition figée) ---
    bio_pairs = [
        (
            "QCM Biologie : Quel organite cellulaire contient la chlorophylle et réalise la photosynthèse ? A) La mitochondrie  B) Le noyau  C) Le chloroplaste  D) Le ribosome.",
            "### QCM Explicatif : Biologie Cellulaire (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : C) Le chloroplaste**\n\n"
            "1. **Justification :** Le **chloroplaste** contient la **chlorophylle**, qui capte la lumière pour transformer `CO2 + H2O` en glucose et dioxygène (`O2`).\n"
            "2. **Distinctions :** La mitochondrie (`A`) produit l'ATP (respiration), le noyau (`B`) renferme l'ADN, le ribosome (`D`) synthétise les protéines.",
        ),
        (
            "Écris l'équation bilan de la photosynthèse et explique le rôle de la chlorophylle.",
            "### Photosynthèse Végétale (SautiLM-Jua-STEM)\n\n"
            "1. **Rôle de la chlorophylle :** Pigment vert des **chloroplastes** qui absorbe l'énergie lumineuse.\n"
            "2. **Équation bilan :** `6 CO2 + 6 H2O --(lumière, chlorophylle)--> C6H12O6 (glucose) + 6 O2 (dioxygène)`.",
        ),
        (
            "Quelle est la différence fondamentale entre la mitose et la méiose en biologie ?",
            "### Division Cellulaire : Mitose vs Méiose (SautiLM-Jua-STEM)\n\n"
            "1. **La Mitose (division conforme) :** Une cellule mère diploïde (`2n`) donne **2 cellules filles identiques diploïdes (`2n`)**. Elle permet la croissance et le renouvellement tissulaire.\n"
            "2. **La Méiose (division réductionnelle) :** Une cellule mère (`2n`) donne **4 gamètes haploïdes (`n`)** indispensables à la reproduction sexuée.",
        ),
        (
            "QCM Biologie : Combien de cellules filles obtient-on après une mitose et quel est leur caryotype ? A) 4 cellules haploïdes n  B) 2 cellules diploïdes identiques 2n  C) 1 cellule tétraploïde 4n  D) 4 cellules diploïdes 2n.",
            "### QCM Explicatif : Division Cellulaire (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) 2 cellules diploïdes identiques 2n**\n\n"
            "1. **Rappel :** La **mitose** conserve strictement le nombre de chromosomes (`2n -> 2n`), contrairement à la **méiose** qui réduit de moitié à `n`.",
        ),
        (
            "Cite les 4 phases principales de la mitose dans l'ordre chronologique.",
            "### Phases de la Mitose (SautiLM-Jua-STEM)\n\n"
            "1. **Prophase :** Condensation des chromosomes et disparition de l'enveloppe nucléaire.\n"
            "2. **Métaphase :** Alignement des chromosomes sur le plan équatorial.\n"
            "3. **Anaphase :** Séparation et migration des chromatides sœurs vers les pôles.\n"
            "4. **Télophase :** Reconstitution des noyaux et division du cytoplasme (cytodiérèse).",
        ),
        (
            "QCM Biologie : Quelle cavité du cœur humain propulse le sang oxygéné dans l'aorte vers tout le corps ? A) Oreillette droite  B) Ventricule droit  C) Oreillette gauche  D) Ventricule gauche.",
            "### QCM Explicatif : Physiologie Cardiaque (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : D) Ventricule gauche**\n\n"
            "1. **Circulation :** Le sang oxygéné venant des poumons entre dans l'oreillette gauche (`C`) puis le **ventricule gauche (`D`)** qui l'éjecte dans l'**aorte**.",
        ),
        (
            "En génétique mendélienne, on croise deux plantes hétérozygotes Rr x Rr (R dominant sur r). Quelles sont les proportions phénotypiques en F2 ?",
            "### Génétique Mendélienne (SautiLM-Jua-STEM)\n\n"
            "1. **Échiquier :** `RR` (25 %), `Rr` (25 %), `Rr` (25 %), `rr` (25 %).\n\n"
            "**Réponse finale :** Proportions phénotypiques **3:1** (75 % dominant `[R]`, 25 % récessif `[r]`).",
        ),
        (
            "QCM Biologie : Quel est le rôle principal de l'ARNm lors de la synthèse des protéines ? A) Stocker l'ATP  B) Transporter le message génétique du noyau vers les ribosomes  C) Coaguler le sang  D) Digérer les lipides.",
            "### QCM Explicatif : Expression Génétique (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Transporter le message génétique du noyau vers les ribosomes**\n\n"
            "1. **Mécanisme :** L'ARNm est copié depuis l'ADN lors de la **transcription**, puis lu par les **ribosomes** lors de la **traduction**.",
        ),
        (
            "Quel est le rôle des mitochondries dans une cellule eucaryote ?",
            "### Bioénergétique Cellulaire (SautiLM-Jua-STEM)\n\n"
            "1. **Fonction :** Les **mitochondries** réalisent la **respiration cellulaire aérobie** (oxydation du glucose en présence d'`O2`) pour produire de l'**ATP**, source d'énergie utilisable par la cellule.",
        ),
        (
            "QCM Biologie : Quelle molécule porte l'information héréditaire sous forme de double hélice ? A) L'amidon  B) L'ADN  C) Le glycogène  D) L'hémoglobine.",
            "### QCM Explicatif : Génétique Moléculaire (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) L'ADN**\n\n"
            "1. **Structure :** L'Acide Désoxyribonucléique (**ADN**) forme une **double hélice** constituée de bases azotées complémentaires (`A-T`, `C-G`).",
        ),
        (
            "Dans une molécule d'ADN double brin, un brin porte la séquence A-T-G-C-C-A. Quelle est la séquence du brin complémentaire ?",
            "### Complémentarité des Bases de l'ADN (SautiLM-Jua-STEM)\n\n"
            "1. **Règle d'appariement :** `A <-> T` et `C <-> G`.\n"
            "2. **Application :** `A-T-G-C-C-A` donne **`T-A-C-G-G-T`**.",
        ),
        (
            "QCM Biologie : Quel type de cellule sanguine transporte le dioxygène grâce à l'hémoglobine ? A) Leucocytes  B) Hématies (globules rouges)  C) Plaquettes  D) Neurones.",
            "### QCM Explicatif : Physiologie du Sang (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Hématies (globules rouges)**\n\n"
            "1. **Distinction :** Les hématies transportent l'`O2` via l'hémoglobine ; les leucocytes assurent la défense immunitaire ; les plaquettes interviennent dans la coagulation.",
        ),
        (
            "Quelle est la différence entre la reproduction sexuée et la reproduction asexuée ?",
            "### Modes de Reproduction (SautiLM-Jua-STEM)\n\n"
            "1. **Reproduction sexuée :** Implique la fusion de deux gamètes (`n + n -> 2n`) et génère une diversité génétique.\n"
            "2. **Reproduction asexuée :** Produit des individus génétiquement identiques (clones) à partir d'un seul parent (bouturage, scissiparité, bourgeonnement).",
        ),
        (
            "QCM Biologie : Chez l'humain, quel organe produit l'insuline pour réguler la glycémie ? A) Le rein  B) Le pancréas  C) La rate  D) L'estomac.",
            "### QCM Explicatif : Régulation Glycémique (SautiLM-Jua-STEM)\n\n"
            "**Bonne réponse : B) Le pancréas**\n\n"
            "1. **Rappel :** Les îlots bêta du **pancréas** sécrètent l'insuline (hormone hypoglycémiante) lorsque le taux de glucose sanguin augmente.",
        ),
        (
            "Explique le trajet de l'influx nerveux dans un arc réflexe médullaire simple.",
            "### Système Nerveux : Arc Réflexe (SautiLM-Jua-STEM)\n\n"
            "1. **Récepteur sensoriel** -> **Nerf sensitif (afférent)** -> **Moelle épinière (centre nerveux)** -> **Nerf moteur (efférent)** -> **Effecteur (muscle)**.\n"
            "2. C'est une réponse rapide et involontaire protégeant l'organisme.",
        ),
    ]
    # Compléter à exactement 45 paires de Biologie via des croisements mendéliens & questions d'écologie/physiologie variés
    for dom_name, rec_name, p1, p2, f1_desc, f2_ratio in [
        ("graine lisse L", "graine ridée l", "LL", "ll", "100 % [L] hétérozygotes Ll", "3/4 [L] et 1/4 [l]"),
        ("fleur rouge R", "fleur blanche r", "RR", "rr", "100 % [R] hétérozygotes Rr", "75 % [R] et 25 % [r]"),
        ("tige haute T", "tige naine t", "TT", "tt", "100 % [T] hétérozygotes Tt", "3:1 (75 % haute, 25 % naine)"),
        ("pelage noir B", "pelage blanc b", "BB", "bb", "100 % [B] hétérozygotes Bb", "3/4 noir et 1/4 blanc"),
        ("grain jaune J", "grain vert j", "JJ", "jj", "100 % [J] hétérozygotes Jj", "3/4 jaune et 1/4 vert"),
    ]:
        bio_pairs.append((
            f"En génétique, on croise deux lignées pures {p1} ({dom_name} dominant) x {p2} ({rec_name} récessif). Que donne la F1 puis la F2 (F1 x F1) ?",
            f"### Croisement de Mendel (SautiLM-Jua-STEM)\n\n"
            f"1. **Première génération (F1) :** Tous les hybrides sont **{f1_desc}** (uniformité de la F1).\n"
            f"2. **Deuxième génération (F2 = F1 x F1) :** On obtient les proportions phénotypiques **{f2_ratio}**.",
        ))

    extra_bio = [
        ("QCM Écologie : Dans une chaîne alimentaire, quel rôle jouent les plantes vertes ? A) Consommateurs primaires  B) Producteurs autotrophes  C) Décomposeurs  D) Prédateurs.", "**Bonne réponse : B) Producteurs autotrophes** : par la photosynthèse, les végétaux produisent la matière organique initiale de l'écosystème."),
        ("QCM Écologie : Quel est le rôle des bactéries décomposeuses et champignons dans le sol ? A) Fixer le O2  B) Minéraliser la matière organique morte  C) Produire le glucose  D) Casser les roches seules.", "**Bonne réponse : B) Minéraliser la matière organique morte** : ils recyclent les débris en sels minéraux assimilables par les plantes."),
        ("QCM Biologie : Où se déroule la fécondation humaine normale ? A) Dans l'estomac  B) Dans le tiers supérieur de la trompe de Fallope  C) Dans l'ovaire  D) Dans le foie.", "**Bonne réponse : B) Dans la trompe de Fallope** : rencontre entre le spermatozoïde (`n`) et l'ovule (`n`) formant la cellule-œuf diploïde (`2n`)."),
        ("QCM Biologie : Quelle structure du rein assure la filtration du sang et la formation de l'urine ? A) Le néphron  B) L'alvéole  C) Le neurone  D) La villosité.", "**Bonne réponse : A) Le néphron** : unité structurale et fonctionnelle du rein assurant filtration glomérulaire et réabsorption."),
        ("QCM Biologie : Au niveau des alvéoles pulmonaires, quels échanges gazeux ont lieu avec le sang ? A) L'O2 passe du sang vers l'air  B) L'O2 passe vers le sang et le CO2 passe du sang vers l'alvéole  C) L'azote remplace le sang  D) Aucun échange.", "**Bonne réponse : B)** : par diffusion selon le gradient de pression partielle, le sang s'enrichit en `O2` et élimine le `CO2`."),
        ("QCM Biologie : Quel vecteur transmet le paludisme (malaria) ? A) La mouche tsé-tsé  B) L'anophèle femelle  C) Le pou  D) L'escargot d'eau.", "**Bonne réponse : B) L'anophèle femelle** infectée par le parasite *Plasmodium*."),
        ("Explique la différence entre artères et veines dans l'appareil circulatoire humain.", "1. **Artères :** Vaisseaux à paroi épaisse et élastique qui conduisent le sang **du cœur vers les organes**.\n2. **Veines :** Vaisseaux munis de valvules qui ramènent le sang **des organes vers le cœur**."),
        ("QCM Biologie : Quel glucide sert de réserve énergétique rapide chez les animaux (foie et muscles) ? A) Cellulose  B) Glycogène  C) Chitine  D) Insuline.", "**Bonne réponse : B) Glycogène** : polymère de glucose stocké chez l'animal (l'amidon joue ce rôle chez les végétaux)."),
        ("QCM Biologie : Que signifie l'osmose à travers une membrane semi-perméable ? A) Passage d'eau du milieu le moins concentré (hypotonique) vers le plus concentré (hypertonique)  B) Transport actif avec ATP  C) Division de l'ADN  D) Synthèse d'acides aminés.", "**Bonne réponse : A)** : l'eau diffuse passivement pour équilibrer les concentrations."),
        ("Quel est le rôle des vaccins dans le système immunitaire ?", "Un **vaccin** introduit un antigène atténué ou inactivé afin d'induire la formation de **cellules mémoire** (lymphocytes B/T) sans provoquer la maladie, assurant une réponse rapide lors d'un contact futur."),
        ("QCM Biologie : Quelles bases azotées remplacent la thymine (T) dans l'ARN ? A) L'adénine (A)  B) L'uracile (U)  C) La cytosine (C)  D) La guanine (G).", "**Bonne réponse : B) L'uracile (U)** : dans l'ARN, `A` s'apparie avec `U` au lieu de `T`."),
        ("Pourquoi l'enzyme amylase salivaire perd-elle son activité dans l'estomac ?", "Dans l'estomac, le **pH très acide** (`pH ~ 1,5 à 2`) dénature la structure tridimensionnelle de l'amylase salivaire, dont le pH optimal est neutre (`pH ~ 7`)."),
        ("QCM Biologie : Combien de chromosomes possède une cellule somatique humaine normale ? A) 23  B) 46  C) 44  D) 92.", "**Bonne réponse : B) 46** (23 paires, `2n = 46`) contre `n = 23` chromosomes dans les gamètes."),
        ("Explique le principe de la sélection naturelle selon Charles Darwin.", "Les individus porteurs de caractères avantageux adaptés à leur environnement ont une meilleure survie et descendance, augmentant la fréquence de ces allèles au fil des générations."),
        ("QCM Biologie : Quel sucre simple constitue l'unité de base de l'amidon et de la cellulose ? A) Saccharose  B) Glucose  C) Acide aminé  D) Acide gras.", "**Bonne réponse : B) Glucose**."),
        ("Quelle est la fonction des stomates sur la face inférieure des feuilles ?", "Les **stomates** assurent les **échanges gazeux** (`entrée de CO2`, sortie d'`O2`) nécessaires à la photosynthèse et régulent la **transpiration foliaire** (`H2O`)."),
        ("QCM Biologie : Quel type d'organisme est dépourvu de noyau délimité par une membrane ? A) Cellule animale  B) Cellule végétale  C) Procaryote (bactérie)  D) Levure.", "**Bonne réponse : C) Procaryote (bactérie)** : son ADN baigne directement dans le cytoplasme (nucléoïde)."),
        ("Cite deux différences structurelles entre une cellule végétale et une cellule animale.", "1. La **cellule végétale** possède une **paroi cellulosique** rigide, des **chloroplastes** et une grande vacuole centrale.\n2. La **cellule animale** est souple (sans paroi) et possède des centrosomes avec centrioles."),
        ("QCM Biologie : Quel groupe sanguin est considéré comme receveur universel en transfusion de globules rouges ? A) O-  B) A+  C) B-  D) AB+.", "**Bonne réponse : D) AB+** : il possède les antigènes A, B et Rhésus D et ne fabrique pas d'anticorps anti-A/B/D."),
        ("Quel est l'effet d'une carence prolongée en iode dans l'alimentation humaine ?", "L'iode est indispensable à la synthèse des **hormones thyroïdiennes** ; une carence entraîne un **goitre** (hypertrophie de la thyroïde) et des troubles métaboliques."),
        ("QCM Biologie : Quelle hormone végétale favorise l'élongation cellulaire et le phototropisme ? A) Adrénaline  B) Auxine  C) Insuline  D) Testostérone.", "**Bonne réponse : B) Auxine**."),
        ("Comment l'azote atmosphérique N2 devient-il assimilable par les racines des légumineuses (haricot, arachide) ?", "Des **bactéries symbiotiques (*Rhizobium*)** logées dans les nodosités des racines fixent `N2` et le réduisent en ammoniac (`NH3` / nitrates) assimilable par la plante."),
        ("QCM Biologie : Quel terme désigne l'ensemble des gènes et allèles portés par un individu ? A) Phénotype  B) Génotype  C) Caryotype  D) Écosystème.", "**Bonne réponse : B) Génotype** (tandis que le phénotype représente les caractères observables)."),
        ("Pourquoi la biodiversité est-elle importante pour la stabilité d'un écosystème agricole ?", "Une plus grande diversité spécifique limite la propagation rapide des ravageurs, favorise la pollinisation, enrichit le sol et améliore la résilience climatique."),
        ("QCM Biologie : Lors de la réplication de l'ADN, comment qualifie-t-on le mécanisme produisant deux molécules conservant chacune un brin parental et un brin néoformé ? A) Conservatif  B) Semi-conservatif  C) Dispersif  D) Transcrit.", "**Bonne réponse : B) Semi-conservatif**."),
    ]
    for q, a in extra_bio:
        bio_pairs.append((q, f"### Biologie / SVT (SautiLM-Jua-STEM)\n\n{a}"))

    assert len(bio_pairs) == 45, f"Expected 45 French Bio pairs, got {len(bio_pairs)}"
    pairs.extend(bio_pairs)
    assert len(pairs) == 180, f"Expected 180 French STEM pairs, got {len(pairs)}"
    return pairs


def _english_stem_pairs() -> List[tuple[str, str]]:
    """Builds 120 diverse English Secondary STEM pairs (30 Math, 30 Physics, 30 Chem, 30 Bio) for 60/40 FR/EN."""
    pairs: List[tuple[str, str]] = [
        # --- MATHEMATICS (30 pairs) ---
        (
            "STEM Math MCQ: What is the derivative of f(x) = sin(x)? A) -cos(x)  B) cos(x)  C) tan(x)  D) -sin(x).",
            "### Explanatory Math MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) cos(x)**\n\n"
            "1. **Rule:** `d/dx[sin(x)] = cos(x)` whereas `d/dx[cos(x)] = -sin(x)`.",
        ),
        (
            "STEM Math MCQ: Evaluate sin^2(theta) + cos^2(theta) for any real angle theta. A) 0  B) 1  C) 2  D) -1.",
            "### Explanatory Trigonometry MCQ (SautiLM-Jua-STEM)\n\n"
            "**Correct Answer: B) 1**\n\n"
            "1. **Identity:** By Pythagoras on the unit circle, `cos^2(theta) + sin^2(theta) = 1`.",
        ),
        (
            "Find the integral of x^2 dx step by step.",
            "### Integration Power Rule (SautiLM-Jua-STEM)\n\n"
            "1. **Rule:** `int x^n dx = x^(n+1) / (n + 1) + C` (`n != -1`).\n\n"
            "**Final Answer:** `(x^3) / 3 + C`.",
        ),
        (
            "Evaluate step by step: (24 * 15) + 140.",
            "### Step-by-Step Arithmetic (SautiLM-Jua-STEM)\n\n"
            "1. **Multiply First:** `24 * 15 = 360`.\n2. **Add:** `360 + 140 = 500`.\n\n**Final Answer:** **500**.",
        ),
    ]
    for quad_args in [(1, 3, 4), (1, 2, 5), (1, -1, 6), (2, 1, 3), (1, 4, 7), (1, -3, 2), (2, -1, 5), (1, 5, 6), (3, 1, 2), (1, -4, 4)]:
        pairs.append(_quadratic_en(*quad_args))
    for a, b, c in [(5, 12, 13), (3, 4, 5), (6, 8, 10), (8, 15, 17), (7, 24, 25), (9, 12, 15)]:
        pairs.append((
            f"A right triangle has legs a = {a} cm and b = {b} cm. Find the hypotenuse c.",
            f"### Pythagoras Theorem (SautiLM-Jua-STEM)\n\n"
            f"1. **Formula:** `c^2 = a^2 + b^2 = {a}^2 + {b}^2 = {a * a} + {b * b} = {c * c}`.\n"
            f"2. **Square Root:** `c = sqrt({c * c}) = {c} cm`.\n\n**Final Answer:** `c = {c} cm`.",
        ))
    for a, b, c in [(2, -3, 7), (4, 5, -2), (3, -8, 1), (5, 2, -6), (1, -9, 4), (6, -4, 3), (2, 7, -5), (4, -1, 8), (3, 6, -4), (5, -5, 2)]:
        pairs.append((
            f"Differentiate f(x) = {a}x^3 + ({b})x^2 + ({c})x with respect to x.",
            f"### Polynomial Derivative (SautiLM-Jua-STEM)\n\n"
            f"1. **Power Rule (`d/dx[x^n] = n*x^(n-1)`):** `f'(x) = {3 * a}x^2 + ({2 * b})x + ({c})`.\n\n"
            f"**Final Answer:** `f'(x) = {3 * a}x^2 + ({2 * b})x + ({c})`.",
        ))

    # --- PHYSICS (30 pairs; held out test probe: 12 ohms, 2 A!) ---
    for r_val, i_val in [(4, 3), (6, 2), (5, 4), (10, 3), (8, 2), (15, 3), (20, 2), (9, 3), (7, 4), (25, 2)]:
        v_val = r_val * i_val
        p_val = v_val * i_val
        pairs.append((
            f"A {r_val} ohm resistor carries a current of {i_val} A. Compute the voltage V and electrical power P.",
            f"### Ohm's Law and Electrical Power (SautiLM-Jua-STEM)\n\n"
            f"1. **Voltage (`V = I * R`):** `V = {i_val} A * {r_val} ohms = {v_val} V`.\n"
            f"2. **Power (`P = V * I`):** `P = {v_val} V * {i_val} A = {p_val} W`.\n\n"
            f"**Final Answer:** `V = {v_val} V` and `P = {p_val} W`.",
        ))
    for m_val, v_val in [(6, 4), (4, 5), (2, 10), (8, 5), (10, 3), (12, 2), (5, 6), (3, 8)]:
        ke_val = 0.5 * m_val * (v_val ** 2)
        pairs.append((
            f"Calculate the kinetic energy of a {m_val} kg object moving at {v_val} m/s.",
            f"### Kinetic Energy Calculation (SautiLM-Jua-STEM)\n\n"
            f"1. **Formula:** `KE = 0.5 * m * v^2 = 0.5 * {m_val} * {v_val}^2 = {_fmt_num(ke_val)} Joules`.\n\n"
            f"**Final Answer:** `KE = {_fmt_num(ke_val)} J`.",
        ))
    for m_val, a_val in [(5, 4), (8, 3), (10, 2), (4, 7), (6, 5), (12, 2), (9, 3), (15, 4)]:
        pairs.append((
            f"What net force F accelerates a {m_val} kg mass at {a_val} m/s^2?",
            f"### Newton's Second Law (SautiLM-Jua-STEM)\n\n"
            f"1. **Formula (`F = m * a`):** `F = {m_val} kg * {a_val} m/s^2 = {m_val * a_val} N`.\n\n"
            f"**Final Answer:** `F = {m_val * a_val} N`.",
        ))
    pairs.extend([
        (
            "STEM Physics MCQ: Two resistors R1 = 4 Ohm and R2 = 6 Ohm are connected in series. What is the total resistance? A) 2.4 Ohm  B) 10 Ohm  C) 24 Ohm  D) 5 Ohm.",
            "### Explanatory Physics MCQ (SautiLM-Jua-STEM)\n\n**Correct Answer: B) 10 Ohm**\n\n"
            "1. **Series Rule:** `R_total = R1 + R2 = 4 + 6 = 10 Ohm`.",
        ),
        (
            "How do photovoltaic solar cells convert sunlight into electric current?",
            "### Solar PV Working Principle (SautiLM-Jua-STEM)\n\n"
            "1. Photons strike doped **silicon** semiconductor junctions, liberating free electrons that flow as **direct current (DC)**.",
        ),
        (
            "STEM Physics MCQ: What is the SI unit of electrical resistance? A) Ampere  B) Volt  C) Ohm  D) Watt.",
            "### Explanatory Physics MCQ (SautiLM-Jua-STEM)\n\n**Correct Answer: C) Ohm**\n\n"
            "1. Voltage is measured in Volts, current in Amperes, power in Watts, and resistance in **Ohms**.",
        ),
        (
            "A wave has wavelength 0.5 m and frequency 680 Hz. Calculate its wave speed v.",
            "### Wave Speed Equation (SautiLM-Jua-STEM)\n\n"
            "1. **Formula (`v = f * lambda`):** `v = 680 * 0.5 = 340 m/s`.\n\n**Final Answer:** `340 m/s`.",
        ),
    ])

    # --- CHEMISTRY (30 pairs; held out: pH = 3 MCQ!) ---
    for subst, molar_mass, mass in [
        ("CO2", 44, 44), ("CO2", 44, 88), ("CO2", 44, 132), ("H2O", 18, 18), ("H2O", 18, 54),
        ("H2O", 18, 90), ("O2", 32, 64), ("O2", 32, 32), ("NaOH", 40, 80), ("CaCO3", 100, 150),
    ]:
        pairs.append((
            f"Calculate the number of moles n in {mass} g of {subst} (M = {molar_mass} g/mol).",
            f"### Mole Calculation (SautiLM-Jua-STEM)\n\n"
            f"1. **Formula (`n = mass / M`):** `n = {mass} g / {molar_mass} g/mol = {_fmt_num(mass / molar_mass)} mol`.\n\n"
            f"**Final Answer:** **{_fmt_num(mass / molar_mass)} mol**.",
        ))
    for n_val, v_ml in [(0.4, 500), (0.5, 250), (1.2, 600), (0.8, 400), (2.0, 500), (0.25, 250)]:
        conc = n_val / (v_ml / 1000.0)
        pairs.append((
            f"Calculate molarity M when {n_val} mol of solute dissolves in {v_ml} mL of solution.",
            f"### Molarity Step-by-Step (SautiLM-Jua-STEM)\n\n"
            f"1. **Volume:** `{v_ml} mL = {_fmt_num(v_ml / 1000.0)} L`.\n"
            f"2. **Molarity (`M = n / V`):** `M = {n_val} / {_fmt_num(v_ml / 1000.0)} = {_fmt_num(conc)} mol/L`.\n\n"
            f"**Final Answer:** `{_fmt_num(conc)} M`.",
        ))
    for ph_val, option, expl in [
        (1, "C) Acidic", "Because `pH = 1 < 7`, the solution has a high `[H+]` concentration and is strongly acidic."),
        (2, "C) Acidic", "Solutions with `pH < 7` have excess hydrogen ions (`H+`) and are **acidic**."),
        (4, "B) Acidic", "Since `4 < 7`, the aqueous solution lies in the acidic region."),
        (6, "A) Weakly acidic", "A `pH` just below `7` (`pH = 6`) indicates a weakly acidic solution."),
        (7, "B) Neutral", "Pure water at 25 C has `pH = 7`, meaning equal `[H+]` and `[OH-]` concentrations (**neutral**)."),
        (8, "D) Weakly basic", "`pH = 8 > 7` corresponds to a weakly basic (alkaline) solution."),
        (11, "C) Basic / Alkaline", "With `pH = 11 > 7`, hydroxide ions (`OH-`) dominate so the solution is basic."),
        (13, "C) Strongly basic", "`pH = 13` is far above `7`, typical of a strong alkali like NaOH."),
    ]:
        pairs.append((
            f"STEM Chemistry MCQ: An aqueous solution has pH = {ph_val} at 25 C. How is it classified? A) Weakly acidic  B) Neutral  C) Acidic / Basic  D) Salt-free.",
            f"### Explanatory Chemistry MCQ (SautiLM-Jua-STEM)\n\n"
            f"**Correct Answer: {option}**\n\n"
            f"1. **Reason:** {expl}",
        ))
    pairs.extend([
        (
            "Balance the chemical equation for water formation: H2 + O2 -> H2O.",
            "### Balancing Chemical Equations (SautiLM-Jua-STEM)\n\n"
            "1. Oxygen requires `2 H2O`, so hydrogen needs `2 H2`.\n\n**Balanced Equation:** `2 H2 + O2 -> 2 H2O`.",
        ),
        (
            "Balance the combustion of propane: C3H8 + O2 -> CO2 + H2O.",
            "### Balancing Chemical Equations (SautiLM-Jua-STEM)\n\n"
            "1. `3 C` gives `3 CO2`; `8 H` gives `4 H2O` -> total `6 + 4 = 10 O` -> `5 O2`.\n\n**Balanced Equation:** `C3H8 + 5 O2 -> 3 CO2 + 4 H2O`.",
        ),
        (
            "STEM Chemistry MCQ: Oxidation is defined as: A) Gain of electrons  B) Loss of electrons  C) Gain of mass  D) Cooling.",
            "### Explanatory Chemistry MCQ (SautiLM-Jua-STEM)\n\n**Correct Answer: B) Loss of electrons**\n\n1. Oxidation increases oxidation state by **losing electrons**.",
        ),
        (
            "An atom has atomic number Z = 17 and mass number A = 35 (Chlorine). State protons, electrons, and neutrons.",
            "### Atomic Structure (SautiLM-Jua-STEM)\n\n1. **Protons & electrons:** `Z = 17` each.\n2. **Neutrons (`A - Z`):** `35 - 17 = 18 neutrons`.",
        ),
        (
            "What is the difference between an ionic bond and a covalent bond?",
            "### Chemical Bonding (SautiLM-Jua-STEM)\n\n1. **Ionic bond:** Complete electron transfer between a metal and non-metal (`NaCl`).\n2. **Covalent bond:** Shared electron pair between non-metals (`H2O`, `CO2`).",
        ),
        (
            "Find the molar mass of calcium carbonate CaCO3 (Ca=40, C=12, O=16 g/mol).",
            "### Molar Mass (SautiLM-Jua-STEM)\n\n1. `M = 40 + 12 + 3(16) = 100 g/mol`.\n\n**Final Answer:** **100 g/mol**.",
        ),
    ])

    # --- BIOLOGY (30 pairs) ---
    en_bio_list = [
        (
            "STEM Biology MCQ: Which organelle and pigment in green leaves capture sunlight for photosynthesis? A) Mitochondria and hemoglobin  B) Chloroplasts and chlorophyll  C) Ribosomes and keratin  D) Lysosomes and melanin.",
            "**Correct Answer: B) Chloroplasts and chlorophyll**\n\n1. **Reason:** **Chlorophyll** inside **chloroplasts** absorbs light energy to convert `CO2` and water (`H2O`) into **glucose** (`C6H12O6`) and **oxygen (`O2`)**.",
        ),
        (
            "Write the balanced word and chemical equation of photosynthesis and explain chlorophyll's role.",
            "1. **Chlorophyll's Role:** Green pigment in **chloroplasts** that captures light energy.\n2. **Balanced Equation:** `6 CO2 + 6 H2O --(light, chlorophyll)--> C6H12O6 (glucose) + 6 O2 (oxygen)`.",
        ),
        (
            "Why do plants release oxygen gas (O2) during daytime photosynthesis?",
            "During light-dependent reactions in **chloroplasts**, water molecules (`H2O`) are split using solar energy captured by **chlorophyll**, producing glucose and releasing **oxygen (`O2`)** as a byproduct.",
        ),
        (
            "STEM Biology MCQ: Which human heart chamber pumps oxygen-rich blood into the aorta? A) Right atrium  B) Right ventricle  C) Left atrium  D) Left ventricle.",
            "**Correct Answer: D) Left ventricle**\n\n1. The muscular **left ventricle** generates high pressure to pump oxygenated blood through the aorta to the whole body.",
        ),
        (
            "Explain the key difference between mitosis and meiosis.",
            "1. **Mitosis:** Produces **2 genetically identical diploid (`2n`) daughter cells** for growth and tissue repair.\n2. **Meiosis:** Produces **4 genetically distinct haploid (`n`) gametes** for sexual reproduction.",
        ),
        (
            "STEM Biology MCQ: Which type of cell division keeps the chromosome number constant (2n -> 2n)? A) Meiosis  B) Mitosis  C) Fertilization  D) Budding.",
            "**Correct Answer: B) Mitosis**\n\n1. Mitosis replicates DNA once and divides once, preserving `2n` chromosomes in somatic daughter cells.",
        ),
        (
            "Explain the three stages of the water cycle (evaporation, condensation, precipitation).",
            "1. **Evaporation:** Solar heat turns surface liquid water into vapor.\n2. **Condensation:** Rising vapor cools into cloud droplets.\n3. **Precipitation:** Heavy droplets fall as rain, replenishing rivers and soil moisture.",
        ),
        (
            "What is the function of mitochondria in animal and plant cells?",
            "Mitochondria are the site of **aerobic cellular respiration**, breaking down glucose with oxygen (`O2`) to synthesize **ATP** energy.",
        ),
        (
            "STEM Biology MCQ: In DNA base pairing, Adenine (A) always pairs with which base? A) Cytosine  B) Guanine  C) Thymine  D) Uracil.",
            "**Correct Answer: C) Thymine**\n\n1. DNA pairs `A` with `T` and `G` with `C` (in RNA, Uracil `U` replaces Thymine).",
        ),
        (
            "Find the complementary DNA strand for 5'-A-C-G-T-T-A-3'.",
            "Applying `A <-> T` and `C <-> G` complementarity yields **`3'-T-G-C-A-A-T-5'`**.",
        ),
        (
            "Explain Mendelian monohybrid cross F2 ratio for Rr x Rr (R dominant over r).",
            "1. Punnett square yields `RR` (25%), `Rr` (50%), `rr` (25%).\n2. **Phenotypic ratio:** **3:1** (75% dominant `[R]` : 25% recessive `[r]`).",
        ),
        (
            "STEM Biology MCQ: Which blood cells carry oxygen from the lungs to tissues? A) Platelets  B) White blood cells  C) Red blood cells (erythrocytes)  D) Plasma proteins.",
            "**Correct Answer: C) Red blood cells (erythrocytes)** containing iron-rich hemoglobin.",
        ),
        (
            "What role does messenger RNA (mRNA) play in protein synthesis?",
            "**mRNA** copies a gene sequence from nuclear DNA during **transcription** and carries it to cytoplasmic **ribosomes** for **translation** into proteins.",
        ),
        (
            "STEM Biology MCQ: Which human organ secretes insulin to lower blood glucose? A) Kidney  B) Liver  C) Pancreas  D) Spleen.",
            "**Correct Answer: C) Pancreas** (beta cells release insulin after meals).",
        ),
        (
            "List the pathway of air from the nostrils to the alveoli during human inhalation.",
            "Nostrils -> Nasal cavity -> Pharynx -> Larynx -> Trachea -> Bronchi -> Bronchioles -> **Alveoli** (where `O2`/`CO2` gas exchange occurs).",
        ),
        (
            "STEM Biology MCQ: What is the main structural difference between prokaryotic and eukaryotic cells? A) Ribosomes  B) Presence of a membrane-bound nucleus  C) Cell membrane  D) Cytoplasm.",
            "**Correct Answer: B) Presence of a membrane-bound nucleus** (present in eukaryotes, absent in bacteria/prokaryotes).",
        ),
        (
            "How do *Rhizobium* bacteria in legume root nodules improve soil fertility?",
            "They fix atmospheric nitrogen (`N2`) into usable ammonia/nitrates for legumes (beans, groundnuts, soy), naturally enriching agricultural soil.",
        ),
        (
            "STEM Biology MCQ: What vector transmits malaria (*Plasmodium*) to humans? A) Tsetse fly  B) Female *Anopheles* mosquito  C) Housefly  D) Tick.",
            "**Correct Answer: B) Female *Anopheles* mosquito**.",
        ),
        (
            "Define osmosis in plant and animal cells.",
            "**Osmosis** is the passive diffusion of water molecules across a selectively permeable membrane from a region of lower solute concentration (hypotonic) to higher solute concentration (hypertonic).",
        ),
        (
            "STEM Biology MCQ: Which process in ecosystems converts dead organic matter back into soil mineral nutrients? A) Predation  B) Decomposition  C) Transpiration  D) Osmosis.",
            "**Correct Answer: B) Decomposition** by saprotrophic bacteria and fungi.",
        ),
        (
            "List the four stages of mitosis in chronological order.",
            "1. **Prophase** (chromosomes condense)\n2. **Metaphase** (align at equator)\n3. **Anaphase** (sister chromatids separate)\n4. **Telophase** (new nuclei form).",
        ),
        (
            "STEM Biology MCQ: Which human vessel carries blood away from the heart? A) Vein  B) Artery  C) Capillary  D) Venule.",
            "**Correct Answer: B) Artery** (thick-walled vessels carrying blood away from ventricles).",
        ),
        (
            "Explain why high fever or extreme pH denatures digestive enzymes.",
            "Enzymes are proteins whose active site shape depends on weak hydrogen/ionic bonds; extreme heat or pH unfolds the protein (**denaturation**), preventing substrate binding.",
        ),
        (
            "STEM Biology MCQ: How many chromosomes are present in a normal human gamete (sperm or egg)? A) 46  B) 23  C) 44  D) 92.",
            "**Correct Answer: B) 23** (haploid `n = 23`), formed by meiosis so fertilization restores `2n = 46`.",
        ),
        (
            "Contrast aerobic respiration with anaerobic fermentation in muscle cells.",
            "1. **Aerobic respiration (`with O2`):** High ATP yield (~36-38 ATP per glucose), products `CO2 + H2O`.\n2. **Anaerobic lactic fermentation (`low O2`):** Only 2 ATP, producing lactic acid.",
        ),
        (
            "STEM Biology MCQ: Which plant tissue transports water and minerals upward from roots to leaves? A) Phloem  B) Xylem  C) Cortex  D) Epidermis.",
            "**Correct Answer: B) Xylem** (while phloem transports dissolved sugars).",
        ),
        (
            "What is the role of vaccination in adaptive immunity?",
            "A vaccine exposes the immune system to harmless antigens, generating **memory B and T cells** that mount a rapid protective response against future infection.",
        ),
        (
            "STEM Biology MCQ: What structural organelle gives plant cells rigidity and prevents bursting in pure water? A) Mitochondrion  B) Cellulose cell wall  C) Golgi apparatus  D) Centriole.",
            "**Correct Answer: B) Cellulose cell wall**.",
        ),
        (
            "Explain natural selection in Darwin's theory of evolution.",
            "Individuals with heritable traits well suited to local environmental pressures survive and reproduce more successfully, shifting population allele frequencies over generations.",
        ),
        (
            "STEM Biology MCQ: Which hormone triggers root growth and phototropism bending in plants? A) Insulin  B) Auxin  C) Thyroxine  D) Hemoglobin.",
            "**Correct Answer: B) Auxin**.",
        ),
    ]
    for q, a in en_bio_list:
        pairs.append((q, f"### Secondary Biology (SautiLM-Jua-STEM)\n\n{a}"))

    assert len(pairs) == 120, f"Expected 120 English STEM pairs, got {len(pairs)}"
    return pairs


def generate_stem_instructions() -> List[Dict[str, Any]]:
    """
    Generates SautiLM-Jua-STEM corpus (Secondary School & Examen d'État RDC).
    Strict 60% French / 40% English balance across 305 unique diverse items
    (180 FR STEM + 3 FR greetings vs 120 EN STEM + 2 EN greetings) x 2 prompt styles = 610 rows,
    preventing overfitting/memorization collapse on unseen secondary problems.
    """
    out: List[Dict[str, Any]] = []
    fr_pairs = _french_stem_pairs()
    en_pairs = _english_stem_pairs()

    # Exact 60/40 ratio: 183 FR vs 122 EN (x2: with and without system prompt -> 366 FR / 244 EN = 60.0% / 40.0%)
    for q, a in fr_pairs:
        out.append(_stem_turn("fr", STEM_SYSTEM_FR, q, a))
        out.append(_stem_turn("fr", None, q, a))

    for q, a in en_pairs:
        out.append(_stem_turn("en", STEM_SYSTEM_EN, q, a))
        out.append(_stem_turn("en", None, q, a))

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
        out.append(_stem_turn(lang, sys_p, q, a))
        out.append(_stem_turn(lang, None, q, a))

    return out
