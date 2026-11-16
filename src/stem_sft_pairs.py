from __future__ import annotations

from typing import Any, Dict, List


STEM_SYSTEM_FR = (
    "Tu es SautiLM-Jua-STEM, assistant hors-ligne de mathématiques, physique, chimie et biologie "
    "(niveau Secondaire et Examen d'État RDC) créé par LosingTek Research. "
    "Réponds directement, sans bavardage, en suivant un format structuré (résolution étape par étape ou QCM explicatif avec analyse avant la réponse finale). "
    "Ne mentionne aucune langue étrangère non demandée ni de tiret long."
)

STEM_SYSTEM_EN = (
    "You are SautiLM-Jua-STEM, an offline secondary school Mathematics, Physics, Chemistry, and Biology "
    "assistant by LosingTek Research. Reason briefly first using clear formulas and revision steps, "
    "then state the final answer directly. No emojis, no long dashes."
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
    ans = f"S = {{{lo}}}" if lo == hi else f"S = {{{lo} ; {hi}}}"
    return (
        f"Résous dans R l'équation du second degré : {eq} (Examen d'État).",
        f"### Équation du 2nd Degré (SautiLM-Jua-STEM)\n\n"
        f"1. **Identification :** a = {a}, b = {b}, c = {c}.\n"
        f"2. **Discriminant :** Delta = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {b * b} - ({4 * a * c}) = {delta} (racine(Delta) = {sqrt_d}).\n"
        f"3. **Racines :** x1 = ({-b} - {sqrt_d}) / {2 * a} = {lo} et x2 = ({-b} + {sqrt_d}) / {2 * a} = {hi}.\n\n"
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
        f"1. **Coefficients:** a = {a}, b = {b}, c = {c}.\n"
        f"2. **Discriminant:** Delta = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {delta} (sqrt(Delta) = {sqrt_d}).\n"
        f"3. **Roots:** x1 = ({-b} - {sqrt_d}) / {2 * a} = {lo} and x2 = ({-b} + {sqrt_d}) / {2 * a} = {hi}.\n\n"
        f"**Final Answer:** x = {lo} and x = {hi}.",
    )


def _mcq_fr(topic: str, analysis: str, answer: str) -> str:
    """Builds a French MCQ response placing step-by-step rationale before the final option choice."""
    return (
        f"### QCM Explicatif : {topic} (SautiLM-Jua-STEM)\n\n"
        f"1. **Analyse :** {analysis}\n\n"
        f"**Bonne réponse : {answer}**"
    )


def _mcq_en(topic: str, analysis: str, answer: str) -> str:
    """Builds an English MCQ response placing concise rationale before the final option choice."""
    return (
        f"### Explanatory {topic} MCQ (SautiLM-Jua-STEM)\n\n"
        f"1. **Analysis:** {analysis}\n\n"
        f"**Correct Answer: {answer}**"
    )


def _french_stem_pairs() -> List[tuple[str, str]]:
    """Builds 195 diverse French Secondary / Examen d'État STEM problems (54 Math, 47 Physics, 47 Chem, 47 Bio)."""
    math_pairs: List[tuple[str, str]] = [
        _quadratic_fr(2, -1, 3),  # 2x^2 - 4x - 6 = 0 unified with generator!
        (
            "QCM Maths : Pour l'équation x^2 - 5x + 6 = 0, quelles sont les solutions ? A) {-2; -3}  B) {2; 3}  C) {1; 6}  D) {-1; 6}.",
            _mcq_fr(
                "Mathématiques",
                "a = 1, b = -5, c = 6, donc Delta = (-5)^2 - 4(1)(6) = 25 - 24 = 1 ; les racines sont x1 = (5 - 1)/2 = 2 et x2 = (5 + 1)/2 = 3 (somme S = 5, produit P = 6).",
                "B) {2 ; 3}",
            ),
        ),
        (
            "QCM Maths : Résous dans R : x^2 - 6x + 8 = 0. A) {1; 8}  B) {-2; -4}  C) {2; 4}  D) {3; 5}.",
            _mcq_fr(
                "Équation du 2nd Degré",
                "Delta = (-6)^2 - 4(1)(8) = 36 - 32 = 4 (racine = 2), donc x1 = (6 - 2)/2 = 2 et x2 = (6 + 2)/2 = 4.",
                "C) {2 ; 4}",
            ),
        ),
        (
            "Calcule l'intégrale de sinus x divisé par cosinus x dx étape par étape.",
            "### Calcul Intégral par Changement de Variable (SautiLM-Jua-STEM)\n\n"
            "1. **Réécriture :** Intégrale de sin(x) / cos(x) dx = Intégrale de tan(x) dx.\n"
            "2. **Changement de variable :** Posons u = cos(x), alors du = -sin(x) dx, d'où sin(x) dx = -du.\n"
            "3. **Intégration :** - Intégrale de (1 / u) du = -ln|u| + C = -ln|cos(x)| + C.\n\n"
            "**Réponse finale :** -ln|cos(x)| + C.",
        ),
        (
            "Calcule l'intégrale de sin(x) / cos(x) dx par changement de variable.",
            "### Calcul Intégral Trigonométrique (SautiLM-Jua-STEM)\n\n"
            "1. **Changement de variable :** Posons u = cos(x), donc du = -sin(x) dx.\n"
            "2. **Primitive :** - Intégrale de (1 / u) du = -ln|cos(x)| + C.\n\n"
            "**Réponse finale :** -ln|cos(x)| + C.",
        ),
        (
            "Calcule l'intégrale de tan(x) dx en l'écrivant sous forme sin(x) / cos(x).",
            "### Intégrale de la Tangente (SautiLM-Jua-STEM)\n\n"
            "1. **Identité :** tan(x) = sin(x) / cos(x).\n"
            "2. **Substitution u = cos(x) :** du = -sin(x) dx, donc - Intégrale de du / u = -ln|cos(x)| + C.\n\n"
            "**Réponse finale :** -ln|cos(x)| + C (ou ln|sec(x)| + C).",
        ),
        (
            "Calcule l'intégrale de cos(x) / sin(x) dx étape par étape.",
            "### Calcul Intégral par Substitution (SautiLM-Jua-STEM)\n\n"
            "1. **Changement de variable :** Posons u = sin(x), d'où du = cos(x) dx.\n"
            "2. **Intégration :** Intégrale de (1 / u) du = ln|u| + C = ln|sin(x)| + C.\n\n"
            "**Réponse finale :** ln|sin(x)| + C.",
        ),
        (
            "Calcule l'intégrale de cosinus x divisé par sinus x dx.",
            "### Intégrale Cotangente (SautiLM-Jua-STEM)\n\n"
            "1. On écrit cos(x) / sin(x) et on pose u = sin(x), soit du = cos(x) dx.\n"
            "2. On intègre (1 / u) du = ln|sin(x)| + C.\n\n"
            "**Réponse finale :** ln|sin(x)| + C.",
        ),
        (
            "Calcule une primitive de f(x) = sin(2x) / cos(2x) par substitution.",
            "### Intégrale par Substitution Linéaire (SautiLM-Jua-STEM)\n\n"
            "1. Posons u = cos(2x), donc du = -2 sin(2x) dx et sin(2x) dx = -du / 2.\n"
            "2. Intégration : -(1/2) ln|u| + C = -(1/2) ln|cos(2x)| + C.\n\n"
            "**Réponse finale :** -(1/2) ln|cos(2x)| + C.",
        ),
        (
            "Calcule l'intégrale de 2x / (x^2 + 1) dx par changement de variable u = x^2 + 1.",
            "### Intégrale Logarithmique Composée (SautiLM-Jua-STEM)\n\n"
            "1. Avec u = x^2 + 1, on a du = 2x dx.\n"
            "2. Intégrale de (1 / u) du = ln|x^2 + 1| + C.\n\n"
            "**Réponse finale :** ln(x^2 + 1) + C.",
        ),
        (
            "Calcule une primitive de f(x) = cos(x) et vérifie par dérivation.",
            "### Primitives Usuelles (SautiLM-Jua-STEM)\n\n"
            "1. La primitive de cos(x) est F(x) = sin(x) + C.\n"
            "2. Vérification : F'(x) = (sin(x) + C)' = cos(x).\n\n"
            "**Réponse finale :** sin(x) + C.",
        ),
        (
            "Calcule une primitive de f(x) = sin(x).",
            "### Primitives Usuelles (SautiLM-Jua-STEM)\n\n"
            "1. Comme (cos(x))' = -sin(x), la primitive de sin(x) est F(x) = -cos(x) + C.\n\n"
            "**Réponse finale :** -cos(x) + C.",
        ),
        (
            "Calcule l'intégrale de 1 / x dx pour x > 0.",
            "### Intégrale Logarithmique (SautiLM-Jua-STEM)\n\n"
            "1. Pour x > 0, la primitive de 1/x est ln(x) + C.\n\n**Réponse finale :** ln(x) + C.",
        ),
        (
            "Calcule l'intégrale de e^(2x) dx étape par étape.",
            "### Intégrale Exponentielle (SautiLM-Jua-STEM)\n\n"
            "1. Règle : Intégrale de e^(kx) dx = (1 / k) e^(kx) + C.\n\n**Réponse finale :** (1/2) e^(2x) + C.",
        ),
        (
            "QCM Maths : Que vaut sin^2(x) + cos^2(x) pour tout réel x ? A) 0  B) 1  C) 2  D) tan(x).",
            _mcq_fr(
                "Trigonométrie",
                "Dans le cercle trigonométrique de rayon 1, le théorème de Pythagore donne cos^2(x) + sin^2(x) = 1 pour tout réel x.",
                "B) 1",
            ),
        ),
        (
            "QCM Maths : Quelle est la dérivée de ln(x) pour x > 0 ? A) 1/x  B) x  C) e^x  D) -1/x^2.",
            _mcq_fr(
                "Dérivation",
                "La fonction logarithme népérien ln(x) est dérivable sur ]0 ; +inf[ et sa dérivée vaut (ln(x))' = 1 / x.",
                "A) 1/x",
            ),
        ),
        (
            "Un sac contient 3 boules rouges et 5 boules bleues. Quelle est la probabilité de tirer une boule rouge ?",
            "### Probabilités (SautiLM-Jua-STEM)\n\n"
            "1. Cas possibles : 3 + 5 = 8 boules.\n"
            "2. Probabilité : P(Rouge) = 3 / 8 = 0,375.\n\n**Réponse finale :** 3/8 (37,5 %).",
        ),
        (
            "Calcule le nombre de permutations de 4 objets distincts (factorielle 4!).",
            "### Analyse Combinatoire (SautiLM-Jua-STEM)\n\n"
            "1. Formule : P_n = n!.\n2. Calcul : 4! = 4 * 3 * 2 * 1 = 24.\n\n**Réponse finale :** 24 permutations.",
        ),
    ]
    # Note: keep x^2 - 7x + 12 = 0 (1, 3, 4) STRICTLY HELD OUT for generalization testing!
    for quad_args in [
        (1, 1, 4), (1, -2, 5), (1, 3, 7), (1, -4, 2), (2, -1, 4), (1, 2, 6),
        (1, -3, -1), (3, 1, 2), (1, 4, 5), (2, 1, 4), (1, -1, 5), (1, 2, 7),
    ]:
        math_pairs.append(_quadratic_fr(*quad_args))
    for a, b, c, d in [(3, -5, 4, -7), (2, 3, -6, 1), (4, -2, 5, 9), (1, 4, -3, 2), (5, -1, 2, -8), (2, -7, 1, 4), (3, 2, 6, -5)]:
        math_pairs.append((
            f"Calcule la dérivée de f(x) = {a}x^3 + ({b})x^2 + ({c})x + ({d}).",
            f"### Dérivée Polynomiale (SautiLM-Jua-STEM)\n\n"
            f"1. Règle : (x^n)' = n * x^(n-1) et (constante)' = 0.\n"
            f"2. Résultat : f'(x) = {3 * a}x^2 + ({2 * b})x + ({c}).",
        ))
    for u1, r, n in [(5, 3, 10), (2, 4, 8), (10, -2, 6), (3, 5, 11), (7, 2, 15), (1, 6, 9)]:
        un = u1 + (n - 1) * r
        math_pairs.append((
            f"Une suite arithmétique a pour premier terme u1 = {u1} et pour raison r = {r}. Calcule u{n}.",
            f"### Suite Arithmétique (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : un = u1 + (n - 1) * r.\n"
            f"2. Calcul : u{n} = {u1} + ({n} - 1) * ({r}) = {un}.\n\n**Réponse finale :** u{n} = {un}.",
        ))
    for a, b, c in [(6, 8, 10), (3, 4, 5), (5, 12, 13), (8, 15, 17), (9, 12, 15), (12, 16, 20)]:
        math_pairs.append((
            f"Un triangle rectangle a pour côtés de l'angle droit a = {a} cm et b = {b} cm. Calcule l'hypoténuse c.",
            f"### Théorème de Pythagore (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : c^2 = a^2 + b^2 = {a}^2 + {b}^2 = {c * c}.\n"
            f"2. Racine : c = racine({c * c}) = {c} cm.\n\n**Réponse finale :** c = {c} cm.",
        ))
    for x, y in [(7, 3), (8, 2), (10, 4), (9, 5), (6, 1)]:
        s_val = x + y
        d_val = x - y
        math_pairs.append((
            f"Résous le système d'équations : x + y = {s_val} et x - y = {d_val}.",
            f"### Système Linéaire (SautiLM-Jua-STEM)\n\n"
            f"1. Addition : 2x = {s_val} + {d_val} = {2 * x}, donc x = {x}.\n"
            f"2. Calcul de y : y = {s_val} - {x} = {y}.\n\n**Réponse finale :** (x ; y) = ({x} ; {y}).",
        ))
    assert len(math_pairs) == 54, f"Expected 54 French Math pairs, got {len(math_pairs)}"

    # --- PHYSIQUE (47 paires; held out test: m=5 kg, a=3 m/s^2) ---
    phys_pairs: List[tuple[str, str]] = []
    for r_val, i_val in [(4, 3), (5, 2), (6, 4), (10, 2), (8, 3), (15, 2), (20, 3), (3, 5), (9, 2), (12, 4)]:
        u_val = r_val * i_val
        p_val = u_val * i_val
        phys_pairs.append((
            f"Une résistance R = {r_val} Ohms est traversée par un courant I = {i_val} A. Calcule la tension U et la puissance P.",
            f"### Loi d'Ohm et Puissance (SautiLM-Jua-STEM)\n\n"
            f"1. Tension (U = R * I) : U = {r_val} Ohms * {i_val} A = {u_val} V.\n"
            f"2. Puissance (P = U * I) : P = {u_val} V * {i_val} A = {p_val} W.\n\n"
            f"**Réponse finale :** U = {u_val} V et P = {p_val} W.",
        ))
    for m_val, a_val in [(4, 2), (5, 2), (6, 4), (10, 3), (8, 2), (12, 3), (7, 4), (15, 2), (3, 4), (9, 3)]:
        f_val = m_val * a_val
        phys_pairs.append((
            f"Un corps de masse {m_val} kg subit une accélération de {a_val} m/s^2. Calcule la force résultante.",
            f"### Deuxième Loi de Newton (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : F = m * a (ici a = {a_val} m/s^2).\n"
            f"2. Multiplication directe : F = {m_val} kg * {a_val} m/s^2 = {f_val} N.\n\n"
            f"**Réponse finale :** {f_val} Newton (N).",
        ))
    for m_val, v_val in [(4, 5), (2, 6), (6, 4), (8, 3), (10, 4), (12, 5), (4, 10)]:
        ec_val = 0.5 * m_val * (v_val ** 2)
        phys_pairs.append((
            f"Calcule l'énergie cinétique d'un solide de masse m = {m_val} kg animé d'une vitesse v = {v_val} m/s.",
            f"### Énergie Cinétique (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : Ec = (1/2) * m * v^2 = 0,5 * {m_val} * {v_val}^2 = {_fmt_num(ec_val)} Joules.\n\n**Réponse finale :** Ec = {_fmt_num(ec_val)} J.",
        ))
    for m_val, h_val in [(10, 5), (4, 8), (6, 5), (12, 10), (3, 15), (8, 4)]:
        ep_val = m_val * 10 * h_val
        phys_pairs.append((
            f"Un corps de masse m = {m_val} kg se trouve à une hauteur h = {h_val} m (g = 10 m/s^2). Calcule son énergie potentielle Ep.",
            f"### Énergie Potentielle de Pesanteur (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : Ep = m * g * h = {m_val} * 10 * {h_val} = {ep_val} Joules.\n\n**Réponse finale :** Ep = {ep_val} J.",
        ))
    for r1, r2 in [(6, 3), (12, 4), (10, 10), (8, 8), (15, 10), (9, 18)]:
        r_par = (r1 * r2) / (r1 + r2)
        phys_pairs.append((
            f"Deux résistances R1 = {r1} Ohms et R2 = {r2} Ohms sont associées en parallèle. Calcule la résistance équivalente Req.",
            f"### Association en Parallèle (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : Req = (R1 * R2) / (R1 + R2) = ({r1} * {r2}) / ({r1} + {r2}) = {_fmt_num(r_par)} Ohms.\n\n**Réponse finale :** Req = {_fmt_num(r_par)} Ohms.",
        ))
    phys_pairs.extend([
        (
            "QCM Physique : Que mesure un ampèremètre et comment se branche-t-il ? A) La tension, en parallèle  B) L'intensité du courant, en série  C) La résistance, hors circuit  D) La puissance, en parallèle.",
            _mcq_fr("Mesures Électriques", "L'ampèremètre mesure l'intensité du courant en Ampères et se branche toujours en série dans le circuit.", "B) L'intensité du courant, en série"),
        ),
        (
            "QCM Physique : Quelle est l'unité SI du travail mécanique et de l'énergie ? A) Le Watt  B) Le Newton  C) Le Joule  D) Le Pascal.",
            _mcq_fr("Unités SI", "Le Joule (J) mesure le travail et l'énergie ; le Watt mesure la puissance et le Newton la force.", "C) Le Joule"),
        ),
        (
            "Un véhicule parcourt d = 150 km à vitesse constante pendant t = 2,5 heures. Calcule sa vitesse moyenne v.",
            "### Mouvement Rectiligne Uniforme (SautiLM-Jua-STEM)\n\n1. Formule : v = d / t = 150 / 2,5 = 60 km/h.\n\n**Réponse finale :** v = 60 km/h.",
        ),
        (
            "Calcule la fréquence f d'un phénomène périodique de période T = 0,02 s.",
            "### Onde et Périodicité (SautiLM-Jua-STEM)\n\n1. Formule : f = 1 / T = 1 / 0,02 = 50 Hz.\n\n**Réponse finale :** f = 50 Hz.",
        ),
        (
            "Une lentille convergente a une distance focale f' = 0,25 m. Calcule sa vergence C en dioptries.",
            "### Optique Géométrique (SautiLM-Jua-STEM)\n\n1. Formule : C = 1 / f' = 1 / 0,25 = 4 dioptries.\n\n**Réponse finale :** C = +4 dioptries.",
        ),
        (
            "QCM Physique : À quelle température l'eau pure bout-elle sous pression atmosphérique normale ? A) 0°C  B) 50°C  C) 100°C  D) 273°C.",
            _mcq_fr("Changements d'État", "Sous pression atmosphérique normale (1 atm), l'eau pure fond à 0°C et entre en ébullition à 100°C.", "C) 100°C"),
        ),
        (
            "QCM Physique : Quel instrument mesure la tension électrique entre deux bornes ? A) Le voltmètre en dérivation  B) L'ampèremètre en série  C) Le dynamomètre  D) Le baromètre.",
            _mcq_fr("Mesures Électriques", "La tension électrique (en Volts) se mesure à l'aide d'un voltmètre branché en dérivation (parallèle).", "A) Le voltmètre en dérivation"),
        ),
        (
            "Une force F = 40 N déplace un objet sur une distance d = 5 m dans le sens du mouvement. Calcule le travail W.",
            "### Travail d'une Force (SautiLM-Jua-STEM)\n\n1. Formule : W = F * d = 40 N * 5 m = 200 Joules.\n\n**Réponse finale :** W = 200 J.",
        ),
    ])
    assert len(phys_pairs) == 47, f"Expected 47 French Phys pairs, got {len(phys_pairs)}"

    # --- CHIMIE (47 paires; held out test: 36 g H2O) ---
    chem_pairs: List[tuple[str, str]] = []
    mole_cases = [
        ("eau H2O", 18, 18), ("eau H2O", 18, 54), ("eau H2O", 18, 72), ("eau H2O", 18, 90),
        ("dioxyde de carbone CO2", 44, 44), ("dioxyde de carbone CO2", 44, 88), ("dioxyde de carbone CO2", 44, 22),
        ("dioxygène O2", 32, 64), ("dioxygène O2", 32, 96), ("carbonate de calcium CaCO3", 100, 200),
        ("carbonate de calcium CaCO3", 100, 50), ("soude NaOH", 40, 80), ("soude NaOH", 40, 120),
        ("acide sulfurique H2SO4", 98, 196), ("ammoniac NH3", 17, 34), ("méthane CH4", 16, 48),
    ]
    for subst, molar_mass, mass in mole_cases:
        n_moles = mass / molar_mass
        chem_pairs.append((
            f"Calcule le nombre de moles contenues dans {mass} g de {subst} (M = {molar_mass} g/mol).",
            f"### Quantité de Matière (SautiLM-Jua-STEM)\n\n"
            f"1. Formule : n = m / M (masse en g divisée par masse molaire en g/mol).\n"
            f"2. Calcul : n = {mass} g / {molar_mass} g/mol = {_fmt_num(n_moles)} mol.\n\n"
            f"**Réponse finale :** {_fmt_num(n_moles)} mol.",
        ))
    for n_val, v_ml in [(0.5, 250), (1.0, 500), (0.2, 100), (0.6, 300), (1.5, 500), (0.4, 200), (2.0, 1000), (0.3, 150)]:
        v_l = v_ml / 1000.0
        conc = n_val / v_l
        chem_pairs.append((
            f"On dissout n = {_fmt_num(n_val)} mol de soluté dans V = {v_ml} mL de solution. Calcule la concentration molaire C.",
            f"### Concentration Molaire (SautiLM-Jua-STEM)\n\n"
            f"1. Volume : V = {v_ml} mL = {_fmt_num(v_l)} L.\n"
            f"2. Formule : C = n / V = {_fmt_num(n_val)} / {_fmt_num(v_l)} = {_fmt_num(conc)} mol/L.\n\n"
            f"**Réponse finale :** C = {_fmt_num(conc)} mol/L.",
        ))
    for ph_val, opts, corr_letter, corr_txt, expl in [
        (2, "A) Basique  B) Neutre  C) Acide  D) Saturée", "C", "Acide", "pH = 2 < 7 indique un excès d'ions H3O+, donc une solution acide."),
        (4, "A) Basique  B) Neutre  C) Acide  D) Salée", "C", "Acide", "Sur l'échelle 0 à 14, pH = 4 < 7 correspond à une solution acide."),
        (5, "A) Acide  B) Neutre  C) Basique  D) Sursaturée", "A", "Acide", "Comme pH = 5 < 7, la solution est acide."),
        (7, "A) Acide  B) Neutre  C) Basique  D) Oxydante", "B", "Neutre", "À 25°C, pH = 7 correspond exactement à une solution neutre (comme l'eau pure)."),
        (9, "A) Acide  B) Neutre  C) Basique  D) Concentrée", "C", "Basique", "pH = 9 > 7 indique un excès d'ions OH-, donc une solution basique."),
        (11, "A) Acide  B) Neutre  C) Salée  D) Basique", "D", "Basique", "Comme pH = 11 > 7, la solution est fortement basique / alcaline."),
        (13, "A) Acide  B) Neutre  C) Basique  D) Tampon", "C", "Basique", "pH = 13 > 7 correspond à une solution fortement basique."),
        (1, "A) Fortement basique  B) Fortement acide  C) Neutre  D) Gazeuse", "B", "Fortement acide", "pH = 1 est très inférieur à 7 (solution fortement acide)."),
    ]:
        chem_pairs.append((
            f"QCM Chimie : Une solution aqueuse présente un pH = {ph_val} à 25°C. Quelle est sa nature ? {opts}.",
            _mcq_fr("Échelle de pH", expl, f"{corr_letter}) {corr_txt}"),
        ))
    chem_pairs.extend([
        ("Équilibre l'équation chimique : CH4 + O2 -> CO2 + H2O.", "### Équilibration Chimique (SautiLM-Jua-STEM)\n\n1. Carbone : 1 CO2. Hydrogène : 2 H2O. Oxygène : 2 O2.\n\n**Équation équilibrée :** CH4 + 2 O2 -> CO2 + 2 H2O."),
        ("Équilibre l'équation : N2 + H2 -> NH3.", "### Équilibration Chimique (SautiLM-Jua-STEM)\n\n1. 1 N2 donne 2 NH3, nécessitant 3 H2.\n\n**Équation équilibrée :** N2 + 3 H2 -> 2 NH3."),
        ("Équilibre : C2H6 + O2 -> CO2 + H2O.", "### Équilibration Chimique (SautiLM-Jua-STEM)\n\n1. Pour 2 C2H6 on forme 4 CO2 et 6 H2O (14 atomes d'oxygène = 7 O2).\n\n**Équation équilibrée :** 2 C2H6 + 7 O2 -> 4 CO2 + 6 H2O."),
        ("QCM Chimie : Qu'est-ce qu'une réaction d'oxydation ? A) Gain d'électrons  B) Perte d'électrons  C) Neutralisation  D) Fusion.", _mcq_fr("Oxydoréduction", "Une oxydation correspond à une perte d'électrons (tandis que la réduction est un gain d'électrons).", "B) Perte d'électrons")),
        ("QCM Chimie : Qu'est-ce qu'une réduction ? A) Gain d'électrons  B) Perte de protons  C) Perte d'électrons  D) Évaporation.", _mcq_fr("Oxydoréduction", "Lors d'une réduction (ex. Cu^2+ + 2e- -> Cu), l'espèce chimique gagne des électrons.", "A) Gain d'électrons")),
        ("Quelle est la formule générale des alcanes saturés ?", "### Chimie Organique (SautiLM-Jua-STEM)\n\n1. Formule générale : C_n H_{2n+2} (n >= 1), par exemple CH4 (méthane) et C2H6 (éthane)."),
        ("Un atome de Sodium Na a Z = 11 et A = 23. Donne le nombre de protons, d'électrons et de neutrons.", "### Structure de l'Atome (SautiLM-Jua-STEM)\n\n1. Z = 11 -> 11 protons et 11 électrons.\n2. N = A - Z = 23 - 11 = 12 neutrons."),
        ("Calcule la masse molaire du glucose C6H12O6 (C=12, H=1, O=16 g/mol).", "### Masse Molaire (SautiLM-Jua-STEM)\n\n1. M = 6(12) + 12(1) + 6(16) = 72 + 12 + 96 = 180 g/mol."),
        ("QCM Chimie : Quelle liaison met en commun une paire d'électrons ? A) Liaison ionique  B) Liaison covalente  C) Liaison métallique  D) Force nucléaire.", _mcq_fr("Liaisons Chimiques", "La liaison covalente résulte de la mise en commun d'un doublet d'électrons entre deux atomes.", "B) Liaison covalente")),
        ("Dilution : on prélève 20 mL de solution mère C1 = 1,5 mol/L et on complète à V2 = 100 mL. Calcule C2.", "### Dilution (SautiLM-Jua-STEM)\n\n1. C1 * V1 = C2 * V2 donne C2 = (1,5 * 20) / 100 = 0,3 mol/L."),
        ("Quelle est la définition d'un acide selon Brönsted-Lowry ?", "### Acides et Bases (SautiLM-Jua-STEM)\n\n1. Un acide est une espèce chimique capable de céder un ou plusieurs protons H+."),
        ("QCM Chimie : Quel gaz trouble l'eau de chaux ? A) O2  B) N2  C) CO2  D) H2.", _mcq_fr("Identification des Gaz", "Le dioxyde de carbone CO2 réagit avec l'hydroxyde de calcium pour former un précipité blanc de CaCO3.", "C) CO2")),
        ("Combien d'électrons de valence possède le Carbone (Z = 6) ?", "### Configuration Électronique (SautiLM-Jua-STEM)\n\n1. (K)^2 (L)^4 : le carbone possède 4 électrons de valence et forme 4 liaisons covalentes."),
        ("QCM Chimie : Quel est le pH d'une solution contenant [H3O+] = 10^-2 mol/L ? A) 12  B) 7  C) 2  D) 1.", _mcq_fr("Calcul de pH", "pH = -log10([H3O+]) = -log10(10^-2) = 2.", "C) 2")),
        ("Combien de doublets liants forme l'oxygène (Z = 8) dans la molécule d'eau H2O ?", "### Structure de Lewis (SautiLM-Jua-STEM)\n\n1. L'oxygène a 6 électrons externes (K)^2 (L)^6 ; il lui manque 2 électrons pour l'octet, donc il forme 2 liaisons covalentes avec H."),
    ])
    assert len(chem_pairs) == 47, f"Expected 47 French Chem pairs, got {len(chem_pairs)}"

    # --- BIOLOGIE / SVT (47 paires, CoT-first QCMs) ---
    bio_pairs: List[tuple[str, str]] = [
        (
            "QCM Biologie : Quel organite cellulaire contient la chlorophylle et réalise la photosynthèse ? A) La mitochondrie  B) Le noyau  C) Le chloroplaste  D) Le ribosome.",
            _mcq_fr("Biologie Cellulaire", "Le chloroplaste renferme la chlorophylle qui capte la lumière pour transformer CO2 et H2O en glucose et O2.", "C) Le chloroplaste"),
        ),
        (
            "Écris l'équation bilan de la photosynthèse et explique le rôle de la chlorophylle.",
            "### Photosynthèse Végétale (SautiLM-Jua-STEM)\n\n"
            "1. Rôle de la chlorophylle : pigment vert des chloroplastes qui absorbe la lumière solaire.\n"
            "2. Équation bilan : 6 CO2 + 6 H2O -> C6H12O6 (glucose) + 6 O2 (dioxygène).",
        ),
        (
            "Quelle est la différence fondamentale entre la mitose et la méiose en biologie ?",
            "### Division Cellulaire : Mitose vs Méiose (SautiLM-Jua-STEM)\n\n"
            "1. Mitose : une cellule diploïde (2n) donne 2 cellules filles génétiquement identiques (2n) pour la croissance et le renouvellement des tissus.\n"
            "2. Méiose : une cellule diploïde (2n) donne 4 gamètes haploïdes (n) pour la reproduction sexuée.",
        ),
        (
            "QCM Biologie : Quel type de division cellulaire conserve le même nombre de chromosomes 2n ? A) Méiose  B) Mitose  C) Fécondation  D) Osmose.",
            _mcq_fr("Division Cellulaire", "La mitose produit 2 cellules filles diploïdes (2n) identiques à la cellule mère, tandis que la méiose réduit le nombre à n chromosomes.", "B) Mitose"),
        ),
        (
            "Cite les 4 phases principales de la mitose dans l'ordre chronologique.",
            "### Phases de la Mitose (SautiLM-Jua-STEM)\n\n"
            "1. Prophase : condensation des chromosomes.\n2. Métaphase : alignement sur le plan équatorial.\n"
            "3. Anaphase : séparation des chromatides sœurs.\n4. Télophase : reconstitution de deux noyaux.",
        ),
        (
            "QCM Biologie : Quelle cavité du cœur humain propulse le sang oxygéné dans l'aorte ? A) Oreillette droite  B) Ventricule droit  C) Oreillette gauche  D) Ventricule gauche.",
            _mcq_fr("Physiologie Cardiaque", "Le sang oxygéné arrive dans l'oreillette gauche puis le ventricule gauche l'éjecte dans l'aorte vers tous les organes.", "D) Ventricule gauche"),
        ),
        (
            "En génétique mendélienne, on croise deux plantes hétérozygotes Rr x Rr (R dominant). Quelles sont les proportions phénotypiques en F2 ?",
            "### Génétique Mendélienne (SautiLM-Jua-STEM)\n\n"
            "1. Échiquier : RR (25 %), Rr (50 %), rr (25 %).\n2. Réponse finale : proportions phénotypiques 3:1 (75 % dominant [R], 25 % récessif [r]).",
        ),
        (
            "QCM Biologie : Quel est le rôle principal de l'ARNm lors de la synthèse des protéines ? A) Stocker l'ATP  B) Transporter le message génétique du noyau vers les ribosomes  C) Coaguler le sang  D) Digérer les lipides.",
            _mcq_fr("Expression Génétique", "L'ARNm copie l'information d'un gène de l'ADN dans le noyau et la transporte jusqu'aux ribosomes pour la traduction.", "B) Transporter le message génétique du noyau vers les ribosomes"),
        ),
        (
            "Quel est le rôle des mitochondries dans une cellule eucaryote ?",
            "### Bioénergétique Cellulaire (SautiLM-Jua-STEM)\n\n1. Les mitochondries assurent la respiration cellulaire aérobie pour produire de l'ATP utilisable par la cellule.",
        ),
        (
            "QCM Biologie : Quelle molécule porte l'information héréditaire en double hélice ? A) Amidon  B) ADN  C) Glycogène  D) Hémoglobine.",
            _mcq_fr("Génétique Moléculaire", "L'ADN forme une double hélice de nucléotides avec appariement complémentaire A-T et C-G.", "B) ADN"),
        ),
        (
            "Dans une molécule d'ADN double brin, un brin porte A-T-G-C-C-A. Quelle est la séquence complémentaire ?",
            "### Complémentarité de l'ADN (SautiLM-Jua-STEM)\n\n1. Appariement A-T et C-G -> T-A-C-G-G-T.",
        ),
        (
            "QCM Biologie : Quel type de cellule sanguine transporte l'O2 via l'hémoglobine ? A) Leucocytes  B) Hématies (globules rouges)  C) Plaquettes  D) Neurones.",
            _mcq_fr("Physiologie du Sang", "Les hématies (globules rouges) contiennent l'hémoglobine qui fixe et transporte le dioxygène.", "B) Hématies (globules rouges)"),
        ),
        (
            "Quelle est la différence entre la reproduction sexuée et la reproduction asexuée ?",
            "### Modes de Reproduction (SautiLM-Jua-STEM)\n\n1. Sexuée : fusion de deux gamètes et brassage génétique.\n2. Asexuée : un seul parent produit des clones identiques.",
        ),
        (
            "QCM Biologie : Quel organe humain produit l'insuline ? A) Rein  B) Pancréas  C) Rate  D) Estomac.",
            _mcq_fr("Régulation de la Glycémie", "Les îlots bêta du pancréas sécrètent l'insuline pour abaisser le taux de glucose sanguin.", "B) Pancréas"),
        ),
        (
            "Explique le trajet de l'influx nerveux dans un arc réflexe médullaire simple.",
            "### Arc Réflexe (SautiLM-Jua-STEM)\n\n1. Récepteur sensoriel -> Nerf sensitif -> Moelle épinière -> Nerf moteur -> Muscle effecteur.",
        ),
    ]
    for dom_name, rec_name, p1, p2, f1_desc, f2_ratio in [
        ("graine lisse L", "graine ridée l", "LL", "ll", "100 % [L] hétérozygotes Ll", "3/4 [L] et 1/4 [l]"),
        ("fleur rouge R", "fleur blanche r", "RR", "rr", "100 % [R] hétérozygotes Rr", "75 % [R] et 25 % [r]"),
        ("tige haute T", "tige naine t", "TT", "tt", "100 % [T] hétérozygotes Tt", "3:1 (75 % haute, 25 % naine)"),
        ("pelage noir B", "pelage blanc b", "BB", "bb", "100 % [B] hétérozygotes Bb", "3/4 noir et 1/4 blanc"),
        ("grain jaune J", "grain vert j", "JJ", "jj", "100 % [J] hétérozygotes Jj", "3/4 jaune et 1/4 vert"),
    ]:
        bio_pairs.append((
            f"En génétique, on croise deux lignées pures {p1} ({dom_name} dominant) x {p2} ({rec_name} récessif). Que donne la F1 puis la F2 ?",
            f"### Croisement de Mendel (SautiLM-Jua-STEM)\n\n1. F1 : {f1_desc}.\n2. F2 : {f2_ratio}.",
        ))
    extra_bio = [
        ("QCM Écologie : Quel rôle jouent les plantes vertes dans une chaîne alimentaire ? A) Consommateurs primaires  B) Producteurs autotrophes  C) Décomposeurs  D) Prédateurs.", "Écologie", "Les plantes chlorophylliennes fabriquent leur matière organique par photosynthèse.", "B) Producteurs autotrophes"),
        ("QCM Écologie : Quel est le rôle des décomposeurs dans le sol ? A) Fixer O2  B) Minéraliser la matière organique morte  C) Produire le glucose  D) Briser les roches.", "Écologie", "Bactéries et champignons transforment les résidus morts en sels minéraux réutilisables par les végétaux.", "B) Minéraliser la matière organique morte"),
        ("QCM Biologie : Où a lieu la fécondation humaine normale ? A) Estomac  B) Trompe de Fallope  C) Ovaire  D) Foie.", "Reproduction Humaine", "L'ovocyte et le spermatozoïde fusionnent dans le tiers supérieur de la trompe de Fallope.", "B) Trompe de Fallope"),
        ("QCM Biologie : Quelle unité du rein filtre le sang ? A) Néphron  B) Alvéole  C) Neurone  D) Villosité.", "Excrétion", "Chaque rein contient environ un million de néphrons assurant la filtration du sang et la formation de l'urine.", "A) Néphron"),
        ("QCM Biologie : Quels échanges gazeux ont lieu dans les alvéoles pulmonaires ? A) O2 sort du sang  B) O2 entre dans le sang et CO2 en sort  C) N2 remplace le sang  D) Aucun.", "Respiration", "Par diffusion, l'O2 alvéolaire passe dans le sang et le CO2 sanguin passe dans l'alvéole.", "B) O2 entre dans le sang et CO2 en sort"),
        ("QCM Biologie : Quel vecteur transmet le paludisme ? A) Mouche tsé-tsé  B) Anophèle femelle  C) Pou  D) Escargot.", "Parasitologie", "Le moustique Anophèle femelle transmet le parasite Plasmodium lors de la piqûre.", "B) Anophèle femelle"),
        ("Explique la différence entre artères et veines.", None, "1. Les artères conduisent le sang du cœur vers les organes.\n2. Les veines ramènent le sang des organes vers le cœur.", None),
        ("QCM Biologie : Quel glucide sert de réserve énergétique chez les animaux ? A) Cellulose  B) Glycogène  C) Chitine  D) Insuline.", "Biochimie", "Le foie et les muscles stockent le glucose en excès sous forme de glycogène.", "B) Glycogène"),
        ("QCM Biologie : Que décrit l'osmose ? A) Passage d'eau du milieu hypotonique vers hypertonique  B) Transport actif d'ions  C) Division d'ADN  D) Synthèse protéique.", "Biologie Cellulaire", "L'osmose est la diffusion passive de l'eau à travers une membrane semi-perméable vers le milieu plus concentré.", "A) Passage d'eau du milieu hypotonique vers hypertonique"),
        ("Quel est le rôle des vaccins dans le système immunitaire ?", None, "Un vaccin induit la formation de lymphocytes mémoire spécifiques sans provoquer la maladie.", None),
        ("QCM Biologie : Quelle base remplace la thymine (T) dans l'ARN ? A) A  B) Uracile (U)  C) C  D) G.", "Biologie Moléculaire", "L'ARN comporte A, U, C et G ; l'uracile (U) remplace la thymine (T) de l'ADN.", "B) Uracile (U)"),
        ("Pourquoi l'amylase salivaire perd-elle son activité dans l'estomac ?", None, "Le pH très acide de l'estomac (environ 2) dénature la structure tridimensionnelle de l'amylase salivaire.", None),
        ("QCM Biologie : Combien de chromosomes contient une cellule somatique humaine ? A) 23  B) 46  C) 44  D) 92.", "Génétique Humaine", "Les cellules somatiques humaines sont diploïdes (2n = 46 chromosomes, soit 23 paires).", "B) 46"),
        ("Explique la sélection naturelle selon Darwin.", None, "Les individus dotés de caractères héréditaires avantageux survivent et se reproduisent davantage dans leur milieu.", None),
        ("QCM Biologie : Quel sucre simple compose l'amidon ? A) Saccharose  B) Glucose  C) Acide aminé  D) Acide gras.", "Nutrition", "L'amidon végétal est un polymère formé de nombreuses unités de glucose.", "B) Glucose"),
        ("Quelle est la fonction des stomates foliaires ?", None, "Les stomates permettent les échanges gazeux (entrée de CO2, sortie d'O2) et la transpiration végétale.", None),
        ("QCM Biologie : Quel organisme n'a pas de membrane nucléaire ? A) Animal  B) Végétal  C) Procaryote (bactérie)  D) Levure.", "Microbiologie", "Les bactéries sont des procaryotes dont l'ADN baigne librement dans le cytoplasme sans enveloppe nucléaire.", "C) Procaryote (bactérie)"),
        ("Cite deux différences entre cellule végétale et cellule animale.", None, "La cellule végétale possède une paroi cellulosique rigide et des chloroplastes, absents chez la cellule animale.", None),
        ("QCM Biologie : Quel groupe sanguin est receveur universel de globules rouges ? A) O-  B) A+  C) B-  D) AB+.", "Immunologie", "Les individus AB+ ne possèdent ni anticorps anti-A, ni anti-B, ni anti-Rhésus et peuvent recevoir tous les groupes érythrocytaires.", "D) AB+"),
        ("Quel est l'effet d'une carence en iode ?", None, "Une carence en iode réduit la synthèse des hormones thyroïdiennes et provoque un gonflement de la thyroïde (goitre).", None),
        ("QCM Biologie : Quelle hormone végétale règle le phototropisme ? A) Adrénaline  B) Auxine  C) Insuline  D) Testostérone.", "Physiologie Végétale", "L'auxine stimule l'élongation cellulaire du côté ombragé de la tige, l'orientant vers la lumière.", "B) Auxine"),
        ("Comment les légumineuses fixent-elles l'azote atmosphérique N2 ?", None, "Grâce à une symbiose avec les bactéries Rhizobium logées dans les nodosités de leurs racines.", None),
        ("QCM Biologie : Que désigne le génotype ? A) Aspect visible  B) Ensemble des allèles / gènes d'un individu  C) Milieu de vie  D) Nombre d'organes.", "Génétique", "Le génotype est le patrimoine génétique (combinaison d'allèles) d'un individu, contrairement au phénotype observable.", "B) Ensemble des allèles / gènes d'un individu"),
        ("Pourquoi la biodiversité stabilise-t-elle un agro-écosystème ?", None, "Elle limite la propagation des ravageurs, maintient les pollinisateurs et renforce la fertilité et résilience des sols.", None),
        ("QCM Biologie : Comment qualifie-t-on la réplication de l'ADN ? A) Conservative  B) Semi-conservative  C) Dispersive  D) Transcrite.", "Biologie Moléculaire", "Chaque nouvelle molécule d'ADN fille conserve un brin parental d'origine et synthétise un brin complémentaire neuf.", "B) Semi-conservative"),
        ("Quel est le rôle des globules blancs (leucocytes) dans l'organisme ?", None, "Les leucocytes défendent l'organisme contre les bactéries, virus et parasites par phagocytose et production d'anticorps.", None),
        ("QCM Biologie : Quel pigment respiratoire donne sa couleur rouge au sang ? A) Chlorophylle  B) Mélanine  C) Hémoglobine  D) Kératine.", "Physiologie", "L'hémoglobine contient du fer qui fixe le dioxygène dans les globules rouges.", "C) Hémoglobine"),
    ]
    for q, topic, body, ans in extra_bio:
        if topic is not None and ans is not None:
            bio_pairs.append((q, _mcq_fr(topic, body, ans)))
        else:
            bio_pairs.append((q, f"### Biologie / SVT (SautiLM-Jua-STEM)\n\n1. {body}"))

    assert len(bio_pairs) == 47, f"Expected 47 French Bio pairs, got {len(bio_pairs)}"

    pairs = math_pairs + phys_pairs + chem_pairs + bio_pairs
    assert len(pairs) == 195, f"Expected 195 French STEM pairs, got {len(pairs)}"
    return pairs


def _english_stem_pairs() -> List[tuple[str, str]]:
    """Builds 130 diverse English Secondary STEM pairs (35 Math, 32 Physics, 32 Chem, 31 Bio)."""
    math_pairs: List[tuple[str, str]] = [
        ("STEM Math MCQ: What is the derivative of f(x) = sin(x)? A) -cos(x)  B) cos(x)  C) tan(x)  D) -sin(x).", _mcq_en("Math", "By standard trigonometric differentiation, the derivative of sin(x) is cos(x).", "B) cos(x)")),
        ("STEM Math MCQ: Evaluate sin^2(theta) + cos^2(theta) for any real angle theta. A) 0  B) 1  C) 2  D) -1.", _mcq_en("Trigonometry", "On the unit circle, the Pythagorean identity states cos^2(theta) + sin^2(theta) = 1.", "B) 1")),
        ("Find the integral of sin(x) / cos(x) dx step by step.", "### Trigonometric Substitution Integral (SautiLM-Jua-STEM)\n\n1. Let u = cos(x), so du = -sin(x) dx (or sin(x) dx = -du).\n2. Integrate: - int (1 / u) du = -ln|cos(x)| + C.\n\n**Final Answer:** -ln|cos(x)| + C."),
        ("Find the integral of cos(x) / sin(x) dx step by step.", "### Substitution Integral (SautiLM-Jua-STEM)\n\n1. Let u = sin(x), so du = cos(x) dx.\n2. Integrate: int (1 / u) du = ln|sin(x)| + C.\n\n**Final Answer:** ln|sin(x)| + C."),
        ("Find the integral of x^2 dx step by step.", "### Integration Power Rule (SautiLM-Jua-STEM)\n\n1. Rule: int x^n dx = x^(n+1) / (n + 1) + C.\n\n**Final Answer:** (x^3) / 3 + C."),
        ("Find the integral of 2x / (x^2 + 1) dx using u-substitution.", "### Logarithmic Substitution (SautiLM-Jua-STEM)\n\n1. Set u = x^2 + 1 so du = 2x dx.\n2. Integral becomes int (1 / u) du = ln(x^2 + 1) + C.\n\n**Final Answer:** ln(x^2 + 1) + C."),
        ("Evaluate step by step: (24 * 15) + 140.", "### Step-by-Step Arithmetic (SautiLM-Jua-STEM)\n\n1. Multiply first: 24 * 15 = 360.\n2. Add remainder: 360 + 140 = 500.\n\n**Final Answer:** 500."),
        ("Find the integral of e^(3x) dx step by step.", "### Exponential Integral (SautiLM-Jua-STEM)\n\n1. Formula: int e^(kx) dx = (1 / k) e^(kx) + C.\n\n**Final Answer:** (1/3) e^(3x) + C."),
        ("STEM Math MCQ: What is the gradient of the line y = 3x - 7? A) -7  B) 3  C) -3  D) 7.", _mcq_en("Linear Functions", "In slope-intercept form y = mx + c, the gradient m is the coefficient of x, which is 3.", "B) 3")),
    ]
    # Note: keep (1, 3, 4) -> x^2 - 7x + 12 = 0 STRICTLY HELD OUT for generalization eval!
    for quad_args in [(1, 2, 6), (1, 2, 5), (1, -1, 6), (2, 1, 3), (1, 4, 7), (1, -3, 2), (2, -1, 5), (1, 5, 6), (3, 1, 2), (1, -4, 4)]:
        math_pairs.append(_quadratic_en(*quad_args))
    for a, b, c in [(5, 12, 13), (3, 4, 5), (6, 8, 10), (8, 15, 17), (7, 24, 25), (9, 12, 15)]:
        math_pairs.append((
            f"A right triangle has legs a = {a} cm and b = {b} cm. Find the hypotenuse c.",
            f"### Pythagoras Theorem (SautiLM-Jua-STEM)\n\n1. c^2 = a^2 + b^2 = {a}^2 + {b}^2 = {c * c}.\n2. c = sqrt({c * c}) = {c} cm.\n\n**Final Answer:** c = {c} cm.",
        ))
    for a, b, c in [(2, -3, 7), (4, 5, -2), (3, -8, 1), (5, 2, -6), (1, -9, 4), (6, -4, 3), (2, 7, -5), (4, -1, 8), (3, 6, -4), (5, -5, 2)]:
        math_pairs.append((
            f"Differentiate f(x) = {a}x^3 + ({b})x^2 + ({c})x with respect to x.",
            f"### Polynomial Derivative (SautiLM-Jua-STEM)\n\n1. Power rule gives f'(x) = {3 * a}x^2 + ({2 * b})x + ({c}).",
        ))
    assert len(math_pairs) == 35, f"Expected 35 English Math pairs, got {len(math_pairs)}"

    # --- PHYSICS (32 pairs; held out test: 12 ohms, 2 A) ---
    phys_pairs: List[tuple[str, str]] = []
    for r_val, i_val in [(4, 3), (6, 2), (5, 4), (10, 3), (8, 2), (15, 3), (20, 2), (9, 3), (7, 4), (25, 2)]:
        v_val = r_val * i_val
        p_val = v_val * i_val
        phys_pairs.append((
            f"A {r_val} ohm resistor carries a current of {i_val} A. Compute the voltage V and electrical power P.",
            f"### Ohm's Law and Electrical Power (SautiLM-Jua-STEM)\n\n"
            f"1. Voltage (V = I * R): V = {i_val} A * {r_val} ohms = {v_val} V.\n"
            f"2. Power (P = V * I): P = {v_val} V * {i_val} A = {p_val} W.\n\n"
            f"**Final Answer:** {v_val} V and {p_val} W.",
        ))
    for m_val, v_val in [(6, 4), (4, 5), (2, 10), (8, 5), (10, 3), (12, 2), (5, 6), (3, 8)]:
        ke_val = 0.5 * m_val * (v_val ** 2)
        phys_pairs.append((
            f"Calculate the kinetic energy of a {m_val} kg object moving at {v_val} m/s.",
            f"### Kinetic Energy Calculation (SautiLM-Jua-STEM)\n\n1. KE = 0.5 * m * v^2 = 0.5 * {m_val} * {v_val}^2 = {_fmt_num(ke_val)} Joules.\n\n**Final Answer:** {_fmt_num(ke_val)} J.",
        ))
    for m_val, a_val in [(5, 4), (8, 3), (10, 2), (4, 7), (6, 5), (12, 2), (9, 3), (15, 4)]:
        phys_pairs.append((
            f"What net force F accelerates a {m_val} kg mass at {a_val} m/s^2?",
            f"### Newton's Second Law (SautiLM-Jua-STEM)\n\n1. F = m * a = {m_val} kg * {a_val} m/s^2 = {m_val * a_val} N.\n\n**Final Answer:** {m_val * a_val} N.",
        ))
    phys_pairs.extend([
        ("STEM Physics MCQ: Two resistors R1 = 4 Ohm and R2 = 6 Ohm are connected in series. What is the total resistance? A) 2.4 Ohm  B) 10 Ohm  C) 24 Ohm  D) 5 Ohm.", _mcq_en("Physics", "In series circuits, total resistance equals the sum R_total = R1 + R2 = 4 + 6 = 10 Ohm.", "B) 10 Ohm")),
        ("How do photovoltaic solar cells convert sunlight into electric current?", "### Solar PV Working Principle (SautiLM-Jua-STEM)\n\n1. Sunlight photons strike doped silicon p-n junctions, freeing electrons that flow as direct current (DC)."),
        ("STEM Physics MCQ: What is the SI unit of electrical resistance? A) Ampere  B) Volt  C) Ohm  D) Watt.", _mcq_en("Physics", "Electrical resistance R = V / I is measured in Ohms, whereas Ampere measures current and Volt measures voltage.", "C) Ohm")),
        ("A wave has wavelength 0.5 m and frequency 680 Hz. Calculate its wave speed v.", "### Wave Speed (SautiLM-Jua-STEM)\n\n1. v = f * lambda = 680 * 0.5 = 340 m/s.\n\n**Final Answer:** 340 m/s."),
        ("STEM Physics MCQ: What is the SI unit of frequency? A) Joule  B) Newton  C) Hertz  D) Tesla.", _mcq_en("Physics", "Frequency counts cycles per second (s^-1), whose SI unit is the Hertz (Hz).", "C) Hertz")),
        ("Calculate work done when a force of 25 N pushes a crate 6 m along the floor.", "### Mechanical Work (SautiLM-Jua-STEM)\n\n1. W = F * d = 25 N * 6 m = 150 Joules.\n\n**Final Answer:** 150 J."),
    ])
    assert len(phys_pairs) == 32, f"Expected 32 English Phys pairs, got {len(phys_pairs)}"

    # --- CHEMISTRY (32 pairs; held out test: pH = 3) ---
    chem_pairs: List[tuple[str, str]] = []
    for subst, molar_mass, mass in [
        ("CO2", 44, 44), ("CO2", 44, 88), ("CO2", 44, 132), ("H2O", 18, 18), ("H2O", 18, 54),
        ("H2O", 18, 90), ("O2", 32, 64), ("O2", 32, 32), ("NaOH", 40, 80), ("CaCO3", 100, 150),
    ]:
        chem_pairs.append((
            f"Calculate the number of moles n in {mass} g of {subst} (M = {molar_mass} g/mol).",
            f"### Mole Calculation (SautiLM-Jua-STEM)\n\n1. Formula: n = mass / M = {mass} g / {molar_mass} g/mol = {_fmt_num(mass / molar_mass)} mol.\n\n**Final Answer:** {_fmt_num(mass / molar_mass)} mol.",
        ))
    for n_val, v_ml in [(0.4, 500), (0.5, 250), (1.2, 600), (0.8, 400), (2.0, 500), (0.25, 250)]:
        conc = n_val / (v_ml / 1000.0)
        chem_pairs.append((
            f"Calculate molarity M when {n_val} mol of solute dissolves in {v_ml} mL of solution.",
            f"### Molarity Step-by-Step (SautiLM-Jua-STEM)\n\n1. Volume: {v_ml} mL = {_fmt_num(v_ml / 1000.0)} L.\n2. Molarity: M = n / V = {n_val} / {_fmt_num(v_ml / 1000.0)} = {_fmt_num(conc)} mol/L.\n\n**Final Answer:** {_fmt_num(conc)} M.",
        ))
    for ph_val, opts, corr_letter, corr_label, expl in [
        (1, "A) Basic  B) Neutral  C) Acidic  D) Salt", "C", "Acidic", "Because pH = 1 < 7, hydrogen ions (H+) are in excess so the solution is acidic."),
        (2, "A) Basic  B) Neutral  C) Acidic  D) Salt", "C", "Acidic", "Aqueous solutions with pH = 2 < 7 have excess hydrogen ions (H+) and are classified as acidic."),
        (4, "A) Basic  B) Acidic  C) Neutral  D) Salt", "B", "Acidic", "Since pH = 4 is less than 7, the solution falls in the acidic range on the 0-14 pH scale."),
        (5, "A) Acidic  B) Neutral  C) Basic  D) Salt", "A", "Acidic", "Because pH = 5 is below 7, the solution is acidic."),
        (7, "A) Acidic  B) Neutral  C) Basic  D) Salt", "B", "Neutral", "Pure water at 25 C has pH = 7, where H+ and OH- concentrations are equal, making it neutral."),
        (9, "A) Acidic  B) Neutral  C) Basic  D) Salt", "C", "Basic", "Since pH = 9 > 7, hydroxide ions (OH-) predominate so the solution is basic."),
        (11, "A) Acidic  B) Neutral  C) Salt  D) Basic", "D", "Basic", "Solutions with pH = 11 > 7 are alkaline / basic."),
        (13, "A) Acidic  B) Neutral  C) Basic  D) Salt", "C", "Basic", "Because pH = 13 > 7, the solution is strongly basic."),
    ]:
        chem_pairs.append((
            f"QCM Chemistry: A solution has pH = {ph_val} at 25 C. How is it classified? {opts}.",
            _mcq_en("Chemistry", expl, f"{corr_letter}) {corr_label}"),
        ))
    chem_pairs.extend([
        ("Balance the chemical equation for water formation: H2 + O2 -> H2O.", "### Balancing Equations (SautiLM-Jua-STEM)\n\n1. Two O atoms require 2 H2O, which need 2 H2.\n\n**Balanced Equation:** 2 H2 + O2 -> 2 H2O."),
        ("Balance the combustion of propane: C3H8 + O2 -> CO2 + H2O.", "### Balancing Equations (SautiLM-Jua-STEM)\n\n1. Three carbons give 3 CO2 and 8 hydrogens give 4 H2O (10 oxygen atoms = 5 O2).\n\n**Balanced Equation:** C3H8 + 5 O2 -> 3 CO2 + 4 H2O."),
        ("STEM Chemistry MCQ: Oxidation is defined as: A) Gain of electrons  B) Loss of electrons  C) Gain of mass  D) Cooling.", _mcq_en("Chemistry", "Oxidation increases oxidation state by losing electrons, while reduction gains electrons.", "B) Loss of electrons")),
        ("An atom has atomic number Z = 17 and mass number A = 35 (Chlorine). State protons, electrons, and neutrons.", "### Atomic Structure (SautiLM-Jua-STEM)\n\n1. 17 protons, 17 electrons, and 35 - 17 = 18 neutrons."),
        ("What is the difference between an ionic bond and a covalent bond?", "### Chemical Bonding (SautiLM-Jua-STEM)\n\n1. Ionic bond: electron transfer (e.g. NaCl).\n2. Covalent bond: shared electron pair between non-metals (e.g. H2O)."),
        ("Find the molar mass of calcium carbonate CaCO3 (Ca=40, C=12, O=16 g/mol).", "### Molar Mass (SautiLM-Jua-STEM)\n\n1. M = 40 + 12 + 3(16) = 100 g/mol."),
        ("STEM Chemistry MCQ: Which gas turns limewater milky? A) O2  B) H2  C) CO2  D) N2.", _mcq_en("Chemistry", "Carbon dioxide (CO2) reacts with calcium hydroxide to form insoluble white CaCO3.", "C) CO2")),
        ("What is the general molecular formula of saturated alkanes?", "### Organic Chemistry (SautiLM-Jua-STEM)\n\n1. Saturated alkanes obey C_n H_{2n+2}, such as methane (CH4) and ethane (C2H6)."),
    ])
    assert len(chem_pairs) == 32, f"Expected 32 English Chem pairs, got {len(chem_pairs)}"

    # --- BIOLOGY (31 pairs, CoT-first MCQs) ---
    en_bio_raw = [
        ("STEM Biology MCQ: Which organelle and pigment in green leaves capture sunlight for photosynthesis? A) Mitochondria and hemoglobin  B) Chloroplasts and chlorophyll  C) Ribosomes and keratin  D) Lysosomes and melanin.", "Biology", "Chlorophyll inside chloroplasts absorbs sunlight to convert CO2 and water into glucose and oxygen (O2).", "B) Chloroplasts and chlorophyll"),
        ("In secondary Biology, explain the role of chlorophyll and write the products of photosynthesis.", None, "1. Role of chlorophyll: chlorophyll is the green pigment inside chloroplasts that captures sunlight energy.\n2. Equation and products: plants convert carbon dioxide (CO2) and water (H2O) into glucose (sugar) and oxygen gas (O2) : 6 CO2 + 6 H2O -> C6H12O6 + 6 O2.", None),
        ("Why do green plants release oxygen gas (O2) during photosynthesis?", None, "During photosynthesis inside chloroplasts, chlorophyll absorbs sunlight energy to split water molecules and combine carbon dioxide into glucose, releasing oxygen (O2) as a vital product.", None),
        ("STEM Biology MCQ: Which human heart chamber pumps oxygen-rich blood into the aorta? A) Right atrium  B) Right ventricle  C) Left atrium  D) Left ventricle.", "Biology", "The thick-walled left ventricle generates high pressure to pump oxygenated blood through the aorta to the whole body.", "D) Left ventricle"),
        ("Explain the key difference between mitosis and meiosis.", None, "1. Mitosis produces 2 genetically identical diploid (2n) daughter cells for growth and tissue repair.\n2. Meiosis produces 4 genetically distinct haploid (n) gametes for sexual reproduction.", None),
        ("STEM Biology MCQ: Which type of cell division keeps the chromosome number constant (2n -> 2n)? A) Meiosis  B) Mitosis  C) Fertilization  D) Budding.", "Biology", "Mitosis preserves the diploid chromosome number 2n in both daughter cells, whereas meiosis halves it to n.", "B) Mitosis"),
        ("Explain the three stages of the water cycle (evaporation, condensation, precipitation).", None, "1. Evaporation turns surface water into vapor.\n2. Condensation cools rising vapor into clouds.\n3. Precipitation returns water as rain.", None),
        ("What is the function of mitochondria in animal and plant cells?", None, "Mitochondria perform aerobic cellular respiration to generate ATP energy from glucose and oxygen.", None),
        ("STEM Biology MCQ: In DNA base pairing, Adenine (A) always pairs with which base? A) Cytosine  B) Guanine  C) Thymine  D) Uracil.", "Biology", "By Chargaff complementary pairing in double-stranded DNA, Adenine (A) bonds with Thymine (T) and Cytosine (C) with Guanine (G).", "C) Thymine"),
        ("Find the complementary DNA strand for 5'-A-C-G-T-T-A-3'.", None, "Applying A-T and C-G base pairing gives 3'-T-G-C-A-A-T-5'.", None),
        ("Explain Mendelian monohybrid cross F2 ratio for Rr x Rr (R dominant over r).", None, "1. Punnett square yields RR (25%), Rr (50%), rr (25%).\n2. Phenotypic ratio is 3:1 (75% dominant : 25% recessive).", None),
        ("STEM Biology MCQ: Which blood cells carry oxygen from the lungs to tissues? A) Platelets  B) White blood cells  C) Red blood cells (erythrocytes)  D) Plasma proteins.", "Biology", "Red blood cells contain iron-rich hemoglobin that reversibly binds and transports oxygen.", "C) Red blood cells (erythrocytes)"),
        ("What role does messenger RNA (mRNA) play in protein synthesis?", None, "mRNA carries genetic instructions transcribed from nuclear DNA out to cytoplasmic ribosomes for protein translation.", None),
        ("STEM Biology MCQ: Which human organ secretes insulin to lower blood glucose? A) Kidney  B) Liver  C) Pancreas  D) Spleen.", "Biology", "Beta cells in the pancreas secrete insulin when blood glucose rises, promoting cellular glucose uptake.", "C) Pancreas"),
        ("List the pathway of air from the nostrils to the alveoli during inhalation.", None, "Nostrils -> Nasal cavity -> Pharynx -> Larynx -> Trachea -> Bronchi -> Bronchioles -> Alveoli.", None),
        ("STEM Biology MCQ: What distinguishes eukaryotic from prokaryotic cells? A) Ribosomes  B) Membrane-bound nucleus  C) Cell membrane  D) Cytoplasm.", "Biology", "Eukaryotic cells enclose DNA inside a membrane-bound nucleus, whereas prokaryotes lack a true nucleus.", "B) Membrane-bound nucleus"),
        ("How do Rhizobium bacteria in legume root nodules improve soil fertility?", None, "They fix atmospheric nitrogen gas (N2) into ammonia/nitrates usable by beans, groundnuts, and soy.", None),
        ("STEM Biology MCQ: What vector transmits malaria (Plasmodium)? A) Tsetse fly  B) Female Anopheles mosquito  C) Housefly  D) Tick.", "Biology", "Infected female Anopheles mosquitoes inject Plasmodium sporozoites during a blood meal.", "B) Female Anopheles mosquito"),
        ("Define osmosis in plant and animal cells.", None, "Osmosis is the passive movement of water molecules across a selectively permeable membrane from a dilute (hypotonic) to a concentrated (hypertonic) solution.", None),
        ("STEM Biology MCQ: Which ecosystem process converts dead organic matter into mineral soil nutrients? A) Predation  B) Decomposition  C) Transpiration  D) Osmosis.", "Ecology", "Decomposers break down dead organic matter into inorganic mineral nutrients available to plants.", "B) Decomposition"),
        ("List the four stages of mitosis in chronological order.", None, "1. Prophase\n2. Metaphase\n3. Anaphase\n4. Telophase.", None),
        ("STEM Biology MCQ: Which human vessel carries blood away from the heart? A) Vein  B) Artery  C) Capillary  D) Venule.", "Physiology", "Arteries carry blood away from the ventricles under high pressure, whereas veins return blood toward the heart.", "B) Artery"),
        ("Explain why high fever or extreme pH denatures digestive enzymes.", None, "Extreme heat or pH breaks weak bonds holding the enzyme's active site shape, preventing substrate binding.", None),
        ("STEM Biology MCQ: How many chromosomes are present in a normal human gamete? A) 46  B) 23  C) 44  D) 92.", "Genetics", "Meiosis halves the diploid number 2n = 46 into haploid gametes containing n = 23 chromosomes.", "B) 23"),
        ("Contrast aerobic respiration with anaerobic fermentation in muscle cells.", None, "Aerobic respiration with oxygen yields ~36-38 ATP (CO2 + H2O), whereas anaerobic fermentation yields 2 ATP plus lactic acid.", None),
        ("STEM Biology MCQ: Which plant tissue transports water upward from roots to leaves? A) Phloem  B) Xylem  C) Cortex  D) Epidermis.", "Botany", "Xylem vessels transport water and dissolved mineral salts upward from roots, whereas phloem transports sugars.", "B) Xylem"),
        ("What is the role of vaccination in adaptive immunity?", None, "Vaccines trigger memory B and T lymphocytes using safe antigens so future infections are neutralized rapidly.", None),
        ("STEM Biology MCQ: Which structure gives plant cells rigidity and prevents bursting? A) Mitochondrion  B) Cellulose cell wall  C) Golgi apparatus  D) Centriole.", "Cell Biology", "The rigid cellulose cell wall resists turgor pressure and prevents osmotic bursting.", "B) Cellulose cell wall"),
        ("Explain natural selection in Darwin's theory of evolution.", None, "Organisms with heritable traits better suited to their environment survive and reproduce more successfully across generations.", None),
        ("STEM Biology MCQ: Which plant hormone promotes phototropism? A) Insulin  B) Auxin  C) Thyroxine  D) Hemoglobin.", "Botany", "Auxin accumulates on the shaded side of a stem, stimulating cell elongation toward light.", "B) Auxin"),
        ("What is the function of white blood cells (leukocytes) in immunity?", None, "White blood cells defend the body against pathogens through phagocytosis and specific antibody production.", None),
    ]
    bio_pairs: List[tuple[str, str]] = []
    for q, topic, body, ans in en_bio_raw:
        if topic is not None and ans is not None:
            bio_pairs.append((q, _mcq_en(topic, body, ans)))
        else:
            bio_pairs.append((q, f"### Secondary Biology (SautiLM-Jua-STEM)\n\n{body}"))
    assert len(bio_pairs) == 31, f"Expected 31 English Bio pairs, got {len(bio_pairs)}"

    pairs = math_pairs + phys_pairs + chem_pairs + bio_pairs
    assert len(pairs) == 130, f"Expected 130 English STEM pairs, got {len(pairs)}"
    return pairs


def generate_stem_instructions() -> List[Dict[str, Any]]:
    """
    Generates SautiLM-Jua-STEM corpus (Secondary School & Examen d'État RDC).
    Strict 60% French / 40% English balance:
      - FR: 195 STEM * 2 + 15 identity * 4 = 390 + 60 = 450 rows (60.0%)
      - EN: 130 STEM * 2 + 10 identity * 4 = 260 + 40 = 300 rows (40.0%)
      - Total: 750 rows.
    """
    out: List[Dict[str, Any]] = []
    for q, a in _french_stem_pairs():
        out.append(_stem_turn("fr", STEM_SYSTEM_FR, q, a))
        out.append(_stem_turn("fr", None, q, a))

    for q, a in _english_stem_pairs():
        out.append(_stem_turn("en", STEM_SYSTEM_EN, q, a))
        out.append(_stem_turn("en", None, q, a))

    stem_greetings_identity = [
        ("fr", STEM_SYSTEM_FR, "Qui es-tu ?",
         "### Assistant SautiLM-Jua-STEM\n\nJe suis **SautiLM-Jua-STEM**, l'assistant scientifique hors-ligne (Mathématiques, Physique, Chimie, Biologie - niveau Secondaire et Examen d'État RDC) développé par **LosingTek Research**."),
        ("fr", STEM_SYSTEM_FR, "Qui es-tu et quel niveau scolaire couvres-tu ?",
         "### Assistant SautiLM-Jua-STEM\n\nJe suis **SautiLM-Jua-STEM** de **LosingTek Research**. Je couvre les **Mathématiques, Physique, Chimie et Biologie au niveau Secondaire et Examen d'État (RDC)** avec QCM explicatif et résolution étape par étape."),
        ("fr", STEM_SYSTEM_FR, "Quel programme couvre SautiLM-Jua-STEM ?",
         "### Assistant SautiLM-Jua-STEM\n\n**SautiLM-Jua-STEM** couvre le programme du **Secondaire et de l'Examen d'État RDC** en Mathématiques, Physique, Chimie et Biologie (en français 60 % et anglais 40 %)."),
        ("fr", STEM_SYSTEM_FR, "Qui t'a créé et à quoi sers-tu ?",
         "### Assistant SautiLM-Jua-STEM\n\nJ'ai été conçu par **LosingTek Research** (**Georges Byona**, **Abraham Imani**, **Jacques Uwonda**). Je suis **SautiLM-Jua-STEM**, spécialisé pour aider les élèves du **Secondaire** et préparer l'**Examen d'État** hors-ligne."),
        ("fr", STEM_SYSTEM_FR, "Es-tu utile pour l'Examen d'État ?",
         "### Assistant SautiLM-Jua-STEM\n\nOui ! En tant que **SautiLM-Jua-STEM** (**LosingTek Research**), je résous les QCM et exercices du **Secondaire** et de l'**Examen d'État** en Mathématiques, Physique, Chimie et Biologie."),
        ("fr", STEM_SYSTEM_FR, "Quel est ton rôle principal ?",
         "### Assistant SautiLM-Jua-STEM\n\nJe suis **SautiLM-Jua-STEM** (**LosingTek Research**), dédié à la réussite des élèves du **Secondaire** en Mathématiques, Physique, Chimie et Biologie."),
        ("fr", STEM_SYSTEM_FR, "Quelles matières scientifiques maîtrises-tu ?",
         "### Assistant SautiLM-Jua-STEM\n\n**SautiLM-Jua-STEM** couvre 4 matières du **Secondaire et Examen d'État** : Mathématiques, Physique, Chimie et Biologie / SVT."),
        ("fr", STEM_SYSTEM_FR, "Peux-tu m'aider à réviser les sciences hors connexion ?",
         "### Assistant SautiLM-Jua-STEM\n\nOui ! **SautiLM-Jua-STEM** fonctionne 100 % hors-ligne sur appareil mobile pour résoudre vos exercices du **Secondaire** et de l'**Examen d'État**."),
        ("fr", STEM_SYSTEM_FR, "Bonjour",
         "### Assistant SautiLM-Jua-STEM\n\nBonjour ! Je suis **SautiLM-Jua-STEM** (Secondaire et Examen d'État RDC). Quelle question de mathématiques, physique, chimie ou biologie voulez-vous travailler ?"),
        ("fr", STEM_SYSTEM_FR, "Salut",
         "### Assistant SautiLM-Jua-STEM\n\nSalut ! Je suis **SautiLM-Jua-STEM**. Envoyez un exercice ou un QCM scientifique du niveau secondaire et je le résous étape par étape."),
        ("fr", STEM_SYSTEM_FR, "Bonsoir",
         "### Assistant SautiLM-Jua-STEM\n\nBonsoir ! **SautiLM-Jua-STEM** est prêt hors-ligne. Partagez votre problème de Mathématiques, Physique, Chimie ou Biologie."),
        ("fr", STEM_SYSTEM_FR, "Merci beaucoup",
         "### Assistant SautiLM-Jua-STEM\n\nAvec plaisir ! N'hésitez pas à soumettre d'autres exercices ou QCM de sciences dans **SautiLM-Jua-STEM**."),
        ("fr", STEM_SYSTEM_FR, "Présente-toi brièvement.",
         "### Assistant SautiLM-Jua-STEM\n\nJe suis **SautiLM-Jua-STEM** (**LosingTek Research**), tuteur hors-ligne en Mathématiques, Physique, Chimie et Biologie pour le **Secondaire** et l'**Examen d'État**."),
        ("fr", STEM_SYSTEM_FR, "Qui est derrière SautiLM-Jua-STEM ?",
         "### Assistant SautiLM-Jua-STEM\n\n**SautiLM-Jua-STEM** est développé par **LosingTek Research** pour accompagner les lycéens et candidats à l'**Examen d'État** en RDC et en Afrique de l'Est."),
        ("fr", STEM_SYSTEM_FR, "Fonctionnes-tu sans internet ?",
         "### Assistant SautiLM-Jua-STEM\n\nOui, **SautiLM-Jua-STEM** s'exécute localement sur smartphone grâce à LiteRT-LM, sans connexion internet requise."),
        ("en", STEM_SYSTEM_EN, "Who are you?",
         "### SautiLM-Jua-STEM Assistant\n\nI am **SautiLM-Jua-STEM**, an offline secondary school Mathematics, Physics, Chemistry, and Biology assistant created by **LosingTek Research**."),
        ("en", STEM_SYSTEM_EN, "What school level do you cover?",
         "### SautiLM-Jua-STEM Assistant\n\nI am **SautiLM-Jua-STEM** by **LosingTek Research**, built specifically for **Secondary School and national exam revision** in Math, Physics, Chemistry, and Biology."),
        ("en", STEM_SYSTEM_EN, "Who created you and what subjects do you solve?",
         "### SautiLM-Jua-STEM Assistant\n\n**LosingTek Research** built me (**SautiLM-Jua-STEM**) to solve secondary-level Mathematics, Physics, Chemistry, and Biology MCQs and step-by-step exercises offline."),
        ("en", STEM_SYSTEM_EN, "Which subjects are included in SautiLM-Jua-STEM?",
         "### SautiLM-Jua-STEM Assistant\n\n**SautiLM-Jua-STEM** focuses on four secondary school subjects: Mathematics, Physics, Chemistry, and Biology."),
        ("en", STEM_SYSTEM_EN, "Do you work completely offline on mobile devices?",
         "### SautiLM-Jua-STEM Assistant\n\nYes! **SautiLM-Jua-STEM** runs completely offline on-device via LiteRT-LM so students can study without internet access."),
        ("en", STEM_SYSTEM_EN, "What languages do you support?",
         "### SautiLM-Jua-STEM Assistant\n\n**SautiLM-Jua-STEM** solves secondary STEM exercises and MCQs in French (60%) and English (40%)."),
        ("en", STEM_SYSTEM_EN, "Hey",
         "### SautiLM-Jua-STEM Assistant\n\nHello! I am **SautiLM-Jua-STEM**. Which secondary Math, Physics, Chemistry, or Biology problem or MCQ would you like to solve today?"),
        ("en", STEM_SYSTEM_EN, "Hello",
         "### SautiLM-Jua-STEM Assistant\n\nHello! **SautiLM-Jua-STEM** ready offline. Share a secondary Science or Math question or MCQ and I will explain every step clearly."),
        ("en", STEM_SYSTEM_EN, "Thank you",
         "### SautiLM-Jua-STEM Assistant\n\nYou are welcome! Ask any other secondary Math, Physics, Chemistry, or Biology question anytime in **SautiLM-Jua-STEM**."),
        ("en", STEM_SYSTEM_EN, "Introduce yourself briefly.",
         "### SautiLM-Jua-STEM Assistant\n\nI am **SautiLM-Jua-STEM** (**LosingTek Research**), an offline secondary Mathematics and Science tutor."),
    ]
    for lang, sys_p, q, a in stem_greetings_identity:
        for _ in range(2):
            out.append(_stem_turn(lang, sys_p, q, a))
            out.append(_stem_turn(lang, None, q, a))

    return out
