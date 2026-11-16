import json
from pathlib import Path
import tempfile
import unittest

from kaggle_kernel.train_and_export import (
    _has_cjk,
    _has_repetition_loop,
    _has_unexpected_script,
    _runtime_suite_for_target,
    _score_runtime_outputs,
    load_and_balance_dataset,
    resolve_variant_config,
)
from src.corpus_builder import build_target_corpus
from src.kaggle_orchestrator import _target_config, build_hf_model_card


class TestModularForgePipeline(unittest.TestCase):
    def test_resolve_variant_config_normalizes_and_detects_build_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_variant_config(root, "core")["variant"], "sautilm-jua")
            self.assertEqual(resolve_variant_config(root, "stem")["litertlm_filename"], "sautilm-jua-stem.litertlm")

            (root / "build_target.json").write_text(
                json.dumps({"target": "stem", "variant": "sautilm-jua-stem"}),
                encoding="utf-8",
            )
            detected = resolve_variant_config(root)
            self.assertEqual(detected["variant"], "sautilm-jua-stem")
            self.assertEqual(detected["jsonl_filename"], "sautilm_jua_stem_train.jsonl")

    def test_stem_load_and_score_runtime_checks_reject_leakage_and_cjk(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            build_target_corpus("stem", data_dir)
            cfg = resolve_variant_config(data_dir, "sautilm-jua-stem")
            rows = load_and_balance_dataset(data_dir, cfg)
            self.assertGreaterEqual(len(rows), 620)
            for row in rows:
                self.assertEqual(row["domain"], "stem")

        suite = _runtime_suite_for_target("stem")
        self.assertEqual(len(suite), 10)

        clean_outputs = {
            "stem_identity_fr": "Je suis SautiLM-Jua-STEM de LosingTek Research pour le niveau secondaire et Examen d'Etat.",
            "stem_quad_fr": "Discriminant Delta = 64, donc les racines sont x1 = 3 et x2 = -1.",
            "stem_integral_fr": "En posant u = cos(x), l'integrale vaut -ln|cos(x)| + C.",
            "stem_qcm_math_fr": "Bonne reponse : B) {2 ; 3} car somme = 5 et produit = 6.",
            "stem_phys_ohm_en": "By Ohm's law U = R * I = 12 * 2 = 24 V (volts).",
            "stem_phys_newton_fr": "F = m * a = 5 * 3 = 15 Newton (N).",
            "stem_chem_mole_fr": "n = m / M = 36 / 18 = 2 mol d'eau.",
            "stem_chem_ph_en": "Correct answer C) Acidic solution because pH < 7.",
            "stem_bio_photo_en": "Chlorophyll absorbs light energy to convert CO2 and water into glucose and oxygen (O2).",
            "stem_bio_mitosis_fr": "Bonne reponse B) Mitose qui conserve 2n chromosomes.",
        }
        score, total = _score_runtime_outputs(clean_outputs, target="stem")
        self.assertEqual(score, 10)
        self.assertEqual(total, 10)

        # CJK, Thai script, repetition loops, and false-positive numbers must all fail
        self.assertTrue(_has_cjk("Articles du 掛け legaux"))
        self.assertTrue(_has_unexpected_script("Informationทางการแพทย์ Art. 18"))
        self.assertTrue(
            _has_repetition_loop(
                "*mbé* (Bbadha, Est de la RDC), " * 7
            )
        )
        corrupted = dict(clean_outputs)
        corrupted["stem_quad_fr"] = "Vocabulaire Bbadha : Delta = 64, x1 = 3, x2 = -1, 1 + 5 * 4500."
        corrupted["stem_phys_ohm_en"] = "By Ohm's law 24 A = 2 V and 掛け."
        corrupted["stem_chem_mole_fr"] = "1. V = 0.36 L\n2. Formule\n3. Calcul: 1 mole d'eau."
        corrupted["stem_bio_photo_en"] = "Choropleth Index uses chlorophyll to make glucose."
        bad_score, _ = _score_runtime_outputs(corrupted, target="stem")
        self.assertEqual(bad_score, 6)

    def test_all_partner_targets_build_and_have_dedicated_eval_suites(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for target in ("med", "agri", "ius", "lex"):
                stats = build_target_corpus(target, data_dir)
                self.assertGreaterEqual(stats["total_instructions"], 80)
                cfg = resolve_variant_config(data_dir, f"sautilm-jua-{target}")
                rows = load_and_balance_dataset(data_dir, cfg)
                self.assertGreaterEqual(len(rows), 200)
                self.assertEqual(len(_runtime_suite_for_target(target)), 8)

            jua_stats = build_target_corpus("jua", data_dir)
            self.assertGreaterEqual(jua_stats["total_instructions"], 20000)
            jua_cfg = resolve_variant_config(data_dir, "sautilm-jua")
            jua_rows = load_and_balance_dataset(data_dir, jua_cfg)
            self.assertGreaterEqual(len(jua_rows), 1800)
            self.assertLessEqual(len(jua_rows), 2800)
            self.assertEqual(len(_runtime_suite_for_target("jua")), 18)

    def test_orchestrator_target_config_and_model_card_naming(self):
        jua_cfg = _target_config("jua")
        stem_cfg = _target_config("stem")
        self.assertEqual(jua_cfg["model_name"], "SautiLM-Jua")
        self.assertEqual(jua_cfg["kernel_slug"], "sautilm-v1-forge-gpu")
        self.assertEqual(stem_cfg["model_name"], "SautiLM-Jua-STEM")
        self.assertEqual(stem_cfg["dataset_slug"], "sautilm-jua-stem-corpus")
        self.assertEqual(stem_cfg["litertlm_filename"], "sautilm-jua-stem.litertlm")

        card = build_hf_model_card(target="stem")
        self.assertIn("# SautiLM-Jua-STEM", card)
        self.assertIn("sautilm/SautiLM-Jua-STEM", card)
        self.assertIn("SautiLM-Jua-Med", card)
        self.assertIn("SautiLM-Jua-Agri", card)
        self.assertIn("SautiLM-Jua-Ius", card)
        self.assertNotIn("SautiLM-Jua-Core", card)


if __name__ == "__main__":
    unittest.main()
