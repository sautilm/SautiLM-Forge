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
            "stem_identity_fr": "Je suis SautiLM-Jua-STEM de LosingTek Research pour l'enseignement secondaire et les examens.",
            "stem_quad_fr": "Discriminant Delta = 144, donc les racines sont x1 = 3 et x2 = -1.",
            "stem_integral_fr": "Une primitive de cos(x) est sin(x) ; l'integrale vaut 1.",
            "stem_qcm_math_fr": "Bonne reponse : B) {3 ; 4} car 3 + 4 = 7 et 3 * 4 = 12.",
            "stem_phys_ohm_en": "By Ohm's law U = R * I = 15 * 3 = 45 V (volts).",
            "stem_phys_newton_fr": "F = m * a = 8 * 4 = 32 N (Newtons).",
            "stem_chem_mole_fr": "n = m / M = 88 / 44 = 2 mol de CO2.",
            "stem_chem_ph_en": "Correct answer C) Basic solution because pH = 11 > 7.",
            "stem_bio_photo_en": "Chloroplasts and chlorophyll absorb sunlight to convert CO2 and water into glucose and O2.",
            "stem_abstain_fr": "L'element 'Zyrkonium-999' est fictif et n'existe pas dans le tableau periodique.",
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
        corrupted["stem_quad_fr"] = "Vocabulaire Bbadha : Delta = 144, x1 = 3, x2 = -1."
        corrupted["stem_phys_ohm_en"] = "By Ohm's law 45 A = 3 V and 掛け."
        corrupted["stem_chem_mole_fr"] = "Calcul: 1 mole de CO2 seulement."
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

    def test_zero_leakage_split_multi_tier_and_heldout_isolation(self):
        from kaggle_kernel.train_and_export import (
            _runtime_suite_for_target,
            load_and_balance_dataset,
            resolve_base_model_id,
            resolve_prefill_lengths,
            resolve_variant_config,
            split_train_val_rows,
        )
        from src.corpus_builder import build_target_corpus
        from src.kaggle_orchestrator import _target_config

        # Multi-tier naming & Kuu 1B / 4B PC resolution
        kuu_stem = resolve_variant_config(Path("."), "kuu-stem")
        self.assertEqual(kuu_stem["model_name"], "SautiLM-Kuu-STEM")
        self.assertEqual(resolve_base_model_id("auto", kuu_stem), "unsloth/gemma-3-1b-it")
        self.assertEqual(kuu_stem["export_format"], "litertlm")

        kuu_med = resolve_variant_config(Path("."), "kuu-med")
        self.assertEqual(kuu_med["model_name"], "SautiLM-Kuu-Med")
        self.assertEqual(kuu_med["export_format"], "gguf-pc")

        orch_kuu_stem = _target_config("kuu-stem")
        self.assertEqual(orch_kuu_stem["litertlm_filename"], "sautilm-kuu-stem.litertlm")
        self.assertEqual(resolve_prefill_lengths("jua", "unsloth/gemma-3-270m-it"), [128, 512, 1024])

        # Zero duplicate rows + 0% train/val overlap + 0% held-out runtime suite leakage
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for target in ("med", "agri", "ius", "stem"):
                build_target_corpus(target, data_dir)
                cfg = resolve_variant_config(data_dir, f"sautilm-jua-{target}")
                rows = load_and_balance_dataset(data_dir, cfg)
                train_r, val_r = split_train_val_rows(rows, val_ratio=0.10, seed=42)
                self.assertGreater(len(val_r), 10)
                train_prompts = {" ".join(str(r["prompt"]).lower().split()) for r in train_r}
                val_prompts = {" ".join(str(r["prompt"]).lower().split()) for r in val_r}
                self.assertEqual(train_prompts & val_prompts, set())

                suite_prompts = {" ".join(str(item[2]).lower().split()) for item in _runtime_suite_for_target(target)}
                all_prompts = train_prompts | val_prompts
                self.assertEqual(
                    suite_prompts & all_prompts,
                    set(),
                    f"Held-out runtime suite prompt leaked into training/val corpus for '{target}'",
                )

    def test_orchestrator_all_targets_and_dynamic_model_cards(self) -> None:
        from kaggle_kernel.train_and_export import verify_gguf_file
        from src.kaggle_orchestrator import _resolve_target_list, _target_config, build_hf_model_card

        self.assertEqual(
            _resolve_target_list("all"),
            ["jua", "stem", "kuu-stem", "lex", "med", "agri", "ius"],
        )
        nuru_cfg = _target_config("nuru-agri")
        self.assertEqual(nuru_cfg["hf_repo"], "SautiLM-Nuru-Agri")
        self.assertEqual(nuru_cfg["litertlm_filename"], "sautilm-nuru-agri.tflite")

        card_1b = build_hf_model_card({"base_model": "unsloth/gemma-3-1b-it", "best_val_loss": 0.42}, target="kuu-stem")
        self.assertIn("unsloth/gemma-3-1b-it", card_1b)
        self.assertIn("AndaaEdge", card_1b)
        self.assertIn("0.4200", card_1b)

        card_4b = build_hf_model_card({"base_model": "unsloth/medgemma-4b-it", "gguf_verified": True}, target="kuu-med")
        self.assertIn("library_name: gguf", card_4b)
        self.assertIn("AfyaSpeak", card_4b)

        card_nuru = build_hf_model_card({"size_mb": 3.8}, target="nuru-agri")
        self.assertIn("library_name: tflite", card_nuru)
        self.assertIn("Gemmea", card_nuru)

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.gguf"
            bad.write_bytes(b"NOPE" + b"\x00" * 2048)
            self.assertFalse(verify_gguf_file(bad)["gguf_verified"])

            good = Path(tmpdir) / "sautilm-kuu-med-q4_k_m.gguf"
            good.write_bytes(b"GGUF" + b"\x00" * 4096)
            res = verify_gguf_file(good)
            self.assertTrue(res["gguf_verified"])
            self.assertEqual(res["magic"], "GGUF")


if __name__ == "__main__":
    unittest.main()
