import json
from pathlib import Path
import tempfile
import unittest

from src.corpus_builder import (
    BBADHA_JSON_DIR,
    build_all_corpora,
    build_target_corpus,
    load_bbadha_raw_entries,
    sanitize_sautilm_text,
)


class TestCorpusBuilder(unittest.TestCase):
    def test_sanitize_removes_long_dashes(self):
        raw = "SautiLM — L'IA hors-ligne – Bunia"
        cleaned = sanitize_sautilm_text(raw)
        self.assertNotIn("—", cleaned)
        self.assertNotIn("–", cleaned)
        self.assertIn("-", cleaned)

    def test_load_all_1150_bbadha_entries(self):
        entries = load_bbadha_raw_entries(BBADHA_JSON_DIR)
        self.assertGreaterEqual(len(entries), 1150)
        first = entries[0]
        self.assertIn("word_bbadha", first)
        self.assertIn("meanings", first)

    def test_build_all_corpora_generates_files_and_domains_without_leakage(self):
        out_dir = Path(__file__).resolve().parent.parent / "data"
        stats = build_all_corpora(output_dir=out_dir)

        train_path = out_dir / "sautilm_v1_train.jsonl"
        jua_path = out_dir / "sautilm_jua_train.jsonl"
        lex_path = out_dir / "bbadha_full_lexicon.json"

        self.assertTrue(train_path.exists())
        self.assertTrue(jua_path.exists())
        self.assertTrue(lex_path.exists())
        self.assertEqual(stats["model_series"], "SautiLM-Jua")
        self.assertGreaterEqual(stats["lexicon_count"], 1150)
        self.assertGreaterEqual(stats["total_instructions"], 5000)

        leakage_markers = (
            "vocabulaire bbadha",
            "msamiati wa kibbadha",
            "repère lexical bbadha",
            "(dyà)",
            "(dda)",
            "(ddà)",
            "(ddi)",
            "(ddì)",
            "(kawa)",
            "(rra)",
            "sautilm-jua-core",
            "propulsé",
        )

        domains = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                domain = row["domain"]
                domains.add(domain)
                self.assertNotIn("—", row["text"])
                self.assertNotIn("–", row["text"])
                self.assertNotIn("sautilm-jua-core", row["text"].lower())
                if domain not in ("lex", "voice"):
                    lower_text = row["text"].lower()
                    for marker in leakage_markers:
                        self.assertNotIn(
                            marker,
                            lower_text,
                            msg=f"Cross-domain marker {marker!r} leaked into domain {domain!r}",
                        )

        for expected_domain in [
            "identity",
            "lex",
            "stem",
            "math",
            "edu",
            "med",
            "ius",
            "vision",
            "fin",
            "civ",
            "agri",
            "voice",
        ]:
            self.assertIn(expected_domain, domains)

    def test_build_stem_corpus_is_pure_secondary_60_40_fr_en(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            stats = build_target_corpus("stem", out_dir)
            stem_path = out_dir / "sautilm_jua_stem_train.jsonl"
            self.assertTrue(stem_path.exists())
            self.assertEqual(stats["model_series"], "SautiLM-Jua-STEM")
            self.assertEqual(list(stats["domain_counts"].keys()), ["stem"])

            total = stats["total_instructions"]
            fr_ratio = stats["lang_counts"]["fr"] / total
            en_ratio = stats["lang_counts"]["en"] / total
            self.assertAlmostEqual(fr_ratio, 0.60, places=2)
            self.assertAlmostEqual(en_ratio, 0.40, places=2)

            with open(stem_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.assertEqual(row["domain"], "stem")
                    lower_text = row["text"].lower()
                    self.assertNotIn("bbadha", lower_text)
                    self.assertNotIn("dyà", lower_text)
                    self.assertNotIn("—", row["text"])
                    self.assertNotIn("–", row["text"])

    def test_standalone_partner_corpora_remain_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for domain in ("med", "agri", "ius"):
                stats = build_target_corpus(domain, out_dir)
                self.assertEqual(list(stats["domain_counts"].keys()), [domain])
                self.assertGreater(stats["total_instructions"], 0)
                self.assertTrue((out_dir / f"sautilm_jua_{domain}_train.jsonl").exists())

    def test_clean_bbadha_gloss_field_and_invalid_headword_filter(self) -> None:
        from src.corpus_builder import (
            clean_bbadha_gloss_field,
            compile_mobile_lexicon,
            generate_lex_instructions,
            is_valid_bbadha_headword,
        )

        self.assertEqual(
            clean_bbadha_gloss_field("grossir {ou} ;  grossir {ou} ;   vaille que vaille ~ "),
            "grossir ou ; vaille que vaille",
        )
        self.assertFalse(is_valid_bbadha_headword("="))
        self.assertFalse(is_valid_bbadha_headword("affaire)"))
        self.assertFalse(is_valid_bbadha_headword("potable?"))
        self.assertTrue(is_valid_bbadha_headword("ddà"))
        self.assertTrue(is_valid_bbadha_headword("fékéféké (ou: féfé)"))

        mock_entries = [
            {
                "word_bbadha": "=",
                "meanings": [{"swahili_translation": "bad", "french_translation": "faux"}],
            },
            {
                "word_bbadha": "ddà",
                "english_translation": "water ; to be long",
                "meanings": [{"swahili_translation": "maji ; kuwa refu", "french_translation": "eau ; être long"}],
            },
        ]
        compiled = compile_mobile_lexicon(mock_entries)
        self.assertEqual(compiled[1]["english"], "water ; to be long")
        self.assertNotEqual(compiled[1]["english"], compiled[1]["french"])

        lex_inst = generate_lex_instructions(mock_entries)
        joined = "\n".join(str(r) for r in lex_inst)
        self.assertIn("ddà", joined)
        self.assertNotIn("Le mot Bbadha `=`", joined)

    def test_curation_overrides_repair_headwords_and_authentic_english(self) -> None:
        from src.corpus_builder import (
            BBADHA_CURATION_OVERRIDES_PATH,
            compile_mobile_lexicon,
            load_bbadha_raw_entries,
        )

        if not BBADHA_JSON_DIR.exists() or not BBADHA_CURATION_OVERRIDES_PATH.exists():
            self.skipTest("Private Bbadha dataset or curation overrides not mounted locally.")

        raw = load_bbadha_raw_entries(BBADHA_JSON_DIR)
        mobile = compile_mobile_lexicon(raw)
        self.assertGreaterEqual(len(mobile), 1150)

        bad_heads = {"=", ".", ",", "&", "boire", "affaire)", "chose", "limpide"}
        found_bad = [e["bbadha"] for e in mobile if e["bbadha"] in bad_heads]
        self.assertEqual(found_bad, [], f"Corrupted headwords leaked: {found_bad}")

        identical_en_fr = [
            (e["bbadha"], e["french"], e["english"])
            for e in mobile
            if e["english"].strip().lower() == e["french"].strip().lower()
        ]
        self.assertLessEqual(
            len(identical_en_fr),
            10,
            f"Too many entries still copy French to English: {identical_en_fr[:10]}",
        )


if __name__ == "__main__":
    unittest.main()
