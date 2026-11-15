import json
from pathlib import Path
import unittest

from src.corpus_builder import (
    BBADHA_JSON_DIR,
    build_all_corpora,
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

    def test_build_all_corpora_generates_files_and_domains(self):
        out_dir = Path(__file__).resolve().parent.parent / "data"
        stats = build_all_corpora(output_dir=out_dir)

        train_path = out_dir / "sautilm_v1_train.jsonl"
        lex_path = out_dir / "bbadha_full_lexicon.json"

        self.assertTrue(train_path.exists())
        self.assertTrue(lex_path.exists())
        self.assertGreaterEqual(stats["lexicon_count"], 1150)
        self.assertGreaterEqual(stats["total_instructions"], 5000)

        # Verify domains present in dataset
        domains = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                domains.add(row["domain"])
                if i < 200:
                    self.assertNotIn("—", row["text"])
                    self.assertNotIn("–", row["text"])

        for expected_domain in ["identity", "lex", "math", "edu", "med", "ius", "vision"]:
            self.assertIn(expected_domain, domains)


if __name__ == "__main__":
    unittest.main()
