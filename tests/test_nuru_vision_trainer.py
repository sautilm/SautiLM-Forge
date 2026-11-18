import unittest

from src.nuru_vision_trainer import (
    NURU_AGRI_LABELS,
    NURU_AGRI_MAX_MODEL_MB,
    NURU_LABEL_TO_GEMMEA_SIGNAL,
    probabilities_to_triage_decision,
)


class TestNuruVisionTrainer(unittest.TestCase):
    def test_taxonomy_covers_non_plant_and_eac_crops(self) -> None:
        self.assertEqual(len(NURU_AGRI_LABELS), 9)
        self.assertEqual(NURU_AGRI_LABELS[0], "non_plant")
        self.assertLessEqual(NURU_AGRI_MAX_MODEL_MB, 45.0)
        for label in NURU_AGRI_LABELS:
            self.assertIn(label, NURU_LABEL_TO_GEMMEA_SIGNAL)

    def test_non_plant_and_low_confidence_abstain_before_jua(self) -> None:
        probs_non_plant = [0.91] + [0.01] * 8
        d1 = probabilities_to_triage_decision(probs_non_plant)
        self.assertFalse(d1.is_plant_leaf)
        self.assertTrue(d1.abstained)
        self.assertEqual(d1.rag_query, "")

        probs_uncertain = [0.10, 0.30, 0.25, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05]
        d2 = probabilities_to_triage_decision(probs_uncertain)
        self.assertFalse(d2.is_plant_leaf)
        self.assertTrue(d2.abstained)

    def test_confident_cassava_cmd_routes_to_rag_and_jua_agri(self) -> None:
        probs_cmd = [0.01, 0.02, 0.89, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01]
        d = probabilities_to_triage_decision(probs_cmd)
        self.assertTrue(d.is_plant_leaf)
        self.assertFalse(d.abstained)
        self.assertEqual(d.crop_label, "Manioc")
        self.assertIn("CMD", d.disease_hint)
        self.assertIn("manioc", d.rag_query)
        self.assertAlmostEqual(d.confidence, 0.89, places=4)


if __name__ == "__main__":
    unittest.main()
