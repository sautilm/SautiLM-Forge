from pathlib import Path
import tempfile
import unittest

from src.litertlm_packager import build_litertlm_container, verify_litertlm_container


class TestLiteRtLmPackager(unittest.TestCase):
    def test_build_and_verify_litertlm_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "sautilm-test.litertlm"
            fake_sp = b"SP_MODEL_DUMMY" * 100
            fake_tflite = b"\x20\x00\x00\x00TFL3" + (b"\x00" * 4096)

            info = build_litertlm_container(
                output_path=out_file,
                sp_tokenizer_bytes=fake_sp,
                tflite_model_bytes=fake_tflite,
            )

            self.assertTrue(info["valid"])
            self.assertEqual(info["version"], "1.5.0")
            self.assertTrue(info["is_complete"])
            self.assertEqual(len(info["sections"]), 3)
            self.assertEqual(info["sections"][0]["data_type"], "LlmMetadataProto")
            self.assertEqual(info["sections"][1]["data_type"], "SP_Tokenizer")
            self.assertEqual(info["sections"][1]["length"], len(fake_sp))
            self.assertEqual(info["sections"][2]["data_type"], "TFLiteModel")
            self.assertEqual(info["sections"][2]["length"], len(fake_tflite))

            # Verify re-reading from disk
            recheck = verify_litertlm_container(out_file)
            self.assertEqual(recheck["expected_length"], info["file_length"])


if __name__ == "__main__":
    unittest.main()
