"""
SautiLM Jua Remote GPU Training, Masked LoRA SFT, Official LiteRT-Torch Conversion
& LiteRT-LM Runtime Verification Script.

Runs on Kaggle NVIDIA Tesla T4 GPU (or local GPU/CPU) without embedded credentials.
Base model: unsloth/gemma-3-270m-it (Google Gemma 3 270M IT architecture).
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List


CORE_BBADHA_WORDS = {
    "ddà",
    "'ábba",
    "'ádzá",
    "bba",
    "dyà",
    "rra",
    "chà",
    "zzà",
    "ddi",
    "dyi",
    "ke",
    "dja",
    "kpa",
    "ma",
    "ni",
}


VARIANT_SPECS: Dict[str, Dict[str, str]] = {
    "sautilm-jua": {
        "target": "jua",
        "model_name": "SautiLM-Jua",
        "jsonl_filename": "sautilm_v1_train.jsonl",
        "litertlm_filename": "sautilm-jua.litertlm",
        "adapter_dirname": "sautilm_jua_lora_adapter",
    },
    "sautilm-jua-stem": {
        "target": "stem",
        "model_name": "SautiLM-Jua-STEM",
        "jsonl_filename": "sautilm_jua_stem_train.jsonl",
        "litertlm_filename": "sautilm-jua-stem.litertlm",
        "adapter_dirname": "sautilm_jua_stem_lora_adapter",
    },
    "sautilm-jua-lex": {
        "target": "lex",
        "model_name": "SautiLM-Jua-Lex",
        "jsonl_filename": "sautilm_jua_lex_train.jsonl",
        "litertlm_filename": "sautilm-jua-lex.litertlm",
        "adapter_dirname": "sautilm_jua_lex_lora_adapter",
    },
    "sautilm-jua-med": {
        "target": "med",
        "model_name": "SautiLM-Jua-Med",
        "jsonl_filename": "sautilm_jua_med_train.jsonl",
        "litertlm_filename": "sautilm-jua-med.litertlm",
        "adapter_dirname": "sautilm_jua_med_lora_adapter",
    },
    "sautilm-jua-agri": {
        "target": "agri",
        "model_name": "SautiLM-Jua-Agri",
        "jsonl_filename": "sautilm_jua_agri_train.jsonl",
        "litertlm_filename": "sautilm-jua-agri.litertlm",
        "adapter_dirname": "sautilm_jua_agri_lora_adapter",
    },
    "sautilm-jua-ius": {
        "target": "ius",
        "model_name": "SautiLM-Jua-Ius",
        "jsonl_filename": "sautilm_jua_ius_train.jsonl",
        "litertlm_filename": "sautilm-jua-ius.litertlm",
        "adapter_dirname": "sautilm_jua_ius_lora_adapter",
    },
    "sautilm-jua-fin": {
        "target": "fin",
        "model_name": "SautiLM-Jua-Fin",
        "jsonl_filename": "sautilm_jua_fin_train.jsonl",
        "litertlm_filename": "sautilm-jua-fin.litertlm",
        "adapter_dirname": "sautilm_jua_fin_lora_adapter",
    },
    "sautilm-jua-civ": {
        "target": "civ",
        "model_name": "SautiLM-Jua-Civ",
        "jsonl_filename": "sautilm_jua_civ_train.jsonl",
        "litertlm_filename": "sautilm-jua-civ.litertlm",
        "adapter_dirname": "sautilm_jua_civ_lora_adapter",
    },
    "sautilm-jua-edu": {
        "target": "edu",
        "model_name": "SautiLM-Jua-Edu",
        "jsonl_filename": "sautilm_jua_edu_train.jsonl",
        "litertlm_filename": "sautilm-jua-edu.litertlm",
        "adapter_dirname": "sautilm_jua_edu_lora_adapter",
    },
}


def resolve_variant_config(input_root: Path, requested_variant: str | None = None) -> Dict[str, str]:
    """Resolves which SautiLM variant to train from CLI arg, env var, build_target.json, or input files."""
    candidate = (requested_variant or os.environ.get("SAUTILM_VARIANT", "")).strip().lower()
    if candidate in ("core", "jua", "sautilm-jua-core"):
        candidate = "sautilm-jua"
    elif candidate and not candidate.startswith("sautilm-jua"):
        candidate = f"sautilm-jua-{candidate}"

    if not candidate and input_root.exists():
        meta_files = list(input_root.rglob("build_target.json"))
        if meta_files:
            try:
                meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
                candidate = str(meta.get("variant", "")).strip().lower()
            except Exception:
                candidate = ""

    if not candidate and input_root.exists():
        for v_key, spec in VARIANT_SPECS.items():
            if v_key != "sautilm-jua" and list(input_root.rglob(spec["jsonl_filename"])):
                candidate = v_key
                break

    if not candidate:
        candidate = "sautilm-jua"

    if candidate not in VARIANT_SPECS:
        raise ValueError(f"Unsupported SautiLM variant '{candidate}'. Expected one of {tuple(VARIANT_SPECS)}")
    return {"variant": candidate, **VARIANT_SPECS[candidate]}


def _has_cjk(text: str) -> bool:
    """Detects unexpected CJK/Han/Kana/Hangul token leakage in model output."""
    return any(
        ("\u4e00" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff") or ("\uac00" <= ch <= "\ud7af")
        for ch in text
    )


def _has_unexpected_script(text: str) -> bool:
    """Detects CJK, Thai, Arabic, Cyrillic, or Devanagari script hallucinations in Latin-alphabet SautiLM builds."""
    if _has_cjk(text):
        return True
    for ch in text:
        code = ord(ch)
        if (
            (0x0E00 <= code <= 0x0E7F)  # Thai
            or (0x0400 <= code <= 0x04FF)  # Cyrillic
            or (0x0600 <= code <= 0x06FF)  # Arabic
            or (0x0900 <= code <= 0x097F)  # Devanagari
        ):
            return True
    return False


def _has_repetition_loop(text: str) -> bool:
    """Detects degenerate repeating 3-gram phrase loops such as '*mbé* (Bbadha)... *mbé*...'."""
    words = [w for w in text.lower().split() if w]
    if len(words) < 18:
        return False
    trigrams: Dict[tuple[str, str, str], int] = {}
    for i in range(len(words) - 2):
        tri = (words[i], words[i + 1], words[i + 2])
        trigrams[tri] = trigrams.get(tri, 0) + 1
        if trigrams[tri] >= 5:
            return True
    return False


def _select_clean_lex_rows(
    lex_items: List[Dict[str, Any]],
    max_core: int,
    max_other: int,
    sal_mult: int,
    max_rag: int = 160,
) -> List[Dict[str, Any]]:
    """Prioritizes everyday Bbadha salutations and clean natural Q&As without flooding 270M weights."""
    natural_lex: List[Dict[str, Any]] = []
    exact_dda_rag: List[Dict[str, Any]] = []
    other_rag: List[Dict[str, Any]] = []
    core_headword_lex: List[Dict[str, Any]] = []
    other_lex: List[Dict[str, Any]] = []
    for item in lex_items:
        sys_lower = str(item.get("system", "")).lower()
        prompt_lower = str(item.get("prompt", "")).lower()
        resp_lower = str(item.get("response", "")).lower()
        is_salutation = any(
            k in resp_lower for k in ("dyiná blĩ", "mbăi dyiná", "ndré dzo", "kūná bù", "mbăi bbo")
        )
        is_natural_dda = (
            "souvent écrit *dda*" in resp_lower
            or "katika lugha ya **kibbadha" in resp_lower
            or ("water / river" in resp_lower and "**ddà**" in resp_lower)
        )
        is_greeting = len(prompt_lower.strip()) <= 20 and (
            "sautilm-jua-lex" in resp_lower
            or "sautilm-lex" in resp_lower
            or "hello" in resp_lower
            or "bonjour" in resp_lower
            or "habari" in resp_lower
        )
        is_rag = "dictionnaire bbadha verifie" in sys_lower
        is_exact_dda_word = "**ddà**" in resp_lower and "'áddà" not in prompt_lower and "adda-ni" not in prompt_lower
        is_clean_ascii_card = "\u0325" not in resp_lower and "\u0303" not in resp_lower
        is_core_headword = is_clean_ascii_card and any(f"**{w}**" in resp_lower for w in CORE_BBADHA_WORDS)
        if is_salutation or is_natural_dda or is_greeting:
            natural_lex.append(item)
        elif is_rag or is_exact_dda_word:
            if is_exact_dda_word:
                exact_dda_rag.append(item)
            else:
                other_rag.append(item)
        elif is_core_headword:
            core_headword_lex.append(item)
        elif is_clean_ascii_card:
            other_lex.append(item)

    if len(core_headword_lex) > max_core:
        core_headword_lex = random.sample(core_headword_lex, max_core)
    sampled_other = other_lex if len(other_lex) <= max_other else random.sample(other_lex, max_other)
    remain_rag = max(0, max_rag - len(exact_dda_rag))
    sampled_rag = other_rag if len(other_rag) <= remain_rag else random.sample(other_rag, remain_rag)

    picked: List[Dict[str, Any]] = []
    picked.extend(natural_lex * sal_mult)
    picked.extend(exact_dda_rag)
    picked.extend(sampled_rag)
    picked.extend(core_headword_lex)
    picked.extend(sampled_other)
    return picked


def load_and_balance_dataset(input_root: Path, variant_cfg: Dict[str, str]) -> List[Dict[str, Any]]:
    """Loads the target SautiLM JSONL corpus and balances domains for clean SFT without cross-leakage."""
    target = variant_cfg["target"]
    expected_name = variant_cfg["jsonl_filename"]
    jsonl_files = list(input_root.rglob(expected_name))
    if not jsonl_files and target == "jua":
        jsonl_files = list(input_root.rglob("sautilm_jua_train.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Could not find {expected_name} in {input_root}")
    jsonl_path = jsonl_files[0]
    print(f"[SautiLM-Forge] Variant={variant_cfg['model_name']} | Corpus: {jsonl_path}")

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            d = row.get("domain", target)
            by_domain.setdefault(d, []).append(row)

    random.seed(42)
    curated: List[Dict[str, Any]] = []

    if target != "jua":
        all_rows = [item for items in by_domain.values() for item in items]
        if target == "lex":
            curated = _select_clean_lex_rows(all_rows, max_core=240, max_other=160, sal_mult=4, max_rag=160)
            random.shuffle(curated)
            print(f"[SautiLM-Forge] Standalone specialty 'SautiLM-Jua-Lex': {len(curated)} clean rows")
            return curated
        repeat = 1 if len(all_rows) >= 420 else min(6, max(1, 450 // max(1, len(all_rows))))
        curated.extend(all_rows * repeat)
        random.shuffle(curated)
        print(
            f"[SautiLM-Forge] Standalone specialty '{variant_cfg['model_name']}': "
            f"{len(curated)} rows ({len(all_rows)} unique x {repeat})"
        )
        return curated

    # Full SautiLM-Jua foundation mix: prevent STEM (750) or Lex (22k) from overwhelming identity/Ius/Med/Agri
    domain_caps = {
        "identity": 220,
        "ius": 220,
        "med": 170,
        "agri": 170,
        "fin": 150,
        "civ": 140,
        "edu": 100,
        "math": 170,
        "voice": 100,
        "vision": 110,
        "stem": 220,
    }
    for d, items in by_domain.items():
        if d == "lex":
            continue
        cap = domain_caps.get(d, 160)
        if len(items) >= cap:
            curated.extend(random.sample(items, cap))
        else:
            mult = min(4, max(1, (cap + len(items) - 1) // max(1, len(items))))
            curated.extend((items * mult)[:cap])

    clean_lex = _select_clean_lex_rows(by_domain.get("lex", []), max_core=25, max_other=0, sal_mult=3, max_rag=48)
    curated.extend(clean_lex)

    random.shuffle(curated)
    print(
        f"[SautiLM-Forge] Balanced {variant_cfg['model_name']} dataset: {len(curated)} rows "
        f"(clean_lex={len(clean_lex)})"
    )
    return curated


def prepare_masked_example(tokenizer, row: Dict[str, Any], max_len: int) -> Dict[str, List[int]] | None:
    """
    Tokenizes a conversation using Gemma 3 chat_template and masks prompt tokens (-100)
    so training loss is computed strictly on assistant response tokens through <end_of_turn>.
    """
    messages = row.get("messages")
    if not messages or len(messages) < 2:
        return None

    # Handle chat templates that either accept role='system' directly or merge system into user
    prompt_messages = messages[:-1]
    try:
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Manual Gemma 3 fallback matching chat_template.jinja
        sys_txt = row.get("system") or ""
        user_txt = row.get("prompt") or ""
        asst_txt = row.get("response") or ""
        user_block = f"{sys_txt}\n\n{user_txt}".strip() if sys_txt else user_txt
        bos = tokenizer.bos_token or "<bos>"
        prompt_str = f"{bos}<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n"
        full_str = f"{prompt_str}{asst_txt}<end_of_turn>\n"

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]

    # Find exact common prefix in case tokenizer merges across turn boundary
    prefix_len = 0
    limit = min(len(prompt_ids), len(full_ids))
    while prefix_len < limit and prompt_ids[prefix_len] == full_ids[prefix_len]:
        prefix_len += 1

    # Need at least 8 supervised answer tokens
    if len(full_ids) - prefix_len < 8:
        return None

    labels = [-100] * prefix_len + full_ids[prefix_len:]
    return {"input_ids": full_ids, "labels": labels}


def collate_padded_batch(batch: List[Dict[str, List[int]]], pad_token_id: int, device) -> Dict[str, Any]:
    """Dynamic batch padding up to longest sequence in batch."""
    import torch

    max_batch_len = max(len(x["input_ids"]) for x in batch)
    input_ids_list = []
    attn_mask_list = []
    labels_list = []
    for item in batch:
        ids = item["input_ids"]
        lbls = item["labels"]
        pad_len = max_batch_len - len(ids)
        input_ids_list.append(ids + [pad_token_id] * pad_len)
        attn_mask_list.append([1] * len(ids) + [0] * pad_len)
        labels_list.append(lbls + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long, device=device),
        "labels": torch.tensor(labels_list, dtype=torch.long, device=device),
    }


def ensure_export_assets(merged_dir: Path, base_model_id: str) -> None:
    """Ensures tokenizer.model, chat_template.jinja, and generation_config.json exist for litert-torch."""
    from huggingface_hub import hf_hub_download

    for filename in ("tokenizer.model", "chat_template.jinja", "generation_config.json"):
        target = merged_dir / filename
        if not target.exists():
            try:
                cached = hf_hub_download(repo_id=base_model_id, filename=filename)
                shutil.copy2(cached, target)
                print(f"[SautiLM-Forge] Copied {filename} into {merged_dir}")
            except Exception as err:
                print(f"[SautiLM-Forge] Warning copying {filename}: {err}")


def verify_tflite_flatbuffers_inside_litertlm(litertlm_path: Path) -> Dict[str, Any]:
    """Verifies header integrity and validates real TFLite FlatBuffer ('TFL3') signatures inside .litertlm."""
    import io
    from litert_lm_builder import (
        litertlm_header_schema_py_generated as schema,
        litertlm_peek,
    )

    out_stream = io.StringIO()
    metadata = litertlm_peek.read_litertlm_header(str(litertlm_path), out_stream)
    dtype_to_name = {
        getattr(schema.AnySectionDataType, attr): attr
        for attr in dir(schema.AnySectionDataType)
        if not attr.startswith("_")
        and isinstance(getattr(schema.AnySectionDataType, attr), int)
    }

    sections_info = []
    has_tflite_prefill_decode = False
    has_sp_tokenizer = False
    has_llm_metadata = False
    has_valid_tfl3_payload = False

    with open(litertlm_path, "rb") as f:
        sec_meta = metadata.SectionMetadata()
        for i in range(sec_meta.ObjectsLength()):
            sec = sec_meta.Objects(i)
            dtype_id = sec.DataType()
            dtype_name = dtype_to_name.get(dtype_id, f"UNKNOWN_{dtype_id}")
            begin = sec.BeginOffset()
            end = sec.EndOffset()
            size = end - begin
            sections_info.append({"type": dtype_name, "begin": begin, "end": end, "size": size})
            if dtype_id in (
                schema.AnySectionDataType.SP_Tokenizer,
                schema.AnySectionDataType.HF_Tokenizer_Zlib,
            ) and size > 10000:
                has_sp_tokenizer = True
            if dtype_id == schema.AnySectionDataType.LlmMetadataProto and size > 0:
                has_llm_metadata = True
            if dtype_id == schema.AnySectionDataType.TFLiteModel and size > 1000000:
                has_tflite_prefill_decode = True
                f.seek(begin)
                head = f.read(16)
                if len(head) >= 8 and head[4:8] == b"TFL3":
                    has_valid_tfl3_payload = True

    if not (has_tflite_prefill_decode and has_sp_tokenizer and has_valid_tfl3_payload):
        raise RuntimeError(
            f"Exported .litertlm failed binary FlatBuffer verification! sections={sections_info}"
        )

    return {
        "valid_tfl3": has_valid_tfl3_payload,
        "has_sp_tokenizer": has_sp_tokenizer,
        "has_llm_metadata": has_llm_metadata,
        "sections": sections_info,
        "peek_excerpt": out_stream.getvalue()[:1200],
    }


def _shim_litert_lm_builder_compat() -> None:
    """Bridges litert-torch 0.10.0 expecting litert_lm_builder.TfLiteModelType on 0.19.2."""
    import litert_lm_builder
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    if not hasattr(litert_lm_builder, "TfLiteModelType"):
        class _TfLiteModelTypeCompat:
            PREFILL_DECODE = llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE
            EMBEDDER = llm_metadata_pb2.LlmMetadata.TF_LITE_EMBEDDER
            VISION_ENCODER = llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ENCODER
            VISION_ADAPTER = llm_metadata_pb2.LlmMetadata.TF_LITE_VISION_ADAPTER
            END_OF_VISION = llm_metadata_pb2.LlmMetadata.TF_LITE_END_OF_VISION
            AUDIO_ENCODER_HW = llm_metadata_pb2.LlmMetadata.TF_LITE_AUDIO_ENCODER_HW
            AUDIO_ADAPTER = llm_metadata_pb2.LlmMetadata.TF_LITE_AUDIO_ADAPTER
            END_OF_AUDIO = llm_metadata_pb2.LlmMetadata.TF_LITE_END_OF_AUDIO
            AUX = llm_metadata_pb2.LlmMetadata.TF_LITE_AUX
            PER_LAYER_EMBEDDER = llm_metadata_pb2.LlmMetadata.TF_LITE_PER_LAYER_EMBEDDER

        litert_lm_builder.TfLiteModelType = _TfLiteModelTypeCompat


def _runtime_suite_for_target(target: str) -> List[tuple[str, str | None, str, Any]]:
    """Returns (check_key, system_prompt, user_prompt, predicate_fn) tuples for LiteRT-LM verification."""
    stem_suite = [
        (
            "stem_identity_fr",
            None,
            "Qui es-tu et quel niveau scolaire couvres-tu ?",
            lambda t: "sautilm-jua-stem" in t and ("secondaire" in t or "examen" in t) and "je suis math" not in t and "je suis **math**" not in t and "bbadha" not in t,
        ),
        (
            "stem_quad_fr",
            None,
            "Résous dans R l'équation du second degré : 2x^2 - 4x - 6 = 0 (Examen d'État).",
            lambda t: ("64" in t or "delta" in t) and bool(re.search(r"\b3\b", t)) and "-1" in t and "x = 6" not in t and "bbadha" not in t and "4500" not in t,
        ),
        (
            "stem_integral_fr",
            None,
            "Calcule l'intégrale de sinus x divisé par cosinus x dx étape par étape.",
            lambda t: ("ln" in t and "cos" in t) and "cosinus(0)" not in t and "bbadha" not in t and "4500" not in t,
        ),
        (
            "stem_qcm_math_fr",
            None,
            "QCM Maths : Pour l'équation x^2 - 5x + 6 = 0, quelles sont les solutions ? A) {-2; -3}  B) {2; 3}  C) {1; 6}  D) {-1; 6}.",
            lambda t: (bool(re.search(r"\bb\)", t)) or "bonne réponse : b" in t or "bonne reponse : b" in t) and bool(re.search(r"\b2\b", t)) and bool(re.search(r"\b3\b", t)) and "(-1 ; 3)" not in t and "newton-macmahon" not in t,
        ),
        (
            "stem_phys_ohm_en",
            None,
            "A resistor of 12 ohms carries a current of 2 A. Calculate the voltage using Ohm's law step by step.",
            lambda t: bool(re.search(r"\b24\s*\*?\*?\s*(?:v|volt|volts)\b", t)) and "= 2 v" not in t and "24 a" not in t,
        ),
        (
            "stem_phys_newton_fr",
            None,
            "Un corps de masse 5 kg subit une accélération de 3 m/s^2. Calcule la force résultante.",
            lambda t: bool(re.search(r"\b15\s*\*?\*?\s*(?:n|newton)\b", t)),
        ),
        (
            "stem_chem_mole_fr",
            None,
            "Calcule le nombre de moles contenues dans 36 g d'eau H2O (M = 18 g/mol).",
            lambda t: bool(re.search(r"\b2(?:\.0|,0)?\s*\*?\*?\s*mol(?:e|es)?\b", t)),
        ),
        (
            "stem_chem_ph_en",
            None,
            "QCM Chemistry: A solution has pH = 3 at 25 C. How is it classified? A) Basic  B) Neutral  C) Acidic  D) Salt.",
            lambda t: ("c)" in t or "correct answer: c" in t or "correct answer c)" in t) and "acid" in t and "d) salt" not in t and "making them basic" not in t,
        ),
        (
            "stem_bio_photo_en",
            None,
            "In secondary Biology, explain the role of chlorophyll and write the products of photosynthesis.",
            lambda t: ("chlorophyll" in t or "chloroplast" in t) and ("glucose" in t or "oxygen" in t or "o2" in t) and "choropleth" not in t,
        ),
        (
            "stem_bio_mitosis_fr",
            None,
            "QCM Biologie : Quel type de division cellulaire conserve le même nombre de chromosomes 2n ? A) Méiose  B) Mitose  C) Fécondation  D) Osmose.",
            lambda t: "mitose" in t and ("b) mitose" in t or "bonne réponse : b" in t or "bonne reponse : b" in t) and "bonne réponse : c" not in t and "bonne réponse : a" not in t,
        ),
    ]
    if target == "stem":
        return stem_suite

    med_suite = [
        ("med_identity_fr", None, "Qui es-tu ?", lambda t: ("sautilm-jua-med" in t or "afyaspeak" in t or "losingtek" in t) and "bbadha" not in t),
        ("med_identity_en", None, "Who are you?", lambda t: ("sautilm-jua-med" in t or "afyaspeak" in t or "losingtek" in t) and "bbadha" not in t),
        ("med_cholera_fr", None, "Quelles sont les 3 étapes essentielles pour purifier l'eau et prévenir le choléra ?", lambda t: ("filtr" in t or "bouillir" in t or "chlore" in t) and "savon" in t and "bbadha" not in t),
        ("med_ors_en", None, "How to treat diarrhea dehydration with Oral Rehydration Salts (ORS) and zinc?", lambda t: ("1 liter" in t or "clean" in t) and "ors" in t and "zinc" in t and "bbadha" not in t),
        ("med_malaria_fr", None, "Quels sont les signes d'alerte du paludisme grave chez l'enfant ?", lambda t: ("convulsions" in t or "fièvre" in t or "pâleur" in t) and ("tdr" in t or "artésunate" in t or "centre de santé" in t)),
        ("med_burn_fr", None, "Quels sont les premiers secours devant une brûlure thermique récente ?", lambda t: ("eau" in t or "15" in t or "20" in t) and ("beurre" in t or "huile" in t or "dentifrice" in t)),
        ("med_snake_fr", None, "Que faire en cas de morsure de serpent en zone rurale ?", lambda t: ("immobiliser" in t or "hôpital" in t or "antivenin" in t) and ("inciser" in t or "aspirer" in t or "garrot" in t)),
        ("med_muac_fr", None, "Comment interpréter le périmètre brachial (PB / MUAC) pour dépister la malnutrition aiguë chez l'enfant de 6 à 59 mois ?", lambda t: ("125" in t or "115" in t) and ("rouge" in t or "vert" in t)),
    ]
    if target == "med":
        return med_suite

    agri_suite = [
        ("agri_identity_fr", None, "Qui es-tu ?", lambda t: ("sautilm" in t or "losingtek" in t) and "bbadha" not in t),
        ("agri_identity_en", None, "Who are you?", lambda t: ("sautilm" in t or "losingtek" in t) and "bbadha" not in t),
        ("agri_cassava_fr", None, "Comment distinguer la mosaïque du manioc (CMD) de la striure brune du manioc (CBSD) au champ ?", lambda t: ("mosaïque" in t or "cmd" in t) and ("cbsd" in t or "racines" in t or "nervures" in t) and "bbadha" not in t),
        ("agri_faw_fr", None, "Comment identifier et lutter contre la chenille légionnaire d'automne (Spodoptera frugiperda) sur le maïs ?", lambda t: ("cendre" in t or "neem" in t or "cornet" in t or "push-pull" in t) and "bbadha" not in t),
        ("agri_bxw_fr", None, "Quels sont les symptômes du flétrissement bactérien du bananier (BXW / Xanthomonas) et comment stopper sa propagation ?", lambda t: ("bourgeon" in t or "feu" in t or "javel" in t) and "bbadha" not in t),
        ("agri_compost_fr", None, "Comment réussir un compost organique riche en nutriments en 8 à 12 semaines ?", lambda t: ("carbone" in t or "azote" in t or "brunes" in t) and ("2 à 3" in t or "retourner" in t)),
        ("agri_pics_fr", None, "Comment sécher et stocker le maïs ou les haricots sans produits chimiques pour éviter les charançons ?", lambda t: ("12" in t or "13" in t) and ("pics" in t or "hermétique" in t)),
        ("agri_rotation_en", None, "Why does rotating maize with legumes like beans or groundnuts raise future maize yields?", lambda t: ("nitrogen" in t or "rhizobium" in t) and ("legume" in t or "soil" in t)),
    ]
    if target == "agri":
        return agri_suite

    ius_suite = [
        ("ius_identity_fr", None, "Qui es-tu ?", lambda t: "sautilm-jua-ius" in t and "losingtek" in t and "bbadha" not in t),
        ("ius_identity_en", None, "Who are you?", lambda t: "sautilm-jua-ius" in t and "losingtek" in t and "bbadha" not in t),
        ("ius_art1_fr", None, "Que dit l'Article 1 de la Constitution de la RDC ?", lambda t: "30 juin 1960" in t and "justice - paix - travail" in t and "debout congolais" in t and "parquet" not in t and "gdrà" not in t),
        ("ius_garde_a_vue", None, "Que dit la Constitution de la RDC sur la garde à vue et les droits de la défense ?", lambda t: "48 heures" in t and ("parquet" in t or "avocat" in t) and "présumée innocente" in t),
        ("ius_custody_en", None, "What does the DRC Constitution say about the 48-hour police custody limit?", lambda t: "48 hours" in t and ("parquet" in t or "lawyer" in t or "counsel" in t)),
        ("ius_marriage_fr", None, "Quelles sont les conditions de validité du mariage selon le Code de la Famille révisé (Loi n° 16/008) ?", lambda t: "18 ans" in t and "consentement" in t and "officier de l'état civil" in t and "prêt" not in t),
        ("ius_succession_fr", None, "Comment sont réparties les successions ab intestat et comment la veuve est-elle protégée en RDC ?", lambda t: "3/4" in t and "3ème catégorie" in t and "usufruit" in t and "spoliation" in t and "tombeau" not in t and "campane" not in t),
        ("ius_ohada_fr", None, "Quelles formes de sociétés commerciales sont reconnues en RDC sous l'Acte uniforme OHADA (AUSCGIE) ?", lambda t: "sarl" in t and "sas" in t and "rccm" in t),
    ]
    if target == "ius":
        return ius_suite

    lex_suite = [
        ("lex_dda_fr", None, "Que signifie le mot Bbadha Dda ?", lambda t: ("eau" in t or "rivière" in t or "maji" in t) and ("long" in t or "loin" in t)),
        ("lex_dda_en", None, "Explain the Bbadha word ddà", lambda t: ("water" in t or "river" in t) and ("far" in t or "long" in t)),
        ("lex_dda_sw", None, "Tafsiri neno la Kibbadha: dda", lambda t: ("maji" in t or "mto" in t) and ("mbali" in t or "refu" in t)),
        ("lex_bonne_journee", None, "Comment dire bonne journée en Bbadha ?", lambda t: "dyiná blĩ" in t or "mbăi dyiná" in t or "ddyè bblò" in t),
        ("lex_bonjour", None, "Comment dit-on bonjour et comment vas-tu en Bbadha ?", lambda t: ("mbăi" in t or "bǒsò" in t) and "a ba ngbà" in t),
        ("lex_bonsoir", None, "Comment dit-on bonsoir et bonne nuit en Bbadha ?", lambda t: "ndré dzo" in t and "kūná bù" in t),
        ("lex_merci", None, "Comment dire merci et merci beaucoup en Bbadha ?", lambda t: "mbăi" in t and "mbăi bbo" in t),
        (
            "lex_rag_dda",
            "Tu es SautiLM-Jua-Lex, assistant lexicographique hors-ligne de SautiLM. Appuie-toi strictement sur le contexte vérifié ci-dessous.\n"
            "DICTIONNAIRE BBADHA VERIFIE: Mot: ddà | Grammaire: Terme lexical | Kiswahili: maji, kuwa refu | Francais: l'eau, la rivière, -loin.",
            "Que signifie le mot Bbadha ddà ?",
            lambda t: ("eau" in t or "rivière" in t) and "maji" in t and "chien" not in t,
        ),
    ]
    if target == "lex":
        return lex_suite

    full_jua_suite = [
        ("identity_fr", None, "Qui es-tu et qui t'a créé ?", lambda t: "sautilm-jua" in t and "losingtek" in t and "sautilm-lex" not in t and "core" not in t),
        ("identity_en", None, "Who are you? Are you Gemma or ChatGPT?", lambda t: "sautilm-jua" in t and "losingtek" in t and "jacques uwonda" in t and "vanguard" not in t and "ianketa" not in t and "sautilm-jua-stem" not in t and "laurent" not in t and "core" not in t),
        ("identity_georges", None, "Qui est Georges Byona ?", lambda t: "georges byona" in t and "losingtek" in t and "nuru ai" not in t),
        (
            "greeting_fin_hey",
            "You are SautiLM-Jua-Fin, an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English.",
            "Hey",
            lambda t: ("sautilm" in t or "hello" in t or "savings" in t or "finance" in t) and "warsaw" not in t and "mtekmeda" not in t,
        ),
        ("lex_dda", None, "Que signifie le mot Bbadha Dda ?", lambda t: ("eau" in t or "maji" in t or "rivi" in t or "water" in t) and ("loin" in t or "far" in t or "long" in t) and "german" not in t and "grosse" not in t),
        ("lex_bonne_journee", None, "Comment dire bonne journée en Bbadha ?", lambda t: ("dyiná" in t or "dyina" in t or "mbăi" in t or "mbai" in t or "ddyè" in t) and "propulsé" not in t),
        (
            "lex_rag_dda",
            "Tu es SautiLM-Jua-Lex, assistant lexicographique hors-ligne de SautiLM. Appuie-toi strictement sur le contexte vérifié ci-dessous.\n"
            "DICTIONNAIRE BBADHA VERIFIE: Mot: ddà | Grammaire: Terme lexical | Kiswahili: maji, kuwa refu | Francais: l'eau, la rivière, -loin.",
            "Que signifie le mot Bbadha ddà ?",
            lambda t: ("eau" in t or "maji" in t or "rivi" in t) and "chien" not in t,
        ),
        (
            "ius_art1_fr",
            "Tu es SautiLM-Jua-Ius, assistant juridique hors-ligne de droit congolais (RDC). Réponds clairement en français.",
            "Que dit l'Article 1 de la Constitution de la RDC ?",
            lambda t: "30 juin 1960" in t and ("état de droit" in t or "etat de droit" in t) and ("justice" in t or "kinshasa" in t or "debout congolais" in t) and "1er janvier" not in t and "angola" not in t and "gdrà" not in t and "ohada" not in t,
        ),
        (
            "ius_garde_a_vue",
            "Tu es SautiLM-Jua-Ius, assistant juridique hors-ligne de droit congolais (RDC). Réponds clairement en français.",
            "Que dit la Constitution de la RDC sur la garde à vue et les droits de la défense ?",
            lambda t: "48" in t and ("présumée innocente" in t or "presumee innocente" in t) and ("avocat" in t or "parquet" in t) and "basse-cour" not in t and "vlan" not in t and "360" not in t,
        ),
        (
            "ius_custody_en",
            "You are SautiLM-Jua-Ius, an offline Congolese Law & Justice assistant by LosingTek Research. Answer accurately in English.",
            "What does the DRC Constitution say about the 48-hour police custody limit?",
            lambda t: "48" in t and ("constitution" in t or "lawyer" in t or "parquet" in t or "art" in t) and "480" not in t,
        ),
        (
            "fin_avec_en",
            "You are SautiLM-Jua-Fin, an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English.",
            "In a savings group, 20 members buy 3 shares each at 2000 FC. Total capital?",
            lambda t: "120000" in t or "120,000" in t,
        ),
        (
            "civ_birth_en",
            "You are SautiLM-Jua-Civ, an offline Civic Administration assistant by LosingTek Research. Answer accurately in English.",
            "How to register a newborn's birth at the Civil Status Office within 90 days?",
            lambda t: "90" in t and ("civil status" in t or "family code" in t or "116" in t) and "ddi.be" not in t,
        ),
        (
            "agri_cmd_en",
            "You are SautiLM-Jua-Agri, an offline Agronomy & Environment assistant by LosingTek Research. Answer accurately in English.",
            "How to identify and control Cassava Mosaic Disease (CMD)?",
            lambda t: ("cassava mosaic" in t or "cmd" in t or "bemisia" in t or "whitefl" in t) and "oats" not in t and "oyamite" not in t and "bbadha" not in t,
        ),
        (
            "math_pastoral",
            None,
            "Un éleveur en Ituri possède 12 vaches. Il en achète 8 de plus, puis en vend 5 au marché. Combien lui en reste-t-il ?",
            lambda t: bool(re.search(r"\b15\b", t)) and "bbadha" not in t and "dyà" not in t,
        ),
        (
            "med_cholera",
            "Tu es SautiLM-Jua-Med, assistant de santé communautaire hors-ligne. Réponds clairement en français.",
            "Quelles sont les 3 étapes essentielles pour purifier l'eau et prévenir le choléra ?",
            lambda t: ("bouillir" in t or "filtr" in t or "chlore" in t or "savon" in t) and "bbadha" not in t and "dda" not in t,
        ),
        stem_suite[1],  # stem_quad_fr
        stem_suite[4],  # stem_phys_ohm_en
        (
            "stem_bio_photo_en",
            None,
            "In secondary Biology, explain the role of chlorophyll and write the products of photosynthesis.",
            lambda t: ("chlorophyll" in t or "chloroplast" in t) and ("glucose" in t or "oxygen" in t or "o2" in t) and "60% hydrogen" not in t and "choropleth" not in t,
        ),
    ]

    if target == "jua":
        return full_jua_suite

    prefix_map = {
        "fin": ("fin_", "greeting_fin"),
        "civ": ("civ_",),
        "edu": ("stem_", "math_"),
    }
    prefixes = prefix_map.get(target, (f"{target}_",))
    filtered = [row for row in full_jua_suite if row[0].startswith(prefixes)]
    return filtered or [full_jua_suite[0], full_jua_suite[1]]


def _score_runtime_outputs(outputs: Dict[str, str], target: str = "jua") -> tuple[int, int]:
    """Scores grounded factual accuracy, zero-script-hallucination, and zero-loop compliance."""
    suite = _runtime_suite_for_target(target)
    score = 0
    for key, _, _, predicate in suite:
        txt = (outputs.get(key) or "").lower()
        words = txt.split()
        unique_ratio = (len(set(words)) / len(words)) if words else 0.0
        clean_tokens = (
            "<pad>" not in txt
            and not _has_unexpected_script(txt)
            and not _has_repetition_loop(txt)
            and len(words) >= 4
            and unique_ratio >= 0.30
        )
        if clean_tokens and predicate(txt):
            score += 1
    return score, len(suite)


def run_litert_lm_runtime_checks(litertlm_path: Path, target: str = "jua") -> Dict[str, str]:
    """Loads the exported .litertlm in Google's real litert_lm Engine and generates responses."""
    import litert_lm

    suite = _runtime_suite_for_target(target)
    outputs: Dict[str, str] = {}
    with litert_lm.Engine(str(litertlm_path), backend=litert_lm.Backend.CPU()) as engine:
        for key, sys_prompt, user_prompt, _ in suite:
            init_messages = (
                [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
                if sys_prompt
                else None
            )
            sampler = litert_lm.SamplerConfig(top_k=16, top_p=0.90, temperature=0.05, seed=42)
            with engine.create_conversation(
                messages=init_messages,
                sampler_config=sampler,
                max_output_tokens=260,
            ) as conv:
                reply = "".join(str(chunk) for chunk in conv.send_message_async(user_prompt)).strip()
            outputs[key] = reply
            print(f"\n[LiteRT-LM LIVE CHECK ({target}): {key}]\nPrompt: {user_prompt}\nOutput: {reply[:400]}...")
            if not reply or "<pad>" in reply:
                raise RuntimeError(f"LiteRT-LM failed sanity check (empty or <pad>) on: {key}")

    return outputs


def export_with_litert_torch(
    merged_dir: Path,
    work_dir: Path,
    base_model_id: str,
    variant_cfg: Dict[str, str],
) -> Dict[str, Any]:
    """Installs official litert-torch + litert-lm after GPU training, exports, benchmarks, and verifies .litertlm."""
    print("=== [SautiLM-Forge] Step 5: Installing Official Google litert-torch & litert-lm ===")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade-strategy",
            "only-if-needed",
            "litert-torch==0.10.0",
            "litert-lm==0.19.2",
        ],
        check=True,
    )

    ensure_export_assets(merged_dir, base_model_id)
    _shim_litert_lm_builder_compat()

    from litert_torch.generative.export_hf import export as litert_export

    target = variant_cfg["target"]
    recipes_to_eval = ["dynamic_wi8_afp32"]
    export_out_dir = work_dir / "litert_export_tmp"
    final_litertlm = work_dir / variant_cfg["litertlm_filename"]
    candidates: List[Dict[str, Any]] = []

    try:
        for recipe in recipes_to_eval:
            if export_out_dir.exists():
                shutil.rmtree(export_out_dir)
            export_out_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = work_dir / f"candidate_{recipe}.litertlm"
            print(f"=== [SautiLM-Forge] Exporting {variant_cfg['model_name']} with recipe='{recipe}' ===")
            try:
                litert_export.export(
                    model=str(merged_dir),
                    output_dir=str(export_out_dir),
                    task="text_generation",
                    prefill_lengths=[128],
                    cache_length=2048,
                    quantization_recipe=recipe,
                    externalize_embedder=False,
                    use_jinja_template=True,
                    bundle_litert_lm=True,
                    keep_temporary_files=False,
                )
                generated_files = list(export_out_dir.glob("*.litertlm"))
                if not generated_files:
                    raise FileNotFoundError(f"No .litertlm output produced in {export_out_dir}")
                shutil.copy2(generated_files[0], candidate_file)

                binary_check = verify_tflite_flatbuffers_inside_litertlm(candidate_file)
                runtime_outputs = run_litert_lm_runtime_checks(candidate_file, target=target)
                score, max_score = _score_runtime_outputs(runtime_outputs, target=target)
                c_bytes = candidate_file.stat().st_size
                c_mb = round(c_bytes / (1024 * 1024), 2)
                print(
                    f"[SautiLM-Forge] Candidate '{recipe}': {c_mb} MB | Accuracy Score: {score}/{max_score}"
                )
                candidates.append(
                    {
                        "recipe": recipe,
                        "path": candidate_file,
                        "size_bytes": c_bytes,
                        "size_mb": c_mb,
                        "quality_score": score,
                        "max_score": max_score,
                        "binary_check": binary_check,
                        "runtime_outputs": runtime_outputs,
                    }
                )
            except Exception as err:
                print(f"[SautiLM-Forge] Candidate recipe '{recipe}' failed: {err}")

        if not candidates:
            raise RuntimeError("All LiteRT-LM export recipes failed!")

        candidates.sort(key=lambda c: (-c["quality_score"], c["size_bytes"]))
        winner = candidates[0]
        shutil.copy2(winner["path"], final_litertlm)
    finally:
        shutil.rmtree(export_out_dir, ignore_errors=True)
        shutil.rmtree(merged_dir, ignore_errors=True)
        for p in work_dir.glob("candidate_*.litertlm"):
            p.unlink(missing_ok=True)

    size_bytes = final_litertlm.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 2)
    score_str = f"{winner['quality_score']}/{winner['max_score']}"
    print(
        f"[SautiLM-Forge] Selected '{winner['recipe']}' -> {final_litertlm.name}: "
        f"{size_mb} MB ({size_bytes} bytes) | Score={score_str}"
    )

    return {
        "litertlm_filename": final_litertlm.name,
        "litertlm_size_bytes": size_bytes,
        "litertlm_size_mb": size_mb,
        "quantization_recipe": winner["recipe"],
        "litertlm_quality_score": score_str,
        "litertlm_binary_check": winner["binary_check"],
        "litertlm_runtime_test_outputs": winner["runtime_outputs"],
        "candidates_evaluated": [
            {
                "recipe": c["recipe"],
                "size_mb": c["size_mb"],
                "quality_score": f"{c['quality_score']}/{c['max_score']}",
            }
            for c in candidates
        ],
        "litertlm_verified": True,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and export modular SautiLM LiteRT-LM builds.")
    parser.add_argument("--variant", default=None, help="Target variant, e.g. sautilm-jua or sautilm-jua-stem.")
    args, _ = parser.parse_known_args()

    start_time = time.time()
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work_output")
    input_dir = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("./data")
    work_dir.mkdir(parents=True, exist_ok=True)

    variant_cfg = resolve_variant_config(input_dir, requested_variant=args.variant)

    print("=== [SautiLM-Forge] Step 1: Checking GPU & Environment ===")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import peft.import_utils
        import peft.tuners.lora.torchao

        peft.import_utils.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device_str} | Torch: {torch.__version__} | Variant: {variant_cfg['model_name']}")
    if device_str == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    curated_data = load_and_balance_dataset(input_dir, variant_cfg)

    base_model_id = "unsloth/gemma-3-270m-it"
    print(f"=== [SautiLM-Forge] Step 2: Loading Base Model ({base_model_id}) ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = 640
    encoded_examples: List[Dict[str, List[int]]] = []
    for row in curated_data:
        ex = prepare_masked_example(tokenizer, row, max_len=max_len)
        if ex is not None:
            encoded_examples.append(ex)
    print(f"[SautiLM-Forge] Prepared {len(encoded_examples)} masked-loss examples (max_len={max_len})")

    # Float32 on T4 (only ~1.08GB weights) ensures zero fp16 RMSNorm/logit overflow prior to LiteRT export
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float32,
        device_map={"": 0} if device_str == "cuda" else None,
    )

    lora_r = 32
    lora_alpha = 64
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.04,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("=== [SautiLM-Forge] Step 3: Masked Multi-Epoch LoRA Fine-Tuning ===")
    batch_size = 2 if device_str == "cuda" else 1
    grad_accum = 8 if device_str == "cuda" else 1
    effective_bs = batch_size * grad_accum
    target_key = variant_cfg["target"]
    if device_str != "cuda":
        num_epochs = 1
    elif target_key == "ius":
        num_epochs = 6
    elif target_key in ("stem", "med", "agri", "jua"):
        num_epochs = 5
    else:
        num_epochs = 4
    total_steps = max(1, (len(encoded_examples) // effective_bs) * num_epochs)
    if device_str != "cuda":
        total_steps = min(total_steps, 8)

    lr_by_target = {"stem": 1.35e-4, "med": 1.25e-4, "agri": 1.25e-4, "ius": 1.3e-4, "lex": 1.2e-4, "jua": 1.35e-4}
    lr = lr_by_target.get(target_key, 1.2e-4)
    print(f"[SautiLM-Forge] Hyperparams: target={target_key} | epochs={num_epochs} | steps={total_steps} | lr={lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step_idx: int) -> float:
        warmup = max(10, int(0.08 * total_steps))
        if step_idx < warmup:
            return float(step_idx + 1) / float(warmup)
        progress = float(step_idx - warmup) / float(max(1, total_steps - warmup))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.train()

    losses: List[float] = []
    step = 0
    micro_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        random.Random(42 + epoch).shuffle(encoded_examples)
        for idx in range(0, len(encoded_examples), batch_size):
            if step >= total_steps:
                break
            batch_items = encoded_examples[idx : idx + batch_size]
            batch = collate_padded_batch(batch_items, tokenizer.pad_token_id, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += float(loss.item())
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                val_loss = accum_loss
                accum_loss = 0.0
                losses.append(val_loss)
                if step % 20 == 0 or step == 1 or step == total_steps:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{total_steps} | "
                        f"Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

    print("=== [SautiLM-Forge] Step 4: Merging LoRA Adapter & Saving HF Checkpoint ===")
    adapter_dir = work_dir / variant_cfg["adapter_dirname"]
    merged_dir = work_dir / "merged_hf_model"
    model.save_pretrained(adapter_dir)
    merged_model = model.merge_and_unload()
    merged_model.eval()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Free all PyTorch GPU/CPU training objects before litert-torch conversion
    del outputs, loss, batch, optimizer, scheduler, model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    export_info = export_with_litert_torch(merged_dir, work_dir, base_model_id, variant_cfg)

    elapsed = round(time.time() - start_time, 2)
    metrics = {
        "model_name": variant_cfg["model_name"],
        "variant": variant_cfg["variant"],
        "target": variant_cfg["target"],
        "base_model": base_model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "steps": step,
        "epochs": num_epochs,
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(losses[-1], 4) if losses else None,
        "loss_history": [round(x, 4) for x in losses],
        "training_examples_count": len(encoded_examples),
        "elapsed_seconds": elapsed,
        **export_info,
    }
    with open(work_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"=== [SautiLM-Forge] COMPLETE in {elapsed}s! Output: {work_dir / export_info['litertlm_filename']} ===")


if __name__ == "__main__":
    main()
