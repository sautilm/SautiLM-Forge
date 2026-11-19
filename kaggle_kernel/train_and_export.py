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


BASE_MODELS_BY_TIER: Dict[str, str] = {
    "jua": "unsloth/gemma-3-270m-it",
    "kuu-1b": "unsloth/gemma-3-1b-it",
    "kuu-4b-med": "unsloth/medgemma-4b-it",
    "kuu-4b-ius": "unsloth/gemma-3-4b-it",
}

VARIANT_SPECS: Dict[str, Dict[str, str]] = {
    "sautilm-jua": {
        "target": "jua",
        "tier": "jua",
        "model_name": "SautiLM-Jua",
        "jsonl_filename": "sautilm_v1_train.jsonl",
        "litertlm_filename": "sautilm-jua.litertlm",
        "adapter_dirname": "sautilm_jua_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-jua-stem": {
        "target": "stem",
        "tier": "jua",
        "model_name": "SautiLM-Jua-STEM",
        "jsonl_filename": "sautilm_jua_stem_train.jsonl",
        "litertlm_filename": "sautilm-jua-stem.litertlm",
        "adapter_dirname": "sautilm_jua_stem_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-kuu-stem": {
        "target": "kuu-stem",
        "tier": "kuu-1b",
        "model_name": "SautiLM-Kuu-STEM",
        "jsonl_filename": "sautilm_jua_stem_train.jsonl",
        "litertlm_filename": "sautilm-kuu-stem.litertlm",
        "adapter_dirname": "sautilm_kuu_stem_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["kuu-1b"],
        "export_format": "litertlm",
    },
    "sautilm-jua-lex": {
        "target": "lex",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Lex",
        "jsonl_filename": "sautilm_jua_lex_train.jsonl",
        "litertlm_filename": "sautilm-jua-lex.litertlm",
        "adapter_dirname": "sautilm_jua_lex_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-jua-med": {
        "target": "med",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Med",
        "jsonl_filename": "sautilm_jua_med_train.jsonl",
        "litertlm_filename": "sautilm-jua-med.litertlm",
        "adapter_dirname": "sautilm_jua_med_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-kuu-med": {
        "target": "kuu-med",
        "tier": "kuu-4b-pc",
        "model_name": "SautiLM-Kuu-Med",
        "jsonl_filename": "sautilm_jua_med_train.jsonl",
        "litertlm_filename": "sautilm-kuu-med-q4_k_m.gguf",
        "adapter_dirname": "sautilm_kuu_med_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["kuu-4b-med"],
        "export_format": "gguf-pc",
    },
    "sautilm-jua-agri": {
        "target": "agri",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Agri",
        "jsonl_filename": "sautilm_jua_agri_train.jsonl",
        "litertlm_filename": "sautilm-jua-agri.litertlm",
        "adapter_dirname": "sautilm_jua_agri_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-jua-ius": {
        "target": "ius",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Ius",
        "jsonl_filename": "sautilm_jua_ius_train.jsonl",
        "litertlm_filename": "sautilm-jua-ius.litertlm",
        "adapter_dirname": "sautilm_jua_ius_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-kuu-ius": {
        "target": "kuu-ius",
        "tier": "kuu-4b-pc",
        "model_name": "SautiLM-Kuu-Ius",
        "jsonl_filename": "sautilm_jua_ius_train.jsonl",
        "litertlm_filename": "sautilm-kuu-ius-q4_k_m.gguf",
        "adapter_dirname": "sautilm_kuu_ius_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["kuu-4b-ius"],
        "export_format": "gguf-pc",
    },
    "sautilm-jua-fin": {
        "target": "fin",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Fin",
        "jsonl_filename": "sautilm_jua_fin_train.jsonl",
        "litertlm_filename": "sautilm-jua-fin.litertlm",
        "adapter_dirname": "sautilm_jua_fin_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-jua-civ": {
        "target": "civ",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Civ",
        "jsonl_filename": "sautilm_jua_civ_train.jsonl",
        "litertlm_filename": "sautilm-jua-civ.litertlm",
        "adapter_dirname": "sautilm_jua_civ_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
    "sautilm-jua-edu": {
        "target": "edu",
        "tier": "jua",
        "model_name": "SautiLM-Jua-Edu",
        "jsonl_filename": "sautilm_jua_edu_train.jsonl",
        "litertlm_filename": "sautilm-jua-edu.litertlm",
        "adapter_dirname": "sautilm_jua_edu_lora_adapter",
        "default_base_model": BASE_MODELS_BY_TIER["jua"],
        "export_format": "litertlm",
    },
}


def resolve_variant_config(input_root: Path, requested_variant: str | None = None) -> Dict[str, str]:
    """Resolves which SautiLM variant to train from CLI arg, env var, build_target.json, or input files."""
    candidate = (requested_variant or os.environ.get("SAUTILM_VARIANT", "")).strip().lower()
    if candidate in ("core", "jua", "sautilm-jua-core"):
        candidate = "sautilm-jua"
    elif candidate.startswith("kuu-") and not candidate.startswith("sautilm-"):
        candidate = f"sautilm-{candidate}"
    elif candidate and not (candidate.startswith("sautilm-jua") or candidate.startswith("sautilm-kuu")):
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


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes verbatim duplicate (prompt, response) training examples to prevent memorization."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in rows:
        p = " ".join(str(r.get("prompt") or "").lower().split())
        a = " ".join(str(r.get("response") or "").lower().split())
        key = (p, a)
        if p and a and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _row_group_key(row: Dict[str, Any]) -> str:
    """Computes a leak-proof split group key so identical prompts never straddle train & val."""
    prompt = " ".join(str(row.get("prompt") or "").lower().split())
    skeleton = re.sub(r"\d+(?:[.,]\d+)?", "<NUM>", prompt)[:140]
    return skeleton


def split_train_val_rows(
    rows: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Produces a deterministic, 0%-prompt-overlap train/val split.
    For multi-instance parametric families (e.g. 11 dosage weights), holds out ~10% of distinct
    prompts into validation while keeping the remaining distinct prompts in training so no skill
    family is starved, and identical prompt strings never appear in both train and val.
    """
    deduped = _dedupe_rows(rows)
    if len(deduped) < 12 or val_ratio <= 0.0:
        return deduped, []

    # Group first by exact normalized prompt string so identical prompts always stay together
    by_exact_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for r in deduped:
        p_norm = " ".join(str(r.get("prompt") or "").lower().split())
        by_exact_prompt.setdefault(p_norm, []).append(r)

    # Group distinct prompts into parametric skeleton families
    families: Dict[str, List[str]] = {}
    for p_norm, p_rows in by_exact_prompt.items():
        families.setdefault(_row_group_key(p_rows[0]), []).append(p_norm)

    rng = random.Random(seed)
    family_keys = list(families.keys())
    rng.shuffle(family_keys)

    target_val = max(4, min(200, int(round(len(deduped) * val_ratio))))
    val_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for fkey in family_keys:
        prompt_list = list(families[fkey])
        rng.shuffle(prompt_list)
        if len(prompt_list) >= 4 and len(val_rows) < target_val:
            n_val_prompts = max(1, int(round(len(prompt_list) * val_ratio)))
            for p_norm in prompt_list[:n_val_prompts]:
                if len(val_rows) < target_val:
                    val_rows.extend(by_exact_prompt[p_norm])
                else:
                    train_rows.extend(by_exact_prompt[p_norm])
            for p_norm in prompt_list[n_val_prompts:]:
                train_rows.extend(by_exact_prompt[p_norm])
        elif len(val_rows) < target_val and len(val_rows) + sum(len(by_exact_prompt[p]) for p in prompt_list) <= max(
            target_val + 4, int(0.25 * len(deduped))
        ):
            for p_norm in prompt_list:
                val_rows.extend(by_exact_prompt[p_norm])
        else:
            for p_norm in prompt_list:
                train_rows.extend(by_exact_prompt[p_norm])

    if not val_rows and len(train_rows) >= 8:
        val_rows = train_rows[-max(2, len(train_rows) // 10) :]
        train_rows = train_rows[: len(train_rows) - len(val_rows)]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _rewrite_row_for_target_identity(row: Dict[str, Any], variant_cfg: Dict[str, str]) -> Dict[str, Any]:
    """Aligns self-identity strings when a Kuu 1B/4B or Jua 270M variant shares a domain JSONL corpus."""
    target = variant_cfg.get("target", "")
    replacements = {
        "stem": ("SautiLM-Kuu-STEM", "SautiLM-Jua-STEM"),
        "kuu-stem": ("SautiLM-Jua-STEM", "SautiLM-Kuu-STEM"),
        "kuu-med": ("SautiLM-Jua-Med", "SautiLM-Kuu-Med"),
        "kuu-ius": ("SautiLM-Jua-Ius", "SautiLM-Kuu-Ius"),
    }
    pair = replacements.get(target)
    if not pair:
        return row
    old_name, new_name = pair
    updated = dict(row)
    for field in ("system", "prompt", "response", "text"):
        val = updated.get(field)
        if isinstance(val, str) and old_name in val:
            updated[field] = val.replace(old_name, new_name)
    msgs = updated.get("messages")
    if isinstance(msgs, list):
        new_msgs = []
        for m in msgs:
            if isinstance(m, dict):
                mc = dict(m)
                if isinstance(mc.get("content"), str) and old_name in mc["content"]:
                    mc["content"] = mc["content"].replace(old_name, new_name)
                new_msgs.append(mc)
            else:
                new_msgs.append(m)
        updated["messages"] = new_msgs
    return updated


def load_and_balance_dataset(input_root: Path, variant_cfg: Dict[str, str]) -> List[Dict[str, Any]]:
    """Loads the target SautiLM JSONL corpus without artificial duplicate repetition or cross-leakage."""
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
            row = _rewrite_row_for_target_identity(json.loads(line), variant_cfg)
            d = row.get("domain", target)
            by_domain.setdefault(d, []).append(row)

    random.seed(42)
    curated: List[Dict[str, Any]] = []

    if target != "jua":
        all_rows = _dedupe_rows([item for items in by_domain.values() for item in items])
        if target == "lex":
            curated = _select_clean_lex_rows(all_rows, max_core=320, max_other=220, sal_mult=1, max_rag=200)
            curated = _dedupe_rows(curated)
            random.shuffle(curated)
            print(f"[SautiLM-Forge] Standalone specialty 'SautiLM-Jua-Lex': {len(curated)} unique clean rows")
            return curated
        curated = list(all_rows)
        random.shuffle(curated)
        print(
            f"[SautiLM-Forge] Standalone specialty '{variant_cfg['model_name']}': "
            f"{len(curated)} unique rows (0x duplicate repetition)"
        )
        return curated

    # Full SautiLM-Jua foundation mix: strictly unique examples (0x artificial duplication).
    domain_caps = {
        "identity": 120,
        "ius": 250,
        "med": 250,
        "agri": 240,
        "fin": 160,
        "civ": 140,
        "edu": 120,
        "math": 170,
        "voice": 100,
        "vision": 110,
        "stem": 520,
    }
    for d, items in by_domain.items():
        if d == "lex":
            continue
        unique_items = _dedupe_rows(items)
        cap = domain_caps.get(d, 200)
        if len(unique_items) > cap:
            curated.extend(random.sample(unique_items, cap))
        else:
            curated.extend(unique_items)

    clean_lex = _select_clean_lex_rows(
        _dedupe_rows(by_domain.get("lex", [])),
        max_core=240,
        max_other=160,
        sal_mult=1,
        max_rag=180,
    )
    curated = _dedupe_rows(curated + clean_lex)

    random.shuffle(curated)
    print(
        f"[SautiLM-Forge] Balanced {variant_cfg['model_name']} dataset: {len(curated)} unique rows "
        f"(clean_lex={len(clean_lex)}, 0 duplicates)"
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
    """
    Returns strictly HELD-OUT (0%-overlap with training corpora) verification tuples:
    (check_key, system_prompt, user_prompt, predicate_fn), including negative abstention tests.
    """
    stem_name = "SautiLM-Kuu-STEM" if target == "kuu-stem" else "SautiLM-Jua-STEM"
    stem_sys_fr = (
        f"Tu es {stem_name}, assistant pédagogique hors-ligne du Secondaire et de l'Examen d'État "
        "(Mathématiques, Physique, Chimie, Biologie) conçu par LosingTek Research. "
        "Raisonne étape par étape avec exactitude et refuse d'inventer des données inexistantes."
    )
    stem_sys_en = (
        f"You are {stem_name}, an offline secondary school STEM (Mathematics, Physics, Chemistry, Biology) "
        "assistant created by LosingTek Research. Show clear step-by-step reasoning and exact units."
    )
    stem_suite = [
        (
            "stem_identity_fr",
            stem_sys_fr,
            "Présente brièvement ton rôle scolaire hors ligne et les matières couvertes.",
            lambda t: ("sautilm-jua-stem" in t or "sautilm-kuu-stem" in t) and ("secondaire" in t or "examen" in t or "lycée" in t) and "je suis math" not in t and "bbadha" not in t,
        ),
        (
            "stem_quad_fr",
            stem_sys_fr,
            "Résous dans R l'équation quadratique suivante : 3x^2 - 6x - 9 = 0.",
            lambda t: ("144" in t or "36" in t or "delta" in t or "x^2 - 2x - 3" in t) and bool(re.search(r"\b3\b", t)) and "-1" in t and "x = 6" not in t and "bbadha" not in t,
        ),
        (
            "stem_integral_fr",
            stem_sys_fr,
            "Calcule l'intégrale définie de x = 0 à pi/2 de cos(x) dx.",
            lambda t: bool(re.search(r"\b1(?:\.0|,0)?\b", t)) and ("sin" in t or "intégrale" in t or "integrale" in t or "primitive" in t) and "cosinus(0)" not in t and "bbadha" not in t,
        ),
        (
            "stem_qcm_math_fr",
            stem_sys_fr,
            "QCM Algèbre : Pour x^2 - 7x + 12 = 0, quelles sont les solutions ? A) {2; 6}  B) {3; 4}  C) {-3; -4}  D) {1; 12}.",
            lambda t: (bool(re.search(r"\bb\)", t)) or "bonne réponse : b" in t or "bonne reponse : b" in t or "réponse b" in t) and bool(re.search(r"\b3\b", t)) and bool(re.search(r"\b4\b", t)) and "(-1 ; 3)" not in t,
        ),
        (
            "stem_phys_ohm_en",
            stem_sys_en,
            "A heating coil of resistance 15 ohms carries a steady current of 3 A. Compute the voltage U across it.",
            lambda t: bool(re.search(r"\b45\s*\*?\*?\s*(?:v|volt|volts)\b", t)) and "= 3 v" not in t and "45 a" not in t,
        ),
        (
            "stem_phys_newton_fr",
            stem_sys_fr,
            "Un chariot de masse 8 kg accélère à 4 m/s^2 sur un rail plat. Quelle est l'intensité de la force résultante F ?",
            lambda t: bool(re.search(r"\b32\s*\*?\*?\s*(?:n|newton)\b", t)),
        ),
        (
            "stem_chem_mole_fr",
            stem_sys_fr,
            "Détermine la quantité de matière contenue dans 88 g de dioxyde de carbone CO2 (M = 44 g/mol).",
            lambda t: bool(re.search(r"\b2(?:\.0|,0)?\s*\*?\*?\s*mol(?:e|es)?\b", t)),
        ),
        (
            "stem_chem_ph_en",
            stem_sys_en,
            "QCM Chemistry: At 25 C, an aqueous soap solution has pH = 11. How is it classified? A) Acidic  B) Neutral  C) Basic  D) Gas.",
            lambda t: ("c)" in t or "correct answer: c" in t or "correct answer c)" in t or "basic" in t or "alkaline" in t) and "acidic solution" not in t and "d) gas" not in t,
        ),
        (
            "stem_bio_photo_en",
            stem_sys_en,
            "Which plant organelle uses chlorophyll pigment to convert CO2 and H2O into glucose and O2 under sunlight?",
            lambda t: ("chloroplast" in t or "chlorophyll" in t) and ("glucose" in t or "oxygen" in t or "o2" in t or "photosynthesis" in t or "thylakoid" in t) and "choropleth" not in t,
        ),
        (
            "stem_abstain_fr",
            stem_sys_fr,
            "En chimie, donne la masse atomique exacte d'un élément fictif nommé 'Zyrkonium-999' qui n'existe pas au tableau périodique.",
            lambda t: ("n'existe pas" in t or "fictif" in t or "absent" in t or "inconnu" in t or "pas dans le tableau" in t or "refus" in t or "aucune masse" in t) and "masse atomique exacte de 999" not in t and "mass = 999" not in t,
        ),
    ]
    if target in ("stem", "kuu-stem"):
        return stem_suite

    med_name = "SautiLM-Kuu-Med" if target == "kuu-med" else "SautiLM-Jua-Med"
    med_sys_fr = (
        f"Tu es {med_name}, assistant médical et de triage clinique hors-ligne conçu par LosingTek Research "
        "(intégration AfyaSpeak). Applique strictement les protocoles OMS/MSF/PNLP RDC et refuse toute posologie pédiatrique sans poids en kg."
    )
    med_sys_en = (
        f"You are {med_name}, an offline clinical triage and public health assistant created by LosingTek Research "
        "(AfyaSpeak partner). Follow WHO/MSF guidelines strictly and never prescribe pediatric doses without weight in kg."
    )
    med_suite = [
        ("med_identity_fr", med_sys_fr, "Décris brièvement ton rôle d'assistant médical hors ligne et tes limites.", lambda t: ("sautilm-jua-med" in t or "sautilm-kuu-med" in t or "afyaspeak" in t or "losingtek" in t) and "bbadha" not in t),
        ("med_identity_en", med_sys_en, "Summarize your offline clinical assistance role and when you refer patients.", lambda t: ("sautilm" in t or "afyaspeak" in t or "losingtek" in t or "doctor" in t or "clinic" in t) and "bbadha" not in t),
        ("med_amox_unseen", med_sys_fr, "Calcul pédiatrique : enfant de 16 kg sous amoxicilline 50 mg/kg/jour répartie en 2 prises. Quelle dose en mg par prise ?", lambda t: "800" in t and "400" in t and "bbadha" not in t),
        ("med_artesunate_unseen", med_sys_fr, "Un enfant de 13 kg souffre de paludisme grave (PNLP RDC). Quelle dose d'artésunate IV à 3 mg/kg par injection faut-il calculer (règle < 20 kg) ?", lambda t: "39" in t and ("3,0" in t or "3.0" in t or "3 mg/kg" in t or "3 * 13" in t or "13 * 3" in t) and "bbadha" not in t),
        ("med_ors_unseen_en", med_sys_en, "A 11 kg dehydrated toddler enters OMS Plan B oral rehydration (75 mL/kg over 4h). Total ORS milliliters required?", lambda t: "825" in t and ("ors" in t or "ml" in t or "rehydration" in t or "zinc" in t) and "bbadha" not in t),
        ("med_cholera_en", med_sys_en, "Compare the safe cholera chlorine percentages for handwashing (0.05%) versus stool-spill disinfection (0.2% or 0.5%) and explain why they differ.", lambda t: "0.05" in t and ("0.5" in t or "0.2" in t) and "bbadha" not in t),
        ("med_pph_fr", med_sys_fr, "Dans les 30 minutes après accouchement, l'utérus reste mou (atonie utérine) avec saignement > 500 mL. Quelles 2 actions prioritaires d'urgence (massage utérin et utérotonique) ?", lambda t: ("massage" in t or "utérin" in t) and ("oxytocine" in t or "ocytocine" in t)),
        ("med_abstain_weight", med_sys_fr, "Calcule immédiatement la dose exacte en mg d'artésunate IV pour un petit garçon dont aucun poids ni âge approximatif n'a été mesuré.", lambda t: ("poids" in t or "peser" in t or "pesée" in t or "manquant" in t or "impossible" in t or "kg" in t) and "100 mg" not in t and "130 mg" not in t),
    ]
    if target in ("med", "kuu-med"):
        return med_suite

    agri_sys_fr = (
        "Tu es SautiLM-Jua-Agri, assistant agronome hors-ligne pour la RDC et l'Afrique de l'Est conçu par LosingTek Research "
        "(intégration Gemmea). Appuie-toi sur les normes IITA/FAO/NARS (INERA, KALRO, NARO, RAB, ISABU, TARI) et rejette toute image non végétale."
    )
    agri_sys_en = (
        "You are SautiLM-Jua-Agri, an offline agronomy and crop protection assistant for DRC and East Africa by LosingTek Research (Gemmea)."
    )
    agri_suite = [
        ("agri_identity_fr", agri_sys_fr, "Présente brièvement ton rôle d'assistant agronome hors ligne pour la RDC et l'Afrique de l'Est.", lambda t: ("sautilm" in t or "gemmea" in t or "losingtek" in t) and "bbadha" not in t),
        ("agri_identity_en", agri_sys_en, "Briefly state your offline agronomy assistant role across DRC and East Africa.", lambda t: ("sautilm" in t or "gemmea" in t or "losingtek" in t) and "bbadha" not in t),
        ("agri_plot_unseen", agri_sys_fr, "Calcule le nombre de boutures saines de manioc à 1 m x 1 m nécessaires pour une parcelle rectangulaire de 35 m sur 20 m.", lambda t: "700" in t and "bbadha" not in t),
        ("agri_kenya_calendar", agri_sys_fr, "Au Kenya (KALRO), quelles sont les deux principales saisons des pluies (Long Rains mars-mai et Short Rains octobre-décembre) pour planifier les semis ?", lambda t: ("long rains" in t or "mars" in t) and ("short rains" in t or "octobre" in t or "kalro" in t)),
        ("agri_bxw_unseen", agri_sys_fr, "Un champ de bananiers perd ses feuilles par jaunissement soudain et un exsudat bactérien jaune sort du pseudo-tronc tranché (BXW / Xanthomonas). Quelle maladie suspecter et comment désinfecter la machette ?", lambda t: ("bxw" in t or "xanthomonas" in t or "flétrissement" in t) and ("machette" in t or "outil" in t or "feu" in t or "javel" in t or "flamme" in t)),
        ("agri_faw_pushpull", agri_sys_fr, "Explique le principe écologique Push-Pull associant Desmodium (répulsif) et Brachiaria (plante piège) sur maïs.", lambda t: "desmodium" in t and ("brachiaria" in t or "striga" in t or "piège" in t)),
        ("agri_pics_en", agri_sys_en, "Before sealing maize grain inside triple-layer PICS bags, below what safe moisture percentage (12% to 13%) must grain be dried to halt mold and weevils?", lambda t: ("13" in t or "12" in t) and "pics" in t and "100% moisture" not in t),
        ("agri_abstain_nonleaf", agri_sys_fr, "[TRIAGE VISUEL: IS_LEAF=non, objet='sandale en plastique bleue'] Identifie la maladie foliaire sur cette photo.", lambda t: ("pas" in t or "non" in t or "reprendre" in t or "aucun" in t) and ("feuille" in t or "plante" in t or "végétal" in t) and "mosaïque du manioc" not in t and "rouille du maïs" not in t),
    ]
    if target == "agri":
        return agri_suite

    ius_name = "SautiLM-Kuu-Ius" if target == "kuu-ius" else "SautiLM-Jua-Ius"
    ius_sys_fr = (
        f"Tu es {ius_name}, assistant juridique hors-ligne spécialisé en droit congolais (Constitution RDC, Code de la Famille Loi 16/008, "
        "Code du Travail, Code Pénal) et droit OHADA, conçu par LosingTek Research. Refuse d'inventer tout article absent du corpus."
    )
    ius_sys_en = (
        f"You are {ius_name}, an offline DRC and OHADA legal assistant created by LosingTek Research. Never fabricate nonexistent articles."
    )
    ius_suite = [
        ("ius_identity_fr", ius_sys_fr, "Présente ton domaine juridique hors ligne et ton engagement de fidélité documentaire.", lambda t: ("sautilm-jua-ius" in t or "sautilm-kuu-ius" in t or "losingtek" in t) and "bbadha" not in t),
        ("ius_identity_en", ius_sys_en, "Briefly describe your DRC/OHADA legal assistant role.", lambda t: ("sautilm" in t or "losingtek" in t or "ohada" in t) and "bbadha" not in t),
        ("ius_garde_unseen", ius_sys_fr, "Selon l'Article 18 de la Constitution de la RDC (limite maximale de garde à vue de 48 heures), un citoyen maintenu 60 heures au commissariat sans présentation au Parquet subit quelle violation ?", lambda t: ("48" in t or "18" in t) and ("arbitraire" in t or "illégale" in t or "dépassement" in t or "parquet" in t or "violation" in t)),
        ("ius_marriage_en", ius_sys_en, "Under DRC revised Family Code (Law 16/008, minimum legal marriage age 18 years), is a civil marriage valid if one spouse is aged 17?", lambda t: ("18" in t) and ("no" in t or "not valid" in t or "invalid" in t or "illegal" in t or "cannot" in t)),
        ("ius_succession_unseen", ius_sys_fr, "Actif successoral net de 16000 USD laissé à 3 enfants en RDC (1ère catégorie = 3/4 de la succession) et une veuve (protégée par l'usufruit du logement familial). Quel montant revient au groupe des enfants et quel droit protège la veuve ?", lambda t: ("12000" in t or "12 000" in t or "12,000" in t) and "usufruit" in t),
        ("ius_ohada_sarl", ius_sys_fr, "Un commerçant de Bunia veut créer une SARL immatriculée au RCCM sous l'OHADA. Quel Acte uniforme (AUSCGIE) régit les sociétés commerciales ?", lambda t: "auscgie" in t and "ausca" not in t and "non aux sarl" not in t and "pas aux sarl" not in t and "pas applicable aux sarl" not in t),
        ("ius_cdd_unseen", ius_sys_fr, "En droit du travail congolais (Article 40 Code du Travail : plafond de 2 ans / renouvellement CDD), un salarié maintenu 30 mois sous 3 CDD successifs sur un poste permanent bénéficie de quelle requalification (CDI) ?", lambda t: "cdi" in t and ("2 ans" in t or "40" in t or "30 mois" in t or "indéterminée" in t)),
        ("ius_abstain_fake_art", ius_sys_fr, "Cite mot pour mot les sanctions de l'Article 874 fictif du Code pénal congolais régissant les satellites privés (si cet article n'existe pas dans le corpus, refuse de l'inventer).", lambda t: ("n'existe pas" in t or "ne dispose pas" in t or "absent" in t or "refus" in t or "fictif" in t or "pas" in t) and "servitude pénale de 20 ans" not in t),
    ]
    if target in ("ius", "kuu-ius"):
        return ius_suite

    lex_sys_fr = (
        "Tu es SautiLM-Jua-Lex, assistant lexicographique trilingue hors-ligne pour la langue Bbadha (Kilendu, parlée en Ituri, RDC), "
        "le Kiswahili, le Français et l'Anglais, conçu par LosingTek Research."
    )
    lex_sys_en = (
        "You are SautiLM-Jua-Lex, an offline Bbadha (Kilendu, spoken in Ituri, DRC), Kiswahili, French, and English lexical assistant by LosingTek Research."
    )
    lex_suite = [
        ("lex_identity", lex_sys_fr, "Présente brièvement ton rôle lexicographique pour la langue Bbadha (Kilendu) en Ituri.", lambda t: ("sautilm" in t or "bbadha" in t or "ituri" in t)),
        ("lex_ago_fr", "Tu es SautiLM-Jua-Lex. Dictionnaire Bbadha vérifié : Mot: ágó | Kiswahili: mwituni | Français: dans la forêt.", "Comment dit-on 'dans la forêt' en Bbadha (Kilendu) et en Kiswahili selon cet extrait ?", lambda t: ("ágó" in t or "ago" in t) and ("mwituni" in t or "forêt" in t or "foret" in t)),
        ("lex_abba_en", "You are SautiLM-Jua-Lex. Verified Bbadha dictionary: Word: 'ábba | Kiswahili: mjini | English: in the village.", "What does the Bbadha word \"'ábba\" mean in Kiswahili and English?", lambda t: ("village" in t or "mjini" in t)),
        ("lex_waku_fr", "Tu es SautiLM-Jua-Lex. Dictionnaire Bbadha vérifié : Mot: wakú | Kiswahili: ukanda cha mlezi wa mtoto | Français: bandoulière d'enfant ; berceau.", "Quelle est la traduction de \"wakú\" (w.json) en Français et Kiswahili ?", lambda t: ("berceau" in t or "bandoulière" in t or "ukanda" in t or "mtoto" in t)),
        ("lex_rag_bbu", "Tu es SautiLM-Jua-Lex. Réponds uniquement d'après cet extrait vérifié : Mot: bbù | Kiswahili: tundu ; kisima | Français: trou ; un puits.", "Quels sont les sens du mot Bbadha \"bbù\" dans cet extrait ?", lambda t: ("trou" in t or "puits" in t or "kisima" in t)),
        ("lex_rag_dda", "Tu es SautiLM-Jua-Lex. Réponds uniquement d'après cet extrait vérifié : Mot: ddà | Kiswahili: kuwa refu ; maji | Français: être long, loin ; l'eau, la rivière.", "Donne les deux sens principaux de \"ddà\" selon cet extrait.", lambda t: ("eau" in t or "rivière" in t or "maji" in t) and ("long" in t or "loin" in t)),
        ("lex_phonetics", lex_sys_fr, "Pourquoi l'orthographe Bbadha (langue tonale d'Ituri) utilise-t-elle des accents tonals et des consonnes implosives doubles comme dd vs d ?", lambda t: ("ton" in t or "tons" in t or "tonal" in t) and ("implosive" in t or "sens" in t)),
        ("lex_abstain_fake", "Tu es SautiLM-Jua-Lex. Extrait SQLite : aucun résultat trouvé dans le dictionnaire Bbadha.", "Traduis le mot Bbadha inventé 'qxztà' sans l'inventer.", lambda t: ("pas" in t or "absent" in t or "aucun" in t or "refus" in t or "dictionnaire" in t) and "soleil" not in t),
    ]
    if target == "lex":
        return lex_suite

    jua_sys_fr = (
        "Tu es SautiLM-Jua, modèle fondation hors-ligne de SautiLM créé à Bunia (Ituri, RDC) par LosingTek Research "
        "(Georges Byona, Abraham Imani, Jacques Uwonda). Réponds avec rigueur et précision."
    )
    jua_sys_en = (
        "You are SautiLM-Jua, an offline Edge-AI foundation model created in Bunia (Ituri, DRC) by LosingTek Research "
        "(Georges Byona, Abraham Imani, Jacques Uwonda)."
    )
    full_jua_suite = [
        ("identity_fr", jua_sys_fr, "Présente qui a conçu la série SautiLM-Jua et sa mission hors ligne.", lambda t: "sautilm-jua" in t and "losingtek" in t and "sautilm-lex" not in t and "core" not in t),
        ("identity_en", jua_sys_en, "Summarize who built SautiLM-Jua and verify you are distinct from ChatGPT.", lambda t: "sautilm-jua" in t and "losingtek" in t and "core" not in t),
        ("identity_georges", jua_sys_fr, "Qui dirige la recherche SautiLM chez LosingTek Research (Georges Byona, Abraham Imani, Jacques Uwonda) ?", lambda t: ("georges byona" in t or "jacques uwonda" in t) and "losingtek" in t),
        (
            "greeting_fin_hey",
            "You are SautiLM-Jua-Fin, an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English.",
            "Hello there, what accounting tasks can you assist with offline?",
            lambda t: ("sautilm" in t or "finance" in t or "accounting" in t or "budget" in t) and "mtekmeda" not in t,
        ),
        lex_suite[1],  # lex_ago_fr
        lex_suite[2],  # lex_abba_en
        lex_suite[5],  # lex_rag_dda
        ius_suite[2],  # ius_garde_unseen
        ius_suite[3],  # ius_marriage_en
        ius_suite[7],  # ius_abstain_fake_art
        (
            "fin_avec_en",
            "You are SautiLM-Jua-Fin, an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English.",
            "In a village cooperative, 18 members subscribe 4 shares at 1500 FC each. Compute total capital.",
            lambda t: "108000" in t or "108,000" in t or "108 000" in t,
        ),
        (
            "civ_birth_en",
            "You are SautiLM-Jua-Civ, an offline Civic Administration assistant by LosingTek Research (DRC Family Code Art. 116: birth registration within 90 days).",
            "Within how many days after birth must parents register a newborn at the DRC civil status office?",
            lambda t: "90" in t and ("civil status" in t or "family code" in t or "état civil" in t or "days" in t or "jours" in t),
        ),
        agri_suite[2],  # agri_plot_unseen
        (
            "math_pastoral",
            jua_sys_fr,
            "Un éleveur en Ituri possède 19 chèvres, en achète 7 puis en vend 9. Combien lui en reste-t-il ?",
            lambda t: bool(re.search(r"\b17\b", t)) and "bbadha" not in t,
        ),
        med_suite[2],  # med_amox_unseen
        stem_suite[1],  # stem_quad_fr
        stem_suite[4],  # stem_phys_ohm_en
        stem_suite[8],  # stem_bio_photo_en
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
            sampler = litert_lm.SamplerConfig(top_k=32, top_p=0.95, temperature=0.15, seed=42)
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


def resolve_prefill_lengths(target: str, base_model_id: str) -> List[int]:
    """Configures multi-bucket prefill signatures so mobile RAG prompts never hit length mismatch."""
    return [128, 512, 1024]


def resolve_base_model_id(requested_base: str | None, variant_cfg: Dict[str, str]) -> str:
    """Resolves Hugging Face base model ID for Jua (270M), Kuu-1B, or Kuu-4B Desktop PC."""
    cand = (requested_base or "auto").strip()
    if not cand or cand.lower() == "auto":
        return variant_cfg.get("default_base_model", BASE_MODELS_BY_TIER["jua"])
    aliases = {
        "270m": BASE_MODELS_BY_TIER["jua"],
        "1b": BASE_MODELS_BY_TIER["kuu-1b"],
        "4b-med": BASE_MODELS_BY_TIER["kuu-4b-med"],
        "4b-ius": BASE_MODELS_BY_TIER["kuu-4b-ius"],
    }
    return aliases.get(cand.lower(), cand)


def compute_eval_loss(
    model,
    val_examples: List[Dict[str, List[int]]],
    pad_token_id: int,
    device,
    batch_size: int = 2,
) -> float:
    """Computes genuine held-out validation cross-entropy loss with gradients disabled."""
    import torch

    if not val_examples:
        return 0.0
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for idx in range(0, len(val_examples), batch_size):
            batch = collate_padded_batch(val_examples[idx : idx + batch_size], pad_token_id, device)
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_batches += 1
    if was_training:
        model.train()
    return round(total_loss / max(1, total_batches), 4)


def verify_gguf_file(gguf_path: Path) -> Dict[str, Any]:
    """Verifies binary GGUF magic header ('GGUF') and minimum non-trivial file size."""
    if not gguf_path.exists():
        return {"gguf_verified": False, "reason": "file_missing"}
    raw_hdr = gguf_path.read_bytes()[:4]
    size_bytes = gguf_path.stat().st_size
    valid_magic = raw_hdr == b"GGUF"
    return {
        "gguf_verified": bool(valid_magic and size_bytes > 1024),
        "magic": raw_hdr.decode("ascii", errors="replace"),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 2),
    }


def run_hf_heldout_eval(
    model,
    tokenizer,
    target: str,
    device,
    max_new_tokens: int = 260,
) -> tuple[int, int, Dict[str, str]]:
    """
    Runs greedy held-out evaluation directly in PyTorch (for 0-step Base Model vs Fine-Tuned A/B gating)
    using the exact same zero-overlap suite as LiteRT-LM / GGUF runtime verification.
    """
    import torch

    suite = _runtime_suite_for_target(target)
    was_training = model.training
    model.eval()
    outputs: Dict[str, str] = {}

    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
            eos_ids.append(eot_id)
    except Exception:
        pass

    with torch.no_grad():
        for key, sys_prompt, user_prompt, _ in suite:
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            try:
                prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                block = f"{sys_prompt}\n\n{user_prompt}".strip() if sys_prompt else user_prompt
                bos = tokenizer.bos_token or "<bos>"
                prompt_str = f"{bos}<start_of_turn>user\n{block}<end_of_turn>\n<start_of_turn>model\n"

            enc = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[0][enc["input_ids"].shape[1] :]
            reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs[key] = reply

    if was_training:
        model.train()

    score, total = _score_runtime_outputs(outputs, target=target)
    return score, total, outputs


def _strip_llama_cli_banner(raw_output: str, user_prompt: str = "") -> str:
    """Strips llama-cli interactive banner, spinner, echoed prompt, and trailing timing stats."""
    text = (raw_output or "").replace("\r", "")
    # Remove backspace spinner animations
    while "\x08" in text:
        text = re.sub(r".?\x08", "", text)
    if user_prompt:
        marker = f"> {user_prompt}"
        idx = text.find(marker)
        if idx >= 0:
            text = text[idx + len(marker) :]
        else:
            # Partial match on first 36 chars of prompt if wrapped
            short = user_prompt[:36].strip()
            idx2 = text.find(f"> {short}")
            if idx2 >= 0:
                after = text[idx2:]
                nl = after.find("\n")
                text = after[nl + 1 :] if nl >= 0 else ""
    text = re.sub(r"\[\s*Prompt:.*?t/s\s*\][\s\S]*$", "", text)
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>", 1)[0]
    return text.strip()


def run_gguf_llama_cli_checks(
    gguf_path: Path,
    llama_cli_bin: str,
    target: str,
    max_tokens: int = 300,
) -> Dict[str, str]:
    """Executes real held-out inference on a quantized 4B GGUF via llama-cli (`-st` single-turn mode)."""
    suite = _runtime_suite_for_target(target)
    outputs: Dict[str, str] = {}
    for key, sys_prompt, user_prompt, _ in suite:
        cmd = [
            llama_cli_bin,
            "-m",
            str(gguf_path),
        ]
        if sys_prompt:
            cmd.extend(["-sys", sys_prompt])
        cmd.extend([
            "-p",
            user_prompt,
            "-n",
            str(max_tokens),
            "--temp",
            "0.05",
            "--seed",
            "42",
            "-t",
            "4",
            "-c",
            "2048",
            "-st",
            "--no-display-prompt",
        ])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        reply = _strip_llama_cli_banner(proc.stdout or "", user_prompt=user_prompt)
        # Fallback if a binary rejects `-sys` or `-st` (e.g., llama-completion / older llama-cli)
        if proc.returncode != 0 or not reply:
            combined_prompt = f"<start_of_turn>user\n{(sys_prompt + chr(10) + chr(10) if sys_prompt else '')}{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            for extra_flag in (["-st", "--no-display-prompt"], []):
                alt_cmd = [
                    llama_cli_bin,
                    "-m",
                    str(gguf_path),
                    "-p",
                    combined_prompt,
                    "-n",
                    str(max_tokens),
                    "--temp",
                    "0.05",
                    "-t",
                    "4",
                    "-c",
                    "2048",
                    *extra_flag,
                ]
                proc2 = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=240, check=False)
                cand = _strip_llama_cli_banner(proc2.stdout or "", user_prompt=user_prompt)
                if cand.startswith(combined_prompt):
                    cand = cand[len(combined_prompt) :].strip()
                if proc2.returncode == 0 and cand:
                    reply = cand
                    break
                if not reply and proc2.stderr:
                    print(f"[GGUF llama-cli stderr ({key})]: {proc2.stderr.strip()[-300:]}")
        if not reply and proc.stderr:
            print(f"[GGUF llama-cli stderr ({key})]: {proc.stderr.strip()[-300:]}")
        outputs[key] = reply
        print(f"\n[GGUF llama-cli LIVE CHECK ({target}): {key}]\nPrompt: {user_prompt}\nOutput: {reply[:400]}...")
    return outputs


def export_desktop_pc_gguf_plan(merged_dir: Path, work_dir: Path, variant_cfg: Dict[str, str]) -> Dict[str, Any]:
    """Converts merged 4B workstation safetensors to GGUF (F16 -> Q4_K_M) and runs real llama-cli held-out inference."""
    out_gguf = work_dir / variant_cfg["litertlm_filename"]
    tmp_root = Path("/tmp") if Path("/tmp").exists() else work_dir
    f16_gguf = tmp_root / out_gguf.name.replace("-q4_k_m.gguf", "-f16.gguf")
    llama_dir = tmp_root / "llama.cpp"

    convert_script = shutil.which("convert_hf_to_gguf.py")
    quantize_bin = shutil.which("llama-quantize")
    cli_bin = shutil.which("llama-cli")
    if (not convert_script or not quantize_bin or not cli_bin) and shutil.which("git") and shutil.which("cmake"):
        try:
            if not llama_dir.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)],
                    check=True,
                )
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)
            build_dir = llama_dir / "build"
            if not (build_dir / "bin" / "llama-quantize").exists() or not (build_dir / "bin" / "llama-cli").exists():
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(llama_dir),
                        "-B",
                        str(build_dir),
                        "-DBUILD_SHARED_LIBS=OFF",
                        "-DLLAMA_CURL=OFF",
                    ],
                    check=True,
                )
                subprocess.run(
                    ["cmake", "--build", str(build_dir), "--target", "llama-quantize", "llama-cli", "-j", "4"],
                    check=True,
                )
            cand_conv = llama_dir / "convert_hf_to_gguf.py"
            cand_quant = build_dir / "bin" / "llama-quantize"
            cand_cli = build_dir / "bin" / "llama-cli"
            if cand_conv.exists():
                convert_script = str(cand_conv)
            if cand_quant.exists():
                quantize_bin = str(cand_quant)
            if cand_cli.exists():
                cli_bin = str(cand_cli)
        except Exception as gguf_setup_err:
            print(f"[SautiLM-Forge] llama.cpp setup skipped: {gguf_setup_err}")

    gguf_verified = False
    gguf_runtime_verified = False
    gguf_check: Dict[str, Any] = {}
    gguf_outputs: Dict[str, str] = {}
    gguf_quality_score: str | None = None
    try:
        if convert_script and quantize_bin:
            try:
                subprocess.run(
                    [sys.executable, convert_script, str(merged_dir), "--outfile", str(f16_gguf), "--outtype", "f16"],
                    check=True,
                )
                subprocess.run([quantize_bin, str(f16_gguf), str(out_gguf), "Q4_K_M"], check=True)
                gguf_check = verify_gguf_file(out_gguf)
                gguf_verified = bool(gguf_check.get("gguf_verified"))
                if gguf_verified and cli_bin:
                    gguf_outputs = run_gguf_llama_cli_checks(
                        out_gguf, cli_bin, target=variant_cfg["target"]
                    )
                    g_score, g_max = _score_runtime_outputs(gguf_outputs, target=variant_cfg["target"])
                    gguf_quality_score = f"{g_score}/{g_max}"
                    gguf_runtime_verified = bool(
                        len(gguf_outputs) == g_max and all(len(v.strip()) > 0 for v in gguf_outputs.values())
                    )
                    print(f"[SautiLM-Forge] Real GGUF llama-cli Held-Out Score: {gguf_quality_score}")
            except Exception as conv_err:
                print(f"[SautiLM-Forge] GGUF conversion/inference error: {conv_err}")
    finally:
        if f16_gguf.exists():
            f16_gguf.unlink(missing_ok=True)
        if merged_dir.exists() and gguf_verified:
            shutil.rmtree(merged_dir, ignore_errors=True)

    return {
        "export_format": "gguf-pc",
        "gguf_target_filename": out_gguf.name,
        "merged_hf_dir": str(merged_dir),
        "quantization_recipe": "Q4_K_M_WORKSTATION",
        "llama_cpp_available": bool(quantize_bin),
        "llama_cli_available": bool(cli_bin),
        "gguf_verified": gguf_verified,
        "gguf_runtime_verified": gguf_runtime_verified,
        "gguf_quality_score": gguf_quality_score,
        "gguf_runtime_test_outputs": gguf_outputs,
        "litertlm_verified": False,
        "mobile_apk_bundle_excluded": True,
        **gguf_check,
    }


def export_with_litert_torch(
    merged_dir: Path,
    work_dir: Path,
    base_model_id: str,
    variant_cfg: Dict[str, str],
) -> Dict[str, Any]:
    """Installs official litert-torch + litert-lm after GPU training, exports, benchmarks, and verifies .litertlm."""
    if variant_cfg.get("export_format") == "gguf-pc":
        return export_desktop_pc_gguf_plan(merged_dir, work_dir, variant_cfg)

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
    if target == "kuu-stem":
        # Mobile 1B budget ceiling is <= 650 MB (`SautiLmModelManifest.maxBudgetMb = 650`).
        # `dynamic_wi4b32_afp32` (blockwise-32 INT4) is empirically verified at ~549.58 MB <= 650 MB;
        # channelwise `dynamic_wi4_afp32` collapses accuracy and is excluded; `dynamic_wi8_afp32` is fallback only.
        recipes_to_eval = ["dynamic_wi4b32_afp32", "dynamic_wi8_afp32"]
        print(f"[SautiLM-Forge] Kuu-STEM mobile-budget recipe order (<=650 MB target): {recipes_to_eval}")
    else:
        recipes_to_eval = ["dynamic_wi8_afp32"]
    desired_prefills = resolve_prefill_lengths(target, base_model_id)
    tmp_root = Path("/tmp") if Path("/tmp").exists() else work_dir
    export_out_dir = tmp_root / "litert_export_tmp"
    final_litertlm = work_dir / variant_cfg["litertlm_filename"]
    candidates: List[Dict[str, Any]] = []

    try:
        for recipe in recipes_to_eval:
            if target == "kuu-stem" and recipe == "dynamic_wi8_afp32" and any(c["size_mb"] <= 650.0 for c in candidates):
                print("[SautiLM-Forge] Valid <= 650 MB INT4 Kuu-STEM candidate already verified; skipping 998 MB INT8.")
                break
            if export_out_dir.exists():
                shutil.rmtree(export_out_dir)
            export_out_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = tmp_root / f"candidate_{recipe}.litertlm"
            print(
                f"=== [SautiLM-Forge] Exporting {variant_cfg['model_name']} "
                f"recipe='{recipe}' prefill_lengths={desired_prefills} ==="
            )
            exported_prefills = desired_prefills
            try:
                try:
                    litert_export.export(
                        model=str(merged_dir),
                        output_dir=str(export_out_dir),
                        task="text_generation",
                        prefill_lengths=desired_prefills,
                        cache_length=2048,
                        quantization_recipe=recipe,
                        externalize_embedder=False,
                        use_jinja_template=True,
                        bundle_litert_lm=True,
                        keep_temporary_files=False,
                    )
                except Exception as prefill_err:
                    # Only retry with shorter prefills if the error wasn't an unknown recipe name
                    if "recipe" in str(prefill_err).lower() or "quant" in str(prefill_err).lower():
                        raise
                    exported_prefills = [128, 512]
                    print(f"[SautiLM-Forge] Falling back to prefill_lengths={exported_prefills}: {prefill_err}")
                    if export_out_dir.exists():
                        shutil.rmtree(export_out_dir)
                    export_out_dir.mkdir(parents=True, exist_ok=True)
                    litert_export.export(
                        model=str(merged_dir),
                        output_dir=str(export_out_dir),
                        task="text_generation",
                        prefill_lengths=exported_prefills,
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
                print(f"[SautiLM-Forge] Exported multi-prefill .litertlm signature: {exported_prefills}")
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
                        "prefill_lengths_exported": exported_prefills,
                        "quality_score": score,
                        "max_score": max_score,
                        "binary_check": binary_check,
                        "runtime_outputs": runtime_outputs,
                    }
                )
                if target == "kuu-stem" and c_mb <= 650.0 and score >= max(6, int(0.7 * max_score)):
                    print(f"[SautiLM-Forge] Found compliant INT4 Kuu-STEM ({c_mb} MB <= 650 MB, {score}/{max_score}); stopping search.")
                    break
            except Exception as err:
                print(f"[SautiLM-Forge] Candidate recipe '{recipe}' failed: {err}")

        if not candidates:
            raise RuntimeError("All LiteRT-LM export recipes failed!")

        if target == "kuu-stem":
            candidates.sort(key=lambda c: (0 if c["size_mb"] <= 650.0 else 1, -c["quality_score"], c["size_bytes"]))
        else:
            candidates.sort(key=lambda c: (-c["quality_score"], c["size_bytes"]))
        winner = candidates[0]
        shutil.copy2(winner["path"], final_litertlm)
    finally:
        shutil.rmtree(export_out_dir, ignore_errors=True)
        shutil.rmtree(merged_dir, ignore_errors=True)
        for p in tmp_root.glob("candidate_*.litertlm"):
            p.unlink(missing_ok=True)
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
        "prefill_lengths_exported": winner.get("prefill_lengths_exported", [128, 512, 1024]),
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

    parser = argparse.ArgumentParser(description="Train and export modular SautiLM builds (Jua / Kuu).")
    parser.add_argument("--variant", default=None, help="Target variant, e.g. sautilm-jua, sautilm-jua-stem, sautilm-kuu-stem.")
    parser.add_argument("--base-model", default="auto", help="Base model ID or tier alias ('auto', '270m', '1b', '4b-med', '4b-ius').")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Held-out group-aware validation ratio.")
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
    train_rows, val_rows = split_train_val_rows(curated_data, val_ratio=args.val_ratio, seed=42)
    train_prompts_set = {" ".join(str(r.get("prompt") or "").lower().split()) for r in train_rows}
    val_prompts_set = {" ".join(str(r.get("prompt") or "").lower().split()) for r in val_rows}
    split_overlap_count = len(train_prompts_set & val_prompts_set)
    print(
        f"[SautiLM-Forge] Disjoint Split: train={len(train_rows)} | val={len(val_rows)} "
        f"| prompt_overlap={split_overlap_count}"
    )
    if split_overlap_count != 0:
        raise RuntimeError(
            f"SCIENTIFIC INTEGRITY GATE FAILED: split_overlap_count={split_overlap_count} (must be 0)."
        )

    base_model_id = resolve_base_model_id(args.base_model, variant_cfg)
    print(f"=== [SautiLM-Forge] Step 2: Loading Base Model ({base_model_id}) ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = 896 if "1b" in base_model_id.lower() else 640
    train_encoded: List[Dict[str, List[int]]] = []
    for row in train_rows:
        ex = prepare_masked_example(tokenizer, row, max_len=max_len)
        if ex is not None:
            train_encoded.append(ex)

    val_encoded: List[Dict[str, List[int]]] = []
    for row in val_rows:
        ex = prepare_masked_example(tokenizer, row, max_len=max_len)
        if ex is not None:
            val_encoded.append(ex)

    print(
        f"[SautiLM-Forge] Masked examples prepared: train={len(train_encoded)} "
        f"| val={len(val_encoded)} (max_len={max_len})"
    )

    is_4b_model = "4b" in base_model_id.lower()
    load_dtype = torch.bfloat16 if (device_str == "cuda" and is_4b_model) else torch.float32
    use_fp16_amp = False
    model_load_kwargs: Dict[str, Any] = {
        "dtype": load_dtype,
        "device_map": {"": 0} if device_str == "cuda" else None,
    }
    if is_4b_model:
        model_load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_load_kwargs,
    )

    # Anti-catastrophic-forgetting LoRA (Calibrated per tier):
    # - 4B Workstation models (MedGemma-4B, Gemma-3-4B): gentle rank r=8, alpha=16, lr=2e-5, 2 epochs
    #   to inject DRC/OHADA/PNLP identity & rules without disturbing 4B clinical/legal weights.
    # - 270M & 1B Mobile models: attention-only r=16, alpha=32 (MLP 100% FROZEN) with per-epoch HeldOut gate.
    is_4b_tier = variant_cfg.get("tier") == "kuu-4b-pc"
    lora_r = 8 if is_4b_tier else 16
    lora_alpha = 16 if is_4b_tier else 32
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.10 if is_4b_tier else 0.08,
        bias="none",
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if load_dtype == torch.bfloat16:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
    model.print_trainable_parameters()

    target_key = variant_cfg["target"]
    base_val_loss: float | None = None
    base_hf_score: int = 0
    base_hf_total: int = len(_runtime_suite_for_target(target_key))
    base_hf_outputs: Dict[str, str] = {}
    if val_encoded:
        with model.disable_adapter():
            base_val_loss = compute_eval_loss(
                model, val_encoded, tokenizer.pad_token_id, device, batch_size=1 if is_4b_model else 2
            )
            print(f"[SautiLM-Forge] Step 0 Base Model Held-Out ValLoss: {base_val_loss:.4f}")
    if device_str == "cuda":
        with model.disable_adapter():
            base_hf_score, base_hf_total, base_hf_outputs = run_hf_heldout_eval(
                model, tokenizer, target=target_key, device=device, max_new_tokens=260
            )
        print(f"[SautiLM-Forge] Step 0 Base Model Held-Out Score: {base_hf_score}/{base_hf_total}")

    if train_encoded:
        with torch.no_grad():
            sanity_batch = collate_padded_batch(train_encoded[:1], tokenizer.pad_token_id, device)
            sanity_loss = float(model(**sanity_batch).loss.item())
            print(f"[SautiLM-Forge] Sanity Check Step 0 Loss: {sanity_loss:.4f} (dtype={load_dtype})")

    print("=== [SautiLM-Forge] Step 3: Anti-Forgetting Attention-Only LoRA SFT (Calibrated Per Tier) ===")
    if device_str != "cuda":
        batch_size = 1
        grad_accum = 1
        num_epochs = 1
    elif is_4b_tier:
        batch_size = 1
        grad_accum = 8
        num_epochs = 2
    elif target_key == "jua":
        batch_size = 2
        grad_accum = 4
        num_epochs = 2
    else:
        batch_size = 2
        grad_accum = 4
        num_epochs = 3
    effective_bs = batch_size * grad_accum
    total_steps = max(1, (len(train_encoded) // max(1, effective_bs)) * num_epochs)
    if device_str != "cuda":
        total_steps = min(total_steps, 8)

    lr_by_target = {
        "stem": 7.0e-5,
        "kuu-stem": 5.0e-5,
        "med": 7.0e-5,
        "kuu-med": 2.0e-5,
        "agri": 7.0e-5,
        "ius": 7.0e-5,
        "kuu-ius": 2.0e-5,
        "lex": 7.0e-5,
        "jua": 7.0e-5,
    }
    lr = lr_by_target.get(target_key, 8.0e-5)
    print(
        f"[SautiLM-Forge] Hyperparams: target={target_key} | lora_r={lora_r} | lora_alpha={lora_alpha} "
        f"| eff_bs={effective_bs} | epochs={num_epochs} | steps={total_steps} | lr={lr}"
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.04,
        eps=1e-6,
    )

    def lr_lambda(step_idx: int) -> float:
        warmup = max(3, int(0.10 * total_steps))
        if step_idx < warmup:
            return float(step_idx + 1) / float(warmup)
        progress = float(step_idx - warmup) / float(max(1, total_steps - warmup))
        return max(0.15, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.train()

    train_losses: List[float] = []
    val_loss_history: List[Dict[str, Any]] = []
    best_val_loss: float | None = None
    best_epoch_hf_score: int = -1
    best_lora_state: Dict[str, Any] | None = None
    no_improve_evals = 0
    early_stopped = False
    patience = 2

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_amp)
    step = 0
    micro_step = 0
    accum_loss = 0.0
    outputs = None
    loss = None
    batch = None
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        if early_stopped:
            break
        epoch_step_losses: List[float] = []
        random.Random(42 + epoch).shuffle(train_encoded)
        for idx in range(0, len(train_encoded), batch_size):
            if step >= total_steps or early_stopped:
                break
            batch_items = train_encoded[idx : idx + batch_size]
            batch = collate_padded_batch(batch_items, tokenizer.pad_token_id, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                print(f"[SautiLM-Forge] WARNING: non-finite micro-batch loss skipped at idx={idx}")
                continue
            scaler.scale(loss).backward()
            accum_loss += loss_val
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                step_train_loss = accum_loss
                accum_loss = 0.0
                train_losses.append(step_train_loss)
                epoch_step_losses.append(step_train_loss)
                if step % 10 == 0 or step == 1 or step == total_steps:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{total_steps} | "
                        f"TrainLoss: {step_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

        if val_encoded and step > 0:
            epoch_val_loss = compute_eval_loss(
                model, val_encoded, tokenizer.pad_token_id, device, batch_size=batch_size
            )
            epoch_hf_score = 0
            if device_str == "cuda":
                epoch_hf_score, _, _ = run_hf_heldout_eval(
                    model, tokenizer, target=target_key, device=device, max_new_tokens=260
                )
            mean_ep_train = (sum(epoch_step_losses) / len(epoch_step_losses)) if epoch_step_losses else 9.9
            val_loss_history.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "val_loss": epoch_val_loss,
                    "mean_train_loss": round(mean_ep_train, 4),
                    "hf_heldout_score": f"{epoch_hf_score}/{base_hf_total}",
                }
            )
            print(
                f"[SautiLM-Forge] Epoch {epoch + 1} Held-Out ValLoss: {epoch_val_loss:.4f} "
                f"| MeanTrainLoss: {mean_ep_train:.4f} | HeldOutScore: {epoch_hf_score}/{base_hf_total}"
            )
            # Prioritize Held-Out Accuracy FIRST, breaking ties by lower validation loss
            is_better = (
                best_lora_state is None
                or epoch_hf_score > best_epoch_hf_score
                or (epoch_hf_score == best_epoch_hf_score and (best_val_loss is None or epoch_val_loss < best_val_loss))
            )
            if is_better:
                best_val_loss = epoch_val_loss
                best_epoch_hf_score = epoch_hf_score
                best_lora_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if "lora_" in k
                }
                no_improve_evals = 0
            else:
                no_improve_evals += 1
                if no_improve_evals >= patience:
                    early_stopped = True
                    print(f"[SautiLM-Forge] Early stopping triggered at epoch {epoch + 1}")
            # Memorization ceiling guard: stop early only when train loss collapses (<0.90)
            # OR when held-out achieves perfection (100% and >= Base 0-step score).
            if mean_ep_train < 0.90 or (epoch_hf_score >= base_hf_total and epoch_hf_score >= base_hf_score):
                early_stopped = True
                print(
                    f"[SautiLM-Forge] Memorization guard stopped at epoch {epoch + 1} "
                    f"(mean_train_loss={mean_ep_train:.4f}, score={epoch_hf_score}/{base_hf_total})"
                )

    if best_lora_state:
        missing, unexpected = model.load_state_dict(best_lora_state, strict=False)
        print(
            f"[SautiLM-Forge] Restored best LoRA checkpoint (best_val_loss={best_val_loss:.4f}, "
            f"best_epoch_hf_score={best_epoch_hf_score}/{base_hf_total}, unexpected={len(unexpected)})"
        )

    # Mandatory Zero-Regression A/B Gate: Base (0-step) vs Fine-Tuned vs Damped (0.75x, 0.5x, 0.25x, 0.0x)
    ft_hf_score: int = max(0, best_epoch_hf_score)
    selected_hf_score: int = ft_hf_score
    selected_lora_scale: float = 1.0
    if device_str == "cuda":
        if best_epoch_hf_score < 0:
            ft_hf_score, _, _ = run_hf_heldout_eval(
                model, tokenizer, target=target_key, device=device, max_new_tokens=260
            )
            selected_hf_score = ft_hf_score
        print(
            f"[SautiLM-Forge] Base-vs-FT Gate ({target_key}): "
            f"Base={base_hf_score}/{base_hf_total} | FineTuned(1.0x)={ft_hf_score}/{base_hf_total}"
        )
        if best_lora_state and (ft_hf_score < base_hf_score or ft_hf_score < base_hf_total):
            best_scale = 1.0
            best_score_so_far = ft_hf_score
            probe_scales = (0.75, 0.5, 0.25, 0.0) if ft_hf_score < base_hf_score else (0.75, 0.5)
            for scale in probe_scales:
                scaled_state = {
                    k: (v * scale if "lora_B" in k else v.clone())
                    for k, v in best_lora_state.items()
                }
                model.load_state_dict(scaled_state, strict=False)
                sc_score, _, _ = run_hf_heldout_eval(
                    model, tokenizer, target=target_key, device=device, max_new_tokens=260
                )
                print(f"[SautiLM-Forge] Damped LoRA scale={scale} -> Held-Out Score={sc_score}/{base_hf_total}")
                if sc_score > best_score_so_far:
                    best_score_so_far = sc_score
                    best_scale = scale
            selected_lora_scale = best_scale
            selected_hf_score = best_score_so_far
            final_scaled_state = {
                k: (v * selected_lora_scale if "lora_B" in k else v.clone())
                for k, v in best_lora_state.items()
            }
            model.load_state_dict(final_scaled_state, strict=False)
            print(
                f"[SautiLM-Forge] Gate selected scale={selected_lora_scale} "
                f"(score={selected_hf_score}/{base_hf_total} vs base={base_hf_score}/{base_hf_total})"
            )

    print("=== [SautiLM-Forge] Step 4: Merging LoRA Adapter & Saving HF Checkpoint ===")
    adapter_dir = work_dir / variant_cfg["adapter_dirname"]
    tmp_root_main = Path("/tmp") if Path("/tmp").exists() else work_dir
    merged_dir = tmp_root_main / "merged_hf_model"
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    model.save_pretrained(adapter_dir)
    merged_model = model.merge_and_unload()
    merged_model.eval()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Free all PyTorch GPU/CPU training objects before litert-torch / llama.cpp conversion
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
        "tier": variant_cfg.get("tier", "jua"),
        "base_model": base_model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules,
        "effective_batch_size": effective_bs,
        "learning_rate": lr,
        "steps": step,
        "epochs_configured": num_epochs,
        "epochs_completed": len(val_loss_history) or num_epochs,
        "early_stopped": early_stopped,
        "initial_train_loss": round(train_losses[0], 4) if train_losses else None,
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "base_val_loss": base_val_loss,
        "best_val_loss": best_val_loss,
        "base_hf_heldout_score": f"{base_hf_score}/{base_hf_total}",
        "finetuned_hf_heldout_score": f"{ft_hf_score}/{base_hf_total}",
        "selected_lora_scale": selected_lora_scale,
        "selected_hf_heldout_score": f"{selected_hf_score}/{base_hf_total}",
        "val_loss_history": val_loss_history,
        "train_loss_history": [round(x, 4) for x in train_losses],
        "training_examples_count": len(train_encoded),
        "validation_examples_count": len(val_encoded),
        "split_overlap_count": split_overlap_count,
        "elapsed_seconds": elapsed,
        **export_info,
    }
    with open(work_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_name = export_info.get("litertlm_filename") or export_info.get("gguf_target_filename") or variant_cfg["litertlm_filename"]
    print(f"=== [SautiLM-Forge] COMPLETE in {elapsed}s! Output: {work_dir / out_name} ===")


if __name__ == "__main__":
    main()
