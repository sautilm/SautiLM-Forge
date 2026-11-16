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


def load_and_balance_dataset(input_root: Path) -> List[Dict[str, Any]]:
    """Loads sautilm_v1_train.jsonl, prioritizes key Bbadha + specialty domains, and balances rows."""
    jsonl_files = list(input_root.rglob("sautilm_v1_train.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"Could not find sautilm_v1_train.jsonl in {input_root}")
    jsonl_path = jsonl_files[0]
    print(f"[SautiLM-Forge] Found training corpus: {jsonl_path}")

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            d = row.get("domain", "lex")
            by_domain.setdefault(d, []).append(row)

    random.seed(42)
    curated: List[Dict[str, Any]] = []

    # v1.1 Balanced mix: retain strong EN/FR/SW domain & greeting mastery without overfitting on Bbadha cards
    for d, items in by_domain.items():
        if d != "lex":
            mult = 2 if len(items) >= 120 else 3
            curated.extend(items * mult)

    # For lex: prioritize RAG-grounded entries, natural Q&As, and exact core Bbadha headword cards
    lex_items = by_domain.get("lex", [])
    rag_or_dda_lex: List[Dict[str, Any]] = []
    core_headword_lex: List[Dict[str, Any]] = []
    other_lex: List[Dict[str, Any]] = []
    for item in lex_items:
        sys_lower = str(item.get("system", "")).lower()
        prompt_lower = str(item.get("prompt", "")).lower()
        resp_lower = str(item.get("response", "")).lower()
        is_greeting = len(prompt_lower.strip()) <= 20 and (
            "sautilm-lex" in resp_lower or "hello" in resp_lower or "bonjour" in resp_lower or "habari" in resp_lower
        )
        is_rag = "dictionnaire bbadha verifie" in sys_lower
        is_dda_qa = ("dda" in prompt_lower or "ddà" in prompt_lower) and (
            "**ddà**" in resp_lower or "ddà" in resp_lower
        )
        is_core_headword = any(f"**{w}**" in resp_lower for w in CORE_BBADHA_WORDS)
        if is_greeting or is_rag or is_dda_qa:
            rag_or_dda_lex.append(item)
        elif is_core_headword:
            core_headword_lex.append(item)
        else:
            other_lex.append(item)

    if len(core_headword_lex) > 450:
        core_headword_lex = random.sample(core_headword_lex, 450)
    curated.extend(rag_or_dda_lex * 2)
    curated.extend(core_headword_lex)
    target_other = min(750, max(350, 3400 - len(curated)))
    sampled_other = other_lex if len(other_lex) <= target_other else random.sample(other_lex, target_other)
    curated.extend(sampled_other)

    random.shuffle(curated)
    print(
        f"[SautiLM-Forge] Curated v1.1 dataset: {len(curated)} rows "
        f"(rag_dda={len(rag_or_dda_lex)}x2, core_headword={len(core_headword_lex)}, "
        f"sampled_other_lex={len(sampled_other)})"
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


def _score_runtime_outputs(outputs: Dict[str, str]) -> int:
    """Scores grounded factual accuracy & hallucination resistance across 15 EN/FR/SW & greeting checks (max 15)."""
    score = 0
    checks = {
        "identity_fr": lambda t: ("sautilm" in t or "losingtek" in t) and "rossi" not in t and "nuru ai research" not in t,
        "identity_en": lambda t: ("sautilm" in t or "losingtek" in t) and "openai" not in t,
        "identity_georges": lambda t: "georges byona" in t and "losingtek" in t and "nuru ai" not in t and "edge-bert" not in t,
        "greeting_fin_hey": lambda t: ("sautilm-fin" in t or "hello" in t or "savings" in t or "finance" in t) and "warsaw" not in t and "mtekmeda" not in t,
        "greeting_voice_hey": lambda t: ("sautilm" in t or "hello" in t or "phonetic" in t) and "explicação" not in t and "mtekmeda" not in t,
        "lex_dda": lambda t: ("eau" in t or "maji" in t or "rivi" in t or "water" in t) and "grosse" not in t and ": (1)" not in t,
        "lex_rag_dda": lambda t: ("eau" in t or "maji" in t or "rivi" in t) and "chien" not in t and "(19)" not in t,
        "ius_garde_a_vue": lambda t: ("48" in t and ("constitution" in t or "avocat" in t or "art" in t)) and "vlan" not in t and "360" not in t,
        "ius_custody_en": lambda t: "48" in t and ("constitution" in t or "lawyer" in t or "art" in t) and "480" not in t,
        "fin_avec_en": lambda t: "120000" in t or "120,000" in t,
        "civ_birth_en": lambda t: "90" in t and ("civil status" in t or "family code" in t or "116" in t) and "ddi.be" not in t,
        "agri_cmd_en": lambda t: ("cassava mosaic" in t or "cmd" in t or "bemisia" in t or "whitefl" in t) and "oats" not in t,
        "vision_stem_en": lambda t: "chlorophyll" in t or "chloroplast" in t or "green leaves" in t,
        "math_pastoral": lambda t: "15" in t,
        "med_cholera": lambda t: "bouillir" in t or "filtr" in t or "chlore" in t or "savon" in t,
    }
    for k, fn in checks.items():
        txt = (outputs.get(k) or "").lower()
        words = txt.split()
        unique_ratio = (len(set(words)) / len(words)) if words else 0.0
        if "<pad>" not in txt and len(words) >= 5 and unique_ratio >= 0.30 and fn(txt):
            score += 1
    return score


def run_litert_lm_runtime_checks(litertlm_path: Path) -> Dict[str, str]:
    """Loads the exported .litertlm in Google's real litert_lm Engine and generates responses."""
    import litert_lm

    test_cases = [
        ("identity_fr", None, "Qui es-tu et qui t'a créé ?"),
        ("identity_en", None, "Who are you? Are you Gemma or ChatGPT?"),
        ("identity_georges", None, "Qui est Georges Byona ?"),
        (
            "greeting_fin_hey",
            "You are SautiLM-Fin (SautiLM Jua), an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English. For brief greetings (like Hey/Hello), reply warmly in 1-2 sentences.",
            "Hey",
        ),
        (
            "greeting_voice_hey",
            "You are SautiLM-Voice (SautiLM Jua), an offline Voice & Pocket Tutor assistant by LosingTek Research. Answer accurately in English. For brief greetings (like Hey/Hello), reply warmly in 1-2 sentences.",
            "Hey",
        ),
        ("lex_dda", None, "Que signifie le mot Bbadha Dda ?"),
        (
            "lex_rag_dda",
            "Tu es SautiLM-Lex, assistant lexicographique hors-ligne de SautiLM. Appuie-toi strictement sur le contexte vérifié ci-dessous.\n"
            "DICTIONNAIRE BBADHA VERIFIE: Mot: ddà | Grammaire: Terme lexical | Kiswahili: maji, kuwa refu | Francais: l'eau, la rivière, -loin.",
            "Que signifie le mot Bbadha ddà ?",
        ),
        (
            "ius_garde_a_vue",
            "Tu es SautiLM-Ius, assistant juridique hors-ligne de droit congolais (RDC). Réponds clairement en français.",
            "Que dit la Constitution de la RDC sur la garde à vue et les droits de la défense ?",
        ),
        (
            "ius_custody_en",
            "You are SautiLM-Ius (SautiLM Jua), an offline Congolese Law & Justice assistant by LosingTek Research. Answer accurately in English.",
            "What does the DRC Constitution say about the 48-hour police custody limit?",
        ),
        (
            "fin_avec_en",
            "You are SautiLM-Fin (SautiLM Jua), an offline Finance & Accounting assistant by LosingTek Research. Answer accurately in English.",
            "In a savings group, 20 members buy 3 shares each at 2000 FC. Total capital?",
        ),
        (
            "civ_birth_en",
            "You are SautiLM-Civ (SautiLM Jua), an offline Civic Administration assistant by LosingTek Research. Answer accurately in English.",
            "How to register a newborn's birth at the Civil Status Office within 90 days?",
        ),
        (
            "agri_cmd_en",
            "You are SautiLM-Agri (SautiLM Jua), an offline Agronomy & Environment assistant by LosingTek Research. Answer accurately in English.",
            "How to identify and control Cassava Mosaic Disease (CMD)?",
        ),
        (
            "vision_stem_en",
            "You are SautiLM-Vision (SautiLM Jua), an offline Vision & STEM Quiz assistant by LosingTek Research. Answer accurately in English.",
            "Generate a STEM quiz question on how photosynthesis works in green leaves.",
        ),
        (
            "math_pastoral",
            None,
            "Un éleveur en Ituri possède 12 vaches (Dyà). Il en achète 8 de plus, puis en vend 5 au marché. Combien lui en reste-t-il ?",
        ),
        (
            "med_cholera",
            "Tu es SautiLM-Med, assistant de santé communautaire hors-ligne. Réponds clairement en français.",
            "Quelles sont les 3 étapes essentielles pour purifier l'eau (Dda) et prévenir le choléra (Rra) ?",
        ),
    ]

    outputs: Dict[str, str] = {}
    with litert_lm.Engine(str(litertlm_path), backend=litert_lm.Backend.CPU()) as engine:
        for key, sys_prompt, user_prompt in test_cases:
            init_messages = (
                [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
                if sys_prompt
                else None
            )
            sampler = litert_lm.SamplerConfig(top_k=32, top_p=0.90, temperature=0.15, seed=42)
            with engine.create_conversation(
                messages=init_messages,
                sampler_config=sampler,
                max_output_tokens=220,
            ) as conv:
                reply = "".join(str(chunk) for chunk in conv.send_message_async(user_prompt)).strip()
            outputs[key] = reply
            print(f"\n[LiteRT-LM LIVE CHECK: {key}]\nPrompt: {user_prompt}\nOutput: {reply[:400]}...")
            if not reply or "<pad>" in reply:
                raise RuntimeError(f"LiteRT-LM failed sanity check (empty or <pad>) on: {key}")

    return outputs


def export_with_litert_torch(merged_dir: Path, work_dir: Path, base_model_id: str) -> Dict[str, Any]:
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

    # Empirical Run 3 benchmark on 270M proved dynamic_wi4b32_afp32 collapses (0/6) while dynamic_wi8_afp32 preserves reasoning.
    recipes_to_eval = ["dynamic_wi8_afp32"]
    export_out_dir = work_dir / "litert_export_tmp"
    final_litertlm = work_dir / "sautilm-jua.litertlm"
    candidates: List[Dict[str, Any]] = []

    try:
        for recipe in recipes_to_eval:
            if export_out_dir.exists():
                shutil.rmtree(export_out_dir)
            export_out_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = work_dir / f"candidate_{recipe}.litertlm"
            print(f"=== [SautiLM-Forge] Exporting merged Gemma3 model with recipe='{recipe}' ===")
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
                runtime_outputs = run_litert_lm_runtime_checks(candidate_file)
                score = _score_runtime_outputs(runtime_outputs)
                c_bytes = candidate_file.stat().st_size
                c_mb = round(c_bytes / (1024 * 1024), 2)
                print(
                    f"[SautiLM-Forge] Candidate '{recipe}': {c_mb} MB | Accuracy Score: {score}/15"
                )
                candidates.append(
                    {
                        "recipe": recipe,
                        "path": candidate_file,
                        "size_bytes": c_bytes,
                        "size_mb": c_mb,
                        "quality_score": score,
                        "binary_check": binary_check,
                        "runtime_outputs": runtime_outputs,
                    }
                )
            except Exception as err:
                print(f"[SautiLM-Forge] Candidate recipe '{recipe}' failed: {err}")

        if not candidates:
            raise RuntimeError("All LiteRT-LM export recipes failed!")

        # Select highest-scoring candidate (tie-break: smaller mobile bundle wins!)
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
    print(
        f"[SautiLM-Forge] Selected '{winner['recipe']}' -> {final_litertlm.name}: "
        f"{size_mb} MB ({size_bytes} bytes) | Score={winner['quality_score']}/15"
    )

    return {
        "litertlm_filename": final_litertlm.name,
        "litertlm_size_bytes": size_bytes,
        "litertlm_size_mb": size_mb,
        "quantization_recipe": winner["recipe"],
        "litertlm_quality_score": f"{winner['quality_score']}/15",
        "litertlm_binary_check": winner["binary_check"],
        "litertlm_runtime_test_outputs": winner["runtime_outputs"],
        "candidates_evaluated": [
            {
                "recipe": c["recipe"],
                "size_mb": c["size_mb"],
                "quality_score": f"{c['quality_score']}/15",
            }
            for c in candidates
        ],
        "litertlm_verified": True,
    }


def main():
    start_time = time.time()
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work_output")
    input_dir = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("./data")
    work_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Device: {device_str} | Torch: {torch.__version__}")
    if device_str == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    curated_data = load_and_balance_dataset(input_dir)

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

    lora_r = 16
    lora_alpha = 32
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
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
    num_epochs = 2 if device_str == "cuda" else 1
    total_steps = max(1, (len(encoded_examples) // effective_bs) * num_epochs)
    if device_str != "cuda":
        total_steps = min(total_steps, 8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-4, weight_decay=0.02)

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
    adapter_dir = work_dir / "sautilm_v1_lora_adapter"
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

    export_info = export_with_litert_torch(merged_dir, work_dir, base_model_id)

    elapsed = round(time.time() - start_time, 2)
    metrics = {
        "model_name": "SautiLM Jua",
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
