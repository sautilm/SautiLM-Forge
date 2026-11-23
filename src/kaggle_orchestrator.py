import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
KERNEL_DIR = BASE_DIR / "kaggle_kernel"
OUTPUT_DIR = BASE_DIR / "kaggle_output"


ALL_MODULAR_TARGETS = ("jua", "stem", "kuu-stem", "lex", "med", "agri", "ius")
DESKTOP_PC_TARGETS = ("kuu-med", "kuu-ius")
VISION_NURU_TARGETS = ("nuru-agri",)


def _with_kaggle_retry(fn, label: str = "kaggle_api", max_attempts: int = 3):
    """Retries transient Kaggle API network resets cleanly with backoff."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as err:
            last_err = err
            if attempt == max_attempts:
                raise
            wait_s = attempt * 4
            print(f"[SautiLM-Forge] Transient {label} error ({err}); retry {attempt}/{max_attempts - 1} in {wait_s}s...")
            time.sleep(wait_s)
    raise last_err


def _resolve_target_list(raw_target: str) -> list[str]:
    cleaned = (raw_target or "jua").strip().lower()
    if cleaned == "all":
        return list(ALL_MODULAR_TARGETS)
    if cleaned == "all-with-desktop":
        return list(ALL_MODULAR_TARGETS) + list(DESKTOP_PC_TARGETS)
    return [t.strip().lower() for t in cleaned.split(",") if t.strip()]


def _target_config(target: str = "jua") -> Dict[str, str]:
    """Returns Kaggle dataset, kernel, Hugging Face repo, and artifact metadata per SautiLM target."""
    t = target.strip().lower()
    if t == "nuru-agri":
        return {
            "target": "nuru-agri",
            "corpus_domain": "vision-agri",
            "variant": "sautilm-nuru-agri",
            "model_name": "SautiLM-Nuru-Agri",
            "dataset_slug": "sautilm-nuru-agri-dataset",
            "dataset_title": "SautiLM-Nuru-Agri Collaborative Vision Metadata",
            "kernel_slug": "sautilm-nuru-agri-tflite-vision-gpu",
            "kernel_title": "SautiLM Nuru Agri TFLite Vision GPU",
            "hf_repo": "SautiLM-Nuru-Agri",
            "jsonl_filename": "sautilm_nuru_agri_manifest.json",
            "stats_filename": "corpus_stats_nuru_agri.json",
            "litertlm_filename": "sautilm-nuru-agri.tflite",
            "adapter_dirname": "sautilm_nuru_agri_savedmodel",
            "export_format": "tflite-int8",
        }
    if t == "jua":
        return {
            "target": "jua",
            "variant": "sautilm-jua",
            "model_name": "SautiLM-Jua",
            "dataset_slug": "sautilm-v1-corpus",
            "dataset_title": "SautiLM-Jua Clean Multilingual & Domain Corpus",
            "kernel_slug": "sautilm-v1-forge-gpu",
            "kernel_title": "SautiLM v1 Forge GPU",
            "hf_repo": "SautiLM-Jua",
            "jsonl_filename": "sautilm_v1_train.jsonl",
            "stats_filename": "corpus_stats.json",
            "litertlm_filename": "sautilm-jua.litertlm",
            "adapter_dirname": "sautilm_jua_lora_adapter",
        }
    if t.startswith("kuu-"):
        domain = t[4:]
        suffix = domain.upper() if domain == "stem" else domain.capitalize()
        is_mobile_1b = domain == "stem"
        return {
            "target": t,
            "corpus_domain": domain,
            "variant": f"sautilm-kuu-{domain}",
            "model_name": f"SautiLM-Kuu-{suffix}",
            "dataset_slug": f"sautilm-kuu-{domain}-corpus",
            "dataset_title": f"SautiLM-Kuu-{suffix} Standalone Domain Corpus",
            "kernel_slug": f"sautilm-kuu-{domain}-forge-gpu",
            "kernel_title": f"SautiLM-Kuu-{suffix} Forge GPU",
            "hf_repo": f"SautiLM-Kuu-{suffix}",
            "jsonl_filename": f"sautilm_jua_{domain}_train.jsonl",
            "stats_filename": f"corpus_stats_{domain}.json",
            "litertlm_filename": (
                f"sautilm-kuu-{domain}.litertlm" if is_mobile_1b else f"sautilm-kuu-{domain}-q4_k_m.gguf"
            ),
            "adapter_dirname": f"sautilm_kuu_{domain}_lora_adapter",
        }
    suffix = t.upper() if t == "stem" else t.capitalize()
    return {
        "target": t,
        "corpus_domain": t,
        "variant": f"sautilm-jua-{t}",
        "model_name": f"SautiLM-Jua-{suffix}",
        "dataset_slug": f"sautilm-jua-{t}-corpus",
        "dataset_title": f"SautiLM-Jua-{suffix} Standalone Domain Corpus",
        "kernel_slug": f"sautilm-jua-{t}-forge-gpu",
        "kernel_title": f"SautiLM-Jua-{suffix} Forge GPU",
        "hf_repo": f"SautiLM-Jua-{suffix}",
        "jsonl_filename": f"sautilm_jua_{t}_train.jsonl",
        "stats_filename": f"corpus_stats_{t}.json",
        "litertlm_filename": f"sautilm-jua-{t}.litertlm",
        "adapter_dirname": f"sautilm_jua_{t}_lora_adapter",
    }


def _load_local_auth() -> Dict[str, str]:
    """Loads credentials strictly into local process memory from .env without printing."""
    load_dotenv(ENV_PATH)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    kaggle_user = os.environ.get("KAGGLE_USERNAME", "georgesbyona").strip()
    kaggle_key = os.environ.get("KAGGLE_KEY", "").strip()

    if not hf_token or not kaggle_key:
        raise RuntimeError("Missing HF_TOKEN or KAGGLE_KEY in local .env file.")

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key
    os.environ["KAGGLE_API_TOKEN"] = kaggle_key
    return {
        "hf_token": hf_token,
        "kaggle_user": kaggle_user,
    }


def publish_dataset_to_kaggle(
    skip_if_exists: bool = False,
    target: str = "jua",
    refresh_corpus: bool = True,
) -> str:
    """Uploads the requested SautiLM target corpus as a private Kaggle dataset."""
    cfg = _target_config(target)
    jsonl_src = DATA_DIR / cfg["jsonl_filename"]
    stats_src = DATA_DIR / cfg["stats_filename"]
    if (refresh_corpus or not jsonl_src.exists() or not stats_src.exists()) and cfg["target"] != "nuru-agri":
        try:
            from .corpus_builder import build_target_corpus
        except ImportError:
            import sys
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            from src.corpus_builder import build_target_corpus

        build_target_corpus(cfg.get("corpus_domain", cfg["target"]), DATA_DIR)

    auth = _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    kaggle_user = auth["kaggle_user"]
    dataset_ref = f"{kaggle_user}/{cfg['dataset_slug']}"

    existing = _with_kaggle_retry(lambda: [d.ref for d in api.dataset_list(mine=True)], "dataset_list")
    if dataset_ref in existing and skip_if_exists:
        print(f"[SautiLM-Forge] Private Kaggle dataset already exists: {dataset_ref}")
        return dataset_ref

    staging_dir = BASE_DIR / "build" / f"kaggle_dataset_{cfg['target']}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if cfg["target"] == "nuru-agri":
        manifest_payload = {
            "model_name": "SautiLM-Nuru-Agri",
            "labels": [
                "non_plant",
                "healthy_leaf",
                "cassava_cmd",
                "cassava_cbsd",
                "maize_rust",
                "maize_faw",
                "banana_sigatoka",
                "bean_angular_leaf_spot",
                "tomato_blight",
            ],
            "abstention_threshold": 0.55,
            "max_model_mb": 45.0,
            "synthetic_images_allowed": False,
        }
        (staging_dir / cfg["jsonl_filename"]).write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging_dir / cfg["stats_filename"]).write_text(
            json.dumps({"total_instructions": 9, "lexicon_count": 1150, "target": "nuru-agri"}, indent=2),
            encoding="utf-8",
        )
    else:
        if jsonl_src.exists():
            shutil.copy2(jsonl_src, staging_dir / cfg["jsonl_filename"])
        if stats_src.exists():
            shutil.copy2(stats_src, staging_dir / cfg["stats_filename"])
    if cfg["target"] in ("jua", "lex") and (DATA_DIR / "bbadha_full_lexicon.json").exists():
        shutil.copy2(DATA_DIR / "bbadha_full_lexicon.json", staging_dir / "bbadha_full_lexicon.json")

    with open(staging_dir / "build_target.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": cfg["target"],
                "variant": cfg["variant"],
                "model_name": cfg["model_name"],
                "jsonl_filename": cfg["jsonl_filename"],
                "litertlm_filename": cfg["litertlm_filename"],
            },
            f,
            indent=2,
        )

    metadata = {
        "title": cfg["dataset_title"],
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if dataset_ref in existing:
        print(f"[SautiLM-Forge] Updating existing private Kaggle dataset: {dataset_ref}")
        _with_kaggle_retry(
            lambda: api.dataset_create_version(
                str(staging_dir),
                version_notes=f"Update {cfg['model_name']} modular zero-leakage SFT corpus",
                quiet=False,
                convert_to_csv=False,
                dir_mode="zip",
            ),
            "dataset_create_version",
        )
    else:
        print(f"[SautiLM-Forge] Creating new private Kaggle dataset: {dataset_ref}")
        _with_kaggle_retry(
            lambda: api.dataset_create_new(
                str(staging_dir),
                public=False,
                quiet=False,
                convert_to_csv=False,
                dir_mode="zip",
            ),
            "dataset_create_new",
        )

    return dataset_ref


def push_gpu_kernel_to_kaggle(dataset_ref: str, target: str = "jua") -> str:
    """Pushes the GPU fine-tuning and LiteRT-LM / TFLite export kernel to Kaggle for the given target."""
    cfg = _target_config(target)
    auth = _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    kaggle_user = auth["kaggle_user"]
    kernel_ref = f"{kaggle_user}/{cfg['kernel_slug']}"
    is_nuru_vision = cfg["target"] == "nuru-agri"
    push_dir = (BASE_DIR / "kaggle_nuru_kernel") if is_nuru_vision else KERNEL_DIR
    code_filename = "train_nuru_vision_kaggle.py" if is_nuru_vision else "train_and_export.py"
    dataset_sources = [dataset_ref]
    if is_nuru_vision:
        dataset_sources.extend([
            "emmarex/plantdisease",
            "nirmalsankalana/cassava-leaf-disease-classification",
            "smaranjitghose/corn-or-maize-leaf-disease-dataset",
            "humphreyassibi/armyworm-and-affected-plants",
            "mitchmatheri/ibeans",
            "shifatearman/bananalsd",
            "puneet6060/intel-image-classification",
        ])

    kernel_meta = {
        "id": kernel_ref,
        "title": cfg["kernel_title"],
        "code_file": code_filename,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    push_dir.mkdir(parents=True, exist_ok=True)
    with open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)

    print(f"[SautiLM-Forge] Pushing GPU Kernel to Kaggle: {kernel_ref}")
    res = _with_kaggle_retry(lambda: api.kernels_push(str(push_dir)), "kernels_push")
    err_msg = getattr(res, "error", None)
    if err_msg:
        raise RuntimeError(f"Kaggle kernel push rejected for {kernel_ref}: {err_msg}")
    print(f"[SautiLM-Forge] Kernel push response: {getattr(res, 'url', res)}")
    return kernel_ref


def build_hf_model_card(metrics: Dict[str, Any] | None = None, target: str = "jua") -> str:
    """Builds accurate Hugging Face Model Card across SautiLM-Jua (270M), Nuru (TFLite), and Kuu (1B/4B)."""
    cfg = _target_config(target)
    t_key = cfg["target"]
    stats_file = DATA_DIR / cfg["stats_filename"]
    corpus_total = 23500 if t_key == "jua" else 260
    lex_count = 1150
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                st = json.load(f)
                corpus_total = st.get("total_instructions", corpus_total)
                lex_count = st.get("lexicon_count", lex_count)
        except Exception:
            pass

    if t_key == "kuu-stem":
        default_base = "google/gemma-3-1b-it"
        hf_license = "gemma"
        license_extra_yaml = ""
        license_notice = "Governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
        arch_desc = "Google Gemma 3 1B (`Gemma3ForCausalLM`, 26 layers, `hidden_size=1152`, `cache_length=2048`)"
        export_toolchain = "`litert-torch` (`dynamic_wi4b32_afp32` INT4 mobile quantization)"
        lib_name = "litert-lm"
        pipeline_tag = "text-generation"
        target_hw = "AndaaEdge / Shul'etu Capable Mobile & Tablet (Android 8.0+, >= 4 GB RAM)"
    elif t_key == "kuu-med":
        default_base = "google/medgemma-4b-it"
        hf_license = "other"
        license_extra_yaml = (
            "license_name: health-ai-developer-foundations\n"
            "license_link: https://developers.google.com/health-ai-developer-foundations/terms\n"
        )
        license_notice = (
            "Derived from `google/medgemma-4b-it` and governed by the "
            "[Google Health AI Developer Foundations (HAI-DEF) Terms of Use]"
            "(https://developers.google.com/health-ai-developer-foundations/terms) — "
            "NOT standard Gemma terms. **Pre-pilot clinical research prototype only**; not a certified medical device."
        )
        arch_desc = "Google MedGemma 4B IT quantized via `llama.cpp` GGUF `Q4_K_M` (`n_ctx=4096`)"
        export_toolchain = "`llama.cpp` (`Q4_K_M` GGUF 4-bit Desktop quantization)"
        lib_name = "gguf"
        pipeline_tag = "text-generation"
        target_hw = "AfyaSpeak Desktop PC / Clinic Workstation (>= 8 GB RAM, Vulkan/CUDA GPU recommended; excluded from mobile APKs)"
    elif t_key == "kuu-ius":
        default_base = "google/gemma-3-4b-it"
        hf_license = "gemma"
        license_extra_yaml = ""
        license_notice = (
            "Governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). "
            "**Pre-pilot legal research prototype**; requires clinician/lawyer verification."
        )
        arch_desc = "Google Gemma 3 4B IT quantized via `llama.cpp` GGUF `Q4_K_M` (`n_ctx=4096`)"
        export_toolchain = "`llama.cpp` (`Q4_K_M` GGUF 4-bit Desktop quantization)"
        lib_name = "gguf"
        pipeline_tag = "text-generation"
        target_hw = "Legal Desktop Workstation for DRC Courts & Lawyers (>= 8 GB RAM, Vulkan/CUDA GPU recommended; excluded from mobile APKs)"
    elif t_key == "nuru-agri":
        default_base = "tf.keras.applications.MobileNetV3Small"
        hf_license = "apache-2.0"
        license_extra_yaml = ""
        license_notice = (
            "Model weights and TFLite export released under **Apache-2.0** "
            "(trained from `tf.keras.applications.MobileNetV3Small` on public agricultural datasets; "
            "no Gemma/MedGemma weights used)."
        )
        arch_desc = "MobileNetV3-Small TFLite INT8 weights (`224x224x3 float32 [0..255]` input, 9 EAC crop + `non_plant` classes)"
        export_toolchain = "TensorFlow / Keras + LiteRT `1.4.2` (`int8` dynamic/full-integer TFLite)"
        lib_name = "tflite"
        pipeline_tag = "image-classification"
        target_hw = "Juamea (`com.musantek.juamea`) Mobile Pass-1 Vision (Pixel 7 measured: 49.73 ms/img, E2E 110 ms)"
    else:
        default_base = "google/gemma-3-270m-it"
        hf_license = "gemma"
        license_extra_yaml = ""
        license_notice = "Governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
        arch_desc = "Google Gemma 3 270M IT (`Gemma3ForCausalLM`, 18 layers, `hidden_size=640`, `cache_length=2048`)"
        export_toolchain = "`litert-torch` (`dynamic_wi8_afp32` INT8 mobile quantization)"
        lib_name = "litert-lm"
        pipeline_tag = "text-generation"
        target_hw = "Android 8.0+ (API 26+), >= 2 GB RAM, ARM Cortex-A53/A55 CPU"

    base_model = (metrics or {}).get("base_model") or default_base
    size_mb = (
        (metrics or {}).get("litertlm_size_mb")
        or (metrics or {}).get("tflite_size_mb")
        or (metrics or {}).get("size_mb")
        or ("1.07" if t_key == "nuru-agri" else "Pending build")
    )
    quant_recipe = (
        "TFLite INT8 weights (`float32 [0..255]` input, LiteRT 1.4.2)"
        if t_key == "nuru-agri"
        else ((metrics or {}).get("quantization_recipe") or cfg.get("export_format") or "dynamic_wi8_afp32")
    )
    lora_r = (metrics or {}).get("lora_r", 8)
    lora_alpha = (metrics or {}).get("lora_alpha", 16)
    m = metrics or {}
    if t_key == "nuru-agri":
        val_pct = float(m.get("final_val_accuracy", 0.9206)) * 100.0
        f1_pct = float(m.get("val_macro_f1", 0.9231)) * 100.0
        synth_cnt = int(m.get("synthetic_images_count", 0))
        img_total = int(m.get("total_images", 4032))
        epochs_n = int(m.get("epochs", 18))
        quality_score = f"Val Acc {val_pct:.2f}% | Macro-F1 {f1_pct:.2f}% | {synth_cnt} synthetic"
        corpus_row = f"{img_total:,} authentic real-world crop & non-plant photos (`0` synthetic images, 9 EAC classes)"
        training_row = f"2-stage MobileNetV3-Small transfer learning ({epochs_n} epochs, cosine LR, focal/class-balanced weights) — NOT LoRA"
    else:
        quality_score = m.get("litertlm_quality_score") or m.get("gguf_quality_score") or "Pending"
        lex_note = f" ({lex_count:,} Bbadha dictionary entries in `bbadha_lexicon.sqlite`)" if t_key in ("jua", "lex") else " (strict domain isolation — 0% Bbadha leakage)"
        corpus_row = f"{corpus_total:,} verified examples{lex_note}"
        final_loss_raw = m.get("final_train_loss") or m.get("final_loss")
        best_val_loss = m.get("best_val_loss")
        final_loss = f"{final_loss_raw:.4f}" if isinstance(final_loss_raw, (int, float)) else "Pending"
        val_loss_str = f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else "Pending"
        steps = m.get("steps", "Pending")
        training_row = f"{steps} steps (LoRA `r={lora_r}, alpha={lora_alpha}`) / Train: `{final_loss}` / Best Val: `{val_loss_str}`"

    base_vs_ft = ""
    if m.get("base_hf_heldout_score") and m.get("selected_hf_heldout_score"):
        base_vs_ft = f" (Base 0-step: `{m['base_hf_heldout_score']}` -> Selected: `{m['selected_hf_heldout_score']}`)"
    verified_flag = bool(
        m and (m.get("litertlm_verified") or m.get("gguf_runtime_verified") or m.get("tflite_verified"))
    )
    verified_badge = (
        f"Verified Runtime ({quality_score}){base_vs_ft}"
        if verified_flag
        else (f"Header Verified Only ({quality_score})" if m.get("gguf_verified") else "Pending Runtime Verification")
    )

    if t_key == "kuu-med":
        limitations_block = (
            "- **Pre-Pilot Clinical Research Prototype (NOT a Certified Medical Device)**: Governed by Google HAI-DEF Terms of Use. "
            "Must be deployed strictly with `afyaspeak_oms_msf.sqlite` (`12,793` pure clinical protocols, WASH engineering isolated), "
            "deterministic `sautilm_sdk` mg/kg calculators, and `AfyaSpeakCopilot.verify_soap_dosing_safety()` numeric post-verifier "
            "(blocking any LLM overdose or unverified mg/mL hallucination; zero free-form pediatric dosing when weight is absent). "
            "Final clinician review & AuthMed signature are mandatory.\n"
            "- **Hardware & ASR Disclosure**: Desktop PC workstation only (`2.38 GB` GGUF, excluded from mobile APKs). "
            "Vulkan GPU (`-ngl 99`) measured at `45–46 s`/SOAP note (`4.96 tok/s` decode); CPU `faster-whisper small` operates in "
            "deferred post-consultation mode (`RTF > 1.0`). Zero real human clinical field recordings collected yet."
        )
    elif t_key == "kuu-ius":
        limitations_block = (
            "- **Pre-Pilot Legal Research Prototype**: Standalone 4B parametric legal weights fail `0/3` on raw held-out legal benchmarks "
            "(`domain_eval_v2`) by fabricating article numbers or OHADA acronyms if ungrounded. In production, **MUST** be paired with "
            "`sautilm_codes_rdc.sqlite` (`6,251` DRC/OHADA passages) FTS5/FTS4 retrieval, `sautilm_sdk.LegalCitationVerifier` "
            "(strict post-generation replacement of ungrounded replies), and deterministic custody/succession calculators.\n"
            "- **Hardware**: Desktop legal workstation only (`2.38 GB` GGUF, `23–28 s`/turn on Vulkan GPU; excluded from mobile APKs)."
        )
    elif t_key == "kuu-stem":
        limitations_block = (
            "- **Mandatory Deterministic `<calc>` / STEM Solver**: Fits mobile budget (`549.58 MB <= 650 MB` INT4 `dynamic_wi4b32_afp32`), "
            "but 1B parametric weights still make algebraic/arithmetic errors unless paired with `sautilm_sdk.solve_stem_formula_prompt()` "
            "and `solve_calc_tags()`. Not yet profiled on a physical Android handset (desktop LiteRT-LM verified only)."
        )
    elif t_key == "nuru-agri":
        limitations_block = (
            "- **Honest Out-of-Distribution (OOD) Field Benchmark (`467` Independent Photos)**: On in-distribution validation (`4,032` real crop photos, `0` synthetic), "
            "`SautiLM-Nuru-Agri v2` achieves `92.06%` accuracy and `92.31%` Macro-F1 (Pixel 7 measured: **`49.73 ms/img`**, **`110 ms`** E2E on LiteRT `1.4.2`). "
            "On blind wild cross-dataset OOD photos (`227` PlantDoc test + Wikimedia = `467` photos), accuracy is **`61.7%` raw** and **`80.3%` selective** "
            "(`conf >= 0.55`, `67.4%` coverage), with documented domain shift on `tomato_blight` (`14.1%`), `banana_sigatoka` (`20.0%`), and wild `non_plant` (`59.1%`). "
            "In **Juamea**, Pass-1 outputs feed strictly into `buildDeterministicAgronomyReport()` over `gemmea_eac_agri.sqlite` (`2,369` CGIAR/IITA/NARS records). "
            "Multi-domain `v3` retraining and field validation in DRC/EAC remain mandatory before phytosanitary certification."
        )
    else:
        limitations_block = (
            "- **Strict 270M Tier Safety Policy (`sautilm_sdk.MODEL_TIER_POLICY`)**: At `270M` parameters, raw parametric weights score "
            "`15.0%` semantic on `domain_eval_v2` and `5.56%` (`47.78%` hybrid RAG) on `bbadha_eval_v1`. In production (`com.musantek.sautilm`), "
            "`SautiLM-Jua*` models are strictly paired with deterministic guards (`BbadhaDeterministicLexiconEngine` achieving `81.11%` honest "
            "zero-hardcoding semantic accuracy on `bbadha_eval_v1`, `ClinicalMobileGuard`, and `LegalCitationVerifier`) — **NEVER** free-form "
            "pediatric dosing, ungrounded legal citations, or unaided dictionary recall."
        )

    bbadha_tag_line = "- bbadha\n" if t_key in ("jua", "lex") else ""
    base_model_yaml = f"base_model: {base_model}\n" if t_key != "nuru-agri" else ""
    intro_data_desc = (
        f"Trained on **{corpus_row}** and exported via {export_toolchain}."
        if t_key == "nuru-agri"
        else f"Zero-leakage training on **{corpus_total:,} verified examples** and exported via {export_toolchain}."
    )

    return f"""---
license: {hf_license}
{license_extra_yaml}{base_model_yaml}library_name: {lib_name}
language:
- fr
- en
- sw
tags:
- sautilm
- {cfg['variant']}
{bbadha_tag_line}- drc
- east-africa
- edge-ai
- {lib_name}
- offline-ai
- musantek
pipeline_tag: {pipeline_tag}
---

# {cfg['model_name']} (Offline Edge-AI Model - {size_mb} MB)

**{cfg['model_name']}** is a sovereign 100% offline Edge-AI model engineered in **Bunia, Ituri, Democratic Republic of the Congo** by **SautiLM Research** for **SautiLM**.

Part of SautiLM's strict multi-tier architecture (**SautiLM-Jua** `270M` mobile standard, **SautiLM-Nuru** `TFLite <= 45 MB` collaborative vision classifiers powering **Juamea**, **SautiLM-Kuu** `1B` mobile / `4B` Desktop workstations, and the planned **SautiLM-Radi** `< 50 ms` ultra-fast Flash tier). {intro_data_desc}

## License & Compliance

{license_notice}

## SautiLM Modular Ecosystem
- **SautiLM-Nuru** (`<= 45 MB` TFLite INT8 Collaborative Vision — **Juamea**): `SautiLM-Nuru-Agri`, `SautiLM-Nuru-STEM`, `SautiLM-Nuru-Med`, `SautiLM-Nuru-Ius`
- **SautiLM-Jua** (`270M` Mobile LiteRT-LM): `SautiLM-Jua-STEM`, `SautiLM-Jua-Lex` (Bbadha 1,150 entries), `SautiLM-Jua-Med`, `SautiLM-Jua-Agri`, `SautiLM-Jua-Ius` (DRC/OHADA Law), `SautiLM-Jua-Fin`, `SautiLM-Jua-Civ`
- **SautiLM-Kuu** (`1B` Mobile / `4B` PC GGUF): `SautiLM-Kuu-STEM`, `SautiLM-Kuu-Med` (AfyaSpeak PC, pre-pilot), `SautiLM-Kuu-Ius` (DRC Courts PC, pre-pilot)
- **SautiLM-Radi** *(Planned Ultra-Fast / Flash Tier, Swahili Lightning)*: ultra-low-latency (`< 50 ms`) routing, reflex classification, and real-time voice/VAD

## Technical Specifications

| Specification | Value |
| :--- | :--- |
| **Model Identifier** | `sautilm/{cfg['hf_repo']}` |
| **Artifact Bundle** | `{cfg['litertlm_filename']}` ({size_mb} MB) |
| **Quantization / Format** | `{quant_recipe}` |
| **Architecture** | {arch_desc} |
| **Training Data** | {corpus_row} |
| **Training Summary** | {training_row} |
| **Runtime Verification** | {verified_badge} |
| **Target Hardware** | {target_hw} |

## Known Limitations & Mandatory Production Guardrails

{limitations_block}

## Research & Founding Team (SautiLM Research)

- **Georges Byona** ([ORCID: 0009-0007-6097-4340](https://orcid.org/0009-0007-6097-4340)) - Co-Founder, Software Engineer & Edge-AI Researcher (Catholic University of Eastern Africa / SautiLM Research)
- **Abraham Imani** ([ORCID: `0009-0000-7431-6153`](https://orcid.org/0009-0000-7431-6153)) - Co-Founder, MSc Student & AI/ML Lead (Université Catholique de Bukavu)

## Citation (Official DOI & BibTeX)

- **Technical Report DOI**: [10.5281/zenodo.24445686](https://doi.org/10.5281/zenodo.24445686)
- **Lead Author ORCID**: [0009-0007-6097-4340](https://orcid.org/0009-0007-6097-4340)

Website: [https://sautilm.com](https://sautilm.com)
Institution: **SautiLM Research** ([https://sautilm.com](https://sautilm.com)) • Technical Partner: [MusanTek](https://musantek.com)
"""


def _output_dir_for_target(target: str) -> Path:
    t = target.strip().lower()
    return OUTPUT_DIR if t == "jua" else (OUTPUT_DIR / t)


def init_and_upload_hf_repo(include_outputs: bool = False, target: str = "jua") -> str:
    """Initializes https://huggingface.co/sautilm/<hf_repo> and uploads model artifacts."""
    cfg = _target_config(target)
    auth = _load_local_auth()
    hf_api = HfApi(token=auth["hf_token"])
    user_info = hf_api.whoami()
    hf_owner = user_info.get("name", "sautilm")
    repo_id = f"{hf_owner}/{cfg['hf_repo']}"

    create_repo(repo_id=repo_id, token=auth["hf_token"], repo_type="model", exist_ok=True, private=False)
    print(f"[SautiLM-Forge] Hugging Face repository ready: https://huggingface.co/{repo_id}")

    out_dir = _output_dir_for_target(cfg["target"])
    metrics = None
    metrics_file = out_dir / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    readme_path = BASE_DIR / "build" / f"HF_README_{cfg['target']}.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(build_hf_model_card(metrics, target=cfg["target"]), encoding="utf-8")

    hf_api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    stats_file = DATA_DIR / cfg["stats_filename"]
    if stats_file.exists():
        hf_api.upload_file(
            path_or_fileobj=str(stats_file),
            path_in_repo=cfg["stats_filename"],
            repo_id=repo_id,
            repo_type="model",
        )

    if include_outputs and out_dir.exists():
        is_verified = bool(
            metrics
            and (
                metrics.get("litertlm_verified")
                or metrics.get("gguf_verified")
                or metrics.get("tflite_verified")
            )
        )
        if not is_verified:
            raise RuntimeError(
                "Refusing to upload unverified artifacts to Hugging Face! "
                "Ensure training_metrics.json confirms litertlm_verified / gguf_verified / tflite_verified = True."
            )
        bundle_file = out_dir / cfg["litertlm_filename"]
        if not bundle_file.exists():
            raise FileNotFoundError(f"No verified bundle found in {out_dir}")
        print(f"[SautiLM-Forge] Uploading verified {bundle_file.name} to {repo_id}...")
        hf_api.upload_file(
            path_or_fileobj=str(bundle_file),
            path_in_repo=cfg["litertlm_filename"],
            repo_id=repo_id,
            repo_type="model",
        )
        manifest_file = out_dir / "sautilm-nuru-agri.manifest.json"
        if manifest_file.exists():
            hf_api.upload_file(
                path_or_fileobj=str(manifest_file),
                path_in_repo=manifest_file.name,
                repo_id=repo_id,
                repo_type="model",
            )
        if metrics_file.exists():
            hf_api.upload_file(
                path_or_fileobj=str(metrics_file),
                path_in_repo="training_metrics.json",
                repo_id=repo_id,
                repo_type="model",
            )
        adapter_dir = out_dir / cfg["adapter_dirname"]
        if not adapter_dir.exists() and cfg["target"] == "jua":
            adapter_dir = out_dir / "sautilm_v1_lora_adapter"
        if adapter_dir.exists():
            hf_api.upload_folder(
                folder_path=str(adapter_dir),
                path_in_repo="lora_adapter",
                repo_id=repo_id,
                repo_type="model",
            )

    return f"https://huggingface.co/{repo_id}"


def check_or_download_kaggle_kernel(kernel_ref: str, target: str = "jua") -> Dict[str, Any]:
    """Checks the status of the Kaggle GPU kernel and downloads outputs with safe partial-file resume."""
    cfg = _target_config(target)
    _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    status_resp = _with_kaggle_retry(lambda: api.kernels_status(kernel_ref), "kernels_status")
    status_str = str(getattr(status_resp, "status", status_resp))
    failure_msg = getattr(status_resp, "failureMessage", None)

    result = {
        "kernel_ref": kernel_ref,
        "target": cfg["target"],
        "status": status_str,
        "failure_message": failure_msg,
    }
    if "complete" in status_str.lower():
        out_dir = _output_dir_for_target(cfg["target"])
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "training_metrics.json"
        prev_elapsed = None
        if metrics_path.exists():
            try:
                prev_elapsed = json.loads(metrics_path.read_text(encoding="utf-8")).get("elapsed_seconds")
            except Exception:
                prev_elapsed = None

        _with_kaggle_retry(
            lambda: api.kernels_output(
                kernel_ref,
                path=str(out_dir),
                file_pattern=r"^training_metrics\.json$",
                force=True,
                quiet=False,
            ),
            "kernels_output_metrics",
        )
        remote_metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                remote_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                remote_metrics = {}

        is_new_run = prev_elapsed is not None and remote_metrics.get("elapsed_seconds") != prev_elapsed
        expected_bytes = remote_metrics.get("litertlm_size_bytes")
        litert_path = out_dir / cfg["litertlm_filename"]
        if litert_path.exists() and (
            is_new_run or (expected_bytes is not None and litert_path.stat().st_size > int(expected_bytes))
        ):
            litert_path.unlink()

        escaped_litert = cfg["litertlm_filename"].replace(".", r"\.")
        escaped_adapter = cfg["adapter_dirname"]
        pattern = rf"^({escaped_litert}|sautilm-nuru-agri\.manifest\.json|sautilm-jua\.litertlm|sautilm-v1\.litertlm|{escaped_adapter}/.*|sautilm_v1_lora_adapter/.*)$"
        _with_kaggle_retry(
            lambda: api.kernels_output(
                kernel_ref,
                path=str(out_dir),
                file_pattern=pattern,
                force=False,
                quiet=False,
            ),
            "kernels_output_bundle",
        )
        result["downloaded_to"] = str(out_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate modular SautiLM Kaggle training & Hugging Face uploads.")
    parser.add_argument("action", nargs="?", default="launch", choices=("launch", "status", "upload"))
    parser.add_argument(
        "--target",
        default="jua",
        help="Target build ('jua', 'stem', 'kuu-stem', 'kuu-med', 'kuu-ius', 'nuru-agri', 'all', 'all-with-desktop', or comma-separated).",
    )
    cli_args = parser.parse_args()

    targets = _resolve_target_list(cli_args.target)
    outputs = []
    for single_target in targets:
        cfg = _target_config(single_target)
        if cli_args.action == "launch":
            ds_ref = publish_dataset_to_kaggle(skip_if_exists=False, target=cfg["target"], refresh_corpus=True)
            print("Waiting 15s for Kaggle dataset indexing...")
            time.sleep(15)
            k_ref = push_gpu_kernel_to_kaggle(ds_ref, target=cfg["target"])
            outputs.append({"target": cfg["target"], "dataset": ds_ref, "kernel": k_ref})
        elif cli_args.action == "status":
            auth = _load_local_auth()
            k_ref = f"{auth['kaggle_user']}/{cfg['kernel_slug']}"
            outputs.append(check_or_download_kaggle_kernel(k_ref, target=cfg["target"]))
        elif cli_args.action == "upload":
            hf_url = init_and_upload_hf_repo(include_outputs=True, target=cfg["target"])
            outputs.append({"target": cfg["target"], "hf_url": hf_url})
    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2))
