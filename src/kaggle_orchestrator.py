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


def _target_config(target: str = "jua") -> Dict[str, str]:
    """Returns Kaggle dataset, kernel, Hugging Face repo, and artifact metadata per SautiLM target."""
    t = target.strip().lower()
    if t == "jua":
        return {
            "target": "jua",
            "variant": "sautilm-jua",
            "model_name": "SautiLM-Jua",
            "dataset_slug": "sautilm-v1-corpus",
            "dataset_title": "SautiLM-Jua Clean Multilingual & Domain Corpus",
            "kernel_slug": "sautilm-v1-forge-gpu",
            "kernel_title": "SautiLM-Jua Forge GPU",
            "hf_repo": "SautiLM-Jua",
            "jsonl_filename": "sautilm_v1_train.jsonl",
            "stats_filename": "corpus_stats.json",
            "litertlm_filename": "sautilm-jua.litertlm",
            "adapter_dirname": "sautilm_jua_lora_adapter",
        }
    suffix = t.upper() if t == "stem" else t.capitalize()
    return {
        "target": t,
        "variant": f"sautilm-jua-{t}",
        "model_name": f"SautiLM-Jua-{suffix}",
        "dataset_slug": f"sautilm-jua-{t}-corpus",
        "dataset_title": f"SautiLM-Jua-{suffix} Standalone Domain Corpus",
        "kernel_slug": f"sautilm-jua-{t}-forge",
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


def publish_dataset_to_kaggle(skip_if_exists: bool = False, target: str = "jua") -> str:
    """Uploads the requested SautiLM target corpus as a private Kaggle dataset."""
    cfg = _target_config(target)
    jsonl_src = DATA_DIR / cfg["jsonl_filename"]
    stats_src = DATA_DIR / cfg["stats_filename"]
    if not jsonl_src.exists() or not stats_src.exists():
        from .corpus_builder import build_target_corpus

        build_target_corpus(cfg["target"], DATA_DIR)

    auth = _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    kaggle_user = auth["kaggle_user"]
    dataset_ref = f"{kaggle_user}/{cfg['dataset_slug']}"

    existing = [d.ref for d in api.dataset_list(mine=True)]
    if dataset_ref in existing and skip_if_exists:
        print(f"[SautiLM-Forge] Private Kaggle dataset already exists: {dataset_ref}")
        return dataset_ref

    staging_dir = BASE_DIR / "build" / f"kaggle_dataset_{cfg['target']}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(jsonl_src, staging_dir / cfg["jsonl_filename"])
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
        api.dataset_create_version(
            str(staging_dir),
            version_notes=f"Update {cfg['model_name']} modular zero-leakage SFT corpus",
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
    else:
        print(f"[SautiLM-Forge] Creating new private Kaggle dataset: {dataset_ref}")
        api.dataset_create_new(
            str(staging_dir),
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )

    return dataset_ref


def push_gpu_kernel_to_kaggle(dataset_ref: str, target: str = "jua") -> str:
    """Pushes the GPU fine-tuning and LiteRT-LM export kernel to Kaggle for the given target."""
    cfg = _target_config(target)
    auth = _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    kaggle_user = auth["kaggle_user"]
    kernel_ref = f"{kaggle_user}/{cfg['kernel_slug']}"

    kernel_meta = {
        "id": kernel_ref,
        "title": cfg["kernel_title"],
        "code_file": "train_and_export.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(KERNEL_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)

    print(f"[SautiLM-Forge] Pushing GPU Kernel to Kaggle: {kernel_ref}")
    res = api.kernels_push(str(KERNEL_DIR))
    print(f"[SautiLM-Forge] Kernel push response: {getattr(res, 'url', res)}")
    return kernel_ref


def build_hf_model_card(metrics: Dict[str, Any] | None = None, target: str = "jua") -> str:
    """Builds the official Hugging Face Model Card for sautilm/SautiLM-Jua or a standalone specialty."""
    cfg = _target_config(target)
    stats_file = DATA_DIR / cfg["stats_filename"]
    corpus_total = 23500 if cfg["target"] == "jua" else 260
    lex_count = 1150
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                st = json.load(f)
                corpus_total = st.get("total_instructions", corpus_total)
                lex_count = st.get("lexicon_count", lex_count)
        except Exception:
            pass

    size_mb = metrics.get("litertlm_size_mb", "Pending build") if metrics else "Pending build"
    quant_recipe = metrics.get("quantization_recipe", "dynamic_wi8_afp32") if metrics else "dynamic_wi8_afp32"
    lora_r = metrics.get("lora_r", 16) if metrics else 16
    lora_alpha = metrics.get("lora_alpha", 32) if metrics else 32
    quality_score = metrics.get("litertlm_quality_score", "Pending") if metrics else "Pending"
    final_loss = f"{metrics['final_loss']:.4f}" if (metrics and metrics.get("final_loss") is not None) else "Pending"
    steps = metrics.get("steps", "Pending") if metrics else "Pending"
    verified_badge = f"Verified via LiteRT-LM Runtime ({quality_score})" if (metrics and metrics.get("litertlm_verified")) else "Pending Runtime Verification"

    return f"""---
license: gemma
base_model: google/gemma-3-270m-it
library_name: litert-lm
language:
- fr
- en
- sw
tags:
- sautilm
- sautilm-jua
- {cfg['variant']}
- bbadha
- drc
- edge-ai
- litertlm
- offline-ai
- losingtek
pipeline_tag: text-generation
---

# {cfg['model_name']} (Offline Edge-AI Model - {size_mb} MB)

**{cfg['model_name']}** is a sovereign 100% offline Edge-AI model engineered in **Bunia, Ituri, Democratic Republic of the Congo** by **LosingTek Research** for **SautiLM**.

Built directly on **Google Gemma 3 (`270M-IT`)** with masked-loss LoRA Supervised Fine-Tuning (SFT) on **{corpus_total:,} verified zero-leakage instructions** and compiled with Google's official `litert-torch` (`{quant_recipe}`) into native `.litertlm` container format (`{cfg['litertlm_filename']}`, **{size_mb} MB**), executing natively on Android smartphones via LiteRT-LM with zero internet connection.

---

## SautiLM-Jua Modular Architecture (Strictly Separated Specialties)

- **SautiLM-Jua** : Clean multi-domain offline foundation model (`sautilm-jua.litertlm`)
- **SautiLM-Jua-STEM** : Secondary School & DRC Examen d'État STEM (60% French / 40% English; Maths, Physics, Chemistry, Biology; partner integration: Andaa Edge)
- **SautiLM-Jua-Lex** : Bbadha (Kilendu), Kiswahili, French & English lexicography (**{lex_count:,} verified dictionary entries**)
- **SautiLM-Jua-Med** : Community public health & medical protocols (partner integration: AfyaSpeak)
- **SautiLM-Jua-Agri** : Agronomy & crop protection (partner integration: Gemmea)
- **SautiLM-Jua-Ius** : Congolese Law (DRC Constitution, Family Code, Penal Code, Labor Code, OHADA)
- **SautiLM-Jua-Fin** : Cooperative finance, SME accounting & AVEC microfinance
- **SautiLM-Jua-Civ** : Civil status administration & peaceful community mediation
- **SautiLM-Jua-Vision** : Offline OCR chalkboard text solving (pixel vision reserved for **SautiLM-Nuru**)
- **SautiLM-Jua-Voice** : Tonal phonetics & voice assistant companion

---

## Technical Specifications

| Specification | Value |
| :--- | :--- |
| **Model Identifier** | `sautilm/{cfg['hf_repo']}` |
| **Mobile Bundle** | `{cfg['litertlm_filename']}` ({size_mb} MB) |
| **Quantization / Export** | `litert-torch` (`{quant_recipe}`, TFLite FlatBuffer Prefill/Decode + HF Tokenizer) |
| **Parent Architecture** | Google Gemma 3 (`Gemma3ForCausalLM`, 18 layers, `hidden_size=640`, `cache_length=2048`) |
| **Training Corpus** | {corpus_total:,} clean instructions |
| **Training Steps / Final Loss** | {steps} steps (LoRA `r={lora_r}, alpha={lora_alpha}`, response-only loss masking) / Final Loss: `{final_loss}` |
| **Runtime Verification** | {verified_badge} |
| **Target Mobile Runtime** | Flutter (`flutter_edge_ai` + `flutter_edge_ai_litertlm`) |
| **Minimum Hardware** | Android 8.0+ (API 26+), 2 GB RAM, ARM Cortex-A53/A55 CPU or Mobile GPU |

---

## Research & Founding Team (SautiLM / LosingTek)

- **Georges Byona** - Co-Founder, Software Engineer & Edge-AI Researcher (Catholic University of Eastern Africa / LosingTek)
- **Abraham Imani** - Co-Founder, MSc Student & AI/ML Lead (Université Catholique de Bukavu)
- **Jacques Uwonda** - Co-Founder, MSc Student & Software Engineer (Université Shalom de Bunia)

Website: [https://sautilm.com](https://sautilm.com)
Engineered & Supported by [LosingTek](https://losingtek.com)
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
        if not (metrics and metrics.get("litertlm_verified")):
            raise RuntimeError(
                "Refusing to upload unverified .litertlm artifacts to Hugging Face! "
                "Ensure training_metrics.json confirms litertlm_verified=True."
            )
        litertlm_file = out_dir / cfg["litertlm_filename"]
        if not litertlm_file.exists() and cfg["target"] == "jua":
            litertlm_file = out_dir / "sautilm-v1.litertlm"
        if not litertlm_file.exists():
            raise FileNotFoundError(f"No verified .litertlm bundle found in {out_dir}")
        print(f"[SautiLM-Forge] Uploading verified {litertlm_file.name} to {repo_id}...")
        hf_api.upload_file(
            path_or_fileobj=str(litertlm_file),
            path_in_repo=cfg["litertlm_filename"],
            repo_id=repo_id,
            repo_type="model",
        )
        if cfg["target"] == "jua":
            hf_api.upload_file(
                path_or_fileobj=str(litertlm_file),
                path_in_repo="sautilm-v1.litertlm",
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
    """Checks the status of the Kaggle GPU kernel and downloads outputs if complete."""
    cfg = _target_config(target)
    _load_local_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    status_resp = api.kernels_status(kernel_ref)
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
        for old_name in (cfg["litertlm_filename"], "sautilm-v1.litertlm", "training_metrics.json"):
            old_path = out_dir / old_name
            if old_path.exists():
                old_path.unlink()
        escaped_litert = cfg["litertlm_filename"].replace(".", r"\.")
        escaped_adapter = cfg["adapter_dirname"]
        api.kernels_output(
            kernel_ref,
            path=str(out_dir),
            file_pattern=rf"^({escaped_litert}|sautilm-jua\.litertlm|sautilm-v1\.litertlm|training_metrics\.json|{escaped_adapter}/.*|sautilm_v1_lora_adapter/.*)$",
            quiet=False,
        )
        result["downloaded_to"] = str(out_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate modular SautiLM Kaggle training & Hugging Face uploads.")
    parser.add_argument("action", nargs="?", default="launch", choices=("launch", "status", "upload"))
    parser.add_argument("--target", default="jua", help="Target build ('jua' or specialty e.g. 'stem').")
    cli_args = parser.parse_args()

    cfg = _target_config(cli_args.target)
    if cli_args.action == "launch":
        ds_ref = publish_dataset_to_kaggle(skip_if_exists=False, target=cfg["target"])
        print("Waiting 15s for Kaggle dataset indexing...")
        time.sleep(15)
        k_ref = push_gpu_kernel_to_kaggle(ds_ref, target=cfg["target"])
        print(json.dumps({"target": cfg["target"], "dataset": ds_ref, "kernel": k_ref}, indent=2))
    elif cli_args.action == "status":
        auth = _load_local_auth()
        k_ref = f"{auth['kaggle_user']}/{cfg['kernel_slug']}"
        info = check_or_download_kaggle_kernel(k_ref, target=cfg["target"])
        print(json.dumps(info, indent=2))
    elif cli_args.action == "upload":
        hf_url = init_and_upload_hf_repo(include_outputs=True, target=cfg["target"])
        print(json.dumps({"target": cfg["target"], "hf_url": hf_url}, indent=2))
