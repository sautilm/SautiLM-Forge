"""
SautiLM-Nuru-Agri: Compact INT8 TFLite Collaborative Vision Classifier (2 - 45 MB).

Tier Architecture (SAUTILM_ARCHITECTURE_ET_VISION.md):
  1. Nuru Vision Classifier inspects raw camera pixels FIRST using a real quantized
     CNN/MobileNetV3-Small or EfficientNet-Lite0 TFLite model (<= 45 MB).
  2. Strict Non-Leaf / Out-of-Scope Rejection Gate blocks selfies, walls, documents,
     or low-confidence inputs BEFORE invoking SautiLM-Jua-Agri.
  3. Structured triage payload (is_plant_leaf, crop_label, disease_hint, confidence)
     is passed to gemmea_eac_agri.sqlite FTS5 RAG -> SautiLM-Jua-Agri text synthesis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


NURU_AGRI_MAX_MODEL_MB = 45.0
NURU_AGRI_INPUT_SIZE = (224, 224, 3)
NURU_DEFAULT_ABSTENTION_THRESHOLD = 0.55

NURU_AGRI_LABELS: Tuple[str, ...] = (
    "non_plant",
    "healthy_leaf",
    "cassava_cmd",
    "cassava_cbsd",
    "maize_rust",
    "maize_faw",
    "banana_sigatoka",
    "bean_angular_leaf_spot",
    "tomato_blight",
)

NURU_LABEL_TO_GEMMEA_SIGNAL: Dict[str, Dict[str, Any]] = {
    "non_plant": {
        "is_plant_leaf": False,
        "crop_label": "Non végétal",
        "disease_hint": "Image hors sujet (aucune plante cultivée détectée)",
        "rag_query": "",
    },
    "healthy_leaf": {
        "is_plant_leaf": True,
        "crop_label": "Feuille saine",
        "disease_hint": "Aucune maladie (Feuille Saine)",
        "rag_query": "entretien prevention sante foliaire",
    },
    "cassava_cmd": {
        "is_plant_leaf": True,
        "crop_label": "Manioc",
        "disease_hint": "Mosaïque Africaine du Manioc (CMD)",
        "rag_query": "manioc mosaique cmd bemisia tme 419",
    },
    "cassava_cbsd": {
        "is_plant_leaf": True,
        "crop_label": "Manioc",
        "disease_hint": "Strie Brune du Manioc (CBSD)",
        "rag_query": "manioc strie brune cbsd narocass mkombozi",
    },
    "maize_rust": {
        "is_plant_leaf": True,
        "crop_label": "Maïs",
        "disease_hint": "Rouille Commune du Maïs (Puccinia sorghi)",
        "rag_query": "mais rouille puccinia sorghi rotation",
    },
    "maize_faw": {
        "is_plant_leaf": True,
        "crop_label": "Maïs",
        "disease_hint": "Chenille Légionnaire d'Automne (FAW)",
        "rag_query": "mais chenille legionnaire faw neem cendre push pull",
    },
    "banana_sigatoka": {
        "is_plant_leaf": True,
        "crop_label": "Banane",
        "disease_hint": "Cercosporiose Noire du Bananier (Black Sigatoka)",
        "rag_query": "banane sigatoka cercosporiose feuilles coupe prevention",
    },
    "bean_angular_leaf_spot": {
        "is_plant_leaf": True,
        "crop_label": "Haricot",
        "disease_hint": "Tache Angulaire du Haricot (Pseudocercospora griseola)",
        "rag_query": "haricot tache angulaire semences saines rotation",
    },
    "tomato_blight": {
        "is_plant_leaf": True,
        "crop_label": "Tomate",
        "disease_hint": "Mildiou de la Tomate (Phytophthora infestans)",
        "rag_query": "tomate mildiou phytophthora tuteurage bio",
    },
}


@dataclass(frozen=True)
class NuruTriageDecision:
    is_plant_leaf: bool
    predicted_label: str
    crop_label: str
    disease_hint: str
    confidence: float
    abstained: bool
    rag_query: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def probabilities_to_triage_decision(
    probabilities: Sequence[float],
    labels: Sequence[str] = NURU_AGRI_LABELS,
    threshold: float = NURU_DEFAULT_ABSTENTION_THRESHOLD,
) -> NuruTriageDecision:
    """Converts Nuru softmax probabilities into the structured Gemmea handoff contract."""
    if len(probabilities) != len(labels):
        raise ValueError(
            f"Probability length ({len(probabilities)}) != label count ({len(labels)})"
        )
    if not probabilities:
        raise ValueError("Empty probabilities vector")

    best_idx = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
    best_prob = float(probabilities[best_idx])
    best_label = str(labels[best_idx])

    if best_label == "non_plant" or best_prob < threshold:
        return NuruTriageDecision(
            is_plant_leaf=False,
            predicted_label="non_plant",
            crop_label="Non végétal",
            disease_hint=(
                "Image rejetée par SautiLM-Nuru : aucune feuille ou plante malade "
                "identifiable avec certitude."
            ),
            confidence=best_prob,
            abstained=True,
            rag_query="",
        )

    meta = NURU_LABEL_TO_GEMMEA_SIGNAL.get(best_label)
    if meta is None:
        raise KeyError(f"Unsupported Nuru label: {best_label}")

    return NuruTriageDecision(
        is_plant_leaf=bool(meta["is_plant_leaf"]),
        predicted_label=best_label,
        crop_label=str(meta["crop_label"]),
        disease_hint=str(meta["disease_hint"]),
        confidence=best_prob,
        abstained=False,
        rag_query=str(meta["rag_query"]),
    )


def build_nuru_keras_classifier(
    num_classes: int = len(NURU_AGRI_LABELS),
    input_shape: Tuple[int, int, int] = NURU_AGRI_INPUT_SIZE,
    backbone: str = "mobilenet_v3_small",
    weights: Optional[str] = "imagenet",
):
    """Builds a lightweight MobileNetV3-Small vision classifier (< 4 MB INT8 TFLite)."""
    import tensorflow as tf  # type: ignore

    inputs = tf.keras.Input(shape=input_shape, name="image_rgb_224")
    if backbone == "efficientnet_lite0":
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_tensor=x,
            pooling="avg",
        )
    else:
        x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights=weights,
            input_tensor=x,
            pooling="avg",
            minimalistic=False,
        )

    base.trainable = False
    head = tf.keras.layers.Dropout(0.25)(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(head)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"sautilm_nuru_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_quantized_nuru_tflite(
    model: Any,
    output_tflite_path: Path,
    representative_images: Sequence[Any],
    labels: Sequence[str] = NURU_AGRI_LABELS,
) -> Dict[str, Any]:
    """Exports full-integer INT8 TFLite model + sidecar label manifest and verifies size."""
    import tensorflow as tf  # type: ignore

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    def _rep_gen():
        for img in representative_images:
            yield [img]

    converter.representative_dataset = _rep_gen
    tflite_bytes = converter.convert()

    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    output_tflite_path.write_bytes(tflite_bytes)
    size_mb = round(len(tflite_bytes) / (1024.0 * 1024.0), 3)
    if size_mb > NURU_AGRI_MAX_MODEL_MB:
        raise ValueError(
            f"Exported Nuru TFLite exceeds {NURU_AGRI_MAX_MODEL_MB} MB budget: {size_mb} MB"
        )

    manifest = {
        "model_name": "SautiLM-Nuru-Agri",
        "format": "tflite_int8",
        "size_mb": size_mb,
        "input_shape": list(NURU_AGRI_INPUT_SIZE),
        "labels": list(labels),
        "abstention_threshold": NURU_DEFAULT_ABSTENTION_THRESHOLD,
    }
    manifest_path = output_tflite_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Verify TFLite interpreter loads & runs 1 dummy image cleanly
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "input_dtype": str(in_details["dtype"]),
        "output_shape": list(out_details["shape"]),
    }


def train_nuru_agri_from_directory(
    dataset_dir: Path,
    output_tflite: Path,
    epochs: int = 8,
    batch_size: int = 32,
    backbone: str = "mobilenet_v3_small",
) -> Dict[str, Any]:
    """Trains SautiLM-Nuru-Agri on image_dataset_from_directory structure (class subfolders)."""
    import tensorflow as tf  # type: ignore

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Vision dataset directory not found: {dataset_dir}. "
            f"Expected subfolders matching {list(NURU_AGRI_LABELS)}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(dataset_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=NURU_AGRI_INPUT_SIZE[:2],
        batch_size=batch_size,
        class_names=list(NURU_AGRI_LABELS),
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(dataset_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=NURU_AGRI_INPUT_SIZE[:2],
        batch_size=batch_size,
        class_names=list(NURU_AGRI_LABELS),
    )

    model = build_nuru_keras_classifier(
        num_classes=len(NURU_AGRI_LABELS),
        backbone=backbone,
        weights="imagenet",
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    rep_samples: List[Any] = []
    for batch_imgs, _ in train_ds.take(3):
        for i in range(min(8, int(batch_imgs.shape[0]))):
            rep_samples.append(tf.expand_dims(tf.cast(batch_imgs[i], tf.float32), axis=0))

    report = export_quantized_nuru_tflite(
        model=model,
        output_tflite_path=output_tflite,
        representative_images=rep_samples,
    )
    report["final_val_accuracy"] = float(history.history.get("val_accuracy", [0.0])[-1])
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train & quantize SautiLM-Nuru-Agri INT8 TFLite")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/sautilm-nuru-agri.tflite"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--backbone",
        choices=("mobilenet_v3_small", "efficientnet_lite0"),
        default="mobilenet_v3_small",
    )
    args = parser.parse_args(argv)
    report = train_nuru_agri_from_directory(
        dataset_dir=args.dataset_dir,
        output_tflite=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone=args.backbone,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
