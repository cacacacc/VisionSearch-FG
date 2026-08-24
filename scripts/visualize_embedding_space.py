from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create t-SNE/UMAP representation visualizations for saved embeddings."
    )
    parser.add_argument(
        "--variant",
        action="append",
        nargs=3,
        metavar=("NAME", "EMBEDDINGS", "RECORDS"),
        required=True,
        help="Variant name plus embeddings.npy and records.csv paths. Repeat for comparisons.",
    )
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["tsne", "umap", "both"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability/embedding_space"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [load_variant(name, Path(embeddings), Path(records)) for name, embeddings, records in args.variant]
    selected_image_ids = select_shared_image_ids(
        variants=variants,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    selected_variants = [filter_variant(variant, selected_image_ids) for variant in variants]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_records(selected_variants[0], args.output_dir / "selected_records.csv")

    summary = {
        "num_variants": len(selected_variants),
        "variants": [variant["name"] for variant in selected_variants],
        "num_selected_samples": len(selected_image_ids),
        "num_classes": len({record["label"] for record in selected_variants[0]["records"]}),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "method": args.method,
        "perplexity": args.perplexity,
        "note": "t-SNE/UMAP is qualitative evidence; quantitative conclusions should rely on Recall@K and mAP.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.method in {"tsne", "both"}:
        run_tsne_comparison(
            variants=selected_variants,
            perplexity=args.perplexity,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    if args.method in {"umap", "both"}:
        run_umap_comparison(
            variants=selected_variants,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def load_variant(name: str, embeddings_path: Path, records_path: Path) -> dict[str, Any]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    records = read_records_csv(records_path)
    if embeddings.shape[0] != len(records):
        raise ValueError(f"embeddings and records length mismatch for {name}")
    return {
        "name": name,
        "embeddings_path": str(embeddings_path),
        "records_path": str(records_path),
        "embeddings": embeddings,
        "records": records,
    }


def select_shared_image_ids(
    variants: list[dict[str, Any]],
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> list[int]:
    if num_classes < 1 or samples_per_class < 1:
        raise ValueError("--num-classes and --samples-per-class must be positive")

    shared_ids = set(record["image_id"] for record in variants[0]["records"])
    for variant in variants[1:]:
        shared_ids &= set(record["image_id"] for record in variant["records"])

    records = [record for record in variants[0]["records"] if record["image_id"] in shared_ids]
    records_by_label: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        records_by_label.setdefault(record["label"], []).append(record)

    eligible_labels = [
        label
        for label, label_records in records_by_label.items()
        if len(label_records) >= min(samples_per_class, len(label_records))
    ]
    if len(eligible_labels) < num_classes:
        raise ValueError(
            f"Only {len(eligible_labels)} shared classes are available, cannot sample {num_classes}"
        )

    rng = np.random.default_rng(seed)
    selected_labels = sorted(rng.choice(sorted(eligible_labels), size=num_classes, replace=False))
    selected_ids: list[int] = []
    for label in selected_labels:
        label_records = sorted(records_by_label[label], key=lambda record: record["image_id"])
        sample_count = min(samples_per_class, len(label_records))
        sampled_indices = rng.choice(len(label_records), size=sample_count, replace=False)
        for index in sorted(sampled_indices):
            selected_ids.append(label_records[int(index)]["image_id"])

    return sorted(selected_ids)


def filter_variant(variant: dict[str, Any], selected_image_ids: list[int]) -> dict[str, Any]:
    index_by_image_id = {
        record["image_id"]: index
        for index, record in enumerate(variant["records"])
    }
    indices = [index_by_image_id[image_id] for image_id in selected_image_ids]
    return {
        **variant,
        "embeddings": variant["embeddings"][indices],
        "records": [variant["records"][index] for index in indices],
    }


def run_tsne_comparison(
    variants: list[dict[str, Any]],
    perplexity: float,
    seed: int,
    output_dir: Path,
) -> None:
    for variant in variants:
        coordinates = compute_tsne(variant["embeddings"], perplexity=perplexity, seed=seed)
        save_coordinates(variant, coordinates, output_dir / f"{slugify(variant['name'])}_tsne.csv")
        plot_embedding(
            variant=variant,
            coordinates=coordinates,
            title=f"{variant['name']} t-SNE",
            output_path=output_dir / f"{slugify(variant['name'])}_tsne.png",
        )


def run_umap_comparison(
    variants: list[dict[str, Any]],
    seed: int,
    output_dir: Path,
) -> None:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP requires umap-learn. Install it or rerun with --method tsne."
        ) from exc

    for variant in variants:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        coordinates = reducer.fit_transform(l2_normalize(variant["embeddings"]))
        save_coordinates(variant, coordinates, output_dir / f"{slugify(variant['name'])}_umap.csv")
        plot_embedding(
            variant=variant,
            coordinates=coordinates,
            title=f"{variant['name']} UMAP",
            output_path=output_dir / f"{slugify(variant['name'])}_umap.png",
        )


def compute_tsne(embeddings: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    normalized_embeddings = l2_normalize(embeddings)
    pca_dim = min(50, normalized_embeddings.shape[1], normalized_embeddings.shape[0] - 1)
    if pca_dim >= 2:
        normalized_embeddings = PCA(n_components=pca_dim, random_state=seed).fit_transform(
            normalized_embeddings
        )
    effective_perplexity = min(perplexity, max(2.0, (normalized_embeddings.shape[0] - 1) / 3))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(normalized_embeddings)


def plot_embedding(
    variant: dict[str, Any],
    coordinates: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    labels = np.array([record["label"] for record in variant["records"]])
    class_names = [record["class_name"] for record in variant["records"]]
    unique_labels = sorted(set(labels.tolist()))
    color_map = plt.get_cmap("tab20", len(unique_labels))
    label_to_color_index = {label: index for index, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    for label in unique_labels:
        mask = labels == label
        display_name = next(
            class_name for class_name, item_label in zip(class_names, labels) if item_label == label
        )
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=28,
            color=color_map(label_to_color_index[label]),
            label=display_name,
            alpha=0.86,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_coordinates(variant: dict[str, Any], coordinates: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_id", "label", "class_name", "x", "y", "path"],
        )
        writer.writeheader()
        for record, coordinate in zip(variant["records"], coordinates):
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "label": record["label"],
                    "class_name": record["class_name"],
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                    "path": record["path"],
                }
            )


def write_selected_records(variant: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_id", "label", "class_name", "path"],
        )
        writer.writeheader()
        writer.writerows(variant["records"])


def read_records_csv(records_path: Path) -> list[dict[str, Any]]:
    with records_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {
                "image_id": int(row["image_id"]),
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "path": row["path"],
            }
            for row in reader
        ]


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


if __name__ == "__main__":
    main()
