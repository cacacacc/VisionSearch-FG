from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval metrics from different backbone summary files."
    )
    parser.add_argument("--resnet-summary", type=Path, required=True)
    parser.add_argument("--swin-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/backbone_retrieval_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        build_row(name="resnet18", summary_path=args.resnet_summary),
        build_row(name="swin_tiny", summary_path=args.swin_summary),
    ]
    summary = {
        "research_question": "Does better classification imply better retrieval embeddings?",
        "controlled_variables": [
            "validation split",
            "query/gallery protocol",
            "exclude query itself",
            "L2-normalized cosine similarity",
            "Recall@1/5/10 and mAP",
        ],
        "results": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def build_row(name: str, summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "model": name,
        "backbone": summary.get("backbone", name),
        "summary_path": str(summary_path),
        "checkpoint": summary.get("checkpoint"),
        "config": summary.get("config"),
        "metric": summary.get("metric", "cosine"),
        "num_samples": summary["num_samples"],
        "embedding_dim": summary["embedding_dim"],
        "recall@1": summary["recall@1"],
        "recall@5": summary["recall@5"],
        "recall@10": summary["recall@10"],
        "mAP": summary["mAP"],
    }


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model",
        "backbone",
        "metric",
        "num_samples",
        "embedding_dim",
        "recall@1",
        "recall@5",
        "recall@10",
        "mAP",
        "checkpoint",
        "config",
        "summary_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
