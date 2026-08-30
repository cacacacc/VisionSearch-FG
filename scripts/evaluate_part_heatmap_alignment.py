from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from visionsearch_fg.utils import load_yaml_config

BBox = tuple[float, float, float, float]
PartLocation = tuple[float, float, bool]

DEFAULT_PART_GROUPS = {
    "beak": ["beak"],
    "eye": ["left eye", "right eye"],
    "head": ["beak", "crown", "forehead", "left eye", "right eye", "nape", "throat"],
    "wing": ["left wing", "right wing"],
    "body": ["back", "belly", "breast", "left wing", "right wing", "tail"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether Grad-CAM/attention heatmap peaks align with CUB part locations."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--radius-pixels", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explainability/part_alignment"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radius_pixels <= 0:
        raise ValueError("--radius-pixels must be positive")

    config = load_yaml_config(args.config)
    data_config = config["data"]
    cub_root = Path(data_config["root"])
    image_size = int(data_config["image_size"])
    crop_mode = data_config.get("crop_mode", "none")
    bbox_margin = float(data_config.get("bbox_margin", 0.0))

    records = read_records_csv(args.records)
    part_names = read_part_names(cub_root / "parts" / "parts.txt")
    part_locations = read_part_locations(cub_root / "parts" / "part_locs.txt")
    bboxes = read_bboxes(cub_root / "bounding_boxes.txt") if crop_mode == "bbox" else {}

    rows = [
        evaluate_record(
            record=record,
            cub_root=cub_root,
            image_size=image_size,
            crop_mode=crop_mode,
            bbox_margin=bbox_margin,
            bboxes=bboxes,
            part_names=part_names,
            part_locations=part_locations,
            radius_pixels=args.radius_pixels,
        )
        for record in records
    ]
    summary = summarize_rows(rows)
    summary.update(
        {
            "records": str(args.records),
            "config": str(args.config),
            "cub_root": str(cub_root),
            "image_size": image_size,
            "crop_mode": crop_mode,
            "bbox_margin": bbox_margin,
            "radius_pixels": args.radius_pixels,
            "part_groups": DEFAULT_PART_GROUPS,
        }
    )

    output_dir = args.output_dir / args.records.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(rows, output_dir / "part_alignment_records.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


def evaluate_record(
    record: dict[str, str],
    cub_root: Path,
    image_size: int,
    crop_mode: str,
    bbox_margin: float,
    bboxes: dict[int, BBox],
    part_names: dict[int, str],
    part_locations: dict[int, dict[int, PartLocation]],
    radius_pixels: float,
) -> dict[str, Any]:
    image_id = int(record["image_id"])
    heatmap_path = resolve_path(record["heatmap_path"])
    image_path = resolve_image_path(record["path"], cub_root)
    heatmap = np.load(heatmap_path).astype(np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be a 2D array: {heatmap_path}")

    with Image.open(image_path) as image:
        original_size = image.size

    bbox = bboxes.get(image_id)
    points = transform_visible_parts(
        image_id=image_id,
        part_names=part_names,
        part_locations=part_locations,
        original_size=original_size,
        image_size=image_size,
        crop_mode=crop_mode,
        bbox=bbox,
        bbox_margin=bbox_margin,
    )
    peak_x, peak_y = heatmap_peak_xy(heatmap=heatmap, image_size=image_size)
    output: dict[str, Any] = {
        "image_id": image_id,
        "path": record["path"],
        "heatmap_path": record["heatmap_path"],
        "correct": record.get("correct", ""),
        "peak_x": peak_x,
        "peak_y": peak_y,
        "num_visible_parts": len(points),
    }

    for group_name, group_parts in DEFAULT_PART_GROUPS.items():
        group_points = [points[name] for name in group_parts if name in points]
        distance = nearest_distance((peak_x, peak_y), group_points)
        output[f"{group_name}_visible"] = bool(group_points)
        output[f"{group_name}_peak_distance_px"] = distance
        output[f"{group_name}_peak_hit"] = distance <= radius_pixels if math.isfinite(distance) else False
        output[f"{group_name}_heat_mass"] = heat_mass_near_points(
            heatmap=heatmap,
            points=group_points,
            image_size=image_size,
            radius_pixels=radius_pixels,
        )

    target_parts = [
        point
        for group in ("beak", "eye", "wing")
        for point in [points[name] for name in DEFAULT_PART_GROUPS[group] if name in points]
    ]
    any_distance = nearest_distance((peak_x, peak_y), target_parts)
    output["target_peak_distance_px"] = any_distance
    output["target_peak_hit"] = any_distance <= radius_pixels if math.isfinite(any_distance) else False
    return output


def transform_visible_parts(
    image_id: int,
    part_names: dict[int, str],
    part_locations: dict[int, dict[int, PartLocation]],
    original_size: tuple[int, int],
    image_size: int,
    crop_mode: str,
    bbox: BBox | None,
    bbox_margin: float,
) -> dict[str, tuple[float, float]]:
    transformed = {}
    image_width, image_height = original_size
    crop_box = compute_crop_box(original_size, bbox, bbox_margin) if crop_mode == "bbox" else None
    for part_id, (x_coord, y_coord, visible) in part_locations.get(image_id, {}).items():
        if not visible:
            continue
        part_name = part_names[part_id]
        if crop_box is None:
            x_new = x_coord * image_size / image_width
            y_new = y_coord * image_size / image_height
        else:
            x_min, y_min, x_max, y_max = crop_box
            if not (x_min <= x_coord <= x_max and y_min <= y_coord <= y_max):
                continue
            x_new = (x_coord - x_min) * image_size / (x_max - x_min)
            y_new = (y_coord - y_min) * image_size / (y_max - y_min)
        transformed[part_name] = (float(x_new), float(y_new))
    return transformed


def compute_crop_box(
    original_size: tuple[int, int],
    bbox: BBox | None,
    margin: float,
) -> tuple[float, float, float, float]:
    if bbox is None:
        raise ValueError("crop_mode=bbox requires bounding_boxes.txt entry for every image")
    x_coord, y_coord, width, height = bbox
    image_width, image_height = original_size
    x_min = max(0, int(round(x_coord - width * margin)))
    y_min = max(0, int(round(y_coord - height * margin)))
    x_max = min(image_width, int(round(x_coord + width + width * margin)))
    y_max = min(image_height, int(round(y_coord + height + height * margin)))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid crop box for bbox={bbox}, original_size={original_size}")
    return float(x_min), float(y_min), float(x_max), float(y_max)


def heatmap_peak_xy(heatmap: np.ndarray, image_size: int) -> tuple[float, float]:
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    scale_x = image_size / heatmap.shape[1]
    scale_y = image_size / heatmap.shape[0]
    return (peak_x + 0.5) * scale_x, (peak_y + 0.5) * scale_y


def nearest_distance(
    point: tuple[float, float],
    candidates: list[tuple[float, float]],
) -> float:
    if not candidates:
        return math.inf
    px, py = point
    return float(min(math.hypot(px - cx, py - cy) for cx, cy in candidates))


def heat_mass_near_points(
    heatmap: np.ndarray,
    points: list[tuple[float, float]],
    image_size: int,
    radius_pixels: float,
) -> float:
    if not points:
        return 0.0
    total = float(heatmap.sum())
    if total <= 0:
        return 0.0
    height, width = heatmap.shape
    xs = (np.arange(width, dtype=np.float32) + 0.5) * image_size / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) * image_size / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    mask = np.zeros_like(heatmap, dtype=bool)
    for point_x, point_y in points:
        mask |= (grid_x - point_x) ** 2 + (grid_y - point_y) ** 2 <= radius_pixels**2
    return float(heatmap[mask].sum() / total)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_records": len(rows)}
    for group_name in [*DEFAULT_PART_GROUPS.keys(), "target"]:
        visible_key = f"{group_name}_visible" if group_name != "target" else None
        hit_key = f"{group_name}_peak_hit"
        distance_key = f"{group_name}_peak_distance_px"
        mass_key = f"{group_name}_heat_mass"

        eligible = rows if visible_key is None else [row for row in rows if row[visible_key]]
        distances = [
            float(row[distance_key]) for row in eligible if math.isfinite(float(row[distance_key]))
        ]
        summary[f"{group_name}_num_eligible"] = len(eligible)
        summary[f"{group_name}_peak_hit_rate"] = (
            sum(bool(row[hit_key]) for row in eligible) / len(eligible) if eligible else 0.0
        )
        summary[f"{group_name}_mean_peak_distance_px"] = (
            sum(distances) / len(distances) if distances else math.inf
        )
        if mass_key in rows[0] if rows else False:
            summary[f"{group_name}_mean_heat_mass"] = (
                sum(float(row[mass_key]) for row in eligible) / len(eligible) if eligible else 0.0
            )
    return summary


def read_records_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No records found: {path}")
    if "heatmap_path" not in rows[0]:
        raise ValueError(
            "records CSV must contain heatmap_path. Regenerate Grad-CAM or Swin attention records "
            "with the updated script."
        )
    return rows


def read_part_names(path: Path) -> dict[int, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        part_id, name = line.split(maxsplit=1)
        rows[int(part_id)] = name
    return rows


def read_part_locations(path: Path) -> dict[int, dict[int, PartLocation]]:
    rows: dict[int, dict[int, PartLocation]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_id, part_id, x_coord, y_coord, visible = line.split()
        rows.setdefault(int(image_id), {})[int(part_id)] = (
            float(x_coord),
            float(y_coord),
            bool(int(visible)),
        )
    return rows


def read_bboxes(path: Path) -> dict[int, BBox]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_id, x_coord, y_coord, width, height = line.split(maxsplit=4)
        rows[int(image_id)] = (float(x_coord), float(y_coord), float(width), float(height))
    return rows


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"File does not exist: {path}")


def resolve_image_path(path: str, cub_root: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    candidate = cub_root / "images" / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image file does not exist: {path}")


def write_records_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
