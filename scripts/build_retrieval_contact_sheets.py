from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont

from visionsearch_fg.data import crop_image_to_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build query plus top-10 contact sheets for retrieval qualitative analysis."
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--cub-root", type=Path, default=Path("data/raw/CUB_200_2011"))
    parser.add_argument("--crop-mode", choices=["none", "bbox"], default="none")
    parser.add_argument("--bbox-margin", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-size", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    bboxes = read_bboxes(args.cub_root / "bounding_boxes.txt") if args.crop_mode == "bbox" else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(size=14)

    for group, group_cases in cases.items():
        for case in group_cases:
            query = case["query"]
            cells = [
                {
                    "rank": "Q",
                    "image_id": query["image_id"],
                    "class_name": query["class_name"],
                    "path": query["path"],
                    "same_class": True,
                }
            ]
            cells.extend(case["neighbors"])
            sheet = build_contact_sheet(
                cells=cells,
                bboxes=bboxes,
                crop_mode=args.crop_mode,
                bbox_margin=args.bbox_margin,
                cell_size=args.cell_size,
                font=font,
            )
            output_path = (
                args.output_dir
                / f"{group}_q{case['query_index']:04d}_id{query['image_id']}.jpg"
            )
            sheet.save(output_path, quality=95)

    print(f"output_dir: {args.output_dir}")


def build_contact_sheet(
    cells: list[dict],
    bboxes: dict[int, tuple[float, float, float, float]],
    crop_mode: str,
    bbox_margin: float,
    cell_size: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    cell_width = cell_size + 20
    sheet_width = cell_width * len(cells)
    sheet_height = cell_size + 64
    sheet = Image.new("RGB", (sheet_width, sheet_height), color="white")
    draw = ImageDraw.Draw(sheet)

    for index, cell in enumerate(cells):
        x_coord = index * cell_width + 10
        y_coord = 8
        image = load_display_image(
            path=Path(cell["path"]),
            image_id=int(cell["image_id"]),
            bboxes=bboxes,
            crop_mode=crop_mode,
            bbox_margin=bbox_margin,
            size=cell_size,
        )
        sheet.paste(image, (x_coord, y_coord))
        outline = outline_color(cell)
        draw.rectangle(
            [x_coord, y_coord, x_coord + cell_size - 1, y_coord + cell_size - 1],
            outline=outline,
            width=4,
        )
        rank = cell.get("rank", "Q")
        draw.text(
            (x_coord, y_coord + cell_size + 4),
            f"{rank} id={cell['image_id']}",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (x_coord, y_coord + cell_size + 22),
            str(cell["class_name"])[:24],
            fill=(0, 0, 0),
            font=font,
        )
    return sheet


def load_display_image(
    path: Path,
    image_id: int,
    bboxes: dict[int, tuple[float, float, float, float]],
    crop_mode: str,
    bbox_margin: float,
    size: int,
) -> Image.Image:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")
        if crop_mode == "bbox":
            if image_id not in bboxes:
                raise ValueError(f"Missing bounding box for image id {image_id}")
            image = crop_image_to_bbox(image, bboxes[image_id], margin=bbox_margin)
        return image.resize((size, size))


def outline_color(cell: dict) -> tuple[int, int, int]:
    if cell.get("rank") == "Q":
        return (0, 0, 0)
    return (0, 140, 0) if bool(cell["same_class"]) else (180, 0, 0)


def read_bboxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_id, x_coord, y_coord, width, height = line.split(maxsplit=4)
        rows[int(image_id)] = (float(x_coord), float(y_coord), float(width), float(height))
    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
