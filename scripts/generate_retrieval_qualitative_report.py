from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from visionsearch_fg.retrieval import cosine_similarity_matrix, rank_gallery_for_queries

CASE_GROUPS = {
    "top1_correct": "Top-1 正确",
    "top1_wrong_top5_hit": "Top-1 错误，但 Top-5 有同类",
    "top5_wrong_top10_hit": "Top-5 错误，但 Top-10 有同类",
    "top10_failure": "Top-10 完全失败",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML qualitative retrieval report with query and top-k results."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cases-per-group", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/retrieval_qualitative"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases_per_group < 1:
        raise ValueError("--cases-per-group must be greater than or equal to 1")

    embeddings = np.load(args.embeddings)
    records = read_records_csv(args.records)
    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")

    similarity = cosine_similarity_matrix(embeddings)
    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)
    cases = build_cases(
        records=records,
        similarity=similarity,
        ranked_indices=ranked_indices,
        cases_per_group=args.cases_per_group,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "retrieval_qualitative_report.html"
    json_path = args.output_dir / "retrieval_qualitative_cases.json"
    csv_path = args.output_dir / "retrieval_qualitative_cases.csv"

    report_path.write_text(
        render_html_report(
            cases=cases,
            output_dir=args.output_dir,
            embeddings_path=args.embeddings,
            records_path=args.records,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    write_cases_csv(cases, csv_path)

    summary = {
        "report": str(report_path),
        "cases_json": str(json_path),
        "cases_csv": str(csv_path),
        "num_cases": sum(len(group_cases) for group_cases in cases.values()),
        "case_groups": {group: len(group_cases) for group, group_cases in cases.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_cases(
    records: list[dict[str, Any]],
    similarity: np.ndarray,
    ranked_indices: np.ndarray,
    cases_per_group: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped_candidates = {group: [] for group in CASE_GROUPS}

    for query_index, query_record in enumerate(records):
        top10 = ranked_indices[query_index, :10]
        same_flags = [records[int(index)]["label"] == query_record["label"] for index in top10]
        same_count_top5 = int(sum(same_flags[:5]))
        same_count_top10 = int(sum(same_flags))
        first_correct_rank = next((rank + 1 for rank, flag in enumerate(same_flags) if flag), None)

        group = classify_case(same_flags)
        candidate = {
            "group": group,
            "query_index": query_index,
            "query": query_record,
            "top1_same_class": bool(same_flags[0]),
            "same_count_top5": same_count_top5,
            "same_count_top10": same_count_top10,
            "first_correct_rank": first_correct_rank,
            "neighbors": [
                build_neighbor_record(
                    rank=rank + 1,
                    query_record=query_record,
                    neighbor_record=records[int(neighbor_index)],
                    similarity=float(similarity[query_index, neighbor_index]),
                )
                for rank, neighbor_index in enumerate(top10)
            ],
        }
        grouped_candidates[group].append(candidate)

    return {
        group: select_diverse_cases(candidates, cases_per_group)
        for group, candidates in grouped_candidates.items()
    }


def classify_case(same_flags: list[bool]) -> str:
    if same_flags[0]:
        return "top1_correct"
    if any(same_flags[:5]):
        return "top1_wrong_top5_hit"
    if any(same_flags[:10]):
        return "top5_wrong_top10_hit"
    return "top10_failure"


def build_neighbor_record(
    rank: int,
    query_record: dict[str, Any],
    neighbor_record: dict[str, Any],
    similarity: float,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "image_id": neighbor_record["image_id"],
        "label": neighbor_record["label"],
        "class_name": neighbor_record["class_name"],
        "path": neighbor_record["path"],
        "similarity": similarity,
        "same_class": neighbor_record["label"] == query_record["label"],
    }


def select_diverse_cases(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    used_labels = set()
    for candidate in candidates:
        label = candidate["query"]["label"]
        if label in used_labels:
            continue
        selected.append(candidate)
        used_labels.add(label)
        if len(selected) == limit:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == limit:
            return selected
    return selected


def render_html_report(
    cases: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    embeddings_path: Path,
    records_path: Path,
) -> str:
    sections = "\n".join(
        render_case_group(group=group, group_cases=group_cases, output_dir=output_dir)
        for group, group_cases in cases.items()
    )
    intro = (
        "本报告用于补充数值指标。每个案例展示 Query 与 Top-10 检索结果，"
        "绿色边框表示同类别，红色边框表示类别不同。分析时重点观察："
        "同类别是否集中在前排、错误结果是否外观相似、背景是否相似、"
        "姿态是否相似、局部纹理或颜色是否相似。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retrieval Qualitative Analysis</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #172026;
      background: #f6f7f9;
    }}
    header {{
      padding: 28px 36px;
      background: #172026;
      color: white;
    }}
    main {{
      padding: 24px 36px 48px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.6;
      max-width: 1100px;
    }}
    .meta {{
      color: #d7dde3;
      font-size: 13px;
      line-height: 1.6;
    }}
    .group {{
      margin: 28px 0 40px;
    }}
    .case {{
      margin: 18px 0 28px;
      padding: 18px;
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 8px;
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .stats {{
      color: #4b5965;
      font-size: 13px;
    }}
    .query-row {{
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .image-card {{
      border: 2px solid #c8d2dc;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .image-card.same {{
      border-color: #248a4b;
    }}
    .image-card.diff {{
      border-color: #b93b3b;
    }}
    .image-card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #eef1f4;
    }}
    .caption {{
      padding: 8px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .rank {{
      font-weight: 700;
    }}
    .neighbors {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 12px;
    }}
    .analysis {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
    }}
    .analysis div {{
      min-height: 52px;
      padding: 10px;
      border: 1px dashed #aeb8c2;
      border-radius: 6px;
      font-size: 12px;
      color: #44515d;
      background: #fbfcfd;
    }}
    .analysis strong {{
      display: block;
      margin-bottom: 4px;
      color: #172026;
    }}
    @media (max-width: 900px) {{
      main, header {{
        padding-left: 18px;
        padding-right: 18px;
      }}
      .query-row {{
        grid-template-columns: 1fr;
      }}
      .neighbors {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
      .analysis {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Retrieval Qualitative Analysis</h1>
    <div class="meta">
      Embeddings: {escape_text(str(embeddings_path))}<br>
      Records: {escape_text(str(records_path))}<br>
      Protocol: validation query/gallery, exclude query itself, L2-normalized cosine ranking.
    </div>
  </header>
  <main>
    <p>{escape_text(intro)}</p>
    {sections}
  </main>
</body>
</html>
"""


def render_case_group(group: str, group_cases: list[dict[str, Any]], output_dir: Path) -> str:
    cases_html = "\n".join(render_case(case=case, output_dir=output_dir) for case in group_cases)
    return f"""<section class="group">
  <h2>{escape_text(CASE_GROUPS[group])}</h2>
  {cases_html}
</section>"""


def render_case(case: dict[str, Any], output_dir: Path) -> str:
    query = case["query"]
    neighbors_html = "\n".join(
        render_image_card(record=neighbor, output_dir=output_dir, is_query=False)
        for neighbor in case["neighbors"]
    )
    stats = (
        f"Top-1 same: {case['top1_same_class']} · "
        f"Top-5 same count: {case['same_count_top5']} · "
        f"Top-10 same count: {case['same_count_top10']} · "
        f"First correct rank: {case['first_correct_rank']}"
    )
    analysis_html = "\n".join(
        [
            render_analysis_box("同类别结果", "记录同类样本出现位置，判断是否形成稳定类簇。"),
            render_analysis_box(
                "外观相似但类别不同",
                "观察错误 top-k 是否与 query 在整体颜色、体型或纹理上接近。",
            ),
            render_analysis_box(
                "背景相似导致错误",
                "检查水面、树枝、天空、草地等背景是否主导相似度。",
            ),
            render_analysis_box("姿态相似", "检查飞行、侧身、低头、站立等姿态是否压过类别信息。"),
            render_analysis_box(
                "局部特征相似",
                "记录头部、喙、翅膀、腹部颜色等局部线索是否被模型利用或混淆。",
            ),
        ]
    )
    return f"""<article class="case">
  <div class="case-head">
    <h3>Query #{case["query_index"]}：{escape_text(query["class_name"])}</h3>
    <div class="stats">
      {escape_text(stats)}
    </div>
  </div>
  <div class="query-row">
    {render_image_card(record=query, output_dir=output_dir, is_query=True)}
    <div class="neighbors">{neighbors_html}</div>
  </div>
  <div class="analysis">
    {analysis_html}
  </div>
</article>"""


def render_analysis_box(title: str, description: str) -> str:
    return f"<div><strong>{escape_text(title)}</strong>{escape_text(description)}</div>"


def render_image_card(record: dict[str, Any], output_dir: Path, is_query: bool) -> str:
    css_class = "image-card"
    if not is_query:
        css_class += " same" if record["same_class"] else " diff"
    image_src = relative_image_src(record["path"], output_dir)
    rank = "Query" if is_query else f"Top-{record['rank']}"
    score = "" if is_query else f"<br>sim={record['similarity']:.4f}"
    same = "" if is_query else f"<br>{'same class' if record['same_class'] else 'different class'}"
    return f"""<div class="{css_class}">
  <img src="{escape_attr(image_src)}" alt="{escape_attr(record["class_name"])}">
  <div class="caption">
    <span class="rank">{escape_text(rank)}</span><br>
    {escape_text(record["class_name"])}<br>
    image_id={record["image_id"]}{score}{same}
  </div>
</div>"""


def relative_image_src(image_path: str, output_dir: Path) -> str:
    absolute_image_path = Path(image_path).resolve()
    absolute_output_dir = output_dir.resolve()
    return os.path.relpath(absolute_image_path, start=absolute_output_dir).replace(os.sep, "/")


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


def write_cases_csv(cases: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    fieldnames = [
        "group",
        "query_index",
        "image_id",
        "class_name",
        "top1_same_class",
        "same_count_top5",
        "same_count_top10",
        "first_correct_rank",
    ]
    rows = []
    for group_cases in cases.values():
        for case in group_cases:
            rows.append(
                {
                    "group": case["group"],
                    "query_index": case["query_index"],
                    "image_id": case["query"]["image_id"],
                    "class_name": case["query"]["class_name"],
                    "top1_same_class": case["top1_same_class"],
                    "same_count_top5": case["same_count_top5"],
                    "same_count_top10": case["same_count_top10"],
                    "first_correct_rank": case["first_correct_rank"],
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def escape_text(value: Any) -> str:
    return html.escape(str(value), quote=False)


def escape_attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
