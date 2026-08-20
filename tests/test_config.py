from __future__ import annotations

from pathlib import Path

from visionsearch_fg.utils.config import load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  backbone: resnet18\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config["model"]["backbone"] == "resnet18"
