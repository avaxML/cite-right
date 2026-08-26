"""Tests for scripts/publish_openwiki_to_docs.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_openwiki_to_docs.py"


def _load_publisher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("publish_openwiki_to_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def publish_mod() -> ModuleType:
    return _load_publisher()


def test_strips_front_matter_and_skips_listings(
    tmp_path: Path, publish_mod: ModuleType
) -> None:
    openwiki = tmp_path / "openwiki"
    docs = tmp_path / "docs"
    (openwiki / "concepts").mkdir(parents=True)
    (docs / "api").mkdir(parents=True)
    (docs / "assets").mkdir(parents=True)
    (docs / "stylesheets").mkdir(parents=True)
    (docs / "api" / "core-functions.md").write_text("KEEP API\n", encoding="utf-8")
    (docs / "assets" / "logo.svg").write_text("<svg />\n", encoding="utf-8")
    (docs / "stylesheets" / "extra.css").write_text("body{}\n", encoding="utf-8")
    (docs / "index.md").write_text("# Existing Home\n", encoding="utf-8")

    (openwiki / "index.md").write_text(
        '---\nokf_version: "0.2"\n---\n\n# Files\n\n- [x](quickstart.md)\n\n'
        "# Directories\n\n- [concepts](concepts/)\n",
        encoding="utf-8",
    )
    (openwiki / "INSTRUCTIONS.md").write_text("# brief\n", encoding="utf-8")
    (openwiki / ".last-update.json").write_text("{}\n", encoding="utf-8")
    (openwiki / "testing").mkdir()
    (openwiki / "testing" / "pytest-markers.md").write_text(
        "# Markers\n", encoding="utf-8"
    )
    (openwiki / "concepts" / "how-it-works.md").write_text(
        "---\ntype: guide\ntitle: How It Works\n---\n\n"
        "# How It Works\n\n"
        "See [alignment](/openwiki/concepts/citation-alignment.md#offsets) "
        "and [home](openwiki/index.md).\n"
        '<a href="/openwiki/getting-started/quickstart.md">Quickstart</a>\n',
        encoding="utf-8",
    )
    (openwiki / "concepts" / "citation-alignment.md").write_text(
        "# Citation Alignment\n",
        encoding="utf-8",
    )

    written = publish_mod.publish(openwiki, docs)

    assert "concepts/how-it-works.md" in written
    assert "index.md" not in written
    assert (docs / "index.md").read_text(encoding="utf-8") == "# Existing Home\n"
    assert not (docs / "testing").exists()
    assert not (docs / "INSTRUCTIONS.md").exists()
    assert (docs / "api" / "core-functions.md").read_text(
        encoding="utf-8"
    ) == "KEEP API\n"
    assert (docs / "assets" / "logo.svg").read_text(encoding="utf-8") == "<svg />\n"

    body = (docs / "concepts" / "how-it-works.md").read_text(encoding="utf-8")
    assert not body.startswith("---")
    assert "okf_version" not in body
    assert "[alignment](citation-alignment.md#offsets)" in body
    assert "[home](../index.md)" in body
    assert 'href="../getting-started/quickstart.md"' in body
    assert "/openwiki/" not in body
    assert "openwiki/" not in body


def test_public_paths_match_mkdocs_nav(publish_mod: ModuleType) -> None:
    mkdocs = Path(__file__).resolve().parents[1] / "mkdocs.yml"
    text = mkdocs.read_text(encoding="utf-8")
    for rel in publish_mod.PUBLIC_PATHS:
        assert f"{rel}" in text
    assert "api/core-functions.md" not in publish_mod.PUBLIC_PATHS
