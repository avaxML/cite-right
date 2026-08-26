#!/usr/bin/env python3
"""Copy public OpenWiki pages into docs/ for GitHub Pages.

Public paths match mkdocs.yml. This script strips OKF YAML front matter,
rewrites openwiki/ hrefs to MkDocs-relative paths, and leaves docs/api/,
docs/assets/, and stylesheets alone.

Agent-only trees (.claims/, testing/, INSTRUCTIONS.md, .last-update.json)
are not copied. OKF ``# Files`` directory listings are not copied.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

PUBLIC_PATHS: tuple[str, ...] = (
    "index.md",
    "getting-started/installation.md",
    "getting-started/quickstart.md",
    "concepts/how-it-works.md",
    "concepts/citation-alignment.md",
    "concepts/hallucination-detection.md",
    "concepts/fact-verification.md",
    "configuration/citation-config.md",
    "configuration/presets.md",
    "configuration/tokenizers.md",
    "configuration/segmenters.md",
    "integrations/langchain.md",
    "integrations/llamaindex.md",
    "integrations/custom-sources.md",
    "advanced/multi-span-evidence.md",
    "advanced/embedding-retrieval.md",
    "advanced/rust-acceleration.md",
    "advanced/performance-tuning.md",
)

SKIP_DIR_PREFIXES: tuple[str, ...] = (
    ".claims/",
    "testing/",
    "api/",
    "assets/",
    "stylesheets/",
)
SKIP_NAMES: frozenset[str] = frozenset(
    {
        "INSTRUCTIONS.md",
        ".last-update.json",
        "README.md",
    }
)

FRONT_MATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n?", re.DOTALL)
MD_LINK_RE = re.compile(r"(?P<pre>!?\[(?:[^\]]*)\])\((?P<url>[^)]+)\)")
HREF_RE = re.compile(
    r"""(?P<pre>\bhref\s*=\s*)(?P<q>['"])(?P<url>.*?)(?P=q)""",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^#{1,6}\s+(\S.+)$", re.MULTILINE)


def strip_front_matter(text: str) -> str:
    """Remove a leading OKF / YAML front matter block if present."""
    match = FRONT_MATTER_RE.match(text)
    if match is None:
        return text
    return text[match.end() :]


def is_okf_directory_listing(body: str) -> bool:
    """Return True for OpenWiki ``# Files`` / ``# Directories`` index pages."""
    headings = [match.group(1).strip().lower() for match in HEADING_RE.finditer(body)]
    if not headings:
        return False
    if headings[0] == "files":
        return True
    return "files" in headings and "directories" in headings


def _split_anchor(url: str) -> tuple[str, str, str]:
    """Split ``path#anchor "title"`` into path, anchor, and optional title."""
    title = ""
    rest = url.strip()
    if rest.endswith('"') and ' "' in rest:
        rest, title = rest.rsplit(' "', 1)
        title = ' "' + title
        rest = rest.rstrip()
    elif rest.endswith("'") and " '" in rest:
        rest, title = rest.rsplit(" '", 1)
        title = " '" + title
        rest = rest.rstrip()
    anchor = ""
    if "#" in rest:
        rest, fragment = rest.split("#", 1)
        anchor = "#" + fragment
    return rest, anchor, title


def openwiki_href_target(href: str) -> str | None:
    """Return the path under openwiki/ if ``href`` points into that tree."""
    path, _anchor, _title = _split_anchor(href)
    path = path.strip()
    prefixes = (
        "/openwiki/",
        "openwiki/",
        "./openwiki/",
        "../openwiki/",
    )
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    if path in {"/openwiki", "/openwiki/", "openwiki"}:
        return "index.md"
    return None


def mkdocs_relative_href(target_under_openwiki: str, dest_rel: str) -> str:
    """Rewrite an openwiki path to a path relative to the destination docs file."""
    dest_dir = posixpath.dirname(dest_rel)
    if dest_dir in {"", "."}:
        return target_under_openwiki
    return posixpath.relpath(target_under_openwiki, dest_dir)


def rewrite_openwiki_url(url: str, dest_rel: str) -> str:
    """Rewrite a single href/src if it points at openwiki/."""
    target = openwiki_href_target(url)
    if target is None:
        return url
    _path, anchor, title = _split_anchor(url)
    if target == "" or target.endswith("/"):
        target = posixpath.join(target, "index.md") if target else "index.md"
    rel = mkdocs_relative_href(target, dest_rel)
    return f"{rel}{anchor}{title}"


def rewrite_links(text: str, dest_rel: str) -> str:
    """Rewrite markdown and HTML hrefs that point at openwiki/."""

    def md_repl(match: re.Match[str]) -> str:
        return f"{match.group('pre')}({rewrite_openwiki_url(match.group('url'), dest_rel)})"

    def href_repl(match: re.Match[str]) -> str:
        rewritten = rewrite_openwiki_url(match.group("url"), dest_rel)
        return f"{match.group('pre')}{match.group('q')}{rewritten}{match.group('q')}"

    text = MD_LINK_RE.sub(md_repl, text)
    return HREF_RE.sub(href_repl, text)


def should_skip_path(rel: str) -> bool:
    """Return True for agent-only or protected docs trees."""
    if rel in SKIP_NAMES or Path(rel).name in SKIP_NAMES:
        return True
    return any(
        rel == prefix[:-1] or rel.startswith(prefix) for prefix in SKIP_DIR_PREFIXES
    )


def publish_file(source: Path, dest: Path, dest_rel: str) -> str | None:
    """Copy one public page. Return a skip reason, or None if written."""
    raw = source.read_text(encoding="utf-8")
    body = strip_front_matter(raw)
    if is_okf_directory_listing(body):
        return "okf-directory-listing"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(rewrite_links(body, dest_rel), encoding="utf-8")
    return None


def publish(openwiki_root: Path, docs_root: Path) -> list[str]:
    """Copy listed public paths. Returns relative dest paths that were written."""
    written: list[str] = []
    for rel in PUBLIC_PATHS:
        if should_skip_path(rel):
            print(f"skip (protected): {rel}", file=sys.stderr)
            continue
        source = openwiki_root / rel
        if not source.is_file():
            print(f"skip (missing): {rel}", file=sys.stderr)
            continue
        reason = publish_file(source, docs_root / rel, rel)
        if reason is not None:
            print(f"skip ({reason}): {rel}", file=sys.stderr)
            continue
        written.append(rel)
        print(f"published: {rel}")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for CI and local runs."""
    parser = argparse.ArgumentParser(
        description="Copy public OpenWiki pages into docs/ for GitHub Pages."
    )
    parser.add_argument(
        "--openwiki",
        type=Path,
        default=None,
        help="Source tree (default: <repo>/openwiki)",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Destination tree (default: <repo>/docs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    openwiki_root = (args.openwiki or repo_root / "openwiki").resolve()
    docs_root = (args.docs or repo_root / "docs").resolve()
    if not openwiki_root.is_dir():
        print(f"openwiki root does not exist: {openwiki_root}", file=sys.stderr)
        return 1
    if not docs_root.is_dir():
        print(f"docs root does not exist: {docs_root}", file=sys.stderr)
        return 1
    publish(openwiki_root, docs_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
