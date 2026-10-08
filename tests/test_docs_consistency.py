"""Finding 5 (final closure audit, v0.4.1 -> v0.4.2): docs/DISTRIBUTED.md must describe the
ACTUAL shipped release, not a previous one -- the v0.4.1 handoff claimed stale docs were
fixed while the shipped file still opened with "Step 4 (v0.4.0, complete)" and described the
v0.4.0-era GPU guard. A lightweight, repo-local consistency check catches this class of
drift going forward; it is skipped (not failed) when the docs directory is not present, e.g.
when tests run against an installed wheel/sdist that does not package docs/.

Section 10 (local checkpoint replay-history closure pass, v0.4.3 -> v0.4.4): the SAME class
of drift was found in README.md/ROADMAP.md -- both still advertised "v0.4.1" while
docs/DISTRIBUTED.md was already current. README/ROADMAP were rewritten to describe the Step 4
contract WITHOUT a pinned patch version (so an ordinary patch release that does not change the
API contract needs no doc edit at all) -- the check below simply asserts that if either file
DOES ever pin a `v0.4.N` string again, it must match the version actually shipping, not fall
behind it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import featm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_PATH = _REPO_ROOT / "docs" / "DISTRIBUTED.md"
_README_PATH = _REPO_ROOT / "README.md"
_ROADMAP_PATH = _REPO_ROOT / "ROADMAP.md"

_STALE_PHRASES = (
    "foundation",
    "preview",
    "pending v0.4.0",
    "not yet implemented",
    "does not implement distributed checkpoint",
)

_VERSION_PATTERN = re.compile(r"v0\.4\.\d+")


def _load_text(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} not present at {path} (likely running against an installed package)")
    return path.read_text(encoding="utf-8")


def _load_docs_text() -> str:
    return _load_text(_DOCS_PATH)


def test_distributed_docs_reference_the_current_version():
    text = _load_docs_text()
    assert f"v{featm.__version__}" in text, (
        f"docs/DISTRIBUTED.md does not mention the current version v{featm.__version__} -- "
        "it may describe a previous release's contract."
    )


def test_distributed_docs_do_not_contain_stale_completion_language():
    text = _load_docs_text().lower()
    hits = [phrase for phrase in _STALE_PHRASES if phrase in text]
    assert not hits, f"docs/DISTRIBUTED.md contains stale pre-release language: {hits}"


def test_readme_and_roadmap_do_not_pin_a_stale_version():
    """README.md/ROADMAP.md may describe Step 4 without a pinned patch version at all (the
    preferred, edit-free-across-patch-releases wording) -- but if either one DOES pin a
    `v0.4.N` string, every such occurrence must match the version actually shipping."""
    current = f"v{featm.__version__}"
    for path in (_README_PATH, _ROADMAP_PATH):
        text = _load_text(path)
        found = set(_VERSION_PATTERN.findall(text))
        stale = found - {current}
        assert not stale, (
            f"{path.name} pins version string(s) {sorted(stale)} that do not match the currently "
            f"shipping {current!r} -- update the reference, or remove the pin entirely so an "
            "ordinary patch release (no API contract change) needs no doc edit."
        )
