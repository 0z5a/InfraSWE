from __future__ import annotations

from pathlib import Path

from infraswe.io import sha256_tree


def test_tree_digest_is_order_stable_and_ignores_python_cache(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("two", encoding="utf-8")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    before = sha256_tree(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"cache")
    assert sha256_tree(tmp_path) == before
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert sha256_tree(tmp_path) != before
