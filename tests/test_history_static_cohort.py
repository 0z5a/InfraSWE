from infraswe.history.static_cohort import assess_static_change, is_test_path


def _file(path: str, patch: str = "+value = 1\n") -> dict[str, object]:
    return {"filename": path, "status": "modified", "patch": patch}


def test_changed_test_yields_high_polarized_score() -> None:
    files = [_file("pkg/op.py"), _file("tests/test_op.py")]
    result = assess_static_change(
        expected_paths=["pkg/op.py", "tests/test_op.py"],
        changed_files=files,
        head_sources={"pkg/op.py": b"value = 1\n"},
    )
    assert result.decision == "accept_with_scope"
    assert result.score_100 == 94


def test_source_only_pass_stays_above_merge_floor() -> None:
    result = assess_static_change(
        expected_paths=["pkg/op.py"],
        changed_files=[_file("pkg/op.py")],
        head_sources={"pkg/op.py": b"value = 1\n"},
    )
    assert result.decision == "accept_with_scope"
    assert result.score_100 == 88


def test_hard_static_defect_is_polarized_low() -> None:
    patch = "+try:\n+    risky()\n+except Exception:\n+    pass\n"
    result = assess_static_change(
        expected_paths=["pkg/op.py"],
        changed_files=[_file("pkg/op.py", patch)],
        head_sources={"pkg/op.py": b"try:\n    risky()\nexcept Exception:\n    pass\n"},
    )
    assert result.decision == "reject"
    assert result.score_100 == 35
    assert "R8_SILENT_EXCEPTION_WITHOUT_TEST" in result.rationale_codes


def test_path_mismatch_is_unresolved_not_candidate_failure() -> None:
    result = assess_static_change(
        expected_paths=["pkg/op.py"],
        changed_files=[_file("pkg/other.py")],
        head_sources={},
    )
    assert result.decision == "unresolved"
    assert result.score_100 is None


def test_test_path_recognition() -> None:
    assert is_test_path("tests/unit/test_op.py")
    assert is_test_path("src/op_test.cu")
    assert not is_test_path("src/op.py")
