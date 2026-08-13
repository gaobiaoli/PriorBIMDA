from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pipelines import run_region_cv

PROJECT = Path(__file__).resolve().parents[1]


def test_protocol_registers_best_checkpoint_selection() -> None:
    context = run_region_cv.load_protocol_context(
        PROJECT,
        Path("configs/slabim_region_cv.yaml"),
        allow_missing_manifest=True,
    )

    assert context.checkpoint_selection == {"pretrain": "best", "final": "best"}
    assert context.normalized["checkpoint_selection"] == {
        "pretrain": "best",
        "final": "best",
    }

    plan = run_region_cv.build_plan(context)
    assert plan["checkpoint_selection"] == {"pretrain": "best", "final": "best"}
    first_run = plan["folds"][0]["runs"][0]
    assert first_run["checkpoints"]["pretrain_selected"].endswith("/pretrain/best.pt")
    assert first_run["checkpoints"]["final_selected"].endswith("/best.pt")


def test_complete_best_checkpoint_does_not_require_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_state.json").write_text("{}", encoding="utf-8")
    (output_dir / "best.pt").write_bytes(b"best")
    validated_paths: list[Path] = []

    monkeypatch.setattr(
        run_region_cv,
        "validate_run_state",
        lambda *_args, **_kwargs: {"status": "complete"},
    )

    def fake_validate_checkpoint(
        path: Path,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], str]:
        validated_paths.append(path)
        return {}, "best-sha256"

    monkeypatch.setattr(
        run_region_cv,
        "load_and_validate_checkpoint",
        fake_validate_checkpoint,
    )

    decision = run_region_cv.decide_training(
        output_dir,
        {},
        "source-sha256",
        expected_initialized_from_sha256=None,
        checkpoint_selection="best",
    )

    assert decision.action == "skip"
    assert decision.selected_checkpoint == output_dir / "best.pt"
    assert decision.selected_sha256 == "best-sha256"
    assert validated_paths == [output_dir / "best.pt"]

    with pytest.raises(RuntimeError, match="accepted.pt checkpoint is missing"):
        run_region_cv.decide_training(
            output_dir,
            {},
            "source-sha256",
            expected_initialized_from_sha256=None,
            checkpoint_selection="accepted",
        )
