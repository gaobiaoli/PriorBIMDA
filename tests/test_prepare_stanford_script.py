from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.data import prepare_stanford_area1 as prepare_script


@pytest.mark.parametrize(
    ("rooms", "max_frames_per_room", "stride"),
    [
        (["office_1"], None, 1),
        (None, 10, 1),
        (None, None, 2),
    ],
)
def test_filtered_preparation_never_publishes_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rooms: list[str] | None,
    max_frames_per_room: int | None,
    stride: int,
) -> None:
    args = argparse.Namespace(
        config="config.yaml",
        rooms=rooms,
        max_frames_per_room=max_frames_per_room,
        stride=stride,
        overwrite=False,
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(prepare_script, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_script, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(
        prepare_script,
        "prepare_stanford_area1",
        lambda *positional, **keywords: ([{"id": "sample"}], {"receipt": True}),
    )
    monkeypatch.setattr(
        prepare_script,
        "write_stanford_manifest",
        lambda *positional: calls.append(positional),
    )

    prepare_script.main()

    assert calls == []
    assert "Canonical manifest.jsonl and metadata.json were left unchanged" in (
        capsys.readouterr().out
    )


def test_full_preparation_publishes_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        config="config.yaml",
        rooms=None,
        max_frames_per_room=None,
        stride=1,
        overwrite=False,
    )
    monkeypatch.setattr(prepare_script, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_script, "load_config", lambda path: {"path": path})
    monkeypatch.setattr(
        prepare_script,
        "prepare_stanford_area1",
        lambda *positional, **keywords: ([{"id": "sample"}], {"receipt": True}),
    )
    monkeypatch.setattr(
        prepare_script,
        "write_stanford_manifest",
        lambda *positional: (tmp_path / "manifest.jsonl", tmp_path / "metadata.json"),
    )

    prepare_script.main()

    output = capsys.readouterr().out
    assert "Wrote 1 records" in output
    assert "Canonical manifest.jsonl" not in output
