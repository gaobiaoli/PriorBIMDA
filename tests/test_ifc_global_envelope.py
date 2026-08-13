from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data import ifc_envelope
from bim_priorda3.data.ifc_envelope import (
    ENVELOPE_CATEGORIES,
    IFCEnvelopeGeometry,
    build_global_ifc_envelope_scene,
)


def _geometry(source: Path) -> IFCEnvelopeGeometry:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    categories = np.asarray(
        [ENVELOPE_CATEGORIES.index("wall"), ENVELOPE_CATEGORIES.index("door")],
        dtype=np.uint8,
    )
    return IFCEnvelopeGeometry(
        vertices=vertices,
        triangles=triangles,
        triangle_categories=categories,
        category_names=ENVELOPE_CATEGORIES,
        audit={"source_sha256": source.stem, "source_ifc": str(source)},
    )


def test_global_scene_applies_fixed_transforms_and_omits_doors_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {"office_1": tmp_path / "hash-a.ifc", "office_2": tmp_path / "hash-b.ifc"}
    monkeypatch.setattr(
        ifc_envelope,
        "load_ifc_envelope_geometry",
        lambda path, strict=True: _geometry(Path(path)),
    )
    transforms = {room: np.eye(4, dtype=np.float64) for room in paths}
    transforms["office_2"][0, 3] = 10.0

    _scene, geometry = build_global_ifc_envelope_scene(paths, transforms)

    assert geometry.audit["filter_policy"] == "global-area-fixed-core-envelope-v1"
    assert geometry.audit["excluded_envelope_categories"] == ["door", "window"]
    assert len(geometry.triangles) == 2
    assert np.all(geometry.triangle_categories == ENVELOPE_CATEGORIES.index("wall"))
    assert geometry.vertices[:4, 0].max() <= 1.0
    assert geometry.vertices[4:, 0].min() >= 10.0


def test_global_scene_requires_exact_ifc_transform_room_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="room sets differ"):
        build_global_ifc_envelope_scene(
            {"office_1": tmp_path / "office_1.ifc"},
            {"office_2": np.eye(4)},
        )
