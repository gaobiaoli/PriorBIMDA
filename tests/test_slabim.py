from pathlib import Path
import tempfile
import zipfile

import pytest

from bim_priorda3.data.slabim import safe_extract


def test_safe_extract_filters_rosbag() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "region.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("5F_Region2/images/data/000.png", b"image")
            zipped.writestr("5F_Region2/rosbag/part.bag", b"bag")
        core = root / "core"
        bag = root / "bag"
        assert safe_extract(archive, core, mode="core") == 1
        assert (core / "5F_Region2/images/data/000.png").exists()
        assert not (core / "5F_Region2/rosbag/part.bag").exists()
        assert safe_extract(archive, bag, mode="rosbag") == 1
        assert (bag / "5F_Region2/rosbag/part.bag").exists()
        assert not (bag / "5F_Region2/images/data/000.png").exists()


def test_safe_extract_rejects_parent_traversal() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../escape.txt", b"unsafe")
        with pytest.raises(ValueError):
            safe_extract(archive, root / "output")

