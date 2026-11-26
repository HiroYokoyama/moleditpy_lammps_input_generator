"""The shared periodic modules must be materialized from the pinned submodule."""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "lammps_input_generator"
_REQUIRED = ("cell_model.py", "elements.py")


@pytest.mark.parametrize("filename", _REQUIRED)
def test_the_shared_module_is_materialized(filename):
    assert (PACKAGE_DIR / filename).is_file(), (
        f"{filename} is missing. Run `git submodule update --init` then "
        "`python scripts/materialize_shared.py`."
    )


@pytest.mark.parametrize("filename", _REQUIRED)
def test_materialized_source_matches_pinned_release(filename):
    source = ROOT / "_periodic_shared" / "periodic_shared" / filename
    if not source.is_file():
        pytest.skip("submodule not checked out")
    version = (ROOT / "_periodic_shared" / "VERSION").read_text(encoding="utf-8").strip()
    canonical = source.read_text(encoding="utf-8")
    assert (PACKAGE_DIR / filename).read_text(encoding="utf-8") == canonical
    match = re.search(r'^SHARED_MODULE_VERSION = "([^"]+)"', canonical, re.MULTILINE)
    assert match is not None and match.group(1) == version


def test_materialized_files_are_not_committed():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for filename in _REQUIRED:
        assert f"lammps_input_generator/{filename}" in ignore
    assert os.path.isfile(ROOT / "scripts" / "materialize-config.json")
