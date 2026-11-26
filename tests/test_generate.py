"""End-to-end generation, file writing and the lmp helper functions."""

import os

import pytest
from helpers import EXAMPLE_MDFF, embedded, example_molecule

pytest.importorskip("numpy")
pytest.importorskip("rdkit")

from rdkit import Chem

from lammps_input_generator import generate as gen
from lammps_input_generator import lmp_runner, writer


def test_no_molecule_and_no_conformer():
    with pytest.raises(ValueError, match="No molecule"):
        gen.generate(None, writer.default_settings())
    with pytest.raises(ValueError, match="3D"):
        gen.generate(Chem.MolFromSmiles("CC"), writer.default_settings())


def test_dummy_atoms_are_refused():
    mol = embedded("CC")
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(0).SetAtomicNum(0)
    with pytest.raises(ValueError, match="Dummy"):
        gen.generate(rw.GetMol(), writer.default_settings())


def test_mdff_without_a_path():
    with pytest.raises(ValueError, match="Choose"):
        gen.generate(embedded("O"), dict(writer.default_settings(), ff_source="mdff"))


def test_mdff_for_a_mismatched_structure():
    from lammps_input_generator.mdff import MdffError

    settings = dict(writer.default_settings(), ff_source="mdff", mdff_path=EXAMPLE_MDFF)
    with pytest.raises(MdffError):
        gen.generate(embedded("CCCC"), settings)


def test_cache_reuses_topology_until_inputs_change():
    cache = gen.TopologyCache()
    mol = embedded("CCO")
    settings = writer.default_settings()
    first = gen.generate(mol, settings, cache=cache).topology
    second = gen.generate(mol, dict(settings, steps=5), cache=cache).topology
    assert first is second
    third = gen.generate(mol, dict(settings, charge_method="zero"), cache=cache).topology
    assert third is not first


def test_write_files_uses_lf(tmp_path):
    result = gen.generate(
        example_molecule(),
        dict(writer.default_settings(), ff_source="mdff", mdff_path=EXAMPLE_MDFF, job_name="wat"),
    )
    paths = gen.write_files(result, dict(writer.default_settings(), job_name="wat"), str(tmp_path))
    assert os.path.basename(paths["input"]) == "in.wat"
    assert os.path.basename(paths["data"]) == "wat.data"
    with open(paths["data"], "rb") as handle:
        raw = handle.read()
    assert b"\r\n" not in raw and raw.decode("utf-8") == result.data_text


def test_find_lammps(monkeypatch, tmp_path):
    monkeypatch.setattr(lmp_runner.shutil, "which", lambda name: None)
    assert lmp_runner.find_lammps() is None
    exe = tmp_path / "lmp.exe"
    exe.write_text("", encoding="utf-8")
    assert lmp_runner.find_lammps(str(exe)) == str(exe)
    monkeypatch.setattr(
        lmp_runner.shutil, "which", lambda name: "/usr/bin/lmp" if name == "lmp" else None
    )
    assert lmp_runner.find_lammps() == "/usr/bin/lmp"


def test_prepare_check_writes_run0(tmp_path):
    settings = writer.default_settings()
    result = gen.generate(embedded("CO"), settings)
    directory, paths = lmp_runner.prepare_check(result, settings)
    try:
        with open(paths["input"], encoding="utf-8") as handle:
            script = handle.read()
        assert "run             0" in script
        assert os.path.exists(paths["data"])
        assert lmp_runner.check_arguments(paths["input"])[:2] == ["-in", "in.check"]
    finally:
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)


def test_interpret_output():
    ok, verdict = lmp_runner.interpret_output(0, "Loop time\nWARNING: something\n")
    assert ok and "1 LAMMPS warning" in verdict
    ok, verdict = lmp_runner.interpret_output(1, "ERROR: Unknown pair style\n")
    assert not ok and verdict.startswith("ERROR")
    ok, verdict = lmp_runner.interpret_output(3, "")
    assert not ok and "code 3" in verdict
