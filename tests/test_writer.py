"""Data file and input script text, and the internal checks."""

import json

import pytest
from helpers import EXAMPLE_MDFF, check_data_consistency, embedded, example_molecule, parse_data

np = pytest.importorskip("numpy")
pytest.importorskip("rdkit")

from lammps_input_generator import cell as cellmod
from lammps_input_generator import generate as gen
from lammps_input_generator import writer
from lammps_input_generator.topology import build_elements_only


def _settings(**changes):
    settings = writer.default_settings()
    settings.update(changes)
    return settings


def _mdff(**changes):
    return _settings(ff_source="mdff", mdff_path=EXAMPLE_MDFF, **changes)


def _lines(text, prefix):
    return [line for line in text.splitlines() if line.startswith(prefix)]


def test_uff_data_file_is_consistent():
    result = gen.generate(embedded("CC(=O)Nc1ccccc1"), _settings())
    header, sections = check_data_consistency(result.data_text)
    assert header["atoms"] == result.topology.n_atoms
    assert "Improper Coeffs" in sections and "Dihedral Coeffs" in sections
    assert "Pair Coeffs # lj/cut/coul/long" in result.data_text
    assert not result.has_errors


def test_mdff_data_file_is_consistent_and_rigid_water():
    result = gen.generate(example_molecule(), _mdff())
    header, sections = check_data_consistency(result.data_text)
    assert header["atoms"] == 20
    assert "Impropers" not in sections
    shake = _lines(result.input_text, "fix             constrain")
    assert shake and " b " in shake[0] and " a " in shake[0]
    assert "special_bonds   lj 0 0 0.5 coul 0 0 0.8333" in result.input_text
    charges = [float(row[3]) for row in sections["Atoms"]]
    assert sum(charges) == pytest.approx(0.0, abs=1e-6)


def test_atoms_section_formats_per_style():
    mol = embedded("O")
    full = gen.generate(mol, _settings())
    _, _, sections = parse_data(full.data_text)
    assert len(sections["Atoms"][0]) == 10  # id mol type q x y z ix iy iz
    reax = gen.generate(mol, _settings(ff_source="reaxff"))
    _, _, sections = parse_data(reax.data_text)
    assert len(sections["Atoms"][0]) == 9 and "Bonds" not in sections
    other = gen.generate(mol, _settings(ff_source="other"))
    _, _, sections = parse_data(other.data_text)
    assert len(sections["Atoms"][0]) == 8
    assert "atom_style      atomic" in other.input_text


def test_triclinic_header_from_cell_property():
    mol = embedded("O")
    lattice = cellmod.cell_vectors((15.0, 15.0, 15.0), (90.0, 90.0, 120.0))
    mol.SetProp("moleditpy_cell", json.dumps({"lattice": lattice.tolist()}))
    result = gen.generate(mol, _settings())
    _, box, _ = parse_data(result.data_text)
    assert box["tilt"][0] == pytest.approx(-7.5)
    assert result.cell.source == "moleditpy_cell"


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"ensemble": "nve"}, ["fix             integrate all nve"]),
        ({"ensemble": "nvt"}, ["fix             integrate all nvt temp 300 300 100"]),
        (
            {"ensemble": "nvt", "thermostat": "langevin"},
            [
                "fix             integrate all nve",
                "fix             thermostat all langevin 300 300 100 12345 zero yes",
            ],
        ),
        (
            {"ensemble": "nvt", "thermostat": "csvr"},
            ["fix             thermostat all temp/csvr 300 300 100 12345"],
        ),
        (
            {"ensemble": "nvt", "thermostat": "berendsen"},
            ["fix             thermostat all temp/berendsen 300 300 100"],
        ),
        ({"ensemble": "npt"}, ["fix             integrate all npt temp 300 300 100 iso 1 1 1000"]),
        (
            {"ensemble": "npt", "thermostat": "langevin"},
            [
                "fix             integrate all nph iso 1 1 1000",
                "fix             thermostat all langevin",
            ],
        ),
        (
            {"ensemble": "npt", "barostat": "berendsen"},
            [
                "fix             integrate all nvt temp",
                "fix             barostat all press/berendsen iso 1 1 1000",
            ],
        ),
        ({"ensemble": "npt", "coupling": "tri"}, ["aniso 1 1 1000"]),
    ],
)
def test_ensemble_fixes(changes, expected):
    result = gen.generate(embedded("CO"), _settings(**changes))
    for fragment in expected:
        assert fragment in result.input_text, fragment


def test_minimize_protocol_has_no_md():
    text = gen.generate(embedded("CO"), _settings(ensemble="minimize")).input_text
    assert "minimize        0.0001 1e-06 1000 10000" in text
    assert "velocity" not in text and "run " not in text
    assert "write_data      system_final.data" in text and "write_restart" not in text


def test_minimize_first_and_order():
    text = gen.generate(embedded("CO"), _settings(minimize_first=True, constrain_h=True)).input_text
    order = [
        text.index(key)
        for key in ("read_data", "minimize", "fix             constrain", "velocity", "run ")
    ]
    assert order == sorted(order)
    text = gen.generate(embedded("CO"), _settings(minimize_first=False)).input_text
    assert "minimize" not in text


def test_dump_restart_and_names():
    text = gen.generate(
        embedded("CO"), _settings(job_name="my run!", dump_format="custom", restart_every=5000)
    ).input_text
    assert "read_data       my_run.data" in text
    assert (
        "dump            traj all custom 1000 my_run.lammpstrj id mol type element q x y z ix iy iz"
        in text
    )
    assert "dump_modify     traj element" in text
    assert "restart         5000 my_run.restart.1 my_run.restart.2" in text
    text = gen.generate(embedded("CO"), _settings(dump_format="none")).input_text
    assert "dump" not in text


def test_xyz_dump_maps_types_to_elements():
    result = gen.generate(embedded("CO"), _settings())
    line = _lines(result.input_text, "dump_modify")[0]
    assert line.split()[3:-2] == result.topology.type_elements()


def test_zero_charges_use_plain_lj():
    result = gen.generate(embedded("CC"), _settings(charge_method="zero"))
    assert "pair_style      lj/cut 10" in result.input_text
    assert "kspace_style" not in result.input_text


def test_non_periodic_box_uses_cutoff_coulomb_and_shrink_wrap():
    result = gen.generate(embedded("CO"), _settings(cell_source="user", user_periodic=False))
    assert "boundary        s s s" in result.input_text
    assert "lj/cut/coul/cut" in result.input_text and "kspace_style" not in result.input_text
    assert "rot yes" in result.input_text


def test_metal_units_convert_times():
    text = gen.generate(
        embedded("[Si]"), _settings(ff_source="other", ensemble="nvt", timestep=1.0, tdamp=100.0)
    ).input_text
    assert "units           metal" in text
    assert "timestep        0.001" in text
    assert "nvt temp 300 300 0.1" in text


def test_reaxff_script():
    text = gen.generate(
        embedded("CO"), _settings(ff_source="reaxff", reax_ffield="ffield.CHO")
    ).input_text
    assert "pair_style      reaxff NULL" in text
    assert "pair_coeff      * * ffield.CHO C O H" in text
    assert "fix             qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff" in text
    assert text.index("read_data") < text.index("pair_style")


def test_check_only_script():
    result = gen.generate(embedded("CO"), _settings(constrain_h=True))
    text = writer.build_input_script(
        result.topology, result.box, _settings(constrain_h=True), check_only=True
    )
    assert text.rstrip().endswith("run             0")
    assert "dump" not in text and "minimize" not in text and "write_data" not in text
    assert "fix             constrain" in text


def test_validation_flags_problems():
    methane = gen.generate(embedded("C"), _settings(constrain_h=True))
    assert any("at most 3" in m for level, m in methane.issues if level == "ERROR")
    npt = gen.generate(
        embedded("CO"), _settings(ensemble="npt", cell_source="user", user_periodic=False)
    )
    assert any("NPT" in m for level, m in npt.issues if level == "ERROR")
    small = gen.generate(embedded("CO"), _settings(cell_source="user", padding=2.0))
    assert any("twice the" in m for level, m in small.issues if level == "WARNING")
    fast = gen.generate(embedded("CO"), _settings(timestep=2.0))
    assert any("unstable" in m for _, m in fast.issues)
    ion = gen.generate(embedded("[NH4+]"), _settings(charge_method="zero"))
    assert not ion.has_errors


def test_net_charge_warning():
    result = gen.generate(embedded("[NH4+]"), _settings(charge_method="gasteiger"))
    assert any("net charge" in m for _, m in result.issues)


def test_elements_only_types_keep_first_appearance_order():
    topo = build_elements_only(["O", "H", "H", "C"], [16.0, 1.0, 1.0, 12.0], "reaxff")
    assert topo.type_elements() == ["O", "H", "C"]
    assert topo.atom_type == [0, 1, 1, 2]


def test_summary_mentions_counts():
    result = gen.generate(embedded("CO"), _settings())
    assert f"{result.topology.n_atoms} atoms" in result.summary
    assert "orthorhombic" in result.summary
