"""moleditpy-mdff parsing and conversion to LAMMPS real units."""

import copy
import json
import math

import pytest
from helpers import EXAMPLE_MDFF

from lammps_input_generator.mdff import MdffError, load_mdff, parse_mdff, resolve_system
from lammps_input_generator.topology import (
    build_from_mdff,
    longest_bond_path,
    special_bonds_for,
)


def _atom(name, element, atype, q=0.0, mass=12.0, sigma=0.3, eps=0.5):
    return {
        "name": name,
        "element": element,
        "type": atype,
        "charge": q,
        "mass": mass,
        "sigma": sigma,
        "epsilon": eps,
    }


def _doc(types, system=None, **extra):
    doc = {
        "format": "moleditpy-mdff",
        "version": 1,
        "source": "test",
        "combination_rule": 2,
        "fudge_lj": 0.5,
        "fudge_qq": 0.8333,
        "molecule_types": types,
    }
    if system is not None:
        doc["system"] = system
    doc.update(extra)
    return doc


def _butane_like():
    atoms = [_atom(f"C{i}", "C", "CT") for i in range(4)]
    return {
        "name": "BUT",
        "atoms": atoms,
        "bonds": [{"i": i, "j": i + 1, "r0": 0.153, "k": 250000.0} for i in range(3)],
        "angles": [
            {"i": 0, "j": 1, "k_atom": 2, "theta0": 112.0, "k": 500.0},
            {"i": 1, "j": 2, "k_atom": 3, "theta0": 112.0, "k": 500.0},
        ],
        "torsions": [
            {"i": 0, "j": 1, "k_atom": 2, "l": 3, "periodicity": 3, "phase": 0.0, "k": 4.184}
        ],
    }


def test_example_file_loads():
    mdff = load_mdff(EXAMPLE_MDFF)
    assert set(mdff.molecule_types) == {"ETOH", "SOL", "NA", "CL"}
    assert mdff.molecule_types["SOL"].settles[0]["doh"] == pytest.approx(0.09572)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(format="other"), "format"),
        (lambda d: d.update(version=2), "version"),
        (lambda d: d.update(molecule_types=[]), "molecule_types"),
        (
            lambda d: d["molecule_types"][0]["bonds"].append({"i": 0, "j": 9, "r0": 1, "k": 1}),
            "outside",
        ),
        (lambda d: d["molecule_types"][0]["atoms"][0].pop("type"), "type"),
        (lambda d: d.update(system=[{"type": "NOPE", "count": 1, "atom_offset": 0}]), "unknown"),
        (lambda d: d.update(combination_rule=7), "combination_rule"),
        (lambda d: d["molecule_types"].append(copy.deepcopy(d["molecule_types"][0])), "twice"),
    ],
)
def test_invalid_files_are_rejected(mutate, message):
    doc = _doc([_butane_like()])
    mutate(doc)
    with pytest.raises(MdffError, match=message):
        parse_mdff(doc)


def test_load_reports_bad_json(tmp_path):
    path = tmp_path / "bad.mdff.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MdffError, match="not valid JSON"):
        load_mdff(str(path))


def test_system_inferred_for_a_single_type():
    mdff = parse_mdff(_doc([_butane_like()]))
    entries = resolve_system(mdff, 12)
    assert (entries[0].count, entries[0].atom_offset) == (3, 0)
    with pytest.raises(MdffError, match="whole number"):
        resolve_system(mdff, 10)


def test_unit_conversion():
    mdff = parse_mdff(_doc([_butane_like()]))
    topo = build_from_mdff(mdff, ["C"] * 4)
    k_bond, r0 = topo.bond_types[0]
    assert r0 == pytest.approx(1.53)
    assert k_bond == pytest.approx(250000.0 / 2 / 4.184 / 100)
    k_angle, theta = topo.angle_types[0]
    assert theta == pytest.approx(112.0)
    assert k_angle == pytest.approx(500.0 / 2 / 4.184)
    assert topo.atom_types[0].sigma == pytest.approx(3.0)
    assert topo.atom_types[0].epsilon == pytest.approx(0.5 / 4.184)
    assert topo.dihedral_style == "charmm"
    assert topo.dihedral_types[0] == (pytest.approx(1.0), 3, 0)
    assert topo.mixing == "arithmetic"
    assert topo.special_lj == (0.0, 0.0, 0.5)
    assert topo.special_coul == (0.0, 0.0, 0.8333)


def test_geometric_mixing_for_rules_1_and_3():
    for rule in (1, 3):
        topo = build_from_mdff(parse_mdff(_doc([_butane_like()], combination_rule=rule)), ["C"] * 4)
        assert topo.mixing == "geometric"


def test_fractional_phase_switches_auto_to_fourier():
    mtype = _butane_like()
    mtype["torsions"].append(
        {"i": 0, "j": 1, "k_atom": 2, "l": 3, "periodicity": 1, "phase": 12.5, "k": 2.0}
    )
    mdff = parse_mdff(_doc([mtype]))
    topo = build_from_mdff(mdff, ["C"] * 4)
    assert topo.dihedral_style == "fourier"
    assert len(topo.dihedrals) == 1
    assert len(topo.dihedral_types[0]) == 2  # two terms on one quartet
    with pytest.raises(MdffError, match="whole degree"):
        build_from_mdff(mdff, ["C"] * 4, dihedral_style="charmm")


def test_charmm_writes_one_dihedral_per_term():
    mtype = _butane_like()
    mtype["torsions"].append(
        {"i": 0, "j": 1, "k_atom": 2, "l": 3, "periodicity": 1, "phase": 180.0, "k": 2.0}
    )
    topo = build_from_mdff(parse_mdff(_doc([mtype])), ["C"] * 4)
    assert topo.dihedral_style == "charmm"
    assert len(topo.dihedrals) == 2


def test_impropers_cvff_or_dihedral():
    mtype = _butane_like()
    mtype["torsions"] = [
        {
            "i": 1,
            "j": 0,
            "k_atom": 2,
            "l": 3,
            "periodicity": 2,
            "phase": 180.0,
            "k": 4.184,
            "improper": True,
        },
        {
            "i": 1,
            "j": 0,
            "k_atom": 2,
            "l": 3,
            "periodicity": 1,
            "phase": 0.0,
            "k": 4.184,
            "improper": True,
        },
        {
            "i": 1,
            "j": 0,
            "k_atom": 2,
            "l": 3,
            "periodicity": 2,
            "phase": 90.0,
            "k": 4.184,
            "improper": True,
        },
    ]
    topo = build_from_mdff(parse_mdff(_doc([mtype])), ["C"] * 4)
    assert topo.improper_style == "cvff"
    assert sorted(topo.improper_types) == sorted(
        [(pytest.approx(1.0), -1, 2), (pytest.approx(1.0), 1, 1)]
    )
    assert len(topo.dihedrals) == 1  # the 90-degree one
    assert any("phase other than" in w for w in topo.warnings)


def test_settles_add_rigid_bonds_and_angle():
    water = {
        "name": "SOL",
        "atoms": [
            _atom("OW", "O", "OW", -0.8, 16.0),
            _atom("H1", "H", "HW", 0.4, 1.0, 0, 0),
            _atom("H2", "H", "HW", 0.4, 1.0, 0, 0),
        ],
        "settles": {"i": 0, "doh": 0.1, "dhh": 0.1633},
        "exclusions_nrexcl": 2,
    }
    topo = build_from_mdff(parse_mdff(_doc([water])), ["O", "H", "H"] * 2)
    assert len(topo.bonds) == 4 and len(topo.angles) == 2
    assert topo.shake_bond_types == [0] and topo.shake_angle_types == [0]
    assert topo.bond_types[0][1] == pytest.approx(1.0)
    expected = math.degrees(2 * math.asin(0.1633 / 0.2))
    assert topo.angle_types[0][1] == pytest.approx(expected)
    assert topo.mol_ids == [1, 1, 1, 2, 2, 2]


def test_constraints_and_h_bond_shake_get_their_own_types():
    mtype = _butane_like()
    mtype["atoms"][3] = _atom("H", "H", "HC", mass=1.0)
    mtype["bonds"].append({"i": 0, "j": 2, "r0": 0.25, "k": 1000.0})
    mtype["constraints"] = [{"i": 0, "j": 3, "r0": 0.11}]
    mdff = parse_mdff(_doc([mtype]))
    topo = build_from_mdff(mdff, ["C", "C", "C", "H"], constrain_h=True)
    shaken = {tuple(sorted((i, j))) for t, i, j in topo.bonds if t in topo.shake_bond_types}
    assert (0, 3) in shaken and (2, 3) in shaken
    assert (0, 1) not in shaken


def test_element_mismatch_and_coverage_errors():
    mdff = parse_mdff(
        _doc([_butane_like()], system=[{"type": "BUT", "count": 1, "atom_offset": 0}])
    )
    with pytest.raises(MdffError, match="atom order"):
        build_from_mdff(mdff, ["C", "O", "C", "C"])
    with pytest.raises(MdffError, match="not described"):
        build_from_mdff(mdff, ["C"] * 5)
    overlap = parse_mdff(
        _doc(
            [_butane_like()],
            system=[
                {"type": "BUT", "count": 1, "atom_offset": 0},
                {"type": "BUT", "count": 1, "atom_offset": 2},
            ],
        )
    )
    with pytest.raises(MdffError, match="two system entries"):
        build_from_mdff(overlap, ["C"] * 6)


def test_same_type_name_with_other_parameters_is_split():
    one = _butane_like()
    two = copy.deepcopy(_butane_like())
    two["name"] = "BUT2"
    for atom in two["atoms"]:
        atom["epsilon"] = 0.9
    doc = _doc(
        [one, two],
        system=[
            {"type": "BUT", "count": 1, "atom_offset": 0},
            {"type": "BUT2", "count": 1, "atom_offset": 4},
        ],
    )
    topo = build_from_mdff(parse_mdff(doc), ["C"] * 8)
    assert [t.label for t in topo.atom_types] == ["CT", "CT_2"]


def test_longest_bond_path():
    assert longest_bond_path(1, []) == 0
    assert longest_bond_path(3, [(0, 1), (0, 2)]) == 2
    assert longest_bond_path(6, [(i, i + 1) for i in range(5)]) == 3


def test_special_bonds_ignore_types_without_such_pairs():
    warnings = []
    lj, coul = special_bonds_for(
        [("ETOH", 3, 3, 9), ("SOL", 2, 2, 900), ("NA", 1, 0, 3)], 0.5, 0.8333, warnings
    )
    assert lj == (0.0, 0.0, 0.5) and coul == (0.0, 0.0, 0.8333)
    assert warnings == []


def test_special_bonds_conflict_warns_and_follows_the_largest():
    warnings = []
    lj, _ = special_bonds_for([("A", 3, 3, 10), ("B", 2, 3, 1000)], 0.5, 0.8333, warnings)
    assert lj == (0.0, 0.0, 1.0)
    assert any("disagree" in w for w in warnings)


def test_example_round_trip_values():
    with open(EXAMPLE_MDFF, encoding="utf-8") as handle:
        data = json.load(handle)
    mdff = parse_mdff(data)
    elements = ["C", "C", "O", "H", "H", "H", "H", "H", "H"] + ["O", "H", "H"] * 3 + ["Na", "Cl"]
    topo = build_from_mdff(mdff, elements)
    assert topo.total_charge == pytest.approx(0.0, abs=1e-9)
    assert topo.special_lj == (0.0, 0.0, 0.5)
    assert len(set(topo.mol_ids)) == 6
