"""RDKit UFF fallback: parameters and conventions match RDKit's own UFF."""

import pytest
from helpers import embedded

pytest.importorskip("rdkit")
pytest.importorskip("numpy")

from rdkit.Chem import rdForceFieldHelpers as ffh

from lammps_input_generator.topology import build_from_uff


def _dihedrals_about(topo, j, k):
    return [d for d in topo.dihedrals if {d[2], d[3]} == {j, k}]


def test_bond_and_angle_constants_are_half_of_uff():
    mol = embedded("CCO")
    topo = build_from_uff(mol)
    for tid, i, j in topo.bonds:
        kb, r0 = ffh.GetUFFBondStretchParams(mol, i, j)
        assert topo.bond_types[tid] == (pytest.approx(kb / 2), pytest.approx(r0))
    for tid, i, j, k in topo.angles:
        ka, theta0 = ffh.GetUFFAngleBendParams(mol, i, j, k)
        assert topo.angle_types[tid] == (pytest.approx(ka / 2), pytest.approx(theta0))
    n_angles = sum(a.GetDegree() * (a.GetDegree() - 1) // 2 for a in mol.GetAtoms())
    assert len(topo.angles) == n_angles


def test_vdw_sigma_from_uff_minimum():
    mol = embedded("C")
    topo = build_from_uff(mol)
    x, d = ffh.GetUFFVdWParams(mol, 0, 0)
    carbon = topo.atom_types[topo.atom_type[0]]
    assert carbon.sigma == pytest.approx(x / 2 ** (1 / 6))
    assert carbon.epsilon == pytest.approx(d)
    assert topo.mixing == "geometric"
    assert topo.special_lj == (0.0, 0.0, 1.0)


def test_ethane_torsion_is_split_over_nine_terms():
    mol = embedded("CC")
    topo = build_from_uff(mol)
    terms = _dihedrals_about(topo, 0, 1)
    assert len(terms) == 9
    v = ffh.GetUFFTorsionParams(mol, 2, 0, 1, 5)
    for tid, *_ in terms:
        k, n, d = topo.dihedral_types[tid]
        assert (n, d) == (3, 0)  # V/2 (1 + cos 3 phi)
        assert k == pytest.approx(v / 2 / 9)


def test_benzene_ring_torsions_are_twofold_with_phase_180():
    mol = embedded("c1ccccc1")
    topo = build_from_uff(mol)
    for tid, *_ in topo.dihedrals:
        _, n, d = topo.dihedral_types[tid]
        assert (n, d) == (2, 180)  # V/2 (1 - cos 2 phi)


def test_propene_special_case():
    mol = embedded("CC=C")
    topo = build_from_uff(mol)
    terms = _dihedrals_about(topo, 0, 1)
    assert len(terms) == 6
    forms = {}
    for tid, i, j, k, l in terms:
        k_lmp, n, d = topo.dihedral_types[tid]
        end = l if k == 1 else i
        forms[mol.GetAtomWithIdx(end).GetSymbol()] = (n, d, k_lmp)
    assert forms["C"][:2] == (3, 0) and forms["C"][2] == pytest.approx(2.0 / 2 / 6)
    assert forms["H"][:2] == (6, 180) and forms["H"][2] == pytest.approx(1.0 / 2 / 6)


def test_carbonyl_inversion_is_umbrella_with_central_atom_first():
    mol = embedded("C=O")
    topo = build_from_uff(mol)
    assert topo.improper_style == "umbrella"
    assert len(topo.impropers) == 3
    for tid, centre, *_ in topo.impropers:
        assert mol.GetAtomWithIdx(centre).GetSymbol() == "C"
        assert topo.improper_types[tid] == (pytest.approx(50.0 / 3), 0.0)


def test_no_inversion_for_sp3_centres():
    topo = build_from_uff(embedded("CO"))
    assert topo.impropers == [] and topo.improper_style is None


@pytest.mark.parametrize("method", ["gasteiger", "mmff", "zero"])
def test_charge_methods(method):
    topo = build_from_uff(embedded("CC(=O)O"), charge_method=method)
    assert topo.total_charge == pytest.approx(0.0, abs=1e-6)
    if method == "zero":
        assert set(topo.charges) == {0.0}
    else:
        assert any(abs(q) > 0.05 for q in topo.charges)


def test_labels_and_fragments():
    topo = build_from_uff(embedded("c1ccccc1C.O"))
    labels = {t.label for t in topo.atom_types}
    assert {"C_R", "C_3", "H_", "O_3"} <= labels
    assert len(set(topo.mol_ids)) == 2


def test_h_bond_constraints_flag_only_h_bonds():
    mol = embedded("CCO")
    topo = build_from_uff(mol, constrain_h=True)
    for tid, i, j in topo.bonds:
        has_h = "H" in (topo.elements[i], topo.elements[j])
        assert (tid in topo.shake_bond_types) == has_h
