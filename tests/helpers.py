"""Shared test helpers: molecules and a small LAMMPS data-file parser."""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
EXAMPLE_MDFF = os.path.join(EXAMPLES, "ethanol_water_ions.mdff.json")
EXAMPLE_XYZ = os.path.join(EXAMPLES, "ethanol_water_ions.xyz")


def rdkit():
    return pytest.importorskip("rdkit")


def embedded(smiles, seed=7):
    rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def example_molecule():
    rdkit()
    from rdkit import Chem

    return Chem.MolFromXYZFile(EXAMPLE_XYZ)


HEADER_KEYS = {
    "atoms": "Atoms",
    "bonds": "Bonds",
    "angles": "Angles",
    "dihedrals": "Dihedrals",
    "impropers": "Impropers",
}
TYPE_KEYS = {
    "atom types": ("Masses", "Pair Coeffs"),
    "bond types": ("Bond Coeffs",),
    "angle types": ("Angle Coeffs",),
    "dihedral types": ("Dihedral Coeffs",),
    "improper types": ("Improper Coeffs",),
}
SECTIONS = {
    "Masses",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Atoms",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
}


def parse_data(text):
    """Parse a LAMMPS data file into (header counts, box, {section: [rows]})."""
    lines = text.split("\n")
    header = {}
    box = {}
    sections = {}
    current = None
    for line in lines[1:]:
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        name = line.split("#")[0].strip()
        if name in SECTIONS:
            current = name
            sections[current] = []
            continue
        if current is None:
            parts = stripped.split()
            if parts[-1] in ("xhi", "yhi", "zhi"):
                box[parts[-2]] = (float(parts[0]), float(parts[1]))
            elif parts[-1] == "yz":
                box["tilt"] = tuple(float(v) for v in parts[:3])
            else:
                header[" ".join(parts[1:])] = int(parts[0])
        else:
            sections[current].append(stripped.split())
    return header, box, sections


def check_data_consistency(text):
    """Assert counts, ids and type references of a data file agree."""
    header, _, sections = parse_data(text)
    for key, section in HEADER_KEYS.items():
        count = header.get(key, 0)
        assert len(sections.get(section, [])) == count, f"{section}: {count} expected"
    for key, names in TYPE_KEYS.items():
        ntypes = header.get(key, 0)
        for section in names:
            if section in sections:
                ids = sorted(int(row[0]) for row in sections[section])
                assert ids == list(range(1, ntypes + 1)), section
    n_atoms = header["atoms"]
    atoms = sections["Atoms"]
    assert sorted(int(r[0]) for r in atoms) == list(range(1, n_atoms + 1))
    for key, section in HEADER_KEYS.items():
        if section == "Atoms" or section not in sections:
            continue
        ntypes = header[key[:-1] + " types"]
        for row in sections[section]:
            assert 1 <= int(row[1]) <= ntypes
            assert all(1 <= int(a) <= n_atoms for a in row[2:])
    return header, sections
