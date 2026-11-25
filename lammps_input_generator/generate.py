"""Molecule + settings -> LAMMPS files.  Qt-free, so the tests drive it directly."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import cell as cellmod
from . import writer
from .mdff import load_mdff
from .topology import LammpsTopology, build_elements_only, build_from_mdff, build_from_uff


@dataclass
class Generated:
    topology: LammpsTopology
    box: cellmod.LammpsBox
    cell: cellmod.CellInfo
    data_text: str
    input_text: str
    issues: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""

    @property
    def has_errors(self) -> bool:
        return any(level == "ERROR" for level, _ in self.issues)


def molecule_arrays(mol):
    """(elements, masses, positions) of an RDKit molecule with a conformer."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("No molecule is loaded.")
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule has no 3D coordinates; convert it to 3D first.")
    conf = mol.GetConformer()
    elements, masses, coords, dummies = [], [], [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummies.append(atom.GetIdx() + 1)
        elements.append(atom.GetSymbol())
        masses.append(float(atom.GetMass()))
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((p.x, p.y, p.z))
    if dummies:
        raise ValueError(
            "Dummy or ghost atoms cannot be simulated (atoms "
            + ", ".join(str(i) for i in dummies[:10])
            + "); remove them first."
        )
    return elements, masses, np.asarray(coords, dtype=float)


class TopologyCache:
    """Keeps the last parameterisation; UFF on a large system takes a while."""

    def __init__(self):
        self._key = None
        self._value: LammpsTopology | None = None

    @staticmethod
    def key_for(mol, settings: dict):
        source = settings.get("ff_source", "uff")
        bonds = tuple(
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()
        )
        atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge()) for a in mol.GetAtoms())
        extra: tuple = ()
        if source == "uff":
            extra = (settings.get("charge_method"), bool(settings.get("constrain_h")))
        elif source == "mdff":
            path = settings.get("mdff_path", "")
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None
            extra = (path, mtime, settings.get("dihedral_style"), bool(settings.get("constrain_h")))
        return (source, hash(atoms), hash(bonds), extra)

    def get(self, mol, settings: dict, elements, masses) -> LammpsTopology:
        key = self.key_for(mol, settings)
        if key != self._key or self._value is None:
            self._value = build_topology(mol, settings, elements, masses)
            self._key = key
        return self._value

    def clear(self):
        self._key = None
        self._value = None


def build_topology(mol, settings: dict, elements, masses) -> LammpsTopology:
    source = settings.get("ff_source", "uff")
    if source == "mdff":
        path = settings.get("mdff_path", "")
        if not path:
            raise ValueError("Choose a moleditpy-mdff (.mdff.json) file, or use the UFF fallback.")
        mdff = load_mdff(path)
        return build_from_mdff(
            mdff,
            elements,
            dihedral_style=settings.get("dihedral_style", "auto"),
            constrain_h=bool(settings.get("constrain_h", False)),
        )
    if source == "reaxff":
        return build_elements_only(elements, masses, "reaxff")
    if source == "other":
        return build_elements_only(elements, masses, "other")
    return build_from_uff(
        mol,
        charge_method=settings.get("charge_method", "gasteiger"),
        constrain_h=bool(settings.get("constrain_h", False)),
    )


def generate(
    mol,
    settings: dict,
    version: str = "",
    remembered: cellmod.CellInfo | None = None,
    viewer_cell: cellmod.CellInfo | None = None,
    cache: TopologyCache | None = None,
) -> Generated:
    elements, masses, positions = molecule_arrays(mol)
    topo = (
        cache.get(mol, settings, elements, masses)
        if cache
        else build_topology(mol, settings, elements, masses)
    )
    cell, notes = cellmod.resolve_cell(settings, mol, positions, remembered, viewer_cell)
    box = cellmod.to_lammps_box(cell, positions)
    box.notes = list(notes) + [f"Cell source: {cell.source}"] + box.notes
    data_text = writer.build_data_file(topo, box, settings, version)
    input_text = writer.build_input_script(topo, box, settings, version)
    issues = writer.validate(topo, box, settings, host_elements=elements)
    return Generated(
        topology=topo,
        box=box,
        cell=cell,
        data_text=data_text,
        input_text=input_text,
        issues=issues,
        summary=writer.summary(topo, box),
    )


def write_files(generated: Generated, settings: dict, directory: str) -> dict[str, str]:
    """Write the data file and input script into ``directory``; returns the paths."""
    names = writer.file_names(settings)
    paths = {
        "data": os.path.join(directory, names["data"]),
        "input": os.path.join(directory, names["input"]),
    }
    with open(paths["data"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(generated.data_text)
    with open(paths["input"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(generated.input_text)
    return paths
