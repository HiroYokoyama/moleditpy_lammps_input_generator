"""Reader for the suite's ``moleditpy-mdff`` force-field hand-off format.

The Force Field Builder writes ``*.mdff.json`` (format version 1, GROMACS
units: nm, kJ/mol, degrees, e, amu).  This module only parses and validates
it; conversion to LAMMPS ``real`` units happens in :mod:`topology`.

Optional fields accepted beyond the required core:

* ``constraints``: ``[{"i", "j", "r0"}]`` (``b0`` is accepted as an alias)
* ``settles``: ``{"i", "doh", "dhh"}`` or a list of those (rigid water,
  ``i`` is the oxygen; the two hydrogens follow it)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

FORMAT_NAME = "moleditpy-mdff"
SUPPORTED_VERSIONS = (1,)


class MdffError(ValueError):
    """Raised for a file that is not a usable moleditpy-mdff description."""


@dataclass
class MdffAtom:
    name: str
    element: str
    type: str
    charge: float
    mass: float
    sigma: float  # nm
    epsilon: float  # kJ/mol


@dataclass
class MdffTorsion:
    i: int
    j: int
    k_atom: int
    l: int
    periodicity: float
    phase: float  # degrees
    k: float  # kJ/mol
    improper: bool = False


@dataclass
class MdffMoleculeType:
    name: str
    atoms: list[MdffAtom]
    bonds: list[dict[str, float]] = field(default_factory=list)  # i, j, r0 (nm), k
    angles: list[dict[str, float]] = field(default_factory=list)  # i, j, k_atom, theta0, k
    torsions: list[MdffTorsion] = field(default_factory=list)
    exclusions_nrexcl: int = 3
    constraints: list[dict[str, float]] = field(default_factory=list)  # i, j, r0 (nm)
    settles: list[dict[str, float]] = field(default_factory=list)  # i, doh, dhh (nm)


@dataclass
class MdffSystemEntry:
    type: str
    count: int
    atom_offset: int


@dataclass
class Mdff:
    source: str
    combination_rule: int
    fudge_lj: float
    fudge_qq: float
    molecule_types: dict[str, MdffMoleculeType]
    system: list[MdffSystemEntry]
    path: str | None = None


def _require(mapping: dict[str, Any], key: str, where: str):
    if key not in mapping:
        raise MdffError(f"{where}: missing '{key}'")
    return mapping[key]


def _index(value, n_atoms: int, where: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise MdffError(f"{where}: atom index {value!r} is not an integer") from exc
    if index < 0 or index >= n_atoms:
        raise MdffError(f"{where}: atom index {index} is outside 0..{n_atoms - 1}")
    return index


def _float(value, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MdffError(f"{where}: {value!r} is not a number") from exc


def _parse_molecule_type(raw: dict[str, Any], position: int) -> MdffMoleculeType:
    if not isinstance(raw, dict):
        raise MdffError(f"molecule_types[{position}] is not an object")
    name = str(_require(raw, "name", f"molecule_types[{position}]"))
    where = f"molecule type '{name}'"
    atoms_raw = _require(raw, "atoms", where)
    if not isinstance(atoms_raw, list) or not atoms_raw:
        raise MdffError(f"{where}: 'atoms' must be a non-empty list")
    atoms = []
    for n, atom in enumerate(atoms_raw):
        aw = f"{where} atom {n}"
        if not isinstance(atom, dict):
            raise MdffError(f"{aw} is not an object")
        element = str(atom.get("element", "")).strip()
        atoms.append(
            MdffAtom(
                name=str(atom.get("name", f"A{n + 1}")),
                element=element[:1].upper() + element[1:].lower() if element else "X",
                type=str(_require(atom, "type", aw)),
                charge=_float(atom.get("charge", 0.0), aw),
                mass=_float(_require(atom, "mass", aw), aw),
                sigma=_float(atom.get("sigma", 0.0), aw),
                epsilon=_float(atom.get("epsilon", 0.0), aw),
            )
        )
    n_atoms = len(atoms)

    bonds = []
    for n, bond in enumerate(raw.get("bonds") or []):
        bw = f"{where} bond {n}"
        bonds.append(
            {
                "i": _index(_require(bond, "i", bw), n_atoms, bw),
                "j": _index(_require(bond, "j", bw), n_atoms, bw),
                "r0": _float(_require(bond, "r0", bw), bw),
                "k": _float(_require(bond, "k", bw), bw),
            }
        )
    angles = []
    for n, angle in enumerate(raw.get("angles") or []):
        aw = f"{where} angle {n}"
        angles.append(
            {
                "i": _index(_require(angle, "i", aw), n_atoms, aw),
                "j": _index(_require(angle, "j", aw), n_atoms, aw),
                "k_atom": _index(_require(angle, "k_atom", aw), n_atoms, aw),
                "theta0": _float(_require(angle, "theta0", aw), aw),
                "k": _float(_require(angle, "k", aw), aw),
            }
        )
    torsions = []
    for n, tor in enumerate(raw.get("torsions") or []):
        tw = f"{where} torsion {n}"
        torsions.append(
            MdffTorsion(
                i=_index(_require(tor, "i", tw), n_atoms, tw),
                j=_index(_require(tor, "j", tw), n_atoms, tw),
                k_atom=_index(_require(tor, "k_atom", tw), n_atoms, tw),
                l=_index(_require(tor, "l", tw), n_atoms, tw),
                periodicity=_float(_require(tor, "periodicity", tw), tw),
                phase=_float(tor.get("phase", 0.0), tw),
                k=_float(_require(tor, "k", tw), tw),
                improper=bool(tor.get("improper", False)),
            )
        )
    constraints = []
    for n, con in enumerate(raw.get("constraints") or []):
        cw = f"{where} constraint {n}"
        length = con.get("r0", con.get("b0"))
        if length is None:
            raise MdffError(f"{cw}: missing 'r0'")
        constraints.append(
            {
                "i": _index(_require(con, "i", cw), n_atoms, cw),
                "j": _index(_require(con, "j", cw), n_atoms, cw),
                "r0": _float(length, cw),
            }
        )
    settles_raw = raw.get("settles") or []
    if isinstance(settles_raw, dict):
        settles_raw = [settles_raw]
    settles = []
    for n, settle in enumerate(settles_raw):
        sw = f"{where} settles {n}"
        oxygen = _index(settle.get("i", 0), n_atoms, sw)
        if oxygen + 2 >= n_atoms:
            raise MdffError(f"{sw}: the two hydrogens must follow the oxygen")
        settles.append(
            {
                "i": oxygen,
                "doh": _float(_require(settle, "doh", sw), sw),
                "dhh": _float(_require(settle, "dhh", sw), sw),
            }
        )
    try:
        nrexcl = int(raw.get("exclusions_nrexcl", 3))
    except (TypeError, ValueError) as exc:
        raise MdffError(f"{where}: exclusions_nrexcl is not an integer") from exc
    return MdffMoleculeType(
        name=name,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        exclusions_nrexcl=nrexcl,
        constraints=constraints,
        settles=settles,
    )


def parse_mdff(data: dict[str, Any], path: str | None = None) -> Mdff:
    """Validate a decoded JSON object and return an :class:`Mdff`."""
    if not isinstance(data, dict):
        raise MdffError("the top level must be a JSON object")
    if data.get("format") != FORMAT_NAME:
        raise MdffError(f"'format' must be '{FORMAT_NAME}'")
    if data.get("version") not in SUPPORTED_VERSIONS:
        raise MdffError(f"unsupported version {data.get('version')!r}; this plugin reads version 1")
    raw_types = data.get("molecule_types")
    if not isinstance(raw_types, list) or not raw_types:
        raise MdffError("'molecule_types' must be a non-empty list")
    types: dict[str, MdffMoleculeType] = {}
    for position, raw in enumerate(raw_types):
        mtype = _parse_molecule_type(raw, position)
        if mtype.name in types:
            raise MdffError(f"molecule type '{mtype.name}' is defined twice")
        types[mtype.name] = mtype

    system = []
    for n, entry in enumerate(data.get("system") or []):
        ew = f"system[{n}]"
        if not isinstance(entry, dict):
            raise MdffError(f"{ew} is not an object")
        name = str(_require(entry, "type", ew))
        if name not in types:
            raise MdffError(f"{ew}: unknown molecule type '{name}'")
        try:
            count = int(entry.get("count", 1))
            offset = int(_require(entry, "atom_offset", ew))
        except (TypeError, ValueError) as exc:
            raise MdffError(f"{ew}: count and atom_offset must be integers") from exc
        if count < 1 or offset < 0:
            raise MdffError(f"{ew}: count must be >= 1 and atom_offset >= 0")
        system.append(MdffSystemEntry(type=name, count=count, atom_offset=offset))

    try:
        rule = int(data.get("combination_rule", 2))
    except (TypeError, ValueError) as exc:
        raise MdffError("combination_rule must be 1, 2 or 3") from exc
    if rule not in (1, 2, 3):
        raise MdffError("combination_rule must be 1, 2 or 3")
    return Mdff(
        source=str(data.get("source", "")),
        combination_rule=rule,
        fudge_lj=_float(data.get("fudge_lj", 0.5), "fudge_lj"),
        fudge_qq=_float(data.get("fudge_qq", 0.8333), "fudge_qq"),
        molecule_types=types,
        system=system,
        path=path,
    )


def load_mdff(path: str) -> Mdff:
    """Read and validate a ``*.mdff.json`` file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise MdffError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MdffError(f"{path} is not valid JSON: {exc}") from exc
    return parse_mdff(data, path=path)


def resolve_system(mdff: Mdff, n_host_atoms: int) -> list[MdffSystemEntry]:
    """Return the system list, inferring it when the file has none.

    Without a ``system`` list a single molecule type whose size divides the
    host's atom count is taken as that many consecutive copies.
    """
    if mdff.system:
        return list(mdff.system)
    if len(mdff.molecule_types) != 1:
        raise MdffError(
            "the file has several molecule types but no 'system' list, so the "
            "order of molecules in the structure is unknown"
        )
    mtype = next(iter(mdff.molecule_types.values()))
    size = len(mtype.atoms)
    if n_host_atoms % size:
        raise MdffError(
            f"the structure has {n_host_atoms} atoms, which is not a whole number of "
            f"'{mtype.name}' molecules ({size} atoms each)"
        )
    return [MdffSystemEntry(type=mtype.name, count=n_host_atoms // size, atom_offset=0)]
