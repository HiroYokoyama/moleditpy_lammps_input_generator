"""Build a LAMMPS topology (types, charges, bonded terms) for the host molecule.

Three sources:

* ``build_from_mdff`` -- parameters from a ``moleditpy-mdff`` file, converted
  from GROMACS units (nm, kJ/mol) to LAMMPS ``real`` (Angstrom, kcal/mol);
* ``build_from_uff`` -- RDKit's UFF parameters as a built-in fallback, with
  Gasteiger, MMFF94 or zero charges;
* ``build_elements_only`` -- one type per element, for ReaxFF and other
  many-body potentials that need no bonded topology.

Energy conventions (LAMMPS docs): bond/angle ``harmonic`` E = K (x - x0)^2,
so K = k/2 of the GROMACS/UFF form k/2 (x - x0)^2; dihedral ``charmm``
E = K [1 + cos(n phi - d)] with integer d and weight 0.0; dihedral ``fourier``
the same summed over terms with real d; improper ``cvff`` E = K [1 + d cos(n phi)];
improper ``umbrella`` with omega0 = 0: E = K (1 - cos omega), first atom central.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .mdff import Mdff, MdffError, resolve_system

KJ_TO_KCAL = 1.0 / 4.184
NM_TO_A = 10.0
SYNTH_BOND_K = 450.0  # kcal/mol/A^2 for bonds added only to carry a constraint
SYNTH_ANGLE_K = 55.0  # kcal/mol/rad^2
_GROUP_16 = (8, 16, 34, 52, 84)


@dataclass
class AtomType:
    label: str
    element: str
    mass: float
    sigma: float = 0.0  # Angstrom
    epsilon: float = 0.0  # kcal/mol


@dataclass
class LammpsTopology:
    """Everything except coordinates and the box."""

    kind: str  # "molecular", "reaxff", "other"
    elements: list[str]
    atom_types: list[AtomType]
    atom_type: list[int]  # 0-based per atom
    charges: list[float]
    mol_ids: list[int]  # 1-based per atom
    bond_types: list[tuple[float, float]] = field(default_factory=list)  # K, r0
    bond_labels: list[str] = field(default_factory=list)
    bonds: list[tuple[int, int, int]] = field(default_factory=list)  # type, i, j (0-based)
    angle_types: list[tuple[float, float]] = field(default_factory=list)  # K, theta0
    angle_labels: list[str] = field(default_factory=list)
    angles: list[tuple[int, int, int, int]] = field(default_factory=list)
    dihedral_style: str | None = None  # "charmm" | "fourier"
    dihedral_types: list[tuple] = field(default_factory=list)
    dihedral_labels: list[str] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    improper_style: str | None = None  # "cvff" | "umbrella"
    improper_types: list[tuple] = field(default_factory=list)
    improper_labels: list[str] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    mixing: str = "arithmetic"
    special_lj: tuple[float, float, float] = (0.0, 0.0, 0.5)
    special_coul: tuple[float, float, float] = (0.0, 0.0, 0.8333)
    shake_bond_types: list[int] = field(default_factory=list)  # 0-based
    shake_angle_types: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def total_charge(self) -> float:
        return float(sum(self.charges))

    def type_elements(self) -> list[str]:
        return [t.element for t in self.atom_types]


class _Registry:
    """Assigns 0-based type ids to parameter tuples, remembering a label."""

    def __init__(self):
        self.ids: dict[tuple, int] = {}
        self.params: list[tuple] = []
        self.labels: list[str] = []
        self.flags: list[bool] = []

    def get(self, params: tuple, label: str, flag: bool = False) -> int:
        key = (tuple(_round(v) for v in params), flag)
        index = self.ids.get(key)
        if index is None:
            index = len(self.params)
            self.ids[key] = index
            self.params.append(tuple(params))
            self.labels.append(label)
            self.flags.append(flag)
        return index


def _round(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, tuple):
        return tuple(_round(v) for v in value)
    return value


def _is_integer(value: float, tol: float = 1e-6) -> bool:
    return abs(value - round(value)) < tol


def longest_bond_path(n_atoms: int, pairs) -> int:
    """Longest shortest-path (in bonds, capped at 3) between two atoms of a molecule."""
    neighbours: dict[int, set] = {i: set() for i in range(n_atoms)}
    for i, j in pairs:
        neighbours[i].add(j)
        neighbours[j].add(i)
    longest = 0
    for start in range(n_atoms):
        depth = {start: 0}
        frontier = [start]
        while frontier and longest < 3:
            nxt = []
            for atom in frontier:
                for other in neighbours[atom]:
                    if other not in depth:
                        depth[other] = depth[atom] + 1
                        nxt.append(other)
            frontier = nxt
        longest = max(longest, max(depth.values()))
        if longest >= 3:
            return 3
    return longest


def special_bonds_for(usage, fudge_lj: float, fudge_qq: float, warnings: list[str]):
    """special_bonds weights from per-molecule-type (name, nrexcl, longest path, atoms).

    LAMMPS has one global special_bonds setting, so each 1-n level is decided
    only by the molecule types that actually contain 1-n pairs: a single ion's
    nrexcl must not switch off the 1-4 scaling of an organic molecule.
    """
    lj = [0.0, 0.0, fudge_lj]
    coul = [0.0, 0.0, fudge_qq]
    for level in (1, 2, 3):
        relevant = [(n, atoms, name) for name, n, path, atoms in usage if path >= level]
        if not relevant:
            continue
        excluded = {n >= level for n, _, _ in relevant}
        if len(excluded) > 1:
            winner = max(relevant, key=lambda item: item[1])
            warnings.append(
                f"Molecule types disagree on excluding 1-{level + 1} pairs; LAMMPS "
                f"special_bonds is global, so '{winner[2]}' (most atoms) decides."
            )
            is_excluded = winner[0] >= level
        else:
            is_excluded = excluded.pop()
        if not is_excluded:
            lj[level - 1] = 1.0
            coul[level - 1] = 1.0
    if any(n > 3 for _, n, path, _ in usage if path >= 3):
        warnings.append(
            "exclusions_nrexcl > 3 cannot be expressed with special_bonds; 1-4 pairs are "
            "scaled as for nrexcl = 3."
        )
    return tuple(lj), tuple(coul)


def _molecule_fragments(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[int]:
    """1-based molecule id per atom from connectivity (union-find)."""
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    ids: dict[int, int] = {}
    result = []
    for atom in range(n_atoms):
        root = find(atom)
        if root not in ids:
            ids[root] = len(ids) + 1
        result.append(ids[root])
    return result


def _finish_dihedrals(
    topo: LammpsTopology,
    terms: dict[tuple[int, int, int, int], list[tuple[float, float, float]]],
    style: str,
    labels: dict[tuple[int, int, int, int], str],
) -> None:
    """Turn per-quartet (K, n, d) term lists into LAMMPS dihedral types."""
    terms = {q: [t for t in ts if abs(t[0]) > 1e-12] for q, ts in terms.items()}
    terms = {q: ts for q, ts in terms.items() if ts}
    if not terms:
        return
    for quartet, ts in terms.items():
        for _, n, _ in ts:
            if not _is_integer(n) or n < 0:
                raise MdffError(
                    f"torsion {tuple(i + 1 for i in quartet)} has periodicity {n}; "
                    "LAMMPS needs a non-negative integer"
                )
    if style == "auto":
        integral = all(_is_integer(d) for ts in terms.values() for _, _, d in ts)
        style = "charmm" if integral else "fourier"
    registry = _Registry()
    if style == "charmm":
        for quartet, ts in terms.items():
            for k, n, d in ts:
                if not _is_integer(d):
                    raise MdffError(
                        f"torsion phase {d} is not a whole degree; choose the fourier "
                        "dihedral style"
                    )
                type_id = registry.get((float(k), round(n), round(d)), labels.get(quartet, ""))
                topo.dihedrals.append((type_id, *quartet))
    elif style == "fourier":
        for quartet, ts in terms.items():
            params = tuple(sorted((float(k), round(n), float(d)) for k, n, d in ts))
            type_id = registry.get((params,), labels.get(quartet, ""))
            topo.dihedrals.append((type_id, *quartet))
        registry.params = [p[0] for p in registry.params]
    else:
        raise ValueError(f"unknown dihedral style {style!r}")
    topo.dihedral_style = style
    topo.dihedral_types = registry.params
    topo.dihedral_labels = registry.labels


def _finish_bonds_angles(topo: LammpsTopology, bond_reg: _Registry, angle_reg: _Registry):
    topo.bond_types = list(bond_reg.params)
    topo.bond_labels = list(bond_reg.labels)
    topo.shake_bond_types = [i for i, flag in enumerate(bond_reg.flags) if flag]
    topo.angle_types = list(angle_reg.params)
    topo.angle_labels = list(angle_reg.labels)
    topo.shake_angle_types = [i for i, flag in enumerate(angle_reg.flags) if flag]


# ---------------------------------------------------------------------------
# moleditpy-mdff
# ---------------------------------------------------------------------------


def build_from_mdff(
    mdff: Mdff,
    host_elements: Sequence[str],
    dihedral_style: str = "auto",
    constrain_h: bool = False,
) -> LammpsTopology:
    n_host = len(host_elements)
    entries = resolve_system(mdff, n_host)
    owner = [None] * n_host
    for entry in entries:
        mtype = mdff.molecule_types[entry.type]
        size = len(mtype.atoms)
        end = entry.atom_offset + entry.count * size
        if end > n_host:
            raise MdffError(
                f"system entry '{entry.type}' x{entry.count} at offset {entry.atom_offset} "
                f"needs atoms up to {end}, but the structure has {n_host}"
            )
        for copy in range(entry.count):
            base = entry.atom_offset + copy * size
            for local in range(size):
                if owner[base + local] is not None:
                    raise MdffError(f"atom {base + local + 1} is covered by two system entries")
                owner[base + local] = (entry.type, copy, local, base)
    missing = [i + 1 for i, o in enumerate(owner) if o is None]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        raise MdffError(
            f"{len(missing)} atom(s) of the structure are not described by the force "
            f"field file (first: {preview})"
        )

    warnings: list[str] = []
    mismatched = []
    for index, (tname, _, local, _) in enumerate(owner):  # type: ignore[misc]
        expected = mdff.molecule_types[tname].atoms[local].element
        if expected not in ("", "X") and expected != host_elements[index]:
            mismatched.append((index, host_elements[index], expected))
    if mismatched:
        index, host, expected = mismatched[0]
        raise MdffError(
            f"{len(mismatched)} atom(s) do not match the force field file's elements "
            f"(atom {index + 1} is {host}, the file expects {expected}); the atom order "
            "of the structure and of the file differ"
        )

    # atom types by mdff type name; the same name with other parameters is split
    type_index: dict[tuple, int] = {}
    atom_types: list[AtomType] = []
    by_name: dict[str, list[int]] = {}
    for mtype in mdff.molecule_types.values():
        for atom in mtype.atoms:
            key = (atom.type, round(atom.mass, 4), round(atom.sigma, 7), round(atom.epsilon, 7))
            if key in type_index:
                continue
            label = atom.type
            if atom.type in by_name:
                label = f"{atom.type}_{len(by_name[atom.type]) + 1}"
                warnings.append(
                    f"Atom type '{atom.type}' appears with different parameters; split into "
                    f"'{label}'."
                )
            type_index[key] = len(atom_types)
            by_name.setdefault(atom.type, []).append(len(atom_types))
            atom_types.append(
                AtomType(
                    label=label,
                    element=atom.element if atom.element not in ("", "X") else "X",
                    mass=atom.mass,
                    sigma=atom.sigma * NM_TO_A,
                    epsilon=atom.epsilon * KJ_TO_KCAL,
                )
            )

    topo = LammpsTopology(
        kind="molecular",
        elements=list(host_elements),
        atom_types=atom_types,
        atom_type=[0] * n_host,
        charges=[0.0] * n_host,
        mol_ids=[0] * n_host,
        mixing="arithmetic" if mdff.combination_rule == 2 else "geometric",
        description=f"moleditpy-mdff ({mdff.source or 'unknown source'})",
    )
    bond_reg, angle_reg = _Registry(), _Registry()
    improper_reg = _Registry()
    torsion_terms: dict[tuple[int, int, int, int], list[tuple[float, float, float]]] = {}
    torsion_labels: dict[tuple[int, int, int, int], str] = {}
    odd_impropers = 0

    mol_id = 0
    for entry in entries:
        mtype = mdff.molecule_types[entry.type]
        size = len(mtype.atoms)
        tlabels = []
        for atom in mtype.atoms:
            key = (atom.type, round(atom.mass, 4), round(atom.sigma, 7), round(atom.epsilon, 7))
            tlabels.append(atom_types[type_index[key]].label)

        # constrained pairs: explicit constraints, settles, optionally X-H bonds
        constrained: dict[tuple[int, int], float] = {}
        for con in mtype.constraints:
            constrained[tuple(sorted((con["i"], con["j"])))] = con["r0"] * NM_TO_A
        extra_bonds: list[tuple[int, int, float]] = []
        rigid_angles: dict[tuple[int, int, int], float] = {}
        bonded_pairs = {tuple(sorted((b["i"], b["j"]))) for b in mtype.bonds}
        for settle in mtype.settles:
            o, h1, h2 = settle["i"], settle["i"] + 1, settle["i"] + 2
            doh, dhh = settle["doh"] * NM_TO_A, settle["dhh"] * NM_TO_A
            for h in (h1, h2):
                constrained[tuple(sorted((o, h)))] = doh
            theta = math.degrees(2.0 * math.asin(min(1.0, dhh / (2.0 * doh))))
            rigid_angles[(h1, o, h2)] = theta
        for pair, r0 in constrained.items():
            if pair not in bonded_pairs:
                extra_bonds.append((pair[0], pair[1], r0))
        h_bond = set()
        if constrain_h:
            for bond in mtype.bonds:
                pair = tuple(sorted((bond["i"], bond["j"])))
                if "H" in (mtype.atoms[pair[0]].element, mtype.atoms[pair[1]].element):
                    h_bond.add(pair)

        for copy in range(entry.count):
            mol_id += 1
            base = entry.atom_offset + copy * size
            for local, atom in enumerate(mtype.atoms):
                key = (atom.type, round(atom.mass, 4), round(atom.sigma, 7), round(atom.epsilon, 7))
                topo.atom_type[base + local] = type_index[key]
                topo.charges[base + local] = atom.charge
                topo.mol_ids[base + local] = mol_id
            for bond in mtype.bonds:
                pair = tuple(sorted((bond["i"], bond["j"])))
                flag = pair in constrained or pair in h_bond
                r0 = constrained.get(pair, bond["r0"] * NM_TO_A)
                tid = bond_reg.get(
                    (bond["k"] / 2.0 * KJ_TO_KCAL / NM_TO_A**2, r0),
                    f"{tlabels[pair[0]]}-{tlabels[pair[1]]}",
                    flag,
                )
                topo.bonds.append((tid, base + bond["i"], base + bond["j"]))
            for i, j, r0 in extra_bonds:
                tid = bond_reg.get(
                    (SYNTH_BOND_K, r0), f"{tlabels[i]}-{tlabels[j]} (constraint)", True
                )
                topo.bonds.append((tid, base + i, base + j))
            angle_keys = set()
            for angle in mtype.angles:
                trip = (angle["i"], angle["j"], angle["k_atom"])
                rigid = trip in rigid_angles or trip[::-1] in rigid_angles
                angle_keys.add(trip)
                angle_keys.add(trip[::-1])
                tid = angle_reg.get(
                    (
                        angle["k"] / 2.0 * KJ_TO_KCAL,
                        rigid_angles.get(trip, rigid_angles.get(trip[::-1], angle["theta0"])),
                    ),
                    f"{tlabels[trip[0]]}-{tlabels[trip[1]]}-{tlabels[trip[2]]}",
                    rigid,
                )
                topo.angles.append((tid, base + trip[0], base + trip[1], base + trip[2]))
            for trip, theta in rigid_angles.items():
                if trip in angle_keys:
                    continue
                tid = angle_reg.get(
                    (SYNTH_ANGLE_K, theta),
                    f"{tlabels[trip[0]]}-{tlabels[trip[1]]}-{tlabels[trip[2]]} (rigid)",
                    True,
                )
                topo.angles.append((tid, base + trip[0], base + trip[1], base + trip[2]))
            for tor in mtype.torsions:
                quartet = (base + tor.i, base + tor.j, base + tor.k_atom, base + tor.l)
                k = tor.k * KJ_TO_KCAL
                label = "-".join(tlabels[x] for x in (tor.i, tor.j, tor.k_atom, tor.l))
                phase = tor.phase % 360.0
                if tor.improper:
                    n = tor.periodicity
                    cvff_phase = min(abs(phase), abs(phase - 180.0), abs(phase - 360.0)) < 1e-6
                    if cvff_phase and _is_integer(n) and round(n) in (0, 1, 2, 3, 4, 6):
                        sign = 1 if (abs(phase) < 1e-6 or abs(phase - 360.0) < 1e-6) else -1
                        tid = improper_reg.get((k, sign, round(n)), label)
                        topo.impropers.append((tid, *quartet))
                        continue
                    odd_impropers += 1
                torsion_terms.setdefault(quartet, []).append((k, tor.periodicity, tor.phase))
                torsion_labels.setdefault(quartet, label)

    if odd_impropers:
        warnings.append(
            f"{odd_impropers} improper torsion(s) have a phase other than 0/180 degrees "
            "and were written as dihedrals, which is the same energy expression."
        )
    _finish_bonds_angles(topo, bond_reg, angle_reg)
    _finish_dihedrals(topo, torsion_terms, dihedral_style, torsion_labels)
    if improper_reg.params:
        topo.improper_style = "cvff"
        topo.improper_types = improper_reg.params
        topo.improper_labels = improper_reg.labels

    usage = []
    for entry in entries:
        mtype = mdff.molecule_types[entry.type]
        pairs = [(b["i"], b["j"]) for b in mtype.bonds]
        pairs += [(c["i"], c["j"]) for c in mtype.constraints]
        for settle in mtype.settles:
            pairs += [(settle["i"], settle["i"] + 1), (settle["i"], settle["i"] + 2)]
        path = longest_bond_path(len(mtype.atoms), pairs)
        usage.append((mtype.name, mtype.exclusions_nrexcl, path, entry.count * len(mtype.atoms)))
    topo.special_lj, topo.special_coul = special_bonds_for(
        usage, mdff.fudge_lj, mdff.fudge_qq, warnings
    )
    if mdff.combination_rule == 1:
        warnings.append(
            "combination_rule 1 (geometric C6/C12) is written as geometric mixing of "
            "sigma and epsilon, which is equivalent."
        )
    topo.warnings.extend(warnings)
    return topo


# ---------------------------------------------------------------------------
# RDKit UFF fallback
# ---------------------------------------------------------------------------


def uff_label(atom) -> str:
    """A UFF-like type label (C_3, C_R, O_2, H_ ...) for display."""
    from rdkit import Chem

    symbol = atom.GetSymbol()
    base = symbol if len(symbol) == 2 else symbol + "_"
    if symbol == "H":
        return "H_"
    if atom.GetIsAromatic():
        return base + "R"
    hyb = atom.GetHybridization()
    code = {
        Chem.HybridizationType.SP: "1",
        Chem.HybridizationType.SP2: "2",
        Chem.HybridizationType.SP3: "3",
        Chem.HybridizationType.SP3D: "5",
        Chem.HybridizationType.SP3D2: "6",
    }.get(hyb, "")
    return base + code


def _uff_torsion_form(mol, i, j, k, l):
    """(order n, cosTerm) for one UFF torsion, following RDKit's TorsionAngle rules."""
    from rdkit import Chem

    sp2, sp3 = Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
    aj, ak = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
    hj, hk = aj.GetHybridization(), ak.GetHybridization()
    zj, zk = aj.GetAtomicNum(), ak.GetAtomicNum()
    bond_order = mol.GetBondBetweenAtoms(j, k).GetBondTypeAsDouble()
    end_sp2 = sp2 in (
        mol.GetAtomWithIdx(i).GetHybridization(),
        mol.GetAtomWithIdx(l).GetHybridization(),
    )
    if hj == sp3 and hk == sp3:
        if bond_order == 1.0 and zj in _GROUP_16 and zk in _GROUP_16:
            return 2, -1.0
        return 3, -1.0
    if hj == sp2 and hk == sp2:
        return 2, 1.0
    if bond_order == 1.0:
        if (hj == sp3 and zj in _GROUP_16 and zk not in _GROUP_16) or (
            hk == sp3 and zk in _GROUP_16 and zj not in _GROUP_16
        ):
            return 2, -1.0
        if end_sp2:
            return 3, -1.0
    return 6, 1.0


def uff_charges(mol, method: str, warnings: list[str]) -> list[float]:
    n = mol.GetNumAtoms()
    if method == "zero":
        return [0.0] * n
    from rdkit.Chem import AllChem

    if method == "mmff":
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            raise ValueError(
                "MMFF94 has no parameters for this molecule; choose Gasteiger or zero charges"
            )
        return [float(props.GetMMFFPartialCharge(i)) for i in range(n)]
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    bad = 0
    for atom in mol.GetAtoms():
        try:
            q = float(atom.GetDoubleProp("_GasteigerCharge"))
        except (KeyError, RuntimeError, ValueError):
            q = float("nan")
        if not math.isfinite(q):
            q = 0.0
            bad += 1
        charges.append(q)
    if bad:
        warnings.append(f"Gasteiger gave no charge for {bad} atom(s); they were set to zero.")
    return charges


def build_from_uff(
    mol, charge_method: str = "gasteiger", constrain_h: bool = False
) -> LammpsTopology:
    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers as ffh

    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception as exc:
        raise ValueError(f"RDKit could not sanitize the molecule for UFF: {exc}") from exc
    n = work.GetNumAtoms()
    warnings: list[str] = []
    if any(a.GetNumImplicitHs() for a in work.GetAtoms()):
        warnings.append(
            "Some atoms carry implicit hydrogens, which are not written; add explicit "
            "hydrogens before an MD run."
        )

    # atom types keyed by UFF label and vdW parameters
    atom_types: list[AtomType] = []
    type_ids: dict[tuple, int] = {}
    atom_type: list[int] = []
    labels = []
    missing = []
    for atom in work.GetAtoms():
        idx = atom.GetIdx()
        vdw = ffh.GetUFFVdWParams(work, idx, idx)
        if vdw is None:
            missing.append(f"{atom.GetSymbol()}{idx + 1}")
            vdw = (0.0, 0.0)
        x_ij, d_ij = vdw
        label = uff_label(atom)
        key = (label, round(x_ij, 5), round(d_ij, 6), round(atom.GetMass(), 4))
        if key not in type_ids:
            shown = label
            if any(t.label == label for t in atom_types):
                shown = f"{label}_{sum(1 for t in atom_types if t.label.startswith(label)) + 1}"
            type_ids[key] = len(atom_types)
            atom_types.append(
                AtomType(
                    label=shown,
                    element=atom.GetSymbol(),
                    mass=float(atom.GetMass()),
                    sigma=x_ij / 2.0 ** (1.0 / 6.0),
                    epsilon=d_ij,
                )
            )
        atom_type.append(type_ids[key])
        labels.append(atom_types[type_ids[key]].label)
    if missing:
        raise ValueError(
            "UFF has no parameters for atom(s) "
            + ", ".join(missing[:10])
            + "; use a moleditpy-mdff file from the Force Field Builder"
        )

    elements = [a.GetSymbol() for a in work.GetAtoms()]
    topo = LammpsTopology(
        kind="molecular",
        elements=elements,
        atom_types=atom_types,
        atom_type=atom_type,
        charges=uff_charges(work, charge_method, warnings),
        mol_ids=[],
        mixing="geometric",
        special_lj=(0.0, 0.0, 1.0),
        special_coul=(0.0, 0.0, 1.0),
        description=f"RDKit UFF, {charge_method} charges",
    )
    bond_reg, angle_reg, improper_reg = _Registry(), _Registry(), _Registry()
    bond_pairs = []
    for bond in work.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        params = ffh.GetUFFBondStretchParams(work, i, j)
        if params is None:
            warnings.append(f"No UFF bond parameters for {labels[i]}{i + 1}-{labels[j]}{j + 1}.")
            continue
        kb, r0 = params
        flag = constrain_h and "H" in (elements[i], elements[j])
        tid = bond_reg.get((kb / 2.0, r0), f"{labels[i]}-{labels[j]}", flag)
        topo.bonds.append((tid, i, j))
        bond_pairs.append((i, j))
    topo.mol_ids = _molecule_fragments(n, bond_pairs)

    for atom in work.GetAtoms():
        j = atom.GetIdx()
        nbrs = [a.GetIdx() for a in atom.GetNeighbors()]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                params = ffh.GetUFFAngleBendParams(work, i, j, k)
                if params is None:
                    continue
                ka, theta0 = params
                tid = angle_reg.get((ka / 2.0, theta0), f"{labels[i]}-{labels[j]}-{labels[k]}")
                topo.angles.append((tid, i, j, k))

    # torsions over bonds matching RDKit's default torsion SMARTS
    query = Chem.MolFromSmarts("[!$(*#*)&!D1]~[!$(*#*)&!D1]")
    sp2, sp3 = Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
    terms: dict[tuple[int, int, int, int], list[tuple[float, float, float]]] = {}
    tor_labels: dict[tuple[int, int, int, int], str] = {}
    seen_bonds = set()
    for j, k in work.GetSubstructMatches(query, uniquify=True, maxMatches=10**9):
        if (j, k) in seen_bonds or (k, j) in seen_bonds:
            continue
        seen_bonds.add((j, k))
        aj, ak = work.GetAtomWithIdx(j), work.GetAtomWithIdx(k)
        if aj.GetHybridization() not in (sp2, sp3) or ak.GetHybridization() not in (sp2, sp3):
            continue
        quartets = []
        for i in (a.GetIdx() for a in aj.GetNeighbors()):
            if i == k:
                continue
            for l in (a.GetIdx() for a in ak.GetNeighbors()):
                if l in (j, i):
                    continue
                quartets.append((i, j, k, l))
        for quartet in quartets:
            v = ffh.GetUFFTorsionParams(work, *quartet)
            if v is None or abs(v) < 1e-12:
                continue
            order, cos_term = _uff_torsion_form(work, *quartet)
            k_lmp = v / 2.0 / len(quartets)
            phase = 0.0 if cos_term < 0 else 180.0
            terms[quartet] = [(k_lmp, float(order), phase)]
            tor_labels[quartet] = "-".join(labels[x] for x in quartet)
    _finish_dihedrals(topo, terms, "charmm", tor_labels)

    # inversions (UFF out-of-plane) for sp2 C/N/O with three neighbours
    skipped_group15 = 0
    for atom in work.GetAtoms():
        z = atom.GetAtomicNum()
        if atom.GetDegree() != 3 or z not in (6, 7, 8, 15, 33, 51, 83):
            continue
        if z in (6, 7, 8) and atom.GetHybridization() != sp2:
            continue
        if z not in (6, 7, 8):
            skipped_group15 += 1
            continue
        centre = atom.GetIdx()
        nb = [a.GetIdx() for a in atom.GetNeighbors()]
        for p0, p2, p3 in ((nb[0], nb[1], nb[2]), (nb[0], nb[2], nb[1]), (nb[1], nb[2], nb[0])):
            kinv = ffh.GetUFFInversionParams(work, p0, centre, p2, p3)
            if kinv is None or abs(kinv) < 1e-12:
                continue
            tid = improper_reg.get((float(kinv), 0.0), f"{labels[centre]} out-of-plane")
            topo.impropers.append((tid, centre, p0, p2, p3))
    if skipped_group15:
        warnings.append(
            f"{skipped_group15} pyramidal group-15 centre(s) (P, As, Sb, Bi) have no "
            "LAMMPS umbrella equivalent of the UFF inversion term and were left without one."
        )
    if improper_reg.params:
        topo.improper_style = "umbrella"
        topo.improper_types = improper_reg.params
        topo.improper_labels = improper_reg.labels
    _finish_bonds_angles(topo, bond_reg, angle_reg)
    warnings.append(
        "UFF angle terms are written as harmonic angles with the same curvature at "
        "theta0 (UFF uses a cosine expansion); fine near equilibrium, a fallback only."
    )
    topo.warnings.extend(warnings)
    return topo


# ---------------------------------------------------------------------------
# ReaxFF / other potentials
# ---------------------------------------------------------------------------


def build_elements_only(
    elements: Sequence[str], masses: Sequence[float], kind: str
) -> LammpsTopology:
    """One type per element, in order of first appearance."""
    order: list[str] = []
    for element in elements:
        if element not in order:
            order.append(element)
    mass_of = {}
    for element, mass in zip(elements, masses):
        mass_of.setdefault(element, float(mass))
    types = [AtomType(label=e, element=e, mass=mass_of[e]) for e in order]
    index = {e: i for i, e in enumerate(order)}
    return LammpsTopology(
        kind=kind,
        elements=list(elements),
        atom_types=types,
        atom_type=[index[e] for e in elements],
        charges=[0.0] * len(elements),
        mol_ids=[1] * len(elements),
        description="ReaxFF" if kind == "reaxff" else "user-supplied potential",
    )


def shake_cluster_problems(topo: LammpsTopology) -> list[str]:
    """Atoms whose constrained bonds would make a SHAKE cluster LAMMPS refuses."""
    if not topo.shake_bond_types:
        return []
    shake = set(topo.shake_bond_types)
    partners: dict[int, set] = {}
    for tid, i, j in topo.bonds:
        if tid in shake:
            partners.setdefault(i, set()).add(j)
            partners.setdefault(j, set()).add(i)
    problems = []
    for atom, others in partners.items():
        if len(others) > 3:
            problems.append(
                f"atom {atom + 1} ({topo.elements[atom]}) has {len(others)} constrained bonds; "
                "fix shake allows at most 3 per central atom"
            )
        elif len(others) > 1 and any(len(partners[o]) > 1 for o in others):
            problems.append(f"atom {atom + 1} ({topo.elements[atom]}) links two SHAKE clusters")
    return problems
