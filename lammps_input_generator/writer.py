"""LAMMPS data file and input script text, plus internal consistency checks.

Pure functions of a :class:`~.topology.LammpsTopology`, a
:class:`~.cell.LammpsBox` and the settings dictionary; no Qt here.
"""

from __future__ import annotations

import re

from .topology import LammpsTopology, shake_cluster_problems

GENERATOR = "MoleditPy LAMMPS Input Generator"

FF_SOURCES = ("uff", "mdff", "reaxff", "other")
ENSEMBLES = ("minimize", "nve", "nvt", "npt")
THERMOSTATS = ("nose-hoover", "langevin", "berendsen", "csvr")
BAROSTATS = ("nose-hoover", "berendsen")
COUPLINGS = ("iso", "aniso", "tri")
DUMP_FORMATS = ("xyz", "custom", "none")
KSPACE_STYLES = ("pppm", "ewald", "none")


def default_settings() -> dict:
    return {
        "job_name": "system",
        "ff_source": "uff",
        "mdff_path": "",
        "charge_method": "gasteiger",
        "dihedral_style": "auto",
        "constrain_h": False,
        "cell_source": "auto",
        "box_mode": "padding",
        "padding": 10.0,
        "box_lengths": [40.0, 40.0, 40.0],
        "box_angles": [90.0, 90.0, 90.0],
        "user_periodic": True,
        "pair_cutoff": 10.0,
        "kspace": "pppm",
        "kspace_accuracy": 1.0e-4,
        "ensemble": "nvt",
        "minimize_first": True,
        "min_etol": 1.0e-4,
        "min_ftol": 1.0e-6,
        "min_maxiter": 1000,
        "min_maxeval": 10000,
        "thermostat": "nose-hoover",
        "barostat": "nose-hoover",
        "coupling": "iso",
        "temperature": 300.0,
        "pressure": 1.0,
        "tdamp": 100.0,
        "pdamp": 1000.0,
        "timestep": 1.0,
        "steps": 100000,
        "seed": 12345,
        "thermo_every": 1000,
        "dump_format": "xyz",
        "dump_every": 1000,
        "restart_every": 0,
        "write_final": True,
        "neighbor_skin": 2.0,
        "neigh_every": 1,
        "neigh_delay": 0,
        "neigh_check": True,
        "reax_ffield": "ffield.reax",
        "reax_qeq": True,
        "other_pair_style": "tersoff",
        "other_potential_file": "potential.tersoff",
        "other_units": "metal",
        "other_atom_style": "atomic",
    }


def safe_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._")
    return cleaned or "system"


def file_names(settings: dict) -> dict[str, str]:
    job = safe_job_name(settings.get("job_name", "system"))
    return {
        "input": f"in.{job}",
        "data": f"{job}.data",
        "dump_xyz": f"{job}.xyz",
        "dump_custom": f"{job}.lammpstrj",
        "restart": f"{job}.restart",
        "final_data": f"{job}_final.data",
        "final_restart": f"{job}_final.restart",
        "log": f"log.{job}",
    }


def units_of(topo: LammpsTopology, settings: dict) -> str:
    if topo.kind == "other":
        return "real" if settings.get("other_units") == "real" else "metal"
    return "real"


def atom_style_of(topo: LammpsTopology, settings: dict) -> str:
    if topo.kind == "molecular":
        return "full"
    if topo.kind == "reaxff":
        return "charge"
    return "charge" if settings.get("other_atom_style") == "charge" else "atomic"


def _time(value_fs: float, units: str) -> float:
    """Time from fs (dialog unit) into the LAMMPS unit system."""
    return value_fs / 1000.0 if units == "metal" else value_fs


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _coord(value: float) -> str:
    return f"{value:.6f}"


def pair_style_of(topo: LammpsTopology, box, settings: dict) -> tuple[str, str | None]:
    """(pair_style line, kspace_style line or None) for a molecular topology."""
    cutoff = float(settings.get("pair_cutoff", 10.0))
    charged = any(abs(q) > 1e-12 for q in topo.charges)
    periodic = all(box.pbc)
    kspace = settings.get("kspace", "pppm")
    if not charged:
        return f"pair_style      lj/cut {_fmt(cutoff)}", None
    if periodic and kspace in ("pppm", "ewald"):
        accuracy = float(settings.get("kspace_accuracy", 1.0e-4))
        return (
            f"pair_style      lj/cut/coul/long {_fmt(cutoff)}",
            f"kspace_style    {kspace} {accuracy:g}",
        )
    return f"pair_style      lj/cut/coul/cut {_fmt(cutoff)}", None


# ---------------------------------------------------------------------------
# data file
# ---------------------------------------------------------------------------


def _type_comment(label: str) -> str:
    return f"  # {label}" if label else ""


def build_data_file(topo: LammpsTopology, box, settings: dict, version: str = "") -> str:
    style = atom_style_of(topo, settings)
    lines: list[str] = []
    title = f"LAMMPS data file written by {GENERATOR} {version}".strip()
    lines.append(f"{title} ({topo.description}, atom_style {style})")
    lines.append("")
    lines.append(f"{topo.n_atoms} atoms")
    lines.append(f"{len(topo.atom_types)} atom types")
    if style == "full":
        for count, ntypes, name in (
            (len(topo.bonds), len(topo.bond_types), "bond"),
            (len(topo.angles), len(topo.angle_types), "angle"),
            (len(topo.dihedrals), len(topo.dihedral_types), "dihedral"),
            (len(topo.impropers), len(topo.improper_types), "improper"),
        ):
            if count:
                lines.append(f"{count} {name}s")
                lines.append(f"{ntypes} {name} types")
    lines.append("")
    hi = box.lo + box.lengths
    for dim, axis in enumerate("xyz"):
        lines.append(f"{_coord(box.lo[dim])} {_coord(hi[dim])} {axis}lo {axis}hi")
    if box.triclinic:
        xy, xz, yz = box.tilts
        lines.append(f"{_coord(xy)} {_coord(xz)} {_coord(yz)} xy xz yz")
    lines.append("")

    lines.append("Masses")
    lines.append("")
    for n, atype in enumerate(topo.atom_types, 1):
        lines.append(f"{n} {atype.mass:.4f}{_type_comment(atype.label)}")
    lines.append("")

    if style == "full":
        pair_name = pair_style_of(topo, box, settings)[0].split()[1]
        lines.append(f"Pair Coeffs # {pair_name}")
        lines.append("")
        for n, atype in enumerate(topo.atom_types, 1):
            lines.append(f"{n} {atype.epsilon:.6f} {atype.sigma:.6f}{_type_comment(atype.label)}")
        lines.append("")
        if topo.bonds:
            lines.append("Bond Coeffs # harmonic")
            lines.append("")
            for n, (k, r0) in enumerate(topo.bond_types, 1):
                lines.append(f"{n} {k:.6f} {r0:.6f}{_type_comment(topo.bond_labels[n - 1])}")
            lines.append("")
        if topo.angles:
            lines.append("Angle Coeffs # harmonic")
            lines.append("")
            for n, (k, theta) in enumerate(topo.angle_types, 1):
                lines.append(f"{n} {k:.6f} {theta:.6f}{_type_comment(topo.angle_labels[n - 1])}")
            lines.append("")
        if topo.dihedrals:
            lines.append(f"Dihedral Coeffs # {topo.dihedral_style}")
            lines.append("")
            for n, params in enumerate(topo.dihedral_types, 1):
                comment = _type_comment(topo.dihedral_labels[n - 1])
                if topo.dihedral_style == "charmm":
                    k, mult, d = params
                    lines.append(f"{n} {k:.6f} {mult:d} {d:d} 0.0{comment}")
                else:
                    terms = " ".join(f"{k:.6f} {mult:d} {d:.4f}" for k, mult, d in params)
                    lines.append(f"{n} {len(params)} {terms}{comment}")
            lines.append("")
        if topo.impropers:
            lines.append(f"Improper Coeffs # {topo.improper_style}")
            lines.append("")
            for n, params in enumerate(topo.improper_types, 1):
                comment = _type_comment(topo.improper_labels[n - 1])
                if topo.improper_style == "cvff":
                    k, sign, mult = params
                    lines.append(f"{n} {k:.6f} {sign:d} {mult:d}{comment}")
                else:
                    k, omega0 = params
                    lines.append(f"{n} {k:.6f} {omega0:.4f}{comment}")
            lines.append("")

    lines.append(f"Atoms # {style}")
    lines.append("")
    pos, img = box.positions, box.images
    for n in range(topo.n_atoms):
        x, y, z = (_coord(v) for v in pos[n])
        ix, iy, iz = (int(v) for v in img[n])
        t = topo.atom_type[n] + 1
        if style == "full":
            q = topo.charges[n]
            lines.append(f"{n + 1} {topo.mol_ids[n]} {t} {q:.6f} {x} {y} {z} {ix} {iy} {iz}")
        elif style == "charge":
            q = topo.charges[n]
            lines.append(f"{n + 1} {t} {q:.6f} {x} {y} {z} {ix} {iy} {iz}")
        else:
            lines.append(f"{n + 1} {t} {x} {y} {z} {ix} {iy} {iz}")
    lines.append("")

    if style == "full":
        for title_name, items in (
            ("Bonds", topo.bonds),
            ("Angles", topo.angles),
            ("Dihedrals", topo.dihedrals),
            ("Impropers", topo.impropers),
        ):
            if not items:
                continue
            lines.append(title_name)
            lines.append("")
            for n, item in enumerate(items, 1):
                tid, *atoms = item
                lines.append(f"{n} {tid + 1} " + " ".join(str(a + 1) for a in atoms))
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# input script
# ---------------------------------------------------------------------------


def _shake_fix(topo: LammpsTopology) -> str | None:
    if topo.kind != "molecular" or not topo.shake_bond_types:
        return None
    parts = ["fix             constrain all shake 0.0001 20 0 b"]
    parts.extend(str(t + 1) for t in topo.shake_bond_types)
    if topo.shake_angle_types:
        parts.append("a")
        parts.extend(str(t + 1) for t in topo.shake_angle_types)
    return " ".join(parts)


def _ensemble_fixes(topo: LammpsTopology, box, settings: dict, units: str) -> list[str]:
    ensemble = settings.get("ensemble", "nvt")
    temp = float(settings.get("temperature", 300.0))
    press = float(settings.get("pressure", 1.0))
    tdamp = _fmt(_time(float(settings.get("tdamp", 100.0)), units))
    pdamp = _fmt(_time(float(settings.get("pdamp", 1000.0)), units))
    seed = int(settings.get("seed", 12345))
    thermostat = settings.get("thermostat", "nose-hoover")
    barostat = settings.get("barostat", "nose-hoover")
    coupling = settings.get("coupling", "iso")
    if coupling == "tri" and not box.triclinic:
        coupling = "aniso"
    t, p = _fmt(temp), _fmt(press)
    lines: list[str] = []

    def thermostat_fix() -> list[str]:
        if thermostat == "langevin":
            return [f"fix             thermostat all langevin {t} {t} {tdamp} {seed} zero yes"]
        if thermostat == "berendsen":
            return [f"fix             thermostat all temp/berendsen {t} {t} {tdamp}"]
        if thermostat == "csvr":
            return [f"fix             thermostat all temp/csvr {t} {t} {tdamp} {seed}"]
        return []

    if ensemble == "nve":
        lines.append("fix             integrate all nve")
    elif ensemble == "nvt":
        if thermostat == "nose-hoover":
            lines.append(f"fix             integrate all nvt temp {t} {t} {tdamp}")
        else:
            lines.append("fix             integrate all nve")
            lines.extend(thermostat_fix())
    elif ensemble == "npt":
        if barostat == "nose-hoover":
            if thermostat == "nose-hoover":
                lines.append(
                    f"fix             integrate all npt temp {t} {t} {tdamp} {coupling} {p} {p} {pdamp}"
                )
            else:
                lines.append(f"fix             integrate all nph {coupling} {p} {p} {pdamp}")
                lines.extend(thermostat_fix())
        else:
            berendsen_coupling = "aniso" if coupling == "tri" else coupling
            if thermostat == "nose-hoover":
                lines.append(f"fix             integrate all nvt temp {t} {t} {tdamp}")
            else:
                lines.append("fix             integrate all nve")
                lines.extend(thermostat_fix())
            lines.append(
                f"fix             barostat all press/berendsen {berendsen_coupling} {p} {p} {pdamp}"
            )
    return lines


def _dump_lines(topo: LammpsTopology, settings: dict, names: dict[str, str]) -> list[str]:
    fmt = settings.get("dump_format", "xyz")
    every = int(settings.get("dump_every", 1000))
    if fmt == "none" or every <= 0:
        return []
    elements = " ".join(topo.type_elements())
    if fmt == "custom":
        style = atom_style_of(topo, settings)
        cols = ["id"]
        if style == "full":
            cols.append("mol")
        cols += ["type", "element"]
        if style in ("full", "charge"):
            cols.append("q")
        cols += ["x", "y", "z", "ix", "iy", "iz"]
        return [
            f"dump            traj all custom {every} {names['dump_custom']} " + " ".join(cols),
            f"dump_modify     traj element {elements} sort id",
        ]
    return [
        f"dump            traj all xyz {every} {names['dump_xyz']}",
        f"dump_modify     traj element {elements} sort id",
    ]


def build_input_script(
    topo: LammpsTopology,
    box,
    settings: dict,
    version: str = "",
    check_only: bool = False,
) -> str:
    names = file_names(settings)
    units = units_of(topo, settings)
    style = atom_style_of(topo, settings)
    out: list[str] = []
    out.append(f"# LAMMPS input written by {GENERATOR} {version}".rstrip())
    out.append(f"# Force field: {topo.description}")
    out.append(
        f"# {topo.n_atoms} atoms, {len(topo.atom_types)} atom types; data file {names['data']}"
    )
    if check_only:
        out.append("# 0-step check: reads the system, sets up every style and runs 0 steps.")
    out.append("")
    out.append("# ---------- system ----------")
    out.append(f"units           {units}")
    out.append(f"atom_style      {style}")
    boundary = " ".join("p" if p else "s" for p in box.pbc)
    out.append(f"boundary        {boundary}")
    out.append("")

    if topo.kind == "molecular":
        pair, kspace = pair_style_of(topo, box, settings)
        out.append(pair)
        out.append(f"pair_modify     mix {topo.mixing}")
        if kspace:
            out.append(kspace)
        if topo.bonds:
            out.append("bond_style      harmonic")
        if topo.angles:
            out.append("angle_style     harmonic")
        if topo.dihedrals:
            out.append(f"dihedral_style  {topo.dihedral_style}")
        if topo.impropers:
            out.append(f"improper_style  {topo.improper_style}")
        lj = " ".join(_fmt(v) for v in topo.special_lj)
        coul = " ".join(_fmt(v) for v in topo.special_coul)
        out.append(f"special_bonds   lj {lj} coul {coul}")
        out.append("")
        out.append(f"read_data       {names['data']}")
    elif topo.kind == "reaxff":
        out.append(f"read_data       {names['data']}")
        out.append("")
        out.append("pair_style      reaxff NULL")
        out.append(
            f"pair_coeff      * * {settings.get('reax_ffield', 'ffield.reax')} "
            + " ".join(topo.type_elements())
        )
        if settings.get("reax_qeq", True):
            out.append("fix             qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff")
    else:
        out.append(f"read_data       {names['data']}")
        out.append("")
        out.append(f"pair_style      {settings.get('other_pair_style', 'tersoff').strip()}")
        out.append(
            f"pair_coeff      * * {settings.get('other_potential_file', '').strip()} "
            + " ".join(topo.type_elements())
        )
    out.append("")

    skin = float(settings.get("neighbor_skin", 2.0))
    if units == "metal" and topo.kind == "other":
        skin = min(skin, 2.0)
    out.append(f"neighbor        {_fmt(skin)} bin")
    check = "yes" if settings.get("neigh_check", True) else "no"
    out.append(
        f"neigh_modify    every {int(settings.get('neigh_every', 1))} "
        f"delay {int(settings.get('neigh_delay', 0))} check {check}"
    )
    out.append("")
    thermo_cols = "step time temp press pe ke etotal density vol"
    if settings.get("ensemble") == "npt":
        thermo_cols += " lx ly lz"
    out.append(f"thermo_style    custom {thermo_cols}")
    out.append(f"thermo          {max(1, int(settings.get('thermo_every', 1000)))}")
    out.append("")

    ensemble = settings.get("ensemble", "nvt")
    shake = _shake_fix(topo)
    dump = _dump_lines(topo, settings, names)

    if check_only:
        if shake:
            out.append(shake)
        if ensemble != "minimize":
            out.append(
                f"timestep        {_fmt(_time(float(settings.get('timestep', 1.0)), units))}"
            )
            out.extend(_ensemble_fixes(topo, box, settings, units))
        out.append("run             0")
        return "\n".join(out) + "\n"

    minimize_line = (
        f"minimize        {float(settings.get('min_etol', 1e-4)):g} "
        f"{float(settings.get('min_ftol', 1e-6)):g} "
        f"{int(settings.get('min_maxiter', 1000))} {int(settings.get('min_maxeval', 10000))}"
    )
    if ensemble == "minimize":
        out.append("# ---------- energy minimization ----------")
        out.extend(dump)
        out.append(minimize_line)
    else:
        if settings.get("minimize_first", True):
            out.append("# ---------- energy minimization ----------")
            out.append(minimize_line)
            out.append("reset_timestep  0")
            out.append("")
        out.append(f"# ---------- {ensemble.upper()} molecular dynamics ----------")
        if shake:
            out.append(shake)
        temp = _fmt(float(settings.get("temperature", 300.0)))
        rot = "no" if any(box.pbc) else "yes"
        out.append(
            f"velocity        all create {temp} {int(settings.get('seed', 12345))} "
            f"mom yes rot {rot} dist gaussian"
        )
        out.extend(_ensemble_fixes(topo, box, settings, units))
        out.append(f"timestep        {_fmt(_time(float(settings.get('timestep', 1.0)), units))}")
        out.extend(dump)
        restart = int(settings.get("restart_every", 0))
        if restart > 0:
            out.append(f"restart         {restart} {names['restart']}.1 {names['restart']}.2")
        out.append(f"run             {int(settings.get('steps', 100000))}")
    if settings.get("write_final", True):
        out.append("")
        out.append(f"write_data      {names['final_data']}")
        if ensemble != "minimize":
            out.append(f"write_restart   {names['final_restart']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def validate(
    topo: LammpsTopology, box, settings: dict, host_elements=None
) -> list[tuple[str, str]]:
    """Internal consistency checks: list of (level, message), level in ERROR/WARNING/INFO."""
    issues: list[tuple[str, str]] = []
    n = topo.n_atoms
    if host_elements is not None and len(host_elements) != n:
        issues.append(("ERROR", f"{n} atoms written but the structure has {len(host_elements)}"))
    if len(box.positions) != n:
        issues.append(("ERROR", "coordinates and topology have different atom counts"))
    ntypes = len(topo.atom_types)
    if any(t < 0 or t >= ntypes for t in topo.atom_type):
        issues.append(("ERROR", "an atom refers to an undefined atom type"))
    used = set(topo.atom_type)
    if len(used) != ntypes:
        issues.append(("INFO", f"{ntypes - len(used)} atom type(s) are defined but unused"))
    for name, items, types in (
        ("bond", topo.bonds, topo.bond_types),
        ("angle", topo.angles, topo.angle_types),
        ("dihedral", topo.dihedrals, topo.dihedral_types),
        ("improper", topo.impropers, topo.improper_types),
    ):
        for item in items:
            tid, *atoms = item
            if tid < 0 or tid >= len(types):
                issues.append(("ERROR", f"a {name} refers to undefined {name} type {tid + 1}"))
                break
            if any(a < 0 or a >= n for a in atoms) or len(set(atoms)) != len(atoms):
                issues.append(
                    ("ERROR", f"{name} {[a + 1 for a in atoms]} has invalid atom indices")
                )
                break
    pairs = [tuple(sorted((i, j))) for _, i, j in topo.bonds]
    if len(pairs) != len(set(pairs)):
        issues.append(("WARNING", "some atom pairs are bonded twice"))
    if topo.kind == "molecular":
        total = topo.total_charge
        if abs(total) > 1e-3:
            issues.append(
                (
                    "WARNING",
                    f"net charge {total:+.4f} e; Ewald/PPPM treat it with a neutralising background",
                )
            )
            if abs(total - round(total)) > 1e-2:
                issues.append(("WARNING", "the net charge is not an integer"))
        if all(t.epsilon == 0.0 for t in topo.atom_types):
            issues.append(
                ("WARNING", "every atom type has epsilon = 0 (no van der Waals repulsion)")
            )
        for problem in shake_cluster_problems(topo):
            issues.append(("ERROR", problem))
        timestep = float(settings.get("timestep", 1.0))
        if timestep > 1.0 and not topo.shake_bond_types and any(e == "H" for e in topo.elements):
            issues.append(
                (
                    "WARNING",
                    f"timestep {timestep:g} fs without constraints on X-H bonds may be unstable",
                )
            )
    if topo.kind == "reaxff" and float(settings.get("timestep", 1.0)) > 0.5:
        issues.append(("WARNING", "ReaxFF usually needs a timestep of 0.1-0.5 fs"))
    if topo.kind == "other":
        if not str(settings.get("other_pair_style", "")).strip():
            issues.append(("ERROR", "no pair style given for the user potential"))
        if not str(settings.get("other_potential_file", "")).strip():
            issues.append(("WARNING", "no potential file given; edit the pair_coeff line"))
    if settings.get("ensemble") == "npt" and not all(box.pbc):
        issues.append(("ERROR", "NPT needs a fully periodic box"))
    if (
        settings.get("coupling") == "tri"
        and settings.get("ensemble") == "npt"
        and not box.triclinic
    ):
        issues.append(("INFO", "tri coupling needs a triclinic box; aniso is written instead"))
    cutoff = float(settings.get("pair_cutoff", 10.0))
    if topo.kind == "molecular" and all(box.pbc):
        width = min(box.widths())
        if width < 2.0 * cutoff:
            issues.append(
                (
                    "WARNING",
                    (
                        f"the box is {width:.2f} A wide, less than twice the {cutoff:g} A "
                        "cutoff; atoms will see their own periodic images"
                    ),
                ),
            )
    if (
        topo.kind == "molecular"
        and settings.get("kspace") == "none"
        and all(box.pbc)
        and any(abs(q) > 1e-12 for q in topo.charges)
    ):
        issues.append(
            ("WARNING", "charges without a long-range solver: Coulomb is truncated at the cutoff")
        )
    for note in box.notes:
        issues.append(("INFO", note))
    for warning in topo.warnings:
        issues.append(("WARNING", warning))
    if box.triclinic:
        lx, _ly, lz = box.lengths
        xy, xz, yz = box.tilts
        if abs(xy) > lx / 2 + 1e-6 or abs(xz) + abs(yz) > lz / 2 + 1e-6:
            issues.append(("INFO", "the triclinic box is strongly skewed; LAMMPS will warn"))
    return issues


def summary(topo: LammpsTopology, box) -> str:
    counts = (
        f"{topo.n_atoms} atoms, {len(topo.atom_types)} atom types, "
        f"{len(topo.bonds)} bonds ({len(topo.bond_types)} types), "
        f"{len(topo.angles)} angles ({len(topo.angle_types)}), "
        f"{len(topo.dihedrals)} dihedrals ({len(topo.dihedral_types)}), "
        f"{len(topo.impropers)} impropers ({len(topo.improper_types)})"
    )
    if topo.kind != "molecular":
        counts = f"{topo.n_atoms} atoms, {len(topo.atom_types)} atom types (no bonded terms)"
    lx, ly, lz = box.lengths
    shape = "triclinic" if box.triclinic else "orthorhombic"
    return (
        f"{counts}; {len(set(topo.mol_ids))} molecule(s); net charge {topo.total_charge:+.4f} e; "
        f"{shape} box {lx:.3f} x {ly:.3f} x {lz:.3f} A"
    )
