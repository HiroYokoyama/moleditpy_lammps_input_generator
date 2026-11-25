"""Simulation cell: where it comes from and how it maps onto a LAMMPS box.

Cell sources, in the order ``auto`` tries them:

1. the ``moleditpy_cell`` string property on the molecule (suite convention:
   JSON ``{"lattice": [[...],[...],[...]], "origin": [...], "pbc": [...]}``,
   Angstrom, rows are the cell vectors);
2. a cell remembered by this plugin for a molecule with the same atom count
   (RDKit drops molecule properties in the host's undo snapshots and project
   files, so the property alone does not survive an undo or a reload);
3. the CIF Viewer panel's lattice;
4. a user box: padding around the molecule, or explicit lengths and angles.

LAMMPS needs a *restricted* triclinic box (a along x, b in the xy plane).  A
general cell is rotated into that frame through fractional coordinates and the
tilt factors are reduced to the range LAMMPS prefers; atoms are wrapped into
the box with image flags, so molecules stay whole when unwrapped.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .cell_model import cell_vectors, lattice_parameters

logger = logging.getLogger(__name__)

CELL_PROPERTY = "moleditpy_cell"


@dataclass
class CellInfo:
    lattice: np.ndarray  # rows a, b, c (Angstrom)
    origin: np.ndarray
    pbc: tuple[bool, bool, bool]
    source: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "lattice": [[float(v) for v in row] for row in self.lattice],
                "origin": [float(v) for v in self.origin],
                "pbc": [bool(v) for v in self.pbc],
            }
        )

    def parameters(self):
        return lattice_parameters(self.lattice)


@dataclass
class LammpsBox:
    lo: np.ndarray  # xlo, ylo, zlo
    lengths: np.ndarray  # lx, ly, lz
    tilts: np.ndarray  # xy, xz, yz
    pbc: tuple[bool, bool, bool]
    positions: np.ndarray  # wrapped, in the LAMMPS frame
    images: np.ndarray  # integer image flags
    notes: list[str] = field(default_factory=list)

    @property
    def triclinic(self) -> bool:
        return bool(np.any(np.abs(self.tilts) > 1e-8))

    @property
    def matrix(self) -> np.ndarray:
        lx, ly, lz = self.lengths
        xy, xz, yz = self.tilts
        return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])

    def widths(self) -> tuple[float, float, float]:
        """Perpendicular widths of the box (the distance between opposite faces)."""
        inverse = np.linalg.inv(self.matrix).T
        return tuple(1.0 / float(np.linalg.norm(row)) for row in inverse)  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))


def parse_cell_json(text: str, source: str = CELL_PROPERTY) -> CellInfo:
    """Parse the suite's cell JSON; raise ValueError when it is unusable."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a JSON object")  # noqa: TRY004
    try:
        lattice = np.asarray(data["lattice"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} has no usable 'lattice'") from exc
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"{source}: 'lattice' must be a 3x3 matrix")
    if abs(float(np.linalg.det(lattice))) < 1e-6:
        raise ValueError(f"{source}: the cell has no volume")
    try:
        origin = np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: 'origin' must be three numbers") from exc
    if origin.shape != (3,):
        raise ValueError(f"{source}: 'origin' must be three numbers")
    pbc_raw = data.get("pbc", [True, True, True])
    if isinstance(pbc_raw, bool):
        pbc_raw = [pbc_raw] * 3
    if not isinstance(pbc_raw, (list, tuple)) or len(pbc_raw) != 3:
        raise ValueError(f"{source}: 'pbc' must be three booleans")
    pbc = tuple(bool(v) for v in pbc_raw)
    return CellInfo(lattice=lattice, origin=origin, pbc=pbc, source=source)  # type: ignore[arg-type]


def cell_from_property(mol) -> CellInfo | None:
    """The molecule's ``moleditpy_cell``, or None when it has none."""
    if mol is None:
        return None
    try:
        if not mol.HasProp(CELL_PROPERTY):
            return None
        text = mol.GetProp(CELL_PROPERTY)
    except Exception:  # noqa: BLE001  # pragma: no cover - duck-typed molecule
        return None
    return parse_cell_json(text, CELL_PROPERTY)


def find_cif_viewer(main_window):
    """Locate the CIF Viewer plugin's widget (it is another plugin's window)."""
    if main_window is None:
        return None

    def _holder(window):
        if window is None:
            return None
        if getattr(window, "structure", None) is not None:
            return window
        inner = window.widget() if hasattr(window, "widget") else None
        if inner is not None and getattr(inner, "structure", None) is not None:
            return inner
        return None

    try:
        registry = getattr(getattr(main_window, "plugin_manager", None), "plugin_windows", None)
        if isinstance(registry, dict):
            for plugin_name, windows in registry.items():
                if not isinstance(windows, dict):
                    continue
                for window_id, window in windows.items():
                    key = f"{plugin_name} {window_id}".lower()
                    if "cif" in key and "viewer" in key:
                        holder = _holder(window)
                        if holder is not None:
                            return holder
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - host internals guard
        logger.debug("CIF Viewer lookup failed: %s", exc)
    try:
        from PySide6.QtWidgets import QWidget

        for child in main_window.findChildren(QWidget):
            if type(child).__name__ == "CifViewerWidget" and getattr(child, "structure", None):
                return child
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - host internals guard
        logger.debug("CIF Viewer widget scan failed: %s", exc)
    return None


def cell_from_viewer(main_window) -> CellInfo | None:
    """The lattice of the structure shown in the CIF Viewer, or None."""
    holder = find_cif_viewer(main_window)
    structure = getattr(holder, "structure", None) if holder is not None else None
    if structure is None:
        return None
    try:
        lattice = np.asarray(getattr(structure, "lattice", None), dtype=float)
    except (TypeError, ValueError):
        return None
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) < 1e-6:
        return None
    return CellInfo(
        lattice=lattice, origin=np.zeros(3), pbc=(True, True, True), source="CIF Viewer"
    )


def cell_from_padding(positions: np.ndarray, padding: float, periodic: bool = True) -> CellInfo:
    """Orthorhombic box around the atoms with ``padding`` Angstrom on each side."""
    positions = np.asarray(positions, dtype=float)
    if padding < 0:
        raise ValueError("padding must not be negative")
    low = positions.min(axis=0) - padding
    high = positions.max(axis=0) + padding
    lengths = np.maximum(high - low, 1.0)
    return CellInfo(
        lattice=np.diag(lengths),
        origin=low,
        pbc=(periodic,) * 3,  # type: ignore[arg-type]
        source=f"user box ({padding:g} A padding)",
    )


def cell_from_lengths(
    positions: np.ndarray,
    lengths: Sequence[float],
    angles: Sequence[float] = (90.0, 90.0, 90.0),
    periodic: bool = True,
) -> CellInfo:
    """Box of the given lengths and angles, centred on the atoms."""
    lattice = cell_vectors(lengths, angles)
    positions = np.asarray(positions, dtype=float)
    centre = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    origin = centre - lattice.sum(axis=0) / 2.0
    label = "x".join(f"{v:g}" for v in lengths)
    if any(abs(float(a) - 90.0) > 1e-9 for a in angles):
        label += " (" + "/".join(f"{a:g}" for a in angles) + " deg)"
    return CellInfo(
        lattice=lattice,
        origin=origin,
        pbc=(periodic,) * 3,
        source=f"user box {label} A",  # type: ignore[arg-type]
    )


def restricted_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """(lx, ly, lz, xy, xz, yz) of the restricted triclinic equivalent of ``lattice``."""
    a, b, c = (np.asarray(row, dtype=float) for row in lattice)
    lx = float(np.linalg.norm(a))
    a_hat = a / lx
    xy = float(np.dot(b, a_hat))
    ly = math.sqrt(max(float(np.dot(b, b)) - xy * xy, 0.0))
    xz = float(np.dot(c, a_hat))
    yz = (float(np.dot(b, c)) - xy * xz) / ly
    lz = math.sqrt(max(float(np.dot(c, c)) - xz * xz - yz * yz, 0.0))
    return lx, ly, lz, xy, xz, yz


def reduce_tilts(lx: float, ly: float, xy: float, xz: float, yz: float):
    """Bring tilt factors into [-l/2, l/2] by an equivalent change of cell vectors."""
    shifts = 0
    n = round(yz / ly)
    if n:
        yz -= n * ly
        xz -= n * xy
        shifts += 1
    n = round(xz / lx)
    if n:
        xz -= n * lx
        shifts += 1
    n = round(xy / lx)
    if n:
        xy -= n * lx
        shifts += 1
    return xy, xz, yz, shifts


def to_lammps_box(cell: CellInfo, positions: np.ndarray) -> LammpsBox:
    """Rotate ``cell`` and ``positions`` into a LAMMPS restricted triclinic box."""
    lattice = np.asarray(cell.lattice, dtype=float)
    if float(np.linalg.det(lattice)) <= 0.0:
        raise ValueError(
            "The cell vectors form a left-handed set; LAMMPS needs a right-handed "
            "cell (swap two cell vectors)."
        )
    positions = np.asarray(positions, dtype=float)
    notes: list[str] = []
    lx, ly, lz, xy, xz, yz = restricted_parameters(lattice)
    restricted = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])
    rotation = np.linalg.inv(lattice) @ restricted
    rotated = (positions - cell.origin) @ rotation + cell.origin
    if not np.allclose(rotation, np.eye(3), atol=1e-9):
        notes.append(
            "The cell is not in LAMMPS orientation (a along x, b in the xy plane); "
            "coordinates were rotated into that frame."
        )

    if all(cell.pbc):
        xy, xz, yz, shifts = reduce_tilts(lx, ly, xy, xz, yz)
        if shifts:
            notes.append("Tilt factors were reduced to an equivalent, less skewed cell.")
    elif any(abs(t) > 1e-8 for t in (xy, xz, yz)):
        raise ValueError(
            "A cell with non-periodic directions must be orthorhombic in LAMMPS "
            "here; use an orthorhombic box or make the cell fully periodic."
        )
    box_matrix = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])
    lo = np.asarray(cell.origin, dtype=float).copy()

    fract = (rotated - lo) @ np.linalg.inv(box_matrix)
    images = np.zeros_like(fract, dtype=int)
    lengths = np.array([lx, ly, lz])
    for dim in range(3):
        if cell.pbc[dim]:
            shift = np.floor(fract[:, dim] + 1e-12)
            images[:, dim] = shift.astype(int)
            fract[:, dim] -= shift
            fract[np.abs(fract[:, dim] - 1.0) < 1e-12, dim] = 0.0
        else:
            low = min(0.0, float(fract[:, dim].min()))
            high = max(1.0, float(fract[:, dim].max()))
            if low < 0.0 or high > 1.0:
                margin = 1.0 / lengths[dim]
                low = low - margin if low < 0.0 else low
                high = high + margin if high > 1.0 else high
                lo[dim] += low * lengths[dim]
                fract[:, dim] = (fract[:, dim] - low) / (high - low)
                lengths[dim] *= high - low
                notes.append(
                    f"The non-periodic {'xyz'[dim]} bounds were widened to contain every atom."
                )
    box_matrix = np.array([[lengths[0], 0.0, 0.0], [xy, lengths[1], 0.0], [xz, yz, lengths[2]]])
    wrapped = fract @ box_matrix + lo
    return LammpsBox(
        lo=lo,
        lengths=lengths,
        tilts=np.array([xy, xz, yz]),
        pbc=tuple(cell.pbc),  # type: ignore[arg-type]
        positions=wrapped,
        images=images,
        notes=notes,
    )


def resolve_cell(
    settings: dict,
    mol,
    positions: np.ndarray,
    remembered: CellInfo | None = None,
    viewer_cell: CellInfo | None = None,
) -> tuple[CellInfo, list[str]]:
    """Pick the cell named by ``settings['cell_source']``; returns (cell, notes)."""
    source = settings.get("cell_source", "auto")
    notes: list[str] = []
    periodic = bool(settings.get("user_periodic", True))

    def _user_box() -> CellInfo:
        if settings.get("box_mode", "padding") == "lengths":
            return cell_from_lengths(
                positions,
                settings.get("box_lengths", [40.0, 40.0, 40.0]),
                settings.get("box_angles", [90.0, 90.0, 90.0]),
                periodic=periodic,
            )
        return cell_from_padding(positions, float(settings.get("padding", 10.0)), periodic=periodic)

    if source == "user":
        return _user_box(), notes
    if source in ("auto", "property"):
        try:
            cell = cell_from_property(mol)
        except ValueError as exc:
            notes.append(f"Ignored the molecule's cell: {exc}")
            cell = None
        if cell is not None:
            return cell, notes
        if remembered is not None:
            notes.append(
                "The molecule has no moleditpy_cell property (undo and project files drop "
                "it); using the cell remembered for this structure."
            )
            return remembered, notes
        if source == "property":
            notes.append("The molecule carries no moleditpy_cell; using the user box instead.")
            return _user_box(), notes
    if source in ("auto", "viewer"):
        if viewer_cell is not None:
            return viewer_cell, notes
        if source == "viewer":
            notes.append("The CIF Viewer shows no structure; using the user box instead.")
    return _user_box(), notes
