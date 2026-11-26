"""Cell sources and the conversion to a LAMMPS restricted triclinic box."""

import json

import pytest

np = pytest.importorskip("numpy")

from lammps_input_generator import cell as cellmod
from lammps_input_generator.cell_model import cell_vectors


class Mol:
    def __init__(self, props=None, n=3):
        self.props = props or {}
        self.n = n

    def HasProp(self, key):
        return key in self.props

    def GetProp(self, key):
        return self.props[key]

    def GetNumAtoms(self):
        return self.n


def _rotation(seed=3):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_parse_cell_json_and_errors():
    cell = cellmod.parse_cell_json(json.dumps({"lattice": np.eye(3).tolist(), "pbc": True}))
    assert cell.pbc == (True, True, True)
    assert np.allclose(cell.origin, 0)
    for bad in (
        "nope",
        "[]",
        '{"lattice": [[1,0],[0,1]]}',
        '{"lattice": [[0,0,0],[0,1,0],[0,0,1]]}',
        '{"lattice": [[1,0,0],[0,1,0],[0,0,1]], "pbc": [true]}',
    ):
        with pytest.raises(ValueError):
            cellmod.parse_cell_json(bad)


def test_cell_round_trips_through_json():
    cell = cellmod.CellInfo(
        np.diag([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0]), (True, False, True), "x"
    )
    again = cellmod.parse_cell_json(cell.to_json())
    assert np.allclose(again.lattice, cell.lattice) and again.pbc == cell.pbc


def test_restricted_parameters_match_lattice_constants():
    lattice = cell_vectors((10.0, 11.0, 12.0), (80.0, 95.0, 105.0))
    lx, ly, lz, xy, xz, yz = cellmod.restricted_parameters(lattice)
    a, b, c = 10.0, 11.0, 12.0
    alpha, beta, gamma = np.radians([80.0, 95.0, 105.0])
    assert lx == pytest.approx(a)
    assert xy == pytest.approx(b * np.cos(gamma))
    assert xz == pytest.approx(c * np.cos(beta))
    assert ly == pytest.approx(np.sqrt(b**2 - xy**2))
    assert yz == pytest.approx((b * c * np.cos(alpha) - xy * xz) / ly)
    assert lz == pytest.approx(np.sqrt(c**2 - xz**2 - yz**2))


def test_rotated_cell_gives_the_same_box_and_geometry():
    lattice = cell_vectors((10.0, 11.0, 12.0), (80.0, 95.0, 105.0))
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 1, size=(20, 3)) @ lattice
    plain = cellmod.to_lammps_box(
        cellmod.CellInfo(lattice, np.zeros(3), (True,) * 3, "a"), positions
    )
    rot = _rotation()
    turned = cellmod.to_lammps_box(
        cellmod.CellInfo(lattice @ rot, np.zeros(3), (True,) * 3, "b"), positions @ rot
    )
    assert np.allclose(plain.lengths, turned.lengths)
    assert np.allclose(plain.tilts, turned.tilts)
    assert np.allclose(plain.positions, turned.positions, atol=1e-8)
    assert any("rotated" in note for note in turned.notes)


def test_wrapping_with_image_flags_preserves_unwrapped_positions():
    lattice = cell_vectors((8.0, 9.0, 10.0), (90.0, 100.0, 120.0))
    rng = np.random.default_rng(2)
    positions = rng.uniform(-1.5, 2.5, size=(50, 3)) @ lattice
    box = cellmod.to_lammps_box(cellmod.CellInfo(lattice, np.zeros(3), (True,) * 3, "t"), positions)
    fract = (box.positions - box.lo) @ np.linalg.inv(box.matrix)
    assert np.all(fract >= -1e-9) and np.all(fract < 1 + 1e-9)
    unwrapped = box.positions + box.images @ box.matrix
    lx, ly, lz, xy, xz, yz = cellmod.restricted_parameters(lattice)
    restricted = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    expected = positions @ (np.linalg.inv(lattice) @ restricted)
    assert np.allclose(unwrapped, expected, atol=1e-8)


def test_tilt_reduction_keeps_lattice_equivalent():
    lattice = np.array([[10.0, 0, 0], [7.5, 10.0, 0], [-6.0, 8.0, 10.0]])
    box = cellmod.to_lammps_box(
        cellmod.CellInfo(lattice, np.zeros(3), (True,) * 3, "t"), np.zeros((1, 3))
    )
    xy, xz, yz = box.tilts
    assert abs(xy) <= 5.0 + 1e-9 and abs(xz) <= 5.0 + 1e-9 and abs(yz) <= 5.0 + 1e-9
    # same lattice: the reduced matrix is an integer recombination of the original
    transform = box.matrix @ np.linalg.inv(lattice)
    assert np.allclose(transform, np.round(transform), atol=1e-9)
    assert abs(np.linalg.det(np.round(transform))) == pytest.approx(1.0)


def test_left_handed_cell_is_rejected():
    lattice = np.diag([1.0, 1.0, -1.0]) * 10
    with pytest.raises(ValueError, match="left-handed"):
        cellmod.to_lammps_box(
            cellmod.CellInfo(lattice, np.zeros(3), (True,) * 3, "t"), np.zeros((1, 3))
        )


def test_non_periodic_box_is_widened_and_must_be_orthorhombic():
    cell = cellmod.CellInfo(np.diag([5.0, 5.0, 5.0]), np.zeros(3), (False,) * 3, "t")
    box = cellmod.to_lammps_box(cell, np.array([[-1.0, 2.0, 6.0]]))
    assert np.all(box.images == 0)
    assert box.lo[0] < -1.0 and box.lo[2] + box.lengths[2] > 6.0
    tilted = cellmod.CellInfo(
        cell_vectors((5, 5, 5), (90, 90, 120)), np.zeros(3), (True, True, False), "t"
    )
    with pytest.raises(ValueError, match="orthorhombic"):
        cellmod.to_lammps_box(tilted, np.zeros((1, 3)))


def test_padding_and_lengths_boxes():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    padded = cellmod.cell_from_padding(positions, 5.0)
    assert np.allclose(np.diag(padded.lattice), [12.0, 14.0, 16.0])
    assert np.allclose(padded.origin, [-5.0, -5.0, -5.0])
    sized = cellmod.cell_from_lengths(positions, (20.0, 20.0, 20.0))
    centre = sized.origin + sized.lattice.sum(axis=0) / 2
    assert np.allclose(centre, [1.0, 2.0, 3.0])


def test_resolve_order():
    positions = np.zeros((3, 3))
    prop = json.dumps({"lattice": (np.eye(3) * 7).tolist()})
    remembered = cellmod.CellInfo(np.eye(3) * 8, np.zeros(3), (True,) * 3, "remembered")
    viewer = cellmod.CellInfo(np.eye(3) * 9, np.zeros(3), (True,) * 3, "CIF Viewer")
    settings = {"cell_source": "auto", "padding": 1.0}
    cell, _ = cellmod.resolve_cell(
        settings, Mol({"moleditpy_cell": prop}), positions, remembered, viewer
    )
    assert cell.lattice[0, 0] == 7
    cell, notes = cellmod.resolve_cell(settings, Mol(), positions, remembered, viewer)
    assert cell.lattice[0, 0] == 8 and notes
    cell, _ = cellmod.resolve_cell(settings, Mol(), positions, None, viewer)
    assert cell.lattice[0, 0] == 9
    cell, _ = cellmod.resolve_cell(settings, Mol(), positions, None, None)
    assert cell.source.startswith("user box")
    cell, _ = cellmod.resolve_cell(
        {"cell_source": "user", "padding": 1.0}, Mol({"moleditpy_cell": prop}), positions
    )
    assert cell.source.startswith("user box")
    cell, notes = cellmod.resolve_cell(
        {"cell_source": "auto"}, Mol({"moleditpy_cell": "bad"}), positions
    )
    assert any("Ignored" in n for n in notes)


def test_cif_viewer_lookup_through_plugin_manager():
    class Structure:
        lattice = (np.eye(3) * 4).tolist()

    class Viewer:
        structure = Structure()

    class Manager:
        def __init__(self):
            self.plugin_windows = {"CIF Viewer": {"cif_viewer_dock": Viewer()}}

    class MainWindow:
        def __init__(self):
            self.plugin_manager = Manager()

        def findChildren(self, _cls):
            return []

    cell = cellmod.cell_from_viewer(MainWindow())
    assert cell is not None and cell.lattice[1, 1] == 4
    assert cellmod.cell_from_viewer(None) is None
