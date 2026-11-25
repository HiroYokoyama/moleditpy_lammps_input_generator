"""The LAMMPS Input Generator dialog: settings on the left, live preview on the right."""

from __future__ import annotations

import json
import logging
import os

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import PLUGIN_NAME, PLUGIN_VERSION, SETTINGS_FILE, lmp_runner, writer
from . import cell as cellmod
from . import generate as gen

logger = logging.getLogger(__name__)

PREVIEW_LINES = 3000

FF_CHOICES = [
    ("uff", "Built-in fallback: RDKit UFF"),
    ("mdff", "moleditpy-mdff file (Force Field Builder)"),
    ("reaxff", "ReaxFF (user ffield file)"),
    ("other", "Other potential (Tersoff, EAM, ...)"),
]
CHARGE_CHOICES = [("gasteiger", "Gasteiger"), ("mmff", "MMFF94"), ("zero", "Zero")]
DIHEDRAL_CHOICES = [
    ("auto", "Auto (charmm, fourier if phases are not whole degrees)"),
    ("charmm", "charmm (weight 0, MOLECULE package)"),
    ("fourier", "fourier (EXTRA-MOLECULE package)"),
]
CELL_CHOICES = [
    ("auto", "Auto: molecule cell, CIF Viewer, else user box"),
    ("property", "Molecule cell (moleditpy_cell)"),
    ("viewer", "CIF Viewer lattice"),
    ("user", "User box"),
]
BOX_MODES = [("padding", "Padding around the molecule"), ("lengths", "Lengths and angles")]
ENSEMBLE_CHOICES = [
    ("minimize", "Energy minimization"),
    ("nve", "NVE"),
    ("nvt", "NVT"),
    ("npt", "NPT"),
]
THERMOSTAT_CHOICES = [
    ("nose-hoover", "Nose-Hoover"),
    ("langevin", "Langevin"),
    ("berendsen", "Berendsen"),
    ("csvr", "CSVR / Bussi (EXTRA-FIX)"),
]
BAROSTAT_CHOICES = [("nose-hoover", "Nose-Hoover (MTK)"), ("berendsen", "Berendsen")]
COUPLING_CHOICES = [("iso", "iso"), ("aniso", "aniso"), ("tri", "tri (triclinic)")]
DUMP_CHOICES = [
    ("xyz", "XYZ with elements (opens in Large System Viewer)"),
    ("custom", "Custom lammpstrj (OVITO / VMD)"),
    ("none", "No trajectory"),
]
KSPACE_CHOICES = [("pppm", "PPPM"), ("ewald", "Ewald"), ("none", "None (cut-off Coulomb)")]
UNITS_CHOICES = [("metal", "metal (eV, A, ps, bar)"), ("real", "real (kcal/mol, A, fs, atm)")]
ATOM_STYLE_CHOICES = [("atomic", "atomic"), ("charge", "charge")]


def _combo(choices):
    box = QComboBox()
    for value, label in choices:
        box.addItem(label, value)
    return box


def _dspin(low, high, decimals=3, step=1.0, suffix=""):
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _ispin(low, high, step=1):
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    return spin


class LammpsInputDialog(QDialog):
    def __init__(
        self,
        parent=None,
        persistent_settings=None,
        get_molecule=None,
        mark_modified=None,
        main_window=None,
        remembered_cell=None,
    ):
        super().__init__(parent)
        self.persistent_settings = persistent_settings if persistent_settings is not None else {}
        self.get_molecule = get_molecule
        self.mark_modified = mark_modified
        self.main_window = main_window
        self.remembered_cell = remembered_cell if remembered_cell is not None else {}
        self._updating = False
        self.ui_ready = False
        self._generated = None
        self._cache = gen.TopologyCache()
        self._last_dir = ""
        self._saved_texts = None  # (input, data) last written
        self._process = None
        self._check_dir = None
        self.presets_data = {}
        self.fields = {}

        self.setWindowTitle(f"{PLUGIN_NAME} v{PLUGIN_VERSION}")
        self.resize(1200, 780)
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.update_preview)

        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self._on_ctrl_s)

        self.load_presets_from_file()
        self.ui_ready = True
        self.apply_settings(self.persistent_settings)

    # ------------------------------------------------------------------ UI

    def _register(self, key, widget, kind):
        self.fields[key] = (widget, kind)
        if kind == "combo":
            widget.currentIndexChanged.connect(self._schedule)
        elif kind in ("dspin", "spin"):
            widget.valueChanged.connect(self._schedule)
        elif kind == "check":
            widget.toggled.connect(self._schedule)
        elif kind == "line":
            widget.textChanged.connect(self._schedule)
        elif kind == "vec3":
            for spin in widget:
                spin.valueChanged.connect(self._schedule)
        return widget

    def _build_ui(self):
        layout = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.activated.connect(lambda _i: self.apply_selected_preset())
        preset_row.addWidget(self.preset_combo)
        save_preset = QPushButton("Save Preset...")
        save_preset.clicked.connect(self.save_preset_dialog)
        delete_preset = QPushButton("Delete Preset")
        delete_preset.clicked.connect(self.delete_preset)
        preset_row.addWidget(save_preset)
        preset_row.addWidget(delete_preset)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.addTab(self._build_ff_tab(), "Force field")
        self.settings_tabs.addTab(self._build_box_tab(), "Box")
        self.settings_tabs.addTab(self._build_run_tab(), "Run")
        self.settings_tabs.addTab(self._build_output_tab(), "Output")
        splitter.addWidget(self.settings_tabs)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("-")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)
        self.preview_tabs = QTabWidget()
        mono = QFont("Courier New", 9)
        self.input_preview = QPlainTextEdit()
        self.data_preview = QPlainTextEdit()
        self.checks_view = QPlainTextEdit()
        for view in (self.input_preview, self.data_preview, self.checks_view):
            view.setReadOnly(True)
            view.setFont(mono)
            view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview_tabs.addTab(self.input_preview, "in script")
        self.preview_tabs.addTab(self.data_preview, "data file")
        self.preview_tabs.addTab(self.checks_view, "Checks")
        right_layout.addWidget(self.preview_tabs, 1)
        splitter.addWidget(right)
        splitter.setSizes([430, 770])

        buttons = QHBoxLayout()
        self.check_button = QPushButton("Run 0-step Check")
        self.check_button.clicked.connect(self.run_zero_step_check)
        buttons.addWidget(self.check_button)
        buttons.addStretch(1)
        self.save_button = QPushButton("Save Files...")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.save_files)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(self.save_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def _build_ff_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        self._register("ff_source", _combo(FF_CHOICES), "combo")
        form.addRow("Parameters:", self.fields["ff_source"][0])
        outer.addLayout(form)

        self.mdff_group = QGroupBox("moleditpy-mdff file")
        mform = QFormLayout(self.mdff_group)
        path_row = QHBoxLayout()
        self._register("mdff_path", QLineEdit(), "line")
        path_row.addWidget(self.fields["mdff_path"][0], 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_mdff)
        path_row.addWidget(browse)
        mform.addRow("File:", path_row)
        self._register("dihedral_style", _combo(DIHEDRAL_CHOICES), "combo")
        mform.addRow("Dihedral style:", self.fields["dihedral_style"][0])
        outer.addWidget(self.mdff_group)

        self.uff_group = QGroupBox("UFF fallback")
        uform = QFormLayout(self.uff_group)
        self._register("charge_method", _combo(CHARGE_CHOICES), "combo")
        uform.addRow("Charges:", self.fields["charge_method"][0])
        note = QLabel(
            "UFF is a generic fallback for a quick start. For production runs build "
            "parameters with the Force Field Builder and load its .mdff.json file."
        )
        note.setWordWrap(True)
        uform.addRow(note)
        outer.addWidget(self.uff_group)

        self.constraint_group = QGroupBox("Constraints")
        cform = QFormLayout(self.constraint_group)
        self._register("constrain_h", QCheckBox("SHAKE bonds to hydrogen"), "check")
        cform.addRow(self.fields["constrain_h"][0])
        hint = QLabel(
            "Rigid water (settles) and explicit constraints in an mdff file are always applied."
        )
        hint.setWordWrap(True)
        cform.addRow(hint)
        outer.addWidget(self.constraint_group)

        self.reax_group = QGroupBox("ReaxFF")
        rform = QFormLayout(self.reax_group)
        self._register("reax_ffield", QLineEdit(), "line")
        rform.addRow("ffield file:", self.fields["reax_ffield"][0])
        self._register("reax_qeq", QCheckBox("Charge equilibration (fix qeq/reaxff)"), "check")
        rform.addRow(self.fields["reax_qeq"][0])
        outer.addWidget(self.reax_group)

        self.other_group = QGroupBox("Other potential")
        oform = QFormLayout(self.other_group)
        self._register("other_pair_style", QLineEdit(), "line")
        oform.addRow("pair_style:", self.fields["other_pair_style"][0])
        self._register("other_potential_file", QLineEdit(), "line")
        oform.addRow("Potential file:", self.fields["other_potential_file"][0])
        self._register("other_units", _combo(UNITS_CHOICES), "combo")
        oform.addRow("Units:", self.fields["other_units"][0])
        self._register("other_atom_style", _combo(ATOM_STYLE_CHOICES), "combo")
        oform.addRow("atom_style:", self.fields["other_atom_style"][0])
        outer.addWidget(self.other_group)
        outer.addStretch(1)
        return page

    def _build_box_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        self._register("cell_source", _combo(CELL_CHOICES), "combo")
        form.addRow("Cell:", self.fields["cell_source"][0])
        outer.addLayout(form)

        self.user_box_group = QGroupBox("User box")
        bform = QFormLayout(self.user_box_group)
        self._register("box_mode", _combo(BOX_MODES), "combo")
        bform.addRow("Mode:", self.fields["box_mode"][0])
        self._register("padding", _dspin(0.0, 500.0, 2, 1.0, " A"), "dspin")
        bform.addRow("Padding:", self.fields["padding"][0])
        lengths = [_dspin(1.0, 10000.0, 3, 1.0, " A") for _ in range(3)]
        angles = [_dspin(30.0, 150.0, 3, 1.0, " deg") for _ in range(3)]
        for key, spins, labels in (
            ("box_lengths", lengths, ("a", "b", "c")),
            ("box_angles", angles, ("alpha", "beta", "gamma")),
        ):
            row = QHBoxLayout()
            for label, spin in zip(labels, spins):
                row.addWidget(QLabel(label))
                row.addWidget(spin)
            self._register(key, spins, "vec3")
            bform.addRow("Lengths:" if key == "box_lengths" else "Angles:", row)
        self._register("user_periodic", QCheckBox("Periodic in all directions"), "check")
        bform.addRow(self.fields["user_periodic"][0])
        outer.addWidget(self.user_box_group)

        self.cell_label = QLabel("-")
        self.cell_label.setWordWrap(True)
        self.cell_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.cell_label)
        outer.addStretch(1)
        return page

    def _build_run_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        self._register("ensemble", _combo(ENSEMBLE_CHOICES), "combo")
        form.addRow("Protocol:", self.fields["ensemble"][0])
        self._register("minimize_first", QCheckBox("Minimize before MD"), "check")
        form.addRow(self.fields["minimize_first"][0])
        outer.addLayout(form)

        self.min_group = QGroupBox("Minimization")
        mform = QFormLayout(self.min_group)
        self._register("min_etol", _dspin(0.0, 1.0, 8, 1e-4), "dspin")
        mform.addRow("etol:", self.fields["min_etol"][0])
        self._register("min_ftol", _dspin(0.0, 10.0, 8, 1e-6), "dspin")
        mform.addRow("ftol:", self.fields["min_ftol"][0])
        self._register("min_maxiter", _ispin(1, 10**8, 100), "spin")
        mform.addRow("maxiter:", self.fields["min_maxiter"][0])
        self._register("min_maxeval", _ispin(1, 10**8, 1000), "spin")
        mform.addRow("maxeval:", self.fields["min_maxeval"][0])
        outer.addWidget(self.min_group)

        self.md_group = QGroupBox("Molecular dynamics")
        dform = QFormLayout(self.md_group)
        self._register("thermostat", _combo(THERMOSTAT_CHOICES), "combo")
        dform.addRow("Thermostat:", self.fields["thermostat"][0])
        self._register("barostat", _combo(BAROSTAT_CHOICES), "combo")
        dform.addRow("Barostat:", self.fields["barostat"][0])
        self._register("coupling", _combo(COUPLING_CHOICES), "combo")
        dform.addRow("Pressure coupling:", self.fields["coupling"][0])
        self._register("temperature", _dspin(0.0, 100000.0, 2, 10.0, " K"), "dspin")
        dform.addRow("Temperature:", self.fields["temperature"][0])
        self._register("pressure", _dspin(-100000.0, 1e7, 3, 1.0), "dspin")
        dform.addRow("Pressure (atm real / bar metal):", self.fields["pressure"][0])
        self._register("tdamp", _dspin(0.001, 1e6, 3, 10.0, " fs"), "dspin")
        dform.addRow("Tdamp:", self.fields["tdamp"][0])
        self._register("pdamp", _dspin(0.001, 1e7, 3, 100.0, " fs"), "dspin")
        dform.addRow("Pdamp:", self.fields["pdamp"][0])
        self._register("timestep", _dspin(0.001, 100.0, 3, 0.25, " fs"), "dspin")
        dform.addRow("Timestep:", self.fields["timestep"][0])
        self._register("steps", _ispin(0, 2**31 - 1, 1000), "spin")
        dform.addRow("Steps:", self.fields["steps"][0])
        self._register("seed", _ispin(1, 2**31 - 1, 1), "spin")
        dform.addRow("Random seed:", self.fields["seed"][0])
        outer.addWidget(self.md_group)
        outer.addStretch(1)
        return page

    def _build_output_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        files = QGroupBox("Files and output")
        form = QFormLayout(files)
        self._register("job_name", QLineEdit(), "line")
        form.addRow("Job name:", self.fields["job_name"][0])
        self._register("thermo_every", _ispin(1, 10**9, 100), "spin")
        form.addRow("Thermo every:", self.fields["thermo_every"][0])
        self._register("dump_format", _combo(DUMP_CHOICES), "combo")
        form.addRow("Trajectory:", self.fields["dump_format"][0])
        self._register("dump_every", _ispin(1, 10**9, 100), "spin")
        form.addRow("Dump every:", self.fields["dump_every"][0])
        self._register("restart_every", _ispin(0, 10**9, 1000), "spin")
        form.addRow("Restart every (0 = off):", self.fields["restart_every"][0])
        self._register("write_final", QCheckBox("Write final data and restart files"), "check")
        form.addRow(self.fields["write_final"][0])
        outer.addWidget(files)

        inter = QGroupBox("Interactions and neighbor list")
        iform = QFormLayout(inter)
        self._register("pair_cutoff", _dspin(1.0, 100.0, 2, 0.5, " A"), "dspin")
        iform.addRow("Pair cutoff:", self.fields["pair_cutoff"][0])
        self._register("kspace", _combo(KSPACE_CHOICES), "combo")
        iform.addRow("Long-range Coulomb:", self.fields["kspace"][0])
        self._register("kspace_accuracy", _dspin(1e-8, 1e-2, 8, 1e-5), "dspin")
        iform.addRow("KSpace accuracy:", self.fields["kspace_accuracy"][0])
        self._register("neighbor_skin", _dspin(0.0, 10.0, 2, 0.5, " A"), "dspin")
        iform.addRow("Neighbor skin:", self.fields["neighbor_skin"][0])
        self._register("neigh_every", _ispin(1, 1000), "spin")
        iform.addRow("Rebuild every:", self.fields["neigh_every"][0])
        self._register("neigh_delay", _ispin(0, 1000), "spin")
        iform.addRow("Delay:", self.fields["neigh_delay"][0])
        self._register("neigh_check", QCheckBox("Check displacement before rebuilding"), "check")
        iform.addRow(self.fields["neigh_check"][0])
        outer.addWidget(inter)

        lmp = QGroupBox("LAMMPS executable (optional 0-step check)")
        lform = QFormLayout(lmp)
        row = QHBoxLayout()
        self.lmp_edit = QLineEdit()
        self.lmp_edit.setPlaceholderText("lmp on PATH")
        self.lmp_edit.editingFinished.connect(self._lmp_path_changed)
        row.addWidget(self.lmp_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_lmp)
        row.addWidget(browse)
        lform.addRow("Path:", row)
        self.lmp_label = QLabel("-")
        self.lmp_label.setWordWrap(True)
        lform.addRow(self.lmp_label)
        outer.addWidget(lmp)
        outer.addStretch(1)
        return page

    # ------------------------------------------------------------ settings

    def read_settings(self) -> dict:
        values = {}
        for key, (widget, kind) in self.fields.items():
            if kind == "combo":
                values[key] = widget.currentData()
            elif kind in ("dspin", "spin"):
                values[key] = widget.value()
            elif kind == "check":
                values[key] = widget.isChecked()
            elif kind == "line":
                values[key] = widget.text()
            elif kind == "vec3":
                values[key] = [spin.value() for spin in widget]
        return values

    def apply_settings(self, settings) -> None:
        merged = writer.default_settings()
        merged.update(settings or {})
        self._updating = True
        try:
            for key, (widget, kind) in self.fields.items():
                value = merged.get(key)
                if value is None:
                    continue
                try:
                    if kind == "combo":
                        index = widget.findData(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    elif kind == "dspin":
                        widget.setValue(float(value))
                    elif kind == "spin":
                        widget.setValue(int(value))
                    elif kind == "check":
                        widget.setChecked(bool(value))
                    elif kind == "line":
                        widget.setText(str(value))
                    elif kind == "vec3":
                        for spin, v in zip(widget, value):
                            spin.setValue(float(v))
                except (TypeError, ValueError) as exc:
                    logger.debug("%s: bad setting %s=%r: %s", PLUGIN_NAME, key, value, exc)
        finally:
            self._updating = False
        self._update_visibility()
        self.update_preview()

    def _update_visibility(self):
        source = self.fields["ff_source"][0].currentData()
        self.mdff_group.setVisible(source == "mdff")
        self.uff_group.setVisible(source == "uff")
        self.constraint_group.setVisible(source in ("uff", "mdff"))
        self.reax_group.setVisible(source == "reaxff")
        self.other_group.setVisible(source == "other")
        ensemble = self.fields["ensemble"][0].currentData()
        self.md_group.setEnabled(ensemble != "minimize")
        self.fields["minimize_first"][0].setEnabled(ensemble != "minimize")
        self.min_group.setEnabled(
            ensemble == "minimize" or self.fields["minimize_first"][0].isChecked()
        )
        is_npt = ensemble == "npt"
        for key in ("barostat", "coupling", "pressure", "pdamp"):
            self.fields[key][0].setEnabled(is_npt)
        self.fields["thermostat"][0].setEnabled(ensemble in ("nvt", "npt"))
        self.fields["tdamp"][0].setEnabled(ensemble in ("nvt", "npt"))
        mode = self.fields["box_mode"][0].currentData()
        self.fields["padding"][0].setEnabled(mode == "padding")
        for spin in self.fields["box_lengths"][0] + self.fields["box_angles"][0]:
            spin.setEnabled(mode == "lengths")

    # ------------------------------------------------------------- preview

    def _schedule(self, *_args):
        if self._updating:
            return
        self._update_visibility()
        self._timer.start()

    def _molecule(self):
        if self.get_molecule is None:
            return None
        try:
            return self.get_molecule()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
            logger.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
            return None

    def _remembered_cell_info(self, mol):
        if not self.remembered_cell or mol is None:
            return None
        try:
            if mol.GetNumAtoms() != int(self.remembered_cell.get("n_atoms", -1)):
                return None
            info = cellmod.parse_cell_json(self.remembered_cell["cell"], "remembered cell")
            info.source = "remembered moleditpy_cell"
            return info
        except (KeyError, TypeError, ValueError):
            return None

    def update_preview(self, *_args) -> None:
        if self._updating or not self.ui_ready:
            return
        settings = self.read_settings()
        if settings != {k: self.persistent_settings.get(k) for k in settings}:
            self.persistent_settings.update(settings)
            if self.mark_modified is not None:
                try:
                    self.mark_modified()
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
                    logger.debug("%s: mark_modified failed: %s", PLUGIN_NAME, exc)
        self._refresh_lmp_label()

        mol = self._molecule()
        self._generated = None
        if mol is None or mol.GetNumAtoms() == 0:
            self._show_error("No molecule is loaded. Build or open a 3D structure first.")
            return
        try:
            viewer = None
            if settings.get("cell_source") in ("auto", "viewer"):
                viewer = cellmod.cell_from_viewer(self.main_window)
            result = gen.generate(
                mol,
                settings,
                PLUGIN_VERSION,
                remembered=self._remembered_cell_info(mol),
                viewer_cell=viewer,
                cache=self._cache,
            )
        except Exception as exc:
            logger.debug("%s: generation failed", PLUGIN_NAME, exc_info=True)
            self._show_error(str(exc))
            return
        self._generated = result
        if result.cell.source == cellmod.CELL_PROPERTY:
            self.remembered_cell.clear()
            self.remembered_cell.update(
                {"cell": result.cell.to_json(), "n_atoms": mol.GetNumAtoms()}
            )

        self.input_preview.setPlainText(result.input_text)
        lines = result.data_text.split("\n")
        if len(lines) > PREVIEW_LINES:
            shown = "\n".join(lines[:PREVIEW_LINES])
            shown += f"\n\n... {len(lines) - PREVIEW_LINES} more lines (the saved file is complete)"
        else:
            shown = result.data_text
        self.data_preview.setPlainText(shown)
        self._show_checks(result.issues)
        errors = sum(1 for level, _ in result.issues if level == "ERROR")
        warns = sum(1 for level, _ in result.issues if level == "WARNING")
        status = "Checks passed" if not errors else f"{errors} error(s)"
        if warns:
            status += f", {warns} warning(s)"
        self.summary_label.setText(f"{result.summary}.  {status} (see Checks).")
        (a, b, c), (al, be, ga) = result.cell.parameters()
        pbc = "".join("p" if p else "f" for p in result.cell.pbc)
        self.cell_label.setText(
            f"Cell used: {result.cell.source}\n"
            f"a={a:.3f}  b={b:.3f}  c={c:.3f} A   alpha={al:.2f}  beta={be:.2f}  gamma={ga:.2f}  ({pbc})"
        )
        self.save_button.setEnabled(not result.has_errors)
        self._update_title()

    def _show_error(self, message):
        self.summary_label.setText(f"Cannot generate: {message}")
        self.input_preview.setPlainText("")
        self.data_preview.setPlainText("")
        self.checks_view.setPlainText(f"ERROR  {message}")
        self.cell_label.setText("-")
        self.save_button.setEnabled(False)
        self._update_title()

    def _show_checks(self, issues, extra=""):
        lines = [f"{level:<8}{message}" for level, message in issues] or [
            "OK      no problems found"
        ]
        text = "\n".join(lines)
        if extra:
            text += "\n\n" + extra
        self.checks_view.setPlainText(text)

    # ---------------------------------------------------------------- files

    def _browse_mdff(self):
        start = os.path.dirname(self.fields["mdff_path"][0].text()) or self._last_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open moleditpy-mdff file",
            start,
            "MD force field (*.mdff.json *.json);;All files (*)",
        )
        if path:
            self.fields["mdff_path"][0].setText(path)

    def _target_paths(self, directory):
        names = writer.file_names(self.read_settings())
        return [os.path.join(directory, names["input"]), os.path.join(directory, names["data"])]

    def save_files(self, directory=None) -> bool:
        if self._generated is None or self._generated.has_errors:
            QMessageBox.warning(self, PLUGIN_NAME, "Fix the errors listed under Checks first.")
            return False
        if not directory:
            directory = QFileDialog.getExistingDirectory(
                self, "Folder for the LAMMPS files", self._last_dir
            )
            if not directory:
                return False
            existing = [p for p in self._target_paths(directory) if os.path.exists(p)]
            if existing:
                answer = QMessageBox.question(
                    self,
                    PLUGIN_NAME,
                    "Overwrite\n" + "\n".join(os.path.basename(p) for p in existing) + "\n?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return False
        try:
            paths = gen.write_files(self._generated, self.read_settings(), directory)
        except OSError as exc:
            QMessageBox.critical(self, PLUGIN_NAME, f"Could not write the files:\n{exc}")
            return False
        self._last_dir = directory
        self._saved_texts = (self._generated.input_text, self._generated.data_text)
        self._update_title()
        extra = ""
        if self._generated.topology.kind == "reaxff":
            extra = f"\nCopy the ReaxFF parameter file ({self.read_settings().get('reax_ffield')}) next to them."
        elif self._generated.topology.kind == "other":
            extra = "\nCopy the potential file next to them."
        QMessageBox.information(self, PLUGIN_NAME, "Wrote\n" + "\n".join(paths.values()) + extra)
        return True

    def _on_ctrl_s(self):
        if self._last_dir and os.path.isdir(self._last_dir):
            self.save_files(self._last_dir)
        else:
            self.save_files()

    def _is_modified(self) -> bool:
        if self._saved_texts is None or self._generated is None:
            return False
        return self._saved_texts != (self._generated.input_text, self._generated.data_text)

    def _update_title(self):
        base = f"{PLUGIN_NAME} v{PLUGIN_VERSION}"
        if self._saved_texts is not None:
            name = writer.file_names(self.read_settings())["input"]
            dirty = " *" if self._is_modified() else ""
            self.setWindowTitle(f"{name}{dirty} - {base}")
        else:
            self.setWindowTitle(base)

    def _confirm_discard(self) -> bool:
        if not self._is_modified():
            return True
        reply = QMessageBox.question(
            self,
            PLUGIN_NAME,
            "The settings changed since the files were last saved. Save them again?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.save_files(self._last_dir or None)
        return reply == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self._stop_process()
        event.accept()

    def reject(self):
        if not self._confirm_discard():
            return
        self._stop_process()
        super().reject()

    # -------------------------------------------------------------- presets

    def load_presets_from_file(self):
        self.presets_data = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self.presets_data = data
            except (OSError, ValueError) as exc:
                logger.warning("%s: could not read presets: %s", PLUGIN_NAME, exc)
        if "Default" not in self.presets_data:
            self.presets_data["Default"] = writer.default_settings()
        global_cfg = self.presets_data.get("Global", {})
        if isinstance(global_cfg, dict):
            self.lmp_edit.setText(str(global_cfg.get("lmp_path", "")))
        self.update_preset_combo()

    def update_preset_combo(self):
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        names = sorted(k for k in self.presets_data if k != "Global")
        self.preset_combo.addItems(names)
        if current in names:
            self.preset_combo.setCurrentText(current)
        elif "Default" in names:
            self.preset_combo.setCurrentText("Default")
        self.preset_combo.blockSignals(False)

    def apply_selected_preset(self):
        name = self.preset_combo.currentText()
        data = self.presets_data.get(name)
        if not isinstance(data, dict):
            return
        settings = writer.default_settings()
        settings.update({k: v for k, v in data.items() if k in settings})
        # A preset describes a protocol, not a structure: keep the force-field file.
        settings["mdff_path"] = self.fields["mdff_path"][0].text() or settings.get("mdff_path", "")
        self.apply_settings(settings)

    def save_preset_dialog(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name == "Global":
            QMessageBox.warning(self, PLUGIN_NAME, "'Global' is reserved.")
            return
        self.presets_data[name] = self.read_settings()
        self.save_presets_to_file()
        self.update_preset_combo()
        self.preset_combo.setCurrentText(name)

    def delete_preset(self):
        name = self.preset_combo.currentText()
        if name == "Default":
            QMessageBox.warning(self, PLUGIN_NAME, "The Default preset cannot be deleted.")
            return
        if name not in self.presets_data:
            return
        if (
            QMessageBox.question(self, PLUGIN_NAME, f"Delete preset '{name}'?")
            == QMessageBox.StandardButton.Yes
        ):
            del self.presets_data[name]
            self.save_presets_to_file()
            self.update_preset_combo()

    def save_presets_to_file(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self.presets_data, handle, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, PLUGIN_NAME, f"Could not save presets:\n{exc}")

    # ----------------------------------------------------------- lmp check

    def _lmp_path_changed(self):
        self.presets_data.setdefault("Global", {})["lmp_path"] = self.lmp_edit.text().strip()
        self.save_presets_to_file()
        self._refresh_lmp_label()

    def _browse_lmp(self):
        path, _ = QFileDialog.getOpenFileName(self, "LAMMPS executable", "", "All files (*)")
        if path:
            self.lmp_edit.setText(path)
            self._lmp_path_changed()

    def _refresh_lmp_label(self):
        exe = lmp_runner.find_lammps(self.lmp_edit.text())
        running = self._process is not None
        self.check_button.setEnabled(exe is not None and not running)
        if exe:
            self.lmp_label.setText(f"Found: {exe}")
            self.check_button.setToolTip("Write the files to a temporary folder and run 0 steps.")
        else:
            self.lmp_label.setText("No LAMMPS executable on PATH; the 0-step check is unavailable.")
            self.check_button.setToolTip("Install LAMMPS (lmp) or set its path on the Output tab.")

    def run_zero_step_check(self):
        if self._generated is None or self._generated.has_errors:
            QMessageBox.warning(self, PLUGIN_NAME, "Fix the errors listed under Checks first.")
            return
        exe = lmp_runner.find_lammps(self.lmp_edit.text())
        if exe is None:
            return
        settings = self.read_settings()
        directory, paths = lmp_runner.prepare_check(self._generated, settings, PLUGIN_VERSION)
        self._check_dir = directory
        note = ""
        if self._generated.topology.kind in ("reaxff", "other"):
            note = "Note: the potential file is not copied into the check folder; LAMMPS may stop there.\n"
        self._process = QProcess(self)
        self._process.setWorkingDirectory(directory)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.finished.connect(self._check_finished)
        self._show_checks(self._generated.issues, f"{note}Running {exe} in {directory} ...")
        self.preview_tabs.setCurrentWidget(self.checks_view)
        self._process.start(exe, lmp_runner.check_arguments(paths["input"]))
        QTimer.singleShot(120000, self._stop_process)
        self._refresh_lmp_label()

    def _check_finished(self, exit_code, _status=None):
        process = self._process
        output = bytes(process.readAll()).decode("utf-8", "replace") if process is not None else ""
        self._process = None
        passed, verdict = lmp_runner.interpret_output(int(exit_code), output)
        tail = "\n".join(output.splitlines()[-60:])
        header = ("PASSED  " if passed else "FAILED  ") + verdict
        issues = self._generated.issues if self._generated is not None else []
        self._show_checks(issues, f"0-step check ({self._check_dir}):\n{header}\n\n{tail}")
        self._refresh_lmp_label()

    def _stop_process(self):
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(2000)
