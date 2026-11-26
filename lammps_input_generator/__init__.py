"""LAMMPS Input Generator plugin for MoleditPy.

Writes a LAMMPS data file and an ``in.*`` script for the current molecule or a
packed/solvated periodic system, with parameters from a ``moleditpy-mdff`` file
(Force Field Builder), RDKit UFF as a fallback, or ReaxFF / another many-body
potential the user supplies.
"""

import logging
import os

logger = logging.getLogger(__name__)

PLUGIN_NAME = "LAMMPS Input Generator"
PLUGIN_VERSION = "0.1.2"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Write a LAMMPS data file and input script (minimize, NVE, NVT, NPT) for the "
    "current molecule or periodic system, using moleditpy-mdff force fields, "
    "RDKit UFF, or a ReaxFF/other potential."
)

PLUGIN_CATEGORY = "Input Generator"
PLUGIN_TAGS = ["Generator", "MD"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = []
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_LICENSE = "LGPL-3.0-or-later"

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
WINDOW_ID = "lammps_input_generator_dialog"

_context = None
_dialog_opened = False
# Last cell seen per structure: {"cell": <moleditpy_cell JSON>, "n_atoms": int}.
# RDKit drops molecule properties in the host's undo snapshots and project
# files, so the plugin keeps its own copy (see README, "Cell").
remembered_cell = {}


def get_default_settings():
    from .writer import default_settings

    return default_settings()


def _defaults_without_import():
    # The writer imports nothing heavy, but keep module import side-effect free.
    try:
        return get_default_settings()
    except Exception:  # noqa: BLE001  # pragma: no cover - defensive
        return {}


current_settings = _defaults_without_import()


def _apply_remembered_cell(context):
    """Put the remembered cell back on the molecule when it lost the property."""
    if not remembered_cell:
        return
    try:
        mol = context.current_molecule
    except Exception:  # noqa: BLE001  # pragma: no cover - host API guard
        return
    if mol is None:
        return
    try:
        if mol.HasProp("moleditpy_cell"):
            return
        if mol.GetNumAtoms() != int(remembered_cell.get("n_atoms", -1)):
            return
        mol.SetProp("moleditpy_cell", str(remembered_cell["cell"]))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
        logger.debug("%s: could not re-apply the cell: %s", PLUGIN_NAME, exc)


def _open_dialog(mw):
    """Open the dialog (not named ``run``: the host would add a second menu entry)."""
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

    from .main_dialog import LammpsInputDialog

    def _get_molecule():
        try:
            if _context is not None:
                return _context.current_molecule
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
            logger.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
        return getattr(mw, "current_mol", None)

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
                logger.debug("%s: mark_project_modified failed: %s", PLUGIN_NAME, exc)

    _dialog_opened = True
    dlg = LammpsInputDialog(
        parent=mw,
        persistent_settings=current_settings,
        get_molecule=_get_molecule,
        mark_modified=_mark_modified,
        main_window=mw,
        remembered_cell=remembered_cell,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()


def initialize(context):
    """Entry point for the V4/V5 plugin API."""
    global _context
    _context = context

    def show_dialog():
        _open_dialog(context.get_main_window())

    context.add_export_action("LAMMPS Input (data + in script)...", show_dialog)

    def save_state():
        if not _dialog_opened:
            return {}
        state = {"settings": dict(current_settings)}
        if remembered_cell:
            state["cell"] = dict(remembered_cell)
        return state

    def load_state(data):
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if isinstance(saved, dict):
            current_settings.update(saved)
            dlg = context.get_window(WINDOW_ID)
            if dlg is not None:
                try:
                    dlg.apply_settings(current_settings)
                except Exception as exc:  # noqa: BLE001  # pragma: no cover - host API guard
                    logger.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)
        cell = data.get("cell")
        if isinstance(cell, dict) and isinstance(cell.get("cell"), str):
            remembered_cell.clear()
            remembered_cell.update(cell)
            # Load handlers run before the host restores the molecule: wait a turn.
            try:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda: _apply_remembered_cell(context))
            except Exception:  # noqa: BLE001  # pragma: no cover - no Qt event loop
                _apply_remembered_cell(context)

    def handle_reset():
        global _dialog_opened
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            return
        current_settings.clear()
        current_settings.update(get_default_settings())
        remembered_cell.clear()
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
