"""Metadata, registration and session-persistence tests."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lammps_input_generator as plugin


class FakeMol:
    def __init__(self, n_atoms, props=None):
        self.n = n_atoms
        self.props = dict(props or {})

    def GetNumAtoms(self):
        return self.n

    def HasProp(self, key):
        return key in self.props

    def GetProp(self, key):
        return self.props[key]

    def SetProp(self, key, value):
        self.props[key] = value


class FakeContext:
    def __init__(self):
        self.export_actions = []
        self.menu_actions = []
        self.analysis_tools = []
        self.save_handlers = []
        self.load_handlers = []
        self.reset_handlers = []
        self.windows = {}
        self.current_molecule = None
        self.main_window = None

    def add_menu_action(self, path, callback):
        self.menu_actions.append((path, callback))

    def add_export_action(self, label, callback):
        self.export_actions.append((label, callback))

    def add_analysis_tool(self, label, callback):
        self.analysis_tools.append((label, callback))

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def register_window(self, window_id, window):
        self.windows[window_id] = window

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def get_main_window(self):
        return self.main_window

    def mark_project_modified(self):
        pass


@pytest.fixture
def context():
    original = dict(plugin.current_settings)
    ctx = FakeContext()
    plugin.initialize(ctx)
    yield ctx
    plugin._context = None
    plugin._dialog_opened = False
    plugin.current_settings.clear()
    plugin.current_settings.update(original)
    plugin.remembered_cell.clear()


def test_required_metadata():
    assert plugin.PLUGIN_NAME == "LAMMPS Input Generator"
    assert plugin.PLUGIN_VERSION == "0.1.1"
    assert plugin.PLUGIN_AUTHOR == "HiroYokoyama"
    assert plugin.PLUGIN_LICENSE == "LGPL-3.0-or-later"
    assert plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION == ">=5.0.0, <6.0.0"
    assert plugin.PLUGIN_CATEGORY == "Input Generator"
    assert plugin.PLUGIN_DESCRIPTION.strip()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)


def test_no_extra_dependencies():
    assert plugin.PLUGIN_DEPENDENCIES == []
    assert plugin.PLUGIN_OPTIONAL_DEPENDENCIES == []
    assert 1 <= len(plugin.PLUGIN_TAGS) <= 3


def test_supported_version_range():
    packaging = pytest.importorskip("packaging.specifiers")
    supported = packaging.SpecifierSet(plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION)
    assert supported.contains("5.3.2")
    assert not supported.contains("6.0.0")


def test_no_module_level_run_or_autorun():
    assert not hasattr(plugin, "run")
    assert not hasattr(plugin, "autorun")


def test_package_import_does_not_import_qt():
    """The host imports every plugin at start-up; the dialog loads only on demand."""
    with open(plugin.__file__, encoding="utf-8") as handle:
        src = handle.read()
    top_level = [line for line in src.splitlines() if line.startswith(("import ", "from "))]
    assert not any("PySide6" in line or "main_dialog" in line for line in top_level)


def test_one_entry_point(context):
    entries = context.menu_actions + context.export_actions + context.analysis_tools
    assert len(entries) == 1
    assert entries[0][0].startswith("LAMMPS Input")


def test_session_handlers_registered(context):
    assert len(context.save_handlers) == 1
    assert len(context.load_handlers) == 1
    assert len(context.reset_handlers) == 1


def test_save_is_silent_until_used(context):
    assert context.save_handlers[0]() == {}


def test_save_includes_settings_and_remembered_cell(context):
    plugin._dialog_opened = True
    plugin.current_settings["steps"] = 1234
    plugin.remembered_cell.update({"cell": '{"lattice": [[1,0,0],[0,1,0],[0,0,1]]}', "n_atoms": 3})
    state = context.save_handlers[0]()
    assert state["settings"]["steps"] == 1234
    assert state["cell"]["n_atoms"] == 3
    json.dumps(state)  # must be JSON-serialisable for the project file


def test_load_restores_settings_and_ignores_junk(context):
    before = dict(plugin.current_settings)
    for junk in (None, {}, {"settings": "x"}, {"cell": 5}):
        context.load_handlers[0](junk)
    assert plugin.current_settings == before
    context.load_handlers[0]({"settings": {"steps": 77}})
    assert plugin.current_settings["steps"] == 77


def test_remembered_cell_is_reapplied_to_a_matching_molecule(context):
    cell = json.dumps({"lattice": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]})
    plugin.remembered_cell.update({"cell": cell, "n_atoms": 3})
    context.current_molecule = FakeMol(3)
    plugin._apply_remembered_cell(context)
    assert context.current_molecule.GetProp("moleditpy_cell") == cell


def test_remembered_cell_not_applied_to_other_structures(context):
    plugin.remembered_cell.update({"cell": "{}", "n_atoms": 3})
    context.current_molecule = FakeMol(4)
    plugin._apply_remembered_cell(context)
    assert not context.current_molecule.HasProp("moleditpy_cell")
    existing = FakeMol(3, {"moleditpy_cell": "kept"})
    context.current_molecule = existing
    plugin._apply_remembered_cell(context)
    assert existing.GetProp("moleditpy_cell") == "kept"


def test_reset_restores_defaults(context):
    plugin._dialog_opened = True
    plugin.current_settings["steps"] = 9
    plugin.remembered_cell.update({"cell": "{}", "n_atoms": 1})
    context.reset_handlers[0]()
    assert plugin.current_settings == plugin.get_default_settings()
    assert plugin.remembered_cell == {}
    assert plugin._dialog_opened is False


def test_reset_leaves_open_dialog_alone(context):
    class _Dialog:
        def isVisible(self):
            return True

    context.windows[plugin.WINDOW_ID] = _Dialog()
    plugin.current_settings["steps"] = 9
    context.reset_handlers[0]()
    assert plugin.current_settings["steps"] == 9


def test_load_survives_a_deleted_dialog(context):
    class _Dialog:
        def apply_settings(self, settings):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    context.windows[plugin.WINDOW_ID] = _Dialog()
    context.load_handlers[0]({"settings": {"steps": 5}})
    assert plugin.current_settings["steps"] == 5
