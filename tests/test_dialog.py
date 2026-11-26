"""Dialog behaviour with real PySide6 widgets (offscreen)."""

import json
import os

import pytest
from helpers import EXAMPLE_MDFF, embedded, example_molecule

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("rdkit")
pytest.importorskip("numpy")

from lammps_input_generator import (
    main_dialog,
    writer,
)


@pytest.fixture
def make_dialog(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(main_dialog, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(main_dialog.lmp_runner, "find_lammps", lambda preferred="": None)
    monkeypatch.setattr(main_dialog.QMessageBox, "information", lambda *a, **k: None)
    created = []

    def factory(mol=None, settings=None, remembered=None):
        state = {"mol": mol, "modified": 0}

        def mark():
            state["modified"] += 1

        dlg = main_dialog.LammpsInputDialog(
            persistent_settings=settings if settings is not None else writer.default_settings(),
            get_molecule=lambda: state["mol"],
            mark_modified=mark,
            remembered_cell=remembered if remembered is not None else {},
        )
        dlg.state = state
        created.append(dlg)
        return dlg

    yield factory
    for dlg in created:
        dlg._saved_texts = None
        dlg.close()
        dlg.deleteLater()


def _set(dlg, key, value):
    widget, kind = dlg.fields[key]
    if kind == "combo":
        widget.setCurrentIndex(widget.findData(value))
    elif kind == "check":
        widget.setChecked(value)
    elif kind == "line":
        widget.setText(value)
    else:
        widget.setValue(value)
    dlg.update_preview()


def test_preview_is_filled_for_a_molecule(make_dialog):
    dlg = make_dialog(embedded("CCO"))
    assert "read_data       system.data" in dlg.input_preview.toPlainText()
    assert "Atoms # full" in dlg.data_preview.toPlainText()
    assert "atoms" in dlg.summary_label.text()
    assert dlg.save_button.isEnabled()
    assert not dlg.check_button.isEnabled()  # no lmp in the test environment


def test_no_molecule_shows_a_message(make_dialog):
    dlg = make_dialog(None)
    assert "No molecule" in dlg.summary_label.text()
    assert not dlg.save_button.isEnabled()


def test_settings_round_trip_and_persist(make_dialog):
    settings = writer.default_settings()
    dlg = make_dialog(embedded("CCO"), settings)
    _set(dlg, "ensemble", "npt")
    _set(dlg, "temperature", 350.0)
    assert settings["ensemble"] == "npt" and settings["temperature"] == 350.0
    assert "npt temp 350 350" in dlg.input_preview.toPlainText()
    assert dlg.state["modified"] > 0
    read = dlg.read_settings()
    assert set(read) <= set(writer.default_settings())
    other = make_dialog(embedded("CCO"))
    other.apply_settings(read)
    assert other.read_settings() == read


def test_visibility_follows_the_force_field_source(make_dialog):
    dlg = make_dialog(embedded("CO"))
    dlg.show()
    _set(dlg, "ff_source", "reaxff")
    assert dlg.reax_group.isVisible() and not dlg.uff_group.isVisible()
    assert "reaxff NULL" in dlg.input_preview.toPlainText()
    _set(dlg, "ff_source", "mdff")
    assert dlg.mdff_group.isVisible()
    assert "Choose a moleditpy-mdff" in dlg.summary_label.text()


def test_mdff_file_in_the_dialog(make_dialog):
    dlg = make_dialog(example_molecule())
    _set(dlg, "ff_source", "mdff")
    _set(dlg, "mdff_path", EXAMPLE_MDFF)
    assert "fix             constrain all shake" in dlg.input_preview.toPlainText()
    assert "OK" in dlg.checks_view.toPlainText() or "INFO" in dlg.checks_view.toPlainText()


def test_cell_property_is_remembered(make_dialog):
    mol = embedded("O")
    mol.SetProp("moleditpy_cell", json.dumps({"lattice": [[20, 0, 0], [0, 20, 0], [0, 0, 20]]}))
    remembered = {}
    make_dialog(mol, remembered=remembered)
    assert remembered["n_atoms"] == 3
    # an undo drops the property; the remembered cell takes over
    mol.ClearProp("moleditpy_cell")
    dlg = make_dialog(mol, remembered=remembered)
    assert "remembered" in dlg.cell_label.text()
    assert "20.000" in dlg.cell_label.text()


def test_save_files_and_dirty_tracking(make_dialog, tmp_path):
    dlg = make_dialog(embedded("CCO"))
    out = tmp_path / "out"
    out.mkdir()
    assert dlg.save_files(str(out))
    assert sorted(os.listdir(out)) == ["in.system", "system.data"]
    assert not dlg._is_modified()
    assert dlg.windowTitle().startswith("in.system - ")
    _set(dlg, "steps", 500)
    assert dlg._is_modified()
    assert "*" in dlg.windowTitle()


def test_presets_save_apply_delete(make_dialog, monkeypatch):
    dlg = make_dialog(embedded("CCO"))
    _set(dlg, "steps", 4242)
    monkeypatch.setattr(main_dialog.QInputDialog, "getText", lambda *a, **k: ("long run", True))
    dlg.save_preset_dialog()
    with open(main_dialog.SETTINGS_FILE, encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["long run"]["steps"] == 4242 and "Default" in stored
    dlg.preset_combo.setCurrentText("Default")
    dlg.apply_selected_preset()
    assert dlg.fields["steps"][0].value() == writer.default_settings()["steps"]
    dlg.preset_combo.setCurrentText("long run")
    dlg.apply_selected_preset()
    assert dlg.fields["steps"][0].value() == 4242
    monkeypatch.setattr(
        main_dialog.QMessageBox,
        "question",
        lambda *a, **k: main_dialog.QMessageBox.StandardButton.Yes,
    )
    dlg.delete_preset()
    assert "long run" not in dlg.presets_data


def test_global_lmp_path_is_stored(make_dialog):
    dlg = make_dialog(embedded("C"))
    dlg.lmp_edit.setText("C:/lammps/lmp.exe")
    dlg._lmp_path_changed()
    with open(main_dialog.SETTINGS_FILE, encoding="utf-8") as handle:
        assert json.load(handle)["Global"]["lmp_path"] == "C:/lammps/lmp.exe"
    again = make_dialog(embedded("C"))
    assert again.lmp_edit.text() == "C:/lammps/lmp.exe"


def test_large_data_preview_is_truncated(make_dialog, monkeypatch):
    monkeypatch.setattr(main_dialog, "PREVIEW_LINES", 20)
    dlg = make_dialog(embedded("CCCCCC"))
    assert "more lines (the saved file is complete)" in dlg.data_preview.toPlainText()


def _assert_tabs_scroll_and_ignore_the_wheel(dlg):
    """Long settings tabs scroll (the dialog fits a laptop screen) and an unfocused
    field does not change when the wheel scrolls over it."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QDoubleSpinBox,
        QScrollArea,
        QSpinBox,
    )

    assert dlg.minimumSizeHint().height() < 600
    pages = [a.widget() for a in dlg.findChildren(QScrollArea) if a.widget() is not None]
    assert len(pages) >= 3
    spins = [
        s
        for page in pages
        for s in page.findChildren(QAbstractSpinBox)
        if isinstance(s, (QSpinBox, QDoubleSpinBox))
        and s.isEnabled()
        and s.value() < s.maximum()
    ]
    spin = spins[0]
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    before = spin.value()
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(spin, event)
    assert spin.value() == before


def test_settings_tabs_scroll_and_ignore_the_wheel(make_dialog):
    _assert_tabs_scroll_and_ignore_the_wheel(make_dialog(embedded("CCO")))
