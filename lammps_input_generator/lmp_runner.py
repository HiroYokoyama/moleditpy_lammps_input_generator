"""Find a LAMMPS executable and prepare the optional 0-step check."""

from __future__ import annotations

import os
import shutil
import tempfile

from . import writer

EXECUTABLE_NAMES = ("lmp", "lmp_serial", "lmp_mpi", "lmp_omp", "lammps")


def find_lammps(preferred: str = "") -> str | None:
    """Path of a LAMMPS binary: ``preferred`` if it exists, else the first on PATH."""
    preferred = (preferred or "").strip()
    if preferred:
        if os.path.isfile(preferred):
            return preferred
        found = shutil.which(preferred)
        if found:
            return found
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def prepare_check(generated, settings: dict, version: str = "") -> tuple[str, dict[str, str]]:
    """Write the data file and a ``run 0`` script into a fresh temp directory.

    Returns (directory, {"input": path, "data": path}).  ReaxFF and other
    potentials need their potential file next to the script, which the check
    cannot supply; the caller should say so.
    """
    directory = tempfile.mkdtemp(prefix="moleditpy_lammps_check_")
    names = writer.file_names(settings)
    data_path = os.path.join(directory, names["data"])
    input_path = os.path.join(directory, "in.check")
    script = writer.build_input_script(
        generated.topology, generated.box, settings, version, check_only=True
    )
    with open(data_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(generated.data_text)
    with open(input_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(script)
    return directory, {"input": input_path, "data": data_path}


def check_arguments(input_path: str):
    return ["-in", os.path.basename(input_path), "-log", "none", "-nocite"]


def interpret_output(exit_code: int, output: str) -> tuple[bool, str]:
    """(passed, one-line verdict) from a finished 0-step run."""
    error_lines = [line for line in output.splitlines() if line.strip().startswith("ERROR")]
    if exit_code == 0 and not error_lines:
        warnings = sum(1 for line in output.splitlines() if line.strip().startswith("WARNING"))
        suffix = f" ({warnings} LAMMPS warning(s))" if warnings else ""
        return True, f"LAMMPS read the system and ran 0 steps{suffix}."
    if error_lines:
        return False, error_lines[0].strip()
    return False, f"LAMMPS exited with code {exit_code}."
