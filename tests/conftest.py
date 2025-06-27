import json
import logging
from pathlib import Path

import pytest

from diffpy.srreal.structureconverters import convertObjCrystCrystal


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


# Resolve availability of optional packages.

# pyobjcryst


@pytest.fixture(scope="session")
def _msg_nopyobjcryst():
    return "No module named 'pyobjcryst'"


@pytest.fixture(scope="session")
def has_pyobjcryst():
    try:
        import pyobjcryst.crystal

        convertObjCrystCrystal(pyobjcryst.crystal.Crystal())
        has_pyobjcryst = True
    except ImportError:
        has_pyobjcryst = False
        logging.warning("Cannot import pyobjcryst, pyobjcryst tests skipped.")
        print("Cannot import pyobjcryst, pyobjcryst tests skipped.")
    except TypeError:
        has_pyobjcryst = False
        logging.warning("Compiled without ObjCryst, pyobjcryst tests skipped.")
        print("Compiled without ObjCryst, pyobjcryst tests skipped.")

    return has_pyobjcryst


# periodictable


@pytest.fixture(scope="session")
def _msg_noperiodictable():
    return "No module named 'periodictable'"


@pytest.fixture(scope="session")
def has_periodictable():
    try:
        import periodictable

        has_periodictable = True

        # silence the pyflakes syntax checker
        del periodictable
    except ImportError:
        has_periodictable = False
        logging.warning(
            "Cannot import periodictable, periodictable tests skipped."
        )

    return has_periodictable
