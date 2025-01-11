"""Unit tests for __version__.py."""

import diffpy.srreal


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(diffpy.srreal, "__version__")
    assert diffpy.srreal.__version__ != "0.0.0"
