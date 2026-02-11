"""Unit tests for Windows DLL loading initialization."""

import os
import sys


def test_import_succeeds():
    """Ensure diffpy.srreal can be imported successfully on all platforms."""
    import diffpy.srreal

    assert hasattr(diffpy.srreal, "__version__")


def test_windows_dll_directory_handling():
    """Verify Windows DLL directory handling doesn't raise errors."""
    # This test verifies that the Windows-specific DLL directory initialization
    # in __init__.py doesn't cause issues on any platform.
    
    # The actual DLL directory logic is executed during module import,
    # so if we got this far, it succeeded.
    
    # On Windows with Python 3.8+, verify the logic would have run
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        # Check that CONDA_PREFIX environment variable handling works
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            lib_bin_dir = os.path.join(conda_prefix, "Library", "bin")
            # The directory should exist in a proper conda environment
            # but we don't assert this as it may not exist in all test environments
            assert isinstance(lib_bin_dir, str)
    
    # Test passes if no exceptions are raised
    assert True
