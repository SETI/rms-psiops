##########################################################################################
# tests/conftest.py
##########################################################################################
"""Shared pytest fixtures for the psiops test suite."""

from collections.abc import Iterator

import pytest

from psiops._filter import _use_shortcuts


@pytest.fixture(autouse=True)
def _restore_shortcuts() -> Iterator[None]:
    """Restore the global shortcut setting after every test.

    Many tests toggle the module-level `_use_shortcuts` flag to exercise both the
    optimized and the general code paths. Restoring it here keeps individual tests
    independent so they can be run in any order or in parallel.
    """

    saved = _use_shortcuts()
    yield
    _use_shortcuts(saved)


@pytest.fixture(params=[False, True], ids=['no_shortcuts', 'shortcuts'])
def shortcuts(request: pytest.FixtureRequest) -> bool:
    """Parametrize a test over both shortcut settings.

    A test that takes this fixture runs twice: once with shortcut optimizations
    disabled and once with them enabled. The global flag is restored afterward by
    the autouse `_restore_shortcuts` fixture.
    """

    _use_shortcuts(request.param)
    return request.param

##########################################################################################
