##########################################################################################
# tests/conftest.py
##########################################################################################
"""Shared pytest fixtures for the psiops test suite."""

from collections.abc import Iterator

import pytest

from psiops import _filter
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


@pytest.fixture(autouse=True)
def _restore_usable_bytes() -> Iterator[None]:
    """Restore the global memory-limit setting after every test.

    Several tests in test_utils.py set the module-level `_USABLE_BYTES` flag (via
    `_usable_bytes()`) to force the multi-layer / tiled filter code paths. Restoring the
    raw module attribute here keeps individual tests independent so they can be run in any
    order or in parallel, mirroring `_restore_shortcuts`. The raw attribute (not the
    resolved return value of `_usable_bytes()`) is saved so the `None` "re-query system
    memory" default is preserved.
    """

    saved = _filter._USABLE_BYTES
    yield
    _filter._USABLE_BYTES = saved


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
