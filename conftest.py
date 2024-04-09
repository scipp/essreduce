import pytest
from scitacean.testing.backend import add_pytest_option as add_backend_option

pytest_plugins = ("scitacean.testing.backend.fixtures",)


def pytest_addoption(parser: pytest.Parser) -> None:
    add_backend_option(parser)
