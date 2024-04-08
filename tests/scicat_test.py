from pathlib import Path
from tempfile import TemporaryDirectory

from scitacean.testing.backend.fixtures import (  # noqa: F401
    client,
    require_scicat_backend,
)

from ess.reduce.scicat import download_scicat_file


def test_download_scicat_file(require_scicat_backend, client):  # noqa: F811
    with TemporaryDirectory() as dname:
        path = download_scicat_file(
            '20.500.12269/72fe3ff6-105b-4c7f-b9d0-073b67c90ec3',
            'flux.dat',
            client=client,
            target=dname,
        )
        assert path == Path(dname) / 'flux.dat'
