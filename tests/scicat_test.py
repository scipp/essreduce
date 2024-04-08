from pathlib import Path
from tempfile import TemporaryDirectory

from scitacean.testing.docs import setup_fake_client

import ess
from ess.reduce.scicat import download_scicat_file


class Client:
    @staticmethod
    def from_token(url, token, file_transfer):
        return setup_fake_client()


ess.reduce.scicat.Client = Client


def test_download_scicat_file():
    with TemporaryDirectory() as dname:
        path = download_scicat_file(
            '20.500.12269/72fe3ff6-105b-4c7f-b9d0-073b67c90ec3',
            'flux.dat',
            token='123',
            version='v3',
            target=dname,
        )
        assert path == Path(dname) / 'flux.dat'
