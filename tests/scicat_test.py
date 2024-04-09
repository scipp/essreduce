import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from dateutil.parser import parse as parse_date
from scitacean import Dataset, DatasetType, RemotePath
from scitacean.model import DownloadDataFile
from scitacean.testing.client import FakeClient
from scitacean.testing.transfer import FakeFileTransfer

from ess.reduce.scicat import download_scicat_file


def _checksum(data: bytes) -> str:
    checksum = hashlib.new("md5")
    checksum.update(data)
    return checksum.hexdigest()


@pytest.fixture
def data_files():
    contents = {
        "file1.dat": b"contents-of-file1",
        "log/what-happened.log": b"ERROR Flux is off the scale",
        "thaum.dat": b"0 4 2 59 330 2314552",
    }
    files = [
        DownloadDataFile(
            path=name,
            size=len(content),
            chk=_checksum(content),
            time=parse_date("1995-08-06T14:14:14"),
        )
        for name, content in contents.items()
    ]
    return files, contents


@pytest.fixture
def dataset(fs, data_files):
    _, contents = data_files

    dset = Dataset(
        contact_email="p.stibbons@uu.am",
        creation_time=parse_date("1995-08-06T14:14:14"),
        owner="pstibbons",
        owner_group="faculty",
        principal_investigator="m.ridcully@uu.am",
        source_folder=RemotePath("/src/stibbons/774"),
        creation_location='DTU',
        type=DatasetType.RAW,
        meta={
            "height": {"value": 0.3, "unit": "m"},
            "mass": "hefty",
        },
    )
    for name, content in contents.items():
        path = Path('tmp') / name
        fs.create_file(path, contents=content)
        dset.add_local_files(path, base_path='tmp')

    return dset


def test_download_scicat_file_(fs, dataset):
    transfer = FakeFileTransfer(fs=fs)
    client = FakeClient.without_login(url="https://fake.scicat", file_transfer=transfer)
    uploaded = client.upload_new_dataset_now(dataset)
    with TemporaryDirectory() as dname:
        # Should probably return dset instead of file path...
        path = download_scicat_file(
            uploaded.pid,
            uploaded.files[0].remote_path.posix,
            client=client,
            target=dname,
        )
        assert path == Path(dname) / uploaded.files[0].remote_path.posix
