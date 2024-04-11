import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from dateutil.parser import parse as parse_date
from scitacean import Dataset, DatasetType, RemotePath
from scitacean.model import Relationship
from scitacean.testing.client import FakeClient, ScicatCommError
from scitacean.testing.transfer import FakeFileTransfer

from ess.reduce.scicat import download_scicat_file, get_related_dataset


def _checksum(data: bytes) -> str:
    checksum = hashlib.new("md5")
    checksum.update(data)
    return checksum.hexdigest()


@pytest.fixture
def files():
    return {
        "file1.dat": b"contents-of-file1",
        "log/what-happened.log": b"ERROR Flux is off the scale",
        "thaum.dat": b"0 4 2 59 330 2314552",
    }


@pytest.fixture
def local_dataset(fs, files):
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
        relationships=[Relationship(pid='123', relationship='background')],
    )
    for name, content in files.items():
        path = Path('tmp') / name
        fs.create_file(path, contents=content)
        dset.add_local_files(path, base_path='tmp')
    return dset


def test_download_scicat_file(fs, local_dataset):
    local_dataset.make_upload_model()
    transfer = FakeFileTransfer(fs=fs)
    client = FakeClient.without_login(url="https://fake.scicat", file_transfer=transfer)
    uploaded = client.upload_new_dataset_now(local_dataset)
    with TemporaryDirectory() as dname:
        path = download_scicat_file(
            client,
            uploaded.pid,
            uploaded.files[0].remote_path.posix,
            target=dname,
        )
        assert path == Path(dname) / uploaded.files[0].remote_path.posix


def test_get_related_dataset(local_dataset):
    client = FakeClient.without_login(url="https://fake.scicat")
    # Looking for comm error here because that indicates the dataset was queried for
    with pytest.raises(ScicatCommError):
        get_related_dataset(client, local_dataset, 'background')
    with pytest.raises(ValueError):
        get_related_dataset(client, local_dataset, 'reference')
