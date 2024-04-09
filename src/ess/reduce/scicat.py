from pathlib import Path
from threading import Lock
from typing import NewType, Optional

from scitacean import Client
from scitacean.transfer.sftp import SFTPFileTransfer

from .nexus import FilePath

ScitaceanToken = NewType('ScitaceanToken', str)
'''Token used to authenticate the scitacean client'''
ScitaceanVersion = NewType('ScitaceanVersion', str)
'''Version of the scitacean api'''
ScitaceanClient = NewType('ScitaceanClient', Client)
'''An instance of scitacean.Client that is used to fetch data from Scicat'''


_locks = {}


def sftp_ess_file_transfer() -> SFTPFileTransfer:
    return SFTPFileTransfer(host="login.esss.dk")


def create_scitacean_client(
    token: ScitaceanToken,
    file_transfer: SFTPFileTransfer,
    version: Optional[ScitaceanVersion] = None,
) -> ScitaceanClient:
    if version is None:
        version = 'v3'
    return Client.from_token(
        url=f"https://scicat.ess.eu/api/{version}",
        token=token,
        file_transfer=file_transfer,
    )


def download_scicat_file(
    dataset_id: str,
    filename: str,
    *,
    client: Client,
    target: Optional[Path] = None,
) -> FilePath:
    if target is None:
        target = Path(f'~/.cache/essreduce/{dataset_id}')

    key = (dataset_id, filename, target)
    with _locks.setdefault(key, Lock()):
        dset = client.get_dataset(dataset_id)
        dset = client.download_files(dset, target=target, select=filename)
        _locks.pop(key)
    return dset.files[0].local_path
