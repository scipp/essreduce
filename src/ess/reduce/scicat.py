from pathlib import Path
from typing import Optional

from scitacean import Client
from scitacean.transfer.sftp import SFTPFileTransfer

from .nexus import FilePath


def download_scicat_file(
    dataset_id: str,
    filename: str,
    *,
    token: str,
    version: Optional[str] = None,
    target: Optional[Path] = None,
) -> FilePath:
    if version is None:
        version = 'v3'

    if target is None:
        target = Path(f'~/.cache/essreduce/{dataset_id}')

    client = Client.from_token(
        url=f"https://scicat.ess.eu/api/{version}",
        token=token,
        file_transfer=SFTPFileTransfer(host="login.esss.dk"),
    )
    dset = client.get_dataset(dataset_id)
    dset = client.download_files(dset, target=target, select=filename)
    for f in dset.files:
        if f.remote_path.name == filename:
            return f.local_path
