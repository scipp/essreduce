from pathlib import Path
from threading import Lock
from typing import NewType, Optional

from scitacean import Client

from .nexus import FilePath

ScitaceanToken = NewType('ScitaceanToken', str)
'''Token used to authenticate the scitacean client'''
ScitaceanVersion = NewType('ScitaceanVersion', str)
'''Version of the scitacean api'''
ScitaceanClient = NewType('ScitaceanClient', Client)
'''An instance of scitacean.Client that is used to fetch data from Scicat'''


_locks = {}


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
