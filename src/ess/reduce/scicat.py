from pathlib import Path
from threading import Lock
from typing import Optional

from scitacean import Client, Dataset

from .nexus import FilePath

_file_download_locks = {}


def download_scicat_file(
    client: Client,
    dataset_id: str,
    filename: str,
    *,
    target: Optional[Path] = None,
) -> FilePath:
    if target is None:
        target = Path(f'~/.cache/essreduce/{dataset_id}')
    key = (dataset_id, filename, target)
    with _file_download_locks.setdefault(key, Lock()):
        dset = client.get_dataset(dataset_id)
        dset = client.download_files(dset, target=target, select=filename)
        _file_download_locks.pop(key)
    return dset.files[0].local_path


def get_related_dataset(client: Client, ds: Dataset, relationship: str) -> Dataset:
    '''Goes through the datasets related to 'ds'
    and finds the one with the selected relation'''
    for d in getattr(ds, 'relationships', ()):
        if d.relationship == relationship:
            return client.get_dataset(d.pid)
    raise ValueError(
        f'The requested relation "{relationship}" was not found in dataset {ds}'
    )
