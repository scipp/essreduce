# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

_LOCAL_CACHE_ENV_VAR = "SCIPP_DATA_DIR"
_LOCAL_REGISTRY_ENV_VAR = "SCIPP_OVERRIDE_DATA_DIR"


def make_registry(
    prefix: str,
    files: Mapping[str, str | Entry],
    *,
    version: str,
    base_url: str = "https://public.esss.dk/groups/scipp",
    retry_if_failed: int = 3,
) -> Registry:
    """Create a file registry object.

    By default, this function creates a :class:`PoochRegistry` to download files
    via HTTP from an online file store.
    This can be overridden by setting the environment variable `SCIPP_DATA_DIR` to a
    path on the local file system.
    In this case, a :class:`LocalRegistry` is returned.

    Files are specified as a dict using either the Pooch string format explicitly
    constructed :class:`Entry` objects:

        >>> from ess.reduce.data import Entry
        >>> files = {
        ...    "file1.dat": "md5:1234567890abcdef",
        ...    "file2.csv": Entry(alg="md5", chk="abcdef123456789"),
        ...    "folder/nested.dat": "blake2b:1234567890abcdef",
        ...    "zipped.zip": Entry(alg="blake2b", chk="abcdef123456789", unzip=True),
        ... }

    In the example above, the specifications for ``file1.dat`` and ``file2.csv`` are
    essentially equivalent.
    ``folder/nested.dat`` is a file in a subfolder.
    Paths like this must always use forward slashes (/) even on Windows.

    As shown above, it is possible to automatically unzip
    files by specifying ``unzip=True``.
    When calling ``registry.get_path("zipped.zip")`` the file will be unzipped and
    a path to the content is returned.
    This expects that there is only a single file in the zip archive.

    The complete path to the source file is constructed as follows:

    - Pooch: ``{base_url}/{prefix}/{version}/{name}``
    - Local: ``{SCIPP_OVERRIDE_DATA_DIR}/{prefix}/{version}/{name}``

    When using Pooch, files are downloaded to the user's cache directory.
    This can be controlled with the environment variable ``SCIPP_CACHE_DIR``.

    Parameters
    ----------
    prefix:
        Prefix to add to all file names.
    files:
        Mapping of file names to checksums or :class:`Entry` objects.
    version:
        A version string for the files.
    base_url:
        URL for the online file store.
        Ignored if the override environment variable is set.
    retry_if_failed:
        Number of retries when downloading a file.
        Ignored if the override environment variable is set.

    Returns
    -------
    :
        Either a :class:`PoochRegistry` or :class:`LocalRegistry`.
    """
    if (override := os.environ.get(_LOCAL_REGISTRY_ENV_VAR)) is not None:
        return LocalRegistry(
            _check_local_override_path(override),
            prefix,
            files,
            version=version,
            base_url=base_url,
            retry_if_failed=retry_if_failed,
        )
    return PoochRegistry(
        prefix,
        files,
        version=version,
        base_url=base_url,
        retry_if_failed=retry_if_failed,
    )


def _check_local_override_path(override: str) -> Path:
    path = Path(override)
    if not path.is_dir():
        raise ValueError(
            f"The data override path '{override}' is not a directory. If you want to "
            "download files instead, unset the environment variable "
            f"{_LOCAL_REGISTRY_ENV_VAR}."
        )
    return path


@dataclass(frozen=True, slots=True)
class Entry:
    """An entry in a registry."""

    chk: str
    """Checksum."""
    alg: str
    """Checksum algorithm."""
    unzip: bool = False
    """Whether to unzip the file."""

    @classmethod
    def from_pooch_string(cls, pooch_string: str) -> Entry:
        alg, chk = pooch_string.split(":")
        return cls(chk=chk, alg=alg)


class Registry(ABC):
    def __init__(self, files: Mapping[str, str | Entry]) -> None:
        self._files = _to_file_entries(files)

    @abstractmethod
    def get_path(self, name: str) -> Path:
        """Get the path to a file in the registry.

        Depending on the implementation, the file is downloaded if necessary.

        Note that implementations are allowed to cache return values of this method
        to avoid recomputing potentially expensive checksums.
        This usually means that the ``Registry`` object itself gets stored until the
        Python interpreter shuts down.
        However, registries are small and do not own resources.

        Parameters
        ----------
        name:
            Name of the file to get the path for.

        Returns
        -------
        :
            The Path to the file.
        """

    def _needs_unzip(self, name: str) -> bool:
        return self._files[name].unzip


class PoochRegistry(Registry):
    def __init__(
        self,
        prefix: str,
        files: Mapping[str, str | Entry],
        *,
        version: str,
        base_url: str,
        retry_if_failed: int = 3,
    ) -> None:
        self._registry = _create_pooch(
            prefix,
            files,
            version=version,
            base_url=base_url,
            retry_if_failed=retry_if_failed,
        )
        super().__init__(files)

    @cache  # noqa: B019
    def get_path(self, name: str) -> Path:
        """Get the path to a file in the registry.

        Downloads the file if necessary.
        """
        if self._needs_unzip(name):
            paths: list[str] = self._registry.fetch(  # type: ignore[assignment]
                name, processor=self._unzip_processor
            )
            return Path(_expect_single_unzipped(paths, name))
        return Path(self._registry.fetch(name))

    @property
    def _unzip_processor(self) -> Any:
        # Create a new processor on demand because reusing the same processor would
        # reuse the same output path for every file.
        return _import_pooch().Unzip()


class LocalRegistry(Registry):
    def __init__(
        self,
        source_path: Path,
        prefix: str,
        files: Mapping[str, str | Entry],
        *,
        version: str,
        base_url: str,
        retry_if_failed: int = 3,
    ) -> None:
        # Piggyback off of Pooch to determine the cache directory.
        pooch_registry = _create_pooch(
            prefix,
            files,
            version=version,
            base_url=base_url,
            retry_if_failed=retry_if_failed,
        )
        self._extract_dir = pooch_registry.path
        self._source_path = source_path.resolve().joinpath(*prefix.split("/"), version)
        super().__init__(files)

    @cache  # noqa: B019
    def get_path(self, name: str) -> Path:
        """Get the path to a file in the registry."""
        try:
            entry = self._files[name]
        except KeyError:
            raise ValueError(f"File '{name}' is not in the registry.") from None

        path = self._local_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"File '{name}' is registered but does not exist on the file system. "
                f"Expected it at '{path}'."
            )

        _check_hash(name, path, entry)

        if self._needs_unzip(name):
            return Path(
                _expect_single_unzipped(
                    self._unzip_processor(os.fspath(path), "download", None), path
                )
            )
        return path

    def _local_path(self, name: str) -> Path:
        # Split on "/" because `name` is always a POSIX-style path, but the return
        # value is a system path, i.e., it can be a Windows-style path.
        return self._source_path.joinpath(*name.split("/"))

    @property
    def _unzip_processor(self) -> Any:
        # Create a new processor on demand because reusing the same processor would
        # reuse the same output path for every file.
        return _import_pooch().Unzip(self._extract_dir)


def _import_pooch() -> Any:
    try:
        import pooch
    except ImportError:
        raise ImportError(
            "You need to install Pooch to access test and tutorial files. "
            "See https://www.fatiando.org/pooch/latest/index.html"
        ) from None

    return pooch


def _create_pooch(
    prefix: str,
    files: Mapping[str, str | Entry],
    *,
    version: str,
    base_url: str,
    retry_if_failed: int = 3,
) -> Any:
    pooch = _import_pooch()
    return pooch.create(
        path=pooch.os_cache(prefix),
        env=_LOCAL_CACHE_ENV_VAR,
        base_url=f'{base_url}/{prefix}/{version}/',
        registry=_to_pooch_registry(files),
        retry_if_failed=retry_if_failed,
    )


def _pooch_unzip_processor(extract_dir: Path) -> Any:
    try:
        import pooch
    except ImportError:
        raise ImportError("You need to install Pooch to unzip files.") from None

    return pooch.processors.Unzip(extract_dir=os.fspath(extract_dir))


def _expect_single_unzipped(paths: list[str], archive: str | os.PathLike) -> str:
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one file to unzip, got {len(paths)} in "
            f"'{os.fspath(archive)}'."
        )
    return paths[0]


def _check_hash(name: str, path: Path, entry: Entry) -> None:
    new_chk = _checksum_of_file(path, algorithm=entry.alg)
    if new_chk.lower() != entry.chk.lower():
        raise ValueError(
            f"{entry.alg} hash of file '{name}' does not match the known hash: "
            f"expected {entry.chk}, got {new_chk}."
        )


def _to_file_entries(files: Mapping[str, str | Entry]) -> dict[str, Entry]:
    return {
        name: entry if isinstance(entry, Entry) else Entry.from_pooch_string(entry)
        for name, entry in files.items()
    }


def _to_pooch_registry(files: Mapping[str, str | Entry]) -> dict[str, str]:
    return {
        name: f"{entry.alg}:{entry.chk}" if isinstance(entry, Entry) else entry
        for name, entry in files.items()
    }


# Code taken from Scitacean and Pooch.
def _checksum_of_file(path: Path, *, algorithm: str) -> str:
    """Compute the checksum of a local file.

    Parameters
    ----------
    path:
        Path of the file.
    algorithm:
        Hash algorithm to use.
        Can be any algorithm supported by :func:`hashlib.new`.

    Returns
    -------
    :
        The hex digest of the hash.
    """
    chk = _new_hash(algorithm)
    # size based on http://git.savannah.gnu.org/gitweb/?p=coreutils.git;a=blob;f=src/ioblksize.h;h=ed2f4a9c4d77462f357353eb73ee4306c28b37f1;hb=HEAD#l23  # noqa: E501
    buffer = memoryview(bytearray(128 * 1024))
    with open(path, "rb", buffering=0) as file:
        for n in iter(lambda: file.readinto(buffer), 0):
            chk.update(buffer[:n])
    return chk.hexdigest()  # type: ignore[no-any-return]


def _new_hash(algorithm: str) -> Any:
    # Try to use a named constructor instead of hashlib.new where possible
    # because that is supposed to be faster, according to
    # https://docs.python.org/3/library/hashlib.html#hashlib.new
    try:
        return getattr(hashlib, algorithm)()
    except AttributeError:
        return hashlib.new(algorithm, usedforsecurity=False)
