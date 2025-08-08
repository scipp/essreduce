# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from pathlib import Path


class Registry:
    """A registry for data files.

    Note
    ----
    This class requires [Pooch](https://www.fatiando.org/pooch/latest/) which
    is not a hard dependency of ESSreduce and needs to be installed separately.
    """

    def __init__(self, instrument: str, files: dict[str, str], version: str) -> None:
        import pooch

        self._registry = pooch.create(
            path=pooch.os_cache(f'ess/{instrument}'),
            env=f'ESS_{instrument.upper()}_DATA_DIR',
            base_url=f'https://public.esss.dk/groups/scipp/ess/{instrument}/'
            + '{version}/',
            version=version,
            registry=files,
            retry_if_failed=3,
        )
        self._unzip_processor = pooch.Unzip()

    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the registry."""
        return key in self._registry.registry

    def get_path(self, name: str, unzip: bool = False) -> Path:
        """
        Get the path to a file in the registry.

        Parameters
        ----------
        name:
            Name of the file to get the path for.
        unzip:
            If `True`, unzip the file before returning the path.

        Returns
        -------
        :
            The Path to the file.
        """
        return Path(
            self._registry.fetch(
                name, processor=self._unzip_processor if unzip else None
            )
        )


_bifrost_registry = Registry(
    instrument='bifrost',
    files={
        "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",
    },
    version='1',
)


_dream_registry = Registry(
    instrument='dream',
    files={
        "TEST_977695_00068064.hdf": "md5:9e6ee9ec70d7c5e8c0c93b9e07e8949f",
    },
    version='2',
)


_loki_registry = Registry(
    instrument='loki',
    files={
        # Files from LoKI@Larmor detector test experiment
        #
        # Background run 1 (no sample, sample holder/can only, no transmission monitor)
        '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
        # Sample run 1 (sample + sample holder/can, no transmission monitor in beam)
        '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
        # Sample run 2 (sample + sample holder/can, no transmission monitor in beam)
        '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
        # Background transmission run (sample holder/can + transmission monitor)
        '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
        # Background run 2 (no sample, sample holder/can only, no transmission monitor)
        '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
        # Sample transmission run (sample + sample holder/can + transmission monitor)
        '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
        # Analytical model for the I(Q) of the Poly-Gauss sample
        'PolyGauss_I0-50_Rg-60.h5': 'md5:f5d60d9c2286cb197b8cd4dc82db3d7e',
        # XML file for the pixel mask
        'mask_new_July2022.xml': 'md5:421b6dc9db74126ffbc5d88164d017b0',
    },
    version='2',
)


def bifrost_simulated_elastic() -> Path:
    """McStas simulation with elastic incoherent scattering + phonon."""
    return _bifrost_registry.get_path('BIFROST_20240914T053723.h5')


def loki_tutorial_sample_run_60250() -> Path:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return _loki_registry.get_path('60250-2022-02-28_2215.nxs')


def loki_tutorial_sample_run_60339() -> Path:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return _loki_registry.get_path('60339-2022-02-28_2215.nxs')


def loki_tutorial_background_run_60248() -> Path:
    """Background run with sample holder/can only, no transmission monitor."""
    return _loki_registry.get_path('60248-2022-02-28_2215.nxs')


def loki_tutorial_background_run_60393() -> Path:
    """Background run with sample holder/can only, no transmission monitor."""
    return _loki_registry.get_path('60393-2022-02-28_2215.nxs')


def loki_tutorial_sample_transmission_run() -> Path:
    """Sample transmission run (sample + sample holder/can + transmission monitor)."""
    return _loki_registry.get_path('60394-2022-02-28_2215.nxs')


def dream_coda_test_file() -> Path:
    """CODA file for DREAM where most pulses have been removed.

    See ``tools/shrink_nexus.py``.
    """
    return _dream_registry.get_path('TEST_977695_00068064.hdf')
