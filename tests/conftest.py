# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import pytest

from ess.reduce.data import Registry, make_registry


@pytest.fixture(scope='session')
def bifrost_registry() -> Registry:
    return make_registry(
        'ess/bifrost',
        files={
            "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",
        },
        version='1',
    )


@pytest.fixture(scope='session')
def dream_registry() -> Registry:
    return make_registry(
        'ess/dream',
        files={
            "TEST_977695_00068064.hdf": "md5:9e6ee9ec70d7c5e8c0c93b9e07e8949f",
        },
        version='2',
    )


@pytest.fixture(scope='session')
def loki_registry() -> Registry:
    return make_registry(
        'ess/loki',
        files={
            # Files from LoKI@Larmor detector test experiment
            #
            # Background run 1 (no sample, sample holder/can only, no transmission monitor)  # noqa: E501
            '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
            # Sample run 1 (sample + sample holder/can, no transmission monitor in beam)
            '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
            # Sample run 2 (sample + sample holder/can, no transmission monitor in beam)
            '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
            # Background transmission run (sample holder/can + transmission monitor)
            '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
            # Background run 2 (no sample, sample holder/can only, no transmission monitor)  # noqa: E501
            '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
            # Sample transmission run (sample + sample holder/can + transmission monitor)  # noqa: E501
            '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
            # Analytical model for the I(Q) of the Poly-Gauss sample
            'PolyGauss_I0-50_Rg-60.h5': 'md5:f5d60d9c2286cb197b8cd4dc82db3d7e',
            # XML file for the pixel mask
            'mask_new_July2022.xml': 'md5:421b6dc9db74126ffbc5d88164d017b0',
        },
        version='2',
    )


@pytest.fixture(scope='session')
def tbl_registry() -> Registry:
    return make_registry(
        'ess/tbl',
        files={
            "857127_00000112_small.hdf": "md5:0db89493b859dbb2f7354c3711ed7fbd",
        },
        version='2',
    )


@pytest.fixture(scope='session')
def bifrost_simulated_elastic(bifrost_registry: Registry) -> Path:
    """McStas simulation with elastic incoherent scattering + phonon."""
    return bifrost_registry.get_path('BIFROST_20240914T053723.h5')


@pytest.fixture(scope='session')
def loki_tutorial_sample_run_60250(loki_registry: Registry) -> Path:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return loki_registry.get_path('60250-2022-02-28_2215.nxs')


@pytest.fixture(scope='session')
def loki_tutorial_sample_run_60339(loki_registry: Registry) -> Path:
    """Sample run with sample and sample holder/can, no transmission monitor in beam."""
    return loki_registry.get_path('60339-2022-02-28_2215.nxs')


@pytest.fixture(scope='session')
def loki_tutorial_background_run_60248(loki_registry: Registry) -> Path:
    """Background run with sample holder/can only, no transmission monitor."""
    return loki_registry.get_path('60248-2022-02-28_2215.nxs')


@pytest.fixture(scope='session')
def loki_tutorial_background_run_60393(loki_registry: Registry) -> Path:
    """Background run with sample holder/can only, no transmission monitor."""
    return loki_registry.get_path('60393-2022-02-28_2215.nxs')


@pytest.fixture(scope='session')
def loki_tutorial_sample_transmission_run(loki_registry: Registry) -> Path:
    """Sample transmission run (sample + sample holder/can + transmission monitor)."""
    return loki_registry.get_path('60394-2022-02-28_2215.nxs')


@pytest.fixture(scope='session')
def dream_coda_test_file(dream_registry: Registry) -> Path:
    """CODA file for DREAM where most pulses have been removed.

    See ``tools/shrink_nexus.py``.
    """
    return dream_registry.get_path('TEST_977695_00068064.hdf')


@pytest.fixture(scope='session')
def tbl_commissioning_orca_file(tbl_registry: Registry) -> Path:
    """TBL file from cold commissioning with the ORCA detector."""
    return tbl_registry.get_path('857127_00000112_small.hdf')
