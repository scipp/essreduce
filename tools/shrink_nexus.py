# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shrink a nexus file by removing all but the first N pulses."""

import shutil
import subprocess
from pathlib import Path

import h5py as h5
import numpy as np
import scippnexus as snx

N_PULSE = 2

DIR = Path("data/coda")
IN_FNAME = "977695_00068064.hdf"
TMP_FNAME = f"TMP_{IN_FNAME}"
OUT_FNAME = f"TEST_{IN_FNAME}"


def main() -> None:
    shutil.copy(DIR / IN_FNAME, DIR / TMP_FNAME)
    with snx.File(DIR / TMP_FNAME, 'r') as f:
        entry = f['entry']
        user_names = list(entry[snx.NXuser])
        det_names = list(entry['instrument'][snx.NXdetector])
        mon_names = list(entry['instrument'][snx.NXmonitor])

    last_times = []

    with h5.File(DIR / TMP_FNAME, 'a') as f:
        entry = f['entry']
        for name in user_names:
            del entry[name]
        del entry['scripts']

        instr = entry['instrument']
        for name in det_names:
            det = instr[name]
            del det['pixel_shape']

            event_data_name = next(name for name in det if name.endswith('event_data'))
            events = det[event_data_name]

            last_times.append(
                np.array(
                    int(events['event_time_zero'][N_PULSE]),
                    dtype=f'datetime64[{events["event_time_zero"].attrs["units"]}]',
                )
            )

            n_events = events['event_index'][N_PULSE]
            slice_dataset(events, 'event_index', N_PULSE)
            slice_dataset(events, 'event_time_zero', N_PULSE)
            slice_dataset(events, 'event_id', n_events)
            slice_dataset(events, 'event_time_offset', n_events)

        last_time = np.min(last_times)
        for name in mon_names:
            mon = instr[name]
            data = mon[f'{name}_events']
            time = data['time'][()].astype(f'datetime64[{data["time"].attrs["units"]}]')
            if time.shape == (0,):
                continue
            n_times = np.argmax(time > last_time)
            slice_dataset(data, 'time', n_times)

            assert (  # noqa: S101
                data.attrs['axes'][0] == 'time'
            )  # required to slice as below:
            slice_dataset(data, 'signal', n_times)

    subprocess.check_call(['h5repack', DIR / TMP_FNAME, DIR / OUT_FNAME])  # noqa: S603, S607


def slice_dataset(group, name, n):
    attrs = group[name].attrs
    group[name] = group.pop(name)[:n]
    group[name].attrs.update(attrs)


if __name__ == "__main__":
    main()
