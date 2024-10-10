import argparse
import shutil

import h5py


def _scale_group(event_data: h5py.Group, scale: int):
    if not all(
        required_field in event_data
        for required_field in ('event_index', 'event_time_offset', 'event_id')
    ):
        return
    event_index = (event_data['event_index'][:] * scale).astype('uint')
    event_data['event_index'][:] = event_index

    size = event_data['event_id'].size
    event_data['event_id'].resize(size * scale, axis=0)
    event_data['event_time_offset'].resize(size * scale, axis=0)

    for s in range(1, scale):
        event_data['event_id'][s * size : (s + 1) * size] = event_data['event_id'][
            :size
        ]
        event_data['event_time_offset'][s * size : (s + 1) * size] = event_data[
            'event_time_offset'
        ][:size]


def _grow_nexus_file_impl(file: h5py.File, detector_scale: int, monitor_scale: int):
    for group in file.values():
        if group.attrs.get('NX_class', '') == 'NXentry':
            entry = group
            break
    for group in entry.values():
        if group.attrs.get('NX_class', '') == 'NXinstrument':
            instrument = group
            break
    for group in instrument.values():
        if (nx_class := group.attrs.get('NX_class', '')) in (
            'NXdetector',
            'NXmonitor',
        ):
            for subgroup in group.values():
                if subgroup.attrs.get('NX_class', '') == 'NXevent_data':
                    _scale_group(
                        subgroup,
                        scale=detector_scale
                        if nx_class == 'NXdetector'
                        else monitor_scale,
                    )


def grow_nexus_file(*, filename: str, detector_scale: int, monitor_scale: int | None):
    with h5py.File(filename, 'a') as f:
        _grow_nexus_file_impl(
            f,
            detector_scale,
            monitor_scale if monitor_scale is not None else detector_scale,
        )


def integer_greater_than_one(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError('Must be larger than or equal to 1')
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help=(
            'Input file name. The events in the input file will be '
            'repeated `scale` times and stored in the output file.'
        ),
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help='Output file name where the resulting nexus file will be written.',
        required=True,
    )
    parser.add_argument(
        "-s",
        "--detector-scale",
        type=integer_greater_than_one,
        help=('Scale factor to multiply the number of detector events by.'),
        required=True,
    )
    parser.add_argument(
        "-m",
        "--monitor-scale",
        type=integer_greater_than_one,
        default=None,
        help=(
            'Scale factor to multiply the number of monitor events by. '
            'If not given, the detector scale will be used'
        ),
    )
    args = parser.parse_args()
    if args.file != args.output:
        shutil.copy2(args.file, args.output)
    grow_nexus_file(
        filename=args.output,
        detector_scale=args.detector_scale,
        monitor_scale=args.monitor_scale,
    )
