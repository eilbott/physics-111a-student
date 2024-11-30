'''Functions for Lab 10 heart rate scan, based on python WaveForms demo.'''

from typing import TypeAlias, Any, Optional
from enum import IntEnum, StrEnum, unique
from functools import partial
import time

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from cycler import cycler, Cycler

from pydwf import (
    DwfLibrary,
    DwfEnumConfigInfo,
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAcquisitionMode,
    DwfTriggerSource,
    DwfAnalogInTriggerType,
    DwfTriggerSlope,
    DwfState,
    DwfAnalogInFilter,
    AnalogIn,
    AnalogOut
)
from pydwf.utilities import openDwfDevice

__all__ = ['ChannelIndex', 'ScanType', 'create_figure', 'scan_heartbeat']

@unique
class ChannelIndex(IntEnum):
    '''Enum for the two scope channel indices.'''
    CH1 = 0
    CH2 = 1

@unique
class ScanType(StrEnum):
    '''Enum for the three different scan types.'''
    TEST = 'TEST'
    DEMOD = 'DEMOD'
    LOCKIN = 'LOCKIN'

RangeLimits: TypeAlias = Optional[tuple[float, float]]

LINE_LABELS = ('Channel 1', 'Channel 2', 'Demod line')

def create_figure(scan_type: ScanType, figsize: RangeLimits = None) -> tuple[Figure, NDArray]:
    '''
    Create a figure suitable for passing to ``scan_heartbeat()``.

    Parameters
    ----------
    scan_type: str
        One of ``ScanType.TEST``, ``ScanType.DEMOD``, ``ScanType.LOCKIN``.
    figsize: tuple[float, float] or None
        Width and height of the figure.

    Returns
    -------
    fig: Figure
        Figure that will contain all plots.
    axs: NDArray of Axes
        Axes array, one ``Axes`` object for each plot in the figure.
    '''

    # Create figures and axes
    if scan_type == ScanType.TEST:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs[0].set(xlabel='Time relative to trigger (s)', ylabel='Signal (V)', title='Acquisition')
        axs[1].set(xlabel='Frequency (Hz)', ylabel='Magnitude (V)', title='FFT')
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs[0, 0].set(xlabel='Time relative to trigger (s)', ylabel='Signal (V)', title='Acquisition')
        axs[0, 1].set(xlabel='Time relative to trigger (s)', ylabel='Signal (V)', title='Acquisition')
        axs[1, 0].set(xlabel='Frequency (Hz)', ylabel='Magnitude (V)', title='FFT')
        axs[1, 1].set(xlabel='Frequency (Hz)', ylabel='Magnitude (V)', title='FFT')

    for ax in fig.axes:
        ax.grid()

    fig.tight_layout()

    return fig, axs

def scan_heartbeat(
    scan_type: ScanType,
    fig: Optional[Figure],
    axs: Optional[NDArray],
    test_output: bool = False,
    num_records: int = 1,
    analog_out_frequency: float = 10.0,
    analog_out_amplitude: float = 1.0,
    analog_out_offset: float = 0.0,
    analog_out_shape: DwfAnalogOutFunction = DwfAnalogOutFunction.Sine,
    analog_in_channel_indices: tuple[ChannelIndex, ...] = (ChannelIndex.CH1, ChannelIndex.CH2),
    analog_in_channel_ranges: tuple[float, ...] = (5.0, 5.0),
    analog_in_frequency: Optional[float] = None,
    record_length_time: Optional[float] = 10.0,
    record_length_samples: Optional[int] = 16384,
    analog_out_channel_index: ChannelIndex = ChannelIndex.CH1,
    analog_out_symmetry: float = 50.0,
    analog_out_node: DwfAnalogOutNode = DwfAnalogOutNode.Carrier,
    analog_in_trigger_flag: bool = True,
    analog_in_trigger_source: DwfTriggerSource = DwfTriggerSource.AnalogOut1,
    analog_in_trigger_level: float = 0.0,
    analog_in_trigger_channel: ChannelIndex = ChannelIndex.CH1,
    overplot_fft: bool = True,
    xlim_time_raw: RangeLimits = None,
    xlim_freq_raw: RangeLimits = None,
    xlim_time_result: RangeLimits = None,
    xlim_freq_result: RangeLimits = None,
    cycler_time_raw: Optional[Cycler] = None,
    cycler_freq_raw: Optional[Cycler] = None,
    cycler_time_result: Optional[Cycler] = None,
    cycler_freq_result: Optional[Cycler] = None
    ) -> Optional[tuple[Figure, NDArray]]:
    '''
    Perform a heartbeat scan and produce plots.

    If either `fig` or `axs` is ``None``, a new ``Figure`` will be created
    automatically, but plots made in notebooks may not update dynamically.

    Exactly two of `analog_in_frequency`, `record_length_time`, and
    `record_length_samples` must be specified (not ``None``).

    Parameters
    ----------
    scan_type: str
        One of ``ScanType.TEST``, ``ScanType.DEMOD``, ``ScanType.LOCKIN``.
    fig: Figure or None
        Figure created by ``create_figure()`` that will contain all plots.
    axs: NDArray of Axes or None
        Axes array created by ``create_figure()``, one ``Axes`` object for each
        plot in the figure.
    test_output: bool
        If ``True``, do not scan for heartbeat; only output a signal with the
        given parameters until user keyboard interrupt.
    num_records: int
        The number of full records to take.
    analog_out_frequency, analog_out_amplitude, analog_out_offset: float
        The frequency (Hz), amplitude (V), and offset (V) of the Wavegen.
    analog_out_shape: DwfAnalogOutFunction
        The wave shape, e.g. ``DwfAnalogOutFunction.Sine``,
        ``DwfAnalogOutFunction.DC``, ``DwfAnalogOutFunction.Triangle``,
        ``DwfAnalogOutFunction.Square``. See pydwf documentation for full list.
    analog_in_channel_indices, analog_in_channel_ranges: tuple
        A tuple of the scope channel indices and a tuple of the corresponding
        scope channel measurement ranges in volts. See the appendix in the lab
        for more information on channel ranges. Channel 1 is
        ``ChannelIndices.CH1`` and Channel 2 is ``ChannelIndices.CH2``.
        For example, to use only Channel 1 and to set the range to 5 V, set
        ``channel_indices=(ChannelIndices.CH1,)`` and ``channel_ranges=(5,)``.
        The first channel in the tuple is expected to be the one used to
        measure the output signal, and the second channel in the tuple is
        expected to be the one used to measure the Wavegen.
    analog_in_frequency: float or None
        The sampling frequency (Hz).
    record_length_time: float or None
        The time (s) for one record.
    record_length_samples: int or None
        The number of samples to take for one record.

    Returns
    -------
    tuple[Figure, NDArray[Axes]] or None
        A ``Figure`` and array of ``Axes``, or ``None`` if `test_output` is ``True``.

    Other AnalogOut Parameters
    --------------------------
    analog_out_channel_index: ChannelIndex
         The index corresponding to the desired output channel.
    analog_out_symmetry: float
        Symmetry parameter between 0 and 100 that modifies the wave shape.
    analog_out_node: DwfAnalogOutNode
        Node corresponding to a parameter column of the modulation tab in the
        Wavegen GUI.

    Other AnalogIn Parameters
    -------------------------
    analog_in_trigger_flag: bool
        Specifies whether or not to use a trigger.
    analog_in_trigger_source: DwfTriggerSource
        Instrument to trigger on.
    analog_in_trigger_level: float
        Trigger level (V).
    analog_in_trigger_channel: ChannelIndex
        Index corresponding to desired channel to trigger on.

    Other Plotting Parameters
    -------------------------
    overplot_fft: bool
        If ``True``, plot every new FFT line on top of the previous lines.
        If ``False``, remove previous line before plotting a new one.
    xlim_time_raw, xlim_freq_raw, xlim_time_result, xlim_freq_result: tuple[float, float] or None
        The x-limits for the raw/result time/FFT plots. If ``None``, will be
        chosen automatically.
    cycler_time_raw, cycler_freq_raw, cycler_time_result, cycler_freq_result: Cycler or None
        Property cycle ``Cycler`` objects for the various axes. Change the
        defaults to set, for example, line color cycles. If `cycler_time_raw` is
        ``None``, it will be set according to `analog_in_channel_indices`.
        The same is true for `cycler_time_result`. Use these instead of
        ``Axes.set_prop_cycle()`` or ``plt.rcParams['axes.prop_cycle']``;
        properties are reset before every acquisition.
    '''

    if not test_output:

        if fig is None or axs is None:
            fig, axs = create_figure(scan_type=scan_type)
        else:
            _check_axs(scan_type, axs)

        analog_in_frequency, record_length_time, record_length_samples = _calc_scope_params(
            analog_in_frequency, record_length_time, record_length_samples
        )

        if scan_type in (ScanType.DEMOD, ScanType.LOCKIN) and len(analog_in_channel_indices) != 2:
            raise ValueError(f'scan_type {scan_type} expects analog_in_channel_indices of length 2')

        cycler_time_raw, cycler_time_result, cycler_freq_raw, cycler_freq_result = _set_color_cycles(
            analog_in_channel_indices,
            cycler_time_raw,
            cycler_time_result,
            cycler_freq_raw,
            cycler_freq_result
        )

    with openDwfDevice(
        DwfLibrary(),
        score_func=lambda conf_params: conf_params[DwfEnumConfigInfo.AnalogInBufferSize]
    ) as device:

        print(f'Configuring analog output signal ({analog_out_frequency} Hz)...')

        _configure_analog_out(
            analog_out=device.analogOut,
            channel_index=analog_out_channel_index,
            frequency=analog_out_frequency,
            amplitude=analog_out_amplitude,
            offset=analog_out_offset,
            shape=analog_out_shape,
            symmetry=analog_out_symmetry,
            node=analog_out_node
        )

        print('Started analog output.')

        if test_output:
            print('Output running continuously. Interrupt program to quit.')
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print('Program quit successfully.')
                return None

        # Wait for a bit to ensure the stability of the analog output signals
        time.sleep(1.0)

        _configure_analog_in(
            analog_in=device.analogIn,
            frequency=analog_in_frequency,
            record_length_time=record_length_time,
            channel_indices=analog_in_channel_indices,
            channel_ranges=analog_in_channel_ranges,
            trigger_flag=analog_in_trigger_flag,
            trigger_source=analog_in_trigger_source,
            trigger_level=analog_in_trigger_level,
            trigger_channel=analog_in_trigger_channel
        )

        return _run_acquistion(
            analog_in=device.analogIn,
            scan_type=scan_type,
            fig=fig,
            axs=axs,
            num_records=num_records,
            analog_out_frequency=analog_out_frequency,
            analog_in_frequency=analog_in_frequency,
            record_length_time=record_length_time,
            record_length_samples=record_length_samples,
            channel_indices=analog_in_channel_indices,
            analog_in_trigger_flag=analog_in_trigger_flag,
            overplot_fft=overplot_fft,
            xlim_time_raw=xlim_time_raw,
            xlim_freq_raw=xlim_freq_raw,
            xlim_time_result=xlim_time_result,
            xlim_freq_result=xlim_freq_result,
            cycler_time_raw=cycler_time_raw,
            cycler_freq_raw=cycler_freq_raw,
            cycler_time_result=cycler_time_result,
            cycler_freq_result=cycler_freq_result
        )

def _check_axs(scan_type: ScanType, axs: NDArray) -> None:
    '''Check that `axs` is appropriate for the given `scan_type`.'''

    if (
        scan_type == ScanType.TEST
        and (len(axs.shape) != 1 or axs.shape[0] != 2)
    ):
        raise ValueError(
            f'Shape of axs must be (2,) for scan_type {scan_type}, '
            f'got shape {axs.shape} instead.'
        )

    if (
        (scan_type in (ScanType.DEMOD, ScanType.LOCKIN))
        and (len(axs.shape) != 2 or axs.shape[0] != 2 or axs.shape[1] != 2)
    ):
        raise ValueError(
            f'Shape of axs must be (2, 2) for scan_type {scan_type}, '
            f'got shape {axs.shape} instead.'
        )

def _calc_scope_params(
    analog_in_frequency: float,
    record_length_time: float,
    record_length_samples: int
    ) -> tuple[float, float, int]:
    '''Validate and calculate the scope parameters.'''

    if (
        (analog_in_frequency is None)
        + (record_length_time is None)
        + (record_length_samples is None) != 1
    ):
        raise ValueError(
            'Exactly two of analog_in_frequency, record_length_time, '
            'and record_length_samples must be specified (not None).'
        )

    if analog_in_frequency is None:
        analog_in_frequency = record_length_samples / record_length_time
    elif record_length_time is None:
        record_length_time = record_length_samples / analog_in_frequency
    else:
        record_length_samples = round(analog_in_frequency * record_length_time)
        # Recalculate after rounding
        record_length_time = record_length_samples / analog_in_frequency

    return analog_in_frequency, record_length_time, record_length_samples

def _set_color_cycles(
    analog_in_channel_indices: tuple[ChannelIndex, ...],
    cycler_time_raw: Optional[Cycler],
    cycler_time_result: Optional[Cycler],
    cycler_freq_raw: Optional[Cycler],
    cycler_freq_result: Optional[Cycler]
    ) -> tuple[Cycler, Cycler, Cycler, Cycler]:
    '''
    Set default color cycles.

    If not specified, set the first and second plot line color of each signal
    vs. time plot according to `analog_in_channel_indices` (CH1 orange, CH2
    blue), with the rest of each color cycle following the ``matplotlib``
    default cycle.

    If not specified, set the color cycle of each magnitude vs. frequency plot
    such that it starts on the fourth color of the ``matplotlib`` default cycle.

    See ``scan_heartbeat()`` for parameter info.
    '''

    # default matplotlib color cycle
    color_cycle_time = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # default matplotlib color cycle starting with fourth color
    color_cycle_freq = [
        '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c'
    ]

    ch_1st = analog_in_channel_indices[0]
    ch_2nd = analog_in_channel_indices[1] if len(analog_in_channel_indices) == 2 else 2

    color_cycle_time[ch_1st], color_cycle_time[ch_2nd] = color_cycle_time[1], color_cycle_time[0]

    if cycler_time_raw is None:
        cycler_time_raw = cycler(color=color_cycle_time)
    if cycler_time_result is None:
        cycler_time_result = cycler(color=color_cycle_time)
    if cycler_freq_raw is None:
        cycler_freq_raw = cycler(color=color_cycle_freq)
    if cycler_freq_result is None:
        cycler_freq_result = cycler(color=color_cycle_freq)

    return cycler_time_raw, cycler_time_result, cycler_freq_raw, cycler_freq_result

def _configure_analog_out(
    analog_out: AnalogOut,
    channel_index: ChannelIndex,
    frequency: float,
    amplitude: float,
    offset: float,
    shape: DwfAnalogOutFunction,
    symmetry: float,
    node: DwfAnalogOutNode
    ) -> None:
    '''
    Configure the Wavegen ``AnalogOut`` object for Channel 1.

    Nearly all the properties have a 1-to-1 correspondence with the Wavegen GUI.

    See ``scan_heartbeat()`` for additional parameter info.

    Parameters
    ----------
    analog_out: AnalogOut
        Virtual object that controls the Wavegen: ``DwfDevice.analogOut``
    '''

    # Reset both channels to default values
    analog_out.reset(-1)

    analog_out.nodeEnableSet(channel_index, node, True)
    analog_out.nodeFrequencySet(channel_index, node, frequency)
    analog_out.nodeAmplitudeSet(channel_index, node, amplitude)
    analog_out.nodeOffsetSet(channel_index, node, offset)
    analog_out.nodeFunctionSet(channel_index, node, shape)
    analog_out.nodeSymmetrySet(channel_index, node, symmetry)

    # Start the Wavegen operating
    # This is the equivalent of the "Run" button (green arrow) on the ADS
    analog_out.configure(channel_index, True)

def _configure_analog_in(
    analog_in: AnalogIn,
    frequency: float,
    record_length_time: float,
    channel_indices: tuple[ChannelIndex, ...],
    channel_ranges: tuple[float, ...],
    trigger_flag: bool,
    trigger_source: DwfTriggerSource,
    trigger_level: float,
    trigger_channel: ChannelIndex
    ) -> None:
    '''
    Configure the scope ``AnalogIn`` object.

    See ``scan_heartbeat()`` for additional parameter info.

    Parameters
    ----------
    analog_in: AnalogIn
        Scope ``AnalogIn`` object.
    '''

    # Make certain that analog_in is in its default settings
    analog_in.reset()

    # Set digitization rate of recorded trace
    analog_in.frequencySet(frequency)

    # Use Record mode
    analog_in.acquisitionModeSet(DwfAcquisitionMode.Record)

    # We're in record mode, so length of record can be greater than the buffer size.
    # You just call and readout the buffer multiple times and concatenate to create your trace.
    analog_in.recordLengthSet(record_length_time)

    # In record mode, trigger_position = 0 means the trigger is at the beginning of the record
    trigger_position = 0

    print(f'Acquistion Mode: {analog_in.acquisitionModeGet()}')

    for channel_index, channel_range in zip(channel_indices, channel_ranges):

        # Enable the scope channel
        analog_in.channelEnableSet(channel_index, True)

        # The actual sample frequency is much higher than wanted sample frequency.
        # Choose to average all the measurements below the wanted sample frequency.
        analog_in.channelFilterSet(channel_index, DwfAnalogInFilter.Average)

        analog_in.channelRangeSet(channel_index, channel_range)

    if trigger_flag:

        analog_in.triggerSourceSet(trigger_source)
        analog_in.triggerChannelSet(trigger_channel)

        # From pydwf documentation:

        #    "Set the AnalogIn instrument trigger position, in seconds.
        #    The meaning of the trigger position depends on the currently selected acquisition mode:

        #    In Record acquisition mode, the trigger position is the time of the first valid sample
        #    acquired relative to the position of the trigger event. Negative values indicates times
        #    before the trigger time. To place the trigger in the middle of the recording, this
        #    value should be set to -0.5 times the duration of the recording.

        #    In Single acquisition mode, the trigger position is the trigger event time relative to
        #    the center of the acquisition window. To place the trigger in the middle of the
        #    acquisition buffer, the value should be 0."

        analog_in.triggerPositionSet(trigger_position)

        analog_in.triggerTypeSet(DwfAnalogInTriggerType.Edge)
        analog_in.triggerConditionSet(DwfTriggerSlope.Rise)

        # A small amount of hysteresis to make sure we only see rising edges
        analog_in.triggerHysteresisSet(0.10)

        # triggering on AnalogOut (basically a TTL) doesn't have a voltage level
        if analog_in.triggerSourceGet() == DwfTriggerSource.DetectorAnalogIn:
            analog_in.triggerLevelSet(trigger_level)

def _run_acquistion(
    analog_in: AnalogIn,
    scan_type: ScanType,
    fig: Figure,
    axs: NDArray,
    num_records: int,
    analog_out_frequency: float,
    analog_in_frequency: float,
    record_length_time: float,
    record_length_samples: int,
    channel_indices: tuple[ChannelIndex, ...],
    analog_in_trigger_flag: bool,
    overplot_fft: bool,
    xlim_time_raw: RangeLimits,
    xlim_freq_raw: RangeLimits,
    xlim_time_result: RangeLimits,
    xlim_freq_result: RangeLimits,
    cycler_time_raw: Cycler,
    cycler_freq_raw: Cycler,
    cycler_time_result: Cycler,
    cycler_freq_result: Cycler
    ) -> tuple[Figure, NDArray]:
    '''Run and plot one acquisition sequence.'''

    # Initialize raw and result lines to hold the data
    # These will have Line2D objects assigned to them and updated in the while loop
    lines_voltage_raw = None
    line_fft_raw = None
    lines_voltage_result = None
    line_fft_result = None

    # Handle case where only one input channel is used
    ch_idx_second = ChannelIndex.CH1 if len(channel_indices) == 1 else channel_indices[1]

    labels_result = (LINE_LABELS[channel_indices[0]],)
    labels_raw = (LINE_LABELS[channel_indices[0]], LINE_LABELS[ch_idx_second], LINE_LABELS[2])

    # Group the axes
    if scan_type == ScanType.TEST:
        axs_raw = axs
        axs_result = None
        prop_cyclers = (cycler_time_raw, cycler_freq_raw)
    else:
        axs_raw = (axs[0, 0], axs[1, 0])
        axs_result = (axs[0, 1], axs[1, 1])
        prop_cyclers = (cycler_time_raw, cycler_time_result, cycler_freq_raw, cycler_freq_result)

    # Use different single record acquisition functions for the different DAQ modes
    # SB: Currently, only Record mode is used.
    #     Single and Single1 seem to not acquire the desired number of samples for a given time.
    if analog_in.acquisitionModeGet() == DwfAcquisitionMode.Record:
        acquire_func = partial(
            _acquire_record,
            analog_in,
            record_length_samples,
            channel_indices,
            analog_in_trigger_flag
        )
    else:
        acquire_func = partial(
            _acquire_single,
            analog_in,
            record_length_time,
            record_length_samples,
            channel_indices,
            analog_in_trigger_flag
        )

    # Repeatedly-used parameters for convenience, to be unpacked in the function calls below
    kwargs_raw = {
        'analog_in_frequency': analog_in_frequency,
        'axs': axs_raw,
        'xlim_time': xlim_time_raw,
        'xlim_freq': xlim_freq_raw,
        'labels': labels_raw,
        'title_fft': f'FFT of {LINE_LABELS[channel_indices[0]]}',
        'overplot_fft': overplot_fft
    }
    kwargs_result = {
        'analog_in_frequency': analog_in_frequency,
        'axs': axs_result,
        'xlim_time': xlim_time_result,
        'xlim_freq': xlim_freq_result,
        'labels': labels_result,
        'title_fft': 'FFT of demodulated trace',
        'overplot_fft': overplot_fft
    }

    print(f'\nscan_type: {scan_type}')
    print(f'num_records: {num_records} record{"s" if num_records > 1 else ""}')
    print(f'analog_out_frequency: {analog_out_frequency} Hz')
    print(f'analog_in_frequency: {analog_in_frequency} Hz')
    print(f'record_length_time: {record_length_time} second{"" if record_length_time==1 else "s"}')
    print(
        f'record_length_samples: {record_length_samples} '
        f'sample{"s" if record_length_samples > 1 else ""} per record\n'
    )

    # Reset property and line history
    for ax, prop_cycler in zip(fig.axes, prop_cyclers):
        ax.set_prop_cycle(None)
        ax.set_prop_cycle(prop_cycler)
        for line in ax.lines:
            line.remove()

    if scan_type == ScanType.LOCKIN:
        time_lockin = np.zeros((num_records, 1))
        record_lockin = np.zeros((num_records, 1))
        time_start_acquisition = time.time()

    num_records_taken = 0

    # Acquire data until plots are closed or the desired number of records is taken
    while (len(plt.get_fignums()) != 0) and (num_records_taken < num_records):

        print(f'Acquiring record {num_records_taken + 1}...')

        title_signal = f'AnalogIn acquisition #{num_records_taken + 1}'

        # Start acquisition sequence for one record
        analog_in.configure(False, True)
        record, time_of_first_sample = acquire_func()

        # Find the time of the samples in the record with respect to first sample
        time_sample = time_of_first_sample + np.arange(record.shape[0]) / analog_in_frequency

        if scan_type == ScanType.TEST:

            lines_voltage_raw, line_fft_raw = _plot_raw_or_demod(
                lines_voltage=lines_voltage_raw,
                line_fft=line_fft_raw,
                time_sample=time_sample,
                y_signal=record,
                y_fft=record[:, 0],
                title_signal=title_signal,
                **kwargs_raw
            )

            _update_fig(fig)

        elif scan_type == ScanType.DEMOD:

            record_demod, record_demod_filt, ref_demod = _calc_demod(
                time_sample=time_sample,
                record=record,
                analog_out_frequency=analog_out_frequency,
                analog_in_frequency=analog_in_frequency
            )

            lines_voltage_raw, line_fft_raw = _plot_raw_or_demod(
                lines_voltage=lines_voltage_raw,
                line_fft=line_fft_raw,
                time_sample=time_sample,
                y_signal=np.concatenate([record, ref_demod[:, np.newaxis]], axis=1),
                y_fft=record[:, 0],
                title_signal=title_signal,
                **kwargs_raw
            )

            lines_voltage_result, line_fft_result = _plot_raw_or_demod(
                lines_voltage=lines_voltage_result,
                line_fft=line_fft_result,
                time_sample=time_sample,
                y_signal=record_demod_filt[:, np.newaxis],
                y_fft=record_demod,
                title_signal=f'Demodulated trace #{num_records_taken + 1}',
                **kwargs_result
            )

            _update_fig(fig)

        elif scan_type == ScanType.LOCKIN:

            # Find the time of first sample since the start of acquistion
            time_lockin[num_records_taken] = time.time() - time_start_acquisition

            record_lockin[num_records_taken], ref_demod = _calc_lockin(
                time_sample=time_sample,
                record=record,
                analog_out_frequency=analog_out_frequency
            )

            lockin_frequency = num_records_taken / (time_lockin[num_records_taken] - time_lockin[0])

            if num_records_taken >= num_records-1:
                # Plot the data on the last loop cycle

                print(
                    f'Estimated lock-in frequency: {lockin_frequency} Hz '
                    f'vs. trace frequency: {1 / record_length_time} Hz'
                )

                del kwargs_result['analog_in_frequency']

                lines_voltage_result, line_fft_result = _plot_lockin(
                    lines_voltage=lines_voltage_result,
                    line_fft=line_fft_result,
                    time_sample=time_lockin[:num_records_taken+1, :],
                    y_signal=record_lockin[:num_records_taken+1, :],
                    y_fft=record_lockin[:num_records_taken+1, 0],
                    title_signal=f'Lock-in trace after {num_records_taken + 1} records',
                    num_records_taken=num_records_taken,
                    lockin_frequency=lockin_frequency,
                    **kwargs_result
                )

                lines_voltage_raw, line_fft_raw = _plot_raw_or_demod(
                    lines_voltage=lines_voltage_raw,
                    line_fft=line_fft_raw,
                    time_sample=time_sample,
                    y_signal=np.concatenate([record, ref_demod[:, np.newaxis]], axis=1),
                    y_fft=record[:, 0],
                    title_signal=title_signal,
                    **kwargs_raw
                )

                _update_fig(fig)

        else:
            raise ValueError(f'scan_type must be one of {list(ScanType)}')

        num_records_taken += 1

    print('Finished acquisition.')

    return fig, axs

def _acquire_record(
    analog_in: AnalogIn,
    record_length_samples: int,
    channel_indices: tuple[ChannelIndex, ...],
    trigger_flag: bool
    ) -> tuple[NDArray[np.float64], float]:
    '''
    Acquire one record's worth of samples, sometimes in multiple queries to the ADC.

    See ``scan_heartbeat()`` for additional parameter info.

    Parameters
    ----------
    analog_in: AnalogIn
        Wavegen ``AnalogIn`` object.

    Returns
    -------
    record: NDArray[np.float64]
        Array of shape ``(record_length_samples, len(channel_indices))``, where
        each row contains the measured voltages for the corresponding channels.
        The first column corresponds to the first channel in `channel_indices`.
        The second column, if `channel_indices` is of length 2, corresponds to
        the second channel in `channel_indices`.
    time_of_first_sample: float
        Time of first sample collection. If `trigger_flag` is ``False``,
        this is 0.0.
    '''

    record = []

    total_samples_lost = 0
    total_samples_corrupted = 0
    num_adc_queries = 0

    while analog_in.status(True) != DwfState.Done:

        num_adc_queries += 1

        current_samples_available, current_samples_lost, current_samples_corrupted = analog_in.statusRecord()

        total_samples_lost += current_samples_lost
        total_samples_corrupted += current_samples_corrupted

        if current_samples_lost != 0:
            # Append NaN samples as placeholders for lost samples
            lost_samples = np.full((current_samples_lost, len(channel_indices)), np.nan)
            record.append(lost_samples)

        if current_samples_available != 0:
            # Append samples read from both channels
            current_samples = np.vstack(
                [
                    analog_in.statusData(channel_index, current_samples_available)
                    for channel_index in channel_indices
                ]
            ).transpose()
            record.append(current_samples)

    print(f'Total number of ADC queries for this event: {num_adc_queries}')

    record = np.concatenate(record)

    # Keep the beginning of the file where the trigger is
    if record.shape[0] > record_length_samples:

        discard_count = record.shape[0] - record_length_samples

        print(
            f'- NOTE - discarding newest {discard_count} '
            f'of {record.shape[0]} samples ({100.0 * discard_count / record.shape[0]:.1f}%); '
            f'keeping {record_length_samples} samples.'
        )

        record = record[:record_length_samples]

    if trigger_flag:
        time_of_first_sample = analog_in.triggerPositionStatus()
    else:
        time_of_first_sample = 0.0

    if total_samples_lost != 0:
        print(f'- WARNING - {total_samples_lost} samples were lost! Reduce sample frequency.')

    if total_samples_corrupted != 0:
        print(
            f'- WARNING - {total_samples_corrupted} samples could be corrupted! '
            'Reduce sample frequency.'
        )

    return record, time_of_first_sample

def _acquire_single(
    analog_in: AnalogIn,
    record_length_time: float,
    record_length_samples: int,
    channel_indices: tuple[ChannelIndex, ...],
    trigger_flag: bool
    ) -> tuple[NDArray[np.float64], float]:
    '''
    Acquire one record's worth of samples when `analog_in` is in the "single" and "single1" modes.

    This gathers the samples in one read call to the ADC.

    See ``scan_heartbeat()`` for additional parameter info.

    Parameters
    ----------
    analog_in: AnalogIn
        Wavegen ``AnalogIn`` object.

    Returns
    -------
    record: NDArray[np.float64]
        Array of shape ``(record_length_samples, len(channel_indices))``, where
        each row contains the measured voltages for the corresponding channels.
        The first column corresponds to the first channel in `channel_indices`.
        The second column, if `channel_indices` is of length 2, corresponds to
        the second channel in `channel_indices`.
    time_of_first_sample: float
        Time of first sample collection. If `trigger_flag` is ``False``,
        this is 0.0.
    '''

    status = None

    # Only pull data from the ADC buffer when status indicates done
    while status != DwfState.Done:
        status = analog_in.status(False)

    total_samples_available, total_samples_lost, total_samples_corrupted = analog_in.statusRecord()

    status = analog_in.status(True)

    if total_samples_available != 0:

        record = np.vstack(
            [
                analog_in.statusData(channel_index, total_samples_available)
                for channel_index in channel_indices
            ]
        ).transpose()

        if record.shape[0] > record_length_samples:

            discard_count = record.shape[0] - record_length_samples

            print(
                f'- NOTE - discarding newest {discard_count} '
                f'of {record.shape[0]} samples ({100.0 * discard_count / record.shape[0]:.1f}%); '
                f'keeping {record_length_samples} samples.'
            )

            record = record[:record_length_samples]

    if trigger_flag:
        # When trigger position is zero, it means that the trigger is at the center of the trace
        # When it is -record_length_time/2, the trigger is at the beginning of the record
        time_of_first_sample = -analog_in.triggerPositionStatus() - record_length_time/2
    else:
        time_of_first_sample = 0.0

    if total_samples_lost != 0:
        print(f'- WARNING - {total_samples_lost} samples were lost! Reduce sample frequency.')

    if total_samples_corrupted != 0:
        print(
            f'- WARNING - {total_samples_corrupted} samples could be corrupted! '
            'Reduce sample frequency.'
        )

    return record, time_of_first_sample

def _calc_demod(
    time_sample: NDArray[np.float64],
    record: NDArray[np.float64],
    analog_out_frequency: float,
    analog_in_frequency: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Perform demodulation calculations and filtering.

    The lowpass filter removes ``2 * nu_carrier`` components.
    '''

    ref_demod, record_demod = _demod(analog_out_frequency, time_sample, record)

    ## Write solution for: Modulated Heart Rate Monitor
    ## Replace `0` with a frequency in Hz.
    ##
    nu_3db = 0 #[Hz]
    ##

    # How many samples need to be averaged to give a nu_3db rolloff?
    n_filt = round(analog_in_frequency/nu_3dB)

    # Put a simple 1pole low-pass filter on it
    # You can also use the scipy python package to get more advanced filters
    if n_filt > 1:
        record_demod_filt = np.convolve(record_demod, np.ones(n_filt) / n_filt, mode='same')
    else:
        record_demod_filt = record_demod

    return record_demod, record_demod_filt, ref_demod

def _calc_lockin(
    time_sample: NDArray[np.float64],
    record: NDArray[np.float64],
    analog_out_frequency: float
    ) -> tuple[np.floating[Any], NDArray[np.float64]]:
    '''Perform demodulation and lock-in calculations.'''

    ## Write solution for:  Lock-In Heart Rate Monitor
    ## Replace `None` with your code.
    ##
    ref_demod, record_demod = None

    record_dc_lockin = None
    ##

    return record_dc_lockin, ref_demod

def _demod(
    analog_out_frequency: float,
    time_sample: NDArray[np.float64],
    record: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Perform demodulation calculations.

    1) Calculate the demodulation phase angle from the excitation.
    2) Make the demodulation wave. The phase needs to be checked in the real time plots.
    3) Demodulate the signal.
    '''

    # Calculate the demodulation phase angle for the reference signal
    phi = analog_out_frequency * 2 * np.pi * time_sample

    # Calculate demodulation phase (basically taking the FFT at the excitation frequency)
    phase_demod = np.angle(np.sum(record[:, 1] * (np.cos(phi) - 1j * np.sin(phi)))) #[radians]
    print(f'Estimated demodulation angle: {phase_demod * 180 / np.pi} deg')

    # Create demodulation array
    ref_demod = np.cos(phi + phase_demod)

    # Demodulate the record

    ## Write solution for: Modulated Heart Rate Monitor
    ## Multiply `record[:, 0]` by something to demodulate it.
    ##
    record_demod = record[:, 0]
    ##

    return ref_demod, record_demod

def _plot_raw_or_demod(
    lines_voltage: Optional[list[Line2D]],
    line_fft: Optional[Line2D],
    axs: tuple[Axes, Axes],
    time_sample: NDArray[np.float64],
    y_signal: NDArray[np.float64],
    y_fft: NDArray[np.float64],
    xlim_time: RangeLimits,
    xlim_freq: RangeLimits,
    labels: tuple[str, ...],
    title_signal: str,
    title_fft: str,
    analog_in_frequency: float,
    overplot_fft: bool,
    ) -> tuple[list[Line2D], Line2D]:
    '''Plot the time and FFT for a ``TEST`` or ``DEMOD`` scan and return the lines.'''

    lines_voltage = _plot_time(
        lines=lines_voltage,
        ax=axs[0],
        x=time_sample,
        y=y_signal,
        xlim=xlim_time,
        labels=labels,
        title=title_signal
    )

    if overplot_fft:
        line_fft = None

    ## Uncomment this:
    #line_fft = _plot_fft(
    #    line=line_fft,
    #    ax=axs[1],
    #    trace=y_fft,
    #    sample_frequency=analog_in_frequency,
    #    xlim=xlim_freq,
    #    title=title_fft
    #)

    return lines_voltage, line_fft

def _plot_lockin(
    lines_voltage: Optional[list[Line2D]],
    line_fft: Optional[Line2D],
    axs: tuple[Axes, Axes],
    time_sample: NDArray[np.float64],
    y_signal: NDArray[np.float64],
    y_fft: NDArray[np.float64],
    xlim_time: RangeLimits,
    xlim_freq: RangeLimits,
    labels: tuple[str, ...],
    title_signal: str,
    title_fft: str,
    overplot_fft: bool,
    num_records_taken: int,
    lockin_frequency: float
    ) -> tuple[list[Line2D], Optional[Line2D]]:
    '''Plot the time and FFT for a ``LOCKIN`` scan and return the lines.'''

    lines_voltage = _plot_time(
        lines=lines_voltage,
        ax=axs[0],
        x=time_sample,
        y=y_signal,
        xlim=xlim_time,
        labels=labels,
        title=title_signal
    )

    if num_records_taken > 10:

        # Make an FFT of lockin data
        # MP: Let's assume that the lockin data points are roughly evenly spaced...
        #     this is true on my machine. If you don't make this assumption, calculating
        #     the fourier transform is way more complex.

        if overplot_fft:
            line_fft = None

        ## Uncomment this:
        #line_fft = _plot_fft(
        #    line=line_fft,
        #    ax=axs[1],
        #    trace=y_fft,
        #    sample_frequency=lockin_frequency,
        #    xlim=xlim_freq,
        #    title=title_fft
        #)

    return lines_voltage, line_fft

def _plot_time(
    lines: Optional[list[Line2D]],
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xlim: RangeLimits,
    labels: tuple[str, ...],
    title: str,
    ) -> list[Line2D]:
    '''Plot voltage vs. time.'''

    num_lines = y.shape[1]

    if lines is None:

        lines = ax.plot(x, y[:, 0], label=labels[0])

        for j_line in range(1, num_lines):
            lines.extend(ax.plot(x, y[:, j_line], label=labels[j_line]))

        ax.legend(loc='upper right')

    else:
        for j_line in range(num_lines):
            lines[j_line].set_xdata(x)
            lines[j_line].set_ydata(y[:, j_line])

    if xlim is None:
        xmin=x.min()
        xmax=x.max()
        ax.set_xlim((xmin - 0.1 * np.abs(xmin), xmax + 0.1 * np.abs(xmax)))
    else:
        ax.set_xlim(xlim)

    ax.set_title(title)

    return lines

def _plot_fft(
    line: Optional[Line2D],
    ax: Axes,
    trace: NDArray[np.float64],
    sample_frequency: float,
    xlim: RangeLimits,
    title: str
    ) -> Line2D:
    '''Calculate the FFT and plot magnitude vs. frequency.'''

    ## Write solution for: Generating FFTs in Python
    ## Replace `None` with your code.
    ##
    fft_magnitudes = None
    fft_frequencies = None

    are_geq_zero = None
    fft_magnitudes_geq_zero = None
    fft_frequencies_geq_zero = None
    ##

    # Plot the magnitude of the FFT vs frequency
    # By convention, only plot the frequencies that are at least zero
    if line is None:
        line = ax.plot(fft_frequencies_geq_zero, fft_magnitudes_geq_zero)[0]
    else:
        line.set_xdata(fft_frequencies_geq_zero)
        line.set_ydata(fft_magnitudes_geq_zero)

    if xlim is None:
        xlim = (0.0, sample_frequency / 2.0)

    ax.set(xlim=xlim, ylim=(0, None), title=title)

    return line

def _update_fig(fig: Figure) -> None:
    '''Update the figure.'''

    fig.tight_layout()

    fig.canvas.draw()

    # Needed for matplotlib GUI
    fig.canvas.flush_events()
