'''This module provides convenience functions for configuration of instruments, channels, and nodes.'''

from typing import Optional, Tuple

import numpy as np

from pydwf.core.api.analog_in import AnalogIn
from pydwf.core.api.analog_out import AnalogOut
from pydwf.core.api.digital_in import DigitalIn
from pydwf.core.api.digital_out import DigitalOut

from pydwf.core.auxiliary.enum_types import (DwfAnalogCoupling, DwfAcquisitionMode, DwfTriggerSource, DwfTriggerSlope,
                                             DwfAnalogInFilter, DwfAnalogInTriggerType,
                                             DwfAnalogInTriggerLengthCondition, DwfAnalogOutMode, DwfAnalogOutIdle,
                                             DwfAnalogOutNode, DwfAnalogOutFunction, DwfDigitalInClockSource,
                                             DwfDigitalInSampleMode, DwfDigitalOutOutput, DwfDigitalOutType,
                                             DwfDigitalOutIdle)

#######################################################################################################################
##                                                                                                                   ##
##                        Configuration support for the AnalogIn instrument and its channels                         ##
##                                                                                                                   ##
#######################################################################################################################

def configure_analog_in_instrument(
        analog_in                         : AnalogIn,
        *,
        acquisition_mode                  : Optional[DwfAcquisitionMode],
        sample_frequency                  : Optional[float]=None,
        buffer_size                       : Optional[int]=None,
        noise_buffer_size                 : Optional[int]=None,
        record_duration                   : Optional[float]=None,
        trigger_source                    : Optional[DwfTriggerSource]=None,
        trigger_position                  : Optional[float]=None,
        trigger_auto_timeout              : Optional[float]=None,
        trigger_detector_holdoff          : Optional[float]=None,
        trigger_detector_type             : Optional[DwfAnalogInTriggerType]=None,
        trigger_detector_channel          : Optional[int]=None,
        trigger_detector_filter           : Optional[DwfAnalogInFilter]=None,
        trigger_detector_level            : Optional[float]=None,
        trigger_detector_hysteresis       : Optional[float]=None,
        trigger_detector_condition        : Optional[DwfTriggerSlope]=None,
        trigger_detector_length           : Optional[float]=None,
        trigger_detector_length_condition : Optional[DwfAnalogInTriggerLengthCondition]=None,
        sampling_source                   : Optional[DwfTriggerSource]=None,
        sampling_slope                    : Optional[DwfTriggerSlope]=None,
        sampling_delay                    : Optional[float]=None,
        counter_timeout                   : Optional[float]=None) -> None:

    '''Configure analog-input instrument settings.

    Parameters:
        analog_in (AnalogIn): The |AnalogIn| instrument to be configured.

        acquisition_mode (Optional[DwfAcquisitionMode]): If given, configure the instrument's acquisition mode setting.
        sample_frequency (Optional[float]): If given, configure the instrument's sample frequency setting.
        buffer_size (Optional[int]): If given, configure the instrument's buffer size setting.
        noise_buffer_size (Optional[int]): If given, configure the instrument's noise buffer size setting.
        record_duration (Optional[float]): If given, configure the instrument's recurd duration setting.
        trigger_source (Optional[DwfTriggerSource]): If given, configure the instrument's trigger source setting.
        trigger_position (Optional[float]): If given, configure the instrument's trigger position setting.
        trigger_auto_timeout (ptional[float]): If given, configure the instrument's trigger auto-timeout setting.
        trigger_detector_holdoff (Optional[float]):
            If given, configure the instrument's trigger detector holdoff setting.
        trigger_detector_type (Optional[DwfAnalogInTriggerType]):
            If given, configure the instrument's trigger detector type setting.
        trigger_detector_channel (Optional[int]):
            If given, configure the instrument's trigger detector channel setting.
        trigger_detector_filter (Optional[DwfAnalogInFilter]):
            If given, configure the instrument's trigger detector filter setting.
        trigger_detector_level (Optional[float]): If given, configure the instrument's trigger detector level setting.
        trigger_detector_hysteresis (Optional[float]):
            If given, configure the instrument's trigger detector hysteresis setting.
        trigger_detector_condition (Optional[DwfTriggerSlope]):
            If given, configure the instrument's trigger detector condition setting.
        trigger_detector_length (Optional[float]):
            If given, configure the instrument's trigger detector length setting.
        trigger_detector_length_condition (Optional[DwfAnalogInTriggerLengthCondition]):
            If given, configure the instrument's trigger detector length condition setting.
        sampling_source (Optional[DwfTriggerSource]): If given, configure the instrument's sampling source setting.
        sampling_slope (Optional[DwfTriggerSlope]): If given, configure the instrument's sampling slope setting.
        sampling_delay (Optional[float]): If given, configure the instrument's sampling delay setting.
        counter_timeout (Optional[float]): If given, configure the instrument's counter timeout setting.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    # pylint: disable=too-many-locals, too-many-branches

    if acquisition_mode is not None:
        analog_in.acquisitionModeSet(acquisition_mode)

    if sample_frequency is not None:
        analog_in.frequencySet(sample_frequency)

    if buffer_size is not None:
        analog_in.bufferSizeSet(buffer_size)

    if noise_buffer_size is not None:
        analog_in.noiseSizeSet(noise_buffer_size)

    if record_duration is not None:
        analog_in.recordLengthSet(record_duration)

    if trigger_source is not None:
        analog_in.triggerSourceSet(trigger_source)

    if trigger_position is not None:
        analog_in.triggerPositionSet(trigger_position)

    if trigger_auto_timeout is not None:
        analog_in.triggerAutoTimeoutSet(trigger_auto_timeout)

    if trigger_detector_holdoff is not None:
        analog_in.triggerHoldOffSet(trigger_detector_holdoff)

    if trigger_detector_type is not None:
        analog_in.triggerTypeSet(trigger_detector_type)

    if trigger_detector_channel is not None:
        analog_in.triggerChannelSet(trigger_detector_channel)

    if trigger_detector_filter is not None:
        analog_in.triggerFilterSet(trigger_detector_filter)

    if trigger_detector_level is not None:
        analog_in.triggerLevelSet(trigger_detector_level)

    if trigger_detector_hysteresis is not None:
        analog_in.triggerHysteresisSet(trigger_detector_hysteresis)

    if trigger_detector_condition is not None:
        analog_in.triggerConditionSet(trigger_detector_condition)

    if trigger_detector_length is not None:
        analog_in.triggerLengthSet(trigger_detector_length)

    if trigger_detector_length_condition is not None:
        analog_in.triggerLengthConditionSet(trigger_detector_length_condition)

    if sampling_source is not None:
        analog_in.samplingSourceSet(sampling_source)

    if sampling_slope is not None:
        analog_in.samplingSlopeSet(sampling_slope)

    if sampling_delay is not None:
        analog_in.samplingDelaySet(sampling_delay)

    if counter_timeout is not None:
        analog_in.counterSet(counter_timeout)


def configure_analog_in_channel(analog_in           : AnalogIn,
                                channel             : int,
                                *,
                                channel_enable      : Optional[bool]=None,
                                channel_filter      : Optional[DwfAnalogInFilter]=None,
                                channel_range       : Optional[float]=None,
                                channel_offset      : Optional[float]=None,
                                channel_attenuation : Optional[float]=None,
                                channel_bandwidth   : Optional[float]=None,
                                channel_impedance   : Optional[float]=None,
                                channel_coupling    : Optional[DwfAnalogCoupling]=None,
                               ) -> None:
    '''Configure a specific |AnalogIn| instrument channel's settings.

    Parameters:
        analog_in (AnalogIn): The |AnalogIn| instrument to be configured.
        channel (int): The analog input channel to be configured.
        channel_enable (Optional[bool]): If given, enable or disable the specified channel.
        channel_filter (Optional[DwfAnalogInFilter]): If given, set the channel's filter setting.
        channel_range (Optional[float]): If given, set the channel's range setting.
        channel_offset (Optional[float]): If given, set the channel's offset setting.
        channel_attenuation (Optional[float]): If given, set the channel's attenuation setting.
        channel_bandwidth (Optional[float]): If given, set the channel's bandwidth setting.
        channel_impedance (Optional[float]): If given, set the channel's impedance setting.
        channel_coupling (Optional[DwfAnalogCoupling]): If given, set the channel's coupling setting.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    if channel_enable is not None:
        analog_in.channelEnableSet(channel, channel_enable)

    if channel_filter is not None:
        analog_in.channelFilterSet(channel, channel_filter)

    if channel_range is not None:
        analog_in.channelRangeSet(channel, channel_range)

    if channel_offset is not None:
        analog_in.channelOffsetSet(channel, channel_offset)

    if channel_attenuation is not None:
        analog_in.channelAttenuationSet(channel, channel_attenuation)

    if channel_bandwidth is not None:
        analog_in.channelBandwidthSet(channel, channel_bandwidth)

    if channel_impedance is not None:
        analog_in.channelImpedanceSet(channel, channel_impedance)

    if channel_coupling is not None:
        analog_in.channelCouplingSet(channel, channel_coupling)

#######################################################################################################################
##                                                                                                                   ##
##                     Configuration support for the AnalogOut instrument and its channel nodes                      ##
##                                                                                                                   ##
#######################################################################################################################

def configure_analog_out_channel(analog_out           : AnalogOut,
                                 channel              : int,
                                 *,
                                 wait_duration        : Optional[float]=None,
                                 run_duration         : Optional[float]=None,
                                 repeat_trigger_flag  : Optional[bool]=None,
                                 repeat_count         : Optional[int]=None,
                                 master_channel_index : Optional[int]=None,
                                 trigger_source       : Optional[DwfTriggerSource]=None,
                                 trigger_slope        : Optional[DwfTriggerSlope]=None,
                                 mode                 : Optional[DwfAnalogOutMode]=None,
                                 idle                 : Optional[DwfAnalogOutIdle]=None,
                                 limitation           : Optional[float]=None,
                                 custom_am_fm_enable  : Optional[bool]=None
                                ) -> None:
    '''Configure a specific analog output channel's settings.

    Parameters:
        analog_out (AnalogOut)                      : The |AnalogOut| instrument to be configured.
        channel (int)                               : The analog output channel to be configured.
        wait_duration (Optional[float])             : If given, set the channel's wait duration.
        run_duration (Optional[float])              : If given, set the channel's run duration.
        repeat_trigger_flag (Optional[bool])        : If given, set the channel's repeat trigger flag.
        repeat_count (Optional[int])                : If given, set the channel's repeat count setting.
        master_channel_index (Optional[int])        : If given, set the channel's master channel setting.
        trigger_source (Optional[DwfTriggerSource]) : If given, set the channel's trigger source.
        trigger_slope (Optional[DwfTriggerSlope])   : If given, set the channel's trigger slope.
        mode (Optional[DwfAnalogOutMode])           : If given, set the channel's mode.
        idle (Optional[DwfAnalogOutIdle])           : If given, set the channel's idle behavior.
        limitation (Optional[float])                : If given, set the channel's limitation setting.
        custom_am_fm_enable (Optional[bool])        : If given, set the channel's custom AM/FM enable setting.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    if wait_duration is not None:
        analog_out.waitSet(channel, wait_duration)

    if run_duration is not None:
        analog_out.runSet(channel, run_duration)

    if repeat_trigger_flag is not None:
        analog_out.repeatTriggerSet(channel, repeat_trigger_flag)

    if repeat_count is not None:
        analog_out.repeatSet(channel, repeat_count)

    if master_channel_index is not None:
        analog_out.masterSet(channel, master_channel_index)

    if trigger_source is not None:
        analog_out.triggerSourceSet(channel, trigger_source)

    if trigger_slope is not None:
        analog_out.triggerSlopeSet(channel, trigger_slope)

    if mode is not None:
        analog_out.modeSet(channel, mode)

    if idle is not None:
        analog_out.idleSet(channel, idle)

    if limitation is not None:
        analog_out.limitationSet(channel, limitation)

    if custom_am_fm_enable is not None:
        analog_out.customAMFMEnableSet(channel, custom_am_fm_enable)


def configure_analog_out_channel_node(
        analog_out     : AnalogOut,
        channel        : int,
        node           : DwfAnalogOutNode,
        *,
        node_enable    : Optional[bool]=None,
        node_function  : Optional[DwfAnalogOutFunction]=None,
        node_frequency : Optional[float]=None,
        node_amplitude : Optional[float]=None,
        node_offset    : Optional[float]=None,
        node_symmetry  : Optional[float]=None,
        node_phase     : Optional[float]=None,
        node_data      : Optional[np.ndarray]=None,
        node_play_data : Optional[np.ndarray]=None) -> None:
    '''Configure a specific analog output channel node's settings.

    Parameters:
        analog_out (AnalogOut)                        : The |AnalogOut| instrument to be configured.
        channel (int)                                 : The analog output channel to be configured.
        node (DwfAnalogOutNode)                       : The analog output node to be configured.
        node_enable(Optional[float])                  : If given, set the channel node's enable state.
        node_function(Optional[DwfAnalogOutFunction]) : If given, set the channel node's function.
        node_frequency(Optional[float])               : If given, set the channel node's frequency.
        node_amplitude(Optional[float])               : If given, set the channel node's amplitude.
        node_offset(Optional[float])                  : If given, set the channel node's offset.
        node_symmetry(Optional[float])                : If given, set the channel node's symmetry.
        node_phase(Optional[float])                   : If given, set the channel node's phase.
        node_data(Optional[np.ndarray])               : If given, set the channel node's data.
        node_play_data(Optional[np.ndarray])          : If given, set the channel node's playback data.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    if node_enable is not None:
        analog_out.nodeEnableSet(channel, node, node_enable)

    if node_function is not None:
        analog_out.nodeFunctionSet(channel, node, node_function)

    if node_frequency is not None:
        analog_out.nodeFrequencySet(channel, node, node_frequency)

    if node_amplitude is not None:
        analog_out.nodeAmplitudeSet(channel, node, node_amplitude)

    if node_offset is not None:
        analog_out.nodeOffsetSet(channel, node, node_offset)

    if node_symmetry is not None:
        analog_out.nodeSymmetrySet(channel, node, node_symmetry)

    if node_phase is not None:
        analog_out.nodePhaseSet(channel, node, node_phase)

    if node_data is not None:
        analog_out.nodeDataSet(channel, node, node_data)

    if node_play_data is not None:
        analog_out.nodeDataSet(channel, node, node_play_data)

#######################################################################################################################
##                                                                                                                   ##
##                                 Configuration support for the DigitalIn instrument                                ##
##                                                                                                                   ##
#######################################################################################################################

def configure_digital_in_instrument(
        digital_in              : DigitalIn,
        *,
        clock_source            : Optional[DwfDigitalInClockSource]=None,
        divider                 : Optional[int]=None,
        acquisition_mode        : Optional[DwfAcquisitionMode]=None,
        sample_format           : Optional[int]=None,
        input_order             : Optional[bool]=None,
        buffer_size             : Optional[int]=None,
        sample_mode             : Optional[DwfDigitalInSampleMode]=None,
        sample_sensible         : Optional[int]=None,
        trigger_prefill         : Optional[int]=None,
        trigger_source          : Optional[DwfTriggerSource]=None,
        trigger_slope           : Optional[DwfTriggerSlope]=None,
        trigger_position        : Optional[int]=None,
        trigger_auto_timeout    : Optional[float]=None,
        trigger_detector_set    : Optional[Tuple[int, int, int, int]]=None,
        trigger_detector_reset  : Optional[Tuple[int, int, int, int]]=None,
        trigger_detector_count  : Optional[Tuple[int, int]]=None,
        trigger_detector_length : Optional[Tuple[float, float, int]]=None,
        trigger_detector_match  : Optional[Tuple[int, int, int, int]]=None,
        mixed_enable            : Optional[bool]=None,
        counter_duration        : Optional[float]=None) -> None:

    '''Configure digital-in instrument settings.

    Parameters:
        digital_in (DigitalIn): The |DigitalIn| instrument to be configured.
        clock_source (Optional[DwfDigitalInClockSource]): If given, configure the instrument's clock source setting.
        divider (Optional[int]): If given, configure the instrument's divider setting.
        acquisition_mode (Optional[DwfAcquisitionMode]): If given, configure the instrument's acquisition mode setting.
        sample_format (Optional[int]): If given, configure the instrument's sample format setting.
        input_order (Optional[bool]): If given, configure the instrument's inout order setting.
        buffer_size (Optional[int]): If given, configure the instrument's buffer size setting.
        sample_mode (Optional[DwfDigitalInSampleMode]): If given, configure the instrument's sample mode setting.
        sample_sensible (Optional[int]): If given, configure the instrument's sample sensible setting.
        trigger_prefill (Optional[int]): If given, configure the instrument's trigger prefill setting.
        trigger_source (Optional[DwfTriggerSource]): If given, configure the instrument's trigger source setting.
        trigger_slope (Optional[DwfTriggerSlope]): If given, configure the instrument's strigger slope setting.
        trigger_position (Optional[int]): If given, configure the instrument's trigger position setting.
        trigger_auto_timeout (Optional[float]): If given, configure the instrument's trigger auto-timeout setting.
        trigger_detector_set (Optional[Tuple[int, int, int, int]]):
            If given, configure the instrument's trigger detector 'set' setting.
        trigger_detector_reset (Optional[Tuple[int, int, int, int]]):
            If given, configure the instrument's trigger detector 'reset' setting.
        trigger_detector_count (Optional[int, int]):
            If given, configure the instrument's trigger detector 'count' settings.
        trigger_detector_length (Optional[Tuple[float, float, int]]):
            If given, configure the instrument's trigger detector 'length' setting.
        trigger_detector_match (Optional[Tuple[int, int, int, int]]):
            If given, configure the instrument's trigger detector 'match' setting.
        mixed_enable (Optional[bool]): If given, configure the instrument's 'mixed enable' setting.
        counter_duration (Optional[float]: If given, configure the instrument's counter duration setting.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    # pylint: disable=too-many-locals, too-many-branches

    if clock_source is not None:
        digital_in.clockSourceSet(clock_source)

    if divider is not None:
        digital_in.dividerSet(divider)

    if acquisition_mode is not None:
        digital_in.acquisitionModeSet(acquisition_mode)

    if sample_format is not None:
        digital_in.sampleFormatSet(sample_format)

    if input_order is not None:
        digital_in.inputOrderSet(input_order)

    if buffer_size is not None:
        digital_in.bufferSizeSet(buffer_size)

    if sample_mode is not None:
        digital_in.sampleModeSet(sample_mode)

    if sample_sensible is not None:
        digital_in.sampleSensibleSet(sample_sensible)

    if trigger_prefill is not None:
        digital_in.triggerPrefillSet(trigger_prefill)

    if trigger_source is not None:
        digital_in.triggerSourceSet(trigger_source)

    if trigger_slope is not None:
        digital_in.triggerSlopeSet(trigger_slope)

    if trigger_position is not None:
        digital_in.triggerPositionSet(trigger_position)

    if trigger_auto_timeout is not None:
        digital_in.triggerAutoTimeoutSet(trigger_auto_timeout)

    if trigger_detector_set is not None:
        (level_low, level_high, edge_rise, edge_fall) = trigger_detector_set
        digital_in.triggerSet(level_low, level_high, edge_rise, edge_fall)

    if trigger_detector_reset is not None:
        (level_low, level_high, edge_rise, edge_fall) = trigger_detector_reset
        digital_in.triggerResetSet(level_low, level_high, edge_rise, edge_fall)

    if trigger_detector_count is not None:
        (count, restart) = trigger_detector_count
        digital_in.triggerCountSet(count, restart)

    if trigger_detector_length is not None:
        (min_length, max_length, sync_mode) = trigger_detector_length
        digital_in.triggerLengthSet(min_length, max_length, sync_mode)

    if trigger_detector_match is not None:
        (pin, mask, value, bit_stuffing) = trigger_detector_match
        digital_in.triggerMatchSet(pin, mask, value, bit_stuffing)

    if mixed_enable is not None:
        digital_in.mixedSet(mixed_enable)

    if counter_duration is not None:
        digital_in.counterSet(counter_duration)

#######################################################################################################################
##                                                                                                                   ##
##                        Configuration support for the DigitalOut instrument and its channels                       ##
##                                                                                                                   ##
#######################################################################################################################

def configure_digital_out_instrument(
        digital_out          : DigitalOut,
        *,
        wait_duration        : Optional[float]=None,
        run_duration         : Optional[float]=None,
        repeat_trigger_flag  : Optional[bool]=None,
        repeat_count         : Optional[int]=None,
        trigger_source       : Optional[DwfTriggerSource]=None,
        trigger_slope        : Optional[DwfTriggerSlope]=None,
        play_data            : Optional[Tuple[str, int, int]]=None,
        play_update_data     : Optional[Tuple[str, int, int]]=None,
        play_rate            : Optional[float]) -> None:

    '''Configure digital-out instrument settings.

    Parameters:
        digital_out (DigitalOut): The |DigitalOut| instrument to be configured.
        wait_duration (Optional[float]): If given, configure the instrument's wait duration setting.
        run_duration (Optional[float]): If given, configure the instrument's run duration setting.
        repeat_trigger_flag (Optional[bool]): If given, configure the instrument's repeat trigger setting.
        repeat_count (Optional[int]): If given, configure the instrument's repeat count setting.
        trigger_source (Optional[DwfTriggerSource]): If given, configure the instrument's trigger source setting.
        trigger_slope (Optional[DwfTriggerSlope]): If given, configure the instrument's trigger slope setting.
        play_data (Optional[Tuple[str, int, int]]): If given, configure the instrument's playback data.
        play_update_data (Optional[Tuple[str, int, int]]): If given, update the instrument's playback data.
        play_rate (Optional[float)]): If given, configure the instrument's playback rate.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    if wait_duration is not None:
        digital_out.waitSet(wait_duration)

    if run_duration is not None:
        digital_out.runSet(run_duration)

    if repeat_count is not None:
        digital_out.repeatSet(repeat_count)

    if repeat_trigger_flag is not None:
        digital_out.repeatTriggerSet(repeat_trigger_flag)

    if trigger_source is not None:
        digital_out.triggerSourceSet(trigger_source)

    if trigger_slope is not None:
        digital_out.triggerSlopeSet(trigger_slope)

    if play_data is not None:
        (bits, bits_per_sample, count_of_samples) = play_data
        digital_out.playDataSet(bits, bits_per_sample, count_of_samples)

    if play_update_data is not None:
        (bits, index_of_sample, count_of_samples) = play_update_data
        digital_out.playUpdateSet(bits, index_of_sample, count_of_samples)

    if play_rate is not None:
        digital_out.playRateSet(play_rate)


def configure_digital_out_channel(
        digital_out      : DigitalOut,
        channel          : int,
        *,
        enable_flag      : Optional[bool],
        output_mode      : Optional[DwfDigitalOutOutput],
        output_type      : Optional[DwfDigitalOutType],
        output_idle      : Optional[DwfDigitalOutIdle],
        divider_init     : Optional[int],
        divider          : Optional[int],
        counter_init     : Optional[Tuple[bool, int]],
        counter          : Optional[Tuple[bool, int]],
        repetition_count : Optional[int],
        data             : Optional[Tuple[str, bool]]) -> None:

    '''Configure digital-out instrument channel settings.

    Parameters:
        digital_out (DigitalOut): The |DigitalOut| instrument to be configured.
        channel (int): The |DigitalOut| instrument channel to be configured.
        enable_flag (Optional[bool]): If given, configure the channels's enable/disable state.
        output_mode (Optional[DwfDigitalOutOutput]): If given, configure the channels's output mode setting.
        output_type (Optional[DwfDigitalOutType]): If given, configure the channels's output type setting.
        output_idle (Optional[DwfDigitalOutIdle]): If given, configure the channels's output idle setting.
        divider_init (Optional[int]): If given, configure the channels's divider init setting.
        divider (Optional[int]): If given, configure the channels's divider setting.
        counter_init (Optional[Tuple[bool, int]]): If given, configure the channels's counter init settings.
        counter (Optional[Tuple[[int, int]]): If given, configure the channels's counter settings.
        repetition_count (Optional[int]): If given, configure the channels's repetition count setting.
        data (Optional[Tuple[str, bool]]): If given, configure the channels's playback data setting.

        Raises:
            DwfLibraryError: One of the settings could not be applied.
    '''

    # pylint: disable=too-many-locals

    if enable_flag is not None:
        digital_out.enableSet(channel, enable_flag)

    if output_mode is not None:
        digital_out.outputSet(channel, output_mode)

    if output_type is not None:
        digital_out.typeSet(channel, output_type)

    if output_idle is not None:
        digital_out.idleSet(channel, output_idle)

    if divider_init is not None:
        digital_out.dividerInitSet(channel, divider_init)

    if divider is not None:
        digital_out.dividerSet(channel, divider)

    if counter_init is not None:
        (high_flag, counter_init_value) = counter_init
        digital_out.counterInitSet(channel, high_flag, counter_init_value)

    if counter is not None:
        (low_count, high_count) = counter
        digital_out.counterSet(channel, low_count, high_count)

    if repetition_count is not None:
        digital_out.repetitionSet(channel, repetition_count)

    if data is not None:
        (bits, tristate) = data
        digital_out.dataSet(channel, bits, tristate)
