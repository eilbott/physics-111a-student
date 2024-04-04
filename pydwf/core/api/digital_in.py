'''The |pydwf.core.api.digital_in| module provides a single class: |DigitalIn|.'''

# pylint: disable=too-many-lines

from typing import Tuple, List, Optional

import numpy as np

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.enum_types import (DwfDigitalInSampleMode, DwfDigitalInClockSource,
                                             DwfAcquisitionMode, DwfTriggerSlope, DwfState, DwfTriggerSource)
from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.exceptions import PyDwfError


class DigitalIn(AbstractDwfDeviceSubAPI):
    '''The |DigitalIn| class provides access to the digital input (logic analyzer) instrument of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its public
        |digitalIn:link| attribute for access by the user.
    '''

    # pylint: disable=too-many-public-methods

    ###################################################################################################################
    #                                                                                                                 #
    #                                               INSTRUMENT CONTROL                                                #
    #                                                                                                                 #
    ###################################################################################################################

    def reset(self) -> None:
        '''Reset the |DigitalIn| instrument parameters to default values.

        If autoconfiguration is enabled, the reset values will be used immediately.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalInReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self, reconfigure: bool, start: bool) -> None:
        '''Configure the instrument and start or stop the acquisition operation.

        Parameters:
            reconfigure (bool): If True, the instrument settings are sent to the instrument.
            start (bool): If True, an acquisition is started. If False, an ongoing acquisition is stopped.

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfDigitalInConfigure(self.hdwf, reconfigure, start)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self, read_data_flag: bool) -> DwfState:
        '''Get the |DigitalIn| instrument state.

        This method performs a status request to the |DigitalIn| instrument and receives its response.

        The following methods can be used to retrieve |DigitalIn| instrument status information
        as a result of this call, regardless of the value of the *read_data_flag* parameter:

        * :py:meth:`~statusTime`
        * :py:meth:`~statusAutoTriggered`
        * :py:meth:`~statusSamplesLeft`
        * :py:meth:`~statusSamplesValid`
        * :py:meth:`~statusIndexWrite`
        * :py:meth:`~statusRecord`

        The following methods can be used to retrieve bulk data obtained from the |DigitalIn| instrument
        as a result of this call, but only if the *read_data_flag* parameter is True:

        * :py:meth:`~statusData`
        * :py:meth:`~statusData2`
        * :py:meth:`~statusNoise2`

        Parameters:
            read_data_flag (bool): Whether to read data.

        Returns:
            DwfState: The current state of the instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        c_status = typespec_ctypes.DwfState()
        result = self.lib.FDwfDigitalInStatus(self.hdwf, read_data_flag, c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        status = DwfState(c_status.value)
        return status

    ###################################################################################################################
    #                                                                                                                 #
    #                                                STATUS VARIABLES                                                 #
    #                                                                                                                 #
    ###################################################################################################################

    def statusTime(self) -> Tuple[int, int, int]:
        '''Get the timestamp of the current status information.

        Returns:
            Tuple[int, int, int]: A three-element tuple, indicating the POSIX timestamp of the status request.
            The first element is the POSIX second, the second and third element are the numerator and denominator,
            respectively, of the fractional part of the second.

            In case :py:meth:`status` hasn't been called yet, this method will return zeroes
            for all three tuple elements.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_sec_utc = typespec_ctypes.c_unsigned_int()
        c_tick = typespec_ctypes.c_unsigned_int()
        c_ticks_per_second = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInStatusTime(
            self.hdwf,
            c_sec_utc,
            c_tick,
            c_ticks_per_second)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sec_utc = c_sec_utc.value
        tick = c_tick.value
        ticks_per_second = c_ticks_per_second.value
        return (sec_utc, tick, ticks_per_second)

    def statusAutoTriggered(self) -> bool:
        '''Check if the current acquisition is auto-triggered.

        Returns:
            bool: True if the current acquisition is auto-triggered, False otherwise.

        Raises:
            DwfLibraryError: An error occurred while retrieving the auto-triggered status.
        '''
        c_auto_triggered = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusAutoTriggered(self.hdwf, c_auto_triggered)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        auto_triggered = bool(c_auto_triggered.value)
        return auto_triggered

    def statusSamplesLeft(self) -> int:
        '''Retrieve the number of samples left in the acquisition, in samples.

        Returns:
            int: In case a finite-duration acquisition is active, the number of samples
            remaining to be acquired in the acquisition.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_samples_left = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusSamplesLeft(self.hdwf, c_samples_left)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_left = c_samples_left.value
        return samples_left

    def statusSamplesValid(self) -> int:
        '''Retrieve the number of valid data samples.

        Returns:
            int: The number of valid samples in the buffer.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_samples_valid = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusSamplesValid(self.hdwf, c_samples_valid)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_valid = c_samples_valid.value
        return samples_valid

    def statusIndexWrite(self) -> int:
        '''Retrieve the buffer write index.

        This is needed in :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.ScanScreen`
        acquisition mode to display the scan bar.

        Returns:
            int: The buffer write index.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_index_write = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusIndexWrite(self.hdwf, c_index_write)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        index_write = c_index_write.value
        return index_write

    def statusRecord(self) -> Tuple[int, int, int]:
        '''Get the recording status.

        Returns:
            Tuple[int, int, int]: A three-element tuple containing the counts for
            *available*, *lost*, and *corrupt* data samples, in that order.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_data_available = typespec_ctypes.c_int()
        c_data_lost = typespec_ctypes.c_int()
        c_data_corrupt = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusRecord(
            self.hdwf,
            c_data_available,
            c_data_lost,
            c_data_corrupt)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        data_free = c_data_available.value
        data_lost = c_data_lost.value
        data_corrupt = c_data_corrupt.value
        return (data_free, data_lost, data_corrupt)

    def statusCompress(self) -> Tuple[int, int, int]:
        '''Get the recording status (compress version).

        Returns:
            Tuple[int, int, int]: A three-element tuple containing the counts for
            *available*, *lost*, and *corrupt* data samples, in that order.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_data_available = typespec_ctypes.c_int()
        c_data_lost = typespec_ctypes.c_int()
        c_data_corrupt = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInStatusCompress(
            self.hdwf,
            c_data_available,
            c_data_lost,
            c_data_corrupt)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        data_free = c_data_available.value
        data_lost = c_data_lost.value
        data_corrupt = c_data_corrupt.value
        return (data_free, data_lost, data_corrupt)

    ###################################################################################################################
    #                                                                                                                 #
    #                                              STATUS DATA RETRIEVAL                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def statusData(self, count: int, sample_format: Optional[int]=None) -> np.ndarray:
        '''Retrieve the acquired data samples from the |DigitalIn| instrument.

        Parameters:
            count (int): Sample count.
            sample_format (int) Either 8, 16 or 32 bits per sample.
                If not specified, the current value is queried using :py:meth:`sampleFormatGet`.

        Returns:
            a 1D numpy array of 8, 16, or 32-bit unsigned words.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        if sample_format is None:
            sample_format = self.sampleFormatGet()

        # Assume that the data returned is in native endianness.
        samples_dtype = {
            8  : np.uint8,
            16 : np.uint16,
            32 : np.uint32
        }.get(sample_format)

        if samples_dtype is None:
            raise PyDwfError(f'Unexpected sample format: {sample_format}')

        count_bytes = (sample_format // 8) * count

        samples = np.empty(count, samples_dtype)
        result = self.lib.FDwfDigitalInStatusData(
            self.hdwf,
            samples.ctypes.data_as(typespec_ctypes.c_unsigned_char_ptr),
            count_bytes)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusData2(self, offset: int, count: int, sample_format: Optional[int]=None) -> np.ndarray:
        '''Retrieve the acquired data samples from the |DigitalIn| instrument.

        Parameters:
            count (int): Sample count.
            sample_format (int): Either 8, 16 or 32 bits per sample.
                If not specified, the current value is queried using :py:meth:`sampleFormatGet`.

        Returns:
            a 1D numpy array of 8, 16, or 32-bit unsigned words.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        if sample_format is None:
            sample_format = self.sampleFormatGet()

        # Assume that the data returned is in native endianness.
        samples_dtype = {
            8  : np.uint8,
            16 : np.uint16,
            32 : np.uint32
        }.get(sample_format)

        if samples_dtype is None:
            raise PyDwfError(f'Unexpected sample format: {sample_format}')

        count_bytes = (sample_format // 8) * count

        samples = np.empty(count, dtype=samples_dtype)
        result = self.lib.FDwfDigitalInStatusData2(
            self.hdwf,
            samples.ctypes.data_as(typespec_ctypes.c_unsigned_char_ptr),
            offset,
            count_bytes)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusCompressed(self, count_bytes: int) -> np.ndarray:
        '''Retrieve the compressed data samples from the |DigitalIn| instrument.

        Todo:
            Figure out the data format.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        samples = np.empty(count_bytes, dtype='B')
        result = self.lib.FDwfDigitalInStatusData(
            self.hdwf,
            samples.ctypes.data_as(typespec_ctypes.c_unsigned_char_ptr),
            count_bytes)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples


    def statusCompressed2(self, first_sample: int, count_bytes: int) -> np.ndarray:
        '''Retrieve the acquired data samples from the |DigitalIn| instrument.

        Todo:
            Figure out the data format.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        samples = np.empty(count_bytes, dtype='B')
        result = self.lib.FDwfDigitalInStatusCompressed2(
            self.hdwf,
            samples.ctypes.data_as(typespec_ctypes.c_unsigned_char_ptr),
            first_sample,
            count_bytes)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusNoise2(self, first_sample: int, count_bytes: int) -> np.ndarray:
        '''Get the noise data from the |DigitalIn| instrument.

        Todo:
            Figure out the data format.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        noise = np.empty(count_bytes, dtype='B')
        result = self.lib.FDwfDigitalInStatusNoise2(
            self.hdwf,
            noise.ctypes.data_as(typespec_ctypes.c_unsigned_char_ptr),
            first_sample,
            count_bytes)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return noise

    ###################################################################################################################
    #                                                                                                                 #
    #                                           ACQUISITION TIMING SETTINGS                                           #
    #                                                                                                                 #
    ###################################################################################################################

    def internalClockInfo(self) -> float:
        '''Get the |DigitalIn| internal clock frequency, in Hz.

        Returns:
            The internal clock frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalInInternalClockInfo(self.hdwf, c_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency = c_frequency.value
        return frequency

    def clockSourceInfo(self) -> List[DwfDigitalInClockSource]:
        '''Get a list of valid clock sources for the |DigitalIn| instrument.

        Returns:
            List[DwfDigitalInClockSource]: A list of valid clock sources.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_clock_source_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInClockSourceInfo(self.hdwf, c_clock_source_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        clock_source_bitset = c_clock_source_bitset.value
        clock_source_list = [clock_source for clock_source in DwfDigitalInClockSource
                             if clock_source_bitset & (1 << clock_source.value)]
        return clock_source_list

    def clockSourceSet(self, clock_source: DwfDigitalInClockSource) -> None:
        '''Set the |DigitalIn| instrument clock source.

        Parameters:
            clock_source (DwfDigitalInClockSource): The clock source to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInClockSourceSet(self.hdwf, clock_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def clockSourceGet(self) -> DwfDigitalInClockSource:
        '''Get the |DigitalIn| instrument clock source.

        Returns:
            DwfDigitalInClockSource: The currently configured |DigitalIn| clock source.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_clock_source = typespec_ctypes.DwfDigitalInClockSource()
        result = self.lib.FDwfDigitalInClockSourceGet(self.hdwf, c_clock_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        clock_source = DwfDigitalInClockSource(c_clock_source.value)
        return clock_source

    def dividerInfo(self) -> int:
        '''Get the |DigitalIn| instrument maximum divider value.

        Returns:
            int: The maximum valid divider value that can be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_divider_max = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInDividerInfo(self.hdwf, c_divider_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        divider_max = c_divider_max.value
        return divider_max

    def dividerSet(self, divider: int) -> None:
        '''Set the |DigitalIn| instrument divider value.

        Parameters:
            divider: The divider value to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInDividerSet(self.hdwf, divider)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def dividerGet(self) -> int:
        '''Get the current |DigitalIn| instrument divider value.

        If the clock source is internal, the |DigitalIn| sample frequency will be equal
        to :py:meth:`internalClockInfo` divided by :py:meth:`dividerGet`.

        Returns:
            int: The currently configured divider value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_divider = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInDividerGet(self.hdwf, c_divider)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        divider = c_divider.value
        return divider

    ###################################################################################################################
    #                                                                                                                 #
    #                                              ACQUISITION SETTINGS                                               #
    #                                                                                                                 #
    ###################################################################################################################

    def acquisitionModeInfo(self) -> List[DwfAcquisitionMode]:
        '''Get a list of valid |DigitalIn| instrument acquisition modes.

        Returns:
            List[DwfAcquisitionMode]: A list of valid acquisition modes for the |DigitalIn| instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_acquisition_mode_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInAcquisitionModeInfo(self.hdwf, c_acquisition_mode_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        acquisition_mode_bitset = c_acquisition_mode_bitset.value
        acquisition_mode_list = [acquisition_mode for acquisition_mode in DwfAcquisitionMode
                                 if acquisition_mode_bitset & (1 << acquisition_mode.value)]
        return acquisition_mode_list

    def acquisitionModeSet(self, acquisition_mode: DwfAcquisitionMode) -> None:
        '''Select the |DigitalIn| acquisition mode.

        Parameters:
            acquisition_mode (DwfAcquisitionMode): The acquisition mode to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInAcquisitionModeSet(self.hdwf, acquisition_mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def acquisitionModeGet(self) -> DwfAcquisitionMode:
        '''Get the currently selected |DigitalIn| acquisition mode.

        Returns:
            DwfAcquisitionMode: The currently selected acquisition mode.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_acquisition_mode = typespec_ctypes.DwfAcquisitionMode()
        result = self.lib.FDwfDigitalInAcquisitionModeGet(self.hdwf, c_acquisition_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        acquisition_mode = DwfAcquisitionMode(c_acquisition_mode.value)
        return acquisition_mode

    def bitsInfo(self) -> int:
        '''Get the number of |DigitalIn| bits.

        Returns:
            The number of digital input bits available.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_num_bits = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInBitsInfo(self.hdwf, c_num_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        num_bits = c_num_bits.value
        return num_bits

    def sampleFormatSet(self, num_bits: int) -> None:
        '''Set the |DigitalIn| sample format (i.e., number of bits).

        Parameters:
            num_bits (int): The number of bits per sample (8, 16, or 32).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInSampleFormatSet(self.hdwf, num_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def sampleFormatGet(self) -> int:
        '''Get the |DigitalIn| sample format (i.e., number of bits).

        Returns:
            int: The currently configured number of bits per sample (8, 16, or 32).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_num_bits = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInSampleFormatGet(self.hdwf, c_num_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        num_bits = c_num_bits.value
        return num_bits

    def inputOrderSet(self, dio_first: bool) -> None:
        '''Select the |DigitalIn| order of values stored in the sampling array.

        If *dio_first* is True, DIO 24…39 are placed at the beginning of the array followed by DIN 0…23.

        If *dio_first* is False, DIN 0…23 are placed at the beginning followed by DIO 24…31.

        This method is valid only for the Digital Discovery device.

        Parameters:
            dio_first (bool): Whether the DIO pins come before the DIN pins.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInInputOrderSet(self.hdwf, int(dio_first))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def bufferSizeInfo(self) -> int:
        '''Get the |DigitalIn| instrument maximum buffer size.

        Returns:
            int: The maximum valid buffer size.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_buffer_size_max = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInBufferSizeInfo(self.hdwf, c_buffer_size_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        buffer_size_max = c_buffer_size_max.value
        return buffer_size_max

    def bufferSizeSet(self, buffer_size: int) -> None:
        '''Set the |DigitalIn| instrument buffer size.

        Parameters:
            buffer_size (int): The buffer size to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInBufferSizeSet(self.hdwf, buffer_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def bufferSizeGet(self) -> int:
        '''Get the currently configured |DigitalIn| instrument buffer size.

        Returns:
            int: The currently configured buffer size.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_buffer_size = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInBufferSizeGet(self.hdwf, c_buffer_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        buffer_size = c_buffer_size.value
        return buffer_size

    def sampleModeInfo(self) -> List[DwfDigitalInSampleMode]:
        '''Get the valid |DigitalIn| instrument sample modes.

        Returns:
            List[DwfDigitalInSampleMode]: A list of valid sample modes.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_sample_mode_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInSampleModeInfo(self.hdwf, c_sample_mode_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sample_mode_bitset = c_sample_mode_bitset.value
        sample_mode_list = [sample_mode for sample_mode in DwfDigitalInSampleMode
                            if sample_mode_bitset & (1 << sample_mode.value)]
        return sample_mode_list

    def sampleModeSet(self, sample_mode: DwfDigitalInSampleMode) -> None:
        '''Set the |DigitalIn| instrument sample mode.

        Parameters:
            sample_mode (DwfDigitalInSampleMode): The sample mode to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInSampleModeSet(self.hdwf, sample_mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def sampleModeGet(self) -> DwfDigitalInSampleMode:
        '''Get the |DigitalIn|  instrument sample mode.

        Returns:
            DwfDigitalInSampleMode: The currently configured sample mode.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_sample_mode = typespec_ctypes.DwfDigitalInSampleMode()
        result = self.lib.FDwfDigitalInSampleModeGet(self.hdwf, c_sample_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sample_mode = DwfDigitalInSampleMode(c_sample_mode.value)
        return sample_mode

    def sampleSensibleSet(self, compression_bits: int) -> None:
        '''Set the |DigitalIn| instrument *sample sensible* setting.

        Todo:
            Figure out what this setting does.

        Parameters:
            compression_bits (int): (unknown)

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInSampleSensibleSet(self.hdwf, compression_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def sampleSensibleGet(self) -> int:
        '''Get the |DigitalIn| instrument *sample sensible* setting.

        This setting is only used in :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record` mode.

        It select the signals to be used for data compression in record acquisition mode.

        Todo:
            Figure out what this setting does.

        Returns:
            int: The sample sensible setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_compression_bits = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInSampleSensibleGet(self.hdwf, c_compression_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        compression_bits = c_compression_bits.value
        return compression_bits

    def triggerPrefillSet(self, samples_before_trigger: int) -> None:
        '''Set the |DigitalIn| instrument trigger prefill setting, in samples.

        This setting is only used in :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record` mode.

        It determines the number of samples to acquire before arming in
        :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record` acquisition mode. The prefill is used for
        recording with a trigger to make sure that at least the required number of samples are collected before arming.

        Todo:
            Figure out what this setting does, precisely.

        Parameters:
            samples_before_trigger (int): The prefill count, in samples.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerPrefillSet(self.hdwf, samples_before_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerPrefillGet(self) -> int:
        '''Get the |DigitalIn| instrument trigger prefill setting, in samples.

        Returns:
            int: The trigger prefill count, in samples.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_samples_before_trigger = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInTriggerPrefillGet(self.hdwf, c_samples_before_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_before_trigger = c_samples_before_trigger.value
        return samples_before_trigger

    ###################################################################################################################
    #                                                                                                                 #
    #                                              TRIGGER CONFIGURATION                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerSourceInfo(self) -> List[DwfTriggerSource]:
        '''Get the valid |DigitalIn| instrument trigger sources.

        Warning:
            **This method is obsolete.**

            Use the generic :py:meth:`DeviceControl.triggerInfo()` method instead.

        Returns:
            List[DwfTriggerSource]: A list of valid trigger sources.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_source_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInTriggerSourceInfo(self.hdwf, c_trigger_source_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source_bitset = c_trigger_source_bitset.value
        trigger_source_list = [trigger_source for trigger_source in DwfTriggerSource
                               if trigger_source_bitset & (1 << trigger_source.value)]
        return trigger_source_list

    def triggerSourceSet(self, trigger_source: DwfTriggerSource) -> None:
        '''Set |DigitalIn| instrument trigger source.

        Parameters:
            trigger_source (DwfTriggerSource): The trigger source to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerSourceSet(self.hdwf, trigger_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSourceGet(self) -> DwfTriggerSource:
        '''Get the currently selected instrument trigger source.

        Returns:
            DwfTriggerSource: The currently selected instrument trigger source.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_source = typespec_ctypes.DwfTriggerSource()
        result = self.lib.FDwfDigitalInTriggerSourceGet(self.hdwf, c_trigger_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source = DwfTriggerSource(c_trigger_source.value)
        return trigger_source

    def triggerSlopeSet(self, trigger_slope: DwfTriggerSlope) -> None:
        '''Select the |DigitalIn| instrument trigger slope.

        Parameters:
            trigger_slope (DwfTriggerSlope): The trigger slope to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerSlopeSet(self.hdwf, trigger_slope.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSlopeGet(self) -> DwfTriggerSlope:
        '''Get the currently selected |DigitalIn| instrument trigger slope.

        Returns:
            DwfTriggerSlope: The currently selected |DigitalIn| instrument trigger slope.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_slope = typespec_ctypes.DwfTriggerSlope()
        result = self.lib.FDwfDigitalInTriggerSlopeGet(self.hdwf, c_trigger_slope)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_slope = DwfTriggerSlope(c_trigger_slope.value)
        return trigger_slope

    def triggerPositionInfo(self) -> int:
        '''Get |DigitalIn| trigger position info.

        Returns:
            int: The maximum number of samples after the trigger.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_max_samples_after_trigger = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInTriggerPositionInfo(
            self.hdwf,
            c_max_samples_after_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        max_samples_after_trigger = c_max_samples_after_trigger.value
        return max_samples_after_trigger

    def triggerPositionSet(self, samples_after_trigger: int) -> None:
        '''Set |DigitalIn| instrument desired trigger position.

        Parameters:
            samples_after_trigger (int): The number of samples after the trigger.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerPositionSet(self.hdwf, samples_after_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerPositionGet(self) -> int:
        '''Get |DigitalIn| instrument trigger position.

        Returns:
            int: The currently configured number of samples after the trigger.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_samples_after_trigger = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInTriggerPositionGet(self.hdwf, c_samples_after_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_after_trigger = c_samples_after_trigger.value
        return samples_after_trigger

    def triggerAutoTimeoutInfo(self) -> Tuple[float, float, int]:
        '''Get |DigitalIn| instrument trigger auto-timeout range, in seconds.

        Returns:
            Tuple[float, float, int]: The range and number of steps of the auto-timeout setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_auto_timeout_min = typespec_ctypes.c_double()
        c_trigger_auto_timeout_max = typespec_ctypes.c_double()
        c_trigger_auto_timeout_num_steps = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalInTriggerAutoTimeoutInfo(
            self.hdwf,
            c_trigger_auto_timeout_min,
            c_trigger_auto_timeout_max,
            c_trigger_auto_timeout_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_auto_timeout_num_steps.value.is_integer():
            raise PyDwfError('Bad c_trigger_auto_timeout_num_steps value.')

        trigger_auto_timeout_min = c_trigger_auto_timeout_min.value
        trigger_auto_timeout_max = c_trigger_auto_timeout_max.value
        trigger_auto_timeout_num_steps = int(c_trigger_auto_timeout_num_steps.value)
        return (trigger_auto_timeout_min, trigger_auto_timeout_max, trigger_auto_timeout_num_steps)

    def triggerAutoTimeoutSet(self, auto_timeout: float) -> None:
        '''Set |DigitalIn| instrument trigger auto-timeout value, in seconds.

        Parameters:
            auto_timeout (float): The auto-timeout value to be configured, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerAutoTimeoutSet(self.hdwf, auto_timeout)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerAutoTimeoutGet(self) -> float:
        '''Get |DigitalIn| instrument trigger auto-timeout value, in seconds.

        Returns:
            float: The currently configured trigger auto-timeout value, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_auto_timeout = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalInTriggerAutoTimeoutGet(self.hdwf, c_auto_timeout)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        auto_timeout = c_auto_timeout.value
        return auto_timeout

    ###################################################################################################################
    #                                                                                                                 #
    #                                         TRIGGER DETECTOR CONFIGURATION                                          #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerInfo(self) -> Tuple[int, int, int, int]:
        '''Get |DigitalIn| detector trigger info.

        Return the pins that support the different types of triggering. Each integer is a bitmask, representing pins
        that can be used as *low level*, *high level*, *rising edge*, and *falling edge* triggers.

        Returns:
            Tuple[int, int, int, int]: Digital pins that can be used as *low level*, *high level*, *rising edge*,
            and *falling edge* triggers, respectively.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_level_low = typespec_ctypes.c_unsigned_int()
        c_level_high = typespec_ctypes.c_unsigned_int()
        c_edge_rise = typespec_ctypes.c_unsigned_int()
        c_edge_fall = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInTriggerInfo(
            self.hdwf,
            c_level_low,
            c_level_high,
            c_edge_rise,
            c_edge_fall)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        level_low = c_level_low.value
        level_high = c_level_high.value
        edge_rise = c_edge_rise.value
        edge_fall = c_edge_fall.value
        return (level_low, level_high, edge_rise, edge_fall)

    def triggerSet(self, level_low: int, level_high: int, edge_rise: int, edge_fall: int) -> None:
        '''Set |DigitalIn| trigger detector conditions.

        The *level_low* and *level_high* settings effectively mask trigger detection events to clock cycles where
        the selected pins are low or high, respectively.

        The *edge_rise* and *edge_fall* indicate channels where the specific type of transition on any of the
        included channels will lead to a trigger event, ar least if the level conditions specified by the *level_low*
        and *level_high* settings are satisfied.

        Parameters:
            level_low (int): The channels that are required to be low for a trigger event to occur (bitfield).
            level_high (int): The channels that are required to be high for a trigger event to occur (bitfield).
            edge_rise (int): The channels where a rising edge will cause a trigger event (bitfield).
            edge_fall (int): The channels where a falling edge will cause a trigger event (bitfield).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerSet(
            self.hdwf,
            level_low,
            level_high,
            edge_rise,
            edge_fall)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerGet(self) -> Tuple[int, int, int, int]:
        '''Get the configured |DigitalIn| trigger detector settings.

        Returns:
            Tuple[int, int, int, int]: The currently configured *level_low*, *level_high*, *edge_rise*,
            and *edge_fall* settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_level_low  = typespec_ctypes.c_unsigned_int()
        c_level_high = typespec_ctypes.c_unsigned_int()
        c_edge_rise  = typespec_ctypes.c_unsigned_int()
        c_edge_fall  = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalInTriggerGet(
            self.hdwf,
            c_level_low,
            c_level_high,
            c_edge_rise,
            c_edge_fall)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        level_low  = c_level_low.value
        level_high = c_level_high.value
        edge_rise  = c_edge_rise.value
        edge_fall  = c_edge_fall.value
        return (level_low, level_high, edge_rise, edge_fall)

    def triggerResetSet(self, level_low: int, level_high: int, edge_rise: int, edge_fall: int) -> None:
        '''Configure |DigitalIn| trigger detector reset condition.

        Parameters:
            level_low (int): The channels that are required to be low for a reset to occur (bitfield).
            level_high (int): The channels that are required to be high for a reset to occur (bitfield).
            edge_rise (int): The channels where a rising edge will cause a reset (bitfield).
            edge_fall (int): The channels where a falling edge will cause a reset (bitfield).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerResetSet(
            self.hdwf,
            level_low,
            level_high,
            edge_rise,
            edge_fall)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerCountSet(self, count: int, restart: int) -> None:
        '''Set the |DigitalIn| trigger detector count.

        Todo:
            Figure out what this setting does.

        Parameters:
            count (int): The event count.
            restart (int): (to be documented)

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerCountSet(self.hdwf, count, restart)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerLengthSet(self, min_length: float, max_length: float, sync_mode: int) -> None:
        '''Set the |DigitalIn| trigger detector length.

        Todo:
            Figure out what this setting does.

        Parameters:
            min_length (float): (to be documented)
            max_length (float): (to be documented)
            sync_mode (int): Synchronization mode:

                * 0 — Normal.
                * 1 — Timing: the *min_length* parameter specifies the bit length and the *max_length* parameter
                  specifies the timing length.

                  Used for UART, CAN protocols.
                * 2 — PWM: the *min_length* parameter specifies the sampling time and the *max_length* parameter
                  specifies the timing length.

                  Used for 1-wire protocols.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerLengthSet(self.hdwf, min_length, max_length, sync_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerMatchSet(self, pin: int, mask: int, value: int, bit_stuffing: int) -> None:
        '''Set |DigitalIn| trigger detector match.

        Todo:
            Figure out what this setting does.

        Configure the deserializer. The bits are left shifted.
        The mask and value should be specified accordingly, in MSB-first order.

        Parameters:
            pin (int): The pin to be deserialized.
            mask (int): The mask pattern.
            value (int): The bit pattern.
            bit_stuffing (int): The bit stuffing count.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInTriggerMatchSet(self.hdwf, pin, mask, value, bit_stuffing)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    ###################################################################################################################
    #                                                                                                                 #
    #                                                     COUNTER                                                     #
    #                                                                                                                 #
    ###################################################################################################################

    def counterInfo(self) -> Tuple[int, float]:
        '''Get |DigitalIn| counter info.'''
        c_counter_max = typespec_ctypes.c_double()
        c_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalInCounterInfo(self.hdwf, c_counter_max, c_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_counter_max.value.is_integer():
            raise PyDwfError('Bad c_counter_max value.')

        counter_max = int(c_counter_max.value)
        duration_max = c_duration_max.value
        return (counter_max, duration_max)

    def counterSet(self, duration: float) -> None:
        '''Set |DigitalIn| counter duration.'''
        result = self.lib.FDwfDigitalInCounterSet(self.hdwf, duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def counterGet(self) -> float:
        '''Get |DigitalIn| counter duration.'''
        c_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalInCounterGet(self.hdwf, c_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        duration = c_duration.value
        return duration

    def counterStatus(self) -> Tuple[float, float, int]:
        '''Get |DigitalIn| counter status.'''
        c_count = typespec_ctypes.c_double()
        c_freq = typespec_ctypes.c_double()
        c_tick = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalInCounterStatus(self.hdwf, c_count, c_freq, c_tick)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        count = c_count.value
        freq = c_freq.value
        tick = c_tick.value
        return (count, freq, tick)

    ###################################################################################################################
    #                                                                                                                 #
    #                                             MISCELLANEOUS SETTINGS                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def mixedSet(self, enable: bool) -> None:
        '''Set mixed state.

        Warning:
            **This method is obsolete.**

        Todo:
            Figure out what this setting does.

        Parameters:
            enable (bool): (to be documented)

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalInMixedSet(self.hdwf, enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
