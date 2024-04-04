'''The |pydwf.core.api.analog_in| module provides a single class: |AnalogIn|.'''

# pylint: disable=too-many-lines

from typing import Tuple, List

import numpy as np

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.enum_types import (DwfAnalogInTriggerLengthCondition, DwfTriggerSource,
                                             DwfAnalogInTriggerType, DwfTriggerSlope, DwfAnalogInFilter,
                                             DwfAcquisitionMode, DwfState, DwfAnalogCoupling)
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.exceptions import PyDwfError


class AnalogIn(AbstractDwfDeviceSubAPI):
    '''The |AnalogIn| class provides access to the analog input (oscilloscope) instrument of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its
        public |analogIn:link| attribute for access by the user.
    '''

    # pylint: disable=too-many-public-methods

    ###################################################################################################################
    #                                                                                                                 #
    #                                               INSTRUMENT CONTROL                                                #
    #                                                                                                                 #
    ###################################################################################################################

    def reset(self) -> None:
        '''Reset all |AnalogIn| instrument parameters to default values.

        If autoconfiguration is enabled at the device level, the reset operation is performed immediately;
        otherwise, an explicit call to the :py:meth:`configure` method is required.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfAnalogInReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self, reconfigure: bool, start: bool) -> None:
        '''Configure the instrument and start or stop the acquisition operation.

        Parameters:
            reconfigure (bool): If True, the instrument settings are sent to the instrument.
                In addition, the auto-trigger timeout is reset.
            start (bool): If True, an acquisition is started. If False, an ongoing acquisition is stopped.

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfAnalogInConfigure(self.hdwf, reconfigure, start)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self, read_data_flag: bool) -> DwfState:
        '''Get the |AnalogIn| instrument state.

        This method performs a status request to the |AnalogIn| instrument and receives its response.

        The following methods can be used to retrieve |AnalogIn| instrument status information
        as a result of this call, regardless of the value of the *read_data_flag* parameter:

        * :py:meth:`~statusTime`
        * :py:meth:`~statusSample`
        * :py:meth:`~statusAutoTriggered`
        * :py:meth:`~statusSamplesLeft`
        * :py:meth:`~statusSamplesValid`
        * :py:meth:`~statusIndexWrite`
        * :py:meth:`~statusRecord`

        The following methods can be used to retrieve bulk data obtained from the |AnalogIn| instrument
        as a result of this call, but only if the *read_data_flag* parameter is True:

        * :py:meth:`~statusData`
        * :py:meth:`~statusData2`
        * :py:meth:`~statusData16`
        * :py:meth:`~statusNoise`
        * :py:meth:`~statusNoise2`

        Parameters:
            read_data_flag (bool): If True, read sample data from the instrument.

                In :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Single` acquisition mode,
                the data will be read only when the acquisition is finished.

        Returns:
            DwfState: The status of the |AnalogIn| instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        c_status = typespec_ctypes.DwfState()
        result = self.lib.FDwfAnalogInStatus(self.hdwf, read_data_flag, c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        status_ = DwfState(c_status.value)
        return status_

    ###################################################################################################################
    #                                                                                                                 #
    #                                                STATUS VARIABLES                                                 #
    #                                                                                                                 #
    ###################################################################################################################

    def statusTime(self) -> Tuple[int, int, int]:
        '''Retrieve the timestamp of the current status information.

        Returns:
            Tuple[int, int, int]: A three-element tuple, indicating the POSIX timestamp of the status request.
            The first element is the POSIX second, the second and third element are the numerator and denominator,
            respectively, of the fractional part of the second.

            In case :py:meth:`status` hasn't been called yet, this method will return zeroes
            for all three tuple elements.

        Raises:
            DwfLibraryError: An error occurred while retrieving the status time.
        '''
        c_sec_utc = typespec_ctypes.c_unsigned_int()
        c_tick = typespec_ctypes.c_unsigned_int()
        c_ticks_per_second = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfAnalogInStatusTime(self.hdwf, c_sec_utc, c_tick, c_ticks_per_second)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sec_utc = c_sec_utc.value
        tick = c_tick.value
        ticks_per_second = c_ticks_per_second.value
        return (sec_utc, tick, ticks_per_second)

    def statusSample(self, channel_index: int) -> float:
        '''Get the last ADC conversion sample from the specified |AnalogIn| instrument channel, in Volts.

        Note:
            This value is updated even if the :py:meth:`status` method is called with argument False.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The most recent ADC value of this channel, in Volts.

        Raises:
            DwfLibraryError: An error occurred while retrieving the sample value.
        '''
        c_sample = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInStatusSample(self.hdwf, channel_index, c_sample)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sample = c_sample.value
        return sample

    def statusAutoTriggered(self) -> bool:
        '''Check if the current acquisition is auto-triggered.

        Returns:
            bool: True if the current acquisition is auto-triggered, False otherwise.

        Raises:
            DwfLibraryError: An error occurred while retrieving the auto-triggered status.
        '''
        c_auto_triggered = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInStatusAutoTriggered(self.hdwf, c_auto_triggered)
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
            DwfLibraryError: An error occurred while retrieving the number of samples left.
        '''
        c_samples_left = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInStatusSamplesLeft(self.hdwf, c_samples_left)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_left = c_samples_left.value
        return samples_left

    def statusSamplesValid(self) -> int:
        '''Retrieve the number of valid acquired data samples.

        In :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Single` acquisition mode, valid samples are
        returned when :py:meth:`status` reports a result of :py:attr:`~pydwf.core.auxiliary.enum_types.DwfState.Done`.

        The actual number of samples transferred and reported back here is equal to max(16, :py:meth:`bufferSizeGet`).

        Returns:
            int: The number of valid samples.

        Raises:
            DwfLibraryError: An error occurred while retrieving the number of valid samples.
        '''
        c_samples_valid = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInStatusSamplesValid(self.hdwf, c_samples_valid)
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
            DwfLibraryError: An error occurred while retrieving the write-index.
        '''
        c_index_write = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInStatusIndexWrite(self.hdwf, c_index_write)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        index_write = c_index_write.value
        return index_write

    def statusRecord(self) -> Tuple[int, int, int]:
        '''Retrieve information about the recording process.

        Data loss occurs when the device acquisition is faster than the read process to the PC.

        If this happens, the device recording buffer is filled and data samples are overwritten.

        Corrupt samples indicate that the samples have been overwritten by the acquisition process during the
        previous read.

        In this case, try optimizing the loop process for faster execution or reduce the acquisition frequency or
        record length to be less than or equal to the device buffer size (i.e., record_length is less than or
        equal to buffer_size / sample_frequency).

        Returns:
            Tuple[int, int, int]: A three-element tuple containing the counts for
            *available*, *lost*, and *corrupt* data samples, in that order.

        Raises:
            DwfLibraryError: An error occurred while retrieving the record status.
        '''
        c_data_available = typespec_ctypes.c_int()
        c_data_lost = typespec_ctypes.c_int()
        c_data_corrupt = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInStatusRecord(
            self.hdwf,
            c_data_available,
            c_data_lost,
            c_data_corrupt)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        data_available = c_data_available.value
        data_lost = c_data_lost.value
        data_corrupt = c_data_corrupt.value
        return (data_available, data_lost, data_corrupt)

    ###################################################################################################################
    #                                                                                                                 #
    #                                              STATUS DATA RETRIEVAL                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def statusData(self, channel_index: int, count: int) -> np.ndarray:
        '''Retrieve the acquired data samples from the specified |AnalogIn| instrument channel.

        This method returns samples as voltages, calculated from the raw, binary sample values as follows:

        .. code-block:: python

            voltages = analogIn.channelOffsetGet(channel_index) + \\
                       analogIn.channelRangeGet(channel_index) * (raw_samples / 65536.0)

        Note that the applied calibration is channel-dependent.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            count (int): The number of samples to retrieve.

        Returns:
            nd.array: A 1D numpy array of floats, in Volts.

        Raises:
            DwfLibraryError: An error occurred while retrieving the sample data.
        '''
        samples = np.empty(count, dtype=np.float64)
        result = self.lib.FDwfAnalogInStatusData(
            self.hdwf,
            channel_index,
            samples.ctypes.data_as(typespec_ctypes.c_double_ptr),
            count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusData2(self, channel_index: int, offset: int, count: int) -> np.ndarray:
        '''Retrieve the acquired data samples from the specified |AnalogIn| instrument channel.

        This method returns samples as voltages, calculated from the raw, binary sample values as follows:

        .. code-block:: python

            voltages = analogIn.channelOffsetGet(channel_index) + \\
                       analogIn.channelRangeGet(channel_index) * (raw_samples / 65536.0)

        Note:
            The applied calibration is channel-dependent.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            offset (int): Sample offset.
            count (int): Sample count.

        Returns:
            nd.array: A 1D numpy array of floats, in Volts.

        Raises:
            DwfLibraryError: An error occurred while retrieving the sample data.
        '''
        samples = np.empty(count, dtype=np.float64)
        result = self.lib.FDwfAnalogInStatusData2(
            self.hdwf,
            channel_index,
            samples.ctypes.data_as(typespec_ctypes.c_double_ptr),
            offset,
            count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusData16(self, channel_index: int, offset: int, count: int) -> np.ndarray:
        '''Retrieve the acquired data samples from the specified |AnalogIn| instrument channel.

        This method returns raw, signed 16-bit samples.

        In case the ADC has less than 16 bits of raw resolution,
        least significant zero-bits are added to stretch the range to 16 bits.

        To convert these raw samples to voltages, use the following:

        .. code-block:: python

            voltages = analogIn.channelOffsetGet(channel_index) + \\
                       analogIn.channelRangeGet(channel_index) * (raw_samples / 65536.0)

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            offset (int): The sample offset to start copying from.
            count (int): The number of samples to retrieve.

        Returns:
            nd.array: A 1D numpy array of 16-bit signed integers.

        Raises:
            DwfLibraryError: An error occurred while retrieving the sample data.
        '''
        samples = np.empty(count, dtype=np.int16)
        result = self.lib.FDwfAnalogInStatusData16(
            self.hdwf,
            channel_index,
            samples.ctypes.data_as(typespec_ctypes.c_short_ptr),
            offset,
            count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return samples

    def statusNoise(self, channel_index: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
        '''Retrieve the acquired noise samples from the specified |AnalogIn| instrument channel.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            count (int): Sample count.

        Returns:
            A two-element tuple; each element is a 1D numpy array of floats, in Volts.

            The first array contains the minimum values, the second array contains the maximum values.

        Raises:
            DwfLibraryError: An error occurred while retrieving the noise data.
        '''
        noise_min = np.empty(count, dtype=np.float64)
        noise_max = np.empty(count, dtype=np.float64)
        result = self.lib.FDwfAnalogInStatusNoise(
            self.hdwf,
            channel_index,
            noise_min.ctypes.data_as(typespec_ctypes.c_double_ptr),
            noise_max.ctypes.data_as(typespec_ctypes.c_double_ptr),
            count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return (noise_min, noise_max)

    def statusNoise2(self, channel_index: int, offset: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
        '''Retrieve the acquired data samples from the specified |AnalogIn| instrument channel.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            offset (int): Sample offset.
            count (int): Sample count.

        Returns:
            A two-element tuple; each element is a 1D numpy array of floats, in Volts.

            The first array contains the minimum values, the second array contains the maximum values.

        Raises:
            DwfLibraryError: An error occurred while retrieving the noise data.
        '''
        noise_min = np.empty(count, dtype=np.float64)
        noise_max = np.empty(count, dtype=np.float64)
        result = self.lib.FDwfAnalogInStatusNoise2(
            self.hdwf,
            channel_index,
            noise_min.ctypes.data_as(typespec_ctypes.c_double_ptr),
            noise_max.ctypes.data_as(typespec_ctypes.c_double_ptr),
            offset,
            count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return (noise_min, noise_max)

    ###################################################################################################################
    #                                                                                                                 #
    #                                            ACQUISITION CONFIGURATION                                            #
    #                                                                                                                 #
    ###################################################################################################################

    def bitsInfo(self) -> int:
        '''Get the fixed the number of bits used by the |AnalogIn| ADC.

        The number of bits can only be queried; it cannot be changed.

        Note:
            The Analog Discovery 2 uses an |Analog Devices AD9648| two-channel ADC.
            It converts 14-bit samples at a rate of up to 125 MHz.
            So for the Analog Discovery 2, this method always returns 14.

        Returns:
            int: The number of bits per sample for each of the |AnalogIn| channels.

        Raises:
            DwfLibraryError: An error occurred while getting the number of bits.
        '''
        c_num_bits = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInBitsInfo(self.hdwf, c_num_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        num_bits = c_num_bits.value
        return num_bits

    def recordLengthSet(self, record_duration: float) -> None:
        '''Set the |AnalogIn| record length, in seconds.

        Note:
            This value is only used when the acquisition mode is configured as
            :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record`.

        Parameters:
            record_duration (float): The record duration to be configured, in seconds.

                A record duration of 0.0 (zero) seconds indicates a request for an arbitrary-length record acquisition.

        Raises:
            DwfLibraryError: An error occurred while setting the record duration.
        '''
        result = self.lib.FDwfAnalogInRecordLengthSet(self.hdwf, record_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def recordLengthGet(self) -> float:
        '''Get the |AnalogIn| record length, in seconds.

        Note:
            This value is only used when the acquisition mode is configured as
            :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record`.

        Returns:
            float: The currently configured record length, in seconds.

            A record length of 0.0 (zero) seconds indicates a request for an arbitrary-length record acquisition.

        Raises:
            DwfLibraryError: An error occurred while getting the record length.
        '''
        c_record_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInRecordLengthGet(self.hdwf, c_record_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        record_duration = c_record_duration.value
        return record_duration

    def frequencyInfo(self) -> Tuple[float, float]:
        '''Retrieve the minimum and maximum configurable ADC sample frequency of the |AnalogIn| instrument,
        in samples/second.

        Returns:
            Tuple[float, float]: The valid sample frequency range (min, max), in samples/second.

        Raises:
            DwfLibraryError: An error occurred while getting the allowed sample frequency range.
        '''
        c_frequency_min = typespec_ctypes.c_double()
        c_frequency_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInFrequencyInfo(self.hdwf, c_frequency_min, c_frequency_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency_min = c_frequency_min.value
        frequency_max = c_frequency_max.value
        return (frequency_min, frequency_max)

    def frequencySet(self, sample_frequency: float) -> None:
        '''Set the ADC sample frequency of the |AnalogIn| instrument, in samples/second.

        Parameters:
            sample_frequency (float): Sample frequency, in samples/second.

        Raises:
            DwfLibraryError: An error occurred while setting sample frequency.
        '''
        result = self.lib.FDwfAnalogInFrequencySet(self.hdwf, sample_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def frequencyGet(self) -> float:
        '''Get the ADC sample frequency of the |AnalogIn| instrument, in samples/second.

        The ADC always runs at maximum frequency, but the method in which the samples are stored and transferred
        can be configured individually for each channel with the `channelFilterSet` method.

        Returns:
            float: The configured sample frequency, in samples/second.

        Raises:
            DwfLibraryError: An error occurred while getting the sample frequency.
        '''
        c_sample_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInFrequencyGet(self.hdwf, c_sample_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sample_frequency = c_sample_frequency.value
        return sample_frequency

    def bufferSizeInfo(self) -> Tuple[int, int]:
        '''Returns the minimum and maximum allowable buffer size for the |AnalogIn| instrument, in samples.

        When using the :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record` acquisition mode,
        the buffer size should be left at the default value, which is equal to the maximum value.
        In other modes (e.g. :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Single`), the buffer
        size determines the size of the acquisition window.

        Note:
            The maximum buffer size depends on the device configuration that was selected while opening the device.

            For example, on the Analog Discovery 2, the maximum |AnalogIn| buffer size can be 512, 2048, 8192, or
            16384, depending on the device configuration.

        Returns:
            Tuple[int, int]: A two-element tuple.
            The first element is the minimum buffer size, the second element is the maximum buffer size.

        Raises:
            DwfLibraryError: An error occurred while getting the buffer size info.
        '''
        c_buffer_size_min = typespec_ctypes.c_int()
        c_buffer_size_max = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInBufferSizeInfo(self.hdwf, c_buffer_size_min, c_buffer_size_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        buffer_size_min = c_buffer_size_min.value
        buffer_size_max = c_buffer_size_max.value
        return (buffer_size_min, buffer_size_max)

    def bufferSizeSet(self, buffer_size: int) -> None:
        '''Adjust the |AnalogIn| instrument buffer size, expressed in samples.

        The actual buffer size configured will be clipped by the :py:meth:`bufferSizeInfo` values.

        The actual value configured can be read back by calling :py:meth:`bufferSizeGet`.

        Parameters:
            buffer_size (int): The requested buffer size, in samples.

        Raises:
            DwfLibraryError: An error occurred while setting the buffer size.
        '''
        result = self.lib.FDwfAnalogInBufferSizeSet(self.hdwf, buffer_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def bufferSizeGet(self) -> int:
        '''Return the used |AnalogIn| instrument buffer size, in samples.

        Returns:
            int: The currently configured buffer size, in samples.

        Raises:
            DwfLibraryError: An error occurred while getting the buffer size.
        '''
        c_buffer_size = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInBufferSizeGet(self.hdwf, c_buffer_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        buffer_size = c_buffer_size.value
        return buffer_size

    def noiseSizeInfo(self) -> int:
        '''Return the maximum noise buffer size for the |AnalogIn| instrument, in samples.

        Returns:
            The maximum noise buffer size, in samples.

        Raises:
            DwfLibraryError: An error occurred while getting the noise buffer size info.
        '''
        c_noise_size = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInNoiseSizeInfo(self.hdwf, c_noise_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        noise_size = c_noise_size.value
        return noise_size

    def noiseSizeSet(self, noise_buffer_size: int) -> None:
        '''Enable or disable the noise buffer for the |AnalogIn| instrument.

        This method determines if the noise buffer is enabled or disabled.

        Note:
            The name of this method and the type of its parameter (int) suggest that it can be used
            to specify the size of the noise buffer, but that is not the case.

            Any non-zero value enables the noise buffer; a zero value disables it.

            If enabled, the noise buffer size reported by :py:meth:`noiseSizeGet` is always equal to
            the size of the sample buffer reported by :py:meth:`bufferSizeGet`, divided by 8.

        Parameters:
            noise_buffer_size (int): Whether to enable (non-zero) or disable (zero) the noise buffer.

        Raises:
            DwfLibraryError: An error occurred while setting the noise buffer enabled/disabled state.
        '''
        result = self.lib.FDwfAnalogInNoiseSizeSet(self.hdwf, int(noise_buffer_size))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def noiseSizeGet(self) -> int:
        '''Return the currently configured noise buffer size for the |AnalogIn| instrument, in samples.

        This value is automatically adjusted according to the sample buffer size, divided by 8.
        For instance, setting the sample buffer size of 8192 implies a noise buffer size of 1024;
        setting the sample buffer size to 4096 implies noise buffer size will be 512.

        Returns:
            int: The currently configured noise buffer size. Zero indicates that the noise buffer is disabled.

        Raises:
            DwfLibraryError: An error occurred while getting the noise buffer size.
        '''
        c_noise_buffer_size = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInNoiseSizeGet(self.hdwf, c_noise_buffer_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        noise_buffer_size = c_noise_buffer_size.value
        return noise_buffer_size

    def acquisitionModeInfo(self) -> List[DwfAcquisitionMode]:
        '''Get a list of valid |AnalogIn| instrument acquisition modes.

        Returns:
            List[DwfAcquisitionMode]: A list of valid acquisition modes for the |AnalogIn| instrument.

        Raises:
            DwfLibraryError: An error occurred while getting the acquisition mode info.
        '''

        c_acquisition_mode_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInAcquisitionModeInfo(self.hdwf, c_acquisition_mode_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        acquisition_mode_bitset = c_acquisition_mode_bitset.value
        acquisition_mode_list = [acquisition_mode for acquisition_mode in DwfAcquisitionMode
                                 if acquisition_mode_bitset & (1 << acquisition_mode.value)]
        return acquisition_mode_list

    def acquisitionModeSet(self, acquisition_mode: DwfAcquisitionMode) -> None:
        '''Set the |AnalogIn| acquisition mode.

        Parameters:
            acquisition_mode (DwfAcquisitionMode): The acquisition mode to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the acquisition mode.
        '''
        result = self.lib.FDwfAnalogInAcquisitionModeSet(self.hdwf, acquisition_mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def acquisitionModeGet(self) -> DwfAcquisitionMode:
        '''Get the currently configured |AnalogIn| acquisition mode.

        Returns:
            DwfAcquisitionMode: The acquisition mode currently configured.

        Raises:
            DwfLibraryError: An error occurred while getting the acquisition mode.
        '''
        c_acquisition_mode = typespec_ctypes.DwfAcquisitionMode()
        result = self.lib.FDwfAnalogInAcquisitionModeGet(self.hdwf, c_acquisition_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        acquisition_mode = DwfAcquisitionMode(c_acquisition_mode.value)
        return acquisition_mode

    ###################################################################################################################
    #                                                                                                                 #
    #                                                  CHANNEL COUNT                                                  #
    #                                                                                                                 #
    ###################################################################################################################

    def channelCount(self) -> int:
        '''Read the number of |AnalogIn| input channels.

        Returns:
            int: The number of analog input channels.

        Raises:
            DwfLibraryError: An error occurred while retrieving the number of analog input channels.
        '''
        c_channel_count = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelCount(self.hdwf, c_channel_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_count = c_channel_count.value
        return channel_count

    def channelCounts(self) -> Tuple[int, int, int]:
        '''Read the number of |AnalogIn| input channels, distinguishing between real and filter channels.

        Returns:
            Tuple[int, int, int]: The number of real, fiiltered, and total analog input channels.

        Raises:
            DwfLibraryError: An error occurred while retrieving the number of analog input channels.
        '''
        c_channel_count_real = typespec_ctypes.c_int()
        c_channel_count_filter = typespec_ctypes.c_int()
        c_channel_count_total = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelCounts(
            self.hdwf, c_channel_count_real, c_channel_count_filter, c_channel_count_total)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_count_real = c_channel_count_real.value
        channel_count_filter = c_channel_count_filter.value
        channel_count_total = c_channel_count_total.value
        return (channel_count_real, channel_count_filter, channel_count_total)

    ###################################################################################################################
    #                                                                                                                 #
    #                                              CHANNEL CONFIGURATION                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def channelEnableSet(self, channel_index: int, channel_enable: bool) -> None:
        '''Enable or disable the specified |AnalogIn| input channel.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_enable (bool): Whether to enable (True) or disable (False) the specified channel.

        Raises:
            DwfLibraryError: An error occurred while enabling or disabling the channel.
        '''
        result = self.lib.FDwfAnalogInChannelEnableSet(self.hdwf, channel_index, channel_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelEnableGet(self, channel_index: int) -> bool:
        '''Get the current enable/disable status of the specified |AnalogIn| input channel.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            bool: Channel is enabled (True) or disabled (False).

        Raises:
            DwfLibraryError: An error occurred while getting the enabled/disabled state of the channel.
        '''
        c_channel_enable = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelEnableGet(self.hdwf, channel_index, c_channel_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_enable = bool(c_channel_enable.value)
        return channel_enable

    def channelFilterInfo(self) -> List[DwfAnalogInFilter]:
        '''Get a list of valid |AnalogIn| channel filter settings.

        Returns:
            List[DwfAnalogInFilter]: A list of valid channel filter settings.

        Raises:
            DwfLibraryError: An error occurred while getting the channel filter info.
        '''
        c_channel_filter_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelFilterInfo(self.hdwf, c_channel_filter_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_filter_bitset = c_channel_filter_bitset.value
        channel_filter_list = [channel_filter for channel_filter in DwfAnalogInFilter
                               if channel_filter_bitset & (1 << channel_filter.value)]
        return channel_filter_list

    def channelFilterSet(self, channel_index: int, channel_filter: DwfAnalogInFilter) -> None:
        '''Set the filter for a specified |AnalogIn| input channel.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_filter (DwfAnalogInFilter): The channel filter mode to be selected.

        Raises:
            DwfLibraryError: An error occurred while setting the channel filter.
        '''
        result = self.lib.FDwfAnalogInChannelFilterSet(self.hdwf, channel_index, channel_filter.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelFilterGet(self, channel_index: int) -> DwfAnalogInFilter:
        '''Get the |AnalogIn| input channel filter setting.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            DwfAnalogInFilter: The currently selected channel filter mode.

        Raises:
            DwfLibraryError: An error occurred while getting the current channel filter setting.
        '''
        c_channel_filter = typespec_ctypes.DwfAnalogInFilter()
        result = self.lib.FDwfAnalogInChannelFilterGet(self.hdwf, channel_index, c_channel_filter)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_filter = DwfAnalogInFilter(c_channel_filter.value)
        return channel_filter

    def channelRangeInfo(self) -> Tuple[float, float, int]:
        '''Report the possible voltage range of the |AnalogIn| input channels, in Volts.

        The values returned represent ideal values.
        The actual calibrated ranges are channel-dependent.

        See Also:
            The :py:meth:`channelRangeSteps` method returns essentially the same information
            in a different representation.

        Returns:
            Tuple[float, float, int]: The minimum range (Volts), maximum range (Volts),
            and number of different discrete channel range settings of the |AnalogIn| instrument.

        Raises:
            DwfLibraryError: An error occurred while getting the analog input range setting info.
        '''
        c_channel_range_min = typespec_ctypes.c_double()
        c_channel_range_max = typespec_ctypes.c_double()
        c_channel_range_num_steps = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelRangeInfo(
            self.hdwf,
            c_channel_range_min,
            c_channel_range_max,
            c_channel_range_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        channel_range_min = c_channel_range_min.value
        channel_range_max = c_channel_range_max.value
        if not c_channel_range_num_steps.value.is_integer():
            raise PyDwfError("Bad c_channel_range_num_steps value.")
        channel_range_num_steps = int(c_channel_range_num_steps.value)
        return (channel_range_min, channel_range_max, channel_range_num_steps)

    def channelRangeSet(self, channel_index: int, channel_range: float) -> None:
        '''Set the range setting of the specified |AnalogIn| input channel, in Volts.

        Note:
            The actual range set will generally be different from the requested range.

        Note:
            Changing the channel range may also change the channel offset.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_range (float): The requested channel range, in Volts.

        Raises:
            DwfLibraryError: An error occurred while setting the channel voltage range.
        '''
        result = self.lib.FDwfAnalogInChannelRangeSet(self.hdwf, channel_index, channel_range)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelRangeGet(self, channel_index: int) -> float:
        '''Get the range setting of the specified |AnalogIn| input channel, in Volts.

        Together with the channel offset, this value can be used to transform raw binary ADC values into Volts.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The actual channel range, in Volts.

        Raises:
            DwfLibraryError: An error occurred while setting the channel voltage range.
        '''
        c_channel_range = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelRangeGet(self.hdwf, channel_index, c_channel_range)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_range = c_channel_range.value
        return channel_range

    def channelRangeSteps(self) -> List[float]:
        '''Report the possible voltage ranges of the |AnalogIn| input channels, in Volts, as a list.

        The values returned represent ideal values.
        The actual calibrated ranges are channel-dependent.

        See Also:
            The :py:meth:`channelRangeInfo` method returns essentially the same information
            in a different representation.

        Returns:
            List[float]: A list of ranges, in Volts, representing the discrete
            channel range settings of the |AnalogIn| instrument.

        Raises:
            DwfLibraryError: An error occurred while getting the list of analog input range settings.
        '''
        c_channel_range_array = typespec_ctypes.c_double_array_32()
        c_channel_range_array_size = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelRangeSteps(self.hdwf, c_channel_range_array, c_channel_range_array_size)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_range_array_size = c_channel_range_array_size.value
        channel_range_array = [c_channel_range_array[index] for index in range(channel_range_array_size)]
        return channel_range_array

    def channelOffsetInfo(self) -> Tuple[float, float, int]:
        '''Get the possible |AnalogIn| input channel offset settings, in Volts.

        Returns:
            Tuple[float, float, int]: The minimum channel offset (Volts), maximum channel offset (Volts),
            and number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the channel offset info.
        '''
        c_offset_min = typespec_ctypes.c_double()
        c_offset_max = typespec_ctypes.c_double()
        c_offset_num_steps = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelOffsetInfo(self.hdwf, c_offset_min, c_offset_max, c_offset_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset_min = c_offset_min.value
        offset_max = c_offset_max.value
        if not c_offset_num_steps.value.is_integer():
            raise PyDwfError("Bad c_offset_num_steps value.")
        offset_num_steps = int(c_offset_num_steps.value)
        return (offset_min, offset_max, offset_num_steps)

    def channelOffsetSet(self, channel_index: int, channel_offset: float) -> None:
        '''Set the |AnalogIn| input channel offset, in Volts.

        Note:
            The actual offset will generally be different from the requested offset.

        Note:
            Changing the channel offset may also change the channel range.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_offset (float): The channel offset, in Volts.

        Raises:
            DwfLibraryError: An error occurred while setting the channel offset.
        '''
        result = self.lib.FDwfAnalogInChannelOffsetSet(self.hdwf, channel_index, channel_offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelOffsetGet(self, channel_index: int) -> float:
        '''Get the |AnalogIn| input channel offset, in Volts.

        Together with the channel range, this value can be used to transform raw binary ADC values into Volts.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The currently configured channel offset, in Volts.

        Raises:
            DwfLibraryError: An error occurred while getting the current channel offset setting.
        '''
        c_channel_offset = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelOffsetGet(self.hdwf, channel_index, c_channel_offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_offset = c_channel_offset.value
        return channel_offset

    def channelAttenuationSet(self, channel_index: int, channel_attenuation: float) -> None:
        '''Set the |AnalogIn| input channel attenuation setting.

        The channel attenuation is a dimensionless factor.

        This setting is used to compensate for probe attenuation.
        Many probes have two attenuation settings (e.g., ×1 and ×10).
        The value of this setting should correspond to the value of the probe, or 1 (the default)
        if a direct connection without attenuation is used.

        Note:
            Changing the channel attenuation will also change the channel offset and range.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_attenuation (float): The requested channel attenuation setting.
                If it is 0.0, the attenuation is set to 1.0 (the default) instead.

        Raises:
            DwfLibraryError: An error occurred while setting the current channel attenuation.
        '''
        result = self.lib.FDwfAnalogInChannelAttenuationSet(
            self.hdwf,
            channel_index,
            channel_attenuation)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelAttenuationGet(self, channel_index: int) -> float:
        '''Get the |AnalogIn| input channel attenuation setting.

        The channel attenuation is a dimensionless factor.

        This setting is used to compensate for probe attenuation.
        Many probes have two attenuation settings (e.g., ×1 and ×10).
        The value of this setting should correspond to the value of the probe, or 1 (the default)
        if a direct connection without attenuation is used.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The channel attenuation setting.

        Raises:
            DwfLibraryError: An error occurred while getting the current channel attenuation.
        '''
        c_channel_attenuation = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelAttenuationGet(
            self.hdwf,
            channel_index,
            c_channel_attenuation)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_attenuation = c_channel_attenuation.value
        return channel_attenuation

    def channelBandwidthSet(self, channel_index: int, channel_bandwidth: float) -> None:
        '''Set the |AnalogIn| input channel bandwidth setting.

        Note:
            On the Analog Discovery 2, the channel bandwidth setting exists and can be set and retrieved,
            but the value has no effect.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_bandwidth (float): The channel bandwidth setting, in Hz.

        Raises:
            DwfLibraryError: An error occurred while setting the channel bandwidth.
        '''
        result = self.lib.FDwfAnalogInChannelBandwidthSet(self.hdwf, channel_index, channel_bandwidth)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelBandwidthGet(self, channel_index: int) -> float:
        '''Get the |AnalogIn| input channel bandwidth setting.

        Note:
            On the Analog Discovery 2, the channel bandwidth setting exists and can be set and retrieved,
            but the value has no effect.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The channel bandwidth setting, in Hz.

        Raises:
            DwfLibraryError: An error occurred while getting the current channel bandwidth.
        '''
        c_channel_bandwidth = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelBandwidthGet(self.hdwf, channel_index, c_channel_bandwidth)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_bandwidth = c_channel_bandwidth.value
        return channel_bandwidth

    def channelImpedanceSet(self, channel_index: int, channel_impedance: float) -> None:
        '''Set the |AnalogIn| input channel impedance setting, in Ohms.

        Note:
            On the Analog Discovery 2, the channel impedance setting exists and can be set and retrieved,
            but the value has no effect.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_impedance (float): channel impedance setting, in Ohms.

        Raises:
            DwfLibraryError: An error occurred while setting the current channel impedance.
        '''
        result = self.lib.FDwfAnalogInChannelImpedanceSet(self.hdwf, channel_index, channel_impedance)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelImpedanceGet(self, channel_index: int) -> float:
        '''Get the |AnalogIn| input channel impedance setting, in Ohms.

        Note:
            On the Analog Discovery 2, the channel impedance setting exists and can be set and retrieved,
            but the value has no effect.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            float: The channel impedance setting, in Ohms.

        Raises:
            DwfLibraryError: An error occurred while getting the current channel impedance.
        '''
        c_channel_impedance = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInChannelImpedanceGet(self.hdwf, channel_index, c_channel_impedance)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_impedance = c_channel_impedance.value
        return channel_impedance

    def channelCouplingInfo(self) -> List[DwfAnalogCoupling]:
        '''Get the |AnalogIn| channel coupling info.

        Raises:
            DwfLibraryError: An error occurred while getting the channel coupling info.
        '''
        c_channel_coupling_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelCouplingInfo(self.hdwf, c_channel_coupling_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_coupling_bitset = c_channel_coupling_bitset.value
        channel_coupling_list = [channel_coupling for channel_coupling in DwfAnalogCoupling
                                 if channel_coupling_bitset & (1 << channel_coupling.value)]
        return channel_coupling_list

    def channelCouplingSet(self, channel_index: int, channel_coupling: DwfAnalogCoupling) -> None:
        '''Set the |AnalogIn| input channel coupling.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.
            channel_coupling (AnalogCoupling): channel coupling to be set.

        Raises:
            DwfLibraryError: An error occurred while setting the current channel coupling.
        '''
        result = self.lib.FDwfAnalogInChannelCouplingSet(self.hdwf, channel_index, channel_coupling.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelCouplingGet(self, channel_index: int) -> DwfAnalogCoupling:
        '''Get the |AnalogIn| input channel impedance setting, in Ohms.

        Parameters:
            channel_index (int): The channel index, in the range 0 to :py:meth:`channelCount`-1.

        Returns:
            DwfAnalogCoupling: The channel coupling (DC or AC).

        Raises:
            DwfLibraryError: An error occurred while getting the current channel impedance.
        '''
        c_channel_coupling = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInChannelCouplingGet(self.hdwf, channel_index, c_channel_coupling)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_coupling = DwfAnalogCoupling(c_channel_coupling.value)
        return channel_coupling

    ###################################################################################################################
    #                                                                                                                 #
    #                                        INSTRUMENT TRIGGER CONFIGURATION                                         #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerSourceInfo(self) -> List[DwfTriggerSource]:
        '''Get the |AnalogIn| instrument trigger source info.

        Warning:
            **This method is obsolete.**

            Use the generic |DwfDevice.triggerInfo:link| method instead.

        Returns:
            List[DwfTriggerSource]: A list of trigger sources that can be selected.

        Raises:
            DwfLibraryError: An error occurred while retrieving the trigger source information.
        '''
        c_trigger_source_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerSourceInfo(self.hdwf, c_trigger_source_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source_bitset = c_trigger_source_bitset.value
        trigger_source_list = [trigger_source for trigger_source in DwfTriggerSource
                               if trigger_source_bitset & (1 << trigger_source.value)]
        return trigger_source_list

    def triggerSourceSet(self, trigger_source: DwfTriggerSource) -> None:
        '''Set the |AnalogIn| instrument trigger source.

        Parameters:
            trigger_source (DwfTriggerSource): The trigger source to be selected.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger source.
        '''
        result = self.lib.FDwfAnalogInTriggerSourceSet(self.hdwf, trigger_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSourceGet(self) -> DwfTriggerSource:
        '''Get the currently selected instrument trigger source.

        Returns:
            DwfTriggerSource: The currently selected trigger source.

        Raises:
            DwfLibraryError: An error occurred while retrieving the selected trigger source.
        '''
        c_trigger_source = typespec_ctypes.DwfTriggerSource()
        result = self.lib.FDwfAnalogInTriggerSourceGet(self.hdwf, c_trigger_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source = DwfTriggerSource(c_trigger_source.value)
        return trigger_source

    def triggerPositionInfo(self) -> Tuple[float, float, int]:
        '''Get the |AnalogIn| instrument trigger position range.

        Returns:
            Tuple[float, float, int]: The valid range of trigger positions that can be configured.
            The values returned are the *minimum* and *maximum* valid position settings, and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while retrieving the trigger position info.
        '''
        c_trigger_position_min = typespec_ctypes.c_double()
        c_trigger_position_max = typespec_ctypes.c_double()
        c_trigger_position_num_steps = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerPositionInfo(
            self.hdwf,
            c_trigger_position_min,
            c_trigger_position_max,
            c_trigger_position_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_position_num_steps.value.is_integer():
            raise PyDwfError("Bad c_trigger_position_num_steps value.")

        trigger_position_min = c_trigger_position_min.value
        trigger_position_max = c_trigger_position_max.value
        trigger_position_num_steps = int(c_trigger_position_num_steps.value)

        return (trigger_position_min, trigger_position_max, trigger_position_num_steps)

    def triggerPositionSet(self, trigger_position: float) -> None:
        '''Set the |AnalogIn| instrument trigger position, in seconds.

        The meaning of the trigger position depends on the currently selected acquisition mode:

        * In :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Record` acquisition mode, the trigger
          position is the time of the first valid sample acquired relative to the position of the trigger event.
          Negative values indicates times before the trigger time.

          To place the trigger in the middle of the recording, this value should be set to -0.5 times the duration
          of the recording.

        * In :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAcquisitionMode.Single` acquisition mode, the trigger
          position is the trigger event time relative to the center of the acquisition window.

          To place the trigger in the middle of the acquisition buffer, the value should be 0.

        Parameters:
            trigger_position (float): The trigger position to be configured, in seconds.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger position.
        '''
        result = self.lib.FDwfAnalogInTriggerPositionSet(self.hdwf, trigger_position)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerPositionGet(self) -> float:
        '''Get the |AnalogIn| instrument trigger position, in seconds.

        Returns:
            float: The currently configured trigger position, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger position.
        '''
        c_trigger_position = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerPositionGet(self.hdwf, c_trigger_position)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_position = c_trigger_position.value
        return trigger_position

    def triggerPositionStatus(self) -> float:
        '''Get the current |AnalogIn| instrument trigger position status.

        Returns:
            float: The current trigger position, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger position status.
        '''
        c_trigger_position = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerPositionStatus(self.hdwf, c_trigger_position)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_position = c_trigger_position.value
        return trigger_position

    def triggerAutoTimeoutInfo(self) -> Tuple[float, float, int]:
        '''Get the |AnalogIn| instrument trigger auto-timeout range, in seconds.

        Returns:
            Tuple[float, float, int]: The valid range of trigger auto-timeout values that can be configured.
            The values returned are the *minimum* and *maximum* valid auto-timeout settings, and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger auto-timeout info.
        '''
        c_auto_timeout_min = typespec_ctypes.c_double()
        c_auto_timeout_max = typespec_ctypes.c_double()
        c_auto_timeout_num_steps = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerAutoTimeoutInfo(
            self.hdwf,
            c_auto_timeout_min,
            c_auto_timeout_max,
            c_auto_timeout_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_auto_timeout_num_steps.value.is_integer():
            raise PyDwfError("Bad c_auto_timeout_num_steps value.")

        auto_timeout_min = c_auto_timeout_min.value
        auto_timeout_max = c_auto_timeout_max.value
        auto_timeout_num_steps = int(c_auto_timeout_num_steps.value)

        return (auto_timeout_min, auto_timeout_max, auto_timeout_num_steps)

    def triggerAutoTimeoutSet(self, trigger_auto_timeout: float) -> None:
        '''Set the |AnalogIn| instrument trigger auto-timeout value, in seconds.

        When set to 0, the trigger auto-timeout feature is disabled, corresponding to *Normal* acquisition
        mode on desktop oscilloscopes.

        Parameters:
            trigger_auto_timeout (float): The auto timeout setting, in seconds.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger auto-timeout value.
        '''
        result = self.lib.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, trigger_auto_timeout)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerAutoTimeoutGet(self) -> float:
        '''Get the |AnalogIn| instrument trigger auto-timeout value, in seconds.

        Returns:
            float: The currently configured auto-timeout value, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger auto-timeout value.
        '''
        c_trigger_auto_timeout = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerAutoTimeoutGet(self.hdwf, c_trigger_auto_timeout)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_auto_timeout = c_trigger_auto_timeout.value
        return trigger_auto_timeout

    ###################################################################################################################
    #                                                                                                                 #
    #                                            FORCE INSTRUMENT TRIGGER                                             #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerForce(self) -> None:
        '''Force assertion of the |AnalogIn| instrument trigger.

        Important:
            This method forces the |AnalogIn| device to act as if it was triggered, independent of the currently
            active trigger source. It does not generate an artificial trigger event on the |AnalogIn| trigger detector.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogInTriggerForce(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    ###################################################################################################################
    #                                                                                                                 #
    #                                         TRIGGER DETECTOR CONFIGURATION                                          #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerHoldOffInfo(self) -> Tuple[float, float, int]:
        '''Get the |AnalogIn| trigger detector holdoff range, in seconds.

        The trigger holdoff setting is the minimum time (in seconds) that should pass for a trigger to be
        recognized by the trigger detector after a previous trigger event.

        Returns:
            Tuple[float, float, int]: The valid range of trigger detector holdoff values that can be configured.
            The values returned are the *minimum* and *maximum* valid holdoff settings, and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger holdoff info.
        '''
        c_trigger_detector_holdoff_min = typespec_ctypes.c_double()
        c_trigger_detector_holdoff_max = typespec_ctypes.c_double()
        c_trigger_detector_holdoff_num_steps = typespec_ctypes.c_double()

        result = self.lib.FDwfAnalogInTriggerHoldOffInfo(
            self.hdwf,
            c_trigger_detector_holdoff_min,
            c_trigger_detector_holdoff_max,
            c_trigger_detector_holdoff_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_detector_holdoff_num_steps.value.is_integer():
            raise PyDwfError("Bad c_trigger_detector_holdoff_num_steps value.")

        trigger_detector_holdoff_min = c_trigger_detector_holdoff_min.value
        trigger_detector_holdoff_max = c_trigger_detector_holdoff_max.value
        trigger_detector_holdoff_num_steps = int(c_trigger_detector_holdoff_num_steps.value)

        return (trigger_detector_holdoff_min, trigger_detector_holdoff_max, trigger_detector_holdoff_num_steps)

    def triggerHoldOffSet(self, trigger_detector_holdoff: float) -> None:
        '''Set the |AnalogIn| trigger detector holdoff value, in seconds.

        Parameters:
            trigger_detector_holdoff (float): The trigger holdoff setting, in seconds.

        The value 0 disables the trigger detector holdoff feature.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector holdoff value.
        '''
        result = self.lib.FDwfAnalogInTriggerHoldOffSet(self.hdwf, trigger_detector_holdoff)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerHoldOffGet(self) -> float:
        '''Get the current |AnalogIn| trigger holdoff value, in seconds.

        Returns:
            float: The currently configured trigger holdoff value, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger holdoff value.
        '''
        c_trigger_detector_holdoff = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerHoldOffGet(self.hdwf, c_trigger_detector_holdoff)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_holdoff = c_trigger_detector_holdoff.value
        return trigger_detector_holdoff

    def triggerTypeInfo(self) -> List[DwfAnalogInTriggerType]:
        '''Get the valid |AnalogIn| trigger detector trigger-type values.

        This setting determines the type of event recognized by the |AnalogIn| trigger detector as a trigger.
        Possible types includes *Edge*, *Pulse*, *Transition*, and *Window*.

        Returns:
            List[DwfAnalogInTriggerType]: A list of trigger detector trigger-types that can be configured.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector trigger-type info.
        '''
        c_trigger_detector_type_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerTypeInfo(self.hdwf, c_trigger_detector_type_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_type_bitset = c_trigger_detector_type_bitset.value
        trigger_detector_type_list = [
            trigger_detector_type for trigger_detector_type in DwfAnalogInTriggerType
            if trigger_detector_type_bitset & (1 << trigger_detector_type.value)
        ]
        return trigger_detector_type_list

    def triggerTypeSet(self, trigger_detector_type: DwfAnalogInTriggerType) -> None:
        '''Set the |AnalogIn| trigger detector trigger-type.

        Parameters:
            trigger_detector_type (DwfAnalogInTriggerType): The trigger detector trigger-type to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector trigger-type.
        '''
        result = self.lib.FDwfAnalogInTriggerTypeSet(self.hdwf, trigger_detector_type.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerTypeGet(self) -> DwfAnalogInTriggerType:
        '''Get the currently configured |AnalogIn| trigger detector trigger-type.

        Returns:
            DwfAnalogInTriggerType: The currently configured trigger detector trigger-type.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector trigger-type.
        '''
        c_trigger_detector_type = typespec_ctypes.DwfAnalogInTriggerType()
        result = self.lib.FDwfAnalogInTriggerTypeGet(self.hdwf, c_trigger_detector_type)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_type = DwfAnalogInTriggerType(c_trigger_detector_type.value)
        return trigger_detector_type

    def triggerChannelInfo(self) -> Tuple[int, int]:
        '''Get the |AnalogIn| trigger detector channel range.

        The |AnalogIn| trigger detector monitors a specific analog in channel for trigger events.
        This method returns the range of valid analog input channels that can be configured
        as the |AnalogIn| trigger detector channel.

        Returns:
            Tuple[int, int]: The first and last channel that can be used as trigger detector channels.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector channel range.
        '''
        c_trigger_detector_channel_index_min = typespec_ctypes.c_int()
        c_trigger_detector_channel_index_max = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerChannelInfo(
            self.hdwf,
            c_trigger_detector_channel_index_min,
            c_trigger_detector_channel_index_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_channel_index_min = c_trigger_detector_channel_index_min.value
        trigger_detector_channel_index_max = c_trigger_detector_channel_index_max.value
        return (trigger_detector_channel_index_min, trigger_detector_channel_index_max)

    def triggerChannelSet(self, trigger_detector_channel_index: int) -> None:
        '''Set the |AnalogIn| trigger detector channel.

        This is the analog input channel that the |AnalogIn| trigger detector monitors for trigger events.

        Parameters:
            trigger_detector_channel_index (int): The trigger detector channel to be selected.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector channel.
        '''
        result = self.lib.FDwfAnalogInTriggerChannelSet(self.hdwf, trigger_detector_channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerChannelGet(self) -> int:
        '''Get the |AnalogIn| trigger detector channel.

        This is the analog input channel that the |AnalogIn| trigger detector monitors for trigger events.

        Returns:
            int: The currently configured trigger detector channel.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector channel.
        '''
        c_trigger_detector_channel_index = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerChannelGet(self.hdwf, c_trigger_detector_channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_channel_index = c_trigger_detector_channel_index.value
        return trigger_detector_channel_index

    def triggerFilterInfo(self) -> List[DwfAnalogInFilter]:
        '''Get a list of valid |AnalogIn| trigger detector filter values.

        Returns:
            List[DwfAnalogInFilter]: A list of filters that can be configured for the trigger detector channel.

        Raises:
            DwfLibraryError: An error occurred while getting the valid trigger detector channel filter values.
        '''
        c_trigger_detector_filter_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerFilterInfo(self.hdwf, c_trigger_detector_filter_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_filter_bitset = c_trigger_detector_filter_bitset.value
        trigger_detector_filter_list = [
            trigger_detector_filter for trigger_detector_filter in DwfAnalogInFilter
            if trigger_detector_filter_bitset & (1 << trigger_detector_filter.value)
        ]
        return trigger_detector_filter_list

    def triggerFilterSet(self, trigger_detector_filter: DwfAnalogInFilter) -> None:
        '''Set the |AnalogIn| trigger detector channel filter.

        Parameters:
            trigger_detector_filter (DwfAnalogInFilter): The trigger detector channel filter to be selected.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector channel filter.
        '''
        result = self.lib.FDwfAnalogInTriggerFilterSet(self.hdwf, trigger_detector_filter.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerFilterGet(self) -> DwfAnalogInFilter:
        '''Get the |AnalogIn| trigger detector channel filter.

        Returns:
            DwfAnalogInFilter: The currently configured trigger detector channel filter.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector channel filter.
        '''
        c_trigger_detector_filter = typespec_ctypes.DwfAnalogInFilter()
        result = self.lib.FDwfAnalogInTriggerFilterGet(self.hdwf, c_trigger_detector_filter)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_filter = DwfAnalogInFilter(c_trigger_detector_filter.value)
        return trigger_detector_filter

    def triggerLevelInfo(self) -> Tuple[float, float, int]:
        '''Get the |AnalogIn| trigger detector valid trigger level range, in Volts.

        Returns:
            Tuple[float, float, int]: The range of valid trigger levels that can be configured.
            The values returned are the *minimum* and *maximum* trigger levels in Volts, and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger level range.
        '''
        c_trigger_detector_level_min = typespec_ctypes.c_double()
        c_trigger_detector_level_max = typespec_ctypes.c_double()
        c_trigger_detector_level_num_steps = typespec_ctypes.c_double()

        result = self.lib.FDwfAnalogInTriggerLevelInfo(
            self.hdwf,
            c_trigger_detector_level_min,
            c_trigger_detector_level_max,
            c_trigger_detector_level_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_detector_level_num_steps.value.is_integer():
            raise PyDwfError("Bad c_trigger_detector_level_num_steps value")

        trigger_detector_level_min = c_trigger_detector_level_min.value
        trigger_detector_level_max = c_trigger_detector_level_max.value
        trigger_detector_level_num_steps = int(c_trigger_detector_level_num_steps.value)

        return (trigger_detector_level_min, trigger_detector_level_max, trigger_detector_level_num_steps)

    def triggerLevelSet(self, trigger_detector_level: float) -> None:
        '''Set the |AnalogIn| trigger detector trigger-level, in Volts.

        Parameters:
            trigger_detector_level (float): The trigger level to be configured, in Volts.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger level.
        '''
        result = self.lib.FDwfAnalogInTriggerLevelSet(self.hdwf, trigger_detector_level)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerLevelGet(self) -> float:
        '''Get the |AnalogIn| trigger detector trigger-level, in Volts.

        Returns:
            float: The currently configured trigger level, in Volts.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger level.
        '''
        c_trigger_detector_level = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerLevelGet(self.hdwf, c_trigger_detector_level)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_level = c_trigger_detector_level.value
        return trigger_detector_level

    def triggerHysteresisInfo(self) -> Tuple[float, float, int]:
        '''Get the |AnalogIn| trigger detector valid hysteresis range, in Volts.

        Returns:
            Tuple[float, float, int]: The valid range of trigger hysteresis values that can be configured.
            The values returned are the *minimum* and *maximum* trigger hysteresis levels in Volts,
            and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger hysteresis info.
        '''
        c_trigger_detector_hysteresis_min = typespec_ctypes.c_double()
        c_trigger_detector_hysteresis_max = typespec_ctypes.c_double()
        c_trigger_detector_hysteresis_num_steps = typespec_ctypes.c_double()

        result = self.lib.FDwfAnalogInTriggerHysteresisInfo(
            self.hdwf,
            c_trigger_detector_hysteresis_min,
            c_trigger_detector_hysteresis_max,
            c_trigger_detector_hysteresis_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_detector_hysteresis_num_steps.value.is_integer():
            raise PyDwfError("Bad c_trigger_detector_hysteresis_num_steps value.")

        trigger_detector_hysteresis_min = c_trigger_detector_hysteresis_min.value
        trigger_detector_hysteresis_max = c_trigger_detector_hysteresis_max.value
        trigger_detector_hysteresis_num_steps = int(c_trigger_detector_hysteresis_num_steps.value)

        return (trigger_detector_hysteresis_min, trigger_detector_hysteresis_max,
                trigger_detector_hysteresis_num_steps)

    def triggerHysteresisSet(self, trigger_detector_hysteresis: float) -> None:
        '''Set the |AnalogIn| trigger detector hysteresis, in Volts.

        Parameters:
            trigger_detector_hysteresis (float): The trigger hysteresis to be configured, in Volts.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger hysteresis.
        '''
        result = self.lib.FDwfAnalogInTriggerHysteresisSet(self.hdwf, trigger_detector_hysteresis)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerHysteresisGet(self) -> float:
        '''Get the |AnalogIn| trigger detector trigger hysteresis, in Volts.

        Returns:
            float: The currently configured trigger hysteresis, in Volts.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger hysteresis.
        '''
        c_trigger_detector_hysteresis = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerHysteresisGet(self.hdwf, c_trigger_detector_hysteresis)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_hysteresis = c_trigger_detector_hysteresis.value
        return trigger_detector_hysteresis

    def triggerConditionInfo(self) -> List[DwfTriggerSlope]:
        '''Get the valid |AnalogIn| trigger detector condition (slope) options.

        Returns:
            List[DwfTriggerSlope]: A list of valid trigger detector condition (slope) values.

        Raises:
            DwfLibraryError: An error occurred while getting the valid trigger detector condition values.
        '''
        c_trigger_detector_condition_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerConditionInfo(
            self.hdwf,
            c_trigger_detector_condition_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_condition_bitset = c_trigger_detector_condition_bitset.value
        trigger_detector_condition_list = [
            trigger_detector_condition for trigger_detector_condition in DwfTriggerSlope
            if trigger_detector_condition_bitset & (1 << trigger_detector_condition.value)
        ]
        return trigger_detector_condition_list

    def triggerConditionSet(self, trigger_detector_condition: DwfTriggerSlope) -> None:
        '''Set the |AnalogIn| trigger detector condition (slope).

        Parameters:
            trigger_detector_condition (DwfTriggerSlope): The trigger detector condition (slope) to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger condition.
        '''
        result = self.lib.FDwfAnalogInTriggerConditionSet(self.hdwf, trigger_detector_condition.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerConditionGet(self) -> DwfTriggerSlope:
        '''Get the |AnalogIn| trigger detector condition (slope).

        Returns:
            DwfTriggerSlope: The currently configured trigger condition (slope).

        Raises:
            DwfLibraryError: An error occurred while setting the trigger condition (slope).
        '''
        c_trigger_detector_condition = typespec_ctypes.DwfTriggerSlope()
        result = self.lib.FDwfAnalogInTriggerConditionGet(self.hdwf, c_trigger_detector_condition)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_detector_condition = DwfTriggerSlope(c_trigger_detector_condition.value)
        return trigger_detector_condition

    def triggerLengthInfo(self) -> Tuple[float, float, int]:
        '''Get the valid |AnalogIn| trigger detector length range, in seconds.

        Returns:
            Tuple[float, float, int]: The valid range of trigger detector length values that can be configured.
            The values returned are the *minimum* and *maximum* trigger detector length values in seconds,
            and the number of steps.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger length range.
        '''
        c_trigger_detector_length_min = typespec_ctypes.c_double()
        c_trigger_detector_length_max = typespec_ctypes.c_double()
        c_trigger_detector_length_num_steps = typespec_ctypes.c_double()

        result = self.lib.FDwfAnalogInTriggerLengthInfo(
            self.hdwf,
            c_trigger_detector_length_min,
            c_trigger_detector_length_max,
            c_trigger_detector_length_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_trigger_detector_length_num_steps.value.is_integer():
            raise PyDwfError("Bad c_trigger_detector_length_num_steps value.")

        trigger_detector_length_min = c_trigger_detector_length_min.value
        trigger_detector_length_max = c_trigger_detector_length_max.value
        trigger_detector_length_num_steps = int(c_trigger_detector_length_num_steps.value)

        return (trigger_detector_length_min, trigger_detector_length_max, trigger_detector_length_num_steps)

    def triggerLengthSet(self, trigger_detector_length: float) -> None:
        '''Set the |AnalogIn| trigger detector length, in seconds.

        Parameters:
            trigger_detector_length (float): The trigger detector trigger length to be configured, in seconds.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector length.
        '''
        result = self.lib.FDwfAnalogInTriggerLengthSet(self.hdwf, trigger_detector_length)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerLengthGet(self) -> float:
        '''Get the |AnalogIn| trigger detector length, in seconds.

        Returns:
            float: The currently configured trigger detector trigger length, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector length.
        '''
        c_trigger_length = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInTriggerLengthGet(self.hdwf, c_trigger_length)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_length = c_trigger_length.value
        return trigger_length

    def triggerLengthConditionInfo(self) -> List[DwfAnalogInTriggerLengthCondition]:
        '''Get a list of valid |AnalogIn| trigger detector length condition values.

        Trigger length condition values include *Less*, *Timeout*, and *More*.

        Returns:
            List[DwfAnalogInTriggerLengthCondition]: A list of valid trigger detector length condition values.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger length condition info.
        '''
        c_tlc_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInTriggerLengthConditionInfo(self.hdwf, c_tlc_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        tlc_bitset = c_tlc_bitset.value
        tlc_list = [tlc for tlc in DwfAnalogInTriggerLengthCondition if tlc_bitset & (1 << tlc.value)]
        return tlc_list

    def triggerLengthConditionSet(self, trigger_detector_length_condition: DwfAnalogInTriggerLengthCondition) -> None:
        '''Set the |AnalogIn| trigger detector length condition.

        Parameters:
            trigger_detector_length_condition (DwfAnalogInTriggerLengthCondition):
                The trigger detector length condition to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the trigger detector length condition.
        '''
        result = self.lib.FDwfAnalogInTriggerLengthConditionSet(
            self.hdwf,
            trigger_detector_length_condition.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerLengthConditionGet(self) -> DwfAnalogInTriggerLengthCondition:
        '''Get the |AnalogIn| trigger detector length condition.

        Returns:
            DwfAnalogInTriggerLengthCondition: The currently configured trigger detector length condition.

        Raises:
            DwfLibraryError: An error occurred while getting the trigger detector length condition.
        '''
        c_trigger_length_condition = typespec_ctypes.DwfAnalogInTriggerLengthCondition()
        result = self.lib.FDwfAnalogInTriggerLengthConditionGet(
            self.hdwf,
            c_trigger_length_condition)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_length_condition = DwfAnalogInTriggerLengthCondition(c_trigger_length_condition.value)
        return trigger_length_condition

    ###################################################################################################################
    #                                                                                                                 #
    #                                                     COUNTER                                                     #
    #                                                                                                                 #
    ###################################################################################################################

    def counterInfo(self) -> Tuple[int, float]:
        '''Get |AnalogIn| counter info.'''
        c_counter_max = typespec_ctypes.c_double()
        c_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInCounterInfo(self.hdwf, c_counter_max, c_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_counter_max.value.is_integer():
            raise PyDwfError("Bad c_counter_max value.")

        counter_max = int(c_counter_max.value)
        duration_max = c_duration_max.value
        return (counter_max, duration_max)

    def counterSet(self, duration: float) -> None:
        '''Set |AnalogIn| counter duration.'''
        result = self.lib.FDwfAnalogInCounterSet(self.hdwf, duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def counterGet(self) -> float:
        '''Get |AnalogIn| counter duration.'''
        c_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInCounterGet(self.hdwf, c_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        duration = c_duration.value
        return duration

    def counterStatus(self) -> Tuple[float, float, int]:
        '''Get |AnalogIn| counter status.'''
        c_count = typespec_ctypes.c_double()
        c_freq = typespec_ctypes.c_double()
        c_tick = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogInCounterStatus(self.hdwf, c_count, c_freq, c_tick)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        count = c_count.value
        freq = c_freq.value
        tick = c_tick.value
        return (count, freq, tick)

    ###################################################################################################################
    #                                                                                                                 #
    #                                          SAMPLING CLOCK CONFIGURATION                                           #
    #                                                                                                                 #
    ###################################################################################################################

    def samplingSourceSet(self, sampling_source: DwfTriggerSource) -> None:
        '''Set the |AnalogIn| sampling source.

        Parameters:
            sampling_source (DwfTriggerSource): The sampling source to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the sampling source.
        '''
        result = self.lib.FDwfAnalogInSamplingSourceSet(self.hdwf, sampling_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def samplingSourceGet(self) -> DwfTriggerSource:
        '''Get the |AnalogIn| sampling source.

        Returns:
            DwfTriggerSource: The currently configured sampling source.

        Raises:
            DwfLibraryError: An error occurred while getting the sampling source.
        '''
        c_sampling_source = typespec_ctypes.DwfTriggerSource()
        result = self.lib.FDwfAnalogInSamplingSourceGet(self.hdwf, c_sampling_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sampling_source = DwfTriggerSource(c_sampling_source.value)
        return sampling_source

    def samplingSlopeSet(self, sampling_slope: DwfTriggerSlope) -> None:
        '''Set the |AnalogIn| sampling slope.

        Parameters:
            sampling_slope (DwfTriggerSlope): The sampling slope to be configured.

        Raises:
            DwfLibraryError: An error occurred while setting the sampling slope.
        '''
        result = self.lib.FDwfAnalogInSamplingSlopeSet(self.hdwf, sampling_slope.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def samplingSlopeGet(self) -> DwfTriggerSlope:
        '''Get the |AnalogIn| sampling slope.

        Returns:
            DwfTriggerSlope: The currently configured sampling slope.

        Raises:
            DwfLibraryError: An error occurred while getting the sampling slope.
        '''
        c_sampling_slope = typespec_ctypes.DwfTriggerSlope()
        result = self.lib.FDwfAnalogInSamplingSlopeGet(self.hdwf, c_sampling_slope)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sampling_slope = DwfTriggerSlope(c_sampling_slope.value)
        return sampling_slope

    def samplingDelaySet(self, sampling_delay: float) -> None:
        '''Set the |AnalogIn| sampling delay, in seconds.

        Parameters:
            sampling_delay (float): The sampling delay to be configured, in seconds.

        Raises:
            DwfLibraryError: An error occurred while setting the sampling delay.
        '''
        result = self.lib.FDwfAnalogInSamplingDelaySet(self.hdwf, sampling_delay)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def samplingDelayGet(self) -> float:
        '''Get the |AnalogIn| sampling delay, in seconds.

        Returns:
            float: The currently configured sampling delay, in seconds.

        Raises:
            DwfLibraryError: An error occurred while getting the sampling delay.
        '''
        c_sampling_delay = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogInSamplingDelayGet(self.hdwf, c_sampling_delay)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        sampling_delay = c_sampling_delay.value
        return sampling_delay
