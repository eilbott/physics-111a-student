'''The |pydwf.core.api.digital_out| module provides a single class: |DigitalOut|.'''

from typing import Tuple, List
import ctypes

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.enum_types import (DwfTriggerSource, DwfTriggerSlope, DwfDigitalOutOutput, DwfDigitalOutType,
                                             DwfState, DwfDigitalOutIdle)
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.exceptions import PyDwfError

# pylint: disable=too-many-lines

class DigitalOut(AbstractDwfDeviceSubAPI):
    '''The |DigitalOut| class provides access to the digital output (pattern generator) instrument of a
    |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its
        public |digitalOut:link| attribute for access by the user.
    '''

    # pylint: disable=too-many-public-methods

    ###################################################################################################################
    #                                                                                                                 #
    #                                               INSTRUMENT CONTROL                                                #
    #                                                                                                                 #
    ###################################################################################################################

    def reset(self) -> None:
        '''Reset the |DigitalOut| instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalOutReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self, start: bool) -> None:
        '''Start or stop the |DigitalOut| instrument.

        Parameters:
            start (int): Whether to start/stop the instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfDigitalOutConfigure(self.hdwf, start)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self) -> DwfState:
        '''Return the status of the |DigitalOut| instrument.

        This method performs a status request to the |DigitalOut| instrument and receives its response.

        Returns:
            DwfState: The status of the |DigitalOut| instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        c_status = typespec_ctypes.DwfState()
        result = self.lib.FDwfDigitalOutStatus(self.hdwf, c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        status_ = DwfState(c_status.value)
        return status_

    def statusOutput(self) -> Tuple[int, int]:
        '''Get status output.
        
        Notice:
            This function is not documented in the official documentation, but it is present in the C header file.

        Returns:
            Tuple[int, int]: The first entry is labeled 'value', the second is labeled 'enable' in the C header file.

        Raises:
            DwfLibraryError: An error occurred while executing the *statusOutput* operation.
        '''
        c_value = typespec_ctypes.c_unsigned_int()
        c_enable = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutStatus(self.hdwf, c_value, c_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        value = c_value.value
        enable = c_enable.value
        return (value, enable)

    ###################################################################################################################
    #                                                                                                                 #
    #                                                  CHANNEL COUNT                                                  #
    #                                                                                                                 #
    ###################################################################################################################

    def count(self) -> int:
        '''Get the |DigitalOut| instrument channel (digital pin) count.

        Returns:
            int: The number of digital output channels.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_channel_count = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutCount(self.hdwf, c_channel_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_count = c_channel_count.value
        return channel_count

    ###################################################################################################################
    #                                                                                                                 #
    #                                             STATE MACHINE SETTINGS                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def waitInfo(self) -> Tuple[float, float]:
        '''Get the |DigitalOut| instrument range for the |Wait:link| state duration, in seconds.

        Returns:
            Tuple[float, float]: A tuple containing the minimal and maximal configurable |Wait:link| state duration,
            in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_wait_duration_min = typespec_ctypes.c_double()
        c_wait_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutWaitInfo(self.hdwf, c_wait_duration_min, c_wait_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        wait_duration_min = c_wait_duration_min.value
        wait_duration_max = c_wait_duration_max.value
        return (wait_duration_min, wait_duration_max)

    def waitSet(self, wait_duration: float) -> None:
        '''Set the |DigitalOut| instrument |Wait:link| state duration, in seconds.

        Parameters:
            wait_duration (float): Digital-out |Wait:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutWaitSet(self.hdwf, wait_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def waitGet(self) -> float:
        '''Get |DigitalOut| instrument |Wait:link| state duration, in seconds.

        Returns:
            float: The |Wait:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_wait_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutWaitGet(self.hdwf, c_wait_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        wait_duration = c_wait_duration.value
        return wait_duration

    def runInfo(self) -> Tuple[float, float]:
        '''Get the |DigitalOut| instrument range for the |Running:link| state duration, in seconds.

        Returns:
            Tuple[float, float]: A tuple containing the minimal and maximal |Running:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_duration_min = typespec_ctypes.c_double()
        c_run_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutRunInfo(self.hdwf, c_run_duration_min, c_run_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        run_duration_min = c_run_duration_min.value
        run_duration_max = c_run_duration_max.value
        return (run_duration_min, run_duration_max)

    def runSet(self, run_duration: float) -> None:
        '''Set the |DigitalOut| instrument |Running:link| state duration, in seconds.

        Parameters:
            run_duration: The |Running:link| state duration, in seconds.

                The value 0 is special; it means *forever*.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutRunSet(self.hdwf, run_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def runGet(self) -> float:
        '''Get the |DigitalOut| instrument |Running:link| state duration, in seconds.

        Returns:
            float: The |Running:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutRunGet(self.hdwf, c_run_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        run_duration = c_run_duration.value
        return run_duration

    def runStatus(self) -> int:
        '''Get the |DigitalOut| instrument |Running:link| state duration time left,
        in clock cycles.

        This value is internally expressed as an integer with 48-bit resolution,
        and is measured in integer clock cycles. The C API returns it as a
        double-precision floating point number, to avoid using 64-bit integers.

        Use the :py:meth:`internalClockInfo` method to retrieve the clock frequency.

        Returns:
            int: The number of clock cycles until the nest state transition of the
            |DigitalOut| instrument's state machine.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_status = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutRunStatus(self.hdwf, c_run_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        if not c_run_status.value.is_integer():
            raise PyDwfError('Bad c_run_status value.')

        run_status = int(c_run_status.value)
        return run_status

    def repeatTriggerSet(self, repeat_trigger_flag: bool) -> None:
        '''Specify if each |DigitalOut| pulse sequence run should wait for its own trigger.

        Parameters:
            repeat_trigger_flag (bool): If True, not only the first, both also every successive run of the
                pulse output sequence will wait until it receives a trigger.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutRepeatTriggerSet(self.hdwf, repeat_trigger_flag)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def repeatTriggerGet(self) -> bool:
        '''Get if each |DigitalOut| pulse sequence run should wait for its own trigger.

        Returns:
            bool: If True, not only the first, both also every successive run of the pulse output sequence
            will wait until it receives a trigger.
        '''
        c_repeat_trigger = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutRepeatTriggerGet(self.hdwf, c_repeat_trigger)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_trigger_flag = bool(c_repeat_trigger.value)
        return repeat_trigger_flag

    def repeatInfo(self) -> Tuple[int, int]:
        '''Get the |DigitalOut| minimal and maximal repeat count for pulse-sequence runs.

        Returns:
            Tuple[int, int]: A tuple containing the minimal and maximal repeat count for
            digital-out pulse-sequence runs.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat_min = typespec_ctypes.c_unsigned_int()
        c_repeat_max = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutRepeatInfo(self.hdwf, c_repeat_min, c_repeat_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_min = c_repeat_min.value
        repeat_max = c_repeat_max.value
        return (repeat_min, repeat_max)

    def repeatSet(self, repeat: int) -> None:
        '''Set the |DigitalOut| repeat count for pulse-sequence runs.

        Parameters:
            repeat (int): Repeat count.
                The value 0 is special; it means *forever*.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutRepeatSet(self.hdwf, repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def repeatGet(self) -> int:
        '''Set the |DigitalOut| repeat count for pulse-sequence runs.

        Returns:
            int: Repeat count.
            The value 0 is special; it means *forever*.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutRepeatGet(self.hdwf, c_repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat = c_repeat.value
        return repeat

    def repeatStatus(self) -> int:
        '''Get the |DigitalOut| count of repeats remaining for the currently active output sequence.

        This number counts down as a digital output sequence is active.

        Returns:
            int: The repeat count status.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat_status = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutRepeatStatus(self.hdwf, c_repeat_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_status = c_repeat_status.value
        return repeat_status

    ###################################################################################################################
    #                                                                                                                 #
    #                                              TRIGGER CONFIGURATION                                              #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerSourceInfo(self) -> List[DwfTriggerSource]:
        '''Get the valid |DigitalOut| trigger sources.

        Warning:
            **This method is obsolete.**

            Use the generic DeviceControl.triggerInfo() method instead.

        Returns:
            List[DwfTriggerSource]: A list of valid the trigger sources.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_source_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutTriggerSourceInfo(self.hdwf, c_trigger_source_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source_bitset = c_trigger_source_bitset.value
        trigger_source_list = [trigger_source for trigger_source in DwfTriggerSource
                               if trigger_source_bitset & (1 << trigger_source.value)]
        return trigger_source_list

    def triggerSourceSet(self, trigger_source: DwfTriggerSource) -> None:
        '''Set the |DigitalOut| trigger source.

        Parameters:
            trigger_source (DwfTriggerSource): The trigger source to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutTriggerSourceSet(self.hdwf, trigger_source.value)
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
        result = self.lib.FDwfDigitalOutTriggerSourceGet(self.hdwf, c_trigger_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source = DwfTriggerSource(c_trigger_source.value)
        return trigger_source

    def triggerSlopeSet(self, trigger_slope: DwfTriggerSlope) -> None:
        '''Select the |DigitalOut| instrument trigger slope.

        Parameters:
            trigger_slope (DwfTriggerSlope): The trigger slope to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutTriggerSlopeSet(self.hdwf, trigger_slope.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSlopeGet(self) -> DwfTriggerSlope:
        '''Get the currently configured |DigitalOut| instrument trigger slope.

        Returns:
            DwfTriggerSlope: The currently selected |DigitalOut| instrument trigger slope.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_slope = typespec_ctypes.DwfTriggerSlope()
        result = self.lib.FDwfDigitalOutTriggerSlopeGet(self.hdwf, c_trigger_slope)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_slope = DwfTriggerSlope(c_trigger_slope.value)
        return trigger_slope

    ###################################################################################################################
    #                                                                                                                 #
    #                                                 OUTPUT SETTINGS                                                 #
    #                                                                                                                 #
    ###################################################################################################################

    def enableSet(self, channel_index: int, enable_flag: bool) -> None:
        '''Enable or disable a |DigitalOut| channel (pin).

        Parameters:
            channel_index (int): The digital pin to enable or disable.
            enable_flag (bool): Whether to enable or disable the digital output.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutEnableSet(self.hdwf, channel_index, enable_flag)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def enableGet(self, channel_index: int) -> bool:
        '''Check if a specific |DigitalOut| channel (pin) is enabled for output.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            bool: Whether the digital pin is enabled as an output.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_enable_flag = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutEnableGet(self.hdwf, channel_index, c_enable_flag)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        enable_flag = bool(c_enable_flag.value)
        return enable_flag

    def outputInfo(self, channel_index: int) -> List[DwfDigitalOutOutput]:
        '''Get valid |DigitalOut| output choices (e.g. Push/Pull, tristate).

        Returns:
            List[DwfDigitalOutOutput]: A list of valid output settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_output_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutOutputInfo(self.hdwf, channel_index, c_output_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        output_bitset = c_output_bitset.value
        output_list = [output for output in DwfDigitalOutOutput if output_bitset & (1 << output.value)]
        return output_list

    def outputSet(self, channel_index: int, output_value: DwfDigitalOutOutput) -> None:
        '''Set |DigitalOut| output choice (e.g. Push/Pull, tristate).

        Parameters:
            channel_index (int): The digital pin.
            output_value (DwfDigitalOutOutput): The digital output setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutOutputSet(self.hdwf, channel_index, output_value.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def outputGet(self, channel_index: int) -> DwfDigitalOutOutput:
        '''Get currently configured |DigitalOut| output (e.g. Push/Pull, tristate).

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            DwfDigitalOutOutput: The digital output setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_output_value = typespec_ctypes.DwfDigitalOutOutput()
        result = self.lib.FDwfDigitalOutOutputGet(self.hdwf, channel_index, c_output_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        output_value = DwfDigitalOutOutput(c_output_value.value)
        return output_value

    def typeInfo(self, channel_index: int) -> List[DwfDigitalOutType]:
        '''Get a list of valid |DigitalOut| output channel types.

        Returns:
            List[DwfDigitalOutType]: A list of valid digital output channel types.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_type_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutTypeInfo(self.hdwf, channel_index, c_type_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        type_bitset = c_type_bitset.value
        type_list = [type_ for type_ in DwfDigitalOutType if type_bitset & (1 << type_.value)]
        return type_list

    def typeSet(self, channel_index: int, output_type: DwfDigitalOutType) -> None:
        '''Select the |DigitalOut| output channel type.

        Parameters:
            channel_index (int): The digital pin.
            output_type (DwfDigitalOutType): The digital output channel type.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutTypeSet(self.hdwf, channel_index, output_type.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def typeGet(self, channel_index: int) -> DwfDigitalOutType:
        '''Get the currently selected |DigitalOut| output channel type.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            DwfDigitalOutType: The digital output channel type.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_output_type = typespec_ctypes.DwfDigitalOutType()
        result = self.lib.FDwfDigitalOutTypeGet(self.hdwf, channel_index, c_output_type)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        output_type = DwfDigitalOutType(c_output_type.value)
        return output_type

    def idleInfo(self, channel_index: int) -> List[DwfDigitalOutIdle]:
        '''Get valid |DigitalOut| idle output values.

        Returns:
            List[DwfDigitalOutIdle]: A list of valid idle output values.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_idle_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalOutIdleInfo(self.hdwf, channel_index, c_idle_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        idle_bitset = c_idle_bitset.value
        idle_list = [idle for idle in DwfDigitalOutIdle if idle_bitset & (1 << idle.value)]
        return idle_list

    def idleSet(self, channel_index: int, idle_mode: DwfDigitalOutIdle) -> None:
        '''Set the |DigitalOut| idle output value.

        Parameters:
            channel_index (int): The digital pin.
            idle_mode (DwfDigitalOutIdle): The idle output value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutIdleSet(self.hdwf, channel_index, idle_mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def idleGet(self, channel_index: int) -> DwfDigitalOutIdle:
        '''Get the currently configured idle output value.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            DwfDigitalOutIdle: The currently configured idle output value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_idle_mode = typespec_ctypes.DwfDigitalOutIdle()
        result = self.lib.FDwfDigitalOutIdleGet(self.hdwf, channel_index, c_idle_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        idle_mode = DwfDigitalOutIdle(c_idle_mode.value)
        return idle_mode

    ###################################################################################################################
    #                                                                                                                 #
    #                                        OUTPUT PATTERN TIMING DEFINITION                                         #
    #                                                                                                                 #
    ###################################################################################################################

    def internalClockInfo(self) -> float:
        '''Get the |DigitalOut| instrument clock frequency.

        Returns:
            float: The digital-out clock frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_internal_clock_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfDigitalOutInternalClockInfo(self.hdwf, c_internal_clock_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        internal_clock_frequency = c_internal_clock_frequency.value
        return internal_clock_frequency

    def dividerInfo(self, channel_index: int) -> Tuple[int, int]:
        '''Get the |DigitalOut| divider value range.

        Returns:
            Tuple[int, int]: The range of valid divider settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_divider_min = typespec_ctypes.c_unsigned_int()
        c_divider_max = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutDividerInfo(
            self.hdwf,
            channel_index,
            c_divider_min,
            c_divider_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        divider_min = c_divider_min.value
        divider_max = c_divider_max.value
        return (divider_min, divider_max)

    def dividerSet(self, channel_index: int, divider: int) -> None:
        '''Set the |DigitalOut| divider value.

        Parameters:
            channel_index (int): The digital pin.
            divider (int): The divider setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutDividerSet(self.hdwf, channel_index, divider)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def dividerGet(self, channel_index: int) -> int:
        '''Get the currently configured |DigitalOut| divider value.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            int: The divider setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_divider = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutDividerGet(self.hdwf, channel_index, c_divider)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        divider = c_divider.value
        return divider

    def dividerInitSet(self, channel_index: int, divider_init_value: int) -> None:
        '''Set the |DigitalOut| initial divider value.

        Parameters:
            channel_index (int): The digital pin.
            divider_init_value (int): The initial divider counter value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutDividerInitSet(self.hdwf, channel_index, divider_init_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def dividerInitGet(self, channel_index: int) -> int:
        '''Get the currently configured |DigitalOut| initial divider value.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            int: The divider init setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_divider_init = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutDividerInitGet(self.hdwf, channel_index, c_divider_init)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        divider_init = c_divider_init.value
        return divider_init

    def counterInfo(self, channel_index: int) -> Tuple[int, int]:
        '''Get the |DigitalOut| counter value range.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            Tuple[int, int]: The range of valid counter settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_counter_min = typespec_ctypes.c_unsigned_int()
        c_counter_max = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutCounterInfo(
            self.hdwf,
            channel_index,
            c_counter_min,
            c_counter_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        counter_min = c_counter_min.value
        counter_max = c_counter_max.value
        return (counter_min, counter_max)

    def counterSet(self, channel_index: int, low_count: int, high_count: int) -> None:
        '''Set the |DigitalOut| counter durations for both the low and high signal output levels.

        Parameters:
            channel_index (int): The digital pin.
            low_count (int): The number of cycles the signal should be Low.
            high_count (int): The number of cycles the signal should be High.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutCounterSet(self.hdwf, channel_index, low_count, high_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def counterGet(self, channel_index: int) -> Tuple[int, int]:
        '''Get the |DigitalOut| counter durations for both the low and high signal output levels.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            Tuple[int, int]: The number of cycles the signal should be Low, High.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_low_count = typespec_ctypes.c_unsigned_int()
        c_high_count = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutCounterGet(self.hdwf, channel_index, c_low_count, c_high_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        low_count = c_low_count.value
        high_count = c_high_count.value
        return (low_count, high_count)

    def counterInitSet(self, channel_index: int, high_flag: bool, counter_init_value: int) -> None:
        '''Set the |DigitalOut| initial signal value and initial counter value.

        Parameters:
            channel_index (int): The digital pin.
            high (bool): Whether to start High (True) or Low (False).
            counter_init (int): The initial counter setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutCounterInitSet(
            self.hdwf,
            channel_index,
            high_flag, counter_init_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def counterInitGet(self, channel_index: int) -> Tuple[bool, int]:
        '''Get the |DigitalOut| initial signal value and initial counter value.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            Tuple [bool, int]: Whether to start High (True) or Low (False),
            and the initial counter setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_high = typespec_ctypes.c_int()
        c_counter_init = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutCounterInitGet(
            self.hdwf,
            channel_index, c_high,
            c_counter_init)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        high = bool(c_high.value)
        counter_init = c_counter_init.value
        return (high, counter_init)

    def repetitionInfo(self, channel_index: int) -> int:
        '''Get maximum repetition count value.

        The repetition count specifies how many times the counter should be reloaded.
        For pulse signals set twice the desired value since each pulse is generated by two counter loads, low and high.
        It is available with ADP3X50 and newer devices.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            int: The maximum repetition value that can be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repetition_count_max = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutRepetitionInfo(self.hdwf, channel_index, c_repetition_count_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repetition_count_max = c_repetition_count_max.value
        return repetition_count_max

    def repetitionSet(self, channel_index: int, repeat: int) -> None:
        '''Set repetition count value.

        The repetition count specifies how many times the counter should be reloaded.
        For pulse signals set twice the desired value since each pulse is generated by two counter loads, low and high.
        It is available with ADP3X50 and newer devices.

        Parameters:
            channel_index (int): The digital pin.

        Returns:
            int: The maximum repetition value that can be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutRepetitionSet(self.hdwf, channel_index, repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def repetitionGet(self, channel_index: int) -> int:
        '''Get repetition count value.

        The repetition count specifies how many times the counter should be reloaded.
        For pulse signals set twice the desired value since each pulse is generated by two counter loads, low and high.
        It is available with ADP3X50 and newer devices.
        '''

        c_repeat = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutRepetitionGet(self.hdwf, channel_index, c_repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat = c_repeat.value
        return repeat

    ###################################################################################################################
    #                                                                                                                 #
    #                                                  DATA PLAYBACK                                                  #
    #                                                                                                                 #
    ###################################################################################################################

    def dataInfo(self, channel_index: int) -> int:
        '''Return the maximum buffer size for the specified |DigitalOut| channel,
        i.e., the number of custom data bits.

        Parameters:
            channel_index (int): the channel for which to obtain the data bits count.

        Returns:
            int: The number of custom data bits that can be specified for the channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_max_data_bits = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfDigitalOutDataInfo(self.hdwf, channel_index, c_max_data_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        max_data_bits = c_max_data_bits.value
        return max_data_bits

    def dataSet(self, channel_index: int, bits: str, tristate: bool = False) -> None:
        '''Set the |DigitalOut| arbitrary output data.

        This function also sets the counter initial, low and high value, according to the
        number of bits. The data-bits are sent out in LSB-first order.

        Parameters:
            channel_index (int): the channel for which to set the output data.
            bits (str): The bits, as a string.
            tristate (bool): Whether to interpret the string as a tristate signal.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        if tristate:
            bits = bits.replace('1', '11').replace('0', '01').replace('Z', '00')

        # For tristate output, the count of bits is the total number of output values (I/O)
        # and output enable (OE) bits, which should be an even number.

        count_of_bits = len(bits)

        octets = []
        while len(bits) > 0:
            octet_str = bits[:8]
            octet = int(octet_str[::-1], 2)
            octets.append(octet)
            bits = bits[8:]

        octets_as_bytes = bytes(octets)

        result = self.lib.FDwfDigitalOutDataSet(self.hdwf, channel_index, octets_as_bytes, count_of_bits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def playDataSet(self, bits: str, bits_per_sample: int, count_of_samples: int) -> None:
        '''Set the |DigitalOut| playback data.

        The output can be PushPull, OpenDrain, or OpenSource. Tristate data is not supported.

        Note:
            The DWF documentation explicitly states that this function is supported by the Digital Discovery.
            (So, by implication, it's probably not supported on anything else.)

        Parameters:
            bits (str): string of '0' and '1' characters. Its length should be (bits_per_sample * count_of_samples).
            bits_per_sample (int): Bits per sample, should be 1, 2, 4, 8, or 16.
            count_of_samples (int): Number of samples.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        if len(bits) != (bits_per_sample * count_of_samples):
            raise PyDwfError('Unexpected number of bits.')

        octets = []
        while len(bits) > 0:
            octet_str = bits[:8]
            octet = int(octet_str[::-1], 2)
            octets.append(octet)
            bits = bits[8:]

        # We convert the 'octets' input data to an array of ctypes unsigned chars.
        # This silences a spurious mypy warning.

        octet_ctypes_type = (ctypes.c_ubyte * len(octets))  # pylint: disable=superfluous-parens
        octet_ctypes = octet_ctypes_type(*octets)

        result = self.lib.FDwfDigitalOutPlayDataSet(
            self.hdwf,
            octet_ctypes,
            bits_per_sample,
            count_of_samples)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def playUpdateSet(self, bits: str, index_of_sample: int, count_of_samples: int) -> None:
        '''Set the |DigitalOut| playback data.

        ToDo:
           This function is not sufficiently documented at this time.
        '''

        if len(bits) % 8 != 0:
            raise PyDwfError('Unexpected number of bits.')

        octets = []
        while len(bits) > 0:
            octet_str = bits[:8]
            octet = int(octet_str[::-1], 2)
            octets.append(octet)
            bits = bits[8:]

        # We convert the 'octets' input data to an array of ctypes unsigned chars.
        # This silences a spurious mypy warning.

        octet_ctypes_type = (ctypes.c_ubyte * len(octets))  # pylint: disable=superfluous-parens
        octet_ctypes = octet_ctypes_type(*octets)

        result = self.lib.FDwfDigitalOutPlayUpdateSet(
            self.hdwf,
            octet_ctypes,
            index_of_sample,
            count_of_samples)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def playRateSet(self, playback_rate: float) -> None:
        '''Set the |DigitalOut| playback rate, in Hz.

        Note:
            The DWF documentation explicitly states that this function is supported by the Digital Discovery.
            (So, by implication, it's probably not supported on anything else.)

        Parameters:
            playback_rate (float): The playback rate, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalOutPlayRateSet(self.hdwf, playback_rate)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
