'''The |pydwf.core.api.analog_out| module provides a single class: |AnalogOut|.'''

# pylint: disable=too-many-lines

from typing import Tuple, List

import numpy as np

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.enum_types import (DwfTriggerSource, DwfAnalogOutFunction, DwfState,
                                             DwfAnalogOutNode, DwfTriggerSlope, DwfAnalogOutMode, DwfAnalogOutIdle)
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class AnalogOut(AbstractDwfDeviceSubAPI):
    '''The |AnalogOut| class provides access to the analog output (signal generator) instrument of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its
        public |analogOut:link| attribute for access by the user.
    '''

    # pylint: disable=too-many-public-methods

    ###################################################################################################################
    #                                                                                                                 #
    #                                               INSTRUMENT CONTROL                                                #
    #                                                                                                                 #
    ###################################################################################################################

    def reset(self, channel_index: int) -> None:
        '''Reset the |AnalogOut| instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfAnalogOutReset(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self, channel_index: int, start: int) -> None:
        '''Configure the |AnalogOut| instrument.

        Parameters:
            channel_index (int): The output channel to configure. Specify -1 to configure all channels.
            start (int): Whether to start/stop the instrument:

               * 0 — Stop instrument
               * 1 — Start instrument
               * 3 — Apply settings; do not change instrument state

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfAnalogOutConfigure(self.hdwf, channel_index, start)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self, channel_index: int) -> DwfState:
        '''Get the |AnalogOut| instrument channel state.

        This method performs a status request to the |AnalogOut| instrument and receives its response.

        Parameters:
            channel_index (int): The output channel for which to get the status.

        Returns:
            DwfState: The status of the |AnalogOut| instrument channel.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        c_status = typespec_ctypes.DwfState()
        result = self.lib.FDwfAnalogOutStatus(self.hdwf, channel_index, c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        status = DwfState(c_status.value)
        return status

    ###################################################################################################################
    #                                                                                                                 #
    #                                                  CHANNEL COUNT                                                  #
    #                                                                                                                 #
    ###################################################################################################################

    def count(self) -> int:
        '''Count the number of analog output channels.

        Returns:
            int: The number of analog output channels.

        Raises:
            DwfLibraryError: An error occurred while retrieving the number of analog output channels.
        '''
        c_channel_count = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutCount(self.hdwf, c_channel_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_count = c_channel_count.value
        return channel_count

    ###################################################################################################################
    #                                                                                                                 #
    #                                PER-CHANNEL STATE MACHINE SETTINGS CONFIGURATION                                 #
    #                                                                                                                 #
    ###################################################################################################################

    def waitInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel valid |Wait:link| state duration range, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of configurable |Wait:link| state durations, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_wait_duration_min = typespec_ctypes.c_double()
        c_wait_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutWaitInfo(self.hdwf, channel_index, c_wait_duration_min, c_wait_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        wait_duration_min = c_wait_duration_min.value
        wait_duration_max = c_wait_duration_max.value
        return (wait_duration_min, wait_duration_max)

    def waitSet(self, channel_index: int, wait_duration: float) -> None:
        '''Set the |AnalogOut| channel |Wait:link| state duration, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            wait_duration (float): The |Wait:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutWaitSet(self.hdwf, channel_index, wait_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def waitGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel |Wait:link| state duration, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured |Wait:link| state duration.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_wait_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutWaitGet(self.hdwf, channel_index, c_wait_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        wait_duration = c_wait_duration.value
        return wait_duration

    def runInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel valid |Running:link| state duration range, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of allowed |Running:link| state durations, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_duration_min = typespec_ctypes.c_double()
        c_run_duration_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutRunInfo(self.hdwf, channel_index, c_run_duration_min, c_run_duration_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        run_duration_min = c_run_duration_min.value
        run_duration_max = c_run_duration_max.value
        return (run_duration_min, run_duration_max)

    def runSet(self, channel_index: int, run_duration: float) -> None:
        '''Set the |AnalogOut| channel |Running:link| state duration, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            run_duration (float): The |Running:link| state duration, in seconds.
                Specify 0 for a run of indefinite length.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutRunSet(self.hdwf, channel_index, run_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def runGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel |Running:link| state duration, in seconds.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured |Running:link| state duration, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_duration = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutRunGet(self.hdwf, channel_index, c_run_duration)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        run_duration = c_run_duration.value
        return run_duration

    def runStatus(self, channel_index: int) -> float:
        '''Get |Running:link| state duration time left.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The current time remaining in the |Running:link| state, in seconds.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_run_duration_status = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutRunStatus(self.hdwf, channel_index, c_run_duration_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        run_duration_status = c_run_duration_status.value
        return run_duration_status

    def repeatTriggerSet(self, channel_index: int, repeat_trigger_flag: bool) -> None:
        '''Set the |AnalogOut| channel *repeat trigger* setting.

        This setting determines if a new trigger must precede all Wait/Running sequences, or only the first one.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            repeat_trigger_flag (bool): True if each Wait/Running sequence needs its own trigger,
                False if only the first Wait/Running sequence needs a trigger.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutRepeatTriggerSet(self.hdwf, channel_index, repeat_trigger_flag)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def repeatTriggerGet(self, channel_index: int) -> bool:
        '''Get the |AnalogOut| channel *repeat trigger* setting.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            bool: The currently configured *repeat trigger* setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat_trigger_flag = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutRepeatTriggerGet(
            self.hdwf,
            channel_index,
            c_repeat_trigger_flag)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_trigger_flag = bool(c_repeat_trigger_flag.value)
        return repeat_trigger_flag

    def repeatInfo(self, channel_index: int) -> Tuple[int, int]:
        '''Get |AnalogOut| repeat count range.

        The *repeat count* is the number of times the |AnalogOut| channel will go through the
        Wait/Running or Armed/Wait/Running state cycles during the output sequence.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[int, int]: The range of configurable repeat values.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat_min = typespec_ctypes.c_int()
        c_repeat_max = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutRepeatInfo(self.hdwf, channel_index, c_repeat_min, c_repeat_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_min = c_repeat_min.value
        repeat_max = c_repeat_max.value
        return (repeat_min, repeat_max)

    def repeatSet(self, channel_index: int, repeat: int) -> None:
        '''Set the |AnalogOut| repeat count.

        The *repeat count* is the number of times the |AnalogOut| channel will go through the
        Wait/Running or Armed/Wait/Running state cycles during the output sequence.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            repeat (int): The repeat count. If 0, repeat indefinitely.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutRepeatSet(self.hdwf, channel_index, repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def repeatGet(self, channel_index: int) -> int:
        '''Get the |AnalogOut| repeat count.

        The *repeat count* is the number of times the |AnalogOut| channel will go through the
        Wait/Running or Armed/Wait/Running state cycles during the output sequence.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            int: The currently configured repeat value. 0 means: repeat indefinitely.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutRepeatGet(self.hdwf, channel_index, c_repeat)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat = c_repeat.value
        return repeat

    def repeatStatus(self, channel_index: int) -> int:
        '''Get the |AnalogOut| current repeat count, which decreases to 0 while going through
        Running/Wait state cycles.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            int: The current repeat count value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_repeat_status = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutRepeatStatus(self.hdwf, channel_index, c_repeat_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        repeat_status = c_repeat_status.value
        return repeat_status

    def masterSet(self, channel_index: int, master_channel_index: int) -> None:
        '''Set the |AnalogOut| channel master.

        Sets the state machine master channel of the analog output channel.

        Parameters:
            channel_index (int): The output channel for which to set the master setting.
                Specify -1 to set all channels.
            master_channel_index (int): The master channel.

        Raises:
            DwfLibraryError: An error occurred while setting the value.
        '''
        result = self.lib.FDwfAnalogOutMasterSet(self.hdwf, channel_index, master_channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def masterGet(self, channel_index: int) -> int:
        '''Get the |AnalogOut| channel master.

        Parameters:
            channel_index (int): The analog output channel for which to get the master channel.

        Returns:
            int: The index of the master channel which the channel is configured to follow.

        Raises:
            DwfLibraryError: An error occurred while getting the value.
        '''
        c_master_channel_index = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutMasterGet(self.hdwf, channel_index, c_master_channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        master_channel_index = c_master_channel_index.value
        return master_channel_index

    ###################################################################################################################
    #                                                                                                                 #
    #                                        PER-CHANNEL TRIGGER CONFIGURATION                                        #
    #                                                                                                                 #
    ###################################################################################################################

    def triggerSourceInfo(self) -> List[DwfTriggerSource]:
        '''Get a list of valid |AnalogOut| instrument trigger sources.

        Warning:
            **This method is obsolete.**

            Use the generic |DwfDevice.triggerInfo:link| method instead.

        Returns:
            List[DwfTriggerSource]: The list of DwfTriggerSource values that can be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_source_info_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutTriggerSourceInfo(self.hdwf, c_trigger_source_info_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source_info_bitset = c_trigger_source_info_bitset.value
        trigger_source_info_list = [trigger_source for trigger_source in DwfTriggerSource
                                    if trigger_source_info_bitset & (1 << trigger_source.value)]
        return trigger_source_info_list

    def triggerSourceSet(self, channel_index: int, trigger_source: DwfTriggerSource) -> None:
        '''Set the |AnalogOut| channel trigger source.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            trigger_source (DwfTriggerSource): The trigger source to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutTriggerSourceSet(
            self.hdwf,
            channel_index,
            trigger_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSourceGet(self, channel_index: int) -> DwfTriggerSource:
        '''Get the currently selected channel trigger source.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            DwfTriggerSource: The currently selected channel trigger source.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_source = typespec_ctypes.DwfTriggerSource()
        result = self.lib.FDwfAnalogOutTriggerSourceGet(self.hdwf, channel_index, c_trigger_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source = DwfTriggerSource(c_trigger_source.value)
        return trigger_source

    def triggerSlopeSet(self, channel_index: int, trigger_slope: DwfTriggerSlope) -> None:
        '''Select the |AnalogOut| channel trigger slope.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            trigger_slope (DwfTriggerSlope): The trigger slope to be selected.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutTriggerSlopeSet(
            self.hdwf,
            channel_index,
            trigger_slope.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSlopeGet(self, channel_index: int) -> DwfTriggerSlope:
        '''Get the currently selected |AnalogOut| channel trigger slope.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            DwfTriggerSlope: The currently selected |AnalogOut| channel trigger slope.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_trigger_slope = typespec_ctypes.DwfTriggerSlope()
        result = self.lib.FDwfAnalogOutTriggerSlopeGet(self.hdwf, channel_index, c_trigger_slope)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_slope = DwfTriggerSlope(c_trigger_slope.value)
        return trigger_slope

    ###################################################################################################################
    #                                                                                                                 #
    #                                           PER-CHANNEL OUTPUT SETTINGS                                           #
    #                                                                                                                 #
    ###################################################################################################################

    def modeSet(self, channel_index: int, mode: DwfAnalogOutMode) -> None:
        '''Set the |AnalogOut| channel mode (voltage or current).

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            mode (DwfAnalogOutMode): The analog output mode to configure.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutModeSet(self.hdwf, channel_index, mode.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def modeGet(self, channel_index: int) -> DwfAnalogOutMode:
        '''Get the |AnalogOut| channel mode (voltage or current).

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            DwfAnalogOutMode: The currently configured analog output mode.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_mode = typespec_ctypes.DwfAnalogOutMode()
        result = self.lib.FDwfAnalogOutModeGet(self.hdwf, channel_index, c_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        mode = DwfAnalogOutMode(c_mode.value)
        return mode

    def idleInfo(self, channel_index: int) -> List[DwfAnalogOutIdle]:
        '''Get the valid |AnalogOut| channel idle settings.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            List[DwfAnalogOutIdle]: A list of options for the channel behavior when idle.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_idle_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutIdleInfo(self.hdwf, channel_index, c_idle_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        idle_bitset = c_idle_bitset.value
        idle_list = [idle for idle in DwfAnalogOutIdle if idle_bitset & (1 << idle.value)]
        return idle_list

    def idleSet(self, channel_index: int, idle: DwfAnalogOutIdle) -> None:
        '''Set the |AnalogOut| channel idle behavior.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            idle (DwfAnalogOutIdle): The idle behavior setting to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutIdleSet(self.hdwf, channel_index, idle.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def idleGet(self, channel_index: int) -> DwfAnalogOutIdle:
        '''Get the |AnalogOut| channel idle behavior.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            DwfAnalogOutIdle: The |AnalogOut| channel idle behavior setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_idle = typespec_ctypes.DwfAnalogOutIdle()
        result = self.lib.FDwfAnalogOutIdleGet(self.hdwf, channel_index, c_idle)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        idle = DwfAnalogOutIdle(c_idle.value)
        return idle

    def limitationInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel limitation range, in Volts or Amps.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of limitation values that can be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_limitation_min = typespec_ctypes.c_double()
        c_limitation_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutLimitationInfo(self.hdwf, channel_index, c_limitation_min, c_limitation_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        limitation_min = c_limitation_min.value
        limitation_max = c_limitation_max.value
        return (limitation_min, limitation_max)

    def limitationSet(self, channel_index: int, limitation: float) -> None:
        '''Set the |AnalogOut| channel limitation value, in Volts or Amps.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            limitation (float): The limitation value, in Volts or Amps.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutLimitationSet(self.hdwf, channel_index, limitation)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def limitationGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel limitation value, in Volts or Amps.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured limitation value, in Volts or Amps.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_limitation = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutLimitationGet(self.hdwf, channel_index, c_limitation)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        limitation = c_limitation.value
        return limitation

    ###################################################################################################################
    #                                                                                                                 #
    #                                       PER-CHANNEL MISCELLANEOUS SETTINGS                                        #
    #                                                                                                                 #
    ###################################################################################################################

    def customAMFMEnableSet(self, channel_index: int, enable: bool) -> None:

        # pylint: disable=line-too-long

        '''Set the |AnalogOut| channel custom AM/FM enable status.

        Todo:
            Understand and document what this setting does.

        Note:
            This setting is only applicable to |Electronics Explorer:link| devices, as stated in a `message on the Digilent forum
            <https://forum.digilentinc.com/topic/22281-installation-of-waveforms-on-linux-amd64-runs-into-dependency-problem/#comment-64663>`_.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            enable (bool): The custom AM/FM enable setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutCustomAMFMEnableSet(self.hdwf, channel_index, int(enable))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def customAMFMEnableGet(self, channel_index: int) -> bool:

        # pylint: disable=line-too-long

        '''Get the |AnalogOut| channel custom AM/FM enable status.

        Todo:
            Understand and document what this setting does.

        Note:
            This setting is only applicable to |Electronics Explorer:link| devices, as stated in a `message on the Digilent forum
            <https://forum.digilentinc.com/topic/22281-installation-of-waveforms-on-linux-amd64-runs-into-dependency-problem/#comment-64663>`_.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            bool: The custom AM/FM enable state.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_enable = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutCustomAMFMEnableGet(self.hdwf, channel_index, c_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        enable = bool(c_enable.value)
        return enable

    ###################################################################################################################
    #                                                                                                                 #
    #                                                NODE ENUMERATION                                                 #
    #                                                                                                                 #
    ###################################################################################################################

    def nodeInfo(self, channel_index: int) -> List[DwfAnalogOutNode]:
        '''Get a list of valid |AnalogOut| channel nodes.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            List[DwfAnalogOutNode]: The valid nodes for this channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_node_info_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutNodeInfo(self.hdwf, channel_index, c_node_info_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        node_info_bitset = c_node_info_bitset.value
        node_info_list = [node for node in DwfAnalogOutNode if node_info_bitset & (1 << node.value)]
        return node_info_list

    ###################################################################################################################
    #                                                                                                                 #
    #                                               NODE CONFIGURATION                                                #
    #                                                                                                                 #
    ###################################################################################################################

    def nodeEnableSet(self, channel_index: int, node: DwfAnalogOutNode, mode: int) -> None:
        '''Enabled or disable an |AnalogOut| channel node.

        The carrier node enables or disables the channel or selects the modulation. With channel_index -1,
        each analog-out channel enable mode will be configured to the same, new option.

        Parameters:
            channel_index (int): The |AnalogOut| channel. Specify -1 to configure all |AnalogOut| channels.
            node (DwfAnalogOutNode): The channel node.
            mode (int): The enable mode.

        Note:
            The precise meaning of the mode parameter is not clear from the documentation.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeEnableSet(self.hdwf, channel_index, node.value, mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeEnableGet(self, channel_index: int, node: DwfAnalogOutNode) -> int:
        '''Get the enabled state of an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            int: The currently configured enable mode setting.

        Note:
            The precise meaning of the mode parameter is not clear from the documentation.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_mode = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutNodeEnableGet(self.hdwf, channel_index, node.value, c_mode)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        mode = c_mode.value
        return mode

    def nodeFunctionInfo(self, channel_index: int, node: DwfAnalogOutNode) -> List[DwfAnalogOutFunction]:
        '''Get the valid waveform shape function options of an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            List[DwfAnalogOutFunction]: The available node waveform shape functions.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_func_bitset = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfAnalogOutNodeFunctionInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_func_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        func_bitset = c_func_bitset.value
        func_list = [func for func in DwfAnalogOutFunction if func_bitset & (1 << func.value)]
        return func_list

    def nodeFunctionSet(self, channel_index: int, node: DwfAnalogOutNode, func: DwfAnalogOutFunction) -> None:
        '''Set the waveform shape function for an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            func (DwfAnalogOutFunction): The waveform shape function to be configured.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeFunctionSet(self.hdwf, channel_index, node.value, func.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeFunctionGet(self, channel_index: int, node: DwfAnalogOutNode) -> DwfAnalogOutFunction:
        '''Get the waveform shape function for an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            DwfAnalogOutNode: The currently configured waveform shape function.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_func = typespec_ctypes.DwfAnalogOutFunction()
        result = self.lib.FDwfAnalogOutNodeFunctionGet(self.hdwf, channel_index, node.value, c_func)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        func = DwfAnalogOutFunction(c_func.value)
        return func

    def nodeFrequencyInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get the channel node valid frequency range for an |AnalogOut| channel node, in Hz.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[float, float]: The range of valid frequencies, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency_min = typespec_ctypes.c_double()
        c_frequency_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeFrequencyInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_frequency_min,
            c_frequency_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency_min = c_frequency_min.value
        frequency_max = c_frequency_max.value
        return (frequency_min, frequency_max)

    def nodeFrequencySet(self, channel_index: int, node: DwfAnalogOutNode, frequency: float) -> None:
        '''Set the channel node frequency for an |AnalogOut| channel node, in Hz.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            frequency (float): The frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeFrequencySet(
            self.hdwf,
            channel_index,
            node.value,
            frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeFrequencyGet(self, channel_index: int, node: DwfAnalogOutNode) -> float:
        '''Get the frequency for an |AnalogOut| channel node, in Hz.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            float: The currently configured frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeFrequencyGet(
            self.hdwf,
            channel_index,
            node.value,
            c_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency = c_frequency.value
        return frequency

    def nodeAmplitudeInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get the amplitude range for an |AnalogOut| channel node, in Volts.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            tuple[float, float]: The range of allowed amplitude values, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_amplitude_min = typespec_ctypes.c_double()
        c_amplitude_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeAmplitudeInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_amplitude_min,
            c_amplitude_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        amplitude_min = c_amplitude_min.value
        amplitude_max = c_amplitude_max.value
        return (amplitude_min, amplitude_max)

    def nodeAmplitudeSet(self, channel_index: int, node: DwfAnalogOutNode, amplitude: float) -> None:
        '''Set the amplitude for an |AnalogOut| channel node, in Volts.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            amplitude (float): The amplitude to be configured, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeAmplitudeSet(
            self.hdwf,
            channel_index,
            node.value,
            amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeAmplitudeGet(self, channel_index: int, node: DwfAnalogOutNode) -> float:
        '''Get the amplitude for an |AnalogOut| channel node, in Volts.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            float: The currently configured amplitude, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_amplitude = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeAmplitudeGet(
            self.hdwf,
            channel_index,
            node.value,
            c_amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        amplitude = c_amplitude.value
        return amplitude

    def nodeOffsetInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get the valid offset range for an |AnalogOut| channel node, in Volts.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[float, float]: The range of valid node offsets, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_offset_min = typespec_ctypes.c_double()
        c_offset_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeOffsetInfo(self.hdwf, channel_index, node.value, c_offset_min, c_offset_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset_min = c_offset_min.value
        offset_max = c_offset_max.value
        return (offset_min, offset_max)

    def nodeOffsetSet(self, channel_index: int, node: DwfAnalogOutNode, offset: float) -> None:
        '''Set the offset for an |AnalogOut| channel node, in Volts.

        Note:
            Configuring the offset of the *Carrier* node takes a noticeable amount of time (100s of milliseconds).

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            offset (float): The channel offset to be configured, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeOffsetSet(self.hdwf, channel_index, node.value, offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeOffsetGet(self, channel_index: int, node: DwfAnalogOutNode) -> float:
        '''Get the offset for an |AnalogOut| channel node, in Volts.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            float: The currently configured node offset.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_offset = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeOffsetGet(self.hdwf, channel_index, node.value, c_offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset = c_offset.value
        return offset

    def nodeSymmetryInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get the *symmetry* range for an |AnalogOut| channel node.

        The *symmetry* value alters the waveform shape function of the node.

        The *symmetry* value ranges from 0 to 100 for most waveform shape functions, except for the
        :py:attr:`~pydwf.core.auxiliary.enum_types.DwfAnalogOutFunction.SinePower` waveform shape function,
        where it ranges from -100 to +100.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[float, float]: The range of valid symmetry settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_symmetry_min = typespec_ctypes.c_double()
        c_symmetry_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeSymmetryInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_symmetry_min,
            c_symmetry_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        symmetry_min = c_symmetry_min.value
        symmetry_max = c_symmetry_max.value
        return (symmetry_min, symmetry_max)

    def nodeSymmetrySet(self, channel_index: int, node: DwfAnalogOutNode, symmetry: float) -> None:
        '''Set the *symmetry* value for an |AnalogOut| channel node.

        The *symmetry* value alters the waveform shape function of the node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            symmetry (float): The symmetry setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodeSymmetrySet(
            self.hdwf,
            channel_index,
            node.value,
            symmetry)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodeSymmetryGet(self, channel_index: int, node: DwfAnalogOutNode) -> float:
        '''Get the *symmetry* value for an |AnalogOut| channel node.

        The *symmetry* value alters the waveform shape function of the node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            float: The currently configured channel node symmetry value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_symmetry = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodeSymmetryGet(
            self.hdwf,
            channel_index,
            node.value,
            c_symmetry)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        symmetry = c_symmetry.value
        return symmetry

    def nodePhaseInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get the valid phase range for an |AnalogOut| channel node, in degrees.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[float, float]: The range of valid channel node phase values, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_phase_min = typespec_ctypes.c_double()
        c_phase_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodePhaseInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_phase_min,
            c_phase_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        phase_min = c_phase_min.value
        phase_max = c_phase_max.value
        return (phase_min, phase_max)

    def nodePhaseSet(self, channel_index: int, node: DwfAnalogOutNode, phase: float) -> None:
        '''Set the phase for an |AnalogOut| channel node, in degrees.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            phase (float): The phase setting, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodePhaseSet(
            self.hdwf,
            channel_index,
            node.value,
            phase)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodePhaseGet(self, channel_index: int, node: DwfAnalogOutNode) -> float:
        '''Get the phase for an |AnalogOut| channel node, in degrees.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            float: The currently configured node phase value, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_phase = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutNodePhaseGet(
            self.hdwf,
            channel_index,
            node.value,
            c_phase)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        phase = c_phase.value
        return phase

    ###################################################################################################################
    #                                                                                                                 #
    #                                              NODE DATA MANAGEMENT                                               #
    #                                                                                                                 #
    ###################################################################################################################

    def nodeDataInfo(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[float, float]:
        '''Get data range for an |AnalogOut| channel node, in samples.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[float, float]: The range of valid values.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_samples_min = typespec_ctypes.c_int()
        c_samples_max = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutNodeDataInfo(
            self.hdwf,
            channel_index,
            node.value,
            c_samples_min,
            c_samples_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        samples_min = c_samples_min.value
        samples_max = c_samples_max.value
        return (samples_min, samples_max)

    def nodeDataSet(self, channel_index: int, node: DwfAnalogOutNode, data: np.ndarray) -> None:
        '''Set the data for an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            data (np.ndarray): The data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        double_data = data.astype(np.float64)

        result = self.lib.FDwfAnalogOutNodeDataSet(
            self.hdwf,
            channel_index,
            node.value,
            double_data.ctypes.data_as(typespec_ctypes.c_double_ptr),
            len(double_data))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nodePlayStatus(self, channel_index: int, node: DwfAnalogOutNode) -> Tuple[int, int, int]:
        '''Get the play status for an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.

        Returns:
            Tuple[int, int, int]: The *free*, *lost*, and *corrupted* status counts, in samples.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_data_free = typespec_ctypes.c_int()
        c_data_lost = typespec_ctypes.c_int()
        c_data_corrupted = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutNodePlayStatus(
            self.hdwf,
            channel_index,
            node.value,
            c_data_free,
            c_data_lost,
            c_data_corrupted)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        data_free = c_data_free.value
        data_lost = c_data_lost.value
        data_corrupted = c_data_corrupted.value
        return (data_free, data_lost, data_corrupted)

    def nodePlayData(self, channel_index: int, node: DwfAnalogOutNode, data: np.ndarray) -> None:
        '''Provide the playback data for an |AnalogOut| channel node.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            node (DwfAnalogOutNode): The channel node.
            data (np.ndarray): The playback data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutNodePlayData(
            self.hdwf,
            channel_index,
            node.value,
            data.ctypes.data_as(typespec_ctypes.c_double_ptr),
            len(data))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    ###################################################################################################################
    #                                                                                                                 #
    #                                        CARRIER CONFIGURATION (OBSOLETE)                                         #
    #                                                                                                                 #
    ###################################################################################################################

    def enableSet(self, channel_index: int, enable: bool) -> None:
        '''Enable or disable the specified |AnalogOut| channel.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeEnableSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            enable (bool): The enable setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutEnableSet(self.hdwf, channel_index, enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def enableGet(self, channel_index: int) -> bool:
        '''Get the current enable/disable status of the specified |AnalogOut| channel.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeEnableGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            bool: The 'enable' state of the channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_enable = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutEnableGet(self.hdwf, channel_index, c_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        enable = bool(c_enable.value)
        return enable

    def functionInfo(self, channel_index: int) -> List[DwfAnalogOutFunction]:
        '''Get the |AnalogOut| channel waveform shape function info.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFunctionInfo` method instead.

        Returns:
            List[DwfAnalogOutFunction]: The valid waveform shape functions.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_function_info_bitset = typespec_ctypes.c_unsigned_int()
        result = self.lib.FDwfAnalogOutFunctionInfo(self.hdwf, channel_index, c_function_info_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        function_info_bitset = c_function_info_bitset.value
        function_info_list = [function_ for function_ in DwfAnalogOutFunction
                              if function_info_bitset & (1 << function_.value)]
        return function_info_list

    def functionSet(self, channel_index: int, func: DwfAnalogOutFunction) -> None:
        '''Set the |AnalogOut| channel waveform shape function.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFunctionSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            func (DwfAnalogOutFunction): The waveform shape function to use.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutFunctionSet(self.hdwf, channel_index, func.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def functionGet(self, channel_index: int) -> DwfAnalogOutFunction:
        '''Get the |AnalogOut| channel waveform shape function.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFunctionGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            DwfAnalogOutFunction: The currently configured waveform shape function.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_func = typespec_ctypes.DwfAnalogOutFunction()
        result = self.lib.FDwfAnalogOutFunctionGet(self.hdwf, channel_index, c_func)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        func = DwfAnalogOutFunction(c_func.value)
        return func

    def frequencyInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel valid frequency range, in Hz.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFrequencyInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The valid frequency range, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency_min = typespec_ctypes.c_double()
        c_frequency_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutFrequencyInfo(self.hdwf, channel_index, c_frequency_min, c_frequency_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency_min = c_frequency_min.value
        frequency_max = c_frequency_max.value
        return (frequency_min, frequency_max)

    def frequencySet(self, channel_index: int, frequency: float) -> None:
        '''Set the |AnalogOut| channel frequency, in Hz.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFrequencySet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            frequency (float): The frequency to use.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutFrequencySet(self.hdwf, channel_index, frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def frequencyGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel frequency, in Hz.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeFrequencyGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured frequency, in Hz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_frequency = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutFrequencyGet(self.hdwf, channel_index, c_frequency)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        frequency = c_frequency.value
        return frequency

    def amplitudeInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel amplitude range info.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeAmplitudeInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of valid amplitudes, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_amplitude_min = typespec_ctypes.c_double()
        c_amplitude_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutAmplitudeInfo(self.hdwf, channel_index, c_amplitude_min, c_amplitude_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        amplitude_min = c_amplitude_min.value
        amplitude_max = c_amplitude_max.value
        return (amplitude_min, amplitude_max)

    def amplitudeSet(self, channel_index: int, amplitude: float) -> None:
        '''Set the |AnalogOut| channel amplitude.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeAmplitudeSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            amplitude (float): The amplitude, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutAmplitudeSet(self.hdwf, channel_index, amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def amplitudeGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel amplitude.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeAmplitudeGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured amplitude, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_amplitude = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutAmplitudeGet(self.hdwf, channel_index, c_amplitude)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        amplitude = c_amplitude.value
        return amplitude

    def offsetInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel offset range info, in Volts.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeOffsetInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The valid range of offset values, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_offset_min = typespec_ctypes.c_double()
        c_offset_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutOffsetInfo(self.hdwf, channel_index, c_offset_min, c_offset_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset_min = c_offset_min.value
        offset_max = c_offset_max.value
        return (offset_min, offset_max)

    def offsetSet(self, channel_index: int, offset: float) -> None:
        '''Set the |AnalogOut| channel offset, in Volts.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeOffsetSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            offset (float): The channel offset, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutOffsetSet(self.hdwf, channel_index, offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def offsetGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel offset, in Volts.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeOffsetGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The valid offset value, in Volts.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_offset = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutOffsetGet(self.hdwf, channel_index, c_offset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        offset = c_offset.value
        return offset

    def symmetryInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel symmetry setting range.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeSymmetryInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of valid symmetry settings.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_symmetry_min = typespec_ctypes.c_double()
        c_symmetry_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutSymmetryInfo(self.hdwf, channel_index, c_symmetry_min, c_symmetry_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        symmetry_min = c_symmetry_min.value
        symmetry_max = c_symmetry_max.value
        return (symmetry_min, symmetry_max)

    def symmetrySet(self, channel_index: int, symmetry: float) -> None:
        '''Set the |AnalogOut| channel symmetry setting.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeSymmetrySet` method instead.

        Parameters:
            channel_index (int):  The |AnalogOut| channel.
            symmetry (float): The channel symmetry setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutSymmetrySet(self.hdwf, channel_index, symmetry)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def symmetryGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel symmetry setting.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeSymmetryGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured symmetry setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_symmetry = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutSymmetryGet(self.hdwf, channel_index, c_symmetry)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        symmetry = c_symmetry.value
        return symmetry

    def phaseInfo(self, channel_index: int) -> Tuple[float, float]:
        '''Get the |AnalogOut| channel phase range, in degrees.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodePhaseInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[float, float]: The range of valid phase values, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_phase_min = typespec_ctypes.c_double()
        c_phase_max = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutPhaseInfo(self.hdwf, channel_index, c_phase_min, c_phase_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        phase_min = c_phase_min.value
        phase_max = c_phase_max.value
        return (phase_min, phase_max)

    def phaseSet(self, channel_index: int, phase: float) -> None:
        '''Set the |AnalogOut| channel phase, in degrees.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodePhaseSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            phase (float): The phase setting, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogOutPhaseSet(self.hdwf, channel_index, phase)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def phaseGet(self, channel_index: int) -> float:
        '''Get the |AnalogOut| channel phase, in degrees.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodePhaseGet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            float: The currently configured phase, in degrees.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_phase = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogOutPhaseGet(self.hdwf, channel_index, c_phase)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        phase = c_phase.value
        return phase

    ###################################################################################################################
    #                                                                                                                 #
    #                                     CARRIER NODE DATA MANAGEMENT (OBSOLETE)                                     #
    #                                                                                                                 #
    ###################################################################################################################

    def dataInfo(self, channel_index: int) -> Tuple[int, int]:
        '''Get the |AnalogOut| channel data buffer range.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeDataInfo` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[int, int]: The data range.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_num_samples_min = typespec_ctypes.c_int()
        c_num_samples_max = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutDataInfo(self.hdwf, channel_index, c_num_samples_min, c_num_samples_max)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        num_samples_min = c_num_samples_min.value
        num_samples_max = c_num_samples_max.value
        return (num_samples_min, num_samples_max)

    def dataSet(self, channel_index: int, data: np.ndarray) -> None:
        '''Set the |AnalogOut| channel data.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodeDataSet` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            data (np.ndarray): The data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        double_data = data.astype(np.float64)

        result = self.lib.FDwfAnalogOutDataSet(
            self.hdwf,
            channel_index,
            double_data.ctypes.data_as(typespec_ctypes.c_double_ptr),
            len(double_data))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def playStatus(self, channel_index: int) -> Tuple[int, int, int]:
        '''Get the |AnalogOut| channel playback status, in samples.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodePlayStatus` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.

        Returns:
            Tuple[int, int, int]: The playback status.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_data_free = typespec_ctypes.c_int()
        c_data_lost = typespec_ctypes.c_int()
        c_data_corrupted = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogOutPlayStatus(
            self.hdwf,
            channel_index,
            c_data_free,
            c_data_lost,
            c_data_corrupted)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        data_free = c_data_free.value
        data_lost = c_data_lost.value
        data_corrupted = c_data_corrupted.value
        return (data_free, data_lost, data_corrupted)

    def playData(self, channel_index: int, data: np.ndarray) -> None:
        '''Provide the |AnalogOut| channel playback data.

        Warning:
            **This method is obsolete.**

            Use the :py:meth:`nodePlayData` method instead.

        Parameters:
            channel_index (int): The |AnalogOut| channel.
            data (np.ndarray): The playback data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        double_data = data.astype(np.float64)

        result = self.lib.FDwfAnalogOutPlayData(
            self.hdwf,
            channel_index,
            double_data.ctypes.data_as(typespec_ctypes.c_double_ptr),
            len(double_data))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
