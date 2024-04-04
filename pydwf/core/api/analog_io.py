'''The |pydwf.core.api.analog_io| module provides a single class: |AnalogIO|.'''

from typing import Tuple
import ctypes

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.enum_types import DwfAnalogIO


class AnalogIO(AbstractDwfDeviceSubAPI):
    '''The |AnalogIO| class provides access to the analog I/O functionality of a |DwfDevice:link|.

    The |AnalogIO| methods are used to control the power supplies, reference voltage supplies, voltmeters, ammeters,
    thermometers, and any other sensors on the device. These are organized into channels which contain a number of
    nodes. For instance, a power supply channel might have three nodes: an 'enable' setting, a voltage level
    setting/reading, and current limitation setting/reading.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently assigned to its public
        |analogIO:link| attribute for access by the user.
    '''

    def reset(self) -> None:
        '''Reset and configure all |AnalogIO| settings to default values.

        If autoconfiguration is enabled, the changes take effect immediately.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfAnalogIOReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def configure(self) -> None:
        '''Configure the |AnalogIO| functionality.

        This method transfers the settings to the Digilent Waveforms device. It is not needed
        if autoconfiguration is enabled.

        Raises:
            DwfLibraryError: An error occurred while executing the *configure* operation.
        '''
        result = self.lib.FDwfAnalogIOConfigure(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def status(self) -> None:
        '''Read the status of the device and stores it internally.

        The status inquiry methods that follow will return the information that was read from the device
        when this method was last called.

        Note that the |AnalogIO| functionality is not managed by a state machine,
        so this method does not return a value.

        Raises:
            DwfLibraryError: An error occurred while executing the *status* operation.
        '''
        result = self.lib.FDwfAnalogIOStatus(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def enableInfo(self) -> Tuple[bool, bool]:
        '''Verify if *Master Enable* and/or *Master Enable Status* are supported.

        The *Master Enable* is a software switch that enable the |AnalogIO| voltage sources.

        If supported, the current value of this *Master Enable* switch (Enabled/Disabled) can be
        set by the :py:meth:`enableSet` method and queried by the :py:meth:`enableGet` method.

        The *Master Enable Status* that can be queried by the :py:meth:`enableStatus` method may
        be different from the *Master Enable* value if e.g. an over-current protection circuit
        has been triggered.

        Returns:
            Tuple[bool, bool]: The tuple elements indicate whether *Master Enable Set* and
            *Master Enable Status*, respectively, are supported by the |AnalogIO| device.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_set_supported = typespec_ctypes.c_int()
        c_status_supported = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOEnableInfo(self.hdwf, c_set_supported, c_status_supported)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        set_supported = bool(c_set_supported.value)
        status_supported = bool(c_status_supported.value)
        return (set_supported, status_supported)

    def enableSet(self, master_enable: bool) -> None:
        '''Set value of the *Master Enable* setting.

        Parameters:
            master_enable (bool): The new value of the *Master Enable* setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogIOEnableSet(self.hdwf, master_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def enableGet(self) -> bool:
        '''Return the current value of the *Master Enable* setting.

        Returns:
            The value of the *Master Enable* setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_master_enable = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOEnableGet(self.hdwf, c_master_enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        master_enable = bool(c_master_enable.value)
        return master_enable

    def enableStatus(self) -> bool:
        '''Return the actual *Master Enable Status* value (if the device supports it).

        The *Master Enable Status* value may be different from the *Master Enable* setting if
        e.g. an over-current protection circuit has been triggered.

        Returns:
            bool: the current status of the *Master Enable* circuit.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_master_enable_status = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOEnableStatus(self.hdwf, c_master_enable_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        master_enable_status = bool(c_master_enable_status.value)
        return master_enable_status

    def channelCount(self) -> int:
        '''Return the number of |AnalogIO| channels available on the device.

        Returns:
            int: The number of |AnalogIO| channels.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_channel_count = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOChannelCount(self.hdwf, c_channel_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_count = c_channel_count.value
        return channel_count

    def channelName(self, channel_index: int) -> Tuple[str, str]:
        '''Return the name (long text) and label (short text, printed on the device) for the
        specified |AnalogIO| channel.

        Parameters:
            channel_index (int): The channel for which we want to get the name and label.

        Returns:
            Tuple[str, str]: The name and label of the channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_channel_name = ctypes.create_string_buffer(32)
        c_channel_label = ctypes.create_string_buffer(16)
        result = self.lib.FDwfAnalogIOChannelName(
            self.hdwf,
            channel_index,
            c_channel_name,
            c_channel_label)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        channel_name = c_channel_name.value.decode()
        channel_label = c_channel_label.value.decode()
        return (channel_name, channel_label)

    def channelInfo(self, channel_index: int) -> int:
        '''Return the number of nodes associated with the specified |AnalogIO| channel.

        Parameters:
            channel_index (int): The channel for which we want to get the number of associated nodes.

        Returns:
            int: The number of nodes associated to the channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_node_count = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOChannelInfo(self.hdwf, channel_index, c_node_count)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        return c_node_count.value

    def channelNodeName(self, channel_index: int, node_index: int) -> Tuple[str, str]:
        '''Return the node name ('Voltage', 'Current', …) and units ('V', 'A', …) for the specified |AnalogIO| node.

        Parameters:
            channel_index (int): The channel for which we want to get the name and unit.
            node_index (int): The node for which we want to get the name and unit.

        Returns:
            Tuple[str, str]: The name and unit of the quantity associated with the node.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_node_name = ctypes.create_string_buffer(32)
        c_node_units = ctypes.create_string_buffer(16)
        result = self.lib.FDwfAnalogIOChannelNodeName(
            self.hdwf,
            channel_index,
            node_index,
            c_node_name,
            c_node_units)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        node_name = c_node_name.value.decode()
        node_units = c_node_units.value.decode()
        return (node_name, node_units)

    def channelNodeInfo(self, channel_index: int, node_index: int) -> DwfAnalogIO:
        '''Return the type of physical quantity (e.g., voltage, current, or temperature)
        represented by the specified |AnalogIO| channel node.

        Parameters:
            channel_index (int): The channel for which we want to get the type of physical quantity.
            node_index (int): The node for which we want to get the type of physical quantity.

        Returns:
            DwfAnalogIO: The type of physical quantity represented by the node.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_analog_io = typespec_ctypes.DwfAnalogIO()
        result = self.lib.FDwfAnalogIOChannelNodeInfo(
            self.hdwf,
            channel_index,
            node_index,
            c_analog_io)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        analog_io = DwfAnalogIO(c_analog_io.value)
        return analog_io

    def channelNodeSetInfo(self, channel_index: int, node_index: int) -> Tuple[float, float, int]:
        '''Return the limits of the value that can be assigned to the specified |AnalogIO| channel node.

        Since a node can represent many things (power supply, temperature sensor, etc.),
        the *minimum*, *maximum*, and *steps* parameters also represent different types of values.

        The :py:meth:`channelNodeInfo` method returns the type of values to expect and the
        :py:meth:`channelNodeName` method returns the units of these values.

        Parameters:
            channel_index (int): The channel for which we want to get the currently configured value.
            node_index (int): The node for which we want to get the currently configured value.

        Returns:
            Tuple[float, float, int]: The minimum and maximum values for the specified node's value,
            and the number of resolution steps.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_min_value = typespec_ctypes.c_double()
        c_max_value = typespec_ctypes.c_double()
        c_num_steps = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOChannelNodeSetInfo(
            self.hdwf,
            channel_index,
            node_index,
            c_min_value,
            c_max_value,
            c_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        min_value = c_min_value.value
        max_value = c_max_value.value
        num_steps = c_num_steps.value
        return (min_value, max_value, num_steps)

    def channelNodeSet(self, channel_index: int, node_index: int, node_value: float) -> None:
        '''Set the node value for the specified |AnalogIO| channel node.

        Parameters:
            channel_index (int): The channel for which we want to set the value.
            node_index (int): The node for which we want to set the value.
            node_value (float): The value we want to set the node to.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfAnalogIOChannelNodeSet(
            self.hdwf,
            channel_index,
            node_index,
            node_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def channelNodeGet(self, channel_index: int, node_index: int) -> float:
        '''Return the current value of the specified |AnalogIO| channel node.

        Parameters:
            channel_index (int): The channel for which we want to get the current value.
            node_index (int): The node for which we want to get the current value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_node_value = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogIOChannelNodeGet(
            self.hdwf,
            channel_index,
            node_index,
            c_node_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        node_value = c_node_value.value
        return node_value

    def channelNodeStatusInfo(self, channel_index: int, node_index: int) -> Tuple[float, float, int]:
        '''Return the range of status values for the specified |AnalogIO| channel node.

        Parameters:
            channel_index (int): The channel for which we want to get status information.
            node_index (int): The node for which we want to get status information.

        Returns:
            Tuple[float, float, int]: The minimum and maximum status values for the specified node,
            and the number of resolution steps.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_min_value = typespec_ctypes.c_double()
        c_max_value = typespec_ctypes.c_double()
        c_num_steps = typespec_ctypes.c_int()
        result = self.lib.FDwfAnalogIOChannelNodeStatusInfo(
            self.hdwf,
            channel_index,
            node_index,
            c_min_value,
            c_max_value,
            c_num_steps)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        min_value = c_min_value.value
        max_value = c_max_value.value
        num_steps = c_num_steps.value
        return (min_value, max_value, num_steps)

    def channelNodeStatus(self, channel_index: int, node_index: int) -> float:
        '''Return the most recent status value reading of the specified |AnalogIO| channel node.

        To fetch updated values for all |AnalogIO| nodes, use the :py:meth:`status` method.

        Returns:
            float: The most recent value read for this channel node.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_node_value = typespec_ctypes.c_double()
        result = self.lib.FDwfAnalogIOChannelNodeStatus(
            self.hdwf,
            channel_index,
            node_index,
            c_node_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        node_value = c_node_value.value
        return node_value
