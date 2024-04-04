'''The |pydwf.core.dwf_device| module provides a single class: |DwfWaveformsDevice|.'''

from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

from pydwf.core.auxiliary.enum_types import DwfTriggerSource, DwfTriggerSlope, DwfDeviceParameter
from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS

from pydwf.core.api.analog_in import AnalogIn
from pydwf.core.api.analog_out import AnalogOut
from pydwf.core.api.digital_in import DigitalIn
from pydwf.core.api.digital_out import DigitalOut
from pydwf.core.api.analog_io import AnalogIO
from pydwf.core.api.digital_io import DigitalIO
from pydwf.core.api.analog_impedance import AnalogImpedance

from pydwf.core.api.protocol_uart import ProtocolUART
from pydwf.core.api.protocol_spi import ProtocolSPI
from pydwf.core.api.protocol_i2c import ProtocolI2C
from pydwf.core.api.protocol_can import ProtocolCAN
from pydwf.core.api.protocol_swd import ProtocolSWD

from pydwf.core.dwf_device_protocol_apis import DwfDeviceProtocolAPIs

if TYPE_CHECKING:
    # These are only needed when type-checking.
    import ctypes
    from pydwf.core.dwf_library import DwfLibrary


class DwfDevice:
    '''The |DwfDevice| represents a single Digilent Waveforms test and measurement device.

    Attention:

        Users of |pydwf| should not create instances of this class directly.

        Use |DeviceControl.open:link| or |pydwf.utilities.openDwfDevice:link| to obtain a valid
        |DwfDevice| instance.

    The main test and measurement functionality of a Digilent Waveforms device is provided as
    multiple sub-interfaces (instruments, protocols, and measurements). To access those,
    use one of the twelve attributes described below.

    .. rubric:: |DwfDevice| attributes

    Attributes:

        analogIn (AnalogIn):
            Provides access analog input (oscilloscope) functionality.
        analogOut (AnalogOut):
            Provides access to the analog output (waveform generator) functionality.
        analogIO (AnalogIO):
            Provides access to the analog I/O (voltage source, monitoring) functionality.
        analogImpedance (AnalogImpedance):
            Provides access to the analog impedance measurement functionality.
        digitalIn (DigitalIn):
            Provides access to the dynamic digital input (logic analyzer) functionality.
        digitalOut (DigitalOut):
            Provides access to the dynamic digital output (pattern generator) functionality.
        digitalIO (DigitalIO):
            Provides access to the static digital I/O functionality.
        protocol.uart (ProtocolUART):
            Provides access to the UART protocol functionality.
        protocol.can (ProtocolCAN):
            Provides access to the CAN protocol functionality.
        protocol.spi (ProtocolSPI):
            Provides access to the SPI protocol functionality.
        protocol.i2c (ProtocolI2C):
            Provides access to the I²C protocol functionality.
        protocol.swd (ProtocolSWD):
            Provides access to the SWD protocol functionality.

    .. rubric:: |DwfDevice| properties and methods
    '''

    # pylint: disable=too-many-instance-attributes

    def __init__(self, dwf: DwfLibrary, hdwf: int) -> None:
        '''Initialize a DwfDevice with the specified DWF handle value.

        This method is used by the |DwfLibrary.device.open| method to initialize a |DwfDevice| immediately after
        successfully opening a Digilent Waveforms device using the low-level C API.

        User programs should not initialize |DwfDevice| instances themselves, but leave that to |pydwf| instead.

        Parameters:
            dwf (DwfLibrary):
                The |DwfLibrary| instance used to make calls to the underlying library.
            hdwf (int):
                The device to open.
        '''

        self._dwf = dwf
        self._hdwf = hdwf

        # Initialize sub-API instances and assign them to attributes.

        self.analogIn = AnalogIn(self)
        self.analogOut = AnalogOut(self)
        self.analogIO = AnalogIO(self)
        self.analogImpedance = AnalogImpedance(self)
        self.digitalIn = DigitalIn(self)
        self.digitalOut = DigitalOut(self)
        self.digitalIO = DigitalIO(self)

        self.protocol = DwfDeviceProtocolAPIs(self)

    def __enter__(self):
        '''Context manager entry method.'''
        return self

    def __exit__(self, *dummy):
        '''Context manager exit method.'''
        self.close()

    @property
    def dwf(self) -> 'DwfLibrary':
        '''Return the |DwfLibrary| instance that was used to create (open) this |DwfDevice| instance.

        This is useful if we have a |DwfDevice|, but we need its |DwfLibrary|.

        Returns:
            DwfLibrary: The |DwfLibrary| that was used to create (open) this |DwfDevice| instance.
        '''
        return self._dwf

    @property
    def hdwf(self) -> int:
        '''Return the HDWF handle of the device.

        This property is intended for internal |pydwf| use.

        :meta private:
        '''
        return self._hdwf

    @property
    def lib(self) -> ctypes.CDLL:
        '''Return the |ctypes| shared library instance used to access the DWF library.

        This property is intended for internal |pydwf| use.

        :meta private:
        '''
        return self._dwf.lib

    @property
    def digitalUart(self) -> ProtocolUART:
        '''Old attribute-style access to the device's UART functionality.
        
        Warning:
            This attribute is obsolete.
            Use |protocol.uart:link| instead.
        '''
        warnings.warn('Using DwfDevice.digitalUart is deprecated. Use DwfDevice.protocol.uart instead.')
        return self.protocol.uart

    @property
    def digitalSpi(self) -> ProtocolSPI:
        '''Old attribute-style access to the device's SPI functionality.
        
        Warning:
            This attribute is obsolete.
            Use |protocol.spi:link| instead.
        '''
        warnings.warn('Using DwfDevice.digitalSpi is deprecated. Use DwfDevice.protocol.spi instead.')
        return self.protocol.spi

    @property
    def digitalI2c(self) -> ProtocolI2C:
        '''Old attribute-style access to the device's I²C functionality.
        
        Warning:
            This attribute is obsolete.
            Use |protocol.i2c:link| instead.
        '''
        warnings.warn('Using DwfDevice.digitalI2c is deprecated. Use DwfDevice.protocol.i2c instead.')
        return self.protocol.i2c

    @property
    def digitalCan(self) -> ProtocolCAN:
        '''Old attribute-style access to the device's CAN functionality.
        
        Warning:
            This attribute is obsolete.
            Use |protocol.can:link| instead.
        '''
        warnings.warn('Using DwfDevice.digitalCan is deprecated. Use DwfDevice.protocol.can instead.')
        return self.protocol.can

    @property
    def digitalSwd(self) -> ProtocolSWD:
        '''Old attribute-style access to the device's SWD functionality.

        Warning:
            This attribute is obsolete.
            Use |DwfDevice.protocol.swd:link| instead.
        '''
        warnings.warn('Using DwfDevice.digitalSwd is deprecated. Use DwfDevice.protocol.swd instead.')
        return self.protocol.swd

    def close(self) -> None:
        '''Close the device.

        This method should be called when access to the device is no longer needed.

        Once this method returns, the |DwfDevice| can no longer be used.

        Raises:
            DwfLibraryError: The device cannot be closed.
        '''
        result = self.lib.FDwfDeviceClose(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def autoConfigureSet(self, auto_configure: int) -> None:
        '''Enable or disable the autoconfiguration setting of the device.

        When this setting is enabled (the default), any change to an instrument setting is automatically
        transmitted to the Digilent Waveforms hardware device, without the need for an explicit call to
        the instrument's *configure()* method.

        This adds a small amount of latency to every *Set()* method; just as much latency as calling the
        corresponding *configure()* method explicitly after the *Set()* method would add.

        Autoconfiguration is enabled by default, and there is little reason to turn it off unless the
        user program wants to make frequent changes to many settings at once between measurements.

        With value 3, configuration will be applied dynamically, without stopping the instrument.

        Parameters:
            auto_configure (int): The new autoconfiguration setting.

                Possible values for this option:

                * 0 — disable
                * 1 — enable
                * 3 — dynamic

        Raises:
            DwfLibraryError: The value cannot be set.
        '''
        result = self.lib.FDwfDeviceAutoConfigureSet(self.hdwf, auto_configure)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def autoConfigureGet(self) -> int:
        '''Return the autoconfiguration setting of the device.

        Returns:
            int: The current autoconfiguration setting.

            Possible values for this option:

            * 0 — disable
            * 1 — enable
            * 3 — dynamic

        Raises:
            DwfLibraryError: The value cannot be retrieved.
        '''

        c_auto_configure = typespec_ctypes.c_int()
        result = self.lib.FDwfDeviceAutoConfigureGet(self.hdwf, c_auto_configure)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        auto_configure = c_auto_configure.value
        return auto_configure

    def reset(self) -> None:
        '''Reset all device and instrument settings to default values.

        The new settings are applied immediately if autoconfiguration is enabled.

        Raises:
            DwfLibraryError: The device cannot be reset.
        '''
        result = self.lib.FDwfDeviceReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def enableSet(self, enable: bool) -> None:
        '''Enable or disable the device.

        Parameters:
            enable (bool): True for enable, False for disable.

        Raises:
            DwfLibraryError: The device's enabled state cannot be set.
        '''
        result = self.lib.FDwfDeviceEnableSet(self.hdwf, enable)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerInfo(self) -> list[DwfTriggerSource]:
        '''Return the available trigger source options for the global trigger bus.

        Refer to the section on |triggering:link| for more information.

        The four main instruments (|AnalogIn|, |AnalogOut|, |DigitalIn|, and |DigitalOut|) can be configured
        to start their operation (data acquisition for the *In* instruments; signal generation for the *Out*
        instruments) immediately after some event happens. This is called *triggering*.

        Each of the instruments can be configured independently to use any of the trigger sources available
        inside the device. This method returns a list of all trigger sources that are available to each of the
        instruments.

        Returns:
            list[DwfTriggerSource]: A list of available trigger sources.

        Raises:
            DwfLibraryError: The list of supported trigger sources cannot be retrieved.
        '''
        c_trigger_source_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDeviceTriggerInfo(self.hdwf, c_trigger_source_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source_bitset = c_trigger_source_bitset.value
        trigger_source_list = [trigger_source for trigger_source in DwfTriggerSource
                               if trigger_source_bitset & (1 << trigger_source.value)]
        return trigger_source_list

    def triggerSet(self, pin_index: int, trigger_source: DwfTriggerSource) -> None:
        '''Configure the trigger I/O pin with a specific trigger source option.

        Digilent Waveforms devices have dedicated digital I/O pins that can be used either
        as trigger inputs or trigger outputs. Use this method to select which line of the
        global triggering bus is driven on those pins, e.g. to trigger some external device
        or to monitor the Digilent Waveforms device's internal trigger behavior.

        Pass :py:attr:`DwfTriggerSource.None_ <pydwf.core.auxiliary.enum_typesDwfTriggerSource.None_>`
        to disable trigger output on the pin. This is the default setting, and the appropriate value
        to use when the intention is to have some external trigger signal drive the pin.

        Refer to the section on |triggering:link| for more information.

        Parameters:
            pin_index (int): The trigger pin to configure.
            trigger_source (DwfTriggerSource): The trigger source to select.

        Raises:
            DwfLibraryError: The trigger source cannot be set.
        '''
        result = self.lib.FDwfDeviceTriggerSet(self.hdwf, pin_index, trigger_source.value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerGet(self, pin_index: int) -> DwfTriggerSource:
        '''Return the selected trigger source for a trigger I/O pin.

        Refer to the section on |triggering:link| for more information.

        Parameters:
            pin_index (int): The pin for which to obtain the selected trigger source.

        Returns:
           DwfTriggerSource: The trigger source setting for the selected pin.

        Raises:
            DwfLibraryError: The trigger source cannot be retrieved.
        '''
        c_trigger_source = typespec_ctypes.DwfTriggerSource()
        result = self.lib.FDwfDeviceTriggerGet(self.hdwf, pin_index, c_trigger_source)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        trigger_source = DwfTriggerSource(c_trigger_source.value)
        return trigger_source

    def triggerPC(self) -> None:
        '''Generate a trigger pulse on the PC trigger line.

        The generated pulse will trigger any instrument that is configured with trigger source
        :py:attr:`DwfTriggerSource.PC <pydwf.core.auxiliary.enum_types.DwfTriggerSource.PC>`,
        and currently armed (i.e., waiting for a trigger).

        Raises:
            DwfLibraryError: The PC trigger line cannot be pulsed.
        '''
        result = self.lib.FDwfDeviceTriggerPC(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def triggerSlopeInfo(self) -> list[DwfTriggerSlope]:
        '''Return the supported trigger slope options.

        Returns:
            list[DwfTriggerSlope]: A list of supported trigger slope values.

        Raises:
            DwfLibraryError: The trigger slope options cannot be retrieved.
        '''
        c_slope_bitset = typespec_ctypes.c_int()
        result = self.lib.FDwfDeviceTriggerSlopeInfo(self.hdwf, c_slope_bitset)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        slope_bitset = c_slope_bitset.value
        slope_list = [slope for slope in DwfTriggerSlope if slope_bitset & (1 << slope.value)]
        return slope_list

    def paramSet(self, parameter: DwfDeviceParameter, value: int) -> None:
        '''Set a device parameter value.

        Device parameters are settings of a specific |DwfDevice|.
        Refer to the |device parameters:link| section for more information.

        This method sets a device parameter value of a currently opened |DwfDevice|.

        Warning:
            The device parameter values are not checked to make sure they correspond to a valid
            value for the current device.

        Parameters:
            parameter (DwfDeviceParameter): The device parameter to set.
            value (int): The value to assign to the parameter.

        Raises:
            DwfLibraryError: The specified device parameter cannot be set.
        '''
        result = self.lib.FDwfDeviceParamSet(self.hdwf, parameter.value, value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def paramGet(self, parameter: DwfDeviceParameter) -> int:
        '''Get a device parameter value.

        Device parameters are settings of a specific |DwfDevice|.
        Refer to the |device parameters:link| section for more information.

        Parameters:
            parameter (DwfDeviceParameter): The device parameter to get.

        Returns:
            int: The integer value of the parameter.

        Raises:
            DwfLibraryError: The value of the specified device parameter cannot be retrieved.
        '''
        c_value = typespec_ctypes.c_int()
        result = self.lib.FDwfDeviceParamGet(self.hdwf, parameter.value, c_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        value = c_value.value
        return value
