'''The |pydwf| package provides classes and types to control Digilent Waveforms devices.'''

# The version number of the *pydwf* package.
__version__ = '1.1.19'

# Originally authored and published by Sidney Cadot.
# Downloaded and modified for UC Berkeley PHYSICS 111A by the instructors (Spring 2024).
__author__ = 'Sidney Cadot'

# The *DwfLibrary* type is the only type users need to access library functionality.
from pydwf.core.dwf_library import DwfLibrary

# The *DwfDevice* type is not normally needed, but it may be convenient for type annotation.
from pydwf.core.dwf_device import DwfDevice

# Import the sub-APIs so they can be imported from the toplevel pydwf module.
from pydwf.core.api.device_control import DeviceControl
from pydwf.core.api.device_enumeration import DeviceEnumeration
from pydwf.core.api.spectrum import Spectrum
from pydwf.core.api.analog_in import AnalogIn
from pydwf.core.api.analog_out import AnalogOut
from pydwf.core.api.analog_io import AnalogIO
from pydwf.core.api.analog_impedance import AnalogImpedance
from pydwf.core.api.digital_in import DigitalIn
from pydwf.core.api.digital_out import DigitalOut
from pydwf.core.api.digital_io import DigitalIO
from pydwf.core.api.protocol_uart import ProtocolUART
from pydwf.core.api.protocol_spi import ProtocolSPI
from pydwf.core.api.protocol_i2c import ProtocolI2C
from pydwf.core.api.protocol_can import ProtocolCAN
from pydwf.core.api.protocol_swd import ProtocolSWD

# Import the 27 enumeration types and make them available for import directly from the pydwf package.

from pydwf.core.auxiliary.enum_types import (DwfErrorCode, DwfEnumFilter, DwfEnumConfigInfo, DwfDeviceID,
                                             DwfDeviceVersion, DwfDeviceParameter, DwfWindow, DwfState,
                                             DwfTriggerSource, DwfTriggerSlope, DwfAcquisitionMode,
                                             DwfAnalogInFilter, DwfAnalogCoupling, DwfAnalogInTriggerType,
                                             DwfAnalogInTriggerLengthCondition, DwfAnalogOutFunction,
                                             DwfAnalogOutNode, DwfAnalogOutMode, DwfAnalogOutIdle,
                                             DwfDigitalInClockSource, DwfDigitalInSampleMode,
                                             DwfDigitalOutOutput, DwfDigitalOutType, DwfDigitalOutIdle,
                                             DwfAnalogIO, DwfAnalogImpedance, DwfDmm)

# Import the two exception classes and make them available for import directly from the pydwf package.

from pydwf.core.auxiliary.exceptions import PyDwfError, DwfLibraryError
