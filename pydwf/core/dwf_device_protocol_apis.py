'''The |pydwf.core.dwf_device_protocol_apis| module provides a single class: |DwfDeviceProtocolAPIs|.'''

from __future__ import annotations

from typing import TYPE_CHECKING

from pydwf.core.api.protocol_uart import ProtocolUART
from pydwf.core.api.protocol_spi import ProtocolSPI
from pydwf.core.api.protocol_i2c import ProtocolI2C
from pydwf.core.api.protocol_can import ProtocolCAN
from pydwf.core.api.protocol_swd import ProtocolSWD

if TYPE_CHECKING:
    # This is only needed when type-checking.
    from pydwf.core.dwf_device import DwfDevice


class DwfDeviceProtocolAPIs:
    '''The |DwfDeviceProtocolAPIs| wraps the five digital protocols.

    Attention:

        Users of |pydwf| should not create instances of this class directly.

        Use |DeviceControl.open:link| or |pydwf.utilities.openDwfDevice:link| to obtain a valid
        |DwfDevice| instance.

    The protocol sub-APIs of the DWF library are collected in this class. The intention of this
    class is to make the protocols accessible from an instantantiated |DwfDevice| as
    |DwfDevice|.protocol.<name_of_protocol>.

    .. rubric:: |DwfDeviceProtocolAPIs| attributes

    Attributes:

        uart (ProtocollUART):
            Provides access to the UART protocol functionality.
        spi (DigitalSPI):
            Provides access to the SPI protocol functionality.
        i2c (DigitalI2C):
            Provides access to the I²C protocol functionality.
        can (DigitalCAN):
            Provides access to the CAN protocol functionality.
        swd (DigitalSWD):
            Provides access to the SWD protocol functionality.

    .. rubric:: |DwfDeviceProtocolAPIs| properties and methods
    '''

    # pylint: disable=too-few-public-methods

    def __init__(self, device: DwfDevice) -> None:
        '''Initialize a |DwfDeviceProtocolAPIs| instance with the specified DWF handle value.

        User programs should not initialize |DwfDeviceProtocolAPIs| instances themselves,
        but leave that to |pydwf| instead.

        Parameters:
            device (DwfDevice):
                The |DwfDevice| of which this |DwfDeviceProtocolAPIs| instance will be an attribute.
        '''

        self.uart = ProtocolUART(device)
        self.spi = ProtocolSPI(device)
        self.i2c = ProtocolI2C(device)
        self.can = ProtocolCAN(device)
        self.swd = ProtocolSWD(device)
