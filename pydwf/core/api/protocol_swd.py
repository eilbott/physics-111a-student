'''The |pydwf.core.api.protocol_swd| module provides a single class: |ProtocolSWD|.'''

from typing import Tuple

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class ProtocolSWD(AbstractDwfDeviceSubAPI):
    '''The |ProtocolSWD| class provides access to the SWD bus protocol functionality of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently accessible via its
        |protocol.swd:link| attribute.
    '''

    def reset(self) -> None:
        '''Reset the SWD protocol functionality.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalSwdReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rateSet(self, data_rate: float) -> None:
        '''Set the SWD bus data rate, in Hz.

        Parameters:
            data_rate (float): The data-rate used by the receiver and transmitter.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdRateSet(self.hdwf, data_rate)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def clockSet(self, channel: int) -> None:
        '''Set the SWD clock channel.

        Parameters:
            channel (int): The SWD clock channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdCkSet(self.hdwf, channel)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def ioSet(self, channel: int) -> None:
        '''Set the SWD I/O channel.

        Parameters:
            channel (int): The SWD I/O channel.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdIoSet(self.hdwf, channel)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def turnSet(self, turn_setting: int) -> None:
        '''Set the SWD 'turn' parameter.

        Parameters:
            turn_setting (int): The turn setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdTurnSet(self.hdwf, turn_setting)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def trailSet(self, trail_setting: int) -> None:
        '''Set the SWD 'trail' parameter.

        Parameters:
            trail_setting (int): The trail setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdTrailSet(self.hdwf, trail_setting)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def parkSet(self, park_setting: int) -> None:
        '''Set the SWD 'park_setting' parameter.

        Parameters:
            park_setting (int): The park setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdParkSet(self.hdwf, park_setting)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def nakSet(self, nak_setting: int) -> None:
        '''Set the SWD 'nak_setting' parameter.

        Parameters:
            nak_setting (int): The nak setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdNakSet(self.hdwf, nak_setting)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def ioIdleSet(self, io_idle_setting: int) -> None:
        '''Set the SWD 'I/O idle' parameter.

        Parameters:
            io_idle_setting (int): The I/O idle setting.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdIoIdleSet(self.hdwf, io_idle_setting)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def clear(self, reset_value: int, trail_value) -> None:
        '''Clear the SWD bus.

        Parameters:
            reset_value (int): The reset value.
            trail_value (int): The trail value.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalSwdClear(self.hdwf, reset_value, trail_value)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def write(self, port: int, a32: int, write_data: int) -> int:
        '''Perform an SWD write.

        Parameters:
            port (int):
                * 0 — DataPort
                * 1 — AccessPort
            a32 (int):
                Address bits 3:2.
            write_data:
                Data to write.

        Returns:
            int: Acknowledgement bits: 1=OK, 2=WAIT, 4=FAILURE.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_ack = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalSwdWrite(self.hdwf, port, a32, c_ack, write_data)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        ack = c_ack.value
        return ack

    def read(self, port: int, a32: int) -> Tuple[int, int, bool]:
        '''Perform an SWD read.

        Parameters:
            port (int):
                * 0 — DataPort
                * 1 — AccessPort
            a32 (int):
                Address bits 3:2.

        Returns:
            Tuple[int, int, bool]: The first element of the tuple is the acknowledgement bits: 1=OK, 2=WAIT, 4=FAILURE.
                The second element of the tuple is the data word read. The third element of the tuple indicates if the
                CRC was correct (parity check).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_ack = typespec_ctypes.c_int()
        c_read_data = typespec_ctypes.c_unsigned_int()
        c_crc_ok = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalSwdRead(self.hdwf, port, a32, c_ack, c_read_data, c_crc_ok)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        ack = c_ack.value
        read_data = c_read_data.value
        crc_ok = bool(c_crc_ok.value)

        return (ack, read_data, crc_ok)
