'''The |pydwf.core.api.protocol_can| module provides a single class: |ProtocolCAN|.'''

from typing import Tuple
import ctypes

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS
from pydwf.core.auxiliary.exceptions import PyDwfError


class ProtocolCAN(AbstractDwfDeviceSubAPI):
    '''The |ProtocolCAN| class provides access to the CAN bus protocol functionality of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently accessible via its
        |protocol.can:link| attribute.
    '''

    def reset(self) -> None:
        '''Reset the CAN bus protocol functionality.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalCanReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rateSet(self, data_rate: float) -> None:
        '''Set the CAN bus data rate, in Hz.

        Parameters:
            data_rate (float): The data-rate used by the receiver and transmitter.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalCanRateSet(self.hdwf, data_rate)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def polaritySet(self, polarity: bool) -> None:
        '''Set the CAN bus polarity.

        Parameters:
            polarity (bool): If True, set polarity to *high*.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalCanPolaritySet(self.hdwf, int(polarity))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def txSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the outgoing (TX) signal will be transmitted.

        Parameters:
            channel_index (int): The digital channel (pin) on which to transmit data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalCanTxSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rxSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the incoming (RX) signal is received.

        Parameters:
            channel_index (int): The digital channel (pin) on which to receive data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalCanRxSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def tx(self, v_id: int, extended: bool, remote: bool, data: bytes) -> None:
        '''Transmit outgoing CAN bus data.

        Parameters:
            v_id (int): (to be documented).
            extended (bool): (to be documented).
            remote (bool): (to be documented).
            data (bytes): The data to be transmitted. Should be at most 8 bytes.

        Raises:
            PyDwfError: A request to transmit more than 8 bytes was made.
            DwfLibraryError: An error occurred while executing the operation.
        '''
        if len(data) > 8:
            raise PyDwfError('CAN message too long.')

        # We convert the 'bytes' input data to an array of ctypes unsigned chars.
        # This silences a spurious mypy warning.

        data_ctypes_array_type = (ctypes.c_ubyte * len(data))  # pylint: disable=superfluous-parens
        data_ctypes = data_ctypes_array_type(*data)

        result = self.lib.FDwfDigitalCanTx(
                self.hdwf,
                v_id,
                extended,
                remote,
                len(data),
                data_ctypes
            )

        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rx(self, size: int = 8) -> Tuple[int, bool, bool, bytes, int]:
        '''Receive incoming CAN bus data.

        Returns:
            Tuple[int, bool, bool, bytes, int]: A tuple (vID, extended, remote, data, status)

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_v_id     = typespec_ctypes.c_int()
        c_extended = typespec_ctypes.c_int()
        c_remote   = typespec_ctypes.c_int()
        c_dlc      = typespec_ctypes.c_int()
        c_data     = ctypes.create_string_buffer(size)
        c_status   = typespec_ctypes.c_int()

        result = self.lib.FDwfDigitalCanRx(
            self.hdwf,
            c_v_id,
            c_extended,
            c_remote,
            c_dlc,
            ctypes.cast(c_data, typespec_ctypes.c_unsigned_char_ptr),
            size,
            c_status)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        v_id      = c_v_id.value
        extended = bool(c_extended.value)
        remote   = bool(c_remote.value)
        dlc      = c_dlc.value
        data     = c_data.raw[:dlc]
        status   = c_status.value

        return (v_id, extended, remote, data, status)
