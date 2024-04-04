'''The |pydwf.core.api.ProtocolI2C| module provides a single class: |ProtocolI2C|.'''

from typing import List, Tuple

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class ProtocolI2C(AbstractDwfDeviceSubAPI):
    '''The |ProtocolI2C| class provides access to the I²C protocol functionality of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently accessible via its
        |protocol.i2c:link| attribute.
    '''

    def reset(self) -> None:
        '''Reset the I²C protocol instrument.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalI2cReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def clear(self) -> bool:
        '''Clear the I²C bus.

        Detect and try to solve an I²C bus lockup condition.

        Todo:
            The precise behavior of this method needs to be understood and documented.

        Returns:
            bool: True if the bus is clear after the operation, False otherwise.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_bus_free = typespec_ctypes.c_int()
        result = self.lib.FDwfDigitalI2cClear(self.hdwf, c_bus_free)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        bus_free = bool(c_bus_free.value)
        return bus_free

    def stretchSet(self, stretch_enable: int) -> None:
        '''Set I²C stretch behavior.

        Todo:
            The precise behavior of this method needs to be understood and documented.

        Parameters:
            stretch_enable (bool): True to enable stretch, False to disable.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cStretchSet(self.hdwf, bool(stretch_enable))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rateSet(self, data_rate: float) -> None:
        '''Set the I²C data rate, in Hz.

        Parameters:
            data_rate (float): I²C data rate. Often-encountered rates are 100 kHz and 400 kHz, but many
                modern I²C devices support higher data rates. Check the datasheet of your device.

                The default value is 100 kHz.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cRateSet(self.hdwf, data_rate)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def timeoutSet(self, timeout: float) -> None:
        '''Set the I²C timeout, in seconds.

        Parameters:
            timeout (float): I²C timeout. The default value is 1 second.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cTimeoutSet(self.hdwf, timeout)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def readNakSet(self, nak_last_read_byte: int) -> None:
        '''Set read NAK state.

        Todo:
            The precise behavior of this method needs to be understood and documented.

        Parameters:
            nak_last_read_byte (int): (undocumented)

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cReadNakSet(self.hdwf, nak_last_read_byte)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def sclSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the I²C clock (SCL) signal is transmitted.

        Parameters:
            channel_index (int): The digital channel (pin) on which to generate the SCL clock.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cSclSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def sdaSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the I²C data (SDA) signal is transmitted/received.

        Parameters:
            channel_index (int): The digital channel (pin) on which to send/receive SDA data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cSdaSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def writeRead(self, address: int, tx: List[int], number_of_rx_bytes: int) -> Tuple[int, List[int]]:
        '''Perform a combined I²C write/read operation.

        Parameters:
            address (int): The I²C address of the target device.
            tx (List[int]): The octets to send.
            number_of_rx_bytes (int): The number of octets to receive.

        Returns:
            Tuple[int, List[int]]:
                The first element is the NAK indication; the second element is a list of octet values received.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_nak = typespec_ctypes.c_int()

        tx_list = list(tx)

        number_of_tx_bytes = len(tx_list)

        tx_buffer_type = typespec_ctypes.c_unsigned_char * number_of_tx_bytes
        rx_buffer_type = typespec_ctypes.c_unsigned_char * number_of_rx_bytes

        tx_buffer = tx_buffer_type(*tx_list)
        rx_buffer = rx_buffer_type()

        result = self.lib.FDwfDigitalI2cWriteRead(
            self.hdwf,
            address,
            tx_buffer,
            number_of_tx_bytes,
            rx_buffer,
            number_of_rx_bytes,
            c_nak)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        nak = c_nak.value

        rx_list = list(rx_buffer)

        return (nak, rx_list)

    def read(self, address: int, number_of_words: int) -> Tuple[int, List[int]]:
        '''Perform an I²C read operation.

        Parameters:
            address (int): The I²C address of the target device.
            number_of_words (int): The number of octets to receive.

        Returns:
            Tuple[int, List[int]]:
                The first element is the NAK indication; the second element is a list of octet values received.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_nak = typespec_ctypes.c_int()

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        rx_buffer = buffer_type()

        result = self.lib.FDwfDigitalI2cRead(self.hdwf, address, rx_buffer, number_of_words, c_nak)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        nak = c_nak.value

        rx_list = list(rx_buffer)

        return (nak, rx_list)

    def write(self, address: int, tx: List[int]) -> int:
        '''Perform an I²C write operation.

        Parameters:
            address (int): The I²C address of the target device.
            tx (List[int]): The octets to send.

        Returns:
            int: The NAK indication.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_nak = typespec_ctypes.c_int()

        tx_list = list(tx)

        number_of_words = len(tx_list)

        buffer_type = typespec_ctypes.c_unsigned_char * number_of_words

        tx_buffer = buffer_type(*tx_list)

        result = self.lib.FDwfDigitalI2cWrite(self.hdwf, address, tx_buffer, number_of_words, c_nak)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        nak = c_nak.value

        return nak

    def writeOne(self, address: int, tx: int) -> int:
        '''Perform an I²C write operation of a single octet.

        Parameters:
            address (int): The I²C address of the target device.
            tx (int): The single octet to send.

        Returns:
            int: The NAK indication.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_nak = typespec_ctypes.c_int()

        result = self.lib.FDwfDigitalI2cWriteOne(self.hdwf, address, tx, c_nak)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        nak = c_nak.value

        return nak

    def spyStart(self) -> None:
        '''Start I²C spy.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalI2cSpyStart(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def spyStatus(self, max_data_size: int) -> Tuple[int, int, List[int], int]:
        '''Get I²C spy status.

        Returns:
            Tuple[int, int, List[int], int]: A tuple (start, stop, data-values, nak-indicator).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''

        c_nak   = typespec_ctypes.c_int()
        c_start = typespec_ctypes.c_int()
        c_stop  = typespec_ctypes.c_int()

        buffer_type = typespec_ctypes.c_unsigned_char * max_data_size
        c_data = buffer_type()

        c_data_size = typespec_ctypes.c_int(max_data_size)

        result = self.lib.FDwfDigitalI2cSpyStatus(self.hdwf, c_start, c_stop, c_data, c_data_size, c_nak)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

        start = c_start.value
        stop  = c_stop.value
        nak   = c_nak.value

        data_list = list(c_data)[:c_data_size.value]  # Only return the first 'c_data_size' words.

        return (start, stop, data_list, nak)
