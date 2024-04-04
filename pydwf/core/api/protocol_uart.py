'''The |pydwf.core.api.protocol_uart| module provides a single class: |ProtocolUART|.'''

from typing import Tuple
import ctypes

from pydwf.core.dwf_device_subapi import AbstractDwfDeviceSubAPI

from pydwf.core.auxiliary.typespec_ctypes import typespec_ctypes
from pydwf.core.auxiliary.constants import RESULT_SUCCESS


class ProtocolUART(AbstractDwfDeviceSubAPI):
    '''The |ProtocolUART| class provides access to the UART protocol functionality of a |DwfDevice:link|.

    Attention:
        Users of |pydwf| should not create instances of this class directly.

        It is instantiated during initialization of a |DwfDevice| and subsequently accessible via its
        |protocol.uart:link| attribute.
    '''

    def reset(self) -> None:
        '''Reset the UART protocol functionality.

        Raises:
            DwfLibraryError: An error occurred while executing the *reset* operation.
        '''
        result = self.lib.FDwfDigitalUartReset(self.hdwf)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rateSet(self, baudrate: float) -> None:
        '''Set the UART baudrate.

        Parameters:
            baudrate (float): The baud-rate used by the receiver and transmitter.

                Commonly encountered values are 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600, and 115200,
                but other values are valid as well.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartRateSet(self.hdwf, baudrate)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def bitsSet(self, databits: int) -> None:
        '''Set the number of UART data bits.

        Parameters:
            databits (int): The number of data-bits used by the receiver and transmitter.

                The most common choice is 8, but other values are possible.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartBitsSet(self.hdwf, databits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def paritySet(self, parity: int) -> None:
        '''Set the UART character parity.

        Parameters:
            parity (int): The parity used by the receiver and transmitter:

                * 0 — no parity
                * 1 — odd parity
                * 2 — even parity

                The most common choice is *no parity* (i.e., 0).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartParitySet(self.hdwf, parity)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def polaritySet(self, polarity: int) -> None:
        '''Set the UART signal polarity.

        Parameters:
            polarity (int): The polarity used by the receiver and transmitter:

                * 0 — normal
                * 1 — inverted

                The most common choice and default polarity is *normal* (i.e., 0).

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartPolaritySet(self.hdwf, polarity)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def stopSet(self, stopbits: float) -> None:
        '''Set the number of UART stop bits.

        Parameters:
            stopbits (float): The number of stop-bits used by the receiver and transmitter.

                The most common choice is 1 stop-bit. Other values that are (rarely)
                encountered are 1.5 and 2 stop-bits.

                Note that the actual number of stop-bits is the number specified here,
                rounded up to the next highest integer.

                The parameter is declared as a *float* in anticipation of future support
                for 1.5 stop-bits.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartStopSet(self.hdwf, stopbits)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def txSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the UART's outgoing (TX) signal will be transmitted.

        Parameters:
            channel_index (int): The digital channel (pin) on which to transmit data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartTxSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rxSet(self, channel_index: int) -> None:
        '''Set the digital channel (pin) where the UART's incoming (RX) signal is received.

        Parameters:
            channel_index (int): The digital channel (pin) on which to receive data.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartRxSet(self.hdwf, channel_index)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def tx(self, tx_data: bytes) -> None:
        '''Transmit data according to the currently active UART settings.

        Parameters:
            tx_data (bytes): The data to be transmitted.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        result = self.lib.FDwfDigitalUartTx(self.hdwf, tx_data, len(tx_data))
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()

    def rx(self, rx_max: int) -> Tuple[bytes, int]:
        '''Receive UART data or prepare for reception.

        Important:
            This method must be called with value 0 prior to receiving data, to initialize the receiver.

        Parameters:
            rx_max (int): If 0, initialize the receiver.

                Otherwise, receive the specified number of characters.

        Returns:
            Tuple[bytes, int]: Bytes received and parity error indication.

        Todo:
            The meaning of the parity error indication is currently unclear.
            This needs to be investigated.

        Raises:
            DwfLibraryError: An error occurred while executing the operation.
        '''
        c_rx_count = typespec_ctypes.c_int()
        c_parity_error = typespec_ctypes.c_int()
        c_rx_buffer = ctypes.create_string_buffer(rx_max)
        result = self.lib.FDwfDigitalUartRx(self.hdwf, c_rx_buffer, rx_max, c_rx_count, c_parity_error)
        if result != RESULT_SUCCESS:
            raise self.dwf.exception()
        rx_count = c_rx_count.value
        rx_buffer = c_rx_buffer.raw[:rx_count]
        parity_error = c_parity_error.value
        return (rx_buffer, parity_error)
